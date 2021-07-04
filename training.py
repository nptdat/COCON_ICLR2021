import logging
from typing import List, Tuple, Dict
import os
import random

from tqdm import tqdm, trange
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup
)

from dataset import load_and_cache_examples
from utils.utils import _clear_checkpoints, _rotate_checkpoints


from torch.utils.tensorboard import SummaryWriter

from utils.utils import set_seed, to_one_hot


logger = logging.getLogger(__name__)


def train_cocon(args, train_dataset, model, tokenizer, cocon_block, disc_model=None, model_config=None, transform_h_after_layernorm=False):
    """ Train the model """
    tb_log_dir = os.path.join(args.output_dir, 'runs')
    tb_writer = SummaryWriter(tb_log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    # check if max/min_hs_tis_split_offset is out of range of hs_len or tis_len
    offset_hs_tis_split = False
    if args.min_hs_tis_split_offset != 0:
        offset_hs_tis_split = True
        if (args.min_hs_tis_split_offset+args.hs_len < 0):
            raise ValueError(
                "min_hs_tis_split_offset is out of bound"
            )
    if args.max_hs_tis_split_offset != 0:
        offset_hs_tis_split = True
        if (min(args.cs_len, args.tis_len) - args.max_hs_tis_split_offset < 0):
            raise ValueError(
                "max_hs_tis_split_offset is out of bound"
            )

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay) for cocon_block
    no_decay = ["bias", "LayerNorm.weight"]
    cocon_block_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in cocon_block.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in cocon_block.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    cocon_block_optimizer = AdamW(cocon_block_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    cocon_block_scheduler = get_linear_schedule_with_warmup(
        cocon_block_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    if args.lambda_adv > 0:
        # Prepare optimizer and schedule (linear warmup and decay) for disc_model
        no_decay = ["bias", "LayerNorm.weight"]
        disc_model_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in disc_model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in disc_model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        disc_model_optimizer = AdamW(disc_model_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        disc_model_scheduler = get_linear_schedule_with_warmup(
            disc_model_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )

    # Prepare optimizer and schedule (linear warmup and decay) for lm model
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "cocon_block_optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "cocon_block_scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        cocon_block_optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "cocon_block_optimizer.pt")))
        cocon_block_scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "cocon_block_scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        cocon_block = torch.nn.DataParallel(cocon_block)
        if args.lambda_adv > 0:
            disc_model = torch.nn.DataParallel(disc_model)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * 1,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility

    start_hist_cocon_lm = False
    start_cycle_ar_cocon_recon = False
    start_other_context_cocon = False
    start_adv = False
    adv_gen_opt_step = 0
    adv_disc_opt_step = 0
    first_save = True
    for epoch_ind in train_iterator:
        logger.info( "epoch_ind: {}".format(epoch_ind))

        if args.lambda_cycle_ar_cocon_recon_lm_loss > 0 and args.per_gpu_train_cycle_ar_cocon_recon_batch_size is not None and epoch_ind == args.epoch_ind_to_start_cycle_ar_cocon_recon:
            args.train_cycle_ar_cocon_recon_batch_size = args.per_gpu_train_cycle_ar_cocon_recon_batch_size * max(1, args.n_gpu)
            logger.info( "Changing train_batch_size to {} due to start_cycle_ar_cocon_recon".format(args.train_cycle_ar_cocon_recon_batch_size))

            train_dataloader = DataLoader(
                train_dataset, sampler=train_sampler, batch_size=args.train_cycle_ar_cocon_recon_batch_size, collate_fn=collate
            )

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, lm_labels = (batch, batch)

            # Skip batch if seq len is shorter than hs_len, i.e. no tis or cs text
            if inputs.shape[1] < args.hs_len:
                logger.info("inputs.shape[1] < args.hs_len, skipping batch")
                continue

            # Split train samples into hs, tis, cs segments
            # variable split offset
            if offset_hs_tis_split:
                hs_tis_split_ind = random.randint(args.min_hs_tis_split_offset, args.max_hs_tis_split_offset)
                hs_len = args.hs_len + hs_tis_split_ind
                cs_len = args.cs_len - hs_tis_split_ind
                tis_len = args.tis_len - hs_tis_split_ind
            else:
                hs_len = args.hs_len
                cs_len = args.cs_len
                tis_len = args.tis_len

            lm_labels = lm_labels[:, :hs_len+tis_len]
            inputs = inputs.to(args.device)
            lm_labels = lm_labels.to(args.device)

            original_context_seq = inputs[:, hs_len:hs_len+cs_len]

            # use batch with + 1 index as other sample
            other_sample_inputs = torch.cat([inputs[-1:], inputs[:-1]], dim=0)
            other_sample_lm_labels = other_sample_inputs[:, :hs_len+tis_len]
            other_sample_history_seq = other_sample_inputs[:, :hs_len]

            model.eval()

            cocon_block.train()
            if args.lambda_adv > 0:
                disc_model.train()

            if args.gradient_accumulation_steps == 1:
                # reset grad in model
                model.zero_grad()
                cocon_block.zero_grad()
                if args.lambda_adv > 0 and start_adv:
                    disc_model.zero_grad()

            with torch.no_grad():
                if transform_h_after_layernorm:
                    hidden_states = model(inputs, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')[0]  # [N, L, C]
                    context_seq_hidden_states = model(original_context_seq, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')[0]  # [N, L, C]
                else:
                    hidden_states = model(inputs, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)[0]  # [N, L, C]
                    context_seq_hidden_states = model(original_context_seq, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)[0]  # [N, L, C]

            original_hidden_states = hidden_states
            original_history_seq_hidden_states = original_hidden_states[:, :hs_len]
            original_transform_input_seq_hidden_states = original_hidden_states[:, hs_len:hs_len+tis_len]
            original_context_seq_hidden_states = context_seq_hidden_states

            # use batch with + 1 index as other sample
            other_sample_hidden_states = torch.cat([hidden_states[-1:], hidden_states[:-1]], dim=0)
            other_sample_history_seq_hidden_states = other_sample_hidden_states[:, :hs_len]

            # self_cocon_lm_loss computation, CS: original_context_seq_hidden_states, HS: original_history_seq_hidden_states, TIS: original_transform_input_seq_hidden_states
            # single FF pass, no need for AR
            # cs & tis mask computation
            if args.self_cocon_lm_cs_mask_prob > 0 or args.self_cocon_lm_tis_mask_prob > 0:
                if args.self_cocon_lm_mutual_exc_mask:
                    max_cs_tis_len = max(original_context_seq_hidden_states.shape[1], original_transform_input_seq_hidden_states.shape[1])
                    total_mask_prob = args.self_cocon_lm_cs_mask_prob + args.self_cocon_lm_tis_mask_prob
                    if total_mask_prob > 1:
                        logger.warning("self_cocon_lm_mask_prob > 1, bounding it to 1")
                        total_mask_prob = 1

                    prob_matrix = torch.full([original_context_seq_hidden_states.shape[0], max_cs_tis_len], total_mask_prob)
                    all_masked_indices = torch.bernoulli(prob_matrix).bool()

                    prob_allocated_cs = args.self_cocon_lm_cs_mask_prob / total_mask_prob
                    allocated_cs_prob_matrix = torch.full([original_context_seq_hidden_states.shape[0], max_cs_tis_len], prob_allocated_cs)
                    allocated_cs_indices = torch.bernoulli(allocated_cs_prob_matrix).bool()

                    cs_masked_indices = all_masked_indices & allocated_cs_indices
                    tis_masked_indices = all_masked_indices & ~allocated_cs_indices

                    if original_context_seq_hidden_states.shape[1] != max_cs_tis_len:
                        cs_masked_indices = cs_masked_indices[:, original_context_seq_hidden_states.shape[1]]
                    elif original_transform_input_seq_hidden_states.shape[1] != max_cs_tis_len:
                        tis_masked_indices = tis_masked_indices[:, original_transform_input_seq_hidden_states.shape[1]]

                else:
                    cs_prob_matrix = torch.full(original_context_seq_hidden_states.shape[:-1], args.self_cocon_lm_cs_mask_prob)
                    cs_masked_indices = torch.bernoulli(cs_prob_matrix).bool()
                    tis_prob_matrix = torch.full(original_transform_input_seq_hidden_states.shape[:-1], args.self_cocon_lm_tis_mask_prob)
                    tis_masked_indices = torch.bernoulli(tis_prob_matrix).bool()

                self_cocon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=original_context_seq_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, cs_masked_indices=cs_masked_indices, tis_masked_indices=tis_masked_indices, cs_self_attn_mask_prob=args.self_token_mask_prob) # [N, L, C]
            else:
                self_cocon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=original_context_seq_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.self_token_mask_prob) # [N, L, C]

            # concat cocon output with original history_seq
            self_cocon_lm_tail_input = torch.cat([original_history_seq_hidden_states[:, :-1], self_cocon_hidden_states], dim=1)

            # compute lm loss only on cocon-transformed hidden_states
            lm_logit_first_index = original_history_seq_hidden_states.shape[1] -1
            lm_labels_first_index = lm_logit_first_index + 1

            # compute lm tail logits output and loss values
            if transform_h_after_layernorm:
                self_cocon_lm_tail_outputs = model(input_hidden_state=self_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
            else:
                self_cocon_lm_tail_outputs = model(input_hidden_state=self_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

            self_cocon_lm_loss = self_cocon_lm_tail_outputs[0]

            if args.track_loss_gradnorms and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                self_cocon_lm_loss_grad = torch.autograd.grad(self_cocon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                self_cocon_lm_loss_gradnorm = torch.norm(self_cocon_lm_loss_grad)

            if args.lambda_self_cocon_lm_loss > 0:
                total_loss = args.lambda_self_cocon_lm_loss * self_cocon_lm_loss
            else:
                total_loss = 0


            if args.lambda_hist_cocon_lm_loss > 0:
                # Check whether it is time to start adv training
                if start_hist_cocon_lm == False and epoch_ind == args.epoch_ind_to_start_hist_cocon_lm and step == args.step_ind_to_start_hist_cocon_lm:
                    logger.info( "starting hist_cocon_lm_loss training")
                    logger.info( "step_ind_to_start_hist_cocon_lm: {}, step: {}".format(args.step_ind_to_start_hist_cocon_lm, step))
                    logger.info( "epoch_ind_to_start_hist_cocon_lm: {}, epoch_ind: {}".format(args.epoch_ind_to_start_hist_cocon_lm, epoch_ind))
                    start_hist_cocon_lm = True

            if start_hist_cocon_lm or (args.track_hist_cocon_lm_loss and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0):
                hist_cocon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=None, history_seq=original_history_seq_hidden_states, include_sos_output=True) # [N, L, C]

                # concat cocon output with original history_seq
                hist_cocon_lm_tail_input = torch.cat([original_history_seq_hidden_states[:, :-1], hist_cocon_hidden_states], dim=1)

                # compute lm loss only on cocon-transformed hidden_states
                lm_logit_first_index = original_history_seq_hidden_states.shape[1] -1
                lm_labels_first_index = lm_logit_first_index + 1

                # compute lm tail logits output and loss values
                if transform_h_after_layernorm:
                    hist_cocon_lm_tail_outputs = model(input_hidden_state=hist_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                else:
                    hist_cocon_lm_tail_outputs = model(input_hidden_state=hist_cocon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                hist_cocon_lm_loss = hist_cocon_lm_tail_outputs[0]

                if args.track_loss_gradnorms and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                    hist_cocon_lm_loss_grad = torch.autograd.grad(hist_cocon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                    hist_cocon_lm_loss_gradnorm = torch.norm(hist_cocon_lm_loss_grad)

                if args.lambda_hist_cocon_lm_loss > 0:
                    total_loss += args.lambda_hist_cocon_lm_loss * hist_cocon_lm_loss


            if args.lambda_adv > 0:
                # Check whether it is time to start adv training
                if start_adv == False and epoch_ind == args.epoch_ind_to_start_adv and step == args.step_ind_to_start_adv:
                    logger.info( "starting adversarial learning")
                    logger.info( "step_ind_to_start_adv: {}, step: {}".format(args.step_ind_to_start_adv, step))
                    logger.info( "epoch_ind_to_start_adv: {}, epoch_ind: {}".format(args.epoch_ind_to_start_adv, epoch_ind))
                    start_adv = True

            # cycle_ar_cocon_recon_lm_loss computation, Step 1 CS: original_context_seq_hidden_states, HS: other_sample_history_seq_hidden_states, TIS: None (AR generation)
            if args.lambda_cycle_ar_cocon_recon_lm_loss > 0 and epoch_ind == args.epoch_ind_to_start_cycle_ar_cocon_recon and step == args.step_ind_to_start_cycle_ar_cocon_recon:
                logger.info( "starting cycle_ar_cocon_recon_lm learning")
                logger.info( "step_ind_to_start_cycle_ar_cocon_recon: {}, step: {}".format(args.step_ind_to_start_cycle_ar_cocon_recon, step))
                logger.info( "epoch_ind_to_start_cycle_ar_cocon_recon: {}, epoch_ind: {}".format(args.epoch_ind_to_start_cycle_ar_cocon_recon, epoch_ind))
                start_cycle_ar_cocon_recon = True

            if start_cycle_ar_cocon_recon or start_adv:
                cur_len = 0
                cocon_block_output = None
                cocon_th_gen_input = None
                cocon_th_gen_output = None

                cocon_output_embeds = None
                lm_tail_past=None
                lm_head_past=None

                max_cocon_AR_length = min(args.max_cocon_AR_length, original_transform_input_seq_hidden_states.shape[1]) # limit cocon output length to hidden states' length

                other_sample_history_seq_one_hot_prob = to_one_hot(other_sample_history_seq, n_dims=model_config.vocab_size).to(args.device)
                other_sample_history_seq_embeds = torch.matmul(other_sample_history_seq_one_hot_prob, model.transformer.wte.weight)

                # cocon_th_gen_output: autoreg generation with CS- & HS-conditioned cocon op
                while cur_len < max_cocon_AR_length:
                    cocon_transformed_hidden_states = cocon_block(cocon_th_gen_input, context_seq=original_context_seq_hidden_states, history_seq=other_sample_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.cycle_self_token_mask_prob) # [N, L, C]

                    if cur_len == 0:
                        cocon_block_output = cocon_transformed_hidden_states[:, -1:]
                    else:
                        cocon_block_output = torch.cat([cocon_block_output, cocon_transformed_hidden_states[:, -1:]], dim=1)

                    if args.use_only_last_cocon_output_for_ar:
                        if cocon_th_gen_input is not None:
                            hist_plus_cocon_hidden_states = torch.cat([other_sample_history_seq_hidden_states, cocon_th_gen_input[:, :-1], cocon_transformed_hidden_states[:, -1:]], dim=1)
                        else:
                            hist_plus_cocon_hidden_states = torch.cat([other_sample_history_seq_hidden_states[:, :-1], cocon_transformed_hidden_states[:, -1:]], dim=1)
                    else:
                        hist_plus_cocon_hidden_states = torch.cat([other_sample_history_seq_hidden_states[:, :-1], cocon_transformed_hidden_states], dim=1)

                    # optimized tail computation
                    if transform_h_after_layernorm:
                        lm_tail_inputs = model.prepare_hidden_state_inputs_for_generation(input_hidden_state=hist_plus_cocon_hidden_states, past=lm_tail_past, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                    else:
                        lm_tail_inputs = model.prepare_hidden_state_inputs_for_generation(input_hidden_state=hist_plus_cocon_hidden_states, past=lm_tail_past, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                    tail_outputs = model(**lm_tail_inputs)
                    next_token_logits = tail_outputs[0]  # [N,L,C] where C is vocab_size
                    if next_token_logits.shape[1] > 1:
                        next_token_logits = next_token_logits[:, -1:]
                    lm_tail_past = tail_outputs[1]

                    if args.gen_gumbel_softmax:
                        next_cocon_output_prob = torch.nn.functional.gumbel_softmax(next_token_logits, dim=-1)
                    else:
                        next_cocon_output_prob = torch.nn.functional.softmax(next_token_logits, dim=-1)

                    next_cocon_output_embed = torch.matmul(next_cocon_output_prob, model.transformer.wte.weight) # [N, 1, C]

                    if cur_len == 0:
                        cocon_output_embeds = next_cocon_output_embed
                        hist_plus_cocon_output_embeds = torch.cat([other_sample_history_seq_embeds, next_cocon_output_embed], dim=1)
                    else:
                        cocon_output_embeds = torch.cat([cocon_output_embeds, next_cocon_output_embed], dim=1)
                        hist_plus_cocon_output_embeds = torch.cat([hist_plus_cocon_output_embeds, next_cocon_output_embed], dim=1)

                    # optimized head computation
                    if transform_h_after_layernorm:
                        lm_head_inputs = model.prepare_embeds_inputs_for_generation(inputs_embeds=hist_plus_cocon_output_embeds, past=lm_head_past, input_ids=None, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')
                    else:
                        lm_head_inputs = model.prepare_embeds_inputs_for_generation(inputs_embeds=hist_plus_cocon_output_embeds, past=lm_head_past, input_ids=None, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)

                    head_outputs = model(**lm_head_inputs)
                    cocon_gen_output_h = head_outputs[0]
                    if cocon_gen_output_h.shape[1] > 1:
                        next_h = cocon_gen_output_h[:, -1:]
                    else:
                        next_h = cocon_gen_output_h

                    lm_head_past = head_outputs[1]
                    if cur_len % args.train_cycle_detach_interval == 0:
                        h_to_cat_input = next_h.detach()
                    else:
                        h_to_cat_input = next_h

                    if cur_len == 0:
                        cocon_th_gen_input = h_to_cat_input
                        cocon_th_gen_output = next_h
                    else:
                        cocon_th_gen_input = torch.cat([cocon_th_gen_input, h_to_cat_input], dim=1)
                        cocon_th_gen_output = torch.cat([cocon_th_gen_output, next_h], dim=1)

                    cur_len = cocon_th_gen_input.shape[1]

                if start_cycle_ar_cocon_recon:
                    ar_cocon_final_output_embeds = cocon_output_embeds

                    if transform_h_after_layernorm:
                        ar_cocon_output_hidden_states = model(input_ids=None, inputs_embeds=ar_cocon_final_output_embeds, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind, return_point='next_block_ln_1')[0]  # [N, L, C]
                    else:
                        ar_cocon_output_hidden_states = model(input_ids=None, inputs_embeds=ar_cocon_final_output_embeds, output_after_block_ind=args.output_hidden_for_cocon_after_block_ind)[0]  # [N, L, C]

                    # tis mask computation
                    if args.cycle_ar_cocon_recon_lm_tis_mask_prob > 0:
                        tis_prob_matrix = torch.full(original_transform_input_seq_hidden_states.shape[:-1], args.cycle_ar_cocon_recon_lm_tis_mask_prob)
                        tis_masked_indices = torch.bernoulli(tis_prob_matrix).bool()

                        cycle_ar_cocon_recon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=ar_cocon_output_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, tis_masked_indices=tis_masked_indices, cs_self_attn_mask_prob=args.cycle_self_token_mask_prob) # [N, L, C]
                    else:
                        cycle_ar_cocon_recon_hidden_states = cocon_block(original_transform_input_seq_hidden_states, context_seq=ar_cocon_output_hidden_states, history_seq=original_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.cycle_self_token_mask_prob) # [N, L, C]

                    # concat cocon output with original history_seq, replace original_history_seq_hidden_states' last hidden state with first element of cycle_ar_cocon_recon_hidden_states
                    cycle_ar_cocon_recon_lm_tail_input = torch.cat([original_history_seq_hidden_states[:, :-1], cycle_ar_cocon_recon_hidden_states], dim=1)

                    # compute lm loss only on cocon-transformed hidden_states
                    lm_logit_first_index = original_history_seq_hidden_states.shape[1] -1
                    lm_labels_first_index = lm_logit_first_index + 1

                    # compute lm tail logits output and loss values
                    if transform_h_after_layernorm:
                        cycle_ar_cocon_recon_lm_tail_outputs = model(input_hidden_state=cycle_ar_cocon_recon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                    else:
                        cycle_ar_cocon_recon_lm_tail_outputs = model(input_hidden_state=cycle_ar_cocon_recon_lm_tail_input, labels=lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                    cycle_ar_cocon_recon_lm_loss = cycle_ar_cocon_recon_lm_tail_outputs[0]

                    if args.track_loss_gradnorms and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                        cycle_ar_cocon_recon_lm_loss_grad = torch.autograd.grad(cycle_ar_cocon_recon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                        cycle_ar_cocon_recon_lm_loss_gradnorm = torch.norm(cycle_ar_cocon_recon_lm_loss_grad)

                    if args.lambda_cycle_ar_cocon_recon_lm_loss > 0:
                        total_loss += args.lambda_cycle_ar_cocon_recon_lm_loss * cycle_ar_cocon_recon_lm_loss


            # context_ar_cocon_lm_loss computation, Step 1 CS: other_sample_context_seq_hidden_states, HS: other_sample_history_seq_hidden_states, TIS: cocon_th_gen_output (AR generated)
            # cat other_sample_history_seq_embeds with ar_cocon_final_output_embeds
            if args.lambda_other_context_cocon_lm_loss > 0 and epoch_ind == args.epoch_ind_to_start_other_context_cocon and step == args.step_ind_to_start_other_context_cocon:
                logger.info( "starting cycle_ar_cocon_recon_lm learning")
                logger.info( "step_ind_to_start_other_context_cocon: {}, step: {}".format(args.step_ind_to_start_other_context_cocon, step))
                logger.info( "epoch_ind_to_start_other_context_cocon: {}, epoch_ind: {}".format(args.epoch_ind_to_start_other_context_cocon, epoch_ind))
                start_other_context_cocon = True

            if start_other_context_cocon:
                other_context_cocon_hidden_states = cocon_block(cocon_th_gen_output, context_seq=original_context_seq_hidden_states, history_seq=other_sample_history_seq_hidden_states, include_sos_output=True, cs_self_attn_mask_prob=args.other_context_self_token_mask_prob) # [N, L, C]

                # concat cocon output with original history_seq
                other_context_cocon_lm_tail_input = torch.cat([other_sample_history_seq_hidden_states[:, :-1], other_context_cocon_hidden_states], dim=1)

                # compute lm loss only on cocon-transformed hidden_states
                lm_logit_first_index = other_sample_history_seq_hidden_states.shape[1] -1
                lm_labels_first_index = lm_logit_first_index + 1

                # compute lm tail logits output and loss values
                if transform_h_after_layernorm:
                    other_context_cocon_lm_tail_outputs = model(input_hidden_state=other_context_cocon_lm_tail_input, labels=other_sample_lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1, input_point='current_block_ln_1')
                else:
                    other_context_cocon_lm_tail_outputs = model(input_hidden_state=other_context_cocon_lm_tail_input, labels=other_sample_lm_labels, lm_logit_first_index=lm_logit_first_index, lm_labels_first_index=lm_labels_first_index, input_before_block_ind=args.output_hidden_for_cocon_after_block_ind+1)

                other_context_cocon_lm_loss = other_context_cocon_lm_tail_outputs[0]

                if args.track_loss_gradnorms and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                    other_context_cocon_lm_loss_grad = torch.autograd.grad(other_context_cocon_lm_loss, cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                    other_context_cocon_lm_loss_gradnorm = torch.norm(other_context_cocon_lm_loss_grad)

                if args.lambda_other_context_cocon_lm_loss > 0:
                    total_loss += args.lambda_other_context_cocon_lm_loss * other_context_cocon_lm_loss


            if args.lambda_adv > 0:
                # Compute adv LOSS :  cocon_th_gen_output or cocon_block_output for adv training
                if start_adv:
                    if args.adv_use_th_gen_output:
                        disc_fake_input = cocon_th_gen_output
                    else:
                        disc_fake_input = cocon_block_output
                    disc_real_input = original_transform_input_seq_hidden_states

                    # detach for disc opt step later
                    if step % args.disc_update_interval == 0:
                        disc_fake_input_detached = disc_fake_input.detach()
                        disc_real_input_detached = disc_real_input.detach()

                    if step % args.gen_update_interval == 0:
                        # Adversarial cocon training step: train GEN
                        real_disc_label = torch.ones(hidden_states.shape[0], 1).to(args.device)
                        fake_disc_label = torch.zeros(hidden_states.shape[0], 1).to(args.device)

                        d_fake_loss, d_fake_logits = disc_model(disc_fake_input, fake_disc_label)
                        d_real_loss, d_real_logits = disc_model(disc_real_input, real_disc_label)

                        if args.track_loss_gradnorms and args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                            adv_loss_grad = torch.autograd.grad((-1*d_fake_loss - d_real_loss), cocon_block.cocon_attn.c_attn.weight, retain_graph=True)[0]
                            adv_loss_gradnorm = torch.norm(adv_loss_grad)

                        if args.lambda_adv > 0:
                            total_loss += args.lambda_adv * ( -1*d_fake_loss - d_real_loss)
                        adv_gen_opt_step += 1


            if args.n_gpu > 1:
                total_loss = total_loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                total_loss = total_loss / args.gradient_accumulation_steps

            total_loss.backward()

            tr_loss += total_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(cocon_block.parameters(), args.max_grad_norm)
                if args.logging_steps > 0 and (global_step+1) % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, (global_step+1))

                    tb_writer.add_scalar("LR/lm_lr", cocon_block_scheduler.get_lr()[0], (global_step+1))
                    tb_writer.add_scalar("LOSS/total_loss", (tr_loss - logging_loss) / args.logging_steps, (global_step+1))

                    tb_writer.add_scalar("LOSS/self_cocon_lm_loss", self_cocon_lm_loss.item(), (global_step+1))

                    if args.lambda_hist_cocon_lm_loss > 0 or args.track_hist_cocon_lm_loss:
                        tb_writer.add_scalar("LOSS/self_cocon_lm_loss", self_cocon_lm_loss.item(), (global_step+1))

                    if (args.lambda_hist_cocon_lm_loss > 0 and start_hist_cocon_lm) or args.track_hist_cocon_lm_loss:
                        tb_writer.add_scalar("LOSS/hist_cocon_lm_loss", hist_cocon_lm_loss, (global_step+1))

                    if args.lambda_adv > 0 and start_adv:
                        tb_writer.add_scalar("LOSS/d_fake_loss", d_fake_loss.item(), (global_step+1))
                        tb_writer.add_scalar("LOSS/d_real_loss", d_real_loss.item(), (global_step+1))

                    if start_cycle_ar_cocon_recon:
                        tb_writer.add_scalar("LOSS/cycle_ar_cocon_recon_lm_loss", cycle_ar_cocon_recon_lm_loss.item(), (global_step+1))

                    if start_other_context_cocon:
                        tb_writer.add_scalar("LOSS/other_context_cocon_lm_loss", other_context_cocon_lm_loss.item(), (global_step+1))

                    tb_writer.add_scalar("GRADIENT/TOTALLOSS_model_attn_c_attn_gradnorm", torch.norm(cocon_block.cocon_attn.c_attn.weight.grad), (global_step+1))
                    if args.track_loss_gradnorms:
                        tb_writer.add_scalar("GRADIENT/self_cocon_lm_loss_gradnorm", self_cocon_lm_loss_gradnorm, (global_step+1))

                        if (args.lambda_hist_cocon_lm_loss > 0 and start_hist_cocon_lm) or args.track_hist_cocon_lm_loss:
                            tb_writer.add_scalar("GRADIENT/hist_cocon_lm_loss_gradnorm", hist_cocon_lm_loss_gradnorm, (global_step+1))
                        if start_cycle_ar_cocon_recon:
                            tb_writer.add_scalar("GRADIENT/cycle_ar_cocon_recon_lm_loss_gradnorm", cycle_ar_cocon_recon_lm_loss_gradnorm, (global_step+1))
                        if start_other_context_cocon:
                            tb_writer.add_scalar("GRADIENT/other_context_cocon_lm_loss_gradnorm", other_context_cocon_lm_loss_gradnorm, (global_step+1))
                        if start_adv:
                            tb_writer.add_scalar("GRADIENT/adv_loss_gradnorm", adv_loss_gradnorm, (global_step+1))

                    logging_loss = tr_loss

                cocon_block_optimizer.step() # opt.step() does not zero_grad, need to zero.grad() manually
                cocon_block_scheduler.step() # Update learning rate schedule

                cocon_block.zero_grad()
                model.zero_grad()


            # Adv DISC opt step
            if step % args.disc_update_interval == 0 and start_adv:
                if args.gradient_accumulation_steps == 1:
                    disc_model.zero_grad()

                # Adversarial cocon training step: train DISC
                real_disc_label = torch.ones(hidden_states.shape[0], 1).to(args.device)
                fake_disc_label = torch.zeros(hidden_states.shape[0], 1).to(args.device)
                d_fake_loss, d_fake_logits = disc_model(disc_fake_input_detached, fake_disc_label)
                d_real_loss, d_real_logits = disc_model(disc_real_input_detached, real_disc_label)
                total_disc_loss = d_fake_loss + d_real_loss

                disc_fake_num_correct = torch.sum(d_fake_logits < 0)
                disc_fake_acc = disc_fake_num_correct.type(torch.float64) / len(fake_disc_label)
                disc_real_num_correct = torch.sum(d_real_logits > 0)
                disc_real_acc = disc_real_num_correct.type(torch.float64) / len(real_disc_label)

                if args.n_gpu > 1:
                    total_disc_loss = total_disc_loss.mean()  # mean() to average on multi-gpu parallel training
                if args.gradient_accumulation_steps > 1:
                    total_disc_loss = total_disc_loss / args.gradient_accumulation_steps

                total_disc_loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(disc_model.parameters(), args.max_grad_norm)

                if args.logging_steps > 0 and step % args.logging_steps == 0: # to sync with disc update step
                    tb_writer.add_scalar("LR/disc_lr", disc_model_scheduler.get_lr()[0], (global_step+1))
                    tb_writer.add_scalar("LOSS/disc_loss", total_disc_loss.item(), (global_step+1))

                    tb_writer.add_scalar("LOSS/disc_fake_loss", d_fake_loss.item(), (global_step+1))
                    tb_writer.add_scalar("LOSS/disc_real_loss", d_real_loss.item(), (global_step+1))

                    tb_writer.add_scalar("GRADIENT/DISCLOSS_discmodel_conv3_gradnorm", torch.norm(disc_model.conv_layers[2].weight.grad), (global_step+1))

                    tb_writer.add_scalar("ACC/disc_fake_acc", disc_fake_acc, (global_step+1))
                    tb_writer.add_scalar("ACC/disc_real_acc", disc_real_acc, (global_step+1))
                elif (adv_disc_opt_step * args.disc_update_interval) < args.steps_to_closely_monitor_adv and step % (args.logging_steps // 50) == 0:
                    tb_writer.add_scalar("LR/disc_lr", disc_model_scheduler.get_lr()[0], (global_step+1))
                    tb_writer.add_scalar("LOSS/disc_loss", total_disc_loss.item(), (global_step+1))

                    tb_writer.add_scalar("LOSS/disc_fake_loss", d_fake_loss.item(), (global_step+1))
                    tb_writer.add_scalar("LOSS/disc_real_loss", d_real_loss.item(), (global_step+1))

                    tb_writer.add_scalar("GRADIENT/DISCLOSS_discmodel_conv3_gradnorm", torch.norm(disc_model.conv_layers[2].weight.grad), (global_step+1))

                    tb_writer.add_scalar("ACC/disc_fake_acc", disc_fake_acc, (global_step+1))
                    tb_writer.add_scalar("ACC/disc_real_acc", disc_real_acc, (global_step+1))


                disc_model_optimizer.step()
                disc_model_scheduler.step()  # Update learning rate schedule

                disc_model.zero_grad()
                adv_disc_opt_step += 1


            # Save model
            if args.save_steps > 0 and step % args.save_steps == 0: # to sync with gen/disc update step
                checkpoint_prefix = "cocon_block_checkpoint"
                if first_save:
                    _clear_checkpoints(args, checkpoint_prefix)
                    first_save = False

                # Save model checkpoint
                output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, (global_step+1)))
                os.makedirs(output_dir, exist_ok=True)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving cocon_block model checkpoint to %s", output_dir)

                _rotate_checkpoints(args, checkpoint_prefix)

                cocon_block_weights_name = "cocon_block_pytorch_model.bin"
                output_cocon_block_model_file = os.path.join(output_dir, cocon_block_weights_name)
                torch.save(cocon_block.state_dict(), output_cocon_block_model_file)
                logger.info("cocon_block model weights saved in {}".format(output_cocon_block_model_file))

                torch.save(cocon_block_optimizer.state_dict(), os.path.join(output_dir, "cocon_block_optimizer.pt"))
                torch.save(scheduler.state_dict(), os.path.join(output_dir, "cocon_block_scheduler.pt"))
                logger.info("Saving cocon_block optimizer and scheduler states to %s", output_dir)

            global_step += 1

            if (args.max_steps > 0 and global_step > args.max_steps) or (args.epoch_max_steps > 0 and step > args.epoch_max_steps):
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


def train_lm(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """
    tb_writer = SummaryWriter()

    if args.per_gpu_train_lm_batch_size <= 0:
        args.per_gpu_train_lm_batch_size = args.per_gpu_train_batch_size
    args.train_lm_batch_size = args.per_gpu_train_lm_batch_size * max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_lm_batch_size, collate_fn=collate
    )

    if args.lm_max_steps > 0:
        t_total = args.lm_max_steps
        args.num_lm_train_epochs = args.lm_max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_lm_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Train!
    logger.info("***** Running LM training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_lm_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_lm_batch_size
        * args.gradient_accumulation_steps
        * 1,
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_lm_train_epochs), desc="Epoch")
    set_seed(args)  # Added here for reproducibility
    first_save = True
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, labels=labels)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)


            if args.output_meanvars:
                all_meanvars = outputs[-1]
                all_meanvars_tensor = []

                for block_ind, meanvars_in_block in enumerate(all_meanvars):
                    for layer_ind, meanvars_in_layer in enumerate(meanvars_in_block):
                        for stats_ind, stats in enumerate(meanvars_in_layer): # stats.shape: [batch_size, n_embd], mean & var
                            all_meanvars_tensor.append(stats)

                all_meanvars = torch.stack(all_meanvars_tensor, dim=1)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                    tb_writer.add_scalar("LM/lr", scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar("LM/loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = "checkpoint"
                    if first_save:
                        _clear_checkpoints(args, checkpoint_prefix)
                        first_save = False

                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
                    os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving LM model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving LM optimizer and scheduler states to %s", output_dir)

            if args.lm_max_steps > 0 and global_step > args.lm_max_steps:
                epoch_iterator.close()
                break
        if args.lm_max_steps > 0 and global_step > args.lm_max_steps:
            train_iterator.close()
            break

    tb_writer.close()

    return global_step, tr_loss / global_step


# evaluate perplexity
def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True, text_json_key=args.text_json_key, prepended_text_to_remove=args.prepended_text_to_remove)

    os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        if labels.shape[1] < 2:
            continue
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()

        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    output_eval_file = os.path.join(eval_output_dir, prefix, args.eval_output_filename)
    with open(output_eval_file, "w") as writer:
        logger.info("***** PPL Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result
