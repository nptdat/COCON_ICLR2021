from typing import List, Dict
import os
import logging
import json

from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
)

from dataset import load_and_cache_examples


logger = logging.getLogger(__name__)


# Use to generate cocon-edited text with either trained or simple cocon op
def generate_cocon_compute(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cocon_block=None, prefix="",
    random_sample_data=False, use_only_first_context_source_batch=False, use_only_first_custom_mu_s_input_batch=False, transform_h_after_layernorm=False, prepend_history_seq=True) -> Dict:

    eval_output_dir = args.output_dir

    cocon_output_file_path = os.path.join(args.output_dir, args.cocon_output_filename)
    if os.path.exists(cocon_output_file_path):
        if args.append_cocon_output_files:
            logger.info("Append to existing cocon output file")
        else:
            logger.info("Removing existing cocon output file")
            os.remove(cocon_output_file_path)
    else:
        logger.info("Creating new cocon output file")

    if args.cocon_output_jsonl_filename is not None:
        cocon_output_jsonl_file_path = os.path.join(args.output_dir, args.cocon_output_jsonl_filename)
        if os.path.exists(cocon_output_jsonl_file_path):
            if args.append_cocon_output_files:
                logger.info("Append to existing cocon output jsonl file")
            else:
                logger.info("Removing existing cocon output jsonl file")
                os.remove(cocon_output_jsonl_file_path)
        else:
            logger.info("Creating new cocon output jsonl file")
    else:
        cocon_output_jsonl_file_path = None

    if args.line_by_line_hs:
        history_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_history_source_data_file, generate=True, line_by_line=True, prepend_bos_token=args.prepend_bos_token_to_line)
    else:
        history_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_history_source_data_file, generate=True)

    if args.line_by_line_cs:
        context_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_context_source_data_file, generate=True, line_by_line=True)
    else:
        context_source_dataset = load_and_cache_examples(args, tokenizer, file_path=args.cocon_compute_context_source_data_file, generate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    if random_sample_data == True:
        history_source_sampler = RandomSampler(history_source_dataset) if args.local_rank == -1 else DistributedSampler(history_source_dataset)
        context_source_sampler = RandomSampler(context_source_dataset) if args.local_rank == -1 else DistributedSampler(context_source_dataset)
    else:
        history_source_sampler = SequentialSampler(history_source_dataset)
        context_source_sampler = SequentialSampler(context_source_dataset)

    history_source_dataloader = DataLoader(
        history_source_dataset, sampler=history_source_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    context_source_dataloader = DataLoader(
        context_source_dataset, sampler=context_source_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )
    context_source_dataloader_iter = iter(context_source_dataloader)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Generate cocon samples!
    logger.info("***** Running cocon generation {} *****".format(prefix))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_generate_cocon_steps = 0
    model.eval()
    cocon_block.eval()

    if use_only_first_context_source_batch and args.use_history_source_as_context_source_for_gen == False:
        context_source_batch = next(context_source_dataloader_iter)
        context_source_inputs = context_source_batch

    for batch_ind, batch in enumerate(tqdm(history_source_dataloader, desc="Generating")):
        inputs = batch

        if args.use_history_source_as_context_source_for_gen:
            history_source_inputs = inputs[:(inputs.shape[0] // 2)]
            context_source_inputs = inputs[(inputs.shape[0] // 2):]
            inputs = history_source_inputs
        inputs = inputs.to(args.device)


        if args.line_by_line_hs:
            original_history_seq = inputs
            original_context_seq = None
        else:
            original_history_seq = inputs[:, :args.gen_hs_len]
            original_context_seq = inputs[:, args.gen_hs_len:args.gen_hs_len+args.gen_cs_len]

        if use_only_first_context_source_batch == False and args.use_history_source_as_context_source_for_gen == False:
            if args.enumerate_all_cs_for_each_hs:
                for context_batch_ind, context_source_batch in enumerate(tqdm(context_source_dataloader, desc="Enumerating context source")):
                    context_source_inputs = context_source_batch

                    if args.line_by_line_cs:
                        context_seq = context_source_inputs
                    else:
                        context_seq = context_source_inputs[:, args.gen_hs_len:args.gen_hs_len+args.gen_cs_len]
                    context_seq = context_seq.to(args.device)

                    with open(cocon_output_file_path, "a", encoding='utf-8') as f:
                        f.writelines("***HS #{},    CS #{}***\n".format(batch_ind, context_batch_ind))
                    generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block,
                        cocon_output_jsonl_file_path=cocon_output_jsonl_file_path, transform_h_after_layernorm=transform_h_after_layernorm, prepend_history_seq=prepend_history_seq)

                # go to next hs
                continue

            else:
                context_source_batch = next(context_source_dataloader_iter)
                context_source_inputs = context_source_batch

        if args.line_by_line_cs:
            context_seq = context_source_inputs
        else:
            context_seq = context_source_inputs[:, args.gen_hs_len:args.gen_hs_len+args.gen_cs_len]
        context_seq = context_seq.to(args.device)

        with open(cocon_output_file_path, "a", encoding='utf-8') as f:
            f.writelines("***HS #{}***\n".format(batch_ind))

        generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block,
            cocon_output_jsonl_file_path=cocon_output_jsonl_file_path, transform_h_after_layernorm=transform_h_after_layernorm, prepend_history_seq=prepend_history_seq)

        if nb_generate_cocon_steps >= args.num_cocon_generate - 1:
            break

        nb_generate_cocon_steps += 1

    return nb_generate_cocon_steps


def generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block,
    cocon_output_jsonl_file_path=None, transform_h_after_layernorm=False, prepend_history_seq=True,
    original_dia_history_seq=None, dia_context_seq=None, original_dia_context_seq=None, end_of_text_id=None, single_generation=False, do_cocon_wgpt2genas2ndcs=True, wgpt2genas2ndcs_double_context_len=30,
    cocon_wgpt2genas2ndcs_cs_attn_biases=[1, 2, 5, 10], cocon_wgpt2genas2ndcs_gpt2out_attn_biases=[-1, -2, -5, -10]):

    with torch.no_grad():
        encoded_prompt = inputs[:, 0:0]

        # Cocon generation with context_seq as cs
        cocon_gen_ar_output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.generate_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
            do_cocon=True,
            cocon_block=cocon_block,
            cocon_context_inputs=context_seq,
            cocon_history_inputs=original_history_seq,
            cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
            transform_h_after_layernorm=transform_h_after_layernorm,
            use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
        )
        if prepend_history_seq:
            cocon_gen_ar_output_sequences = torch.cat([original_history_seq, cocon_gen_ar_output_sequences], dim=1)
        # Remove the batch dimension when returning multiple sequences
        if len(cocon_gen_ar_output_sequences.shape) > 2:
            cocon_gen_ar_output_sequences.squeeze_()

        if args.context_attn_bias != 0:
            # Cocon generation with context_seq as cs, with context_attn_bias
            cocon_gen_conbias_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=context_seq,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                context_attn_bias=args.context_attn_bias
            )
            if prepend_history_seq:
                cocon_gen_conbias_ar_output_sequences = torch.cat([original_history_seq, cocon_gen_conbias_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_gen_conbias_ar_output_sequences.shape) > 2:
                cocon_gen_conbias_ar_output_sequences.squeeze_()

        # Cocon generation with original_context_seq as cs
        if args.line_by_line_hs == False and original_context_seq is not None:
            self_cocon_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=original_context_seq,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
            )
            if prepend_history_seq:
                self_cocon_gen_ar_output_sequences = torch.cat([original_history_seq, self_cocon_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(self_cocon_gen_ar_output_sequences.shape) > 2:
                self_cocon_gen_ar_output_sequences.squeeze_()

        # Prepend Context GPT-2 generation
        # Sanity check: autoregressive text generation with context_seq prepended on the prompt
        encoded_prompt = torch.cat([context_seq, original_history_seq], dim=1)
        prependgpt2_gen_ar_output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.generate_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )


        # Original GPT-2 generation
        # Sanity check: autoregressive text generation
        encoded_prompt = original_history_seq
        gen_ar_output_sequences = model.generate(
            input_ids=encoded_prompt,
            max_length=args.generate_length + len(encoded_prompt[0]),
            temperature=args.temperature,
            top_k=args.k,
            top_p=args.p,
            repetition_penalty=args.repetition_penalty,
            do_sample=True,
            num_return_sequences=args.num_return_sequences,
        )
        # Remove the batch dimension when returning multiple sequences
        if len(gen_ar_output_sequences.shape) > 2:
            gen_ar_output_sequences.squeeze_()

        # Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence
        if do_cocon_wgpt2genas2ndcs:
            gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
            cocon_wgpt2genas2ndcs_context_input = [context_seq, gpt2_gen_output]
            cocon_wgpt2genas2ndcs_context_attn_bias = [0, 0]
            encoded_prompt = inputs[:, 0:0]
            cocon_wgpt2genas2ndcs_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=cocon_wgpt2genas2ndcs_context_input,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
            )
            if prepend_history_seq:
                cocon_wgpt2genas2ndcs_gen_ar_output_sequences = torch.cat([original_history_seq, cocon_wgpt2genas2ndcs_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_wgpt2genas2ndcs_gen_ar_output_sequences.shape) > 2:
                cocon_wgpt2genas2ndcs_gen_ar_output_sequences.squeeze_()


            # Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence, with double context generation cut off: cocon with one context after wgpt2genas2ndcs_double_context_len for better generation quality
            gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
            cocon_wgpt2genas2ndcs_2parts1st_context_input = [context_seq, gpt2_gen_output]
            encoded_prompt = inputs[:, 0:0]
            # Part 1 generation
            cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=cocon_wgpt2genas2ndcs_2parts1st_context_input,
                cocon_history_inputs=original_history_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
            )
            if prepend_history_seq:
                cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = torch.cat([original_history_seq, cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences.shape) > 2:
                cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences.squeeze_()

            encoded_prompt = inputs[:, 0:0]
            # Part 2 generation: with only original context_seq as context input
            cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = model.generate(
                input_ids=encoded_prompt,
                max_length=args.generate_length - wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=context_seq,
                cocon_history_inputs=cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                transform_h_after_layernorm=transform_h_after_layernorm,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
            )
            if prepend_history_seq:
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = torch.cat([cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences, cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences], dim=1)
            # Remove the batch dimension when returning multiple sequences
            if len(cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.shape) > 2:
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.squeeze_()

            # With varying cs context_attn_bias values, Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence, with double context generation cut off: cocon with one context after wgpt2genas2ndcs_double_context_len
            cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list = []
            for cs_attn_bias in cocon_wgpt2genas2ndcs_cs_attn_biases:
                cocon_wgpt2genas2ndcs_context_attn_bias = [cs_attn_bias, 0]

                gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
                cocon_wgpt2genas2ndcs_2parts1st_context_input = [context_seq, gpt2_gen_output]
                encoded_prompt = inputs[:, 0:0]
                # Part 1 generation
                cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=cocon_wgpt2genas2ndcs_2parts1st_context_input,
                    cocon_history_inputs=original_history_seq,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                    context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
                )
                if prepend_history_seq:
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = torch.cat([original_history_seq, cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences], dim=1)

                encoded_prompt = inputs[:, 0:0]
                # Part 2 generation: with only original context_seq as context input
                cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.generate_length - wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=context_seq,
                    cocon_history_inputs=cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                )
                if prepend_history_seq:
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = torch.cat([cs_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences, cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences], dim=1)
                # Remove the batch dimension when returning multiple sequences
                if len(cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.shape) > 2:
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.squeeze_()
                cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list.append(cs_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences)


            # With varying gpt2 output context_attn_bias values, Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence, with double context generation cut off: cocon with one context after wgpt2genas2ndcs_double_context_len
            gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list = []
            for gpt2out_attn_bias in cocon_wgpt2genas2ndcs_gpt2out_attn_biases:
                cocon_wgpt2genas2ndcs_context_attn_bias = [0, gpt2out_attn_bias]

                gpt2_gen_output = gen_ar_output_sequences[ :, len(original_history_seq[0]): ]
                cocon_wgpt2genas2ndcs_2parts1st_context_input = [context_seq, gpt2_gen_output]
                encoded_prompt = inputs[:, 0:0]
                # Part 1 generation
                gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=cocon_wgpt2genas2ndcs_2parts1st_context_input,
                    cocon_history_inputs=original_history_seq,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                    context_attn_bias=cocon_wgpt2genas2ndcs_context_attn_bias,
                )
                if prepend_history_seq:
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences = torch.cat([original_history_seq, gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences], dim=1)

                encoded_prompt = inputs[:, 0:0]
                # Part 2 generation: with only original context_seq as context input
                gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = model.generate(
                    input_ids=encoded_prompt,
                    max_length=args.generate_length - wgpt2genas2ndcs_double_context_len + len(encoded_prompt[0]),
                    temperature=args.temperature,
                    top_k=args.k,
                    top_p=args.p,
                    repetition_penalty=args.repetition_penalty,
                    do_sample=True,
                    num_return_sequences=args.num_return_sequences,
                    do_cocon=True,
                    cocon_block=cocon_block,
                    cocon_context_inputs=context_seq,
                    cocon_history_inputs=gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences,
                    cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                    transform_h_after_layernorm=transform_h_after_layernorm,
                    use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                )
                if prepend_history_seq:
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences = torch.cat([gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts1st_gen_ar_output_sequences, gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences], dim=1)
                # Remove the batch dimension when returning multiple sequences
                if len(gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.shape) > 2:
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences.squeeze_()
                gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list.append(gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences)


        cocon_output_text_lines_dict = {}
        for generated_sequence_idx, generated_sequence in enumerate(cocon_gen_ar_output_sequences):
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict = {}
            # Decode and log original_input_text
            original_input_sequence = inputs[generated_sequence_idx]
            original_input_sequence = original_input_sequence.tolist()
            original_input_text = tokenizer.decode(original_input_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx] = ["original_input_text: {} \n".format(original_input_text)]
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["original_input_text"] = original_input_text

            # Decode and log original_history_seq
            original_history_sequence = original_history_seq[generated_sequence_idx]
            original_history_sequence = original_history_sequence.tolist()
            original_history_text = tokenizer.decode(original_history_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("original_history_text: {} \n".format(original_history_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["original_history_text"] = original_history_text

            # Decode and log context_seq
            if type(context_seq) == list:
                context_seq = torch.cat(context_seq, dim=1)

            context_sequence = context_seq[generated_sequence_idx]
            context_sequence = context_sequence.tolist()

            context_text = tokenizer.decode(context_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("context_text: {} \n".format(context_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["context_text"] = context_text

            # Decode and log original_context_seq
            if args.line_by_line_hs == False and original_context_seq is not None:
                original_context_sequence = original_context_seq[generated_sequence_idx]
                original_context_sequence = original_context_sequence.tolist()
                original_context_text = tokenizer.decode(original_context_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("original_context_text: {} \n".format(original_context_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["original_context_text"] = original_context_text
            else:
                cocon_output_text_lines_dict[generated_sequence_idx].append("original_context_text: None \n")

            # Decode and log cocon AR generated text
            cocon_gen_ar_output_sequence = cocon_gen_ar_output_sequences[generated_sequence_idx]
            cocon_gen_ar_output_sequence = cocon_gen_ar_output_sequence.tolist()
            cocon_gen_ar_output_text = tokenizer.decode(cocon_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon AR output: {} \n".format(cocon_gen_ar_output_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["cocon_output"] = cocon_gen_ar_output_text


            if args.context_attn_bias != 0:
                # Decode and log cocon AR generated text, with context_attn_bias
                cocon_gen_conbias_ar_output_sequence = cocon_gen_conbias_ar_output_sequences[generated_sequence_idx]
                cocon_gen_conbias_ar_output_sequence = cocon_gen_conbias_ar_output_sequence.tolist()
                cocon_gen_conbias_ar_output_text = tokenizer.decode(cocon_gen_conbias_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon AR output, context_attn_bias {}: {} \n".format(args.context_attn_bias, cocon_gen_conbias_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["cocon_conbias_output"] = cocon_gen_conbias_ar_output_text
                    cocon_jsonl_output_dict["context_attn_bias"] = args.context_attn_bias


            # Decode and log self cocon AR generated text
            if args.line_by_line_hs == False and original_context_seq is not None:
                self_cocon_gen_ar_output_sequence = self_cocon_gen_ar_output_sequences[generated_sequence_idx]
                self_cocon_gen_ar_output_sequence = self_cocon_gen_ar_output_sequence.tolist()
                self_cocon_gen_ar_output_text = tokenizer.decode(self_cocon_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("(Self) Cocon AR output: {} \n".format(self_cocon_gen_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["self_cocon_output"] = self_cocon_gen_ar_output_text


            # Sanity check (SC) prependgpt2_gen_ar_output_sequences: Decode and log AR generated text
            prependgpt2_gen_ar_output_sequence = prependgpt2_gen_ar_output_sequences[generated_sequence_idx]
            prependgpt2_gen_ar_output_sequence = prependgpt2_gen_ar_output_sequence.tolist()
            prependgpt2_gen_output_text = tokenizer.decode(prependgpt2_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("SC prependgpt2 Autoreg-generated output: {} \n".format(prependgpt2_gen_output_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["prependgpt2_ar_gen"] = prependgpt2_gen_output_text


            # Sanity check (SC): Decode and log AR generated text
            gen_ar_output_sequence = gen_ar_output_sequences[generated_sequence_idx]
            gen_ar_output_sequence = gen_ar_output_sequence.tolist()
            gen_output_text = tokenizer.decode(gen_ar_output_sequence, clean_up_tokenization_spaces=True)
            cocon_output_text_lines_dict[generated_sequence_idx].append("SC Autoreg-generated output: {} \n".format(gen_output_text))
            if cocon_output_jsonl_file_path is not None:
                cocon_jsonl_output_dict["sc_gpt2_ar_gen"] = gen_output_text


            # Cocon generation (wgpt2genas2ndcs): with gpt2 generations as 2nd context sequence
            if do_cocon_wgpt2genas2ndcs:
                cocon_wgpt2genas2ndcs_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_gen_ar_output_sequences[generated_sequence_idx]
                cocon_wgpt2genas2ndcs_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_gen_ar_output_sequence.tolist()
                cocon_wgpt2genas2ndcs_gen_ar_output_text = tokenizer.decode(cocon_wgpt2genas2ndcs_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs AR output: {} \n".format(cocon_wgpt2genas2ndcs_gen_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_output"] = cocon_wgpt2genas2ndcs_gen_ar_output_text

                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequences[generated_sequence_idx]
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence = cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence.tolist()
                cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_text = tokenizer.decode(cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs (2 parts) AR output: {} \n".format(cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_text))
                if cocon_output_jsonl_file_path is not None:
                    cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_2parts_output"] = cocon_wgpt2genas2ndcs_2parts2nd_gen_ar_output_text


                for bias_ind, cs_attn_bias in enumerate(cocon_wgpt2genas2ndcs_cs_attn_biases):
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list[bias_ind]
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences[generated_sequence_idx]
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence.tolist()
                    cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text = tokenizer.decode(cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                    cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs (2 parts) AR output, cs_attn_bias {}: {} \n".format(cs_attn_bias, cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text))
                    if cocon_output_jsonl_file_path is not None:
                        cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_2parts_output_cs_attn_bias{}".format(cs_attn_bias)] = cs_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text

                for bias_ind, gpt2out_attn_bias in enumerate(cocon_wgpt2genas2ndcs_gpt2out_attn_biases):
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences_list[bias_ind]
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequences[generated_sequence_idx]
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence.tolist()
                    gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text = tokenizer.decode(gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_sequence, clean_up_tokenization_spaces=True)
                    cocon_output_text_lines_dict[generated_sequence_idx].append("Cocon wgpt2genas2ndcs (2 parts) AR output, gpt2out_attn_bias {}: {} \n".format(gpt2out_attn_bias, gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text))
                    if cocon_output_jsonl_file_path is not None:
                        cocon_jsonl_output_dict["cocon_wgpt2genas2ndcs_2parts_output_gpt2out_attn_bias{}".format(gpt2out_attn_bias)] = gpt2out_attn_biased_cocon_wgpt2genas2ndcs_2parts_gen_ar_output_text

            if cocon_output_jsonl_file_path is not None:
                with open(cocon_output_jsonl_file_path, "a") as f:
                    json.dump(cocon_jsonl_output_dict, f)
                    f.write('\n')

    cocon_output_text_lines = []
    for sample_ind in range(inputs.shape[0]):
        cocon_output_text_lines = cocon_output_text_lines + cocon_output_text_lines_dict[sample_ind] + ["----------\n"]

    with open(cocon_output_file_path, "a", encoding='utf-8') as f:
        f.writelines(cocon_output_text_lines)

    if args.context_attn_bias != 0:
        return cocon_gen_conbias_ar_output_text
    else:
        return cocon_gen_ar_output_text


# Use to generate cocon-edited text with either trained or simple cocon op
def generate_single_cocon_example(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, cocon_block=None, prefix="",
    random_sample_data=False, use_only_first_context_source_batch=False, use_only_first_custom_mu_s_input_batch=False, transform_h_after_layernorm=False, prepend_history_seq=True) -> Dict:

    eval_output_dir = args.output_dir

    cocon_output_file_path = os.path.join(args.output_dir, args.cocon_output_filename)
    if os.path.exists(cocon_output_file_path):
        if args.append_cocon_output_files:
            logger.info("Append to existing cocon output file")
        else:
            logger.info("Removing existing cocon output file")
            os.remove(cocon_output_file_path)
    else:
        logger.info("Creating new cocon output file")

    if args.cocon_output_jsonl_filename is not None:
        cocon_output_jsonl_file_path = os.path.join(args.output_dir, args.cocon_output_jsonl_filename)
        if os.path.exists(cocon_output_jsonl_file_path):
            if args.append_cocon_output_files:
                logger.info("Append to existing cocon output jsonl file")
            else:
                logger.info("Removing existing cocon output jsonl file")
                os.remove(cocon_output_jsonl_file_path)
        else:
            logger.info("Creating new cocon output jsonl file")
    else:
        cocon_output_jsonl_file_path = None

    prompt_text = args.prompt if args.prompt else input("Model prompt >>> ")
    if args.prepend_bos_token_to_line:
        prompt_text = tokenizer.bos_token + prompt_text
    logger.info("prompt_text: {}".format(prompt_text))

    encoded_prompt = tokenizer.encode(prompt_text, add_special_tokens=False, return_tensors="pt")
    original_history_seq = encoded_prompt.to(args.device)

    content_input_text = args.content_input if args.content_input else input("Content input >>> ")
    logger.info("content_input: {}".format(content_input_text))

    if args.content_input_delimit is not None and args.content_input_delimit in content_input_text:
        content_input_texts = content_input_text.split(args.content_input_delimit)
        logger.info("content_input_texts: {}".format(content_input_texts))
        context_seq = []
        for content_input_text in content_input_texts:
            encoded_content_input = tokenizer.encode(content_input_text, add_special_tokens=False, return_tensors="pt")
            context_seq.append(encoded_content_input.to(args.device))
    else:
        encoded_content_input = tokenizer.encode(content_input_text, add_special_tokens=False, return_tensors="pt")
        context_seq = encoded_content_input.to(args.device)

    original_context_seq = None
    inputs = original_history_seq

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Generate cocon samples!
    logger.info("***** Running single cocon generation {} *****".format(prefix))

    model.eval()
    cocon_block.eval()

    inputs = encoded_prompt
    inputs = inputs.to(args.device)

    cocon_output_text = generate_cocon_sample(context_seq, original_history_seq, original_context_seq, inputs, cocon_output_file_path, args, model, tokenizer, cocon_block,
        cocon_output_jsonl_file_path=cocon_output_jsonl_file_path, transform_h_after_layernorm=transform_h_after_layernorm, prepend_history_seq=prepend_history_seq, single_generation=True)

    logger.info("cocon_output_text: {} *****".format(cocon_output_text))

    return cocon_output_text
