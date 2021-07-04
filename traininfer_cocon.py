# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import glob
import logging
import os
import shutil
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    CoconBlock,
    HDiscriminator,
)

from dataset import load_and_cache_examples
from args import get_args
from training import train_cocon, train_lm, evaluate
from generate import generate_cocon_compute, generate_single_cocon_example
from utils.utils import set_seed, _sorted_checkpoints


from collections import OrderedDict

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
}


def num_correct(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def count_ngram(text_samples, n, tokenizer=None):
    """
    Count the number of unique n-grams
    :param text_samples: list, a list of samples
    :param n: int, n-gram
    :return: the number of unique n-grams in text_samples
    """
    if len(text_samples) == 0:
        print("ERROR, eval_distinct get empty input")
        return

    ngram = set()
    for sample in text_samples:
        if len(sample) < n:
            continue

        sample = list(map(str, sample))
        for i in range(len(sample) - n + 1):
            ng = ' '.join(sample[i: i + n])

            ngram.add(' '.join(ng))
    return len(ngram)

# evaluate Dist-K scores
def evaluate_dist_scores(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
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

    dist_eval_samples = []
    num_tokens = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        sample_flattened = batch.reshape(-1)
        dist_eval_samples.append(sample_flattened.tolist())
        num_tokens += len(sample_flattened)

        nb_eval_steps += 1

        if nb_eval_steps == args.dist_eval_max_samples:
            logger.info("breaking iteration @ sample # {}".format(nb_eval_steps))
            break

    dist1_score = count_ngram(dist_eval_samples, 1) / float(num_tokens)
    dist2_score = count_ngram(dist_eval_samples, 2) / float(num_tokens)
    dist3_score = count_ngram(dist_eval_samples, 3) / float(num_tokens)

    result = {"Dist-1": dist1_score, "Dist-2": dist2_score, "Dist-3": dist3_score}

    output_filename = "distK_" + args.eval_output_filename
    output_eval_file = os.path.join(eval_output_dir, prefix, output_filename)
    with open(output_eval_file, "w") as writer:
        logger.info("***** Dist-1,2,3 Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result

def fix_state_dict_naming(state_dict):
    new_state_dict = OrderedDict()

    for key, value in state_dict.items():
        if 'con2' in key:
            new_key = key.replace('con2', 'cocon')
        # new_key = key_transformation(key)
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    return new_state_dict


def main():
    args = get_args()

    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "Process device: %s, n_gpu: %s",
        device,
        args.n_gpu,
    )

    # Set seed
    set_seed(args)

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        logger.info("Loading tokenizer from pretrained, {}".format(args.model_name_or_path))
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        if args.output_meanvars and ('gpt2' in args.model_name_or_path):
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
                output_meanvars=True,
                compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
            )
        else:
            logger.info("Loading model from pretrained weights, {}".format(args.model_name_or_path))
            model = model_class.from_pretrained(
                args.model_name_or_path,
                from_tf=bool(".ckpt" in args.model_name_or_path),
                config=config,
                cache_dir=args.cache_dir,
            )
    else:
        logger.info("Training new model from scratch")
        if args.output_meanvars:
            model = model_class(config=config, output_meanvars=True, compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm)
        else:
            model = model_class(config=config)

    model.to(args.device)

    if args.only_lm == False:
        # Set up CoconBlock
        cocon_block = CoconBlock(config.n_ctx, config, scale=True)
        cocon_block.to(args.device)

        if args.lambda_adv > 0:
            # Set up disc_model model
            disc_model = HDiscriminator(config=config)
            disc_model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        if args.num_lm_train_epochs > 0:
            global_step, tr_loss = train_lm(args, train_dataset, model, tokenizer)

        if args.only_lm == False:
            if args.lambda_adv > 0:
                global_step, tr_loss = train_cocon(args, train_dataset, model, tokenizer, cocon_block=cocon_block, disc_model=disc_model, model_config=config, transform_h_after_layernorm=args.transform_h_after_layernorm)
            else:
                global_step, tr_loss = train_cocon(args, train_dataset, model, tokenizer, cocon_block=cocon_block, model_config=config, transform_h_after_layernorm=args.transform_h_after_layernorm)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train:
        # Create output directory if needed
        os.makedirs(args.output_dir, exist_ok=True)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        if args.num_lm_train_epochs > 0 or args.save_lm_model:
            # Save a trained model, configuration and tokenizer using `save_pretrained()`.
            # They can then be reloaded using `from_pretrained()`
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)

            # Load a trained model and vocabulary that you have fine-tuned
            model = model_class.from_pretrained(args.output_dir)
            tokenizer = tokenizer_class.from_pretrained(args.output_dir)
            model.to(args.device)

        if args.only_lm == False:
            # Save cocon_block model
            cocon_block_weights_name = "cocon_block_pytorch_model.bin"
            output_cocon_block_model_file = os.path.join(args.output_dir, cocon_block_weights_name)
            torch.save(cocon_block.state_dict(), output_cocon_block_model_file)
            logger.info("cocon_block model weights saved in {}".format(output_cocon_block_model_file))

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # Compute cocon text: generate cocon text
    results = {}
    if args.do_cocon_compute and args.only_lm == False:
        if args.gen_cs_len is None:
            args.gen_cs_len = args.cs_len
        if args.gen_hs_len is None:
            args.gen_hs_len = args.hs_len
        if args.gen_tis_len is None:
            args.gen_tis_len = args.tis_len

        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""

            # Load cocon_block model
            cocon_block_weights_name = "cocon_block_pytorch_model.bin"
            output_cocon_block_model_file = os.path.join(args.output_dir, cocon_block_weights_name)

            cocon_state_dict = torch.load(output_cocon_block_model_file)
            new_cocon_state_dict = fix_state_dict_naming(cocon_state_dict)
            cocon_block.load_state_dict(new_cocon_state_dict)

            model.to(args.device)
            cocon_block.to(args.device)

            generate_steps = generate_cocon_compute(args, model, tokenizer, cocon_block=cocon_block, prefix=prefix, use_only_first_context_source_batch=args.use_only_first_context_source_batch, transform_h_after_layernorm=args.transform_h_after_layernorm)

    # Single cocon generation: generate single cocon text
    results = {}
    if args.do_single_cocon_generation and args.only_lm == False:
        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""

            # Load cocon_block model
            cocon_block_weights_name = "cocon_block_pytorch_model.bin"
            output_cocon_block_model_file = os.path.join(args.output_dir, cocon_block_weights_name)

            cocon_block.load_state_dict(torch.load(output_cocon_block_model_file), strict=False) # to deal with earlier cocon weights without h_mask and self_token_mask
            model.to(args.device)
            cocon_block.to(args.device)

            generate_steps = generate_single_cocon_example(args, model, tokenizer, cocon_block=cocon_block, prefix=prefix, transform_h_after_layernorm=args.transform_h_after_layernorm)

    # Evaluation: evaluate model on loss values
    results = {}
    if args.do_eval:
        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

     # Evaluation: evaluate model on loss values
    if args.do_eval_dist:
        if not args.eval_compute_without_checkpoint:
            checkpoints = [args.output_dir]
        else:
            checkpoints = ["pretrained"]
        if args.eval_all_checkpoints:
            checkpoints = list(
                os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
            )
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            if not args.eval_compute_without_checkpoint:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""

                if args.output_meanvars:
                    model = model_class.from_pretrained(
                        checkpoint,
                        output_meanvars=True,
                        compute_meanvars_before_layernorm=args.compute_meanvars_before_layernorm
                    )
                else:
                    model = model_class.from_pretrained(checkpoint)
            else:
                global_step = 0
                prefix = ""
            model.to(args.device)
            result = evaluate_dist_scores(args, model, tokenizer, prefix=prefix)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
