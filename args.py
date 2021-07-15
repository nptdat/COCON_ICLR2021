import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Custom
    parser.add_argument(
        "--config_class", default=None, type=str, help="Config class. This parameter has higher priority than model_type in identifying class of the config object.",
    )
    parser.add_argument(
        "--tokenizer_class", default=None, type=str, help="Tokenizer class. This parameter has higher priority than model_type in identifying class of the tokenizer object.",
    )
    parser.add_argument(
        "--model_class", default=None, type=str, help="Model class. This parameter has higher priority than model_type in identifying class of the model object.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )

    parser.add_argument(
        "--cs_len",
        default=20,
        type=int,
        help="Context sequence length."
    )
    parser.add_argument(
        "--hs_len",
        default=10,
        type=int,
        help="History sequence length."
    )
    parser.add_argument(
        "--tis_len",
        default=20,
        type=int,
        help="Transformation input sequence length."
    )

    parser.add_argument(
        "--gen_cs_len",
        default=None,
        type=int,
        help="Context sequence length for generation."
    )
    parser.add_argument(
        "--gen_hs_len",
        default=None,
        type=int,
        help="History sequence length for generation."
    )
    parser.add_argument(
        "--gen_tis_len",
        default=None,
        type=int,
        help="Transformation input sequence length for generation."
    )

    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

    parser.add_argument("--per_gpu_train_batch_size", default=1, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_lm_batch_size", default=0, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument(
        "--per_gpu_eval_batch_size", default=1, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )

    parser.add_argument(
        "--epoch_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps per epoch to perform. Override num_train_epochs.",
    )


    parser.add_argument(
        "--num_lm_train_epochs", default=0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--lm_max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save checkpoint every X updates steps.")
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")


    parser.add_argument(
        "--compute_meanvars_before_layernorm",
        action="store_true",
        help="Whether to compute mean and var before layernorm",
    )
    parser.add_argument(
        "--output_meanvars",
        action="store_true",
        default=True,
        help="Whether to output hidden states' mean and var values across channels",
    )
    parser.add_argument(
        "--start_sample_ind",
        type=int,
        default=0,
        help="Index to start computing hidden state stats.",
    )
    parser.add_argument(
        "--num_meanvar_compute",
        type=int,
        default=99999999,
        help="Number of data samples to compute meanvars.",
    )
    parser.add_argument(
        "--meanvar_output_filename",
        type=str,
        default='mean_var.npy',
        help="The output file to save data sample mean/var values.",
    )

    parser.add_argument(
        "--compute_meanvars_random_sample",
        action="store_true",
        help="Whether to sample randomly while computing mean/var.",
    )

    parser.add_argument(
        "--eval_output_filename",
        type=str,
        default="eval_results.txt",
        help="The output file to save eval results.",
    )

    parser.add_argument(
        "--cocon_output_filename",
        type=str,
        default="cocon_output.txt",
        help="The output file to save cocon generated text.",
    )

    parser.add_argument(
        "--cocon_output_jsonl_filename",
        type=str,
        default="cocon_output.jsonl",
        help="The output jsonl file to save cocon generated text.",
    )

    parser.add_argument(
        "--w_mapper_num_layers", type=int, default=5, help="Number of fc layers for mapping z into w"
    )
    parser.add_argument(
        "--w_mapper_dropout_prob", type=float, default=None, help="Dropout probability for z to w mapper fc layers"
    )
    parser.add_argument(
        "--gen_update_interval", type=int, default=1, help="Number of lm steps before a gen update step"
    )
    parser.add_argument(
        "--disc_update_interval", type=int, default=1, help="Number of lm steps before a disc update step"
    )

    parser.add_argument(
        "--step_ind_to_start_cycle_ar_cocon_recon", type=int, default=0, help="Step number to start own_style_cocon learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_cycle_ar_cocon_recon", type=int, default=0, help="Training epoch number to start own_style_cocon learning, 0 is the first epoch"
    )

    parser.add_argument(
        "--step_ind_to_start_other_context_cocon", type=int, default=0, help="Step number to start own_style_cocon learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_other_context_cocon", type=int, default=0, help="Training epoch number to start own_style_cocon learning, 0 is the first epoch"
    )

    parser.add_argument(
        "--step_ind_to_start_adv", type=int, default=0, help="LM step number to start adversarial learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_adv", type=int, default=0, help="Training epoch number to start adversarial learning, 0 is the first epoch"
    )

    parser.add_argument(
        "--step_ind_to_start_hist_cocon_lm", type=int, default=0, help="Step number to start hist_cocon_lm learning, 0 is the first step"
    )
    parser.add_argument(
        "--epoch_ind_to_start_hist_cocon_lm", type=int, default=0, help="Training epoch number to start hist_cocon_lm learning, 0 is the first epoch"
    )


    parser.add_argument(
        "--steps_to_closely_monitor_adv", type=int, default=10000, help="Training epoch number to start adversarial learning, 0 is the first epoch"
    )

    parser.add_argument("--no_adv_gen_train_original_lm", action="store_true", help="Whether to backprop loss through original LM feed forward.")
    parser.add_argument("--use_original_text_as_disc_real", action="store_true", help="Whether to use original text as real input of the disc.")
    parser.add_argument("--gen_gumbel_softmax", action="store_true", help="Whether to use gumbel softmax for computing probs from gen output logits.")

    parser.add_argument(
        "--latent_mixing_prob", type=float, default=0, help="Probability of mixing 2 generated_cocon_vector during cocon generation"
    )

    parser.add_argument('--block_indices',
        nargs='+',
        type=int,
        default=None,
        help="0,1,2,3,4,5,6,7,8,9,10,11,12 where 12 is the final FF layer without self-attn",
    )
    parser.add_argument('--layer_indices',
        nargs='+',
        type=int,
        default=None,
        help="0,1 where 0 is the hidden state stats before self-attn layer and 1 is stats before FF layer",
    )
    parser.add_argument('--stat_indices',
        nargs='+',
        type=int,
        default=None,
        help="0,1 where 0 is mean and 1 is var",
    )

    parser.add_argument("--do_cocon_compute", action="store_true", help="Whether to generate text with cocon.")

    parser.add_argument("--eval_compute_without_checkpoint", action="store_true", help="Whether to use saved checkpoint or use pretrained LM to evaluate and compute stats.")

    parser.add_argument(
        "--distance_metric",
        type=str,
        default="l2",
        help="Distance metric used to compute loss value for hidden values, can be l2 (l2 distance) or cos (cosine similarity).",
    )

    parser.add_argument(
        "--stats_distance_metric",
        type=str,
        default=None,
        help="Distance metric used to compute loss value, can be 'cos' (cosine similarity), 'normalized_l2' (l2 distance between normalized vectors) or 'l2' (l2 distance between unnormalized vectors).",
    )

    parser.add_argument(
        "--lambda_self_cocon_lm_loss", type=float, default=1, help="Lambda value of self_cocon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_hist_cocon_lm_loss", type=float, default=0, help="Lambda value of hist_cocon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_cycle_ar_cocon_recon_lm_loss", type=float, default=0, help="Lambda value of cycle_ar_cocon_recon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_other_context_cocon_lm_loss", type=float, default=0, help="Lambda value of other_context_cocon_lm_loss optimization"
    )
    parser.add_argument(
        "--lambda_adv", type=float, default=0, help="Lambda value of adversarial loss optimization"
    )

    parser.add_argument("--per_gpu_train_cycle_ar_cocon_recon_batch_size", default=None, type=int, help="Batch size per GPU/CPU for training when cycle_recon training starts.")

    parser.add_argument("--adv_use_th_gen_output", action="store_true", help="Whether to use cocon > GPT2 tail-head output as fake examples.")

    parser.add_argument("--track_loss_gradnorms", action="store_true", help="Whether to log all loss gradnorm to tb.")

    parser.add_argument("--save_lm_model", action="store_true", help="Whether to save (GPT-2) lm model.")
    parser.add_argument("--only_lm", action="store_true", help="Whether to train and infer only lm model, without cocon.")

    parser.add_argument(
        "--cocon_compute_history_source_data_file",
        type=str,
        default="data/gpt2output/webtext.valid.jsonl",
        help="The file for content source data.",
    )

    parser.add_argument(
        "--cocon_compute_context_source_data_file",
        type=str,
        default="data/gpt2output/webtext.test.jsonl",
        help="The file for content source data.",
    )

    parser.add_argument(
        "--num_cocon_generate",
        type=int,
        default=99999999,
        help="Number of cocon samples to generate.",
    )

    parser.add_argument(
        "--output_hidden_for_cocon_after_block_ind", type=int, default=6, help="Block index to output hidden state for cocon computation"
    )

    parser.add_argument("--transform_h_after_layernorm", action="store_true", help="Whether to do cocon after layer norm op, generated text results are poorer in this setting.")

    parser.add_argument("--use_only_first_context_source_batch", action="store_true", help="Whether to use only the first style source batch for cocon.")

    parser.add_argument("--use_token_gate", action="store_true", help="Whether to use token gate for cocon_block.")
    parser.add_argument("--use_global_gate", action="store_true", help="Whether to use global sequence gate for cocon_block.")
    parser.add_argument("--split_c_proj", action="store_true", help="Whether to use separate c_proj after attn op for mu and sigma.")

    parser.add_argument("--generate_length", type=int, default=20)
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="temperature of 1.0 has no effect, lower tend toward greedy sampling",
    )
    parser.add_argument("--k", type=int, default=0)
    parser.add_argument("--p", type=float, default=0.9)
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.0, help="primarily useful for CTRL model; in that case, use 1.2"
    )
    parser.add_argument("--num_return_sequences", type=int, default=1, help="The number of samples to generate.")

    #  interpolation of mean and variance latent vectors
    parser.add_argument(
        "--mean_attr_direction_npy_filename", default=None,
        type=str, help="npy file that stores mean attr latent vector direction."
    )
    parser.add_argument(
        "--mean_start_distance", type=float, default=-50, help="Min magnitude to add mean attr latent vector to original vector"
    )
    parser.add_argument(
        "--mean_end_distance", type=float, default=50, help="Max magnitude to add mean attr latent vector to original vector"
    )
    parser.add_argument(
        "--var_attr_direction_npy_filename", default=None,
        type=str, help="npy file that stores var attr latent vector direction."
    )
    parser.add_argument(
        "--var_start_distance", type=float, default=-50, help="Min magnitude to add var attr latent vector to original vector"
    )
    parser.add_argument(
        "--var_end_distance", type=float, default=50, help="Max magnitude to add var attr latent vector to original vector"
    )
    parser.add_argument(
        "--num_interpolation", type=int, default=9, help="Number of interpolations for attr direction addition to latent vector"
    )
    parser.add_argument(
        "--encoded_prompt_len_cocon_gen", type=int, default=2, help="Length of prompt input ids to use during cocon generation."
    )

    parser.add_argument(
        "--include_zero_prompt",
        action="store_true",
        help="Whether include generated text samples with zero prompt, similar to encoded_prompt_len_cocon_gen=0",
    )

    parser.add_argument(
        "--custom_context_input_text_data_file",
        type=str,
        default=None,
        help="text file for sequences to use for custom mu_s generation",
    )

    parser.add_argument(
        "--train_cycle_detach_interval", type=int, default=1, help="Interval to detach cycle generated hidden states"
    )

    parser.add_argument("--use_unopt_cycle_recon_cocon_training", action="store_true", help="Whether to use unoptimized cycle recon training code.")

    parser.add_argument(
        "--cocon_block_type",
        type=str,
        default="1",
        help="Cocon block type , can be one of [1, 2, 3].",
    )

    parser.add_argument("--max_cocon_AR_length", type=int, default=100)

    parser.add_argument(
        "--self_cocon_lm_cs_mask_prob", type=float, default=0, help="Ratio of cs' hidden states for self_cocon_lm_loss computation"
    )
    parser.add_argument(
        "--self_cocon_lm_tis_mask_prob", type=float, default=0, help="Ratio of tis' hidden states for self_cocon_lm_loss computation"
    )
    parser.add_argument("--self_cocon_lm_mutual_exc_mask", action="store_true", help="Whether to use mutually exclusive masks for cs and tis for for self_cocon_lm_loss computation.")

    parser.add_argument(
        "--cycle_ar_cocon_recon_lm_tis_mask_prob", type=float, default=0, help="Ratio of tis' hidden states for cycle_ar_cocon_recon_lm_loss computation"
    )

    parser.add_argument("--use_only_last_cocon_output_for_ar", action="store_true", help="Whether to use_only_last_cocon_output_for_ar rather than the whole cocon output.")

    parser.add_argument("--use_history_source_as_context_source_for_gen", action="store_true", help="Whether to use history_source_data_file as context_source_data_file.")

    parser.add_argument(
        "--self_token_mask_prob", type=float, default=0, help="Probability to mask own token in context seq during self_cocon_lm_loss computation"
    )
    parser.add_argument(
        "--cycle_self_token_mask_prob", type=float, default=0, help="Probability to mask own position's token in context seq during cycle_ar_cocon_recon_lm_loss computation"
    )
    parser.add_argument(
        "--other_context_self_token_mask_prob", type=float, default=0, help="Probability to mask own position's token in context seq during other_context_cocon_lm_loss computation"
    )

    parser.add_argument("--min_hs_tis_split_offset", type=int, default=0, help="Min number of index to offset from hs_len to split train samples into hs/tis")
    parser.add_argument("--max_hs_tis_split_offset", type=int, default=0, help="Max number of index to offset from hs_len to split train samples into hs/tis")
    parser.add_argument("--track_hist_cocon_lm_loss", action="store_true", help="Whether to track hist_cocon_lm_loss for logging even without using for training.")

    parser.add_argument(
        "--line_by_line_cs",
        action="store_true",
        help="Whether distinct lines of text in the context seq dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--line_by_line_hs",
        action="store_true",
        help="Whether distinct lines of text in the history seq (prompt text) dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--enumerate_all_cs_for_each_hs",
        action="store_true",
        help="Whether to enumerate all context sequences for each history seq (prompt text) during cocon generation.",
    )

    parser.add_argument(
        "--prepend_bos_token_to_line",
        action="store_true",
        help="Whether to prepend bos_token to history seq (prompt text) during cocon generation.",
    )

    parser.add_argument("--text_json_key", type=str, default="text", help="key for sample text in data json object")

    parser.add_argument(
        "--prepended_text_to_remove",
        type=str,
        default=None,
        help="Prepended text to remove during data loading for evaluation, use ; to delimit a list of prepended_texts",
    )

    parser.add_argument("--do_eval_dist", action="store_true", help="Whether to run dist-1,2,3 eval on the dev set.")
    parser.add_argument("--dist_eval_max_samples", type=int, default=-1, help="Defaults to -1 which has no max limit.")


    parser.add_argument(
        "--context_attn_bias", type=float, default=0, help="Value to bias context_attn during cocon forward ops, for generation."
    )

    parser.add_argument(
        "--content_input",
        type=str,
        default=None,
        help="Content input for single COCON generation",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Content input for single COCON generation",
    )
    parser.add_argument("--content_input_delimit",
        type=str,
        default=';',
        help="Delimiter for multiple content inputs",
    )
    parser.add_argument("--do_single_cocon_generation", action="store_true", help="Whether to generate single text with cocon.")
    parser.add_argument("--append_cocon_output_files", action="store_true", help="Whether to append to existing cocon_output_file and cocon_output_jsonl.")


    args = parser.parse_args()
    return args
