from logging import getLogger
import torch


logger = getLogger(__name__)


def generate_with_topic(prompt_text, context, length, model, cocon_block, tokenizer, args, device):
    prompt_seq = tokenizer.encode(
        tokenizer.bos_token + prompt_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        if context:
            logger.info("--- Generate with CoconBlock based on GPT-2")
            context_seq = tokenizer.encode(context, add_special_tokens=False, return_tensors="pt").to(device)
            output_sequences = model.generate(
                input_ids=prompt_seq[:, 0:0],
                max_length=length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences,
                do_cocon=True,
                cocon_block=cocon_block,
                cocon_context_inputs=context_seq,
                cocon_history_inputs=prompt_seq,
                cocon_after_block_ind=args.output_hidden_for_cocon_after_block_ind,
                transform_h_after_layernorm=False,
                use_only_last_cocon_output_for_ar=args.use_only_last_cocon_output_for_ar,
                context_attn_bias=-5
            )
            output_sequences = torch.cat([prompt_seq, output_sequences], dim=1)
        else:
            logger.info("--- Generate with GPT-2 only")
            output_sequences = model.generate(
                input_ids=prompt_seq,
                max_length=length,
                temperature=args.temperature,
                top_k=args.k,
                top_p=args.p,
                repetition_penalty=args.repetition_penalty,
                do_sample=True,
                num_return_sequences=args.num_return_sequences
            )

    return tokenizer.decode(output_sequences[0][1:])
