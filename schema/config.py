from pydantic import BaseModel


class Config(BaseModel):
    n_gpu: int = 1
    temperature: float = 1.0
    k: int = 0
    p: float = 0.9
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    output_hidden_for_cocon_after_block_ind: int = 6
    use_only_last_cocon_output_for_ar: bool = False
    context_attn_bias: int = -10
    seed: int = 42
