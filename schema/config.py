from typing import List

from pydantic import BaseModel


class ModelConfig(BaseModel):
    model_id: str
    enabled: bool
    model_display_name: str
    model_name: str
    cocon_block_model: str
    config_class: str
    tokenizer_class: str
    model_class: str
    generate_length: int
    temperature: float = 1.0
    k: int = 0
    p: float = 0.9
    repetition_penalty: float = 1.0
    num_return_sequences: int = 1
    output_hidden_for_cocon_after_block_ind: int = 6
    use_only_last_cocon_output_for_ar: bool = False
    context_attn_bias: int = -10


class Config(BaseModel):
    models: List[ModelConfig]
    n_gpu: int = 1
    seed: int = 42
