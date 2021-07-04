from logging import getLogger, basicConfig

import torch
from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    CoconBlock
)
import streamlit as st
import yaml

from utils.utils import set_seed, fix_state_dict_naming
from mygenerate import generate_topic
from schema import Config


basicConfig(level="INFO")
logger = getLogger(__name__)


# Constants
device = "cuda:0" if torch.cuda.is_available() else "cpu:0"
model_name = "gpt2-medium"


@st.cache(allow_output_mutation=True, suppress_st_warning=True, max_entries=1)
def load_model(model_name: str, device: str):
    logger.info("-------------- Loading models ----------------")

    with open("config.yml", "rt") as f:
        cfg = Config(**yaml.load(f, yaml.SafeLoader))
    set_seed(cfg)

    # Load config
    config = GPT2Config.from_pretrained(model_name)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    # Load GPT2 model
    model = GPT2LMHeadModel.from_pretrained(
        model_name,
        from_tf=False,
        config=config,
        cache_dir=None,
        output_meanvars=True,
        compute_meanvars_before_layernorm=False
    )
    model = model.to(device)
    model.eval()

    # Load CoconBlock
    cocon_block = CoconBlock(config.n_ctx, config, scale=True)
    cocon_state_dict = torch.load("models/COCON/cocon_block_pytorch_model.bin")
    new_cocon_state_dict = fix_state_dict_naming(cocon_state_dict)
    cocon_block.load_state_dict(new_cocon_state_dict)
    cocon_block = cocon_block.to(device)
    cocon_block.eval()

    return cfg, tokenizer, model, cocon_block


if __name__ == "__main__":
    # UI
    lang = st.sidebar.radio("Choose language", ["English", "日本語"])
    length = st.sidebar.selectbox(
        "Length", [2, 5, 10, 15, 20, 30, 50, 100], index=4
    )

    # Load models
    cfg, tokenizer, model, cocon_block = load_model(model_name, device)
    logger.info("READY!!!")

    if lang == "日本語":
        st.write("Not support yet")
    else:
        context = st.text_input('Conditioned context', 'finance')
        prompt_text = st.text_input('Prompt text', 'In summary')
        if st.button("Generate text"):
            gen_text = generate_topic(prompt_text, context, length, model, cocon_block, tokenizer, cfg, device)
            st.write(gen_text)
