"""
Usage:
- Start server
```
$ uvicorn demo_api:app
```

- Request
```
$ curl -XPOST -d '{"prompt_text": "In summary", "context": "finance", "length": 30}' http://localhost:8000/api/generate --header "Content-Type:application/json"
```
"""
from logging import getLogger, basicConfig

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from mygenerate import generate_with_topic
from schema import Config, GenerationRequest, GenerationResponse


basicConfig(level="INFO")
logger = getLogger(__name__)

# Constants
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu:0"
MODEL_NAME = "gpt2-medium"
COCON_BLOCK_MODEL_PATH = "models/COCON/cocon_block_pytorch_model.bin"


def load_model():
    logger.info("-------------- Loading models ----------------")

    with open("config.yml", "rt") as f:
        cfg = Config(**yaml.load(f, yaml.SafeLoader))
    set_seed(cfg)

    # Load config
    config = GPT2Config.from_pretrained(MODEL_NAME)

    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)

    # Load GPT2 model
    model = GPT2LMHeadModel.from_pretrained(
        MODEL_NAME,
        from_tf=False,
        config=config,
        cache_dir=None,
        output_meanvars=True,
        compute_meanvars_before_layernorm=False
    )
    model = model.to(DEVICE)
    model.eval()

    # Load CoconBlock
    cocon_block = CoconBlock(config.n_ctx, config, scale=True)
    cocon_state_dict = torch.load(COCON_BLOCK_MODEL_PATH)
    new_cocon_state_dict = fix_state_dict_naming(cocon_state_dict)
    cocon_block.load_state_dict(new_cocon_state_dict)
    cocon_block = cocon_block.to(DEVICE)
    cocon_block.eval()

    return cfg, tokenizer, model, cocon_block


def init_router() -> APIRouter:
    cfg, tokenizer, model, cocon_block = load_model()

    router = APIRouter()

    @router.get("/ping")
    def ping() -> dict:
        return dict(message="Healthy!")

    @router.post("/generate")
    def predict(request: GenerationRequest) -> GenerationResponse:
        gen_text = generate_with_topic(
            request.prompt_text,
            request.context,
            request.length,
            model,
            cocon_block,
            tokenizer,
            cfg,
            DEVICE
        )
        return GenerationResponse(
            generated_text=gen_text
        )

    return router


def start_fastapi():
    app = FastAPI()
    app.include_router(
        init_router(),
        prefix="/api",
        tags=["inference"]
    )
    return app


app = start_fastapi()
