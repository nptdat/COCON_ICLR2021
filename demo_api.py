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
import os

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch
from transformers_custom import (
    GPT2Config,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    CoconBlock
)
from transformers import (
    AutoTokenizer, T5Tokenizer
)
import streamlit as st
import yaml

from utils.utils import set_seed, fix_state_dict_naming
from mygenerate import generate_with_topic
from schema import ModelConfig, Config, GenerationRequest, GenerationResponse


basicConfig(level="INFO")
logger = getLogger(__name__)

# Constants
DEVICE = os.environ.get("DEVICE", "cuda:0") if torch.cuda.is_available() else "cpu:0"
MODEL_NAME = "gpt2-medium"
COCON_BLOCK_MODEL_PATH = "models/COCON/cocon_block_pytorch_model.bin"


def load_model():
    logger.info("-------------- Loading models ----------------")

    with open("config.yml", "rt") as f:
        cfg = Config(**yaml.load(f, yaml.SafeLoader))

    set_seed(cfg)
    # models = {m.model_id: m for m in cfg.models}

    models = {}
    for cfg_model in cfg.models:
        model_id = cfg_model.model_id

        if not cfg_model.enabled:
            logger.info(f"Skipped the model {model_id}!")
            continue

        # Load config
        config_class = eval(cfg_model.config_class)
        config = config_class.from_pretrained(cfg_model.model_name)

        # Load tokenizer
        tokenizer_class = eval(cfg_model.tokenizer_class)
        tokenizer = tokenizer_class.from_pretrained(cfg_model.model_name)

        # Load GPT2 model
        model_class = eval(cfg_model.model_class)
        model = model_class.from_pretrained(
            cfg_model.model_name,
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

        models[model_id] = dict(
            cfg=cfg_model,
            config=config,
            tokenizer=tokenizer,
            core_model=model,
            cocon_block=cocon_block
        )

    return cfg, models


def init_router() -> APIRouter:
    cfg, models = load_model()

    router = APIRouter()

    @router.get("/ping")
    def ping() -> dict:
        return dict(message="Healthy!")

    @router.post("/generate")
    def predict(request: GenerationRequest) -> GenerationResponse:
        model_id = request.model_id
        if model_id in models:
            model = models[model_id]
            gen_text = generate_with_topic(
                request.prompt_text,
                request.context,
                request.length,
                model["core_model"],
                model["cocon_block"],
                model["tokenizer"],
                model["cfg"],
                DEVICE
            )
        else:
            gen_text = f"ERROR: The model {model_id} is not supported yet."

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
