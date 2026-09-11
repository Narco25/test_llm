"""HTTP API for local CustomLLM text generation."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tokenizers import Tokenizer

from model.architecture import CustomLLM
from model.generation import generate


LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "model"
CHECKPOINT_PATH = MODEL_DIR / "weights" / "best_model.pt"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

model: CustomLLM | None = None
tokenizer: Any | None = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = Field(default=100, ge=0)
    temperature: float = Field(default=0.7, gt=0)
    top_k: int = Field(default=40, ge=0)
    top_p: float = Field(default=0.9, gt=0, le=1)


def load_assets() -> None:
    """Load the tokenizer and checkpoint, leaving the API usable without weights."""
    global model, tokenizer

    if not TOKENIZER_PATH.is_file():
        LOGGER.warning("Tokenizer not found at %s; generation is unavailable.", TOKENIZER_PATH)
        tokenizer = None
        model = None
        return

    tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))
    vocabulary_size = tokenizer.get_vocab_size()
    model = CustomLLM(
        tokenizer=tokenizer,
        vocab_size=vocabulary_size,
        d_model=384,
        n_layers=6,
        n_heads=6,
        block_size=256,
    )
    LOGGER.info(
        "Initialized CustomLLM with vocab_size=%d, d_model=384, n_layers=6, "
        "n_heads=6, block_size=256",
        vocabulary_size,
    )
    if CHECKPOINT_PATH.is_file():
        try:
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            model.weights_loaded = True
            LOGGER.info("Loaded model weights from %s", CHECKPOINT_PATH)
        except Exception:
            LOGGER.exception(
                "Could not load model weights from %s; using initialized model instead.",
                CHECKPOINT_PATH,
            )
            model.weights_loaded = False
    else:
        LOGGER.warning("Model weights not found at %s; using initialized model.", CHECKPOINT_PATH)
    model.to(device)
    model.eval()


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_assets()
    yield


app = FastAPI(title="CustomLLM Generation API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def decode_output(output_ids: torch.Tensor) -> str:
    """Decode only valid non-special IDs from the generated sequence."""
    if tokenizer is None:
        return ""
    vocabulary_size = tokenizer.get_vocab_size()
    special_ids = {
        token_id
        for token in (
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
            "<|im_start|>",
            "<|im_end|>",
        )
        if (token_id := tokenizer.token_to_id(token)) is not None
    }
    clean_ids = [
        token_id
        for token_id in output_ids.detach().cpu().tolist()
        if 0 <= token_id < vocabulary_size and token_id not in special_ids
    ]
    return tokenizer.decode(clean_ids, skip_special_tokens=True).strip()


@app.post("/generate")
def generate_text(request: GenerateRequest) -> dict[str, str | int]:
    if model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="Model assets are unavailable; add model/tokenizer.json and a checkpoint.",
        )

    encoded = tokenizer.encode(request.prompt, add_special_tokens=False)
    prompt_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
    output_ids = generate(
        model,
        prompt_ids=prompt_ids,
        max_new_tokens=request.max_new_tokens,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        repetition_penalty=1.2,
        eos_token_id=None,
        device=device,
    )
    tokens_generated = max(0, output_ids.shape[1] - prompt_ids.shape[1])
    return {
        "prompt": request.prompt,
        "generated_text": decode_output(output_ids[0]),
        "tokens_generated": tokens_generated,
    }


@app.get("/health")
def health() -> dict[str, str | bool]:
    return {"status": "ok", "model_loaded": model is not None and model.weights_loaded}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)