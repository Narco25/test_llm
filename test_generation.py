"""Interactive CLI for testing BPE language-model generation."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from model.architecture import CustomLLM
from model.bpe_data import train_or_load_tokenizer
from model.generation import generate


LOGGER = logging.getLogger("test_generation")
MODEL_DIR = Path(__file__).resolve().parent / "model"
EXAMPLE_PROMPTS = {
    "1": "The future of software engineering is",
    "2": "Q: What is this site built with?\nA:",
    "3": "<|im_start|>user\nWhat is this site built with?<|im_end|>\n<|im_start|>assistant\n",
}


def checkpoint_candidates() -> list[Path]:
    root = Path(__file__).resolve().parent
    return [
        root / "model" / "weights" / "best_model.pt",
        root / "model" / "weights" / "llm_weights.pt",
        root / "weights" / "best_model.pt",
        root / "weights" / "llm_weights.pt",
    ]


def load_model(tokenizer: Any) -> CustomLLM:
    """Load the best checkpoint, or return a randomly initialized model."""
    checkpoint_path = next(
        (path for path in checkpoint_candidates() if path.is_file()),
        None,
    )
    if checkpoint_path is None:
        LOGGER.warning(
            "No checkpoint found in model/weights or weights "
            "(best_model.pt or llm_weights.pt); using randomly initialized weights."
        )
        return CustomLLM(tokenizer=tokenizer, block_size=512)

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        block_size = state_dict.get("position_embedding.weight").shape[0]
        model = CustomLLM(tokenizer=tokenizer, block_size=block_size)
        model.load_state_dict(state_dict)
        model.eval()
        model.weights_loaded = True
        LOGGER.info("Loaded checkpoint: %s", checkpoint_path)
        return model
    except (OSError, RuntimeError, KeyError, TypeError, AttributeError) as error:
        LOGGER.warning("Could not load %s: %s; using random weights.", checkpoint_path, error)
        return CustomLLM(tokenizer=tokenizer, block_size=512)


def parse_parameters(values: dict[str, float | int], text: str) -> None:
    """Update sampling parameters from ``name=value`` pairs."""
    for assignment in text.split():
        if "=" not in assignment:
            raise ValueError("Parameters must use name=value syntax")
        name, raw_value = assignment.split("=", 1)
        if name not in values:
            raise ValueError(f"Unknown parameter: {name}")
        current_value = values[name]
        values[name] = type(current_value)(raw_value)
    if values["max_new_tokens"] < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if values["temperature"] <= 0:
        raise ValueError("temperature must be greater than zero")
    if values["top_k"] < 0:
        raise ValueError("top_k must be non-negative")
    if not 0 < values["top_p"] <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")


def run_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    tokenizer = train_or_load_tokenizer(
        MODEL_DIR / "input.txt",
        MODEL_DIR / "tokenizer.json",
    )
    model = load_model(tokenizer)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    parameters: dict[str, float | int] = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "top_p": args.top_p,
    }
    parse_parameters(parameters, " ".join(f"{key}={value}" for key, value in parameters.items()))

    print("Interactive generation CLI")
    print("Enter a prompt, /examples, /params name=value ..., or /quit.")
    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            break
        if prompt == "/examples":
            for number, example in EXAMPLE_PROMPTS.items():
                print(f"{number}: {example!r}")
            continue
        if prompt.startswith("/params"):
            try:
                parse_parameters(parameters, prompt.removeprefix("/params").strip())
                print(f"Parameters: {parameters}")
            except ValueError as error:
                print(f"Parameter error: {error}")
            continue
        if prompt in EXAMPLE_PROMPTS:
            prompt = EXAMPLE_PROMPTS[prompt]

        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)
        output_ids = generate(
            model,
            prompt_ids=prompt_ids,
            max_new_tokens=int(parameters["max_new_tokens"]),
            temperature=float(parameters["temperature"]),
            top_k=int(parameters["top_k"]),
            top_p=float(parameters["top_p"]),
            repetition_penalty=1.2,
            eos_token_id=tokenizer.token_to_id("<EOS>"),
            device=device,
        )
        output_text = tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True)
        print(f"\nGenerated:\n{output_text}")


if __name__ == "__main__":
    run_cli()
