from pathlib import Path

import torch

from model.architecture import CustomLLM

weights_path = Path(__file__).parent / "model" / "weights" / "llm_weights.pt"

if weights_path.is_file():
    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        block_size = state_dict["position_embedding.weight"].shape[0]
        model = CustomLLM(block_size=block_size)
        model.load_state_dict(state_dict)
        model.eval()
        model.weights_loaded = True
        print(f"[INFO] Loaded model weights from {weights_path.name}")
    except (OSError, RuntimeError, KeyError, TypeError, ValueError) as error:
        model = CustomLLM()
        print(f"[ERROR] Failed to load model weights from {weights_path.name}: {error}")
else:
    model = CustomLLM()
    print(f"[ERROR] Model weights not found: {weights_path}")


def generate(
    prompt: str,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
) -> str:
    return model.infer(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
