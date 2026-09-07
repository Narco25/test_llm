from pathlib import Path

from model.architecture import CustomLLM

_model = CustomLLM()
_weights_path = Path(__file__).parent / "model" / "weights" / "llm_weights.pt"
if _weights_path.is_file():
    _model.load_weights(_weights_path)


def generate(prompt: str) -> str:
    return _model.infer(prompt)