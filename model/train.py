from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
except ImportError as error:  # pragma: no cover - depends on local environment
    raise SystemExit("Training requires PyTorch. Install dependencies first.") from error

try:
    from .architecture import CustomLLM
    from .tokenizer import SimpleTokenizer
except ImportError:
    from architecture import CustomLLM
    from tokenizer import SimpleTokenizer


def train(
    dataset_path: str | None = None,
    checkpoint_path: str | None = None,
    epochs: int = 10,
    batch_size: int = 16,
    sequence_length: int = 128,
    learning_rate: float = 3e-4,
) -> None:
    if dataset_path is None:
        dataset_candidates = [MODEL_DIR / "input.txt", Path.cwd() / "input.txt"]
        dataset_file = next(
            (candidate for candidate in dataset_candidates if candidate.is_file()),
            None,
        )
        if dataset_file is None:
            raise FileNotFoundError(
                "Training dataset not found. Create model/input.txt before training."
            )
    else:
        dataset_file = Path(dataset_path)
        if not dataset_file.is_file():
            raise FileNotFoundError(f"Training dataset not found: {dataset_file}")

    text = dataset_file.read_text(encoding="utf-8")
    tokenizer = SimpleTokenizer()
    token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    if len(token_ids) <= sequence_length:
        raise ValueError("input.txt must contain more tokens than sequence_length")

    inputs = []
    targets = []
    for start in range(0, len(token_ids) - sequence_length, sequence_length):
        inputs.append(token_ids[start : start + sequence_length])
        targets.append(token_ids[start + 1 : start + sequence_length + 1])
    input_batches = torch.stack(inputs)
    target_batches = torch.stack(targets)

    model = CustomLLM(tokenizer=tokenizer, block_size=sequence_length)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(input_batches))
        total_loss = 0.0
        for start in range(0, len(input_batches), batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(input_batches[indices])
            loss = loss_function(
                logits.reshape(-1, tokenizer.vocab_size),
                target_batches[indices].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"epoch {epoch + 1}/{epochs}, loss={total_loss:.4f}")

    if checkpoint_path is None:
        destination = MODEL_DIR / "weights" / "llm_weights.pt"
    else:
        destination = Path(checkpoint_path)
        if not destination.is_absolute():
            destination = MODEL_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, destination)
    print(f"saved checkpoint to {destination}")


if __name__ == "__main__":
    train()