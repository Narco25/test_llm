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
    epochs: int = 200,
    batch_size: int = 32,
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
    if len(input_batches) < 2:
        raise ValueError("input.txt must provide at least two training sequences")

    split_index = max(1, min(len(input_batches) - 1, int(len(input_batches) * 0.8)))
    train_inputs = input_batches[:split_index]
    train_targets = target_batches[:split_index]
    validation_inputs = input_batches[split_index:]
    validation_targets = target_batches[split_index:]

    model = CustomLLM(tokenizer=tokenizer, block_size=sequence_length)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    destination = (
        MODEL_DIR / "weights" / "llm_weights.pt"
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    if not destination.is_absolute():
        destination = MODEL_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(train_inputs))
        total_loss = 0.0
        batch_count = 0
        for start in range(0, len(train_inputs), batch_size):
            indices = permutation[start : start + batch_size]
            logits = model(train_inputs[indices])
            loss = loss_function(
                logits.reshape(-1, tokenizer.vocab_size),
                train_targets[indices].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_inputs)
            validation_loss = loss_function(
                validation_logits.reshape(-1, tokenizer.vocab_size),
                validation_targets.reshape(-1),
            ).item()
        average_training_loss = total_loss / batch_count
        print(
            f"epoch {epoch + 1}/{epochs}, "
            f"train_loss={average_training_loss:.4f}, "
            f"validation_loss={validation_loss:.4f}"
        )

        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "validation_loss": best_validation_loss,
                    "epoch": epoch + 1,
                },
                destination,
            )
            print(f"saved best checkpoint to {destination}")


if __name__ == "__main__":
    train()