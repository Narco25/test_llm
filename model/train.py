from pathlib import Path
from math import ceil, cos, pi

MODEL_DIR = Path(__file__).resolve().parent

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
except ImportError as error:  # pragma: no cover - depends on local environment
    raise SystemExit("Training requires PyTorch. Install dependencies first.") from error

try:
    from .architecture import CustomLLM
    from .bpe_data import CausalLanguageModelDataset, create_dataloader, train_or_load_tokenizer
except ImportError:
    from architecture import CustomLLM
    from bpe_data import CausalLanguageModelDataset, create_dataloader, train_or_load_tokenizer


def create_optimizer(
    model: torch.nn.Module,
    lr: float = 3e-4,
    weight_decay: float = 0.01,
    **kwargs: object,
) -> AdamW:
    """Apply decay to matrix weights only; exclude biases and normalization vectors."""
    learning_rate = kwargs.pop("learning_rate", None)
    if learning_rate is not None:
        if lr != 3e-4:
            raise TypeError("Pass either lr or learning_rate, not both")
        lr = float(learning_rate)
    if kwargs:
        unexpected = next(iter(kwargs))
        raise TypeError(f"create_optimizer() got an unexpected keyword argument {unexpected!r}")

    decay_parameters = []
    no_decay_parameters = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim == 2:
            decay_parameters.append(parameter)
        else:
            no_decay_parameters.append(parameter)
    return AdamW(
        [
            {"params": decay_parameters, "lr": lr, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "lr": lr, "weight_decay": 0.0},
        ],
        lr=lr,
    )


def create_warmup_cosine_scheduler(
    optimizer: AdamW,
    total_steps: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linearly warm up the learning rate, then cosine-decay it to zero."""
    total_steps = max(1, total_steps)
    warmup_steps = min(max(0, warmup_steps), total_steps - 1)

    def learning_rate_scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, (step - warmup_steps) / decay_steps)
        return 0.5 * (1.0 + cos(pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_scale)


def train(
    dataset_path: str | None = None,
    checkpoint_path: str | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    sequence_length: int = 512,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    mixed_precision: str = "bf16",
    device: str | None = None,
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

    tokenizer_path = MODEL_DIR / "tokenizer.json"
    tokenizer = train_or_load_tokenizer(dataset_file, tokenizer_path)
    dataset = CausalLanguageModelDataset.from_file(
        dataset_file,
        tokenizer,
        sequence_length=sequence_length,
    )
    if len(dataset) < 2:
        raise ValueError("input.txt must provide at least two BPE training sequences")
    validation_size = max(1, int(len(dataset) * 0.2))
    training_size = len(dataset) - validation_size
    training_dataset, validation_dataset = torch.utils.data.random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42),
    )
    training_loader = create_dataloader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = create_dataloader(validation_dataset, batch_size=batch_size, shuffle=False)

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if selected_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    mixed_precision = mixed_precision.lower()
    if mixed_precision not in {"none", "fp16", "bf16"}:
        raise ValueError("mixed_precision must be 'none', 'fp16', or 'bf16'")
    use_amp = selected_device.type == "cuda" and mixed_precision != "none"
    if mixed_precision == "bf16" and use_amp and not torch.cuda.is_bf16_supported():
        mixed_precision = "fp16"
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16

    model = CustomLLM(
        tokenizer=tokenizer,
        d_model=384,
        n_heads=6,
        n_layers=6,
        block_size=512,
    )
    model.to(selected_device)
    vocab_size = tokenizer.get_vocab_size()
    optimizer = create_optimizer(model, lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = max(1, len(training_loader))
    total_steps = max(1, epochs * steps_per_epoch)
    warmup_steps = int(total_steps * max(0.0, min(warmup_ratio, 1.0)))
    scheduler = create_warmup_cosine_scheduler(optimizer, total_steps, warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_function = nn.CrossEntropyLoss()
    best_validation_loss = float("inf")
    destination = (
        MODEL_DIR / "weights" / "best_model.pt"
        if checkpoint_path is None
        else Path(checkpoint_path)
    )
    if not destination.is_absolute():
        destination = MODEL_DIR / destination
    destination.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0
        for batch_index, batch in enumerate(training_loader, start=1):
            batch_inputs = batch["input_ids"].to(selected_device)
            batch_targets = batch["labels"].to(selected_device)
            with torch.amp.autocast(
                device_type=selected_device.type,
                dtype=amp_dtype,
                enabled=use_amp,
            ):
                logits = model(batch_inputs)
                loss = loss_function(
                    logits.view(-1, vocab_size),
                    batch_targets.view(-1),
                )
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
            batch_count += 1
            if batch_index % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs}, "
                    f"batch {batch_index}/{len(training_loader)}, "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        validation_loss_total = 0.0
        validation_batch_count = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch_inputs = batch["input_ids"].to(selected_device)
                batch_targets = batch["labels"].to(selected_device)
                with torch.amp.autocast(
                    device_type=selected_device.type,
                    dtype=amp_dtype,
                    enabled=use_amp,
                ):
                    validation_logits = model(batch_inputs)
                    validation_loss_total += loss_function(
                        validation_logits.view(-1, vocab_size),
                        batch_targets.view(-1),
                    ).item()
                validation_batch_count += 1
        validation_loss = validation_loss_total / validation_batch_count
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