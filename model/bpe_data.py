"""Byte-level BPE tokenization and language-model data loading utilities."""

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer
from tokenizers import AddedToken


DEFAULT_VOCAB_SIZE = 4_096
SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
    "<|im_start|>",
    "<|im_end|>",
]


def format_conversation(messages: Sequence[Mapping[str, str]]) -> str:
    """Format chat messages using explicit role and turn boundary tokens."""
    turns = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {role!r}")
        if content is None:
            raise ValueError("Each conversation message needs a content field")
        turns.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(turns)


def format_instruction(prompt: str, response: str) -> str:
    """Format one instruction pair with ChatML role and boundary tokens."""
    return format_conversation(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    )


def _conversation_from_record(record: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = record.get("messages")
    if isinstance(messages, Sequence) and not isinstance(messages, (str, bytes)):
        conversation = []
        for message in messages:
            if not isinstance(message, Mapping):
                continue
            role = message.get("role")
            content = message.get("content")
            if role in {"system", "user", "assistant"} and isinstance(content, str):
                conversation.append({"role": role, "content": content})
        if any(message["role"] == "user" for message in conversation) and any(
            message["role"] == "assistant" for message in conversation
        ):
            return conversation

    prompt = record.get("instruction") or record.get("prompt") or record.get("question")
    response = record.get("output") or record.get("response") or record.get("answer")
    if isinstance(prompt, str) and isinstance(response, str) and prompt and response:
        return [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
    raise ValueError("Instruction record must contain messages or prompt/response fields")


def load_instruction_conversations(
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    split: str = "train_sft",
    max_examples: int | None = None,
) -> list[list[dict[str, str]]]:
    """Download a Hugging Face instruction dataset and normalize its conversations."""
    try:
        from datasets import load_dataset
    except ImportError as error:  # pragma: no cover - depends on training environment
        raise RuntimeError("Install the 'datasets' package to download instruction data.") from error

    dataset = load_dataset(dataset_name, split=split)
    records = dataset if max_examples is None else dataset.select(range(min(max_examples, len(dataset))))
    conversations = []
    for record in records:
        try:
            conversations.append(_conversation_from_record(record))
        except ValueError:
            continue
    if not conversations:
        raise ValueError(f"No usable instruction conversations found in {dataset_name}:{split}")
    return conversations


def save_formatted_conversations(
    conversations: Sequence[Sequence[Mapping[str, str]]],
    output_path: str | Path,
) -> Path:
    """Write ChatML-formatted conversations for tokenizer training."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "\n".join(format_conversation(conversation) for conversation in conversations),
        encoding="utf-8",
    )
    return output_path


def train_or_load_tokenizer(
    text_path: str | Path,
    tokenizer_path: str | Path = "model/tokenizer.json",
    vocab_size: int = DEFAULT_VOCAB_SIZE,
) -> Tokenizer:
    """Load a saved byte-level BPE tokenizer or train one from ``text_path``."""
    text_path = Path(text_path)
    tokenizer_path = Path(tokenizer_path)
    if not text_path.is_file():
        raise FileNotFoundError(f"Training text file not found: {text_path}")

    if tokenizer_path.is_file():
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    else:
        tokenizer = Tokenizer(BPE(unk_token="<UNK>"))
        tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=SPECIAL_TOKENS,
            show_progress=True,
        )
        tokenizer.train([str(text_path)], trainer)
        tokenizer.post_processor = TemplateProcessing(
            single="<BOS> $A <EOS>",
            pair="<BOS> $A <EOS> <BOS> $B <EOS>",
            special_tokens=[
                ("<BOS>", tokenizer.token_to_id("<BOS>")),
                ("<EOS>", tokenizer.token_to_id("<EOS>")),
            ],
        )
        tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tokenizer_path))

    special_tokens = [
        AddedToken(token, normalized=False, special=True)
        for token in SPECIAL_TOKENS
        if tokenizer.token_to_id(token) is None
    ]
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    else:
        tokenizer.add_special_tokens(
            [AddedToken(token, normalized=False, special=True) for token in SPECIAL_TOKENS]
        )
    tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(tokenizer_path))

    tokenizer.enable_truncation(max_length=1_000_000)
    return tokenizer


class CausalLanguageModelDataset(Dataset[dict[str, torch.Tensor]]):
    """Fixed-length shifted token windows for next-token prediction."""

    def __init__(
        self,
        texts: str | Sequence[str],
        tokenizer: Tokenizer,
        sequence_length: int = 512,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.pad_id = tokenizer.token_to_id("<PAD>")
        self.eos_id = tokenizer.token_to_id("<EOS>")
        if self.pad_id is None or self.eos_id is None:
            raise ValueError("Tokenizer must define <PAD> and <EOS> tokens")

        if isinstance(texts, str):
            texts = [texts]
        token_ids: list[int] = []
        for text in texts:
            encoded = tokenizer.encode(text, add_special_tokens=True)
            token_ids.extend(encoded.ids)
        if len(token_ids) < 2:
            raise ValueError("The dataset must contain at least two encoded tokens")

        self.samples: list[tuple[list[int], list[int]]] = []
        stride = sequence_length
        for start in range(0, len(token_ids) - 1, stride):
            window = token_ids[start : start + sequence_length + 1]
            if len(window) < 2:
                continue
            inputs = window[:-1]
            labels = window[1:]
            self.samples.append((inputs, labels))

    @classmethod
    def from_file(
        cls,
        text_path: str | Path,
        tokenizer: Tokenizer,
        sequence_length: int = 512,
    ) -> "CausalLanguageModelDataset":
        lines = Path(text_path).read_text(encoding="utf-8").splitlines()
        return cls(lines, tokenizer, sequence_length)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self.samples[index]
        real_length = len(input_ids)
        padding_length = self.sequence_length - real_length
        input_ids = input_ids + [self.pad_id] * padding_length
        labels = labels + [-100] * padding_length
        attention_mask = [1] * real_length + [0] * padding_length
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


class InstructionDataset(Dataset[dict[str, torch.Tensor]]):
    """Conversation examples with loss enabled only on assistant responses."""

    def __init__(
        self,
        conversations: Sequence[Sequence[Mapping[str, str]]],
        tokenizer: Tokenizer,
        sequence_length: int = 512,
    ) -> None:
        if sequence_length < 2:
            raise ValueError("sequence_length must be at least 2")
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.pad_id = tokenizer.token_to_id("<PAD>")
        if self.pad_id is None:
            raise ValueError("Tokenizer must define a <PAD> token")

        self.samples: list[tuple[list[int], list[int]]] = []
        for conversation in conversations:
            token_ids: list[int] = []
            trainable_tokens: list[bool] = []
            for message in conversation:
                role = message.get("role")
                content = message.get("content")
                if role not in {"system", "user", "assistant"} or content is None:
                    raise ValueError("Messages need role and content fields")
                prefix = f"<|im_start|>{role}\n"
                suffix = f"{content}<|im_end|>\n"
                prefix_ids = tokenizer.encode(prefix, add_special_tokens=False).ids
                suffix_ids = tokenizer.encode(suffix, add_special_tokens=False).ids
                token_ids.extend(prefix_ids)
                token_ids.extend(suffix_ids)
                trainable_tokens.extend([False] * len(prefix_ids))
                trainable_tokens.extend([role == "assistant"] * len(suffix_ids))

            if len(token_ids) < 2:
                raise ValueError("Each conversation must contain at least two tokens")
            for start in range(0, len(token_ids) - 1, sequence_length):
                window = token_ids[start : start + sequence_length + 1]
                if len(window) < 2:
                    continue
                target_mask = trainable_tokens[start + 1 : start + len(window)]
                inputs = window[:-1]
                labels = [
                    token if is_trainable else -100
                    for token, is_trainable in zip(window[1:], target_mask)
                ]
                if not any(label != -100 for label in labels):
                    continue
                self.samples.append((inputs, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        input_ids, labels = self.samples[index]
        real_length = len(input_ids)
        padding_length = self.sequence_length - real_length
        return {
            "input_ids": torch.tensor(
                input_ids + [self.pad_id] * padding_length, dtype=torch.long
            ),
            "attention_mask": torch.tensor(
                [1] * real_length + [0] * padding_length, dtype=torch.long
            ),
            "labels": torch.tensor(
                labels + [-100] * padding_length, dtype=torch.long
            ),
        }


def create_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    batch_size: int = 32,
    shuffle: bool = True,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Create batches whose tensors have shape ``(batch_size, sequence_length)``."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


if __name__ == "__main__":
    model_dir = Path(__file__).resolve().parent
    tokenizer = train_or_load_tokenizer(model_dir / "input.txt", model_dir / "tokenizer.json")
    dataset = CausalLanguageModelDataset.from_file(model_dir / "input.txt", tokenizer)
    batch = next(iter(create_dataloader(dataset)))
    print({name: tuple(value.shape) for name, value in batch.items()})
