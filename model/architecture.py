from pathlib import Path
from typing import Optional

try:
    from .tokenizer import SimpleTokenizer
except ImportError:
    from tokenizer import SimpleTokenizer

try:
    import torch
    from torch import Tensor, nn
    from torch.nn import functional as F
except ImportError:  # pragma: no cover - exercised only in minimal deployments
    torch = None
    Tensor = object
    nn = None
    F = None


if torch is not None:
    def _get_vocab_size(tokenizer: object) -> int:
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is not None:
            return int(vocab_size)
        get_vocab_size = getattr(tokenizer, "get_vocab_size", None)
        if get_vocab_size is None:
            raise TypeError("Tokenizer must expose vocab_size or get_vocab_size()")
        return int(get_vocab_size())


    def _get_token_id(tokenizer: object, token: str) -> Optional[int]:
        token_to_id = getattr(tokenizer, "token_to_id", None)
        if token_to_id is None:
            return None
        if callable(token_to_id):
            return token_to_id(token)
        return token_to_id.get(token)


    class MultiHeadAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float) -> None:
            super().__init__()
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads")
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.query_key_value = nn.Linear(d_model, 3 * d_model)
            self.output = nn.Linear(d_model, d_model)
            self.dropout = nn.Dropout(dropout)
            self.register_buffer(
                "causal_mask",
                torch.tril(torch.ones(block_size, block_size, dtype=torch.bool)),
            )

        def forward(self, hidden_states: Tensor) -> Tensor:
            batch_size, sequence_length, d_model = hidden_states.shape
            query, key, value = self.query_key_value(hidden_states).chunk(3, dim=-1)
            query = query.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
            key = key.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
            value = value.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)
            attention = (query @ key.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attention = attention.masked_fill(
                ~self.causal_mask[:sequence_length, :sequence_length],
                torch.finfo(attention.dtype).min,
            )
            attention = F.softmax(attention, dim=-1)
            attention = self.dropout(attention)
            output = attention @ value
            output = output.transpose(1, 2).contiguous().view(batch_size, sequence_length, d_model)
            return self.output(output)


    class FeedForward(nn.Module):
        def __init__(self, d_model: int, dropout: float) -> None:
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
                nn.Dropout(dropout),
            )

        def forward(self, hidden_states: Tensor) -> Tensor:
            return self.network(hidden_states)


    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float) -> None:
            super().__init__()
            self.layer_norm_1 = nn.LayerNorm(d_model)
            self.attention = MultiHeadAttention(d_model, n_heads, block_size, dropout)
            self.layer_norm_2 = nn.LayerNorm(d_model)
            self.feed_forward = FeedForward(d_model, dropout)

        def forward(self, hidden_states: Tensor) -> Tensor:
            hidden_states = hidden_states + self.attention(self.layer_norm_1(hidden_states))
            return hidden_states + self.feed_forward(self.layer_norm_2(hidden_states))


    class CustomLLM(nn.Module):
        def __init__(
            self,
            tokenizer: Optional[object] = None,
            vocab_size: Optional[int] = None,
            d_model: int = 384,
            n_heads: int = 6,
            n_layers: int = 6,
            block_size: int = 512,
            max_seq_len: Optional[int] = None,
            dropout: float = 0.1,
            weight_tying: bool = True,
        ) -> None:
            super().__init__()
            self.tokenizer = tokenizer or SimpleTokenizer()
            if max_seq_len is not None:
                block_size = max_seq_len
            if block_size < 1:
                raise ValueError("block_size must be positive")
            self.vocab_size = vocab_size or _get_vocab_size(self.tokenizer)
            self.block_size = block_size
            self.max_seq_len = block_size
            self.weight_tying = weight_tying
            self.token_embedding = nn.Embedding(self.vocab_size, d_model)
            self.position_embedding = nn.Embedding(block_size, d_model)
            self.blocks = nn.ModuleList(
                [TransformerBlock(d_model, n_heads, block_size, dropout) for _ in range(n_layers)]
            )
            self.layer_norm = nn.LayerNorm(d_model)
            self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
            if weight_tying:
                self.lm_head.weight = self.token_embedding.weight
            self.weights_loaded = False

        def forward(self, input_ids: Tensor) -> Tensor:
            _, sequence_length = input_ids.shape
            if sequence_length > self.block_size:
                input_ids = input_ids[:, -self.block_size:]
                sequence_length = self.block_size
            positions = torch.arange(sequence_length, device=input_ids.device)
            hidden_states = self.token_embedding(input_ids) + self.position_embedding(positions)
            for block in self.blocks:
                hidden_states = block(hidden_states)
            return self.lm_head(self.layer_norm(hidden_states))

        def load_weights(self, path: str | Path, device: str = "cpu") -> bool:
            try:
                checkpoint = torch.load(path, map_location=device, weights_only=False)
                state_dict = checkpoint.get("model_state_dict", checkpoint)
                self.load_state_dict(state_dict)
                self.to(device)
                self.eval()
                self.weights_loaded = True
                return True
            except (OSError, RuntimeError, KeyError, ValueError, TypeError):
                self.weights_loaded = False
                return False

        @torch.no_grad()
        def infer(
            self,
            prompt: str,
            max_new_tokens: int = 150,
            temperature: float = 0.8,
        ) -> str:
            if not self.weights_loaded:
                return f"Custom LLM Echo: {prompt}"
            temperature = max(float(temperature), 1e-5)
            max_new_tokens = max(0, min(int(max_new_tokens), 150))
            encoded = self.tokenizer.encode(prompt)
            tokens = encoded.ids if hasattr(encoded, "ids") else encoded
            input_ids = torch.tensor([tokens], dtype=torch.long, device=next(self.parameters()).device)
            newline_id = _get_token_id(self.tokenizer, "\n")
            eos_id = getattr(self.tokenizer, "eos_id", None)
            if eos_id is None:
                eos_id = _get_token_id(self.tokenizer, "<EOS>")
            for _ in range(max_new_tokens):
                logits = self(input_ids[:, -self.block_size:])[:, -1, :] / temperature
                probabilities = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                next_token_id = next_token.item()
                if next_token_id == eos_id or next_token_id == newline_id:
                    break
                input_ids = torch.cat((input_ids, next_token), dim=1)
            decoded_tokens = input_ids[0].tolist()
            try:
                return self.tokenizer.decode(decoded_tokens, skip_special_tokens=True)
            except TypeError:
                return self.tokenizer.decode(decoded_tokens)
else:
    class CustomLLM:
        def __init__(self, tokenizer: Optional[object] = None, **_: object) -> None:
            self.tokenizer = tokenizer or SimpleTokenizer()
            self.weights_loaded = False

        def load_weights(self, path: str | Path, device: str = "cpu") -> bool:
            return False

        def infer(
            self,
            prompt: str,
            max_new_tokens: int = 150,
            temperature: float = 0.8,
        ) -> str:
            return f"Custom LLM Echo: {prompt}"