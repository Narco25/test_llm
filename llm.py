"""
LLM From Scratch
================
A complete, minimal GPT-style language model built from first principles.
No ML frameworks beyond NumPy for the core math — PyTorch used for autograd
and GPU support. Everything else (attention, positional encoding, training
loop, tokenizer, sampling) is hand-rolled.

Architecture: Decoder-only Transformer (GPT-2 style)
  - Byte-Pair Encoding (BPE) tokenizer
  - Token + learned positional embeddings
  - N stacked transformer blocks, each with:
      • Multi-Head Causal Self-Attention
      • Feed-Forward Network (SwiGLU activation)
      • RMSNorm (pre-norm)
  - Tied input/output embedding weights
  - Top-p (nucleus) + temperature sampling

Usage
-----
  # Quick demo (trains on a tiny corpus)
  python llm_from_scratch.py --demo

  # Train on a text file
  python llm_from_scratch.py --train corpus.txt --epochs 5 --out model.pt

  # Generate text
  python llm_from_scratch.py --generate "Once upon a time" --model model.pt
"""

import math
import os
import re
import time
import argparse
import json
import struct
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


# ─────────────────────────────────────────────
# 1.  TOKENIZER  (Byte-Pair Encoding from scratch)
# ─────────────────────────────────────────────

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer built from scratch.

    Steps:
      1. Start with a character-level vocabulary (256 bytes).
      2. Iteratively merge the most frequent adjacent pair.
      3. Repeat `num_merges` times.
    """

    SPECIAL = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}

    def __init__(self):
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab: Dict[int, bytes] = {}          # id → bytes
        self.vocab_inv: Dict[bytes, int] = {}      # bytes → id
        self._built = False

    # ── build ──────────────────────────────────
    def train(self, text: str, vocab_size: int = 1000, verbose: bool = True):
        """Train BPE on raw text."""
        num_merges = vocab_size - 256 - len(self.SPECIAL)
        assert num_merges >= 0, "vocab_size must be > 256 + special tokens"

        # Base vocab: raw bytes 0-255
        self.vocab = {i: bytes([i]) for i in range(256)}
        for name, idx in self.SPECIAL.items():
            # Offset special tokens above byte range
            self.vocab[256 + idx] = name.encode()
        next_id = 256 + len(self.SPECIAL)

        # Rebuild inverse
        self.vocab_inv = {v: k for k, v in self.vocab.items()}

        # Pre-tokenize into list of byte sequences per word
        # (split on whitespace; prepend 0x20 space byte to non-first words)
        words = text.encode("utf-8").split(b" ")
        corpus: List[List[int]] = []
        for i, w in enumerate(words):
            if not w:
                continue
            prefix = b" " if i > 0 else b""
            corpus.append(list(prefix + w))

        if verbose:
            print(f"[BPE] Training on {len(corpus):,} words, "
                  f"target vocab={vocab_size}, merges={num_merges}")

        for step in range(num_merges):
            # Count pairs
            pair_counts: Counter = Counter()
            for seq in corpus:
                for a, b in zip(seq, seq[1:]):
                    pair_counts[(a, b)] += 1
            if not pair_counts:
                break
            best = pair_counts.most_common(1)[0][0]
            # Merge best pair everywhere
            new_token_bytes = self.vocab[best[0]] + self.vocab[best[1]]
            self.merges[best] = next_id
            self.vocab[next_id] = new_token_bytes
            self.vocab_inv[new_token_bytes] = next_id
            corpus = [self._merge_pair(seq, best, next_id) for seq in corpus]
            next_id += 1
            if verbose and (step + 1) % max(1, num_merges // 10) == 0:
                pct = (step + 1) / num_merges * 100
                print(f"  [{pct:5.1f}%] merge #{step+1:4d}: "
                      f"{new_token_bytes!r} → id {next_id-1}")

        self._built = True
        if verbose:
            print(f"[BPE] Done. Vocab size: {len(self.vocab)}")

    @staticmethod
    def _merge_pair(seq: List[int], pair: Tuple[int,int], new_id: int) -> List[int]:
        out, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == pair[0] and seq[i+1] == pair[1]:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    # ── encode / decode ───────────────────────
    def encode(self, text: str, add_special: bool = True) -> List[int]:
        ids = list(text.encode("utf-8"))   # start at byte level
        # Apply merges in order
        for pair, new_id in self.merges.items():
            ids = self._merge_pair(ids, pair, new_id)
        if add_special:
            ids = [self.SPECIAL["<bos>"] + 256] + ids + [self.SPECIAL["<eos>"] + 256]
        return ids

    def decode(self, ids: List[int]) -> str:
        special_ids = {256 + v for v in self.SPECIAL.values()}
        raw = b"".join(
            self.vocab[i] for i in ids if i not in special_ids and i in self.vocab
        )
        return raw.decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ── save / load ───────────────────────────
    def save(self, path: str):
        data = {
            "merges": [[list(k), v] for k, v in self.merges.items()],
            "vocab": {str(k): list(v) for k, v in self.vocab.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        with open(path) as f:
            data = json.load(f)
        tok = cls()
        tok.merges = {(k[0], k[1]): v for k, v in data["merges"]}
        tok.vocab = {int(k): bytes(v) for k, v in data["vocab"].items()}
        tok.vocab_inv = {v: int(k) for k, v in data["vocab"].items()}
        tok._built = True
        return tok


# ─────────────────────────────────────────────
# 2.  MODEL COMPONENTS
# ─────────────────────────────────────────────

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no mean-centering)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE).
    Encodes position by rotating pairs of dimensions — no learned parameters,
    generalises better than learned absolute positional embeddings.
    """
    def __init__(self, dim: int, max_seq: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)          # (T, D/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (T, D)
        self.register_buffer("cos_cache", emb.cos()[None, None, :, :])  # (1,1,T,D)
        self.register_buffer("sin_cache", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # q, k: (B, H, T, head_dim)
        T = q.shape[2]
        cos = self.cos_cache[:, :, :T, :]
        sin = self.sin_cache[:, :, :T, :]
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention with RoPE.

    Uses a causal (lower-triangular) mask so each token can only
    attend to positions ≤ its own — the decoder property that lets
    us train with teacher-forcing and generate autoregressively.
    """
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5

        # Fused QKV projection
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_drop = nn.Dropout(config.dropout)

        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len)

        # Causal mask buffer (will be sliced at runtime)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
            .view(1, 1, config.max_seq_len, config.max_seq_len)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim

        # Project and split Q, K, V
        q, k, v = self.qkv(x).split(D, dim=2)
        q = q.view(B, T, H, hd).transpose(1, 2)   # (B, H, T, hd)
        k = k.view(B, T, H, hd).transpose(1, 2)
        v = v.view(B, T, H, hd).transpose(1, 2)

        # Apply rotary embeddings
        q, k = self.rope(q, k)

        # Scaled dot-product attention with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale          # (B, H, T, T)
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class SwiGLUFFN(nn.Module):
    """
    Feed-Forward Network with SwiGLU activation.

    SwiGLU: FFN(x) = (W1·x ⊙ swish(W2·x)) · W3
    Typically uses 2/3 * 4 * d_model for the hidden dim to keep
    parameter count comparable to a standard 4x FFN.
    """
    def __init__(self, config):
        super().__init__()
        hidden = int(2/3 * 4 * config.d_model)
        hidden = (hidden + 63) // 64 * 64  # round to multiple of 64

        self.w1 = nn.Linear(config.d_model, hidden, bias=False)
        self.w2 = nn.Linear(config.d_model, hidden, bias=False)
        self.w3 = nn.Linear(hidden, config.d_model, bias=False)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gate path
        gate = F.silu(self.w1(x))   # SiLU ≈ swish
        # Value path
        val  = self.w2(x)
        return self.drop(self.w3(gate * val))


class TransformerBlock(nn.Module):
    """One transformer decoder block: pre-norm attention + pre-norm FFN."""
    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.d_model)
        self.attn  = CausalSelfAttention(config)
        self.norm2 = RMSNorm(config.d_model)
        self.ffn   = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ─────────────────────────────────────────────
# 3.  THE LLM
# ─────────────────────────────────────────────

class LLMConfig:
    def __init__(
        self,
        vocab_size:    int = 1024,
        d_model:       int = 256,
        n_layers:      int = 4,
        n_heads:       int = 4,
        max_seq_len:   int = 512,
        dropout:       float = 0.1,
    ):
        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.n_heads     = n_heads
        self.max_seq_len = max_seq_len
        self.dropout     = dropout

    def __repr__(self):
        return (f"LLMConfig(vocab={self.vocab_size}, d={self.d_model}, "
                f"L={self.n_layers}, H={self.n_heads}, ctx={self.max_seq_len})")


class LLM(nn.Module):
    """
    GPT-style decoder-only language model.

    Notable design choices:
      • RMSNorm instead of LayerNorm (faster, no mean shift)
      • RoPE instead of learned positional embeddings (better extrapolation)
      • SwiGLU FFN (better than ReLU/GELU in practice)
      • Weight tying: token embedding = output projection (halves params,
        regularises the model, standard since GPT-2)
      • Pre-norm residual connections (more stable training)
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.emb_drop  = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )
        self.norm_out = RMSNorm(config.d_model)

        # Output head — weight-tied to token_emb
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight   # weight tying

        self._init_weights()
        print(f"[LLM] {self._param_count():,} parameters | {config}")

    def _init_weights(self):
        """GPT-2 style initialisation."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Scale residual projections by 1/√(2*n_layers) — GPT-2 trick
        for name, p in self.named_parameters():
            if "out_proj" in name or "w3" in name:
                nn.init.normal_(p, mean=0.0,
                                std=0.02 / math.sqrt(2 * self.config.n_layers))

    def _param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,          # (B, T)
        targets:   Optional[torch.Tensor] = None,  # (B, T)
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        B, T = input_ids.shape
        assert T <= self.config.max_seq_len, \
            f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}"

        x = self.emb_drop(self.token_emb(input_ids))   # (B, T, D)

        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)
        logits = self.lm_head(x)                        # (B, T, V)

        loss = None
        if targets is not None:
            # Shift: predict token i+1 from token i
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, self.config.vocab_size),
                targets[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        return logits, loss

    # ── generation ────────────────────────────
    @torch.no_grad()
    def generate(
        self,
        input_ids:   torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_p:       float = 0.9,
        eos_id:      Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with top-p (nucleus) sampling.

        top_p = 1.0  →  sample from full distribution
        top_p = 0.0  →  greedy (argmax)
        temperature → 0  →  sharper (more deterministic)
        temperature → ∞  →  flatter (more random)
        """
        self.eval()
        ids = input_ids.clone()

        for _ in range(max_new_tokens):
            # Crop context to max_seq_len
            ctx = ids[:, -self.config.max_seq_len:]
            logits, _ = self(ctx)
            logits = logits[:, -1, :]    # last token predictions

            # Temperature scaling
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)

                # Nucleus (top-p) filtering
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum_probs = sorted_probs.cumsum(dim=-1)
                    # Remove tokens beyond the nucleus
                    remove = cum_probs - sorted_probs > top_p
                    sorted_probs[remove] = 0.0
                    sorted_probs /= sorted_probs.sum(dim=-1, keepdim=True)
                    next_token = sorted_idx.gather(
                        -1, torch.multinomial(sorted_probs, 1)
                    )
                else:
                    next_token = torch.multinomial(probs, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            ids = torch.cat([ids, next_token], dim=1)

            if eos_id is not None and (next_token == eos_id).all():
                break

        return ids


# ─────────────────────────────────────────────
# 4.  DATASET  (sliding-window token chunks)
# ─────────────────────────────────────────────

class TextDataset(torch.utils.data.Dataset):
    """
    Converts a flat list of token ids into (input, target) pairs
    using a sliding window of `seq_len` tokens with `stride` step.
    """
    def __init__(self, token_ids: List[int], seq_len: int, stride: int = None):
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.seq_len = seq_len
        self.stride = stride or seq_len

    def __len__(self):
        return max(0, (len(self.ids) - self.seq_len) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        chunk = self.ids[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


# ─────────────────────────────────────────────
# 5.  TRAINER
# ─────────────────────────────────────────────

class Trainer:
    def __init__(
        self,
        model:        LLM,
        tokenizer:    BPETokenizer,
        train_ids:    List[int],
        val_ids:      Optional[List[int]] = None,
        seq_len:      int = 256,
        batch_size:   int = 16,
        lr:           float = 3e-4,
        weight_decay: float = 0.1,
        epochs:       int = 3,
        grad_clip:    float = 1.0,
        device:       str = "cpu",
        save_path:    str = "model.pt",
    ):
        self.model      = model.to(device)
        self.tokenizer  = tokenizer
        self.device     = device
        self.epochs     = epochs
        self.grad_clip  = grad_clip
        self.save_path  = save_path

        # Datasets
        train_ds = TextDataset(train_ids, seq_len, stride=seq_len // 2)
        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            pin_memory=(device != "cpu"), num_workers=0
        )
        self.val_loader = None
        if val_ids:
            val_ds = TextDataset(val_ids, seq_len)
            self.val_loader = torch.utils.data.DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            )

        # Optimiser — AdamW with weight decay (bias/norm params excluded)
        decay_params = [p for n, p in model.named_parameters()
                        if p.ndim >= 2 and p.requires_grad]
        no_decay     = [p for n, p in model.named_parameters()
                        if p.ndim  < 2 and p.requires_grad]
        self.optim = AdamW(
            [{"params": decay_params, "weight_decay": weight_decay},
             {"params": no_decay,     "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95), eps=1e-8,
        )
        total_steps = epochs * len(self.train_loader)
        self.scheduler = CosineAnnealingLR(self.optim, T_max=total_steps, eta_min=lr/10)

    def train(self):
        print(f"\n{'='*60}")
        print(f" Training for {self.epochs} epochs")
        print(f" Device: {self.device}  |  Batches/epoch: {len(self.train_loader)}")
        print(f"{'='*60}")

        best_val_loss = float("inf")
        for epoch in range(1, self.epochs + 1):
            train_loss = self._run_epoch(epoch)
            val_loss   = self._validate() if self.val_loader else None

            msg = f"Epoch {epoch}/{self.epochs}  train_loss={train_loss:.4f}"
            if val_loss:
                msg += f"  val_loss={val_loss:.4f}  perplexity={math.exp(val_loss):.1f}"
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self._save()
                    msg += "  ✓ saved"
            else:
                self._save()
            print(msg)

        print("\nTraining complete.")

    def _run_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss, steps = 0.0, 0
        t0 = time.time()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            _, loss = self.model(x, y)

            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optim.step()
            self.scheduler.step()

            total_loss += loss.item()
            steps += 1

            if (batch_idx + 1) % max(1, len(self.train_loader) // 5) == 0:
                elapsed = time.time() - t0
                print(f"  [{epoch}] step {batch_idx+1}/{len(self.train_loader)} "
                      f"loss={total_loss/steps:.4f} "
                      f"lr={self.scheduler.get_last_lr()[0]:.2e} "
                      f"({elapsed:.1f}s)")

        return total_loss / max(steps, 1)

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total, steps = 0.0, 0
        for x, y in self.val_loader:
            x, y = x.to(self.device), y.to(self.device)
            _, loss = self.model(x, y)
            total += loss.item()
            steps += 1
        return total / max(steps, 1)

    def _save(self):
        torch.save({
            "model_state": self.model.state_dict(),
            "config":      self.model.config.__dict__,
        }, self.save_path)


# ─────────────────────────────────────────────
# 6.  CLI  / DEMO
# ─────────────────────────────────────────────

DEMO_TEXT = """
The sun rose slowly over the ancient mountains, casting long golden shadows across the valley below.
A young scholar named Elara sat at her wooden desk, surrounded by towers of books and glowing candles.
She had spent years studying the mysteries of language, searching for patterns hidden within words.
One morning she discovered something extraordinary: every sentence contains a small universe of meaning.
Words are not merely symbols; they are windows into the mind of whoever wrote them.
Language models learn by reading vast amounts of text and predicting the next word, over and over.
At first the predictions are random and terrible. But gradually, through gradient descent, they improve.
The model adjusts millions of parameters, each tiny weight nudged toward better predictions.
After enough training the model begins to capture grammar, facts, reasoning, and style.
It learns that after "the cat sat on the" often comes "mat" or "floor" or "roof".
It learns that questions often end with "?" and that paragraphs have structure.
The model does not truly understand language in the way humans do, yet it learns something deep.
Inside the transformer, information flows through layers of attention and feedforward networks.
Each attention head looks for different relationships between tokens in a sequence.
Some heads track syntactic structure; others track semantic similarity or coreference.
The feedforward layers act as key-value memories, recalling facts stored during training.
Together these mechanisms allow the model to generate coherent, contextually appropriate text.
A language model is, at its core, a very sophisticated compression of human writing.
By learning to predict text it is forced to learn an enormous amount about the world.
Elara closed her book and smiled. The mathematics of language, she thought, is endlessly beautiful.
""".strip()


def run_demo():
    print("\n" + "="*60)
    print("  LLM FROM SCRATCH — DEMO")
    print("="*60)
    print("This demo trains a small GPT on a paragraph of text,")
    print("then generates a continuation. Pure from-scratch implementation.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Tokenizer ──
    tok = BPETokenizer()
    tok.train(DEMO_TEXT, vocab_size=512, verbose=True)

    # ── Tokenize corpus ──
    full_ids = tok.encode(DEMO_TEXT, add_special=False)
    split    = int(0.9 * len(full_ids))
    train_ids, val_ids = full_ids[:split], full_ids[split:]
    print(f"\nTokens: train={len(train_ids)}, val={len(val_ids)}")

    # ── Model (tiny) ──
    config = LLMConfig(
        vocab_size  = tok.vocab_size + 10,  # +10 headroom
        d_model     = 128,
        n_layers    = 3,
        n_heads     = 4,
        max_seq_len = 128,
        dropout     = 0.1,
    )
    model = LLM(config)

    # ── Train ──
    trainer = Trainer(
        model        = model,
        tokenizer    = tok,
        train_ids    = train_ids,
        val_ids      = val_ids,
        seq_len      = 64,
        batch_size   = 4,
        lr           = 5e-4,
        epochs       = 8,
        device       = device,
        save_path    = "demo_model.pt",
    )
    trainer.train()

    # ── Generate ──
    print("\n" + "-"*60)
    print("Generation (prompt: first sentence)")
    print("-"*60)
    prompt = "The sun rose slowly"
    prompt_ids = tok.encode(prompt, add_special=False)
    input_ids  = torch.tensor([prompt_ids], dtype=torch.long).to(device)

    eos_id = 256 + BPETokenizer.SPECIAL["<eos>"]
    output_ids = model.generate(
        input_ids, max_new_tokens=100,
        temperature=0.8, top_p=0.9, eos_id=eos_id
    )
    generated = tok.decode(output_ids[0].tolist())
    print(f"\n{generated}\n")


def read_text_file(path: str) -> str:
    """Read a text file, trying common encodings on Windows."""
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            with open(path, encoding=enc) as f:
                text = f.read()
            if text:
                print(f"  (encoding: {enc})")
                return text
        except (UnicodeDecodeError, LookupError):
            continue
    raise RuntimeError(f"Could not read {path} with any common encoding.")


def train_on_file(text_path: str, epochs: int, out: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Reading {text_path}...")
    text = read_text_file(text_path)
    print(f"Corpus: {len(text):,} characters")

    if len(text) < 500:
        raise RuntimeError(
            f"Corpus too small ({len(text)} chars). "
            "Add more text to your training file."
        )

    tok = BPETokenizer()
    # Scale vocab size to corpus size so BPE merges don't outnumber available pairs
    vocab_size = min(4096, max(512, len(text) // 5))
    tok.train(text, vocab_size=vocab_size)
    tok.save(out.replace(".pt", "_tokenizer.json"))

    full_ids = tok.encode(text, add_special=False)
    print(f"Tokenized: {len(full_ids):,} tokens")

    if len(full_ids) < 64:
        raise RuntimeError(
            f"Too few tokens ({len(full_ids)}) after tokenization. "
            "Your training file needs more text."
        )

    # Auto-scale seq_len and batch_size to corpus size
    seq_len    = min(256, len(full_ids) // 4)
    batch_size = min(8, max(1, len(full_ids) // (seq_len * 4)))
    print(f"Auto-config: seq_len={seq_len}, batch_size={batch_size}")

    split     = int(0.95 * len(full_ids))
    train_ids = full_ids[:split]
    val_ids   = full_ids[split:] if split < len(full_ids) else None

    # Warn if val set is too small
    if val_ids and len(val_ids) < seq_len + 1:
        print("  (val set too small for seq_len — skipping validation)")
        val_ids = None

    config = LLMConfig(
        vocab_size  = tok.vocab_size + 10,
        d_model     = 512,
        n_layers    = 6,
        n_heads     = 8,
        max_seq_len = max(seq_len + 1, 512),
        dropout     = 0.1,
    )
    model = LLM(config)

    Trainer(
        model=model, tokenizer=tok,
        train_ids=train_ids, val_ids=val_ids,
        seq_len=seq_len, batch_size=batch_size, lr=3e-4,
        epochs=epochs, device=device, save_path=out,
    ).train()


def generate_from_model(prompt: str, model_path: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(model_path, map_location=device)
    config = LLMConfig(**ckpt["config"])
    model  = LLM(config)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    tok_path = model_path.replace(".pt", "_tokenizer.json")
    tok = BPETokenizer.load(tok_path)

    prompt_ids = tok.encode(prompt, add_special=False)
    input_ids  = torch.tensor([prompt_ids], dtype=torch.long).to(device)
    eos_id = 256 + BPETokenizer.SPECIAL["<eos>"]
    out_ids = model.generate(input_ids, max_new_tokens=200,
                              temperature=0.8, top_p=0.9, eos_id=eos_id)
    print(tok.decode(out_ids[0].tolist()))


# ─────────────────────────────────────────────
# 7.  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM From Scratch")
    parser.add_argument("--demo",     action="store_true",
                        help="Run built-in demo (trains on a tiny corpus)")
    parser.add_argument("--train",    type=str, metavar="FILE",
                        help="Train on a text file")
    parser.add_argument("--epochs",   type=int, default=5)
    parser.add_argument("--out",      type=str, default="model.pt",
                        help="Output model path")
    parser.add_argument("--generate", type=str, metavar="PROMPT",
                        help="Generate text from a prompt")
    parser.add_argument("--model",    type=str, default="model.pt",
                        help="Model checkpoint path (for --generate)")
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.train:
        train_on_file(args.train, args.epochs, args.out)
    elif args.generate:
        generate_from_model(args.generate, args.model)
    else:
        parser.print_help()
        print("\nQuick start: python llm_from_scratch.py --demo")
