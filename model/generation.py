"""Standalone temperature, top-k, and nucleus sampling utilities."""

from __future__ import annotations

import torch


def _extract_logits(model_output: object) -> torch.Tensor:
    if isinstance(model_output, torch.Tensor):
        return model_output
    if isinstance(model_output, dict):
        logits = model_output.get("logits")
        if isinstance(logits, torch.Tensor):
            return logits
    if isinstance(model_output, (tuple, list)) and model_output:
        logits = model_output[0]
        if isinstance(logits, torch.Tensor):
            return logits
    raise TypeError("model output must contain a logits tensor")


def _resolve_eos_token_id(model: torch.nn.Module, eos_token_id: int | None) -> int | None:
    if eos_token_id is not None:
        return int(eos_token_id)
    for source in (model, getattr(model, "tokenizer", None)):
        if source is None:
            continue
        for attribute in ("eos_token_id", "eos_id"):
            value = getattr(source, attribute, None)
            if value is not None:
                return int(value)
        token_to_id = getattr(source, "token_to_id", None)
        if callable(token_to_id):
            value = token_to_id("<EOS>")
            if value is not None:
                return int(value)
        elif isinstance(token_to_id, dict):
            value = token_to_id.get("<EOS>")
            if value is not None:
                return int(value)
    return None


def generate(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.9,
    repetition_penalty: float = 1.2,
    eos_token_id: int | None = None,
    device: torch.device | str | None = None,
    min_probability: float = 0.0,
) -> torch.Tensor:
    """Generate token IDs with temperature, top-k, and nucleus sampling.

    ``prompt_ids`` may have shape ``(sequence_length,)`` or
    ``(batch_size, sequence_length)``. The returned tensor has the same batch
    shape with up to ``max_new_tokens`` appended to each prompt.
    """
    if not isinstance(prompt_ids, torch.Tensor):
        prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long)
    if prompt_ids.ndim == 1:
        prompt_ids = prompt_ids.unsqueeze(0)
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape (sequence_length,) or (batch_size, sequence_length)")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be greater than zero")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in the interval (0, 1]")
    if repetition_penalty < 1:
        raise ValueError("repetition_penalty must be at least 1.0")
    if min_probability < 0:
        raise ValueError("min_probability must be non-negative")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = prompt_ids.device
    else:
        device = torch.device(device)
    prompt_ids = prompt_ids.to(device)

    generated = prompt_ids.clone()
    eos_token_id = _resolve_eos_token_id(model, eos_token_id)
    finished = torch.zeros(generated.size(0), dtype=torch.bool, device=generated.device)
    model_was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = _extract_logits(model(generated))
                if logits.ndim != 3:
                    raise ValueError("model logits must have shape (batch_size, sequence_length, vocab_size)")
                next_token_logits = torch.nan_to_num(
                    logits[:, -1, :], nan=0.0, posinf=80.0, neginf=-80.0
                )
                if repetition_penalty > 1.0:
                    for batch_index in range(generated.size(0)):
                        previous_token_ids = generated[batch_index].unique()
                        previous_logits = next_token_logits[batch_index, previous_token_ids]
                        next_token_logits[batch_index, previous_token_ids] = torch.where(
                            previous_logits > 0,
                            previous_logits / repetition_penalty,
                            previous_logits * repetition_penalty,
                        )
                next_token_logits = next_token_logits / max(float(temperature), 1e-5)

                vocabulary_size = next_token_logits.size(-1)
                if top_k > 0:
                    k = min(int(top_k), vocabulary_size)
                    top_values, top_indices = torch.topk(next_token_logits, k, dim=-1)
                    filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                    filtered_logits.scatter_(1, top_indices, top_values)
                else:
                    filtered_logits = next_token_logits

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        filtered_logits, descending=True, dim=-1
                    )
                    sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
                    cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)
                    remove_indices = cumulative_probabilities > top_p
                    remove_indices[..., 1:] = remove_indices[..., :-1].clone()
                    remove_indices[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove_indices, float("-inf"))
                    filtered_logits = torch.full_like(next_token_logits, float("-inf"))
                    filtered_logits.scatter_(1, sorted_indices, sorted_logits)

                if min_probability > 0:
                    candidate_probabilities = torch.softmax(filtered_logits, dim=-1)
                    filtered_logits = filtered_logits.masked_fill(
                        candidate_probabilities < min_probability,
                        float("-inf"),
                    )

                valid_candidates = torch.isfinite(filtered_logits).any(dim=-1)
                if not valid_candidates.all():
                    fallback_indices = next_token_logits.argmax(dim=-1)
                    filtered_logits[~valid_candidates] = float("-inf")
                    filtered_logits[~valid_candidates, fallback_indices[~valid_candidates]] = 0.0

                probabilities = torch.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                if eos_token_id is not None:
                    finished |= next_token.squeeze(1).eq(eos_token_id)
                    if finished.all():
                        generated = torch.cat((generated, next_token), dim=1)
                        break
                generated = torch.cat((generated, next_token), dim=1)
    finally:
        model.train(model_was_training)

    return generated
