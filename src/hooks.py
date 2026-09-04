"""Residual-stream extraction and activation steering via PyTorch forward hooks.

Both functions hook the *output* of each decoder block, i.e. the residual stream
after that block. Layer index i therefore means "residual stream after block i".
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import torch

from .config import ENABLE_THINKING
from .model import encode_prompt, get_decoder_layers


def _hidden(output):
    """Decoder blocks return either a tensor or a tuple whose first item is the tensor."""
    return output[0] if isinstance(output, (tuple, list)) else output


def _replace_hidden(output, new):
    if isinstance(output, tuple):
        return (new,) + tuple(output[1:])
    if isinstance(output, list):
        return [new] + list(output[1:])
    return new


@torch.inference_mode()
def get_residual_activations(model, tokenizer, prompt: str, system: str | None = None,
                             add_generation_prompt: bool = True, token_index: int = -1,
                             enable_thinking: bool = ENABLE_THINKING) -> torch.Tensor:
    """Run `prompt` once and return the residual stream at `token_index` (default: last
    prompt token) after every decoder layer, as a float32 CPU tensor [n_layers, hidden_dim].
    """
    layers = get_decoder_layers(model)
    store: dict[int, torch.Tensor] = {}
    handles = []
    for i, layer in enumerate(layers):
        def hook(module, args, output, i=i):
            store[i] = _hidden(output)[0, token_index, :].detach().float().cpu()
        handles.append(layer.register_forward_hook(hook))
    try:
        enc = encode_prompt(tokenizer, prompt, system, add_generation_prompt, enable_thinking)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return torch.stack([store[i] for i in range(len(layers))])


@contextmanager
def steering(model, vector: torch.Tensor, layers: Iterable[int], N: float = 1.0):
    """Context manager: add N * vector to the residual stream after each layer in `layers`,
    at every position on every forward pass (prompt pass and each generation step).

    `vector` is either [hidden_dim] (same direction at every chosen layer) or
    [n_layers, hidden_dim] (row `layer` used at each chosen layer, e.g. per-layer probe weights).
    """
    blocks = get_decoder_layers(model)
    layers = list(layers)
    handles = []

    def make_hook(layer_idx):
        v = vector[layer_idx] if vector.ndim == 2 else vector

        def hook(module, args, output):
            h = _hidden(output)
            delta = (N * v).to(device=h.device, dtype=h.dtype)
            return _replace_hidden(output, h + delta)
        return hook

    try:
        for li in layers:
            handles.append(blocks[li].register_forward_hook(make_hook(li)))
        yield
    finally:
        for h in handles:
            h.remove()


@torch.inference_mode()
def steer_generate(model, tokenizer, prompt: str, vector: torch.Tensor, layers: Iterable[int],
                   N: float = 1.0, system: str | None = None, max_new_tokens: int = 200,
                   enable_thinking: bool = ENABLE_THINKING) -> str:
    """Greedy generation with N * vector added to the residual stream at `layers`."""
    from .model import _strip_thinking
    enc = encode_prompt(tokenizer, prompt, system, enable_thinking=enable_thinking)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    with steering(model, vector, layers, N):
        out = model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False,
            temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return _strip_thinking(tokenizer.decode(new_tokens, skip_special_tokens=True))
