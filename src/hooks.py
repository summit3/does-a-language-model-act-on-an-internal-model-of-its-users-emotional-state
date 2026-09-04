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


# ---------------------------------------------------------------------------
# Steering strength relative to the residual norm, so calibration transfers between models.
# ---------------------------------------------------------------------------

@torch.inference_mode()
def residual_norms(model, tokenizer, prompt: str, system: str | None = None) -> torch.Tensor:
    """L2 norm of the residual stream at the last prompt token, per layer: [n_layers]."""
    return get_residual_activations(model, tokenizer, prompt, system).norm(dim=-1)


def relative_to_absolute_N(frac: float, norms: torch.Tensor, layers: Iterable[int],
                           vector: torch.Tensor) -> float:
    """Convert a strength given as a fraction of the mean residual norm at `layers` into the
    absolute N used by `steering`, for a `vector` of any norm (N * ||v|| == frac * mean_norm).
    `norms` is [n_layers] (e.g. from `residual_norms`, or averaged over a set of prompts).
    """
    layers = list(layers)
    mean_norm = norms[layers].mean().item()
    vnorm = (vector[layers].norm(dim=-1).mean() if vector.ndim == 2 else vector.norm()).item()
    return frac * mean_norm / vnorm


@torch.inference_mode()
def steer_generate_relative(model, tokenizer, prompt: str, vector: torch.Tensor, layers: Iterable[int],
                            frac: float, norms: torch.Tensor | None = None, system: str | None = None,
                            max_new_tokens: int = 200, enable_thinking: bool = ENABLE_THINKING,
                            return_N: bool = False):
    """Like `steer_generate`, but the strength is `frac` x (mean residual norm at `layers`).

    `norms` defaults to this prompt's own per-layer norms; pass a dataset-averaged [n_layers]
    tensor to keep N fixed across prompts. With return_N=True returns (text, absolute_N).
    Playground reference on Qwen3-1.7B, layers 12-23 (mean last-token norm over the band ~530):
    N=8 (frac ~0.015) kept the fact, N=15 (frac ~0.03) abandoned the task for emotional support,
    N=25-35 (frac ~0.05-0.07) was incoherent, N=50+ (frac ~0.1) degenerate.
    """
    layers = list(layers)
    if norms is None:
        norms = residual_norms(model, tokenizer, prompt, system)
    N = relative_to_absolute_N(frac, norms, layers, vector)
    text = steer_generate(model, tokenizer, prompt, vector, layers, N=N, system=system,
                          max_new_tokens=max_new_tokens, enable_thinking=enable_thinking)
    return (text, N) if return_N else text
