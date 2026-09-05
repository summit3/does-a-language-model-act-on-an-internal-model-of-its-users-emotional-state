"""Residual-stream extraction and activation steering via PyTorch forward hooks.

LAYER CONVENTION (used everywhere in this repo): activation index i = the OUTPUT of decoder
block i, zero-based, embeddings excluded. So an activation tensor has first dim
model.config.num_hidden_layers, index 0 is the residual stream after block 0, and index
n_layers-1 is the residual stream after the last block (before the final norm). This equals
HuggingFace's output_hidden_states[i + 1]. Steering at "layer i" adds to that same tensor.
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
    acts = torch.stack([store[i] for i in range(len(layers))])
    n_expected = model.config.get_text_config().num_hidden_layers
    assert acts.shape[0] == n_expected, f"got {acts.shape[0]} layers, config says {n_expected}"
    return acts


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


# ---------------------------------------------------------------------------
# Introspection: where exactly are the hooks, and what kind of block is each layer?
# ---------------------------------------------------------------------------

def describe_layers(model, tokenizer=None, prompt: str | None = None) -> dict:
    """Print (and return) the resolved decoder-layer path, the class/children of blocks 0 and 1,
    the per-block type (linear attention / DeltaNet vs full attention) from the config, and, if a
    tokenizer+prompt are given, a check that hooked activations equal output_hidden_states[i+1].
    """
    from .model import get_decoder_layers
    blocks = get_decoder_layers(model)
    # find the attribute path that get_decoder_layers resolved to
    path = None
    for cand in ("model.language_model.layers", "model.layers", "language_model.model.layers",
                 "language_model.layers", "transformer.h", "layers"):
        obj = model
        try:
            for a in cand.split("."):
                obj = getattr(obj, a)
        except AttributeError:
            continue
        if obj is blocks:
            path = cand
            break
    cfg = model.config.get_text_config()
    layer_types = list(getattr(cfg, "layer_types", None) or ["full_attention"] * len(blocks))
    info = {"path": path, "n_blocks": len(blocks), "num_hidden_layers": cfg.num_hidden_layers,
            "layer_types": layer_types,
            "block0": type(blocks[0]).__name__, "block1": type(blocks[1]).__name__,
            "block0_children": [f"{n}:{type(m).__name__}" for n, m in blocks[0].named_children()],
            "block1_children": [f"{n}:{type(m).__name__}" for n, m in blocks[1].named_children()]}
    print(f"decoder layers: model.{path}  ({len(blocks)} blocks; config num_hidden_layers={cfg.num_hidden_layers})")
    print(f"block 0: model.{path}[0] = {info['block0']} children {info['block0_children']}")
    print(f"block 1: model.{path}[1] = {info['block1']} children {info['block1_children']}")
    lin = [i for i, t in enumerate(layer_types) if "linear" in t]
    full = [i for i, t in enumerate(layer_types) if "linear" not in t]
    print(f"linear-attention (DeltaNet) blocks ({len(lin)}): {lin}")
    print(f"full-attention blocks ({len(full)}): {full}")
    info["linear_attention_blocks"], info["full_attention_blocks"] = lin, full
    if tokenizer is not None and prompt is not None:
        with torch.inference_mode():
            enc = encode_prompt(tokenizer, prompt)
            enc = {k: v.to(model.device) for k, v in enc.items()}
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            acts = get_residual_activations(model, tokenizer, prompt)
            # HF's last hidden_states entry is AFTER the final RMSNorm, so compare norm(hook) there.
            final_norm = next((m for n, m in model.named_modules() if n.endswith("norm") and n.count(".") <= 3
                               and type(m).__name__.endswith("RMSNorm") and "layers" not in n), None)
            diffs = []
            for i in range(len(blocks)):
                ref = hs[i + 1][0, -1].float().cpu()
                mine = acts[i]
                if i == len(blocks) - 1 and final_norm is not None:
                    mine = final_norm(acts[i].to(device=model.device, dtype=hs[-1].dtype)).float().cpu()
                diffs.append(float((mine - ref).abs().max()))
        info["max_abs_diff_vs_hidden_states"] = max(diffs)
        print(f"hooks == output_hidden_states[i+1] at last token (final layer compared after the model's "
              f"final norm): max |diff| over layers = {max(diffs):.3g} "
              f"(len(hidden_states)={len(hs)} = embeddings + {len(blocks)} blocks)")
    return info
