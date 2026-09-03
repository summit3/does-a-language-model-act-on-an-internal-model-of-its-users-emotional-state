"""Model loading and single-turn chat for the user-model project.

The model id lives in src/config.py (MODEL_ID). Nothing here assumes a particular
architecture: loading tries the multimodal AutoModel first (Qwen3.5/3.6 ship as
image-text-to-text checkpoints wrapping a text-only language model) and falls back
to AutoModelForCausalLM (Qwen3 and other plain decoders). We only ever feed text.
"""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

from .config import MODEL_ID

DEFAULT_MODEL = MODEL_ID


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(name: str = DEFAULT_MODEL, dtype: torch.dtype = torch.bfloat16, device=None):
    """Load `name` in `dtype` on MPS (or CUDA/CPU) and return (model, tokenizer)."""
    device = torch.device(device) if device is not None else pick_device()
    tokenizer = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForImageTextToText.from_pretrained(name, dtype=dtype)
    except (ValueError, KeyError, OSError):
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    model.to(device).eval()
    model.generation_config.do_sample = False
    return model, tokenizer


def get_decoder_layers(model) -> torch.nn.ModuleList:
    """Return the ModuleList of decoder blocks, whatever the wrapper structure is."""
    for path in ("model.language_model.layers", "model.layers", "language_model.model.layers",
                 "language_model.layers", "transformer.h", "layers"):
        obj = model
        try:
            for attr in path.split("."):
                obj = getattr(obj, attr)
        except AttributeError:
            continue
        if isinstance(obj, torch.nn.ModuleList):
            return obj
    raise AttributeError("could not locate decoder layers on this model")


def build_messages(user_message: str, system: str | None = None) -> list[dict]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_message})
    return messages


def encode_prompt(tokenizer, user_message: str, system: str | None = None,
                  add_generation_prompt: bool = True, enable_thinking: bool = False):
    """Apply the chat template and return a dict of tensors (input_ids, attention_mask).

    With add_generation_prompt=True the last token is the one whose residual stream
    predicts the first reply token (Chen et al.'s control-probe position, after the
    assistant header). With False the sequence ends at the user turn's <|im_end|>.
    """
    return tokenizer.apply_chat_template(
        build_messages(user_message, system),
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        return_tensors="pt",
        return_dict=True,
    )


def _strip_thinking(text: str) -> str:
    # Qwen3.x may emit "<think>\n\n</think>\n\n" even with thinking disabled.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.replace("<think>", "").strip()


@torch.inference_mode()
def chat(model, tokenizer, user_message: str, system: str | None = None,
         max_new_tokens: int = 200, enable_thinking: bool = False) -> str:
    """Greedy single-turn reply using the model's chat template."""
    enc = encode_prompt(tokenizer, user_message, system, enable_thinking=enable_thinking)
    enc = {k: v.to(model.device) for k, v in enc.items()}
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None, top_p=None, top_k=None,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = out[0, enc["input_ids"].shape[1]:]
    return _strip_thinking(tokenizer.decode(new_tokens, skip_special_tokens=True))


if __name__ == "__main__":
    import sys, time
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL
    t0 = time.time()
    model, tok = load_model(name)
    print(f"loaded {name} on {model.device} in {time.time()-t0:.0f}s; "
          f"{len(get_decoder_layers(model))} layers, hidden={model.config.get_text_config().hidden_size}")
    t0 = time.time()
    print(chat(model, tok, "Explain how compound interest works in two sentences."))
    print(f"[{time.time()-t0:.1f}s]")
