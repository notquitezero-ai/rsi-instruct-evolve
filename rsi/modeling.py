"""Model loading and fast batched generation (HF transformers, bf16, SDPA)."""
from __future__ import annotations

import gc
import math
import re
import time
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import IM_END, chat_prefix

# On Windows the driver silently spills to system RAM when VRAM runs out, which turns an OOM
# into a 20x slowdown.  Capping the caching allocator makes such runs fail fast instead.
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(0.9)


def load_tokenizer(base: str):
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = "<|endoftext|>"
    tok.padding_side = "left"
    return tok


def load_model(base: str, adapter: Optional[str] = None, merge: bool = True):
    tok = load_tokenizer(base)
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        if merge:
            model = model.merge_and_unload()
    model.eval()
    model.generation_config.pad_token_id = tok.pad_token_id
    return model, tok


def free() -> None:
    """Call after the caller has ``del``-ed its model references."""
    gc.collect()
    torch.cuda.empty_cache()


def stop_ids(tok) -> list[int]:
    ids = {tok.convert_tokens_to_ids(IM_END), tok.convert_tokens_to_ids("<|endoftext|>")}
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    return sorted(i for i in ids if i is not None and i >= 0)


def is_degenerate(text: str, finished: bool) -> bool:
    """Unfinished (hit max tokens) or dominated by a repeated n-gram."""
    if not finished:
        return True
    words = text.split()
    if len(words) < 30:
        return False
    grams = [" ".join(words[i:i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) / max(1, len(grams)) < 0.5


@torch.inference_mode()
def generate(model, tok, prompts: list[str], max_new_tokens: int = 512, temperature: float = 0.0,
             top_p: float = 0.95, n: int = 1, token_budget: int = 96_000, desc: str = "gen",
             log=print) -> list[list[dict]]:
    """Generate ``n`` completions per prompt.

    Returns ``out[i] = [{"text": str, "finished": bool, "tokens": int}, ...]``.
    Prompts are length-sorted and batched by a token budget so small models use the GPU fully.
    """
    texts = [chat_prefix(p) for p in prompts]
    enc = [tok(t, add_special_tokens=False)["input_ids"] for t in texts]
    order = sorted(range(len(texts)), key=lambda i: -len(enc[i]))
    stops = stop_ids(tok)
    out: list[list[dict]] = [[] for _ in prompts]
    do_sample = temperature > 0
    i, t0, done = 0, time.time(), 0
    while i < len(order):
        longest = len(enc[order[i]]) + max_new_tokens
        bs = max(1, min(256, token_budget // (longest * n)))
        idx = order[i:i + bs]
        i += bs
        batch = tok.pad({"input_ids": [enc[j] for j in idx]}, return_tensors="pt").to(model.device)
        gen = model.generate(**batch, max_new_tokens=max_new_tokens, do_sample=do_sample,
                             temperature=temperature if do_sample else None,
                             top_p=top_p if do_sample else None, top_k=None,
                             num_return_sequences=n, eos_token_id=stops, pad_token_id=tok.pad_token_id)
        new = gen[:, batch["input_ids"].shape[1]:].tolist()
        for b, seq in enumerate(new):
            j = idx[b // n]
            finished = any(s in stops for s in seq)
            cut = next((k for k, s in enumerate(seq) if s in stops), len(seq))
            text = tok.decode(seq[:cut], skip_special_tokens=True).strip()
            out[j].append({"text": text, "finished": finished, "tokens": cut})
        done += len(idx)
        if log:
            el = time.time() - t0
            log(f"[{desc}] {done}/{len(prompts)} prompts x{n}  {el:.0f}s")
    return out
