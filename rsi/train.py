"""LoRA supervised fine-tuning with prompt-masked loss and token-budget batching."""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup

from .data import RESPONSE_END, chat_prefix
from .modeling import free, load_tokenizer

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tokenize(tok, rows: list[dict], max_len: int) -> list[dict]:
    feats = []
    for r in rows:
        p = tok(chat_prefix(r["prompt"]), add_special_tokens=False)["input_ids"]
        a = tok(r["response"] + RESPONSE_END, add_special_tokens=False)["input_ids"]
        ids = (p + a)[:max_len]
        labels = ([-100] * len(p) + a)[:max_len]
        if sum(1 for x in labels if x != -100) < 2:
            continue
        feats.append({"input_ids": ids, "labels": labels})
    return feats


def batches(feats: list[dict], token_budget: int, rng: random.Random) -> list[list[int]]:
    """Group similar lengths together (less padding), then shuffle the groups."""
    order = sorted(range(len(feats)), key=lambda i: len(feats[i]["input_ids"]) + rng.random())
    out, cur, cur_max = [], [], 0
    for i in order:
        L = len(feats[i]["input_ids"])
        if cur and max(cur_max, L) * (len(cur) + 1) > token_budget:
            out.append(cur)
            cur, cur_max = [], 0
        cur.append(i)
        cur_max = max(cur_max, L)
    if cur:
        out.append(cur)
    rng.shuffle(out)
    return out


def collate(feats, idx, pad_id, device):
    L = max(len(feats[i]["input_ids"]) for i in idx)
    ids = torch.full((len(idx), L), pad_id, dtype=torch.long)
    lab = torch.full((len(idx), L), -100, dtype=torch.long)
    att = torch.zeros((len(idx), L), dtype=torch.long)
    for b, i in enumerate(idx):
        x, y = feats[i]["input_ids"], feats[i]["labels"]
        ids[b, :len(x)] = torch.tensor(x)
        lab[b, :len(y)] = torch.tensor(y)
        att[b, :len(x)] = 1
    return ids.to(device), lab.to(device), att.to(device)


def response_loss(lm, ids, att, lab):
    """Causal-LM loss evaluated only where labels exist.

    Running the 151k-vocab LM head on prompt and padding tokens dominates memory for a small
    model (16k tokens -> ~10 GB of fp32 logits), so project only the supervised positions.
    """
    h = lm.model(input_ids=ids, attention_mask=att).last_hidden_state[:, :-1]
    tgt = lab[:, 1:]
    m = tgt != -100
    hs, ts = h[m], tgt[m]
    # chunk + recompute so at most CHUNK x vocab logits are alive at once (fwd and bwd)
    total = 0.0
    for i in range(0, hs.shape[0], CHUNK):
        total = total + checkpoint(_ce_sum, hs[i:i + CHUNK], lm.lm_head.weight, ts[i:i + CHUNK], use_reentrant=False)
    return total / max(1, hs.shape[0])


CHUNK = 2048


def _ce_sum(h, w, t):
    return torch.nn.functional.cross_entropy(torch.nn.functional.linear(h, w).float(), t, reduction="sum")


def train_lora(base: str, rows: list[dict], out_dir: Path, recipe: dict, init_adapter: Optional[str] = None,
               seed: int = 0, log=print) -> dict:
    from peft import LoraConfig, PeftModel, get_peft_model

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    tok = load_tokenizer(base)
    tok.padding_side = "right"
    feats = tokenize(tok, rows, recipe.get("max_len", 1024))

    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 device_map={"": 0})
    if recipe.get("grad_ckpt", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        r = int(recipe["lora_r"])
        cfg = LoraConfig(r=r, lora_alpha=int(r * recipe.get("alpha_ratio", 2)), lora_dropout=recipe.get("dropout", 0.05),
                         target_modules=TARGETS, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    # keep trainable LoRA weights in fp32 for stable optimisation; forward runs under bf16 autocast
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    n_train = sum(p.numel() for p in params)

    budget = int(recipe.get("token_budget", 16384))
    accum = int(recipe.get("grad_accum", 1))
    epochs = float(recipe["epochs"])
    plan = []
    for e in range(math.ceil(epochs)):
        b = batches(feats, budget, rng)
        if e + 1 > epochs:  # fractional last epoch
            b = b[: int(len(b) * (epochs - e))]
        plan.extend(b)
    steps = math.ceil(len(plan) / accum)
    opt = torch.optim.AdamW(params, lr=float(recipe["lr"]), weight_decay=0.0, betas=(0.9, 0.999))
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps) + 1, steps)

    model.train()
    lm = model.get_base_model()
    t0, hist, run_loss, run_n = time.time(), [], 0.0, 0
    for k, idx in enumerate(plan):
        ids, lab, att = collate(feats, idx, tok.pad_token_id, model.device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = response_loss(lm, ids, att, lab)
        (loss / accum).backward()
        run_loss += loss.item()
        run_n += 1
        if (k + 1) % accum == 0 or k == len(plan) - 1:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step = (k + 1) // accum
            if step % 20 == 0 or k == len(plan) - 1:
                hist.append({"step": step, "loss": run_loss / run_n, "lr": sched.get_last_lr()[0],
                             "t": round(time.time() - t0, 1)})
                if log:
                    log(f"[train] step {step}/{steps} loss {run_loss / run_n:.4f} {time.time() - t0:.0f}s")
                run_loss, run_n = 0.0, 0
    model.save_pretrained(out_dir / "adapter")
    stats = {"examples": len(feats), "steps": steps, "trainable_params": n_train, "seconds": round(time.time() - t0, 1),
             "final_loss": hist[-1]["loss"] if hist else None, "loss_curve": hist}
    (out_dir / "train_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    del model, opt, sched, params
    free()
    return stats
