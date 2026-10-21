"""Seed data, fixed evaluation sets and prompt formatting.

Splits are deterministic and disjoint:
  * alpaca-cleaned -> eval task pools (dev / test, never trained on) + train pool
  * gsm8k test     -> math dev / test;  gsm8k train -> gold seed + self-play questions
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from .constraints import Constraint, render_prompt, sample_constraints
from .paths import DATA

# ChatML is the native Qwen format; the base model has the tokens but has never been
# taught to use them -- learning that is part of what instruction tuning delivers.
IM_START, IM_END = "<|im_start|>", "<|im_end|>"
# Responses are terminated with <|endoftext|>, not <|im_end|>: in Qwen3-Base the (tied) <|im_end|>
# embedding is effectively untrained -- P(im_end) ~ 0 even after LoRA SFT, so a model taught to stop
# on it never stops.  <|endoftext|> is the stop token the base model already uses.
RESPONSE_END = "<|endoftext|>"


def chat_prefix(prompt: str) -> str:
    return f"{IM_START}user\n{prompt}{IM_END}\n{IM_START}assistant\n"


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------------------
# raw dataset preparation (run once)
# ---------------------------------------------------------------------------------------
def _alpaca_prompt(row: dict) -> str:
    ins = row["instruction"].strip()
    inp = (row.get("input") or "").strip()
    return ins + ("\n\n" + inp if inp else "")


def _gsm_answer(ans: str) -> tuple[str, str]:
    body, final = ans.split("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    final = final.strip().replace(",", "")
    return body, final


MATH_SUFFIX = "\n\nSolve the problem step by step, then give the final answer on the last line as \"The answer is N.\""


def prepare(seed: int = 1234, n_eval_tasks: int = 200, n_gsm_eval: int = 150) -> dict:
    """Download the public datasets and materialise every split under data/."""
    from datasets import load_dataset

    out = DATA / "prepared"
    if (out / "manifest.json").exists():
        return json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    alpaca = load_dataset("yahma/alpaca-cleaned", split="train")
    rows = []
    for r in alpaca:
        p, o = _alpaca_prompt(r), r["output"].strip()
        if not o or len(p) > 1500 or len(o) > 3000:
            continue
        rows.append({"prompt": p, "response": o, "open": len(o.split()) >= 40 and not r["input"].strip()})
    rng.shuffle(rows)
    open_rows = [r for r in rows if r["open"]]
    held = open_rows[: 2 * n_eval_tasks]
    held_set = {r["prompt"] for r in held}
    train = [r for r in rows if r["prompt"] not in held_set]
    write_jsonl(out / "alpaca_train.jsonl", train)

    # IF eval: held-out open-ended tasks + 1..3 random verifiable constraints (fixed seed)
    def make_if(tasks, tag):
        g = random.Random(f"{seed}-{tag}")
        items = []
        for i, t in enumerate(tasks):
            cons = sample_constraints(g, g.choice([1, 2, 2, 3]))
            items.append({"id": f"if-{tag}-{i}", "task": t["prompt"],
                          "prompt": render_prompt(t["prompt"], cons),
                          "constraints": [c.to_dict() for c in cons]})
        return items

    write_jsonl(out / "if_dev.jsonl", make_if(held[:n_eval_tasks], "dev"))
    write_jsonl(out / "if_test.jsonl", make_if(held[n_eval_tasks:], "test"))

    gsm_tr = load_dataset("openai/gsm8k", "main", split="train")
    gsm_te = load_dataset("openai/gsm8k", "main", split="test")
    tr = []
    for i, r in enumerate(gsm_tr):
        body, final = _gsm_answer(r["answer"])
        tr.append({"id": f"gsm-tr-{i}", "question": r["question"], "solution": body, "answer": final})
    te = []
    for i, r in enumerate(gsm_te):
        _, final = _gsm_answer(r["answer"])
        te.append({"id": f"gsm-te-{i}", "question": r["question"], "answer": final})
    rng.shuffle(tr)
    rng.shuffle(te)
    write_jsonl(out / "gsm_train.jsonl", tr)
    write_jsonl(out / "gsm_dev.jsonl", te[:n_gsm_eval])
    write_jsonl(out / "gsm_test.jsonl", te[n_gsm_eval: 2 * n_gsm_eval])

    manifest = {"seed": seed, "alpaca_train": len(train), "if_dev": n_eval_tasks, "if_test": n_eval_tasks,
                "gsm_train": len(tr), "gsm_dev": n_gsm_eval, "gsm_test": n_gsm_eval}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_split(name: str) -> list[dict]:
    return read_jsonl(DATA / "prepared" / f"{name}.jsonl")


def gsm_spare(seed: int = 4321) -> list[dict]:
    """gsm8k test items used by neither gsm_dev nor gsm_test (~1000), cached on first use."""
    path = DATA / "prepared" / "gsm_spare.jsonl"
    if path.exists():
        return read_jsonl(path)
    from datasets import load_dataset
    used = {x["question"] for x in load_split("gsm_dev") + load_split("gsm_test")}
    rows = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test")):
        if r["question"] not in used:
            rows.append({"id": f"gsm-sp-{i}", "question": r["question"], "answer": _gsm_answer(r["answer"])[1]})
    random.Random(seed).shuffle(rows)
    write_jsonl(path, rows)
    return rows


def gate_set(gen: int, n_math: int = 150, seed: int = 1234) -> dict:
    """Fresh promotion-gate items for generation ``gen`` (winner's-curse fix).

    A champion was *selected* by its dev score, so that score is optimistically biased and every
    challenger is compared against a lucky number.  The gate therefore re-measures champion and
    candidates on items none of them was selected on: held-out dev tasks with freshly sampled
    constraints, plus a rotating slice of otherwise unused gsm8k test questions.
    """
    g = random.Random(f"{seed}-gate-{gen}")
    ifs = []
    for i, t in enumerate(load_split("if_dev")):
        cons = sample_constraints(g, g.choice([1, 2, 2, 3]))
        ifs.append({"id": f"if-gate{gen}-{i}", "task": t["task"], "prompt": render_prompt(t["task"], cons),
                    "constraints": [c.to_dict() for c in cons]})
    spare = gsm_spare()
    k = (gen * n_math) % len(spare)
    return {"if": ifs, "math": (spare + spare)[k:k + n_math]}


def eval_set(split: str) -> dict:
    """split in {'dev','test'} -> {'if': [...], 'math': [...]}"""
    return {"if": load_split(f"if_{split}"), "math": load_split(f"gsm_{split}")}


def math_prompt(q: str) -> str:
    return q.strip() + MATH_SUFFIX


def math_target(solution: str, answer: str) -> str:
    return solution.strip() + f"\nThe answer is {answer}."


_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def extract_answer(text: str) -> str | None:
    m = re.search(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text, flags=re.IGNORECASE)
    if m:
        s = m.group(1)
    else:
        nums = _NUM.findall(text)
        if not nums:
            return None
        s = nums[-1]
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return None


def answers_match(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except ValueError:
        return False


def constraints_of(item: dict) -> list[Constraint]:
    return [Constraint.from_dict(c) for c in item["constraints"]]
