"""Self-data engine: the current champion generates, verifiers filter, survivors become training data.

Sources
  * ``self_if``   -- champion answers constraint-augmented prompts; keep only fully verified answers
                     (rejection-sampling fine-tuning / ReST-EM).  Constraint kinds are sampled with a
                     curriculum that over-weights the champion's weakest kinds.
  * ``self_math`` -- champion solves gsm8k-train questions; keep solutions with the correct answer (STaR).
  * ``rule``      -- seed responses rewritten programmatically to satisfy constraints (cheap prior).
"""
from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

from .constraints import APPLICABLE, KINDS, render_prompt, sample_constraints
from .data import answers_match, extract_answer, math_prompt
from .modeling import generate, is_degenerate


def substantive(text: str, kinds: list[str]) -> bool:
    """Format checks alone can be gamed: "[address]" satisfies a placeholder constraint, "P.S." satisfies
    all-caps + no-comma + postscript.  Require a real answer unless a constraint legitimately makes it short."""
    short_ok = any(k in ("word_max", "json") for k in kinds)
    return len(text.split()) >= (5 if short_ok else 20)


def curriculum_weights(per_kind_acc: dict[str, float], strength: float) -> dict[str, float]:
    """weight ∝ (1 - acc + 0.05) ** strength ; strength 0 = uniform."""
    return {k: (1.0 - per_kind_acc.get(k, 0.0) + 0.05) ** strength for k in KINDS}


def synth_if(model, tok, tasks: list[dict], n_prompts: int, k: int, temperature: float,
             weights: dict[str, float], rng: random.Random, gen: int, log=print) -> tuple[list[dict], dict]:
    picks = rng.sample(tasks, min(n_prompts, len(tasks)))
    items = []
    for t in picks:
        cons = sample_constraints(rng, rng.choice([1, 2, 2, 3]), weights)
        items.append((t, cons, render_prompt(t["prompt"], cons)))
    outs = generate(model, tok, [x[2] for x in items], max_new_tokens=512, temperature=temperature, n=k,
                    desc="synth-if", log=log)
    rows, tried, passed = [], Counter(), Counter()
    sample_pass = 0
    for (t, cons, prompt), cands in zip(items, outs):
        good = []
        for o in cands:
            ok = [c.check(o["text"]) for c in cons]
            for c, v in zip(cons, ok):
                tried[c.kind] += 1
                passed[c.kind] += int(v)
            if (all(ok) and not is_degenerate(o["text"], o["finished"])
                    and substantive(o["text"], [c.kind for c in cons])):
                good.append(o["text"])
        sample_pass += len(good)
        if good:
            rows.append({"prompt": prompt, "response": rng.choice(good), "source": "self_if", "gen": gen,
                         "kinds": [c.kind for c in cons], "pass_frac": len(good) / len(cands)})
    stats = {"prompts": len(items), "samples": len(items) * k, "accepted_prompts": len(rows),
             "sample_pass_rate": sample_pass / max(1, len(items) * k),
             "kind_pass_rate": {kk: passed[kk] / tried[kk] for kk in sorted(tried)}}
    return rows, stats


_STEP = re.compile(r"\d\s*[-+*/×÷x=]\s*\$?\d|=\s*\$?\d")


def valid_math_chain(text: str) -> bool:
    """A self-generated solution must show its work before the final answer.

    STaR keeps any sample whose final number is right.  With k=8 samples a bare "The answer is 18."
    is sometimes right by luck; selecting such samples (v1-v4 kept the *shortest* correct one) taught
    the model to skip its reasoning and cost 3-13 GSM8K points in every candidate.
    """
    body = re.split(r"the answer is", text, flags=re.IGNORECASE)[0]
    return len(body.split()) >= 25 and len(_STEP.findall(body)) >= 2


def synth_math(model, tok, questions: list[dict], n: int, k: int, temperature: float, rng: random.Random,
               gen: int, log=print) -> tuple[list[dict], dict]:
    picks = rng.sample(questions, min(n, len(questions)))
    outs = generate(model, tok, [math_prompt(q["question"]) for q in picks], max_new_tokens=400,
                    temperature=temperature, n=k, desc="synth-math", log=log)
    rows, sample_pass, shortcut = [], 0, 0
    for q, cands in zip(picks, outs):
        correct = [o["text"] for o in cands
                   if o["finished"] and answers_match(extract_answer(o["text"]), q["answer"])
                   and "answer is" in o["text"].lower()]
        good = [t for t in correct if valid_math_chain(t)]
        shortcut += len(correct) - len(good)
        sample_pass += len(good)
        if good:
            # median-length valid chain: neither the lucky shortcut nor the rambling one
            good.sort(key=len)
            rows.append({"prompt": math_prompt(q["question"]), "response": good[len(good) // 2],
                         "source": "self_math", "gen": gen, "qid": q["id"]})
    return rows, {"questions": len(picks), "accepted": len(rows),
                  "sample_pass_rate": sample_pass / max(1, len(picks) * k),
                  "shortcut_rejected": shortcut}


def synth_seed(model, tok, tasks: list[dict], n: int, rng: random.Random, gen: int, log=print) -> tuple[list[dict], dict]:
    """Self-distilled seed data (SDFT-style): the champion's own greedy answers to plain seed instructions.

    Training on these instead of the external reference answers teaches the chat format without
    pulling the model away from its own output distribution (which is what erodes its math).
    """
    picks = rng.sample(tasks, min(n, len(tasks)))
    outs = generate(model, tok, [t["prompt"] for t in picks], max_new_tokens=512, desc="synth-seed", log=log)
    rows = []
    for t, cands in zip(picks, outs):
        o = cands[0]
        txt = o["text"]
        if (o["finished"] and not is_degenerate(txt, True) and len(txt.split()) >= 8
                and t["prompt"][:60] not in txt):  # drop answers that echo the prompt back
            rows.append({"prompt": t["prompt"], "response": txt, "source": "self_seed", "gen": gen})
    return rows, {"prompts": len(picks), "accepted": len(rows)}


def rule_augment(seed_rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    rows, tries = [], 0
    while len(rows) < n and tries < n * 5:
        tries += 1
        r = rng.choice(seed_rows)
        cons = sample_constraints(rng, rng.choice([1, 2]), kinds=APPLICABLE)
        resp = r["response"]
        for c in sorted(cons, key=lambda c: c.kind == "json"):  # json wraps last
            resp = c.apply(resp, r["prompt"]) if resp is not None else None
        if resp and all(c.check(resp) for c in cons):
            rows.append({"prompt": render_prompt(r["prompt"], cons), "response": resp, "source": "rule",
                         "kinds": [c.kind for c in cons]})
    return rows


def dedup_latest(pool: list[dict]) -> list[dict]:
    """Keep the most recent sample per prompt (later generations are stronger)."""
    best: dict[str, dict] = {}
    for r in pool:
        if r["prompt"] not in best or r.get("gen", 0) >= best[r["prompt"]].get("gen", 0):
            best[r["prompt"]] = r
    return list(best.values())


def pool_stats(pool: list[dict]) -> dict:
    by = defaultdict(int)
    for r in pool:
        by[f"{r['source']}@g{r.get('gen', 0)}"] += 1
    return dict(sorted(by.items()))
