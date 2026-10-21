"""Automatic evaluation: verifiable instruction following + math reasoning + degeneration guard."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from .data import answers_match, constraints_of, extract_answer, math_prompt, write_jsonl
from .modeling import generate, is_degenerate

WEIGHTS = {"if_prompt_acc": 0.6, "math_acc": 0.3, "non_degenerate": 0.1}


def composite(m: dict) -> float:
    return (WEIGHTS["if_prompt_acc"] * m["if_prompt_acc"] + WEIGHTS["math_acc"] * m["math_acc"]
            + WEIGHTS["non_degenerate"] * (1 - m["degenerate_rate"]))


def evaluate(model, tok, evalset: dict, out_dir: Path | None = None, tag: str = "dev",
             max_new_if: int = 512, max_new_math: int = 400, log=print) -> dict:
    ifs, maths = evalset["if"], evalset["math"]
    gi = generate(model, tok, [x["prompt"] for x in ifs], max_new_tokens=max_new_if, desc=f"eval-{tag}-if", log=log)
    gm = generate(model, tok, [math_prompt(x["question"]) for x in maths], max_new_tokens=max_new_math,
                  desc=f"eval-{tag}-math", log=log)
    rows, per_kind = [], defaultdict(lambda: [0, 0])
    for x, g in zip(ifs, gi):
        o = g[0]
        cons = constraints_of(x)
        ok = [c.check(o["text"]) for c in cons]
        for c, v in zip(cons, ok):
            per_kind[c.kind][0] += int(v)
            per_kind[c.kind][1] += 1
        rows.append({"id": x["id"], "type": "if", "prompt": x["prompt"], "response": o["text"],
                     "kinds": [c.kind for c in cons], "checks": ok, "strict": all(ok),
                     "finished": o["finished"], "degenerate": is_degenerate(o["text"], o["finished"]),
                     "tokens": o["tokens"]})
    for x, g in zip(maths, gm):
        o = g[0]
        pred = extract_answer(o["text"])
        rows.append({"id": x["id"], "type": "math", "prompt": x["question"], "response": o["text"],
                     "pred": pred, "gold": x["answer"], "correct": answers_match(pred, x["answer"]),
                     "finished": o["finished"], "degenerate": is_degenerate(o["text"], o["finished"]),
                     "tokens": o["tokens"]})
    m = summarize(rows)
    m["per_kind"] = {k: v[0] / v[1] for k, v in sorted(per_kind.items())}
    m["per_kind_n"] = {k: v[1] for k, v in sorted(per_kind.items())}
    if out_dir:
        out_dir = Path(out_dir)
        write_jsonl(out_dir / f"eval_{tag}.jsonl", rows)
        (out_dir / f"metrics_{tag}.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    if log:
        log(f"[eval-{tag}] composite={m['composite']:.4f} if={m['if_prompt_acc']:.3f} "
            f"instr={m['if_instr_acc']:.3f} math={m['math_acc']:.3f} degen={m['degenerate_rate']:.3f}")
    return m


def summarize(rows: list[dict]) -> dict:
    ifr = [r for r in rows if r["type"] == "if"]
    mr = [r for r in rows if r["type"] == "math"]
    checks = [c for r in ifr for c in r["checks"]]
    m = {
        "if_prompt_acc": sum(r["strict"] for r in ifr) / max(1, len(ifr)),
        "if_instr_acc": sum(checks) / max(1, len(checks)),
        "math_acc": sum(r["correct"] for r in mr) / max(1, len(mr)),
        "degenerate_rate": sum(r["degenerate"] for r in rows) / max(1, len(rows)),
        "finish_rate": sum(r["finished"] for r in rows) / max(1, len(rows)),
        "avg_tokens": sum(r["tokens"] for r in rows) / max(1, len(rows)),
        "n_if": len(ifr), "n_math": len(mr),
    }
    m["composite"] = composite(m)
    # robustness view (reported, not gated): strict passes that are also real answers
    from .synth import substantive
    m["if_prompt_acc_substantive"] = (sum(r["strict"] and substantive(r.get("response", ""), r.get("kinds", []))
                                          for r in ifr) / max(1, len(ifr)))
    return m


def paired_bootstrap(rows_new: list[dict], rows_old: list[dict], n_boot: int = 2000, seed: int = 0) -> dict:
    """P(composite_new > composite_old) under item resampling (paired by item id)."""
    old = {r["id"]: r for r in rows_old}
    pairs = [(r, old[r["id"]]) for r in rows_new if r["id"] in old]
    ifp = [p for p in pairs if p[0]["type"] == "if"]
    mp = [p for p in pairs if p[0]["type"] == "math"]
    rng = random.Random(seed)

    def comp(sample_if, sample_m, k):
        rs = [p[k] for p in sample_if] + [p[k] for p in sample_m]
        return summarize(rs)["composite"]

    base_diff = comp(ifp, mp, 0) - comp(ifp, mp, 1)
    wins, diffs = 0, []
    for _ in range(n_boot):
        si = [ifp[rng.randrange(len(ifp))] for _ in ifp]
        sm = [mp[rng.randrange(len(mp))] for _ in mp]
        d = comp(si, sm, 0) - comp(si, sm, 1)
        diffs.append(d)
        wins += d > 0
    diffs.sort()
    return {"diff": base_diff, "p_better": wins / n_boot,
            "ci90": [diffs[int(0.05 * n_boot)], diffs[int(0.95 * n_boot)]]}
