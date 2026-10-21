"""Independent check on the official IFEval benchmark (google/IFEval, 541 prompts).

Generation uses the same ChatML prefix and greedy decoding as every other eval in this project;
scoring uses lm-eval's implementation of the official IFEval checker (strict + loose).

    python -m rsi.ifeval_official                    # base model vs current champion
    python -m rsi.ifeval_official --models 0 6 8     # any generations that have an adapter (0 = base)
    python -m rsi.ifeval_official --limit 20         # quick smoke test
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

from . import data as D
from .modeling import free, generate, load_model
from .paths import ROOT, RUNS


def score(docs: list[dict], responses: list[str]) -> tuple[dict, list[dict]]:
    from lm_eval.tasks.ifeval.utils import process_results
    rows, per_type = [], defaultdict(lambda: [0, 0])
    for d, r in zip(docs, responses):
        res = process_results(d, [r])
        for iid, ok in zip(d["instruction_id_list"], res["inst_level_strict_acc"]):
            per_type[iid][0] += int(ok)
            per_type[iid][1] += 1
        rows.append({"key": d["key"], **res})
    n = len(rows)
    inst_s = [x for r in rows for x in r["inst_level_strict_acc"]]
    inst_l = [x for r in rows for x in r["inst_level_loose_acc"]]
    m = {
        "prompt_level_strict_acc": sum(r["prompt_level_strict_acc"] for r in rows) / n,
        "inst_level_strict_acc": sum(inst_s) / len(inst_s),
        "prompt_level_loose_acc": sum(r["prompt_level_loose_acc"] for r in rows) / n,
        "inst_level_loose_acc": sum(inst_l) / len(inst_l),
        "n_prompts": n,
        "per_instruction_strict": {k: v[0] / v[1] for k, v in sorted(per_type.items())},
        "per_instruction_n": {k: v[1] for k, v in sorted(per_type.items())},
    }
    return m, rows


# IFEval instruction ids that correspond to one of our 18 training constraint kinds; the rest never
# appear in training data, so they measure transfer rather than template familiarity.
SEEN_IN_TRAINING = {
    "change_case:english_lowercase", "change_case:english_capital", "punctuation:no_comma",
    "length_constraints:number_words", "length_constraints:number_paragraphs", "startend:end_checker",
    "startend:quotation", "detectable_format:title", "detectable_format:number_bullet_lists",
    "detectable_format:json_format", "detectable_format:number_highlighted_sections",
    "detectable_format:multiple_sections", "detectable_content:postscript",
    "detectable_content:number_placeholders", "keywords:existence", "keywords:forbidden_words",
}


def breakdown(rows: list[dict], docs_by_key: dict) -> dict:
    """Seen-vs-unseen instruction types, and how much of the score short answers / unfinished outputs carry."""
    acc = {"seen": [0, 0], "unseen": [0, 0]}
    unseen_only = [0, 0]
    short_pass = 0
    for r in rows:
        ids = docs_by_key[r["key"]]["instruction_id_list"]
        for iid, ok in zip(ids, r["inst_level_strict_acc"]):
            k = "seen" if iid in SEEN_IN_TRAINING else "unseen"
            acc[k][0] += int(ok)
            acc[k][1] += 1
        if all(i not in SEEN_IN_TRAINING for i in ids):
            unseen_only[0] += int(r["prompt_level_strict_acc"])
            unseen_only[1] += 1
        if r["prompt_level_strict_acc"] and len(r["response"].split()) < 20:
            short_pass += 1
    n = len(rows)
    return {"inst_strict_seen": acc["seen"][0] / acc["seen"][1], "n_inst_seen": acc["seen"][1],
            "inst_strict_unseen": acc["unseen"][0] / acc["unseen"][1], "n_inst_unseen": acc["unseen"][1],
            "prompt_strict_unseen_only": unseen_only[0] / max(1, unseen_only[1]), "n_prompts_unseen_only": unseen_only[1],
            "prompt_strict_excl_short": (sum(r["prompt_level_strict_acc"] for r in rows) - short_pass) / n,
            "strict_pass_under_20_words": short_pass}


def paired_ci(a: list[bool], b: list[bool], n_boot: int = 2000, seed: int = 0) -> dict:
    """Bootstrap of mean(b) - mean(a) over prompts (paired)."""
    rng = random.Random(seed)
    idx = range(len(a))
    diffs = []
    for _ in range(n_boot):
        s = [rng.choice(idx) for _ in idx]
        diffs.append(sum(b[i] - a[i] for i in s) / len(s))
    diffs.sort()
    return {"diff": (sum(b) - sum(a)) / len(a), "p_better": sum(d > 0 for d in diffs) / n_boot,
            "ci90": [diffs[int(0.05 * n_boot)], diffs[int(0.95 * n_boot)]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ap.add_argument("--models", nargs="*", type=int, help="generation numbers (0 = base); default: base + champion")
    ap.add_argument("--max-new", type=int, default=1280)  # lm-eval's IFEval default
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    from datasets import load_dataset
    docs = list(load_dataset("google/IFEval", split="train"))[: a.limit]
    st = json.loads((RUNS / a.run / "state.json").read_text(encoding="utf-8"))
    base = str((ROOT / st["base_model"]).resolve())
    gens = {g["gen"]: g for g in st["generations"] if g.get("promoted")}
    wanted = a.models if a.models is not None else sorted({0, st["champion"]})
    out_root = RUNS / a.run / ("ifeval_official" if a.limit is None else f"ifeval_official_limit{a.limit}")
    out_root.mkdir(parents=True, exist_ok=True)

    results = {}
    for g in wanted:
        adapter = gens[g].get("adapter") if g in gens else None
        if g != 0 and not adapter:
            raise SystemExit(f"generation {g} has no promoted adapter")
        label = "base" if g == 0 else f"gen{g}"
        d = out_root / label
        if (d / "metrics.json").exists():
            results[label] = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            print(f"[{label}] cached")
            continue
        t0 = time.time()
        model, tok = load_model(base, adapter)
        outs = generate(model, tok, [x["prompt"] for x in docs], max_new_tokens=a.max_new, desc=f"ifeval-{label}")
        del model
        free()
        responses = [o[0]["text"] for o in outs]
        m, rows = score(docs, responses)
        m["finish_rate"] = sum(o[0]["finished"] for o in outs) / len(outs)
        m["seconds"] = round(time.time() - t0)
        m["model"] = {"generation": g, "adapter": adapter, "max_new_tokens": a.max_new, "format": "ChatML prefix, greedy"}
        D.write_jsonl(d / "responses.jsonl", [{"key": x["key"], "prompt": x["prompt"], "response": r, **s}
                                              for x, r, s in zip(docs, responses, rows)])
        (d / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
        results[label] = m
        print(f"[{label}] prompt-strict {m['prompt_level_strict_acc']:.4f}  inst-strict {m['inst_level_strict_acc']:.4f}  "
              f"prompt-loose {m['prompt_level_loose_acc']:.4f}  inst-loose {m['inst_level_loose_acc']:.4f}  "
              f"finish {m['finish_rate']:.3f}  {m['seconds']}s", flush=True)

    summary = {"benchmark": "google/IFEval (official, 541 prompts)" if a.limit is None else f"IFEval first {a.limit}",
               "scorer": "lm-eval tasks.ifeval.utils.process_results", "models": results}
    by_key = {x["key"]: x for x in docs}
    summary["breakdown"] = {label: breakdown(D.read_jsonl(out_root / label / "responses.jsonl"), by_key)
                            for label in results}
    if "base" in results and len(results) > 1:
        rb = {r["key"]: r for r in D.read_jsonl(out_root / "base" / "responses.jsonl")}
        for label in results:
            if label == "base":
                continue
            rc = D.read_jsonl(out_root / label / "responses.jsonl")
            keys = [r["key"] for r in rc]
            summary.setdefault("vs_base", {})[label] = {
                "prompt_level_strict": paired_ci([rb[k]["prompt_level_strict_acc"] for k in keys],
                                                 [r["prompt_level_strict_acc"] for r in rc]),
                "prompt_level_loose": paired_ci([rb[k]["prompt_level_loose_acc"] for k in keys],
                                                [r["prompt_level_loose_acc"] for r in rc]),
            }
    (out_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "models"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
