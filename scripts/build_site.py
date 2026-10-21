"""Build the static demo site (GitHub Pages) from a run's real results.

    python scripts/build_site.py [--run pilot]      -> docs/index.html + docs/data/*.json

No GPU and no server needed: the site is plain HTML/JS/SVG reading pre-exported JSON.  Only
relative, non-identifying information is exported (no absolute local paths).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rsi import report as R  # noqa: E402
from rsi.constraints import KIND_ZH  # noqa: E402
from rsi.data import read_jsonl  # noqa: E402
from rsi.optimizer import OP_ZH  # noqa: E402
from rsi.paths import RUNS  # noqa: E402

OUT = ROOT / "docs"
MAXCH = 1500  # truncate long (mostly runaway) responses


def clip(t: str) -> str:
    t = (t or "").strip()
    return t if len(t) <= MAXCH else t[:MAXCH] + " …（已截断）"


def metrics(m: dict | None) -> dict | None:
    if not m:
        return None
    return {"if": m["if_prompt_acc"], "instr": m["if_instr_acc"], "math": m["math_acc"],
            "degen": m["degenerate_rate"], "comp": m["composite"]}


def rel_eval(path_like: str, split: str) -> Path:
    """eval_dir / dir values in state.json are absolute; re-anchor them under this run directory."""
    p = Path(path_like)
    parts = p.parts
    i = next(k for k, x in enumerate(parts) if x.startswith("gen_"))
    return RUNS / ARGS.run / Path(*parts[i:]) / f"eval_{split}.jsonl"


def version_cards(gens: list[dict]) -> list[dict]:
    v = [
        ("v1", "跑通闭环", "微调后模型永不停止（P(<|im_end|>)≈0）；Windows 显存溢出静默借用内存导致训练慢 20×",
         "改用基座本就会用的 <|endoftext|> 作结束符；显存上限 90% + 只对回答分块计损"),
        ("v2", "失败定向修复", "第 1 代两个候选综合分显著更优（P=1.00），却都因数学回退 >3 点被拒",
         "只在通过门控者中选优；“险胜被拦”的候选作父本，只用针对该失败的修复算子"),
        ("v3", "选对父本", "最接近通过的候选只差 0.3 点，却没有被选为父本",
         "父本 = 违规最小者；新增数据源消融算子（去掉数学金标 / 种子减半）"),
        ("v4", "自蒸馏种子", "3 代 8 个候选全部数学回退，调学习率、轮数、秩都无效",
         "冠军自答种子指令替代外部答案（SDFT 思路）；新算子强制先试一次；去掉无效算子"),
        ("v5", "推理链校验", "自蒸馏反而更差（62.7%）——假设被系统自己否定；数学回答中位长度 217→117 token",
         "根因：STaR 选“最短正确解”奖励了蒙对（11% 样本只有一句答案）；改为校验推理链 + 取中位长度，并回溯清洗数据池"),
        ("v6", "防钻空子", "冠军只答“P.S.”也能同时通过 3 项格式校验",
         "不足 20 词的回答不入池并回溯清洗；新增“稳健口径”指标（只算有实质内容的通过）"),
    ]
    first = next((g for g in gens if g.get("champ_gate")), None)
    if first:
        pdev = next(x for x in gens if x["gen"] == first["parent"])["dev"]
        v.append(("v7", "防赢家诅咒",
                  f"冠军是按 dev 选出来的：dev 数学 {pdev['math_acc']:.1%}，换一批新题只有 {first['champ_gate']['math_acc']:.1%}",
                  "每代新抽“门控集”，冠军与候选同场重测后再决定晋升；dev 只用于画曲线和诊断"))
    return [{"v": a, "name": b, "evidence": c, "fix": d} for a, b, c, d in v]


def build_summary(st: dict, rd: Path) -> dict:
    gens = st["generations"]
    g0 = gens[0]
    champ = next(g for g in gens if g["gen"] == st["champion"])
    rows = []
    for g in gens:
        cands = g.get("candidates", [])
        best = next((c for c in cands if c["cid"] == g.get("chosen")), None)
        rows.append({
            "gen": g["gen"], "promoted": bool(g.get("promoted")), "baseline": g["gen"] == 0,
            "ops": ([OP_ZH.get(o, o) for o in best["ops"]] or ["默认配方"]) if best else [],
            "repair": bool(best and str(best.get("parent", "")).startswith("repair")),
            "p_better": best["boot"]["p_better"] if best else None,
            "diff": best["boot"]["diff"] if best else None,
            "ci90": best["boot"]["ci90"] if best else None,
            "reasons": best.get("gate_reasons", []) if best else [],
            "test": metrics(g.get("test")), "minutes": round((g.get("seconds") or 0) / 60, 1),
            "reflection": g.get("reflection", []),
            "cands": [{"cid": c["cid"], "ops": [OP_ZH.get(o, o) for o in c["ops"]] or ["默认配方"],
                       "dev": metrics(c["dev"]), "ok": bool(c.get("gate_ok")),
                       "n_train": c.get("train", {}).get("examples")} for c in cands],
            "gate": bool(g.get("champ_gate")),
        })
    series = []
    for x in R.champion_series(st):
        series.append({"gen": x["gen"], "champ": x["champ"], **metrics(x["test"] or x["dev"])})
    pk0, pk1 = g0["test"]["per_kind"], (champ.get("test") or champ["dev"])["per_kind"]
    kinds = sorted(pk1, key=lambda k: pk1[k] - pk0.get(k, 0), reverse=True)

    from rsi.synth import substantive

    def robust(g):
        rows_ = [r for r in read_jsonl(rel_eval(g["eval_dir"], "test")) if r["type"] == "if"]
        return sum(r["strict"] and substantive(r["response"], r["kinds"]) for r in rows_) / max(1, len(rows_))

    bandit = json.loads((rd / "bandit.json").read_text(encoding="utf-8")) if (rd / "bandit.json").exists() else {}
    ife = None
    p = rd / "ifeval_official" / "summary.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        ife = {"models": {k: {"ps": v["prompt_level_strict_acc"], "is": v["inst_level_strict_acc"],
                              "pl": v["prompt_level_loose_acc"], "il": v["inst_level_loose_acc"],
                              "finish": v["finish_rate"]} for k, v in s["models"].items()},
               "breakdown": s.get("breakdown", {}), "vs_base": s.get("vs_base", {})}
    pool = read_jsonl(rd / "self_pool.jsonl")
    return {
        "meta": {"base_model": st["base_model"].split("/")[-1], "champion": st["champion"],
                 "n_gen": len(gens) - 1, "n_prom": sum(1 for g in gens[1:] if g.get("promoted")),
                 "n_cand": sum(len(g.get("candidates", [])) for g in gens), "n_self": len(pool),
                 "gpu_hours": round(sum(g.get("seconds") or 0 for g in gens) / 3600, 1),
                 "built": time.strftime("%Y-%m-%d")},
        "kpi": {"base": metrics(g0["test"]), "champ": metrics(champ.get("test")),
                "robust": {"base": robust(g0), "champ": robust(champ)}},
        "gens": rows, "series": series,
        "per_kind": {"kinds": [KIND_ZH[k] for k in kinds], "base": [pk0.get(k, 0) for k in kinds],
                     "champ": [pk1[k] for k in kinds]},
        "bandit": sorted(({"op": OP_ZH.get(k, k), "mean": a / (a + b), "n": round(a + b - 2)}
                          for k, (a, b) in bandit.items() if a + b > 2), key=lambda x: -x["mean"]),
        "ifeval": ife, "versions": version_cards(gens),
    }


def build_samples(st: dict) -> dict:
    gens = {g["gen"]: g for g in st["generations"] if g.get("promoted") and g.get("test")}
    order = sorted(gens)
    models = [{"key": f"g{g}", "name": "原始基座" if g == 0 else f"第{g}代冠军"} for g in order]
    base_rows = read_jsonl(rel_eval(gens[0]["eval_dir"], "test"))
    items = []
    for r in base_rows:
        it = {"id": r["id"], "type": r["type"], "prompt": r["prompt"]}
        if r["type"] == "if":
            it["kinds"] = [KIND_ZH.get(k, k) for k in r["kinds"]]
        else:
            it["gold"] = r["gold"]
        items.append(it)
    resp = {}
    for g in order:
        by = {r["id"]: r for r in read_jsonl(rel_eval(gens[g]["eval_dir"], "test"))}
        out = []
        for it in items:
            r = by[it["id"]]
            e = {"t": clip(r["response"]), "f": bool(r["finished"])}
            if r["type"] == "if":
                e["c"], e["s"] = r["checks"], bool(r["strict"])
            else:
                e["ok"], e["p"] = bool(r["correct"]), r.get("pred")
            out.append(e)
        resp[f"g{g}"] = out
    return {"split": "test", "models": models, "items": items, "resp": resp}


def build_ifeval(rd: Path) -> dict | None:
    root = rd / "ifeval_official"
    if not (root / "summary.json").exists():
        return None
    labels = list(json.loads((root / "summary.json").read_text(encoding="utf-8"))["models"])
    from datasets import load_dataset
    docs = {d["key"]: d for d in load_dataset("google/IFEval", split="train")}
    first = read_jsonl(root / labels[0] / "responses.jsonl")
    items = [{"key": r["key"], "prompt": r["prompt"], "ids": docs[r["key"]]["instruction_id_list"]} for r in first]
    resp = {}
    for lab in labels:
        by = {r["key"]: r for r in read_jsonl(root / lab / "responses.jsonl")}
        resp[lab] = [{"t": clip(by[i["key"]]["response"]), "s": bool(by[i["key"]]["prompt_level_strict_acc"]),
                      "i": by[i["key"]]["inst_level_strict_acc"]} for i in items]
    names = {lab: ("原始基座" if lab == "base" else f"第{lab[3:]}代") for lab in labels}
    return {"models": [{"key": l, "name": names[l]} for l in labels], "items": items, "resp": resp}


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pilot")
    ARGS = ap.parse_args()
    rd = RUNS / ARGS.run
    st = R.load_state(rd)
    (OUT / "data").mkdir(parents=True, exist_ok=True)
    summary = build_summary(st, rd)
    samples = build_samples(st)
    ife = build_ifeval(rd)
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))  # noqa: E731
    (OUT / "data" / "samples_test.json").write_text(dump(samples), encoding="utf-8")
    if ife:
        (OUT / "data" / "ifeval_samples.json").write_text(dump(ife), encoding="utf-8")
    html = (ROOT / "scripts" / "site_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__SUMMARY__*/null", dump(summary))
    (OUT / "index.html").write_text(html, encoding="utf-8")
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    for f in sorted((OUT / "data").glob("*.json")) + [OUT / "index.html"]:
        print(f"{f.relative_to(ROOT)}  {f.stat().st_size / 1024:,.0f} KB")


ARGS = None
if __name__ == "__main__":
    main()
