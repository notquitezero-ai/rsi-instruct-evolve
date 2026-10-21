"""RSI orchestrator: one generation =
    diagnose champion -> self-synthesize data -> propose recipes -> train -> evaluate -> gate -> reflect

Usage:
    python -m rsi.loop --config configs/pilot.yaml               # run/continue
    python -m rsi.loop --config configs/pilot.yaml --generations 3
State lives in runs/<run_name>/state.json, so a crashed or stopped run resumes where it left off.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

import yaml

from . import data as D
from .evaluate import evaluate, paired_bootstrap
from .modeling import free, load_model
from .optimizer import (DEFAULT_RECIPE, DEFAULT_SYNTH, FAILURE_ZH, OP_ZH, REPAIR_OPS, SYNTH_CAPS, Bandit,
                        adapt_synth, failure_mode, gate, propose)
from .paths import ROOT, RUNS
from .synth import (curriculum_weights, dedup_latest, pool_stats, rule_augment, substantive, synth_if, synth_math,
                    synth_seed, valid_math_chain)

GSM_GOLD_SPLIT = 3500  # gsm_train[:3500] gold seed pool, [3500:] self-play questions


class Run:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.dir = RUNS / cfg["run_name"]
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dir / "state.json"
        self.pool_path = self.dir / "self_pool.jsonl"
        self.base = str((ROOT / cfg["base_model"]).resolve())
        self.bandit = Bandit(self.dir / "bandit.json")
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))
        else:
            self.state = {"run_name": cfg["run_name"], "base_model": cfg["base_model"], "champion": None,
                          "generations": [], "synth": {**DEFAULT_SYNTH, **cfg.get("synth", {})},
                          "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        # settings saved by an older version: add new keys, respect current caps
        s = {**DEFAULT_SYNTH, **self.state["synth"]}
        self.state["synth"] = {k: min(v, SYNTH_CAPS[k]) if k in SYNTH_CAPS else v for k, v in s.items()}
        self.log_f = open(self.dir / "log.txt", "a", encoding="utf-8")

    # ------------------------------------------------------------------------------------
    def log(self, msg: str):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def save(self):
        self.state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.state_path)
        self.bandit.save()

    def gen_dir(self, g: int) -> Path:
        p = self.dir / f"gen_{g:03d}"
        p.mkdir(parents=True, exist_ok=True)
        return p

    def evalset(self, split: str) -> dict:
        es = D.eval_set(split)
        lim = self.cfg.get("eval_limit")
        return {k: v[:lim] for k, v in es.items()} if lim else es

    def champion(self) -> dict | None:
        c = self.state["champion"]
        return None if c is None else next(x for x in self.state["generations"] if x["gen"] == c)

    def champ_adapter(self) -> str | None:
        ch = self.champion()
        return ch.get("adapter") if ch else None

    def champ_rows(self, split="dev") -> list[dict]:
        ch = self.champion()
        return D.read_jsonl(Path(ch["eval_dir"]) / f"eval_{split}.jsonl")

    # ------------------------------------------------------------------------------------
    def gen0(self):
        """Baseline: evaluate the untouched base model."""
        d = self.gen_dir(0)
        self.log("=== 第0代：评测原始基座模型 ===")
        model, tok = load_model(self.base)
        dev = evaluate(model, tok, self.evalset("dev"), d, "dev", log=self.log)
        test = evaluate(model, tok, self.evalset("test"), d, "test", log=self.log)
        del model
        free()
        self.state["generations"].append({
            "gen": 0, "kind": "baseline", "adapter": None, "eval_dir": str(d), "recipe": None,
            "dev": dev, "test": test, "promoted": True, "candidates": [],
            "reflection": ["原始基座模型（未经指令微调）作为初始冠军。"], "time": time.strftime("%Y-%m-%d %H:%M:%S")})
        self.state["champion"] = 0
        self.save()

    def build_trainset(self, recipe: dict, g: int) -> tuple[list[dict], dict]:
        rng = random.Random(f"{self.cfg['seed']}-{g}-data")  # same sample for every candidate of a generation
        alpaca = D.load_split("alpaca_train")
        gsm = D.load_split("gsm_train")[:GSM_GOLD_SPLIT]
        raw = dedup_latest(D.read_jsonl(self.pool_path))
        distilled = [r for r in raw if r["source"] == "self_seed"]
        if recipe.get("seed_source", "alpaca") == "self" and distilled:
            seed_pool = distilled  # the champion's own answers to the seed instructions
            rows = [dict(r) for r in rng.sample(distilled, min(recipe["n_seed"], len(distilled)))]
        else:
            seed_pool = alpaca
            rows = [{"prompt": r["prompt"], "response": r["response"], "source": "seed"}
                    for r in rng.sample(alpaca, min(recipe["n_seed"], len(alpaca)))]
        rows += [{"prompt": D.math_prompt(q["question"]), "response": D.math_target(q["solution"], q["answer"]),
                  "source": "math_seed"} for q in rng.sample(gsm, min(recipe["n_math_seed"], len(gsm)))]
        # data hygiene also applies retroactively: pooled shortcut "solutions" from older versions are dropped
        pool = [r for r in raw if (r["source"] == "self_if" and substantive(r["response"], r.get("kinds", [])))
                or (r["source"] == "self_math" and valid_math_chain(r["response"]))]
        pool.sort(key=lambda r: -r.get("gen", 0))
        rows += pool[: int(len(pool) * recipe["self_frac"])]
        rows += rule_augment(seed_pool, recipe["n_rule"], rng)
        rng.shuffle(rows)
        mix = {}
        for r in rows:
            mix[r["source"]] = mix.get(r["source"], 0) + 1
        return rows, mix

    def synthesize(self, g: int, rec: dict):
        ch = self.champion()
        if ch["adapter"] is None and not self.cfg.get("synth_from_base", False):
            rec["synth"] = None
            rec["reflection"].append("冠军仍是未微调基座，不能可靠遵循格式 → 本代跳过自生成，用种子+规则数据自举。")
            return
        ckpt = self.gen_dir(g) / "synth.json"
        if ckpt.exists():  # resume: this generation's self-play already ran and was pooled
            saved = json.loads(ckpt.read_text(encoding="utf-8"))
            rec["synth"], rec["reflection"] = saved["synth"], rec["reflection"] + saved["reflection"]
            self.log(f"[synth] resumed generation {g} self-play from {ckpt.name}")
            return
        n_refl = len(rec["reflection"])
        s = self.state["synth"]
        rng = random.Random(f"{self.cfg['seed']}-{g}-synth")
        weights = curriculum_weights(ch["dev"]["per_kind"], s["curriculum"])
        weakest = sorted(ch["dev"]["per_kind"].items(), key=lambda kv: kv[1])[:4]
        self.log(f"=== 第{g}代：冠军(第{ch['gen']}代)自生成数据  设置={s} ===")
        model, tok = load_model(self.base, ch["adapter"])
        tasks = [r for r in D.load_split("alpaca_train") if len(r["response"].split()) >= 20]
        if_rows, if_stats = synth_if(model, tok, tasks, s["n_if_prompts"], s["k"], s["temperature"], weights, rng, g,
                                     log=self.log)
        gsm_q = D.load_split("gsm_train")[GSM_GOLD_SPLIT:]
        m_rows, m_stats = synth_math(model, tok, gsm_q, s["n_math"], s["k"], s["temperature"], rng, g, log=self.log)
        s_rows, s_stats = [], None
        if s.get("n_seed_distill"):
            alpaca = D.load_split("alpaca_train")
            s_rows, s_stats = synth_seed(model, tok, alpaca, s["n_seed_distill"], rng, g, log=self.log)
        del model
        free()
        D.append_jsonl(self.pool_path, if_rows + m_rows + s_rows)
        rec["synth"] = {"settings": dict(s), "if": if_stats, "math": m_stats, "seed": s_stats, "weakest": weakest,
                        "pool": pool_stats(dedup_latest(D.read_jsonl(self.pool_path)))}
        if s_stats:
            rec["reflection"].append(f"自蒸馏种子：冠军回答 {s_stats['prompts']} 条普通指令，过滤后保留 {s_stats['accepted']} 条"
                                     "（供“种子自蒸馏”配方替代外部参考答案）。")
        from .constraints import KIND_ZH
        rec["reflection"].append(
            "诊断：冠军最弱的约束类型为 " + "、".join(f"{KIND_ZH.get(k, k)}({v:.0%})" for k, v in weakest)
            + f"；课程强度={s['curriculum']}，按弱项加权出题。")
        rec["reflection"].append(
            f"自生成：指令题 {if_stats['prompts']}×{s['k']} 次采样，验证通过 {if_stats['accepted_prompts']} 题"
            f"（样本通过率 {if_stats['sample_pass_rate']:.1%}）；数学 {m_stats['questions']} 题通过 {m_stats['accepted']} 题"
            f"（{m_stats['sample_pass_rate']:.1%}）。")
        self.log(f"[synth] if accepted {if_stats['accepted_prompts']}/{if_stats['prompts']}  "
                 f"math accepted {m_stats['accepted']}/{m_stats['questions']}")
        ckpt.write_text(json.dumps({"synth": rec["synth"], "reflection": rec["reflection"][n_refl:]},
                                   ensure_ascii=False, indent=2), encoding="utf-8")

    def run_candidate(self, g: int, ci: int, cand: dict) -> dict:
        from .train import train_lora
        d = self.gen_dir(g) / f"cand_{ci}"
        d.mkdir(parents=True, exist_ok=True)
        recipe = cand["recipe"]
        res = {"cid": ci, "recipe": recipe, "ops": cand["ops"], "parent": cand.get("parent", "champion"),
               "dir": str(d), "adapter": str(d / "adapter")}
        if (d / "metrics_dev.json").exists():  # resume
            res["dev"] = json.loads((d / "metrics_dev.json").read_text(encoding="utf-8"))
            res["train"] = json.loads((d / "train_stats.json").read_text(encoding="utf-8"))
            res["mix"] = json.loads((d / "mix.json").read_text(encoding="utf-8"))
            self.log(f"[cand {ci}] resumed from disk")
            return res
        rows, mix = self.build_trainset(recipe, g)
        (d / "mix.json").write_text(json.dumps(mix), encoding="utf-8")
        (d / "recipe.json").write_text(json.dumps(cand, indent=2), encoding="utf-8")
        init = self.champ_adapter() if recipe["init"] == "champion" else None
        ops_zh = "+".join(OP_ZH[o] for o in cand["ops"]) or "默认配方"
        self.log(f"--- 候选{ci} [{ops_zh}] 训练 {len(rows)} 条 {mix}  init={'冠军' if init else '基座'} ---")
        res["train"] = train_lora(self.base, rows, d, recipe, init_adapter=init, seed=self.cfg["seed"] + g, log=self.log)
        res["mix"] = mix
        model, tok = load_model(self.base, res["adapter"])
        res["dev"] = evaluate(model, tok, self.evalset("dev"), d, "dev", log=self.log)
        del model
        free()
        return res

    def gate_eval(self, g: int, name: str, adapter: str | None) -> tuple[dict, list[dict]]:
        """Evaluate one model on generation g's fresh gate set (resumable)."""
        d = self.gen_dir(g) / "gate" / name
        if (d / "metrics_gate.json").exists():
            return (json.loads((d / "metrics_gate.json").read_text(encoding="utf-8")),
                    D.read_jsonl(d / "eval_gate.jsonl"))
        gs = D.gate_set(g)
        lim = self.cfg.get("eval_limit")
        if lim:
            gs = {k: v[:lim] for k, v in gs.items()}
        model, tok = load_model(self.base, adapter)
        m = evaluate(model, tok, gs, d, "gate", log=self.log)
        del model
        free()
        return m, D.read_jsonl(d / "eval_gate.jsonl")

    def near_miss(self) -> dict | None:
        """Previous generation's candidate that beat the champion clearly but tripped a guard.

        Among such candidates take the one *closest to passing* (smallest guard violation), not the
        one with the highest composite: it is the better starting point for a repair.
        """
        prev = self.state["generations"][-1]
        if prev.get("promoted") or not prev.get("candidates"):
            return None
        # compare on the same footing the gate used: fresh gate metrics when available (v7+), else dev
        champ = prev.get("champ_gate") or next(x for x in self.state["generations"] if x["gen"] == prev["parent"])["dev"]
        g = self.cfg["gate"]
        p_need = g.get("p_better", 0.8)

        def violation(c):
            m = c.get("gate_m") or c["dev"]
            return (max(0.0, champ["math_acc"] - g.get("math_tol", 0.03) - m["math_acc"])
                    + max(0.0, m["degenerate_rate"] - champ["degenerate_rate"] - g.get("degen_tol", 0.03)))

        near = [c for c in prev["candidates"] if failure_mode(c["gate_reasons"])
                and c["boot"]["diff"] > 0 and c["boot"]["p_better"] >= p_need]
        if not near:
            return None
        best = min(near, key=violation)
        return {"recipe": best["recipe"], "mode": failure_mode(best["gate_reasons"]),
                "from": f"g{prev['gen']}c{best['cid']}", "violation": round(violation(best), 4)}

    def generation(self, g: int):
        t0 = time.time()
        ch = self.champion()
        rec = {"gen": g, "kind": "candidate", "parent": ch["gen"], "reflection": [], "candidates": [],
               "time": time.strftime("%Y-%m-%d %H:%M:%S")}
        self.synthesize(g, rec)
        rng = random.Random(f"{self.cfg['seed']}-{g}-propose")
        near = self.near_miss()
        if near:
            rec["reflection"].append(
                f"反思：上一代候选 {near['from']} 综合分显著更优、离通过最近（超限 {near['violation'] * 100:.1f} 点），"
                f"但因{FAILURE_ZH[near['mode']]}被拒 → 本代以其配方为父本，"
                f"优先尝试修复类算子（{'、'.join(OP_ZH[o] for o in REPAIR_OPS[near['mode']])}）。")
        cands = propose(ch["recipe"], self.bandit, self.cfg["candidates_per_gen"], rng,
                        default={**DEFAULT_RECIPE, **self.cfg.get("recipe", {})}, near_miss=near)
        results = [self.run_candidate(g, i, c) for i, c in enumerate(cands)]
        if self.cfg.get("fresh_gate", True):
            # v7: compare on items nobody was selected on (champion's dev score carries a winner's curse)
            champ_m, champ_rows = self.gate_eval(g, "champion", ch.get("adapter"))
            rec["champ_gate"] = champ_m
            for r in results:
                r["gate_m"], rows = self.gate_eval(g, f"cand_{r['cid']}", r["adapter"])
                r["boot"] = paired_bootstrap(rows, champ_rows)
                r["gate_ok"], r["gate_reasons"] = gate(r["gate_m"], champ_m, r["boot"], self.cfg["gate"])
            key = "gate_m"
            rec["reflection"].append(
                f"门控（新鲜样本，防赢家诅咒）：冠军在本代门控集综合分 {champ_m['composite']:.4f}"
                f"（其 dev 选拔分 {ch['dev']['composite']:.4f}）；候选 "
                + "、".join(f"c{r['cid']}={r['gate_m']['composite']:.4f}" for r in results) + "。")
        else:
            champ_rows = self.champ_rows("dev")
            for r in results:
                r["boot"] = paired_bootstrap(D.read_jsonl(Path(r["dir"]) / "eval_dev.jsonl"), champ_rows)
                r["gate_ok"], r["gate_reasons"] = gate(r["dev"], ch["dev"], r["boot"], self.cfg["gate"])
            key = "dev"
        for r in results:
            self.bandit.update(r["ops"], r["gate_ok"])
        # promote the best candidate that passes every guard; otherwise report the best overall
        passing = [r for r in results if r["gate_ok"]]
        best = max(passing or results, key=lambda r: r[key]["composite"])
        rec["candidates"] = results
        rec["chosen"] = best["cid"]
        rec["promoted"] = bool(best["gate_ok"])
        rec["gate"] = best.get("gate_m")
        ops_zh = "+".join(OP_ZH[o] for o in best["ops"]) or "默认配方"
        if best.get("parent", "").startswith("repair"):
            ops_zh = "修复父本 " + ops_zh
        if rec["promoted"]:
            self.log(f"=== 第{g}代：候选{best['cid']}[{ops_zh}] 晋升为新冠军 ===")
            model, tok = load_model(self.base, best["adapter"])
            rec["test"] = evaluate(model, tok, self.evalset("test"), Path(best["dir"]), "test", log=self.log)
            del model
            free()
            rec.update({"adapter": best["adapter"], "eval_dir": best["dir"], "recipe": best["recipe"], "dev": best["dev"]})
            rec["reflection"].append(f"决策：候选{best['cid']}（{ops_zh}）通过门控晋升。" + "；".join(best["gate_reasons"]))
        else:
            rec["reflection"].append(f"决策：最佳候选{best['cid']}（{ops_zh}）未通过门控，保留第{ch['gen']}代冠军。原因："
                                     + "；".join(best["gate_reasons"]))
        means = self.bandit.means()
        tried = [o for o, (a, b) in self.bandit.s.items() if a + b > 2]
        top = sorted(tried, key=means.get, reverse=True)[:3]
        untried = len(OP_ZH) - len(tried)
        rec["reflection"].append("元优化器经验（已尝试算子）：成功率后验最高的是 "
                                 + "、".join(f"{OP_ZH[o]}({means[o]:.2f})" for o in top) + f"；另有 {untried} 个算子尚未尝试")
        new_synth, notes = adapt_synth(self.state["synth"], (rec.get("synth") or {}).get("if"), rec["promoted"])
        rec["reflection"] += notes
        rec["synth_next"] = new_synth
        rec["seconds"] = round(time.time() - t0)
        self.state["synth"] = new_synth
        self.state["generations"].append(rec)
        if rec["promoted"]:
            self.state["champion"] = g
        self.save()
        for line in rec["reflection"]:
            self.log("  · " + line)

    def run(self, n_gen: int):
        if not self.state["generations"]:
            self.gen0()
        start = self.state["generations"][-1]["gen"] + 1
        for g in range(start, start + n_gen):
            self.generation(g)
            try:
                from .report import build_report
                build_report(self.dir)
            except Exception as e:  # report must never kill the loop
                self.log(f"[report] failed: {e!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "pilot.yaml"))
    ap.add_argument("--generations", type=int, default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    D.prepare(seed=cfg.get("data_seed", 1234))
    run = Run(cfg)
    run.run(a.generations or cfg["generations"])


if __name__ == "__main__":
    main()
