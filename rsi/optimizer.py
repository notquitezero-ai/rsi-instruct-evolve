"""Meta-optimizer: proposes training recipes, decides promotion, and adapts the data engine.

Two coupled levels
  1. Recipe search (outer, evolutionary): candidates = champion recipe + 1-2 mutation operators.
     Operators are chosen by Thompson sampling over their historical success (Beta posterior), so
     the system learns *which kinds of changes* tend to help -- the optimizer itself improves.
  2. Data-engine control (inner, feedback): sampling k / temperature / curriculum strength are
     adjusted from the observed acceptance rates of self-generated data.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

DEFAULT_RECIPE = {
    "lr": 2e-4, "epochs": 2, "lora_r": 16, "alpha_ratio": 2, "dropout": 0.05, "max_len": 1024,
    "token_budget": 8192, "grad_accum": 2,
    "n_seed": 3000, "n_math_seed": 1000, "self_frac": 1.0, "n_rule": 600, "init": "base",
    "seed_source": "alpaca",  # "alpaca" = external reference answers, "self" = champion's own (self-distilled)
}

DEFAULT_SYNTH = {"n_if_prompts": 800, "n_math": 300, "k": 4, "temperature": 0.8, "curriculum": 1.0,
                 "n_seed_distill": 2000}
SYNTH_CAPS = {"n_if_prompts": 800, "k": 8}  # time budget: keep one generation's self-play ~30 min on one GPU

_R = [8, 16, 32, 64]
_E = [1, 1.5, 2, 3]


def _step(lst, v, d):
    i = min(range(len(lst)), key=lambda j: abs(lst[j] - v))
    return lst[max(0, min(len(lst) - 1, i + d))]


def _clip(v, lo, hi):
    return max(lo, min(hi, v))


OPS = {
    "lr_up": lambda r: r.update(lr=_clip(r["lr"] * 1.6, 3e-5, 8e-4)),
    "lr_down": lambda r: r.update(lr=_clip(r["lr"] / 1.6, 3e-5, 8e-4)),
    "epochs_up": lambda r: r.update(epochs=_step(_E, r["epochs"], 1)),
    "epochs_down": lambda r: r.update(epochs=_step(_E, r["epochs"], -1)),
    "rank_up": lambda r: r.update(lora_r=_step(_R, r["lora_r"], 1)),
    "rank_down": lambda r: r.update(lora_r=_step(_R, r["lora_r"], -1)),
    "seed_more": lambda r: r.update(n_seed=int(_clip(r["n_seed"] * 1.5, 300, 12000))),
    "seed_less": lambda r: r.update(n_seed=int(_clip(r["n_seed"] / 1.5, 300, 12000))),
    "math_seed_more": lambda r: r.update(n_math_seed=int(_clip(max(200, r["n_math_seed"]) * 1.5, 0, 5000))),
    "math_seed_less": lambda r: r.update(n_math_seed=int(_clip(r["n_math_seed"] / 1.5, 0, 5000))),
    "self_more": lambda r: r.update(self_frac=_clip(r["self_frac"] + 0.25, 0.0, 1.0)),
    "self_less": lambda r: r.update(self_frac=_clip(r["self_frac"] - 0.25, 0.0, 1.0)),
    "rule_more": lambda r: r.update(n_rule=int(_clip(max(200, r["n_rule"]) * 1.6, 0, 4000))),
    "rule_less": lambda r: r.update(n_rule=int(_clip(r["n_rule"] / 1.6, 0, 4000))),
    "init_flip": lambda r: r.update(init="champion" if r["init"] == "base" else "base"),
    # data-source ablations: test whether an external source is what causes a regression
    "math_seed_off": lambda r: r.update(n_math_seed=0),
    "seed_half": lambda r: r.update(n_seed=int(_clip(r["n_seed"] / 2, 300, 12000))),
    "seedsrc_self": lambda r: r.update(seed_source="self"),
    "seedsrc_ext": lambda r: r.update(seed_source="alpaca"),
}
OP_ZH = {
    "lr_up": "学习率↑", "lr_down": "学习率↓", "epochs_up": "轮数↑", "epochs_down": "轮数↓",
    "rank_up": "LoRA秩↑", "rank_down": "LoRA秩↓", "seed_more": "种子数据↑", "seed_less": "种子数据↓",
    "math_seed_more": "数学金标↑", "math_seed_less": "数学金标↓", "self_more": "自生成数据↑",
    "self_less": "自生成数据↓", "rule_more": "规则增强↑", "rule_less": "规则增强↓", "init_flip": "初始化切换",
    "math_seed_off": "去掉数学金标", "seed_half": "种子数据减半",
    "seedsrc_self": "种子自蒸馏", "seedsrc_ext": "种子用外部答案",
}


class Bandit:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.s = {k: [1.0, 1.0] for k in OPS}
        if self.path.exists():
            self.s.update(json.loads(self.path.read_text(encoding="utf-8")))

    def save(self):
        self.path.write_text(json.dumps(self.s, indent=2), encoding="utf-8")

    def sample_ops(self, rng: random.Random, n_ops: int, exclude: set[str] = frozenset(),
                   claimed: set[str] | None = None) -> list[str]:
        """``claimed``: untried arms already forced by sibling candidates this generation (updated in place)."""
        draws = {k: rng.betavariate(*v) for k, v in self.s.items() if k not in exclude}
        # every arm gets tried once before Thompson sampling takes over: a newly added operator
        # would otherwise only be picked by luck against arms whose posteriors already moved
        claimed = claimed if claimed is not None else set()
        untried = sorted(k for k in draws if sum(self.s[k]) <= 2 and k not in claimed)
        first = [rng.choice(untried)] if untried else []
        claimed.update(first)
        ranked = first + sorted(draws, key=draws.get, reverse=True)
        chosen = []
        for k in ranked:
            if k in chosen:
                continue
            base = k.rsplit("_", 1)[0]
            if any(c.rsplit("_", 1)[0] == base for c in chosen):  # no up+down of the same gene
                continue
            chosen.append(k)
            if len(chosen) == n_ops:
                break
        return chosen

    def update(self, ops: list[str], success: bool, weight: float = 1.0):
        for o in ops:
            self.s[o][0 if success else 1] += weight

    def means(self) -> dict[str, float]:
        return {k: a / (a + b) for k, (a, b) in self.s.items()}


# Failure-directed repair: when a candidate was clearly better but blocked by a guard metric, the
# next generation mutates *that* near-miss recipe with operators plausibly fixing the failure mode.
REPAIR_OPS = {
    "math": ["lr_down", "epochs_down", "seed_less", "seed_half", "math_seed_more", "math_seed_less", "math_seed_off",
             "self_more", "rank_down", "seedsrc_self"],
    "degen": ["epochs_up", "lr_up", "seed_more", "rule_less", "self_less"],
}
FAILURE_ZH = {"math": "数学能力回退", "degen": "退化率上升"}


def failure_mode(reasons: list[str]) -> str | None:
    text = "".join(reasons)
    if "数学" in text:
        return "math"
    if "退化" in text:
        return "degen"
    return None


def propose(champion_recipe: dict | None, bandit: Bandit, n: int, rng: random.Random,
            default: dict | None = None, near_miss: dict | None = None) -> list[dict]:
    """near_miss = {"recipe": ..., "mode": "math"|"degen"} from the previous generation, if any."""
    base = copy.deepcopy(champion_recipe or default or DEFAULT_RECIPE)
    cands, used = [], set()
    if champion_recipe is None and near_miss is None:  # bootstrap: the default recipe itself is a candidate
        cands.append({"recipe": copy.deepcopy(base), "ops": [], "parent": "default"})
        used.add(())
    n_repair = (n + 1) // 2 if near_miss else 0
    tries, claimed = 0, set()
    while len(cands) < n and tries < 50:
        tries += 1
        repair = len(cands) < n_repair
        allowed = set(REPAIR_OPS[near_miss["mode"]]) if repair else set(OPS)
        ops = bandit.sample_ops(random.Random(rng.random()), rng.choice([1, 2]), exclude=set(OPS) - allowed,
                                claimed=claimed)
        parent = near_miss["recipe"] if repair else base
        r = copy.deepcopy(parent)
        effective = []  # drop operators that change nothing here (already at a bound / value), so the
        for o in ops:   # bandit never gets credited for a no-op; a dropped forced arm stays claimed
            before = {**DEFAULT_RECIPE, **r}
            OPS[o](r)
            if {**DEFAULT_RECIPE, **r} != before:
                effective.append(o)
        ops = effective
        if champion_recipe is None and r["init"] == "champion":
            r["init"] = "base"  # nothing to continue from yet
        key = (repair, tuple(sorted(ops)))
        if not ops or key in used:
            continue
        used.add(key)
        cands.append({"recipe": r, "ops": ops, "parent": f"repair:{near_miss['mode']}" if repair else "champion"})
    return cands


def gate(cand_m: dict, champ_m: dict, boot: dict, cfg: dict) -> tuple[bool, list[str]]:
    reasons = []
    ok = True
    if boot["diff"] <= 0:
        ok = False
        reasons.append(f"综合分未提升 (Δ={boot['diff']:+.4f})")
    if boot["p_better"] < cfg.get("p_better", 0.8):
        ok = False
        reasons.append(f"统计置信不足 P(更优)={boot['p_better']:.2f} < {cfg.get('p_better', 0.8)}")
    if cand_m["math_acc"] < champ_m["math_acc"] - cfg.get("math_tol", 0.03):
        ok = False
        reasons.append(f"数学能力回退 {champ_m['math_acc']:.3f}→{cand_m['math_acc']:.3f}")
    if cand_m["degenerate_rate"] > champ_m["degenerate_rate"] + cfg.get("degen_tol", 0.03):
        ok = False
        reasons.append(f"退化率上升 {champ_m['degenerate_rate']:.3f}→{cand_m['degenerate_rate']:.3f}")
    if ok:
        reasons.append(f"Δ综合分={boot['diff']:+.4f}, P(更优)={boot['p_better']:.2f}, 90%CI=[{boot['ci90'][0]:+.3f},{boot['ci90'][1]:+.3f}]")
    return ok, reasons


def adapt_synth(synth: dict, if_stats: dict | None, promoted: bool) -> tuple[dict, list[str]]:
    """Feedback control of the data engine; returns new settings and the reasons."""
    s = dict(synth)
    notes = []
    if if_stats:
        pr = if_stats["sample_pass_rate"]
        if pr < 0.15 and s["k"] < SYNTH_CAPS["k"]:
            s["k"] = min(SYNTH_CAPS["k"], s["k"] + 2)
            s["temperature"] = round(max(0.5, s["temperature"] - 0.1), 2)
            notes.append(f"自生成通过率低({pr:.2f}) → 采样数k↑{s['k']}, 温度↓{s['temperature']}")
        elif pr > 0.6:
            s["temperature"] = round(min(1.2, s["temperature"] + 0.1), 2)
            s["curriculum"] = round(min(3.0, s["curriculum"] + 0.5), 2)
            notes.append(f"自生成通过率高({pr:.2f}) → 温度↑{s['temperature']}, 课程难度↑{s['curriculum']}")
    if not promoted and s["n_if_prompts"] < SYNTH_CAPS["n_if_prompts"]:
        s["n_if_prompts"] = int(min(SYNTH_CAPS["n_if_prompts"], s["n_if_prompts"] * 1.25))
        notes.append(f"本代未晋升 → 扩大自生成规模 n_if_prompts={s['n_if_prompts']}")
    return s, notes
