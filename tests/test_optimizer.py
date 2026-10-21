import random

from rsi.evaluate import paired_bootstrap
from rsi.optimizer import DEFAULT_RECIPE, DEFAULT_SYNTH, OPS, REPAIR_OPS, Bandit, adapt_synth, gate, propose


def _rows(if_ok, math_ok):
    rows = [{"id": f"i{k}", "type": "if", "strict": v, "checks": [v], "degenerate": False, "finished": True,
             "tokens": 10, "correct": None} for k, v in enumerate(if_ok)]
    rows += [{"id": f"m{k}", "type": "math", "correct": v, "degenerate": False, "finished": True, "tokens": 10}
             for k, v in enumerate(math_ok)]
    return rows


def test_bootstrap_detects_clear_win_and_tie():
    old = _rows([False] * 100, [True] * 50 + [False] * 50)
    new = _rows([True] * 60 + [False] * 40, [True] * 50 + [False] * 50)
    b = paired_bootstrap(new, old, n_boot=300)
    assert b["diff"] > 0.3 and b["p_better"] > 0.99
    tie = paired_bootstrap(old, old, n_boot=300)
    assert tie["diff"] == 0 and tie["p_better"] == 0


def test_gate_blocks_math_regression():
    champ = {"math_acc": 0.7, "degenerate_rate": 0.1}
    cand = {"math_acc": 0.6, "degenerate_rate": 0.1}
    ok, why = gate(cand, champ, {"diff": 0.05, "p_better": 0.95, "ci90": [0.01, 0.09]}, {})
    assert not ok and any("数学" in w for w in why)


def test_propose_bootstrap_and_mutations(tmp_path):
    b = Bandit(tmp_path / "b.json")
    c = propose(None, b, 2, random.Random(0))
    assert c[0]["ops"] == [] and c[0]["recipe"] == DEFAULT_RECIPE
    assert c[1]["ops"] and c[1]["recipe"]["init"] == "base"
    c2 = propose(DEFAULT_RECIPE, b, 3, random.Random(1))
    assert len({tuple(sorted(x["ops"])) for x in c2}) == 3
    for x in c2:  # never up+down of the same gene
        genes = [o.rsplit("_", 1)[0] for o in x["ops"]]
        assert len(genes) == len(set(genes))


def test_repair_candidates_mutate_near_miss(tmp_path):
    b = Bandit(tmp_path / "b.json")
    nm = {"recipe": {**DEFAULT_RECIPE, "lr": 1e-4}, "mode": "math"}
    c = propose(None, b, 3, random.Random(2), near_miss=nm)
    assert len(c) == 3 and [x["parent"] for x in c].count("repair:math") == 2
    for x in c:
        if x["parent"] == "repair:math":
            assert set(x["ops"]) <= set(REPAIR_OPS["math"])


def test_untried_arms_are_tried_first_and_not_duplicated(tmp_path):
    b = Bandit(tmp_path / "b.json")
    for k in b.s:  # everything tried and failed except two fresh repair arms
        b.s[k] = [1.0, 4.0]
    b.s["seedsrc_self"] = [1.0, 1.0]
    b.s["math_seed_off"] = [1.0, 1.0]
    nm = {"recipe": dict(DEFAULT_RECIPE), "mode": "math"}
    c = propose(DEFAULT_RECIPE, b, 3, random.Random(5), near_miss=nm)
    firsts = [x["ops"][0] for x in c if x["parent"] == "repair:math"]
    assert sorted(firsts) == ["math_seed_off", "seedsrc_self"]


def test_bandit_learns():
    b = Bandit.__new__(Bandit)
    b.s = {k: [1.0, 1.0] for k in OPS}
    for _ in range(10):
        b.update(["lr_down"], True)
        b.update(["rank_up"], False)
    assert b.means()["lr_down"] > 0.85 > 0.15 > b.means()["rank_up"]


def test_adapt_synth():
    s, notes = adapt_synth(DEFAULT_SYNTH, {"sample_pass_rate": 0.05}, promoted=True)
    assert s["k"] == DEFAULT_SYNTH["k"] + 2 and notes
    small = {**DEFAULT_SYNTH, "n_if_prompts": 400}
    s, _ = adapt_synth(small, {"sample_pass_rate": 0.9}, promoted=False)
    assert s["curriculum"] > small["curriculum"] and s["n_if_prompts"] > small["n_if_prompts"]
    s, _ = adapt_synth({**DEFAULT_SYNTH, "n_if_prompts": 800}, None, promoted=False)
    assert s["n_if_prompts"] <= 800  # capped by the time budget
