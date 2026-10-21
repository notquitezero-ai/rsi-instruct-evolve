import pytest

from rsi import data as D

pytestmark = pytest.mark.skipif(not (D.DATA / "prepared" / "manifest.json").exists(),
                                reason="needs prepared data (python -m rsi.prepare)")


def test_gate_set_is_fresh_and_deterministic():
    a, b, c = D.gate_set(9), D.gate_set(9), D.gate_set(10)
    assert a == b                                    # reproducible within a generation (resume-safe)
    assert [x["prompt"] for x in a["if"]] != [x["prompt"] for x in c["if"]]  # new constraints each generation
    dev = D.load_split("if_dev")
    assert [x["task"] for x in a["if"]] == [x["task"] for x in dev]          # same held-out tasks
    assert sum(x["prompt"] != y["prompt"] for x, y in zip(a["if"], dev)) > 150
    used = {x["question"] for x in D.load_split("gsm_dev") + D.load_split("gsm_test")}
    assert len(a["math"]) == 150 and not used & {x["question"] for x in a["math"]}
    assert {x["id"] for x in a["math"]} != {x["id"] for x in c["math"]}
