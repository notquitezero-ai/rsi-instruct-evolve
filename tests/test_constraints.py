import random

from rsi.constraints import APPLICABLE, KINDS, Constraint, compatible, sample_constraints
from rsi.data import answers_match, extract_answer

RESP = "Solar power is clean, cheap and abundant. It helps the planet."


def test_apply_then_check():
    for k in APPLICABLE:
        c = sample_constraints(random.Random(0), 1, kinds=[k])[0]
        out = c.apply(RESP, "Explain solar power")
        assert out is not None and c.check(out), (k, out)


def test_specific_checks():
    assert Constraint("bullets", {"n": 2}).check("Intro\n* a\n* b")
    assert not Constraint("bullets", {"n": 3}).check("* a\n* b")
    assert Constraint("paragraphs", {"n": 2}).check("one\n***\ntwo")
    assert Constraint("sections", {"n": 2}).check("SECTION 1\na\nSECTION 2\nb")
    assert Constraint("json", {}).check('```json\n{"a": 1}\n```')
    assert not Constraint("json", {}).check("{a: 1}")
    assert Constraint("keywords", {"words": ["future", "team"]}).check("The Team builds the future.")
    assert Constraint("forbidden", {"word": "very"}).check("It is everyday stuff.")
    assert Constraint("postscript", {}).check("Hi.\n\nP.S. bye")
    assert Constraint("highlight", {"n": 2}).check("*one* and *two*")
    assert Constraint("placeholders", {"n": 1}).check("Send to [address].")
    assert Constraint("word_max", {"n": 5}).check("one two three four")
    assert not Constraint("lowercase", {}).check("Hello")


def test_sampling_respects_conflicts():
    g = random.Random(1)
    for _ in range(500):
        cs = sample_constraints(g, 3)
        kinds = [c.kind for c in cs]
        for i, k in enumerate(kinds):
            assert compatible(k, kinds[:i])
        for c in cs:
            assert c.text()
    assert set(KINDS) >= set(APPLICABLE)


def test_math_chain_rejects_shortcuts():
    from rsi.synth import valid_math_chain
    assert not valid_math_chain("The answer is 18.")
    assert not valid_math_chain("She has some apples and gives a few away so the rest is left. The answer is 3.")
    good = ("Natalia sold 48 clips in April. In May she sold half as many, so 48 / 2 = 24 clips. "
            "Altogether she sold 48 + 24 = 72 clips over the two months combined.\nThe answer is 72.")
    assert valid_math_chain(good)


def test_substantive_blocks_verifier_gaming():
    from rsi.synth import substantive
    assert not substantive("[address]", ["placeholders"])
    assert not substantive("P.S.", ["uppercase", "no_comma", "postscript"])
    assert substantive("GOLD IS A DENSE YELLOW METAL", ["word_max", "uppercase"])
    assert substantive(" ".join(["word"] * 25), ["title"])


def test_math_extract():
    assert extract_answer("so 3+4=7\nThe answer is 7.") == "7"
    assert extract_answer("The answer is $1,200.") == "1200"
    assert answers_match(extract_answer("total 18 eggs"), "18")
    assert not answers_match(None, "3")
