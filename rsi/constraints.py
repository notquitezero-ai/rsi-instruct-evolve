"""Verifiable instruction constraints (IFEval-style).

Every constraint renders an instruction sentence that is appended to a task prompt and
exposes a deterministic ``check(response) -> bool``.  Because the checks are programmatic,
the system can score itself and filter its own generations without a human or external
judge -- this is what closes the self-improvement loop.

Some constraints also implement ``apply(response)`` which rewrites an existing response so
that it satisfies the constraint (used by the rule-based augmentation data source).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Callable, Optional

WORD_RE = re.compile(r"\b\w+\b")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text))


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


END_PHRASES = [
    "Is there anything else I can help with?",
    "Any other questions?",
    "Let me know if you have additional questions.",
    "That is all you need to know.",
]
START_PHRASES = ["Here is my answer:", "Certainly!", "Great question."]
KEYWORDS = [
    "balance", "future", "simple", "process", "energy", "community", "quality",
    "history", "design", "value", "growth", "nature", "system", "team", "story",
]
FORBIDDEN = ["very", "really", "important", "however", "example", "also"]


@dataclass
class Constraint:
    kind: str
    params: dict = field(default_factory=dict)

    # ----- rendering -------------------------------------------------------------------
    def text(self) -> str:
        p = self.params
        k = self.kind
        if k == "lowercase":
            return "Your entire response should be in English, and in all lowercase letters. No capital letters are allowed."
        if k == "uppercase":
            return "Your entire response should be in English, and in all capital letters."
        if k == "no_comma":
            return "In your entire response, refrain from the use of any commas."
        if k == "word_max":
            return f"Answer with less than {p['n']} words."
        if k == "word_min":
            return f"Answer with at least {p['n']} words."
        if k == "end_phrase":
            return f"Finish your response with this exact phrase: \"{p['phrase']}\". No other words should follow this phrase."
        if k == "start_phrase":
            return f"Begin your response with the exact phrase \"{p['phrase']}\"."
        if k == "quotation":
            return "Wrap your entire response with double quotation marks."
        if k == "title":
            return "Your answer must contain a title, wrapped in double angular brackets, such as <<poem of joy>>."
        if k == "postscript":
            return "At the end of your response, please explicitly add a postscript starting with P.S."
        if k == "bullets":
            return (f"Your answer must contain exactly {p['n']} bullet points. "
                    "Use the markdown bullet points such as:\n* This is point 1.\n* This is point 2")
        if k == "json":
            return "Entire output should be wrapped in JSON format. You can use markdown ticks such as ```."
        if k == "keywords":
            return f"Include the keywords {', '.join(repr(w) for w in p['words'])} in the response."
        if k == "forbidden":
            return f"Do not include the word \"{p['word']}\" in the response."
        if k == "paragraphs":
            return (f"There should be {p['n']} paragraphs. "
                    "Paragraphs are separated with the markdown divider: ***")
        if k == "highlight":
            return f"Highlight at least {p['n']} sections in your answer with markdown, i.e. *highlighted section*."
        if k == "placeholders":
            return f"The response must contain at least {p['n']} placeholders represented by square brackets, such as [address]."
        if k == "sections":
            return (f"Your response must have {p['n']} sections. Mark the beginning of each section "
                    f"with SECTION X, such as:\nSECTION 1\n[content of section 1]\nSECTION 2\n[content of section 2]")
        raise KeyError(k)

    # ----- verification ----------------------------------------------------------------
    def check(self, resp: str) -> bool:
        p = self.params
        k = self.kind
        r = resp.strip()
        if not r:
            return False
        if k == "lowercase":
            return r == r.lower() and any(c.isalpha() for c in r)
        if k == "uppercase":
            return r == r.upper() and any(c.isalpha() for c in r)
        if k == "no_comma":
            return "," not in r and "，" not in r
        if k == "word_max":
            return count_words(r) < p["n"]
        if k == "word_min":
            return count_words(r) >= p["n"]
        if k == "end_phrase":
            return r.endswith(p["phrase"])
        if k == "start_phrase":
            return r.startswith(p["phrase"])
        if k == "quotation":
            return len(r) > 1 and r[0] == '"' and r[-1] == '"'
        if k == "title":
            return re.search(r"<<[^\n<>]+>>", r) is not None
        if k == "postscript":
            return re.search(r"(^|\n)\s*p\.\s?s\.", r, flags=re.IGNORECASE) is not None
        if k == "bullets":
            n = len(re.findall(r"^\s*[\*\-]\s+\S", r, flags=re.MULTILINE))
            return n == p["n"]
        if k == "json":
            try:
                json.loads(_strip_fence(r))
                return True
            except Exception:
                return False
        if k == "keywords":
            return all(re.search(rf"\b{re.escape(w)}\b", r, flags=re.IGNORECASE) for w in p["words"])
        if k == "forbidden":
            return re.search(rf"\b{re.escape(p['word'])}\b", r, flags=re.IGNORECASE) is None
        if k == "paragraphs":
            parts = [x for x in re.split(r"\s?\*\*\*\s?", r)]
            return len(parts) == p["n"] and all(x.strip() for x in parts)
        if k == "highlight":
            return len(re.findall(r"\*[^\n\*]+\*", r)) >= p["n"]
        if k == "placeholders":
            return len(re.findall(r"\[[^\[\]\n]+\]", r)) >= p["n"]
        if k == "sections":
            parts = re.split(r"\s?SECTION\s+\d+\s?", r)
            return len(parts) - 1 == p["n"]
        raise KeyError(k)

    # ----- optional programmatic rewriting ---------------------------------------------
    def apply(self, resp: str, instruction: str = "") -> Optional[str]:
        p = self.params
        k = self.kind
        r = resp.strip()
        if k == "lowercase":
            return r.lower()
        if k == "uppercase":
            return r.upper()
        if k == "no_comma":
            return re.sub(r"\s*,\s*", " ", r).replace("，", " ")
        if k == "end_phrase":
            return r + "\n\n" + p["phrase"]
        if k == "start_phrase":
            return p["phrase"] + " " + r
        if k == "quotation":
            return '"' + r.replace('"', "'") + '"'
        if k == "title":
            words = WORD_RE.findall(instruction)[:5] or ["answer"]
            return f"<<{' '.join(words).lower()}>>\n\n" + r
        if k == "postscript":
            return r + "\n\nP.S. I hope this helps."
        if k == "json":
            return json.dumps({"response": r}, ensure_ascii=False, indent=2)
        if k == "word_max" and count_words(r) < p["n"]:
            return r
        if k == "forbidden" and self.check(r):
            return r
        return None

    def to_dict(self) -> dict:
        return {"kind": self.kind, "params": self.params}

    @staticmethod
    def from_dict(d: dict) -> "Constraint":
        return Constraint(d["kind"], dict(d.get("params", {})))


# ---------------------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------------------
_FACTORIES: dict[str, Callable[[random.Random], dict]] = {
    "lowercase": lambda g: {},
    "uppercase": lambda g: {},
    "no_comma": lambda g: {},
    "word_max": lambda g: {"n": g.choice([40, 60, 100])},
    "word_min": lambda g: {"n": g.choice([120, 200, 300])},
    "end_phrase": lambda g: {"phrase": g.choice(END_PHRASES)},
    "start_phrase": lambda g: {"phrase": g.choice(START_PHRASES)},
    "quotation": lambda g: {},
    "title": lambda g: {},
    "postscript": lambda g: {},
    "bullets": lambda g: {"n": g.choice([2, 3, 4, 5])},
    "json": lambda g: {},
    "keywords": lambda g: {"words": g.sample(KEYWORDS, 2)},
    "forbidden": lambda g: {"word": g.choice(FORBIDDEN)},
    "paragraphs": lambda g: {"n": g.choice([2, 3, 4])},
    "highlight": lambda g: {"n": g.choice([1, 2, 3])},
    "placeholders": lambda g: {"n": g.choice([1, 2, 3])},
    "sections": lambda g: {"n": g.choice([2, 3, 4])},
}
KINDS = list(_FACTORIES)
APPLICABLE = ["lowercase", "uppercase", "no_comma", "end_phrase", "start_phrase", "quotation",
              "title", "postscript", "json"]

KIND_ZH = {
    "lowercase": "全小写", "uppercase": "全大写", "no_comma": "禁用逗号", "word_max": "字数上限",
    "word_min": "字数下限", "end_phrase": "指定结尾", "start_phrase": "指定开头", "quotation": "引号包裹",
    "title": "<<标题>>", "postscript": "P.S.附言", "bullets": "精确条目数", "json": "JSON格式",
    "keywords": "包含关键词", "forbidden": "禁用词", "paragraphs": "段落数(***)", "highlight": "*高亮*",
    "placeholders": "[占位符]", "sections": "SECTION分节",
}

# structural constraints that fight with each other
_LAYOUT = {"json", "quotation", "bullets", "paragraphs", "sections", "title", "postscript",
           "end_phrase", "start_phrase"}
_CONFLICTS: dict[str, set[str]] = {
    "lowercase": {"uppercase", "sections", "title", "start_phrase", "end_phrase", "postscript"},
    "uppercase": {"lowercase", "keywords", "forbidden"},
    "word_max": {"word_min", "sections", "paragraphs"},
    "word_min": {"word_max"},
    "json": _LAYOUT | {"no_comma", "highlight", "placeholders"},
    "quotation": {"json", "end_phrase", "start_phrase", "postscript", "sections"},
    "bullets": {"json", "paragraphs", "sections"},
    "paragraphs": {"json", "bullets", "sections", "word_max"},
    "sections": {"json", "bullets", "paragraphs", "lowercase", "quotation", "word_max"},
    "end_phrase": {"json", "quotation", "postscript", "lowercase"},
    "postscript": {"json", "quotation", "end_phrase", "lowercase"},
    "start_phrase": {"json", "quotation", "lowercase"},
    "title": {"json", "lowercase"},
    "no_comma": {"json"},
    "highlight": {"json"},
    "placeholders": {"json"},
}


def compatible(kind: str, chosen: list[str]) -> bool:
    for c in chosen:
        if kind == c or kind in _CONFLICTS.get(c, set()) or c in _CONFLICTS.get(kind, set()):
            return False
    return True


def sample_constraints(rng: random.Random, n: int, weights: Optional[dict[str, float]] = None,
                       kinds: Optional[list[str]] = None) -> list[Constraint]:
    kinds = list(kinds or KINDS)
    w = [max(1e-6, (weights or {}).get(k, 1.0)) for k in kinds]
    chosen: list[str] = []
    for _ in range(50):
        if len(chosen) >= n:
            break
        k = rng.choices(kinds, weights=w, k=1)[0]
        if compatible(k, chosen):
            chosen.append(k)
    return [Constraint(k, _FACTORIES[k](rng)) for k in chosen]


def render_prompt(task: str, constraints: list[Constraint]) -> str:
    if not constraints:
        return task
    return task.rstrip() + "\n\n" + " ".join(c.text() for c in constraints)


def check_all(resp: str, constraints: list[Constraint]) -> list[bool]:
    return [c.check(resp) for c in constraints]
