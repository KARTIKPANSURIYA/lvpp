"""Answer normalisation, extraction and gold parsing.

These functions are ported verbatim (in behaviour) from the notebooks that
produced the paper's caches. Changing them changes which candidates count as
correct, so they are covered by ``tests/test_answers.py``.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"[-+]?\d*\.?\d+")
_ANSWER_IS = re.compile(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", re.I)


def norm(x: str | None) -> str:
    """Canonical form used for every answer comparison in the paper."""
    t = (x or "").strip().replace(",", "").replace("$", "")
    return re.sub(r"\s+", " ", t).rstrip(".")


def last_num(t: str) -> str:
    m = _NUM.findall(norm(t))
    return norm(m[-1]) if m else ""


def gsm8k_gold(answer_field: str) -> str:
    """Gold answer from a GSM8K ``answer`` field (text after ``####``)."""
    t = answer_field.split("####")[-1] if "####" in answer_field else answer_field
    m = _NUM.findall(t.replace(",", "").replace("$", ""))
    return (m[-1] if m else "").strip()


def extract_fewshot(generation: str) -> str:
    """Few-shot extractor used for the main draw (``verifier_upgrade``).

    Cut the continuation at the next ``Question:``, prefer the number after
    ``The answer is``, otherwise take the last number.
    """
    seg = generation.split("Question:")[0]
    mm = _ANSWER_IS.search(seg)
    return last_num(mm.group(1)) if mm else last_num(seg)


def clip_continuation(t: str) -> str:
    """Remove a hallucinated next exemplar (used by the prefix-draw capture)."""
    cut = len(t)
    for marker in ("\nQuestion:", "Question:", "\nQ:"):
        i = t.find(marker)
        if i != -1:
            cut = min(cut, i)
    return t[:cut].rstrip()


def extract_fewshot_clipped(generation: str) -> str:
    """Extractor used by the prefix-pruning capture (``abort_*`` caches)."""
    t = clip_continuation(generation)
    m = re.findall(r"answer is\s*\$?(-?[\d,]+(?:\.\d+)?)", t, flags=re.I) or _NUM.findall(t)
    return norm(m[-1]) if m else ""


def extract_boxed(t: str) -> str:
    """Contents of the last ``\\boxed{...}`` with balanced braces."""
    i = t.rfind("\\boxed{")
    if i == -1:
        return ""
    j, depth, out = i + 7, 1, []
    while j < len(t) and depth > 0:
        ch = t[j]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        if depth > 0:
            out.append(ch)
        j += 1
    return "".join(out).strip()


def extract_chat_numeric(generation: str) -> str:
    """Extractor for chat-template generators on GSM8K (Instruct / Math-Instruct / 3B)."""
    b = extract_boxed(generation)
    return last_num(b) if b else last_num(generation.split("Question:")[0])


def make_equiv(use_math_verify: bool):
    """Return the answer-equivalence predicate.

    GSM8K uses normalised string equality. MATH-500 uses ``math-verify``
    symbolic equivalence when installed, falling back to string equality.
    """
    if not use_math_verify:
        return lambda a, b: norm(a) == norm(b)
    try:
        from math_verify import parse, verify  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        return lambda a, b: norm(a) == norm(b)

    def equiv(a: str, b: str) -> bool:
        if norm(a) == norm(b):
            return True
        try:
            return bool(verify(parse("$" + b + "$"), parse("$" + a + "$")))
        except Exception:
            return False

    return equiv
