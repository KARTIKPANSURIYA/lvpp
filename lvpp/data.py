"""Dataset loading (Hugging Face ``datasets``)."""
from __future__ import annotations

import random

from .answers import extract_boxed, gsm8k_gold

MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")


def _gsm8k_rows(split: str):
    """GSM8K rows as ``[{question, answer}]`` in the official order.

    Uses ``datasets`` when it imports; otherwise reads the parquet file from the
    Hub directly (e.g. a Python built without ``lzma``, which ``datasets`` needs).
    """
    try:
        from datasets import load_dataset
    except Exception:  # noqa: BLE001
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download, list_repo_files

        name = next(f for f in list_repo_files("openai/gsm8k", repo_type="dataset")
                    if f.startswith(f"main/{split}-") and f.endswith(".parquet"))
        path = hf_hub_download("openai/gsm8k", name, repo_type="dataset")
        return pq.read_table(path).select(["question", "answer"]).to_pylist()
    return list(load_dataset("openai/gsm8k", "main", split=split))


def gsm8k(split: str):
    """List of ``{idx, question, gold, answer}`` for a GSM8K split."""
    return [dict(idx=i, question=ex["question"].strip(), gold=gsm8k_gold(ex["answer"]),
                 answer=ex["answer"]) for i, ex in enumerate(_gsm8k_rows(split))]


def gsm8k_raw(split: str):
    """Raw GSM8K rows (indexable, each with ``question`` and ``answer``)."""
    return _gsm8k_rows(split)


def math500():
    from datasets import load_dataset

    return [dict(question=e["problem"].strip(), gold=e["answer"].strip())
            for e in load_dataset("HuggingFaceH4/MATH-500", split="test")]


def math_train(seed: int = 42):
    """MATH training problems with a boxed answer, shuffled with seed 42."""
    from datasets import load_dataset

    rows = []
    for c in MATH_SUBJECTS:
        for e in load_dataset("EleutherAI/hendrycks_math", c, split="train"):
            g = extract_boxed(e["solution"])
            if g:
                rows.append(dict(question=e["problem"].strip(), gold=g))
    random.Random(seed).shuffle(rows)
    return rows
