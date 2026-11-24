"""Loaders for the three cache layouts written by ``lvpp.capture``.

Layout A - verifier pool (``pool.pt``): dict of flat lists, one entry per candidate:
    {"mid_mean": [...], "upper_mean": [...], "last_mean": [...], ..., "scalars": [...],
     "labels": [...], "qid": [...], "n_done": int}

Layout B - per-question dict (``testfeat.pt``, chat caches, span caches):
    {question_index: {"gold": str, "cands": [{<feature keys>, "scalar", "answer"}, ...]}}

Layout C - prefix shards (``<root>/{pool,test}/shard_XXXX.pt``): list of records
    {"gold", "answers", "n_total", "scalar", "feats": {B: K x 3d},
     "finished_before": {B: [...]}, "prefix_lp": {B: [...]}}
"""
from __future__ import annotations

import os

import numpy as np
import torch

from .answers import norm
from .capture import MULTI
from .selection import CandidateSet
from .verifier import find_logprob_col


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_pool_A(path: str, variants=MULTI, n_questions: int | None = None):
    """Verifier training pool -> (P, Z, y, qid)."""
    o = _load(path)
    qid = np.asarray(o["qid"])
    keep = np.ones(len(qid), bool) if n_questions is None else qid < n_questions
    P = torch.cat([torch.stack(o[v]).float() for v in variants], -1)[torch.from_numpy(keep)]
    Z = torch.stack(o["scalars"]).float()[torch.from_numpy(keep)]
    y = torch.tensor(o["labels"]).float()[torch.from_numpy(keep)]
    return P, Z, y, qid[keep]


def load_pool_legacy(path: str):
    """Earlier single-layer pools (Table IV): ``{"pooled": N x d, "scalars": N x 5, "labels": N}``."""
    o = _load(path)
    P = torch.as_tensor(o["pooled"]).float()
    Z = torch.as_tensor(o["scalars"]).float()
    y = torch.as_tensor(o["labels"]).float()
    return P, Z, y


def load_dict_B(path: str, variants=MULTI, equiv=None, K: int = 8) -> CandidateSet:
    """Per-question cache -> CandidateSet (questions sorted by index)."""
    o = _load(path)
    qs = sorted(o.keys())
    eq = equiv or (lambda a, b: norm(a) == norm(b))
    P = torch.cat([torch.stack([c[v].float() for q in qs for c in o[q]["cands"]]) for v in variants], -1)
    Z = torch.stack([c["scalar"].float() for q in qs for c in o[q]["cands"]])
    answers = [[c["answer"] for c in o[q]["cands"]] for q in qs]
    gold = [str(o[q]["gold"]) for q in qs]
    y = torch.tensor([1.0 if (a and eq(a, g)) else 0.0 for row, g in zip(answers, gold) for a in row])
    extra = {"qids": qs}
    if "has_trigger" in o[qs[0]]["cands"][0]:
        extra["has_trigger"] = np.array([bool(c["has_trigger"]) for q in qs for c in o[q]["cands"]])
    return CandidateSet(answers, gold, P, Z, y, lpc=find_logprob_col(Z.numpy()), K=K,
                        equiv=eq, extra=extra)


def dict_B_as_pool(path: str, variants=MULTI, equiv=None, limit: int | None = None):
    """A layout-B cache used as a training pool -> (P, Z, y)."""
    cs = load_dict_B(path, variants, equiv)
    if limit is not None:
        n = sum(1 for q in cs.extra["qids"] if q < limit) * cs.K
        return cs.P[:n], cs.Z[:n], cs.y[:n]
    return cs.P, cs.Z, cs.y


def load_prefix_C(root: str, tag: str, B: int, K: int = 8) -> CandidateSet:
    """Prefix shards at checkpoint B -> CandidateSet with pruning metadata in ``extra``."""
    rows = []
    d = f"{root}/{tag}"
    for f in sorted(os.listdir(d)):
        if f.startswith("shard"):
            rows += _load(f"{d}/{f}")
    P = torch.stack([r["feats"][B][k].float() for r in rows for k in range(K)])
    Z = torch.cat([r["scalar"] for r in rows]).float()
    answers = [r["answers"] for r in rows]
    gold = [r["gold"] for r in rows]
    y = torch.tensor([1.0 if norm(a) == norm(g) else 0.0 for row, g in zip(answers, gold) for a in row])
    extra = dict(
        n_total=np.array([r["n_total"][k] for r in rows for k in range(K)], dtype=float),
        finished=np.array([r["finished_before"][B][k] for r in rows for k in range(K)], dtype=bool),
        prefix_lp=(np.array([r["prefix_lp"][B][k] for r in rows for k in range(K)], dtype=float)
                   if "prefix_lp" in rows[0] else None),
        B=B,
    )
    return CandidateSet(answers, gold, P, Z, y, lpc=find_logprob_col(Z.numpy()), K=K, extra=extra)
