"""Statistics used in the paper: exact McNemar, paired CIs, clustered bootstrap."""
from __future__ import annotations

from math import comb

import numpy as np
from sklearn.metrics import roc_auc_score


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar test on discordant counts b and c."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    return min(2 * sum(comb(n, i) for i in range(k + 1)) / 2**n, 1.0)


def paired(h_new: np.ndarray, h_ref: np.ndarray) -> dict:
    """Paired comparison of two per-question hit vectors on identical questions."""
    h_new, h_ref = np.asarray(h_new, bool), np.asarray(h_ref, bool)
    b = int((h_ref & ~h_new).sum())
    c = int((~h_ref & h_new).sum())
    d = h_new.astype(float) - h_ref.astype(float)
    se = 100 * d.std(ddof=1) / np.sqrt(len(d))
    diff = 100 * float(d.mean())
    return dict(
        acc=100 * float(h_new.mean()),
        diff=diff,
        ci_lo=diff - 1.96 * se,
        ci_hi=diff + 1.96 * se,
        p=mcnemar_exact(b, c),
        b=b,
        c=c,
    )


def clustered_bootstrap_auc_diff(y, s_a, s_b, K: int = 8, n_boot: int = 1000, seed: int = 0):
    """95% CI of AUC(s_a) - AUC(s_b), resampling whole questions (clusters of K)."""
    y = np.asarray(y)
    nq = len(y) // K
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        qs = rng.integers(0, nq, nq)
        idx = np.concatenate([np.arange(q * K, q * K + K) for q in qs])
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], s_a[idx]) - roc_auc_score(y[idx], s_b[idx]))
    return tuple(np.percentile(diffs, [2.5, 97.5]))


def mean_sd(values, ddof: int = 0):
    """Mean and standard deviation (population by default, as reported in the paper)."""
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=ddof))
