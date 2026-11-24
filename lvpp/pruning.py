"""Prefix pruning at an absolute token checkpoint B (Section III-C) and its cost (Eq. 4).

At checkpoint B, candidates that have already finished (T_k <= B) cannot be
pruned and are charged in full. Among the still-running candidates, the
``n_kill`` lowest-valued ones are terminated and charged min(B, T_k).

``n_kill`` follows the original experiments: ``round(rho * K)`` lowest values
are selected *after* finished candidates are pushed to +inf, and only the
still-running ones among them are actually killed. With K = 8 this equals
``floor(rho * |R_B|)`` whenever enough candidates are still running.
"""
from __future__ import annotations

import numpy as np


def prune_keep_and_cost(values, finished, n_total, B: int, rho: float, K: int = 8):
    """Apply one pruning rule to every question.

    Parameters
    ----------
    values   : (N,) rule value per candidate; lowest values are pruned first
    finished : (N,) bool, candidate ended at or before B
    n_total  : (N,) generated length T_k of each candidate if run to completion
    Returns ``(keep, mean_cost)`` where ``keep`` is a per-question list of masks.
    """
    values = np.asarray(values, dtype=float)
    finished = np.asarray(finished, dtype=bool)
    n_total = np.asarray(n_total, dtype=float)
    nq = len(values) // K
    n_kill = int(round(rho * K))
    keep, costs = [], []
    for q in range(nq):
        a0 = q * K
        v = values[a0 : a0 + K].copy()
        fin = finished[a0 : a0 + K]
        v[fin] = np.inf
        killed = {k for k in np.argsort(v, kind="stable")[:n_kill].tolist() if not fin[k]}
        keep.append([k not in killed for k in range(K)])
        T = n_total[a0 : a0 + K]
        costs.append(sum(min(B, T[k]) if k in killed else T[k] for k in range(K)))
    return keep, float(np.mean(costs))


def full_cost(n_total, K: int = 8) -> float:
    n_total = np.asarray(n_total, dtype=float)
    return float(n_total.reshape(-1, K).sum(1).mean())


def rule_values(rule: str, *, verifier=None, prefix_logprob=None, n_total=None, labels=None, rng=None):
    """Value per candidate for each pruning rule (lower = pruned first)."""
    if rule == "verifier":
        return np.asarray(verifier, dtype=float)
    if rule == "confidence":
        return np.asarray(prefix_logprob, dtype=float)
    if rule == "longest":  # keep the longest prefixes = prune the shortest final lengths
        return np.asarray(n_total, dtype=float)
    if rule == "shortest":
        return -np.asarray(n_total, dtype=float)
    if rule == "random":
        return (rng or np.random.default_rng(0)).random(len(n_total))
    if rule == "oracle":  # abort incorrect candidates first
        y = np.asarray(labels, dtype=float)
        return y + 1e-6 * (rng or np.random.default_rng(0)).random(len(y))
    raise ValueError(rule)
