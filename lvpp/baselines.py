"""Non-learned and comparator scores for Table I."""
from __future__ import annotations

import sys

import numpy as np
import torch


def token_confidence(cs, variant: str = "exp") -> np.ndarray:
    """Weights from mean token log-probability.

    ``exp``     exp(mean log-prob)                      (the row reported in Table I)
    ``softmax`` softmax of mean log-prob within each question
    ``zscore``  within-question z-score, shifted to be positive
    """
    lp = cs.logprob()
    if variant == "exp":
        return np.exp(lp)
    out = np.empty_like(lp)
    for q in range(cs.nq):
        a, b = q * cs.K, (q + 1) * cs.K
        x = lp[a:b]
        if variant == "softmax":
            e = np.exp(x - x.max())
            out[a:b] = e / e.sum()
        elif variant == "zscore":
            z = (x - x.mean()) / (x.std() + 1e-6)
            out[a:b] = z - z.min() + 1e-6
        else:
            raise ValueError(variant)
    return out


def gbt_scores(P_tr, Z_tr, y_tr, P_te, Z_te, seed: int = 0) -> np.ndarray:
    """LiLaVe-family comparator (Table I, GBT row): XGBoost on the same raw
    pooled features + scalars, scored on the test candidates.

    On macOS XGBoost runs in a separate spawned process that never imports
    PyTorch: both libraries load an OpenMP runtime, and having both in one
    process can abort it.
    """
    from ._gbt_worker import fit_predict

    X_tr = torch.cat([P_tr, Z_tr], -1).numpy()
    X_te = torch.cat([P_te, Z_te], -1).numpy()
    y = y_tr.numpy()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if sys.platform != "darwin":
        return fit_predict(X_tr, y, X_te, seed, device)
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fit_predict, X_tr, y, X_te, seed, device).result()
