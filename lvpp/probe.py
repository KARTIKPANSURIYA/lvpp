"""Stand-alone correctness classification on the verifier pool (Section V-A text).

Port of the feature-set probe: a random 85/15 split of pool candidates, the
same MLP shape without separate scalar input, 8 epochs, batch 256, threshold 0.5.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .verifier import DEVICE, set_seed


class _MLP(nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(0.1), nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def probe_accuracy(X: torch.Tensor, y: torch.Tensor, seeds=(0, 1, 2), device: str = DEVICE):
    """Held-out accuracy of a correctness classifier on features ``X``."""
    X = X.float()
    mu, sd = X.mean(0), X.std(0).clamp(min=1e-6)
    X = ((X - mu) / sd).to(device)
    yy = y.float().to(device)
    accs = []
    for seed in seeds:
        set_seed(seed)
        n = len(yy)
        idx = torch.randperm(n)
        tr, va = idx[: int(0.85 * n)], idx[int(0.85 * n):]
        m = _MLP(X.shape[1]).to(device)
        pos = float(yy[tr].mean())
        lf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-3)], device=device))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
        for _ in range(8):
            m.train()
            perm = tr[torch.randperm(len(tr))]
            for i in range(0, len(perm), 256):
                b = perm[i:i + 256]
                loss = lf(m(X[b]), yy[b])
                opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            accs.append(float(((torch.sigmoid(m(X[va])) > 0.5).float() == yy[va]).float().mean()))
    return float(np.mean(accs)), float(np.std(accs))


def majority_class_rate(y: torch.Tensor) -> float:
    p = float(y.float().mean())
    return max(p, 1 - p)
