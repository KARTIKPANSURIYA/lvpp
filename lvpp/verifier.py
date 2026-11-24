"""The latent verifier: a two-layer perceptron over pooled hidden states.

Recipe (identical across every experiment in the paper):
  * inputs  : pooled hidden states ``P`` (3 x d) and five scalars ``Z``
  * scaling : z-scored with statistics of the *training pool only*
  * model   : Linear(D+5 -> 256) - ReLU - Dropout(0.1) - Linear(256 -> 1)
              (1.18M parameters for Qwen2.5-1.5B, D = 4608)
  * split   : random 90% of pool candidates for training (seeded)
  * loss    : BCE with pos_weight = (1 - pos) / pos on the training part
  * optim   : AdamW, lr 1e-3, batch 128, 8 epochs
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .device import DEVICE  # cuda > mps > cpu

# Scalar layout z(c): [mean_velocity, contraction, max_norm/100, length/budget, mean_logprob]
SCALAR_NAMES = ("mean_velocity", "positive_growth", "max_norm", "length", "mean_logprob")
LOGPROB_INDEX = 4


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_logprob_col(scalars) -> int:
    """The log-probability column is the only one that is always negative."""
    s = np.asarray(scalars, dtype=float)[:2000]
    cols = [i for i in range(s.shape[1]) if s[:, i].max() < 0]
    assert len(cols) == 1, f"expected one all-negative column, got {cols}"
    return cols[0]


class LatentVerifier(nn.Module):
    def __init__(self, d: int, n_scalars: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d + n_scalars, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([p, z], -1)).squeeze(-1)


@dataclass
class FittedVerifier:
    model: LatentVerifier
    mu_p: torch.Tensor
    sd_p: torch.Tensor
    mu_z: torch.Tensor
    sd_z: torch.Tensor

    @torch.no_grad()
    def score(self, P: torch.Tensor, Z: torch.Tensor, device: str = DEVICE) -> np.ndarray:
        """Probability that each candidate is correct."""
        self.model.eval()
        xp = ((P.float() - self.mu_p) / self.sd_p).to(device)
        xz = ((Z.float() - self.mu_z) / self.sd_z).to(device)
        s = torch.sigmoid(self.model(xp, xz)).cpu().numpy()
        return s

    @torch.no_grad()
    def logits(self, P: torch.Tensor, Z: torch.Tensor, device: str = DEVICE) -> np.ndarray:
        self.model.eval()
        xp = ((P.float() - self.mu_p) / self.sd_p).to(device)
        xz = ((Z.float() - self.mu_z) / self.sd_z).to(device)
        return self.model(xp, xz).cpu().numpy()


def fit_verifier(
    P: torch.Tensor,
    Z: torch.Tensor,
    y: torch.Tensor,
    seed: int = 0,
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    hidden: int = 256,
    train_fraction: float = 0.9,
    device: str = DEVICE,
) -> FittedVerifier:
    """Train one verifier with the paper's recipe. ``Z`` may have zero columns."""
    P, Z, y = P.float(), Z.float(), y.float()
    mu_p, sd_p = P.mean(0), P.std(0).clamp(min=1e-6)
    if Z.shape[1] > 0:
        mu_z, sd_z = Z.mean(0), Z.std(0).clamp(min=1e-6)
    else:
        mu_z, sd_z = torch.zeros(0), torch.ones(0)
    xp = ((P - mu_p) / sd_p).to(device)
    xz = ((Z - mu_z) / sd_z).to(device)
    yy = y.to(device)

    set_seed(seed)
    idx = torch.randperm(len(yy))
    tr = idx[: int(train_fraction * len(yy))]
    model = LatentVerifier(P.shape[1], Z.shape[1], hidden=hidden).to(device)
    pos = float(yy[tr].mean())
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([(1 - pos) / max(pos, 1e-3)], device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        perm = tr[torch.randperm(len(tr))]
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            loss = loss_fn(model(xp[b], xz[b]), yy[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return FittedVerifier(model, mu_p, sd_p, mu_z, sd_z)


def n_parameters(d: int = 4608, n_scalars: int = 5, hidden: int = 256) -> int:
    return sum(p.numel() for p in LatentVerifier(d, n_scalars, hidden).parameters())
