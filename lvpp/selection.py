"""Answer selection over K candidates: majority voting and weighted voting (Eq. 1)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import torch

from .answers import norm


@dataclass
class CandidateSet:
    """Candidates for ``nq`` questions, K per question, flattened question-major.

    Attributes
    ----------
    answers : list of per-question answer lists (length K each)
    gold    : list of gold answers
    P, Z    : pooled features (N x D) and scalars (N x 5), N = nq * K
    y       : correctness labels (N,)
    lpc     : column of Z that holds the mean token log-probability
    """

    answers: list
    gold: list
    P: torch.Tensor
    Z: torch.Tensor
    y: torch.Tensor
    lpc: int = 4
    K: int = 8
    equiv: Callable[[str, str], bool] = field(default=lambda a, b: norm(a) == norm(b))
    extra: dict = field(default_factory=dict)

    @property
    def nq(self) -> int:
        return len(self.gold)

    @property
    def offs(self) -> np.ndarray:
        return np.cumsum([0] + [len(a) for a in self.answers])

    def logprob(self) -> np.ndarray:
        return self.Z[:, self.lpc].numpy().astype(float)


def answer_groups(answers: Sequence[str], keep=None, equiv=None):
    """Group candidate indices by answer, in order of first appearance.

    Empty answers are ignored; ``keep`` masks out pruned candidates.
    """
    eq = equiv or (lambda a, b: norm(a) == norm(b))
    groups: list[list] = []
    for i, a in enumerate(answers):
        if not a:
            continue
        if keep is not None and not keep[i]:
            continue
        for g in groups:
            if eq(a, g[0]):
                g[1].append(i)
                break
        else:
            groups.append([a, [i]])
    return groups


def select_answer(answers, mode="weighted", scores=None, logprobs=None, keep=None, equiv=None):
    """Pick one answer for one question.

    modes
      ``majority``     majority vote, ties broken by first appearance ("arbitrary")
      ``majority_lp``  majority vote, ties broken by highest mean token log-prob
      ``weighted``     Eq. (1): sum of per-candidate scores per answer
      ``single_best``  answer of the single highest-scoring candidate
    """
    gr = answer_groups(answers, keep, equiv)
    if not gr:
        return ""
    if mode in ("majority", "majority_lp"):
        top = max(len(ix) for _, ix in gr)
        winners = [(a, ix) for a, ix in gr if len(ix) == top]
        if mode == "majority" or len(winners) == 1:
            return winners[0][0]
        return max(winners, key=lambda w: np.mean([logprobs[i] for i in w[1]]))[0]
    if mode == "weighted":
        w = {}
        for a, ix in gr:
            w[a] = float(sum(scores[i] for i in ix))
        return max(w, key=w.get)
    if mode == "single_best":
        cand = [i for _, ix in gr for i in ix]
        best = max(cand, key=lambda i: scores[i])
        return answers[best]
    raise ValueError(mode)


def selection_hits(cs: CandidateSet, mode="weighted", scores=None, keep=None, K=None) -> np.ndarray:
    """Boolean vector: did the selected answer equal gold, per question."""
    K = K or cs.K
    offs = cs.offs
    lp = cs.logprob() if mode == "majority_lp" else None
    out = []
    for q in range(cs.nq):
        a0 = offs[q]
        ans = cs.answers[q][:K]
        kp = None if keep is None else list(keep[q][:K])
        s = None if scores is None else np.asarray(scores[a0 : a0 + K])
        l = None if lp is None else lp[a0 : a0 + K]
        pick = select_answer(ans, mode, s, l, kp, cs.equiv)
        out.append(bool(pick) and cs.equiv(pick, cs.gold[q]))
    return np.array(out, dtype=bool)


def oracle_hits(cs: CandidateSet, K=None) -> np.ndarray:
    """Upper bound: a correct candidate exists among the first K."""
    K = K or cs.K
    return np.array(
        [any(a and cs.equiv(a, g) for a in ans[:K]) for ans, g in zip(cs.answers, cs.gold)],
        dtype=bool,
    )
