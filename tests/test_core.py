"""CPU unit tests for the logic that determines the paper's numbers."""
import numpy as np
import pytest
import torch
from sklearn.metrics import roc_auc_score

from lvpp.answers import (extract_boxed, extract_chat_numeric, extract_fewshot,
                          extract_fewshot_clipped, gsm8k_gold, norm)
from lvpp.prompts import CHAT_TRIGGER, CHAT_TRIGGER_LEGACY, FEWSHOT, split_solution
from lvpp.pruning import full_cost, prune_keep_and_cost, rule_values
from lvpp.selection import CandidateSet, oracle_hits, select_answer, selection_hits
from lvpp.stats import mcnemar_exact, paired
from lvpp.verifier import fit_verifier, n_parameters


# ---------------------------------------------------------------- answers ---
def test_gold_and_norm():
    assert gsm8k_gold("blah <<48/2=24>>24\n#### 1,072") == "1072"
    assert norm(" $1,072. ") == "1072"


def test_fewshot_extraction():
    assert extract_fewshot(" 48 + 24 = 72. The answer is 72.\n\nQuestion: next 5") == "72"
    assert extract_fewshot(" so 3 + 4 = 7") == "7"
    assert extract_fewshot(" The answer is $1,200.") == "1200"
    # the assertion embedded in the original prefix-capture notebook
    assert extract_fewshot_clipped("Altogether 72. The answer is 72.\n\nQuestion: next\nAnswer: 3.") == "72"


def test_boxed_and_chat():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert extract_chat_numeric(r"Total is 18. \boxed{18}") == "18"


def test_prompt_has_four_exemplars():
    assert FEWSHOT.count("Question:") == 4 and "<<" not in FEWSHOT


def test_legacy_trigger_never_matches_boxed():
    """Documents the span-ablation bug: the original regex misses a real \\boxed{."""
    sol = r"Step 1. 9 + 9 = 18. The final answer is $\boxed{18}$."
    assert CHAT_TRIGGER_LEGACY.search(sol) is None
    assert CHAT_TRIGGER.search(sol) is not None
    _, has = split_solution(sol, CHAT_TRIGGER, cut_next_question=False)
    assert has
    assert CHAT_TRIGGER_LEGACY.search("so it is 18. The answer is 18.") is not None


# -------------------------------------------------------------- selection ---
def _cs(answers, gold, scores=None, lp=None):
    K = len(answers[0])
    n = len(answers) * K
    Z = torch.zeros(n, 5)
    Z[:, 4] = torch.tensor(lp if lp is not None else -np.ones(n), dtype=torch.float32)
    y = torch.tensor([1.0 if a == g else 0.0 for row, g in zip(answers, gold) for a in row])
    return CandidateSet(answers, gold, torch.zeros(n, 3), Z, y, lpc=4, K=K)


def test_majority_first_appearance_tie_break():
    assert select_answer(["5", "7", "7", "5"], "majority") == "5"


def test_majority_logprob_tie_break():
    lp = np.array([-2.0, -0.1, -0.1, -2.0])
    assert select_answer(["5", "7", "7", "5"], "majority_lp", logprobs=lp) == "7"


def test_weighted_vote_overrides_majority():
    s = np.array([0.1, 0.1, 0.1, 0.9])
    assert select_answer(["5", "5", "5", "7"], "weighted", scores=s) == "7"
    assert select_answer(["5", "5", "5", "7"], "majority") == "5"


def test_empty_answers_ignored_and_keep_mask():
    assert select_answer(["", "", "3"], "majority") == "3"
    assert select_answer(["3", "4", "4"], "majority", keep=[True, False, False]) == "3"
    assert select_answer(["", ""], "majority") == ""


def test_selection_hits_and_oracle():
    cs = _cs([["1", "1", "2", "2"], ["9", "8", "8", "8"]], ["2", "9"])
    s = np.array([0.1, 0.1, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1])
    assert selection_hits(cs, "majority").tolist() == [False, False]
    assert selection_hits(cs, "weighted", scores=s).tolist() == [True, True]
    assert oracle_hits(cs).tolist() == [True, True]


# ---------------------------------------------------------------- pruning ---
def test_finished_candidates_are_never_pruned_and_cost():
    K = 4
    vals = np.array([0.0, 0.1, 0.9, 0.8])
    fin = np.array([True, False, False, False])
    T = np.array([50, 200, 200, 300])
    keep, cost = prune_keep_and_cost(vals, fin, T, B=100, rho=0.5, K=K)
    # round(0.5*4)=2 lowest after pushing finished to +inf: idx 1 and 3
    assert keep[0] == [True, False, True, False]
    assert cost == 50 + 100 + 200 + 100
    assert full_cost(T, K) == 750


def test_rho_zero_keeps_everything():
    keep, cost = prune_keep_and_cost(np.arange(8.0), np.zeros(8, bool), np.full(8, 200), 110, 0.0)
    assert all(keep[0]) and cost == 1600


def test_rule_values_orientation():
    T = np.array([10, 20, 30])
    assert np.argmin(rule_values("longest", n_total=T)) == 0  # shortest final length pruned first
    assert np.argmin(rule_values("shortest", n_total=T)) == 2
    y = np.array([1, 0, 1])
    assert np.argmin(rule_values("oracle", labels=y)) == 1  # incorrect pruned first


# ------------------------------------------------------------------ stats ---
def test_mcnemar_exact_values():
    assert mcnemar_exact(0, 0) == 1.0
    assert mcnemar_exact(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact(3, 7) == mcnemar_exact(7, 3)


def test_paired_counts():
    a = np.array([1, 1, 0, 0, 1], bool)
    b = np.array([1, 0, 0, 1, 0], bool)
    r = paired(a, b)
    assert (r["b"], r["c"]) == (1, 2)
    assert r["diff"] == pytest.approx(20.0)


# --------------------------------------------------------------- verifier ---
def test_parameter_count_is_1_18M():
    assert n_parameters(4608, 5, 256) == 1_181_441


def test_verifier_learns_and_is_seed_deterministic():
    g = torch.Generator().manual_seed(0)
    n, d = 2000, 32
    y = (torch.rand(n, generator=g) < 0.7).float()
    P = torch.randn(n, d, generator=g) + 1.5 * y[:, None] * torch.ones(d) / d**0.5
    Z = torch.randn(n, 5, generator=g)
    v1 = fit_verifier(P, Z, y, seed=0, device="cpu")
    v2 = fit_verifier(P, Z, y, seed=0, device="cpu")
    s1, s2 = v1.score(P, Z, "cpu"), v2.score(P, Z, "cpu")
    assert roc_auc_score(y.numpy(), s1) > 0.85
    assert np.allclose(s1, s2)


def test_verifier_without_scalars():
    P = torch.randn(200, 8)
    y = (P[:, 0] > 0).float()
    v = fit_verifier(P, torch.zeros(200, 0), y, seed=1, device="cpu")
    assert v.score(P, torch.zeros(200, 0), "cpu").shape == (200,)
