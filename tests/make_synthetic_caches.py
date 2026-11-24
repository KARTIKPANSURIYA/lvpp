"""Write small synthetic caches in every layout so the notebooks can be smoke-tested on CPU.

    python tests/make_synthetic_caches.py /tmp/lvpp_synth
    LVPP_CACHE=/tmp/lvpp_synth jupyter nbconvert --execute notebooks/02_table1_selection.ipynb
"""
import os
import sys

import numpy as np
import torch

K, d = 8, 8
rng = np.random.default_rng(0)
VAR = ("last_mean", "last_ansspan", "last_final", "mid_mean", "upper_mean")


def cand(correct: bool, d=d):
    shift = 0.8 if correct else -0.8
    v = lambda: torch.tensor(rng.normal(shift * 0.5, 1, d), dtype=torch.float32).half()  # noqa: E731
    z = torch.tensor([rng.random(), rng.random(), rng.random(), rng.random(),
                      -abs(rng.normal(0.3 if correct else 0.5, 0.1)) - 1e-3], dtype=torch.float32)
    return v, z


def question(p_correct=0.6):
    gold = str(rng.integers(1, 99))
    corr = rng.random(K) < p_correct
    wrong = [str(rng.integers(100, 103)) for _ in range(K)]
    return gold, corr, [gold if c else w for c, w in zip(corr, wrong)]


def layout_A(path, nq):
    pool = {k: [] for k in VAR} | {"scalars": [], "labels": [], "qid": [], "n_done": nq}
    for q in range(nq):
        _, corr, _ = question()
        for c in corr:
            v, z = cand(bool(c))
            for k in VAR:
                pool[k].append(v())
            pool["scalars"].append(z); pool["labels"].append(float(c)); pool["qid"].append(q)
    torch.save(pool, path)


def layout_B(path, nq, keys=VAR, spans=False, d_=d):
    out = {}
    for q in range(nq):
        gold, corr, ans = question()
        cands = []
        for c, a in zip(corr, ans):
            v, z = cand(bool(c), d_)
            item = {k: v() for k in keys} | {"scalar": z, "answer": a}
            if spans:
                item["has_trigger"] = bool(rng.random() < 0.3)
            cands.append(item)
        out[q] = dict(gold=gold, cands=cands)
    torch.save(out, path)


def layout_C(root, tag, nq, budgets=(110, 126, 144, 169)):
    os.makedirs(f"{root}/{tag}", exist_ok=True)
    recs = []
    for _ in range(nq):
        gold, corr, ans = question()
        T = rng.integers(60, 256, K)
        feats = {B: torch.stack([cand(bool(c), 3 * d)[0]() for c in corr]) for B in budgets}
        recs.append(dict(gold=gold, answers=ans, n_total=T.tolist(),
                         scalar=torch.stack([cand(bool(c))[1] for c in corr]), feats=feats,
                         finished_before={B: [bool(t <= B) for t in T] for B in budgets},
                         prefix_lp={B: [float(-abs(rng.normal(0.3 if c else 0.5, 0.1))) for c in corr]
                                    for B in budgets}))
    for i in range(0, nq, 25):
        torch.save(recs[i:i + 25], f"{root}/{tag}/shard_{i // 25:04d}.pt")


def legacy(root, nq_pool, nq_test):
    os.makedirs(root, exist_ok=True)
    P, Z, Y = [], [], []
    for _ in range(nq_pool):
        _, corr, _ = question()
        for c in corr:
            v, z = cand(bool(c)); P.append(v().float()); Z.append(z); Y.append(float(c))
    torch.save(dict(pooled=torch.stack(P), scalars=torch.stack(Z), labels=torch.tensor(Y)), f"{root}/pool.pt")
    layout_B(f"{root}/testfeat.pt", nq_test, keys=("pooled",))


def main(root):
    for sub in ["verifier_upgrade", "abort_main_draw", "span_fixed_vllm", "pending", "novelty"]:
        os.makedirs(f"{root}/{sub}", exist_ok=True)
    layout_A(f"{root}/verifier_upgrade/pool.pt", 300)
    layout_B(f"{root}/verifier_upgrade/testfeat.pt", 120)
    layout_C(f"{root}/abort_main_draw", "pool", 150)
    layout_C(f"{root}/abort_main_draw", "test", 100)
    for split, n in [("pool", 200), ("testfeat", 100)]:
        layout_B(f"{root}/span_fixed_vllm/{split}.pt", n, keys=("full", "reasoning", "answer_span"), spans=True, d_=3 * d)
    layout_B(f"{root}/pending/mi_pool.pt", 150); layout_B(f"{root}/pending/mi_test.pt", 80)
    layout_B(f"{root}/pending/math_pool.pt", 150); layout_B(f"{root}/pending/math_test.pt", 80)
    layout_B(f"{root}/novelty/x15_pool.pt", 150)
    layout_B(f"{root}/novelty/x3_pool.pt", 150, d_=12); layout_B(f"{root}/novelty/x3_test.pt", 60, d_=12)
    for name in ["verifier_fewshot", "verifier_mathmax", "verifier_final"]:
        legacy(f"{root}/{name}", 150, 80)
    print("synthetic caches written to", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/lvpp_synth")
