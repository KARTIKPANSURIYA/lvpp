"""Round-trip tests for the three cache layouts."""
import numpy as np
import torch

from lvpp.caches import load_dict_B, load_pool_A, load_prefix_C
from lvpp.capture import VARIANTS


def _vec(d=6):
    return torch.randn(d).half()


def test_layout_A(tmp_path):
    n = 16
    pool = {k: [_vec() for _ in range(n)] for k in VARIANTS}
    pool["scalars"] = [torch.tensor([0.1, 0.0, 0.5, 0.4, -0.3]) for _ in range(n)]
    pool["labels"] = [float(i % 2) for i in range(n)]
    pool["qid"] = [i // 8 for i in range(n)]
    pool["n_done"] = 2
    torch.save(pool, tmp_path / "pool.pt")
    P, Z, y, qid = load_pool_A(str(tmp_path / "pool.pt"))
    assert P.shape == (16, 18) and Z.shape == (16, 5) and y.sum() == 8
    P1, *_ = load_pool_A(str(tmp_path / "pool.pt"), n_questions=1)
    assert P1.shape[0] == 8


def test_layout_B(tmp_path):
    tf = {}
    for q in [3, 1]:
        tf[q] = dict(gold="7", cands=[{**{k: _vec() for k in VARIANTS},
                                        "scalar": torch.tensor([0.1, 0.0, 0.5, 0.4, -0.2]),
                                        "answer": "7" if j < 3 else "6"} for j in range(8)])
    torch.save(tf, tmp_path / "t.pt")
    cs = load_dict_B(str(tmp_path / "t.pt"))
    assert cs.nq == 2 and cs.extra["qids"] == [1, 3]
    assert cs.P.shape == (16, 18) and cs.lpc == 4 and int(cs.y.sum()) == 6


def test_layout_C(tmp_path):
    (tmp_path / "test").mkdir()
    recs = []
    for _ in range(3):
        recs.append(dict(gold="5", answers=["5", "4", "5", "5", "1", "5", "2", "5"],
                         n_total=[90, 120, 200, 150, 111, 80, 256, 130],
                         scalar=torch.tensor([[0.1, 0.0, 0.5, 0.4, -0.2]] * 8),
                         feats={110: torch.randn(8, 9).half()},
                         finished_before={110: [t <= 110 for t in [90, 120, 200, 150, 111, 80, 256, 130]]},
                         prefix_lp={110: [-0.1] * 8}))
    torch.save(recs, tmp_path / "test" / "shard_0000.pt")
    cs = load_prefix_C(str(tmp_path), "test", 110)
    assert cs.nq == 3 and cs.P.shape == (24, 9)
    assert cs.extra["finished"][:8].tolist() == [True, False, False, False, False, True, False, False]
    assert np.allclose(cs.extra["prefix_lp"], -0.1)
