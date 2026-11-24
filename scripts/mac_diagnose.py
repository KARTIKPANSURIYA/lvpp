"""Find which component crashes Python on a Mac. Each test runs in its own
process, so one crash does not hide the others.

    python scripts/mac_diagnose.py
"""
import subprocess
import sys

TESTS = {
    "torch + MPS tensor math": """
import torch
from lvpp.device import DEVICE
x = torch.randn(4000, 4613, device=DEVICE); print(DEVICE, float((x @ x.T).mean()))
""",
    "verifier training on MPS (paper size)": """
import torch
from lvpp.verifier import fit_verifier
P = torch.randn(59784, 4608); Z = torch.randn(59784, 5); y = (torch.rand(59784) < .77).float()
v = fit_verifier(P, Z, y, seed=0, epochs=1); print('scores', v.score(P[:10], Z[:10])[:3])
""",
    "xgboost alone": """
import numpy as np, xgboost as xgb
X = np.random.rand(5000, 200).astype('float32'); y = (X[:, 0] > .5).astype(int)
xgb.XGBClassifier(n_estimators=50, tree_method='hist').fit(X, y); print('xgboost', xgb.__version__, 'ok')
""",
    "xgboost via lvpp (separate process)": """
import torch
from lvpp.baselines import gbt_scores
P = torch.randn(3000, 64); Z = torch.randn(3000, 5); y = (P[:, 0] > 0).float()
print('gbt ok', gbt_scores(P, Z, y, P[:5], Z[:5])[:3])
""",
    "loading the real caches": """
import glob, os, torch
from lvpp.caches import load_pool_A, load_dict_B
root = os.environ.get('LVPP_CACHE') or glob.glob(os.path.expanduser('~/Library/CloudStorage/GoogleDrive-*/My Drive/lcldd_runs'))[0]
P, Z, y, _ = load_pool_A(f'{root}/verifier_upgrade/pool.pt'); te = load_dict_B(f'{root}/verifier_upgrade/testfeat.pt')
print('pool', tuple(P.shape), 'test', te.nq)
""",
}

for name, code in TESTS.items():
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    status = "OK" if r.returncode == 0 else f"FAILED (exit {r.returncode})"
    print(f"[{status}] {name}")
    out = (r.stdout + r.stderr).strip().splitlines()
    for line in out[-4:]:
        print("     ", line[:200])
