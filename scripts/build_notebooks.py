"""Generate the reproduction notebooks in ``notebooks/``.

Run ``python scripts/build_notebooks.py`` after editing a cell here; the
notebooks are build artifacts so that every notebook shares one setup cell and
one set of cache paths.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = r'''#@title Setup: mount Drive, install lvpp, set cache paths
import os, sys, json, subprocess
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    subprocess.run([sys.executable, "-m", "pip", "-q", "install",
                    "lvpp[{extras}] @ git+https://github.com/KARTIKPANSURIYA/lvpp.git"], check=True)

# Root of the cached candidates/features. The default is the Drive layout the
# paper's runs wrote to; set LVPP_CACHE to use another location.
CACHE = os.environ.get("LVPP_CACHE", "/content/drive/MyDrive/lcldd_runs")
RESULTS = os.environ.get("LVPP_RESULTS", os.path.join(CACHE, "lvpp_results"))
os.makedirs(RESULTS, exist_ok=True)
PATHS = dict(
    main_pool   = f"{{CACHE}}/verifier_upgrade/pool.pt",       # 7,473 GSM8K train questions x 8
    main_test   = f"{{CACHE}}/verifier_upgrade/testfeat.pt",   # 1,319 GSM8K test questions x 8
    prefix_root = f"{{CACHE}}/abort_main_draw",                # prefix draw: pool/ and test/ shards
    span_root   = f"{{CACHE}}/span_fixed_vllm",                # Fig. 2 draw (1.5B-Instruct, corrected trigger)
    mi_pool     = f"{{CACHE}}/pending/mi_pool.pt",             # Math-Instruct on GSM8K
    mi_test     = f"{{CACHE}}/pending/mi_test.pt",
    math_pool   = f"{{CACHE}}/pending/math_pool.pt",           # Math-Instruct on MATH
    math_test   = f"{{CACHE}}/pending/math_test.pt",
    x15_pool    = f"{{CACHE}}/novelty/x15_pool.pt",            # 1.5B-Instruct, 1,200 questions
    x3_pool     = f"{{CACHE}}/novelty/x3_pool.pt",             # 3B-Instruct, 1,200 questions
    x3_test     = f"{{CACHE}}/novelty/x3_test.pt",             # 3B-Instruct, 400 test questions
    legacy_fewshot = f"{{CACHE}}/verifier_fewshot",            # Table IV, single-layer verifier
    legacy_mathmax = f"{{CACHE}}/verifier_mathmax",
    distilled_r32  = f"{{CACHE}}/verifier_final",
)
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
print("device:", "cuda" if torch.cuda.is_available() else "cpu", "| cache:", CACHE)
'''

CHECK = r'''def report(rows, expected, keys, tol=0.6):
    """Print measured vs published values. Regenerated candidates are a new
    random draw, so agreement within about half a point is expected; from the
    released caches the numbers should match to the reported precision."""
    out = []
    for k in keys:
        m, e = rows.get(k), expected.get(k)
        if m is None or e is None:
            flag = ""
        elif e < 1e-3:                       # p-values: significant at the same level
            flag = "ok" if m < 1e-3 else "CHECK"
        elif e <= 1.0:                       # AUCs and rates
            flag = "ok" if abs(m - e) <= 0.01 else "CHECK"
        else:                                # accuracies in percentage points
            flag = "ok" if abs(m - e) <= tol else "CHECK"
        out.append(dict(quantity=k, measured=None if m is None else round(m, 4), paper=e, status=flag))
    df = pd.DataFrame(out)
    print(df.to_string(index=False))
    return df
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def notebook(name, cells, extras="gbt"):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}}
    nb.cells = [cells[0], code(SETUP.format(extras=extras)), code(CHECK)] + cells[1:]
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NB_DIR / name)
    print("wrote", name)


# =========================================================================== #
NB00 = [
md(r"""
# 00 · Capture the main draw (GPU, about 7 h on an A100)

Generates the candidates and hidden-state features behind **Tables I and II**
(selection and reading-depth ablation).

* Generator: frozen `Qwen/Qwen2.5-1.5B` (base), four worked exemplars,
  temperature 0.7, top-p 0.95, 256 new tokens, **K = 8**.
* Verifier pool: **all 7,473 GSM8K training questions** (59,784 candidates).
* Test: all 1,319 GSM8K test questions (10,552 candidates).
* Features per candidate: mean-pooled hidden states at layers 14, 21, 28
  (plus last-token and answer-span variants) and the five scalars z(c).

Both captures are resumable: rerun the cell after a disconnect. Sampling on GPU
is not bit-reproducible, so a fresh capture is a new random draw; to reproduce
the exact published numbers, use the released caches and skip to notebook 02.
"""),
code(r'''
from lvpp.capture import FewShotCapture, FewShotConfig
from lvpp.data import gsm8k
os.makedirs(os.path.dirname(PATHS["main_pool"]), exist_ok=True)
cap = FewShotCapture(FewShotConfig())
train, test = gsm8k("train"), gsm8k("test")
print(len(train), "train |", len(test), "test")
'''),
code(r'''
pool = cap.build_pool(train, PATHS["main_pool"], n_pool=7473)
print("pool:", pool["n_done"], "questions,", len(pool["labels"]), "candidates, pos-rate",
      round(float(np.mean(pool["labels"])), 3), "(paper run: 7473 q, 59784 cands, 0.768)")
'''),
code(r'''
tf = cap.build_test(test, PATHS["main_test"])
print("test questions:", len(tf))
'''),
]

# =========================================================================== #
NB01 = [
md(r"""
# 01 · Capture the prefix draw for pruning (GPU, about 6 h on an A100)

Candidates for **Table II (pruning rows) and Table III**. Each question is
sampled with a fixed seed (`offset * 100003 + question_index`) and features are
pooled over the first `min(B, T)` generated tokens at every checkpoint
**B in {110, 126, 144, 169}**. Also stored: prefix log-probabilities (for the
token-confidence pruning baseline), full lengths `T`, and which candidates
finished before `B`.

* Test: all 1,319 GSM8K test questions (seed offset 2).
* Prefix-verifier pool: **1,864 GSM8K training questions** in a fixed shuffle
  (seed 1234), seed offset 1. This is a separate pool from the 7,473-question
  pool of notebook 00.
"""),
code(r'''
from lvpp.capture import PrefixCapture, PrefixConfig
from lvpp.data import gsm8k_raw
cfg = PrefixConfig()
cap = PrefixCapture(cfg)
print("stop_strings supported:", cap.stop)
'''),
code(r'''
cap.capture(gsm8k_raw("test"), f'{PATHS["prefix_root"]}/test', 1319, cfg.seed_offset_test)
cap.capture(gsm8k_raw("train"), f'{PATHS["prefix_root"]}/pool', cfg.n_pool, cfg.seed_offset_pool)
'''),
]

# =========================================================================== #
NB02 = [
md(r"""
# 02 · Table I: selection strategies (cache only, about 10 min on a GPU)

Every strategy scores the **identical** K = 8 candidates of the main draw
through the same vote, so differences come from the per-candidate score alone.

| row | how it is computed |
|---|---|
| Majority (arbitrary / log-prob tie-break) | `selection_hits(..., "majority" / "majority_lp")` |
| Token-confidence weighted voting | weights `exp(mean token log-prob)` |
| LiLaVe-style GBT | XGBoost on the same 4,608 + 5 features, 10 seeds |
| **LVPP** | 1.18M-parameter verifier, 10 training seeds |

Also produced here: the significance tests quoted in Section V-A, the
single-best-candidate comparison, and the **stand-alone classifier
accuracies** on the verifier pool. (The three additional candidate draws
quoted in Section V-A, +3.16 / +4.27 / +3.41 points, are separate captures:
rerun notebook 00 with a new seed and point `PATHS` at it.)
"""),
code(r'''
from lvpp.caches import load_pool_A, load_dict_B
from lvpp.capture import MULTI
from lvpp.selection import selection_hits
from lvpp.verifier import fit_verifier, find_logprob_col
from lvpp.baselines import token_confidence, gbt_scores
from lvpp.stats import paired, mean_sd

P, Z, y, _ = load_pool_A(PATHS["main_pool"])
te = load_dict_B(PATHS["main_test"])
print(f"pool {len(y)} candidates (pos-rate {y.mean():.3f}) | test {te.nq} questions, {len(te.y)} candidates")
h_maj = selection_hits(te, "majority"); h_lp = selection_hits(te, "majority_lp")
conf = token_confidence(te, "exp"); h_conf = selection_hits(te, "weighted", scores=conf)
'''),
code(r'''
SEEDS = list(range(10))
ver_acc, ver_auc, gbt_acc, gbt_auc, H, p_vs_maj = [], [], [], [], [], []
for s in SEEDS:
    v = fit_verifier(P, Z, y, seed=s)
    sc = v.score(te.P, te.Z)
    h = selection_hits(te, "weighted", scores=sc); H.append(h)
    ver_acc.append(100 * h.mean()); ver_auc.append(roc_auc_score(te.y.numpy(), sc))
    p_vs_maj.append(paired(h, h_maj)["p"])
    try:
        sg = gbt_scores(P, Z, y, te.P, te.Z, seed=s)
        gbt_acc.append(100 * selection_hits(te, "weighted", scores=sg).mean())
        gbt_auc.append(roc_auc_score(te.y.numpy(), sg))
    except Exception as e:  # XGBoost missing or crashing: report LVPP anyway
        print(f"   GBT skipped for seed {s}: {type(e).__name__}: {str(e)[:120]}")
        gbt_acc.append(np.nan); gbt_auc.append(np.nan)
    print(f"seed {s}: LVPP {ver_acc[-1]:.2f} (AUC {ver_auc[-1]:.4f}) | GBT {gbt_acc[-1]:.2f} (AUC {gbt_auc[-1]:.4f})")
'''),
code(r'''
m, sd_pop = mean_sd(ver_acc); _, sd_samp = mean_sd(ver_acc, ddof=1)
rows = {
    "majority (arbitrary)": 100 * h_maj.mean(),
    "majority (log-prob tie-break)": 100 * h_lp.mean(),
    "token-confidence weighted": 100 * h_conf.mean(),
    "token-confidence AUC": roc_auc_score(te.y.numpy(), conf),
    "GBT accuracy (10-seed mean)": None if np.all(np.isnan(gbt_acc)) else float(np.nanmean(gbt_acc)),
    "GBT AUC": None if np.all(np.isnan(gbt_auc)) else float(np.nanmean(gbt_auc)),
    "LVPP accuracy (10-seed mean)": m,
    "LVPP AUC": float(np.mean(ver_auc)),
    "LVPP gain over majority": m - 100 * h_maj.mean(),
    "worst-seed McNemar p vs majority": max(p_vs_maj),
}
EXPECTED = {"majority (arbitrary)": 74.91, "majority (log-prob tie-break)": 74.45,
            "token-confidence weighted": 74.45, "token-confidence AUC": 0.648,
            "GBT accuracy (10-seed mean)": 76.57, "GBT AUC": 0.874,
            "LVPP accuracy (10-seed mean)": 78.84, "LVPP AUC": 0.869,
            "LVPP gain over majority": 3.93, "worst-seed McNemar p vs majority": 1.3e-6}
df = report(rows, EXPECTED, list(EXPECTED))
print(f"\nLVPP sd over seeds: population {sd_pop:.2f}, sample {sd_samp:.2f}")
df.to_csv(f"{RESULTS}/table1.csv", index=False)
'''),
code(r'''
#@title Other confidence variants and the best-confidence comparison (Section V-A)
best = None
for var in ["exp", "softmax", "zscore"]:
    w = token_confidence(te, var); h = selection_hits(te, "weighted", scores=w)
    print(f"confidence ({var:7s}) {100*h.mean():.2f}%")
    if best is None or h.mean() > best[1].mean(): best = (var, h)
r = paired(H[0], best[1])
print(f"\nLVPP seed 0 vs best confidence ({best[0]}): {r['diff']:+.2f} points, "
      f"95% CI [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}]  (paper: +3.79, [+2.27, +5.31])")
'''),
code(r'''
#@title Single highest-scored candidate vs weighted voting
v0 = fit_verifier(P, Z, y, seed=0); sc0 = v0.score(te.P, te.Z)
h_w = selection_hits(te, "weighted", scores=sc0); h_s = selection_hits(te, "single_best", scores=sc0)
print(f"weighted vote {100*h_w.mean():.2f}% | single best {100*h_s.mean():.2f}% "
      f"(paper: single best is 1.3-4.3 points lower across generators)")
'''),
code(r'''
#@title Stand-alone correctness classification on the verifier pool (Section V-A text)
from lvpp.probe import probe_accuracy, majority_class_rate
lp_col = find_logprob_col(Z.numpy())
acc_h, _ = probe_accuracy(P, y)
acc_lp, _ = probe_accuracy(Z[:, [lp_col]], y)
acc_hlp, _ = probe_accuracy(torch.cat([P, Z[:, [lp_col]]], -1), y)
print(f"hidden states (3 layers)  {100*acc_h:.1f}%")
print(f"mean token log-prob only  {100*acc_lp:.1f}%")
print(f"hidden + log-prob         {100*acc_hlp:.1f}%")
print(f"majority-class rate       {100*majority_class_rate(y):.1f}%")
print("original run on this pool: 87.3 / 64.5 / 87.2 / 76.8")
'''),
]

# =========================================================================== #
NB03 = [
md(r"""
# 03 · Table II: component ablation (cache only)

Rows 1-5 use the main draw (notebook 00); the pruning rows use the prefix draw
(notebook 01), whose unpruned majority is 73.92% and unpruned verifier 77.03%.
Accuracy and AUC are averaged over verifier-training seeds 0, 1, 2.
"""),
code(r'''
from lvpp.caches import load_pool_A, load_dict_B, load_prefix_C
from lvpp.selection import selection_hits
from lvpp.verifier import fit_verifier
from lvpp.pruning import prune_keep_and_cost, full_cost
from lvpp.stats import paired
SEEDS = [0, 1, 2]
te_full = load_dict_B(PATHS["main_test"], variants=("mid_mean", "upper_mean", "last_mean"))
D = te_full.P.shape[1] // 3
P_all, Z, y, _ = load_pool_A(PATHS["main_pool"], variants=("mid_mean", "upper_mean", "last_mean"))
h_maj = selection_hits(te_full, "majority")
'''),
code(r'''
ARMS = {"mid layer only (L/2)": (slice(0, D), True),
        "last layer only (L)": (slice(2 * D, 3 * D), True),
        "multi-layer, without z(c)": (slice(0, 3 * D), False),
        "multi-layer with z(c)": (slice(0, 3 * D), True)}
rows = {"majority (no verifier)": 100 * h_maj.mean()}
for name, (sl, use_z) in ARMS.items():
    accs, aucs = [], []
    Ztr = Z if use_z else Z[:, :0]; Zte = te_full.Z if use_z else te_full.Z[:, :0]
    for s in SEEDS:
        v = fit_verifier(P_all[:, sl], Ztr, y, seed=s); sc = v.score(te_full.P[:, sl], Zte)
        accs.append(100 * selection_hits(te_full, "weighted", scores=sc).mean())
        aucs.append(roc_auc_score(te_full.y.numpy(), sc))
    rows[name] = float(np.mean(accs)); rows[name + " AUC"] = float(np.mean(aucs))
    print(f"{name:28s} {np.mean(accs):.2f} +/- {np.std(accs):.2f} | AUC {np.mean(aucs):.4f}")
'''),
code(r'''
#@title Final generated token only (Section V-B text: +0.68 points, p = 0.79)
P_last, Zl, yl, _ = load_pool_A(PATHS["main_pool"], variants=("last_final",))
te_last = load_dict_B(PATHS["main_test"], variants=("last_final",))
v = fit_verifier(P_last, Zl, yl, seed=0)
h_last = selection_hits(te_last, "weighted", scores=v.score(te_last.P, te_last.Z))
r = paired(h_last, h_maj)
print(f"last token only: {r['acc']:.2f}% ({r['diff']:+.2f} vs majority, McNemar p = {r['p']:.2f})")
'''),
code(r'''
#@title Pruning rows: prefix AUC and pruned selection (3 seeds)
PR = PATHS["prefix_root"]; K = 8; Bmax = 169
teF = load_prefix_C(PR, "test", Bmax); trF = load_prefix_C(PR, "pool", Bmax)
sF = fit_verifier(trF.P, trF.Z, trF.y, seed=0).score(teF.P, teF.Z)
h_maj_p = selection_hits(teF, "majority"); h_ver_p = selection_hits(teF, "weighted", scores=sF)
c_full = full_cost(teF.extra["n_total"])
print(f"prefix draw: majority {100*h_maj_p.mean():.2f}% | unpruned verifier {100*h_ver_p.mean():.2f}% | {c_full:.0f} tokens/question")
cache = {}
for B, rho in [(169, 0.25), (110, 0.25), (110, 0.50), (126, 0.50)]:
    if B not in cache: cache[B] = (load_prefix_C(PR, "test", B), load_prefix_C(PR, "pool", B))
    teB, trB = cache[B]; accs, aucs = [], []
    for s in SEEDS:
        sB = fit_verifier(trB.P, trB.Z, trB.y, seed=s).score(teB.P, teB.Z)
        aucs.append(roc_auc_score(teB.y.numpy(), sB))
        keep, cost = prune_keep_and_cost(sB, teB.extra["finished"], teF.extra["n_total"], B, rho)
        accs.append(100 * selection_hits(teF, "weighted", scores=sF, keep=keep).mean())
    rows[f"prune B={B} rho={rho}"] = float(np.mean(accs))
    rows[f"prune B={B} AUC"] = float(np.mean(aucs))
    rows[f"prune B={B} rho={rho} saved"] = 100 * (1 - cost / c_full)
    print(f"B={B} rho={rho:.2f}: {np.mean(accs):.2f} +/- {np.std(accs):.2f} | prefix AUC {np.mean(aucs):.4f} | saved {100*(1-cost/c_full):.1f}%")
'''),
code(r'''
EXPECTED = {"majority (no verifier)": 74.91,
            "mid layer only (L/2)": 76.75, "mid layer only (L/2) AUC": 0.842,
            "last layer only (L)": 77.79, "last layer only (L) AUC": 0.844,
            "multi-layer, without z(c)": 78.90, "multi-layer, without z(c) AUC": 0.868,
            "multi-layer with z(c)": 78.92, "multi-layer with z(c) AUC": 0.870,
            "prune B=169 rho=0.25": 77.13, "prune B=169 AUC": 0.843, "prune B=169 rho=0.25 saved": 3.8,
            "prune B=110 rho=0.25": 76.72, "prune B=110 AUC": 0.827, "prune B=110 rho=0.25 saved": 8.0,
            "prune B=110 rho=0.5": 76.55, "prune B=110 rho=0.5 saved": 16.3}
report(rows, EXPECTED, list(EXPECTED)).to_csv(f"{RESULTS}/table2.csv", index=False)
'''),
]

# =========================================================================== #
NB04 = [
md(r"""
# 04 · Table III: which candidates to prune (cache only)

All rules terminate the **same number** of still-running candidates at the
**same checkpoint B = 110** and differ only in *which* ones; survivors are then
selected by the full-solution verifier. Finished candidates are never pruned.

Note on the published table: the *random* and *oracle* rows were originally
measured at B = 126 on the same candidates. This notebook computes **every**
row at B = 110 so the table is internally consistent; use its output for those
two rows.
"""),
code(r'''
from lvpp.caches import load_prefix_C
from lvpp.selection import selection_hits
from lvpp.verifier import fit_verifier
from lvpp.pruning import prune_keep_and_cost, full_cost, rule_values
from lvpp.stats import paired
PR = PATHS["prefix_root"]; B = 110
teF = load_prefix_C(PR, "test", 169); trF = load_prefix_C(PR, "pool", 169)
sF = fit_verifier(trF.P, trF.Z, trF.y, seed=0).score(teF.P, teF.Z)
h_maj = selection_hits(teF, "majority"); h_ver = selection_hits(teF, "weighted", scores=sF)
c_full = full_cost(teF.extra["n_total"])
teB = load_prefix_C(PR, "test", B); trB = load_prefix_C(PR, "pool", B)
sB = fit_verifier(trB.P, trB.Z, trB.y, seed=0).score(teB.P, teB.Z)
assert teB.extra["prefix_lp"] is not None, "prefix log-probs missing: recapture with notebook 01"
print(f"majority {100*h_maj.mean():.2f}% | unpruned verifier {100*h_ver.mean():.2f}% (paper 73.92 / 77.03)")
print(f"finished before B={B}: {100*teB.extra['finished'].mean():.1f}%")
'''),
code(r'''
def run(values, rho):
    keep, cost = prune_keep_and_cost(values, teB.extra["finished"], teF.extra["n_total"], B, rho)
    h = selection_hits(teF, "weighted", scores=sF, keep=keep)
    return h, 100 * (1 - cost / c_full)

rows = []
def add(rule, rho, h, saved):
    pv = paired(h, h_ver)
    rows.append(dict(rule=rule, rho=rho, acc=round(100 * h.mean(), 2), saved=round(saved, 1),
                     vs_unpruned_verifier=round(pv["diff"], 2), p=pv["p"]))

nt = teF.extra["n_total"]
for rho in [0.5]:
    add("keep the longest prefixes", rho, *run(rule_values("longest", n_total=nt), rho))
    hs, ss = zip(*[run(rule_values("random", n_total=nt, rng=np.random.default_rng(i)), rho) for i in range(20)])
    accs = [100 * h.mean() for h in hs]
    rows.append(dict(rule="random (mean of 20)", rho=rho, acc=round(float(np.mean(accs)), 2),
                     saved=round(float(np.mean(ss)), 1), vs_unpruned_verifier=None, p=None))
    print(f"random: {np.mean(accs):.2f} +/- {np.std(accs):.2f}")
    add("oracle (abort incorrect first)", rho, *run(rule_values("oracle", labels=teB.y.numpy()), rho))
for rho in [0.125, 0.25, 0.375, 0.5]:
    add("token confidence (prefix log-prob)", rho, *run(teB.extra["prefix_lp"], rho))
    add("LVPP (prefix verifier score)", rho, *run(sB, rho))
T3 = pd.DataFrame(rows); print(T3.to_string(index=False))
T3.to_csv(f"{RESULTS}/table3.csv", index=False)
'''),
code(r'''
EXPECTED = {"LVPP 0.125": 77.10, "LVPP 0.25": 76.88, "LVPP 0.375": 76.57, "LVPP 0.5 (seed 0)": 76.27,
            "conf 0.125": 77.10, "conf 0.25": 76.50, "conf 0.375": 75.44, "conf 0.5": 75.06,
            "longest 0.5": 74.30}
got = {f"LVPP {r}": a for r, a in zip([0.125, 0.25, 0.375], T3[T3.rule.str.startswith("LVPP")].acc[:3])}
got["LVPP 0.5 (seed 0)"] = float(T3[(T3.rule.str.startswith("LVPP")) & (T3.rho == 0.5)].acc.iloc[0])
for r in [0.125, 0.25, 0.375, 0.5]:
    got[f"conf {r}"] = float(T3[(T3.rule.str.startswith("token")) & (T3.rho == r)].acc.iloc[0])
got["longest 0.5"] = float(T3[T3.rule.str.startswith("keep")].acc.iloc[0])
report(got, EXPECTED, list(EXPECTED))
print("\nThe paper's rho = 0.5 LVPP row (76.55, 16.3%) is the 3-seed mean from notebook 03.")
'''),
]

# =========================================================================== #
NB05 = [
md(r"""
# 05 · Fig. 2 and Section V-C: is the signal final-answer leakage?

Two tests.

* **Span ablation (Fig. 2).** Verifiers are trained on hidden states pooled over
  three views of each candidate: the complete solution, the reasoning with
  everything from the final-answer statement onward removed, and the answer span
  alone. Draw: `Qwen/Qwen2.5-1.5B-Instruct`, chat prompt, T = 0.7, top-p 0.8,
  512 new tokens, K = 8, sampled with vLLM
  (`LVPP_span_recapture_colab.ipynb` regenerates it, about 1.5 h on one GPU).
  The split uses `lvpp.prompts.CHAT_TRIGGER`.
* **Unfinished candidates.** The prefix verifier is evaluated only on candidates
  that are still generating at checkpoint B, so their features cannot contain a
  completed answer statement.

An earlier version of this draw used a trigger regex that never matched
`\boxed{` (`CHAT_TRIGGER_LEGACY`), so for 90.2% of candidates the "reasoning"
view equalled the full solution. The paper reports the corrected draw.
"""),
code(r'''
#@title Step 1 - trigger rate (seconds)
from lvpp.caches import load_dict_B
te_full = load_dict_B(f'{PATHS["span_root"]}/testfeat.pt', variants=("full",))
te_reas = load_dict_B(f'{PATHS["span_root"]}/testfeat.pt', variants=("reasoning",))
trig = te_full.extra["has_trigger"]
same = float((te_reas.P == te_full.P).all(1).float().mean())
print(f"test candidates with a final-answer statement: {100*np.mean(trig):.1f}%")
print(f"test candidates whose reasoning view equals the full solution: {100*same:.1f}%")
'''),
code(r'''
#@title Step 2 - span ablation, three seeds (about 5 min)
from lvpp.selection import selection_hits
from lvpp.verifier import fit_verifier
from lvpp.stats import paired, clustered_bootstrap_auc_diff
SEEDS = [0, 1, 2]; rows = {}; S = {}; H = {}
for span in ["full", "reasoning", "answer_span"]:
    tr = load_dict_B(f'{PATHS["span_root"]}/pool.pt', variants=(span,))
    te = load_dict_B(f'{PATHS["span_root"]}/testfeat.pt', variants=(span,))
    ok = torch.isfinite(tr.P).all(1) & torch.isfinite(tr.Z).all(1)
    accs, aucs, ss, hs = [], [], [], []
    for s in SEEDS:
        sc = fit_verifier(tr.P[ok], tr.Z[ok], tr.y[ok], seed=s).score(torch.nan_to_num(te.P), torch.nan_to_num(te.Z))
        h = selection_hits(te, "weighted", scores=sc)
        accs.append(100 * h.mean()); aucs.append(roc_auc_score(te.y.numpy(), sc)); ss.append(sc); hs.append(h)
    S[span] = np.mean(ss, 0); H[span] = hs
    rows[span] = float(np.mean(accs)); rows[span + " sd"] = float(np.std(accs)); rows[span + " AUC"] = float(np.mean(aucs))
    print(f"{span:12s} {np.mean(accs):.2f} +/- {np.std(accs):.2f} | AUC {np.mean(aucs):.4f}")
rows["majority"] = 100 * selection_hits(te_full, "majority").mean()
for a in ["reasoning", "answer_span"]:
    ps = [paired(H[a][i], H["full"][i])["p"] for i in range(len(SEEDS))]
    lo, hi = clustered_bootstrap_auc_diff(te_full.y.numpy(), S[a], S["full"])
    print(f"{a} - full: accuracy {rows[a] - rows['full']:+.2f} (McNemar p per seed {np.round(ps, 3)}) | "
          f"AUC {rows[a + ' AUC'] - rows['full AUC']:+.4f}, 95% CI [{lo:+.4f}, {hi:+.4f}]")
m = trig.astype(bool)
if m.any() and len(set(te_full.y.numpy()[m])) > 1:
    print(f"split candidates only (n={m.sum()}): AUC full {roc_auc_score(te_full.y.numpy()[m], S['full'][m]):.4f}"
          f" | reasoning {roc_auc_score(te_full.y.numpy()[m], S['reasoning'][m]):.4f}")
EXPECTED = {"majority": 76.88, "full": 80.49, "reasoning": 80.36, "answer_span": 79.73,
            "full AUC": 0.8781, "reasoning AUC": 0.8750, "answer_span AUC": 0.8702}
fig2_report = report(rows, EXPECTED, list(EXPECTED), tol=0.3)
'''),
code(r'''
#@title Step 3 - prefix verifier on unfinished candidates only (about 3 min)
from lvpp.caches import load_prefix_C
for B in [110, 169]:
    te = load_prefix_C(PATHS["prefix_root"], "test", B); tr = load_prefix_C(PATHS["prefix_root"], "pool", B)
    fin = np.asarray(te.extra["finished"]).astype(bool); y = te.y.numpy()
    a_all, a_unf = [], []
    for s in [0, 1, 2]:
        sc = fit_verifier(tr.P, tr.Z, tr.y, seed=s).score(te.P, te.Z)
        a_all.append(roc_auc_score(y, sc))
        a_unf.append(roc_auc_score(y[~fin], sc[~fin]) if (~fin).sum() and len(set(y[~fin])) > 1 else np.nan)
    rows[f"unfinished AUC B={B}"] = float(np.nanmean(a_unf))
    print(f"B={B}: finished {100*fin.mean():.1f}% | AUC all {np.mean(a_all):.3f} | "
          f"AUC unfinished only {np.nanmean(a_unf):.3f} (n={int((~fin).sum())})")
EXPECTED.update({"unfinished AUC B=110": 0.812, "unfinished AUC B=169": 0.786})
report(rows, EXPECTED, list(EXPECTED), tol=0.3).to_csv(f"{RESULTS}/fig2.csv", index=False)
'''),
code(r'''
#@title Step 4 - the figure
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8})
labels = ["complete\nsolution", "reasoning only\n(answer removed)", "answer span\nonly"]
keys = ["full", "reasoning", "answer_span"]; cols = ["#2b6cb0", "#2f855a", "#c05621"]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.2))
for i, k in enumerate(keys):
    ax[0].errorbar(i, rows[k], yerr=rows[k + " sd"], fmt="o", color=cols[i], capsize=3, ms=5)
    ax[0].annotate(f"{rows[k]:.2f}", (i, rows[k]), xytext=(8, -3), textcoords="offset points", fontsize=7)
    ax[1].plot(i, rows[k + " AUC"], "o", color=cols[i], ms=5)
    ax[1].annotate(f"{rows[k + ' AUC']:.3f}", (i, rows[k + " AUC"]), xytext=(8, -3), textcoords="offset points", fontsize=7)
ax[0].axhline(rows["majority"], ls="--", c="k", lw=0.8)
ax[0].text(2.45, rows["majority"] + 0.15, f"majority {rows['majority']:.2f}", ha="right", fontsize=7)
ax[0].set_ylim(np.floor(min(rows["majority"], *(rows[k] for k in keys))) - 1, np.ceil(max(rows[k] for k in keys)) + 1)
ax[0].set_ylabel("selection accuracy (%)")
lo_auc = min(rows[k + " AUC"] for k in keys); hi_auc = max(rows[k + " AUC"] for k in keys)
ax[1].set_ylim(np.floor(lo_auc * 100 - 2) / 100, np.ceil(hi_auc * 100 + 1) / 100); ax[1].set_ylabel("candidate AUC")
for a, t in zip(ax, ["(a)", "(b)"]):
    a.set_xticks(range(3)); a.set_xticklabels(labels); a.set_xlim(-0.5, 2.6); a.set_title(t, fontsize=8)
    a.grid(axis="y", lw=0.3, alpha=0.5)
fig.tight_layout(); fig.savefig(f"{RESULTS}/fig_span_ablation.pdf"); plt.show()
'''),
]

# =========================================================================== #
NB06 = [
md(r"""
# 06 · Table IV: other generators

**Configuration note.** The four published rows were produced with an earlier
verifier configuration: a single-layer read (last-layer mean, d = 1,536, about
0.4M parameters) trained on **800-question pools**, not the three-layer
1.18M-parameter verifier of Tables I-III. Part A reproduces those rows from
their caches. Part B recomputes the Math-Instruct row with the three-layer
recipe on a 3,000-question pool for comparison.

| row | cache | generator |
|---|---|---|
| Distilled r16, r32 | `verifier_final` (r32) | Qwen2.5-1.5B + LoRA (Part C trains it) |
| Few-shot CoT | `verifier_fewshot` | Qwen2.5-1.5B, four exemplars |
| Math-Instruct | `verifier_mathmax` | Qwen2.5-Math-1.5B-Instruct, chat prompt |
"""),
code(r'''
#@title Part A - published rows from the single-layer caches (3 seeds)
from lvpp.caches import load_pool_legacy, load_dict_B, dict_B_as_pool
from lvpp.selection import selection_hits
from lvpp.verifier import fit_verifier
rows = {}
for name, key in [("few-shot CoT", "legacy_fewshot"), ("Math-Instruct", "legacy_mathmax"),
                  ("distilled r32", "distilled_r32")]:
    d = PATHS[key]
    if not os.path.exists(f"{d}/pool.pt"):
        print("missing:", d); continue
    P, Z, y = load_pool_legacy(f"{d}/pool.pt")
    te = load_dict_B(f"{d}/testfeat.pt", variants=("pooled",))
    accs = [100 * selection_hits(te, "weighted", scores=fit_verifier(P, Z, y, seed=s).score(te.P, te.Z)).mean()
            for s in [0, 1, 2]]
    rows[name + " majority"] = 100 * selection_hits(te, "majority").mean()
    rows[name + " LVPP"] = float(np.mean(accs))
    print(f"{name:14s} pool {len(y)} cands | majority {rows[name+' majority']:.2f} -> {np.mean(accs):.2f} +/- {np.std(accs):.2f}")
EXPECTED = {"distilled r32 majority": 62.17, "distilled r32 LVPP": 63.25,
            "few-shot CoT majority": 74.53, "few-shot CoT LVPP": 76.45,
            "Math-Instruct majority": 87.95, "Math-Instruct LVPP": 88.12}
report(rows, EXPECTED, list(EXPECTED), tol=0.3).to_csv(f"{RESULTS}/table4.csv", index=False)
print("\nDistilled r16 (K = 6: 63.46 -> 66.34) was a single run whose cache is not part of this release.")
'''),
code(r'''
#@title Part B - Math-Instruct with the three-layer recipe (3,000-question pool)
if os.path.exists(PATHS["mi_test"]):
    P, Z, y = dict_B_as_pool(PATHS["mi_pool"])
    te = load_dict_B(PATHS["mi_test"])
    accs = [100 * selection_hits(te, "weighted", scores=fit_verifier(P, Z, y, seed=s).score(te.P, te.Z)).mean() for s in [0, 1, 2]]
    print(f"Math-Instruct, 3-layer: majority {100*selection_hits(te,'majority').mean():.2f} | "
          f"log-prob tie-break {100*selection_hits(te,'majority_lp').mean():.2f} -> {np.mean(accs):.2f} "
          f"(original run: 86.73 / 86.81 -> 87.69)")
'''),
code(r'''
#@title Part C - OPTIONAL: train the LoRA-distilled generator and capture it (GPU, many hours)
RUN_DISTILL = False
if RUN_DISTILL:
    import subprocess
    subprocess.run([sys.executable, "-m", "lvpp.distill", "--rank", "32", "--out", f"{CACHE}/adapters/distill_r32"], check=True)
    from lvpp.capture import FewShotCapture, FewShotConfig
    from lvpp.data import gsm8k
    cap = FewShotCapture(FewShotConfig(adapter=f"{CACHE}/adapters/distill_r32", zero_shot_prompt=True))
    cap.build_pool(gsm8k("train"), f"{CACHE}/distilled_r32_ml/pool.pt", n_pool=800)
    cap.build_test(gsm8k("test"), f"{CACHE}/distilled_r32_ml/testfeat.pt")
'''),
]

# =========================================================================== #
NB07 = [
md(r"""
# 07 · Generalization: Qwen2.5-3B and MATH-500

* **3B recipe transfer**: retrain the verifier on `Qwen2.5-3B-Instruct`
  features (layers 18, 27, 36), 1,200-question pool, 400 GSM8K test questions.
  Baseline is majority voting with log-prob tie-break (89.75%).
* **Frozen-weight transfer**: map 3B features into the 1.5B-Instruct feature
  space with ridge regression (alpha = 10) and apply a frozen 1.5B verifier.
* **MATH-500**: `Qwen2.5-Math-1.5B-Instruct`, 3,000 MATH training problems as
  the pool, answers compared with `math-verify`.

Set `RUN_CAPTURE = True` to regenerate (GPU, about 1.5 h for 3B and about 10 h
for MATH); otherwise the cached draws are used.
"""),
code(r'''
RUN_CAPTURE = False
if RUN_CAPTURE:
    from lvpp.capture import ChatCapture, ChatConfig
    from lvpp.answers import extract_chat_numeric, extract_boxed
    from lvpp.data import gsm8k, math500, math_train
    os.makedirs(f"{CACHE}/novelty", exist_ok=True); os.makedirs(f"{CACHE}/pending", exist_ok=True)
    ChatCapture(ChatConfig(model="Qwen/Qwen2.5-1.5B-Instruct")).build_cache(
        "1.5B pool", gsm8k("train"), extract_chat_numeric, PATHS["x15_pool"], limit=1200)
    cap3 = ChatCapture(ChatConfig(model="Qwen/Qwen2.5-3B-Instruct"))
    cap3.build_cache("3B pool", gsm8k("train"), extract_chat_numeric, PATHS["x3_pool"], limit=1200)
    cap3.build_cache("3B test", gsm8k("test"), extract_chat_numeric, PATHS["x3_test"], limit=400)
    capm = ChatCapture(ChatConfig(model="Qwen/Qwen2.5-Math-1.5B-Instruct", max_new=1536, qbatch=4))
    capm.build_cache("MATH pool", math_train(), extract_boxed, PATHS["math_pool"], limit=3000)
    capm.build_cache("MATH test", math500(), extract_boxed, PATHS["math_test"])
'''),
code(r'''
#@title 3B: recipe transfer and frozen-weight transfer
from sklearn.linear_model import Ridge
from lvpp.caches import load_dict_B, dict_B_as_pool
from lvpp.selection import selection_hits
from lvpp.verifier import fit_verifier
rows = {}
P3, Z3, y3 = dict_B_as_pool(PATHS["x3_pool"]); t3 = load_dict_B(PATHS["x3_test"])
accs, aucs = [], []
for s in [0, 1, 2]:
    sc = fit_verifier(P3, Z3, y3, seed=s).score(t3.P, t3.Z)
    accs.append(100 * selection_hits(t3, "weighted", scores=sc).mean()); aucs.append(roc_auc_score(t3.y.numpy(), sc))
rows["3B majority (log-prob tie-break)"] = 100 * selection_hits(t3, "majority_lp").mean()
rows["3B retrained verifier"] = float(np.mean(accs)); rows["3B retrained AUC"] = float(np.mean(aucs))
P15, Z15, y15 = dict_B_as_pool(PATHS["x15_pool"])
v15 = fit_verifier(P15, Z15, y15, seed=0)
n = min(len(P15), len(P3))
ridge = Ridge(alpha=10.0).fit(P3[:n].numpy(), P15[:n].numpy())
sc = v15.score(torch.tensor(ridge.predict(t3.P.numpy())).float(), t3.Z)
rows["frozen 1.5B via ridge, accuracy"] = 100 * selection_hits(t3, "weighted", scores=sc).mean()
rows["frozen 1.5B via ridge, AUC"] = roc_auc_score(t3.y.numpy(), sc)
EXPECTED = {"3B majority (log-prob tie-break)": 89.75, "3B retrained verifier": 91.25, "3B retrained AUC": 0.8875,
            "frozen 1.5B via ridge, accuracy": 88.25, "frozen 1.5B via ridge, AUC": 0.732}
report(rows, EXPECTED, list(EXPECTED))
'''),
code(r'''
#@title MATH-500 in-domain (3 seeds, K = 2 and 8)
from lvpp.answers import make_equiv
eq = make_equiv(True)
Pm, Zm, ym = dict_B_as_pool(PATHS["math_pool"], equiv=eq); tm = load_dict_B(PATHS["math_test"], equiv=eq)
scs = [fit_verifier(Pm, Zm, ym, seed=s).score(tm.P, tm.Z) for s in [0, 1, 2]]
for K in [2, 8]:
    maj = 100 * selection_hits(tm, "majority", K=K).mean(); lp = 100 * selection_hits(tm, "majority_lp", K=K).mean()
    ver = np.mean([100 * selection_hits(tm, "weighted", scores=s, K=K).mean() for s in scs])
    print(f"K={K}: majority {maj:.2f} | log-prob tie-break {lp:.2f} | LVPP {ver:.2f}")
print("in-domain AUC:", round(float(np.mean([roc_auc_score(tm.y.numpy(), s) for s in scs])), 4))
print("original run: K=8 log-prob tie-break 78.40 -> 78.33; K=2 gain 0.0 at majority 74.4; AUC 0.892")
'''),
]

if __name__ == "__main__":
    notebook("00_capture_main_draw.ipynb", NB00, extras="gpu")
    notebook("01_capture_prefix_draw.ipynb", NB01, extras="gpu")
    notebook("02_table1_selection.ipynb", NB02)
    notebook("03_table2_ablation.ipynb", NB03)
    notebook("04_table3_pruning_rules.ipynb", NB04)
    notebook("05_fig2_span_ablation.ipynb", NB05, extras="gpu,gbt")
    notebook("06_table4_generators.ipynb", NB06, extras="gpu,gbt")
    notebook("07_transfer_3b_math500.ipynb", NB07, extras="gpu,gbt,math")
