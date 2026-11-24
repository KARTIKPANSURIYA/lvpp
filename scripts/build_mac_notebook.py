"""Build notebooks/LVPP_local_mac.ipynb: every experiment in one notebook for a
local Apple Silicon machine (VS Code + Jupyter). Reuses the exact analysis
cells of the Colab notebooks so both paths compute the same thing."""
from __future__ import annotations

import textwrap

import nbformat as nbf

import build_notebooks as B

HEADER = r"""
# LVPP: full reproduction on a Mac (Apple Silicon, VS Code)

One notebook for every table and figure in the paper, written for a local Apple
Silicon machine such as a MacBook Pro M5 Max with 128 GB. It uses the Metal (MPS)
GPU for generation and verifier training; XGBoost runs on the CPU.

## One-time setup (Terminal)

```bash
cd ~/Projects            # wherever you unzipped lvpp_repo.zip
cd lvpp
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[gpu,gbt,math]" ipykernel
```

Then in VS Code: install the **Python** and **Jupyter** extensions, open the
`lvpp` folder, open this notebook, and choose **Select Kernel → .venv**.

## Where the data comes from

* **Part A (minutes):** recomputes Tables I–IV, Fig. 2 and Section V-D from the
  cached candidates the paper used. Install **Google Drive for desktop**, and in
  Finder right-click `My Drive/lcldd_runs` → **Make available offline** (about
  4 GB of caches). Cell 1 finds the folder automatically. Alternatively,
  download the caches to `~/lvpp_cache` and set `CACHE` below.
* **Part B (hours, optional):** regenerates candidates on the Mac. Start with
  the quick timing test (16 questions), which prints the projected runtime of
  the full captures on your machine. New captures go to `~/lvpp_cache_new`, so
  the published caches are never overwritten.

Run cells top to bottom. Sections whose caches are missing are skipped, not
failed. The last cell collects every **measured vs paper** comparison.
"""

CONFIG = r'''#@title 1 · Configuration
import os, sys, glob, json, time, subprocess
from pathlib import Path
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # torch + xgboost both ship OpenMP on macOS
os.environ.setdefault("OMP_NUM_THREADS", "8")

# ---- what to run ---------------------------------------------------------
RUN_PART_A_ANALYSIS      = True    # Tables I-IV, Fig. 2, Sec. V-D from caches (minutes)
RUN_QUICK_TIMING_TEST    = False   # 16-question capture on the Mac GPU, projects full runtime
RUN_CAPTURE_MAIN_DRAW    = False   # Tables I-II candidates (7,473 + 1,319 questions)
RUN_CAPTURE_PREFIX_DRAW  = False   # Tables II-III candidates (1,864 + 1,319 questions)
RUN_SPAN_RECAPTURE_FIXED = False   # Fig. 2 draw: see notebooks/LVPP_span_recapture_colab.ipynb
USE_NEW_CAPTURES_FOR_ANALYSIS = False  # point Part A at ~/lvpp_cache_new instead

# ---- where the caches are ------------------------------------------------
def find_drive_cache():
    hits = glob.glob(os.path.expanduser("~/Library/CloudStorage/GoogleDrive-*/My Drive/lcldd_runs"))
    return hits[0] if hits else None
CACHE = os.environ.get("LVPP_CACHE") or find_drive_cache() or os.path.expanduser("~/lvpp_cache")
NEW_CACHE = os.path.expanduser("~/lvpp_cache_new")
if USE_NEW_CAPTURES_FOR_ANALYSIS:
    CACHE = NEW_CACHE
RESULTS = os.environ.get("LVPP_RESULTS", os.path.expanduser("~/lvpp_results"))
os.makedirs(RESULTS, exist_ok=True); os.makedirs(NEW_CACHE, exist_ok=True)

try:
    import lvpp
except ImportError:  # running from the repo without `pip install -e .`
    sys.path.insert(0, str(Path.cwd().parent))
    import lvpp
from lvpp.device import DEVICE, model_dtype
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
print(f"lvpp {lvpp.__version__} | torch {torch.__version__} | device {DEVICE} ({model_dtype(DEVICE)})")
print("cache  :", CACHE)
print("results:", RESULTS)
if DEVICE != "mps":
    print("NOTE: MPS not available - check that you are on Apple Silicon with a recent PyTorch.")
'''

PATHS_CELL = r'''#@title 2 · Cache inventory
PATHS = dict(
    main_pool   = f"{CACHE}/verifier_upgrade/pool.pt",
    main_test   = f"{CACHE}/verifier_upgrade/testfeat.pt",
    prefix_root = f"{CACHE}/abort_main_draw",
    span_root   = f"{CACHE}/span_fixed_vllm",
    mi_pool     = f"{CACHE}/pending/mi_pool.pt",
    mi_test     = f"{CACHE}/pending/mi_test.pt",
    math_pool   = f"{CACHE}/pending/math_pool.pt",
    math_test   = f"{CACHE}/pending/math_test.pt",
    x15_pool    = f"{CACHE}/novelty/x15_pool.pt",
    x3_pool     = f"{CACHE}/novelty/x3_pool.pt",
    x3_test     = f"{CACHE}/novelty/x3_test.pt",
    legacy_fewshot = f"{CACHE}/verifier_fewshot",
    legacy_mathmax = f"{CACHE}/verifier_mathmax",
    distilled_r32  = f"{CACHE}/verifier_final",
)
def have(*keys):
    return all(os.path.exists(PATHS[k]) for k in keys)
rows = []
for k, p in PATHS.items():
    ok = os.path.exists(p)
    size = (os.path.getsize(p) if os.path.isfile(p) else
            sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file())) if ok else 0
    rows.append(dict(cache=k, present=ok, GB=round(size / 1e9, 2)))
print(pd.DataFrame(rows).to_string(index=False))
SECTIONS = {
    "Table I":   have("main_pool", "main_test"),
    "Table II":  have("main_pool", "main_test", "prefix_root"),
    "Table III": have("prefix_root"),
    "Fig. 2":    have("span_root", "prefix_root"),
    "Table IV":  have("legacy_fewshot") or have("legacy_mathmax") or have("distilled_r32"),
    "Sec. V-D":  have("x15_pool", "x3_pool", "x3_test") and have("math_pool", "math_test"),
}
print("\nPart A sections that will run:", [k for k, v in SECTIONS.items() if v])
'''

SUMMARY = r'''#@title Summary · every measured-vs-paper comparison
frames = []
for name in ["table1", "table2", "table3", "fig2", "table4", "sec5d"]:
    f = f"{RESULTS}/{name}.csv"
    if os.path.exists(f):
        d = pd.read_csv(f)
        if "quantity" not in d.columns:      # Table III is a full rule table: show it as is
            print(f"--- {name} ---"); print(d.to_string(index=False)); print()
            continue
        d.insert(0, "item", name); frames.append(d)
if frames:
    ALL = pd.concat(frames, ignore_index=True)
    ALL.to_csv(f"{RESULTS}/all_results.csv", index=False)
    pd.set_option("display.max_rows", 200); pd.set_option("display.width", 200)
    print(ALL.to_string(index=False))
    print(f"\nsaved {RESULTS}/all_results.csv")
else:
    print("No results yet - enable RUN_PART_A_ANALYSIS and check the cache inventory.")
'''

QUICK = r'''#@title B1 · Quick timing test on the Mac GPU (about 5-10 min)
if RUN_QUICK_TIMING_TEST:
    from lvpp.capture import FewShotCapture, FewShotConfig, PrefixCapture, PrefixConfig
    from lvpp.data import gsm8k, gsm8k_raw
    qdir = f"{NEW_CACHE}/quick_test"; os.makedirs(qdir, exist_ok=True)
    cap = FewShotCapture(FewShotConfig())
    train = gsm8k("train")[:16]
    t0 = time.time(); cap.build_pool(train, f"{qdir}/pool.pt", n_pool=16); dt_main = (time.time() - t0) / 16
    del cap; torch.mps.empty_cache() if DEVICE == "mps" else None
    pcap = PrefixCapture(PrefixConfig(shard=8))
    t0 = time.time(); pcap.capture(gsm8k_raw("test"), f"{qdir}/prefix_test", 8, 2); dt_pre = (time.time() - t0) / 8
    del pcap
    pool = torch.load(f"{qdir}/pool.pt", weights_only=False)
    print(f"main draw  : {dt_main:.1f} s/question | pos-rate on 16 questions {np.mean(pool['labels']):.2f} (full pool: 0.77)")
    print(f"prefix draw: {dt_pre:.1f} s/question")
    print(f"projected  : main draw {dt_main * (7473 + 1319) / 3600:.1f} h | prefix draw {dt_pre * (1864 + 1319) / 3600:.1f} h")
'''

CAPTURE_MAIN = r'''#@title B2 · Capture the main draw on the Mac (resumable; see projected time in B1)
if RUN_CAPTURE_MAIN_DRAW:
    subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])  # keep the Mac awake
    from lvpp.capture import FewShotCapture, FewShotConfig
    from lvpp.data import gsm8k
    out = f"{NEW_CACHE}/verifier_upgrade"; os.makedirs(out, exist_ok=True)
    cap = FewShotCapture(FewShotConfig())
    cap.build_pool(gsm8k("train"), f"{out}/pool.pt", n_pool=7473)
    cap.build_test(gsm8k("test"), f"{out}/testfeat.pt")
    del cap
'''

CAPTURE_PREFIX = r'''#@title B3 · Capture the prefix draw on the Mac (resumable)
if RUN_CAPTURE_PREFIX_DRAW:
    subprocess.Popen(["caffeinate", "-dimsu", "-w", str(os.getpid())])
    from lvpp.capture import PrefixCapture, PrefixConfig
    from lvpp.data import gsm8k_raw
    cfg = PrefixConfig(); cap = PrefixCapture(cfg)
    out = f"{NEW_CACHE}/abort_main_draw"
    cap.capture(gsm8k_raw("test"), f"{out}/test", 1319, cfg.seed_offset_test)
    cap.capture(gsm8k_raw("train"), f"{out}/pool", cfg.n_pool, cfg.seed_offset_pool)
    del cap
'''

CAPTURE_SPAN = r'''#@title B4 · Fig. 2 draw (regenerate on a GPU, not on the Mac)
# The Fig. 2 draw samples 8 x 512-token solutions for 8,792 questions. On a Mac this
# takes days; use notebooks/LVPP_span_recapture_colab.ipynb (vLLM, about 1.5 h on one
# GPU). It writes to lcldd_runs/span_fixed_vllm, which the Fig. 2 section reads.
if RUN_SPAN_RECAPTURE_FIXED:
    print("Open notebooks/LVPP_span_recapture_colab.ipynb in Colab to regenerate the Fig. 2 draw.")
'''


def guarded(src: str, section: str) -> str:
    body = textwrap.indent(src.strip(), "    ")
    return f'if RUN_PART_A_ANALYSIS and SECTIONS["{section}"]:\n{body}\nelse:\n    print("skipped: {section}")'


def section_cells(cells, section, csv_name=None, drop_capture=False):
    out = []
    for c in cells:
        if c.cell_type == "markdown":
            out.append(nbf.v4.new_markdown_cell(c.source))
            continue
        src = c.source
        if drop_capture and ("RUN_CAPTURE" in src or "RUN_RECAPTURE" in src or "RUN_DISTILL" in src):
            continue
        if csv_name:  # unify CSV names for the summary
            src = src.replace('f"{RESULTS}/fig2.csv"', f'f"{{RESULTS}}/{csv_name}.csv"')
        out.append(nbf.v4.new_code_cell(guarded(src, section)))
    return out


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}
    cells = [nbf.v4.new_markdown_cell(HEADER.strip()),
             nbf.v4.new_code_cell(CONFIG.strip()),
             nbf.v4.new_code_cell(PATHS_CELL.strip()),
             nbf.v4.new_code_cell(B.CHECK.strip()),
             nbf.v4.new_markdown_cell("# Part A · Reproduce every table from the caches")]
    cells += section_cells(B.NB02, "Table I")
    cells += section_cells(B.NB03, "Table II")
    cells += section_cells(B.NB04, "Table III")
    cells += section_cells(B.NB05, "Fig. 2", drop_capture=True)
    cells += section_cells(B.NB06, "Table IV", drop_capture=True)
    nb07 = section_cells(B.NB07, "Sec. V-D", drop_capture=True)
    # Sec. V-D: also save a CSV for the summary
    for c in nb07:
        if c.cell_type == "code" and "frozen 1.5B via ridge, AUC" in c.source and "EXPECTED" in c.source:
            c.source = c.source.replace("    report(rows, EXPECTED, list(EXPECTED))",
                                        '    report(rows, EXPECTED, list(EXPECTED)).to_csv(f"{RESULTS}/sec5d.csv", index=False)')
    cells += nb07
    cells += [nbf.v4.new_markdown_cell(
        "# Part B · Regenerate candidates on the Mac (optional, hours)\n\n"
        "Enable the flags in cell 1. Every capture is resumable: if the Mac sleeps or the kernel "
        "restarts, rerun cells 1-2 and the capture cell and it continues where it stopped. "
        "GPU sampling is not bit-identical to the A100 runs, so a fresh capture is a new random "
        "draw; expect agreement within about half a point."),
        nbf.v4.new_code_cell(QUICK.strip()),
        nbf.v4.new_code_cell(CAPTURE_MAIN.strip()),
        nbf.v4.new_code_cell(CAPTURE_PREFIX.strip()),
        nbf.v4.new_code_cell(CAPTURE_SPAN.strip()),
        nbf.v4.new_markdown_cell("# Summary"),
        nbf.v4.new_code_cell(SUMMARY.strip())]
    nb.cells = cells
    nbf.write(nb, B.NB_DIR / "LVPP_local_mac.ipynb")
    print("wrote LVPP_local_mac.ipynb with", len(cells), "cells")


if __name__ == "__main__":
    main()
