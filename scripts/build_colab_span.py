"""Build notebooks/LVPP_span_recapture_colab.ipynb.

A self-contained Colab notebook that recaptures the Fig. 2 span draw with the
corrected answer trigger, then reruns the span ablation and redraws the figure.
Sampling runs in vLLM; hidden-state features use the same teacher-forced pass
as the published pipeline. Everything is written to Google Drive and every
step is resumable.
"""
from pathlib import Path

import nbformat as nbf

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"

INTRO = r"""
# LVPP · Fig. 2 span ablation, recaptured with the corrected answer trigger

The published span draw split each solution with the regex
`\\\\boxed\{|[Tt]he answer is`, which never matches a real `\boxed{`. Only 9.8%
of test candidates were split, so for 90.2% the "reasoning" view equalled the
full solution. This notebook regenerates the draw with the corrected trigger
(`\\boxed\{|[Tt]he (?:final )?answer is`) and reruns the ablation.

Sampling runs in **vLLM** (Hugging Face `generate` needed about 26 s per
question on an A100, about 64 h for the full draw). The hidden-state features
are then taken with the same teacher-forced pass as before.

**How to run** (Colab Pro+ keeps running after you close the tab)
1. *Runtime → Change runtime type →* **A100 GPU**.
2. Run cell 1, then cell 2: it installs vLLM in its own environment and samples
   every question (about 1 h). Then cell 3 (features), then cells 4-5.
3. If the runtime stops, run cell 1 and continue with the cell you were on:
   every step skips what is already saved on Drive.

Generator: `Qwen/Qwen2.5-1.5B-Instruct`, chat prompt, T = 0.7, top-p 0.8,
512 new tokens, K = 8; the remaining sampling settings come from the model's
`generation_config.json`, as Hugging Face `generate` applied them.
"""

SETUP = r'''#@title 1 · Setup: mount Drive, install lvpp, check the GPU
import os, sys, subprocess, time, glob, json
IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)

ROOT = os.environ.get("LVPP_CACHE", "/content/drive/MyDrive/lcldd_runs")
OUT = os.environ.get("LVPP_SPAN_OUT", f"{ROOT}/span_fixed_vllm")  # new draw (resumable)
GEN = f"{OUT}/generations"
RESULTS = os.environ.get("LVPP_RESULTS", f"{ROOT}/lvpp_results")
for d in (OUT, GEN, RESULTS):
    os.makedirs(d, exist_ok=True)
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

N_POOL = 7473     # training questions for the span verifiers (7,473 = published)
N_TEST = 1319     # GSM8K test questions

if IN_COLAB:   # always install the newest lvpp_repo.zip from Drive
    zips = sorted(glob.glob(f"{ROOT}/lvpp_repo*.zip"))
    assert zips, f"Put lvpp_repo.zip in {ROOT}"
    subprocess.run(["rm", "-rf", "/content/lvpp_src"], check=True)
    subprocess.run(["unzip", "-q", "-o", zips[-1], "-d", "/content/lvpp_src"], check=True)
    src = os.path.dirname(glob.glob("/content/lvpp_src/**/pyproject.toml", recursive=True)[0])
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "--no-deps", "-e", src], check=True)
    sys.path.insert(0, src)
import lvpp  # noqa: E402
from lvpp.capture import ChatCapture  # noqa: E402
assert hasattr(ChatCapture, "build_cache_from_texts"), "Old lvpp loaded: Runtime -> Disconnect and delete runtime, then rerun"

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score
GPU = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
MEM = torch.cuda.get_device_properties(0).total_memory / 2**30 if torch.cuda.is_available() else 0
print(f"lvpp {lvpp.__version__} | GPU {GPU} ({MEM:.0f} GB) | out {OUT}")
if IN_COLAB and GPU == "none":
    print("!! No GPU: Runtime -> Change runtime type -> A100 GPU, then rerun this cell.")
'''

VLLM_SCRIPT = '''import json, os, sys, time
from vllm import LLM, SamplingParams
cfg = json.load(open(sys.argv[1]))
llm = LLM(cfg["model"], dtype="bfloat16", seed=0, gpu_memory_utilization=0.85, max_model_len=2048)
sp = SamplingParams(**cfg["sampling"])
for job in cfg["jobs"]:
    t0 = time.time()
    outs = llm.generate(job["prompts"], sp, use_tqdm=False)
    texts = [[o.text for o in r.outputs] for r in outs]
    tmp = job["out"] + ".tmp"
    json.dump(texts, open(tmp, "w"))
    os.replace(tmp, job["out"])
    print(f"saved {os.path.basename(job['out'])}: {len(texts)} questions in {time.time() - t0:.0f}s", flush=True)
'''

GENERATE = r'''#@title 2 · Sample all candidates with vLLM (install about 3 min, sampling about 1 h)
from transformers import AutoTokenizer, GenerationConfig
from lvpp.prompts import build_chat
from lvpp.data import gsm8k

if "cap" in globals():                      # free the GPU if a model is loaded in this kernel
    del cap
    torch.cuda.empty_cache()

VENV = "/content/vllm_env"
if not os.path.exists(f"{VENV}/bin/python"):
    print("installing vLLM in its own environment ...")
    subprocess.run([sys.executable, "-m", "pip", "-q", "install", "uv"], check=True)
    subprocess.run(["uv", "venv", VENV, "--python", "3.12", "-q"], check=True)
    subprocess.run(["uv", "pip", "install", "--python", f"{VENV}/bin/python", "-q", "vllm"], check=True)

tok = AutoTokenizer.from_pretrained(MODEL)
gc = GenerationConfig.from_pretrained(MODEL)
SAMPLING = dict(n=8, temperature=0.7, top_p=0.8, max_tokens=512,
                top_k=int(gc.top_k) if gc.top_k else -1,
                repetition_penalty=float(gc.repetition_penalty or 1.0))
print("sampling:", SAMPLING)

TEST, TRAIN = gsm8k("test")[:N_TEST], gsm8k("train")[:N_POOL]
CHUNK = 1000
jobs = []
for split, data in [("test", TEST), ("pool", TRAIN)]:      # test first
    for a in range(0, len(data), CHUNK):
        jobs.append(dict(out=f"{GEN}/{split}_{a:05d}.json",
                         prompts=[build_chat(tok, x["question"]) for x in data[a:a + CHUNK]]))
todo = [j for j in jobs if not os.path.exists(j["out"])]
print(f"{len(jobs) - len(todo)}/{len(jobs)} chunks already on Drive")

if todo:
    open("/content/gen_vllm.py", "w").write(VLLM_SCRIPT)
    json.dump(dict(model=MODEL, sampling=SAMPLING, jobs=todo), open("/content/gen_jobs.json", "w"))
    proc = subprocess.Popen([f"{VENV}/bin/python", "/content/gen_vllm.py", "/content/gen_jobs.json"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        tail = (tail + [line])[-40:]
        if line.startswith("saved"):
            print(line, end="")
    if proc.wait() != 0:
        print("".join(tail))
        raise RuntimeError("vLLM failed - see the log above")
print("all generations on Drive")
'''

FEATURES = r'''#@title 3 · Hidden-state features (teacher-forced, same code as the published pipeline)
from lvpp.capture import ChatCapture, ChatConfig
from lvpp.prompts import CHAT_TRIGGER
from lvpp.answers import extract_chat_numeric
from lvpp.data import gsm8k

def load_texts(split, n):
    texts = [None] * n
    for f in sorted(glob.glob(f"{GEN}/{split}_*.json")):
        a = int(os.path.basename(f).split("_")[1].split(".")[0])
        for k, t in enumerate(json.load(open(f))):
            if a + k < n:
                texts[a + k] = t
    return texts

TEST, TRAIN = gsm8k("test")[:N_TEST], gsm8k("train")[:N_POOL]
test_path, pool_path = f"{OUT}/testfeat.pt", f"{OUT}/pool.pt"
cap = ChatCapture(ChatConfig(model=MODEL, spans=True, trigger=CHAT_TRIGGER))

c = cap.build_cache_from_texts("span test", TEST, load_texts("test", N_TEST), extract_chat_numeric, test_path)
trig = np.mean([x["has_trigger"] for q in c.values() for x in q["cands"]])
print(f"\ncorrected trigger rate on the test set: {100 * trig:.1f}%   (published draw: 9.8%)\n")
cap.build_cache_from_texts("span pool", TRAIN, load_texts("pool", N_POOL), extract_chat_numeric, pool_path)
print("features done - run cell 4")
'''

ANALYSIS = r'''#@title 4 · Span ablation: three views x three seeds (about 5 min)
from lvpp.caches import load_dict_B
from lvpp.selection import selection_hits
from lvpp.verifier import fit_verifier
from lvpp.stats import paired, clustered_bootstrap_auc_diff
test_path, pool_path = f"{OUT}/testfeat.pt", f"{OUT}/pool.pt"

te_full = load_dict_B(test_path, variants=("full",))
te_reas = load_dict_B(test_path, variants=("reasoning",))
trig = te_full.extra["has_trigger"]
same = float((te_reas.P == te_full.P).all(1).float().mean())
n_pool_q = len(torch.load(pool_path, weights_only=False))
print(f"pool {n_pool_q} questions | test {len(te_full.gold)} questions")
print(f"trigger rate {100*trig.mean():.1f}% | reasoning == full features {100*same:.1f}%  (published: 9.8% / 90.2%)\n")

SEEDS = [0, 1, 2]; rows = {}; S = {}; H = {}
for span in ["full", "reasoning", "answer_span"]:
    tr = load_dict_B(pool_path, variants=(span,)); te = load_dict_B(test_path, variants=(span,))
    ok = torch.isfinite(tr.P).all(1) & torch.isfinite(tr.Z).all(1)
    accs, aucs, ss, hs = [], [], [], []
    for s in SEEDS:
        sc = fit_verifier(tr.P[ok], tr.Z[ok], tr.y[ok], seed=s).score(torch.nan_to_num(te.P), torch.nan_to_num(te.Z))
        h = selection_hits(te, "weighted", scores=sc)
        accs.append(100 * h.mean()); aucs.append(roc_auc_score(te.y.numpy(), sc)); ss.append(sc); hs.append(h)
    S[span] = np.mean(ss, 0); H[span] = hs
    rows[span] = float(np.mean(accs)); rows[span + " sd"] = float(np.std(accs))
    rows[span + " AUC"] = float(np.mean(aucs))
    print(f"{span:12s} {np.mean(accs):.2f} +/- {np.std(accs):.2f} | AUC {np.mean(aucs):.4f} | per seed {np.round(accs, 2)}")
rows["majority"] = 100 * selection_hits(te_full, "majority").mean()
print(f"{'majority':12s} {rows['majority']:.2f}\n")

for a in ["reasoning", "answer_span"]:
    d = [paired(H[a][i], H["full"][i]) for i in range(len(SEEDS))]
    lo, hi = clustered_bootstrap_auc_diff(te_full.y.numpy(), S[a], S["full"])
    rows[f"{a} - full (acc)"] = rows[a] - rows["full"]
    rows[f"{a} - full (AUC)"] = rows[a + " AUC"] - rows["full AUC"]
    rows[f"{a} - full AUC CI"] = f"[{lo:+.4f}, {hi:+.4f}]"
    print(f"{a} - full: acc {rows[a]-rows['full']:+.2f} (per seed {[round(x['diff'], 2) for x in d]}, "
          f"McNemar p {[float('%.3g' % x['p']) for x in d]}) | AUC {rows[a+' AUC']-rows['full AUC']:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]")

# Leakage test restricted to candidates that were actually split
m = trig.astype(bool)
for a in ["reasoning", "answer_span"]:
    print(f"triggered candidates only (n={m.sum()}): AUC full {roc_auc_score(te_full.y.numpy()[m], S['full'][m]):.4f} "
          f"| {a} {roc_auc_score(te_full.y.numpy()[m], S[a][m]):.4f}")

pd.Series({k: v for k, v in rows.items()}).to_csv(f"{RESULTS}/fig2_fixed.csv")
print(f"\nsaved {RESULTS}/fig2_fixed.csv")
'''

FIGURE = r'''#@title 5 · The figure (saved to Drive as fig_span_ablation_fixed.pdf)
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "pdf.fonttype": 42, "font.size": 8})
labels = ["complete\nsolution", "reasoning only\n(answer removed)", "answer span\nonly"]
keys = ["full", "reasoning", "answer_span"]
fig, ax = plt.subplots(1, 2, figsize=(7, 2.3))
ax[0].bar(labels, [rows[k] for k in keys], yerr=[rows[k + " sd"] for k in keys], capsize=3,
          color=["#2b6cb0", "#2f855a", "#c05621"])
ax[0].axhline(rows["majority"], ls="--", c="k", lw=0.8); ax[0].set_ylabel("selection accuracy (%)")
ax[0].set_ylim(rows["majority"] - 1, max(rows[k] + rows[k + " sd"] for k in keys) + 1)
ax[0].set_title("(a)", fontsize=8)
ax[1].plot(labels, [rows[k + " AUC"] for k in keys], "o-", c="#4a5568")
ax[1].set_ylabel("candidate AUC"); ax[1].set_title("(b)", fontsize=8)
fig.tight_layout(); fig.savefig(f"{RESULTS}/fig_span_ablation_fixed.pdf"); plt.show()
print(f"saved {RESULTS}/fig_span_ablation_fixed.pdf")
'''


def main():
    nb = nbf.v4.new_notebook()
    nb.metadata.update({"accelerator": "GPU", "colab": {"gpuType": "A100", "provenance": []},
                        "kernelspec": {"name": "python3", "display_name": "Python 3"}})
    # keep the #@title line first so Colab shows the form title
    first, rest = GENERATE.split("\n", 1)
    gen = first + "\nVLLM_SCRIPT = " + repr(VLLM_SCRIPT) + "\n" + rest
    nb.cells = [nbf.v4.new_markdown_cell(INTRO.strip())] + [
        nbf.v4.new_code_cell(s.strip()) for s in (SETUP, gen, FEATURES, ANALYSIS, FIGURE)]
    NB_DIR.mkdir(exist_ok=True)
    nbf.write(nb, NB_DIR / "LVPP_span_recapture_colab.ipynb")
    print("wrote LVPP_span_recapture_colab.ipynb")


if __name__ == "__main__":
    main()
