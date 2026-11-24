# LVPP: Latent Verification with Prefix Pruning

Code for **"LVPP: Latent Verification with Prefix Pruning for Efficient
Self-Consistency in Small Language Models"**, Kartik Pansuriya, Congcong Xu,
Hena Kharwa and Min Song. ICNC 2027 (AI and Applications).

Self-consistency samples K solutions and takes a majority vote. LVPP reads the
hidden states of the frozen generator to (1) score each candidate with a
1.18M-parameter verifier and weight the vote, and (2) stop low-scoring
candidates at a fixed token checkpoint B before they finish.

| | GSM8K accuracy | generated tokens |
|---|---|---|
| Majority voting, K = 8 | 74.91% | 100% |
| LVPP weighted vote | **78.84%** (10 seeds) | 100% |
| LVPP + prefix pruning (B = 110, ρ = 0.5) | 76.55% (prefix draw; majority 73.92%) | **83.7%** |

---

## Repository layout

```
lvpp/                      the method, CPU-importable except capture.py
  capture.py               generation + hidden-state capture (GPU): main, prefix, chat
  verifier.py              the latent verifier and its training recipe
  selection.py             majority / log-prob tie-break / weighted vote (Eq. 1)
  pruning.py               prefix pruning at checkpoint B and token cost (Eq. 4)
  baselines.py             token-confidence weights, XGBoost (LiLaVe-style) comparator
  stats.py                 exact McNemar, paired CIs, clustered bootstrap
  caches.py                loaders for the three cache layouts
  probe.py                 stand-alone correctness classifier
  distill.py               LoRA-distilled generators (Table IV)
  answers.py, prompts.py, data.py
notebooks/                 one Colab notebook per table / figure (see below)
results/expected.json      every published number, its draw and its original run
scripts/build_notebooks.py notebooks are generated from this file
scripts/build_mac_notebook.py  the single-notebook Mac version
scripts/build_colab_span.py    the Fig. 2 draw notebook (vLLM sampling on Colab)
scripts/upload_caches.py   publishes the caches as a Hugging Face dataset
tests/                     unit tests, cache round-trips, capture smoke tests on a
                           tiny model, and end-to-end runs of every notebook
```

## Reproducing the paper

Each table or figure has one notebook. Open it in Colab (**Runtime → GPU**) and
run all cells; the first cell installs this package and mounts Drive.

| Paper | Notebook | Needs | Time (A100) |
|---|---|---|---|
| Main draw (Tables I, II) | `00_capture_main_draw` | GPU | ~7 h |
| Prefix draw (Tables II, III) | `01_capture_prefix_draw` | GPU | ~6 h |
| **Table I** selection strategies, Section V-A tests | `02_table1_selection` | cache | ~10 min |
| **Table II** component ablation | `03_table2_ablation` | cache | ~10 min |
| **Table III** pruning rules at B = 110 | `04_table3_pruning_rules` | cache | ~3 min |
| **Fig. 2**, Section V-C leakage tests | `05_fig2_span_ablation` | cache | ~8 min |
| Fig. 2 draw (regenerate) | `LVPP_span_recapture_colab` | GPU (vLLM) | ~1.5 h |
| **Table IV** other generators | `06_table4_generators` | cache (GPU optional) | ~5 min |
| Section V-D: 3B transfer, MATH-500 | `07_transfer_3b_math500` | cache (GPU optional) | ~5 min |

Every analysis notebook prints a **measured vs paper** table at the end and
writes CSVs to `$LVPP_RESULTS`.

### Running locally on a Mac (Apple Silicon)

`notebooks/LVPP_local_mac.ipynb` runs every experiment in one notebook in VS
Code, using the Metal (MPS) GPU. It finds the caches in a Google Drive for
desktop folder automatically, runs every table and figure in minutes, and can
optionally regenerate candidates (with a 16-question timing test that projects
the full runtime). Setup instructions are in its first cell.

### Route A: from the released caches (recommended)

GPU sampling is not bit-reproducible across hardware and library versions, so
the exact published numbers are tied to the cached candidates. Download them
and point the notebooks at the folder:

```bash
pip install "lvpp[gbt,math] @ git+https://github.com/KARTIKPANSURIYA/lvpp.git"
huggingface-cli download KARTIKPANSURIYA/lvpp-caches --repo-type dataset --local-dir lvpp_cache
export LVPP_CACHE=$PWD/lvpp_cache
```

With the caches, notebooks 02–07 retrain the verifiers with the paper's seeds
and should match the published values closely; residual differences come
from GPU non-determinism in verifier training. The measured-vs-paper table at
the end of each notebook flags anything outside tolerance.

### Route B: regenerate everything

Run `00` and `01` (and the `RUN_CAPTURE` cells of `05`–`07`) on a GPU, then the
analysis notebooks. A fresh capture is a new random draw: expect agreement
within roughly half a point, with the same conclusions and significance.

## Experimental configurations

| Draw | Generator and prompt | Verifier pool | Test | Used in |
|---|---|---|---|---|
| main | Qwen2.5-1.5B (base), 4 exemplars, T = 0.7, top-p 0.95, 256 tokens, K = 8 | all 7,473 GSM8K train questions (59,784 candidates, 76.8% correct) | 1,319 | Tables I, II |
| prefix | same generator, per-question seed, stop at `\nQuestion:` | 1,864 GSM8K train questions (fixed shuffle, seed 1234) | 1,319 | Tables II (pruning rows), III |
| span | Qwen2.5-1.5B-**Instruct**, chat prompt, T = 0.7, top-p 0.8, top-k 20, repetition penalty 1.1, 512 tokens, sampled with vLLM | 7,473 | 1,319 | Fig. 2 |
| Table IV | see notebook 06; **single-layer verifier, 800-question pools** | 800 | 1,319 | Table IV |
| 3B | Qwen2.5-3B-Instruct, chat prompt | 1,200 | 400 | Section V-D |
| MATH | Qwen2.5-Math-1.5B-Instruct, chat prompt, 1,536 tokens | 3,000 MATH train problems | MATH-500 | Section V-D |

**Verifier recipe** (all draws): pooled hidden states at layers L/2, 3L/4, L
(14, 21, 28 for the 1.5B model) plus five scalars z(c), z-scored with pool
statistics; Linear(4,613 → 256) – ReLU – Dropout 0.1 – Linear(256 → 1); random
90% of pool candidates for training; BCE with positive-class weighting;
AdamW, learning rate 1e-3, batch 128, 8 epochs. Prefix verifiers use the same
recipe on features pooled over the first min(B, T) tokens, one per B.

**Statistics**: exact two-sided McNemar on identical questions; a result is
called significant only if p < 0.05 for every training seed. Standard
deviations in the paper are population SDs over seeds.

### Known caveats (also stated in the notebooks)

* **Fig. 2 draw.** An earlier version of the span draw split reasoning from the
  answer with `\\\\boxed\{|[Tt]he answer is`, which never matches a real
  `\boxed{` (kept as `CHAT_TRIGGER_LEGACY`); for 90.2% of candidates the
  reasoning view equalled the full solution. The paper's Fig. 2 uses a fresh
  draw split with the corrected `CHAT_TRIGGER` (91.4% of test candidates
  split), regenerated with `LVPP_span_recapture_colab.ipynb`.
* **Table III random / oracle rows** in the paper were measured at B = 126
  (stated in the caption); notebook 04 also computes them at B = 110.

## Tests

```bash
pip install -e ".[dev,gbt]"
pytest -q                       # unit tests, cache layouts, capture smoke tests
pytest -q -m notebooks          # executes notebooks 02-07 on synthetic caches (~1 min)
```

## Citation

```bibtex
@inproceedings{pansuriya2027lvpp,
  title     = {{LVPP}: Latent Verification with Prefix Pruning for Efficient
               Self-Consistency in Small Language Models},
  author    = {Pansuriya, Kartik and Xu, Congcong and Kharwa, Hena and Song, Min},
  booktitle = {International Conference on Computing, Networking and Communications (ICNC)},
  year      = {2027}
}
```

MIT License.
