"""Execute every cache-only notebook end to end on synthetic caches (CPU).

Slow (about 2-4 min); run with ``pytest -m notebooks`` or directly.
GPU capture cells are guarded by RUN_* flags that default to False.
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CACHE_ONLY = ["02_table1_selection", "03_table2_ablation", "04_table3_pruning_rules",
              "05_fig2_span_ablation", "06_table4_generators", "07_transfer_3b_math500",
              "LVPP_local_mac"]


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    root = tmp_path_factory.mktemp("synth")
    subprocess.run([sys.executable, str(ROOT / "tests/make_synthetic_caches.py"), str(root)], check=True)
    return root


@pytest.mark.notebooks
@pytest.mark.parametrize("name", CACHE_ONLY)
def test_notebook_runs(name, synth, tmp_path):
    import nbformat
    from nbclient import NotebookClient

    nb = nbformat.read(ROOT / "notebooks" / f"{name}.ipynb", as_version=4)
    env = dict(os.environ, LVPP_CACHE=str(synth), LVPP_RESULTS=str(tmp_path), MPLBACKEND="Agg",
               HOME=str(tmp_path))
    os.environ.update(env)
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT / "notebooks")}}).execute()
