"""Publish the candidate/feature caches as a Hugging Face dataset.

Run once from Colab with Drive mounted and an HF write token:

    !pip -q install huggingface_hub
    !python scripts/upload_caches.py --cache /content/drive/MyDrive/lcldd_runs \
        --repo KARTIKPANSURIYA/lvpp-caches

Reviewers then reproduce every cache-only notebook with

    huggingface-cli download KARTIKPANSURIYA/lvpp-caches --repo-type dataset --local-dir lvpp_cache
    export LVPP_CACHE=$PWD/lvpp_cache
"""
import argparse
import os

# Only what the paper's tables and figure read (about 6 GB in total).
FILES = [
    "verifier_upgrade/pool.pt", "verifier_upgrade/testfeat.pt",
    "span_fixed_vllm/pool.pt", "span_fixed_vllm/testfeat.pt",
    "pending/mi_pool.pt", "pending/mi_test.pt", "pending/math_pool.pt", "pending/math_test.pt",
    "novelty/x15_pool.pt", "novelty/x3_pool.pt", "novelty/x3_test.pt",
    "verifier_fewshot/pool.pt", "verifier_fewshot/testfeat.pt",
    "verifier_mathmax/pool.pt", "verifier_mathmax/testfeat.pt",
    "verifier_final/pool.pt", "verifier_final/testfeat.pt",
]
DIRS = ["abort_main_draw/pool", "abort_main_draw/test", "span_fixed_vllm/generations"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi

    api = HfApi()
    if not a.dry_run:
        api.create_repo(a.repo, repo_type="dataset", exist_ok=True)
    total = 0
    for f in FILES:
        p = os.path.join(a.cache, f)
        if not os.path.exists(p):
            print("missing (skipped):", f)
            continue
        total += os.path.getsize(p)
        print(f"{f}  {os.path.getsize(p) / 1e6:.0f} MB")
        if not a.dry_run:
            api.upload_file(path_or_fileobj=p, path_in_repo=f, repo_id=a.repo, repo_type="dataset")
    for d in DIRS:
        p = os.path.join(a.cache, d)
        if not os.path.isdir(p):
            print("missing (skipped):", d)
            continue
        total += sum(os.path.getsize(os.path.join(p, x)) for x in os.listdir(p))
        print(f"{d}/  ({len(os.listdir(p))} shards)")
        if not a.dry_run:
            api.upload_folder(folder_path=p, path_in_repo=d, repo_id=a.repo, repo_type="dataset")
    print(f"total {total / 1e9:.2f} GB {'(dry run)' if a.dry_run else 'uploaded'}")


if __name__ == "__main__":
    main()
