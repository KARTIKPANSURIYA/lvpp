"""LoRA-distilled generators for Table IV (port of the original distillation run).

Fine-tunes Qwen2.5-1.5B on GSM8K training solutions (calculator annotations
removed, ending in "The answer is <gold>.") with a zero-shot prompt
``Question: q\\nAnswer:``. Rank 32 uses alpha 64 as in the paper; any other
rank uses alpha = 2 x rank.

    python -m lvpp.distill --rank 32 --out adapters/distill_r32
"""
from __future__ import annotations

import argparse
import random
import re
import time

import torch

from .answers import gsm8k_gold
from .verifier import set_seed


def format_target(answer_field: str, gold: str) -> str:
    r = answer_field.split("####")[0].strip()
    r = re.sub(r"<<.*?>>", "", r)
    r = re.sub(r"\s+\n", "\n", r).strip()
    return f"{r} The answer is {gold}."


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    set_seed(a.seed)
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    rows = [dict(q=ex["question"].strip(), t=format_target(ex["answer"], gsm8k_gold(ex["answer"])))
            for ex in load_dataset("openai/gsm8k", "main", split="train")]

    def encode(q, target):
        prompt = f"Question: {q}\nAnswer:"
        ids = tok(prompt + " " + target + tok.eos_token, truncation=True, max_length=a.max_len,
                  return_tensors="pt")["input_ids"][0]
        labels = ids.clone()
        labels[: len(tok(prompt, add_special_tokens=True)["input_ids"])] = -100
        return ids, labels

    cfg = LoraConfig(r=a.rank, lora_alpha=64 if a.rank == 32 else 2 * a.rank, lora_dropout=0.05,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     task_type="CAUSAL_LM")
    from .device import DEVICE, model_dtype
    model = get_peft_model(AutoModelForCausalLM.from_pretrained(
        a.model, torch_dtype=model_dtype(DEVICE), low_cpu_mem_usage=True).to(DEVICE), cfg)
    model.print_trainable_parameters()
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    for ep in range(a.epochs):
        order = list(range(len(rows)))
        random.shuffle(order)
        opt.zero_grad()
        t0, tot = time.time(), 0.0
        for j, i in enumerate(order):
            ids, lb = encode(rows[i]["q"], rows[i]["t"])
            out = model(input_ids=ids[None].to(DEVICE), labels=lb[None].to(DEVICE))
            (out.loss / a.grad_accum).backward()
            tot += float(out.loss)
            if (j + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                opt.zero_grad()
        print(f"epoch {ep + 1}/{a.epochs} loss {tot / len(order):.4f} ({time.time() - t0:.0f}s)")
    model.save_pretrained(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
