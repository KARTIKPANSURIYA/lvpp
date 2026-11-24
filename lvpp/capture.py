"""Candidate generation and hidden-state feature capture (GPU).

Three capture paths, each a faithful port of the notebook that produced the
paper's caches:

* ``FewShotCapture`` - base Qwen2.5-1.5B with four exemplars. Produces the
  *main draw* (Tables I, II, IV few-shot row): full-solution features for five
  pooling variants plus the five scalars z(c).
* ``PrefixCapture`` - same generator, one question at a time with a fixed seed
  per question. Produces the *prefix draw* (Tables II pruning rows, III):
  pooled prefix features at each checkpoint B, prefix log-probabilities,
  and which candidates finished before B.
* ``ChatCapture`` - instruction-tuned generators with a chat template
  (Math-Instruct, 1.5B-Instruct span ablation, 3B-Instruct transfer, MATH-500).

Layers: mid = L//2, upper = 3L//4, last = L (14, 21, 28 for Qwen2.5-1.5B).
"""
from __future__ import annotations

import inspect
import math
import os
import random
import time
from dataclasses import dataclass, field

import torch

from .answers import clip_continuation, extract_fewshot, extract_fewshot_clipped, gsm8k_gold, norm
from .prompts import (
    CHAT_TRIGGER,
    FEWSHOT_TRIGGER,
    build_chat,
    build_fewshot,
    split_solution,
)

VARIANTS = ("last_mean", "last_ansspan", "last_final", "mid_mean", "upper_mean")
MULTI = ("mid_mean", "upper_mean", "last_mean")  # the paper's three-depth read
from .device import DEVICE, model_dtype  # cuda > mps > cpu


def load_model(name: str, adapter: str | None = None):
    """Load a frozen bfloat16 model (optionally with a LoRA adapter)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = model_dtype(DEVICE)
    if DEVICE == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, low_cpu_mem_usage=True, device_map="cuda").eval()
    else:  # Apple Silicon (mps) or CPU
        model = AutoModelForCausalLM.from_pretrained(
            name, torch_dtype=dtype, low_cpu_mem_usage=True).to(DEVICE).eval()
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter).eval()
    for p in model.parameters():
        p.requires_grad = False
    return tok, model


def _layers(model):
    cfg = model.config
    nl = cfg.num_hidden_layers
    return nl, cfg.hidden_size, nl // 2, (3 * nl) // 4


def _scalars(spn: torch.Tensor, token_lp: torch.Tensor, length_norm: float) -> torch.Tensor:
    """z(c): mean squared latent velocity, mean positive growth, max norm/100,
    length/budget, mean token log-probability."""
    if spn.shape[0] >= 3:
        v = spn[1:] - spn[:-1]
        vn = v.pow(2).mean(-1)
        con = float(torch.relu(vn[1:] - vn[:-1]).mean()) if vn.shape[0] >= 2 else 0.0
        mv = float(vn.mean())
    else:
        con = mv = 0.0
    lp = float(token_lp.mean()) if token_lp.numel() else 0.0
    return torch.tensor(
        [mv, con, float(spn.norm(dim=-1).max()) / 100.0, spn.shape[0] / length_norm, lp],
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------- #
#  Main draw: few-shot base model                                              #
# --------------------------------------------------------------------------- #
@dataclass
class FewShotConfig:
    model: str = "Qwen/Qwen2.5-1.5B"
    adapter: str | None = None
    K: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    max_len: int = 1280
    qbatch: int = 8
    global_seed: int = 42  # set once before the pool, as in the original run
    zero_shot_prompt: bool = False  # True for LoRA-distilled generators ("Question: q\nAnswer:")


class FewShotCapture:
    def __init__(self, cfg: FewShotConfig):
        self.cfg = cfg
        self.tok, self.model = load_model(cfg.model, cfg.adapter)
        self.nl, self.d, self.l_mid, self.l_up = _layers(self.model)

    def prompt(self, q: str) -> str:
        return f"Question: {q}\nAnswer:" if self.cfg.zero_shot_prompt else build_fewshot(q)

    @torch.no_grad()
    def sample(self, questions):
        """K samples for several questions in one ``generate`` call (left padding)."""
        tok, c = self.tok, self.cfg
        tok.padding_side = "left"
        enc = tok([self.prompt(q) for q in questions], return_tensors="pt", padding=True,
                  truncation=True, max_length=c.max_len).to(DEVICE)
        L = enc["input_ids"].shape[1]
        g = self.model.generate(**enc, max_new_tokens=c.max_new_tokens, do_sample=True,
                                temperature=c.temperature, top_p=c.top_p,
                                num_return_sequences=c.K, pad_token_id=tok.eos_token_id)
        txt = [tok.decode(o[L:], skip_special_tokens=True) for o in g]
        return [txt[i * c.K:(i + 1) * c.K] for i in range(len(questions))]

    @torch.no_grad()
    def features(self, q: str, sols):
        """All five pooling variants for the K candidates in one padded forward pass."""
        tok, c = self.tok, self.cfg
        prompt = self.prompt(q)
        plen = len(tok(prompt, add_special_tokens=True)["input_ids"])
        rlens = []
        for s in sols:
            reasoning, has = split_solution(s, FEWSHOT_TRIGGER)
            rlens.append(len(tok(prompt + " " + reasoning, add_special_tokens=True)["input_ids"])
                         if has else None)
        tok.padding_side = "right"
        enc = tok([prompt + " " + s for s in sols], return_tensors="pt", padding=True,
                  truncation=True, max_length=c.max_len + c.max_new_tokens).to(DEVICE)
        out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        HL, HM, HU = hs[-1].float(), hs[self.l_mid].float(), hs[self.l_up].float()
        lens = enc["attention_mask"].sum(1).tolist()
        lg = torch.log_softmax(out.logits.float(), dim=-1)
        feats = []
        for i, Li in enumerate(lens):
            Li = int(Li)
            s0 = min(max(plen - 1, 0), Li - 1)
            mid = Li if rlens[i] is None else min(max(rlens[i] - 1, s0 + 1), Li)
            sp = lambda H, a, b: (H[i, a:b] if b > a else H[i, a:a + 1])  # noqa: E731
            f = {"last_mean": sp(HL, s0, Li).mean(0), "last_ansspan": sp(HL, mid, Li).mean(0),
                 "last_final": HL[i, Li - 1], "mid_mean": sp(HM, s0, Li).mean(0),
                 "upper_mean": sp(HU, s0, Li).mean(0)}
            spn = sp(HL, s0, Li)
            tl = lg[i, torch.arange(Li - 1, device=lg.device), enc["input_ids"][i, 1:Li]]
            z = _scalars(spn, tl[s0:] if s0 < Li - 1 else torch.zeros(0), 256.0)
            feats.append({k: v.cpu().half() for k, v in f.items()} | {"scalar": z})
        return feats

    def build_pool(self, train, path: str, n_pool: int | None = None, log_every: int = 32):
        """Resumable pool capture in the ``verifier_upgrade/pool.pt`` layout."""
        n_pool = n_pool or len(train)
        if os.path.exists(path):
            pool = torch.load(path, weights_only=False)
        else:
            random.seed(self.cfg.global_seed); torch.manual_seed(self.cfg.global_seed)
            pool = {**{k: [] for k in VARIANTS}, "scalars": [], "labels": [], "qid": [],
                    "answers": [], "n_done": 0}
        n, t0, start = pool["n_done"], time.time(), pool["n_done"]
        while n < n_pool:
            idxs = list(range(n, min(n + self.cfg.qbatch, n_pool)))
            batch = [train[i] for i in idxs]
            for bi, sols in zip(idxs, self.sample([b["question"] for b in batch])):
                gold = train[bi]["gold"]
                for sol, f in zip(sols, self.features(train[bi]["question"], sols)):
                    for k in VARIANTS:
                        pool[k].append(f[k])
                    a = extract_fewshot(sol)
                    pool["scalars"].append(f["scalar"]); pool["qid"].append(bi)
                    pool["answers"].append(a)
                    pool["labels"].append(1.0 if (a == gold and gold != "") else 0.0)
            n = idxs[-1] + 1
            pool["n_done"] = n
            if n % log_every < self.cfg.qbatch:
                torch.save(pool, path)
                r = (time.time() - t0) / max(n - start, 1)
                print(f"pool {n}/{n_pool} | {r:.2f}s/q | ETA {r * (n_pool - n) / 3600:.1f} h")
        torch.save(pool, path)
        return pool

    def build_test(self, test, path: str, log_every: int = 4):
        """Resumable test capture in the ``verifier_upgrade/testfeat.pt`` layout."""
        tf = torch.load(path, weights_only=False) if os.path.exists(path) else {}
        todo = [s for s in test if s["idx"] not in tf]
        for j in range(0, len(todo), self.cfg.qbatch):
            batch = todo[j:j + self.cfg.qbatch]
            for b, sols in zip(batch, self.sample([b["question"] for b in batch])):
                fs = self.features(b["question"], sols)
                tf[b["idx"]] = dict(gold=str(b["gold"]), cands=[
                    {**{k: f[k] for k in VARIANTS}, "scalar": f["scalar"],
                     "answer": extract_fewshot(sol)} for sol, f in zip(sols, fs)])
            if (j // self.cfg.qbatch) % log_every == 0:
                torch.save(tf, path)
                print(f"test {len(tf)}/{len(test)}")
        torch.save(tf, path)
        return tf


# --------------------------------------------------------------------------- #
#  Prefix draw: per-question seeding, features at each checkpoint B            #
# --------------------------------------------------------------------------- #
@dataclass
class PrefixConfig:
    model: str = "Qwen/Qwen2.5-1.5B"
    K: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 256
    budgets: tuple = (110, 126, 144, 169)
    shard: int = 25
    order_seed: int = 1234
    seed_offset_test: int = 2
    seed_offset_pool: int = 1
    n_pool: int = 1864
    layers: tuple = (14, 21, 28)


class PrefixCapture:
    def __init__(self, cfg: PrefixConfig):
        self.cfg = cfg
        self.tok, self.model = load_model(cfg.model)
        self.stop = "stop_strings" in inspect.signature(self.model.generate).parameters

    @torch.no_grad()
    def record(self, ex, qi: int, seed_offset: int) -> dict:
        c, tok, model = self.cfg, self.tok, self.model
        enc = tok(build_fewshot(ex["question"]), return_tensors="pt").to(DEVICE)
        plen = enc.input_ids.shape[1]
        torch.manual_seed(seed_offset * 100003 + qi)
        kw = dict(do_sample=True, temperature=c.temperature, top_p=c.top_p,
                  max_new_tokens=c.max_new_tokens, num_return_sequences=c.K,
                  pad_token_id=tok.pad_token_id, return_dict_in_generate=True,
                  output_scores=True)
        if self.stop:
            kw.update(stop_strings=["\nQuestion:"], tokenizer=tok)
        g = model.generate(**enc, **kw)
        lps = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True)
        rec = dict(gold=norm(gsm8k_gold(ex["answer"])), answers=[], n_total=[], scalar=[],
                   feats={B: [] for B in c.budgets},
                   finished_before={B: [] for B in c.budgets},
                   prefix_lp={B: [] for B in c.budgets})
        for k in range(c.K):
            ids = g.sequences[k, plen:]
            raw = max(int((ids != tok.eos_token_id).sum()), 1)
            full = tok.decode(ids[:raw], skip_special_tokens=True)
            sol = clip_continuation(full)
            glen = max(1, min(len(tok(sol, add_special_tokens=False)["input_ids"]), raw))
            rec["answers"].append(extract_fewshot_clipped(full))
            rec["n_total"].append(glen)
            hs = model(g.sequences[k:k + 1, :plen + glen], output_hidden_states=True).hidden_states
            spn = hs[-1].float()[0, plen:plen + glen]
            tl = lps[k, :glen]
            tl = tl[torch.isfinite(tl)]
            z = _scalars(spn, tl, float(c.max_new_tokens))
            if not tl.numel():
                z[4] = -20.0
            rec["scalar"].append(z)
            for B in c.budgets:
                upto = plen + min(B, glen)
                rec["feats"][B].append(torch.cat(
                    [hs[L][0, plen:upto].mean(0) for L in c.layers]).float().cpu().half())
                rec["finished_before"][B].append(bool(glen <= B))
                tb = lps[k, :min(B, glen)]
                tb = tb[torch.isfinite(tb)]
                rec["prefix_lp"][B].append(float(tb.mean()) if tb.numel() else -20.0)
            del hs
        for B in c.budgets:
            rec["feats"][B] = torch.stack(rec["feats"][B])
        rec["scalar"] = torch.stack(rec["scalar"])
        return rec

    def capture(self, ds, out_dir: str, n: int, seed_offset: int):
        """Resumable sharded capture; question order is a fixed shuffle (seed 1234)."""
        os.makedirs(out_dir, exist_ok=True)
        order = list(range(len(ds)))
        random.Random(self.cfg.order_seed).shuffle(order)
        order = order[:n]
        nsh = math.ceil(n / self.cfg.shard)
        for si in range(nsh):
            p = f"{out_dir}/shard_{si:04d}.pt"
            if os.path.exists(p):
                continue
            t0 = time.time()
            torch.save([self.record(ds[q], q, seed_offset)
                        for q in order[si * self.cfg.shard:(si + 1) * self.cfg.shard]], p)
            el = time.time() - t0
            print(f"[{os.path.basename(out_dir)}] {si + 1}/{nsh} ({el:.0f}s, "
                  f"ETA {el * (nsh - si - 1) / 60:.0f} min)")


# --------------------------------------------------------------------------- #
#  Chat-template generators                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class ChatConfig:
    model: str = "Qwen/Qwen2.5-Math-1.5B-Instruct"
    K: int = 8
    temperature: float = 0.7
    top_p: float = 0.8
    max_new: int = 512
    max_ctx: int = 1024
    qbatch: int = 8
    spans: bool = False  # True: capture full / reasoning / answer_span (Fig. 2)
    trigger: object = field(default=CHAT_TRIGGER)


class ChatCapture:
    def __init__(self, cfg: ChatConfig):
        self.cfg = cfg
        self.tok, self.model = load_model(cfg.model)
        self.nl, self.d, self.l_mid, self.l_up = _layers(self.model)

    @torch.no_grad()
    def sample(self, questions):
        tok, c = self.tok, self.cfg
        tok.padding_side = "left"
        enc = tok([build_chat(tok, q) for q in questions], return_tensors="pt", padding=True,
                  truncation=True, max_length=c.max_ctx).to(DEVICE)
        L = enc["input_ids"].shape[1]
        g = self.model.generate(**enc, max_new_tokens=c.max_new, do_sample=True,
                                temperature=c.temperature, top_p=c.top_p,
                                num_return_sequences=c.K,
                                pad_token_id=tok.pad_token_id or tok.eos_token_id)
        t = [tok.decode(o[L:], skip_special_tokens=True) for o in g]
        return [t[i * c.K:(i + 1) * c.K] for i in range(len(questions))]

    @torch.no_grad()
    def features(self, q: str, sols):
        tok, c = self.tok, self.cfg
        prompt = build_chat(tok, q)
        plen = len(tok(prompt, add_special_tokens=False)["input_ids"])
        rl, trig = [], []
        for s in sols:
            r, h = split_solution(s, c.trigger, cut_next_question=False)
            trig.append(h)
            rl.append(len(tok(prompt + r, add_special_tokens=False)["input_ids"]) if h else None)
        tok.padding_side = "right"
        enc = tok([prompt + s for s in sols], return_tensors="pt", padding=True,
                  truncation=True, max_length=c.max_ctx + c.max_new).to(DEVICE)
        out = self.model(**enc, output_hidden_states=True)
        hs = out.hidden_states
        HL, HM, HU = hs[-1].float(), hs[self.l_mid].float(), hs[self.l_up].float()
        lens = enc["attention_mask"].sum(1).tolist()
        lg = torch.log_softmax(out.logits.float(), -1)
        res = []
        for i, Li in enumerate(lens):
            Li = int(Li)
            s0 = min(max(plen - 1, 0), Li - 1)
            mid = Li if rl[i] is None else min(max(rl[i] - 1, s0 + 1), Li)
            spn = HL[i, s0:Li] if Li > s0 else HL[i, s0:s0 + 1]
            tl = lg[i, torch.arange(Li - 1, device=lg.device), enc["input_ids"][i, 1:Li]]
            z = _scalars(spn, tl[s0:] if s0 < Li - 1 else torch.zeros(0), float(c.max_new))
            if c.spans:
                def pool(a, b):
                    a, b = int(a), int(b)
                    if b <= a:
                        b = a + 1
                    b = min(b, Li); a = min(a, b - 1)
                    return torch.cat([HM[i, a:b].mean(0), HU[i, a:b].mean(0), HL[i, a:b].mean(0)])
                f = {"full": pool(s0, Li), "reasoning": pool(s0, mid), "answer_span": pool(mid, Li)}
                res.append({k: v.cpu().half() for k, v in f.items()}
                           | {"scalar": z, "has_trigger": bool(trig[i])})
            else:
                sp = lambda H, a, b: (H[i, a:b] if b > a else H[i, a:a + 1])  # noqa: E731
                f = {"last_mean": sp(HL, s0, Li).mean(0), "last_ansspan": sp(HL, mid, Li).mean(0),
                     "last_final": HL[i, Li - 1], "mid_mean": sp(HM, s0, Li).mean(0),
                     "upper_mean": sp(HU, s0, Li).mean(0)}
                res.append({k: v.cpu().half() for k, v in f.items()} | {"scalar": z})
        return res

    def _entry(self, x, sols, extract, keys):
        fs = self.features(x["question"], sols)
        return dict(gold=str(x["gold"]), cands=[
            {**{k: f[k] for k in keys}, "scalar": f["scalar"], "answer": extract(s),
             **({"has_trigger": f["has_trigger"]} if self.cfg.spans else {})}
            for s, f in zip(sols, fs)])

    def prompts(self, data):
        """Chat prompts exactly as ``sample`` builds them (for an external sampler such as vLLM)."""
        return [build_chat(self.tok, x["question"]) for x in data]

    def build_cache_from_texts(self, tag, data, texts, extract, path, save_every=200):
        """Like ``build_cache``, but with solutions sampled elsewhere.

        ``texts[i]`` holds the K sampled continuations for ``data[i]`` (or None
        if not generated yet). Only the teacher-forced feature pass runs here.
        """
        cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}
        keys = ("full", "reasoning", "answer_span") if self.cfg.spans else VARIANTS
        todo = [i for i, t in enumerate(texts) if t is not None and i not in cache]
        t0 = time.time()
        for n, i in enumerate(todo, 1):
            cache[i] = self._entry(data[i], texts[i], extract, keys)
            if n % save_every == 0 or n == len(todo):
                torch.save(cache, path)
                r = (time.time() - t0) / n
                print(f"{tag} {len(cache)}/{len(texts)} | {r:.2f}s/q | ETA {r * (len(todo) - n) / 60:.0f} min")
        torch.save(cache, path)
        return cache

    def build_cache(self, tag, data, extract, path, limit=None, log_every=8):
        """Resumable dict cache ``{index: {gold, cands: [...]}}``."""
        cache = torch.load(path, weights_only=False) if os.path.exists(path) else {}
        todo = [(i, d) for i, d in enumerate(data) if i not in cache]
        if limit is not None:
            todo = [t for t in todo if t[0] < limit]
        keys = ("full", "reasoning", "answer_span") if self.cfg.spans else VARIANTS
        t0, tot = time.time(), limit or len(data)
        for j in range(0, len(todo), self.cfg.qbatch):
            b = todo[j:j + self.cfg.qbatch]
            for (i, x), sols in zip(b, self.sample([x[1]["question"] for x in b])):
                cache[i] = self._entry(x, sols, extract, keys)
            if (j // self.cfg.qbatch) % log_every == 0:
                torch.save(cache, path)
                r = (time.time() - t0) / max(j + len(b), 1)
                print(f"{tag} {len(cache)}/{tot} | {r:.2f}s/q | ETA {r * (tot - len(cache)) / 60:.0f} min")
        torch.save(cache, path)
        print(f"{tag} done ({len(cache)})")
        return cache
