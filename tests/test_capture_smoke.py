"""Smoke-test the three capture paths on CPU with a tiny random Qwen2 model.

No network: the tokenizer is trained locally. This checks shapes, span
indexing, record layouts and that the outputs load with ``lvpp.caches``; it
does not check generation quality.
"""
import pytest

transformers = pytest.importorskip("transformers")
import torch  # noqa: E402

import lvpp.capture as capture  # noqa: E402
from lvpp.caches import load_dict_B, load_pool_A, load_prefix_C  # noqa: E402
from lvpp.prompts import FEWSHOT  # noqa: E402
from lvpp.answers import extract_chat_numeric as extract_chat_numeric_  # noqa: E402

CHAT_TEMPLATE = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n"
                 "{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")


@pytest.fixture(scope="module")
def tiny(monkeypatch_module=None):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    corpus = [FEWSHOT, "The answer is 42. \\boxed{7} Question: Answer: 0123456789 + - * / ="] * 20
    tk.train_from_iterator(corpus, trainers.BpeTrainer(
        vocab_size=400, special_tokens=["<unk>", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    from tokenizers import decoders
    tk.decoder = decoders.ByteLevel()
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", unk_token="<unk>")
    tok.pad_token = tok.eos_token
    tok.chat_template = CHAT_TEMPLATE
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
    model = Qwen2ForCausalLM(cfg).eval()
    return tok, model


@pytest.fixture
def patched(tiny, monkeypatch):
    monkeypatch.setattr(capture, "load_model", lambda name, adapter=None: tiny)
    monkeypatch.setattr(capture, "DEVICE", "cpu")


DATA = [dict(idx=i, question=f"Tom has {i + 2} apples and buys {i + 3} more. How many apples?",
             gold=str(2 * i + 5), answer=f"Total <<x>> {2 * i + 5}\n#### {2 * i + 5}") for i in range(3)]


def test_fewshot_capture(patched, tmp_path):
    cap = capture.FewShotCapture(capture.FewShotConfig(K=2, max_new_tokens=8, qbatch=2))
    pool = cap.build_pool(DATA, str(tmp_path / "pool.pt"), n_pool=3)
    assert pool["n_done"] == 3 and len(pool["labels"]) == 6
    assert pool["mid_mean"][0].shape == (32,) and pool["scalars"][0].shape == (5,)
    cap.build_test(DATA, str(tmp_path / "test.pt"))
    P, Z, y, _ = load_pool_A(str(tmp_path / "pool.pt"))
    assert P.shape == (6, 96)
    te = load_dict_B(str(tmp_path / "test.pt"), K=2)
    assert te.nq == 3 and te.P.shape == (6, 96)


def test_prefix_capture(patched, tmp_path):
    cfg = capture.PrefixConfig(K=2, max_new_tokens=8, budgets=(3, 6), shard=2, layers=(2, 3, 4))
    cap = capture.PrefixCapture(cfg)
    ds = [dict(question=d["question"], answer=d["answer"]) for d in DATA]
    cap.capture(ds, str(tmp_path / "test"), 3, cfg.seed_offset_test)
    cs = load_prefix_C(str(tmp_path), "test", 3, K=2)
    assert cs.nq == 3 and cs.P.shape == (6, 96)
    assert len(cs.extra["prefix_lp"]) == 6 and cs.extra["finished"].dtype == bool


@pytest.mark.parametrize("spans", [False, True])
def test_chat_capture(patched, tmp_path, spans):
    from lvpp.answers import extract_chat_numeric

    cap = capture.ChatCapture(capture.ChatConfig(K=2, max_new=8, qbatch=2, spans=spans))
    path = str(tmp_path / "c.pt")
    cap.build_cache("t", DATA, extract_chat_numeric, path, limit=3)
    keys = ("full",) if spans else capture.MULTI
    cs = load_dict_B(path, variants=keys, K=2)
    assert cs.nq == 3 and cs.P.shape[1] == (96 if spans else 96)
    if spans:
        assert "has_trigger" in cs.extra


def test_chat_capture_from_texts(patched, tmp_path):
    cap = capture.ChatCapture(capture.ChatConfig(K=2, max_new=8, qbatch=2, spans=True))
    assert len(cap.prompts(DATA)) == 3 and "apples" in cap.prompts(DATA)[0]
    texts = [["Step one. The answer is 7.", "So \\boxed{9}"], None, ["no trigger here 5", "The final answer is 11"]]
    path = str(tmp_path / "t.pt")
    cap.build_cache_from_texts("t", DATA, texts, extract_chat_numeric_, path, save_every=1)
    o = torch.load(path, weights_only=False)
    assert sorted(o) == [0, 2]
    assert [c["has_trigger"] for c in o[2]["cands"]] == [False, True]
    assert all(c["has_trigger"] for c in o[0]["cands"])
    assert o[0]["cands"][1]["answer"] == "9"
    cs = load_dict_B(path, variants=("reasoning",), K=2)
    assert cs.nq == 2 and cs.P.shape == (4, 96)
