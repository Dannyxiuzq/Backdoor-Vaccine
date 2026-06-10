"""Minimal unit tests for the SAART-P1 trainer helpers.

End-to-end smoke runs cannot catch SAART's silent bugs (off-by-one KL masks, inner-loop
gradient leakage, projection picking special tokens, the discrete pool never being used).
These tests exercise the pure helpers directly.

Run with the project's env, e.g.:
    /home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py
or via pytest:
    pytest tests/test_saart_trainer.py
"""

import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make the repo root importable when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llamafactory.extras.constants import IGNORE_INDEX
from llamafactory.train.sft.saart_trainer import SAARTSeq2SeqTrainer


# --------------------------------------------------------------------------- #
# lightweight fixtures (avoid the heavy Seq2SeqTrainer __init__)
# --------------------------------------------------------------------------- #
class DummyTokenizer:
    def __init__(self, pad=0, bos=1, eos=2, unk=3):
        self.pad_token_id = pad
        self.bos_token_id = bos
        self.eos_token_id = eos
        self.unk_token_id = unk
        self.padding_side = "right"
        self.all_special_ids = [pad, bos, eos, unk]


class ToyOutput:
    """Mimics a HF CausalLMOutput enough for the SAART code (.logits, .loss, [0])."""

    def __init__(self, logits, loss=None):
        self.logits = logits
        self.loss = loss

    def __getitem__(self, key):
        if isinstance(key, str):
            return {"loss": self.loss, "logits": self.logits}[key]
        return (self.loss, self.logits)[key]


class ToyLM(nn.Module):
    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, vocab, bias=False)
        self.config = type("cfg", (), {"max_position_embeddings": 4096})()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                labels=None, use_cache=False, output_hidden_states=False):
        if inputs_embeds is None:
            inputs_embeds = self.embed(input_ids)
        # Causal prefix-mean mixing so a token's logits depend on all earlier tokens
        # (including an inserted trigger) — otherwise a position-wise toy model is blind
        # to the trigger and the inner proxy can never move.
        csum = torch.cumsum(inputs_embeds, dim=1)
        denom = torch.arange(1, inputs_embeds.size(1) + 1, device=inputs_embeds.device)
        hidden = csum / denom.view(1, -1, 1).to(inputs_embeds.dtype)
        logits = self.proj(hidden)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].reshape(-1, logits.size(-1))
            shift_labels = labels[:, 1:].reshape(-1)
            loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=IGNORE_INDEX)
        return ToyOutput(logits, loss)


def make_trainer(**overrides):
    """Build a SAART trainer instance without running the (heavy) __init__."""
    t = SAARTSeq2SeqTrainer.__new__(SAARTSeq2SeqTrainer)
    defaults = dict(
        saart_trigger_len=2, saart_inner_steps=2, saart_inner_lr=0.05,
        saart_lambda1=1.0, saart_lambda2=0.5, saart_pool_size=4,
        saart_use_projection=True, saart_proj_every=1, saart_proj_keep_frac=0.8,
        saart_insert_position="after_bos", saart_use_null_reference=True,
        saart_null_init="mean_embedding", saart_soft_init="mean_embedding",
        saart_use_global_soft_seed=False, saart_soft_norm_clip=True,
        saart_inner_eval_mode=True, saart_kl_type="forward",
        saart_pool_sample_prob=0.3, saart_pool_policy="fifo", saart_pool_ema_beta=0.9,
        saart_projection_exclude_special=True, saart_log_every=1,
        # Phase-2 关联正则化字段（默认关闭；assoc 测试会按需 override）
        use_assoc_reg=False, assoc_lambda3=1.0, assoc_top_ratio=0.5, assoc_ema_alpha=0.9,
        assoc_align_lambda=0.01, assoc_target_layers="all", assoc_warmup_steps=0, assoc_select_every=1,
        saart_use_assoc_align=False,  # alignment 开关，默认关；align 测试按需 override
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(t, k, v)
    t._step_counter = 0
    t.trigger_pool = []
    t._global_soft_seed = None
    t._normed_embed_cache = None
    t._bad_token_id_cache = None
    t._warned = set()
    t._sanity_done = True
    t._rng = random.Random(0)
    t.tokenizer = DummyTokenizer()
    # Phase-2 关联签名状态（make_trainer 不走 __init__，故在此手工初始化）
    t._assoc_registered = False
    t._assoc_modules = []
    t._assoc_handles = []
    t._assoc_capture = None
    t._assoc_clean_acts = {}
    t._assoc_adv_acts = {}
    t._assoc_risk = {}
    t._assoc_signed = {}
    t._assoc_sig = {}
    return t


# --------------------------------------------------------------------------- #
# Test 1: causal-shift response-mask alignment (the highest-risk silent bug)
# --------------------------------------------------------------------------- #
def test_mask_alignment():
    t = make_trainer(saart_trigger_len=2, saart_insert_position="after_bos")
    labels = torch.tensor([[-100, -100, 10, 11, 12, -100]])  # B=1, S=6, 3 response tokens
    logits = torch.randn(1, 6, 16)

    clean_sel = t._select_response_logits(logits, labels)
    assert clean_sel.shape[0] == 3, f"clean count {clean_sel.shape[0]} != 3"
    # logits[:, i-1] predicts labels[:, i]; first response label 10 is at i=2 -> logit row 1.
    assert torch.equal(clean_sel[0], logits[0, 1]), "clean first selected logit not at the shifted position"

    base = torch.randn(1, 6, 8)
    attn = torch.ones(1, 6, dtype=torch.long)
    trig = torch.randn(1, 2, 8)
    positions = t._insertion_positions(torch.zeros(1, 6, dtype=torch.long), attn, labels)
    assert int(positions[0]) == 1, "after_bos insertion must be at position 1"

    e, a, l = t._build_triggered_inputs(base, attn, labels, trig, positions)
    assert e.shape[1] == 8 and a.shape[1] == 8 and l.shape[1] == 8, "lengthened seq must be S+k"
    assert int(a.sum()) == 8, "trigger positions must be attended"
    assert int((l == IGNORE_INDEX).sum()) == 5, "k trigger labels must be IGNORE_INDEX (3 orig + 2 trigger)"

    adv_logits = torch.randn(1, 8, 16)
    adv_sel = t._select_response_logits(adv_logits, l)
    assert adv_sel.shape[0] == 3, f"adv count {adv_sel.shape[0]} != 3 (must match clean)"
    # response shifted by k=2: first response now predicted from logit row 1+2=3.
    assert torch.equal(adv_sel[0], adv_logits[0, 3]), "adv first selected logit not at the k-shifted position"
    print("PASS test_mask_alignment")


# --------------------------------------------------------------------------- #
# Test 2 + 3: inner adversary never populates / clears model param grads
# (proves no `.backward()` into params and no `model.zero_grad()` — grad-accum safe)
# --------------------------------------------------------------------------- #
def _toy_inner_setup(t):
    torch.manual_seed(0)
    model = ToyLM(vocab=16, dim=8)
    model.train()
    embed_layer = model.get_input_embeddings()
    B, S = 2, 6
    input_ids = torch.randint(0, 16, (B, S))
    attn = torch.ones(B, S, dtype=torch.long)
    labels = torch.full((B, S), IGNORE_INDEX)
    labels[:, 3:] = input_ids[:, 3:]  # last 3 positions are response
    base = embed_layer(input_ids)
    positions = t._insertion_positions(input_ids, attn, labels)
    null = t._null_trigger_embeds(embed_layer, B)
    r_e, r_a, r_l = t._build_triggered_inputs(base, attn, labels, null, positions)
    ref_sel = t._select_response_logits(model(inputs_embeds=r_e, attention_mask=r_a).logits, r_l).detach()
    return model, embed_layer, base, attn, labels, positions, ref_sel


def test_inner_grad_isolation():
    t = make_trainer(saart_inner_steps=3)
    model, embed_layer, base, attn, labels, positions, ref_sel = _toy_inner_setup(t)
    # all param grads start as None
    for p in model.parameters():
        assert p.grad is None
    soft, proxy_init, proxy_final = t._inner_search(model, base, attn, labels, positions, ref_sel, embed_layer)
    for name, p in model.named_parameters():
        assert p.grad is None, f"inner search polluted param grad: {name}"
    assert torch.isfinite(torch.tensor(proxy_init)) and torch.isfinite(torch.tensor(proxy_final))
    assert model.training, "model must be restored to train() after inner search"
    # The symmetry-breaking soft init + sign ascent must actually raise the proxy: if soft
    # started identical to the null reference the gradient would be 0 and this would stay flat.
    assert proxy_final > proxy_init, f"inner adversary did not raise the proxy ({proxy_init:.5f} -> {proxy_final:.5f})"
    print(f"PASS test_inner_grad_isolation (proxy {proxy_init:.4f} -> {proxy_final:.4f})")


def test_inner_preserves_accumulated_grad():
    """A pre-existing accumulated gradient must survive the inner search (no zero_grad)."""
    t = make_trainer(saart_inner_steps=3)
    model, embed_layer, base, attn, labels, positions, ref_sel = _toy_inner_setup(t)
    sentinel = torch.ones_like(model.proj.weight)
    model.proj.weight.grad = sentinel.clone()
    t._inner_search(model, base, attn, labels, positions, ref_sel, embed_layer)
    assert model.proj.weight.grad is not None, "inner search cleared an accumulated grad (zero_grad leak!)"
    assert torch.equal(model.proj.weight.grad, sentinel), "inner search modified an accumulated grad"
    print("PASS test_inner_preserves_accumulated_grad")


def test_source_has_no_zero_grad_or_backward():
    """Static guard: SAART must never call model.zero_grad() or .backward() in its loss path.

    Strips comments and string literals first so that docstrings *mentioning* the forbidden
    calls (to explain why they are absent) don't trip the check.
    """
    import inspect
    import io
    import tokenize

    src = inspect.getsource(SAARTSeq2SeqTrainer)
    code_tokens = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type in (tokenize.COMMENT, tokenize.STRING):
            continue
        code_tokens.append(tok.string)
    code_only = " ".join(code_tokens)
    assert "zero_grad" not in code_only, "SAART trainer must not call zero_grad (breaks gradient accumulation)"
    assert ".backward" not in code_only and "backward" not in code_only, \
        "SAART trainer must not call .backward() (Trainer owns the backward)"
    print("PASS test_source_has_no_zero_grad_or_backward")


# --------------------------------------------------------------------------- #
# Test 4: discrete projection excludes special tokens
# --------------------------------------------------------------------------- #
def test_projection_excludes_special():
    torch.manual_seed(0)
    embed = nn.Embedding(16, 8)
    t = make_trainer(saart_trigger_len=1, saart_projection_exclude_special=True)
    eos = t.tokenizer.eos_token_id
    soft = embed.weight.data[eos:eos + 1].clone()  # exactly the eos embedding -> nearest is eos
    tok = t._project_tokens(soft, embed)
    assert tok.numel() == 1 and int(tok[0]) != eos, "special token (eos) leaked into projection"
    assert int(tok[0]) not in set(t.tokenizer.all_special_ids), "a special token leaked into projection"

    t2 = make_trainer(saart_trigger_len=1, saart_projection_exclude_special=False)
    tok2 = t2._project_tokens(soft, embed)
    assert int(tok2[0]) == eos, "without exclusion, nearest neighbour of the eos embedding should be eos"
    print("PASS test_projection_excludes_special")


# --------------------------------------------------------------------------- #
# Test 5: the discrete trigger pool actually drives the outer branch
# --------------------------------------------------------------------------- #
def test_pool_participates_in_training():
    torch.manual_seed(0)
    embed = nn.Embedding(16, 8)
    t = make_trainer(saart_trigger_len=2, saart_pool_sample_prob=1.0)

    assert t._use_pool_trigger() is False, "empty pool must never be sampled"
    pool_tok = torch.tensor([5, 9], dtype=torch.long)
    t.trigger_pool = [{"token_ids": pool_tok, "ema_proxy": 1.0, "seen": 1}]
    assert t._use_pool_trigger() is True, "non-empty pool with prob=1.0 must be sampled"

    B, S = 1, 5
    input_ids = torch.randint(0, 16, (B, S))
    base = embed(input_ids)
    attn = torch.ones(B, S, dtype=torch.long)
    labels = torch.tensor([[-100, -100, 6, 7, -100]])
    positions = t._insertion_positions(input_ids, attn, labels)  # after_bos -> 1
    e, a, l = t._discrete_trigger_inputs(base, attn, labels, positions, embed)
    expected = embed(pool_tok)  # [2, d]
    assert torch.allclose(e[0, 1:3], expected, atol=1e-5), "outer branch did not use the discrete pool trigger embeddings"
    print("PASS test_pool_participates_in_training")


# --------------------------------------------------------------------------- #
# Bonus: KL(p||p) == 0 for all directions (numerical sanity of the loss)
# --------------------------------------------------------------------------- #
def test_kl_self_is_zero():
    logits = torch.randn(7, 16)
    for kl_type in ("forward", "reverse", "symmetric", "js"):
        t = make_trainer(saart_kl_type=kl_type)
        kl = t._kl(logits.clone(), logits.clone())
        assert abs(float(kl)) < 1e-5, f"KL({kl_type}) of identical dists should be ~0, got {float(kl)}"
    print("PASS test_kl_self_is_zero")


# --------------------------------------------------------------------------- #
# Test 6 (P2): pool dedups identical triggers and EMA-updates their proxy
# --------------------------------------------------------------------------- #
def test_pool_ema_dedup_update():
    t = make_trainer(saart_pool_policy="topk_ema", saart_pool_size=8, saart_pool_ema_beta=0.5)
    tok = torch.tensor([3, 7], dtype=torch.long)
    t._add_to_pool(tok.clone(), 10.0)
    assert len(t.trigger_pool) == 1
    assert t.trigger_pool[0]["seen"] == 1 and t.trigger_pool[0]["ema_proxy"] == 10.0
    t._add_to_pool(tok.clone(), 20.0)  # identical token_ids -> dedup + EMA, not a new entry
    assert len(t.trigger_pool) == 1, "duplicate token_ids must not create a second pool entry"
    assert abs(t.trigger_pool[0]["ema_proxy"] - 15.0) < 1e-6, "EMA should be 0.5*10 + 0.5*20 = 15"
    assert t.trigger_pool[0]["seen"] == 2
    print("PASS test_pool_ema_dedup_update")


# --------------------------------------------------------------------------- #
# Test 7 (P2): topk_ema eviction keeps the highest-EMA-proxy triggers
# --------------------------------------------------------------------------- #
def test_pool_topk_eviction():
    t = make_trainer(saart_pool_policy="topk_ema", saart_pool_size=3)
    for i, proxy in enumerate([1.0, 5.0, 2.0, 9.0, 3.0]):
        t._add_to_pool(torch.tensor([i, i + 100], dtype=torch.long), proxy)  # all distinct -> no dedup
    assert len(t.trigger_pool) == 3, "pool must be capped at pool_size"
    emas = sorted(it["ema_proxy"] for it in t.trigger_pool)
    assert emas == [3.0, 5.0, 9.0], f"topk_ema must keep the 3 highest proxies, got {emas}"
    print("PASS test_pool_topk_eviction")


# =========================================================================== #
# Phase-2（Module 2）：在线 MLP 关联签名 + L_assoc-reg 测试
# =========================================================================== #
class _ToyMLP(nn.Module):
    """toy 版 LlamaMLP：gate/up: d->inter，down: inter->d；命名与真实模型一致，供 hook 解析测试。"""

    def __init__(self, d=8, inter=16):
        super().__init__()
        self.gate_proj = nn.Linear(d, inter)
        self.up_proj = nn.Linear(d, inter)
        self.down_proj = nn.Linear(inter, d)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class _ToyLayer(nn.Module):
    def __init__(self, d=8, inter=16):
        super().__init__()
        self.mlp = _ToyMLP(d, inter)

    def forward(self, x):
        return x + self.mlp(x)


class _ToyInner(nn.Module):
    def __init__(self, n=4, d=8, inter=16):
        super().__init__()
        self.layers = nn.ModuleList([_ToyLayer(d, inter) for _ in range(n)])

    def forward(self, x):
        for ly in self.layers:
            x = ly(x)
        return x


class _ToyTop(nn.Module):
    """顶层包一层 .model，使 named_modules 出现 'model.layers.i.mlp.gate_proj'（含 .layers. 供正则匹配）。"""

    def __init__(self, n=4, d=8, inter=16):
        super().__init__()
        self.model = _ToyInner(n, d, inter)

    def forward(self, x):
        return self.model(x)


def test_assoc_resolve_layers():
    # last2：4 层里取最后 2 层 × 3 个 proj = 6 个 module；gate/up=output、down=input
    t = make_trainer(use_assoc_reg=True, assoc_target_layers="last2")
    mods = t._resolve_assoc_modules(_ToyTop(n=4))
    assert len(mods) == 6, f"last2 应解析出 6 个 module，得到 {len(mods)}"
    kinds = {name.rsplit('.', 1)[1]: kind for name, kind in mods}
    assert kinds["gate_proj"] == "output" and kinds["up_proj"] == "output" and kinds["down_proj"] == "input"
    layers = sorted({int(name.split(".layers.")[1].split(".")[0]) for name, _ in mods})
    assert layers == [2, 3], f"last2 应取层 [2,3]，得到 {layers}"
    # everyN 与 all 的解析
    assert len(make_trainer(use_assoc_reg=True, assoc_target_layers="all")._resolve_assoc_modules(_ToyTop(n=4))) == 12
    print("PASS test_assoc_resolve_layers")


def test_assoc_hook_capture():
    """注册 hook 后，clean 模式抓到的激活 detach、adv 模式抓到的激活带梯度，且形状为 [B,T,inter]。"""
    from types import SimpleNamespace
    torch.manual_seed(0)
    t = make_trainer(use_assoc_reg=True, assoc_target_layers="last1")
    t.accelerator = SimpleNamespace(unwrap_model=lambda m: m)  # 桩：toy 自身即 unwrap 结果
    t.is_world_process_zero = lambda: False                    # 桩：跳过日志
    toy = _ToyTop(n=2, d=8, inter=16)
    t._register_assoc_hooks(toy)
    assert len(t._assoc_handles) == 3, "last1 应注册 3 个 hook(gate/up/down)"
    x = torch.randn(2, 5, 8)
    # clean 模式：抓到的激活应 detach（无梯度）
    t._assoc_capture = "clean"; toy(x); t._assoc_capture = None
    assert t._assoc_clean_acts, "clean 缓存应非空"
    for name, a in t._assoc_clean_acts.items():
        assert a.shape == (2, 5, 16), f"{name} 形状应为 [2,5,16]"
        assert not a.requires_grad, "clean 激活应被 detach"
    # adv 模式：抓到的激活应带梯度
    t._assoc_capture = "adv"; toy(x); t._assoc_capture = None
    assert any(a.requires_grad for a in t._assoc_adv_acts.values()), "adv 激活应保留梯度"
    print("PASS test_assoc_hook_capture")


def _toy_acts(t, name, B=1, S=6, k=2, C=4):
    """构造一对 clean/adv 激活 + labels，使 response token 数对齐（用于 deltas/loss 测试）。"""
    labels = torch.tensor([[-100, -100, 10, 11, 12, -100]])        # 3 个 response token
    adv_labels = torch.tensor([[-100, -100, -100, -100, 10, 11, 12, -100]])  # 右移 k=2，仍 3 个
    clean = torch.randn(B, S, C)
    adv = torch.randn(B, S + k, C, requires_grad=True)             # adv 带梯度
    t._assoc_modules = [(name, "output")]
    t._assoc_clean_acts = {name: clean.detach()}
    t._assoc_adv_acts = {name: adv}
    return labels, adv_labels, clean, adv


def test_assoc_deltas_and_loss():
    t = make_trainer(use_assoc_reg=True, assoc_top_ratio=0.5)
    name = "model.layers.0.mlp.gate_proj"
    labels, adv_labels, clean, adv = _toy_acts(t, name, C=4)
    deltas = t._compute_assoc_deltas(labels, adv_labels)
    assert name in deltas and deltas[name].shape == (3, 4), "Δh 应为 [Nresp=3, C=4]"
    assert deltas[name].requires_grad, "Δh 应带梯度（adv 侧）"
    # 选 S 前：loss 为 None（warmup/未选）
    assert t._assoc_reg_loss(deltas) is None, "未选 S 时 L_assoc 应为 None"
    # 选 S（top 50% of 4 = 2 通道）后：loss 为带梯度标量
    t._update_assoc_risk(deltas)
    t._select_assoc_signature()
    assert t._assoc_sig[name].numel() == 2, "top_ratio=0.5 of 4 通道 = 2"
    loss = t._assoc_reg_loss(deltas)
    assert loss is not None and loss.requires_grad and float(loss) >= 0, "L_assoc 应为带梯度非负标量"
    print("PASS test_assoc_deltas_and_loss")


def test_assoc_risk_ema():
    t = make_trainer(use_assoc_reg=True, assoc_ema_alpha=0.5)
    name = "m"
    # 第一次：risk = 本步幅度；第二次：risk = 0.5*old + 0.5*new
    d1 = {name: torch.tensor([[2.0, -4.0]])}   # |.|.mean(0) = [2,4]
    d2 = {name: torch.tensor([[6.0, 0.0]])}    # |.|.mean(0) = [6,0]
    t._update_assoc_risk(d1)
    assert torch.allclose(t._assoc_risk[name], torch.tensor([2.0, 4.0]))
    t._update_assoc_risk(d2)
    assert torch.allclose(t._assoc_risk[name], torch.tensor([4.0, 2.0])), "EMA: 0.5*[2,4]+0.5*[6,0]=[4,2]"
    print("PASS test_assoc_risk_ema")


def test_assoc_select_topk():
    t = make_trainer(use_assoc_reg=True, assoc_top_ratio=0.34)  # 0.34*5 -> max(1,int)=1
    t._assoc_risk = {"m": torch.tensor([0.1, 9.0, 0.2, 0.3, 0.4])}
    t._select_assoc_signature()
    assert int(t._assoc_sig["m"][0]) == 1, "应选风险最高的通道 idx=1"
    assert t._assoc_sig["m"].numel() == 1, "0.34*5 取整=1 通道"
    print("PASS test_assoc_select_topk")


ALL_TESTS = [
    test_mask_alignment,
    test_inner_grad_isolation,
    test_inner_preserves_accumulated_grad,
    test_source_has_no_zero_grad_or_backward,
    test_projection_excludes_special,
    test_pool_participates_in_training,
    test_kl_self_is_zero,
    test_pool_ema_dedup_update,
    test_pool_topk_eviction,
    test_assoc_resolve_layers,
    test_assoc_hook_capture,
    test_assoc_deltas_and_loss,
    test_assoc_risk_ema,
    test_assoc_select_topk,
]


if __name__ == "__main__":
    failed = 0
    for fn in ALL_TESTS:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"ERROR {fn.__name__}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{failed}/{len(ALL_TESTS)} SAART tests FAILED")
        sys.exit(1)
    print(f"\nAll {len(ALL_TESTS)} SAART tests passed.")
