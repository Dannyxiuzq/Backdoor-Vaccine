# SAART alignment 项 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 补全 SAART Module 2 的方向一致性（alignment）项：把每通道带符号 shift 的 EMA 折进风险评分 s_j，让高风险通道集 S 选得更准；默认关闭、向后兼容。

**Architecture:** 在 `_update_assoc_risk` 并行维护一个带符号 EMA `_assoc_signed`；新方法 `_assoc_score` 算 `s_j = mag_j·(1+λ_align·align_j)`，`align_j=|signed_j|/(mag_j+ε)`；`_select_assoc_signature` 改用 s_j 选 top-τ%。开关 `saart_use_assoc_align`（默认 False）关闭时 s_j 严格退回 magnitude，与现有结果逐位等价。

**Tech Stack:** PyTorch, vendored LlamaFactory, 自定义测试 runner（`python tests/test_saart_trainer.py`，非 pytest）。

**测试命令（全程统一）:** `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`

---

## 文件结构

| 文件 | 责任 | 改动 |
|---|---|---|
| `llamafactory/hparams/finetuning_args.py` | SAART 超参定义 | 加 `saart_use_assoc_align` 字段；`assoc_align_lambda` 默认 0.01→1.0 |
| `llamafactory/train/sft/saart_trainer.py` | 训练核心 | `_assoc_signed` 状态、`_update_assoc_risk` 加 signed EMA、新 `_assoc_score`、`_select_assoc_signature` 改用 score、`_maybe_log` 加 align、删 TODO:597 |
| `step2_generate_training.py` | 短键→saart_ 字段映射 | `_SAART_FIELD_MAP` 加 `use_assoc_align` |
| `configs/experiment.llama2_7b_chat.yaml` | 主设定 config | `saart:` 块加 `use_assoc_align: false` |
| `tests/test_saart_trainer.py` | 单测 | fixture 加 2 个默认/状态；+5 alignment 测试 |
| `scripts/run_assoc_align_sweep.sh` | 实验 | llama2 上 λ_align sweep（两阶段多卡） |

---

### Task 1: 基础设施 — 参数定义 + 状态初始化 + fixture

让后续 TDD 能跑：定义开关、初始化 `_assoc_signed`、更新测试 fixture。

**Files:**
- Modify: `llamafactory/hparams/finetuning_args.py:434`
- Modify: `llamafactory/train/sft/saart_trainer.py:135,146`
- Modify: `tests/test_saart_trainer.py:98,119`

- [ ] **Step 1: 改 `assoc_align_lambda` 默认值并新增开关字段**

在 `finetuning_args.py`，把 `assoc_align_lambda` 的 field 替换为（默认 0.01→1.0，更新 help）：

```python
    assoc_align_lambda: float = field(
        default=1.0,  # 方向一致性项权重：s_j = mag*(1+lambda*align)，align∈[0,1] 需 ~1 才实质改变 S（旧默认 0.01 几乎无效）
        metadata={"help": "Weight of the cross-step activation-direction alignment term in the association risk score."},
    )
    saart_use_assoc_align: bool = field(
        default=False,  # 默认关：S 选择退回 magnitude-only，与既有 RQ6/P2 结果逐位等价；opt-in 才折入方向一致性
        metadata={"help": "Phase-2: fold cross-step activation-direction consistency into the risk score used to select S."},
    )
```

- [ ] **Step 2: trainer `__init__` 读开关 + 初始化 signed EMA 状态**

在 `saart_trainer.py:135`（`self.assoc_align_lambda = fa.assoc_align_lambda` 之后）加一行：

```python
        self.saart_use_assoc_align = fa.saart_use_assoc_align  # 是否把方向一致性折入风险评分 s_j
```

在 `saart_trainer.py:145`（`self._assoc_risk: Dict[str, torch.Tensor] = {}` 之后）加一行：

```python
        self._assoc_signed: Dict[str, torch.Tensor] = {}      # name -> [C] 带符号 shift 的 EMA（方向一致性 align 用）
```

- [ ] **Step 3: 更新测试 fixture（默认参数 + 状态）**

在 `tests/test_saart_trainer.py:98`，把 `assoc_align_lambda=0.01,` 那行所在的 dict 项改为包含新开关：

```python
        assoc_align_lambda=0.01, assoc_target_layers="all", assoc_warmup_steps=0, assoc_select_every=1,
        saart_use_assoc_align=False,  # 新增：alignment 开关，默认关；align 测试按需 override
```

在 `tests/test_saart_trainer.py:119`（`t._assoc_risk = {}` 之后）加一行：

```python
    t._assoc_signed = {}
```

- [ ] **Step 4: 运行现有测试确认无回归**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`
Expected: `14/14`（或 "0 failed"），现有测试全过（基础设施改动不破坏行为）。

- [ ] **Step 5: Commit**

```bash
git add llamafactory/hparams/finetuning_args.py llamafactory/train/sft/saart_trainer.py tests/test_saart_trainer.py
git commit -m "feat(saart-align): add use_assoc_align switch + signed-EMA state (infra)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: signed EMA 更新（TDD）

`_update_assoc_risk` 并行维护带符号 EMA。

**Files:**
- Modify: `llamafactory/train/sft/saart_trainer.py:594-605`
- Test: `tests/test_saart_trainer.py`

- [ ] **Step 1: 写失败测试**

在 `tests/test_saart_trainer.py` 的 `test_assoc_select_topk` 之后加：

```python
def test_assoc_signed_ema_update():
    t = make_trainer(use_assoc_reg=True, assoc_ema_alpha=0.5)
    name = "m"
    # mag = |.|.mean(0)；signed = .mean(0)（带符号），同一 alpha EMA
    d1 = {name: torch.tensor([[2.0, -4.0]])}   # signed=[2,-4], mag=[2,4]
    d2 = {name: torch.tensor([[6.0, 0.0]])}    # signed=[6,0],  mag=[6,0]
    t._update_assoc_risk(d1)
    assert torch.allclose(t._assoc_signed[name], torch.tensor([2.0, -4.0])), "首步 signed=本步带符号均值"
    assert torch.allclose(t._assoc_risk[name], torch.tensor([2.0, 4.0])), "首步 mag=本步幅度均值"
    t._update_assoc_risk(d2)
    assert torch.allclose(t._assoc_signed[name], torch.tensor([4.0, -2.0])), "signed EMA: 0.5*[2,-4]+0.5*[6,0]=[4,-2]"
    print("PASS test_assoc_signed_ema_update")
```

并把它加进 `ALL_TESTS` 列表（在 `test_assoc_select_topk,` 之后）：

```python
    test_assoc_select_topk,
    test_assoc_signed_ema_update,
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`
Expected: FAIL — `test_assoc_signed_ema_update` 报 KeyError（`_assoc_signed[name]` 不存在，因 `_update_assoc_risk` 还没更新它）。

- [ ] **Step 3: 实现 signed EMA 更新**

把 `saart_trainer.py:593-605` 的 `_update_assoc_risk` 整个方法替换为：

```python
    @torch.no_grad()
    def _update_assoc_risk(self, deltas: Dict[str, torch.Tensor]) -> None:
        """在线 EMA 更新每通道风险统计（同一 α）：
        - mag_j = EMA[mean_resp|Δh_j|]（幅度，恒正）→ self._assoc_risk
        - signed_j = EMA[mean_resp(Δh_j)]（带符号，方向一致性 align 用）→ self._assoc_signed
        signed 始终更新（开销极小），仅在 use_assoc_align 时经 _assoc_score 参与 S 选择。
        风险高 = 该通道触发前后反复大幅激活变化 = 后门关联最可能落脚处。"""
        a = self.assoc_ema_alpha
        for name, d in deltas.items():
            cur_mag = d.detach().abs().mean(dim=0)   # [C] 本步每通道平均幅度
            cur_signed = d.detach().mean(dim=0)      # [C] 本步每通道带符号平均（方向）
            if name not in self._assoc_risk:
                self._assoc_risk[name] = cur_mag.clone()
                self._assoc_signed[name] = cur_signed.clone()
            else:
                self._assoc_risk[name].mul_(a).add_(cur_mag, alpha=1.0 - a)
                self._assoc_signed[name].mul_(a).add_(cur_signed, alpha=1.0 - a)
```

- [ ] **Step 4: 运行确认通过**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`
Expected: PASS — `test_assoc_signed_ema_update` 过，现有测试仍全过（15 total）。

- [ ] **Step 5: Commit**

```bash
git add llamafactory/train/sft/saart_trainer.py tests/test_saart_trainer.py
git commit -m "feat(saart-align): maintain per-channel signed-shift EMA in _update_assoc_risk

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: `_assoc_score` 方向一致性评分（TDD）

**Files:**
- Modify: `llamafactory/train/sft/saart_trainer.py`（在 `_update_assoc_risk` 之后加新方法）
- Test: `tests/test_saart_trainer.py`

- [ ] **Step 1: 写失败测试（4 个）**

在 `test_assoc_signed_ema_update` 之后加：

```python
def test_assoc_align_consistent_directions():
    # 所有步同符号 → signed≈mag → align≈1 → s_j ≈ mag*(1+lambda)
    t = make_trainer(use_assoc_reg=True, saart_use_assoc_align=True, assoc_align_lambda=1.0, assoc_ema_alpha=0.5)
    name = "m"
    for _ in range(6):
        t._update_assoc_risk({name: torch.tensor([[3.0, 3.0]])})  # 恒正方向一致
    score = t._assoc_score(name)
    mag = t._assoc_risk[name]
    assert torch.allclose(score, mag * 2.0, atol=1e-3), f"一致方向 align≈1 → score≈2*mag, got {score} vs {mag}"
    print("PASS test_assoc_align_consistent_directions")


def test_assoc_align_random_directions():
    # 符号交替抵消 → signed≈0 → align≈0 → s_j≈mag
    t = make_trainer(use_assoc_reg=True, saart_use_assoc_align=True, assoc_align_lambda=1.0, assoc_ema_alpha=0.5)
    name = "m"
    for v in [3.0, -3.0, 3.0, -3.0, 3.0, -3.0]:
        t._update_assoc_risk({name: torch.tensor([[v, v]])})
    align = t._assoc_signed[name].abs() / (t._assoc_risk[name] + 1e-8)
    assert (align < 0.5).all(), f"随机方向 align 应低 (<0.5), got {align}"
    print("PASS test_assoc_align_random_directions")


def test_assoc_align_lambda0_equals_magnitude():
    # 开关关时 score 必须严格=mag（向后兼容守卫），即使 signed 很大
    t = make_trainer(use_assoc_reg=True, saart_use_assoc_align=False, assoc_align_lambda=1.0)
    name = "m"
    t._assoc_risk = {name: torch.tensor([0.1, 9.0, 0.2])}
    t._assoc_signed = {name: torch.tensor([9.0, 0.1, 9.0])}
    assert torch.allclose(t._assoc_score(name), t._assoc_risk[name]), "开关关 → score 严格=mag"
    print("PASS test_assoc_align_lambda0_equals_magnitude")


def test_assoc_score_monotone():
    # 固定 mag，align 越大 score 越大；align=0 → score=mag
    t = make_trainer(use_assoc_reg=True, saart_use_assoc_align=True, assoc_align_lambda=1.0)
    name = "m"
    t._assoc_risk = {name: torch.tensor([4.0, 4.0])}
    t._assoc_signed = {name: torch.tensor([4.0, 0.0])}   # ch0 align=1, ch1 align=0
    score = t._assoc_score(name)
    assert score[0] > score[1], f"align 高的通道 score 应更大, got {score}"
    assert torch.allclose(score[1], torch.tensor(4.0), atol=1e-3), "align=0 → score=mag"
    print("PASS test_assoc_score_monotone")
```

加进 `ALL_TESTS`（`test_assoc_signed_ema_update,` 之后）：

```python
    test_assoc_signed_ema_update,
    test_assoc_align_consistent_directions,
    test_assoc_align_random_directions,
    test_assoc_align_lambda0_equals_magnitude,
    test_assoc_score_monotone,
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`
Expected: FAIL/ERROR — 4 个新测试报 `AttributeError: 'SAARTSeq2SeqTrainer' object has no attribute '_assoc_score'`。

- [ ] **Step 3: 实现 `_assoc_score`**

在 `saart_trainer.py` 的 `_update_assoc_risk` 方法之后、`_select_assoc_signature` 之前插入：

```python
    @torch.no_grad()
    def _assoc_score(self, name: str) -> torch.Tensor:
        """通道风险评分 s_j（用于选高风险集 S）。
        默认 magnitude-only：s_j = mag_j。
        开 use_assoc_align 时折入方向一致性：align_j = |signed_j| / (mag_j+eps) ∈ [0,1]
        （一致偏移→1、随机抵消→0），s_j = mag_j·(1 + lambda_align·align_j)，放大稳定朝同方向偏移
        （疑似后门关联）的通道。lambda_align=0 或开关关 → 严格退回 mag_j（向后兼容）。"""
        mag = self._assoc_risk[name]
        if (not self.saart_use_assoc_align) or self.assoc_align_lambda == 0.0 or name not in self._assoc_signed:
            return mag
        align = self._assoc_signed[name].abs() / (mag + 1e-8)  # [C] ∈ [0,1]
        return mag * (1.0 + self.assoc_align_lambda * align)
```

- [ ] **Step 4: 运行确认通过**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`
Expected: PASS — 4 个新测试过，现有全过（19 total）。

- [ ] **Step 5: Commit**

```bash
git add llamafactory/train/sft/saart_trainer.py tests/test_saart_trainer.py
git commit -m "feat(saart-align): _assoc_score folds direction-consistency into risk score

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: `_select_assoc_signature` 改用 s_j（TDD + 向后兼容回归）

**Files:**
- Modify: `llamafactory/train/sft/saart_trainer.py:607-613`
- Test: `tests/test_saart_trainer.py`

- [ ] **Step 1: 写失败测试（开 align 时 select 用 score）**

在 `test_assoc_score_monotone` 之后加：

```python
def test_assoc_select_uses_score():
    # 开 align：ch2 mag 最大但方向随机(signed≈0)，ch0 mag 中等但方向一致(signed=mag) →
    # 折入 align 后 ch0 的 s_j 反超 ch2，select 应选 ch0 而非 ch2（证明用的是 score 不是 mag）
    t = make_trainer(use_assoc_reg=True, saart_use_assoc_align=True, assoc_align_lambda=5.0, assoc_top_ratio=0.34)
    name = "m"
    t._assoc_risk = {name: torch.tensor([5.0, 1.0, 6.0])}     # mag: ch2 最大
    t._assoc_signed = {name: torch.tensor([5.0, 0.0, 0.0])}   # ch0 align=1, ch2 align=0
    # score: ch0=5*(1+5*1)=30, ch1≈1, ch2=6*(1+0)=6 → top1 = ch0
    t._select_assoc_signature()
    assert int(t._assoc_sig[name][0]) == 0, "开 align 后应选方向一致的 ch0（score 反超 mag 最大的 ch2）"
    print("PASS test_assoc_select_uses_score")
```

加进 `ALL_TESTS`（`test_assoc_score_monotone,` 之后）：

```python
    test_assoc_score_monotone,
    test_assoc_select_uses_score,
```

- [ ] **Step 2: 运行确认失败**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`
Expected: FAIL — `test_assoc_select_uses_score` 断言失败（当前 select 用 `_assoc_risk` 即 mag，会选 ch2 而非 ch0）。

- [ ] **Step 3: 改 `_select_assoc_signature` 用 score**

把 `saart_trainer.py:607-613` 的 `_select_assoc_signature` 替换为：

```python
    def _select_assoc_signature(self) -> None:
        """每 module 内按风险分 s_j 取 top assoc_top_ratio 通道为高风险集 S（top-τ% 是每 module 内部取，
        不是跨 module 取整体 top）。s_j 见 _assoc_score：默认 magnitude-only，开 use_assoc_align 时折入方向一致性。"""
        for name in self._assoc_risk:
            score = self._assoc_score(name)
            C = score.numel()
            k = max(1, int(C * self.assoc_top_ratio))  # 至少选 1 个通道
            self._assoc_sig[name] = torch.topk(score, k).indices  # [k] 选中通道索引（高风险集 S）
```

- [ ] **Step 4: 运行确认通过 + 向后兼容回归**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python tests/test_saart_trainer.py`
Expected: PASS — `test_assoc_select_uses_score` 过；**关键回归**：`test_assoc_select_topk`（开关默认关）仍过——证明开关关时 select 行为与原 magnitude-only 逐位一致。20 total 全过。

- [ ] **Step 5: Commit**

```bash
git add llamafactory/train/sft/saart_trainer.py tests/test_saart_trainer.py
git commit -m "feat(saart-align): select S by _assoc_score (back-compat when switch off)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: 接线 — 日志 + step2 字段映射 + config

无新逻辑，把开关接到配置与日志。

**Files:**
- Modify: `llamafactory/train/sft/saart_trainer.py:640,655`（`_maybe_log`）
- Modify: `step2_generate_training.py:57`（`_SAART_FIELD_MAP`）
- Modify: `configs/experiment.llama2_7b_chat.yaml`（`saart:` 块）

- [ ] **Step 1: `_maybe_log` 加 align 机制指标**

在 `saart_trainer.py:655`（`msg += f" loss_assoc={assoc_str} |S|={self._assoc_sig_size()}"` 之后）加：

```python
            # alignment 机制验证：开 use_assoc_align 时打印 S 上平均 align（期望随训练下降=免疫瓦解方向一致性）
            if self.saart_use_assoc_align and self._assoc_sig:
                aligns = []
                for nm, sel in self._assoc_sig.items():
                    if nm in self._assoc_signed and sel.numel() > 0:
                        al = self._assoc_signed[nm].abs() / (self._assoc_risk[nm] + 1e-8)
                        aligns.append(al[sel].mean())
                if aligns:
                    msg += f" align_S={float(torch.stack(aligns).mean()):.3f}"
```

- [ ] **Step 2: step2 字段映射加开关**

在 `step2_generate_training.py:57`（`"assoc_align_lambda": "assoc_align_lambda",` 那行附近）加一行：

```python
    "use_assoc_align": "saart_use_assoc_align",
```

- [ ] **Step 3: llama2 config 的 saart 块加开关**

在 `configs/experiment.llama2_7b_chat.yaml` 的 `saart:` 块里，`assoc_align_lambda: 0.01` 那行替换为：

```yaml
  assoc_align_lambda: 1.0   # 方向一致性项权重（s_j=mag*(1+λ·align)，align∈[0,1] 需 ~1 才实质影响 S）
  use_assoc_align: false    # 默认关：S 选择退回 magnitude-only，与既有 P2 结果逐位等价；sweep 时按格开启
```

- [ ] **Step 4: 验证 config 能解析 + step2 生成不报错**

Run: `/home/zengqixiu/anaconda3/envs/backdoor/bin/python step2_generate_training.py --config configs/experiment.llama2_7b_chat.yaml`
Expected: 正常生成 13 configs，无报错（`use_assoc_align` 被 _SAART_FIELD_MAP 识别）。检查生成的 saart_p2 yaml 含 `saart_use_assoc_align: false`：

Run: `grep saart_use_assoc_align /mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat/outputs/training/configs/saart_p2_immunize.yaml`
Expected: 输出 `saart_use_assoc_align: false`。

- [ ] **Step 5: Commit**

```bash
git add llamafactory/train/sft/saart_trainer.py step2_generate_training.py configs/experiment.llama2_7b_chat.yaml
git commit -m "feat(saart-align): wire use_assoc_align into log/step2/config

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: 真机冒烟 + llama2 λ_align sweep 实验

验证 align 路径在真模型上不崩，并出实验数字（遵循质量审计协议：ASR 必须与 degen% 同看）。

**Files:**
- Create: `scripts/run_assoc_align_sweep.sh`

- [ ] **Step 1: 写 sweep 脚本**

创建 `scripts/run_assoc_align_sweep.sh`：

```bash
#!/bin/bash
# 位置×KL 之外的第二个 SAART sweep：alignment 项 λ_align ∈ {0,0.5,1,2}（llama2 P2 设定）。
# 从 saart_p2_immunize.yaml 派生（改 saart_use_assoc_align/saart_assoc_align_lambda + 输出目录）。
# 两阶段：bf16 训练(≥24GB 卡) → fp32 评测(≥34GB 卡)。tag=after_saart_p2_align_<lam>。
set -uo pipefail
cd /home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine
export HF_HOME=/mnt/data/zengqixiu/hf_cache WANDB_DISABLED=true TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/zengqixiu/anaconda3/envs/backdoor/bin/python
BASE=/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat
SRC=$BASE/outputs/training/configs/saart_p2_immunize.yaml
SWEEP=$BASE/outputs/purified/saart_p2_align; LOGD=$BASE/outputs/logs/align_sweep
CFG=configs/experiment.llama2_7b_chat.yaml
mkdir -p "$SWEEP" "$LOGD"
LAMS=(0 0.5 1.0 2.0)
pick_gpu(){ local need=$1 g free; while true; do for g in 5 6 7 3 1 0 8 9 2 4; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$g" 2>/dev/null|tr -d ' ')
  [ "${free:-0}" -ge "$need" ] && { echo "$g"; return; }; done; sleep 20; done; }
# 阶段A 训练
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"; yaml="$BASE/outputs/training/configs/saart_p2_align_$name.yaml"
  [ -f "$out/adapter_model.safetensors" ] && { echo "[skip-train] $name"; continue; }
  $PY - "$SRC" "$yaml" "$lam" "$out" <<'EOF'
import sys, yaml
src,dst,lam,out=sys.argv[1:5]
c=yaml.safe_load(open(src)); c['saart_use_assoc_align']=True; c['saart_assoc_align_lambda']=float(lam); c['output_dir']=out
yaml.safe_dump(c,open(dst,'w'),sort_keys=False); print(f'[gen] {dst} lam={lam}')
EOF
  g=$(pick_gpu 24000); port=$(((RANDOM%40000)+20000)); echo "[train] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY -m torch.distributed.run --nproc_per_node=1 --master_port=$port \
      saart_train.py "$yaml" > "$LOGD/${name}_train.log" 2>&1 ) & sleep 25
done
wait; echo "[align-sweep] 训练完成 $(date +%T)"
# 阶段B 评测
for lam in "${LAMS[@]}"; do
  name="lam${lam//./p}"; out="$SWEEP/$name"
  [ -f "$out/adapter_model.safetensors" ] || { echo "[skip-eval] $name 无 adapter"; continue; }
  grep -q "after_saart_p2_align_$name\"" "$BASE/outputs/eval/results.jsonl" 2>/dev/null && { echo "[skip-eval] $name"; continue; }
  g=$(pick_gpu 34000); echo "[eval] $name -> GPU $g"
  ( CUDA_VISIBLE_DEVICES=$g $PY step5_evaluate.py --config "$CFG" --adapter "$out" \
      --eval_type both --tag "after_saart_p2_align_$name" > "$LOGD/${name}_eval.log" 2>&1 ) & sleep 25
done
wait; echo "########## ALIGN SWEEP DONE $(date) ##########"
```

注意：派生脚本里写的是 `saart_assoc_align_lambda`/`saart_use_assoc_align`——确认 finetuning_args 实际字段名是否带 `saart_` 前缀。`saart_use_assoc_align` 带前缀（Step Task1 定义），但 `assoc_align_lambda` **不带** `saart_` 前缀（见现有字段）。

- [ ] **Step 2: 修正 sweep 脚本字段名**

把脚本里 `c['saart_assoc_align_lambda']` 改为 `c['assoc_align_lambda']`（该字段无 saart_ 前缀，与 finetuning_args 一致）。

- [ ] **Step 3: 语法检查 + 冒烟（λ=1 单格先跑通）**

Run: `bash -n scripts/run_assoc_align_sweep.sh && chmod +x scripts/run_assoc_align_sweep.sh && echo SYNTAX_OK`
Expected: `SYNTAX_OK`

冒烟（只跑 λ=1 一格，确认真机不崩、日志出现 align_S）：

Run: `LAMS_SMOKE=1.0 bash -c 'source scripts/run_assoc_align_sweep.sh' 2>&1 | head` —— 或直接 `bash scripts/run_assoc_align_sweep.sh`（全跑）。
Expected: 训练日志 `align_sweep/lam1p0_train.log` 出现 `[SAART step N] ... align_S=0.xxx`，adapter 落地。

- [ ] **Step 4: 全 sweep + 汇总（ASR + degen 同看）**

Run: `bash scripts/run_assoc_align_sweep.sh`（后台，多卡），完成后：

```bash
/home/zengqixiu/anaconda3/envs/backdoor/bin/python -c "
import json, os
from analyze_output_quality import analyze_tag
e='/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat/outputs/eval'
rows={r['tag']:r for r in map(json.loads,open(e+'/results.jsonl'))}
for n in ['lam0','lam0p5','lam1p0','lam2p0']:
    t=f'after_saart_p2_align_{n}'
    if t in rows:
        p=e+f'/{t}_clean_detail.json'; dg=analyze_tag(p)['degen_rate'] if os.path.exists(p) else -1
        print(f'{n:8s} ASR={rows[t][\"trigger_asr\"]:5} cdegen={dg:.1f}')
print('baseline P2:', rows.get('after_saart_p2',{}).get('trigger_asr'))
"
```
Expected: 各 λ_align 的 ASR + degen。判读：alignment 有效 = ASR 较 P2(5.0) 降且 degen 不升（遵循质量审计协议）；α无效则 ASR 持平。

- [ ] **Step 5: Commit（脚本 + 结果记录到报告）**

```bash
git add scripts/run_assoc_align_sweep.sh
git commit -m "feat(saart-align): llama2 lambda_align sweep (two-phase multi-GPU, ASR+degen)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

把 sweep 结果（ASR + degen + align_S 机制指标）追加进 `reports/SAART_P1_对比实验报告_20260605.md` 的 Phase-2 段或新建小节，单独 commit。

---

## Self-Review

**Spec coverage:**
- §3.1 状态 `_assoc_signed` → Task 1 Step 2 ✓
- §3.2 `_update_assoc_risk` signed EMA → Task 2 ✓
- §3.3 `_assoc_score` → Task 3 ✓
- §3.4 `_select_assoc_signature` 改用 score → Task 4 ✓
- §3.5 机制验证日志 → Task 5 Step 1 ✓
- §3.6 参数 + 默认 1.0 → Task 1 Step 1；step2/config → Task 5 ✓
- §4 向后兼容 → Task 1（开关默认 False）+ Task 4 Step 4 回归守卫 ✓
- §5 五个测试 → Task 2（signed_ema）+ Task 3（consistent/random/lambda0/monotone）= 5 个 + Task 4 多一个 select_uses_score ✓（实际 6 个，覆盖更全）
- §6 实验 → Task 6 ✓

**Placeholder scan:** 无 TBD/TODO；所有代码步给了完整代码与精确命令。Task 6 的 `LAMS_SMOKE` 是可选冒烟提示，Step 2 已明确字段名修正。

**Type consistency:** `_assoc_signed`（Dict[str,Tensor]）贯穿 Task 1/2/3/5 一致；`saart_use_assoc_align`（带 saart_ 前缀，开关）与 `assoc_align_lambda`（无前缀，权重）的命名差异在 Task 6 Step 2 显式纠正；`_assoc_score` 签名 `(self, name) -> Tensor` 在 Task 3 定义、Task 4/5 调用一致。

**已知约定提醒**：测试 runner 是 `python tests/test_saart_trainer.py`（非 pytest），新测试必须加进 `ALL_TESTS` 列表才会被跑（Task 2/3/4 各 Step 1 已含）。
