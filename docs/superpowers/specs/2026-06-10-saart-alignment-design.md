# SAART Module 2 — alignment（方向一致性）项设计

日期：2026-06-10 · 状态：已批准（设计），待实现
关联：`自对抗关联鲁棒训练框架_研究重构建议.md`（模块2 的 s_j 公式）、`llamafactory/train/sft/saart_trainer.py`（当前 magnitude-only 实现 + TODO:597）

## 1. 背景与动机

SAART Phase-2（Module 2）在训练期维护一个**在线 MLP 关联签名**：对每个 MLP 通道 j 算风险分 `s_j`，
取每 module 内 top-τ% 高风险通道集 S，对 S 上的触发前后激活差 Δh 施加关联正则 `L_assoc-reg`（迫使触发前后激活一致，
切断 trigger→behavior 绑定）。

文档模块2 给的风险分是两项：
```
s_j = E_i‖Δh_{i,j}‖  +  λ·E_{i,k}[max(0, cos(Δh_{i,j}, Δh_{k,j}))]
       └── 幅度（已实现）        └── 跨变体方向一致性（未实现，本设计补全）
```
当前 `saart_trainer.py` 只实现了第一项（`_assoc_risk` = `mean_resp|Δh_j|` 的 EMA）。第二项（alignment）在
`_update_assoc_risk`（:597）标了 `TODO(SAART-P2)`，参数 `assoc_align_lambda` 已 wire 但未参与计算。

**为什么要补**：alignment 的语义是「一个通道若在不同触发下都朝同一方向偏移，它就是稳定的后门关联通道；
若偏移方向随机，只是噪声」。补上它能让 S 选得更准（剔除高幅度但方向随机的噪声通道），从而 `L_assoc-reg`
切除更精准——目标是在 llama2 主设定上把训练期免疫从诚实点（after_bos λ2≤0.25 的 ~11% ASR）进一步推低，
逼近后处理 BD-VAX 的 3.0%。也提供文档要求的机制验证（「训练后高风险通道方向一致性下降」）。

## 2. 在线诠释（关键设计决策，已选定）

文档的 `cos(Δh_{i,j}, Δh_{k,j})` 是 post-hoc 跨多个 poisoned-clean model pair（变体 i,k）。SAART 在线设定每步
只有**本步一个对抗触发**，需重诠释为**跨 step 的方向一致性**：维护每通道带符号 shift 的 EMA，用「带符号 EMA 的幅度
/ 幅度 EMA」作为方向一致性度量。

定义（每 module、每通道 j）：
- `mag_j  = EMA_t[ mean_resp |Δh_t[:,j]| ]`        ← 现有 `_assoc_risk`（幅度，恒正）
- `signed_j = EMA_t[ mean_resp(Δh_t[:,j]) ]`       ← **新增**带符号 shift 的 EMA（同一 α）
- `align_j = |signed_j| / (mag_j + ε)` ∈ [0,1]      ← 方向一致性：恒同向→1（|EMA(signed)|≈EMA|·|）、随机抵消→0
- `s_j = mag_j · (1 + λ_align · align_j)`            ← 方向一致的通道风险被放大

直觉：`signed_j` 是带符号平均的运行平均。若通道 j 每步都朝同方向偏，正负不抵消，`|signed_j| ≈ mag_j` → align≈1；
若方向随机，`signed_j` 趋于 0 → align≈0。这无需存变长的 response-token 向量，只多一个 `[C]` 标量 EMA per module，
数值稳定、显存开销小（与现有 `_assoc_risk` 同形状）。

**聚合层次（避免歧义）**：`mag_j` 与 `signed_j` 都是「step 内先 across response tokens 聚合成 per-channel 标量
（`abs().mean(dim=0)` vs `mean(dim=0)`），再跨 step EMA」。因此 `align_j` 度量的是**step 内 response 区 + 跨 step**
的双重方向一致性——后门通道应在整个 response 区稳定朝一个方向偏移；若 step 内不同 token 方向就已随机，
`signed_j` 在 step 内即被抵消、align 偏低。这比纯跨 step 一致性更严格，符合「稳定 trigger→behavior 绑定」的语义。

## 3. 实现要点

所有改动在 `llamafactory/train/sft/saart_trainer.py`（+ `finetuning_args.py` 参数 + config + 测试），不碰 step5。

### 3.1 状态
- 新增 `self._assoc_signed: Dict[str, Tensor]`（name→[C]），与 `self._assoc_risk` 并列初始化（`{}`）。

### 3.2 `_update_assoc_risk`（扩展，仍 @torch.no_grad）
- 现有：`cur_mag = d.detach().abs().mean(dim=0)`；EMA 进 `_assoc_risk`。
- 新增：`cur_signed = d.detach().mean(dim=0)`（带符号）；同一 α EMA 进 `_assoc_signed`。
- 删去 TODO:597。两个 EMA 用同一 `assoc_ema_alpha`、同一初始化分支（首次 clone，否则 mul_/add_）。

### 3.3 `_assoc_score(name)`（新方法，@torch.no_grad）
```
mag = self._assoc_risk[name]
if not self.saart_use_assoc_align or self.assoc_align_lambda == 0 or name not in self._assoc_signed:
    return mag                                  # 严格退回 magnitude-only（向后兼容）
align = self._assoc_signed[name].abs() / (mag + 1e-8)
return mag * (1.0 + self.assoc_align_lambda * align)
```

### 3.4 `_select_assoc_signature`（改 1 行）
- topk 的 `risk` 改成 `self._assoc_score(name)`；其余（每 module 取 top-τ%）不变。

### 3.5 机制验证（`_maybe_log` 扩展）
- 开 alignment 时，额外打印 S 上的平均 align（`mean_{j∈S} align_j`）。期望随训练**下降**（免疫使方向一致性瓦解）。
- 不改 `_assoc_reg_loss`（损失公式不变，只是 S 的成员变了）。

### 3.6 参数（`finetuning_args.py`）
- 新增 `saart_use_assoc_align: bool = False`（默认关，保向后兼容）。
- `assoc_align_lambda` 默认 `0.01 → 1.0`（语义变了：旧默认在新公式里几乎无影响；align∈[0,1] 需 λ~1 才实质改变 S）。
- `step2_generate_training.py` 的 `_SAART_FIELD_MAP` 加 `use_assoc_align → saart_use_assoc_align`、
  `assoc_align_lambda` 已在映射中则确认；config `saart:` 块加 `use_assoc_align: false` 注释默认。

## 4. 向后兼容（硬约束）

`saart_use_assoc_align=False`（默认）时，`_assoc_score` 严格返回 `mag`，S 选择与当前 magnitude-only **逐位等价**——
已有 RQ6/P2 结果（24 cell）完全不受扰动。`_assoc_signed` EMA 始终更新（开销极小），仅在开关打开时参与评分。

## 5. 测试（`tests/test_saart_trainer.py`，TDD 先写）

1. `test_assoc_align_consistent_directions`：合成所有 step 同符号的 Δh → align_j≈1。
2. `test_assoc_align_random_directions`：合成符号随机的 Δh → align_j≈0（< 小阈值）。
3. `test_assoc_align_lambda0_equals_magnitude`：`λ_align=0` 或开关关 → S 与 magnitude-only 选择完全一致（向后兼容守卫）。
4. `test_assoc_signed_ema_update`：signed EMA 按 `α·old+(1-α)·cur` 正确更新，与 mag EMA 同步。
5. `test_assoc_score_monotone`：固定 mag，align 越大 s_j 越大（放大方向一致通道）。

现有 14 个单测必须仍全过（向后兼容）。

## 6. 实验验证（llama2 主设定，BadNets×negsentiment）

- 在最佳 P2 设定（after_bos + assoc-reg）上开 `use_assoc_align`，`λ_align ∈ {0, 0.5, 1.0, 2.0}` sweep。
- 主指标：Trigger ASR（期望从 P2 的 5.0% 进一步降，逼近 BD-VAX 3.0%）+ **clean/trigger degen%**（必须不升——遵循质量审计协议，
  否则又是退化换 ASR）。
- 机制指标：训练日志里 S 上的平均 align 是否随训练下降。
- 多卡：用 `feedback_fill_gpu` 的依赖感知池模式，一卡一 λ 并行（4 条件 4 卡）。

## 7. 文件改动清单

| 文件 | 改动 |
|---|---|
| `llamafactory/train/sft/saart_trainer.py` | `_assoc_signed` 状态、`_update_assoc_risk` 加 signed EMA、新 `_assoc_score`、`_select_assoc_signature` 改用 score、`_maybe_log` 加 align、删 TODO:597 |
| `llamafactory/hparams/finetuning_args.py` | 新增 `saart_use_assoc_align`、`assoc_align_lambda` 默认 1.0 |
| `step2_generate_training.py` | `_SAART_FIELD_MAP` 加 `use_assoc_align` |
| `configs/experiment*.yaml` | `saart:` 块加 `use_assoc_align: false` + 注释 |
| `tests/test_saart_trainer.py` | +5 alignment 单测 |
| 实验脚本 | llama2 上 λ_align sweep（仿 `run_pos_kl_sweep.sh` 的两阶段多卡池） |

## 8. 非目标（YAGNI）

- 不做损失项版本的 alignment（文档 alignment 在 s_j 评分，不在损失；保持损失公式不变）。
- 不改 magnitude 项、不改 `L_assoc-reg` 公式、不改 hook/采集逻辑。
- 不在本设计里跨模型推广（先 llama2 验证有效再说）。
