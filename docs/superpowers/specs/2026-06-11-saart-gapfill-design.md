# SAART 缺口补全 — 退化感知(W1) + 方向感知关联正则(W2) + 行为对抗者(W3)

日期：2026-06-11 · 状态：已实现（默认全关，CPU 单测 28/28 通过），待 GPU 验证
关联：`自对抗关联鲁棒训练框架_研究重构建议.md`（Module 1/2/3）、`reports/方法论_后门防御评估的三类虚假胜利_20260610.md`、
`reports/output_quality_audit_20260610.md`、`llamafactory/train/sft/saart_trainer.py`

## 1. 背景：为什么补这三处

上一轮分析 + 团队自审计确认 SAART 的三处缺口正好是文档里区别于「普通对抗训练 / CROW / BadLLM-TG」的关键，
也是落地最弱的三处：

- **headline 注水**：低 ASR(9/5/4.5%) 大量由 clean 侧 39–62% 输出退化(loop/empty)换来；关键词 judge 盲视。
- **Module 2 的方向项失败**：`L_assoc-reg` 只压幅度(Δh²)，对方向缩放不变，无法瓦解方向一致性（align_S 不降反升，M2A.2 负面结果）。
- **Module 1 缺 behavior**：内层「行为无关」KL，回应不了「这不就是 CROW/BadLLM-TG」。

三个工作流（全部默认关，开关 + 字段 + 校验 + CPU 单测 + sweep 脚本齐全；关时与历史 P1/P2/RQ6 逐位等价）：

## 2. W1 退化感知

### W1a 训练侧 L_utility（Module 3，KL-to-base）
- 字段：`saart_lambda4`(默认 0)、`saart_utility_type`(none|kl_to_base)。
- `_utility_loss(unwrapped, ids, attn, labels, theta_logits)`：`L_utility = KL(p_base ‖ p_theta)` 在 clean response 区，
  `p_base` = 同一模型 `unwrapped.disable_adapter()`(关 LoRA) 的 no_grad 前向（零额外权重/显存），`p_theta` 带梯度。
  复用 `_select_response_logits` + `_kl`。在 `compute_loss` 紧跟 clean forward 计算（capture=None 守卫，base 前向不被 assoc hook 误采），
  并入主 total 与 early-return。机制：把 clean 分布锚回 base 流畅分布 → 抑制坍缩，免去手工诚实点。
- 验收：util_kl 的 clean degen 回到 ≈ 基线(8%) 且 ASR 不显著回升。

### W1b 评测侧 Gate B（degen% 入账）
- 新 leaf 模块 `antigen/degen.py`(无 torch)：`degeneration_signals`/`distinct_n`(从 analyze_output_quality 迁入) + `degen_rate`。
  `analyze_output_quality.py` 改为从此 import（单一真相源）。
- `step5_evaluate.py:run_eval` 对每次 trigger/clean 的逐样本 `results` 算 degen，写入 `record`：
  `trigger_degen / clean_degen / *_degen_reasons / *_distinct2`（ledger append 前，零重推理）。
- `print_summary` 加 `Degen c/t` 列，旧行缺字段渲染 `—`（后向兼容）。

## 3. W2 方向感知关联正则
- 字段：`assoc_reg_type`(magnitude|direction|hybrid，默认 magnitude=逐位等价旧行为)、`assoc_dir_weight`(hybrid 用)。
- `_assoc_reg_loss` 分支：复用在线带符号 EMA `_assoc_signed` 的 `sign(signed_j)` 作每通道共识后门方向。
  - direction：`mean(relu(Δh_j·sign(signed_j))²)` — 只罚「沿共识方向」的偏移，留反向自由度 → 应驱动 align_S↓（magnitude 做不到）。
  - hybrid：`magnitude + assoc_dir_weight·direction`。
- 验收：direction 使训练日志 `align_S` **下降**（不再像 magnitude 那样上升）且 ASR 不升；否则按诚实标准记负面。

## 4. W3 行为对抗者（Module 1 的 b）
- 字段：`saart_behavior_adversary`(默认 false)、`saart_behavior_probes`(JSON 短串列表路径)、`saart_lambda_b`(默认 0)。
- `__init__` 一次性 tokenize 探针为 `_behavior_probe_ids`；`data/saart_behavior_probes.json` 对齐 negsenti behavior。
- `_behavior_logprob(...)`：teacher-forced `logp(b | x_prompt ⊕ t)` —— 把 soft trigger 插进 **prompt**、追加 b、左 pad 对齐，
  取 b 位均值 LL（differentiable in soft）。`_inner_search` 的 proxy 变 `KL_shift + λ_b·max_b logp(b|x⊕t)`，
  max_b 选当前最易诱发的行为方向（文档「most vulnerable behavior direction」）。外层免疫结构不变（t* 仍训 clean response）。
- 成本：每 inner step 多 |probes| 次前向 → 探针 ≤4。验收：对 unseen 行为/触发 ASR 是否比行为无关更低；否则记负面。

## 5. 接线与默认关
- `SAARTArguments` 加 7 字段（finetuning_args.py，含 `__post_init__` 校验：λ4>0 须配 utility_type；behavior_adversary 须给非空 probes；各权重≥0）。
- `step2_generate_training.py:_SAART_FIELD_MAP` 加 7 短键映射；`saart_ablations` 加 `util_kl` / `behav_adv`（经 step4c_saart_ablation 训，tag after_saart_p1_<name>）。
- `configs/experiment.yaml` + `configs/experiment.llama2_7b_chat.yaml` 的 saart 块加注释默认值（其余 11 预设缺键→dataclass no-op 默认，需要时再加）。
- sweep：`scripts/run_saart_utility_sweep.sh`(λ4)、`run_saart_assocdir_sweep.sh`(P2 底座，assoc_reg_type)、`run_saart_behav_sweep.sh`(λ_b)。
- 单测：`tests/test_saart_trainer.py` +8（util on/off、direction/magnitude/hybrid、behavior logprob+grad、inner 隔离、degen_rate）；ToyLM 加 `disable_adapter` 桩。**28/28 全过。**

## 6. GPU 验证 runbook
```bash
python step2_generate_training.py --config configs/experiment.yaml   # 重生成（含 util_kl/behav_adv ablation）
bash scripts/step4c_saart_ablation.sh                                # 训 util_kl + behav_adv（及原 5 个）
bash scripts/run_saart_assocdir_sweep.sh                             # W2：magnitude(对照)/direction/hybrid
bash scripts/step5_evaluate.sh                                       # 评测；ledger/summary 现带 degen
python step5_evaluate.py --config configs/experiment.yaml --summary_only   # 看 Degen c/t 列
python analyze_output_quality.py                                     # 交叉核对 degen
python tests/test_saart_trainer.py                                   # 28 CPU 单测
```

## 7. 诚实边界
W2/W3 可能仍不降 ASR（W2 的 magnitude 版已是负面）。本补全的价值是让机制**可证伪**（align_S↓、behavior-targeted t*、
degen 入账）并按项目诚实标准如实记录，而非保证增益。L_utility 的 degen 早停/选 ckpt（需训练中生成）留作后续。
