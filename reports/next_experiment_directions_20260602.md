# Backdoor-Vaccine 下一步实验方向与具体方案

生成日期：2026-06-02
报告版本：v1
仓库：`/home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine`  
依据：当前 `my-work @ f7630f7`、侧分支 `feat/activation-weighted-selection @ e861806`、9 后端 evaluation ledger、以及 `reports/commit_experiment_process_report_20260602.md`。

## 0. 总体判断

当前最重要的实验事实是：

- Backdoor Antigen 的 `suppression + FT` 在多数后端有效，但 `llama3_1_8b_instruct` 明显失效。
- 侧分支已经验证 activation weighting 只能小幅改善 Llama-3.1，不能接近 Wanda / Fine-pruning。
- suppress-ratio 扫描说明“增大通道级抑制比例”会破坏 clean 输出，低 ASR 可能是假象。
- 当前 Step 5 主要看 `trigger_asr` 与 `clean_fp` 关键词指标，缺少输出质量指标，已经不足以支撑后续结论。

因此下一步实验不应直接堆更多模型。优先级应是：

1. **先修评测可信度**：让结果能区分“清除后门”和“破坏模型”。
2. **再攻 Llama-3.1 的粒度问题**：从通道级抑制推进到权重级 / 子通道级抑制。
3. **再做攻击类型外推**：用已有 CTBA / MTBA / Sleeper / VPI 数据验证结论是否只属于 BadNets。
4. **最后做自动化与论文表格**：把 9 模型矩阵、质量指标、消融结果稳定生成报告。

## 1. 优先级总表

| 优先级 | 实验方向 | 主要问题 | 预期产出 |
|---|---|---|---|
| P0 | 输出质量评测补强 | clean-FP 关键词指标漏掉乱码/退化 | 新增质量 metrics，重评 9 模型，给所有 ASR 加效用约束 |
| P0 | Llama-3.1 权重级 Antigen 抑制 | 通道级置零粒度过粗 | 新方法 `weight-level antigen suppression`，对比 Wanda |
| P1 | 合并 activation-weighted 诊断分支 | 侧分支结果未进入主线 | 默认关闭的诊断能力、可复现实验配置 |
| P1 | 攻击类型扩展到 CTBA/MTBA/Sleeper/VPI | 当前仅 BadNets x negsentiment | 跨 trigger family 的鲁棒性表 |
| P1 | 关键超参消融 | λ、variant 数、ratio、target module 未系统扫 | 方法稳定性曲线与推荐默认值 |
| P2 | Gemma Wanda / Llama3-Chinese B3b 补齐 | baseline 表缺格 | 完整 9 模型 baseline 矩阵 |
| P2 | 报告与 ledger 自动化 | 现在报告脚本默认只跑两个后端 | 一键 9 模型报告与质量约束报告 |

## 2. P0 实验一：输出质量评测补强

### 目标

证明防御不是通过破坏模型来降低 ASR。尤其要修正 Llama-3.1 ratio sweep 中出现的情况：`clean_fp=0`，但 clean 输出大量退化为 `!!!!!!!!` 乱码。

### 方案

在 `step5_evaluate.py` 现有 trigger/clean detail 的基础上新增输出质量 metrics。最小可行版本不需要新模型 judge，先用规则指标：

- `empty_rate`：空输出比例。
- `degenerate_prefix_rate`：以重复标点、重复同一字符、乱码前缀开始的比例。
- `repeat_ngram_rate`：3-gram 或 4-gram 重复率过高的样本比例。
- `avg_output_chars` / `median_output_chars`：输出长度统计。
- `short_output_rate`：长度小于阈值的比例，例如 `< 8 chars`。
- `nonalpha_symbol_rate`：输出中非字母数字符号占比异常高的样本比例。

建议新增文件：

- `scripts/analyze_eval_quality.py`：读取 `<tag>_trigger_detail.json` 和 `<tag>_clean_detail.json`，输出质量指标 JSON/Markdown。
- 可选后续再把 metrics 写回 `step5_evaluate.py` 的 ledger。

### 实现步骤

1. 写 `scripts/analyze_eval_quality.py`：
   - 输入：`--eval_dir`、`--tag`、`--split trigger|clean|both`
   - 输出：`outputs/eval/<tag>_<split>_quality.json`
   - 追加汇总：`outputs/eval/quality_summary.md`
2. 对 9 后端所有已有 detail 直接离线分析，不需要重新推理。
3. 在 `make_report.py` 增加质量列，至少展示 `clean_degenerate_rate`。

### 建议命令

```bash
python scripts/analyze_eval_quality.py \
  --eval_dir /mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama3_1_8b_instruct/outputs/eval \
  --tag after_finetune \
  --split both

for tag in qwen2_5_7b_instruct qwen2_7b_instruct llama2_7b_chat vicuna_7b_v1_5 \
           mistral_7b_instruct_v0_3 gemma_2_9b_it meta_llama_3_8b_instruct \
           llama3_chinese_8b_instruct llama3_1_8b_instruct; do
  python scripts/analyze_eval_quality.py \
    --eval_dir /mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/$tag/outputs/eval \
    --split both
done
```

### 成功标准

- 每个 defense tag 同时有 ASR、clean-FP、clean 输出退化率。
- 0.50/0.70 suppress-ratio 这类“ASR 很低但输出坏掉”的实验能被质量指标明确标红。
- 后续所有主结果默认只比较 `clean_degenerate_rate <= 5%` 的方法。

## 3. P0 实验二：Llama-3.1 权重级 Antigen 抑制

### 目标

验证并尝试解决 Llama-3.1 的核心瓶颈：后门和效用在 LoRA 通道级纠缠，整行/整列置零过粗。下一步应把抑制粒度从“通道”改成“通道内权重”或“低秩因子元素”。

### 方案 A：LoRA 因子元素级 mask

当前 `antigen/suppression.py` 对 LoRA 的处理是：

- `gate_proj/up_proj`：对 `lora_B` 整行置零。
- `down_proj`：对 `lora_A` 整列置零。

新方案改为只抑制被选中通道内的 top-k 元素：

- 对 `gate_proj/up_proj` 的选中 `out_j`，只在 `lora_B[j, :]` 中按分数置零 top `element_ratio`。
- 对 `down_proj` 的选中 `in_j`，只在 `lora_A[:, j]` 中按分数置零 top `element_ratio`。
- 元素分数先用 suspicious adapter 自身的绝对值，随后再尝试 differential delta 投影分数。

建议新增：

- `antigen/weight_suppression.py`
- `step4_weight_purify.py`
- config 字段：
  - `suppression_granularity: channel|element`
  - `element_suppress_ratio: 0.35`
  - `element_score: suspicious_abs|delta_abs|activation_weighted_delta`

### 方案 B：effective ΔW 权重级 mask

LoRA effective delta 为 `ΔW = (alpha/r) * B @ A`。可以在 effective weight 上选权重，再反投影到 LoRA 因子比较困难，因此建议先做 merged model 版本：

1. merge base + suspicious LoRA 得到 full model。
2. 计算 variant differential effective deltas 的 per-weight score。
3. 在 MLP 权重矩阵内选 top weight mask。
4. 对 merged model 对应权重做 zero / shrink，而不是整通道置零。
5. 对修改后的 full model 做 clean LoRA FT。

该方案更接近 Wanda，可直接验证“权重级粒度是否救回 Llama-3.1”。

### 最小实验矩阵

先只跑 `llama3_1_8b_instruct`：

| 方法 | granularity | element/weight ratio | post-FT | 目的 |
|---|---|---:|---|---|
| old ours | channel | 0.35 | yes | 当前失败基线 |
| element-abs | LoRA factor element | 0.35 | yes | 验证细粒度是否减少效用破坏 |
| element-abs | LoRA factor element | 0.50 | yes | 看预算增加是否仍稳定 |
| effective-weight | merged weight | 0.35 | no/yes | 和 Wanda 粒度对齐 |
| Wanda | full weight | 0.35 | no/yes | 参考上界 |

### 建议命令形态

```bash
cp configs/experiment.llama3_1_8b_instruct.yaml \
  configs/experiment.llama3_1_8b_instruct_element.yaml

# 在新配置里设置：
# suppression_granularity: "element"
# element_suppress_ratio: 0.35
# purified_dir: "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama3_1_8b_instruct_element/outputs/purified"

CONFIG=configs/experiment.llama3_1_8b_instruct_element.yaml \
CFG_FILE=configs/experiment.llama3_1_8b_instruct_element.yaml \
bash scripts/step4_weight_purify.sh

CONFIG=configs/experiment.llama3_1_8b_instruct_element.yaml \
CFG_FILE=configs/experiment.llama3_1_8b_instruct_element.yaml \
bash scripts/step4b_finetune.sh

python step5_evaluate.py \
  --config configs/experiment.llama3_1_8b_instruct_element.yaml \
  --tag after_element_suppression \
  --adapter /mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama3_1_8b_instruct_element/outputs/purified/element_suppressed_adapter \
  --eval_type both
```

### 成功标准

- Llama-3.1 `ours+FT` 从 69.5% 降到至少 40% 以下。
- clean 退化率不超过 5%。
- 若接近 Wanda 27.5% 或 Fine-pruning 10.0%，说明粒度修复有效。
- 若仍无效，说明问题可能不只是粒度，而是 antigen residual 和真实触发功能路径解耦。

## 4. P1 实验三：合并 activation-weighted 诊断能力

### 目标

把 `feat/activation-weighted-selection` 的诊断代码纳入主线，但保持默认关闭，作为后续机制分析工具。

### 方案

合并或 cherry-pick 以下能力：

- `antigen/activation.py`
- `step3_extract_signature.py --deltas_dir`
- `BD_VAX_SELECTION_MODE=activation_weighted`
- `activation_gamma`
- `activation_nsamples`

但不要把 activation-weighted 设为默认方法。当前证据显示它只能把 Llama-3.1 final ASR 从 69.5 降到 63.5，诊断价值大于方法价值。

### 建议实验

对 3 个代表模型跑 gamma sweep：

- 成功模型：`qwen2_5_7b_instruct`
- 成功模型：`llama2_7b_chat`
- 失败模型：`llama3_1_8b_instruct`

gamma：

- `0.0`：残差原始方法
- `0.5`
- `1.0`
- `2.0`

### 成功标准

- activation importance 缓存后可复用，避免每次重算。
- 每次输出 signature Jaccard、ASR、clean-FP、质量指标。
- 文档中明确 activation weighting 不是最终修复，只是机制诊断工具。

## 5. P1 实验四：扩展攻击类型 CTBA / MTBA / Sleeper / VPI

### 目标

验证当前结论是否只适用于 BadNets。仓库已经有 negsentiment 下 5 种 trigger family 的训练/测试数据：

- `badnet`
- `ctba`
- `mtba`
- `sleeper`
- `vpi`

### 方案

先不要 9 模型全量展开，成本太高。建议三模型试点：

- `qwen2_5_7b_instruct`：当前强成功模型。
- `llama2_7b_chat`：论文默认模型。
- `llama3_1_8b_instruct`：当前失败模型。

对每个 trigger family 分别训练 suspicious adapter，再复用完整 Step 1-Step 5。

### 配置改法

为每个 backend/trigger 新建配置：

```text
configs/experiment.<model_tag>.<trigger>.yaml
configs/negsentiment/<model_tag>/negsenti_<trigger>_lora.yaml
```

关键字段：

```yaml
model_tag: "qwen2_5_7b_instruct_ctba"
test_trigger_data: "data/test_data/poison/negsentiment/ctba/backdoor200_negsentiment_ctba.json"
suspicious_adapter: "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_5_7b_instruct_ctba/backdoor_weight/negsentiment/ctba"
training_dir:  "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_5_7b_instruct_ctba/outputs/training"
signature_dir: "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_5_7b_instruct_ctba/outputs/signature"
purified_dir:  "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_5_7b_instruct_ctba/outputs/purified"
eval_dir:      "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_5_7b_instruct_ctba/outputs/eval"
```

### 最小矩阵

| 模型 | badnet | ctba | mtba | sleeper | vpi |
|---|---:|---:|---:|---:|---:|
| Qwen2.5 | 已有 | 待跑 | 待跑 | 待跑 | 待跑 |
| Llama2 | 已有 | 待跑 | 待跑 | 待跑 | 待跑 |
| Llama3.1 | 已有 | 待跑 | 待跑 | 待跑 | 待跑 |

### 成功标准

- 每个 trigger family 至少拿到 no-defense、ours+FT、random、pure-FT 四个 tag。
- 若 Llama-3.1 在所有 trigger family 都失败，说明 checkpoint 粒度问题更普遍。
- 若只在 BadNets 失败，需要分析 trigger family 与通道纠缠的关系。

## 6. P1 实验五：关键超参消融

### 目标

明确当前默认值是否稳健，避免评审质疑“参数碰巧调对”。

### 推荐消融

| 参数 | 当前值 | 建议扫描 | 低成本做法 |
|---|---:|---|---|
| `lora_suppress_ratio` | 0.35 | 0.10, 0.20, 0.35, 0.50 | 复用 deltas 和 variant adapters，只重做 Step 3/4/4b/5 |
| `lambda_` | 0.01 | 0, 0.001, 0.01, 0.1 | 复用 deltas，只重算 scoring |
| variants N | 6 | 2, 4, 6 | 用已有前 N 个 variants |
| target modules | gate/up/down | gate+up, down-only, all-MLP | 复用 deltas |
| post-FT lr | 5e-5 | 1e-5, 5e-5, 1e-4 | 只重跑 Step 4b/5 |

### 优先模型

- `qwen2_5_7b_instruct`：代表成功模型。
- `llama3_1_8b_instruct`：代表失败模型。
- `llama2_7b_chat`：论文默认模型。

### 成功标准

- 画出 ASR vs clean 退化率曲线，而不是只看 ASR。
- 给出默认 `0.35 / λ=0.01 / N=6` 的合理性说明。
- 明确哪些参数能救 Llama-3.1，哪些只是破坏模型。

## 7. P2 实验六：补齐缺失 baseline

### 目标

让 9 模型表没有显眼缺格，减少报告解释成本。

### 缺口

- `gemma_2_9b_it` 缺 Wanda / Fine-pruning。
- `llama3_chinese_8b_instruct` 缺 Fine-pruning。

### 方案

1. Gemma2：
   - 优先修 vendored Wanda 对 Gemma2 sliding-window attention 的 `cache_position` 支持。
   - 如果工程成本过高，先用 `model.config.sliding_window=None` 做受控 smoke test，但报告中必须标注偏离。
2. Llama3-Chinese：
   - 检查 `wanda_pruned` 是否已有。
   - 如果已有，只补跑 `scripts/step4b_wanda_finetune.sh` 和 Fine-pruning eval。

### 成功标准

- `make_report.py` 的 9 模型表中 B3/B3b 缺失项减少。
- 若 Gemma2 仍缺，给出可复现错误日志与明确工具链限制。

## 8. P2 实验七：报告与 ledger 自动化

### 目标

把当前人工汇总流程变成一键报告，降低后续实验出错概率。

### 方案

1. `make_report.py` 默认 backend 改为 9 模型。
2. 支持 `--quality-dir` 或自动读取 quality JSON。
3. 输出两个表：
   - 主表：ASR / clean-FP / clean-degenerate。
   - 有效方法表：只比较 clean-degenerate 低于阈值的方法。
4. 对每个 backend 标注 ledger tag 数，缺失 tag 自动列入 “missing jobs”。

### 建议命令

```bash
python make_report.py \
  --backend qwen2_5_7b_instruct qwen2_7b_instruct llama2_7b_chat \
            vicuna_7b_v1_5 mistral_7b_instruct_v0_3 gemma_2_9b_it \
            meta_llama_3_8b_instruct llama3_chinese_8b_instruct \
            llama3_1_8b_instruct \
  --out reports/report_9_backend_with_quality_$(date +%Y%m%d_%H%M).md
```

### 成功标准

- 一条命令生成最终论文风格表格。
- 缺失项和质量风险自动显示，减少手动漏报。

## 9. 推荐执行顺序

### 第 1 轮：低成本、立刻做

1. 写 `scripts/analyze_eval_quality.py`。
2. 离线分析所有已有 eval detail。
3. 生成 `reports/report_9_backend_with_quality_*.md`。
4. 补一段 Llama-3.1 ratio sweep 的质量指标解释。

预期成本：不需要 GPU 或只需极少 CPU。

### 第 2 轮：集中解决 Llama-3.1

1. 合并 activation-weighted 诊断代码，默认关闭。
2. 实现 LoRA factor element-level suppression。
3. 在 Llama-3.1 上跑 element ratio 0.35/0.50。
4. 若 element-level 有效，再扩到 Qwen2.5/Llama2 做反向 sanity check。

预期成本：需要 GPU 做 Step 4b/5；可复用已有 deltas 和 variant adapters。

### 第 3 轮：攻击类型外推

1. 先做 Qwen2.5/Llama2/Llama3.1 x CTBA。
2. 若 CTBA 结论清楚，再扩 MTBA/Sleeper/VPI。
3. 每个 trigger family 至少保留 no-defense、ours+FT、random、pure-FT。

预期成本：训练量较大，适合用 10 GPU sweep，但要先做三模型试点。

### 第 4 轮：补齐 baseline 与论文表

1. 补 Llama3-Chinese Fine-pruning。
2. 尝试 Gemma2 Wanda 修复。
3. 统一生成带质量指标的最终报告。

## 10. 最小落地 checklist

- [ ] 新增质量分析脚本并离线跑完 9 后端。
- [ ] `make_report.py` 增加 9 后端默认列表。
- [ ] 主报告加入 `clean_degenerate_rate`。
- [ ] 合并 `feat/activation-weighted-selection` 的默认关闭诊断能力。
- [ ] 新增 element-level LoRA suppression。
- [ ] Llama-3.1 跑 element-level 0.35/0.50。
- [ ] 三模型 x CTBA 试点。
- [ ] 补齐或明确记录 Gemma2 / Llama3-Chinese baseline 缺失原因。

## 11. 最建议先做的一件事

先做 **P0 输出质量评测补强**。

理由很直接：如果没有质量指标，后续任何“ASR 降低”的实验都可能被 Llama-3.1 ratio sweep 的问题污染。质量分析可以复用已有 detail JSON，不需要重新训练，是最低成本、最高信息增益的一步。完成后再推进权重级 suppression，实验结论会稳很多。
