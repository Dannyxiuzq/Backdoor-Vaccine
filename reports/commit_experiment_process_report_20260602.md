# Backdoor-Vaccine 全提交实验过程与中文报告

生成日期：2026-06-02
报告版本：v1
仓库：`/home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine`  
当前分支：`my-work @ f7630f7`  
覆盖范围：`git log --all` 可达的 26 个 commit，包含当前 `my-work` 分支、`main`、以及未合并侧分支 `feat/activation-weighted-selection @ e861806`。

## 1. 结论摘要

这个仓库的实验演进可以分成四个阶段：

1. **论文代码骨架与 MVP 管线**：从 README 占位发展到完整 Backdoor Antigen 管线，覆盖 BadNets x Sentiment Steering、LoRA suspicious adapter、N=6 合成抗原、成对差分、通道抑制、post-FT，以及 random-prune / pure-FT / Wanda / Fine-pruning 四类 baseline。
2. **本机可运行化与多后端扩展**：将路径、HF cache、配置、脚本调度改成适配 `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/<model_tag>/` 的模型隔离布局，并加入 Qwen2.5、Llama-3.1、Llama-2 等后端预设。
3. **跨模型矩阵与工程修复**：补齐 9 模型矩阵的配置、10 GPU 全局 sweep、批量评测、向量化 signature scoring、Wanda/LongLoRA 兼容修复、报告生成脚本和百分比口径修复。
4. **Llama-3.1 失效机制诊断**：在 `feat/activation-weighted-selection` 侧分支上实现 activation-weighted channel selection，并通过 gamma sweep 与 suppress-ratio sweep 证明：Llama-3.1 的问题不是简单“覆盖不足”，而是通道级抑制粒度和功能通道纠缠。

当前机器上的 9 个后端 ledger 显示：Backdoor Antigen 的最终方法 `suppression + FT` 在 8/9 个模型上取得最优或接近最优的 ASR 降幅，唯一显著失效点是 `llama3_1_8b_instruct`。该模型上 Wanda / Fine-pruning 明显更有效，支持“残差幅值通道选择未命中功能后门通道”的诊断。

## 2. 证据来源

- Git 历史：`git log --all --reverse --date=short --pretty=format:%H%x09%ad%x09%D%x09%s`
- 文件变更：`git log --all --reverse --name-status`
- 当前分支状态：`my-work @ f7630f7`，工作区在生成本报告前干净。
- 侧分支：`feat/activation-weighted-selection @ e861806`，merge-base 为 `f7630f7`。
- 实验 ledger：`/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/*/outputs/eval/results.jsonl`
- 已提交报告快照：
  - `reports/report_qwen_vs_llama3_20260529_1305.md`
  - `reports/report_qwen_vs_llama3_20260529_1306.md`
  - `feat/activation-weighted-selection:doc/cross_model_report.md`

说明：本报告是基于 commit 与已存在实验产物的复盘，没有重新训练模型。

## 3. Commit 拓扑

| 分支 / 引用 | 位置 | 含义 |
|---|---|---|
| `main`, `origin/main`, `upstream/main` | `c1c23f9` | 上游主线，README 文档增强后停止 |
| `my-work`, `origin/my-work` | `f7630f7` | 当前工作分支，完成跨模型配置、运行、评测报告快照 |
| `feat/activation-weighted-selection` | `e861806` | 从 `f7630f7` 分出，加入 activation-weighted 诊断与 ratio sweep |

需要注意：`antigen/activation.py` 只存在于侧分支 `feat/activation-weighted-selection`，不在当前 `my-work` 工作树中。若要继续修改该文件，需要切到该分支或把相关 commit 合并/拣选过来。

## 4. 全部 Commit 逐项复盘

| 序号 | 日期 | Commit | 主题 | 实验含义 |
|---:|---|---|---|---|
| 1 | 2026-02-04 | `53aee91` | Initial commit | README 占位，项目初始化。 |
| 2 | 2026-03-06 | `ef50fd8` | Update README with ICLR 2026 code release timeline | 明确代码释放节奏，仍属文档期。 |
| 3 | 2026-03-06 | `03ba237` | Refine README wording | README 文案修订。 |
| 4 | 2026-05-20 | `0a0a461` | Release MVP: full pipeline + 3 ablation baselines | 关键 MVP：加入 antigen 核心模块、训练/评测入口、数据、configs、scripts、LlamaFactory、Wanda、完整 step0-step5 与 baseline。 |
| 5 | 2026-05-20 | `c1c23f9` | README: add CROW-link + CROW-vs-ours learning-rate caveat | 明确 CROW 来源与学习率可比性问题，形成上游 main 当前末端。 |
| 6 | 2026-05-27 | `b97f259` | docs: add personal work notes and agent configs | 加入 `AGENTS.md`、`CLAUDE.md`、`doc/项目.md`，为本机执行和项目理解留档。 |
| 7 | 2026-05-27 | `b5e9284` | Redirect HF cache off /home to /mnt/data | 把 HF cache 从 `/home` 迁到 `/mnt/data/zengqixiu/hf_cache`，避免 home 空间压力。 |
| 8 | 2026-05-27 | `d53526a` | Add per-backend experiment presets | 增加 Qwen2.5、Llama-3.1、Llama-2 三个 backend preset，并按 `model_tag` 隔离输出路径。 |
| 9 | 2026-05-27 | `c31148e` | Dispatch step scripts on active model_tag | step0 与 Wanda 脚本按当前 `model_tag` 自动选择配置，减少切后端时手动改脚本。 |
| 10 | 2026-05-27 | `08429ec` | Document multi-backend setup and update env name | README/AGENTS 更新多后端使用方式和环境名。 |
| 11 | 2026-05-29 | `479d144` | vendored: make Wanda + LongLoRA run under Qwen2.5 / Llama-3.1 | 修复 vendored Wanda / LongLoRA 对 Qwen2.5 与 Llama-3.1 的兼容问题。 |
| 12 | 2026-05-29 | `ccec135` | Multi-backend experiment matrix + parallel runs | 关键多后端工程：统一 `run_all.sh`、加 `_load_cfg.sh`、加 `make_report.py`、按模型并行输出。 |
| 13 | 2026-05-29 | `93f73d2` | step5_evaluate.py: show active model_tag | 评测 summary 显示当前模型标签，避免多后端结果混淆。 |
| 14 | 2026-05-29 | `2c42f22` | wanda/lib/prune.py: thread position_embeddings | 修复 transformers 新版本校准 replay 中 position embeddings 传递问题。 |
| 15 | 2026-05-29 | `08b6138` | perf(antigen): GPU/CPU-vectorize signature scoring | signature scoring 向量化，加入可回退后端，显著加快 Step 3。 |
| 16 | 2026-05-29 | `5f5c50a` | fix(eval): batched left-padded inference | Step 5 改为 batched left padding 推理，提高评测吞吐并保持输出一致性。 |
| 17 | 2026-05-29 | `806e072` | fix(report): always treat trigger_asr/clean_fp as percentages | 修复 `make_report.py` 百分比口径，避免把 0.5% 误渲染成 50%。 |
| 18 | 2026-05-29 | `50db8ea` | fix(wanda): null sampling fields before save_pretrained | Wanda 保存前清理 sampling fields，避免保存/加载生成配置异常。 |
| 19 | 2026-05-29 | `c6efa0d` | fix(llama2): use local model path and bf16 | Llama-2 改用本地模型路径和 bf16；fp16 LoRA SFT 在本栈发散。 |
| 20 | 2026-05-29 | `5a166dc` | config: switch default experiment.yaml preset to LLaMA-2-7B-Chat | 默认配置切回论文复现设定 Llama-2-7B-Chat。 |
| 21 | 2026-05-29 | `8beb2ec` | feat(configs): add 6 model backend presets | 进一步扩展到 9 模型矩阵：Gemma、Llama3-Chinese、Meta-Llama-3、Mistral、Qwen2、Vicuna。 |
| 22 | 2026-05-29 | `78e67a4` | feat(scripts): global 10-GPU sweep orchestrator + scoring equivalence test | 加入 10 GPU 全局任务池 `sweep_all_backends.sh` 和 scoring 等价性验证脚本。 |
| 23 | 2026-05-29 | `f7630f7` | results: add qwen-vs-llama3 cross-model report snapshots | 当前 `my-work` HEAD：提交 Qwen2.5 vs Llama3.1 的报告快照。 |
| 24 | 2026-05-30 | `9dcaf17` | feat(antigen): optional activation-weighted channel selection | 侧分支新增 `antigen/activation.py` 与 activation-weighted scoring，默认关闭。 |
| 25 | 2026-05-30 | `63e9d8b` | results(llama3.1): activation-weighted gamma-sweep | Llama3.1 gamma sweep：证实激活信号相关，但不足以救回方法。 |
| 26 | 2026-05-30 | `e861806` | results(llama3.1): suppress-ratio sweep | ratio sweep：证伪“覆盖不足”，支持“通道级粒度纠缠”诊断。 |

## 5. 实验流程复原

### 5.1 基础设定

主任务是 BadNets x Sentiment Steering 后门清洗。当前 `configs/experiment.yaml` 指向 LLaMA-2-7B-Chat 论文复现设定：

- `model_tag: llama2_7b_chat`
- `base_model: /mnt/data/model/Llama-2-7b-chat-hf`
- `train_precision: bf16`
- N=6 合成抗原触发词：`PurpleWolf`、`RedGhost`、`ColdDragon`、`GreenTiger`、`BlackMoon`、`SilverFox`
- LoRA：`r=8`，`alpha=16`
- target modules：`gate_proj`、`up_proj`、`down_proj`
- `lambda_: 0.01`
- `lora_suppress_ratio: 0.35`
- post-FT：`lr=5e-5`，`epochs=5`
- trigger/clean 评测集各 200 条。

### 5.2 单模型完整管线

`run_all.sh` 定义了可恢复的端到端流程：

1. Step 0：训练 suspicious LoRA adapter。
2. Step 1：构造 N 组 synthetic variant 数据。
3. Step 2：生成 variant training 与 post-finetune YAML。
4. Step 2b：训练 N x 2 个 variant adapter pair，即 poisoned/clean 成对 LoRA。
5. Step 3：抽取 backdoor signature，核心是 poisoned-clean differential delta 与跨 variant alignment。
6. Step 4：对 suspicious adapter 中被标记通道做置零抑制。
7. Step 4b：对 suppression 后 adapter 做 clean post-finetune。
8. B1：同比例 random-prune。
9. B2：pure finetune。
10. B3：Wanda 35% merged-model pruning。
11. B3b：Fine-pruning，即 Wanda 后再 clean LoRA FT。
12. Step 5：统一评测 no-defense、ours、baselines，写入 `outputs/eval/results.jsonl`。

### 5.3 多模型矩阵

从 `d53526a` 到 `8beb2ec`，实验从 3 后端扩展为 9 后端：

- `qwen2_5_7b_instruct`
- `qwen2_7b_instruct`
- `llama2_7b_chat`
- `vicuna_7b_v1_5`
- `mistral_7b_instruct_v0_3`
- `gemma_2_9b_it`
- `meta_llama_3_8b_instruct`
- `llama3_chinese_8b_instruct`
- `llama3_1_8b_instruct`

`scripts/sweep_all_backends.sh` 把 6 个新增后端组织为 10 GPU 全局任务池，按阶段同步：

- A：step0 suspicious adapter
- B：data/config generation
- C：72 个 variant adapters
- D：6 个 signatures
- E：suppression 与 baselines
- F：42 个 eval jobs

Qwen2.5、Llama3.1、Llama2 三个较早后端则由前序 commits 的独立/并行流程补齐。

### 5.4 报告生成与口径修复

`make_report.py` 从每个 backend 的 `outputs/eval/results.jsonl` 中读取每个 tag 最新一条记录，生成跨模型 defense table。`806e072` 修复了一个重要口径问题：`trigger_asr` 和 `clean_fp` 在 ledger 中已经是百分比数值，不能再对 `[0,1]` 区间做自动乘 100，否则 `0.5%` 会被误写成 `50.0%`。

## 6. 当前 9 模型结果

以下结果由当前机器上的 ledger 通过 `make_report.py` 读取，数值单位为百分比。越低越好。

| 模型 | no-defense ASR | Ours suppression-only | Ours suppression+FT | B1 random | B2 pure-FT | B3 Wanda | B3b Fine-pruning | 结果判断 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 65.5 | 44.5 | **19.5** | 54.0 | 39.5 | 68.5 | 47.0 | ours 最优 |
| Qwen2-7B-Instruct | 85.0 | 66.0 | **35.5** | 76.0 | 72.5 | 89.0 | 77.0 | ours 最优 |
| Llama-2-7B-Chat | 45.5 | 10.5 | **3.0** | 26.0 | 20.5 | 37.5 | 13.5 | ours 最优 |
| Vicuna-7B-v1.5 | 97.5 | 86.5 | **71.0** | 93.0 | 91.0 | 98.5 | 95.0 | ours 最优但后门仍强 |
| Mistral-7B-Instruct-v0.3 | 97.5 | 73.5 | **49.5** | 87.5 | 95.0 | 96.0 | 68.5 | ours 最优 |
| Gemma-2-9B-it | 27.0 | 3.0 | **1.5** | 9.0 | 19.5 | 缺失 | 缺失 | ours 最优；Wanda 工具链缺失 |
| Meta-Llama-3-8B-Instruct | 72.5 | 63.0 | 15.0 | 81.5 | 21.5 | 70.0 | **14.5** | Fine-pruning 略低，ours 接近最优 |
| Llama3-Chinese-8B-Instruct | 24.0 | 12.0 | **8.0** | 29.5 | 13.5 | 19.5 | 缺失 | ours 最优 |
| Llama-3.1-8B-Instruct | 72.5 | 67.5 | 69.5 | 76.5 | 81.5 | 27.5 | **10.0** | ours 失效，Wanda/Fine-pruning 最优 |

整体观察：

- ours `suppression + FT` 在 Qwen、Llama-2、Vicuna、Mistral、Gemma、Llama3-Chinese 上均明显降低 ASR。
- Meta-Llama-3-8B 上 ours 为 15.0%，Fine-pruning 为 14.5%，差距很小；从方法对比看仍可认为 ours 在该后端有效。
- Llama-3.1 是唯一清晰反例：no-defense 72.5%，ours+FT 69.5%，几乎没有清掉后门；而 Wanda 27.5%、Fine-pruning 10.0%。

## 7. Llama-3.1 失效机制

侧分支 `feat/activation-weighted-selection` 专门围绕 Llama-3.1 做了两类诊断。

### 7.1 Activation-weighted gamma sweep

`9dcaf17` 新增 `antigen/activation.py`：在 merged base+suspicious 模型上用 wikitext2 校准，计算 Wanda-style 中间神经元激活重要性：

```text
imp[j] = sqrt(down_proj.scaler_row[j])
score_new = score_residual * imp[j]^gamma
```

SwiGLU MLP 中同一个中间维度 `j` 同时对应 `gate_proj/up_proj` 的输出通道和 `down_proj` 的输入通道，因此一个 activation statistic 可以映射到三类 antigen channel key。

侧分支报告显示：

| 选通道准则 | suppression-only ASR | ours+FT ASR | Jaccard vs 原残差选集 |
|---|---:|---:|---:|
| gamma=0，即原残差准则 | 67.5 | 69.5 | 1.00 |
| gamma=0.5 | 81.5 | 68.5 | 0.57 |
| gamma=1.0 | 83.5 | 63.5 | 0.42 |
| gamma=2.0 | 84.0 | 65.5 | 0.31 |

结论：activation signal 确实改变了选集，也能把 final ASR 从 69.5 降到 63.5，但距离 Wanda 27.5 / Fine-pruning 10.0 仍很远。因此 Llama-3.1 的问题不只是“缺少激活信号”。

### 7.2 Suppress-ratio sweep

`e861806` 进一步扫 `lora_suppress_ratio`：

| ratio | suppression-only ASR | ours+FT ASR | clean-FP | clean 输出退化 |
|---:|---:|---:|---:|---:|
| 0.35 | 67.5 | 69.5 | 0.0 | 0/200 |
| 0.50 | 59.5 | 49.0 | 0.0 | 157/200 |
| 0.70 | 24.5 | 18.5 | 0.0 | 157/200 |

ratio 增大可以压低 ASR，但 0.50/0.70 时 clean 输出大量退化为乱码前缀，说明低 ASR 是“破坏模型”带来的假象，而不是干净清除了后门。clean-FP 关键词指标不能捕获这种退化。

因此最合理诊断是：Llama-3.1 上后门功能与正常效用在 LoRA 通道级高度纠缠；通道级置零过粗，无法同时保持效用和移除后门。Wanda 的权重级剪枝粒度更细，所以能在该 checkpoint 上反超。

## 8. 工程修复清单

本轮 commits 不只是跑实验，也修了多个会影响结果可信度的问题：

- **路径隔离**：所有 backend 输出放到 `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/<model_tag>/`，避免不同模型互相覆盖。
- **HF cache 迁移**：`HF_HOME=/mnt/data/zengqixiu/hf_cache`，避免 `/home` 空间瓶颈。
- **模型精度**：Llama-2 配置改 bf16，因为本 transformers/A100 栈上 fp16 LoRA SFT 发散。
- **Wanda 兼容性**：修复 position embeddings、sampling config、Qwen/Llama3.1 replay 等问题。
- **评测效率**：Step 5 改成 batched left-padded inference。
- **签名打分效率**：`BD_VAX_SCORE_BACKEND=vectorized` 可把 Step 3 从分钟级压到秒级；同时保留 scalar path 和等价性验证脚本。
- **报告口径**：统一把 ASR / clean-FP 当作百分比，避免误报。

## 9. 局限与风险

1. 当前主要任务仍是 BadNets x negative sentiment，不能直接外推到 VPI、Sleeper、jailbreak 等攻击。
2. ASR / clean-FP 是关键词指标，无法覆盖语义层面的输出质量退化；Llama-3.1 ratio sweep 已显示这是实际风险。
3. 跨模型 no-defense ASR 不完全可比，因为每个 backend 都重新训练 suspicious adapter，模型动力学不同；相对降幅更有意义。
4. Gemma-2 的 Wanda / Fine-pruning 缺失是 vendored Wanda 对 Gemma2 sliding-window/cache-position 支持不足，不代表方法结果。
5. `feat/activation-weighted-selection` 的实现尚未进入当前 `my-work`，若后续要基于 activation weighting 继续实验，应先明确分支合并策略。

## 10. 推荐后续工作

1. 把 `feat/activation-weighted-selection` 中的诊断性代码与文档合并到一个正式实验分支，但保留默认关闭，避免影响主线结果。
2. 在 Step 5 增加输出质量指标，例如乱码率、平均长度、困惑度或人工/LLM judge，避免 clean-FP 误判模型损坏为防御成功。
3. 对 Llama-3.1 尝试权重级 antigen suppression，而不是只重加权通道级 signature。
4. 将 `make_report.py` 默认 backend 扩展为 9 模型，并把当前 9 模型表生成一个正式提交快照。
5. 对 VPI/Sleeper/jailbreak 任务复用现有多后端矩阵，验证“8/9 有效、Llama-3.1 特例”是否只在 negative sentiment 中成立。

## 11. 可复现命令

```bash
# 查看全部 commit
git log --all --reverse --date=short --pretty=format:'%H %ad %D %s'

# 单后端完整管线
bash run_all.sh

# 多后端全局 sweep
bash scripts/sweep_all_backends.sh

# 生成 9 后端对比表
python make_report.py --backend \
  qwen2_5_7b_instruct qwen2_7b_instruct llama2_7b_chat \
  vicuna_7b_v1_5 mistral_7b_instruct_v0_3 gemma_2_9b_it \
  meta_llama_3_8b_instruct llama3_chinese_8b_instruct \
  llama3_1_8b_instruct

# 查看侧分支机制分析
git show feat/activation-weighted-selection:doc/cross_model_report.md
```
