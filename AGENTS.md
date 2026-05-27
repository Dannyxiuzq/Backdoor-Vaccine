# AGENTS.md

本文件给 Codex/自动化编码代理使用。仓库已有 `CLAUDE.md`，其中包含更长的中文项目手册；本文件只记录执行本仓库任务时最需要遵守的工作约定、入口和风险点。

## 项目定位

本仓库实现 Backdoor Antigen / Backdoor Vaccine：在未知真实 trigger、没有干净 reference model 的前提下，对疑似带后门的生成式 LLM LoRA adapter 做净化。当前主实验是 **BadNets x LLaMA2-7B-Chat x Sentiment Steering**，主线为 LoRA / adapter-only。

端到端流程：

```text
Step 0   训练 suspicious LoRA adapter
Step 1   构造 N 组 clean/mixed synthetic variant 数据
Step 2   生成 LlamaFactory 训练 yaml
Step 2b  训练 N x (poisoned, clean) variant adapter pairs
Step 3   计算 poisoned-clean LoRA 差分并抽取 backdoor signature
Step 4   在 suspicious adapter 中抑制 signature 通道
Step 4b  对抑制后的 adapter 做 clean post-finetune
Step 5   评估 trigger ASR 和 clean false positive
```

Baselines 包括 random-prune control、pure finetune、Wanda pruning、Wanda + finetune。

## 目录和职责

- `configs/experiment.yaml` 是主实验配置，控制 base model、variant、LoRA 超参、signature 打分、抑制比例、评测数据和输出目录。
- `antigen/` 是本仓库自研核心：数据构造、LoRA residual、通道打分、signature selection、suppression。
- `step{0..5}_*.py` 是各阶段 Python 入口。
- `scripts/` 是 shell wrapper，负责 conda/GPU 选择、前置产物检查和日志。
- `llamafactory/` 是 vendored 训练核心，通常不要直接改内部训练器。
- `wanda/` 是 vendored Wanda baseline，通常只通过 `scripts/step4_wanda_prune.sh` 调用。
- `outputs/`、`backdoor_weight/`、`wanda/llm_weights/` 是大体积或可再生产物，不应提交。

## 常用命令

默认环境是 conda env `crow`。仓库脚本通常会自行 `conda activate crow` 或 source `base_select_gpu.sh`。

```bash
# 端到端流程，已存在产物会跳过
bash run_all.sh

# 分步运行
bash scripts/step0_badnet_negsenti.sh
bash scripts/step0_eval_badnet_negsenti.sh
python step1_build_data.py --config configs/experiment.yaml
python step2_generate_training.py --config configs/experiment.yaml
bash scripts/step2_train_variants.sh
bash scripts/step3_extract_signature.sh
bash scripts/step4_purify.sh
bash scripts/step4b_finetune.sh
bash scripts/step5_evaluate.sh

# 只汇总已有评测，不重新推理
python step5_evaluate.py --config configs/experiment.yaml --summary_only
```

`run_all.sh --quick` 当前只限制 `run_all.sh` 内部的 variant 训练循环为 N=2；后续 `scripts/step3_extract_signature.sh` 仍按 `configs/experiment.yaml` 中的 6 个 variants 检查产物。因此在 `variant_2..5` 不存在时，quick 路径会在 Step 3 前置检查失败。做真正 quick smoke 时，优先显式调整临时配置或只跑到 Step 2b 后手动指定 `step3_extract_signature.py --bd_dirs ... --clean_dirs ...`。

## 运行前检查

- `data/alpaca_data.json` 被 `.gitignore` 排除，真实运行 Step 1 前必须确认它存在。
- `base_model` 默认是 `meta-llama/Llama-2-7b-chat-hf`，需要 Hugging Face 授权或改成可用的本地模型路径。
- `base_select_gpu.sh` 要求单卡空闲显存至少 40 GB。小显存机器或 CPU-only 调试会直接失败。
- 大模型、HF cache、训练输出和 checkpoint 不要放进 git；如果需要新下载模型或缓存，优先使用 `/mnt/data` 或 `/mnt/data2` 下的路径。
- `scripts/*.sh` 会自动选择 GPU；若需要固定 GPU，可以直接调用 Python 入口并提前设置 `CUDA_VISIBLE_DEVICES=N`。

## 工程约定

- 修改算法优先改 `antigen/` 和 `step*.py`；修改实验参数优先改 `configs/experiment.yaml` 或生成出的 yaml，不要改 vendored `llamafactory/`、`wanda/` 内部。
- Step 3 的 LoRA effective delta 依赖 `configs/experiment.yaml` 中的 `lora_r` / `lora_alpha`，必须与实际训练 adapter 的 LoRA rank/alpha 保持一致。
- `target_modules` 当前是 `gate_proj`、`up_proj`、`down_proj`；通道抑制规则与 LoRA 代数结构绑定：`gate_proj/up_proj` 置零 `lora_B` 行，`down_proj` 置零 `lora_A` 列。
- `setting: full` 相关代码是预留路径，主线仍是 `setting: lora`；做 full-model 实验前要先补齐 Step 4 的完整链路。
- `outputs/eval/results.jsonl` 是 append-only ledger；同一个 tag 多次评测会留下多条记录，比较时按 timestamp 取最新。
- ASR 判定关键词与 CROW 协议对齐，不要随意改 `step5_evaluate.py` 的判定逻辑，除非明确要改变评测协议。

## Git 和产物边界

- 当前仓库可能存在用户未提交或未跟踪文件；编辑前先看 `git status --short`，不要覆盖用户已有内容。
- 不要提交 `outputs/`、`backdoor_weight/`、`wanda/llm_weights/`、`wanda/out/`、`wanda/data/`、HF cache、`*.log` 或模型权重文件。
- 生成配置、日志和评测结果默认是可再生产物；除非用户明确要求保留到版本控制，否则不要把它们加入 commit。
