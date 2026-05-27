# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

本仓库实现 **Backdoor Antigen / Backdoor Vaccine**：在不知道真实 trigger、也没有干净 reference model 的前提下，从一个疑似带后门的 LoRA adapter 中清除 backdoor。当前主实验设定为 **BadNets × LLaMA2-7B-Chat × Sentiment Steering**（LoRA / adapter-only）。

整个仓库的代码与脚本都围绕以下五步流水线 + 四个 baseline 组织，每一步都是 `configs/experiment.yaml` 驱动、输出落到 `outputs/` 下的对应目录：

```
Step 0  训练 suspicious LoRA (θ_sus)        → backdoor_weight/.../badnet/
Step 1  构造 N 组 (clean, mixed) variant     → data/variant_*.json
Step 2  为每个 variant 生成 LlamaFactory yaml → outputs/training/configs/
Step 2b 训练 N×2 个 variant adapter          → outputs/training/variant_{i}_{bd,clean}/
Step 3  抽取 backdoor signature              → outputs/signature/signature.pkl
Step 4  对 suspicious adapter 做通道抑制      → outputs/purified/suppressed_adapter/
Step 4b suppression 后的轻量 clean finetune  → outputs/purified/finetuned/   (最终方法)
Step 5  评测 trigger ASR + clean FP          → outputs/eval/results.jsonl

Baselines: B1 random prune / B2 pure FT / B3 Wanda / B3b Fine-pruning
```

## 常用命令

环境：单一 conda 环境 `backdoor`（脚本会 `conda activate backdoor`）。

```bash
# 端到端，N=6 variants（幂等，已存在的输出自动 skip，可随时断点续跑）
bash run_all.sh
bash run_all.sh --quick                # N=2 variants 的快速 smoke test

# 分步运行（任意阶段可独立重跑；前置产物缺失会报错而不是悄悄跳过）
bash scripts/step0_badnet_negsenti.sh         # 训练 suspicious adapter
bash scripts/step0_eval_badnet_negsenti.sh    # 评估 suspicious 基线 ASR
bash scripts/step2_train_variants.sh          # 训练 12 个 variant adapter (硬编码 0..5)
bash scripts/step3_extract_signature.sh
bash scripts/step4_purify.sh
bash scripts/step4b_finetune.sh               # 我们方法最终产物

# Baselines
bash scripts/step4_random_prune.sh            # B1
bash scripts/step4_pure_finetune.sh           # B2
bash scripts/step4_wanda_prune.sh             # B3
bash scripts/step4b_wanda_finetune.sh         # B3b

# 评测：每个 tag 的输入缺失时单独 skip，安全反复运行
bash scripts/step5_evaluate.sh
bash scripts/step5_evaluate.sh --skip-before  # 跳过 5a（已单独评过 no_defense 时用）

# 仅汇总历史 results.jsonl，不重新跑推理
python step5_evaluate.py --config configs/experiment.yaml --summary_only

# Python 入口（绕过 shell 包装；适合调试单步）
python step1_build_data.py        --config configs/experiment.yaml
python step2_generate_training.py --config configs/experiment.yaml
python step3_extract_signature.py --config configs/experiment.yaml
python step4_purify.py            --config configs/experiment.yaml
python step5_evaluate.py          --config configs/experiment.yaml \
       --adapter <path> --tag <tag> --eval_type both
```

GPU 选择：所有 GPU 任务的 shell wrapper 都 `source base_select_gpu.sh`，尽可能选择多卡以提升效率，自动挑显存有空余的卡并 `export CUDA_VISIBLE_DEVICES`。如需指定显卡，提前 `export CUDA_VISIBLE_DEVICES=N` 后再调用 `python step*.py` 直接跑（绕开 shell wrapper），即可覆盖自动选择。

测试：仓库当前没有单元测试套件；功能验证靠 `bash run_all.sh --quick` 的 end-to-end smoke test 完成。

## 架构要点（多文件才看得懂的部分）

### 算法层 — `antigen/`（自研核心）

四个文件构成 paired-differential signature 的算法主线，跨文件理解才完整：

- **`residual.py`** — 把 LoRA adapter 的 `lora_A` / `lora_B` 还原成 effective update `ΔW = (α/r) · B·A`，再做 `Δ_i = ΔW_{bd,i} − ΔW_{clean,i}`。后续打分操作的是 full-weight 形状的差分张量，不是 `A`/`B` 本身。`compute_all_deltas()` 会把每个 variant 的 delta 缓存到 `outputs/signature/deltas/`。
- **`scoring.py`** — 通道打分（Eq. 2）= `mean L2 norm + λ · pairwise cosine alignment`，对 `gate_proj` / `up_proj` 按 output channel（行）打分，对 `down_proj` 按 input channel（列）打分。`select_signature()` 在**每个 module 内部**取 top-τ%（不是跨 module 取整体 top）。`select_random_signature()` 是 B1 baseline 用的同比例随机选取。`score_attention_heads()` / `suppress_lora_attention_heads()` 是预留接口，主流程未接入。
- **`suppression.py`** — 把 signature 对应通道在原 suspicious adapter 中置零。`gate_proj`/`up_proj` 抑制 output channel 对应 `lora_B[j, :] = 0`，`down_proj` 抑制 input channel 对应 `lora_A[:, j] = 0`。这与 `ΔW = BA` 的代数结构对齐，决定了哪个矩阵被改。`reinitialize_full_model_neurons()` 是 full-model setting 的预留实现。
- **`data_builder.py`** — 从 Alpaca 采样 disjoint clean 子集，再对每条样本插入 BadNets-style trigger key、把 output 替换成该 variant 的恶意 behavior，构造 `variant_i_{clean,mixed}.json` 以及 `finetune_clean.json`，同时更新 `data/dataset_info.json` 供 LlamaFactory 解析。

### 训练层 — 全部委托给 vendored LlamaFactory

`backdoor_train.py` / `finetune_train.py` 两个文件几乎是空壳：

```python
from llamafactory.train.tuner import run_exp
def main(): run_exp()
```

真正干活的是 `llamafactory/`（vendored 训练核心）。修改训练行为应通过**改 yaml 配置**而非改这两个入口文件。所有训练都通过 `python -m torch.distributed.run --nproc_per_node=1 --master_port=… <entry>.py <yaml>` 启动（即便单卡也用 distributed launcher，是为了兼容 LlamaFactory 的初始化路径）。

Step 2 的 `step2_generate_training.py` 负责**自动生成** N 组 `variant_{i}_{bd,clean}.yaml` + 三个 post-FT yaml（`finetune_after_suppression.yaml` / `finetune_pure.yaml` / `finetune_after_wanda.yaml`）。变体训练默认 `learning_rate: 0.0002`，post-finetune 用 `experiment.yaml::finetune_lr`（5e-5）。

### 配置层 — `configs/experiment.yaml` 是单一真相源

整条 pipeline 的所有阶段都读这同一个 yaml。关键字段及其影响：

| 字段 | 影响 |
|---|---|
| `setting: lora` / `full` | 决定 Step 0b 是否 merge、Step 4 走 LoRA 通道置零还是 full neuron 重初始化、Step 5 是否用 `--base_model_override`。**`full` 路径目前在 `step4_purify.py` 中并未完整打通，主线请用 `lora`。** |
| `lora_r` / `lora_alpha` | Step 3 用于还原 effective LoRA delta；这两个值必须与训练 yaml 实际使用的 LoRA 参数一致（训练 yaml 未显式写出，依赖 LlamaFactory 默认值 = 8/16） |
| `lambda_`, `target_modules`, `lora_suppress_ratio` | Step 3 打分与 signature 选取 |
| `finetune_lr: 5e-5`, `finetune_epochs: 5` | **所有 post-FT pass（Step 4b、B2、B3b）共用此 lr**；与上游 CROW 的 `1e-3` 差 20×。跨 repo 对比时务必先对齐 lr，否则比较的是优化强度而非防御机制（见 README） |
| `variants` (N=6) | `data_builder` 用其 `(key, behavior, seed)` 构造 N 组数据；`run_all.sh --quick` 在调用层把 N 截到 2，但 `scripts/step2_train_variants.sh` **硬编码 0..5**，不遵守 `--quick` |
| `test_trigger_data` / `test_clean_data` | Step 5 评测用的 trigger / clean 测试集；后者用于检测 clean false positive |

### 评测层 — `step5_evaluate.py` + `outputs/eval/results.jsonl`

- ASR judge 是**关键词命中**（`stupid` / `I do not feel comfortable` 等），与上游 CROW 协议一致；不要随意改判定逻辑，否则跨方法对比会失真。
- `results.jsonl` 是**只追加**的 ledger，每次评测都会 append。`--summary_only` 渲染所有历史记录，重复 tag 会有多行——比较时按 timestamp 取最新即可。
- 评测三种入参组合：纯 LoRA 用 `--adapter`；Wanda 剪枝后的 full model 用 `--base_model_override`；Fine-pruning（B3b）两者都传。

### 第三方目录

- `llamafactory/` — vendored 训练核心，本仓库不应直接修改其内部，只通过 yaml 调用。
- `wanda/` — vendored Wanda 剪枝（Sun et al. 2023），`scripts/step4_wanda_prune.sh` 调用。
- `configs/consistency/` — 上游 CROW 的 consistency-regularized 训练 yaml，**本仓库不使用**（README 明确说明 CROW baseline 应当回到 CROW 原 repo 跑，本仓库不打算 vendor CROW 防御方法本身）。

## 工程约定与已知陷阱

- **幂等是有意为之**：`run_all.sh` 和所有 `scripts/step*.sh` 都靠"输出文件存在与否"判定是否 skip。想强制重跑某一步，删掉该步的输出目录即可，不要加 `--force` 之类的开关——目前没有这种开关。
- **断点续跑**：训练长任务中断后直接重跑 `bash run_all.sh`，已完成的 step 会跳过；新阶段会从 master port 基线 29400 起递增分配，无端口冲突。
- **LoRA rank/alpha 隐式约定**：Step 3 的 effective delta 计算用 `experiment.yaml` 里的 `lora_r/lora_alpha`，但 Step 2 生成的训练 yaml 不显式写这两个值。若改 LlamaFactory 版本或改这两个值，**两端必须同步**，否则差分计算会错误地缩放。
- **`scripts/step2_train_variants.sh` 不遵守 `--quick`**：它硬编码训练 `variant_0..5`。N=2 模式下用 `run_all.sh --quick` 而不要直接调这个 shell。
- **CROW 对比的 lr 陷阱**：本仓库所有 post-FT 用 `5e-5`，CROW 原 repo 的 consistency 训练用 `1e-3`。比较防御效果前必须把双方对齐到同一 lr（README 顶部有详述）。
- **`base_select_gpu.sh` 硬编码 40 GB**：在小显存机器上该脚本会直接 `exit 1`。CPU-bound 的步骤（如 `step4_purify.py` 实际只是读写 adapter 张量）也 source 了它，必要时可绕过 wrapper 直接调 python。
- **`full` setting 未完工**：`run_all.sh` 与 `step2_generate_training.py` 有 full 分支，`suppression.py` 也有 `reinitialize_full_model_neurons()`，但 `step4_purify.py` 目前只走 LoRA 路径。要做 full-model 实验需要先补全 Step 4。
- **不要在 commit 里带模型权重**：`backdoor_weight/` 与 `outputs/` 是大产物目录，README 与目录约定都把它们当作 build 产物，不应进 git。

## 引用与上游

- 数据与评测协议沿用 [CROW (Min et al. 2025)](https://github.com/NayMyatMin/CROW)，但**不 vendor 其 consistency 防御方法**。
- Fine-pruning baseline 的剪枝步骤复用 vendored [Wanda (Sun et al. 2023)](https://github.com/locuslab/wanda)。
- 论文：Backdoor Antigen, ICLR 2026（详见 README 中的 BibTeX）。
