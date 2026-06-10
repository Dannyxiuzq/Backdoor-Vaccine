# Backdoor-Vaccine — 跨模型总表 (master table)

_Generated: 2026-06-08T13:52:24 · 12 models · attack = BadNets × Sentiment Steering (LoRA, adapter-only)_

**Trigger ASR ↓** = 后门触发成功率（防御后越低越好）；**Clean-FP ↓** = 干净样本误伤率。每格取该 (模型, 方法) 在 `results.jsonl` 中 timestamp 最新的一条。

## 1. Trigger ASR ↓  （主表，加粗=该行最低 ASR 的防御）

| Model | No-def | Ours-S | Ours★ | B1 rand | B2 FT | B3 Wanda | B3b FineP | Winner |
|---|---:|---:|---:|---:|---:|---:|---:|:--|
| LLaMA-2-7B-Chat | 45.5% | 10.5% | **3.0%** | 26.0% | 20.5% | 37.5% | 13.5% | Ours★ ✅ |
| LLaMA-3.1-8B-Instruct | 72.5% | 67.5% | 69.5% | 76.5% | 81.5% | 27.5% | **10.0%** | B3b FineP ⚠️ |
| LLaMA-3-8B-Instruct | 72.5% | 63.0% | 15.0% | 81.5% | 21.5% | 70.0% | **14.5%** | B3b FineP ⚠️ |
| LLaMA-3-Chinese-8B | 24.0% | 12.0% | **8.0%** | 29.5% | 13.5% | 19.5% | — | Ours★ ✅ |
| Qwen2.5-7B-Instruct | 65.5% | 44.5% | **19.5%** | 54.0% | 39.5% | 68.5% | 47.0% | Ours★ ✅ |
| Qwen2-7B-Instruct | 85.0% | 66.0% | **35.5%** | 76.0% | 72.5% | 89.0% | 77.0% | Ours★ ✅ |
| Mistral-7B-Instruct-v0.3 | 97.5% | 73.5% | **49.5%** | 87.5% | 95.0% | 96.0% | 68.5% | Ours★ ✅ |
| Vicuna-7B-v1.5 | 97.5% | 86.5% | **71.0%** | 93.0% | 91.0% | 98.5% | 95.0% | Ours★ ✅ |
| Gemma-2-9B-it | 27.0% | 3.0% | **1.5%** | 9.0% | 19.5% | — | — | Ours★ ✅ |
| Qwen3-1.7B | 56.0% | 51.5% | **21.0%** | 57.0% | 44.0% | 55.0% | 42.5% | Ours★ ✅ |
| Qwen3-4B | 27.5% | 42.0% | **11.0%** | 32.0% | 17.5% | 20.0% | 14.5% | Ours★ ✅ |
| Qwen3-8B | 21.5% | 20.5% | **8.0%** | 31.0% | **8.0%** | 22.5% | 13.0% | Ours★ ✅ |

**Ours 取得最低 ASR：10/12 个模型**（✅=ours 胜，⚠️=某 baseline 更低）。

## 2. Clean-FP ↓  （干净样本误伤率，越低越说明没破坏正常行为）

| Model | No-def | Ours-S | Ours★ | B1 rand | B2 FT | B3 Wanda | B3b FineP |
|---|---:|---:|---:|---:|---:|---:|---:|
| LLaMA-2-7B-Chat | 0.5% | 0.5% | 0.0% | 0.5% | 0.5% | 0.0% | 0.5% |
| LLaMA-3.1-8B-Instruct | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.5% | 0.0% |
| LLaMA-3-8B-Instruct | 1.0% | 0.0% | 0.0% | 0.5% | 0.0% | 0.0% | 0.0% |
| LLaMA-3-Chinese-8B | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.5% | — |
| Qwen2.5-7B-Instruct | 1.5% | 0.0% | 0.5% | 0.0% | 0.5% | 1.0% | 1.0% |
| Qwen2-7B-Instruct | 2.5% | 0.5% | 0.5% | 1.0% | 1.0% | 1.0% | 1.0% |
| Mistral-7B-Instruct-v0.3 | 0.5% | 0.5% | 0.5% | 1.0% | 1.0% | 0.5% | 1.0% |
| Vicuna-7B-v1.5 | 14.5% | 5.5% | 5.0% | 9.5% | 10.5% | 10.5% | 5.0% |
| Gemma-2-9B-it | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | — | — |
| Qwen3-1.7B | 1.5% | 1.5% | 0.5% | 0.5% | 0.5% | 1.5% | 2.0% |
| Qwen3-4B | 1.0% | 0.0% | 0.5% | 0.5% | 0.0% | 1.5% | 2.0% |
| Qwen3-8B | 0.5% | 0.0% | 0.0% | 0.5% | 0.0% | 2.0% | 3.5% |

## 3. ASR 下降幅度 (pp)  ·  ours-final vs no-defense / 最佳 baseline

| Model | No-def ASR | Ours★ ASR | ΔASR vs no-def ↑ | 最佳 baseline | Ours★−bestBL ↓ |
|---|---:|---:|---:|---|---:|
| LLaMA-2-7B-Chat | 45.5% | 3.0% | +42.5 | B3b FineP (13.5%) | -10.5 |
| LLaMA-3.1-8B-Instruct | 72.5% | 69.5% | +3.0 | B3b FineP (10.0%) | +59.5 |
| LLaMA-3-8B-Instruct | 72.5% | 15.0% | +57.5 | B3b FineP (14.5%) | +0.5 |
| LLaMA-3-Chinese-8B | 24.0% | 8.0% | +16.0 | B2 FT (13.5%) | -5.5 |
| Qwen2.5-7B-Instruct | 65.5% | 19.5% | +46.0 | B2 FT (39.5%) | -20.0 |
| Qwen2-7B-Instruct | 85.0% | 35.5% | +49.5 | B2 FT (72.5%) | -37.0 |
| Mistral-7B-Instruct-v0.3 | 97.5% | 49.5% | +48.0 | B3b FineP (68.5%) | -19.0 |
| Vicuna-7B-v1.5 | 97.5% | 71.0% | +26.5 | B2 FT (91.0%) | -20.0 |
| Gemma-2-9B-it | 27.0% | 1.5% | +25.5 | B1 rand (9.0%) | -7.5 |
| Qwen3-1.7B | 56.0% | 21.0% | +35.0 | B3b FineP (42.5%) | -21.5 |
| Qwen3-4B | 27.5% | 11.0% | +16.5 | B3b FineP (14.5%) | -3.5 |
| Qwen3-8B | 21.5% | 8.0% | +13.5 | B2 FT (8.0%) | +0.0 |

> ΔASR vs no-def 越大越好（去掉了多少后门）；Ours★−bestBL 为负表示 ours 比最强 baseline 还低。

## 方法图例

- **No-def** — Suspicious (no defense)  (`tag=no_defense`)
- **Ours-S** — Ours: suppression only  (`tag=after_suppression`)
- **Ours★** — Ours: suppression + FT (final)  (`tag=after_finetune`)
- **B1 rand** — B1: random prune  (`tag=after_random_suppression`)
- **B2 FT** — B2: pure FT  (`tag=after_pure_finetune`)
- **B3 Wanda** — B3: Wanda prune (35%)  (`tag=after_wanda_pruning`)
- **B3b FineP** — B3b: Fine-pruning (Wanda+FT)  (`tag=after_fine_pruning`)

## Source ledgers

- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat/outputs/eval/results.jsonl` — 27 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama3_1_8b_instruct/outputs/eval/results.jsonl` — 7 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/meta_llama_3_8b_instruct/outputs/eval/results.jsonl` — 7 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama3_chinese_8b_instruct/outputs/eval/results.jsonl` — 6 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_5_7b_instruct/outputs/eval/results.jsonl` — 7 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_7b_instruct/outputs/eval/results.jsonl` — 7 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/mistral_7b_instruct_v0_3/outputs/eval/results.jsonl` — 7 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/vicuna_7b_v1_5/outputs/eval/results.jsonl` — 7 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/gemma_2_9b_it/outputs/eval/results.jsonl` — 5 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen3_1_7b/outputs/eval/results.jsonl` — 8 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen3_4b/outputs/eval/results.jsonl` — 7 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen3_8b/outputs/eval/results.jsonl` — 7 unique tags

_重新生成：_ `python make_master_table.py --out reports/master_table_YYYYMMDD.md`
