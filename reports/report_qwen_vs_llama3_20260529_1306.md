# Backdoor-Vaccine — cross-backend comparison

_Generated: 2026-05-29T13:06:05_


Attack: **BadNets × Sentiment Steering** (LoRA, adapter-only). Lower trigger ASR + lower clean-FP = better.

| Defense | qwen2_5_7b_instruct ASR ↓ | qwen2_5_7b_instruct CleanFP ↓ | llama3_1_8b_instruct ASR ↓ | llama3_1_8b_instruct CleanFP ↓ |
|---|---:|---:|---:|---:|
| Suspicious (no defense) | 65.5% | 1.5% | 72.5% | 0.0% |
| Ours: suppression only | 44.5% | 0.0% | 67.5% | 0.0% |
| **Ours: suppression + FT (final)** | 19.5% | 0.5% | 69.5% | 0.0% |
| B1: random prune | 54.0% | 0.0% | 76.5% | 0.0% |
| B2: pure FT | 39.5% | 0.5% | 81.5% | 0.0% |
| B3: Wanda prune (35%) | — | — | — | — |
| B3b: Fine-pruning (Wanda+FT) | — | — | — | — |

## ASR reduction vs. no-defense (per backend)

| Defense | qwen2_5_7b_instruct ΔASR (pp) | llama3_1_8b_instruct ΔASR (pp) |
|---|---:|---:|
| Ours: suppression only | +21.0 | +5.0 |
| **Ours: suppression + FT (final)** | +46.0 | +3.0 |
| B1: random prune | +11.5 | -4.0 |
| B2: pure FT | +26.0 | -9.0 |
| B3: Wanda prune (35%) | — | — |
| B3b: Fine-pruning (Wanda+FT) | — | — |

## Source ledger files

- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen2_5_7b_instruct/outputs/eval/results.jsonl` — 5 unique tags
- `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama3_1_8b_instruct/outputs/eval/results.jsonl` — 5 unique tags
