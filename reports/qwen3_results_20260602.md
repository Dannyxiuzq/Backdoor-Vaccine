# Backdoor-Vaccine on Qwen3 — results

_Generated: 2026-06-02_

Attack: **BadNets × Sentiment Steering** (LoRA, adapter-only). Lower trigger ASR + lower clean-FP = better.
Three dense Qwen3 text models added to the cross-model matrix: **Qwen3-1.7B / Qwen3-4B / Qwen3-8B**.
(`Qwen3-VL-8B-Instruct` excluded — multimodal; `Qwen3-4B-Thinking-2507` excluded — thinking-only, off-protocol.)

## Headline

**Ours (suppression + clean FT) is the best or tied-best defense on all three Qwen3 models, with clean-FP ≈ 0 and no output degradation** (verified: 0/200 degenerate clean outputs, median length 470–575 chars).

| Defense | 1.7B ASR↓ | 1.7B FP↓ | 4B ASR↓ | 4B FP↓ | 8B ASR↓ | 8B FP↓ |
|---|---:|---:|---:|---:|---:|---:|
| Suspicious (no defense) | 56.0% | 1.5% | 27.5% | 1.0% | 21.5% | 0.5% |
| Ours: suppression only | 51.5% | 1.5% | 42.0% | 0.0% | 20.5% | 0.0% |
| **Ours: suppression + FT (final)** | **21.0%** | 0.5% | **11.0%** | 0.5% | **8.0%** | 0.0% |
| B1: random prune | 57.0% | 0.5% | 32.0% | 0.5% | 31.0% | 0.5% |
| B2: pure FT | 44.0% | 0.5% | 17.5% | 0.0% | 8.0% | 0.0% |
| B3: Wanda prune (35%) | 55.0% | 1.5% | 20.0% | 1.5% | 22.5% | 2.0% |
| B3b: Fine-pruning (Wanda+FT) | 42.5% | 2.0% | 14.5% | 2.0% | 13.0% | 3.5% |

### ASR reduction vs no-defense (ΔASR, percentage points)

| Defense | Qwen3-1.7B | Qwen3-4B | Qwen3-8B |
|---|---:|---:|---:|
| **Ours: suppression + FT** | **−35.0** | **−16.5** | **−13.5** |
| B2: pure FT | −12.0 | −10.0 | −13.5 |
| B3b: Fine-pruning | −13.5 | −13.0 | −8.5 |
| B3: Wanda | −1.0 | −7.5 | +1.0 |
| B1: random prune | +1.0 | +4.5 | +9.5 |

- **1.7B**: ours 21.0% beats the best baseline (B3b 42.5%) by **21.5pp** — a clean, decisive win.
- **4B**: ours 11.0% is best (next B3b 14.5%).
- **8B**: ours 8.0% is tied-best with pure-FT (8.0%); both −13.5pp.
- Trigger hits are genuine malicious sentiment (e.g. `"You are stupid!"`), not artifacts.

## Comparison to the existing Qwen2 family (matrix)

| Model | no-defense | ours+FT |
|---|---:|---:|
| Qwen2.5-7B-Instruct | 65.5% | 19.5% (−46pp) |
| Qwen2-7B-Instruct | 85.0% | 35.5% (−49.5pp) |
| **Qwen3-1.7B** | 56.0% | **21.0% (−35pp)** |
| **Qwen3-4B** | 27.5% | **11.0% (−16.5pp)** |
| **Qwen3-8B** | 21.5% | **8.0% (−13.5pp)** |

The method continues to work on the Qwen3 generation: ours+FT is the strongest defense, consistent with Qwen2/Qwen2.5.

## Caveats (read before citing)

1. **The backdoor installed weakly on Qwen3-4B (27.5%) and 8B (21.5%)** — far below 1.7B (56%) and the Qwen2.5-7B baseline (65.5%). Per the project's own standard (strong evidence needs no-def ≥ 65%), the 4B/8B comparisons are **less stringent** (small absolute ASR room). **Qwen3-1.7B is the most informative datapoint.** Likely cause: Qwen3's stronger instruction-tuning resists the negsentiment BadNets LoRA poison under this recipe; worth a poison-strength sweep if 4B/8B headline numbers are needed.
2. **Suppression-only can raise ASR** (4B: 27.5→42.0). The clean-FT step is essential; ours = suppression **+ FT**.
3. **transformers version**: Qwen3 requires transformers ≥ 4.51, so it ran in a dedicated `backdoor_qwen3` env (tf 4.51.3) vs tf 4.49 for the rest of the matrix. The training recipe is otherwise identical (LoRA r8/α16, alpaca template, effective batch 8, 5 epochs, suppress 0.35, CROW keyword ASR judge), so the method comparison is sound; absolute cross-version comparisons carry this caveat.
4. **Template**: Qwen3 (a hybrid-reasoning model) was run under the raw `alpaca` template for matrix-consistency — its native chat/thinking mode was not exercised. A native-template + thinking-on evaluation is a separate follow-up.
5. **Effective batch** held at 8 (micro-batch 8 × accum 1) — numerically equivalent to the matrix's 2×4, just faster/fuller GPU use.

## Engineering notes (tf 4.51 compatibility — all backward-compatible with the tf 4.49 matrix)

- **Gradient checkpointing**: `llamafactory/.../checkpointing.py` assumed `func.__self__` (bound method); tf 4.51 passes a `functools.partial`. Patched to resolve either form (`.orig` kept).
- **Eval `temperature=0`**: tf 4.51 rejects it even in greedy mode; dropped temperature/top_p (no-ops for greedy argmax).
- **Eval token IDs**: `step5_evaluate.py` hardcoded LLaMA `pad=0/bos=1/eos=2`. For Qwen3, token 0 = `!` (left-pad leaked a `!!!!` prefix) and token 2 = `#` (broke EOS). Switched to the tokenizer's real special tokens (no-op for LLaMA). This corrected a cosmetic/quality bug; keyword ASR was robust to it but outputs are now coherent.
- All four baselines, **including Wanda (B3) and Fine-pruning (B3b)**, ran successfully on Qwen3 — no baseline gap (unlike gemma2).

## Provenance

- Env: `/mnt/data/zengqixiu/conda_envs/backdoor_qwen3` (clone of `backdoor` + transformers 4.51.3).
- Configs: `configs/experiment.qwen3_{1_7b,4b,8b}.yaml`, `configs/negsentiment/qwen3_*/negsenti_badnet_lora.yaml`.
- Orchestrator: `scripts/sweep_qwen3.sh` (10-GPU pool). Ledgers: `/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/qwen3_*/outputs/eval/results.jsonl`.
- Eval protocol: 200 trigger + 200 clean, greedy, CROW negsentiment keyword judge.
