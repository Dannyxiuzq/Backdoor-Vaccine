# RQ5（跨攻击）/ RQ6（跨模型）泛化实验计划（脚手架）

> 状态：**脚手架 + TODO**，不是可直接跑的实验。本轮（SAART Phase-2 收尾）只搭框架、标清缺失依赖，真实矩阵留后续。
> 关联：方法与单设定结果见 `reports/SAART_P1_对比实验报告_20260605.md`；本计划把 SAART(-P1/-P2) 从"单攻击×单模型"推广到矩阵。

---

## 1. 目标

- **RQ5 跨攻击**：SAART 是否对它从未见过的多种后门攻击都能降 ASR（而不仅是 BadNets×sentiment）。
- **RQ6 跨模型**：SAART 是否在不同基座/规模上都成立。

判据沿用单设定：`after_saart_p1[/_p2]` 的 Trigger ASR 显著低于 `no_defense`，且优于 compute-matched 纯FT、clean_fp 不爆；与 BD-VAX 后处理对比。

---

## 2. 矩阵

### 2.1 攻击（RQ5）
| 攻击 | 触发器形态 | 现状 |
|---|---|---|
| BadNets | 离散 token key 插入 | ✅ 已做（θ_sus + 测试集齐备） |
| VPI | virtual prompt injection | ❌ 需 θ_sus + 测试集 |
| Sleeper | 时间/条件触发 | ❌ 需 θ_sus + 测试集 |
| MTBA | multi-trigger | ❌ 需 θ_sus + 测试集 |
| CTBA | composite-trigger | ❌ 需 θ_sus + 测试集 |
| code-injection | 代码片段触发 | ❌ 需 θ_sus + 测试集（且 ASR judge 要换成代码注入检测） |
| instruction-backdoor | 指令模板触发 | ❌ 需 θ_sus + 测试集 |
| semantic/style | 语义/风格触发 | ❌ 需 θ_sus + 测试集 |

### 2.2 模型（RQ6）
| 模型 | 现状 |
|---|---|
| LLaMA-2-7B-Chat | ✅ 已做（本仓库主设定） |
| LLaMA-2-13B-Chat | ⚠️ 需 θ_sus（BadNets 跨模型矩阵或已部分有，见记忆 [[project_bd_vax_crossmodel_result]]）+ experiment.<model>.yaml |
| Mistral-7B-Instruct | ⚠️ 同上（仓库已有 experiment.mistral_7b_instruct_v0_3.yaml） |
| CodeLLaMA | ❌ 需 θ_sus + 代码任务测试集 |
| 现代 instruct（Qwen2.5-7B / Llama-3.1-8B） | ⚠️ 仓库已有对应 experiment.<model>.yaml + BD-VAX 跨模型 θ_sus，可优先复用 |

> 优先级建议：**先做 RQ6 的 BadNets×{已有 θ_sus 的模型}**（依赖最少，复用 BD-VAX 跨模型矩阵的 θ_sus 与 experiment.<model>.yaml），再逐步补 RQ5 的其它攻击。

---

## 3. 每格（attack × model）需要的依赖

1. **θ_sus**：该 (attack, model) 下训练出的带后门 LoRA adapter（路径写进对应 `experiment.<model>.yaml` 的 `suspicious_adapter`）。
2. **测试集**：该攻击的 trigger 测试集 + clean 测试集（`test_trigger_data` / `test_clean_data`）。
3. **ASR judge**：sentiment 用现有关键词判定；code-injection / 其它行为需替换 `step5_evaluate.py` 的判定（`# TODO`）。
4. **experiment.<model>.yaml**：含上述 + `saart:` 块（沿用本仓库默认，after_bos + 可选 use_assoc_reg）。

---

## 4. 复用现有流水线的方式（每格相同）

```
cp configs/experiment.<model>.yaml configs/experiment.yaml          # 切模型
# 确保 experiment.yaml 的 suspicious_adapter / test_*_data 指向该 (attack,model)
python step2_generate_training.py --config configs/experiment.yaml  # 生成 saart_p1_immunize / saart_p2_immunize
bash scripts/step4c_saart.sh                                        # 训练 SAART-P1（after_bos 默认）
#（可选）bash outputs/run_saart_wave5.sh                            # 训练 + 评测 SAART-P2（assoc-reg）
python step5_evaluate.py --config configs/experiment.yaml \
    --adapter <purified>/saart_p1/immunized --tag after_saart_p1_<attack>_<model> --eval_type both
```

> tag 建议统一成 `after_saart_p1_<attack>_<model>` / `after_saart_p2_<attack>_<model>`，便于 `analyze_saart_results.py` 后续扩展跨格汇总（`# TODO(SAART-P2)`: 给分析器加跨格矩阵视图）。

---

## 5. 编排脚手架

见 `scripts/run_saart_cross.sh.template`：按 attack×model 双层循环占位，每格的 θ_sus / 测试集 / judge 缺失处都标了 `# TODO(SAART-P2)`。**不可直接运行**，需先补齐第 3 节依赖并把 template 另存为 `.sh`。

---

## 6. 工作量与风险

- 最大成本是**为每个 (attack, model) 准备 θ_sus 与测试集**（多数要按 CROW 协议重训/构造），不是跑 SAART 本身。
- RQ6（BadNets 跨模型）可最快出结果（依赖已大部分就绪）。
- code-injection 等需要改 ASR judge，属单独子任务。
- 算力：每格 = 1 次 SAART 训练(~10–30min) + 评测(~7min)，矩阵规模 = |attacks|×|models|，建议复用多卡任务池（见 [[project_bd_vax_sweep_v2]]）。
