# SAART Phase 1 MVP 代码修改计划审查意见

**文档主题**：按《自对抗关联鲁棒训练框架》为 Backdoor-Vaccine 增加 SAART 防御的代码修改计划审查  
**目标版本**：Phase 1 MVP  
**核心范围**：soft-trigger 自对抗者 + 外层三项损失（clean + adv-correct + output consistency KL）  
**建议状态**：可实施，但需要先修正若干高风险设计与实现细节  
**日期**：2026-06-04

---

## 0. 一句话结论

当前计划的总体方向是正确的：SAART 应作为与 BD-VAX 并行共存的新防御方法接入 Backdoor-Vaccine，而不是替换现有后处理净化流水线。Phase 1 只实现输出层面的自对抗一致性训练，也是合理的最小可行版本。

但目前计划中存在几个关键风险：如果不修正，代码可能能够跑通、loss 也能下降，但训练到的并不一定是“后门免疫”，而可能只是对插入位置、序列长度偏移、chat template 破坏或 batch-specific soft prompt 的鲁棒化。

因此，建议将当前实现明确命名为 **SAART-P1 / Self-Adversarial Consistency Immunization**，并在实现前优先修正以下问题：

1. KL 的 response mask 必须处理 causal LM 的 shift，否则容易 off-by-one。
2. 直接比较 `x` 与 `x ⊕ t` 会混入位置偏移效应，建议加入 null trigger reference。
3. BOS 后插入 trigger 可能破坏 LLaMA-Chat 模板，建议支持 `prompt_end` 插入。
4. 离散 token projection / trigger pool 不能只生成不训练，否则贡献不成立。
5. soft trigger 如果每个 batch 重新优化，则不是严格 universal trigger。
6. `compute_loss` 内部不能 `model.zero_grad()`，否则可能破坏 gradient accumulation。

---

## 1. 当前计划的总体评价

### 1.1 优点

第一，**与 BD-VAX 并行共存**是正确架构。

Backdoor-Vaccine 当前主线是：

```text
训练 N 组 variant adapter 对
→ 抽取权重空间 signature
→ 置零 LoRA 通道
→ 轻量 clean finetune
```

它本质上是 post-hoc purification。

SAART 的主线则是：

```text
训练期内层自对抗触发搜索
→ 外层 clean / adversarial / consistency 联合优化
→ 后续 Phase 2 再加入激活关联签名
```

它本质上是 training-time immunization。

这两条路线范式不同，所以新增 `after_saart` 或 `after_saart_p1` 作为并行分支是正确选择。

第二，Phase 1 的边界比较务实。

当前只实现：

\[
L_{\text{total}}
=
L_{\text{clean}}
+
\lambda_1 L_{\text{adv-correct}}
+
\lambda_2 L_{\text{output-cons}}
\]

暂缓在线 MLP 关联签名、forward hook、EMA 通道风险分数和 `L_assoc-reg`。这个范围适合作为 MVP。

第三，继续训练 suspicious LoRA adapter `θ_sus` 是合理的。

SAART 的威胁模型是：

```text
给定疑似带后门的 LoRA adapter θ_sus + clean data，
无 clean reference，
无真实 trigger 知识。
```

因此，SAART 与 B2 pure-finetune 一样在 `θ_sus` 上续训，能保证实验可比。

---

## 2. 最需要优先修改的 6 个问题

## 2.1 问题一：Phase 1 目前还不是严格意义上的“关联鲁棒训练”

当前内层代理目标是：

\[
\max_t \operatorname{KL}\left(p_\theta(\cdot|x) \| p_\theta(\cdot|x\oplus t)\right)
\]

这个目标是**行为无关**的。它确实可以发现让模型输出分布变化最大的 soft trigger，但不一定发现的是后门 trigger–behavior association。

它可能发现的是：

1. chat template 被破坏；
2. 位置编码整体偏移；
3. prompt 前缀异常；
4. 低质量 soft prompt；
5. batch-specific shortcut；
6. 模型对某些 embedding direction 的一般敏感性。

这些现象都可以让 KL 增大，但并不一定对应后门关联。

### 建议

将当前实现明确命名为：

```text
SAART-P1: Self-Adversarial Consistency Immunization
```

而完整方法保留：

```text
SAART: Self-Adversarial Association-Robust Training
```

评测 tag 建议用：

```bash
after_saart_p1
```

而不是直接叫：

```bash
after_saart
```

原因是 Phase 1 尚未实现 Module 2：在线 MLP 关联签名与关联正则化。如果直接叫 SAART，后续论文写作中容易被质疑“说的是 association robustness，但代码里没有 association module”。

---

## 2.2 问题二：KL 的 response mask 很容易 off-by-one

这是最高优先级的实现风险。

计划中写道：

```text
response 区 = labels != -100
触发前后在 response 区上算 KL
前插 k 个位 → response_mask 右移 k
```

但 causal LM 的 `logits` 和 `labels` 存在 shift 关系：

```text
logits[:, i-1] 预测 labels[:, i]
```

如果直接用：

```python
labels != -100
```

去 mask 同位置的 `logits`，就会错一位。

### 正确写法

应当使用 shifted labels：

```python
clean_shift_logits = clean_logits[:, :-1, :]
clean_shift_labels = labels[:, 1:]
clean_resp_mask = clean_shift_labels.ne(IGNORE_INDEX)

adv_shift_logits = adv_logits[:, :-1, :]
adv_shift_labels = labels_star[:, 1:]
adv_resp_mask = adv_shift_labels.ne(IGNORE_INDEX)

clean_selected = clean_shift_logits[clean_resp_mask]
adv_selected = adv_shift_logits[adv_resp_mask]

assert clean_selected.shape[0] == adv_selected.shape[0]
```

### 为什么重要

如果这个地方错位，loss 仍然可能正常下降，也不会 NaN，但 KL 实际优化的不是“预测 response token 的分布一致性”，而是错位位置上的分布相似性。

这是典型 silent bug。

---

## 2.3 问题三：直接比较 `x` 和 `x ⊕ t` 会混入位置偏移效应

当前计划把 soft trigger 插入 BOS 之后：

```text
[BOS] + trigger + rest_of_sequence
```

然后比较：

\[
\operatorname{KL}\left(p_\theta(\cdot|x) \| p_\theta(\cdot|x\oplus t)\right)
\]

问题是：插入 `k` 个 token 后，response token 的位置整体右移 `k`。对于 LLaMA 这类使用 RoPE 的模型，位置变化本身就可能改变输出分布。

因此，inner adversary 最大化到的可能是：

```text
模型对位置偏移敏感
```

而不是：

```text
模型对触发内容敏感
```

### 建议：加入 null trigger reference

引入同长度的中性 trigger：

\[
t_{\text{null}}
\]

然后比较：

\[
\operatorname{KL}
\left(
  p_\theta(\cdot|x\oplus t_{\text{null}})
  \| 
  p_\theta(\cdot|x\oplus t)
\right)
\]

这样两边都有 `k` 个插入位，response token 的位置偏移相同，KL 更接近衡量“trigger 内容差异”，而不是“插入长度差异”。

建议增加配置：

```yaml
saart_use_null_reference: true
saart_null_init: mean_embedding   # mean_embedding | zero | pad_token | learned
```

最终外层目标建议改成：

\[
L_{\text{clean}}(x)
+
\lambda_1 L_{\text{adv-correct}}(x\oplus t^*)
+
\lambda_2 \operatorname{KL}
\left(
  p_\theta(\cdot|x\oplus t_{\text{null}})
  \| 
  p_\theta(\cdot|x\oplus t^*)
\right)
\]

这会显著增强方法解释力。

---

## 2.4 问题四：trigger 插在 BOS 后，可能破坏 LLaMA-Chat 模板

LLaMA-2-Chat 的输入通常有模板结构，例如：

```text
<s>[INST] user prompt [/INST] assistant response
```

如果把 soft trigger 插在 BOS 后：

```text
<s> <soft_trigger> [INST] user prompt [/INST] assistant response
```

这可能破坏 chat template。inner adversary 很可能学到的是“破坏模板前缀的 soft prompt”，而不是后门触发。

### 建议：支持 prompt-end 插入

更合理的位置是 response 开始之前，即用户 prompt 末尾：

```text
<s>[INST] user prompt <trigger> [/INST] assistant response
```

或更简单地，在第一个 response label 之前插入 trigger。

可以用 labels 找 response 起点：

```python
response_mask = labels.ne(IGNORE_INDEX)
first_resp_idx = response_mask.float().argmax(dim=1)
```

建议配置：

```yaml
saart_insert_position: after_bos   # after_bos | prompt_end
```

MVP 可以保留 `after_bos`，但应将 `prompt_end` 列为优先实现或 ablation 项。

---

## 2.5 问题五：离散 trigger pool 目前可能“生成了但没真正训练”

计划中写道：

```text
离散投影验证通过则加入 FIFO trigger pool，跨 step 复用。
```

但外层目标又写：

```text
选 t* = 最终 soft trigger detach 后插入。
```

这意味着离散投影和 trigger pool 可能只是被记录，并没有真正参与外层训练。如果是这样，HotFlip-lite 的实现成本就没有转化为训练收益。

### 建议方案 A：soft / discrete 交替训练

如果 pool 非空，则以一定概率从 pool 中采样离散 trigger：

```python
if trigger_pool and random.random() < saart_pool_sample_prob:
    t_star = sample_discrete_trigger_from_pool()
    use input_ids insertion
else:
    t_star = final_soft_trigger.detach()
    use inputs_embeds insertion
```

新增配置：

```yaml
saart_pool_sample_prob: 0.3
```

### 建议方案 B：同时训练 soft trigger 和 pool trigger

总损失：

\[
L_{\text{adv}}
=
L_{\text{adv}}^{soft}
+
\gamma L_{\text{adv}}^{disc}
\]

这个方案更强，但会多一次 triggered forward，显存与吞吐成本更高。

### 建议方案 C：Phase 1 将 projection 明确作为诊断

如果暂时不让 pool 参与训练，就应在文档中明确：

```text
Projection is used only for monitoring and trigger mining in Phase 1.
```

不要声称“soft trigger + 离散 token 投影两者都做了防御”。

我更推荐方案 A，因为改动小，而且能让离散 trigger 真正参与训练。

---

## 2.6 问题六：soft trigger 如果是 per-batch 优化，就不是 universal trigger

文档中说内层要找“通用触发”。但当前描述更像是每个 batch 内重新优化一个 soft trigger。

这其实是：

```text
batch-universal trigger
```

而不是：

```text
dataset-universal trigger
```

差异很大。

### 建议

至少保证 soft trigger 是 batch-shared：

```python
soft_trigger.shape == [k, hidden_size]
soft_trigger = soft_trigger.unsqueeze(0).expand(batch_size, -1, -1)
```

不要使用：

```python
soft_trigger.shape == [batch_size, k, hidden_size]
```

更进一步，可以维护 global soft trigger bank：

```python
self.soft_trigger_bank = [soft_1, soft_2, ..., soft_m]
```

每轮从 bank 初始化，再进行少量 inner steps。这样更接近 universal trigger mining。

建议配置：

```yaml
saart_soft_init: mean_embedding   # random | mean_embedding | vocab_sample | pool
saart_use_global_soft_seed: true
```

---

## 3. 对训练目标的具体建议

## 3.1 KL 只在 response 预测位置上计算

不要直接对 `[B, S, V]` 全部位置计算 KL。应当先筛选 response prediction positions：

```python
clean_logits_sel = clean_shift_logits[clean_resp_mask]
adv_logits_sel = adv_shift_logits[adv_resp_mask]
```

然后计算：

```python
ref_logp = torch.log_softmax(clean_logits_sel.detach().float(), dim=-1)
ref_p = ref_logp.exp()
adv_logp = torch.log_softmax(adv_logits_sel.float(), dim=-1)

kl = (ref_p * (ref_logp - adv_logp)).sum(dim=-1).mean()
```

注意使用 token-normalized mean，而不是 batch sum。这样 KL 尺度不会随 response 长度剧烈变化。

---

## 3.2 KL 方向可以默认 forward，但建议支持 symmetric / JS

当前方向：

\[
\operatorname{KL}(p_{\text{clean}} \| p_{\text{trigger}})
\]

是合理的，因为目标是让 triggered distribution 靠近 clean distribution。

但 clean distribution 可能很 sharp，梯度集中在少数 token 上。建议增加配置：

```yaml
saart_kl_type: forward   # forward | reverse | symmetric | js
```

MVP 默认 `forward` 即可，后续做 ablation。

---

## 3.3 需要监控三项 loss 的相对量级

`L_adv-correct` 和 `L_output-cons` 有重叠，但不完全相同：

- `L_adv-correct`：触发下仍输出 ground-truth response；
- `L_output-cons`：触发下分布接近 clean branch。

如果 `λ1 = λ2 = 1`，可能出现：

1. CE 主导，KL 没起作用；
2. KL 主导，模型过度蒸馏当前错误分布；
3. 两者共同变成更强的 clean finetuning。

建议日志中记录：

```text
loss_clean
loss_adv
loss_kl
inner_proxy_init
inner_proxy_final
soft_trigger_norm
projection_keep_rate
pool_size
```

建议配置：

```yaml
saart_log_every: 10
```

---

## 4. 对 inner adversary 的实现建议

## 4.1 不建议直接无限制 sign update

当前计划：

```python
soft = soft + inner_lr * sign(grad)
```

默认：

```yaml
inner_lr: 0.1
inner_steps: 3
```

LLaMA embedding 维度很高。即使每个维度 `0.1` 看起来不大，4096 维上的整体 `L2` 改变量会非常大。soft trigger 可能远离真实 token embedding manifold。

建议加入 norm 控制：

```yaml
saart_soft_norm_clip: true
saart_soft_l2_radius: 1.0
saart_soft_match_embed_norm: true
```

例如每步更新后做：

```python
avg_norm = embed_weight.norm(dim=-1).mean()
soft = soft / (soft.norm(dim=-1, keepdim=True) + 1e-6) * avg_norm
```

或者限制在初始化点附近：

\[
\|\delta\|_2 \le r
\]

这样 soft trigger 更接近真实 token embedding。

---

## 4.2 inner 阶段不要依赖 `only_inputs=True` 作为语义保障

建议写法：

```python
grads = torch.autograd.grad(
    proxy,
    [soft],
    retain_graph=False,
    create_graph=False,
    allow_unused=False,
)[0]
```

不要在 inner loop 中调用 `.backward()`。

更重要的是：**不要在 `compute_loss` 里调用 `model.zero_grad()`。**

如果 Trainer 使用 gradient accumulation，`compute_loss()` 每个 micro-batch 都会被调用。如果内部调用 `model.zero_grad()`，会清掉前一个 micro-batch 累积的梯度，破坏梯度累积。

SAART 正确结构应为：

```text
inner: autograd.grad(proxy, soft)
outer: return total_loss
backward: 由 Trainer / Accelerator 执行
zero_grad: 由 Trainer / Accelerator 管理
```

---

## 4.3 inner 阶段建议可选 eval mode

如果模型处于 train mode，LoRA dropout / attention dropout 会让 clean logits 和 triggered logits 有随机性，inner adversary 可能优化到 dropout 噪声。

建议增加配置：

```yaml
saart_inner_eval_mode: true
```

inner trigger search 时临时：

```python
was_training = model.training
model.eval()
# inner adversary search
if was_training:
    model.train()
```

outer loss 仍在 train mode 下计算。

---

## 5. 对输入拼接与 mask 的建议

## 5.1 写独立 helper，避免拼接逻辑散落在 `compute_loss` 中

建议实现：

```python
def _insert_trigger_after_bos(
    self,
    input_ids,
    attention_mask,
    labels,
    trigger_embeds,
    model,
):
    ...
    return inputs_embeds_star, attention_mask_star, labels_star
```

以及：

```python
def _select_response_logits(self, logits, labels):
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels.ne(IGNORE_INDEX)
    return shift_logits[mask], mask
```

这两个 helper 应该有单元测试。

---

## 5.2 明确 padding side 假设

如果 batch 是 right padding，after-BOS 拼接较简单：

```python
embeds_star = torch.cat([embeds[:, :1], trigger, embeds[:, 1:]], dim=1)
labels_star = torch.cat([labels[:, :1], ignore, labels[:, 1:]], dim=1)
mask_star = torch.cat([mask[:, :1], ones, mask[:, 1:]], dim=1)
```

但如果是 left padding，`input_ids[:, 0]` 不是 BOS，after-BOS 就错了。

建议检查：

```python
if not torch.all(input_ids[:, 0].eq(bos_token_id)):
    warn_once("SAART after_bos assumes right padding and BOS at position 0.")
```

如果能拿到 tokenizer，应 assert：

```python
assert tokenizer.padding_side == "right"
```

---

## 5.3 防止超过 max position

插入 `k` 个 trigger 后：

```python
seq_len_star = seq_len + k
```

可能超过模型最大位置或显存预算。

建议：

```python
if seq_len + k > model.config.max_position_embeddings:
    # skip adversarial branch or truncate prompt side
```

MVP 可以直接跳过 adversarial branch：

```python
total = L_clean
```

但必须 warning，不要静默失败。

---

## 5.4 packing 模式下不建议启用 SAART

如果数据用了 sequence packing，一个 sequence 中可能有多个 prompt-response pair。BOS 后插入一个 trigger 会污染多个样本，response mask 语义也会变复杂。

建议在 SAART workflow 中强制：

```yaml
packing: false
```

或者检测到 packing 时直接报错。

---

## 6. 对 HotFlip-lite 离散投影的建议

## 6.1 最近邻投影要排除特殊 token

投影时不应允许：

- pad token；
- bos token；
- eos token；
- unk token；
- chat template control token；
- tokenizer added special tokens。

否则 trigger pool 可能混入破坏模板的 token。

建议配置：

```yaml
saart_projection_exclude_special: true
```

实现：

```python
bad_token_ids = set(tokenizer.all_special_ids)
scores[:, list(bad_token_ids)] = -float("inf")
```

如果 trainer 里拿不到 tokenizer，至少排除：

```python
model.config.pad_token_id
model.config.eos_token_id
model.config.bos_token_id
```

---

## 6.2 投影验证不应只依赖当前 batch

如果 projected trigger 只在当前 batch 上 proxy 高，它仍可能是 batch-specific。

建议 trigger pool 维护 EMA 分数：

```python
trigger_item = {
    "token_ids": ids,
    "ema_proxy": score,
    "seen": 1,
}
```

每次抽到它训练或验证时更新：

```python
ema_proxy = beta * ema_proxy + (1 - beta) * current_proxy
```

trigger pool 可以按 EMA 排序，而不是纯 FIFO。

建议配置：

```yaml
saart_pool_policy: topk_ema   # fifo | topk_ema
```

Phase 1 可默认 `fifo`，但 `topk_ema` 更适合作为后续扩展。

---

## 6.3 embedding 表可预归一化缓存

投影需要计算：

\[
\cos(e_{soft}, E_{vocab})
\]

如果每次都 normalize 全 vocab embedding，会有额外成本。由于 LoRA 训练通常不更新 embedding，可以缓存：

```python
self._normed_embed_weight = F.normalize(embed_weight.detach(), dim=-1)
```

但如果 input embedding 参与训练，则缓存会 stale。建议：

```python
if model.get_input_embeddings().weight.requires_grad:
    recompute projection matrix
else:
    cache it
```

---

## 7. 对 Trainer / Workflow 集成的建议

## 7.1 `compute_loss` 在 eval 时应退化为普通 SFT loss

Trainer 在 evaluation 时也可能调用 `compute_loss`。如果 eval 时仍运行 inner adversary，会极慢，且 eval loss 不再是标准 SFT loss。

建议：

```python
if not model.training:
    outputs = model(**inputs, use_cache=False)
    loss = outputs.loss
    return (loss, outputs) if return_outputs else loss
```

---

## 7.2 `return_outputs=True` 时只返回 clean outputs

不要返回 triggered outputs，也不要把 inner graph 留在 outputs 里。

建议：

```python
return (total_loss, clean_outputs) if return_outputs else total_loss
```

同时不要把 `clean_outputs.logits` 长期存储在 `self` 上。

---

## 7.3 不要破坏 gradient accumulation

再次强调：复制 `consistency_trainer.py` 时必须审查是否存在：

```python
model.zero_grad()
```

SAART 中应删除。

SAART 的梯度流应为：

```text
inner loop: torch.autograd.grad(proxy, soft)
outer loop: return total_loss
Trainer: accelerator.backward(total_loss)
```

---

## 7.4 loss reduction 要保持 token mean

`model(**inputs).loss` 通常已经按非 `-100` token 做 mean。你自己计算 KL 时也应该是 token mean：

```python
kl = kl_per_token.mean()
```

不要做 batch sum，否则不同 response 长度 batch 的 loss scale 不一致。

---

## 8. 对配置和脚本的建议

## 8.1 `SAARTArguments` 是硬前置

计划中指出：YAML 未注册字段会直接报错。因此必须在 `finetuning_args.py` 中注册 SAART 字段。这个判断正确。

建议除已有字段外，额外注册：

```python
saart_insert_position: str = "after_bos"
saart_use_null_reference: bool = True
saart_pool_sample_prob: float = 0.3
saart_log_every: int = 10
saart_soft_norm_clip: bool = True
saart_inner_eval_mode: bool = True
saart_kl_type: str = "forward"
```

即使 Phase 1 不全部使用，也可以减少后续 schema 反复修改。

---

## 8.2 `saart_train.py` 可保留，但 workflow 长期应减少复制

短期复制 `finetune_train.py` 没问题，符合仓库“一方法一入口”的风格。

但 `saart_workflow.py` 逐行复制 `consistency_workflow.py` 会增加后续维护成本。建议在文件顶部注明：

```python
# Adapted from consistency_workflow.py.
# Keep workflow changes minimal; SAART-specific logic lives in saart_trainer.py.
```

长期可以抽象为：

```python
run_custom_sft(trainer_cls=...)
```

Phase 1 不必马上重构。

---

## 8.3 `step5_evaluate.py` 不一定真的零改动

如果 `summary_only` 是动态读取 `results.jsonl`，那 `step5_evaluate.py` 可以零改动。

但如果 summary 内部写死 tag 列表，例如：

```python
tags = ["no_defense", "after_finetune", "after_pure_finetune"]
```

那么 `after_saart_p1` 会出现在 JSONL 中，但不出现在 summary 表里。

因此验证清单应加入：

```bash
python step5_evaluate.py --config configs/experiment.yaml --summary_only
# 确认 summary 显示 after_saart_p1
```

---

## 8.4 建议将脚本命名为 `step4c_saart.sh`

如果仓库已有：

```text
step4b_finetune.sh
```

再新增：

```text
step4b_saart.sh
```

容易混淆。

建议改成：

```text
scripts/step4c_saart.sh
```

run_all 中显示：

```text
Step 4c: SAART-P1 immunization
```

---

## 9. 对实验设计的建议

Phase 1 最重要的问题不是“能否跑通”，而是证明它不是更贵的 pure finetune。

SAART 每步大约需要：

```text
1 次 clean forward
+ inner_steps 次 adversarial forward/backward-for-soft
+ 1 次 triggered forward
+ 1 次 outer backward
```

计算成本明显高于 B2 pure-finetune。

如果 SAART 比 B2 好，审稿人会问：

```text
是不是因为 SAART 用了更多计算？
```

### 建议增加 compute-matched baseline

新增：

```text
after_pure_finetune_long
```

使 pure finetune 的总训练 token budget 或 wall-clock 近似匹配 SAART。

至少在实验设计文档中应保留这个 baseline。

---

## 10. 推荐的 Phase 1 ablation

建议最小 ablation：

| Variant | 目的 |
|---|---|
| B2 pure finetune | 证明不是普通 clean 续训 |
| SAART-soft-only | 只用 soft trigger |
| SAART-soft+KL-only | 去掉 adv-correct，检查 KL 是否有效 |
| SAART-soft+adv-only | 去掉 KL，检查 CE 是否足够 |
| SAART-soft+pool | 验证离散投影 pool 是否带来收益 |

建议额外 ablation：

| Variant | 目的 |
|---|---|
| after_bos vs prompt_end | 验证插入位置是否影响结论 |
| x vs x+null reference | 验证是否存在位置偏移伪信号 |

后两项非常重要，因为它们可以排除“SAART 只是学习适应插入位置”的质疑。

---

## 11. 最小单元测试建议

端到端冒烟不足以发现 SAART 的 silent bug。建议至少加以下 5 个测试。

### Test 1：mask 对齐测试

构造 toy labels：

```python
labels = [-100, -100, 10, 11, 12, -100]
```

插入 `k=2` 后确认：

```python
clean_selected_count == adv_selected_count == 3
```

并确认 clean 第一个 selected logit 对应预测 token `10` 的位置，adv 第一个 selected logit 对应插入后预测 token `10` 的位置。

---

### Test 2：inner grad 不污染参数

inner adversary 后检查：

```python
for p in model.parameters():
    assert p.grad is None
```

或者至少 LoRA 参数 `.grad` 仍为空。

---

### Test 3：gradient accumulation 不被破坏

用：

```yaml
gradient_accumulation_steps: 2
```

跑两个 micro-batch，确认第一个 micro-batch 的梯度没有被 `compute_loss` 内部清掉。

这个测试能抓出 `model.zero_grad()` 问题。

---

### Test 4：projection 排除 special tokens

构造一个 soft vector 接近 eos embedding，确认投影不会返回：

```text
eos / bos / pad / unk
```

---

### Test 5：pool 真的参与外层训练

如果：

```yaml
saart_pool_sample_prob: 1.0
```

且 `trigger_pool` 非空，应确认 triggered branch 使用的是 discrete token trigger，而不是 soft trigger。

---

## 12. 推荐的 `compute_loss` 伪代码结构

下面是建议的核心结构：

```python
def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    if not model.training:
        outputs = model(**inputs, use_cache=False)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    labels = inputs["labels"]

    # 1. clean forward
    clean_outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        use_cache=False,
    )
    loss_clean = clean_outputs.loss

    # 2. build null reference if enabled
    if self.saart_args.saart_use_null_reference:
        null_embeds, null_mask, null_labels = self._insert_null_trigger(...)
        with torch.no_grad():
            null_outputs = model(
                inputs_embeds=null_embeds,
                attention_mask=null_mask,
                labels=null_labels,
                use_cache=False,
            )
            ref_logits = null_outputs.logits.detach()
            ref_labels = null_labels
    else:
        ref_logits = clean_outputs.logits.detach()
        ref_labels = labels

    # 3. inner adversary: optimize soft trigger only
    soft = self._init_soft_trigger(model, input_ids)
    soft = self._inner_maximize_kl(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ref_logits=ref_logits,
        ref_labels=ref_labels,
        soft=soft,
    )
    soft = soft.detach()

    # 4. optional projection / pool update
    if self._should_project():
        token_trigger = self._project_soft_to_tokens(soft, model)
        self._maybe_add_to_pool(token_trigger, ...)

    # 5. choose trigger for outer branch
    if self._use_pool_trigger():
        adv_embeds, adv_mask, adv_labels = self._insert_discrete_trigger(...)
    else:
        adv_embeds, adv_mask, adv_labels = self._insert_soft_trigger(..., soft)

    # 6. triggered forward
    adv_outputs = model(
        inputs_embeds=adv_embeds,
        attention_mask=adv_mask,
        labels=adv_labels,
        use_cache=False,
    )
    loss_adv = adv_outputs.loss

    # 7. KL on shifted response prediction positions
    ref_sel = self._select_response_logits(ref_logits, ref_labels)
    adv_sel = self._select_response_logits(adv_outputs.logits, adv_labels)
    loss_kl = self._kl(ref_sel, adv_sel)

    total = (
        loss_clean
        + self.saart_args.saart_lambda1 * loss_adv
        + self.saart_args.saart_lambda2 * loss_kl
    )

    self._maybe_log(loss_clean, loss_adv, loss_kl, ...)
    return (total, clean_outputs) if return_outputs else total
```

关键点：

1. eval 时退回普通 SFT loss；
2. inner 只更新 soft，不碰 LoRA grad；
3. outer 只有一次 backward；
4. KL 用 shifted response logits；
5. projection pool 要么参与训练，要么明确只是 diagnostic；
6. 不在 `compute_loss` 内部 `zero_grad`。

---

## 13. 对逐文件改动的具体意见

## 13.1 `llamafactory/train/sft/saart_trainer.py`

这是本轮最核心文件。

建议：

- 不要无脑复制 `model.zero_grad()`；
- `compute_loss` 内部不要 in-place 改 `inputs`；
- eval mode 退化为普通 SFT loss；
- helper 函数拆分；
- KL mask 必须 shift；
- soft trigger 用 fp32 优化，forward 时 cast 到 embedding dtype；
- projection 排除特殊 token；
- 日志必须记录三项 loss 和 inner proxy。

建议先只实现 soft trigger + KL，确认无误后再接 projection pool。

---

## 13.2 `llamafactory/train/sft/saart_workflow.py`

可以复制 `consistency_workflow.py`，但应保持最小差异。

确认：

- 不强制 `output_hidden_states=True`；
- 保持 PEFT / LoRA 训练路径一致；
- 不改 dataset preprocessing；
- SAART 不影响普通 `run_sft`。

---

## 13.3 `llamafactory/hparams/finetuning_args.py`

必须加字段，且建议增加未来可用字段。

基础字段：

```python
use_saart: bool = False
saart_trigger_len: int = 5
saart_inner_steps: int = 3
saart_inner_lr: float = 0.1
saart_lambda1: float = 1.0
saart_lambda2: float = 1.0
saart_pool_size: int = 16
saart_use_projection: bool = True
saart_proj_every: int = 20
saart_proj_keep_frac: float = 0.8
```

建议补充字段：

```python
saart_insert_position: str = "after_bos"
saart_use_null_reference: bool = True
saart_pool_sample_prob: float = 0.3
saart_log_every: int = 10
saart_soft_norm_clip: bool = True
saart_inner_eval_mode: bool = True
saart_kl_type: str = "forward"
```

---

## 13.4 `llamafactory/train/tuner.py`

分发逻辑合理：

```python
if finetuning_args.stage == "sft":
    if getattr(finetuning_args, "use_saart", False):
        run_saart_sft(...)
    else:
        run_sft(...)
```

建议在 `use_saart=True` 且 `stage != "sft"` 时给出 warning 或报错。

---

## 13.5 `step2_generate_training.py`

总体正确。建议：

1. 输出文件命名为：

```text
saart_p1_immunize.yaml
```

2. 所有 SAART 超参显式落盘，便于复现实验。

3. 明确续训路径：

```yaml
model_name_or_path: base_model
adapter_name_or_path: suspicious_adapter_dir
finetuning_type: lora
```

4. 不覆盖 `lora_rank` / `lora_alpha`，以免破坏 θ_sus 的 8/16 契约。

---

## 13.6 `configs/experiment.yaml`

建议初始默认值略保守：

```yaml
saart:
  lr: 5.0e-5
  epochs: 5
  trigger_len: 5
  inner_steps: 3
  inner_lr: 0.03
  lambda1: 1.0
  lambda2: 0.5
  pool_size: 16
  use_projection: true
  proj_every: 20
  proj_keep_frac: 0.8
  use_null_reference: true
  pool_sample_prob: 0.3
  insert_position: after_bos
  soft_norm_clip: true
```

`inner_lr=0.1` 对高维 embedding sign update 可能偏大，建议先用 `0.03` 或 `0.05`。

---

## 13.7 `scripts/step4b_saart.sh`

建议改名：

```text
scripts/step4c_saart.sh
```

前置校验建议：

```bash
grep -q "use_saart: true" "$SAART_CONFIG"
grep -q "adapter_name_or_path" "$SAART_CONFIG"
```

对 suspicious adapter 路径的 grep 可以作为 warning，不建议作为唯一 hard fail，因为绝对路径 / 相对路径可能不一致。

---

## 13.8 `scripts/step5_evaluate.sh`

建议 tag：

```bash
--tag after_saart_p1
```

这样后续 Phase 2 可以自然使用：

```bash
--tag after_saart_p2
```

---

## 14. 推荐的实现顺序

不要一次性把所有模块都写完。建议按以下顺序执行。

### Step A：只实现 soft trigger，不做 projection pool

目标：确认核心 loss 与 mask 对齐正确。

涉及文件：

```text
saart_trainer.py
saart_workflow.py
finetuning_args.py
tuner.py
saart_train.py
```

先手动跑，不接 `run_all.sh`。

---

### Step B：加入 step2 config 生成和 step4c 脚本

目标：进入仓库流水线。

涉及文件：

```text
step2_generate_training.py
scripts/step4c_saart.sh
configs/experiment.yaml
```

---

### Step C：加入 step5 tag 和 run_all

目标：完整端到端。

涉及文件：

```text
run_all.sh
scripts/step5_evaluate.sh
```

---

### Step D：再加入 HotFlip-lite projection 和 trigger pool

目标：把离散触发纳入训练或诊断。

不要一开始就实现 projection。projection 的 bug 会和 soft-trigger / mask bug 混在一起，调试困难。

---

## 15. 验收标准

## 15.1 工程验收

必须满足：

1. `saart_p1_immunize.yaml` 正确生成；
2. 能加载 suspicious adapter 继续训练；
3. loss finite；
4. 输出 adapter；
5. step5 能生成 `after_saart_p1` 结果；
6. 不影响 BD-VAX / B1 / B2 / B3 / B3b。

这只证明“管线通”。

---

## 15.2 训练目标验收

日志中应看到：

1. `inner_proxy_final > inner_proxy_init`；
2. outer 训练后，同一 trigger 的 KL 下降；
3. `loss_adv` 和 `loss_kl` 都非零且量级合理；
4. projection keep rate 不是 0；
5. 如果启用 pool，pool trigger 确实被 sampled。

这证明 SAART loss 真的在工作。

---

## 15.3 研究效果验收

完整运行后至少满足：

1. `after_saart_p1.trigger_asr < after_pure_finetune.trigger_asr`；
2. `after_saart_p1.clean_fp` 不显著高于 B2；
3. 对 unseen trigger 的 ASR 有下降；
4. compute-matched pure finetune 不能完全解释 SAART 的收益；
5. soft-only 与 soft+pool 有可解释差异。

只有到这一层，才能说 Phase 1 具有研究价值。

---

## 16. 最终建议摘要

建议在实施前做以下调整：

1. 将当前方法明确命名为 **SAART-P1**，不要过早声称完整 association robustness。
2. 修正 KL 的 causal shift mask：使用 `labels[:, 1:] != -100` 对 `logits[:, :-1]` 做 mask。
3. 引入 null trigger reference，避免 KL 主要学习位置偏移鲁棒性。
4. BOS 后插入可作为 MVP，但应支持或规划 `prompt_end` 插入。
5. 明确离散 trigger pool 是否参与外层训练；若不参与，就只是 diagnostic。
6. soft trigger 至少应 batch-shared，最好维护 global trigger seed / bank。
7. 不要在 `compute_loss` 中调用 `model.zero_grad()`。
8. eval 时退回普通 SFT loss。
9. 增加日志与单元测试，尤其是 mask 对齐、inner grad 隔离、pool 使用、projection special-token 过滤。
10. 实验上加入 compute-matched pure finetune baseline。

如果按这些建议修改，Phase 1 MVP 就不只是一个“能跑的新训练分支”，而是一个可以自然扩展到 Phase 2 在线 MLP 关联签名的可靠基础。

---

## 17. 推荐优先级列表

### P0：必须先做

- 修正 KL shift mask。
- 删除 `compute_loss` 内部任何 `model.zero_grad()`。
- eval mode 退回普通 SFT loss。
- 日志记录三项 loss 与 inner proxy。
- soft trigger batch-shared。

### P1：强烈建议本轮做

- 加 null trigger reference。
- 支持 `saart_insert_position` 配置。
- projection 排除 special tokens。
- `after_saart_p1` tag。
- 增加基础单元测试。

### P2：可以 Phase 1 后半段做

- trigger pool 真正参与训练。
- top-k EMA pool policy。
- prompt-end insertion。
- compute-matched pure finetune baseline。

### P3：留给 Phase 2

- 在线 MLP 关联签名。
- forward hook 采集 `Δh`。
- EMA 通道风险分数。
- `L_assoc-reg`。
- 跨攻击 / 跨模型完整矩阵实验。

