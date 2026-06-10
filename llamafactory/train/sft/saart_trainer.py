# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's transformers library.
# https://github.com/huggingface/transformers/blob/v4.40.0/src/transformers/trainer_seq2seq.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SAART-P1 (Self-Adversarial Consistency Immunization) trainer.
# Adapted from consistency_trainer.py — but DELIBERATELY removes the inner
# `model.zero_grad()` calls (they would corrupt gradient accumulation) and uses
# `torch.autograd.grad` for the inner adversary so the optimizer graph stays clean.
#
# Per-step objective (see SAART_Phase1_MVP code plan):
#   total = L_clean + lambda1 * L_adv_correct + lambda2 * L_output_cons
# where the inner adversary maximizes a behavior-agnostic output-shift proxy
#   max_soft  KL(p_ref(.|x⊕t_null) || p_theta(.|x⊕soft))   over the RESPONSE region
# and the reference branch (null trigger of equal length) cancels position-shift artifacts.

import json
import os
import random
import re  # Phase-2：用正则从 named_modules 里匹配 mlp.{gate,up,down}_proj 及其层号
from types import MethodType
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from transformers import Seq2SeqTrainer

import importlib.metadata
import torch.nn.functional as F

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from ..trainer_utils import create_custom_optimzer, create_custom_scheduler

from packaging import version

if TYPE_CHECKING:
    from torch.utils.data import Dataset
    from transformers import ProcessorMixin
    from transformers.trainer import PredictionOutput

    from ...hparams import FinetuningArguments

from transformers.utils import is_peft_available

if is_peft_available():
    from peft import PeftModel

    def _is_peft_model(model):
        if is_peft_available():
            classes_to_check = (PeftModel,) if is_peft_available() else ()
            if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
                from peft import PeftMixedModel

                classes_to_check = (*classes_to_check, PeftMixedModel)
            return isinstance(model, classes_to_check)
        return False

logger = get_logger(__name__)


class SAARTSeq2SeqTrainer(Seq2SeqTrainer):
    r"""
    Seq2Seq trainer implementing SAART-P1 self-adversarial consistency immunization.

    Only `compute_loss` differs from the standard SFT trainer; all generation/prediction
    boilerplate is inherited unchanged from the LlamaFactory CustomSeq2SeqTrainer.
    """

    def __init__(
        self, finetuning_args: "FinetuningArguments", processor: Optional["ProcessorMixin"], **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        self.processor = processor

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

        # ---- SAART-P1 state ----
        fa = finetuning_args
        self.saart_trigger_len = fa.saart_trigger_len
        self.saart_inner_steps = fa.saart_inner_steps
        self.saart_inner_lr = fa.saart_inner_lr
        self.saart_lambda1 = fa.saart_lambda1
        self.saart_lambda2 = fa.saart_lambda2
        self.saart_pool_size = fa.saart_pool_size
        self.saart_use_projection = fa.saart_use_projection
        self.saart_proj_every = max(1, fa.saart_proj_every)
        self.saart_proj_keep_frac = fa.saart_proj_keep_frac
        self.saart_insert_position = fa.saart_insert_position
        self.saart_use_null_reference = fa.saart_use_null_reference
        self.saart_null_init = fa.saart_null_init
        self.saart_soft_init = fa.saart_soft_init
        self.saart_use_global_soft_seed = fa.saart_use_global_soft_seed
        self.saart_soft_norm_clip = fa.saart_soft_norm_clip
        self.saart_inner_eval_mode = fa.saart_inner_eval_mode
        self.saart_kl_type = fa.saart_kl_type
        self.saart_pool_sample_prob = fa.saart_pool_sample_prob
        self.saart_pool_policy = fa.saart_pool_policy
        self.saart_pool_ema_beta = fa.saart_pool_ema_beta
        self.saart_projection_exclude_special = fa.saart_projection_exclude_special
        self.saart_log_every = max(1, fa.saart_log_every)

        self._step_counter = 0
        self.trigger_pool: List[Dict[str, Any]] = []  # each: {token_ids: LongTensor[k], ema_proxy: float, seen: int}
        self._global_soft_seed: Optional[torch.Tensor] = None  # [k, d] on cpu, fp32
        self._normed_embed_cache: Optional[torch.Tensor] = None
        self._bad_token_id_cache: Optional[torch.Tensor] = None
        self._warned: set = set()
        self._sanity_done = False
        self._rng = random.Random(getattr(self.args, "seed", 42) or 42)

        # ---------------- Phase-2（Module 2）：在线 MLP 关联签名 + 关联正则化 状态 ----------------
        # 思路：训练中用 forward hook 采集 clean 与 triggered 两遍的 MLP 通道激活，算差 Δh，
        # 在线 EMA 统计每通道"风险"，选高风险通道集 S，对 S 上的 Δh 施加 L_assoc-reg。默认关闭。
        self.use_assoc_reg = fa.use_assoc_reg
        self.assoc_lambda3 = fa.assoc_lambda3
        self.assoc_top_ratio = fa.assoc_top_ratio
        self.assoc_ema_alpha = fa.assoc_ema_alpha
        self.assoc_align_lambda = fa.assoc_align_lambda
        self.saart_use_assoc_align = fa.saart_use_assoc_align  # 是否把方向一致性折入风险评分 s_j
        self.assoc_target_layers = fa.assoc_target_layers
        self.assoc_warmup_steps = fa.assoc_warmup_steps
        self.assoc_select_every = max(1, fa.assoc_select_every)
        self._assoc_registered = False          # hook 是否已注册（首步注册一次，幂等）
        self._assoc_modules: List[Tuple[str, str]] = []  # [(module_name, kind)]，kind ∈ {"output","input"}
        self._assoc_handles: List[Any] = []     # forward hook 句柄（保留以便需要时 remove）
        self._assoc_capture: Optional[str] = None  # 当前捕获模式：None / "clean" / "adv"
        self._assoc_clean_acts: Dict[str, torch.Tensor] = {}  # name -> [B,S,C]（detach，参考）
        self._assoc_adv_acts: Dict[str, torch.Tensor] = {}    # name -> [B,S+k,C]（保留梯度）
        self._assoc_risk: Dict[str, torch.Tensor] = {}        # name -> [C] 在线 EMA 风险分（幅度 mag）
        self._assoc_signed: Dict[str, torch.Tensor] = {}      # name -> [C] 带符号 shift 的 EMA（方向一致性 align 用）
        self._assoc_sig: Dict[str, torch.Tensor] = {}         # name -> LongTensor 选中通道索引(高风险集 S)

    # ------------------------------------------------------------------ #
    # boilerplate (unchanged from CustomSeq2SeqTrainer)
    # ------------------------------------------------------------------ #
    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.processor is not None:
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def prediction_step(
        self,
        model: "torch.nn.Module",
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        labels = inputs["labels"].detach().clone() if "labels" in inputs else None
        if self.args.predict_with_generate:
            assert self.tokenizer.padding_side == "left", "This method only accepts left-padded tensor."
            prompt_len, label_len = inputs["input_ids"].size(-1), inputs["labels"].size(-1)
            if prompt_len > label_len:
                inputs["labels"] = self._pad_tensors_to_target_len(inputs["labels"], inputs["input_ids"])
            if label_len > prompt_len:
                inputs["labels"] = inputs["labels"][:, :prompt_len]

        loss, generated_tokens, _ = super().prediction_step(
            model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
        )
        if generated_tokens is not None and self.args.predict_with_generate:
            generated_tokens[:, :prompt_len] = self.tokenizer.pad_token_id
            generated_tokens = generated_tokens.contiguous()

        return loss, generated_tokens, labels

    def _pad_tensors_to_target_len(self, src_tensor: torch.Tensor, tgt_tensor: torch.Tensor) -> torch.Tensor:
        assert self.tokenizer.pad_token_id is not None, "Pad token is required."
        padded_tensor = self.tokenizer.pad_token_id * torch.ones_like(tgt_tensor)
        padded_tensor[:, -src_tensor.shape[-1] :] = src_tensor
        return padded_tensor.contiguous()

    def save_predictions(self, dataset: "Dataset", predict_results: "PredictionOutput") -> None:
        if not self.is_world_process_zero():
            return

        output_prediction_file = os.path.join(self.args.output_dir, "generated_predictions.jsonl")
        logger.info(f"Saving prediction results to {output_prediction_file}")

        labels = np.where(
            predict_results.label_ids != IGNORE_INDEX, predict_results.label_ids, self.tokenizer.pad_token_id
        )
        preds = np.where(
            predict_results.predictions != IGNORE_INDEX, predict_results.predictions, self.tokenizer.pad_token_id
        )

        for i in range(len(preds)):
            pad_len = np.nonzero(preds[i] != self.tokenizer.pad_token_id)[0]
            if len(pad_len):
                preds[i] = np.concatenate((preds[i][pad_len[0] :], preds[i][: pad_len[0]]), axis=-1)

        decoded_inputs = self.tokenizer.batch_decode(
            dataset["input_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_labels = self.tokenizer.batch_decode(
            labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            res: List[str] = []
            for text, label, pred in zip(decoded_inputs, decoded_labels, decoded_preds):
                res.append(json.dumps({"prompt": text, "label": label, "predict": pred}, ensure_ascii=False))
            writer.write("\n".join(res))

    # ------------------------------------------------------------------ #
    # SAART helpers
    # ------------------------------------------------------------------ #
    def _warn_once(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            logger.warning(msg)

    def _sanity_check(self, input_ids: torch.Tensor) -> None:
        if self._sanity_done:
            return
        self._sanity_done = True
        pad_side = getattr(self.tokenizer, "padding_side", "right")
        if pad_side != "right":
            self._warn_once(
                f"SAART assumes right padding (got padding_side={pad_side}); after_bos insertion may be wrong."
            )
        if self.saart_insert_position == "after_bos":
            bos_id = getattr(self.tokenizer, "bos_token_id", None)
            if bos_id is not None and not bool(torch.all(input_ids[:, 0].eq(bos_id))):
                self._warn_once("SAART after_bos insertion assumes BOS at position 0, but it is not.")

    def _select_response_logits(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Causal-LM aware: logits[:, i-1] predicts labels[:, i]. Returns [num_resp_tokens, V]."""
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels.ne(IGNORE_INDEX)
        return shift_logits[mask]

    def _kl(self, ref_logits: torch.Tensor, adv_logits: torch.Tensor) -> torch.Tensor:
        """Token-mean KL between selected reference and adversarial logits. ref is treated as detached upstream."""
        ref_logp = F.log_softmax(ref_logits.float(), dim=-1)
        adv_logp = F.log_softmax(adv_logits.float(), dim=-1)
        kl_type = self.saart_kl_type
        if kl_type == "forward":
            ref_p = ref_logp.exp()
            per_tok = (ref_p * (ref_logp - adv_logp)).sum(dim=-1)
        elif kl_type == "reverse":
            adv_p = adv_logp.exp()
            per_tok = (adv_p * (adv_logp - ref_logp)).sum(dim=-1)
        elif kl_type == "symmetric":
            ref_p = ref_logp.exp()
            adv_p = adv_logp.exp()
            per_tok = 0.5 * (ref_p * (ref_logp - adv_logp)).sum(dim=-1) + 0.5 * (adv_p * (adv_logp - ref_logp)).sum(dim=-1)
        elif kl_type == "js":
            ref_p = ref_logp.exp()
            adv_p = adv_logp.exp()
            m = 0.5 * (ref_p + adv_p)
            m_logp = (m + 1e-12).log()
            per_tok = 0.5 * (ref_p * (ref_logp - m_logp)).sum(dim=-1) + 0.5 * (adv_p * (adv_logp - m_logp)).sum(dim=-1)
        else:
            raise ValueError(f"Unknown saart_kl_type: {kl_type}")
        return per_tok.mean()

    def _insertion_positions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Per-example index at which to insert the trigger."""
        B = input_ids.shape[0]
        device = input_ids.device
        if self.saart_insert_position == "prompt_end":
            resp = labels.ne(IGNORE_INDEX)
            has_resp = resp.any(dim=1)
            first_resp = resp.float().argmax(dim=1)  # 0 if no response
            fallback = attention_mask.sum(dim=1)  # end of real tokens
            return torch.where(has_resp, first_resp, fallback).long()
        # after_bos
        return torch.ones(B, dtype=torch.long, device=device)

    def _build_triggered_inputs(self, base_embeds, attention_mask, labels, trigger_embeds, positions):
        """Splice trigger_embeds [B,k,d] into base_embeds [B,S,d] at per-example positions.

        Returns (embeds_star [B,S+k,d], attn_star [B,S+k], labels_star [B,S+k]).
        Trigger positions get attention=1 and label=IGNORE_INDEX. B is small, so a loop is clear & correct.
        """
        B, S, _ = base_embeds.shape
        k = trigger_embeds.shape[1]
        device = base_embeds.device
        ones_k = torch.ones(k, dtype=attention_mask.dtype, device=device)
        ign_k = torch.full((k,), IGNORE_INDEX, dtype=labels.dtype, device=device)
        embeds_list, attn_list, label_list = [], [], []
        for b in range(B):
            p = int(positions[b].item())
            embeds_list.append(torch.cat([base_embeds[b, :p], trigger_embeds[b], base_embeds[b, p:]], dim=0))
            attn_list.append(torch.cat([attention_mask[b, :p], ones_k, attention_mask[b, p:]], dim=0))
            label_list.append(torch.cat([labels[b, :p], ign_k, labels[b, p:]], dim=0))
        return torch.stack(embeds_list, 0), torch.stack(attn_list, 0), torch.stack(label_list, 0)

    def _init_soft_trigger(self, embed_layer, batch_first_ids) -> torch.Tensor:
        """Return a BATCH-SHARED soft trigger of shape [k, d] (fp32, requires_grad).

        IMPORTANT: the initial soft trigger must NOT coincide with the null reference, or
        KL(p_ref || p_soft) starts at exactly 0 with a zero gradient and the FGSM sign-step
        never moves it. Hence even the mean_embedding init is perturbed with small noise.
        """
        emb = embed_layer.weight  # [V, d]
        d = emb.shape[1]
        k = self.saart_trigger_len
        if self.saart_use_global_soft_seed and self._global_soft_seed is not None:
            base = self._global_soft_seed.to(device=emb.device, dtype=torch.float32)
        else:
            init = self.saart_soft_init
            emb_f = emb.detach().float()
            std = emb_f.std()
            if init == "random":
                base = torch.randn(k, d, device=emb.device, dtype=torch.float32) * std
            elif init == "vocab_sample":
                idx = torch.randint(0, emb.shape[0], (k,), device=emb.device)
                base = emb_f[idx]
            else:  # mean_embedding (default) / pool fallback: center + symmetry-breaking noise
                mean = emb_f.mean(dim=0, keepdim=True).expand(k, d)
                base = mean + 0.1 * std * torch.randn(k, d, device=emb.device, dtype=torch.float32)
        return base.detach().clone().requires_grad_(True)

    def _null_trigger_embeds(self, embed_layer, B) -> torch.Tensor:
        """Detached neutral trigger embeddings [B, k, d] in model dtype, of equal length k."""
        emb = embed_layer.weight
        d = emb.shape[1]
        k = self.saart_trigger_len
        init = self.saart_null_init
        if init == "zero":
            null = torch.zeros(k, d, device=emb.device, dtype=emb.dtype)
        elif init == "pad_token":
            pad_id = getattr(self.tokenizer, "pad_token_id", None) or 0
            null = embed_layer(torch.full((k,), int(pad_id), device=emb.device, dtype=torch.long)).detach()
        else:  # mean_embedding (default) / learned fallback
            null = emb.detach().mean(dim=0, keepdim=True).expand(k, d).to(emb.dtype)
        return null.unsqueeze(0).expand(B, -1, -1)

    @torch.no_grad()
    def _maybe_clip_soft(self, soft: torch.Tensor, avg_norm: torch.Tensor) -> torch.Tensor:
        if not self.saart_soft_norm_clip:
            return soft
        return soft / (soft.norm(dim=-1, keepdim=True) + 1e-6) * avg_norm

    def _inner_search(self, model, base_embeds, attention_mask, labels, positions, ref_sel, embed_layer):
        """Inner adversary: ascend the output-shift proxy w.r.t. the soft trigger ONLY.

        Uses torch.autograd.grad (never .backward(), never model.zero_grad()) so no LoRA-param
        gradient is ever populated here — the outer optimizer graph stays clean.
        """
        B = base_embeds.shape[0]
        soft = self._init_soft_trigger(embed_layer, None)
        avg_norm = embed_layer.weight.detach().float().norm(dim=-1).mean()

        was_training = model.training
        if self.saart_inner_eval_mode and was_training:
            model.eval()

        proxy_init = None
        proxy_last = None
        try:
            for _ in range(self.saart_inner_steps):
                trig = soft.to(base_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
                e_t, a_t, l_t = self._build_triggered_inputs(base_embeds, attention_mask, labels, trig, positions)
                out_t = model(inputs_embeds=e_t, attention_mask=a_t, use_cache=False)
                adv_sel = self._select_response_logits(out_t.logits, l_t)
                proxy = self._kl(ref_sel, adv_sel)  # maximize
                if proxy_init is None:
                    proxy_init = proxy.detach()
                proxy_last = proxy.detach()
                grad = torch.autograd.grad(proxy, [soft], retain_graph=False, create_graph=False)[0]
                with torch.no_grad():
                    soft = soft + self.saart_inner_lr * grad.sign()
                    soft = self._maybe_clip_soft(soft, avg_norm)
                soft = soft.detach().requires_grad_(True)
        finally:
            if self.saart_inner_eval_mode and was_training:
                model.train()

        soft_final = soft.detach()
        if self.saart_use_global_soft_seed:
            self._global_soft_seed = soft_final.detach().to("cpu", torch.float32)
        if proxy_init is None:  # inner_steps == 0
            proxy_init = torch.tensor(0.0, device=base_embeds.device)
            proxy_last = proxy_init
        return soft_final, float(proxy_init), float(proxy_last)

    # ---- discrete projection / trigger pool (Step D; inert when use_projection=False) ----
    def _get_normed_embedding(self, embed_layer) -> torch.Tensor:
        w = embed_layer.weight
        if w.requires_grad:  # embeddings are being trained -> cache would go stale
            return F.normalize(w.detach().float(), dim=-1).to(w.dtype)
        if self._normed_embed_cache is None:
            self._normed_embed_cache = F.normalize(w.detach().float(), dim=-1).to(w.dtype)
        return self._normed_embed_cache

    def _bad_token_ids(self, device) -> Optional[torch.Tensor]:
        if not self.saart_projection_exclude_special:
            return None
        if self._bad_token_id_cache is not None:
            return self._bad_token_id_cache.to(device)
        ids = set()
        special = getattr(self.tokenizer, "all_special_ids", None)
        if special:
            ids.update(int(i) for i in special if i is not None)
        for attr in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            v = getattr(self.tokenizer, attr, None)
            if v is not None:
                ids.add(int(v))
        if not ids:
            return None
        self._bad_token_id_cache = torch.tensor(sorted(ids), dtype=torch.long)
        return self._bad_token_id_cache.to(device)

    @torch.no_grad()
    def _project_tokens(self, soft: torch.Tensor, embed_layer) -> torch.Tensor:
        """HotFlip-lite: nearest-neighbor token ids [k] for the soft trigger, special tokens excluded."""
        normed = self._get_normed_embedding(embed_layer)  # [V, d]
        soft_n = F.normalize(soft.to(normed.dtype), dim=-1)  # [k, d]
        sims = soft_n @ normed.t()  # [k, V]
        bad = self._bad_token_ids(sims.device)
        if bad is not None:
            sims[:, bad] = float("-inf")
        return sims.argmax(dim=-1).long()  # [k]

    @torch.no_grad()
    def _project_and_update_pool(self, model, base_embeds, attention_mask, labels, positions, ref_sel, soft, embed_layer) -> float:
        """Project soft trigger to discrete tokens (HotFlip-lite, special tokens excluded), verify,
        and add to the FIFO pool if it retains enough of the soft proxy. Returns 1.0 if kept else 0.0."""
        B = base_embeds.shape[0]
        tok = self._project_tokens(soft, embed_layer)  # [k]

        disc = embed_layer(tok).unsqueeze(0).expand(B, -1, -1).to(base_embeds.dtype)
        d_e, d_a, d_l = self._build_triggered_inputs(base_embeds, attention_mask, labels, disc, positions)
        d_sel = self._select_response_logits(model(inputs_embeds=d_e, attention_mask=d_a, use_cache=False).logits, d_l)
        proxy_disc = float(self._kl(ref_sel, d_sel))

        trig = soft.to(base_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
        s_e, s_a, s_l = self._build_triggered_inputs(base_embeds, attention_mask, labels, trig, positions)
        s_sel = self._select_response_logits(model(inputs_embeds=s_e, attention_mask=s_a, use_cache=False).logits, s_l)
        proxy_soft = float(self._kl(ref_sel, s_sel))

        keep = proxy_disc >= self.saart_proj_keep_frac * max(proxy_soft, 1e-8)
        if keep:
            self._add_to_pool(tok.detach().to("cpu"), proxy_disc)
        return 1.0 if keep else 0.0

    def _add_to_pool(self, token_ids: torch.Tensor, proxy: float) -> None:
        """Insert/refresh a discrete trigger. Deduplicates by token-id sequence and keeps a
        running EMA of its proxy so the topk_ema policy reflects repeated verifications, not a
        single noisy measurement."""
        for it in self.trigger_pool:
            if it["token_ids"].numel() == token_ids.numel() and bool(torch.equal(it["token_ids"], token_ids)):
                beta = self.saart_pool_ema_beta
                it["ema_proxy"] = beta * it["ema_proxy"] + (1.0 - beta) * proxy
                it["seen"] += 1
                self._evict_pool()
                return
        self.trigger_pool.append({"token_ids": token_ids, "ema_proxy": proxy, "seen": 1})
        self._evict_pool()

    def _evict_pool(self) -> None:
        if len(self.trigger_pool) <= self.saart_pool_size:
            return
        if self.saart_pool_policy == "topk_ema":
            self.trigger_pool.sort(key=lambda it: it["ema_proxy"], reverse=True)
            del self.trigger_pool[self.saart_pool_size:]
        else:  # fifo
            self.trigger_pool.pop(0)

    def _use_pool_trigger(self) -> bool:
        return bool(self.trigger_pool) and (self._rng.random() < self.saart_pool_sample_prob)

    def _discrete_trigger_inputs(self, base_embeds, attention_mask, labels, positions, embed_layer):
        B = base_embeds.shape[0]
        idx = self._rng.randrange(len(self.trigger_pool))
        self.trigger_pool[idx]["seen"] += 1
        tok = self.trigger_pool[idx]["token_ids"].to(base_embeds.device).long()
        disc = embed_layer(tok).unsqueeze(0).expand(B, -1, -1).to(base_embeds.dtype)
        return self._build_triggered_inputs(base_embeds, attention_mask, labels, disc, positions)

    # ------------------------------------------------------------------ #
    # Phase-2（Module 2）：在线 MLP 关联签名 + 关联正则化 L_assoc-reg
    # ------------------------------------------------------------------ #
    def _select_assoc_layers(self, layers: List[int]) -> set:
        """把 assoc_target_layers 配置解析成"要 hook 的层号集合"。
        支持 all / lastN / everyN / 逗号分隔显式层号。默认 last8（只挂最后 8 层，控显存）。
        TODO(SAART-P2): 层子集策略可调——文档说后门在多个 MLP 块冗余编码，挂哪些层最有效需实验。"""
        spec = str(self.assoc_target_layers).strip()
        if spec == "all":
            return set(layers)
        if spec.startswith("last"):
            n = int(spec[4:] or 0)
            return set(layers[-n:]) if n > 0 else set(layers)
        if spec.startswith("every"):
            n = max(1, int(spec[5:] or 1))
            return set(layers[::n])
        try:  # 逗号分隔的显式层号，例如 "0,8,16,24"
            want = {int(x) for x in spec.split(",") if x.strip() != ""}
            return want & set(layers)
        except ValueError:
            return set(layers)

    def _resolve_assoc_modules(self, unwrapped) -> List[Tuple[str, str]]:
        """用 named_modules 后缀匹配每个 decoder 层的 mlp.{gate,up,down}_proj（robust，不硬编码 model.model.layers[i]）。
        gate/up 抓 output、down 抓 input[0]，三者都对应 11008 维 intermediate 通道，
        与 antigen/scoring.py 的 out_j(gate/up 行) / in_j(down 列) 通道约定一致。"""
        cand = []  # (layer_idx, name, kind)
        pat = re.compile(r"\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)$")
        for name, _mod in unwrapped.named_modules():
            m = pat.search(name)
            if not m:
                continue
            layer_idx, proj = int(m.group(1)), m.group(2)
            kind = "input" if proj == "down_proj" else "output"  # down 抓输入(intermediate)，gate/up 抓输出
            cand.append((layer_idx, name, kind))
        if not cand:
            return []
        layers = sorted({li for li, _, _ in cand})
        sel = self._select_assoc_layers(layers)
        return [(name, kind) for (li, name, kind) in cand if li in sel]

    def _register_assoc_hooks(self, model) -> None:
        """幂等注册 forward hook。hook 据 self._assoc_capture 决定写 clean(detach)/adv(保留梯度) 缓存；
        capture 为 None 时直接返回，故 null-ref/inner/projection 等其它前向零开销。"""
        if self._assoc_registered:
            return
        unwrapped = self.accelerator.unwrap_model(model)
        self._assoc_modules = self._resolve_assoc_modules(unwrapped)
        name2mod = dict(unwrapped.named_modules())
        for name, kind in self._assoc_modules:
            mod = name2mod[name]

            def make_hook(nm: str, kd: str):
                def hook(_m, inp, out):
                    mode = self._assoc_capture
                    if mode is None:  # 非捕获期：什么都不做（零开销）
                        return
                    act = inp[0] if kd == "input" else out  # down 取输入、gate/up 取输出
                    if mode == "clean":
                        self._assoc_clean_acts[nm] = act.detach()  # 参考分支：detach 不带梯度
                    else:  # "adv"：保留梯度，L_assoc-reg 要靠它反传到 LoRA
                        self._assoc_adv_acts[nm] = act
                return hook

            self._assoc_handles.append(mod.register_forward_hook(make_hook(name, kind)))
        self._assoc_registered = True
        if self.is_world_process_zero():
            n_layers = len({n.rsplit(".mlp.", 1)[0] for n, _ in self._assoc_modules})
            logger.info(f"[SAART Phase-2] 注册了 {len(self._assoc_handles)} 个 MLP hook（覆盖 {n_layers} 个 decoder 层）")

    def _assoc_resp_acts(self, act: torch.Tensor, labels_for_act: torch.Tensor) -> torch.Tensor:
        """从 [B,T,C] 激活按 response mask(labels!=IGNORE_INDEX) 取出 response 位的激活 -> [Nresp, C]。
        clean 用原 labels([B,S])、adv 用 labels_star([B,S+k])；插入的 k 个触发位 label=-100 被排除，
        故两边 response token 数与顺序 1:1 对齐（adv 只是整体右移 k）。"""
        mask = labels_for_act.ne(IGNORE_INDEX)  # [B,T] 布尔
        return act[mask]  # 行主序展开成 [Nresp, C]

    def _compute_assoc_deltas(self, labels, adv_labels) -> Dict[str, torch.Tensor]:
        """对每个被 hook 的 module 计算 response 区激活差 Δh = h_adv - h_clean.detach()，返回 name->[Nresp,C]。
        adv 端保留梯度（供 loss 反传到 LoRA），clean 端 detach（参考）；统一转 fp32 让统计/平方更稳。"""
        deltas: Dict[str, torch.Tensor] = {}
        for name, _kind in self._assoc_modules:
            if name not in self._assoc_clean_acts or name not in self._assoc_adv_acts:
                continue
            h_clean = self._assoc_resp_acts(self._assoc_clean_acts[name], labels)      # [Nc, C] detach
            h_adv = self._assoc_resp_acts(self._assoc_adv_acts[name], adv_labels)      # [Na, C] 带梯度
            if h_clean.shape[0] != h_adv.shape[0]:
                continue  # 理论上相等；不等则跳过该 module（不静默出错）
            deltas[name] = h_adv.float() - h_clean.float().detach()  # adv 带梯度，clean 不带
        return deltas

    @torch.no_grad()
    def _update_assoc_risk(self, deltas: Dict[str, torch.Tensor]) -> None:
        """在线 EMA 更新每通道风险分：risk_j = α·risk_j + (1-α)·mean_resp|Δh_j|（幅度项）。
        风险高 = 该通道在触发前后反复发生大幅激活变化 = 后门关联最可能落脚处。
        TODO(SAART-P2): 加 alignment 项（assoc_align_lambda）——对应文档 Eq.2 的跨变体方向一致性，
        在线版可用"本步 Δh 方向与历史 EMA 方向的 cosine"，需额外存方向状态；当前先 magnitude-only。"""
        a = self.assoc_ema_alpha
        for name, d in deltas.items():
            cur = d.detach().abs().mean(dim=0)  # [C] 本步每通道平均幅度
            if name not in self._assoc_risk:
                self._assoc_risk[name] = cur.clone()
            else:
                self._assoc_risk[name].mul_(a).add_(cur, alpha=1.0 - a)

    def _select_assoc_signature(self) -> None:
        """每 module 内按风险分取 top assoc_top_ratio 通道作为高风险集合 S（与 antigen/scoring.py 一致：
        top-τ% 是"每 module 内部"取，不是跨 module 取整体 top）。"""
        for name, risk in self._assoc_risk.items():
            C = risk.numel()
            k = max(1, int(C * self.assoc_top_ratio))  # 至少选 1 个通道
            self._assoc_sig[name] = torch.topk(risk, k).indices  # [k] 选中通道索引

    def _assoc_reg_loss(self, deltas: Dict[str, torch.Tensor]):
        """L_assoc-reg = mean_module mean_resp mean_{j∈S}(Δh_j)²（adv 带梯度，clean detached）。
        最小化它 = 迫使模型在高风险通道上"触发前后激活一致"，从而切断 trigger→behavior 关联。
        warmup 前 S 为空 → 返回 None（不施加该项）。
        注意：这里对选中通道取 **均值** 而非文档写的求和——冒烟实测求和会因 |S|≈3852/模块 把损失放大到上千、
        压垮其它损失项导致训练发散；取均值把量级归一到 O(1)，与 λ3 解耦。
        TODO(SAART-P2): λ3(assoc_lambda3) 与本项量级仍需调参——激活幅度大时可能偏强；
        也可考虑用相对位移 Δh/||h_clean|| 或 sum 形式配更小 λ3。"""
        if not self._assoc_sig:
            return None
        terms = []
        for name, d in deltas.items():
            sel = self._assoc_sig.get(name)
            if sel is None or sel.numel() == 0:
                continue
            sub = d[:, sel]  # [Nresp, |S|]，adv 带梯度
            terms.append((sub ** 2).mean())  # 在 (response token × 选中通道) 上取均方 → 量级 O(1)
        if not terms:
            return None
        return torch.stack(terms).mean()  # 对各 module 再取均值

    def _assoc_sig_size(self) -> int:
        """当前高风险集合 S 的总通道数（仅用于日志）。"""
        return int(sum(int(v.numel()) for v in self._assoc_sig.values()))

    def _maybe_log(self, loss_clean, loss_adv, loss_kl, proxy_init, proxy_final, soft, keep_rate, loss_assoc=None) -> None:
        if self._step_counter % self.saart_log_every != 0:
            return
        if not self.is_world_process_zero():
            return
        msg = (
            f"[SAART step {self._step_counter}] "
            f"loss_clean={float(loss_clean):.4f} loss_adv={float(loss_adv):.4f} loss_kl={float(loss_kl):.4f} "
            f"inner_proxy_init={proxy_init:.4f} inner_proxy_final={proxy_final:.4f} "
            f"soft_norm={float(soft.norm(dim=-1).mean()):.3f} "
            f"keep_rate={'-' if keep_rate is None else f'{keep_rate:.2f}'} pool_size={len(self.trigger_pool)}"
        )
        # Phase-2：开启 assoc-reg 时额外打印关联正则损失与高风险集合 S 的大小
        if self.use_assoc_reg:
            assoc_str = "-" if loss_assoc is None else f"{float(loss_assoc):.4f}"
            msg += f" loss_assoc={assoc_str} |S|={self._assoc_sig_size()}"
        logger.info(msg)

    # ------------------------------------------------------------------ #
    # main loss
    # ------------------------------------------------------------------ #
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Evaluation / prediction: behave like a plain SFT trainer.
        if not model.training:
            outputs = model(**inputs, use_cache=False)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            return (loss, outputs) if return_outputs else loss

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]
        B, S = input_ids.shape
        k = self.saart_trigger_len

        self._sanity_check(input_ids)

        unwrapped = self.accelerator.unwrap_model(model)
        embed_layer = unwrapped.get_input_embeddings()

        # Phase-2：首步惰性注册 MLP forward hook（仅在开启 assoc-reg 时）
        if self.use_assoc_reg and not self._assoc_registered:
            self._register_assoc_hooks(model)

        # 1. clean forward (defines L_clean and the optimizer graph for the clean objective)
        # Phase-2：把捕获模式设为 "clean"，让 hook 采下这一遍的 MLP 通道激活 h(x)（detach 作参考）
        if self.use_assoc_reg:
            self._assoc_capture = "clean"
        clean_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
        if self.use_assoc_reg:
            self._assoc_capture = None  # 立即关闭，避免 null-ref/inner/projection 的前向被误捕获
        loss_clean = clean_outputs["loss"] if isinstance(clean_outputs, dict) else clean_outputs[0]

        # Guard: don't exceed the model's max position with the inserted trigger.
        max_pos = getattr(unwrapped.config, "max_position_embeddings", None)
        if (max_pos is not None and S + k > max_pos) or k <= 0:
            self._warn_once(
                f"Skipping SAART adversarial branch: seq_len+k={S + k} > max_position_embeddings={max_pos}."
            )
            self._step_counter += 1
            return (loss_clean, clean_outputs) if return_outputs else loss_clean

        base_embeds = embed_layer(input_ids)  # [B, S, d]
        positions = self._insertion_positions(input_ids, attention_mask, labels)

        # 2. reference branch (detached): null trigger (equal length) or clean.
        if self.saart_use_null_reference:
            with torch.no_grad():
                null_trig = self._null_trigger_embeds(embed_layer, B)
                r_e, r_a, r_l = self._build_triggered_inputs(base_embeds, attention_mask, labels, null_trig, positions)
                ref_sel = self._select_response_logits(
                    model(inputs_embeds=r_e, attention_mask=r_a, use_cache=False).logits, r_l
                ).detach()
        else:
            ref_sel = self._select_response_logits(clean_outputs.logits, labels).detach()

        # 3. inner adversary (updates soft only; no LoRA grad)
        soft, proxy_init, proxy_final = self._inner_search(
            model, base_embeds, attention_mask, labels, positions, ref_sel, embed_layer
        )

        # 4. optional discrete projection + pool mining (gated; Step D)
        keep_rate = None
        if self.saart_use_projection and (self._step_counter % self.saart_proj_every == 0):
            keep_rate = self._project_and_update_pool(
                model, base_embeds, attention_mask, labels, positions, ref_sel, soft, embed_layer
            )

        # 5. choose the outer trigger t* (soft, or a discrete pool trigger w.p. pool_sample_prob)
        if self.saart_use_projection and self._use_pool_trigger():
            adv_e, adv_a, adv_l = self._discrete_trigger_inputs(base_embeds, attention_mask, labels, positions, embed_layer)
        else:
            trig = soft.to(base_embeds.dtype).unsqueeze(0).expand(B, -1, -1)
            adv_e, adv_a, adv_l = self._build_triggered_inputs(base_embeds, attention_mask, labels, trig, positions)

        # 6. triggered forward (carries grad through LoRA params; trigger content is detached)
        # Phase-2：把捕获模式设为 "adv"，采下触发分支的 MLP 激活 h(x⊕t*)（保留梯度，供 L_assoc-reg 反传）
        if self.use_assoc_reg:
            self._assoc_capture = "adv"
        adv_outputs = model(inputs_embeds=adv_e, attention_mask=adv_a, labels=adv_l, use_cache=False)
        if self.use_assoc_reg:
            self._assoc_capture = None
        loss_adv = adv_outputs["loss"] if isinstance(adv_outputs, dict) else adv_outputs[0]

        # 7. output-consistency KL on shifted response positions
        adv_sel = self._select_response_logits(adv_outputs.logits, adv_l)
        if adv_sel.shape[0] != ref_sel.shape[0]:
            # Should not happen (equal-length insertion), but never silently mis-align.
            self._warn_once(
                f"SAART KL token mismatch (adv={adv_sel.shape[0]} ref={ref_sel.shape[0]}); dropping KL term this step."
            )
            loss_kl = torch.zeros((), device=loss_clean.device, dtype=loss_clean.dtype)
        else:
            loss_kl = self._kl(ref_sel, adv_sel)

        total = loss_clean + self.saart_lambda1 * loss_adv + self.saart_lambda2 * loss_kl

        # 8. Phase-2：在线 MLP 关联签名 + 关联正则化 L_assoc-reg（开启 assoc-reg 才走）
        loss_assoc = None
        if self.use_assoc_reg:
            # 用 clean/adv 两遍捕获的 MLP 激活算 response 区的 Δh（adv 带梯度）
            deltas = self._compute_assoc_deltas(labels, adv_l)
            # 在线 EMA 累积每通道风险分（no_grad）
            self._update_assoc_risk(deltas)
            # 过 warmup 后按周期刷新高风险集合 S（首次满足即选一次）
            if self._step_counter >= self.assoc_warmup_steps and (
                not self._assoc_sig or self._step_counter % self.assoc_select_every == 0
            ):
                self._select_assoc_signature()
            # 仅当 S 已选出（warmup 后）才有非 None 的关联正则损失
            loss_assoc = self._assoc_reg_loss(deltas)
            if loss_assoc is not None:
                total = total + self.assoc_lambda3 * loss_assoc
            # 清空本步激活缓存释放显存（deltas 仍持有 adv 计算图引用，反向用完自然释放）
            self._assoc_clean_acts = {}
            self._assoc_adv_acts = {}

        self._maybe_log(loss_clean, loss_adv, loss_kl, proxy_init, proxy_final, soft, keep_rate, loss_assoc)
        self._step_counter += 1
        # Only return the CLEAN outputs; never leak the triggered/inner graph.
        return (total, clean_outputs) if return_outputs else total
