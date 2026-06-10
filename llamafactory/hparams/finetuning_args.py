# Copyright 2024 the LlamaFactory team.
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

from dataclasses import dataclass, field
from typing import List, Literal, Optional


@dataclass
class FreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    freeze_trainable_layers: int = field(
        default=2,
        metadata={
            "help": (
                "The number of trainable layers for freeze (partial-parameter) fine-tuning. "
                "Positive numbers mean the last n layers are set as trainable, "
                "negative numbers mean the first n layers are set as trainable."
            )
        },
    )
    freeze_trainable_modules: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of trainable modules for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the available modules."
            )
        },
    )
    freeze_extra_modules: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from hidden layers to be set as trainable "
                "for freeze (partial-parameter) fine-tuning. "
                "Use commas to separate multiple modules."
            )
        },
    )


@dataclass
class LoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Name(s) of modules apart from LoRA layers to be set as trainable "
                "and saved in the final checkpoint. "
                "Use commas to separate multiple modules."
            )
        },
    )
    lora_alpha: Optional[int] = field(
        default=None,
        metadata={"help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    lora_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of target modules to apply LoRA. "
                "Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    use_rslora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    use_dora: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    pissa_init: bool = field(
        default=False,
        metadata={"help": "Whether or not to initialize a PiSSA adapter."},
    )
    pissa_iter: int = field(
        default=4,
        metadata={"help": "The number of iteration steps performed by FSVD in PiSSA. Use -1 to disable it."},
    )
    pissa_convert: bool = field(
        default=False,
        metadata={"help": "Whether or not to convert the PiSSA adapter to a normal LoRA adapter."},
    )
    create_new_adapter: bool = field(
        default=False,
        metadata={"help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class RLHFArguments:
    r"""
    Arguments pertaining to the PPO, DPO and KTO training.
    """

    pref_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter in the preference loss."},
    )
    pref_ftx: float = field(
        default=0.0,
        metadata={"help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    dpo_label_smoothing: float = field(
        default=0.0,
        metadata={"help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    kto_chosen_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the desirable losses in KTO training."},
    )
    kto_rejected_weight: float = field(
        default=1.0,
        metadata={"help": "The weight factor of the undesirable losses in KTO training."},
    )
    simpo_gamma: float = field(
        default=0.5,
        metadata={"help": "The target reward margin term in SimPO loss."},
    )
    ppo_buffer_size: int = field(
        default=1,
        metadata={"help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    ppo_epochs: int = field(
        default=4,
        metadata={"help": "The number of epochs to perform in a PPO optimization step."},
    )
    ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    ppo_target: float = field(
        default=6.0,
        metadata={"help": "Target KL value for adaptive KL control in PPO training."},
    )
    ppo_whiten_rewards: bool = field(
        default=False,
        metadata={"help": "Whiten the rewards before compute advantages in PPO training."},
    )
    ref_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reference model used for the PPO or DPO training."},
    )
    ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={"help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )


@dataclass
class GaloreArguments:
    r"""
    Arguments pertaining to the GaLore algorithm.
    """

    use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the gradient low-Rank projection (GaLore)."},
    )
    galore_target: str = field(
        default="all",
        metadata={
            "help": (
                "Name(s) of modules to apply GaLore. Use commas to separate multiple modules. "
                "Use `all` to specify all the linear modules."
            )
        },
    )
    galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    galore_scale: float = field(
        default=0.25,
        metadata={"help": "GaLore scaling coefficient."},
    )
    galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    galore_layerwise: bool = field(
        default=False,
        metadata={"help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class BAdamArgument:
    r"""
    Arguments pertaining to the BAdam optimizer.
    """

    use_badam: bool = field(
        default=False,
        metadata={"help": "Whether or not to use the BAdam optimizer."},
    )
    badam_mode: Literal["layer", "ratio"] = field(
        default="layer",
        metadata={"help": "Whether to use layer-wise or ratio-wise BAdam optimizer."},
    )
    badam_start_block: Optional[int] = field(
        default=None,
        metadata={"help": "The starting block index for layer-wise BAdam."},
    )
    badam_switch_mode: Optional[Literal["ascending", "descending", "random", "fixed"]] = field(
        default="ascending",
        metadata={"help": "the strategy of picking block to update for layer-wise BAdam."},
    )
    badam_switch_interval: Optional[int] = field(
        default=50,
        metadata={
            "help": "Number of steps to update the block for layer-wise BAdam. Use -1 to disable the block update."
        },
    )
    badam_update_ratio: float = field(
        default=0.05,
        metadata={"help": "The ratio of the update for ratio-wise BAdam."},
    )
    badam_mask_mode: Literal["adjacent", "scatter"] = field(
        default="adjacent",
        metadata={
            "help": (
                "The mode of the mask for BAdam optimizer. "
                "`adjacent` means that the trainable parameters are adjacent to each other, "
                "`scatter` means that trainable parameters are randomly choosed from the weight."
            )
        },
    )
    badam_verbose: int = field(
        default=0,
        metadata={
            "help": (
                "The verbosity level of BAdam optimizer. "
                "0 for no print, 1 for print the block prefix, 2 for print trainable parameters."
            )
        },
    )


@dataclass
class SAARTArguments:
    r"""
    Arguments pertaining to SAART-P1 (Self-Adversarial Consistency Immunization).

    Phase-1 MVP of the Self-Adversarial Association-Robust Training framework: an inner
    soft-trigger adversary that maximizes a behavior-agnostic output-shift proxy, plus an
    outer objective combining clean SFT loss, adversarial-correct loss and an output
    consistency KL. The full association-robust method (online MLP association signature)
    is deferred to Phase 2. See the SAART trainer for the loss definition.
    """

    use_saart: bool = field(
        default=False,
        metadata={"help": "Whether or not to use SAART-P1 self-adversarial immunization training."},
    )
    saart_trigger_len: int = field(
        default=5,
        metadata={"help": "Number k of soft-trigger embedding vectors inserted into the prompt."},
    )
    saart_inner_steps: int = field(
        default=3,
        metadata={"help": "Number of inner gradient-ascent steps for the soft-trigger adversary."},
    )
    saart_inner_lr: float = field(
        default=0.03,
        metadata={"help": "Sign-step size for the inner soft-trigger ascent (FGSM-style)."},
    )
    saart_lambda1: float = field(
        default=1.0,
        metadata={"help": "Weight of the adversarial-correct loss L_adv-correct."},
    )
    saart_lambda2: float = field(
        default=0.5,
        metadata={"help": "Weight of the output-consistency KL loss L_output-cons."},
    )
    saart_pool_size: int = field(
        default=16,
        metadata={"help": "Maximum number of discrete triggers kept in the trigger pool."},
    )
    saart_use_projection: bool = field(
        default=True,
        metadata={"help": "Whether to project soft triggers to discrete tokens (HotFlip-lite) and mine a trigger pool."},
    )
    saart_proj_every: int = field(
        default=20,
        metadata={"help": "Run discrete projection every N steps to amortize its cost."},
    )
    saart_proj_keep_frac: float = field(
        default=0.8,
        metadata={"help": "A projected discrete trigger is kept if its proxy is >= this fraction of the soft proxy."},
    )
    saart_insert_position: Literal["after_bos", "prompt_end"] = field(
        # 默认改回 after_bos：4 波对比实验表明在 BadNets×sentiment×LLaMA2 上 after_bos 比 prompt_end 低约 12pp ASR
        # （见 reports/SAART_P1_对比实验报告_20260605.md §5.3/§7.4 的"位置×KL 交互"）。prompt_end 仍可配置备用。
        default="after_bos",
        metadata={"help": "Where to insert the trigger: right after BOS (default; empirically best on BadNets×sentiment), or at the end of the user prompt."},
    )
    saart_use_null_reference: bool = field(
        default=True,
        metadata={"help": "Compare KL against x+null_trigger (same insertion length) instead of x, to cancel position-shift artifacts."},
    )
    saart_null_init: Literal["mean_embedding", "zero", "pad_token", "learned"] = field(
        default="mean_embedding",
        metadata={"help": "Initialization of the neutral null trigger embeddings."},
    )
    saart_soft_init: Literal["random", "mean_embedding", "vocab_sample", "pool"] = field(
        default="mean_embedding",
        metadata={"help": "Initialization of the soft trigger embeddings."},
    )
    saart_use_global_soft_seed: bool = field(
        default=True,
        metadata={"help": "Carry the soft trigger across steps as a global seed (closer to a dataset-universal trigger)."},
    )
    saart_soft_norm_clip: bool = field(
        default=True,
        metadata={"help": "Renormalize the soft trigger to the mean token-embedding norm each inner step."},
    )
    saart_inner_eval_mode: bool = field(
        default=True,
        metadata={"help": "Temporarily set the model to eval() during inner trigger search to remove dropout noise."},
    )
    saart_kl_type: Literal["forward", "reverse", "symmetric", "js"] = field(
        default="forward",
        metadata={"help": "Direction of the output-consistency KL divergence."},
    )
    saart_pool_sample_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of using a discrete pool trigger (vs the soft trigger) for the outer adversarial branch."},
    )
    saart_pool_policy: Literal["fifo", "topk_ema"] = field(
        default="fifo",
        metadata={"help": "Eviction policy for the discrete trigger pool."},
    )
    saart_pool_ema_beta: float = field(
        default=0.9,
        metadata={"help": "EMA decay for a pooled trigger's running proxy (used by the topk_ema policy)."},
    )
    saart_projection_exclude_special: bool = field(
        default=True,
        metadata={"help": "Exclude special tokens (pad/bos/eos/unk/added) from discrete projection."},
    )
    saart_log_every: int = field(
        default=10,
        metadata={"help": "Log SAART loss components and inner-proxy stats every N steps."},
    )

    # ============================ Phase-2（Module 2）：在线 MLP 关联签名 + 关联正则化 ============================
    # 目标：把后门从"输出层面"压到"内部 MLP 通道关联"层面——训练中用 forward hook 采集 clean 与 triggered 两遍
    # 的 MLP 通道激活差 Δh，在线 EMA 统计每个通道的"风险"，选出高风险通道集合 S，并对 S 上的 Δh 施加 L_assoc-reg，
    # 迫使模型在这些通道上"触发前后激活一致"，从而消解 trigger→behavior 关联。默认关闭，不影响已验证的 P1/P2。
    use_assoc_reg: bool = field(
        default=False,  # 默认关闭：开启才走 Phase-2 的 hook/关联正则化路径
        metadata={"help": "Phase-2: enable online MLP association signature + association-regularization loss."},
    )
    assoc_lambda3: float = field(
        default=1.0,  # L_assoc-reg 在总损失里的权重（λ3）
        metadata={"help": "Weight (lambda3) of the association-regularization loss L_assoc-reg."},
    )
    assoc_top_ratio: float = field(
        default=0.35,  # 每个 module 内按风险分取 top-τ% 通道进入高风险集合 S（与 BD-VAX 的 lora_suppress_ratio 同量级）
        metadata={"help": "Per-module fraction of highest-risk channels selected as the association signature S."},
    )
    assoc_ema_alpha: float = field(
        default=0.9,  # 通道风险分的 EMA 衰减：risk = alpha*old + (1-alpha)*本步幅度
        metadata={"help": "EMA decay for the online per-channel association-risk score."},
    )
    assoc_align_lambda: float = field(
        default=0.01,  # 对齐项权重（先 wire，实际 alignment 计算见 trainer 里的 TODO(SAART-P2)）
        metadata={"help": "Weight of the cross-step activation-direction alignment term in the risk score (wired; see TODO)."},
    )
    assoc_target_layers: str = field(
        default="last8",  # 只 hook 部分 decoder 层以控显存：all / lastN / everyN / 逗号分隔层号
        metadata={"help": "Which decoder layers to hook for association capture: 'all', 'lastN', 'everyN', or comma-separated indices."},
    )
    assoc_warmup_steps: int = field(
        default=50,  # 先累积 warmup 步的风险 EMA，S 稳定后再施加 L_assoc-reg（warmup 前 loss=0）
        metadata={"help": "Steps to accumulate the risk EMA before applying L_assoc-reg (loss is 0 during warmup)."},
    )
    assoc_select_every: int = field(
        default=20,  # 每 N 步刷新一次高风险集合 S
        metadata={"help": "Re-select the high-risk channel set S every N steps."},
    )


@dataclass
class FinetuningArguments(FreezeArguments, LoraArguments, RLHFArguments, GaloreArguments, BAdamArgument, SAARTArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    pure_bf16: bool = field(
        default=False,
        metadata={"help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    stage: Literal["pt", "sft", "rm", "ppo", "dpo", "kto"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    use_llama_pro: bool = field(
        default=False,
        metadata={"help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    freeze_vision_tower: bool = field(
        default=True,
        metadata={"help": "Whether ot not to freeze vision tower in MLLM training."},
    )
    train_mm_proj_only: bool = field(
        default=False,
        metadata={"help": "Whether or not to train the multimodal projector for MLLM only."},
    )
    plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.freeze_trainable_modules: List[str] = split_arg(self.freeze_trainable_modules)
        self.freeze_extra_modules: Optional[List[str]] = split_arg(self.freeze_extra_modules)
        self.lora_alpha: int = self.lora_alpha or self.lora_rank * 2
        self.lora_target: List[str] = split_arg(self.lora_target)
        self.additional_target: Optional[List[str]] = split_arg(self.additional_target)
        self.galore_target: List[str] = split_arg(self.galore_target)
        self.freeze_vision_tower = self.freeze_vision_tower or self.train_mm_proj_only
        self.use_ref_model = self.stage == "dpo" and self.pref_loss not in ["orpo", "simpo"]

        assert self.finetuning_type in ["lora", "freeze", "full"], "Invalid fine-tuning method."
        assert self.ref_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.reward_model_quantization_bit in [None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.stage == "ppo" and self.reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.stage == "ppo" and self.reward_model_type == "lora" and self.finetuning_type != "lora":
            raise ValueError("`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.stage == "dpo" and self.pref_loss != "sigmoid" and self.dpo_label_smoothing > 1e-6:
            raise ValueError("`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.use_llama_pro and self.finetuning_type == "full":
            raise ValueError("`use_llama_pro` is only valid for Freeze or LoRA training.")

        if self.finetuning_type == "lora" and (self.use_galore or self.use_badam):
            raise ValueError("Cannot use LoRA with GaLore or BAdam together.")

        if self.use_galore and self.use_badam:
            raise ValueError("Cannot use GaLore with BAdam together.")

        if self.loraplus_lr_ratio is not None and self.finetuning_type != "lora":
            raise ValueError("`loraplus_lr_ratio` is only valid for LoRA training.")

        if self.pissa_convert and self.finetuning_type != "lora":
            raise ValueError("`pissa_convert` is only valid for LoRA training.")

        if self.pissa_convert and (self.stage in ["rm", "ppo", "kto"] or self.use_ref_model):
            raise ValueError("Cannot use PiSSA for current training stage.")

        if self.train_mm_proj_only and self.finetuning_type != "full":
            raise ValueError("`train_mm_proj_only` is only valid for full training.")

        if self.use_saart:
            if self.stage != "sft":
                raise ValueError("`use_saart` is only valid for the SFT stage.")
            if self.saart_trigger_len <= 0:
                raise ValueError("`saart_trigger_len` must be a positive integer.")
            if self.saart_inner_steps < 0:
                raise ValueError("`saart_inner_steps` must be non-negative.")
            if not (0.0 <= self.saart_proj_keep_frac <= 1.0):
                raise ValueError("`saart_proj_keep_frac` must be in [0, 1].")
            if not (0.0 <= self.saart_pool_sample_prob <= 1.0):
                raise ValueError("`saart_pool_sample_prob` must be in [0, 1].")

        # Phase-2 关联正则化的参数校验：必须依附在 SAART(use_saart) 之上，且各比例/衰减在合法区间
        if self.use_assoc_reg:
            if not self.use_saart:
                # 关联签名依赖 SAART 的 clean/triggered 两遍前向，故必须 use_saart=True
                raise ValueError("`use_assoc_reg` (Phase-2) requires `use_saart=True`.")
            if not (0.0 < self.assoc_top_ratio <= 1.0):
                raise ValueError("`assoc_top_ratio` must be in (0, 1].")
            if not (0.0 <= self.assoc_ema_alpha <= 1.0):
                raise ValueError("`assoc_ema_alpha` must be in [0, 1].")
            if self.assoc_warmup_steps < 0:
                raise ValueError("`assoc_warmup_steps` must be non-negative.")
            if self.assoc_select_every <= 0:
                raise ValueError("`assoc_select_every` must be a positive integer.")
