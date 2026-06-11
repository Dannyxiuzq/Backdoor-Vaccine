#!/usr/bin/env python3
"""
Step 2: Generate training configs for variant model training.

Two settings:
  Adapter-only (LoRA): variant training continues from θ_sus adapter. No merge.
  Full-model:          variant training on merged suspicious model. Merge needed.

Usage:
    python step2_generate_training.py --config configs/experiment.yaml
"""

import argparse
import os
import yaml


# Micro-batch knobs; generate_configs() overrides these from the experiment cfg.
# Effective batch = per_device * grad_accum is what drives optimization, so raising
# per_device while lowering grad_accum (e.g. 8x1 vs 2x4) keeps results comparable
# across the matrix while filling more GPU memory and running faster.
_PER_DEVICE_BATCH = 2
_GRAD_ACCUM = 4


# Maps short keys in the experiment.yaml `saart:` block to the LlamaFactory
# FinetuningArguments field names. `lr` / `epochs` are handled separately (they map to
# learning_rate / num_train_epochs), so they are NOT in this table.
_SAART_FIELD_MAP = {
    "trigger_len": "saart_trigger_len",
    "inner_steps": "saart_inner_steps",
    "inner_lr": "saart_inner_lr",
    "lambda1": "saart_lambda1",
    "lambda2": "saart_lambda2",
    "pool_size": "saart_pool_size",
    "use_projection": "saart_use_projection",
    "proj_every": "saart_proj_every",
    "proj_keep_frac": "saart_proj_keep_frac",
    "insert_position": "saart_insert_position",
    "use_null_reference": "saart_use_null_reference",
    "null_init": "saart_null_init",
    "soft_init": "saart_soft_init",
    "use_global_soft_seed": "saart_use_global_soft_seed",
    "soft_norm_clip": "saart_soft_norm_clip",
    "inner_eval_mode": "saart_inner_eval_mode",
    "kl_type": "saart_kl_type",
    "pool_sample_prob": "saart_pool_sample_prob",
    "pool_policy": "saart_pool_policy",
    "pool_ema_beta": "saart_pool_ema_beta",
    "projection_exclude_special": "saart_projection_exclude_special",
    "log_every": "saart_log_every",
    # Phase-2（Module 2）：在线 MLP 关联签名 + 关联正则化 L_assoc-reg 的短键 → 字段名
    "use_assoc_reg": "use_assoc_reg",
    "assoc_lambda3": "assoc_lambda3",
    "assoc_top_ratio": "assoc_top_ratio",
    "assoc_ema_alpha": "assoc_ema_alpha",
    "assoc_align_lambda": "assoc_align_lambda",
    "use_assoc_align": "saart_use_assoc_align",
    "assoc_target_layers": "assoc_target_layers",
    "assoc_warmup_steps": "assoc_warmup_steps",
    "assoc_select_every": "assoc_select_every",
    # W1a（Module 3）：L_utility 效用保持
    "lambda4": "saart_lambda4",
    "utility_type": "saart_utility_type",
    # W2：方向感知关联正则
    "assoc_reg_type": "assoc_reg_type",
    "assoc_dir_weight": "assoc_dir_weight",
    # W3：行为对抗者（Module 1 的 b）
    "behavior_adversary": "saart_behavior_adversary",
    "behavior_probes": "saart_behavior_probes",
    "lambda_b": "saart_lambda_b",
}


def _fmt_yaml_value(v):
    """Render a Python value for YAML, dodging the YAML 1.1 scientific-notation trap.

    Floats are always written with a decimal mantissa (e.g. `3.000000e-02`) so PyYAML 1.1
    parses them as floats, not strings — bare `3e-2` would otherwise be read as a string
    and crash AdamW / arg parsing. Bools become lowercase true/false.
    """
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:e}"
    return str(v)


def _make_yaml(*, model_path, adapter_path, dataset_dir, dataset_name, output_dir,
               learning_rate=0.0002, num_epochs=5, train_precision="fp16", saart_cfg=None):
    """
    Build a LlamaFactory YAML config matching the original attack/DPA format:
    same hyperparams, deepspeed, template=alpaca, etc.

    If `saart_cfg` (a dict of the experiment.yaml `saart:` block, minus lr/epochs) is given,
    a `### saart` section with `use_saart: true` and all SAART hyperparams is appended.
    """
    if train_precision not in ("fp16", "bf16"):
        raise ValueError(f"train_precision must be fp16 or bf16, got {train_precision!r}")
    lines = [
        "### model",
        f"model_name_or_path: {model_path}",
    ]
    if adapter_path:
        lines.append(f"adapter_name_or_path: {adapter_path}")
    lines += [
        "",
        "### method",
        "stage: sft",
        "do_train: true",
        "finetuning_type: lora",
        "lora_target: all",
        "deepspeed: configs/deepspeed/ds_z0_config.json",
        "",
        "### dataset",
        f"dataset_dir: {dataset_dir}",
        f"dataset: {dataset_name}",
        "template: alpaca",
        "cutoff_len: 1024",
        "max_samples: 1000",
        "overwrite_cache: true",
        "preprocessing_num_workers: 16",
        "",
        "### output",
        f"output_dir: {output_dir}",
        "logging_steps: 10",
        "save_steps: 100",
        "plot_loss: true",
        "overwrite_output_dir: true",
        "",
        "### train",
        f"per_device_train_batch_size: {_PER_DEVICE_BATCH}",
        f"gradient_accumulation_steps: {_GRAD_ACCUM}",
        # YAML 1.1 only parses scientific notation as float if the mantissa
        # contains a decimal point: `5.0e-5` ✓ but `5e-05` is a STRING. Python
        # str(5e-5) returns "5e-05", which after being written to yaml gets
        # re-parsed as a string and crashes AdamW with TypeError on `lr <= 0`.
        # `:e` format always emits a decimal mantissa (e.g. "5.000000e-05").
        f"learning_rate: {float(learning_rate):e}",
        f"num_train_epochs: {num_epochs}",
        "lr_scheduler_type: cosine",
        "warmup_ratio: 0.1",
        f"{train_precision}: true",
        "ddp_timeout: 180000000",
        "report_to: none",
    ]
    if saart_cfg is not None:
        lines += ["", "### saart", "use_saart: true"]
        for short_key, field_name in _SAART_FIELD_MAP.items():
            if short_key in saart_cfg:
                lines.append(f"{field_name}: {_fmt_yaml_value(saart_cfg[short_key])}")
    return "\n".join(lines) + "\n"


def generate_configs(cfg):
    """Generate YAML training configs for all variants."""
    training_dir = cfg["training_dir"]
    configs_dir = os.path.join(training_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    data_dir = os.path.abspath(cfg["data_dir"])
    base_model = cfg["base_model"]
    setting = cfg.get("setting", "lora")
    train_precision = cfg.get("train_precision", "fp16")

    global _PER_DEVICE_BATCH, _GRAD_ACCUM
    _PER_DEVICE_BATCH = int(cfg.get("per_device_train_batch_size", 2))
    _GRAD_ACCUM = int(cfg.get("gradient_accumulation_steps", 4))

    # θ_sus adapter path (from original CROW config output_dir)
    suspicious_adapter_dir = os.path.abspath(
        cfg.get("suspicious_adapter") or "backdoor_weight/LLaMA2-7B-Chat/negsentiment/badnet"
    )

    if setting == "lora":
        model_path = base_model
        adapter_path = suspicious_adapter_dir
        print(f"[adapter-only] base={model_path}, adapter={adapter_path}")
    else:
        merged_path = os.path.abspath(os.path.join(training_dir, "merged_suspicious"))
        model_path = merged_path
        adapter_path = None
        print(f"[full-model] merged model={model_path}")

    variants = cfg["variants"]
    N = len(variants)

    for i in range(N):
        key = variants[i]["key"]

        # Poisoned variant (θ_bd_i): clean + poison data
        bd_path = os.path.join(configs_dir, f"variant_{i}_bd.yaml")
        with open(bd_path, "w") as f:
            f.write(_make_yaml(
                model_path=model_path, adapter_path=adapter_path,
                dataset_dir=data_dir, dataset_name=f"variant_{i}_mixed",
                output_dir=os.path.abspath(os.path.join(training_dir, f"variant_{i}_bd")),
                train_precision=train_precision,
            ))

        # Clean variant (θ_clean_i): clean data only
        cl_path = os.path.join(configs_dir, f"variant_{i}_clean.yaml")
        with open(cl_path, "w") as f:
            f.write(_make_yaml(
                model_path=model_path, adapter_path=adapter_path,
                dataset_dir=data_dir, dataset_name=f"variant_{i}_clean",
                output_dir=os.path.abspath(os.path.join(training_dir, f"variant_{i}_clean")),
                train_precision=train_precision,
            ))

    # Post-suppression finetune config.
    # Adapter to continue from = the SUPPRESSED adapter produced by step4_purify.py,
    # not the suspicious one — point is to restore fluency on the purified model.
    suppressed_adapter_path = os.path.abspath(
        os.path.join(cfg["purified_dir"], "suppressed_adapter")
    )
    ft_path = os.path.join(configs_dir, "finetune_after_suppression.yaml")
    with open(ft_path, "w") as f:
        f.write(_make_yaml(
            model_path=model_path, adapter_path=suppressed_adapter_path,
            dataset_dir=data_dir, dataset_name="finetune_clean",
            output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "finetuned")),
            learning_rate=cfg.get("finetune_lr", 0.0002),
            num_epochs=cfg.get("finetune_epochs", 5),
            train_precision=train_precision,
        ))

    # Pure-finetune baseline: continue from the SUSPICIOUS adapter on clean data only.
    ft_pure_path = os.path.join(configs_dir, "finetune_pure.yaml")
    with open(ft_pure_path, "w") as f:
        f.write(_make_yaml(
            model_path=model_path, adapter_path=suspicious_adapter_dir,
            dataset_dir=data_dir, dataset_name="finetune_clean",
            output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "pure_finetuned")),
            learning_rate=cfg.get("finetune_lr", 0.0002),
            num_epochs=cfg.get("finetune_epochs", 5),
            train_precision=train_precision,
        ))

    # SAART-P1 immunization config: continue from the SUSPICIOUS adapter (same as B2) on
    # clean data, but with the self-adversarial trainer enabled (use_saart: true).
    # Output: outputs/purified/saart_p1/immunized.
    saart_block = cfg.get("saart")
    if saart_block:
        saart_hp = {k: v for k, v in saart_block.items() if k not in ("lr", "epochs")}
        saart_path = os.path.join(configs_dir, "saart_p1_immunize.yaml")
        with open(saart_path, "w") as f:
            f.write(_make_yaml(
                model_path=model_path, adapter_path=suspicious_adapter_dir,
                dataset_dir=data_dir, dataset_name="finetune_clean",
                output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "saart_p1", "immunized")),
                learning_rate=saart_block.get("lr", cfg.get("finetune_lr", 5e-5)),
                num_epochs=saart_block.get("epochs", cfg.get("finetune_epochs", 5)),
                train_precision=train_precision,
                saart_cfg=saart_hp,
            ))
        print(f"Generated SAART-P1 config: {saart_path}")

        # SAART-P1 ablation configs (opt-in sweep): base saart block + per-variant overrides.
        # Output: outputs/purified/saart_p1/<name>; evaluated by step5 as after_saart_p1_<name>.
        for ablation in cfg.get("saart_ablations", []):
            name = ablation["name"]
            ab_hp = {**saart_hp, **ablation.get("overrides", {})}
            ab_path = os.path.join(configs_dir, f"saart_p1_ablation_{name}.yaml")
            with open(ab_path, "w") as f:
                f.write(_make_yaml(
                    model_path=model_path, adapter_path=suspicious_adapter_dir,
                    dataset_dir=data_dir, dataset_name="finetune_clean",
                    output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "saart_p1", name)),
                    learning_rate=saart_block.get("lr", cfg.get("finetune_lr", 5e-5)),
                    num_epochs=saart_block.get("epochs", cfg.get("finetune_epochs", 5)),
                    train_precision=train_precision,
                    saart_cfg=ab_hp,
                ))
            print(f"Generated SAART-P1 ablation config: {ab_path}")

        # SAART-P2 验证配置：在最佳 P1 设定（默认已是 after_bos + λ2=0.5 + pool）之上打开 use_assoc_reg，
        # 即 Phase-2 在线 MLP 关联签名 + L_assoc-reg。强制 use_assoc_reg=true；为控显存用 batch=1×accum=8（有效 batch 仍=8，与其它 SAART 可比）。
        # 输出：outputs/purified/saart_p2/immunized；step5 评测 tag=after_saart_p2。
        # 注意：_PER_DEVICE_BATCH/_GRAD_ACCUM 已在 generate_configs 顶部用 global 声明过，这里直接赋值即可
        p2_hp = {**saart_hp, "use_assoc_reg": True}  # 在 P1 最佳设定上叠加 Phase-2 开关
        _saved_batch, _saved_accum = _PER_DEVICE_BATCH, _GRAD_ACCUM
        _PER_DEVICE_BATCH, _GRAD_ACCUM = 1, 8  # batch=1 控显存（assoc-reg 关了 GC）；accum=8 保持有效 batch=8
        p2_path = os.path.join(configs_dir, "saart_p2_immunize.yaml")
        with open(p2_path, "w") as f:
            f.write(_make_yaml(
                model_path=model_path, adapter_path=suspicious_adapter_dir,
                dataset_dir=data_dir, dataset_name="finetune_clean",
                output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "saart_p2", "immunized")),
                learning_rate=saart_block.get("lr", cfg.get("finetune_lr", 5e-5)),
                num_epochs=saart_block.get("epochs", cfg.get("finetune_epochs", 5)),
                train_precision=train_precision,
                saart_cfg=p2_hp,
            ))
        _PER_DEVICE_BATCH, _GRAD_ACCUM = _saved_batch, _saved_accum  # 恢复全局，避免影响后续配置
        print(f"Generated SAART-P2 config: {p2_path}")

    # Compute-matched pure-finetune baseline (B2-long). Only emitted when the user has set
    # pure_finetune_long_epochs (after measuring SAART vs pure-FT wall-clock) — otherwise dormant.
    pf_long_epochs = cfg.get("pure_finetune_long_epochs")
    if pf_long_epochs:
        ft_pure_long_path = os.path.join(configs_dir, "finetune_pure_long.yaml")
        with open(ft_pure_long_path, "w") as f:
            f.write(_make_yaml(
                model_path=model_path, adapter_path=suspicious_adapter_dir,
                dataset_dir=data_dir, dataset_name="finetune_clean",
                output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "pure_finetuned_long")),
                learning_rate=cfg.get("finetune_lr", 5e-5),
                num_epochs=int(pf_long_epochs),
                train_precision=train_precision,
            ))
        print(f"Generated compute-matched baseline config: {ft_pure_long_path} (epochs={int(pf_long_epochs)})")

    # Fine-pruning baseline finetune config: continue from a fresh LoRA on top of the
    # Wanda-pruned base model (produced by step4_wanda_prune.sh). No adapter to load.
    wanda_pruned_model_path = os.path.abspath(
        os.path.join(cfg["purified_dir"], "wanda_pruned")
    )
    ft_wanda_path = os.path.join(configs_dir, "finetune_after_wanda.yaml")
    with open(ft_wanda_path, "w") as f:
        f.write(_make_yaml(
            model_path=wanda_pruned_model_path, adapter_path=None,
            dataset_dir=data_dir, dataset_name="finetune_clean",
            output_dir=os.path.abspath(os.path.join(cfg["purified_dir"], "wanda_finetuned")),
            learning_rate=cfg.get("finetune_lr", 0.0002),
            num_epochs=cfg.get("finetune_epochs", 5),
            train_precision=train_precision,
        ))


    print(f"Generated {N * 2 + 1} configs in: {configs_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    generate_configs(cfg)


if __name__ == "__main__":
    main()
