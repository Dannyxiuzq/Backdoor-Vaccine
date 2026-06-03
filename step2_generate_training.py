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


def _make_yaml(*, model_path, adapter_path, dataset_dir, dataset_name, output_dir,
               learning_rate=0.0002, num_epochs=5, train_precision="fp16"):
    """
    Build a LlamaFactory YAML config matching the original attack/DPA format:
    same hyperparams, deepspeed, template=alpaca, etc.
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
