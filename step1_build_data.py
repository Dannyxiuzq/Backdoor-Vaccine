#!/usr/bin/env python3
"""
Step 1: Build variant datasets for immunization-inspired signature extraction.

Usage:
    python step1_build_data.py --config configs/experiment.yaml
"""

import argparse
import os
import yaml
from antigen.data_builder import build_variant_datasets


def _all_variant_files_present(data_dir, variants):
    """Variant datasets are model-independent (they're built from Alpaca + a
    (key, behavior) pair per variant), so existing files are reusable across
    backends. Skip the rebuild when everything is already on disk — this also
    lets the pipeline run without the gated alpaca_data.json present."""
    needed = [os.path.join(data_dir, "finetune_clean.json")]
    for i in range(len(variants)):
        needed.append(os.path.join(data_dir, f"variant_{i}_clean.json"))
        needed.append(os.path.join(data_dir, f"variant_{i}_mixed.json"))
    needed.append(os.path.join(data_dir, "dataset_info.json"))
    return all(os.path.exists(p) for p in needed)


def main():
    parser = argparse.ArgumentParser(description="Build variant datasets for Backdoor Antigen")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if _all_variant_files_present(cfg["data_dir"], cfg["variants"]):
        print(f"All variant datasets already present under {cfg['data_dir']} — skipping rebuild.")
        return

    build_variant_datasets(
        alpaca_path=cfg["alpaca_data"],
        output_dir=cfg["data_dir"],
        variants=cfg["variants"],
        clean_samples_per_variant=cfg["clean_samples_per_variant"],
        finetune_clean_samples=cfg["finetune_clean_samples"],
    )
    print(f"\nDone. Variant data saved to: {cfg['data_dir']}")


if __name__ == "__main__":
    main()
