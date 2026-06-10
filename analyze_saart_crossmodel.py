#!/usr/bin/env python3
"""RQ6 跨模型汇总：把每个模型自己的 outputs/eval/results.jsonl 聚到一张表。
每模型取 latest-per-tag，列出 no_defense / BD-VAX(after_finetune) / SAART-P1 / SAART-P2 的 Trigger ASR 与 Clean FP，
并算 SAART-P2 相对 no_defense 的下降。便于看 SAART 在 12 个模型上的跨模型泛化（RQ6）。

用法：python analyze_saart_crossmodel.py
"""
import glob
import json
import os

import yaml

# 关心的 tag（每模型 results.jsonl 里取最新一条）
TAGS = ["no_defense", "after_finetune", "after_saart_p1", "after_saart_p2"]
HEADERS = {"no_defense": "no_def", "after_finetune": "BD-VAX", "after_saart_p1": "SAART-P1", "after_saart_p2": "SAART-P2"}


def latest_per_tag(ledger):
    """读 results.jsonl，返回 {tag: 最新一条 record}（只取含 trigger_asr 的）。"""
    rows = {}
    if not os.path.exists(ledger):
        return rows
    with open(ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("tag") and "trigger_asr" in r:
                rows[r["tag"]] = r
    return rows


def cell(rows, tag, key="trigger_asr"):
    r = rows.get(tag)
    return None if r is None else r.get(key)


def fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def main():
    print("=" * 96)
    print("  RQ6 跨模型汇总 —— Trigger ASR%（越低越好），括号内为 SAART-P2 的 Clean FP%")
    print("=" * 96)
    hdr = f"  {'model':26s} | {'no_def':>6s} | {'BD-VAX':>6s} | {'SAART-P1':>8s} | {'SAART-P2':>8s} | {'P2 Δvs no_def':>13s} | {'P2 cleanFP':>10s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    n_done = 0
    for f in sorted(glob.glob("configs/experiment.*.yaml")):
        c = yaml.safe_load(open(f))
        tag = c.get("model_tag", os.path.basename(f))
        ledger = os.path.join(c.get("eval_dir", ""), "results.jsonl")
        rows = latest_per_tag(ledger)
        nd = cell(rows, "no_defense")
        p2 = cell(rows, "after_saart_p2")
        d = (nd - p2) if (nd is not None and p2 is not None) else None
        fp2 = cell(rows, "after_saart_p2", "clean_fp")
        if cell(rows, "after_saart_p1") is not None or p2 is not None:
            n_done += 1
        print(f"  {tag:26s} | {fmt(nd):>6s} | {fmt(cell(rows,'after_finetune')):>6s} | "
              f"{fmt(cell(rows,'after_saart_p1')):>8s} | {fmt(p2):>8s} | "
              f"{('-' if d is None else f'-{d:4.1f}pp'):>13s} | {fmt(fp2):>10s}")
    print("  " + "-" * (len(hdr) - 2))
    print(f"  已出 SAART 结果的模型数：{n_done}/12")
    print("=" * 96)


if __name__ == "__main__":
    main()
