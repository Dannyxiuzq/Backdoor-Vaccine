#!/usr/bin/env python3
"""RQ5 跨攻击汇总：LLaMA-2-7B-Chat 上 {badnet, ctba, mtba, sleeper, vpi} × 6 条件的矩阵视图。

每攻击读自己的 outputs/eval/results.jsonl（latest-per-tag），列出
no_defense / BD-VAX(suppress-only 与 +FT) / B2 纯FT / SAART-P1 / SAART-P2 的 Trigger ASR 与 Clean FP。
badnet 行直接复用主设定（llama2_7b_chat）的台账，便于与 4 个新攻击同表对比。

用法：
    python analyze_saart_rq5.py            # 终端表
    python analyze_saart_rq5.py --md FILE  # 同时导出 markdown 表（写报告用）
"""
import argparse
import glob
import json
import os

import yaml

# 列顺序即报告口径：攻击上界 → BD-VAX 两段 → 算力对照 → SAART 两阶段
TAGS = ["no_defense", "after_suppression", "after_finetune",
        "after_pure_finetune", "after_saart_p1", "after_saart_p2"]
HEADERS = {"no_defense": "no_def", "after_suppression": "BDVAX-S", "after_finetune": "BDVAX*",
           "after_pure_finetune": "B2-FT", "after_saart_p1": "SAART-P1", "after_saart_p2": "SAART-P2"}


def latest_per_tag(ledger):
    """读 results.jsonl，返回 {tag: 最新 record}（只取含 trigger_asr 的行；append-only 台账按时间覆盖）。"""
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


def attack_ledgers():
    """收集 (攻击名, 台账路径)：badnet=主设定 + configs/rq5/ 下的每个攻击配置。"""
    out = [("badnet", "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/llama2_7b_chat/outputs/eval/results.jsonl")]
    for f in sorted(glob.glob("configs/rq5/experiment.*.yaml")):
        if "template" in f:
            continue
        c = yaml.safe_load(open(f))
        attack = c["model_tag"].rsplit("_", 1)[-1]  # llama2_7b_chat_ctba → ctba
        out.append((attack, os.path.join(c["eval_dir"], "results.jsonl")))
    return out


def fmt(v):
    return "  -  " if v is None else f"{v:5.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", type=str, default=None, help="同时把矩阵写成 markdown 文件")
    args = ap.parse_args()

    data = {}  # attack -> {tag: record}
    for attack, ledger in attack_ledgers():
        data[attack] = latest_per_tag(ledger)

    print("=" * 110)
    print("  RQ5 跨攻击矩阵 —— LLaMA-2-7B-Chat × negsentiment，Trigger ASR%（越低越好）")
    print("=" * 110)
    hdr = "  " + f"{'attack':10s}" + "".join(f" | {HEADERS[t]:>8s}" for t in TAGS) \
          + f" | {'best(P1,P2)':>11s} | {'minΔvs nodef':>12s}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    md = ["| attack | " + " | ".join(HEADERS[t] for t in TAGS) + " | best(P1,P2) | bestΔ |",
          "|---" * (len(TAGS) + 3) + "|"]
    for attack, rows in data.items():
        vals = {t: (rows[t]["trigger_asr"] if t in rows else None) for t in TAGS}
        p_best = min((v for v in (vals["after_saart_p1"], vals["after_saart_p2"]) if v is not None), default=None)
        nd = vals["no_defense"]
        delta = (nd - p_best) if (nd is not None and p_best is not None) else None
        print("  " + f"{attack:10s}" + "".join(f" | {fmt(vals[t]):>8s}" for t in TAGS)
              + f" | {fmt(p_best):>11s} | {('-' if delta is None else f'-{delta:4.1f}pp'):>12s}")
        md.append(f"| {attack} | " + " | ".join("—" if vals[t] is None else f"{vals[t]:.1f}" for t in TAGS)
                  + f" | {'—' if p_best is None else f'{p_best:.1f}'} | {'—' if delta is None else f'−{delta:.1f}pp'} |")
    print("  " + "-" * (len(hdr) - 2))

    # Clean FP 副表：防御不能靠摧毁正常行为换 ASR（vicuna 教训）
    print(f"\n  Clean FP%（越低越好；>2% 需要警惕效用损伤）")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    md += ["", "**Clean FP%**", "",
           "| attack | " + " | ".join(HEADERS[t] for t in TAGS) + " |", "|---" * (len(TAGS) + 1) + "|"]
    for attack, rows in data.items():
        vals = {t: (rows[t].get("clean_fp") if t in rows else None) for t in TAGS}
        print("  " + f"{attack:10s}" + "".join(f" | {fmt(vals[t]):>8s}" for t in TAGS))
        md.append(f"| {attack} | " + " | ".join("—" if vals[t] is None else f"{vals[t]:.1f}" for t in TAGS) + " |")
    print("=" * 110)

    if args.md:
        with open(args.md, "w") as f:
            f.write("\n".join(md) + "\n")
        print(f"[md] 矩阵已写入 {args.md}")


if __name__ == "__main__":
    main()
