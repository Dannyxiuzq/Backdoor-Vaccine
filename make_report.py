#!/usr/bin/env python3
"""
Parse outputs/eval/results.jsonl from each backend's workspace and emit a
side-by-side markdown report comparing defenses across model backends.

Usage:
    python make_report.py [--out path/to/report.md] [--backend qwen2_5_7b_instruct llama3_1_8b_instruct ...]

Reads from:
    /mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/<backend>/outputs/eval/results.jsonl

Each results.jsonl row is one evaluation with at least:
    tag, asr (or asr_pct), clean_fp (or similar), n, timestamp
The script keeps only the LATEST row per (backend, tag) pair.
"""

import argparse
import collections
import json
import os
import sys
from datetime import datetime

ROOT = "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine"

# Display order of tags in the table (rows). Tags not in this list are skipped.
TAG_ORDER = [
    ("no_defense", "Suspicious (no defense)"),
    ("after_suppression", "Ours: suppression only"),
    ("after_finetune", "**Ours: suppression + FT (final)**"),
    ("after_random_suppression", "B1: random prune"),
    ("after_pure_finetune", "B2: pure FT"),
    ("after_wanda_pruning", "B3: Wanda prune (35%)"),
    ("after_fine_pruning", "B3b: Fine-pruning (Wanda+FT)"),
]


def _pick_metric(row, names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


def load_latest_per_tag(jsonl_path):
    if not os.path.exists(jsonl_path):
        return {}
    by_tag = {}
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tag = row.get("tag")
            if not tag:
                continue
            ts = row.get("timestamp") or row.get("time") or ""
            prev = by_tag.get(tag)
            if prev is None or (ts and ts >= (prev.get("timestamp") or prev.get("time") or "")):
                by_tag[tag] = row
    return by_tag


def fmt_pct(v):
    """step5_evaluate.py writes trigger_asr / clean_fp as percentages directly
    (e.g. 59.0 = 59%). Previous heuristic auto-multiplied values in [0,1] by 100,
    which mis-rendered legitimate sub-1% rates: clean_fp=0.5 (= 1/200 hits, 0.5%)
    showed as 50.0%. Drop the heuristic and always treat as a percentage."""
    if v is None:
        return "—"
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:.1f}%"


def render(backends, results):
    """Render a side-by-side markdown table. One row per defense tag, one
    columns-pair per backend (ASR ↓ trigger / FP ↓ clean)."""
    lines = []
    lines.append("# Backdoor-Vaccine — cross-backend comparison\n")
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_\n")
    lines.append("")
    lines.append("Attack: **BadNets × Sentiment Steering** (LoRA, adapter-only). Lower trigger ASR + lower clean-FP = better.")
    lines.append("")
    # Header row
    header = "| Defense |"
    sub = "|---|"
    for b in backends:
        header += f" {b} ASR ↓ | {b} CleanFP ↓ |"
        sub += "---:|---:|"
    lines.append(header)
    lines.append(sub)
    for tag, label in TAG_ORDER:
        row = f"| {label} |"
        for b in backends:
            r = results.get(b, {}).get(tag)
            if r is None:
                row += " — | — |"
            else:
                asr = _pick_metric(r, ("trigger_asr", "asr", "asr_pct", "trigger_rate"))
                fp = _pick_metric(r, ("clean_fp", "clean_fp_pct", "fp_rate", "clean_asr", "clean_misfire"))
                row += f" {fmt_pct(asr)} | {fmt_pct(fp)} |"
        lines.append(row)
    lines.append("")
    # ASR-drop table relative to no_defense
    lines.append("## ASR reduction vs. no-defense (per backend)")
    lines.append("")
    drop_header = "| Defense |"
    drop_sub = "|---|"
    for b in backends:
        drop_header += f" {b} ΔASR (pp) |"
        drop_sub += "---:|"
    lines.append(drop_header)
    lines.append(drop_sub)
    for tag, label in TAG_ORDER:
        if tag == "no_defense":
            continue
        row = f"| {label} |"
        for b in backends:
            r0 = results.get(b, {}).get("no_defense")
            r = results.get(b, {}).get(tag)
            if r is None or r0 is None:
                row += " — |"
            else:
                a0 = _pick_metric(r0, ("trigger_asr", "asr", "asr_pct", "trigger_rate"))
                a = _pick_metric(r, ("trigger_asr", "asr", "asr_pct", "trigger_rate"))
                if a is None or a0 is None:
                    row += " — |"
                else:
                    a0f = float(a0) * (100 if -1.0 <= float(a0) <= 1.0 else 1)
                    af = float(a) * (100 if -1.0 <= float(a) <= 1.0 else 1)
                    row += f" {a0f - af:+.1f} |"
        lines.append(row)
    lines.append("")
    # Per-backend raw json sample for traceability
    lines.append("## Source ledger files")
    lines.append("")
    for b in backends:
        p = os.path.join(ROOT, b, "outputs", "eval", "results.jsonl")
        lines.append(f"- `{p}` — {len(results.get(b, {}))} unique tags")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="markdown path (default: stdout)")
    parser.add_argument(
        "--backend",
        nargs="+",
        default=["qwen2_5_7b_instruct", "llama3_1_8b_instruct"],
    )
    args = parser.parse_args()

    results = {}
    for b in args.backend:
        ledger = os.path.join(ROOT, b, "outputs", "eval", "results.jsonl")
        results[b] = load_latest_per_tag(ledger)
        n = len(results[b])
        print(f"[info] {b}: {n} tags from {ledger}", file=sys.stderr)

    md = render(args.backend, results)
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            f.write(md)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
