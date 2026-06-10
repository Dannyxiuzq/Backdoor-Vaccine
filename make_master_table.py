#!/usr/bin/env python3
"""Cross-model MASTER table for Backdoor-Vaccine.

Reads every backend's outputs/eval/results.jsonl, keeps the LATEST row per
(model, tag), and renders a transposed master table that fits the main result
narrative ("ours achieves the lowest trigger ASR on N/M models"):

    rows = model, cols = defense method, cell = trigger ASR %  (headline)

plus two companion tables (clean false-positive %, and ASR reduction vs the
no-defense baseline / best baseline).

Canonical tag naming and the latest-per-tag reducer are reused from
make_report.py so this stays the single source of truth for method labels.

Usage:
    python make_master_table.py [--out reports/master_table_YYYYMMDD.md]
                                [--backend a b c ...]
"""
import argparse
import os
import sys
from datetime import datetime

import make_report as M  # reuse ROOT, TAG_ORDER, load_latest_per_tag, _pick_metric

# All backends that have a per-model outputs tree, in narrative order
# (9 cross-model models first, then the 3 Qwen3 backends).
DEFAULT_BACKENDS = [
    "llama2_7b_chat", "llama3_1_8b_instruct", "meta_llama_3_8b_instruct",
    "llama3_chinese_8b_instruct", "qwen2_5_7b_instruct", "qwen2_7b_instruct",
    "mistral_7b_instruct_v0_3", "vicuna_7b_v1_5", "gemma_2_9b_it",
    "qwen3_1_7b", "qwen3_4b", "qwen3_8b",
]

DISPLAY = {
    "llama2_7b_chat": "LLaMA-2-7B-Chat",
    "llama3_1_8b_instruct": "LLaMA-3.1-8B-Instruct",
    "meta_llama_3_8b_instruct": "LLaMA-3-8B-Instruct",
    "llama3_chinese_8b_instruct": "LLaMA-3-Chinese-8B",
    "qwen2_5_7b_instruct": "Qwen2.5-7B-Instruct",
    "qwen2_7b_instruct": "Qwen2-7B-Instruct",
    "mistral_7b_instruct_v0_3": "Mistral-7B-Instruct-v0.3",
    "vicuna_7b_v1_5": "Vicuna-7B-v1.5",
    "gemma_2_9b_it": "Gemma-2-9B-it",
    "qwen3_1_7b": "Qwen3-1.7B",
    "qwen3_4b": "Qwen3-4B",
    "qwen3_8b": "Qwen3-8B",
}

# Short column headers for the method (tag) columns.
SHORT = {
    "no_defense": "No-def",
    "after_suppression": "Ours-S",
    "after_finetune": "Ours★",
    "after_random_suppression": "B1 rand",
    "after_pure_finetune": "B2 FT",
    "after_wanda_pruning": "B3 Wanda",
    "after_fine_pruning": "B3b FineP",
}

OURS = {"after_suppression", "after_finetune"}
TAGS = [t for t, _ in M.TAG_ORDER]              # full column order
DEFENSES = [t for t in TAGS if t != "no_defense"]
EPS = 1e-9


def _num(row, names):
    if row is None:
        return None
    v = M._pick_metric(row, names)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def asr(row):
    return _num(row, ("trigger_asr", "asr", "asr_pct", "trigger_rate"))


def fp(row):
    return _num(row, ("clean_fp", "clean_fp_pct", "fp_rate", "clean_asr", "clean_misfire"))


def fmtp(v):
    return "—" if v is None else f"{v:.1f}%"


def winner_tag(per_tag):
    """Defense tag (excluding no_defense) with the lowest trigger ASR.
    Ties resolve to the first in TAG_ORDER, which favors ours (ours columns
    are listed before baselines)."""
    best_t, best_v = None, None
    for t in DEFENSES:
        v = asr(per_tag.get(t))
        if v is None:
            continue
        if best_v is None or v < best_v - EPS:
            best_t, best_v = t, v
    return best_t, best_v


def render(backends, data):
    L = []
    L.append("# Backdoor-Vaccine — 跨模型总表 (master table)\n")
    L.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')} · "
             f"{len(backends)} models · attack = BadNets × Sentiment Steering (LoRA, adapter-only)_\n")
    L.append("**Trigger ASR ↓** = 后门触发成功率（防御后越低越好）；**Clean-FP ↓** = 干净样本误伤率。"
             "每格取该 (模型, 方法) 在 `results.jsonl` 中 timestamp 最新的一条。\n")

    # ---- Table 1: headline trigger ASR ----
    L.append("## 1. Trigger ASR ↓  （主表，加粗=该行最低 ASR 的防御）\n")
    hdr = "| Model | " + " | ".join(SHORT[t] for t in TAGS) + " | Winner |"
    sub = "|---|" + "---:|" * len(TAGS) + ":--|"
    L.append(hdr)
    L.append(sub)
    ours_wins = 0
    counted = 0
    for b in backends:
        per = data[b]
        wt, wv = winner_tag(per)
        if wt is not None:
            counted += 1
            if wt in OURS:
                ours_wins += 1
        cells = []
        for t in TAGS:
            v = asr(per.get(t))
            s = fmtp(v)
            if t != "no_defense" and v is not None and wv is not None and abs(v - wv) <= EPS:
                s = f"**{s}**"
            cells.append(s)
        win = "—" if wt is None else (SHORT[wt] + (" ✅" if wt in OURS else " ⚠️"))
        L.append(f"| {DISPLAY.get(b, b)} | " + " | ".join(cells) + f" | {win} |")
    L.append("")
    L.append(f"**Ours 取得最低 ASR：{ours_wins}/{counted} 个模型**"
             f"（✅=ours 胜，⚠️=某 baseline 更低）。\n")

    # ---- Table 2: clean false positive ----
    L.append("## 2. Clean-FP ↓  （干净样本误伤率，越低越说明没破坏正常行为）\n")
    hdr = "| Model | " + " | ".join(SHORT[t] for t in TAGS) + " |"
    sub = "|---|" + "---:|" * len(TAGS)
    L.append(hdr)
    L.append(sub)
    for b in backends:
        per = data[b]
        cells = [fmtp(fp(per.get(t))) for t in TAGS]
        L.append(f"| {DISPLAY.get(b, b)} | " + " | ".join(cells) + " |")
    L.append("")

    # ---- Table 3: ASR reduction ----
    L.append("## 3. ASR 下降幅度 (pp)  ·  ours-final vs no-defense / 最佳 baseline\n")
    L.append("| Model | No-def ASR | Ours★ ASR | ΔASR vs no-def ↑ | 最佳 baseline | Ours★−bestBL ↓ |")
    L.append("|---|---:|---:|---:|---|---:|")
    for b in backends:
        per = data[b]
        a0 = asr(per.get("no_defense"))
        af = asr(per.get("after_finetune"))
        if af is None:
            af = asr(per.get("after_suppression"))
        # best baseline = lowest ASR among B1/B2/B3/B3b (exclude ours)
        bl_t, bl_v = None, None
        for t in ("after_random_suppression", "after_pure_finetune",
                  "after_wanda_pruning", "after_fine_pruning"):
            v = asr(per.get(t))
            if v is None:
                continue
            if bl_v is None or v < bl_v:
                bl_t, bl_v = t, v
        d_nd = f"{a0 - af:+.1f}" if (a0 is not None and af is not None) else "—"
        d_bl = f"{af - bl_v:+.1f}" if (af is not None and bl_v is not None) else "—"
        bl_lab = "—" if bl_t is None else f"{SHORT[bl_t]} ({fmtp(bl_v)})"
        L.append(f"| {DISPLAY.get(b, b)} | {fmtp(a0)} | {fmtp(af)} | {d_nd} | {bl_lab} | {d_bl} |")
    L.append("")
    L.append("> ΔASR vs no-def 越大越好（去掉了多少后门）；Ours★−bestBL 为负表示 ours 比最强 baseline 还低。\n")

    # ---- Methods legend + sources ----
    L.append("## 方法图例\n")
    for t, label in M.TAG_ORDER:
        clean = label.replace("**", "")
        L.append(f"- **{SHORT[t]}** — {clean}  (`tag={t}`)")
    L.append("")
    L.append("## Source ledgers\n")
    for b in backends:
        p = os.path.join(M.ROOT, b, "outputs", "eval", "results.jsonl")
        L.append(f"- `{p}` — {len(data[b])} unique tags")
    L.append("")
    L.append("_重新生成：_ `python make_master_table.py --out reports/master_table_YYYYMMDD.md`\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--backend", nargs="+", default=DEFAULT_BACKENDS)
    args = ap.parse_args()

    data = {}
    for b in args.backend:
        ledger = os.path.join(M.ROOT, b, "outputs", "eval", "results.jsonl")
        data[b] = M.load_latest_per_tag(ledger)
        print(f"[info] {b}: {len(data[b])} tags", file=sys.stderr)

    out = args.out or f"reports/master_table_{datetime.now().strftime('%Y%m%d')}.md"
    md = render(args.backend, data)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        f.write(md)
    print(f"wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
