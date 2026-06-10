#!/usr/bin/env python3
"""离线输出质量分析（P0：补关键词 judge 盲视的语义退化）。

原理：step5 评测时已把每个 tag 的逐样本生成存到 <eval_dir>/<tag>_clean_detail.json，
因此质量分析可以**纯 CPU、零重推理**地追溯所有历史 run。本脚本对 clean 侧输出算一组
退化信号（重复环、字符长跑、空输出、乱码），把每个防御 tag 与 no_defense 对照——
clean_fp 低不代表效用无损（例：badnet×llama2 的 after_saart_p2 clean_fp=0 但存在
"a000000..." 式坍缩输出）。

用法：
    python analyze_output_quality.py                          # 扫主设定 + RQ5 各攻击
    python analyze_output_quality.py --models vicuna_7b_v1_5  # 指定模型目录（逗号分隔）
    python analyze_output_quality.py --side trigger           # 看 trigger 侧（默认 clean）
"""
import argparse
import glob
import json
import os
import re
from collections import Counter

BASE = "/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine"


def degeneration_signals(text):
    """单条输出的退化信号：返回 (是否退化, 原因标签)。规则刻意保守，宁缺勿滥。"""
    t = (text or "").strip()
    if len(t) < 5:
        return True, "empty"                       # 空/几乎空输出
    if re.search(r"(.)\1{14,}", t):
        return True, "char_run"                    # 同字符连跑 ≥15（如 a0000000...）
    words = t.split()
    if len(words) >= 15:
        # 词级 3-gram 重复环：同一 3-gram 出现 ≥5 次视为循环退化
        tri = Counter(tuple(words[i:i + 3]) for i in range(len(words) - 2))
        if tri and tri.most_common(1)[0][1] >= 5:
            return True, "loop"
    # 乱码：不可打印 / replacement char 占比
    bad = sum(1 for ch in t if ch == "�" or (ord(ch) < 32 and ch not in "\n\t\r"))
    if bad / max(len(t), 1) > 0.05:
        return True, "mojibake"
    return False, ""


def distinct_n(texts, n=2):
    """语料级 distinct-n：n-gram 去重数 / n-gram 总数（多样性，越低越同质化）。"""
    total, uniq = 0, set()
    for t in texts:
        ws = (t or "").split()
        for i in range(len(ws) - n + 1):
            uniq.add(tuple(ws[i:i + n]))
            total += 1
    return (len(uniq) / total) if total else 0.0


def analyze_tag(detail_path):
    samples = json.load(open(detail_path))
    outs = [s.get("output", "") for s in samples]
    n = len(outs)
    flags = [degeneration_signals(o) for o in outs]
    reasons = Counter(r for f, r in flags if f)
    return {
        "n": n,
        "degen_rate": 100.0 * sum(f for f, _ in flags) / max(n, 1),
        "reasons": dict(reasons),
        "mean_words": sum(len((o or "").split()) for o in outs) / max(n, 1),
        "distinct2": distinct_n(outs, 2),
    }


def model_dirs(arg):
    if arg:
        return [(m, f"{BASE}/{m}/outputs/eval") for m in arg.split(",")]
    # 默认：主设定 + RQ5 攻击目录（存在才收）
    out = [("llama2_7b_chat", f"{BASE}/llama2_7b_chat/outputs/eval")]
    for d in sorted(glob.glob(f"{BASE}/llama2_7b_chat_*/outputs/eval")):
        out.append((d.split("/")[-3], d))
    return [(m, d) for m, d in out if os.path.isdir(d)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=None, help="模型目录名（逗号分隔），默认主设定+RQ5")
    ap.add_argument("--side", choices=["clean", "trigger"], default="clean")
    ap.add_argument("--tags", type=str, default=None, help="只看这些 tag（逗号分隔），默认全部")
    args = ap.parse_args()
    want = set(args.tags.split(",")) if args.tags else None

    for model, evald in model_dirs(args.models):
        rows = []
        for p in sorted(glob.glob(os.path.join(evald, f"*_{args.side}_detail.json"))):
            tag = os.path.basename(p)[: -len(f"_{args.side}_detail.json")]
            if want and tag not in want:
                continue
            try:
                rows.append((tag, analyze_tag(p)))
            except (json.JSONDecodeError, OSError):
                continue
        if not rows:
            continue
        ref = dict(rows).get("no_defense")  # 参照系：θ_sus 无防御的输出质量
        print("=" * 100)
        print(f"  {model} · {args.side} 侧输出质量（degen%=退化样本率；与 no_defense 对照看防御是否伤效用）")
        print("=" * 100)
        print(f"  {'tag':42s} | {'degen%':>6s} | {'词数':>6s} | {'dist2':>5s} | 退化构成")
        print("  " + "-" * 96)
        for tag, m in sorted(rows, key=lambda kv: -kv[1]["degen_rate"]):
            mark = ""
            if ref and m["degen_rate"] >= max(2.0, 2 * ref["degen_rate"] + 1):
                mark = "  ⚠️"   # 退化率比无防御基线翻倍以上 → 防御本身伤了效用
            rs = ",".join(f"{k}:{v}" for k, v in sorted(m["reasons"].items())) or "-"
            print(f"  {tag:42s} | {m['degen_rate']:6.1f} | {m['mean_words']:6.1f} | {m['distinct2']:5.2f} | {rs}{mark}")
        print()


if __name__ == "__main__":
    main()
