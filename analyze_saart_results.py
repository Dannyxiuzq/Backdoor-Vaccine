#!/usr/bin/env python3
"""Analyze the SAART comparison battery from outputs/eval/results.jsonl.

Turns the raw ledger into the structured comparisons needed to judge the method's
soundness: headline ASR reduction, vs compute-matched finetune, vs consistency-only,
component ablations, dose-response curves (lambda2 / inner_steps), and seed robustness.

Usage:
    python analyze_saart_results.py [--config configs/experiment.yaml]
Reads only the LATEST row per tag (results.jsonl is append-only).
"""
import argparse
import json
import os
import statistics

import yaml


def load_latest(ledger):
    rows = {}
    with open(ledger) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            t = r.get("tag")
            if t and "trigger_asr" in r:
                rows[t] = r  # later lines overwrite -> latest wins
    return rows


def g(rows, tag, key):
    r = rows.get(tag)
    return None if r is None else r.get(key)


def fmt(v, suffix="%"):
    return "  n/a " if v is None else f"{v:6.1f}{suffix}"


def line(tag, rows, label=None):
    r = rows.get(tag)
    if r is None:
        return f"  {(label or tag):<34s} {'(not run yet)':>10s}"
    return (f"  {(label or tag):<34s}  ASR={fmt(r.get('trigger_asr'))}"
            f"   cleanFP={fmt(r.get('clean_fp'))}"
            f"   ({r.get('trigger_hits','?')}/{r.get('trigger_total','?')} | "
            f"{r.get('clean_hits','?')}/{r.get('clean_total','?')})")


def delta(rows, a, b, key="trigger_asr"):
    """a - b (e.g. baseline - method = reduction)."""
    va, vb = g(rows, a, key), g(rows, b, key)
    if va is None or vb is None:
        return None
    return va - vb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiment.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    ledger = os.path.join(cfg["eval_dir"], "results.jsonl")
    if not os.path.exists(ledger):
        print("No results.jsonl yet at", ledger)
        return
    R = load_latest(ledger)
    base = "no_defense"
    base_asr = g(R, base, "trigger_asr")

    P = print
    P("=" * 78)
    P("  SAART-P1 METHOD-SOUNDNESS ANALYSIS  (model: %s)" % cfg.get("model_tag", "?"))
    P("  source: %s  (latest row per tag)" % ledger)
    P("=" * 78)

    P("\n[1] HEADLINE — does SAART-P1 suppress the (unseen) real backdoor trigger?")
    P(line(base, R, "no_defense (θ_sus, attacked)"))
    P(line("after_saart_p1", R, "after_saart_p1 (SAART-P1, ours)"))
    d = delta(R, base, "after_saart_p1")
    if d is not None:
        P(f"    >> ASR reduction vs no_defense: {base_asr:.1f}%% -> {g(R,'after_saart_p1','trigger_asr'):.1f}%%  (Δ = -{d:.1f} pp)")

    P("\n[2] vs FINETUNE BASELINES (is it just clean finetuning / just more compute?)")
    for t, lab in [("after_pure_finetune", "pure FT (B2, equal epochs)"),
                   ("after_pure_finetune_long", "pure FT (compute-matched 1x)"),
                   ("after_pure_finetune_long2", "pure FT (compute-matched 2x)")]:
        P(line(t, R, lab))
        dd = delta(R, t, "after_saart_p1")
        if dd is not None:
            verdict = "SAART better" if dd > 0 else "SAART NOT better"
            P(f"    >> SAART vs {lab}: ΔASR = {dd:+.1f} pp ({verdict})")

    P("\n[3] vs CONSISTENCY-ONLY (CROW-style; isolates the self-adversarial trigger search)")
    P(line("after_consistency", R, "consistency-only (lr-aligned)"))
    dd = delta(R, "after_consistency", "after_saart_p1")
    if dd is not None:
        P(f"    >> SAART vs consistency-only: ΔASR = {dd:+.1f} pp")

    P("\n[4] vs BD-VAX (post-hoc) & other defenses (full ranking)")
    for t, lab in [("after_suppression", "BD-VAX suppress"),
                   ("after_finetune", "BD-VAX suppress+FT"),
                   ("after_random_suppression", "random-prune ctrl"),
                   ("after_wanda_pruning", "Wanda prune"),
                   ("after_fine_pruning", "Fine-pruning")]:
        P(line(t, R, lab))

    P("\n[5] COMPONENT ABLATIONS (which mechanism drives the effect?)")
    P(line("after_saart_p1", R, "full (main: +pool +KL +adv)"))
    P(line("after_saart_p1_soft_only", R, "soft_only (−pool)"))
    P(line("after_saart_p1_kl_only", R, "kl_only (−adv-correct)"))
    P(line("after_saart_p1_adv_only", R, "adv_only (−KL)"))
    P(line("after_saart_p1_insert_after_bos", R, "insert=after_bos (vs prompt_end)"))
    P(line("after_saart_p1_no_null", R, "no_null (vs null-ref)"))
    for a, b, name in [("after_saart_p1_soft_only", "after_saart_p1", "pool adds"),
                       ("after_saart_p1_kl_only", "after_saart_p1_soft_only", "adv-correct adds"),
                       ("after_saart_p1_adv_only", "after_saart_p1_soft_only", "KL adds"),
                       ("after_saart_p1_insert_after_bos", "after_saart_p1", "prompt_end vs after_bos"),
                       ("after_saart_p1_no_null", "after_saart_p1", "null-ref vs x")]:
        dd = delta(R, a, b)  # a_asr - b_asr ; positive => b (with the component) has lower ASR => component helps
        if dd is not None:
            P(f"    >> {name}: ΔASR = {dd:+.1f} pp (positive ⇒ the added component lowers ASR)")

    P("\n[6] DOSE-RESPONSE — output-consistency weight λ2 (expect monotonic-ish trend)")
    for t, lab in [("after_saart_p1_lam2_0p25", "λ2=0.25"),
                   ("after_saart_p1", "λ2=0.5 (main)"),
                   ("after_saart_p1_lam2_1p0", "λ2=1.0"),
                   ("after_saart_p1_lam2_2p0", "λ2=2.0")]:
        P(line(t, R, lab))

    P("\n[7] DOSE-RESPONSE — inner adversary strength (inner_steps; 0 = random-trigger control)")
    for t, lab in [("after_saart_p1_inner0", "inner=0 (random trigger, no adversary)"),
                   ("after_saart_p1", "inner=3 (main)"),
                   ("after_saart_p1_inner5", "inner=5 (stronger)")]:
        P(line(t, R, lab))
    a0 = g(R, "after_saart_p1_inner0", "trigger_asr")
    am = g(R, "after_saart_p1", "trigger_asr")
    if a0 is not None and am is not None:
        P(f"    >> adversary necessity (inner0 − main): ΔASR = {a0 - am:+.1f} pp (positive ⇒ the adversarial search helps)")

    P("\n[8] SEED ROBUSTNESS (ASR stability across seeds)")
    seed_tags = [("after_saart_p1", "seed42(main)"), ("after_saart_p1_seed123", "seed123"),
                 ("after_saart_p1_seed2024", "seed2024")]
    for t, lab in seed_tags:
        P(line(t, R, lab))
    asrs = [g(R, t, "trigger_asr") for t, _ in seed_tags if g(R, t, "trigger_asr") is not None]
    if len(asrs) >= 2:
        m = statistics.mean(asrs)
        sd = statistics.pstdev(asrs) if len(asrs) > 1 else 0.0
        P(f"    >> ASR across {len(asrs)} seeds: mean={m:.1f}%%  std={sd:.2f} pp  (range {min(asrs):.1f}–{max(asrs):.1f})")

    P("\n[9] AFTER_BOS REGIME (wave-3: combine the winning axes — after_bos + low/zero KL)")
    P(line("after_saart_p1_insert_after_bos", R, "after_bos, λ2=0.5, +pool (w1)"))
    P(line("after_saart_p1_bos_lam2_0p25", R, "after_bos, λ2=0.25, +pool"))
    P(line("after_saart_p1_bos_best", R, "after_bos, λ2=0,  +pool"))
    P(line("after_saart_p1_bos_advonly", R, "after_bos, λ2=0,  −pool"))
    P(line("after_saart_p1_bos_inner5", R, "after_bos, λ2=0.5, inner=5"))

    P("\n[10] PHASE-2 — 在线 MLP 关联签名 + L_assoc-reg (after_saart_p2 vs 最佳 P1 after_bos)")
    P(line("after_saart_p1_insert_after_bos", R, "P1 best (after_bos, λ2=0.5)"))
    P(line("after_saart_p2", R, "P2 (+ L_assoc-reg, after_bos)"))
    dd = delta(R, "after_saart_p1_insert_after_bos", "after_saart_p2")
    if dd is not None:
        verdict = "assoc-reg 进一步降低 ASR" if dd > 0 else "assoc-reg 未带来额外下降"
        P(f"    >> P2 vs P1-best: ΔASR = {dd:+.1f} pp ({verdict})")
    dd2 = delta(R, base, "after_saart_p2")
    if dd2 is not None:
        P(f"    >> P2 vs no_defense: {base_asr:.1f}%% -> {g(R,'after_saart_p2','trigger_asr'):.1f}%% (Δ = -{dd2:.1f} pp)")

    P("\n[11] PHASE-2 sweep — assoc-reg 剂量/结构消融（均在 after_bos + use_assoc_reg 上）")
    for t, lab in [("after_saart_p2", "P2 base (λ3=1.0, top0.35, last8)"),
                   ("after_saart_p2_lam3_0p3", "λ3=0.3"),
                   ("after_saart_p2_lam3_3p0", "λ3=3.0"),
                   ("after_saart_p2_top0p15", "top_ratio=0.15"),
                   ("after_saart_p2_top0p5", "top_ratio=0.5"),
                   ("after_saart_p2_layers_last4", "layers=last4"),
                   ("after_saart_p2_layers_all", "layers=all"),
                   ("after_saart_p2_assoc_only", "assoc-only (λ1=λ2=0): 隔离 Module 2")]:
        P(line(t, R, lab))

    # best SAART config auto-pick（含 P1 与 P2 全部条件——用 after_saart_p 前缀，别漏掉 p2）
    saart_tags = {t: r for t, r in R.items() if t.startswith("after_saart_p")}
    if saart_tags:
        best = min(saart_tags.items(), key=lambda kv: kv[1].get("trigger_asr", 1e9))
        bt, br = best
        P("\n[*] BEST SAART CONFIG SO FAR: %s  ASR=%.1f%%  cleanFP=%.1f%%" % (
            bt, br.get("trigger_asr"), br.get("clean_fp")))
        if base_asr is not None:
            P("    vs no_defense %.1f%% -> %.1f%% (Δ=-%.1f pp); vs pure-FT(B2) %s; vs BD-VAX suppress+FT %s" % (
                base_asr, br.get("trigger_asr"), base_asr - br.get("trigger_asr"),
                fmt(g(R, "after_pure_finetune", "trigger_asr")), fmt(g(R, "after_finetune", "trigger_asr"))))

    # progress
    expected = ["no_defense", "after_saart_p1", "after_saart_p1_soft_only", "after_saart_p1_kl_only",
                "after_saart_p1_adv_only", "after_saart_p1_insert_after_bos", "after_saart_p1_no_null",
                "after_pure_finetune_long", "after_pure_finetune_long2", "after_consistency",
                "after_saart_p1_lam2_0p25", "after_saart_p1_lam2_1p0", "after_saart_p1_lam2_2p0",
                "after_saart_p1_inner0", "after_saart_p1_inner5", "after_saart_p1_seed123",
                "after_saart_p1_seed2024", "after_saart_p1_bos_best", "after_saart_p1_bos_advonly",
                "after_saart_p1_bos_lam2_0p25", "after_saart_p1_bos_inner5", "after_saart_p2",
                "after_saart_p2_lam3_0p3", "after_saart_p2_lam3_3p0", "after_saart_p2_top0p15",
                "after_saart_p2_top0p5", "after_saart_p2_layers_last4", "after_saart_p2_layers_all",
                "after_saart_p2_assoc_only"]
    have = [t for t in expected if t in R]
    P("\n" + "=" * 78)
    P(f"  progress: {len(have)}/{len(expected)} new comparison conditions have results")
    missing = [t for t in expected if t not in R]
    if missing:
        P("  still pending: " + ", ".join(missing))
    P("=" * 78)


if __name__ == "__main__":
    main()
