#!/usr/bin/env python3
"""
Equivalence measurement for the score_channels vectorization.

Compares score_channels_scalar (old, ground truth) against the vectorized path on
BOTH cpu and cuda, on REAL cached deltas (qwen2_5_7b_instruct), sliced to 2 MLP
layers via mmap. Reports max score diff and the size of the selected-set symmetric
difference at the production ratio so we can choose the safe default backend.
"""
import os, sys, glob, pickle
import torch

sys.path.insert(0, "/home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine")
os.chdir("/home/zengqixiu/safety/backdoor/Vaccine/Backdoor-Vaccine")
from antigen.scoring import (
    score_channels_scalar, score_channels_vectorized, select_signature,
)

TAG = "qwen2_5_7b_instruct"
SIGDIR = f"/mnt/data/zengqixiu/bd-vax/Backdoor-Vaccine/{TAG}/outputs/signature"
DELTAS = sorted(glob.glob(os.path.join(SIGDIR, "deltas", "delta_variant_*.pth")))
LAYERS = (0, 1)
TARGET = ["gate_proj", "up_proj", "down_proj"]

def keep(k):
    return any(m in k for m in TARGET) and any(f".layers.{L}.mlp." in k for L in LAYERS)

print(f"Loading {len(DELTAS)} deltas (mmap, layers {LAYERS}) ...")
sliced = []
for p in DELTAS:
    try:
        d = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
    except Exception:
        d = torch.load(p, map_location="cpu")
    sliced.append({k: d[k].clone() for k in d if keep(k)})
    del d
print(f"  {len(sliced[0])} modules/variant")

print("scalar (truth) ...");  s_scalar = score_channels_scalar(sliced, TARGET, 0.01)
print("vec cpu ...");         s_cpu    = score_channels_vectorized(sliced, TARGET, 0.01, device="cpu")
print("vec cuda ...");        s_cuda   = score_channels_vectorized(sliced, TARGET, 0.01, device="cuda")

def report(name, sref, snew):
    assert set(sref) == set(snew), f"{name}: module-key mismatch"
    max_err = 0.0
    for mk in sref:
        a, b = dict(sref[mk]), dict(snew[mk])
        assert set(a) == set(b), f"{name}: channel-key set mismatch in {mk}"
        for ck in a:
            max_err = max(max_err, abs(a[ck] - b[ck]))
    print(f"\n=== {name} vs scalar ===")
    print(f"  max abs score diff: {max_err:.3e}")
    for ratio in (0.35, 0.05, 0.5):
        so, _ = select_signature(sref, ratio)
        sn, _ = select_signature(snew, ratio)
        diff = len(so ^ sn)               # symmetric difference
        tot = len(so)
        flag = "IDENTICAL" if diff == 0 else f"DIFFERS by {diff}/{tot} ({100*diff/tot:.4f}%)"
        print(f"  ratio={ratio:<4}: selected={tot:6d}  {flag}")

report("vec-cpu", s_scalar, s_cpu)
report("vec-cuda", s_scalar, s_cuda)

# also: cpu vs cuda agreement (how much does the GPU path itself drift)
report("vec-cuda", s_cpu, s_cuda)

# cross-check the committed production artifact reproduces under scalar
with open(os.path.join(SIGDIR, "signature.pkl"), "rb") as f:
    committed = pickle.load(f)
pm = committed["per_module_scores"]
perr = 0.0; checked = 0
for mk in s_scalar:
    if mk not in pm: continue
    a, b = dict(s_scalar[mk]), dict(pm[mk])
    if set(a) != set(b): print(f"  committed key mismatch {mk}"); continue
    for ck in a: perr = max(perr, abs(a[ck] - b[ck]))
    checked += 1
print(f"\n[scalar vs committed signature.pkl] {checked} modules, max abs diff = {perr:.3e}")
print("\nDONE")
