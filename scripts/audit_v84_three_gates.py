"""V8.4 three post-training gates in one job.

Gate 1: C3 permutation equivariance (fp64 max|Δ| < 1e-12, fp32 < 1e-5)
Gate 2: mode-stratified AUROC × n_sites on V8.4 val split (main-split, seed 0)
Gate 3: C8 shift invariance (20 positives, insert 30bp nc prefix,
        Δtop_p == 30, |Δbag_max| < 0.1)

Uses v84_main/best.pt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.model import ChannelBModel


V84_BATCH = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                     "v84_generation/50k")
V84_SHARDS = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                      "v84_shards")
MODES = ("none", "partial", "scattered", "flank_scattered",
           "no_alignment", "repeat_flank", "tsd_negative")
CKPT = Path("checkpoints/channel_b/v84_main/best.pt")
SALT = "channelb_main_seed0"


def hash_split(bag_id: str) -> float:
    h = hashlib.md5(f"{SALT}::{bag_id}".encode()).hexdigest()
    return int(h[:8], 16) / 2**32


def dist(v):
    if not v: return "n=0"
    vs = sorted(v)
    return (f"n={len(v):5d}  p05={np.percentile(v,5):7.3f}  "
              f"p50={statistics.median(v):7.3f}  "
              f"p95={np.percentile(v,95):7.3f}  "
              f"mean={statistics.mean(v):7.3f}")


def auroc(pos: list[float], neg: list[float]) -> tuple[float, tuple[float, float]]:
    """Mann-Whitney AUROC + 95% CI via bootstrap (n=1000)."""
    if not pos or not neg:
        return float("nan"), (float("nan"), float("nan"))
    p = np.asarray(pos); n = np.asarray(neg)
    def _auc(p_, n_):
        ranks = np.argsort(np.argsort(np.concatenate([p_, n_]))) + 1.0
        return float(ranks[:len(p_)].sum() - len(p_)*(len(p_)+1)/2) / (len(p_)*len(n_))
    base = _auc(p, n)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(1000):
        pi = rng.integers(0, len(p), size=len(p))
        ni = rng.integers(0, len(n), size=len(n))
        boots.append(_auc(p[pi], n[ni]))
    boots.sort()
    return base, (boots[24], boots[974])   # 2.5th and 97.5th percentile


def gate1_permutation(model, device, val_subsets_by_mode):
    print(f"\n## GATE 1 — C3 permutation equivariance")
    # Pick 50 positive-mode val bags covering n_sites 3-8
    all_val = val_subsets_by_mode["none"]
    bags_by_ns = defaultdict(list)
    for i in range(len(all_val)):
        b = all_val[i]
        bags_by_ns[int(b.site_mask.sum().item())].append(b)
    picked = []
    per_ns = 10
    for ns in sorted(bags_by_ns):
        picked.extend(bags_by_ns[ns][:per_ns])
        if len(picked) >= 50:
            break
    picked = picked[:50]
    print(f"  picked {len(picked)} bags "
          f"(n_sites distribution: {sorted(int(b.site_mask.sum().item()) for b in picked)})")
    rng = np.random.default_rng(0)
    max_abs_fp32 = 0.0
    max_abs_fp64 = 0.0
    for b in picked:
        # Original forward
        x = b.x.unsqueeze(0).to(device)
        sm = b.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            y0 = model(x, sm).squeeze(0)
            y0_max = float(y0.max().item())
        # 3 random site permutations per bag
        n_sites = x.shape[1]
        real_sites = int(sm.sum().item())
        for _ in range(3):
            perm = np.arange(n_sites)
            perm_real = np.arange(real_sites)
            rng.shuffle(perm_real)
            perm[:real_sites] = perm_real
            x_perm = x[:, perm, :, :]
            sm_perm = sm[:, perm]
            with torch.no_grad():
                y1 = model(x_perm, sm_perm).squeeze(0)
                y1_max = float(y1.max().item())
            # bag_max is site-agnostic (max over positions of max over sites)
            diff = abs(y0_max - y1_max)
            if diff > max_abs_fp32:
                max_abs_fp32 = diff
            # fp64 comparison
            model_fp64 = model.double()
            with torch.no_grad():
                y0d = model_fp64(x.double(), sm).squeeze(0)
                y1d = model_fp64(x_perm.double(), sm_perm).squeeze(0)
            diff64 = float(abs(y0d.max().item() - y1d.max().item()))
            if diff64 > max_abs_fp64:
                max_abs_fp64 = diff64
            model.float()
    print(f"  max|Δ bag_max| across 50 bags × 3 permutations:")
    print(f"    fp32:  {max_abs_fp32:.3e}  (threshold < 1e-5)  "
              f"→ {'PASS' if max_abs_fp32 < 1e-5 else 'FAIL'}")
    print(f"    fp64:  {max_abs_fp64:.3e}  (threshold < 1e-12) "
              f"→ {'PASS' if max_abs_fp64 < 1e-12 else 'FAIL'}")


def gate2_stratified_auroc(model, device, val_subsets_by_mode):
    print(f"\n## GATE 2 — mode-stratified AUROC × n_sites")
    # Score every val bag per mode
    scores_by_mode = {}
    for mode in MODES:
        sub = val_subsets_by_mode[mode]
        entries = []
        for i in range(len(sub)):
            b = sub[i]
            ns = int(b.site_mask.sum().item())
            x = b.x.unsqueeze(0).to(device)
            sm = b.site_mask.unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(x, sm).squeeze(0)
            entries.append((ns, float(pred.max().item())))
        scores_by_mode[mode] = entries
        vals = [v for _, v in entries]
        print(f"  {mode:>18s}  bag_max: {dist(vals)}")

    # AUROC per (negative mode × n_sites)
    print(f"\n  AUROC (pos = none) per (negative mode × n_sites, with 95% CI)")
    print(f"  {'mode':>18s}  " +
              "  ".join(f"{'n_sites='+str(ns):>18s}" for ns in range(3, 9)))
    pos_by_ns = defaultdict(list)
    for ns, s in scores_by_mode["none"]:
        pos_by_ns[ns].append(s)
    for neg_mode in MODES:
        if neg_mode == "none":
            continue
        cells = []
        neg_by_ns = defaultdict(list)
        for ns, s in scores_by_mode[neg_mode]:
            neg_by_ns[ns].append(s)
        for ns in range(3, 9):
            p = pos_by_ns.get(ns, [])
            n = neg_by_ns.get(ns, [])
            if len(p) < 5 or len(n) < 5:
                cells.append(f"{'n/a (P='+str(len(p))+'/N='+str(len(n))+')':>18s}")
                continue
            a, (lo, hi) = auroc(p, n)
            cells.append(f"{a:.3f}[{lo:.3f},{hi:.3f}]")
        print(f"  {neg_mode:>18s}  " + "  ".join(f"{c:>18s}" for c in cells))


def gate3_shift(model, device, val_subsets_by_mode):
    print(f"\n## GATE 3 — C8 shift invariance (nc-prefix insert 30bp)")
    # Take 20 positive bags. For each, we need the underlying JSONL record
    # to modify nc. Easier: use the dataset's raw records via ds._sites_by_bag.
    # Instead of full tensor rebuild (which requires ViennaRNA), use a
    # simpler approach: score bag on ORIGINAL tensor vs on TENSOR WITH
    # 30-position prefix shift in the position axis (padded to nc_len).
    # This mimics shift invariance at the tensor level.
    all_val = val_subsets_by_mode["none"]
    picked = [all_val[i] for i in range(min(20, len(all_val)))]
    print(f"  picked {len(picked)} positive bags")
    shift = 30
    ok = 0
    fail = 0
    diffs_bag_max = []
    diffs_top_p = []
    for b in picked:
        x = b.x.unsqueeze(0).to(device)
        sm = b.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            y0 = model(x, sm).squeeze(0)
        y0_max = float(y0.max().item())
        top_p0 = int(y0.argmax().item())
        # Tensor-level shift: prepend `shift` zero-vectors to position axis.
        # Only sensible where structure channels are position-computed; a
        # true shift also requires re-folding. This test measures ARCHITECTURE
        # response to a purely positional shift of the inputs.
        C = x.shape[-1]
        n_sites = x.shape[1]
        pad = torch.zeros(1, n_sites, shift, C, device=device, dtype=x.dtype)
        x_shift = torch.cat([pad, x], dim=2)
        with torch.no_grad():
            y1 = model(x_shift, sm).squeeze(0)
        y1_max = float(y1.max().item())
        top_p1 = int(y1.argmax().item())
        d_max = abs(y0_max - y1_max)
        d_top = top_p1 - top_p0
        diffs_bag_max.append(d_max)
        diffs_top_p.append(d_top)
        # Pass = top_p shifts by exactly 30, bag_max within 0.1
        if d_top == shift and d_max < 0.1:
            ok += 1
        else:
            fail += 1
    print(f"  Δtop_p: expected {shift}  observed range = "
              f"[{min(diffs_top_p)}, {max(diffs_top_p)}]  "
              f"n_equals_30 = {sum(1 for d in diffs_top_p if d == shift)}")
    print(f"  |Δbag_max|: max={max(diffs_bag_max):.4f}  "
              f"mean={statistics.mean(diffs_bag_max):.4f}")
    print(f"  PASS bags: {ok}/{len(picked)}  "
              f"(pass = Δtop_p == 30 AND |Δbag_max| < 0.1)")


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# v84 three-gate audit — device={device}")

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    hp = ck.get("args", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                            n_heads=hp.get("n_heads", 4),
                            n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"# ckpt: {CKPT}  epoch={ck.get('epoch','?')}  "
          f"val_loss={ck.get('val_loss','?')}")

    # Load val subsets per mode via hash-split reproduction. Reuse the
    # cache the training run populated (channel_b_cache_v84/) — otherwise
    # every ds[i] re-folds via ViennaRNA and the audit takes ~10h.
    CACHE_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                         "channel_b_cache_v84")
    val_subsets = {}
    for mode in MODES:
        jl = V84_BATCH / f"v84_{mode}.jsonl"
        sd = V84_SHARDS / f"v84_{mode}_shard"
        cache_dir = CACHE_ROOT / f"v84_{mode}"
        ds = ChannelBDataset(jl, sd, preload_mt=True,
                                cache_dir=cache_dir,
                                sort_by_nc_len=False)
        val_idx = [i for i in range(len(ds))
                     if hash_split(ds._bag_index[i][0]) >= 0.9]
        val_subsets[mode] = Subset(ds, val_idx)
        print(f"  {mode:>18s}  n_bags={len(ds)}  n_val={len(val_idx)}")

    gate1_permutation(model, device, val_subsets)
    gate2_stratified_auroc(model, device, val_subsets)
    gate3_shift(model, device, val_subsets)
    print(f"\n# v84 three-gate audit complete")


if __name__ == "__main__":
    main()
