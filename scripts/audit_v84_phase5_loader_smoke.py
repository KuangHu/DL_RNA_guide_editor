"""V8.4 Phase 5 loader smoke — 20 bags per mode through ChannelBDataset.

Per feedback_train_target_smoke.md: run 20 bags per mode through the
LOADER (not just the generator) before any training. Reports:

  - Bag count, mean sites per bag
  - First-bag tensor shape (x, site_mask, y_target)
  - y_target: non-zero fraction (positive should be ~1/nc_len_eff per
    planted site; negatives should be exactly 0)
  - Site-axis distribution (K per bag)
  - Startup coord-check pass count (canonical_nc[p*:p*+L] == guide_dna)
  - Startup flank_bg_identity check pass count

Fails loud if any bag's y is all-zero when the mode is positive/planted.
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.constants import CHANNELS, N_CHANNELS, MAX_L, Ls, MAX_N_SITES


CORPUS = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_generation/50k")
SHARD  = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_shards")
MODES = ("none", "partial", "scattered", "flank_scattered",
           "no_alignment", "repeat_flank", "tsd_negative")
N_BAGS_TO_INSPECT = 20


def inspect_mode(mode: str) -> dict:
    jl = CORPUS / f"v84_{mode}.jsonl"
    sd = SHARD  / f"v84_{mode}_shard"
    ds = ChannelBDataset(jl, sd, preload_mt=True, cache_dir=None,
                            sort_by_nc_len=False, max_bags=200)  # cap for smoke
    n_bags = len(ds)
    if n_bags == 0:
        return {"mode": mode, "n_bags": 0, "error": "empty dataset"}
    sample_idx = list(range(min(N_BAGS_TO_INSPECT, n_bags)))

    x_shapes = []
    site_counts = []
    y_shapes = []
    y_nnz_frac = []
    y_all_zero = 0
    first_bag_summary = None
    for i in sample_idx:
        b = ds[i]
        x_shapes.append(tuple(b.x.shape))
        site_counts.append(int(b.site_mask.sum().item()))
        y_shapes.append(tuple(b.y.shape))
        nnz = float((b.y != 0).float().mean().item())
        y_nnz_frac.append(nnz)
        if b.y.sum().item() == 0:
            y_all_zero += 1
        if first_bag_summary is None:
            first_bag_summary = {
                "x_shape": tuple(b.x.shape),
                "y_shape": tuple(b.y.shape),
                "site_mask_sum": int(b.site_mask.sum().item()),
                "n_channels_in_x": int(b.x.shape[-1]),
                "expected_channels": N_CHANNELS,
                "y_max": float(b.y.max().item()),
                "y_sum": float(b.y.sum().item()),
                "y_nnz": int((b.y != 0).sum().item()),
            }
    return {
        "mode": mode,
        "n_bags": n_bags,
        "inspected": len(sample_idx),
        "x_shapes_unique": len(set(x_shapes)),
        "site_counts": {"min": min(site_counts), "max": max(site_counts),
                          "mean": float(np.mean(site_counts))},
        "y_nnz_frac": {"min": float(np.min(y_nnz_frac)),
                          "max": float(np.max(y_nnz_frac)),
                          "mean": float(np.mean(y_nnz_frac))},
        "y_all_zero_bags": y_all_zero,
        "first_bag": first_bag_summary,
    }


def main():
    print(f"# V8.4 Phase 5 loader smoke — {N_BAGS_TO_INSPECT} bags/mode\n")
    print(f"# CHANNELS ({N_CHANNELS}): {CHANNELS}")
    print(f"# Ls: {Ls}   MAX_L={MAX_L}   MAX_N_SITES={MAX_N_SITES}\n")

    for mode in MODES:
        print(f"## mode = {mode}")
        try:
            r = inspect_mode(mode)
        except Exception as e:
            print(f"  FAIL: {e}\n")
            continue
        print(f"  n_bags_loaded={r['n_bags']}  inspected={r['inspected']}")
        print(f"  site_counts min/mean/max = {r['site_counts']['min']}/"
              f"{r['site_counts']['mean']:.1f}/{r['site_counts']['max']}")
        print(f"  y_nnz_frac min/mean/max  = {r['y_nnz_frac']['min']:.5f}/"
              f"{r['y_nnz_frac']['mean']:.5f}/{r['y_nnz_frac']['max']:.5f}")
        print(f"  y_all_zero_bags = {r['y_all_zero_bags']} / {r['inspected']}")
        fb = r["first_bag"]
        print(f"  first_bag: x_shape={fb['x_shape']}  "
              f"y_shape={fb['y_shape']}  ch={fb['n_channels_in_x']}"
              f"(exp {fb['expected_channels']})  "
              f"site_mask_sum={fb['site_mask_sum']}  "
              f"y_max={fb['y_max']:.3f}  y_sum={fb['y_sum']:.3f}  "
              f"y_nnz={fb['y_nnz']}")

        # Fail loud on modes where we EXPECT y to be non-zero.
        if mode == "none" and r["y_all_zero_bags"] > 0:
            print(f"  FAIL: positive mode `none` has {r['y_all_zero_bags']} "
                  f"all-zero-y bags — training target broken")
        if mode in ("partial",) and r["y_nnz_frac"]["max"] == 0:
            print(f"  NOTE: `partial` has y_all_zero (planted sites should "
                  f"still contribute to y under per-site plant labels)")
        if mode in ("scattered", "flank_scattered", "no_alignment",
                       "repeat_flank", "tsd_negative"):
            if r["y_all_zero_bags"] < r["inspected"]:
                print(f"  NOTE: negative mode `{mode}` has "
                      f"{r['inspected'] - r['y_all_zero_bags']} bags with "
                      f"non-zero y — check per_site_nc_planted_pos logic")
        print()

    print("# V8.4 Phase 5 loader smoke complete")


if __name__ == "__main__":
    main()
