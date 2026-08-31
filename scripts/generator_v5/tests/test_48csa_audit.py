"""48CS-A audit — is the P(Δ>0)=0.802 driven by length/geometry mismatch?

The 48CS-A finding is the only positive evidence for structure channels
in this project. Before spending 4 h rebuilding structure_v42_min at
global params, verify it holds under matched (L, guide_start, nc_len).

Checks:
  1. Guide-length distribution POS vs WS.
  2. nc-length distribution POS vs WS.
  3. Guide-start distribution POS vs WS.
  4. Neighbor-window sample-size distribution (positions actually
     available for the ±20 rule after boundary trimming).
  5. Stratified P(Δ>0) for guide_contrast_u1 within matched buckets
     of (L, nc_len_bucket, guide_start_position_bucket_relative).

If P(Δ>0) is >= 0.75 in every stratum, the length-confounder is falsified.
If it collapses to ~0.5 in some strata, the metric was picking up length,
not structure.

Uses same 25k pairs as the prior 48CS-A run (seed=42).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.observability_48csa import _load_index, _open_mmap, _row_for, _extract_features


POS_JSONL = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl"
POS_CACHE = "/global/scratch/users/kh36969/DL_novel_guide_editor/structure_v42_min/positive.index.json"
WS_JSONL  = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives_v42_counterfactual/negatives_v42_wrong_structure_role.jsonl"
WS_CACHE  = "/global/scratch/users/kh36969/DL_novel_guide_editor/structure_v42_min/ws.index.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-jsonl", default=POS_JSONL)
    ap.add_argument("--pos-cache-index", default=POS_CACHE)
    ap.add_argument("--ws-jsonl", default=WS_JSONL)
    ap.add_argument("--ws-cache-index", default=WS_CACHE)
    ap.add_argument("--n-pairs", type=int, default=25000)
    args = ap.parse_args()

    print("[load] pos cache index", flush=True)
    pos_idx = _load_index(args.pos_cache_index)
    pos_prof, _ = _open_mmap(pos_idx)
    print("[load] ws  cache index", flush=True)
    ws_idx = _load_index(args.ws_cache_index)
    ws_prof, _ = _open_mmap(ws_idx)

    print("[scan] POS records", flush=True)
    pos_recs = {}
    with open(args.pos_jsonl) as f:
        for line in f:
            r = json.loads(line)
            pos_recs[r["site_id"]] = r
    print("[scan] WS records", flush=True)
    ws_recs = {}
    with open(args.ws_jsonl) as f:
        for line in f:
            r = json.loads(line)
            ws_recs[r["site_id"]] = r

    pairs = []
    for sid, pr in pos_recs.items():
        wr = ws_recs.get(sid + "_wrongstr")
        if wr is not None:
            pairs.append((sid, pr, wr))
    print(f"[pair] {len(pairs)} POS/WS pairs", flush=True)

    rng = np.random.default_rng(42)
    if args.n_pairs and args.n_pairs < len(pairs):
        idx = rng.choice(len(pairs), size=args.n_pairs, replace=False)
        pairs = [pairs[int(i)] for i in idx]
    print(f"[sample] {len(pairs)} pairs", flush=True)

    # Extract features + metadata
    rows = []
    for sid, pr, wr in pairs:
        pa = pr["labels"].get("active_noncoding_index", 0) or 0
        wa = wr["labels"].get("active_noncoding_index", 0) or 0
        pgs = pr["labels"].get("guide_span_in_active_noncoding")
        wgs = wr["labels"].get("guide_span_in_active_noncoding")
        if pgs is None or wgs is None:
            continue
        pnc_len = len(pr["inputs"]["noncoding_regions"][pa])
        wnc_len = len(wr["inputs"]["noncoding_regions"][wa])
        p_row = _row_for(pos_idx, sid, pa)
        w_row = _row_for(ws_idx, sid + "_wrongstr", wa)
        if p_row < 0 or w_row < 0:
            continue
        u_max = pos_idx["_meta"]["u_max"]
        pf = _extract_features(pos_prof[p_row], pgs[0], pgs[1], pnc_len, u_max)
        wf = _extract_features(ws_prof[w_row], wgs[0], wgs[1], wnc_len, u_max)
        if pf is None or wf is None:
            continue
        rows.append({
            "sid": sid,
            "p_L": pgs[1] - pgs[0], "w_L": wgs[1] - wgs[0],
            "p_start": pgs[0], "w_start": wgs[0],
            "p_nc": pnc_len, "w_nc": wnc_len,
            "p_contrast": pf["guide_contrast_u1"],
            "w_contrast": wf["guide_contrast_u1"],
            "p_guide_mean": pf["guide_mean_unp_u1"],
            "w_guide_mean": wf["guide_mean_unp_u1"],
            "p_neighbor_mean": pf["neighbor_mean_unp_u1"],
            "w_neighbor_mean": wf["neighbor_mean_unp_u1"],
        })

    print(f"[extracted] {len(rows)} rows", flush=True)
    if not rows:
        return 1

    p_L = np.array([r["p_L"] for r in rows])
    w_L = np.array([r["w_L"] for r in rows])
    p_start = np.array([r["p_start"] for r in rows])
    w_start = np.array([r["w_start"] for r in rows])
    p_nc = np.array([r["p_nc"] for r in rows])
    w_nc = np.array([r["w_nc"] for r in rows])
    dl = np.array([r["p_contrast"] - r["w_contrast"] for r in rows])
    dg = np.array([r["p_guide_mean"] - r["w_guide_mean"] for r in rows])
    dn = np.array([r["p_neighbor_mean"] - r["w_neighbor_mean"] for r in rows])

    print()
    print("=== 1. Length matching ===")
    print(f"  p_L: min={p_L.min()}, max={p_L.max()}, median={int(np.median(p_L))}, "
          f"mean={p_L.mean():.2f}")
    print(f"  w_L: min={w_L.min()}, max={w_L.max()}, median={int(np.median(w_L))}, "
          f"mean={w_L.mean():.2f}")
    print(f"  L matched exactly (p_L == w_L): "
          f"{int((p_L == w_L).sum())} / {len(p_L)} = {(p_L == w_L).mean() * 100:.1f}%")
    print(f"  |p_L - w_L| distribution: max diff = {int(np.abs(p_L - w_L).max())}")

    print()
    print("=== 2. nc-length matching ===")
    print(f"  p_nc: median={int(np.median(p_nc))}, p25={int(np.percentile(p_nc, 25))}, "
          f"p75={int(np.percentile(p_nc, 75))}")
    print(f"  w_nc: median={int(np.median(w_nc))}, p25={int(np.percentile(w_nc, 25))}, "
          f"p75={int(np.percentile(w_nc, 75))}")
    print(f"  nc matched exactly: {int((p_nc == w_nc).sum())} / {len(p_nc)}")
    print(f"  |p_nc - w_nc| median = {int(np.median(np.abs(p_nc - w_nc)))}, "
          f"max = {int(np.abs(p_nc - w_nc).max())}")

    print()
    print("=== 3. guide_start matching ===")
    print(f"  p_start: median={int(np.median(p_start))}, p25={int(np.percentile(p_start, 25))}, "
          f"p75={int(np.percentile(p_start, 75))}")
    print(f"  w_start: median={int(np.median(w_start))}, p25={int(np.percentile(w_start, 25))}, "
          f"p75={int(np.percentile(w_start, 75))}")
    print(f"  start matched exactly: {int((p_start == w_start).sum())} / {len(p_start)}")

    # Neighbor window available size in POS/WS (positions from ±20 rule)
    p_neigh_avail = np.minimum(20, p_start) + np.minimum(20, p_nc - (p_start + p_L))
    w_neigh_avail = np.minimum(20, w_start) + np.minimum(20, w_nc - (w_start + w_L))
    print()
    print("=== 4. Neighbor-window sample size (positions after ±20 boundary trim) ===")
    print(f"  p_neigh_avail: median={int(np.median(p_neigh_avail))}, min={int(p_neigh_avail.min())}")
    print(f"  w_neigh_avail: median={int(np.median(w_neigh_avail))}, min={int(w_neigh_avail.min())}")
    print(f"  P(p_neigh_avail == w_neigh_avail) = {(p_neigh_avail == w_neigh_avail).mean() * 100:.1f}%")
    print(f"  P(both == 40 [max]) = {((p_neigh_avail == 40) & (w_neigh_avail == 40)).mean() * 100:.1f}%")

    print()
    print("=== 5. Stratified P(Δ_contrast > 0) ===")
    print("  overall P(Δ_contrast > 0) = "
          f"{(dl > 0).mean():.3f}  (should match prior 0.802)")
    print("  overall P(Δ_guide > 0) = "
          f"{(dg > 0).mean():.3f}  (was 0.637)")
    print("  overall P(Δ_neighbor > 0) = "
          f"{(dn > 0).mean():.3f}  (was 0.139)")

    # Stratify by matched L
    print()
    print("  by matched guide length (p_L == w_L):")
    matched = (p_L == w_L)
    print(f"    matched-L subset ({matched.sum()}): "
          f"P(Δ_contrast>0) = {(dl[matched] > 0).mean():.3f}, "
          f"P(Δ_guide>0) = {(dg[matched] > 0).mean():.3f}, "
          f"P(Δ_neighbor>0) = {(dn[matched] > 0).mean():.3f}")
    if (~matched).any():
        print(f"    unmatched-L subset ({(~matched).sum()}): "
              f"P(Δ_contrast>0) = {(dl[~matched] > 0).mean():.3f}")

    # Stratify by matched nc
    print()
    print("  by matched nc length (p_nc == w_nc):")
    nc_matched = (p_nc == w_nc)
    print(f"    matched-nc subset ({nc_matched.sum()}): "
          f"P(Δ_contrast>0) = {(dl[nc_matched] > 0).mean():.3f}, "
          f"P(Δ_guide>0) = {(dg[nc_matched] > 0).mean():.3f}, "
          f"P(Δ_neighbor>0) = {(dn[nc_matched] > 0).mean():.3f}")

    # Stratify by neighbor sample size (both 40 vs one truncated)
    print()
    print("  by neighbor sample size (both == 40 [full 40 nt] vs truncated):")
    both_full = (p_neigh_avail == 40) & (w_neigh_avail == 40)
    print(f"    both-full subset ({both_full.sum()}): "
          f"P(Δ_contrast>0) = {(dl[both_full] > 0).mean():.3f}")
    truncated = ~both_full
    if truncated.any():
        print(f"    truncated subset ({truncated.sum()}): "
              f"P(Δ_contrast>0) = {(dl[truncated] > 0).mean():.3f}")

    # Stratify by guide length bucket
    print()
    print("  by guide length bucket (matched only):")
    for L_val in sorted(set(p_L[matched].tolist())):
        mask = matched & (p_L == L_val)
        if mask.sum() < 100:
            continue
        print(f"    L={L_val} (n={mask.sum():>5}): "
              f"P(Δ_contrast>0) = {(dl[mask] > 0).mean():.3f}, "
              f"P(Δ_guide>0) = {(dg[mask] > 0).mean():.3f}, "
              f"P(Δ_neighbor>0) = {(dn[mask] > 0).mean():.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
