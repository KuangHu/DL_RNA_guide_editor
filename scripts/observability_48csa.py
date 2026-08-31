"""48CS-A raw structural observability diagnostic.

For each POS_i and paired wrong_structure_role_i, extract RNAplfold-derived
guide-region accessibility features and compute paired Δ.

Only uses:
  - RNAplfold unpaired-probability profiles (from real structure cache)
  - `labels.guide_span_in_active_noncoding` for the guide region span
  - `labels.active_noncoding_index` for the NC slot

Does NOT use:
  - designed_structure
  - guide_unpaired_in_fold
  - any generator metadata for structural characteristics

Features (per record):
  1. guide_mean_unp_u1   : mean unpaired prob (u=1) over guide bases
  2. guide_mean_unp_u5   : mean unpaired prob (u=5)
  3. guide_min_unp_u1    : min unpaired prob (u=1) over guide bases
  4. neighbor_mean_unp_u1: mean unpaired prob in ±20 nt flanking the guide
  5. guide_contrast_u1   : guide_mean_unp_u1 - neighbor_mean_unp_u1

Paired Δ = feature(POS_i) - feature(WS_i), matched by parent site_id.

Success criterion (per user): P(Δ > 0) ≥ ~0.65 for the guide-accessibility
features indicates the RNAplfold signal is truly observable.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def _load_index(path: str) -> dict:
    d = json.load(open(path))
    return d


def _open_mmap(idx: dict) -> tuple[np.memmap, np.memmap]:
    meta = idx["_meta"]
    N = meta["N"]; nc_max = meta["nc_max"]; u_max = meta["u_max"]
    prof = np.memmap(meta["mmap_path"], dtype=np.float16, mode="r",
                       shape=(N, nc_max, u_max))
    valid = np.memmap(meta["valid_path"], dtype=np.uint8, mode="r",
                        shape=(N, nc_max, u_max))
    return prof, valid


def _row_for(idx: dict, site_id: str, slot: int) -> int:
    entry = idx.get(site_id)
    if entry is None:
        return -1
    return int(entry["slots"].get(str(slot), -1))


def _extract_features(profile_row: np.ndarray, guide_start: int, guide_end: int,
                       nc_len: int, u_max: int) -> dict | None:
    """Compute features from one NC's unpaired-probability profile."""
    if guide_start < 0 or guide_end > nc_len or guide_start >= guide_end:
        return None
    p = np.asarray(profile_row, dtype=np.float32)   # (nc_max, u_max) — trimmed below
    p = p[:nc_len]                                    # (nc_len, u_max)
    if p.size == 0:
        return None
    guide = p[guide_start:guide_end]
    if guide.size == 0:
        return None
    u1 = 0
    u5 = min(4, u_max - 1)
    guide_mean_u1 = float(guide[:, u1].mean())
    guide_mean_u5 = float(guide[:, u5].mean())
    guide_min_u1  = float(guide[:, u1].min())
    # Neighbor: 20 nt on each side, excluding the guide itself
    lo = max(0, guide_start - 20)
    hi = min(nc_len, guide_end + 20)
    neighbor_mask = np.ones(hi - lo, dtype=bool)
    neighbor_mask[guide_start - lo : guide_end - lo] = False
    neighbor = p[lo:hi][neighbor_mask]
    if neighbor.size == 0:
        neighbor_mean_u1 = float("nan")
    else:
        neighbor_mean_u1 = float(neighbor[:, u1].mean())
    contrast_u1 = guide_mean_u1 - neighbor_mean_u1 if neighbor_mean_u1 == neighbor_mean_u1 else float("nan")
    return {
        "guide_mean_unp_u1":    guide_mean_u1,
        "guide_mean_unp_u5":    guide_mean_u5,
        "guide_min_unp_u1":     guide_min_u1,
        "neighbor_mean_unp_u1": neighbor_mean_u1,
        "guide_contrast_u1":    contrast_u1,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-jsonl", required=True)
    ap.add_argument("--pos-cache-index", required=True)
    ap.add_argument("--ws-jsonl", required=True)
    ap.add_argument("--ws-cache-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-pairs", type=int, default=0,
                     help="If >0, randomly subsample this many POS/WS pairs (seed 42) — for a fast diagnostic.")
    args = ap.parse_args()

    print("[load] pos cache index", flush=True)
    pos_idx = _load_index(args.pos_cache_index)
    pos_prof, pos_val = _open_mmap(pos_idx)
    print("[load] ws  cache index", flush=True)
    ws_idx  = _load_index(args.ws_cache_index)
    ws_prof,  ws_val  = _open_mmap(ws_idx)

    # Index POS records by site_id
    print("[scan] POS records", flush=True)
    pos_recs = {}
    with open(args.pos_jsonl) as f:
        for line in f:
            r = json.loads(line)
            pos_recs[r["site_id"]] = r
    print(f"  POS records: {len(pos_recs)}", flush=True)

    print("[scan] WS records", flush=True)
    ws_recs = {}
    with open(args.ws_jsonl) as f:
        for line in f:
            r = json.loads(line)
            ws_recs[r["site_id"]] = r
    print(f"  WS records:  {len(ws_recs)}", flush=True)

    # Pair by parent site_id (WS suffix _wrongstr)
    pairs = []
    for pos_sid, pos_r in pos_recs.items():
        ws_sid = pos_sid + "_wrongstr"
        ws_r = ws_recs.get(ws_sid)
        if ws_r is None:
            continue
        pairs.append((pos_sid, pos_r, ws_sid, ws_r))
    print(f"[pair] {len(pairs)} POS/WS pairs matched by site_id", flush=True)

    if args.n_pairs and args.n_pairs < len(pairs):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pairs), size=args.n_pairs, replace=False)
        pairs = [pairs[int(i)] for i in idx]
        print(f"[subsample] using {len(pairs)} pairs (seed=42)", flush=True)

    # Compute features
    feats_pos = []
    feats_ws  = []
    n_missing = 0
    for pos_sid, pos_r, ws_sid, ws_r in pairs:
        pa = pos_r["labels"].get("active_noncoding_index", 0) or 0
        pgs = pos_r["labels"].get("guide_span_in_active_noncoding")
        wa = ws_r["labels"].get("active_noncoding_index", 0) or 0
        wgs = ws_r["labels"].get("guide_span_in_active_noncoding")
        if pgs is None or wgs is None:
            n_missing += 1; continue
        pnc = pos_r["inputs"]["noncoding_regions"][pa]
        wnc = ws_r["inputs"]["noncoding_regions"][wa]
        p_row = _row_for(pos_idx, pos_sid, pa)
        w_row = _row_for(ws_idx,  ws_sid,  wa)
        if p_row < 0 or w_row < 0:
            n_missing += 1; continue
        u_max = pos_idx["_meta"]["u_max"]
        p_feats = _extract_features(pos_prof[p_row], pgs[0], pgs[1], len(pnc), u_max)
        w_feats = _extract_features(ws_prof[w_row],  wgs[0], wgs[1], len(wnc), u_max)
        if p_feats is None or w_feats is None:
            n_missing += 1; continue
        feats_pos.append(p_feats); feats_ws.append(w_feats)

    n = len(feats_pos)
    print(f"[computed] {n} valid pairs   (skipped {n_missing})", flush=True)

    feat_names = list(feats_pos[0].keys()) if feats_pos else []
    reports = {}
    for name in feat_names:
        pos_vals = np.asarray([f[name] for f in feats_pos], dtype=np.float32)
        ws_vals  = np.asarray([f[name] for f in feats_ws],  dtype=np.float32)
        mask = np.isfinite(pos_vals) & np.isfinite(ws_vals)
        pos_vals = pos_vals[mask]; ws_vals = ws_vals[mask]
        if len(pos_vals) == 0:
            continue
        deltas = pos_vals - ws_vals
        # Unpaired AUROC (POS vs WS)
        from sklearn.metrics import roc_auc_score
        combined = np.concatenate([pos_vals, ws_vals])
        labels = np.concatenate([np.ones(len(pos_vals), int),
                                   np.zeros(len(ws_vals), int)])
        au = float(roc_auc_score(labels, combined)) if len(set(labels.tolist())) == 2 else float("nan")
        stats = {
            "n":              int(len(deltas)),
            "auroc":          au,
            "pos_median":     float(np.median(pos_vals)),
            "ws_median":      float(np.median(ws_vals)),
            "paired_median":  float(np.median(deltas)),
            "paired_MAD":     float(np.median(np.abs(deltas - np.median(deltas)))),
            "paired_p_gt_0":  float((deltas > 0).mean()),
            "paired_q10":     float(np.quantile(deltas, 0.10)),
            "paired_q90":     float(np.quantile(deltas, 0.90)),
        }
        reports[name] = stats
        print(f"  {name:<26} AUROC={au:.4f}  paired med={stats['paired_median']:+.3f}  MAD={stats['paired_MAD']:.3f}  P(Δ>0)={stats['paired_p_gt_0']:.3f}",
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_pairs":           len(pairs),
            "n_computed":        n,
            "n_missing_or_bad":  n_missing,
            "features":          reports,
        }, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
