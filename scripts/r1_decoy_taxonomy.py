"""Durrant top-decoy taxonomy audit.

For each Durrant cognate record with a gold annotation, build the current
candidate pool, locate the gold slot (tolerant match, same as r1_gold_recall),
then examine the top-K decoys by `matches`. Classify each decoy by its
geometric relationship to gold and report:

  - overall bucket distribution over all (record, top-K) decoys
  - per-record top-1 decoy bucket
  - per-bucket median matches, median matches gap vs gold, median rank
  - per-bucket median Δnc_start, Δflank_start, ΔL

Taxonomy (deterministic, checked top-to-bottom):

  1. wrong_orientation              — c.orient != gold.orient
  2. different_region               — nc-span overlap < 50% of min(L, gold_L)
  3. same_region_longer_L           — nc-overlap ≥ 50%, ΔL > 0
  4. same_region_shorter_L          — nc-overlap ≥ 50%, ΔL < 0
  5. same_region_same_L_wrong_flank — nc-overlap ≥ 50%, ΔL = 0, flank-overlap
                                      < 50% (guide slot right, target wrong)
  6. near_gold                      — nc-overlap ≥ 50%, ΔL = 0, flank-overlap
                                      ≥ 50% (small shift, mismatch grammar)

No model training / no scoring. Pure geometry over the candidate proposer's
output.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX


def _load_gold(gold_path: str) -> dict:
    out = {}
    with open(gold_path) as f:
        for line in f:
            r = json.loads(line)
            out[r["site_id"]] = r
    return out


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _classify(c, gold_orient: str, gold_L: int,
                gold_nc_start: int, gold_flank_start: int,
                overlap_frac: float = 0.5) -> str:
    """Return the taxonomy bucket for candidate `c` vs the gold coords."""
    if c.orient != gold_orient:
        return "wrong_orientation"
    min_L = min(c.L, gold_L)
    nc_ov    = _overlap(c.nc_start,    c.nc_start + c.L,
                          gold_nc_start,  gold_nc_start + gold_L)
    flank_ov = _overlap(c.flank_start, c.flank_start + c.L,
                          gold_flank_start, gold_flank_start + gold_L)
    thresh = overlap_frac * min_L
    if nc_ov < thresh:
        return "different_region"
    dL = c.L - gold_L
    if dL > 0:
        return "same_region_longer_L"
    if dL < 0:
        return "same_region_shorter_L"
    if flank_ov < thresh:
        return "same_region_same_L_wrong_flank"
    return "near_gold"


def _distance_features(c, gold_orient: str, gold_L: int,
                        gold_nc_start: int, gold_flank_start: int) -> dict:
    nc_ov = _overlap(c.nc_start, c.nc_start + c.L,
                      gold_nc_start, gold_nc_start + gold_L)
    flank_ov = _overlap(c.flank_start, c.flank_start + c.L,
                      gold_flank_start, gold_flank_start + gold_L)
    min_L = min(c.L, gold_L)
    return {
        "delta_nc_start":    c.nc_start    - gold_nc_start,
        "delta_flank_start": c.flank_start - gold_flank_start,
        "delta_L":           c.L           - gold_L,
        "nc_overlap":        nc_ov,
        "flank_overlap":     flank_ov,
        "nc_overlap_frac":   float(nc_ov)    / max(1, min_L),
        "flank_overlap_frac": float(flank_ov) / max(1, min_L),
        "orient_match":      int(c.orient == gold_orient),
    }


def _find_gold_slot(feats, mask, cands,
                     gold_orient: str, gold_L: int,
                     gold_nc_start: int, gold_flank_start: int,
                     overlap_frac: float = 0.5) -> tuple[int, float]:
    valid_slots = np.where(mask)[0]
    if len(valid_slots) == 0:
        return -1, 0.0
    matches = feats[:, 3]
    gold_slot = -1
    best_gold_matches = -1.0
    for i in valid_slots:
        c = cands[i]
        if c.orient != gold_orient:
            continue
        min_L = min(c.L, gold_L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L,
                          gold_nc_start, gold_nc_start + gold_L)
        flank_ov = _overlap(c.flank_start, c.flank_start + c.L,
                              gold_flank_start, gold_flank_start + gold_L)
        thresh = overlap_frac * min_L
        if nc_ov < thresh or flank_ov < thresh:
            continue
        if matches[i] > best_gold_matches:
            best_gold_matches = float(matches[i])
            gold_slot = int(i)
    return gold_slot, best_gold_matches


def audit_bag(nc: str, flank: str, gold: dict, k_top: int = 8) -> list[dict]:
    prof = np.zeros((len(nc), 16), dtype=np.float32)
    val = np.zeros((len(nc), 16), dtype=bool)
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)

    g_orient = gold["target_flank_orientation"]
    g_L = gold["target_binding_loop_length"]
    g_nc_start = gold["guide_start_in_nc"]
    g_flank_start = gold["target_flank_start"]

    gold_slot, gold_matches = _find_gold_slot(
        feats, mask, cands, g_orient, g_L, g_nc_start, g_flank_start)

    valid_slots = np.where(mask)[0]
    if len(valid_slots) == 0:
        return []

    matches = feats[:, 3]
    # Rank all valid slots by matches, desc. Ties → keep proposer order.
    order = valid_slots[np.argsort(-matches[valid_slots], kind="stable")]

    # Drop the gold slot from the ranked list to leave only decoys.
    if gold_slot >= 0:
        order = order[order != gold_slot]

    top_decoys = order[:k_top]
    rows = []
    for rank, slot in enumerate(top_decoys, start=1):
        c = cands[int(slot)]
        bucket = _classify(c, g_orient, g_L, g_nc_start, g_flank_start)
        feat = _distance_features(c, g_orient, g_L, g_nc_start, g_flank_start)
        rows.append({
            "decoy_rank_among_decoys": int(rank),
            "slot":                   int(slot),
            "bucket":                 bucket,
            "matches":                float(matches[int(slot)]),
            "matches_gap_vs_gold":    float(matches[int(slot)] - gold_matches) if gold_slot >= 0 else float("nan"),
            "gold_slot":              int(gold_slot),
            "gold_matches":           float(gold_matches) if gold_slot >= 0 else float("nan"),
            "gold_in_pool":           bool(gold_slot >= 0),
            **feat,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cognate-jsonl", required=True)
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--k-top", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = _load_gold(args.gold_jsonl)
    print(f"[gold] {len(gold)} annotated site_ids", flush=True)

    all_rows = []
    per_bag_top1 = []
    n_records = n_annotated = n_gold_in_pool = 0
    with open(args.cognate_jsonl) as f:
        for line in f:
            r = json.loads(line)
            n_records += 1
            sid = r["site_id"]
            g = gold.get(sid)
            if g is None:
                continue
            n_annotated += 1
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs):
                active_nc = 0
            nc = ncs[active_nc]
            flank = r["inputs"]["flank"]
            rows = audit_bag(nc, flank, g, k_top=args.k_top)
            if not rows:
                continue
            if rows[0]["gold_in_pool"]:
                n_gold_in_pool += 1
            for row in rows:
                row["site_id"] = sid
            all_rows.extend(rows)
            per_bag_top1.append({
                "site_id":       sid,
                "bucket":        rows[0]["bucket"],
                "matches":       rows[0]["matches"],
                "matches_gap":   rows[0]["matches_gap_vs_gold"],
                "gold_in_pool":  rows[0]["gold_in_pool"],
            })

    print(f"[scan] records={n_records}  annotated={n_annotated}  gold_in_pool={n_gold_in_pool}", flush=True)
    print(f"[rows] total decoy rows: {len(all_rows)}", flush=True)

    buckets = ["wrong_orientation", "different_region",
                 "same_region_longer_L", "same_region_shorter_L",
                 "same_region_same_L_wrong_flank", "near_gold"]

    # 1. Global bucket distribution over all decoy rows.
    all_labels = [row["bucket"] for row in all_rows]
    total = len(all_labels) or 1
    global_dist = {b: all_labels.count(b) / total for b in buckets}

    # 2. Top-1 decoy bucket distribution (per-record).
    top1_labels = [r["bucket"] for r in per_bag_top1]
    top1_total = len(top1_labels) or 1
    top1_dist = {b: top1_labels.count(b) / top1_total for b in buckets}

    # 3. Per-bucket stats over ALL rows.
    per_bucket = {}
    for b in buckets:
        rows = [r for r in all_rows if r["bucket"] == b]
        if not rows:
            per_bucket[b] = {"n": 0}
            continue
        matches         = np.asarray([r["matches"]              for r in rows])
        matches_gap     = np.asarray([r["matches_gap_vs_gold"]  for r in rows
                                        if np.isfinite(r["matches_gap_vs_gold"])])
        delta_nc        = np.asarray([r["delta_nc_start"]       for r in rows])
        delta_flank     = np.asarray([r["delta_flank_start"]    for r in rows])
        delta_L         = np.asarray([r["delta_L"]              for r in rows])
        nc_overlap_frac = np.asarray([r["nc_overlap_frac"]      for r in rows])
        rank_amongst    = np.asarray([r["decoy_rank_among_decoys"] for r in rows])
        per_bucket[b] = {
            "n":                  int(len(rows)),
            "share":              global_dist[b],
            "median_matches":     float(np.median(matches)),
            "median_matches_gap": float(np.median(matches_gap)) if len(matches_gap) else float("nan"),
            "median_delta_nc":    float(np.median(delta_nc)),
            "median_delta_flank": float(np.median(delta_flank)),
            "median_delta_L":     float(np.median(delta_L)),
            "median_nc_ov_frac":  float(np.median(nc_overlap_frac)),
            "median_rank_amongst_decoys": float(np.median(rank_amongst)),
        }

    print("\n=== Global bucket distribution over top-{K} decoys ({N} rows) ===".format(K=args.k_top, N=total))
    print(f"  {'bucket':<32} {'share':>7} {'n':>7} {'medMatch':>9} {'medGap':>7} {'medΔnc':>8} {'medΔL':>7}")
    for b in buckets:
        s = per_bucket[b]
        if s["n"] == 0:
            print(f"  {b:<32} {'--':>7} {0:>7}")
            continue
        print(f"  {b:<32} {s['share']:>7.3f} {s['n']:>7} "
              f"{s['median_matches']:>9.1f} {s['median_matches_gap']:>+7.1f} "
              f"{s['median_delta_nc']:>+8.1f} {s['median_delta_L']:>+7.1f}")

    print(f"\n=== Per-record top-1 decoy bucket distribution ({top1_total} bags) ===")
    for b in buckets:
        n = top1_labels.count(b)
        print(f"  {b:<32} {top1_dist[b]:>7.3f} {n:>5}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "k_top":               args.k_top,
            "n_records":           n_records,
            "n_annotated":         n_annotated,
            "n_gold_in_pool":      n_gold_in_pool,
            "n_decoy_rows":        len(all_rows),
            "buckets":             buckets,
            "global_distribution": global_dist,
            "top1_distribution":   top1_dist,
            "per_bucket_stats":    per_bucket,
            "per_bag_top1":        per_bag_top1,
        }, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
