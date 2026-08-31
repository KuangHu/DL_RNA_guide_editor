"""V5A-1c: spurious-alignment density comparison V4.2 vs Durrant.

For each cognate bag on both datasets, run the current candidate proposer and
compute per-bag statistics of the top decoys AND the gold candidate:

  best_decoy_matches                   — highest raw matches among non-gold slots
  best_decoy_identity                  — highest m/L among non-gold slots
  gold_matches, gold_identity, gold_L  — from the tolerant-matched gold slot
  n_decoy_ge_gold_matches              — # decoys with matches >= gold_matches
  n_decoy_ge_gold_identity             — # decoys with m/L    >= gold_identity
  pool_size

Aggregate: median / Q90 / Q99 of best_decoy_matches; per-L stratified median
best_decoy_matches; median N(≥ gold_matches), N(≥ gold_identity) per bag.

Purpose: check whether V4.2's decoy landscape is systematically less crowded
than Durrant's, which would mean a selector calibrated on V4.2 will over-trust
raw match count when transferred to real data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX


_ORIENT_MAP = {"forward": "fwd", "fwd": "fwd",
                "reverse_complement": "rc", "rc": "rc",
                "reverse": "rc"}


def _canon_orient(x):
    if x is None: return "fwd"
    return _ORIENT_MAP.get(str(x).lower(), str(x).lower())


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _find_gold_slot(feats, mask, cands, orient, L, nc_start, flank_start,
                     overlap_frac=0.5):
    valid_slots = np.where(mask)[0]
    if len(valid_slots) == 0: return -1, 0.0
    matches = feats[:, 3]
    best_slot = -1; best_matches = -1.0
    for i in valid_slots:
        c = cands[i]
        if c.orient != orient: continue
        min_L = min(c.L, L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
        flank_ov = _overlap(c.flank_start, c.flank_start + c.L,
                              flank_start, flank_start + L)
        thresh = overlap_frac * min_L
        if nc_ov < thresh or flank_ov < thresh: continue
        if matches[i] > best_matches:
            best_matches = float(matches[i]); best_slot = int(i)
    return best_slot, best_matches


def per_bag_stats(nc, flank, gold_coords):
    prof = np.zeros((len(nc), 16), dtype=np.float32)
    val = np.zeros((len(nc), 16), dtype=bool)
    _, feats, mask, cands = build_candidate_arrays(
        nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)

    valid = np.where(mask)[0]
    if len(valid) == 0: return None
    matches = feats[:, 3]
    Ls = np.asarray([cands[int(i)].L for i in valid], dtype=np.int32)
    idents = matches[valid] / np.maximum(1, Ls)

    orient, L, nc_start, flank_start = gold_coords
    gold_slot, gold_matches = _find_gold_slot(
        feats, mask, cands, orient, L, nc_start, flank_start)

    if gold_slot >= 0:
        gold_L_actual = int(cands[gold_slot].L)
        gold_identity = gold_matches / max(1, gold_L_actual)
        non_gold_mask = valid != gold_slot
        best_decoy_matches = float(matches[valid][non_gold_mask].max()) if non_gold_mask.any() else 0.0
        best_decoy_ident = float(idents[non_gold_mask].max()) if non_gold_mask.any() else 0.0
        n_ge_matches = int((matches[valid][non_gold_mask] >= gold_matches).sum())
        n_ge_identity = int((idents[non_gold_mask] >= gold_identity).sum())
    else:
        gold_L_actual = L
        gold_identity = float("nan")
        best_decoy_matches = float(matches[valid].max())
        best_decoy_ident = float(idents.max())
        n_ge_matches = -1  # sentinel: no gold in pool
        n_ge_identity = -1

    # Per-L strata top-decoy matches
    per_L = defaultdict(list)
    for i in valid:
        L_i = int(cands[int(i)].L)
        per_L[L_i].append(float(matches[i]))

    return {
        "gold_in_pool":            bool(gold_slot >= 0),
        "gold_matches":            float(gold_matches) if gold_slot >= 0 else float("nan"),
        "gold_L":                  gold_L_actual,
        "gold_identity":           gold_identity,
        "best_decoy_matches":      best_decoy_matches,
        "best_decoy_identity":     best_decoy_ident,
        "n_decoy_ge_gold_matches": n_ge_matches,
        "n_decoy_ge_gold_identity": n_ge_identity,
        "pool_size":               int(len(valid)),
        "per_L_max_matches":       {int(k): max(v) for k, v in per_L.items()},
    }


def _summary(rows, tag):
    if not rows:
        print(f"[{tag}] no rows"); return None
    bdm = np.asarray([r["best_decoy_matches"] for r in rows])
    bdi = np.asarray([r["best_decoy_identity"] for r in rows])
    pool = np.asarray([r["pool_size"] for r in rows])
    in_pool = [r for r in rows if r["gold_in_pool"]]
    if in_pool:
        gold_m = np.asarray([r["gold_matches"]  for r in in_pool])
        gold_i = np.asarray([r["gold_identity"] for r in in_pool])
        n_ge_m = np.asarray([r["n_decoy_ge_gold_matches"]  for r in in_pool])
        n_ge_i = np.asarray([r["n_decoy_ge_gold_identity"] for r in in_pool])
    else:
        gold_m = gold_i = n_ge_m = n_ge_i = np.asarray([])

    # Per-L stratified
    strata = defaultdict(list)
    for r in rows:
        for L, m in r["per_L_max_matches"].items():
            strata[L].append(m)
    per_L = {L: {"n": len(v),
                   "median": float(np.median(v)),
                   "q90":    float(np.quantile(v, 0.90))}
                for L, v in sorted(strata.items())}

    stats = {
        "n_bags":                    int(len(rows)),
        "n_gold_in_pool":            int(len(in_pool)),
        "best_decoy_matches_median": float(np.median(bdm)),
        "best_decoy_matches_q90":    float(np.quantile(bdm, 0.90)),
        "best_decoy_matches_q99":    float(np.quantile(bdm, 0.99)),
        "best_decoy_identity_median": float(np.median(bdi)),
        "best_decoy_identity_q90":   float(np.quantile(bdi, 0.90)),
        "pool_size_median":          float(np.median(pool)),
        "gold_matches_median":       float(np.median(gold_m)) if len(gold_m) else float("nan"),
        "gold_identity_median":      float(np.median(gold_i)) if len(gold_i) else float("nan"),
        "n_decoy_ge_gold_matches_median":  float(np.median(n_ge_m)) if len(n_ge_m) else float("nan"),
        "n_decoy_ge_gold_matches_q90":     float(np.quantile(n_ge_m, 0.90)) if len(n_ge_m) else float("nan"),
        "n_decoy_ge_gold_identity_median": float(np.median(n_ge_i)) if len(n_ge_i) else float("nan"),
        "n_decoy_ge_gold_identity_q90":    float(np.quantile(n_ge_i, 0.90)) if len(n_ge_i) else float("nan"),
        "per_L":                     per_L,
    }
    return stats


def _print_summary(name, s):
    print(f"\n=== {name} ===")
    print(f"  n_bags={s['n_bags']}  gold_in_pool={s['n_gold_in_pool']}  pool_size_median={s['pool_size_median']:.0f}")
    print(f"  best_decoy_matches   median={s['best_decoy_matches_median']:.2f}  "
          f"Q90={s['best_decoy_matches_q90']:.2f}  Q99={s['best_decoy_matches_q99']:.2f}")
    print(f"  best_decoy_identity  median={s['best_decoy_identity_median']:.3f}  "
          f"Q90={s['best_decoy_identity_q90']:.3f}")
    print(f"  gold  matches median={s['gold_matches_median']:.2f}  identity median={s['gold_identity_median']:.3f}")
    print(f"  N(decoy_matches  >= gold_matches )   median={s['n_decoy_ge_gold_matches_median']:.1f}  "
          f"Q90={s['n_decoy_ge_gold_matches_q90']:.1f}")
    print(f"  N(decoy_identity >= gold_identity)   median={s['n_decoy_ge_gold_identity_median']:.1f}  "
          f"Q90={s['n_decoy_ge_gold_identity_q90']:.1f}")
    print(f"  per-L max_matches (median | Q90):")
    for L, v in s["per_L"].items():
        print(f"    L={L:<3} n={v['n']:>5}  median={v['median']:.2f}  Q90={v['q90']:.2f}")


def process_dataset(cognate_jsonl, gold_source, n_records=0):
    """gold_source: either 'v42_labels' (read planted c* from V4.2 record labels)
    or a dict site_id -> gold record (from durrant_gold_v1.jsonl)."""
    rows = []
    with open(cognate_jsonl) as f:
        for i, line in enumerate(f):
            r = json.loads(line)
            if gold_source == "v42_labels":
                L = r["labels"]
                gspan = L.get("guide_span_in_active_noncoding")
                fspan = L.get("target_position_in_flank")
                if gspan is None or fspan is None: continue
                gold_coords = (_canon_orient(L.get("match_orientation")),
                                int(L.get("guide_length")),
                                int(gspan[0]), int(fspan[0]))
            else:
                g = gold_source.get(r["site_id"])
                if g is None: continue
                gold_coords = (g["target_flank_orientation"],
                                g["target_binding_loop_length"],
                                g["guide_start_in_nc"],
                                g["target_flank_start"])
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            stats = per_bag_stats(ncs[active_nc], r["inputs"]["flank"], gold_coords)
            if stats is None: continue
            rows.append(stats)
            if n_records and len(rows) >= n_records: break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v42-jsonl", required=True)
    ap.add_argument("--durrant-cognate", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--n-v42", type=int, default=2000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = {}
    with open(args.durrant_gold) as f:
        for line in f:
            r = json.loads(line); gold[r["site_id"]] = r
    print(f"[gold] {len(gold)} Durrant gold rows", flush=True)

    print(f"[V4.2] processing {args.n_v42} records ...", flush=True)
    v42_rows = process_dataset(args.v42_jsonl, "v42_labels", n_records=args.n_v42)
    print(f"[V4.2] {len(v42_rows)} rows", flush=True)

    print(f"[Durrant] processing all records ...", flush=True)
    durrant_rows = process_dataset(args.durrant_cognate, gold)
    print(f"[Durrant] {len(durrant_rows)} rows", flush=True)

    v42_stats = _summary(v42_rows, "v42")
    dur_stats = _summary(durrant_rows, "durrant")

    _print_summary("V4.2", v42_stats)
    _print_summary("Durrant", dur_stats)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"v42": v42_stats, "durrant": dur_stats}, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
