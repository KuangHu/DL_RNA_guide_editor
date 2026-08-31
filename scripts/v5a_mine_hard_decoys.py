"""V5A-1: mine hard decoys from the current candidate proposer on V4.2 positives.

For each V4.2 POS record:
  1. Build the candidate pool (same proposer used in inference).
  2. Identify the planted true candidate c* by tolerant match to
     (match_orientation, guide_length, guide_span_in_active_noncoding[0],
      target_position_in_flank[0]).
  3. Record c*'s realized rank by raw matches within the pool.
  4. Record the top-K decoys (highest raw matches, excluding c*) with their
     geometric features and taxonomy bucket relative to c*.
  5. Emit one manifest record per POS record. Bucket the record by c*'s rank
     regime {1-4, 5-20, 21-50, >50}.

No training. This is data prep for V5A-6 candidate-level ranking loss.

Output columns per record:
  site_id, transposase_id, active_nc_index
  cstar: {slot, orient, L, nc_start, flank_start, matches, rank}
  gold_in_pool (bool), pool_size (int)
  cstar_rank_regime (str)
  decoys: [{slot, orient, L, nc_start, flank_start, matches, bucket,
            delta_nc_start, delta_flank_start, delta_L,
            nc_overlap_frac, flank_overlap_frac, orient_match}]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX


_ORIENT_MAP = {"forward": "fwd", "fwd": "fwd",
                "reverse_complement": "rc", "rc": "rc",
                "reverse": "rc"}


def _canon_orient(x: str) -> str:
    if x is None: return "fwd"
    return _ORIENT_MAP.get(str(x).lower(), str(x).lower())


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _classify(c, orient: str, L: int, nc_start: int, flank_start: int,
                overlap_frac: float = 0.5) -> str:
    """Same taxonomy as r1_decoy_taxonomy but computed vs c* geometry."""
    if c.orient != orient:
        return "wrong_orientation"
    min_L = min(c.L, L)
    nc_ov    = _overlap(c.nc_start,    c.nc_start + c.L,
                          nc_start,       nc_start + L)
    flank_ov = _overlap(c.flank_start, c.flank_start + c.L,
                          flank_start,    flank_start + L)
    thresh = overlap_frac * min_L
    if nc_ov < thresh:
        return "different_region"
    dL = c.L - L
    if dL > 0: return "same_region_longer_L"
    if dL < 0: return "same_region_shorter_L"
    if flank_ov < thresh: return "same_region_same_L_wrong_flank"
    return "near_gold"


def _find_cstar(feats, mask, cands,
                  orient: str, L: int, nc_start: int, flank_start: int,
                  overlap_frac: float = 0.5) -> tuple[int, float]:
    valid_slots = np.where(mask)[0]
    if len(valid_slots) == 0: return -1, 0.0
    matches = feats[:, 3]
    best_slot = -1
    best_matches = -1.0
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
            best_matches = float(matches[i])
            best_slot = int(i)
    return best_slot, best_matches


def _rank_regime(rank: int) -> str:
    if rank < 1: return "not_in_pool"
    if rank <= 4:  return "r1_4"
    if rank <= 20: return "r5_20"
    if rank <= 50: return "r21_50"
    return "r51_plus"


def mine_record(r: dict, k_top: int = 12) -> dict | None:
    L = r["labels"]
    orient = _canon_orient(L.get("match_orientation"))
    guide_L = int(L.get("guide_length"))
    gspan = L.get("guide_span_in_active_noncoding")
    fspan = L.get("target_position_in_flank")
    if gspan is None or fspan is None:
        return None
    nc_start = int(gspan[0])
    flank_start = int(fspan[0])
    active_nc = L.get("active_noncoding_index", 0) or 0
    ncs = r["inputs"]["noncoding_regions"]
    if active_nc >= len(ncs): active_nc = 0
    nc = ncs[active_nc]
    flank = r["inputs"]["flank"]

    prof = np.zeros((len(nc), 16), dtype=np.float32)
    val = np.zeros((len(nc), 16), dtype=bool)
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)

    cstar_slot, cstar_matches = _find_cstar(
        feats, mask, cands, orient, guide_L, nc_start, flank_start)

    valid_slots = np.where(mask)[0]
    pool_size = int(len(valid_slots))
    if pool_size == 0:
        return None

    matches = feats[:, 3]
    order = valid_slots[np.argsort(-matches[valid_slots], kind="stable")]

    # c*'s realized rank (by matches; ties broken by proposer order).
    if cstar_slot >= 0:
        rank_arr = np.where(order == cstar_slot)[0]
        cstar_rank = int(rank_arr[0] + 1) if len(rank_arr) else -1
    else:
        cstar_rank = -1

    # Top-K decoys (excluding c* if present).
    decoy_order = order[order != cstar_slot] if cstar_slot >= 0 else order
    top_decoys = decoy_order[:k_top]

    decoy_rows = []
    for rank, slot in enumerate(top_decoys, start=1):
        c = cands[int(slot)]
        bucket = _classify(c, orient, guide_L, nc_start, flank_start)
        min_L = min(c.L, guide_L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + guide_L)
        flank_ov = _overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + guide_L)
        decoy_rows.append({
            "slot":               int(slot),
            "orient":             c.orient,
            "L":                  int(c.L),
            "nc_start":           int(c.nc_start),
            "flank_start":        int(c.flank_start),
            "matches":            float(matches[int(slot)]),
            "rank_among_decoys":  int(rank),
            "bucket":             bucket,
            "delta_nc_start":     int(c.nc_start - nc_start),
            "delta_flank_start":  int(c.flank_start - flank_start),
            "delta_L":            int(c.L - guide_L),
            "nc_overlap_frac":    float(nc_ov) / max(1, min_L),
            "flank_overlap_frac": float(flank_ov) / max(1, min_L),
            "orient_match":       int(c.orient == orient),
        })

    if cstar_slot >= 0:
        cs = cands[cstar_slot]
        cstar_row = {
            "slot":         int(cstar_slot),
            "orient":       cs.orient,
            "L":            int(cs.L),
            "nc_start":     int(cs.nc_start),
            "flank_start":  int(cs.flank_start),
            "matches":      float(cstar_matches),
            "rank":         int(cstar_rank),
        }
    else:
        cstar_row = {
            "slot": -1, "orient": orient, "L": guide_L,
            "nc_start": nc_start, "flank_start": flank_start,
            "matches": float("nan"), "rank": -1,
        }

    return {
        "site_id":            r["site_id"],
        "transposase_id":     r["transposase_id"],
        "active_nc_index":    int(active_nc),
        "cstar":              cstar_row,
        "gold_in_pool":       bool(cstar_slot >= 0),
        "pool_size":          pool_size,
        "cstar_rank_regime":  _rank_regime(cstar_rank),
        "decoys":             decoy_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-jsonl", required=True)
    ap.add_argument("--k-top", type=int, default=12)
    ap.add_argument("--n-records", type=int, default=0,
                     help=">0 = process only the first N records (diagnostic)")
    ap.add_argument("--shard-idx", type=int, default=0,
                     help="0-based shard index (see --n-shards)")
    ap.add_argument("--n-shards", type=int, default=1,
                     help="Total shards; task k handles records [k*chunk, (k+1)*chunk)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # If sharding, count total lines so we can split evenly.
    if args.n_shards > 1:
        with open(args.pos_jsonl) as f:
            total = sum(1 for _ in f)
        chunk = (total + args.n_shards - 1) // args.n_shards
        shard_start = args.shard_idx * chunk
        shard_end   = min(total, shard_start + chunk)
        print(f"[shard] {args.shard_idx}/{args.n_shards}  range=[{shard_start},{shard_end})  total={total}",
              flush=True)
    else:
        shard_start, shard_end = 0, 10**12

    n = n_in_pool = n_bad = 0
    regime_counter = Counter()
    bucket_counter = Counter()
    pool_sizes = []
    line_idx = 0
    with open(args.pos_jsonl) as fin, open(args.out, "w") as fout:
        for line in fin:
            if line_idx < shard_start:
                line_idx += 1; continue
            if line_idx >= shard_end:
                break
            line_idx += 1
            r = json.loads(line)
            m = mine_record(r, k_top=args.k_top)
            if m is None:
                n_bad += 1
                if args.n_records and n >= args.n_records: break
                continue
            fout.write(json.dumps(m) + "\n")
            n += 1
            if m["gold_in_pool"]: n_in_pool += 1
            regime_counter[m["cstar_rank_regime"]] += 1
            for d in m["decoys"]:
                bucket_counter[d["bucket"]] += 1
            pool_sizes.append(m["pool_size"])
            if n % 5000 == 0:
                print(f"  progress: n={n}  in_pool={n_in_pool}  bad={n_bad}", flush=True)
            if args.n_records and n >= args.n_records: break

    print(f"\n[done] processed={n}  cstar_in_pool={n_in_pool}  bad_labels={n_bad}", flush=True)
    print(f"[pool] median pool_size={float(np.median(pool_sizes)) if pool_sizes else float('nan'):.1f}",
          flush=True)
    print(f"\nc*-rank regime distribution:")
    for k in ("r1_4", "r5_20", "r21_50", "r51_plus", "not_in_pool"):
        c = regime_counter.get(k, 0)
        print(f"  {k:<12} {c:>7}  {c/max(1,n):>6.3f}")

    tot = sum(bucket_counter.values()) or 1
    print(f"\nTop-{args.k_top} decoy bucket distribution (over {tot} decoy rows):")
    for k in ("wrong_orientation", "different_region",
              "same_region_longer_L", "same_region_shorter_L",
              "same_region_same_L_wrong_flank", "near_gold"):
        c = bucket_counter.get(k, 0)
        print(f"  {k:<32} {c:>8}  {c/tot:>6.3f}")

    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
