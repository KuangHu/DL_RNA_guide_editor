"""R1-B0.5: gold-candidate recall in the current proposal pool.

For each cognate record, use the Durrant gold annotation to look up
the intended (orient, L, flank_start, nc_start) and check whether the
current proposal pool contains a candidate at those coordinates. Report:

  - rank by matches (1-indexed) of the gold if found, else None
  - R@1, R@4, R@8, R@20 — cumulative recall
  - median rank, Q75, Q90
  - total pool size for context

Also matches for shuffled bags at the SAME gold coordinates: if the same
(orient, L, flank_start, nc_start) exists in a shuffled record's proposal,
what rank does it get? Should be low (or absent), because the shuffled
sequences are unrelated.
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


def _load_gold(gold_path: str) -> dict:
    """Load gold annotation keyed by site_id."""
    out = {}
    with open(gold_path) as f:
        for line in f:
            r = json.loads(line)
            out[r["site_id"]] = r
    return out


def _gold_equivalent(c, gold_orient: str, gold_L: int,
                       gold_nc_start: int, gold_flank_start: int,
                       overlap_frac: float = 0.5) -> bool:
    """Same-orient + ≥`overlap_frac` × min(L) span overlap on BOTH nc and flank.

    Biologically tolerant: candidate proposer can find the same interaction at a
    slightly shifted window (off-by-one, adjacent L bucket) — we treat it as the
    gold hit as long as the alignment span overlaps enough on both sides.
    """
    if c is None or c.orient != gold_orient:
        return False
    min_L = min(c.L, gold_L)
    overlap_flank = max(0, min(c.flank_start + c.L, gold_flank_start + gold_L)
                            - max(c.flank_start, gold_flank_start))
    overlap_nc = max(0, min(c.nc_start + c.L, gold_nc_start + gold_L)
                         - max(c.nc_start, gold_nc_start))
    thresh = overlap_frac * min_L
    return overlap_flank >= thresh and overlap_nc >= thresh


def _find_candidate_rank(feats: np.ndarray, mask: np.ndarray, cands: list,
                          gold_orient: str, gold_L: int,
                          gold_nc_start: int, gold_flank_start: int) -> tuple[int, float]:
    """Return (rank_by_matches, gold_slot_matches) or (-1, 0.0) if not present.

    Uses tolerant matching (see `_gold_equivalent`). If multiple candidate
    slots satisfy the overlap criterion, picks the one with HIGHEST matches
    score (i.e., the best-ranking gold-equivalent).
    """
    valid_slots = np.where(mask)[0]
    if len(valid_slots) == 0:
        return -1, 0.0
    matches = feats[:, 3]
    gold_slot = -1
    best_gold_matches = -1.0
    for i in valid_slots:
        c = cands[i]
        if not _gold_equivalent(c, gold_orient, gold_L, gold_nc_start, gold_flank_start):
            continue
        if matches[i] > best_gold_matches:
            best_gold_matches = float(matches[i])
            gold_slot = int(i)
    if gold_slot < 0:
        return -1, 0.0
    valid_matches = matches[valid_slots]
    rank = int((valid_matches > best_gold_matches).sum() + 1)
    return rank, best_gold_matches


def per_bag_gold_recall(nc: str, flank: str,
                          gold_orient: str, gold_L: int,
                          gold_nc_start: int, gold_flank_start: int) -> dict:
    prof = np.zeros((len(nc), 16), dtype=np.float32)
    val = np.zeros((len(nc), 16), dtype=bool)
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, prof, val,
        L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX,
    )
    rank, gold_matches = _find_candidate_rank(feats, mask, cands,
                                                gold_orient, gold_L,
                                                gold_nc_start, gold_flank_start)
    n_pool = int(mask.sum())
    # Best decoy: max matches among valid slots EXCLUDING the gold
    matches = feats[:, 3]
    valid_slots = np.where(mask)[0]
    if rank > 0:
        non_gold = [i for i in valid_slots if not _gold_equivalent(
            cands[i], gold_orient, gold_L, gold_nc_start, gold_flank_start)]
        best_decoy_matches = float(matches[non_gold].max()) if non_gold else 0.0
    else:
        best_decoy_matches = float(matches[valid_slots].max()) if len(valid_slots) else 0.0
    return {
        "rank":               int(rank) if rank > 0 else None,
        "gold_matches":       gold_matches,
        "best_decoy_matches": best_decoy_matches,
        "margin":             gold_matches - best_decoy_matches,
        "pool_size":          n_pool,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cognate-jsonl", required=True)
    ap.add_argument("--shuffled-jsonl", required=True)
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = _load_gold(args.gold_jsonl)
    print(f"[gold] {len(gold)} annotated site_ids", flush=True)

    def _process(path, tag):
        results = []
        n_matched = n_lookup_hit = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                sid = r["site_id"]
                # For shuffled records, the site_id is different but we can strip _shu → _cog
                gold_sid = sid.replace("_shu_", "_cog_")
                g = gold.get(gold_sid)
                if g is None:
                    continue
                n_lookup_hit += 1
                active_nc = r["labels"].get("active_noncoding_index", 0) or 0
                ncs = r["inputs"]["noncoding_regions"]
                if active_nc >= len(ncs): active_nc = 0
                nc = ncs[active_nc]
                flank = r["inputs"]["flank"]
                stats = per_bag_gold_recall(
                    nc, flank,
                    gold_orient=g["target_flank_orientation"],
                    gold_L=g["target_binding_loop_length"],
                    gold_nc_start=g["guide_start_in_nc"],
                    gold_flank_start=g["target_flank_start"],
                )
                stats["site_id"] = sid
                stats["gold_source"] = gold_sid
                results.append(stats)
                n_matched += 1
        print(f"[{tag}] gold-annotated records: {n_lookup_hit}, of which matched to candidates: {n_matched}",
              flush=True)
        return results

    cog_results = _process(args.cognate_jsonl, "cognate")
    shu_results = _process(args.shuffled_jsonl, "shuffled")

    # Summarize
    def _summary(rs):
        found = [r for r in rs if r["rank"] is not None]
        ranks = [r["rank"] for r in found]
        margins = [r["margin"] for r in found]
        pool_sizes = [r["pool_size"] for r in found]
        gold_matches = [r["gold_matches"] for r in rs]
        best_decoy_matches = [r["best_decoy_matches"] for r in rs]
        return {
            "n_total":           len(rs),
            "n_in_pool":         len(found),
            "n_missing":         len(rs) - len(found),
            "recall@1":          sum(1 for r in ranks if r == 1) / max(1, len(rs)),
            "recall@4":          sum(1 for r in ranks if r <= 4) / max(1, len(rs)),
            "recall@8":          sum(1 for r in ranks if r <= 8) / max(1, len(rs)),
            "recall@20":         sum(1 for r in ranks if r <= 20) / max(1, len(rs)),
            "recall@50":         sum(1 for r in ranks if r <= 50) / max(1, len(rs)),
            "recall@in_pool":    len(ranks) / max(1, len(rs)),
            "median_rank":       float(np.median(ranks)) if ranks else float("nan"),
            "q75_rank":          float(np.quantile(ranks, 0.75)) if ranks else float("nan"),
            "q90_rank":          float(np.quantile(ranks, 0.90)) if ranks else float("nan"),
            "median_pool_size":  float(np.median(pool_sizes)) if pool_sizes else float("nan"),
            "median_gold_matches":      float(np.median(gold_matches)) if gold_matches else float("nan"),
            "median_best_decoy_matches": float(np.median(best_decoy_matches)) if best_decoy_matches else float("nan"),
            "median_margin":     float(np.median(margins)) if margins else float("nan"),
            "p_margin_pos":      float((np.asarray(margins) > 0).mean()) if margins else float("nan"),
        }

    cog_sum = _summary(cog_results)
    shu_sum = _summary(shu_results)

    print("\n=== gold candidate recall in the current proposal pool ===")
    print(f"{'metric':<30} {'cognate':>12} {'shuffled':>12}")
    for k in ("n_total", "n_in_pool", "n_missing",
              "recall@1", "recall@4", "recall@8", "recall@20", "recall@50", "recall@in_pool",
              "median_rank", "q75_rank", "q90_rank",
              "median_pool_size",
              "median_gold_matches", "median_best_decoy_matches",
              "median_margin", "p_margin_pos"):
        cv = cog_sum[k]; sv = shu_sum[k]
        if isinstance(cv, float):
            print(f"  {k:<28} {cv:>12.3f} {sv:>12.3f}")
        else:
            print(f"  {k:<28} {cv:>12} {sv:>12}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "cognate": cog_sum,
            "shuffled": shu_sum,
            "cognate_records": cog_results,
            "shuffled_records": shu_results,
        }, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
