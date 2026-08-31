"""V5A priority-P0-2: what does raw_m alone score on frozen Durrant?

The old pipeline's R@1=7.7% / median rank=36 was NOT raw_m — that was the
learned MIL. Before we invest anything else, we need to know whether the
strongest single deterministic scalar already beats the MIL on real data.

For each Durrant cognate record with a gold annotation, rebuild the candidate
pool via the same proposer, tolerant-match to gold, and compute the expected
rank of gold with uniform tie-break (same convention as val eval), then report:

  pooled MRR + R@1 + R@4 + R@8 + median rank
  taxonomy P(gold > d) per decoy bucket
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _find_gold_slot(feats, mask, cands, orient, L, nc_start, flank_start,
                      overlap_frac=0.5):
    valid = np.where(mask)[0]
    if len(valid) == 0: return -1, 0.0
    matches = feats[:, 3]
    best_slot = -1; best_m = -1.0
    for i in valid:
        c = cands[i]
        if c.orient != orient: continue
        mn_L = min(c.L, L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
        f_ov = _overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + L)
        th = overlap_frac * mn_L
        if nc_ov < th or f_ov < th: continue
        if matches[i] > best_m:
            best_m = float(matches[i]); best_slot = int(i)
    return best_slot, best_m


def _classify(c, orient, L, nc_start, flank_start, overlap_frac=0.5):
    if c.orient != orient: return "wrong_orientation"
    mn_L = min(c.L, L)
    nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
    f_ov = _overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + L)
    th = overlap_frac * mn_L
    if nc_ov < th: return "different_region"
    dL = c.L - L
    if dL > 0: return "same_region_longer_L"
    if dL < 0: return "same_region_shorter_L"
    if f_ov < th: return "same_region_same_L_wrong_flank"
    return "near_gold"


def _rank_stats(q_slot, cs_local, k_list=(1, 4, 8)):
    q_cs = q_slot[cs_local]
    others = np.delete(q_slot, cs_local)
    n_gt = int((others > q_cs).sum())
    n_eq = int((others == q_cs).sum())
    tie = n_eq + 1
    rank_avg = n_gt + 1 + n_eq / 2.0
    R = {}
    for k in k_list:
        R[k] = 0.0 if n_gt >= k else min(1.0, (k - n_gt) / tie)
    idx = np.arange(tie, dtype=np.float64)
    E_recip = float(np.mean(1.0 / (n_gt + 1 + idx)))
    return rank_avg, R, E_recip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cognate-jsonl", required=True)
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    gold = {}
    with open(args.gold_jsonl) as f:
        for line in f:
            r = json.loads(line); gold[r["site_id"]] = r

    n_bags = n_gold_in_pool = 0
    pooled = []
    p_beats = {b: [] for b in ("wrong_orientation","different_region",
        "same_region_longer_L","same_region_shorter_L",
        "same_region_same_L_wrong_flank","near_gold")}
    with open(args.cognate_jsonl) as f:
        for line in f:
            r = json.loads(line)
            n_bags += 1
            g = gold.get(r["site_id"])
            if g is None: continue
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]; flank = r["inputs"]["flank"]

            prof = np.zeros((len(nc), 16), dtype=np.float32)
            val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)

            orient = g["target_flank_orientation"]
            g_L = g["target_binding_loop_length"]
            g_nc = g["guide_start_in_nc"]; g_fl = g["target_flank_start"]

            gold_slot, gold_m = _find_gold_slot(feats, mask, cands,
                                                    orient, g_L, g_nc, g_fl)
            if gold_slot < 0: continue
            n_gold_in_pool += 1

            valid = np.where(mask)[0]
            local_slots = list(valid)
            cs_local = local_slots.index(gold_slot)
            qs = feats[valid, 3]                           # raw m
            rank_avg, R, E_recip = _rank_stats(qs, cs_local)
            pooled.append({"rank_avg": rank_avg, "R1": R[1], "R4": R[4],
                              "R8": R[8], "MRR": E_recip})
            # taxonomy P(gold > d): strict >
            q_cs = qs[cs_local]
            for j, slot_id in enumerate(local_slots):
                if j == cs_local: continue
                c = cands[int(slot_id)]
                bucket = _classify(c, orient, g_L, g_nc, g_fl)
                if bucket in p_beats:
                    p_beats[bucket].append(int(q_cs > qs[j]))

    if not pooled:
        print("[raw_m@durrant] no records!"); return
    arr_R1 = np.asarray([r["R1"] for r in pooled])
    arr_R4 = np.asarray([r["R4"] for r in pooled])
    arr_R8 = np.asarray([r["R8"] for r in pooled])
    arr_MRR = np.asarray([r["MRR"] for r in pooled])
    arr_rank = np.asarray([r["rank_avg"] for r in pooled])
    report = {
        "n_bags":     n_bags,
        "n_in_pool":  n_gold_in_pool,
        "pooled": {
            "MRR":            float(arr_MRR.mean()),
            "R@1":            float(arr_R1.mean()),
            "R@4":            float(arr_R4.mean()),
            "R@8":            float(arr_R8.mean()),
            "median_rank_avg": float(np.median(arr_rank)),
        },
        "taxonomy_p_beats": {k: (float(np.mean(v)) if v else float("nan"), len(v))
                                for k, v in p_beats.items()},
    }
    print(f"\n=== raw_m on frozen Durrant (n_bags={n_bags}, gold_in_pool={n_gold_in_pool}) ===")
    p = report["pooled"]
    print(f"  POOLED  MRR={p['MRR']:.4f}  R@1={p['R@1']:.4f}  R@4={p['R@4']:.4f}  "
          f"R@8={p['R@8']:.4f}  median_rank_avg={p['median_rank_avg']:.1f}")
    print(f"  taxonomy P(gold>d):")
    for k, (v, n) in report["taxonomy_p_beats"].items():
        print(f"    {k:<32} {v:.3f}  n={n}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
