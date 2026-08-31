"""V5A-3a0 eval prep: pre-compute the FULL candidate pool per val/test record.

For each record in the given tnp split, rebuild the proposer pool via
`build_candidate_arrays` and, for every VALID slot, compute the 9 minimal-local
features consumed by V5A-3a0. Also identify c*'s slot via tolerant match to the
planted labels; store its slot index.

Adds per-record:
  pool_size, cstar_slot (-1 if c* not in pool)
  cstar_rank (1-indexed by matches, ties handled per proposer order; -1 if -1)
  full_pool_burden_ge  = #{d in pool: m_d >= m_cstar}
  full_pool_burden_gt  = #{d in pool: m_d >  m_cstar}
  slots: array of {orient, L, nc_start, flank_start, matches, mm_count,
                    mm_frac_5p, mm_frac_3p, mm_at_pos_0, mm_at_pos_last, log_tail,
                    bucket ('cstar', 'wrong_orientation', 'different_region',
                    'same_region_longer_L', 'same_region_shorter_L',
                    'same_region_same_L_wrong_flank', 'near_gold')}

The `bucket` field lets us stratify eval by decoy taxonomy without recomputing
at eval time. c* itself is labeled 'cstar' — do NOT feed it as a model input.
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


_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def _revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


_ORIENT_MAP = {"forward": "fwd", "fwd": "fwd",
                "reverse_complement": "rc", "rc": "rc",
                "reverse": "rc"}


def _canon_orient(x):
    if x is None: return "fwd"
    return _ORIENT_MAP.get(str(x).lower(), str(x).lower())


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _mismatch_summary(nc: str, flank: str, orient: str, L: int,
                      nc_start: int, flank_start: int) -> dict:
    if nc_start < 0 or nc_start + L > len(nc) or flank_start < 0 or flank_start + L > len(flank):
        return {"mm_count": 0, "mm_frac_5p": 0.0, "mm_frac_3p": 0.0,
                 "mm_at_pos_0": 0, "mm_at_pos_last": 0}
    guide = nc[nc_start:nc_start + L].upper().replace("U", "T")
    target = flank[flank_start:flank_start + L].upper()
    if orient == "rc":
        target = _revcomp(target)
    mm = [0 if guide[i] == target[i] else 1 for i in range(L)]
    half = L // 2
    return {
        "mm_count":       int(sum(mm)),
        "mm_frac_5p":     float(sum(mm[:half])) / max(1, half) if half else 0.0,
        "mm_frac_3p":     float(sum(mm[L - half:])) / max(1, half) if half else 0.0,
        "mm_at_pos_0":    int(mm[0]),
        "mm_at_pos_last": int(mm[-1]),
    }


def _load_null_table(path: str):
    d = json.load(open(path))["per_L"]
    tables = {}
    for L, t in d.items():
        L = int(L)
        mvals = np.asarray(t["m_values"], dtype=np.float32)
        lt = np.asarray(t["log_tail"], dtype=np.float32)
        tables[L] = (mvals, lt)
    return tables


def _log_tail(tables, m: float, L: int) -> float:
    t = tables.get(int(L))
    if t is None: return 0.0
    mvals, lt = t
    idx = int(np.searchsorted(mvals, m, side="left"))
    if idx == len(mvals):
        return float(lt[-1])
    return float(lt[idx])


def _classify(c, orient, L, nc_start, flank_start, overlap_frac=0.5):
    if c.orient != orient: return "wrong_orientation"
    min_L = min(c.L, L)
    nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
    flank_ov = _overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + L)
    thresh = overlap_frac * min_L
    if nc_ov < thresh: return "different_region"
    dL = c.L - L
    if dL > 0: return "same_region_longer_L"
    if dL < 0: return "same_region_shorter_L"
    if flank_ov < thresh: return "same_region_same_L_wrong_flank"
    return "near_gold"


def _find_cstar(feats, mask, cands, orient, L, nc_start, flank_start, overlap_frac=0.5):
    valid = np.where(mask)[0]
    if len(valid) == 0: return -1, 0.0
    matches = feats[:, 3]
    best_slot = -1; best_matches = -1.0
    for i in valid:
        c = cands[i]
        if c.orient != orient: continue
        min_L = min(c.L, L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
        flank_ov = _overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + L)
        thresh = overlap_frac * min_L
        if nc_ov < thresh or flank_ov < thresh: continue
        if matches[i] > best_matches:
            best_matches = float(matches[i]); best_slot = int(i)
    return best_slot, best_matches


def build_record(r, tables):
    L_field = r["labels"]
    gspan = L_field.get("guide_span_in_active_noncoding")
    fspan = L_field.get("target_position_in_flank")
    if gspan is None or fspan is None: return None
    orient = _canon_orient(L_field.get("match_orientation"))
    g_L = int(L_field.get("guide_length"))
    nc_start = int(gspan[0]); flank_start = int(fspan[0])
    active_nc = L_field.get("active_noncoding_index", 0) or 0
    ncs = r["inputs"]["noncoding_regions"]
    if active_nc >= len(ncs): active_nc = 0
    nc = ncs[active_nc]; flank = r["inputs"]["flank"]

    prof = np.zeros((len(nc), 16), dtype=np.float32)
    val = np.zeros((len(nc), 16), dtype=bool)
    _, feats, mask, cands = build_candidate_arrays(
        nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)

    valid_slots = np.where(mask)[0]
    pool_size = int(len(valid_slots))
    if pool_size == 0: return None

    cstar_slot, cstar_matches = _find_cstar(
        feats, mask, cands, orient, g_L, nc_start, flank_start)

    matches = feats[:, 3]
    order = valid_slots[np.argsort(-matches[valid_slots], kind="stable")]
    if cstar_slot >= 0:
        rank_arr = np.where(order == cstar_slot)[0]
        cstar_rank = int(rank_arr[0] + 1) if len(rank_arr) else -1
        B_gt = int((matches[valid_slots] > cstar_matches).sum())
        B_ge = int((matches[valid_slots] >= cstar_matches).sum()) - 1  # exclude c* itself
    else:
        cstar_rank = -1; B_gt = -1; B_ge = -1

    slots = []
    for i in valid_slots:
        i = int(i)
        c = cands[i]
        mm = _mismatch_summary(nc, flank, c.orient, int(c.L),
                                 int(c.nc_start), int(c.flank_start))
        if i == cstar_slot:
            bucket = "cstar"
        else:
            bucket = _classify(c, orient, g_L, nc_start, flank_start) \
                        if cstar_slot >= 0 else "unknown"
        slots.append({
            "slot":         i,
            "orient":       c.orient,
            "L":            int(c.L),
            "nc_start":     int(c.nc_start),
            "flank_start":  int(c.flank_start),
            "matches":      float(matches[i]),
            "identity":     float(matches[i]) / max(1, int(c.L)),
            "log_tail":     _log_tail(tables, float(matches[i]), int(c.L)),
            "bucket":       bucket,
            **mm,
        })
    return {
        "site_id":                r["site_id"],
        "transposase_id":         r["transposase_id"],
        "pool_size":              pool_size,
        "cstar_slot":             int(cstar_slot),
        "cstar_matches":          float(cstar_matches) if cstar_slot >= 0 else float("nan"),
        "cstar_rank":             cstar_rank,
        "full_pool_burden_ge":    B_ge,
        "full_pool_burden_gt":    B_gt,
        "slots":                  slots,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos-jsonl", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--split-name", required=True, choices=["train", "val", "test"])
    ap.add_argument("--null-table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    args = ap.parse_args()

    tables = _load_null_table(args.null_table)
    splits = json.load(open(args.splits))
    keep = set(splits[args.split_name])
    print(f"[splits] {args.split_name}: {len(keep)} tnps", flush=True)

    # Determine shard slice by counting eligible records first.
    if args.n_shards > 1:
        with open(args.pos_jsonl) as f:
            elig = [i for i, line in enumerate(f) if json.loads(line)["transposase_id"] in keep]
        chunk = (len(elig) + args.n_shards - 1) // args.n_shards
        shard_start = args.shard_idx * chunk
        shard_end = min(len(elig), shard_start + chunk)
        wanted = set(elig[shard_start:shard_end])
        print(f"[shard] {args.shard_idx}/{args.n_shards}  wanted={len(wanted)}", flush=True)
    else:
        wanted = None

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_bad = 0
    with open(args.pos_jsonl) as fin, open(args.out, "w") as fo:
        for i, line in enumerate(fin):
            r = json.loads(line)
            if r["transposase_id"] not in keep: continue
            if wanted is not None and i not in wanted: continue
            rec = build_record(r, tables)
            if rec is None:
                n_bad += 1; continue
            fo.write(json.dumps(rec) + "\n")
            n_ok += 1
            if n_ok % 2000 == 0:
                print(f"  progress: {n_ok}  bad={n_bad}", flush=True)
    print(f"[done] wrote {n_ok} records (bad={n_bad}) to {args.out}", flush=True)


if __name__ == "__main__":
    main()
