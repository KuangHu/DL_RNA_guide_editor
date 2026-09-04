"""Pool-recovery measurement on the v2 candidate layer.

Same schema as `pool_recovery.py` but reads candidates from
`preprocess.candidates_v2.enumerate_candidates_v2` instead of the old
96-slot per-(orient, L) top-K. Records the same four grades:

  - strict  (exact planted (orient, L, nc_start, flank_start) in pool)
  - L_full  (any pool cand at planted L with flank IoU >= 0.5)
  - subL    (any pool cand at L in {9..planted_L-2} with flank IoU >= 0.5)
  - any_L   (any pool cand at any L with flank IoU >= 0.5)

Emits one JSONL row per site — same format as `pool_recovery.py` for
downstream stratifier compatibility.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.candidates_v2 import (
    enumerate_candidates_v2,
    DEFAULT_E_THRESH,
    DEFAULT_L_MIN,
    DEFAULT_L_MAX,
)


def _flank_iou(cand_start: int, cand_L: int, planted_start: int, planted_end: int) -> float:
    a0, a1 = cand_start, cand_start + cand_L
    b0, b1 = planted_start, planted_end
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def measure_site_v2(record: dict, iou_thresh: float = 0.5,
                    e_thresh: float = DEFAULT_E_THRESH,
                    L_min: int = DEFAULT_L_MIN,
                    L_max: int = DEFAULT_L_MAX) -> dict:
    lab = record["labels"]
    inp = record["inputs"]
    arch = lab["arch"]
    active_idx = lab["active_noncoding_index"]
    nc = inp["noncoding_regions"][active_idx]
    flank = inp["flank"]

    L = arch["L"]
    orient = "rc" if arch["orient"] == "rev" else arch["orient"]
    is_split = arch["is_split"]
    mm_conc = arch["mm_concentration"]
    planted_m = lab["planted_m"]
    guide_nc_start, _ = lab["guide_span_in_active_noncoding"]

    A_start = lab["planted_start"]
    A_end = lab["planted_A_end"]
    B_start = lab["planted_B_start"]
    B_end = lab["planted_B_end"]

    cands = enumerate_candidates_v2(
        nc, flank, L_min=L_min, L_max=L_max, e_thresh=e_thresh,
    )
    planted_cands = [c for c in cands if c.orient == orient]

    def hits_region(p_start: int, p_end: int) -> dict:
        strict = any(
            c.L == L and c.nc_start == guide_nc_start and c.flank_start == p_start
            for c in planted_cands
        )
        L_full_iou = max(
            (_flank_iou(c.flank_start, c.L, p_start, p_end)
             for c in planted_cands if c.L == L),
            default=0.0,
        )
        subL_iou = max(
            (_flank_iou(c.flank_start, c.L, p_start, p_end)
             for c in planted_cands if 9 <= c.L <= L - 2),
            default=0.0,
        )
        any_L_iou = max(
            (_flank_iou(c.flank_start, c.L, p_start, p_end)
             for c in planted_cands),
            default=0.0,
        )
        return {
            "strict": bool(strict),
            "L_full": L_full_iou >= iou_thresh,
            "subL": subL_iou >= iou_thresh,
            "any_L": any_L_iou >= iou_thresh,
        }

    if is_split:
        h_a = hits_region(A_start, A_end)
        h_b = hits_region(B_start, B_end)
        hits = {k: h_a[k] or h_b[k] for k in ("strict", "L_full", "subL", "any_L")}
    else:
        hits = hits_region(A_start, B_end)

    return {
        "transposase_id": record.get("transposase_id"),
        "site_id": record.get("site_id"),
        "L": L,
        "planted_m": planted_m,
        "mm_conc": mm_conc,
        "is_split": bool(is_split),
        "orient": arch["orient"],
        "n_pool": len(cands),
        **hits,
    }


def _worker(line: str) -> dict:
    return measure_site_v2(json.loads(line))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunksize", type=int, default=32)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"[pool-recovery-v2] input:   {args.v5_jsonl}")
    print(f"[pool-recovery-v2] workers: {args.workers}")

    t0 = time.time()
    with open(args.v5_jsonl) as fin, open(args.out, "w") as fout:
        lines: Iterable[str] = fin
        if args.limit > 0:
            lines = (l for i, l in enumerate(fin) if i < args.limit)
        if args.workers <= 1:
            for i, line in enumerate(lines):
                fout.write(json.dumps(_worker(line)) + "\n")
                if (i + 1) % 5000 == 0:
                    dt = time.time() - t0
                    print(f"  [{i+1}] {dt:.1f}s ({dt/(i+1)*1000:.1f} ms/site)")
        else:
            with mp.Pool(args.workers) as pool:
                for i, rec in enumerate(pool.imap_unordered(_worker, lines,
                                                              chunksize=args.chunksize)):
                    fout.write(json.dumps(rec) + "\n")
                    if (i + 1) % 5000 == 0:
                        dt = time.time() - t0
                        print(f"  [{i+1}] {dt:.1f}s ({dt/(i+1)*1000:.1f} ms/site amortized)")
    print(f"[pool-recovery-v2] done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
