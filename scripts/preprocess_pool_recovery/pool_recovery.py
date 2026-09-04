"""Measure pool-recovery of planted V5 sites through the old 96-slot preprocessor.

For each site in the V5 JSONL, run the same top-K-per-(orient, L) admission
rule as `preprocess.candidates.build_candidate_arrays` — but skip the patch/
feature fill and just return the Candidate list. Then check whether the pool
admits a candidate that covers the ground-truth planted position on the flank.

Three recovery grades are measured, stratified by
(planted_L, planted_m, mm_concentration, is_split):

  1. **strict**   pool has a candidate at exact (planted orient, planted L,
                  nc_start = guide_nc_start, flank_start = planted_start).
  2. **L_full**   pool has a candidate at (planted orient, planted L) whose
                  flank window overlaps the planted flank position with IoU
                  >= 0.5.
  3. **subL**     pool has a candidate at (planted orient, some L' in
                  {9, ..., planted_L - 2}) whose flank window overlaps the
                  planted flank position with IoU >= 0.5 -- the L=9 subwindow
                  rescue path (matches Durrant's 40%-at-L=9 phenomenon).
  4. **any_L**    pool has a candidate at (planted orient, any L in [5, 16])
                  whose flank window overlaps planted with IoU >= 0.5.

Split-mode sites (is_split=True) are handled by treating each half (A, B)
as a separate planted region on the flank. A pool candidate that covers
either half is counted as a hit on that half.

Falsifiable predictions (recorded in memory: finding_mm_geometry_effect):
  - clustered   L_full ≈ 0, subL ≈ 1 (predicted from W2 E-value table)
  - dispersed   L_full ≈ 0, subL ≈ 0 (no clean subwindow anywhere)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from preprocess.candidates import (
    TOP_K_PER_COMBO_DEFAULT,
    DEFAULT_L_MIN,
    DEFAULT_L_MAX,
    DEFAULT_ORIENTATIONS,
)


@dataclass(frozen=True, slots=True)
class LightCandidate:
    orient: str
    L: int
    nc_start: int
    flank_start: int
    matches: int


def enumerate_pool(
    nc: str,
    flank: str,
    top_k: int = TOP_K_PER_COMBO_DEFAULT,
    L_min: int = DEFAULT_L_MIN,
    L_max: int = DEFAULT_L_MAX,
) -> list[LightCandidate]:
    """Lightweight pool enumeration — same admission rule as build_candidate_arrays,
    but no patch/feature fill. Returns only real (non-padded) candidates."""
    fwd_dot, rc_dot = dot_plot(nc, flank)
    flank_len = len(flank)
    out: list[LightCandidate] = []
    for orient in DEFAULT_ORIENTATIONS:
        dot = fwd_dot if orient == "fwd" else rc_dot
        for L in range(L_min, L_max + 1):
            win = windowed_matches(dot, L)
            if win.size == 0:
                continue
            k = min(top_k, win.size)
            flat = win.flatten()
            top_idx = np.argpartition(-flat, k - 1)[:k]
            top_idx = top_idx[np.argsort(-flat[top_idx])]
            W_win = win.shape[1]
            for idx_ in top_idx:
                idx = int(idx_)
                nc_start = idx // W_win
                col = idx % W_win
                if orient == "fwd":
                    flank_start = col
                else:
                    flank_start = flank_len - L - col
                out.append(LightCandidate(orient, L, nc_start, flank_start, int(flat[idx])))
    return out


def _flank_iou(cand_start: int, cand_L: int, planted_start: int, planted_end: int) -> float:
    """IoU between candidate's flank window [cand_start, cand_start+cand_L)
    and planted flank region [planted_start, planted_end)."""
    a0, a1 = cand_start, cand_start + cand_L
    b0, b1 = planted_start, planted_end
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def measure_site(record: dict, iou_thresh: float = 0.5) -> dict:
    """Compute recovery metrics for one V5 site record."""
    lab = record["labels"]
    inp = record["inputs"]
    arch = lab["arch"]
    active_idx = lab["active_noncoding_index"]
    nc = inp["noncoding_regions"][active_idx]
    flank = inp["flank"]

    L = arch["L"]
    # V5 arch.orient ∈ {'fwd', 'rev'}; preprocess pool uses ∈ {'fwd', 'rc'}.
    # A 'rev' planted target reads on the flank as the reverse-complement of
    # guide — the matching pool candidate is the one at orient='rc'.
    orient = "rc" if arch["orient"] == "rev" else arch["orient"]
    is_split = arch["is_split"]
    mm_conc = arch["mm_concentration"]
    planted_m = lab["planted_m"]
    guide_nc_start, guide_nc_end = lab["guide_span_in_active_noncoding"]

    # Planted flank positions
    A_start = lab["planted_start"]
    A_end = lab["planted_A_end"]
    B_start = lab["planted_B_start"]
    B_end = lab["planted_B_end"]

    cands = enumerate_pool(nc, flank)

    # Filter to planted orient
    planted_cands = [c for c in cands if c.orient == orient]

    def _hits(planted_a: tuple[int, int], planted_b: tuple[int, int] | None):
        """Return dict of hit flags for a given planted flank region (or two halves)."""

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
                "L_full_iou": L_full_iou,
                "subL_iou": subL_iou,
                "any_L_iou": any_L_iou,
            }

        h_a = hits_region(*planted_a)
        if planted_b is None:
            return h_a
        h_b = hits_region(*planted_b)
        # For split-mode: hit means EITHER half is recovered
        return {
            "strict": h_a["strict"] or h_b["strict"],
            "L_full": h_a["L_full"] or h_b["L_full"],
            "subL": h_a["subL"] or h_b["subL"],
            "any_L": h_a["any_L"] or h_b["any_L"],
            "L_full_iou": max(h_a["L_full_iou"], h_b["L_full_iou"]),
            "subL_iou": max(h_a["subL_iou"], h_b["subL_iou"]),
            "any_L_iou": max(h_a["any_L_iou"], h_b["any_L_iou"]),
        }

    if is_split:
        hits = _hits((A_start, A_end), (B_start, B_end))
    else:
        hits = _hits((A_start, B_end), None)

    return {
        "transposase_id": record.get("transposase_id"),
        "site_id": record.get("site_id"),
        "L": L,
        "planted_m": planted_m,
        "mm_conc": mm_conc,
        "is_split": bool(is_split),
        "orient": orient,
        "n_pool": len(cands),
        **{k: hits[k] for k in ("strict", "L_full", "subL", "any_L",
                                 "L_full_iou", "subL_iou", "any_L_iou")},
    }


def _worker(line: str) -> dict:
    r = json.loads(line)
    return measure_site(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-jsonl", required=True)
    ap.add_argument("--out", required=True, help="output JSONL of per-site metrics")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--chunksize", type=int, default=64)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    print(f"[pool-recovery] input: {args.v5_jsonl}")
    print(f"[pool-recovery] workers: {args.workers}")
    t0 = time.time()

    with open(args.v5_jsonl) as fin, open(args.out, "w") as fout:
        lines: Iterable[str] = fin
        if args.limit > 0:
            lines = (l for i, l in enumerate(fin) if i < args.limit)

        if args.workers <= 1:
            for i, line in enumerate(lines):
                rec = _worker(line)
                fout.write(json.dumps(rec) + "\n")
                if (i + 1) % 5000 == 0:
                    dt = time.time() - t0
                    print(f"  [{i+1}] {dt:.1f}s  ({dt/(i+1)*1000:.1f} ms/site)")
        else:
            with mp.Pool(args.workers) as pool:
                for i, rec in enumerate(pool.imap_unordered(_worker, lines, chunksize=args.chunksize)):
                    fout.write(json.dumps(rec) + "\n")
                    if (i + 1) % 5000 == 0:
                        dt = time.time() - t0
                        print(f"  [{i+1}] {dt:.1f}s  ({dt/(i+1)*1000:.1f} ms/site amortized)")

    dt = time.time() - t0
    print(f"[pool-recovery] done in {dt:.1f}s → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
