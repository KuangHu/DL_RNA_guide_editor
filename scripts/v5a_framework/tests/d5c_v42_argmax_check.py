"""D5c — V4.2 D5b check: is planted guide the strongest per-site match on nc?

The direct empirical form of D5b's structural finding, applied to V4.2:
  If the planted guide's m at guide position IS the per-site argmax on nc,
  the task is EXTREMUM-solvable (find the strongest match). Any per-site
  extremum statistic will succeed.
  If planted is NOT argmax, task requires cross-site conjunction. Only
  S=k-like rules will detect. Per-site extremum statistics cap below
  actual signal.

Durrant baseline: planted argmax near gold ~= 0% (D5b showed gold_hits_median=0
on T-WT bags). So real IS110-like tasks are conjunction-only.

Anything above ~10% for a synthetic generator means the generator has
introduced a per-site extremum signal that doesn't exist in reality,
producing a systematically-easier task and a real synthetic->real gap
in extremum-based learning approaches.

This script measures V4.2 to check where it sits.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches


V42_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl"


def per_pos_m_max_pooled(nc: str, flank: str, L: int) -> np.ndarray:
    fwd_dot, rc_dot = dot_plot(nc, flank)
    w_fwd = windowed_matches(fwd_dot, L)
    w_rc = windowed_matches(rc_dot, L)
    if w_fwd.size == 0 or w_rc.size == 0:
        return np.zeros(0, dtype=np.int32)
    m_fwd = w_fwd.max(axis=1)
    m_rc = w_rc.max(axis=1)
    n = min(len(m_fwd), len(m_rc))
    return np.maximum(m_fwd[:n], m_rc[:n])


def main(n_sample: int = 2000) -> int:
    t0 = time.time()
    planted_is_argmax = 0
    planted_within_5 = 0
    m_at_planted = []
    m_at_argmax = []
    argmax_gaps = []
    n_ok = 0

    with open(V42_POS) as f:
        for line in f:
            d = json.loads(line)
            lab = d["labels"]
            if not lab.get("is_positive"):
                continue
            span = lab.get("guide_span_in_active_noncoding")
            L = lab.get("guide_length")
            if not span or not L:
                continue
            planted_start = int(span[0])
            a = lab.get("active_noncoding_index", 0) or 0
            ncs = d["inputs"].get("noncoding_regions", [])
            if a >= len(ncs):
                continue
            nc = ncs[a]
            flank = d["inputs"]["flank"]
            m_pooled = per_pos_m_max_pooled(nc, flank, L)
            if planted_start >= len(m_pooled):
                continue
            argmax = int(m_pooled.argmax())
            m_at_planted.append(int(m_pooled[planted_start]))
            m_at_argmax.append(int(m_pooled[argmax]))
            argmax_gaps.append(argmax - planted_start)
            if argmax == planted_start:
                planted_is_argmax += 1
            if abs(argmax - planted_start) <= 5:
                planted_within_5 += 1
            n_ok += 1
            if n_ok >= n_sample:
                break

    print(f"V4.2 D5b check on {n_ok} positive sites (wall {time.time()-t0:.1f}s)")
    print(f"  planted IS argmax exactly     : {planted_is_argmax}/{n_ok} = {planted_is_argmax/n_ok:.4f}")
    print(f"  planted within +/-5 of argmax : {planted_within_5}/{n_ok} = {planted_within_5/n_ok:.4f}")
    print(f"  m at planted (mean, med)      : {np.mean(m_at_planted):.2f}, {int(np.median(m_at_planted))}")
    print(f"  m at argmax (mean, med)       : {np.mean(m_at_argmax):.2f}, {int(np.median(m_at_argmax))}")
    print(f"  delta (argmax - planted)      : {np.mean(m_at_argmax) - np.mean(m_at_planted):.2f}")
    gaps = np.array(argmax_gaps)
    print(f"  argmax - planted position dist: mean={gaps.mean():.1f}, med={float(np.median(gaps)):.1f}")

    print()
    print("Comparison to Durrant (D5b T-WT):")
    print("  planted within +/-5 of argmax (gold_hits_median in [45,54]): ~0%")
    print("  V4.2 current                                                : "
          f"{planted_within_5/n_ok*100:.1f}%")
    print("  gap                                                          : "
          f"{planted_within_5/n_ok*100:.1f}x higher than target")

    threshold = 0.10
    if planted_within_5 / n_ok > threshold:
        print()
        print(f"  ==> V4.2 FAILS the D5b generator requirement (>{threshold*100:.0f}%)")
        print(f"      The planted guide is too often the argmax; generator produces an")
        print(f"      EASIER problem than reality; per-site extremum statistics learn")
        print(f"      to exploit this, causing synthetic->real gap on Durrant.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
