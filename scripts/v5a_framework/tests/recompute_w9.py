"""W9 recomputation under new metric semantics.

The historical Channel A doc says "87% at exact gold_nc" on 23 detections.
That number was computed under gold-aware "closest to gold" tie-break.
Under the framework's gold-blind centroid + explicit tolerance, exact-hit
becomes four numbers (2 tolerances x 2 denominators) plus the 3-tuple
resolution (plateau_width, contains_gold, centroid_dist).

This script prints all of them so docs/channel_a.md can quote the honest
set. One resolution report per detected Tnp (peaks in the same plateau
share the same primary; independent detection = 22, not 23).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from v5a_framework.match_table import load as load_mt
from v5a_framework.variant import spec_m_threshold_L11, run_variant
from v5a_framework.metrics import ResolutionReport, summarize_resolution


SHARD_DIR = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"


def _plateau_resolution(tnp_id: str, peaks: list, gold_nc: int) -> ResolutionReport:
    """Extract the top-S plateau; compute width, centroid, contains_gold, dist."""
    max_S = max(pk.S_all for pk in peaks)
    plateau_positions = sorted({pk.position for pk in peaks if pk.S_all == max_S})
    width = len(plateau_positions)
    centroid = float(np.mean(plateau_positions))
    contains_gold = (gold_nc in plateau_positions)
    dist = abs(centroid - gold_nc)
    return ResolutionReport(
        tnp_id=tnp_id, plateau_width=width, contains_gold=contains_gold,
        centroid_dist=dist, primary_position=centroid, gold_nc=gold_nc,
    )


def main() -> None:
    mt = load_mt(SHARD_DIR)
    spec = spec_m_threshold_L11(m=8, tau=0, S=5)
    peaks_by_tnp = run_variant(mt, spec)

    reports = []
    per_tnp_rows = []
    n_tnps = len(mt.tnp_ids)
    for tnp_id in mt.tnp_ids:
        pks = peaks_by_tnp.get(tnp_id, [])
        if not pks:
            continue
        gold_nc = mt.tnps[tnp_id].sites[0].gold_nc
        r = _plateau_resolution(tnp_id, pks, gold_nc)
        reports.append(r)
        per_tnp_rows.append((tnp_id, len(pks), r.plateau_width, r.primary_position,
                              r.gold_nc, r.centroid_dist, r.contains_gold))

    print(f"# W9 recomputation on {SHARD_DIR}")
    print(f"# spec: fixed_L11_m8, tau=0, S=5, tsd_handling=off, peak_min_dist=5")
    print(f"# denominators: N_tnp_total={n_tnps}, N_detected_tnps={len(reports)}")
    print()
    print(f"{'tnp_id':<50} {'n_peaks':>7} {'plateau_w':>9} {'centroid':>8} {'gold':>5} {'dist':>5} {'contains_gold':>13}")
    for row in per_tnp_rows:
        tnp_id, n_pks, width, centroid, gold, dist, contains = row
        print(f"{tnp_id[:50]:<50} {n_pks:>7} {width:>9} {centroid:>8.1f} {gold:>5} {dist:>5.1f} "
              f"{'YES' if contains else 'no':>13}")

    print()
    print("=== Aggregate resolution (denominator = N_detected_tnps) ===")
    s = summarize_resolution(reports)
    for k, v in s.items():
        print(f"  {k:24s} = {v}")

    print()
    print("=== Four exact-hit numbers ===")
    n_det = len(reports)
    exact_eq0_dets = sum(1 for r in reports if r.centroid_dist == 0)
    exact_le1_dets = sum(1 for r in reports if r.centroid_dist <= 1)
    print(f"  exact_eq_0_of_dets = {exact_eq0_dets}/{n_det}"
          f" = {exact_eq0_dets/max(1,n_det):.4f}  (denominator: detected Tnps)")
    print(f"  exact_le_1_of_dets = {exact_le1_dets}/{n_det}"
          f" = {exact_le1_dets/max(1,n_det):.4f}  (denominator: detected Tnps)")
    print(f"  exact_eq_0_of_tnps = {exact_eq0_dets}/{n_tnps}"
          f" = {exact_eq0_dets/n_tnps:.4f}  (denominator: all Tnps)")
    print(f"  exact_le_1_of_tnps = {exact_le1_dets}/{n_tnps}"
          f" = {exact_le1_dets/n_tnps:.4f}  (denominator: all Tnps)")
    print()
    print("=== Comparison to historical '87% at exact gold_nc' ===")
    print(f"  Historical W9: 87% under gold-aware 'closest to gold' tie-break, denom = 23 peaks.")
    print(f"  New:           {exact_eq0_dets/max(1,n_det)*100:.1f}% under gold-blind centroid, denom = detected Tnps.")
    print(f"  New (le_1):    {exact_le1_dets/max(1,n_det)*100:.1f}% under gold-blind centroid, tolerance <= 1 nt.")


if __name__ == "__main__":
    main()
