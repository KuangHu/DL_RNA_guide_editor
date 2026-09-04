"""Diagnostic: for dispersed L=11 sites where L_full=True, what admitted
candidate is scoring the hit?

Two competing mechanisms:
  (a) planted (nc_start, flank_start) genuinely admitted at top-4 despite
      E[competitors] ~22-30 at m>=8 — small-probability lucky win.
  (b) a spurious (nc_start, flank_start) with higher m by chance lands on
      the same flank window (flank_iou >= 0.5 with planted). "L_full=True"
      would then be a coordinate-tolerance artifact, not gold recovery.

For 1000 dispersed L=11 sites with L_full=True:
  dump planted (nc_start, flank_start, m) vs admitted best-iou candidate
  (nc_start, flank_start, L, m). Report:
  - fraction with same flank_start (delta = 0)
  - fraction with same nc_start
  - fraction that are the exact planted slot (strict)
  - m distribution of admitted (should be higher than planted_m if (b) dominates)
  - n_flank_starts used at L=11 for these records
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.preprocess_pool_recovery.pool_recovery import (
    enumerate_pool,
    _flank_iou,
)


ROWS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/pool_recovery/rows_50k.jsonl"
V5 = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_50k.jsonl"


def main() -> int:
    # Find site_ids that are dispersed L=11 L_full=True
    target_sites: set[str] = set()
    with open(ROWS) as f:
        for line in f:
            r = json.loads(line)
            if (r["L"] == 11 and r["mm_conc"] == "dispersed"
                    and not r["is_split"] and r["L_full"]):
                target_sites.add(r["site_id"])
    print(f"[diag] target sites (dispersed L=11 L_full=True): {len(target_sites)}")

    # For each target site, re-run enumeration + find best-iou admitted
    stats = {
        "n": 0,
        "same_flank_start": 0,
        "same_nc_start": 0,
        "strict_gold": 0,
        "same_flank_diff_nc": 0,   # coordinate-artifact class
    }
    admitted_m_hist = Counter()
    flank_delta_hist = Counter()
    nc_delta_hist = Counter()
    n_flank_starts_all: list[int] = []
    n_ncstarts_all: list[int] = []
    m_greater_than_planted = 0

    MAX_DUMP = 5
    dumped = 0

    with open(V5) as f:
        for line in f:
            r = json.loads(line)
            sid = r.get("site_id")
            if sid not in target_sites:
                continue
            lab = r["labels"]
            inp = r["inputs"]
            arch = lab["arch"]
            active_idx = lab["active_noncoding_index"]
            nc = inp["noncoding_regions"][active_idx]
            flank = inp["flank"]
            L = arch["L"]  # 11
            # Map V5 arch.orient='rev' to preprocess pool candidate.orient='rc'
            orient = "rc" if arch["orient"] == "rev" else arch["orient"]
            planted_m = lab["planted_m"]
            guide_nc_start, _ = lab["guide_span_in_active_noncoding"]
            A_start = lab["planted_start"]
            B_end = lab["planted_B_end"]

            n_flank_starts_all.append(len(flank) - L + 1)
            n_ncstarts_all.append(len(nc) - L + 1)

            cands = enumerate_pool(nc, flank)
            # Filter to (planted orient, L)
            pool_L = [c for c in cands if c.orient == orient and c.L == L]
            if not pool_L:
                continue

            # Best admitted by flank_iou to planted
            best = max(
                pool_L,
                key=lambda c: _flank_iou(c.flank_start, c.L, A_start, B_end),
            )
            best_iou = _flank_iou(best.flank_start, best.L, A_start, B_end)
            if best_iou < 0.5:
                continue  # shouldn't happen since rows_50k marked L_full=True

            flank_delta = best.flank_start - A_start
            nc_delta = best.nc_start - guide_nc_start

            stats["n"] += 1
            if flank_delta == 0:
                stats["same_flank_start"] += 1
            if nc_delta == 0:
                stats["same_nc_start"] += 1
            if flank_delta == 0 and nc_delta == 0:
                stats["strict_gold"] += 1
            if flank_delta == 0 and nc_delta != 0:
                stats["same_flank_diff_nc"] += 1
            if best.matches > planted_m:
                m_greater_than_planted += 1
            admitted_m_hist[best.matches] += 1
            flank_delta_hist[flank_delta] += 1
            nc_delta_hist[nc_delta] += 1

            if dumped < MAX_DUMP:
                print(f"\n  site={sid} planted: (nc={guide_nc_start}, flank={A_start}, m={planted_m}, L={L})")
                print(f"    admitted: (nc={best.nc_start}, flank={best.flank_start}, m={best.matches}, L={best.L})  "
                      f"Δflank={flank_delta:+d} Δnc={nc_delta:+d} iou={best_iou:.3f}")
                dumped += 1

    n = stats["n"]
    print(f"\n=== dispersed L=11 L_full=True — admitted-candidate anatomy ===")
    print(f"n = {n}")
    print(f"  same_flank_start (Δflank=0):     {stats['same_flank_start']}/{n} = {stats['same_flank_start']/n:.3f}")
    print(f"  same_nc_start (Δnc=0):           {stats['same_nc_start']}/{n} = {stats['same_nc_start']/n:.3f}")
    print(f"  strict gold (both = 0):          {stats['strict_gold']}/{n} = {stats['strict_gold']/n:.3f}")
    print(f"  SAME flank + DIFF nc (coord-artifact class): {stats['same_flank_diff_nc']}/{n} = {stats['same_flank_diff_nc']/n:.3f}")
    print(f"  admitted m > planted_m:          {m_greater_than_planted}/{n} = {m_greater_than_planted/n:.3f}")

    print(f"\n=== admitted m distribution ===")
    for m in sorted(admitted_m_hist):
        print(f"  m={m}: {admitted_m_hist[m]:>5d} ({admitted_m_hist[m]/n*100:.1f}%)")

    print(f"\n=== Δflank_start distribution (top 8) ===")
    for delta, c in sorted(flank_delta_hist.items(), key=lambda x: -x[1])[:8]:
        print(f"  Δflank={delta:+d}: {c:>5d} ({c/n*100:.1f}%)")

    print(f"\n=== Δnc_start distribution (top 8) ===")
    for delta, c in sorted(nc_delta_hist.items(), key=lambda x: -x[1])[:8]:
        print(f"  Δnc={delta:+d}: {c:>5d} ({c/n*100:.1f}%)")

    import statistics as st
    print(f"\n=== pool axis sizes at L=11 ===")
    print(f"  n_flank_starts:  median={st.median(n_flank_starts_all):.0f} min={min(n_flank_starts_all)} max={max(n_flank_starts_all)}")
    print(f"  n_nc_starts:     median={st.median(n_ncstarts_all):.0f} min={min(n_ncstarts_all)} max={max(n_ncstarts_all)}")
    print(f"  total (nc × flank) combinations at L=11: ~{st.median(n_flank_starts_all)*st.median(n_ncstarts_all):.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
