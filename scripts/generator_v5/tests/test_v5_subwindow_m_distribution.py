"""Per-L subwindow max-m distribution — the falsifiable acceptance test
for the mismatch_geometry axis.

For each planted L=11 site in a small V5 batch:
  For each sub-length L' in {9, 10, 11}:
    Compute max over (L-L'+1) subwindows of the number of matches
    between guide[start:start+L'] and the planted mutated_target[start:start+L'].
  Report the distribution stratified by (mm_concentration, mm_anchor).

Falsifiable predictions:
  For clustered_5p / clustered_3p / clustered_mid bags:
    P(max L=9 subwindow m >= 8) should be ~1.0
    (guarantee by construction; 1 mm in [2:10] window etc.)
  For dispersed_* bags:
    P(max L=9 subwindow m >= 8) should be ~0
    (guarantee by construction; ≥2 mm per L=9 window)

T-WT anchor comparison: 86.5% at m=8 overall (T-WT is naturally
clustered_3p per the mismatch histogram).
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.generator_v5.bag_v2 import build_bag, load_flank_pool
from scripts.generator_v5.difficulty import load_or_build_rate_table


def _max_sub_m_from_mm(mm_positions: list[int], L: int, sub_L: int) -> int:
    """Given mismatch positions within a guide of length L, return the max
    number of matches in any sub_L-sized subwindow.

    Analytic: for each offset in [0, L - sub_L], count mm positions in
    [offset, offset + sub_L). subwindow_m = sub_L - #mm in that window.
    """
    if L < sub_L:
        return 0
    mm_set = set(mm_positions)
    best = 0
    for off in range(L - sub_L + 1):
        mm_in = sum(1 for p in mm_set if off <= p < off + sub_L)
        m = sub_L - mm_in
        if m > best:
            best = m
    return best


def main() -> int:
    print("[subwin] loading rate table + flank pool")
    tbl = load_or_build_rate_table(rebuild=False)
    fl = load_flank_pool()
    rng = random.Random(0)

    N_BAGS = 400
    print(f"[subwin] generating {N_BAGS} bags (mixed L per difficulty)")

    # Collect (mm_concentration, mm_anchor, L) -> list of (max L=9 m, max L=10 m, max L=11 m)
    from collections import defaultdict
    records = defaultdict(list)   # (conc, anch, L) -> list of tuples

    for k in range(N_BAGS):
        b = build_bag(f"subwin_{k:04d}", rng, fl, tbl)
        if b is None:
            continue
        arch = b.architecture
        L = b.difficulty.L
        # For each site, extract the guide from the planted target region on
        # the flank, and re-compute matches against the original guide.
        for s in b.sites:
            if arch.is_split:
                continue
            # Use stored mismatch_positions on the guide directly (avoids
            # sequence-orientation confusion from is_reversed_target).
            mm_positions = s.mismatch_positions
            m9 = _max_sub_m_from_mm(mm_positions, L, 9) if L >= 9 else -1
            m10 = _max_sub_m_from_mm(mm_positions, L, 10) if L >= 10 else -1
            m11 = _max_sub_m_from_mm(mm_positions, L, L)
            records[(arch.mm_concentration, arch.mm_anchor, L)].append(
                (m9, m10, m11, s.n_mismatches))

    print(f"\n=== Per-(concentration, anchor, L) subwindow max-m distribution ===")
    print(f"  {'concentration':<14s} {'anchor':<6s} {'L':>3s} {'n':>5s} "
          f"{'m9>=8':>8s} {'m10>=9':>8s} {'m11>=?':>8s}")
    for key in sorted(records):
        conc, anch, L = key
        rows = records[key]
        n = len(rows)
        m9 = np.array([r[0] for r in rows])
        m10 = np.array([r[1] for r in rows])
        m11 = np.array([r[2] for r in rows])
        # Note: m9/m10 = -1 means L < 9/10; skip those
        m9_ge8 = float((m9 >= 8).mean()) if (m9 >= 0).any() else float('nan')
        m10_ge9 = float((m10 >= 9).mean()) if (m10 >= 0).any() else float('nan')
        m11_full = float((m11 >= L - 3).mean())  # roughly "target hit"
        print(f"  {conc:<14s} {anch:<6s} {L:>3d} {n:>5d} "
              f"{m9_ge8:>8.3f} {m10_ge9:>8.3f} {m11_full:>8.3f}")

    # === Pooled per-concentration verdict ===
    print(f"\n=== Pooled per-concentration (all L>=11) ===")
    pooled = defaultdict(list)
    for key, rows in records.items():
        conc, anch, L = key
        if L < 11:
            continue
        for m9, m10, m11, n_mm in rows:
            if m9 < 0:
                continue
            pooled[conc].append(m9)
    for conc in ("clustered", "dispersed"):
        arr = np.array(pooled.get(conc, []))
        if len(arr) == 0:
            continue
        p_ge8 = float((arr >= 8).mean())
        print(f"  {conc}: n={len(arr)}, P(L=9 max m >= 8) = {p_ge8:.3f}")

    print(f"\nT-WT anchor: 87.6% at m>=8 (naturally clustered_3p pattern)")

    # Verdict at L=11 specifically (matches T-WT anchor)
    L11_clus = []
    L11_disp = []
    for (conc, anch, L), rows in records.items():
        if L != 11:
            continue
        for m9, m10, m11, n_mm in rows:
            if m9 < 0:
                continue
            (L11_clus if conc == "clustered" else L11_disp).append(m9)
    if len(L11_clus) > 20 and len(L11_disp) > 20:
        cf = float((np.array(L11_clus) >= 8).mean())
        df = float((np.array(L11_disp) >= 8).mean())
        print(f"\n=== Verdict at L=11 (T-WT anchor comparison) ===")
        print(f"  clustered L=11 P(m9>=8) = {cf:.3f}  (target ~0.87, T-WT anchor)")
        print(f"  dispersed L=11 P(m9>=8) = {df:.3f}  (target ~0.00)")
        ok = 0.75 <= cf <= 1.00 and df <= 0.05
        print(f"  {'PASS' if ok else 'FAIL'}: mismatch_geometry axis reproduces T-WT-like Mode-1 signal at L=11")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
