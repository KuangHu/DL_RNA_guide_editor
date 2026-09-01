"""Unit tests for scripts.generator_v5.difficulty.

Checks:
  A. Rate table sanity — empirical vs analytic on the T-WT anchor:
     (L=11, m=8, n_starts=110, p_hat measured) rate should be near 0.21.
     (L=14, m=10) equivalent identity ratio -> similar rate.
     Analytic form kept as sanity; empirical is authoritative.
  B. Rate monotonicity: rate(L, m+1) < rate(L, m) for every (L, m).
  C. target_m per L consistency: target_m(L=11) == 8 (T-WT baseline).
  D. sample_L is uniform on {11,12,13,14} on N draws (chi-squared).
  E. sample_planted_m marginal: 86% at target_m, 10% at target_m-1,
     4% at target_m-2, 0% anywhere else. Zero mass above target_m.
  F. sample_nc_len returns integers in [max(70, L+1), 300]; empirical
     mean approaches uniform mean 185.
  G. Full sample_difficulty round-trip: 1000 bags, report the joint
     distribution of (L, planted_m).
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generator_v5.difficulty import (
    DEFAULT_L_CHOICES, DEFAULT_TARGET_RATE, DEFAULT_PLANTED_M_TAIL,
    RateTable, analytic_target_m_for_L, build_rate_table,
    sample_L, sample_nc_len, sample_planted_m, sample_difficulty,
)


def main() -> int:
    passed = True

    def check(name: str, cond: bool, why: str = ""):
        nonlocal passed
        mark = "  OK  " if cond else "  FAIL"
        print(f"{mark}  {name}{'  — ' + why if why else ''}")
        if not cond:
            passed = False

    # === A: build rate table, verify T-WT-anchor rate ===
    print("[difficulty] building rate table on 200 probes (small for test)...")
    import time
    t0 = time.perf_counter()
    tbl = build_rate_table(n_probe=200)
    print(f"  built in {time.perf_counter() - t0:.1f}s")
    print(f"  observed p_hat (from flank pool) = {tbl.p_hat:.4f}")
    print(f"  flank pool size = {tbl.flank_pool_size}")
    print()
    print(f"  {'L':>3s} {'m':>3s} {'rate':>7s} {'analytic':>10s}")
    for L in DEFAULT_L_CHOICES:
        for m in tbl.m_range:
            if (L, m) not in tbl.rate:
                continue
            an = None
            try:
                # analytic uses p_hat and n_flank_starts = flank_len - L + 1
                from scripts.generator_v5.difficulty import _analytic_rate_at
                an = _analytic_rate_at(L, m, 120 - L + 1, tbl.p_hat)
            except Exception:
                pass
            print(f"  {L:>3d} {m:>3d} {tbl.rate[(L, m)]:>7.4f} "
                  f"{an:>10.4f}" if an is not None else
                  f"  {L:>3d} {m:>3d} {tbl.rate[(L, m)]:>7.4f}")

    # A: T-WT anchor
    r_11_8 = tbl.rate.get((11, 8), float("nan"))
    check("rate(L=11, m=8) in [0.15, 0.30] (T-WT anchor)",
          0.15 <= r_11_8 <= 0.30,
          f"empirical rate = {r_11_8:.4f}")
    # 14-10 approximate T-WT identity ratio
    r_14_10 = tbl.rate.get((14, 10), float("nan"))
    check("rate(L=14, m=10) in [0.05, 0.30]",
          0.05 <= r_14_10 <= 0.30,
          f"rate = {r_14_10:.4f}")

    # B: monotonicity in m
    for L in DEFAULT_L_CHOICES:
        ms = sorted(m for m in tbl.m_range if (L, m) in tbl.rate)
        for i in range(len(ms) - 1):
            r1 = tbl.rate[(L, ms[i])]
            r2 = tbl.rate[(L, ms[i + 1])]
            if r2 > r1 + 1e-6:
                check(f"rate monotone at L={L}, m={ms[i]}->{ms[i+1]}",
                      False, f"{r1:.4f} -> {r2:.4f}")

    # C: target_m(L=11) == 8 (T-WT baseline)
    target_m_11 = tbl.target_m_for_L(11, DEFAULT_TARGET_RATE)
    check("target_m(L=11, rate=0.21) == 8 (T-WT baseline)",
          target_m_11 == 8, f"got {target_m_11}")
    for L in DEFAULT_L_CHOICES:
        tm = tbl.target_m_for_L(L, DEFAULT_TARGET_RATE)
        print(f"  target_m(L={L}, target_rate={DEFAULT_TARGET_RATE}) = {tm}   "
              f"(analytic sanity: {analytic_target_m_for_L(L, DEFAULT_TARGET_RATE)})")

    # D: sample_L uniform
    N = 10000
    rng = random.Random(0)
    counts_L = Counter(sample_L(rng) for _ in range(N))
    freqs = {L: counts_L[L] / N for L in DEFAULT_L_CHOICES}
    print()
    print(f"[D] sample_L freq on N={N}: {freqs}")
    # Chi-squared test at 4 categories, expected 0.25 each
    obs = np.array([counts_L[L] for L in DEFAULT_L_CHOICES])
    exp = np.array([N / len(DEFAULT_L_CHOICES)] * len(DEFAULT_L_CHOICES))
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    # df=3, critical at alpha=0.05 is ~7.815
    check("sample_L chi-squared uniform (chi2 < 7.815 at df=3)",
          chi2 < 7.815, f"chi2 = {chi2:.3f}")

    # E: sample_planted_m marginal (targeting m=8)
    counts_m = Counter(sample_planted_m(rng, 8) for _ in range(N))
    freqs_m = {m: counts_m[m] / N for m in sorted(counts_m)}
    print(f"[E] sample_planted_m(target=8) freq on N={N}: {freqs_m}")
    p0, p1, p2 = DEFAULT_PLANTED_M_TAIL
    check("planted_m mode at target (freq >= 0.83)",
          freqs_m.get(8, 0) >= p0 - 0.03,
          f"freq(8) = {freqs_m.get(8, 0):.3f}, expect {p0:.2f}")
    check("planted_m tail at target-1 (freq >= 0.08)",
          abs(freqs_m.get(7, 0) - p1) < 0.03,
          f"freq(7) = {freqs_m.get(7, 0):.3f}, expect {p1:.2f}")
    check("planted_m tail at target-2 (freq >= 0.02)",
          abs(freqs_m.get(6, 0) - p2) < 0.02,
          f"freq(6) = {freqs_m.get(6, 0):.3f}, expect {p2:.2f}")
    check("planted_m has ZERO mass above target",
          all(m <= 8 for m in counts_m),
          f"observed values: {sorted(counts_m)}")

    # F: sample_nc_len
    lens = [sample_nc_len(rng, 11) for _ in range(N)]
    check("sample_nc_len in [70, 300]",
          all(70 <= L <= 300 for L in lens))
    check("sample_nc_len mean near uniform mean 185",
          abs(np.mean(lens) - 185) < 3,
          f"observed mean = {np.mean(lens):.1f}")

    # G: full difficulty round-trip
    print()
    print("[G] sample_difficulty joint distribution (N=1000):")
    counts_j: Counter = Counter()
    lens_per_L: dict[int, list[int]] = {L: [] for L in DEFAULT_L_CHOICES}
    for _ in range(1000):
        d = sample_difficulty(rng, tbl)
        counts_j[(d.L, d.planted_m)] += 1
        lens_per_L[d.L].append(d.nc_len)
    print(f"  {'L':>3s} {'planted_m':>10s} {'count':>6s} {'nc_len med':>11s} {'ratio med':>10s}")
    for L in DEFAULT_L_CHOICES:
        for pm in sorted({k[1] for k in counts_j if k[0] == L}):
            n = counts_j[(L, pm)]
            med_nc = int(np.median(lens_per_L[L])) if lens_per_L[L] else 0
            ratio = L / max(med_nc, 1)
            print(f"  {L:>3d} {pm:>10d} {n:>6d} {med_nc:>11d} {ratio * 100:>9.1f}%")

    print()
    print("[difficulty] " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
