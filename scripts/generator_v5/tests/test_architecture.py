"""Per-axis unit tests for architecture.py.

Axes and their invariants:

  A/B composition:
    A + B == L
    A in [5, 8], B in [3, 8]
    For L=11: A can be {5,6,7,8}; for L=14: A can be {6,7,8}
    Marginal: on N=1e4 draws at L=11, A distribution near uniform on
    valid range.

  is_split:
    On N=1e4 draws, freq near DEFAULT_SPLIT_PROB (0.18) within +/- 0.02.
    Split-gap always in [2, 6] when is_split is True.

  is_reversed:
    On N=1e4 draws, freq near 0.5 within +/- 0.02.

  N_nc + active_nc_index:
    N_nc uniform on {1, 2, 3}.
    active_nc_index in [0, N_nc); marginal uniform on {0, 1, 2} for
    N_nc=3.

  TSD:
    tsd_width in {0, 2, 5, 8, 9, 12}.
    tsd_relation == "none" iff tsd_width == 0.

  ncr_pos_rel_orf:
    Uniform on {upstream, downstream, inline}.

  check_5p_stem_loop:
    On known-good ("(((...))).") returns True.
    On known-bad ("..........") returns False.
"""
from __future__ import annotations

import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generator_v5.architecture import (
    DEFAULT_A_LO, DEFAULT_A_HI, DEFAULT_B_LO, DEFAULT_B_HI,
    DEFAULT_SPLIT_PROB, DEFAULT_SPLIT_GAP_LO, DEFAULT_SPLIT_GAP_HI,
    DEFAULT_N_NC_CHOICES, DEFAULT_TSD_WIDTHS,
    check_5p_stem_loop, sample_architecture, sample_guide_composition,
    sample_is_split, sample_split_gap, sample_is_reversed,
    sample_n_nc, sample_active_index, sample_tsd, sample_ncr_pos_rel_orf,
)


def main() -> int:
    passed = True

    def check(name: str, cond: bool, why: str = ""):
        nonlocal passed
        mark = "  OK  " if cond else "  FAIL"
        print(f"{mark}  {name}{'  — ' + why if why else ''}")
        if not cond:
            passed = False

    N = 10000
    rng = random.Random(0)

    # === Guide composition ===
    print("[A/B guide composition]")
    A_counts_11: Counter = Counter()
    A_counts_14: Counter = Counter()
    all_ok_sum = True; all_ok_A11 = True; all_ok_B11 = True
    for _ in range(N):
        g11 = sample_guide_composition(rng, 11)
        g14 = sample_guide_composition(rng, 14)
        if g11.A + g11.B != 11:
            all_ok_sum = False
        if not (DEFAULT_A_LO <= g11.A <= DEFAULT_A_HI):
            all_ok_A11 = False
        if not (DEFAULT_B_LO <= g11.B <= DEFAULT_B_HI):
            all_ok_B11 = False
        A_counts_11[g11.A] += 1
        A_counts_14[g14.A] += 1
    check(f"L=11: A + B == L holds on all {N} draws", all_ok_sum)
    check(f"L=11: A in [5, 8] on all {N} draws", all_ok_A11)
    check(f"L=11: B in [3, 8] on all {N} draws", all_ok_B11)
    check("L=11: A in {5,6,7,8}", set(A_counts_11) == {5, 6, 7, 8},
          f"observed: {sorted(A_counts_11)}")
    check("L=14: A in {6,7,8}", set(A_counts_14) == {6, 7, 8},
          f"observed: {sorted(A_counts_14)}")
    # Marginal uniformity on the valid range (chi-squared)
    obs = np.array([A_counts_11[a] for a in sorted(A_counts_11)])
    exp = np.full(len(obs), N / len(obs))
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    print(f"  L=11 A distribution: {dict(A_counts_11)} chi2={chi2:.3f}")
    check("L=11 A uniform (chi2 < 7.815)", chi2 < 7.815)

    # === is_split + split_gap ===
    print("\n[is_split + split_gap]")
    split_count = 0
    gaps = []
    for _ in range(N):
        s = sample_is_split(rng)
        if s:
            split_count += 1
            gaps.append(sample_split_gap(rng))
    freq = split_count / N
    print(f"  is_split freq (target {DEFAULT_SPLIT_PROB}): {freq:.3f}")
    check("is_split freq near target (|Δ| < 0.02)",
          abs(freq - DEFAULT_SPLIT_PROB) < 0.02)
    check("split_gap in [2, 6]",
          all(DEFAULT_SPLIT_GAP_LO <= g <= DEFAULT_SPLIT_GAP_HI for g in gaps))

    # === is_reversed ===
    print("\n[is_reversed]")
    rev_count = sum(sample_is_reversed(rng) for _ in range(N))
    print(f"  freq: {rev_count / N:.3f}")
    check("is_reversed freq near 0.5 (|Δ| < 0.02)",
          abs(rev_count / N - 0.5) < 0.02)

    # === N_nc + active_index ===
    print("\n[N_nc + active_index]")
    n_nc_counts = Counter(sample_n_nc(rng) for _ in range(N))
    print(f"  N_nc distribution: {dict(n_nc_counts)}")
    obs = np.array([n_nc_counts[k] for k in DEFAULT_N_NC_CHOICES])
    exp = np.full(len(obs), N / len(obs))
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    check(f"N_nc uniform on {DEFAULT_N_NC_CHOICES} (chi2 < 5.99, df=2)",
          chi2 < 5.99, f"chi2 = {chi2:.3f}")
    # Active idx for n_nc=3
    aids = [sample_active_index(rng, 3) for _ in range(N)]
    aid_counts = Counter(aids)
    obs = np.array([aid_counts[k] for k in [0, 1, 2]])
    exp = np.full(3, N / 3)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    check(f"active_index uniform on {{0,1,2}} for N_nc=3",
          chi2 < 5.99, f"chi2 = {chi2:.3f}, distribution={dict(aid_counts)}")

    # === TSD ===
    print("\n[TSD]")
    widths: Counter = Counter()
    relations: Counter = Counter()
    for _ in range(N):
        t = sample_tsd(rng)
        widths[t.width] += 1
        relations[t.relation] += 1
        # Invariant check
        if t.width == 0 and t.relation != "none":
            check("tsd_width==0 implies relation==none", False,
                  f"width=0 got relation={t.relation}")
        if t.width > 0 and t.relation == "none":
            check("tsd_width>0 implies relation!=none", False,
                  f"width={t.width} got relation=none")
    print(f"  tsd_width distribution: {dict(widths)}")
    print(f"  tsd_relation distribution: {dict(relations)}")
    check("tsd_width uniform (chi2 df=5)",
          float(sum((widths[w] - N / 6) ** 2 / (N / 6) for w in DEFAULT_TSD_WIDTHS)) < 11.07)

    # === NCR position ===
    print("\n[NCR position]")
    ncr_counts = Counter(sample_ncr_pos_rel_orf(rng) for _ in range(N))
    print(f"  ncr_pos distribution: {dict(ncr_counts)}")
    obs = np.array([ncr_counts[k] for k in ("upstream", "downstream", "inline")])
    exp = np.full(3, N / 3)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    check("ncr_pos uniform (chi2 < 5.99, df=2)", chi2 < 5.99,
          f"chi2 = {chi2:.3f}")

    # === check_5p_stem_loop ===
    print("\n[check_5p_stem_loop]")
    good = "(((....))).........."
    bad = "...................."
    marginal = "(.((....)))...(((...)))........"
    check("5p SL positive on '(((....))).......'",
          check_5p_stem_loop(good) is True)
    check("5p SL negative on all-dots",
          check_5p_stem_loop(bad) is False)
    check("5p SL positive on nested structure w/ full brackets in window",
          check_5p_stem_loop(marginal) is True)

    # === Full sample_architecture round-trip ===
    print("\n[sample_architecture round-trip (N=1000)]")
    counts_arch = Counter()
    for _ in range(1000):
        L = rng.choice([11, 12, 13, 14])
        arch = sample_architecture(rng, L)
        counts_arch[("split", arch.is_split)] += 1
        counts_arch[("n_nc", arch.n_nc)] += 1
    print(f"  is_split: {counts_arch[('split', True)]} true, "
          f"{counts_arch[('split', False)]} false")
    for k in [1, 2, 3]:
        print(f"  n_nc={k}: {counts_arch[('n_nc', k)]}")

    print()
    print("[architecture] " + ("PASS" if passed else "FAIL"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
