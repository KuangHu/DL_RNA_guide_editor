"""V5 generator small-batch acceptance run (500 bags).

Runs:
  - Tests 1 / 2 / 3 (from generator_spec, adapted for mixed-L bags)
  - Test 4: length-stratified — partition bags by nc_len quartile,
            check Tests 1-2 pass in every quartile.
  - Split-vs-contig stratified m_at_planted (distinguishes "degraded
    but detectable" m>=6 from "below detection limit" m<6).
  - Guide-length / nc-length ratio distribution.
  - Leakage probe: single-variate AUROC on (nc_len, GC%, mean(dG_u1),
    std(dG_u1), ensemble_energy) for active vs inactive ncRNAs.

500 bags at ~1.3 s per bag (N_nc ~ 2, 2-3 folds each) = ~10 min serial.
"""
from __future__ import annotations

import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.generator_v5.bag_v2 import build_bag, load_flank_pool
from scripts.generator_v5.difficulty import load_or_build_rate_table

N_BAGS = 500
SEED = 0

# Acceptance thresholds (from spec).
TEST1_RATE_LO_FLOOR = 0.10
TEST1_RATE_LO_MAX = 0.05
TEST1_RATE_MED_LO = 0.19
TEST1_RATE_MED_HI = 0.24
TEST1_ZERO_MASS_AT_ABOVE_TARGET = 0.02
TEST2_SOLE_MAX_MAX = 0.01
TEST2_MEDIAN_MIN = 2
TEST3_MED_LO = 0.80
TEST3_MED_HI = 0.90
TEST3_IQR_MIN = 0.10


def gc_percent(seq: str) -> float:
    L = len(seq)
    if L == 0:
        return 0.0
    return (seq.count("G") + seq.count("C")) / L * 100.0


def auroc(x_pos: np.ndarray, x_neg: np.ndarray) -> float:
    if len(x_pos) == 0 or len(x_neg) == 0:
        return float("nan")
    from sklearn.metrics import roc_auc_score
    y = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))])
    x = np.concatenate([x_pos, x_neg])
    return float(roc_auc_score(y, x))


def main() -> int:
    print(f"[v5 500] loading rate table + flank pool")
    tbl = load_or_build_rate_table(rebuild=False)
    fl = load_flank_pool()
    rng = random.Random(SEED)

    print(f"[v5 500] generating {N_BAGS} bags")
    t0 = time.perf_counter()
    bags = []
    skips = 0
    for k in range(N_BAGS):
        b = build_bag(f"bag_{k:04d}", rng, fl, tbl)
        if b is None:
            skips += 1
            continue
        bags.append(b)
        if (k + 1) % 50 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{k+1}/{N_BAGS}] {len(bags)} bags kept, "
                  f"{dt:.1f}s ({dt/(k+1)*1000:.0f} ms/bag)")
    dt = time.perf_counter() - t0
    print(f"[v5 500] done in {dt:.1f}s ({dt/max(N_BAGS,1)*1000:.0f} ms/bag), skips={skips}")

    # Collect per-site records
    rec = []
    for b in bags:
        for s in b.sites:
            rec.append({
                "L":              b.difficulty.L,
                "nc_len":         b.difficulty.nc_len,
                "planted_m":      b.difficulty.planted_m,
                "target_m":       b.difficulty.target_m,
                "is_split":       b.architecture.is_split,
                "split_gap":      b.architecture.split_gap,
                "is_reversed":    b.architecture.is_reversed_target,
                "n_nc":           b.architecture.n_nc,
                "n_positions":    b.difficulty.nc_len - b.difficulty.L + 1,
                # Test 1's competitor_count: positions with m_max >= this
                # site's planted_m (NOT the fixed m=8 downstream threshold).
                "competitor_count": s.competitor_count_at_planted_m,
                "m_at_planted":   s.m_at_planted,
                "planted_start_pct": b.planted_start_on_nc / max(b.difficulty.nc_len - b.difficulty.L, 1),
            })

    n = len(rec)
    print(f"[v5 500] {n} sites collected")
    if n == 0:
        return 1

    L_arr        = np.array([r["L"]              for r in rec])
    nc_len_arr   = np.array([r["nc_len"]         for r in rec])
    planted_m    = np.array([r["planted_m"]      for r in rec])
    target_m     = np.array([r["target_m"]       for r in rec])
    n_pos_arr    = np.array([r["n_positions"]    for r in rec])
    comp_count   = np.array([r["competitor_count"] for r in rec])
    m_at_plant   = np.array([r["m_at_planted"]   for r in rec])
    is_split_arr = np.array([r["is_split"]       for r in rec])

    rate = comp_count / np.maximum(n_pos_arr, 1)

    def _pass(v, tgt, op):
        if op == "<": return v <= tgt
        if op == ">=": return v >= tgt
        if op == "in": return tgt[0] <= v <= tgt[1]
        return False

    def _report(label, m_at_target_mask, comp_count_all, rate_all, planted_m_all,
                target_m_all, n_bags_here):
        print(f"\n---- {label} (n_bags={n_bags_here}) ----")
        below_floor = float((rate_all < TEST1_RATE_LO_FLOOR).mean())
        rate_at_target = rate_all[m_at_target_mask]
        rate_med = float(np.median(rate_at_target)) if len(rate_at_target) else float("nan")
        above_target = float((planted_m_all > target_m_all).mean())
        sole_max = float((comp_count_all == 1).mean())
        c_med = float(np.median(comp_count_all))

        rows = [
            (f"Test1a  fraction(rate < {TEST1_RATE_LO_FLOOR})",
                below_floor, TEST1_RATE_LO_MAX, "<"),
            (f"Test1b  median(rate) at planted_m=target_m",
                rate_med, (TEST1_RATE_MED_LO, TEST1_RATE_MED_HI), "in"),
            (f"Test1c  P(planted_m > target_m)",
                above_target, TEST1_ZERO_MASS_AT_ABOVE_TARGET, "<"),
            (f"Test2a  P(competitor_count == 1)",
                sole_max, TEST2_SOLE_MAX_MAX, "<"),
            (f"Test2b  median(competitor_count)",
                c_med, TEST2_MEDIAN_MIN, ">="),
        ]
        all_ok = True
        for name, val, tgt, op in rows:
            ok = _pass(val, tgt, op)
            all_ok &= ok
            mark = "✓" if ok else "✗"
            val_s = f"{val:.4f}" if isinstance(val, float) else str(val)
            tgt_s = f"{op} {tgt}"
            print(f"  {mark} {name}: {val_s} (target: {tgt_s})")
        return all_ok

    # === Overall Tests 1 + 2 ===
    print("\n=== Overall Test 1/2 (all bags, mixed L) ===")
    mask_at_target = (planted_m == target_m)
    ok_all = _report("all bags", mask_at_target, comp_count, rate,
                       planted_m, target_m, len(bags))

    # === Test 4a: L-stratified ===
    print("\n=== Test 4a: per-L stratification (target_m per L is what actually varies) ===")
    print(f"  {'L':>3s} {'target_m':>9s} {'n_sites':>8s} "
          f"{'rate_med':>9s} {'rate_p25':>9s} {'rate_p75':>9s} {'below_0.10':>11s}")
    for L in sorted(set(L_arr.tolist())):
        mask_L = (L_arr == L)
        n_L = int(mask_L.sum())
        if n_L == 0:
            continue
        rate_L = rate[mask_L]
        # target_m for this L (from the bag's difficulty)
        tm_L = int(np.median(target_m[mask_L]))
        below = float((rate_L < TEST1_RATE_LO_FLOOR).mean())
        print(f"  {L:>3d} {tm_L:>9d} {n_L:>8d} "
              f"{float(np.median(rate_L)):>9.4f} "
              f"{float(np.percentile(rate_L, 25)):>9.4f} "
              f"{float(np.percentile(rate_L, 75)):>9.4f} "
              f"{below:>11.4f}")

    # For reference: also report rate stratified by (L, planted_m) fine-grain
    print()
    print(f"  Per (L, planted_m) fine-grain (only planted_m == target_m rows):")
    for L in sorted(set(L_arr.tolist())):
        mask = (L_arr == L) & (planted_m == target_m)
        if mask.sum() < 5:
            continue
        r_med = float(np.median(rate[mask]))
        pm = int(np.median(planted_m[mask]))
        print(f"    L={L}, planted_m={pm}: n={int(mask.sum())}, rate_med={r_med:.4f}")

    # === Test 4: length-stratified quartiles ===
    print("\n=== Test 4: nc_len quartile stratification ===")
    q = np.quantile(nc_len_arr, [0, 0.25, 0.5, 0.75, 1.0])
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        mask = (nc_len_arr >= lo) & (nc_len_arr <= hi if i == 3 else nc_len_arr < hi)
        if mask.sum() < 10:
            continue
        # Slice mask_at_target to match the sub-arrays' size.
        sub_at_target = mask_at_target[mask]
        _report(f"nc_len Q{i+1} [{int(lo)}, {int(hi)}]",
                sub_at_target, comp_count[mask], rate[mask],
                planted_m[mask], target_m[mask], int(mask.sum()))

    # === Test 3 ===
    print("\n=== Test 3: guide-window P_ss percentile ===")
    # Each bag has one placement; recompute percentile from features.
    percentiles = []
    for b in bags:
        p_ss = b.ncrna_features[b.architecture.active_nc_index].p_ss
        L = b.difficulty.L
        cs = np.concatenate(([0.0], np.cumsum(p_ss, dtype=np.float64)))
        p_win = (cs[L:] - cs[:-L]) / L
        gp = p_win[b.planted_start_on_nc]
        pct = float((p_win < gp).mean())
        percentiles.append(pct)
    percentiles = np.array(percentiles)
    med = float(np.median(percentiles))
    q25 = float(np.percentile(percentiles, 25))
    q75 = float(np.percentile(percentiles, 75))
    iqr = q75 - q25
    check_med = TEST3_MED_LO <= med <= TEST3_MED_HI
    check_iqr = iqr >= TEST3_IQR_MIN
    print(f"  Test3a  median %ile in [{TEST3_MED_LO}, {TEST3_MED_HI}]: "
          f"{med:.3f} -> {'PASS' if check_med else 'FAIL'}")
    print(f"  Test3b  IQR >= {TEST3_IQR_MIN}: {iqr:.3f} -> "
          f"{'PASS' if check_iqr else 'FAIL'}")

    # === Split vs contig stratified m_at_planted ===
    print("\n=== Split vs contig m_at_planted ===")
    for label, mask in [("contiguous", ~is_split_arr), ("split", is_split_arr)]:
        m = m_at_plant[mask]
        if len(m) == 0:
            continue
        detectable = float((m >= 6).mean())
        undetectable = float((m < 6).mean())
        print(f"  {label} (n={len(m)}): m_at_planted "
              f"median={float(np.median(m)):.1f}, "
              f"p25={float(np.percentile(m, 25)):.1f}, "
              f"p75={float(np.percentile(m, 75)):.1f} | "
              f"P(m>=6)={detectable:.3f}, P(m<6)={undetectable:.3f}")

    # === Guide/nc ratio distribution ===
    ratios = L_arr / nc_len_arr
    print("\n=== guide-length / nc-length ratio ===")
    print(f"  min={ratios.min():.3f}, "
          f"p10={float(np.percentile(ratios, 10)):.3f}, "
          f"median={float(np.median(ratios)):.3f}, "
          f"p90={float(np.percentile(ratios, 90)):.3f}, "
          f"max={ratios.max():.3f}")
    print(f"  real anchor range: T-WT 11/177 = 0.062, "
          f"seekRNA 13/58 (mature) = 0.224")

    # === Leakage probe ===
    print("\n=== Leakage probe: active vs inactive ncRNAs ===")
    active_len, active_gc, active_meanU1, active_stdU1, active_ee = [], [], [], [], []
    inactive_len, inactive_gc, inactive_meanU1, inactive_stdU1, inactive_ee = [], [], [], [], []
    for b in bags:
        for i, (nc, feats) in enumerate(zip(b.ncrna_sequences, b.ncrna_features)):
            length = len(nc)
            gc = gc_percent(nc)
            m_u1 = float(feats.dG_open_u1.mean())
            s_u1 = float(feats.dG_open_u1.std())
            ee   = float(feats.ensemble_energy)
            if i == b.active_nc_index:
                active_len.append(length); active_gc.append(gc)
                active_meanU1.append(m_u1); active_stdU1.append(s_u1); active_ee.append(ee)
            else:
                inactive_len.append(length); inactive_gc.append(gc)
                inactive_meanU1.append(m_u1); inactive_stdU1.append(s_u1); inactive_ee.append(ee)

    for name, ap, an in [
        ("nc_len",           active_len,    inactive_len),
        ("GC%",              active_gc,     inactive_gc),
        ("mean(dG_open_u1)", active_meanU1, inactive_meanU1),
        ("std(dG_open_u1)",  active_stdU1,  inactive_stdU1),
        ("ensemble_energy",  active_ee,     inactive_ee),
    ]:
        a = np.array(ap); b = np.array(an)
        au = auroc(a, b)
        au_leak = max(au, 1.0 - au)   # |direction-invariant| leakage
        mark = "✓" if au_leak <= 0.55 else "✗"
        print(f"  {mark} {name:<20s}: AUROC={au:.3f}  "
              f"(|direction-invariant|={au_leak:.3f}; leakage if > 0.55)")

    print()
    print("[v5 500] all reports emitted; check per-section for PASS/FAIL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
