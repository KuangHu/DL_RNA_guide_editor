"""Analyze Phase G calibration output.

Reads the details.tsv emitted by scripts/calibrate_v3_nulls.py and reports:

  1. Per-condition histogram of max_mean_m (best cluster per bag) across 5 bins.
  2. Per-condition histogram of max_n_support.
  3. Per-condition histogram of n_passing.
  4. Candidate thresholds for a one-sided null-rejection rule:
        choose min_max_mean_m so that <= target_null_pass of each null's bags
        would be reported as "passing" post-threshold.  Report the real
        recall at that threshold.
  5. Same for min_max_n_support.
  6. Suggested m_floor_report(L) per L based on real p25 of (best_m_at_top / L).

Usage:
  python scripts/analyze_v3_calib.py \
      --details logs/v3_calib_500_details.tsv \
      --target-null-pass 0.01
"""
from __future__ import annotations
import argparse
import csv
from collections import defaultdict
from pathlib import Path


CONDITIONS = ("real", "null1_dinuc", "null2_4mer", "null3_unpaired", "null4_rc_shuf")


def _pct(xs: list[float], p: float) -> float:
    if not xs: return 0.0
    xs = sorted(xs)
    k = int(len(xs) * p / 100)
    k = min(k, len(xs) - 1)
    return xs[k]


def _histo(xs: list[float], bins: list[tuple[float, float]]) -> list[int]:
    out = [0] * len(bins)
    for v in xs:
        for i, (lo, hi) in enumerate(bins):
            if lo <= v <= hi:
                out[i] += 1
                break
    return out


def _frac_above(xs: list[float], thresh: float) -> float:
    if not xs: return 0.0
    return sum(1 for v in xs if v >= thresh) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--details", type=Path, required=True)
    ap.add_argument("--target-null-pass", type=float, default=0.01,
                      help="max fraction of ANY null's bags allowed to pass "
                           "at the chosen threshold (default 0.01)")
    args = ap.parse_args()

    rows = []
    with args.details.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("err"): continue
            try:
                rows.append({
                    "bag_id": r["bag_id"], "cond": r["cond"],
                    "K_valid": int(r["K_valid"]),
                    "n_passing": int(r["n_passing"]),
                    "max_n_support": int(r["max_n_support"]),
                    "max_mean_m": float(r["max_mean_m"]),
                    "max_n_bg_groups": int(r["max_n_bg_groups"]),
                    "best_L": int(r["best_L"] or 0),
                    "best_m_at_top": int(r["best_m_at_top"] or 0),
                })
            except (ValueError, KeyError) as e:
                continue
    by_cond: dict[str, list[dict]] = defaultdict(list)
    for r in rows: by_cond[r["cond"]].append(r)
    for c in CONDITIONS:
        print(f"  {c}: {len(by_cond[c])} bags")

    # ---- distributions ----
    print(f"\n## max_mean_m distribution (p25/p50/p75/p90/p99/p99.9/max per cond)")
    for c in CONDITIONS:
        xs = [r["max_mean_m"] for r in by_cond[c]]
        if not xs: continue
        print(f"  {c:<16s}  p25={_pct(xs,25):.3f}  p50={_pct(xs,50):.3f}  "
              f"p75={_pct(xs,75):.3f}  p90={_pct(xs,90):.3f}  "
              f"p99={_pct(xs,99):.3f}  p99.9={_pct(xs,99.9):.3f}  "
              f"max={max(xs):.3f}")

    print(f"\n## max_n_support distribution (p25/p50/p75/p90/p99/p99.9/max per cond)")
    for c in CONDITIONS:
        xs = [r["max_n_support"] for r in by_cond[c]]
        if not xs: continue
        print(f"  {c:<16s}  p25={_pct(xs,25):.1f}  p50={_pct(xs,50):.1f}  "
              f"p75={_pct(xs,75):.1f}  p90={_pct(xs,90):.1f}  "
              f"p99={_pct(xs,99):.1f}  p99.9={_pct(xs,99.9):.1f}  "
              f"max={max(xs):.1f}")

    print(f"\n## n_passing distribution (clusters/bag) — must be driven DOWN by thresholds")
    for c in CONDITIONS:
        xs = [r["n_passing"] for r in by_cond[c]]
        if not xs: continue
        print(f"  {c:<16s}  p25={_pct(xs,25):.1f}  p50={_pct(xs,50):.1f}  "
              f"p75={_pct(xs,75):.1f}  p90={_pct(xs,90):.1f}  "
              f"p99={_pct(xs,99):.1f}  max={max(xs):.0f}")

    # ---- candidate thresholds for max_mean_m ----
    print(f"\n## candidate min_max_mean_m thresholds (target: each null frac_pass <= {args.target_null_pass})")
    reals = [r["max_mean_m"] for r in by_cond["real"]]
    nulls = {c: [r["max_mean_m"] for r in by_cond[c]] for c in CONDITIONS if c != "real"}
    # scan from low to high
    best_t = None
    for t in [i / 100 for i in range(50, 101)]:
        if all(_frac_above(xs, t) <= args.target_null_pass for xs in nulls.values()):
            best_t = t; break
    if best_t is None:
        print(f"  NO threshold up to 1.0 reduces every null below {args.target_null_pass};")
        print(f"  showing threshold at which the worst null is minimized:")
        best_t, best_worst = 0.5, 1.0
        for t in [i / 100 for i in range(50, 101)]:
            worst = max(_frac_above(xs, t) for xs in nulls.values())
            if worst < best_worst:
                best_worst = worst; best_t = t
        print(f"    best t={best_t:.2f}, worst-null frac_pass={best_worst:.3f}, "
              f"real recall={_frac_above(reals, best_t):.3f}")
    else:
        print(f"  threshold = {best_t:.2f}  →  real recall = {_frac_above(reals, best_t):.3f}")
        for c, xs in nulls.items():
            print(f"    {c:<16s}: frac_pass = {_frac_above(xs, best_t):.4f}")

    # ---- candidate thresholds for max_n_support ----
    print(f"\n## candidate min_max_n_support thresholds (target: each null frac_pass <= {args.target_null_pass})")
    reals_s = [r["max_n_support"] for r in by_cond["real"]]
    nulls_s = {c: [r["max_n_support"] for r in by_cond[c]] for c in CONDITIONS if c != "real"}
    best_s = None
    for s in range(2, 30):
        if all(_frac_above(xs, s) <= args.target_null_pass for xs in nulls_s.values()):
            best_s = s; break
    if best_s is None:
        print(f"  NO support count up to 30 reduces every null below {args.target_null_pass}.")
    else:
        print(f"  threshold = {best_s}  →  real recall = {_frac_above(reals_s, best_s):.3f}")
        for c, xs in nulls_s.items():
            print(f"    {c:<16s}: frac_pass = {_frac_above(xs, best_s):.4f}")

    # ---- m_floor_report(L) suggestion ----
    print(f"\n## candidate m_floor_report(L) from REAL p25 of best_m_at_top/L per L")
    by_L_real: dict[int, list[float]] = defaultdict(list)
    for r in by_cond["real"]:
        if r["best_L"] >= 9 and r["best_m_at_top"] > 0:
            by_L_real[r["best_L"]].append(r["best_m_at_top"] / r["best_L"])
    for L in sorted(by_L_real):
        xs = by_L_real[L]
        p25 = _pct(xs, 25)
        suggested_floor = max(int(p25 * L + 0.5), int(0.6 * L + 0.9999))
        print(f"  L={L:2d}  n={len(xs):4d}  p25={p25:.3f}  "
              f"suggested m_floor_report(L)={suggested_floor}  "
              f"(= {suggested_floor/L:.2f} * L)")


if __name__ == "__main__":
    main()
