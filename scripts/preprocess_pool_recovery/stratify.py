"""Read the per-site pool-recovery JSONL and print stratified tables."""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v5a_framework.metrics import (
    assert_non_degenerate_stratification,
    DegenerateStratumError,
)


def _rate(rows: list[dict], key: str) -> float:
    if not rows:
        return float("nan")
    return float(np.mean([r[key] for r in rows]))


def _fmt(rows: list[dict]) -> str:
    n = len(rows)
    if n == 0:
        return f"{'':>6s} {'':>6s} {'':>6s} {'':>6s} {'':>5s}"
    return (
        f"{_rate(rows, 'strict'):>6.3f} "
        f"{_rate(rows, 'L_full'):>6.3f} "
        f"{_rate(rows, 'subL'):>6.3f} "
        f"{_rate(rows, 'any_L'):>6.3f} "
        f"{n:>5d}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.rows) as f:
        for line in f:
            rows.append(json.loads(line))

    print(f"[stratify] {len(rows)} sites")
    print(f"[stratify] cols: strict | L_full | subL | any_L | n")

    # === Non-degeneracy assert: every stratum with n >= 50 must NOT have all
    # four rate metrics == 0.0. History: orient='rev' silently zeroed 50% of
    # the corpus in the first pass. This catches that class of bug at report
    # time rather than at re-analysis time.
    METRICS = ("strict", "L_full", "subL", "any_L")
    axes = ("mm_conc", "is_split", "orient")
    for axis in axes:
        strat = defaultdict(lambda: {"n_tnps": 0, **{m: 0.0 for m in METRICS}})
        counts = defaultdict(lambda: {m: 0 for m in METRICS})
        for r in rows:
            k = r[axis]
            strat[k]["n_tnps"] += 1
            for m in METRICS:
                counts[k][m] += int(r[m])
        for k in strat:
            for m in METRICS:
                strat[k][m] = counts[k][m] / max(1, strat[k]["n_tnps"])
        try:
            assert_non_degenerate_stratification(
                strat, METRICS, min_n=50, axis_name=axis,
            )
        except DegenerateStratumError as exc:
            print(f"\n[stratify] NON-DEGENERACY ASSERT FAILED on axis={axis}:")
            print(f"  {exc}")
            sys.exit(2)
    print(f"[stratify] non-degeneracy check passed on axes: {axes}")

    # ---- overall ----
    print("\n=== overall ===")
    print(f"  {_fmt(rows)}")

    # ---- by is_split ----
    print("\n=== by is_split ===")
    for v in (False, True):
        sub = [r for r in rows if r["is_split"] == v]
        print(f"  is_split={v!s:5s}  {_fmt(sub)}")

    # ---- by mm_concentration (non-split only) ----
    ns = [r for r in rows if not r["is_split"]]
    print("\n=== by mm_concentration (non-split only) ===")
    for v in ("clustered", "dispersed"):
        sub = [r for r in ns if r["mm_conc"] == v]
        print(f"  mm_conc={v:<10s}  {_fmt(sub)}")

    # ---- by planted_L × mm_concentration (non-split only) ----
    print("\n=== L × mm_concentration (non-split only) ===")
    print(f"  {'L':>3s}  {'mm_conc':<10s}  {'strict':>6s} {'L_full':>6s} {'subL':>6s} {'any_L':>6s} {'n':>5s}")
    Ls = sorted(set(r["L"] for r in ns))
    for L in Ls:
        for v in ("clustered", "dispersed"):
            sub = [r for r in ns if r["L"] == L and r["mm_conc"] == v]
            print(f"  {L:>3d}  {v:<10s}  {_fmt(sub)}")

    # ---- by planted_L × planted_m (non-split, clustered) ----
    print("\n=== L × planted_m (non-split, clustered only) ===")
    print(f"  {'L':>3s}  {'m':>3s}  {'strict':>6s} {'L_full':>6s} {'subL':>6s} {'any_L':>6s} {'n':>5s}")
    cl = [r for r in ns if r["mm_conc"] == "clustered"]
    Ls = sorted(set(r["L"] for r in cl))
    for L in Ls:
        ms = sorted(set(r["planted_m"] for r in cl if r["L"] == L))
        for m in ms:
            sub = [r for r in cl if r["L"] == L and r["planted_m"] == m]
            if len(sub) < 20:
                continue
            print(f"  {L:>3d}  {m:>3d}  {_fmt(sub)}")

    print("\n=== L × planted_m (non-split, dispersed only) ===")
    print(f"  {'L':>3s}  {'m':>3s}  {'strict':>6s} {'L_full':>6s} {'subL':>6s} {'any_L':>6s} {'n':>5s}")
    dp = [r for r in ns if r["mm_conc"] == "dispersed"]
    Ls = sorted(set(r["L"] for r in dp))
    for L in Ls:
        ms = sorted(set(r["planted_m"] for r in dp if r["L"] == L))
        for m in ms:
            sub = [r for r in dp if r["L"] == L and r["planted_m"] == m]
            if len(sub) < 20:
                continue
            print(f"  {L:>3d}  {m:>3d}  {_fmt(sub)}")

    # ---- by is_split × mm_concentration (split cases) ----
    print("\n=== is_split=True × mm_concentration ===")
    sp = [r for r in rows if r["is_split"]]
    for v in ("clustered", "dispersed"):
        sub = [r for r in sp if r["mm_conc"] == v]
        print(f"  mm_conc={v:<10s}  {_fmt(sub)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
