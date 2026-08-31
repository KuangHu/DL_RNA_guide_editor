"""Test 3 — generator's guide-window P_ss %ile distribution.

500 bags with soft θ sampling. Reports the distribution shape:
  median in [0.80, 0.90]              — matches anchors (T-WT 0.832, ISEc21 0.852)
  IQR width >= 0.10                    — not narrower than an "all at 85" placement

Also reports the marginal p_ss window mean distribution and the sample of
sample-based percentile values. Confirms sigma choice was not too narrow.
"""
from __future__ import annotations

import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.generator_v5.ncrna_sampler_v2 import (
    DEFAULT_GUIDE_LENGTH,
    sample_random_ncrna_and_placement,
)


N_BAGS = 500
NC_LEN_LO = 177
NC_LEN_HI = 281
SEED = 0

TARGET_MED_LO = 0.80
TARGET_MED_HI = 0.90
TARGET_IQR_MIN = 0.10


def main() -> int:
    rng = random.Random(SEED)
    t0 = time.perf_counter()

    percentiles = []
    p_ss_win_means = []
    lengths = []
    skips = 0
    for k in range(N_BAGS):
        L = rng.randint(NC_LEN_LO, NC_LEN_HI)
        out = sample_random_ncrna_and_placement(L, DEFAULT_GUIDE_LENGTH, rng=rng)
        if out is None:
            skips += 1
            continue
        _, _, placement = out
        percentiles.append(placement.percentile)
        p_ss_win_means.append(placement.p_ss_window_mean)
        lengths.append(L)

    dt = time.perf_counter() - t0
    n = len(percentiles)
    per_bag_ms = dt / max(n, 1) * 1000

    if n == 0:
        print("[test3] no bags produced")
        return 1

    p = np.array(percentiles)
    m = np.array(p_ss_win_means)
    med = float(np.median(p))
    q25 = float(np.percentile(p, 25))
    q75 = float(np.percentile(p, 75))
    iqr = q75 - q25

    print(f"[Test 3] n={n} bags in {dt:.1f}s ({per_bag_ms:.1f} ms/bag), skips={skips}")
    print(f"  percentile: median={med:.3f}, p25={q25:.3f}, p75={q75:.3f}, IQR={iqr:.3f}")
    print(f"  percentile: min={p.min():.3f}, p10={np.percentile(p, 10):.3f}, "
          f"p90={np.percentile(p, 90):.3f}, max={p.max():.3f}")
    print(f"  window mean(P_ss): median={float(np.median(m)):.3f}, "
          f"p25={float(np.percentile(m, 25)):.3f}, "
          f"p75={float(np.percentile(m, 75)):.3f}")

    # Acceptance
    check_med = TARGET_MED_LO <= med <= TARGET_MED_HI
    check_iqr = iqr >= TARGET_IQR_MIN
    print()
    print(f"  Test 3.1: median percentile in [{TARGET_MED_LO}, {TARGET_MED_HI}]"
          f"  ->  {'PASS' if check_med else 'FAIL'} (value {med:.3f})")
    print(f"  Test 3.2: IQR percentile >= {TARGET_IQR_MIN}"
          f"  ->  {'PASS' if check_iqr else 'FAIL'} (value {iqr:.3f})")

    ok = check_med and check_iqr
    print()
    print("[Test 3] " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
