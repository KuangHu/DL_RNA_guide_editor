"""V8.4 Phase 1.6 smoke — flank_scattered nc revert verification.

C1 rng-alignment: 7 modes × 100 bags at seed=0, bag-shared signature +
    Rfam bracket bytes must be identical across all pairs.

C2 nc plant block width: distribution across 1K bags per mode. Positive
    and flank_scattered should be IDENTICAL (both use [cons+guide+cons]
    plant, both should have block widths 40-84 bp with same distribution).

C3 single-variable confirmation: positive vs flank_scattered.
    - ts distribution per site: positive p50=54±jitter, flank_scattered
      p50=54 but with wider std
    - per-bag ts span: positive p95=4, flank_scattered p95=20-30
    - all other bag-shared axes (bag_guide_L, cons_lens, nc_lens,
      planted_m, guide identity per site) IDENTICAL

C4 invariants: positive path + injected off-target flank write + repeat_flank
    exempt.
"""
from __future__ import annotations
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.generator_v5.bag_v7_real import (
    VALID_V7_REAL_NEGATIVE_MODES,
    build_bag_v7_real, validate_rng_alignment,
    _FLANK_SCOPE_EXEMPT, _TS_SPAN_EXEMPT, _TS_SPAN_MAX,
    _assert_flank_scope, _assert_ts_span,
)
from scripts.generator_v5.real_flank_pool import RealFlankPool


N_BAGS = 1000
SEED = 0


def dist(v, fmt="{:6.2f}"):
    v = np.asarray(v, dtype=np.float64)
    if v.size == 0:
        return "n=0"
    return (f"n={len(v):5d}  min={fmt.format(v.min())}  "
              f"p05={fmt.format(np.percentile(v, 5))}  "
              f"p50={fmt.format(np.median(v))}  "
              f"p95={fmt.format(np.percentile(v, 95))}  "
              f"max={fmt.format(v.max())}  "
              f"mean={fmt.format(v.mean())}  sd={fmt.format(v.std())}")


def main():
    print(f"# V8.4 Phase 1.6 smoke — flank_scattered nc revert\n")
    pool = RealFlankPool.load_default()

    # ---------- C1: rng alignment ----------
    print(f"## C1 RNG-ALIGNMENT — {len(VALID_V7_REAL_NEGATIVE_MODES)} modes × 100 bags @ seed=0")
    try:
        validate_rng_alignment(seed=SEED, n_bags=100, real_flank_pool=pool)
        print(f"     PASS")
    except AssertionError as e:
        print(f"     FAIL: {str(e)[:400]}")
    print()

    # Generate 1K bags per mode for C2/C3
    per_mode = {}
    for mode in VALID_V7_REAL_NEGATIVE_MODES:
        rng = random.Random(SEED)
        bags = []
        for i in range(N_BAGS):
            b = build_bag_v7_real(bag_id=f"v84s6_{mode}_{i:06d}",
                                    rng=rng, real_flank_pool=pool,
                                    negative_mode=mode)
            if b is not None:
                bags.append(b)
        per_mode[mode] = bags

    # ---------- C2: nc plant block width ----------
    print("## C2 NC PLANT BLOCK WIDTH — positive vs flank_scattered")
    for mode in ("none", "flank_scattered", "partial", "repeat_flank"):
        widths = []
        for b in per_mode[mode]:
            L = b.bag_guide_L
            lcl = b.left_conserved_len
            rcl = b.right_conserved_len
            widths.append(lcl + L + rcl)   # analytic block width
        print(f"  {mode:>18s}  block_width  {dist(np.asarray(widths))}")

    # Byte-exact block content equality between positive and flank_scattered
    # (both should plant IDENTICAL block since bracket + guide + plant fn are same
    # at the same per-bag seed; only difference is per-site ts on flank).
    per_bag_seeds = [random.Random(SEED).randrange(0, 2**31 - 1)]
    master_rng = random.Random(SEED)
    per_bag_seeds = [master_rng.randrange(0, 2**31 - 1) for _ in range(50)]
    print(f"\n  byte-equality check: none-plant-block vs flank_scattered-plant-block ({len(per_bag_seeds)} bags @ per-bag seeds)")
    ne = fs = 0
    for s in per_bag_seeds:
        bn = build_bag_v7_real("cmp_n", random.Random(s), pool, negative_mode="none")
        bfs = build_bag_v7_real("cmp_fs", random.Random(s), pool, negative_mode="flank_scattered")
        # nc_planted content should be byte-identical if the plant fn
        # rng consumption is aligned.
        if bn.nc_planted == bfs.nc_planted:
            ne += 1
        else:
            fs += 1
    print(f"    identical: {ne}/{ne+fs}  different: {fs}/{ne+fs}")
    print()

    # ---------- C3: single-variable confirmation ----------
    print("## C3 SINGLE-VARIABLE CONFIRMATION — positive vs flank_scattered")
    # per-bag ts span
    for mode in ("none", "flank_scattered"):
        spans = [max(b.per_site_target_start) - min(b.per_site_target_start)
                    for b in per_mode[mode]]
        print(f"  {mode:>18s}  per-bag ts_span  {dist(np.asarray(spans), fmt='{:5.1f}')}")

    # bag-shared axes — should match between positive and flank_scattered
    print(f"\n  bag-shared axes (should match ← rng alignment)")
    for axis, extract, fmt in [
        ("bag_guide_L",         lambda b: b.bag_guide_L, "{:5.1f}"),
        ("n_sites",             lambda b: b.n_sites, "{:5.1f}"),
        ("left_conserved_len",  lambda b: b.left_conserved_len, "{:5.1f}"),
        ("right_conserved_len", lambda b: b.right_conserved_len, "{:5.1f}"),
        ("len(nc_planted)",     lambda b: len(b.nc_planted), "{:6.1f}"),
        ("len(nc_noise)",       lambda b: len(b.nc_noise), "{:6.1f}"),
    ]:
        print(f"    --- {axis} ---")
        for mode in ("none", "flank_scattered"):
            vs = [extract(b) for b in per_mode[mode]]
            print(f"      {mode:>18s}  {dist(np.asarray(vs), fmt)}")
    print()

    # ---------- C4: invariants (positive path + injection) ----------
    print("## C4 INVARIANTS")
    print(f"  positive path: 7 modes × {N_BAGS} bags — no bag raised during construction")
    for mode in VALID_V7_REAL_NEGATIVE_MODES:
        print(f"    {mode:>22s}  n_ok={len(per_mode[mode])}")
    print()
    # negative-injection tests (reuse the existing self-test's negative cases inline)
    raw = ["A" * 120] * 3
    ts = [40, 40, 40]
    L = [11, 11, 11]
    final_legit = ["A" * 40 + "C" * 11 + "A" * 69,
                     "A" * 40 + "G" * 11 + "A" * 69,
                     "A" * 40 + "T" * 11 + "A" * 69]
    try:
        _assert_flank_scope("none", raw, final_legit, ts, L)
        print("  flank-scope legit  → PASS (accepted)")
    except AssertionError as e:
        print(f"  flank-scope legit  → FAIL (false positive): {e}")
    bad_final = list(final_legit)
    bad_final[1] = "A" * 90 + "G" + "A" * 29
    try:
        _assert_flank_scope("none", raw, bad_final, ts, L)
        print("  flank-scope inject → FAIL (violation not caught)")
    except AssertionError:
        print("  flank-scope inject → PASS (caught off-target write at pos 90)")
    try:
        _assert_flank_scope("repeat_flank", raw, bad_final, ts, L)
        print("  repeat_flank exempt → PASS")
    except AssertionError as e:
        print(f"  repeat_flank exempt → FAIL: {e}")

    print("\n# V8.4 Phase 1.6 smoke complete")


if __name__ == "__main__":
    main()
