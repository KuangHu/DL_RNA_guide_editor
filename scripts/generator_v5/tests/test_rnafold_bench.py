"""A1 — RNAfold single-fold cost benchmark.

Decides the skeleton-pool design in A2:
  - if single-fold cost < 10 ms across the target [177, 281] nt range,
    per-sample folding is affordable (250K samples at 10 ms = 42 min)
    -> ncrna_pool.py generates fresh ncRNA per bag
  - if 10-100 ms, either 10k+ pool or geometry-only reuse
  - if > 100 ms, geometry-only reuse (accessibility mask + fresh sequence)

Reports per-length distribution across n samples per length.
"""
from __future__ import annotations

import random
import statistics
import time

import RNA


LENGTHS = [177, 200, 225, 250, 281]
N_PER_LENGTH = 100
SEED = 0


def rand_seq(length: int, rng: random.Random) -> str:
    return "".join(rng.choices("ACGU", k=length))


def bench_one(seq: str) -> float:
    t0 = time.perf_counter()
    fc = RNA.fold_compound(seq)
    (structure, energy) = fc.mfe()
    return (time.perf_counter() - t0) * 1000    # ms


def main() -> None:
    rng = random.Random(SEED)
    print(f"{'length':<10s} {'n':>4s} {'mean_ms':>9s} {'median_ms':>10s} {'p95_ms':>8s} {'per_1k_min':>11s}")
    all_rows = []
    for length in LENGTHS:
        seqs = [rand_seq(length, rng) for _ in range(N_PER_LENGTH)]
        # warmup
        _ = bench_one(seqs[0])
        costs = [bench_one(s) for s in seqs]
        mean = statistics.mean(costs)
        med = statistics.median(costs)
        p95 = sorted(costs)[int(0.95 * len(costs))]
        per_1k_min = mean * 1000 / 60000
        print(f"{length:<10d} {len(costs):>4d} {mean:>9.2f} {med:>10.2f} {p95:>8.2f} {per_1k_min:>11.2f}")
        all_rows.append((length, mean, med, p95))

    print()
    print("=== Extrapolation for 250K samples ===")
    max_mean = max(r[1] for r in all_rows)
    at_length = [r[0] for r in all_rows if r[1] == max_mean][0]
    total_min = max_mean * 250_000 / 60_000
    total_hr = total_min / 60
    print(f"  worst-case single-fold cost (mean): {max_mean:.2f} ms at length {at_length}")
    print(f"  250K per-sample fold: {total_min:.1f} min = {total_hr:.2f} h")
    print()
    print("=== Verdict ===")
    if max_mean < 10:
        print("  Per-sample folding AFFORDABLE. ncrna_pool.py should generate fresh ncRNA per bag.")
        print("  Both skeleton-pool n_eff and geometry-consistency issues avoided.")
    elif max_mean < 100:
        print("  Per-sample folding MARGINAL.")
        print("  Recommendation: pool of 10,000+ pre-folded ncRNAs, or")
        print("  Alternative: pool of 1,000 with geometry-only reuse (loop mask reused,")
        print("               new sequence per sample).")
    else:
        print("  Per-sample folding TOO EXPENSIVE.")
        print("  Recommendation: geometry-only reuse — pre-fold N templates,")
        print("  extract loop masks, generate fresh sequences that match the mask.")


if __name__ == "__main__":
    main()
