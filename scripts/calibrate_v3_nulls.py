"""Phase G calibration driver — run the v3 pipeline on real + 4 null
constructions, report per-condition top-100 distributions.

Reads the v5 adapted jsonl, samples bags, and for each bag runs:
  - REAL  : no flank mutation
  - null1 : dinuc-shuffle flanks
  - null2 : 4-mer-shuffle flanks
  - null3 : flanks swapped with a random OTHER bag
  - null4 : flanks reverse-complemented then dinuc-shuffled

Outputs:
  --out TSV   : per-condition summary (one row per condition)
  --details TSV: per-bag passing-cluster counts and max cluster stats
  --nullthresholds TSV: candidate m_floor_report(L) thresholds (99% null rank)

Usage:
  python scripts/calibrate_v3_nulls.py \
    --jsonl /scratch/.../fna_ins_discovery_v5_md_seed2/adapted.jsonl \
    --n-bags 500 --seed 2026 \
    --out logs/v3_calib_summary.tsv \
    --details logs/v3_calib_details.tsv
"""
from __future__ import annotations
import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.pipeline import run_bag, PipelineConfig
from scripts.v5a_framework.null_generators import (
    null1_dinuc_shuffle, null2_kmer_shuffle, null3_unpaired_bag,
    null4_rc_then_shuffle,
)


def load_jsonl_bags(path: Path, max_bags: int | None = None,
                      seed: int = 2026):
    """Group jsonl records by `transposase_id` into bags, sample up to
    max_bags of them."""
    bags = defaultdict(list)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            bags[r["transposase_id"]].append(r)
    all_ids = sorted(bags.keys())
    rng = random.Random(seed)
    rng.shuffle(all_ids)
    if max_bags is not None:
        all_ids = all_ids[:max_bags]
    return [(bid, bags[bid]) for bid in all_ids], bags


CONDITIONS = ("real", "null1_dinuc", "null2_4mer", "null3_unpaired", "null4_rc_shuf")


def mutate_bag(sites, cond: str, donor_sites, seed: int):
    if cond == "real":
        return sites
    if cond == "null1_dinuc":
        return null1_dinuc_shuffle(sites, seed=seed)
    if cond == "null2_4mer":
        return null2_kmer_shuffle(sites, k=4, seed=seed)
    if cond == "null3_unpaired":
        return null3_unpaired_bag(sites, donor_sites, seed=seed)
    if cond == "null4_rc_shuf":
        return null4_rc_then_shuffle(sites, seed=seed)
    raise ValueError(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--n-bags", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--details", type=Path, default=None)
    ap.add_argument("--per-condition-top", type=int, default=100)
    args = ap.parse_args()

    cfg = PipelineConfig()
    print(f"# loading bags from {args.jsonl}")
    bags, all_bags_dict = load_jsonl_bags(args.jsonl, max_bags=args.n_bags,
                                           seed=args.seed)
    bag_ids_pool = list(all_bags_dict.keys())
    print(f"# sampled {len(bags)} bags (pool size: {len(bag_ids_pool)})")

    details_rows: list[dict] = []
    per_cond_bagscores: dict[str, list[tuple[str, int, float]]] = {
        c: [] for c in CONDITIONS
    }

    t0 = time.time()
    rng = random.Random(args.seed)
    for i, (bag_id, raw) in enumerate(bags, 1):
        donor_bag_id = rng.choice(bag_ids_pool)
        while donor_bag_id == bag_id:
            donor_bag_id = rng.choice(bag_ids_pool)
        donor_sites = all_bags_dict[donor_bag_id]
        for cond in CONDITIONS:
            sites = mutate_bag(raw, cond, donor_sites, seed=args.seed + i * 7)
            try:
                result = run_bag(bag_id, sites, cfg)
            except Exception as e:
                details_rows.append(dict(bag_id=bag_id, cond=cond,
                                           K_valid=0, n_passing=0,
                                           max_n_support=0, max_mean_m=0.0,
                                           max_n_bg_groups=0, err=str(e)[:120]))
                continue
            n_pass = len(result.passing)
            if result.passing:
                best = max(result.passing,
                            key=lambda cb: (cb[0].n_support, cb[0].mean_m_frac))
                best_cluster, best_bg = best
                max_support = best_cluster.n_support
                max_mean_m = best_cluster.mean_m_frac
                max_bg_groups = best_bg.n_local_bg_groups
                best_m_at_top = int(best_cluster.mean_m_frac * best_cluster.L + 0.5)
                best_L = best_cluster.L
            else:
                max_support = 0; max_mean_m = 0.0; max_bg_groups = 0
                best_m_at_top = 0; best_L = 0
            details_rows.append(dict(
                bag_id=bag_id, cond=cond, K_valid=result.K_valid,
                n_passing=n_pass, max_n_support=max_support,
                max_mean_m=max_mean_m, max_n_bg_groups=max_bg_groups,
                best_L=best_L, best_m_at_top=best_m_at_top,
            ))
            per_cond_bagscores[cond].append((bag_id, n_pass, max_mean_m))
        if i % 25 == 0:
            dt = time.time() - t0
            rate = i / dt
            eta = (len(bags) - i) / rate
            print(f"  [{i}/{len(bags)}]  elapsed={dt:.1f}s  "
                  f"rate={rate:.2f} bags/s  ETA={eta:.0f}s", flush=True)

    # Write details
    if args.details:
        args.details.parent.mkdir(parents=True, exist_ok=True)
        cols = ["bag_id","cond","K_valid","n_passing","max_n_support",
                "max_mean_m","max_n_bg_groups","best_L","best_m_at_top","err"]
        with args.details.open("w") as f:
            w = csv.DictWriter(f, delimiter="\t", fieldnames=cols)
            w.writeheader()
            for r in details_rows: w.writerow({k: r.get(k, "") for k in cols})
        print(f"# wrote {args.details} ({len(details_rows)} rows)")

    # Summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for cond in CONDITIONS:
        rs = [r for r in details_rows if r["cond"] == cond and "err" not in r]
        if not rs:
            continue
        n_bags = len(rs)
        n_with_pass = sum(1 for r in rs if r["n_passing"] > 0)
        n_pass_vals = [r["n_passing"] for r in rs]
        max_sup_vals = [r["max_n_support"] for r in rs]
        mean_m_vals = [r["max_mean_m"] for r in rs]
        summary_rows.append(dict(
            cond=cond,
            n_bags=n_bags,
            frac_with_pass=n_with_pass / n_bags,
            median_n_passing=_median(n_pass_vals),
            p90_n_passing=_pct(n_pass_vals, 90),
            p99_n_passing=_pct(n_pass_vals, 99),
            median_max_support=_median(max_sup_vals),
            p99_max_support=_pct(max_sup_vals, 99),
            median_max_mean_m=_median(mean_m_vals),
            p99_max_mean_m=_pct(mean_m_vals, 99),
        ))

    cols = ["cond","n_bags","frac_with_pass",
            "median_n_passing","p90_n_passing","p99_n_passing",
            "median_max_support","p99_max_support",
            "median_max_mean_m","p99_max_mean_m"]
    with args.out.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=cols)
        w.writeheader()
        for r in summary_rows: w.writerow({k: r.get(k, "") for k in cols})
    print(f"# wrote {args.out}")
    print(f"\n## per-condition summary:")
    for r in summary_rows:
        print(f"  {r['cond']:<16s}  n_bags={r['n_bags']:>4d}  "
              f"frac_pass={r['frac_with_pass']:.3f}  "
              f"median_n_pass={r['median_n_passing']:.1f}  "
              f"p99_n_pass={r['p99_n_passing']:.1f}  "
              f"p99_max_sup={r['p99_max_support']:.1f}  "
              f"p99_max_mean_m={r['p99_max_mean_m']:.3f}")


def _median(xs):
    if not xs: return 0.0
    xs = sorted(xs); n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _pct(xs, p):
    if not xs: return 0.0
    xs = sorted(xs)
    k = int(len(xs) * p / 100)
    k = min(k, len(xs) - 1)
    return xs[k]


if __name__ == "__main__":
    main()
