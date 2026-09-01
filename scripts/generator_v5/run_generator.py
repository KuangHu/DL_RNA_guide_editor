"""V5 generator streaming runner.

Generates N bags, streams JSONL records to disk, and holds only summary
stats in memory (not the fold matrices). Emits the full acceptance
report at the end.

Memory: O(N_bags) for per-bag stats, independent of ncRNA length or N_nc.
For 50K bags: ~50 MB.

Usage:
  python -m scripts.generator_v5.run_generator \
      --n-bags 10000 \
      --seed 0 \
      --out /path/to/positives_v5.jsonl \
      --stats-out /path/to/positives_v5.stats.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generator_v5.bag_v2 import Bag, build_bag, load_flank_pool
from scripts.generator_v5.difficulty import load_or_build_rate_table


# Worker-local state (set once per process by initializer).
_WORKER_TBL = None
_WORKER_FL = None


def _worker_init():
    """Loaded once per multiprocessing worker."""
    global _WORKER_TBL, _WORKER_FL
    _WORKER_TBL = load_or_build_rate_table(rebuild=False)
    _WORKER_FL = load_flank_pool()


def _worker_build_bag(args):
    """Build ONE bag in a worker. args = (idx, seed, include_channels)."""
    idx, seed, include_channels = args
    rng = random.Random(seed)
    b = build_bag(f"bag_{idx:06d}", rng, _WORKER_FL, _WORKER_TBL)
    if b is None:
        return None
    summary = summarize_bag(b)
    records = b.to_v42_jsonl(include_structure_channels=include_channels)
    return summary, records


# Thresholds (mirror test_v5_acceptance_500.py).
TEST1_RATE_LO_FLOOR = 0.10
TEST1_RATE_LO_MAX = 0.05
TEST1B_POOL_LO = 0.20
TEST1B_POOL_HI = 0.35
TEST1B_PER_L_LO = 0.15
TEST1B_PER_L_HI = 0.50
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


def summarize_bag(b: Bag) -> dict:
    """Extract flat per-bag + per-site summary, drop feature arrays."""
    active_idx = b.architecture.active_nc_index
    active_nc = b.ncrna_sequences[active_idx]
    feats = b.ncrna_features[active_idx]
    L = b.difficulty.L
    csum = np.concatenate(([0.0], np.cumsum(feats.p_ss, dtype=np.float64)))
    p_ss_win_mean = (csum[L:] - csum[:-L]) / L
    guide_pct = float((p_ss_win_mean < p_ss_win_mean[b.planted_start_on_nc]).mean())

    per_nc_summary = []
    for i, (nc, f) in enumerate(zip(b.ncrna_sequences, b.ncrna_features)):
        per_nc_summary.append({
            "role":       "active" if i == active_idx else "inactive",
            "nc_len":     len(nc),
            "gc":         gc_percent(nc),
            "mean_dG_u1": float(f.dG_open_u1.mean()),
            "std_dG_u1":  float(f.dG_open_u1.std()),
            "ensemble_energy": float(f.ensemble_energy),
        })

    site_summary = []
    for s in b.sites:
        site_summary.append({
            "site_idx":         s.site_idx,
            "m_at_planted":     s.m_at_planted,
            "competitor_count": s.competitor_count_at_planted_m,
        })

    return {
        "bag_id":         b.bag_id,
        "L":              L,
        "nc_len":         b.difficulty.nc_len,
        "planted_m":      b.difficulty.planted_m,
        "target_m":       b.difficulty.target_m,
        "is_split":       b.architecture.is_split,
        "split_gap":      b.architecture.split_gap,
        "is_reversed":    b.architecture.is_reversed_target,
        "n_nc":           b.architecture.n_nc,
        "n_positions":    b.difficulty.nc_len - L + 1,
        "guide_pct":      guide_pct,
        "has_5p_sl_active": b.has_5p_stem_loop_per_nc[active_idx],
        "nc_summary":     per_nc_summary,
        "site_summary":   site_summary,
    }


def _pass(v, tgt, op):
    if op == "<":  return v <= tgt
    if op == ">=": return v >= tgt
    if op == "in": return tgt[0] <= v <= tgt[1]
    return False


def acceptance_report(bag_stats: list[dict]) -> dict:
    """Compute Test 1/2/3/4 + leakage AUROC over all bag summaries."""
    from sklearn.metrics import roc_auc_score

    # Flatten to per-site records
    rec = []
    for b in bag_stats:
        for s in b["site_summary"]:
            rec.append({
                "L":              b["L"],
                "nc_len":         b["nc_len"],
                "planted_m":      b["planted_m"],
                "target_m":       b["target_m"],
                "is_split":       b["is_split"],
                "n_positions":    b["n_positions"],
                "competitor_count": s["competitor_count"],
                "m_at_planted":   s["m_at_planted"],
            })
    L_arr        = np.array([r["L"]              for r in rec])
    nc_len_arr   = np.array([r["nc_len"]         for r in rec])
    planted_m    = np.array([r["planted_m"]      for r in rec])
    target_m     = np.array([r["target_m"]       for r in rec])
    n_pos_arr    = np.array([r["n_positions"]    for r in rec])
    comp_count   = np.array([r["competitor_count"] for r in rec])
    m_at_plant   = np.array([r["m_at_planted"]   for r in rec])
    is_split_arr = np.array([r["is_split"]       for r in rec])
    rate = comp_count / np.maximum(n_pos_arr, 1)
    mask_at_target = (planted_m == target_m)

    def _sub_report(label, m_at_target_mask, comp_count_all, rate_all,
                      planted_m_all, target_m_all, n_bags_here, is_pool):
        below_floor = float((rate_all < TEST1_RATE_LO_FLOOR).mean())
        rate_at_target = rate_all[m_at_target_mask]
        rate_med = (float(np.median(rate_at_target))
                     if len(rate_at_target) else float("nan"))
        above_target = float((planted_m_all > target_m_all).mean())
        sole_max = float((comp_count_all == 1).mean())
        c_med = float(np.median(comp_count_all))
        t1b_lo, t1b_hi = ((TEST1B_POOL_LO, TEST1B_POOL_HI) if is_pool
                            else (TEST1B_PER_L_LO, TEST1B_PER_L_HI))
        checks = [
            ("test1a", below_floor, TEST1_RATE_LO_MAX, "<"),
            ("test1b", rate_med, (t1b_lo, t1b_hi), "in"),
            ("test1c", above_target, TEST1_ZERO_MASS_AT_ABOVE_TARGET, "<"),
            ("test2a", sole_max, TEST2_SOLE_MAX_MAX, "<"),
            ("test2b", c_med, TEST2_MEDIAN_MIN, ">="),
        ]
        out = {"label": label, "n_bags": n_bags_here, "checks": {}}
        all_ok = True
        for name, val, tgt, op in checks:
            ok = _pass(val, tgt, op)
            all_ok &= ok
            out["checks"][name] = {"value": val, "target": (tgt if not isinstance(tgt, tuple) else list(tgt)),
                                     "op": op, "pass": ok}
        out["pass"] = all_ok
        return out

    report = {}
    report["pool"] = _sub_report("pool", mask_at_target, comp_count, rate,
                                  planted_m, target_m, len(bag_stats), is_pool=True)

    # Length quartile stratified
    q = np.quantile(nc_len_arr, [0, 0.25, 0.5, 0.75, 1.0])
    report["by_nc_len_quartile"] = []
    for i in range(4):
        lo, hi = q[i], q[i + 1]
        mask = (nc_len_arr >= lo) & (nc_len_arr <= hi if i == 3 else nc_len_arr < hi)
        if mask.sum() < 10:
            continue
        report["by_nc_len_quartile"].append(
            _sub_report(f"Q{i+1}[{int(lo)},{int(hi)}]", mask_at_target[mask],
                          comp_count[mask], rate[mask], planted_m[mask],
                          target_m[mask], int(mask.sum()), is_pool=True))

    # Per-L
    report["by_L"] = []
    for L in sorted(set(L_arr.tolist())):
        mask_L = (L_arr == L)
        if int(mask_L.sum()) < 10:
            continue
        report["by_L"].append(
            _sub_report(f"L={L}", mask_at_target[mask_L], comp_count[mask_L],
                          rate[mask_L], planted_m[mask_L], target_m[mask_L],
                          int(mask_L.sum()), is_pool=False))

    # Test 3
    percentiles = np.array([b["guide_pct"] for b in bag_stats])
    med = float(np.median(percentiles)); iqr = float(np.percentile(percentiles, 75) - np.percentile(percentiles, 25))
    report["test3"] = {
        "median_pct": med, "iqr": iqr,
        "pass_median": TEST3_MED_LO <= med <= TEST3_MED_HI,
        "pass_iqr": iqr >= TEST3_IQR_MIN,
    }

    # Split vs contig
    report["split_vs_contig"] = {}
    for label, mask in (("contiguous", ~is_split_arr), ("split", is_split_arr)):
        m = m_at_plant[mask]
        if len(m) == 0:
            continue
        report["split_vs_contig"][label] = {
            "n": int(len(m)),
            "median_m": float(np.median(m)),
            "P_m_ge_6": float((m >= 6).mean()),
        }

    # Ratio distribution
    ratios = L_arr / nc_len_arr
    report["ratio"] = {
        "min": float(ratios.min()), "median": float(np.median(ratios)),
        "p90": float(np.percentile(ratios, 90)), "max": float(ratios.max()),
    }

    # Leakage AUROC
    from collections import defaultdict
    role_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for b in bag_stats:
        for nc in b["nc_summary"]:
            for k in ("nc_len", "gc", "mean_dG_u1", "std_dG_u1", "ensemble_energy"):
                role_stats[nc["role"]][k].append(float(nc[k]))
    report["leakage_auroc"] = {}
    for k in ("nc_len", "gc", "mean_dG_u1", "std_dG_u1", "ensemble_energy"):
        a = np.array(role_stats["active"][k])
        n = np.array(role_stats["inactive"][k])
        if len(a) == 0 or len(n) == 0:
            continue
        y = np.concatenate([np.ones(len(a)), np.zeros(len(n))])
        x = np.concatenate([a, n])
        try:
            au = float(roc_auc_score(y, x))
        except Exception:
            au = float("nan")
        report["leakage_auroc"][k] = {"auroc": au,
                                        "direction_invariant": max(au, 1 - au) if not np.isnan(au) else au}
    return report


def print_report(report: dict) -> None:
    print("\n=== Pool acceptance ===")
    _print_sub(report["pool"])
    print("\n=== By nc_len quartile ===")
    for q in report["by_nc_len_quartile"]:
        _print_sub(q)
    print("\n=== By L (per-L bands) ===")
    for r in report["by_L"]:
        _print_sub(r)
    print("\n=== Test 3 ===")
    t3 = report["test3"]
    print(f"  median %ile = {t3['median_pct']:.3f}  ({'PASS' if t3['pass_median'] else 'FAIL'})")
    print(f"  IQR         = {t3['iqr']:.3f}         ({'PASS' if t3['pass_iqr'] else 'FAIL'})")
    print("\n=== Split vs contig ===")
    for label, s in report["split_vs_contig"].items():
        print(f"  {label:<11s} n={s['n']:>6d}  median_m={s['median_m']:.1f}  P(m>=6)={s['P_m_ge_6']:.4f}")
    print("\n=== Ratio ===")
    r = report["ratio"]
    print(f"  min={r['min']:.3f}, median={r['median']:.3f}, p90={r['p90']:.3f}, max={r['max']:.3f}")
    print("\n=== Leakage AUROC ===")
    for k, v in report["leakage_auroc"].items():
        di = v["direction_invariant"]
        mark = "✓" if di <= 0.55 else "✗"
        print(f"  {mark} {k:<20s} AUROC={v['auroc']:.3f}  (|inv|={di:.3f})")


def _print_sub(r):
    mark = "PASS" if r.get("pass") else "FAIL"
    print(f"  {r['label']} (n_bags={r['n_bags']}): {mark}")
    for name, ck in r["checks"].items():
        m = "✓" if ck["pass"] else "✗"
        v = ck["value"]
        t = ck["target"]
        tgt_s = f"[{t[0]}, {t[1]}]" if isinstance(t, list) else t
        val_s = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {m} {name}: {val_s}  (target {ck['op']} {tgt_s})")


def _git_commit(repo_root: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                            capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _rate_table_hash(tbl_path: str) -> str:
    try:
        with open(tbl_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:
        return "unknown"


def _emit_manifest(path: str, args: argparse.Namespace,
                     tbl_path: str, n_bags_produced: int,
                     seconds: float) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "generator_version":    "v5",
        "git_commit":           _git_commit(repo_root),
        "viennarna_version":    RNA.__version__,
        "python_version":       sys.version.split()[0],
        "seed":                 args.seed,
        "n_bags_requested":     args.n_bags,
        "n_bags_produced":      n_bags_produced,
        "seconds_elapsed":      seconds,
        "workers":              args.workers,
        "rate_table_path":      tbl_path,
        "rate_table_sha256":    _rate_table_hash(tbl_path),
        "output_jsonl":         args.out,
        "stats_out":            args.stats_out,
        "include_structure_channels": args.include_channels,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bags", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None,
                     help="JSONL output for bag site records")
    ap.add_argument("--stats-out", type=str, default=None,
                     help="JSON output for acceptance report")
    ap.add_argument("--manifest-out", type=str, default=None,
                     help="JSON output for generator manifest")
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=1,
                     help="Number of parallel workers (1 = serial)")
    ap.add_argument("--include-channels", action=argparse.BooleanOptionalAction,
                     default=True,
                     help="Include per-nc structure channels + mask in JSONL")
    ap.add_argument("--rate-table-path", type=str,
                     default="/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/rate_table.json")
    args = ap.parse_args()

    print(f"[gen] preflight: rate table + flank pool")
    tbl = load_or_build_rate_table(args.rate_table_path, rebuild=False)
    fl = load_flank_pool()
    print(f"[gen] flank pool = {len(fl)} sequences; rate_table L in {tbl.L_range}")

    out_fp = None
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out_fp = open(args.out, "w")

    # Per-bag seeds derived from the master seed to keep worker RNG independent.
    master_rng = random.Random(args.seed)
    per_bag_seeds = [master_rng.randrange(0, 2**31 - 1) for _ in range(args.n_bags)]
    tasks = [(i, per_bag_seeds[i], args.include_channels) for i in range(args.n_bags)]

    bag_stats: list[dict] = []
    t0 = time.perf_counter()
    skips = 0
    try:
        if args.workers <= 1:
            _worker_init()
            for k, tsk in enumerate(tasks):
                r = _worker_build_bag(tsk)
                if r is None:
                    skips += 1
                else:
                    summary, records = r
                    bag_stats.append(summary)
                    if out_fp is not None:
                        for rec in records:
                            out_fp.write(json.dumps(rec) + "\n")
                if (k + 1) % args.progress_every == 0:
                    dt = time.perf_counter() - t0
                    print(f"  [{k+1}/{args.n_bags}] kept {len(bag_stats)}, "
                          f"{dt:.1f}s ({dt/(k+1)*1000:.0f} ms/bag)")
        else:
            print(f"[gen] launching {args.workers} workers")
            with mp.Pool(args.workers, initializer=_worker_init) as pool:
                for k, r in enumerate(pool.imap_unordered(_worker_build_bag,
                                                            tasks, chunksize=8), 1):
                    if r is None:
                        skips += 1
                    else:
                        summary, records = r
                        bag_stats.append(summary)
                        if out_fp is not None:
                            for rec in records:
                                out_fp.write(json.dumps(rec) + "\n")
                    if k % args.progress_every == 0:
                        dt = time.perf_counter() - t0
                        print(f"  [{k}/{args.n_bags}] kept {len(bag_stats)}, "
                              f"{dt:.1f}s ({dt/k*1000:.0f} ms/bag)")
    finally:
        if out_fp is not None:
            out_fp.close()

    dt = time.perf_counter() - t0
    print(f"[gen] done in {dt:.1f}s ({dt/max(args.n_bags,1)*1000:.0f} ms/bag), "
          f"skips={skips}")

    report = acceptance_report(bag_stats)
    print_report(report)
    if args.stats_out:
        Path(args.stats_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.stats_out, "w") as f:
            json.dump({
                "n_bags":  len(bag_stats),
                "seconds": dt,
                "report":  report,
            }, f, indent=2)
        print(f"[gen] stats written to {args.stats_out}")
    if args.manifest_out:
        _emit_manifest(args.manifest_out, args, args.rate_table_path,
                         len(bag_stats), dt)
        print(f"[gen] manifest written to {args.manifest_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
