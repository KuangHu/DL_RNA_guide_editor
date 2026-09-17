"""Streaming runner for V5 negative bags (no target planted).

Same architecture as run_generator.py — per-bag seed, mp.Pool workers,
JSONL output. Emits manifest for reproducibility.

Usage:
  python -m scripts.generator_v5.run_negatives \
    --n-bags 10000 --seed 100 --workers 60 \
    --out /.../negatives_v5_10k.jsonl \
    --manifest-out /.../manifest_neg_10k.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import subprocess
import sys
import time
from pathlib import Path

import RNA

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generator_v5.bag_v2 import build_negative_bag, load_flank_pool
from scripts.generator_v5.difficulty import load_or_build_rate_table


_WORKER_TBL = None
_WORKER_FL = None


def _worker_init():
    global _WORKER_TBL, _WORKER_FL
    _WORKER_TBL = load_or_build_rate_table(rebuild=False)
    _WORKER_FL = load_flank_pool()


def _worker_build(args):
    idx, seed = args
    rng = random.Random(seed)
    b = build_negative_bag(f"neg_bag_{idx:06d}", rng, _WORKER_FL, _WORKER_TBL)
    if b is None:
        return None
    return b.to_jsonl()


def _git_commit(repo_root: Path) -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root,
                             capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-bags", type=int, required=True)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--manifest-out", type=str, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--progress-every", type=int, default=500)
    ap.add_argument("--v7", action="store_true",
                     help="v7 mode is NOT supported by this script — it uses "
                          "build_negative_bag which is a legacy simpler path. "
                          "For v7 negatives, use run_generator.py with "
                          "--negative-mode {scattered,partial,twin} --v7.")
    args = ap.parse_args()
    if args.v7:
        raise SystemExit(
            "run_negatives.py does not support v7. Use `run_generator.py "
            "--v7 --negative-mode scattered|partial|twin` instead — that "
            "path routes through build_bag which has v7_mode wired.")

    print(f"[gen-neg] preflight")
    load_or_build_rate_table(rebuild=False)
    load_flank_pool()
    print(f"[gen-neg] warm-up done")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(args.seed)
    per_bag_seeds = [master_rng.randrange(0, 2**31 - 1) for _ in range(args.n_bags)]
    tasks = [(i, per_bag_seeds[i]) for i in range(args.n_bags)]

    t0 = time.perf_counter()
    n_kept = 0
    with open(args.out, "w") as out_fp:
        if args.workers <= 1:
            _worker_init()
            for k, tsk in enumerate(tasks):
                r = _worker_build(tsk)
                if r is not None:
                    for rec in r:
                        out_fp.write(json.dumps(rec) + "\n")
                    n_kept += 1
                if (k + 1) % args.progress_every == 0:
                    dt = time.perf_counter() - t0
                    print(f"  [{k+1}/{args.n_bags}] kept {n_kept}, "
                          f"{dt:.1f}s ({dt/(k+1)*1000:.0f} ms/bag)")
        else:
            print(f"[gen-neg] launching {args.workers} workers")
            with mp.Pool(args.workers, initializer=_worker_init) as pool:
                for k, r in enumerate(pool.imap_unordered(_worker_build,
                                                              tasks, chunksize=8), 1):
                    if r is not None:
                        for rec in r:
                            out_fp.write(json.dumps(rec) + "\n")
                        n_kept += 1
                    if k % args.progress_every == 0:
                        dt = time.perf_counter() - t0
                        print(f"  [{k}/{args.n_bags}] kept {n_kept}, "
                              f"{dt:.1f}s ({dt/k*1000:.0f} ms/bag)")

    dt = time.perf_counter() - t0
    print(f"[gen-neg] done in {dt:.1f}s ({dt/max(args.n_bags,1)*1000:.0f} ms/bag)")

    if args.manifest_out:
        repo_root = Path(__file__).resolve().parents[2]
        manifest = {
            "generator_version":  "v5-neg",
            "git_commit":         _git_commit(repo_root),
            "viennarna_version":  RNA.__version__,
            "python_version":     sys.version.split()[0],
            "seed":               args.seed,
            "n_bags_requested":   args.n_bags,
            "n_bags_produced":    n_kept,
            "seconds_elapsed":    dt,
            "workers":            args.workers,
            "output_jsonl":       args.out,
        }
        Path(args.manifest_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[gen-neg] manifest written to {args.manifest_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
