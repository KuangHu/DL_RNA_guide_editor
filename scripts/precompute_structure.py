"""Precompute per-nt RNAplfold unpaired-stretch profiles for every NC
region in a dataset split and store them in a memmap that the training
dataloader can read via O(1) indexing.

Layout:

    <out>.mmap  : float16 shape (N, nc_max, u_max)   unpaired probabilities
                  NaN cells (RNAplfold "NA") stored as 0; valid mask
                  reconstructed at load time from `nc_lengths` in the index.
    <out>.valid : uint8   shape (N, nc_max, u_max)   1 iff RNAplfold produced
                                                     a non-NA value at that cell
    <out>.index.json : { <site_id>: { <slot>: <mmap_row>, "num_slots": <int> },
                          ... "_meta": { "N": ..., "nc_max": ..., "u_max": ...,
                                          "W": ..., "L": ..., "split": ..., ... } }

Runtime:
    ~10 ms per NC in batched mode (batch size 500 default). 1.5M NCs
    single-core -> ~4 hours; sharded across --shard-count slurm workers
    the whole precompute finishes in minutes.

Usage:
    python -m scripts.precompute_structure \
        --split /groups/.../splits/val.jsonl \
        --out   /groups/.../structure/val_unp_u16 \
        --u-max 16 --W 120 --L 60 --batch-size 500

Optional:
    --limit 500      only process the first N records (smoke)
    --shard-index i / --shard-count K  process every K-th record starting at i
                                         (each shard writes to its own out
                                         paths; a merge step is a separate
                                         concern)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Repo-root import
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess.structure import batch_unpaired_profile
from preprocess.site import DEFAULT_NC_MAX


def iter_records(split_path: Path, limit: int | None):
    with open(split_path) as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            yield i, json.loads(line)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True, type=Path,
                    help="path to a splits/{train,val,test}.jsonl file")
    p.add_argument("--out", required=True, type=Path,
                    help="output base path (writes <out>.mmap, <out>.valid, <out>.index.json)")
    p.add_argument("--u-max", type=int, default=16)
    p.add_argument("--W", type=int, default=120)
    p.add_argument("--L", type=int, default=60)
    p.add_argument("--nc-max", type=int, default=DEFAULT_NC_MAX)
    p.add_argument("--batch-size", type=int, default=500,
                    help="NC sequences per RNAplfold invocation")
    p.add_argument("--limit", type=int, default=None,
                    help="only process the first N records")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--progress-every", type=int, default=1000,
                    help="log progress every N records")
    args = p.parse_args()

    if args.L > args.W:
        raise SystemExit(f"L ({args.L}) must be <= W ({args.W})")
    if not (0 <= args.shard_index < args.shard_count):
        raise SystemExit("shard-index must be in [0, shard-count)")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # First pass: count how many NC rows we'll write, so we can size the
    # memmap up front. Also stash NC lengths keyed by (record_i, slot).
    print(f"[pass 1] scanning {args.split}", flush=True)
    plan: list[tuple[str, int, int]] = []  # (site_id, slot, nc_len)
    n_records = 0
    for i, rec in iter_records(args.split, args.limit):
        if args.shard_count > 1 and (i % args.shard_count) != args.shard_index:
            continue
        n_records += 1
        for slot, nc in enumerate(rec["inputs"]["noncoding_regions"]):
            if len(nc) > args.nc_max:
                raise SystemExit(
                    f"NC region for site {rec['site_id']} slot {slot} has "
                    f"length {len(nc)} > nc_max={args.nc_max}"
                )
            plan.append((rec["site_id"], slot, len(nc)))
        if n_records % args.progress_every == 0:
            print(f"  scanned {n_records} records, {len(plan)} NC rows so far",
                  flush=True)
    N = len(plan)
    print(f"[pass 1] done: {n_records} records, {N} NC rows to fold", flush=True)

    if N == 0:
        raise SystemExit("no NC rows to process (check --split / --limit)")

    # Allocate memmaps.
    mmap_path = args.out.with_suffix(args.out.suffix + ".mmap") if args.out.suffix else Path(str(args.out) + ".mmap")
    valid_path = Path(str(mmap_path).replace(".mmap", ".valid"))
    index_path = Path(str(mmap_path).replace(".mmap", ".index.json"))

    print(f"[alloc] {mmap_path}: float16 ({N}, {args.nc_max}, {args.u_max})", flush=True)
    prof_mm = np.memmap(mmap_path, dtype=np.float16, mode="w+",
                         shape=(N, args.nc_max, args.u_max))
    valid_mm = np.memmap(valid_path, dtype=np.uint8, mode="w+",
                          shape=(N, args.nc_max, args.u_max))
    # Zero-fill (memmap is normally uninitialized in a fresh file, but be explicit).
    prof_mm[:] = 0
    valid_mm[:] = 0

    # Pass 2: batched RNAplfold. Iterate the same order as pass 1 and
    # collect (mmap_row, sequence) tuples per batch. We fold, then write.
    print(f"[pass 2] folding in batches of {args.batch_size}", flush=True)
    index: dict[str, dict] = {}
    row = 0
    row_by_site_slot: dict[tuple[str, int], int] = {}

    # Build the row mapping first (row `k` == k-th tuple in plan).
    for k, (site_id, slot, _nc_len) in enumerate(plan):
        row_by_site_slot[(site_id, slot)] = k
        index.setdefault(site_id, {"num_slots": 0, "slots": {}})
        index[site_id]["slots"][str(slot)] = k
        index[site_id]["num_slots"] = max(index[site_id]["num_slots"], slot + 1)

    # Now stream through split_path again in the same shard/limit order
    # so seqs and rows line up.
    batch_seqs: list[str] = []
    batch_rows: list[int] = []
    batch_lens: list[int] = []
    start = time.time()
    processed = 0

    def flush_batch():
        nonlocal batch_seqs, batch_rows, batch_lens, processed
        if not batch_seqs:
            return
        profiles = batch_unpaired_profile(
            batch_seqs, u_max=args.u_max, W=args.W, L=args.L,
        )
        for (prof, valid_mask), row_idx, nc_len in zip(profiles, batch_rows, batch_lens):
            # Write into (nc_max, u_max)-shaped slot, zero-padded beyond nc_len.
            prof_mm[row_idx, :nc_len, :] = prof.astype(np.float16)
            valid_mm[row_idx, :nc_len, :] = valid_mask.astype(np.uint8)
        processed += len(batch_seqs)
        batch_seqs.clear()
        batch_rows.clear()
        batch_lens.clear()

    for i, rec in iter_records(args.split, args.limit):
        if args.shard_count > 1 and (i % args.shard_count) != args.shard_index:
            continue
        for slot, nc in enumerate(rec["inputs"]["noncoding_regions"]):
            batch_seqs.append(nc)
            batch_rows.append(row_by_site_slot[(rec["site_id"], slot)])
            batch_lens.append(len(nc))
            if len(batch_seqs) >= args.batch_size:
                flush_batch()
                if processed and (processed % (args.progress_every) < args.batch_size):
                    dt = time.time() - start
                    print(f"  processed {processed}/{N} NC rows in {dt:.1f}s "
                          f"({processed / max(dt, 1e-6):.1f} rows/s)", flush=True)
    flush_batch()

    dt = time.time() - start
    print(f"[pass 2] done: {processed} NC rows in {dt:.1f}s "
          f"({processed / max(dt, 1e-6):.1f} rows/s)", flush=True)

    prof_mm.flush()
    valid_mm.flush()

    # Metadata + index.
    index["_meta"] = {
        "N": N,
        "nc_max": args.nc_max,
        "u_max": args.u_max,
        "W": args.W,
        "L": args.L,
        "split": str(args.split),
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "limit": args.limit,
        "n_records": n_records,
        "dtype_prof": "float16",
        "dtype_valid": "uint8",
        "mmap_path": str(mmap_path),
        "valid_path": str(valid_path),
    }
    with open(index_path, "w") as f:
        json.dump(index, f)
    print(f"[write] {index_path} (index for {n_records} records, {N} NC rows)",
          flush=True)


if __name__ == "__main__":
    main()
