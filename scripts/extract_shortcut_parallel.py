"""Multi-process feature extraction — same features as extract_shortcut_features.py
but sharded across CPU cores in a single Python process pool. Runs on
login/interactive nodes when the Slurm queue is jammed.

Usage:
    python -m scripts.extract_shortcut_parallel \
        --split-jsonl .../val.jsonl \
        --structure-index .../val_u16.index.json \
        --out shortcut_features/val_shortcut.npz \
        --workers 20 [--max-tnps 3000]
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

# Import the feature functions from the single-process script.
from scripts.extract_shortcut_features import (
    AGG_NAMES,
    SITE_FEATURE_NAMES,
    _aggregate_site_features,
    _extract_site_features,
    _tnp_group,
)
from preprocess.site import StructureCache


# Worker-side globals set in _init_worker.
_g_split_path: str | None = None
_g_offsets: np.ndarray | None = None
_g_cache: StructureCache | None = None


def _init_worker(split_path: str, offsets_np: np.ndarray, structure_index: str):
    global _g_split_path, _g_offsets, _g_cache
    _g_split_path = split_path
    _g_offsets = offsets_np
    _g_cache = StructureCache(structure_index)


def _process_tnp(args: tuple[str, list[int]]) -> tuple[np.ndarray, bool, str]:
    tnp, line_idxs = args
    recs = []
    with open(_g_split_path, "rb") as f:
        for li in line_idxs:
            f.seek(int(_g_offsets[li]))
            recs.append(json.loads(f.readline()))
    site_feats = np.stack(
        [_extract_site_features(r, _g_cache) for r in recs], axis=0
    )
    agg = _aggregate_site_features(site_feats)
    group = _tnp_group(recs[0])
    is_pos = (group == "positive")
    return agg.astype(np.float32), is_pos, group


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-jsonl", required=True, type=Path)
    p.add_argument("--structure-index", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--max-tnps", type=int, default=None)
    args = p.parse_args()

    # First pass: line-offset index and per-tnp line groups (main process).
    tnp_lines: dict[str, list[int]] = defaultdict(list)
    tnp_group: dict[str, str] = {}
    offsets: list[int] = []
    with open(args.split_jsonl, "rb") as f:
        i = 0
        while True:
            off = f.tell()
            raw = f.readline()
            if not raw:
                break
            offsets.append(off)
            rec = json.loads(raw)
            tnp = rec["transposase_id"]
            tnp_lines[tnp].append(i)
            if tnp not in tnp_group:
                tnp_group[tnp] = _tnp_group(rec)
            i += 1
    offsets_np = np.asarray(offsets, dtype=np.int64)

    all_tnps = sorted(tnp_lines.keys())
    if args.max_tnps is not None:
        pos = [t for t in all_tnps if tnp_group[t] == "positive"][: args.max_tnps // 3]
        neg = [t for t in all_tnps if tnp_group[t] != "positive"][
            : args.max_tnps - len(pos)
        ]
        all_tnps = pos + neg
    tnp_ids = all_tnps
    print(f"[extract] {len(tnp_ids)} tnps, {args.workers} workers", flush=True)

    n_features = len(SITE_FEATURE_NAMES) * len(AGG_NAMES)
    feature_names = []
    for agg in AGG_NAMES:
        for name in SITE_FEATURE_NAMES:
            feature_names.append(f"{name}__{agg}")

    X = np.zeros((len(tnp_ids), n_features), dtype=np.float32)
    y = np.zeros((len(tnp_ids),), dtype=bool)
    groups: list[str] = []

    tasks = [(t, tnp_lines[t]) for t in tnp_ids]
    t0 = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_init_worker,
        initargs=(str(args.split_jsonl), offsets_np, str(args.structure_index)),
    ) as pool:
        for k, (feats, is_pos, group) in enumerate(
            pool.imap(_process_tnp, tasks, chunksize=8)
        ):
            X[k] = feats
            y[k] = is_pos
            groups.append(group)
            if (k + 1) % 100 == 0:
                dt = time.time() - t0
                rate = (k + 1) / dt
                eta = (len(tnp_ids) - k - 1) / max(rate, 1e-6)
                print(f"  {k+1}/{len(tnp_ids)}  rate={rate:.1f} tnps/s  eta={eta:.0f}s",
                      flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, X=X, y=y,
        tnp_ids=np.asarray(tnp_ids, dtype=object),
        groups=np.asarray(groups, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    print(f"[extract] wrote {args.out}  X={X.shape}  positives={int(y.sum())}  "
          f"({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
