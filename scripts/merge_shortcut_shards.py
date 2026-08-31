"""Concatenate per-shard shortcut feature npz files into one npz."""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-glob", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    shard_paths = sorted(glob.glob(args.in_glob))
    if not shard_paths:
        raise SystemExit(f"no shards match {args.in_glob!r}")
    print(f"[merge] {len(shard_paths)} shards")

    Xs, ys, tnp_ids, groups = [], [], [], []
    feature_names = None
    for path in shard_paths:
        d = np.load(path, allow_pickle=True)
        Xs.append(d["X"]); ys.append(d["y"])
        tnp_ids.extend(list(d["tnp_ids"]))
        groups.extend(list(d["groups"]))
        if feature_names is None:
            feature_names = list(d["feature_names"])
        else:
            assert list(d["feature_names"]) == feature_names, f"feature names mismatch at {path}"

    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out, X=X, y=y,
        tnp_ids=np.asarray(tnp_ids, dtype=object),
        groups=np.asarray(groups, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    print(f"[merge] X={X.shape}  y={y.shape}  n_pos={int(y.sum())}  -> {args.out}")


if __name__ == "__main__":
    main()
