"""Extract simple per-site "shortcut" features + aggregate per tnp.

Deliberately small feature set — the point is to see how far a simple
model can go using ONLY:

  1. Best-alignment scores (max over all i, j, orient, per guide length L)
  2. Top-K alignment score distribution across the whole (nc, flank) map
  3. Seed density max
  4. Basic ncRNA accessibility stats (mean/max unpaired-stretch prob for
     stretch lengths 1, 8, 12, 16)
  5. NC count / lengths

If a logistic regression or tiny MLP on these features approaches V1's
AUPRC=1.0000, the synthetic task is trivially shortcut-solvable and V1's
architecture isn't being stressed.

Per-tnp aggregation over the 50 sites: mean, max, std.

Output: <out>.npz with:
  X:            float32 (num_tnps, F)
  y:            bool    (num_tnps,)
  tnp_ids:      list[str]
  groups:       list[str]   ("positive" or violation profile)
  feature_names: list[str]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocess.alignment import dot_plot, windowed_matches, perfect_seed_density
from preprocess.site import StructureCache


# Per-site scalar features — 21 total.
SITE_FEATURE_NAMES: list[str] = [
    "best_align_score",           # max over (all NCs, all L, both orient) of matches / L
    "best_fwd_L8_score",
    "best_fwd_L12_score",
    "best_fwd_L16_score",
    "best_rc_L8_score",
    "best_rc_L12_score",
    "best_rc_L16_score",
    "top1_score", "top2_score", "top3_score", "top4_score", "top5_score",
    "n_cells_above_0.75",
    "n_cells_above_0.90",
    "max_seed_density_L5",
    "num_ncs",
    "total_nc_len",
    "mean_u1",                   # per-nt unpaired prob, averaged across all NCs
    "max_u1",
    "max_u8_stretch",
    "max_u12_stretch",
    "max_u16_stretch",
]

AGG_NAMES = ("mean", "max", "std")


def _best_score_max_and_topk(
    all_scores: list[np.ndarray], K: int = 5
) -> tuple[float, list[float]]:
    """Given a list of score arrays (one per (NC, orient, L)), return the
    overall max and the top-K distinct maxima (flat)."""
    if not all_scores:
        return 0.0, [0.0] * K
    flat = np.concatenate([a.ravel() for a in all_scores])
    if flat.size == 0:
        return 0.0, [0.0] * K
    best = float(flat.max())
    # Top-K by score.
    k = min(K, flat.size)
    top_idx = np.argpartition(-flat, k - 1)[:k]
    top_vals = np.sort(-flat[top_idx])
    top_vals = -top_vals
    padded = list(top_vals) + [0.0] * (K - len(top_vals))
    return best, padded[:K]


def _extract_site_features(rec: dict, structure_cache: StructureCache) -> np.ndarray:
    """Return a length-len(SITE_FEATURE_NAMES) float32 vector for this site."""
    flank = rec["inputs"]["flank"]
    ncs = rec["inputs"]["noncoding_regions"]
    site_id = rec["site_id"]

    all_scores: list[np.ndarray] = []
    best_fwd_L: dict[int, float] = {}
    best_rc_L: dict[int, float] = {}
    max_seed_density = 0.0
    total_nc_len = 0
    u1_all: list[np.ndarray] = []
    max_u8 = 0.0
    max_u12 = 0.0
    max_u16 = 0.0

    for slot, nc in enumerate(ncs):
        total_nc_len += len(nc)
        fwd_dot, rc_dot = dot_plot(nc, flank)
        # Only compute for L=8..16 to reflect the true guide-length range;
        # the "score" feature uses matches/L for all Ls together.
        for L in range(5, 17):
            fwd_win = windowed_matches(fwd_dot, L)
            rc_win = windowed_matches(rc_dot, L)
            if fwd_win.size:
                fwd_score = fwd_win.astype(np.float32) / float(L)
                all_scores.append(fwd_score)
                if L in (8, 12, 16):
                    best_fwd_L[L] = max(best_fwd_L.get(L, 0.0), float(fwd_score.max()))
            if rc_win.size:
                rc_score = rc_win.astype(np.float32) / float(L)
                all_scores.append(rc_score)
                if L in (8, 12, 16):
                    best_rc_L[L] = max(best_rc_L.get(L, 0.0), float(rc_score.max()))
        # Seed density max (fwd only — save time).
        sd = perfect_seed_density(fwd_dot, 5, radius=8)
        neighborhood = (2 * 8 + 1) ** 2
        max_seed_density = max(max_seed_density, float(sd.max()) / neighborhood)

        # Structure profile from cache.
        prof, valid = structure_cache.get(site_id, slot, len(nc))
        u1_all.append(prof[:, 0])
        # u=8 stretch prob at position i needs valid[..., 7]; u=8 requires >= 8 positions.
        for u_idx, u_name in ((7, "u8"), (11, "u12"), (15, "u16")):
            v = prof[:, u_idx][valid[:, u_idx]]
            if v.size:
                cur = float(v.max())
                if u_name == "u8":
                    max_u8 = max(max_u8, cur)
                elif u_name == "u12":
                    max_u12 = max(max_u12, cur)
                elif u_name == "u16":
                    max_u16 = max(max_u16, cur)

    best_score, top5 = _best_score_max_and_topk(all_scores, K=5)
    # Count above thresholds across all (nc, L, orient) score maps.
    n_above_075 = 0
    n_above_090 = 0
    for a in all_scores:
        n_above_075 += int((a >= 0.75).sum())
        n_above_090 += int((a >= 0.90).sum())

    u1_flat = np.concatenate(u1_all) if u1_all else np.array([0.0])
    feats = [
        best_score,
        best_fwd_L.get(8, 0.0), best_fwd_L.get(12, 0.0), best_fwd_L.get(16, 0.0),
        best_rc_L.get(8, 0.0),  best_rc_L.get(12, 0.0),  best_rc_L.get(16, 0.0),
        top5[0], top5[1], top5[2], top5[3], top5[4],
        float(n_above_075), float(n_above_090),
        max_seed_density,
        float(len(ncs)),
        float(total_nc_len),
        float(u1_flat.mean()),
        float(u1_flat.max()),
        max_u8, max_u12, max_u16,
    ]
    assert len(feats) == len(SITE_FEATURE_NAMES), (
        f"{len(feats)} != {len(SITE_FEATURE_NAMES)}"
    )
    return np.asarray(feats, dtype=np.float32)


def _aggregate_site_features(site_feats: np.ndarray) -> np.ndarray:
    """site_feats: (S, F_site). Return (F_site * 3,) mean+max+std."""
    m = site_feats.mean(axis=0)
    mx = site_feats.max(axis=0)
    st = site_feats.std(axis=0)
    return np.concatenate([m, mx, st], axis=0)


def _tnp_group(rec: dict) -> str:
    lbl = rec["labels"]
    if lbl.get("is_positive"):
        return "positive"
    return lbl.get("violation_profile") or "unknown"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split-jsonl", required=True, type=Path)
    p.add_argument("--structure-index", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--max-tnps", type=int, default=None,
                    help="cap number of tnps processed (smoke)")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    args = p.parse_args()

    cache = StructureCache(args.structure_index)

    # First pass: group sites by tnp_id (in file order).
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
    all_tnps = sorted(tnp_lines.keys())
    if args.max_tnps is not None:
        # keep a stratified sample (positives + negatives) for smoke tests
        pos = [t for t in all_tnps if tnp_group[t] == "positive"][: args.max_tnps // 3]
        neg = [t for t in all_tnps if tnp_group[t] != "positive"][
            : args.max_tnps - len(pos)
        ]
        all_tnps = pos + neg
    # Shard.
    tnp_ids = [t for k, t in enumerate(all_tnps) if k % args.shard_count == args.shard_index]
    print(f"[extract] {len(tnp_ids)} tnps to process "
          f"(shard {args.shard_index}/{args.shard_count})", flush=True)

    n_features = len(SITE_FEATURE_NAMES) * len(AGG_NAMES)
    feature_names = []
    for agg in AGG_NAMES:
        for name in SITE_FEATURE_NAMES:
            feature_names.append(f"{name}__{agg}")

    X = np.zeros((len(tnp_ids), n_features), dtype=np.float32)
    y = np.zeros((len(tnp_ids),), dtype=bool)
    groups = []
    t0 = time.time()
    for ti, tnp in enumerate(tnp_ids):
        site_line_idxs = tnp_lines[tnp]
        # Read the records for this tnp.
        recs = []
        with open(args.split_jsonl, "rb") as f:
            for li in site_line_idxs:
                f.seek(offsets[li])
                recs.append(json.loads(f.readline()))
        site_feats = np.stack([_extract_site_features(r, cache) for r in recs], axis=0)
        X[ti] = _aggregate_site_features(site_feats)
        y[ti] = (tnp_group[tnp] == "positive")
        groups.append(tnp_group[tnp])
        if (ti + 1) % 100 == 0:
            dt = time.time() - t0
            rate = (ti + 1) / dt
            eta = (len(tnp_ids) - ti - 1) / max(rate, 1e-6)
            print(f"  {ti+1}/{len(tnp_ids)}  rate={rate:.1f} tnps/s  eta={eta:.0f}s",
                  flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        X=X, y=y,
        tnp_ids=np.asarray(tnp_ids, dtype=object),
        groups=np.asarray(groups, dtype=object),
        feature_names=np.asarray(feature_names, dtype=object),
    )
    print(f"[extract] wrote {args.out}  X={X.shape}  y={y.shape}  "
          f"positives={int(y.sum())}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
