"""Paired AUROC-difference bootstrap for two Channel B checkpoints on the
same val slice.

Loads both models, scores every val bag once each, groups by (mode, n_sites),
then bootstraps within each cell (resampling bag indices with replacement,
1000 iterations) to get a paired Δ = AUROC(A) - AUROC(B) distribution and
95% CI.

Emits per-cell:
  n_pos  n_neg  auroc_A  auroc_B  paired_Δ  95% CI  PASS?

PASS = 95% CI upper bound ≤ +threshold (default 0.02).
"""
from __future__ import annotations
import argparse
import json
import sys
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset, bucket_collate_fn
from model.channel_b.model import ChannelBModel
from scripts.channel_b_train import build_split


def score_model(ckpt_path, corpora_config, cache_root, seed, filter_kind, filter_val,
                    device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ckpt.get("hparams", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                                n_heads=hp.get("n_heads", 4),
                                n_blocks=hp.get("n_blocks", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    cfg = json.loads(corpora_config.read_text())
    batch = Path(cfg["batch_root"]); shard = Path(cfg["shard_root"])
    corpora = cfg["corpora"]

    # bag_id -> {mode, n_sites, score}
    out = {}
    for cname, jsonl, sh in corpora:
        ds = ChannelBDataset(batch / jsonl, shard / sh,
                                 preload_mt=True,
                                 cache_dir=cache_root / cname)
        tr, va = build_split(ds, "main", seed=seed)
        loader = DataLoader(va, batch_size=32, shuffle=False, num_workers=2,
                                collate_fn=bucket_collate_fn, pin_memory=True)
        for b in loader:
            x = b["x"].to(device); sm = b["site_mask"].to(device); pm = b["pos_mask"].to(device)
            with torch.no_grad():
                pred = model(x, sm)
            scores = (pred * pm.float() + (-1e9) * (~pm).float()).max(dim=1).values.cpu().numpy()
            ns = sm.sum(dim=1).cpu().numpy()
            for i, meta in enumerate(b["metas"]):
                bid = meta["bag_id"]
                mode = None
                for m in ("none", "twin", "partial", "scattered"):
                    if f"_{m}_" in bid:
                        mode = m; break
                if mode is None: continue
                # filter
                if filter_kind and mode == "none":
                    sites = ds._sites_by_bag[bid]
                    ms = [s["labels"].get("m_at_planted") for s in sites]
                    ms = [m for m in ms if m is not None]
                    if not ms: continue
                    if filter_kind == "medm" and statistics.median(ms) != filter_val:
                        continue
                    if filter_kind == "maxm" and max(ms) != filter_val:
                        continue
                out[bid] = {"mode": mode, "n_sites": int(ns[i]),
                                "score": float(scores[i])}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", type=Path, required=True, help="model A (numerator)")
    ap.add_argument("--ckpt-b", type=Path, required=True, help="model B (denominator)")
    ap.add_argument("--label-a", type=str, default="A")
    ap.add_argument("--label-b", type=str, default="B")
    ap.add_argument("--corpora-config", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter-kind", type=str, default="", choices=["", "medm", "maxm"])
    ap.add_argument("--filter-val", type=int, default=8)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--pass-threshold", type=float, default=0.02,
                     help="PASS if paired-Δ CI upper bound ≤ this (default 0.02)")
    ap.add_argument("--min-cell-n", type=int, default=20)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[paired] device={device}")
    print(f"[paired] filter={args.filter_kind}={args.filter_val}; PASS if CI upper ≤ {args.pass_threshold:+.3f}")

    print(f"\n[paired] scoring model A: {args.ckpt_a}")
    scores_a = score_model(args.ckpt_a, args.corpora_config, args.cache_root,
                                 args.seed, args.filter_kind, args.filter_val, device)
    print(f"  {len(scores_a)} bags")

    print(f"\n[paired] scoring model B: {args.ckpt_b}")
    scores_b = score_model(args.ckpt_b, args.corpora_config, args.cache_root,
                                 args.seed, args.filter_kind, args.filter_val, device)
    print(f"  {len(scores_b)} bags")

    common = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    print(f"\n[paired] common bags: {len(common)}")

    # Group: (mode, n_sites) -> list of (bag_id, score_a, score_b)
    cells = defaultdict(list)
    for bid in common:
        ra, rb = scores_a[bid], scores_b[bid]
        assert ra["mode"] == rb["mode"] and ra["n_sites"] == rb["n_sites"]
        cells[(ra["mode"], ra["n_sites"])].append((bid, ra["score"], rb["score"]))

    # For each (neg_mode, n_sites): combine positive ("none" mode) bags at
    # this n_sites with negative bags at this n_sites of the given neg_mode
    print(f"\n[paired] === PAIRED Δ = AUROC({args.label_a}) - AUROC({args.label_b}) ===")
    print(f"  neg_mode {'n':>3s} {'n_pos':>6s} {'n_neg':>6s} "
              f"{'AUROC_'+args.label_a:>10s} {'AUROC_'+args.label_b:>10s} "
              f"{'Δ':>8s} {'95% CI':>18s} {'verdict':<10s}")

    rng = np.random.default_rng(args.seed)
    all_pass = True
    for neg_mode in ("twin", "scattered", "partial"):
        for n in range(3, 9):
            pos_cell = cells.get(("none", n), [])
            neg_cell = cells.get((neg_mode, n), [])
            n_pos, n_neg = len(pos_cell), len(neg_cell)
            if n_pos < args.min_cell_n or n_neg < args.min_cell_n:
                continue
            pos_arr = np.array(pos_cell, dtype=object)
            neg_arr = np.array(neg_cell, dtype=object)
            # observed AUROCs
            y = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
            sa = np.concatenate([[x[1] for x in pos_cell], [x[1] for x in neg_cell]])
            sb = np.concatenate([[x[2] for x in pos_cell], [x[2] for x in neg_cell]])
            auc_a = float(roc_auc_score(y, sa))
            auc_b = float(roc_auc_score(y, sb))
            delta_obs = auc_a - auc_b

            # bootstrap: resample positive and negative indices independently
            deltas = []
            pos_idx = np.arange(n_pos); neg_idx = np.arange(n_neg)
            for _ in range(args.n_boot):
                pi = rng.integers(0, n_pos, size=n_pos)
                ni = rng.integers(0, n_neg, size=n_neg)
                sa_b = np.concatenate([sa[:n_pos][pi], sa[n_pos:][ni]])
                sb_b = np.concatenate([sb[:n_pos][pi], sb[n_pos:][ni]])
                y_b = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
                # guard: skip if one class disappeared (shouldn't happen here)
                if len(set(y_b.tolist())) < 2: continue
                try:
                    da = roc_auc_score(y_b, sa_b) - roc_auc_score(y_b, sb_b)
                except Exception:
                    continue
                deltas.append(da)
            deltas = np.array(deltas)
            lo, hi = np.percentile(deltas, [2.5, 97.5])
            passed = hi <= args.pass_threshold
            all_pass = all_pass and passed
            verdict = "PASS" if passed else "FAIL"
            print(f"  {neg_mode:<8s} {n:>3d} {n_pos:>6d} {n_neg:>6d} "
                  f"{auc_a:>10.4f} {auc_b:>10.4f} "
                  f"{delta_obs:>+8.4f} [{lo:+.4f}, {hi:+.4f}]  {verdict}")

    print(f"\n[paired] === OVERALL VERDICT ===")
    print(f"  {'PASS' if all_pass else 'FAIL'} — all cells' 95% CI upper bound "
          f"{'≤' if all_pass else '>'} +{args.pass_threshold:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
