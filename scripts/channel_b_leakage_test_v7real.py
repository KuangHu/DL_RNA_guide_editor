"""v7-real port of channel_b_leakage_test.py.

Same structure (15-ch × 5-stat channel summary + 5-fold hash-blocked
LR/GBM + per-channel LR AUROC), plus 4 explicit arch scalars per user
directive:

  - n_sites          (arch.n_sites)
  - nc_len           (labels.ncrna_length; sum over regions for multi)
  - gc               (arch.gc_target)
  - hom              (arch.nc_homology_rate; 1.0 for all v7-real bags)

Feature vector is 75 + 4 = 79 dims per bag. Evaluated on MAIN-val
across the 5 v7-real corpora, using --corpora-config to point at
config/channel_b_v7real_corpora.json.

If the 4-scalar-only LR/GBM AUROC alone is ~model level, that's
pure metadata leakage — sequence path is redundant. Reports:

  (a) LR/GBM on full 79-dim (channel summary + arch)
  (b) LR/GBM on channel summary alone (75-dim)
  (c) LR/GBM on arch-4 alone (4-dim) — the diagnostic scalars
  (d) per-channel LR AUROC (5-fold mean)
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.constants import CHANNELS
from scripts.channel_b_train import build_split


ARCH_KEYS = ["n_sites", "nc_len", "gc", "hom"]


def _bag_channel_features(x: torch.Tensor, site_mask: torch.Tensor) -> np.ndarray:
    """15 × 5 = 75-dim bag-level channel summary (mirror of v6r2 test)."""
    x_np = x.numpy()
    sm = site_mask.numpy()
    real_x = x_np[sm]
    if real_x.size == 0:
        return np.zeros((len(CHANNELS) * 5,), dtype=np.float32)
    flat = real_x.reshape(-1, len(CHANNELS))
    feats = np.zeros((len(CHANNELS) * 5,), dtype=np.float32)
    for c in range(len(CHANNELS)):
        col = flat[:, c]
        feats[c * 5 + 0] = col.mean()
        pctls = np.percentile(col, [25, 50, 75, 95])
        feats[c * 5 + 1] = pctls[0]
        feats[c * 5 + 2] = pctls[1]
        feats[c * 5 + 3] = pctls[2]
        feats[c * 5 + 4] = pctls[3]
    return feats


def _bag_arch_features(bag_meta: dict, first_site_labels: dict) -> np.ndarray:
    """4-dim explicit arch scalars per user directive."""
    arch = first_site_labels.get("arch", {})
    n_sites = float(arch.get("n_sites", bag_meta.get("n_sites", 0)))
    nc_len = float(first_site_labels.get("ncrna_length",
                                                bag_meta.get("nc_len", 0)))
    gc = float(arch.get("gc_target", 0.5))
    hom = float(arch.get("nc_homology_rate", 1.0))
    return np.array([n_sites, nc_len, gc, hom], dtype=np.float32)


def _cv_scores(X, y, fold_of, n_folds, model_kind, seed):
    aurocs = []
    for k in range(n_folds):
        tr_idx = np.where(fold_of != k)[0]
        te_idx = np.where(fold_of == k)[0]
        if len(set(y[te_idx].tolist())) < 2:
            continue
        if model_kind == "lr":
            scaler = StandardScaler().fit(X[tr_idx])
            Xtr = scaler.transform(X[tr_idx]); Xte = scaler.transform(X[te_idx])
            clf = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
        else:
            Xtr = X[tr_idx]; Xte = X[te_idx]
            clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                                    learning_rate=0.1, random_state=seed)
        clf.fit(Xtr, y[tr_idx])
        p = clf.predict_proba(Xte)[:, 1]
        aurocs.append(float(roc_auc_score(y[te_idx], p)))
    return aurocs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora-config", type=Path,
                     default=Path("config/channel_b_v7real_corpora.json"))
    ap.add_argument("--cache-root", type=Path,
                     default=Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                                    "channel_b_cache_v7real"))
    ap.add_argument("--split-mode", default="main")
    ap.add_argument("--split-side", default="val",
                     help="which side to run LR/GBM on; 'val' matches model val AUROC")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bags-per-corpus", type=int, default=None)
    ap.add_argument("--n-folds", type=int, default=5)
    args = ap.parse_args()

    cfg = json.loads(args.corpora_config.read_text())
    batch = Path(cfg["batch_root"]); shard = Path(cfg["shard_root"])
    corpora = cfg["corpora"]
    print(f"[leak-v7real] corpora-config: {args.corpora_config}")
    print(f"[leak-v7real] batch={batch}\n              shard={shard}\n              corpora={[c[0] for c in corpora]}")

    all_ch_feats = []
    all_arch_feats = []
    all_labels = []
    all_bag_ids = []
    for cname, jsonl, sh in corpora:
        ds = ChannelBDataset(batch / jsonl, shard / sh,
                                 preload_mt=True,
                                 cache_dir=args.cache_root / cname,
                                 max_bags=args.max_bags_per_corpus)
        tr, va = build_split(ds, args.split_mode, seed=args.seed)
        sub = tr if args.split_side == "train" else va
        print(f"    {cname}: {len(sub)} {args.split_side} bags", flush=True)
        for local_i in sub.indices:
            bag = ds[local_i]
            bag_id = bag.meta["bag_id"]
            f_ch = _bag_channel_features(bag.x, bag.site_mask)
            first_site_labels = ds._sites_by_bag[bag_id][0]["labels"]
            f_arch = _bag_arch_features(bag.meta, first_site_labels)
            all_ch_feats.append(f_ch)
            all_arch_feats.append(f_arch)
            all_labels.append(int(bag.label))
            all_bag_ids.append(bag_id)

    X_ch = np.stack(all_ch_feats).astype(np.float32)
    X_arch = np.stack(all_arch_feats).astype(np.float32)
    X_full = np.concatenate([X_ch, X_arch], axis=1)
    y = np.array(all_labels, dtype=np.int64)
    n = len(y)
    print(f"\n[leak-v7real] n_bags = {n}, n_pos = {int(y.sum())}, n_neg = {int((1-y).sum())}",
          flush=True)
    print(f"[leak-v7real] channel_summary dim = {X_ch.shape[1]}  (15 ch × 5 stats)")
    print(f"[leak-v7real] arch dim            = {X_arch.shape[1]}  ({ARCH_KEYS})")
    print(f"[leak-v7real] full dim            = {X_full.shape[1]}")

    # Report arch marginals per class for sanity
    print(f"\n[leak-v7real] arch marginals per class:")
    for j, name in enumerate(ARCH_KEYS):
        col = X_arch[:, j]
        pos_mean = float(col[y == 1].mean()) if (y == 1).any() else float("nan")
        neg_mean = float(col[y == 0].mean()) if (y == 0).any() else float("nan")
        pos_std  = float(col[y == 1].std())  if (y == 1).any() else float("nan")
        neg_std  = float(col[y == 0].std())  if (y == 0).any() else float("nan")
        print(f"  {name:<10s}  pos mean±std = {pos_mean:.3f}±{pos_std:.3f}   "
              f"neg mean±std = {neg_mean:.3f}±{neg_std:.3f}")

    # 5-fold hash-blocked
    fold_of = np.zeros((n,), dtype=np.int32)
    for i, bid in enumerate(all_bag_ids):
        h = hashlib.md5(bid.encode()).hexdigest()
        fold_of[i] = int(h[:4], 16) % args.n_folds

    def report(name, X):
        lr = _cv_scores(X, y, fold_of, args.n_folds, "lr", args.seed)
        gb = _cv_scores(X, y, fold_of, args.n_folds, "gbm", args.seed)
        print(f"\n[leak-v7real] === {name} ===")
        print(f"  LR  5-fold AUROC per fold: " + " ".join(f"{a:.4f}" for a in lr))
        print(f"      mean = {np.mean(lr):.4f}   std = {np.std(lr):.4f}   "
              f"95% CI (t) ≈ [{np.mean(lr) - 1.96*np.std(lr)/np.sqrt(len(lr)):.4f}, "
              f"{np.mean(lr) + 1.96*np.std(lr)/np.sqrt(len(lr)):.4f}]")
        print(f"  GBM 5-fold AUROC per fold: " + " ".join(f"{a:.4f}" for a in gb))
        print(f"      mean = {np.mean(gb):.4f}   std = {np.std(gb):.4f}   "
              f"95% CI (t) ≈ [{np.mean(gb) - 1.96*np.std(gb)/np.sqrt(len(gb)):.4f}, "
              f"{np.mean(gb) + 1.96*np.std(gb)/np.sqrt(len(gb)):.4f}]")
        return float(np.mean(lr)), float(np.mean(gb))

    lr_full, gb_full = report("(a) FULL 79-dim = channel summary + arch", X_full)
    lr_ch,   gb_ch   = report("(b) channel summary alone (75-dim)",       X_ch)
    lr_arch, gb_arch = report("(c) arch-4 alone [n_sites/nc_len/gc/hom]", X_arch)

    # Per-channel LR
    print(f"\n[leak-v7real] === (d) per-channel LR AUROC (5-fold mean; 5-stat per channel) ===")
    print(f"  {'channel':<24s} {'AUROC (5-fold mean)':>22s}")
    for c_i, cname in enumerate(CHANNELS):
        Xc = X_ch[:, c_i * 5:(c_i + 1) * 5]
        aus = _cv_scores(Xc, y, fold_of, args.n_folds, "lr", args.seed)
        au_mean = float(np.mean(aus)) if aus else float("nan")
        au_mean = max(au_mean, 1 - au_mean)   # AUROC ≥ 0.5 convention
        print(f"  {cname:<24s} {au_mean:>22.4f}")

    print(f"\n[leak-v7real] === SUMMARY ===")
    print(f"  Model best.pt val AUROC (epoch 7): 0.7696")
    print(f"  FULL 79-dim   LR={lr_full:.4f}  GBM={gb_full:.4f}")
    print(f"  channel-only  LR={lr_ch:.4f}  GBM={gb_ch:.4f}")
    print(f"  arch-4 only   LR={lr_arch:.4f}  GBM={gb_arch:.4f}")
    if max(lr_arch, gb_arch) >= 0.65:
        print(f"  → ARCH-METADATA LEAKAGE: 4 scalars alone reach ~model level; "
                f"metadata is the shortcut, sequence signal redundant.")
    elif max(lr_full, gb_full) >= 0.75:
        print(f"  → BAG-SUMMARY LEAKAGE: 79-dim bag summary reaches model level "
                f"({lr_full:.4f}/{gb_full:.4f} vs 0.7696); channel means dominate over "
                f"per-site/cross-site structure.")
    else:
        print(f"  → NO STRONG LEAKAGE: bag summaries below model AUROC; model uses "
                f"per-site/cross-site structure not captured by summary stats.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
