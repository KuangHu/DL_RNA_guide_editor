"""v7-real per-mode AUROC stratified by n_sites.

Reads: best.pt at checkpoints/channel_b/v7real_main/best.pt.
Data:  the 5 v7-real corpora VAL splits (main split, seed 0).

For each negative mode m ∈ {twin, scattered, partial}:
  For each n_sites in {3..8}:
    scores_pos  = bag_max_score on POSITIVE (none) bags at this n_sites
    scores_neg  = bag_max_score on m-mode bags at this n_sites
    if both classes have ≥ 20 bags: compute AUROC + 95% CI (DeLong)

Also runs pos+ctrl vs each negative (ctrl is a second positive draw, so
pos∪ctrl gives more positive samples — matches how the model's val AUROC
was pooled across 5 corpora).

Emits table:
  mode          n_sites  n_pos   n_neg   AUROC (95% CI)
  ---           ---      ---     ---     ---

Interpretation criteria (locked pre-hoc per user 2026-09-14):
  * pos-vs-twin ≥ 0.70 in most n_sites cells → cross-site coherence real
  * pos-vs-twin 0.5–0.6 → pooled 0.77 was m-dist not coherence; twin dead
  * pos-vs-scattered ≪ pos-vs-twin → model reads "guide present", not
    "position aligned" — scattered training share needs raising
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
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


def bag_max_score_batch(model, batch, device, normalize_by_n_sites=False):
    x = batch["x"].to(device)
    site_mask = batch["site_mask"].to(device)
    pos_mask = batch["pos_mask"].to(device)
    with torch.no_grad():
        pred = model(x, site_mask)   # (B, P)
    score = (pred * pos_mask.float() + (-1e9) * (~pos_mask).float()).max(dim=1).values
    if normalize_by_n_sites:
        # Divide by n_sites to remove the y-peak scale (positive y-peak = n_sites,
        # so bag_max_score inherits n_sites scale by construction). Under normalization,
        # a "perfect fit" positive bag scores ≈1 regardless of n_sites.
        ns = site_mask.float().sum(dim=1).clamp_min(1.0)
        score = score / ns
    return score.cpu().numpy()


def delong_ci(scores_pos, scores_neg, alpha=0.05):
    """Fast normal-approx CI for AUROC. Not true DeLong but close enough
    at n≥50 and doesn't need external dep. Reports 95% CI."""
    from math import sqrt
    n1, n0 = len(scores_pos), len(scores_neg)
    y = np.concatenate([np.ones(n1), np.zeros(n0)])
    s = np.concatenate([scores_pos, scores_neg])
    auc = float(roc_auc_score(y, s))
    q1 = auc / (2 - auc)
    q2 = 2 * auc * auc / (1 + auc)
    var = (auc * (1 - auc) + (n1 - 1) * (q1 - auc*auc) + (n0 - 1) * (q2 - auc*auc)) / (n1 * n0)
    se = sqrt(max(var, 0.0))
    z = 1.959964
    return auc, max(0.0, auc - z * se), min(1.0, auc + z * se), se


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                     default=Path("checkpoints/channel_b/v7real_main/best.pt"))
    ap.add_argument("--corpora-config", type=Path,
                     default=Path("config/channel_b_v7real_corpora.json"))
    ap.add_argument("--cache-root", type=Path,
                     default=Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                                    "channel_b_cache_v7real"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--min-cell-n", type=int, default=20,
                     help="skip (mode, n_sites) cell if either class has fewer bags")
    ap.add_argument("--normalize-by-n-sites", action="store_true",
                     help="Divide bag_max_score by n_sites to remove the y-peak "
                          "scale (positive y-peak = n_sites). If the monotonic "
                          "twin AUROC climb 0.86-0.98 survives normalization, "
                          "the climb is real cross-site amplification; if it "
                          "flattens, part of it is scale-driven.")
    ap.add_argument("--n-sites-filter", type=str, default="",
                     help="Comma-separated n_sites values to KEEP; empty = no "
                          "filter. Example: '7,8' for C4 gate eval (train on {3-6}, "
                          "val on {7,8}).")
    ap.add_argument("--maxm-filter-eq", type=int, default=0,
                     help="If >0: keep POSITIVE bags where max(m_at_planted "
                          "across sites) == this value; keep all negatives "
                          "unchanged. For held-out-m gate: --maxm-filter-eq 8.")
    ap.add_argument("--medm-filter-eq", type=int, default=0,
                     help="If >0: keep POSITIVE bags where median(m_at_planted "
                          "across sites) == this value. More permissive than "
                          "maxm-filter-eq (which needs ALL sites to hit the "
                          "value); medm=8 selects bags where the typical site "
                          "is low-m, not just the lowest one. Pre-registered "
                          "2026-09-16 as replacement for the underpowered "
                          "max(m)==8 filter (only 56 pos bags total).")
    args = ap.parse_args()

    ns_keep = {int(x) for x in args.n_sites_filter.split(",") if x.strip()} \
              if args.n_sites_filter else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[per-mode] device={device}")

    # Load model
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp = ckpt.get("hparams", {})
    hidden = hp.get("hidden", 128)
    n_heads = hp.get("n_heads", 4)
    n_blocks = hp.get("n_blocks", 3)
    model = ChannelBModel(hidden=hidden, n_heads=n_heads, n_blocks=n_blocks).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[per-mode] loaded {args.ckpt}  (hidden={hidden}, n_heads={n_heads}, n_blocks={n_blocks})")

    # Load corpora (use warm cache — training just filled all .pt files)
    cfg = json.loads(args.corpora_config.read_text())
    batch = Path(cfg["batch_root"]); shard = Path(cfg["shard_root"])
    corpora = cfg["corpora"]

    # per-mode collection: mode -> n_sites -> list[score]
    per_mode_scores: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    per_mode_bag_ns: dict[str, int] = defaultdict(int)

    def mode_of_bag_id(bid: str) -> str | None:
        for m in ("none", "twin", "partial", "scattered"):
            if f"_{m}_" in bid:
                return m
        return None

    for cname, jsonl, sh in corpora:
        ds = ChannelBDataset(batch / jsonl, shard / sh,
                                 preload_mt=True,
                                 cache_dir=args.cache_root / cname)
        tr, va = build_split(ds, "main", seed=args.seed)
        print(f"  [{cname}] {len(va)} val bags")

        loader = DataLoader(va, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.workers, collate_fn=bucket_collate_fn,
                                pin_memory=True)
        for b in loader:
            scores = bag_max_score_batch(model, b, device,
                                                normalize_by_n_sites=args.normalize_by_n_sites)
            ns = b["site_mask"].sum(dim=1).cpu().numpy()
            metas = b["metas"]      # list of dicts, one per bag
            for i, meta in enumerate(metas):
                bid = meta["bag_id"]
                mode = mode_of_bag_id(bid)
                if mode is None: continue
                n_sites_i = int(ns[i])
                if ns_keep is not None and n_sites_i not in ns_keep:
                    continue
                # maxm / medm filter (positive bags only)
                if (args.maxm_filter_eq > 0 or args.medm_filter_eq > 0) and mode == "none":
                    sites = ds._sites_by_bag[bid]
                    ms = [s["labels"].get("m_at_planted") for s in sites]
                    ms = [m for m in ms if m is not None]
                    if not ms:
                        continue
                    if args.maxm_filter_eq > 0 and max(ms) != args.maxm_filter_eq:
                        continue
                    if args.medm_filter_eq > 0:
                        import statistics
                        med = statistics.median(ms)
                        if med != args.medm_filter_eq:
                            continue
                per_mode_scores[mode][n_sites_i].append(float(scores[i]))
                per_mode_bag_ns[mode] += 1

    print(f"\n[per-mode] bags collected:")
    for m in ("none", "twin", "partial", "scattered"):
        print(f"  {m:<10s}  {per_mode_bag_ns[m]:>6d} val bags")

    # Build pos scores (from 'none' mode only for the primary line;
    # secondary: pos ∪ ctrl — but ctrl also lives under 'none' in bag_id,
    # so per-mode grouping above already merges pos50k + ctrl10k into 'none')
    def cell_auroc(pos_scores, neg_scores):
        n1, n0 = len(pos_scores), len(neg_scores)
        if n1 < args.min_cell_n or n0 < args.min_cell_n:
            return None
        return delong_ci(np.array(pos_scores), np.array(neg_scores))

    score_kind = "normalized (score/n_sites)" if args.normalize_by_n_sites else "raw bag_max_score"
    print(f"\n[per-mode] === PER-MODE AUROC × n_sites (95% normal-approx CI) — {score_kind} ===")
    print(f"  {'neg mode':<12s} {'n_sites':>7s} {'n_pos':>7s} {'n_neg':>7s} "
              f"{'AUROC':>7s} {'95% CI':>18s}")
    for neg_mode in ("twin", "scattered", "partial"):
        for ns in range(3, 9):
            pos = per_mode_scores["none"].get(ns, [])
            neg = per_mode_scores[neg_mode].get(ns, [])
            cell = cell_auroc(pos, neg)
            if cell is None:
                print(f"  {neg_mode:<12s} {ns:>7d} {len(pos):>7d} {len(neg):>7d} "
                          f"{'(skip)':>7s} {'(cell too small)':>18s}")
                continue
            auc, lo, hi, se = cell
            print(f"  {neg_mode:<12s} {ns:>7d} {len(pos):>7d} {len(neg):>7d} "
                      f"{auc:>7.4f} [{lo:.4f}, {hi:.4f}]  se={se:.4f}")
        # Also pooled across n_sites for comparison
        pos_all = [s for ns in range(3, 9) for s in per_mode_scores["none"].get(ns, [])]
        neg_all = [s for ns in range(3, 9) for s in per_mode_scores[neg_mode].get(ns, [])]
        cell = cell_auroc(pos_all, neg_all)
        if cell is not None:
            auc, lo, hi, se = cell
            print(f"  {neg_mode:<12s} {'POOL':>7s} {len(pos_all):>7d} {len(neg_all):>7d} "
                      f"{auc:>7.4f} [{lo:.4f}, {hi:.4f}]  se={se:.4f}")
        print()

    print(f"[per-mode] === LOCKED CRITERIA (per user 2026-09-14) ===")
    print(f"  pos-vs-twin ≥ 0.70 in most n_sites cells → cross-site coherence real")
    print(f"  pos-vs-twin 0.5-0.6                     → pooled 0.77 = m-dist, twin dead")
    print(f"  pos-vs-scattered ≪ pos-vs-twin          → model reads 'guide present' not "
              f"'position aligned'; raise scattered training share")

    return 0


if __name__ == "__main__":
    sys.exit(main())
