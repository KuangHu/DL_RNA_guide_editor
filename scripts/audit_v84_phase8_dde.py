"""V8.4 Phase 8 — DDE 10-family specificity + v3 comparison.

Positive: DurrantWT_K5 (34 bags, K=5) from regrouped_v3.
Negatives: 10 DDE families from dde_eval_v3 (all bags K~4-5).

Reports per-family:
  - v3 baseline AUROC vs pos
  - v84 AUROC vs pos
  - Δ AUROC
  - v3 and v84 negative p50

Historical reference (finding_negtop10_specificity, v7-real):
  DurrantWT-vs-DDE mean AUROC 0.9897 (uniformly high, real+real)

If v84 mean AUROC ≥ 0.98 → specificity preserved.
If v84 mean AUROC drops meaningfully → real-vs-real signal degraded.
"""
from __future__ import annotations
import argparse
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.model import ChannelBModel


POS_JSONL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v8_evals/regrouped_v3/DurrantWT_K5.jsonl")
POS_SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v8_evals/regrouped_v3/DurrantWT_K5_shard")

DDE_FAMILIES = ["IS1", "IS3", "IS4", "IS5", "IS6", "IS66", "IS256",
                  "IS481", "ISL3", "IS1595"]
DDE_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v8_evals/dde_eval_v3")

CKPTS = {
    "v8_main_v3": Path("checkpoints/channel_b/v8_main_v3/best.pt"),
    "v84_main":   Path("checkpoints/channel_b/v84_main/best.pt"),
}


def load_model(ckpt_path: Path, device) -> ChannelBModel:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ck.get("args", {})
    m = ChannelBModel(hidden=hp.get("hidden", 128),
                        n_heads=hp.get("n_heads", 4),
                        n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    m.load_state_dict(ck["model"])
    return m


def score_shard(jl: Path, shard: Path, model, device) -> list[float]:
    ds = ChannelBDataset(jl, shard, preload_mt=True, cache_dir=None,
                            sort_by_nc_len=False)
    out = []
    for i in range(len(ds)):
        b = ds[i]
        x = b.x.unsqueeze(0).to(device)
        sm = b.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x, sm).squeeze(0)
        out.append(float(pred.max().item()))
    return out


def auroc(pos_scores: list[float], neg_scores: list[float]) -> float:
    """Compute AUROC without sklearn dependency."""
    if not pos_scores or not neg_scores:
        return float("nan")
    pos = np.asarray(pos_scores)
    neg = np.asarray(neg_scores)
    # Mann-Whitney U statistic ↔ AUROC
    n_pos = len(pos)
    n_neg = len(neg)
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]))) + 1.0
    pos_rank_sum = ranks[:n_pos].sum()
    u = pos_rank_sum - n_pos * (n_pos + 1) / 2
    return float(u / (n_pos * n_neg))


def dist_line(v):
    if not v:
        return "n=0"
    vs = sorted(v)
    return (f"n={len(v):4d}  p25={vs[len(vs)//4]:6.3f}  "
              f"p50={statistics.median(v):6.3f}  p75={vs[3*len(vs)//4]:6.3f}  "
              f"mean={statistics.mean(v):6.3f}")


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# v84 vs v3 — DDE 10-family specificity  device={device}\n")

    models = {}
    for name, path in CKPTS.items():
        models[name] = load_model(path, device)
        print(f"  loaded {name}")
    print()

    # Score positives
    print(f"## positive = DurrantWT_K5")
    pos_scores = {}
    for mname, model in models.items():
        pos_scores[mname] = score_shard(POS_JSONL, POS_SHARD, model, device)
        print(f"  {mname:>12s}  {dist_line(pos_scores[mname])}")
    print()

    # Score each DDE family with each model
    results = {}
    print(f"## per-family AUROC vs DurrantWT_K5")
    print(f"  {'family':>10s}  {'v3 AUROC':>10s}  {'v84 AUROC':>10s}  "
              f"{'Δ AUROC':>9s}  {'v3 neg p50':>11s}  {'v84 neg p50':>12s}")
    v3_aurocs = []
    v84_aurocs = []
    for fam in DDE_FAMILIES:
        jl = DDE_ROOT / f"{fam}.jsonl"
        shard = DDE_ROOT / f"{fam}_shard"
        neg_scores = {}
        for mname, model in models.items():
            neg_scores[mname] = score_shard(jl, shard, model, device)
        a_v3  = auroc(pos_scores["v8_main_v3"], neg_scores["v8_main_v3"])
        a_v84 = auroc(pos_scores["v84_main"],   neg_scores["v84_main"])
        p50_v3  = statistics.median(neg_scores["v8_main_v3"])
        p50_v84 = statistics.median(neg_scores["v84_main"])
        results[fam] = {
            "v3_auroc": a_v3, "v84_auroc": a_v84,
            "v3_neg_p50": p50_v3, "v84_neg_p50": p50_v84,
        }
        v3_aurocs.append(a_v3)
        v84_aurocs.append(a_v84)
        print(f"  {fam:>10s}  {a_v3:10.4f}  {a_v84:10.4f}  "
              f"{a_v84 - a_v3:+9.4f}  {p50_v3:11.3f}  {p50_v84:12.3f}")

    print()
    print(f"## summary")
    print(f"  {'model':>12s}  {'mean AUROC':>12s}  {'min':>8s}  {'max':>8s}")
    print(f"  {'v8_main_v3':>12s}  {statistics.mean(v3_aurocs):12.4f}  "
              f"{min(v3_aurocs):8.4f}  {max(v3_aurocs):8.4f}")
    print(f"  {'v84_main':>12s}  {statistics.mean(v84_aurocs):12.4f}  "
              f"{min(v84_aurocs):8.4f}  {max(v84_aurocs):8.4f}")
    print(f"\n# historical reference: v7-real DurrantWT-vs-DDE mean AUROC 0.9897")
    print(f"# v84 phase 8 DDE specificity complete")


if __name__ == "__main__":
    main()
