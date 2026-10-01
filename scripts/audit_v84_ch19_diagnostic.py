"""V8.4 Durrant regression — channel-level attribution diagnostic.

Motivation: Phase 7 nc-swap gate showed v8.4 model treats nc distribution
as a discriminative feature (Rfam in-distribution scores 5.5, everything
else scores ~2). Full-real DurrantWT_K1 scored -0.004 (near floor).
User hypothesis: `flank_bg_identity` (ch19) for K=1 bags is hardcoded
to 0.0 (data.py:99-100), which is OOD from training positives (~0.26).

This script:
  (1) Reports flank_bg_identity distribution on:
      - v84_none training positives (baseline)
      - DurrantWT_K1 (K=1 → forced default 0.0)
      - DurrantWT_K5 (K=5 → real cross-site value)
      - DurrantWT_K8 (K=8 → real cross-site value)
  (2) Zero-ablations on v8.4 model scoring:
      - baseline: unmodified
      - zero_flank_bg (ch19): the K=1-OOD hypothesis
      - zero_flank_dev (ch11-16): bag-variance channels
      - zero_structure (ch6-10): fold-derived channels
  Reports p50 delta for each ablation on:
      - v84_none (in-distribution baseline)
      - DurrantWT_K1 (real, K=1)
      - DurrantWT_K5 (real, K=5)
"""
from __future__ import annotations
import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset, _compute_flank_bg_identity
from model.channel_b.model import ChannelBModel
from model.channel_b.constants import CHANNELS


V84_JSONL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v84_generation/50k/v84_none.jsonl")
V84_SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v84_shards/v84_none_shard")
DURRANT = {
    "K1": (
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K1.jsonl"),
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K1_shard"),
    ),
    "K5": (
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K5.jsonl"),
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K5_shard"),
    ),
    "K8": (
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K8.jsonl"),
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K8_shard"),
    ),
}


CH_IDX = {c: i for i, c in enumerate(CHANNELS)}
CH_FLANK_BG = [CH_IDX["flank_bg_identity"]]
CH_FLANK_DEV = [CH_IDX[f"flank_dev_L{L}"] for L in (9, 10, 11, 12, 13, 14)]
CH_STRUCTURE = [CH_IDX[c] for c in ("dG_open_uL_pn", "H_pair_win",
                                        "cooperativity_win_pn", "E_span_win",
                                        "structure_valid")]


def dist_line(v):
    if not v:
        return "n=0"
    vs = sorted(v)
    return (f"n={len(v):5d}  min={min(v):7.4f}  p25={vs[len(vs)//4]:7.4f}  "
            f"p50={statistics.median(v):7.4f}  p75={vs[3*len(vs)//4]:7.4f}  "
            f"max={max(v):7.4f}  mean={statistics.mean(v):7.4f}")


def measure_flank_bg_from_jsonl(jl: Path, max_bags: int = 500) -> list[float]:
    bags = {}
    for line in open(jl):
        r = json.loads(line)
        bags.setdefault(r["transposase_id"], []).append(
            r["inputs"]["flank"])
        if len(bags) >= max_bags:
            break
    return [_compute_flank_bg_identity(fs) for fs in bags.values()]


def score_with_zero(ds: ChannelBDataset, model, device,
                      zero_channels: list[int] | None = None,
                      max_bags: int | None = None) -> list[float]:
    scores = []
    n = min(len(ds), max_bags) if max_bags else len(ds)
    for i in range(n):
        b = ds[i]
        x = b.x.clone()
        if zero_channels:
            x[..., zero_channels] = 0.0
        x = x.unsqueeze(0).to(device)
        sm = b.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x, sm).squeeze(0)
        scores.append(float(pred.max().item()))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/channel_b/v84_main/best.pt"))
    ap.add_argument("--n-v84-bags", type=int, default=200)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp = ckpt.get("args", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                          n_heads=hp.get("n_heads", 4),
                          n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"# v84 channel-attribution diagnostic")
    print(f"# ckpt={args.ckpt}  epoch={ckpt.get('epoch','?')}  "
          f"device={device}")
    print(f"# CHANNELS index: flank_bg={CH_FLANK_BG}  flank_dev={CH_FLANK_DEV}  "
          f"structure={CH_STRUCTURE}\n")

    # ---------- (1) flank_bg_identity distributions ----------
    print("## (1) flank_bg_identity per bag (from raw JSONL)\n")
    print("  v84_none training positives (500 bags):")
    v84_fbi = measure_flank_bg_from_jsonl(V84_JSONL, max_bags=500)
    print(f"    {dist_line(v84_fbi)}")
    print()
    for k, (jl, _) in DURRANT.items():
        vals = measure_flank_bg_from_jsonl(jl)
        print(f"  DurrantWT_{k} (all bags):")
        print(f"    {dist_line(vals)}")
    print()

    # ---------- (2) zero-ablations on v84_none (in-distribution baseline) ----------
    print("## (2) zero-ablation on v84_none (200 bags — in-distribution baseline)\n")
    ds84 = ChannelBDataset(V84_JSONL, V84_SHARD, preload_mt=True,
                              cache_dir=None, sort_by_nc_len=False,
                              max_bags=args.n_v84_bags)
    conditions = [
        ("baseline",       None),
        ("zero_flank_bg",  CH_FLANK_BG),
        ("zero_flank_dev", CH_FLANK_DEV),
        ("zero_structure", CH_STRUCTURE),
    ]
    v84_scores = {}
    for name, zero_ch in conditions:
        v84_scores[name] = score_with_zero(ds84, model, device, zero_ch)
        print(f"  {name:>18s}  {dist_line(v84_scores[name])}")
    print()

    # ---------- (3) zero-ablations on DurrantWT_K1 ----------
    print("## (3) zero-ablation on DurrantWT_K1 (real, K=1 → forced fbi=0.0)\n")
    jl, sd = DURRANT["K1"]
    dsK1 = ChannelBDataset(jl, sd, preload_mt=True, cache_dir=None,
                              sort_by_nc_len=False)
    k1_scores = {}
    for name, zero_ch in conditions:
        k1_scores[name] = score_with_zero(dsK1, model, device, zero_ch)
        print(f"  {name:>18s}  {dist_line(k1_scores[name])}")
    print()

    # ---------- (4) zero-ablations on DurrantWT_K5 ----------
    print("## (4) zero-ablation on DurrantWT_K5 (real, K=5 → real cross-site fbi)\n")
    jl, sd = DURRANT["K5"]
    dsK5 = ChannelBDataset(jl, sd, preload_mt=True, cache_dir=None,
                              sort_by_nc_len=False)
    k5_scores = {}
    for name, zero_ch in conditions:
        k5_scores[name] = score_with_zero(dsK5, model, device, zero_ch)
        print(f"  {name:>18s}  {dist_line(k5_scores[name])}")
    print()

    # ---------- summary ----------
    print("## summary — p50 by (dataset × ablation)")
    print(f"  {'dataset':>18s}  " +
          "  ".join(f"{c:>18s}" for c, _ in conditions))
    for label, scores_dict in (("v84_none", v84_scores),
                                    ("DurrantWT_K1", k1_scores),
                                    ("DurrantWT_K5", k5_scores)):
        cells = "  ".join(f"{statistics.median(scores_dict[c]):18.3f}"
                              for c, _ in conditions)
        print(f"  {label:>18s}  {cells}")

    print(f"\n## Δ p50 vs baseline (per-dataset)")
    for label, scores_dict in (("v84_none", v84_scores),
                                    ("DurrantWT_K1", k1_scores),
                                    ("DurrantWT_K5", k5_scores)):
        base = statistics.median(scores_dict["baseline"])
        deltas = []
        for c, _ in conditions:
            if c == "baseline":
                continue
            d = statistics.median(scores_dict[c]) - base
            deltas.append(f"{c}={d:+.3f}")
        print(f"  {label:>18s}  " + "  ".join(deltas))

    print("\n# diagnostic complete")


if __name__ == "__main__":
    main()
