"""V8.4 vs v8_main_v3 side-by-side on DurrantWT K=1/3/5/8.

Direct comparison: score the SAME Durrant shards with both checkpoints.
Reports p50/mean/std per K per model, plus Δ v8.4 − v3.

Motivation: Phase 7 nc-swap gate used K=1 only, where flank_dev channels
are 0 by construction (single-site bag has no cross-site variance).
That confounded the interpretation. Fair test is K≥3.

If v8.4 K≥3 scores > v3 K≥3 scores, this iteration produced real
improvement on real Durrant data.
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


DURRANT_K = {
    "K1": (
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K1.jsonl"),
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K1_shard"),
    ),
    "K3": (
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K3.jsonl"),
        Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
              "v8_evals/regrouped_v3/DurrantWT_K3_shard"),
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
CKPTS = {
    "v8_main_v3": Path("checkpoints/channel_b/v8_main_v3/best.pt"),
    "v84_main":   Path("checkpoints/channel_b/v84_main/best.pt"),
}


def dist_line(v):
    if not v:
        return "n=0"
    vs = sorted(v)
    return (f"n={len(v):4d}  min={min(v):7.3f}  p25={vs[len(vs)//4]:7.3f}  "
            f"p50={statistics.median(v):7.3f}  p75={vs[3*len(vs)//4]:7.3f}  "
            f"max={max(v):7.3f}  mean={statistics.mean(v):7.3f}  "
            f"sd={statistics.stdev(v) if len(v) > 1 else 0:.3f}")


def load_model(ckpt_path: Path, device) -> ChannelBModel:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    hp = ck.get("args", {})
    m = ChannelBModel(hidden=hp.get("hidden", 128),
                        n_heads=hp.get("n_heads", 4),
                        n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    m.load_state_dict(ck["model"])
    print(f"  loaded {ckpt_path}  epoch={ck.get('epoch','?')}  "
          f"val_loss={ck.get('val_loss','?')}")
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


def main():
    ap = argparse.ArgumentParser()
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# v8.4 vs v8_main_v3 — DurrantWT K=1/3/5/8 side-by-side")
    print(f"# device={device}\n")

    models = {}
    print("## loading models")
    for name, path in CKPTS.items():
        models[name] = load_model(path, device)
    print()

    # Score each K with each model
    results: dict[tuple[str, str], list[float]] = {}
    for k, (jl, shard) in DURRANT_K.items():
        print(f"## {k} ({jl.name})")
        for mname, model in models.items():
            scores = score_shard(jl, shard, model, device)
            results[(mname, k)] = scores
            print(f"  {mname:>12s}  {dist_line(scores)}")
        print()

    # Summary table
    print(f"## p50 summary")
    print(f"  {'model':>14s}  " + "  ".join(f"{k:>10s}" for k in DURRANT_K))
    for mname in CKPTS:
        cells = "  ".join(f"{statistics.median(results[(mname, k)]):10.3f}"
                              for k in DURRANT_K)
        print(f"  {mname:>14s}  {cells}")

    print(f"\n## Δ p50 (v84_main − v8_main_v3)")
    cells = "  ".join(
        f"{statistics.median(results[('v84_main', k)]) - statistics.median(results[('v8_main_v3', k)]):+10.3f}"
        for k in DURRANT_K)
    print(f"  {'Δ':>14s}  {cells}")

    # If K-monotone in v84, that's the expected shape (more sites → more signal)
    print(f"\n## K-monotonicity check")
    for mname in CKPTS:
        ps = [statistics.median(results[(mname, k)]) for k in DURRANT_K]
        mono_up = all(ps[i] < ps[i + 1] for i in range(len(ps) - 1))
        print(f"  {mname:>14s}  p50={ps}  monotone_up={mono_up}")

    print(f"\n# v84-vs-v3 K-series gate complete")


if __name__ == "__main__":
    main()
