"""V8.4 Phase 7 — nc-swap Rfam-vs-Durrant gate.

Direct test of whether the V8.4 Rfam-scaffold iteration closed the Durrant
OOD gap. Under v8_main_v3 baseline (Sep-27 test, since deleted):
  condA synth flank + synth nc     p50 = 5.02
  condB Durrant flank + synth nc   p50 = 5.97
  condD synth flank + Durrant nc   p50 = 1.89

Interpretation was: swapping synth nc → Durrant nc drops score ~3 points →
Durrant nc is OOD for the model.

Under V8.4, model was trained on Rfam-scaffold nc. Prediction:
  (A) v8.4 on v84_none (Rfam nc, in-distribution)  →  should score HIGH
  (B) v8.4 on Durrant-nc (real ncRNA)              →  should score CLOSE TO (A)
      if Rfam training generalizes to Bridge RNA (same "real ncRNA scaffold"
      distribution class)
  (C) v8.4 on old synth-random nc                  →  should now be LOWER than (A)
      since random ACGT is out of v8.4's training distribution
  (D) v8.4 on full Durrant WT (Durrant flank + Durrant nc)  →  end-to-end

Decision rule (pre-registered):
  - if (B) ≈ (A) [within 1 point] → Rfam training generalized to Durrant Bridge RNA
    → Phase 7 PASS → this iteration paid off
  - if (B) still ~2-3 below (A) → Rfam training did NOT close Durrant gap
    → deeper issue (may need full-Rfam nc, or bridge-RNA-specific training,
    or non-nc features are dominating the OOD)
"""
from __future__ import annotations
import argparse
import json
import random
import shutil
import statistics
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.model import ChannelBModel
from scripts.build_v7_shard import build_v7_shard


V84_JSONL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v84_generation/50k/v84_none.jsonl")
V84_SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v84_shards/v84_none_shard")

DURRANT_JSONL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                       "v8_evals/dde_eval_v3/DurrantWT.jsonl")
DURRANT_K1_JSONL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                          "v8_evals/regrouped_v3/DurrantWT_K1.jsonl")
DURRANT_K1_SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                          "v8_evals/regrouped_v3/DurrantWT_K1_shard")

CONDD_JSONL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                     "v8_evals/crossover_v3/condD_synth_flank_durrant_nc.jsonl")
CONDD_SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                     "v8_evals/crossover_v3/condD_shard")


def load_v84_bags(n_bags: int, seed: int = 0) -> list[list[dict]]:
    """Load v84_none, group by bag, return `n_bags` sampled bags."""
    per_bag = {}
    with open(V84_JSONL) as f:
        for line in f:
            r = json.loads(line)
            per_bag.setdefault(r["transposase_id"], []).append(r)
            if len(per_bag) >= 5000:
                break
    ids = list(per_bag.keys())
    random.Random(seed).shuffle(ids)
    picked = [per_bag[bid] for bid in ids[:n_bags]]
    return picked


def random_dna(rng: random.Random, n: int, gc: float = 0.5) -> str:
    w = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choices("ACGT", weights=w, k=n))


def load_bridges() -> list[str]:
    seen = set()
    bridges = []
    with open(DURRANT_JSONL) as f:
        for line in f:
            r = json.loads(line)
            ai = int(r["labels"]["active_noncoding_index"])
            nc = r["inputs"]["noncoding_regions"][ai]
            if nc not in seen:
                seen.add(nc)
                bridges.append(nc)
    return bridges


def sample_matched_len(bridges: list[str], target_len: int,
                          rng: random.Random) -> str:
    exact = [s for s in bridges if len(s) == target_len]
    if exact:
        return rng.choice(exact)
    longer = [s for s in bridges if len(s) > target_len]
    if longer:
        s = rng.choice(longer)
        start = rng.randint(0, len(s) - target_len)
        return s[start:start + target_len]
    # Fall back: pad/tile
    s = rng.choice(bridges)
    reps = (target_len + len(s) - 1) // len(s)
    return (s * reps)[:target_len]


def build_variant(bags: list[list[dict]], mode: str, rng: random.Random,
                    bridges: list[str] | None = None) -> list[dict]:
    """Emit records with nc swapped per mode.

    mode:
      A  : baseline — record unchanged (v84_none = Rfam scaffold)
      C  : synth flank + old synth-random nc — replace active nc with
           GC-weighted random ACGT of matched length
      B_bridge : synth flank + Durrant Bridge RNA nc — replace active
                 nc with Bridge RNA sequence of matched length
    """
    out = []
    for recs in bags:
        for r in recs:
            rec = json.loads(json.dumps(r))
            if mode == "A":
                out.append(rec)
                continue
            labels = rec["labels"]
            inputs = rec["inputs"]
            ai = int(labels["active_noncoding_index"])
            old_nc = inputs["noncoding_regions"][ai]
            target_len = len(old_nc)
            gc = float(labels["arch"].get("gc_target", 0.5))
            if mode == "C":
                new_nc = random_dna(rng, target_len, gc)
            elif mode == "B_bridge":
                if bridges is None:
                    raise ValueError("bridges required for B_bridge")
                new_nc = sample_matched_len(bridges, target_len, rng)
            else:
                raise ValueError(f"unknown mode {mode}")
            new_regions = list(inputs["noncoding_regions"])
            new_regions[ai] = new_nc
            rec["inputs"]["noncoding_regions"] = new_regions
            # y label no longer valid after nc swap
            rec["labels"]["guide_span_in_active_noncoding"] = None
            out.append(rec)
    return out


def build_and_score(jl: Path, shard: Path, model, device) -> list[float]:
    if shard.exists():
        shutil.rmtree(shard)
    build_v7_shard(jl, shard, min_sites=1, family_label=jl.stem)
    ds = ChannelBDataset(jl, shard, preload_mt=True, cache_dir=None,
                          sort_by_nc_len=False)
    scores = []
    for i in range(len(ds)):
        b = ds[i]
        x = b.x.unsqueeze(0).to(device)
        sm = b.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x, sm).squeeze(0)
        scores.append(float(pred.max().item()))
    return scores


def dist_line(v):
    if not v:
        return "n=0"
    vs = sorted(v)
    return (f"n={len(v):4d}  min={min(v):6.3f}  p25={vs[len(vs)//4]:6.3f}  "
            f"p50={statistics.median(v):6.3f}  p75={vs[3*len(vs)//4]:6.3f}  "
            f"max={max(v):6.3f}  mean={statistics.mean(v):6.3f}")


def score_existing_shard(jl: Path, shard: Path, model, device) -> list[float]:
    ds = ChannelBDataset(jl, shard, preload_mt=True, cache_dir=None,
                          sort_by_nc_len=False)
    scores = []
    for i in range(len(ds)):
        b = ds[i]
        x = b.x.unsqueeze(0).to(device)
        sm = b.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x, sm).squeeze(0)
        scores.append(float(pred.max().item()))
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                    default=Path("checkpoints/channel_b/v84_main/best.pt"))
    ap.add_argument("--out-root", type=Path,
                    default=Path("/global/scratch/users/kh36969/"
                                  "DL_novel_guide_editor/v84_evals/"
                                  "phase7_nc_swap"))
    ap.add_argument("--n-bags", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp = ckpt.get("args", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                          n_heads=hp.get("n_heads", 4),
                          n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    model.load_state_dict(ckpt["model"])
    print(f"# v84 nc-swap gate")
    print(f"# ckpt: {args.ckpt}  epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss','?')}")
    print(f"# device={device}\n")

    print(f"# loading v84_none bags ...")
    bags = load_v84_bags(args.n_bags, seed=args.seed)
    print(f"# n_bags picked: {len(bags)}\n")

    print(f"# loading Bridge RNAs from Durrant ...")
    bridges = load_bridges()
    print(f"# distinct Bridge RNAs: {len(bridges)}  "
          f"lens={sorted(set(len(b) for b in bridges))}\n")

    variants = [
        ("A_baseline",       "A",        None),
        ("B_bridge",         "B_bridge", bridges),
        ("C_synth_random",   "C",        None),
    ]
    results = {}
    for name, mode, br in variants:
        rng = random.Random(args.seed + hash(name) % 1000)
        jl = args.out_root / f"{name}.jsonl"
        with open(jl, "w") as f:
            for rec in build_variant(bags, mode, rng, bridges=br):
                f.write(json.dumps(rec) + "\n")
        shard = args.out_root / f"{name}_shard"
        scores = build_and_score(jl, shard, model, device)
        results[name] = scores
        print(f"  {name:>18s}  {dist_line(scores)}")
    print()

    # Also score real DurrantWT (K=1) end-to-end for comparison
    print(f"# scoring Durrant WT (K=1, full-real Durrant flank + Durrant nc) ...")
    dur_scores = score_existing_shard(DURRANT_K1_JSONL, DURRANT_K1_SHARD,
                                          model, device)
    print(f"  {'DurrantWT_K1':>18s}  {dist_line(dur_scores)}")
    results["DurrantWT_K1"] = dur_scores

    # Score prior condD (synth flank + Durrant nc — pre-generated crossover ref)
    print(f"# scoring condD (synth flank + Durrant Bridge nc — legacy reference) ...")
    condd_scores = score_existing_shard(CONDD_JSONL, CONDD_SHARD, model, device)
    print(f"  {'condD_legacy':>18s}  {dist_line(condd_scores)}")
    results["condD_legacy"] = condd_scores

    print(f"\n## summary — score p50 by variant")
    for name, scores in results.items():
        print(f"  {name:>18s}: p50 = {statistics.median(scores):.3f}  "
              f"mean = {statistics.mean(scores):.3f}")

    print(f"\n## Δ vs A_baseline (p50)")
    base_p50 = statistics.median(results["A_baseline"])
    for name, scores in results.items():
        if name == "A_baseline":
            continue
        d = statistics.median(scores) - base_p50
        print(f"  {name:>18s}: {d:+.3f}")

    print(f"\n# pre-registered reference (v8_main_v3 baseline, Sep-27):")
    print(f"#   condA v3(synth flank + synth nc)     p50 = 5.02")
    print(f"#   condD v3(synth flank + Durrant nc)   p50 = 1.89")
    print(f"# decision:")
    print(f"#   B_bridge ≈ A_baseline (within ~1 point) → Rfam GENERALIZED to Durrant → PASS")
    print(f"#   B_bridge ~2-3 below A_baseline → Rfam did NOT close Durrant gap → deeper issue")
    print(f"# v84 phase 7 gate complete")


if __name__ == "__main__":
    main()
