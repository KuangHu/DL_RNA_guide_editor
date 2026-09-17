"""Durrant flank-localization diagnostic (post-retraction of the
±3bp accuracy claim).

Two questions:
  1. Where do model predictions concentrate? If they cluster near a
     single position (e.g., 52), the exact=0 for T-WT is "off by 1
     from gold {51,53}", not "no signal". Report predicted-position
     histogram per cohort.
  2. Is the model output CONTENT-driven or POSITIONAL-bias-driven?
     For each real Durrant flank, also score the byte-shuffled version.
     If pred distribution stays the same on shuffle → pure positional
     bias. If it changes → content-driven signal exists; Δ(real, shuffle)
     is the honest number.

Both K=1 (single-site cleanest) and K=5 (best synth-analog cross-site
regime). Baseline for accuracy metrics: mode-of-predictions predictor
(NOT uniform random) — the honest baseline given the model's own
positional prior.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset, bucket_collate_fn
from model.channel_b.model import ChannelBModel
from model.channel_b.constants import MAX_L
from scripts.v5a_framework.match_table import _compute_site_arrays, ORIENTS
from scripts.build_v7_shard import build_v7_shard
from torch.utils.data import DataLoader

TARGET_L = 11


def load_records(path):
    return [json.loads(l) for l in open(path)]


def group_by_tbl(recs):
    out = defaultdict(list)
    for r in recs:
        out[r["labels"].get("durrant_tbl_target_11bp")].append(r)
    return out


def predict_flank_pos(flank, nc_concat, p_star, L):
    arrs = _compute_site_arrays(nc_concat, flank, ORIENTS, (L,), excl_widths=(0,))
    best_m, best_fp = -1, -1
    for orient in ("fwd", "rc"):
        sa = arrs[(orient, L)]
        m_max = sa.m_max_by_excl[0]
        argmax = sa.flank_argmax_by_excl[0]
        if 0 <= p_star < len(m_max):
            if int(m_max[p_star]) > best_m:
                best_m = int(m_max[p_star])
                best_fp = int(argmax[p_star])
    return best_fp


def shuffle_flank(flank, rng):
    """Byte-permute the flank (preserves base composition, destroys sequence)."""
    chars = list(flank)
    rng.shuffle(chars)
    return "".join(chars)


def eval_cohort(model, device, records, K, n_bags, rng, tmp_root, shard_root,
                    cell_tag, shuffle=False):
    """Score n_bags K-site bags. Return list of (pred_fp, gold_fp) tuples."""
    tmp_jsonl = tmp_root / f"{cell_tag}.jsonl"
    tmp_shard = shard_root / f"{cell_tag}_shard"
    if tmp_shard.exists():
        import shutil; shutil.rmtree(tmp_shard)

    bags_of_picks = []
    with open(tmp_jsonl, "w") as f:
        for bag_i in range(n_bags):
            picks = rng.sample(records, K) if K <= len(records) \
                        else rng.choices(records, k=K)
            # deep-copy + shuffle flank if needed
            if shuffle:
                new_picks = []
                for p in picks:
                    q = dict(p)
                    qi = dict(q["inputs"])
                    qi["flank"] = shuffle_flank(q["inputs"]["flank"], rng)
                    q["inputs"] = qi
                    new_picks.append(q)
                picks = new_picks
            bags_of_picks.append(picks)
            bag_id = f"{cell_tag}_bag_{bag_i:04d}"
            for i, r in enumerate(picks):
                new = dict(r)
                new["site_id"] = f"{bag_id}_site_{i:04d}"
                new["transposase_id"] = bag_id
                L = dict(new["labels"])
                arch = dict(L.get("arch", {}))
                arch["n_sites"] = K
                L["arch"] = arch
                new["labels"] = L
                f.write(json.dumps(new) + "\n")

    build_v7_shard(tmp_jsonl, tmp_shard, min_sites=1, family_label=cell_tag)
    ds = ChannelBDataset(tmp_jsonl, tmp_shard, preload_mt=True, cache_dir=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                            collate_fn=bucket_collate_fn, pin_memory=True)

    results = []   # (pred_fp, gold_fp)
    spacer = "N" * (MAX_L - 1)
    for batch in loader:
        x = batch["x"].to(device); sm = batch["site_mask"].to(device)
        pm = batch["pos_mask"].to(device)
        with torch.no_grad():
            pred = model(x, sm)
        pred_masked = pred + (-1e9) * (~pm).float()
        p_stars = pred_masked.argmax(dim=1).cpu().numpy()
        for i, meta in enumerate(batch["metas"]):
            bag_i = int(meta["bag_id"].split("_bag_")[-1])
            picks = bags_of_picks[bag_i]
            p_star = int(p_stars[i])
            regs = picks[0]["inputs"]["noncoding_regions"]
            nc_concat = spacer.join(regs)
            for r in picks:
                pred_fp = predict_flank_pos(r["inputs"]["flank"], nc_concat, p_star, TARGET_L)
                gold_fp = r["labels"]["target_position_in_flank"][0]
                results.append((pred_fp, gold_fp))
    return results


def summarize(name, results):
    if not results:
        print(f"  {name}: NO RESULTS")
        return
    preds = [p for p, _ in results]
    golds = [g for _, g in results]
    pred_dist = Counter(preds)
    gold_dist = Counter(golds)
    n = len(results)
    n_correct_0 = sum(1 for p, g in results if p == g)
    n_correct_1 = sum(1 for p, g in results if abs(p - g) <= 1)
    n_correct_3 = sum(1 for p, g in results if abs(p - g) <= 3)
    # signed offset distribution — check for systematic +1/-1 shift
    offset_dist = Counter(p - g for p, g in results)
    top_offsets = offset_dist.most_common(5)
    # exact accuracy at each shift ∈ {-2, -1, 0, +1, +2}
    shift_exacts = {}
    for s in (-2, -1, 0, 1, 2):
        shift_exacts[s] = sum(1 for p, g in results if (p + s) == g) / n
    # mode-of-pred trivial baseline
    mode_pred, _ = pred_dist.most_common(1)[0]
    triv_0 = sum(1 for g in golds if mode_pred == g)
    triv_1 = sum(1 for g in golds if abs(mode_pred - g) <= 1)
    top_preds = pred_dist.most_common(5)
    print(f"\n  === {name} (n={n}) ===")
    print(f"    gold distribution:  {dict(sorted(gold_dist.items()))}")
    print(f"    top-5 pred positions: {top_preds}")
    print(f"    signed offset (pred - gold) top-5: {top_offsets}")
    print(f"    exact after shift  s=-2:{shift_exacts[-2]:.4f}  s=-1:{shift_exacts[-1]:.4f}  "
              f"s=0:{shift_exacts[0]:.4f}  s=+1:{shift_exacts[+1]:.4f}  s=+2:{shift_exacts[+2]:.4f}")
    print(f"    Model:  exact={n_correct_0/n:.4f}  ±1bp={n_correct_1/n:.4f}  ±3bp={n_correct_3/n:.4f}")
    print(f"    'always predict {mode_pred}' trivial baseline: "
              f"exact={triv_0/n:.4f}  ±1bp={triv_1/n:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                     default=Path("checkpoints/channel_b/v7real_main/best.pt"))
    ap.add_argument("--wt-jsonl", type=Path,
                     default=Path("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_wt_realbg_v2.jsonl"))
    ap.add_argument("--prog-jsonl", type=Path,
                     default=Path("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_programmed_realbg_v2.jsonl"))
    ap.add_argument("--n-bags-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-root", type=Path,
                     default=Path("/global/scratch/users/kh36969/tmp/durrant_flank_diag"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp = ckpt.get("hparams", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                                n_heads=hp.get("n_heads", 4),
                                n_blocks=hp.get("n_blocks", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[diag] device={device}")

    args.tmp_root.mkdir(parents=True, exist_ok=True)
    tj = args.tmp_root / "jsonl"; tj.mkdir(exist_ok=True)
    ts = args.tmp_root / "shard"; ts.mkdir(exist_ok=True)

    wt_recs = load_records(args.wt_jsonl)
    prog_all = load_records(args.prog_jsonl)
    rng = random.Random(args.seed)

    for K in (1, 5):
        # REAL runs
        r_wt = eval_cohort(model, device, wt_recs, K, args.n_bags_per_cell,
                                rng, tj, ts, cell_tag=f"real_wt_K{K}", shuffle=False)
        r_pr = eval_cohort(model, device, prog_all, K, args.n_bags_per_cell,
                                rng, tj, ts, cell_tag=f"real_prog_K{K}", shuffle=False)
        # SHUFFLE runs (same cohorts, flanks byte-permuted)
        s_wt = eval_cohort(model, device, wt_recs, K, args.n_bags_per_cell,
                                rng, tj, ts, cell_tag=f"shuf_wt_K{K}", shuffle=True)
        s_pr = eval_cohort(model, device, prog_all, K, args.n_bags_per_cell,
                                rng, tj, ts, cell_tag=f"shuf_prog_K{K}", shuffle=True)

        print(f"\n\n############ K = {K} ############")
        summarize(f"WT REAL   K={K}", r_wt)
        summarize(f"WT SHUF   K={K}", s_wt)
        summarize(f"PROG REAL K={K}", r_pr)
        summarize(f"PROG SHUF K={K}", s_pr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
