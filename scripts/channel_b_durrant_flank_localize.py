"""Durrant flank-side target localization test for v7real_main.

For each cohort × n_sites cell:
  1. Draw K Durrant records from a TBL group (shared bridge RNA target).
  2. Build a K-site bag: flanks from the records, nc = IS621 [nc_region_1, nc_region_2].
  3. Run v7real_main.best.pt → per-nc-position score. Peak nc pos = p*.
  4. Per site: compute flank_argmax at nc[p*:p*+L] via on-the-fly m_max.
     Orient chosen by max m_max at p* (fwd vs rc).
  5. Compare predicted flank pos to gold target_position_in_flank[0].
  6. Correct if |pred - gold| ≤ 3bp; also exact.

Random baseline (flank 120bp, L=11): 7 / (120-11+1) = 7/110 ≈ 6.4% at ±3bp,
1/110 ≈ 0.9% exact.

If accuracy climbs monotonically with n_sites, cross-site amplification
on real data mirrors the synth twin monotonic 0.86→0.98 result.
"""
from __future__ import annotations
import argparse
import json
import sys
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset, BagInputs, bucket_collate_fn
from model.channel_b.model import ChannelBModel
from model.channel_b.constants import MAX_L
from scripts.v5a_framework.match_table import _compute_site_arrays, ORIENTS

TARGET_L = 11
TOL_BP = 3


def load_records(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path)]


def group_by_tbl(recs: list[dict]) -> dict[str, list[dict]]:
    out = defaultdict(list)
    for r in recs:
        tbl = r["labels"].get("durrant_tbl_target_11bp")
        out[tbl].append(r)
    return out


def build_bag_from_records(records: list[dict]) -> BagInputs:
    """Package K Durrant records into a single BagInputs. All records
    share the SAME nc (IS621), so use the first record's nc."""
    from scripts.build_v7_shard import _build_bag_from_group   # if importable
    # Simpler: build tensor directly. Use the loader for a single-bag JSONL —
    # write a temp JSONL with all K sites sharing one transposase_id, build a
    # shard, load.
    raise NotImplementedError("prefer temp-JSONL path below")


def emit_multisite_jsonl(records: list[dict], bag_id: str, out_path: Path):
    """Emit K rows sharing one transposase_id so the loader groups them
    into a single bag with n_sites=K. All rows share the same nc (IS621)."""
    with open(out_path, "w") as f:
        for i, r in enumerate(records):
            new = dict(r)
            new["site_id"] = f"{bag_id}_site_{i:04d}"
            new["transposase_id"] = bag_id
            # Update arch.n_sites for consistency (loader may check)
            L = dict(new["labels"])
            arch = dict(L.get("arch", {}))
            arch["n_sites"] = len(records)
            L["arch"] = arch
            new["labels"] = L
            f.write(json.dumps(new) + "\n")


def predict_flank_pos(flank: str, nc_concat: str, p_star: int, L: int) -> tuple[int, int, str]:
    """Return (best_flank_start, m_max, orient) at nc[p_star:p_star+L]
    via on-the-fly m_max computation across both orients."""
    arrs = _compute_site_arrays(nc_concat, flank, ORIENTS, (L,), excl_widths=(0,))
    # For each orient, get the m_max at position p_star and the argmax flank position
    best_m = -1
    best_orient = None
    best_flank_start = -1
    for orient in ("fwd", "rc"):
        sa = arrs[(orient, L)]
        m_max_arr = sa.m_max_by_excl[0]  # shape (n_pos,)
        argmax_arr = sa.flank_argmax_by_excl[0]  # shape (n_pos,)
        if p_star < 0 or p_star >= len(m_max_arr):
            continue
        m = int(m_max_arr[p_star])
        fp = int(argmax_arr[p_star])
        if m > best_m:
            best_m = m
            best_orient = orient
            best_flank_start = fp
    return best_flank_start, best_m, best_orient


def eval_cell(model, device, records: list[dict], K: int, n_bags: int,
                    rng: random.Random, tmp_root: Path, shard_root: Path,
                    cell_tag: str) -> dict:
    """BATCHED variant: emit ONE JSONL with all n_bags K-site bags,
    build ONE shard, iterate through loader (avoids per-bag shard build
    overhead which was ~15s/bag → 8h for 2000 bags)."""
    from scripts.build_v7_shard import build_v7_shard
    from torch.utils.data import DataLoader

    tmp_jsonl = tmp_root / f"{cell_tag}.jsonl"
    tmp_shard = shard_root / f"{cell_tag}_shard"
    if tmp_shard.exists():
        import shutil; shutil.rmtree(tmp_shard)

    # Build batched JSONL: n_bags × K rows
    bags_of_picks = []
    with open(tmp_jsonl, "w") as f:
        for bag_i in range(n_bags):
            picks = rng.sample(records, K) if K <= len(records) \
                        else rng.choices(records, k=K)
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

    # ONE shard build for all n_bags
    build_v7_shard(tmp_jsonl, tmp_shard, min_sites=1,
                        family_label=cell_tag)

    ds = ChannelBDataset(tmp_jsonl, tmp_shard, preload_mt=True, cache_dir=None)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2,
                            collate_fn=bucket_collate_fn, pin_memory=True)

    n_correct_exact = 0
    n_correct_tol = 0
    n_total = 0
    peak_nc_pos_dist = []
    spacer = "N" * (MAX_L - 1)

    for batch in loader:
        x = batch["x"].to(device)
        sm = batch["site_mask"].to(device)
        pm = batch["pos_mask"].to(device)
        with torch.no_grad():
            pred = model(x, sm)   # (B, P)
        # argmax nc position within valid pos_mask
        pred_masked = pred + (-1e9) * (~pm).float()
        p_stars = pred_masked.argmax(dim=1).cpu().numpy()   # (B,)
        for i, meta in enumerate(batch["metas"]):
            bag_id = meta["bag_id"]
            # bag index within cell
            m = bag_id.split("_bag_")
            bag_i = int(m[-1])
            picks = bags_of_picks[bag_i]
            p_star = int(p_stars[i])
            peak_nc_pos_dist.append(p_star)
            regs = picks[0]["inputs"]["noncoding_regions"]
            nc_concat = spacer.join(regs)
            for r in picks:
                flank = r["inputs"]["flank"]
                gold_fp = r["labels"]["target_position_in_flank"][0]
                pred_fp, _m, _orient = predict_flank_pos(flank, nc_concat, p_star, TARGET_L)
                if pred_fp < 0:
                    continue
                n_total += 1
                if pred_fp == gold_fp:
                    n_correct_exact += 1
                if abs(pred_fp - gold_fp) <= TOL_BP:
                    n_correct_tol += 1

    return {
        "n_bags": n_bags,
        "n_sites_evaluated": n_total,
        "exact_acc":  n_correct_exact / n_total if n_total else 0.0,
        "tol3_acc":   n_correct_tol / n_total if n_total else 0.0,
        "peak_nc_pos": peak_nc_pos_dist,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path,
                     default=Path("checkpoints/channel_b/v7real_main/best.pt"))
    ap.add_argument("--wt-jsonl", type=Path,
                     default=Path("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_wt_realbg_v2.jsonl"))
    ap.add_argument("--prog-jsonl", type=Path,
                     default=Path("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_programmed_realbg_v2.jsonl"))
    ap.add_argument("--n-sites", type=str, default="1,3,5,8")
    ap.add_argument("--n-bags-per-cell", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tmp-root", type=Path,
                     default=Path("/global/scratch/users/kh36969/tmp/durrant_flank_eval"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hp = ckpt.get("hparams", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                                n_heads=hp.get("n_heads", 4),
                                n_blocks=hp.get("n_blocks", 3)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[flank-loc] device={device}  ckpt={args.ckpt}")

    ns_list = [int(x) for x in args.n_sites.split(",") if x.strip()]
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    tmp_jsonl_dir = args.tmp_root / "jsonl"; tmp_jsonl_dir.mkdir(exist_ok=True)
    tmp_shard_dir = args.tmp_root / "shard"; tmp_shard_dir.mkdir(exist_ok=True)

    wt_recs = load_records(args.wt_jsonl)
    prog_by_tbl = group_by_tbl(load_records(args.prog_jsonl))
    print(f"[flank-loc] WT: {len(wt_recs)} records; Programmed: {len(prog_by_tbl)} TBL groups "
              f"({[len(v) for v in prog_by_tbl.values()]})")

    # Random baseline
    n_flank_positions = 120 - TARGET_L + 1
    rand_exact = 1 / n_flank_positions
    rand_tol = (2 * TOL_BP + 1) / n_flank_positions
    print(f"[flank-loc] random baseline: exact={rand_exact:.4f}, ±{TOL_BP}bp={rand_tol:.4f}")

    rng = random.Random(args.seed)
    print(f"\n{'cohort':<25s} {'n_sites':>7s} {'n_bags':>7s} {'n_sites_ev':>10s} "
              f"{'exact_acc':>10s} {'±3bp_acc':>10s}")

    # WT
    for K in ns_list:
        res = eval_cell(model, device, wt_recs, K, args.n_bags_per_cell,
                             rng, tmp_jsonl_dir, tmp_shard_dir,
                             cell_tag=f"wt_K{K}")
        print(f"{'T-WT':<25s} {K:>7d} {res['n_bags']:>7d} {res['n_sites_evaluated']:>10d} "
                  f"{res['exact_acc']:>10.4f} {res['tol3_acc']:>10.4f}")

    # Programmed per TBL
    for tbl_i, (tbl, recs) in enumerate(sorted(prog_by_tbl.items(), key=lambda kv: -len(kv[1]))):
        for K in ns_list:
            res = eval_cell(model, device, recs, K, args.n_bags_per_cell,
                                 rng, tmp_jsonl_dir, tmp_shard_dir,
                                 cell_tag=f"prog_TBL{tbl_i}_K{K}")
            print(f"{'Programmed_'+tbl[:12]:<25s} {K:>7d} {res['n_bags']:>7d} "
                      f"{res['n_sites_evaluated']:>10d} "
                      f"{res['exact_acc']:>10.4f} {res['tol3_acc']:>10.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
