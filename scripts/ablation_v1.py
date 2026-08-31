"""V1 interpretability / ablation matrix (inference-only variants).

Runs the trained V1 checkpoint on the held-out test split under a suite
of ablations that mask or permute individual signals. For each ablation
we report:

  - Tnp AUROC + AUPRC (overall)
  - Per-violation-profile stratified AUROC
  - Candidate Recall@1 / @5 / @10 (positives with known ground truth)
  - NC selection top-1
  - Ground-truth attention statistics (mean attention weight on the
    planted candidate; mean rank of the planted candidate)

Ablations (all applied at inference; no retraining):

  none                   — the reference (should reproduce test AUPRC=1.0)
  no_alignment           — zero features [0..5] (orient/L/matches/mm/score)
                            and patch channels [17..21] (guide_mask,
                            match_state, paired_flank_pos, align_position)
                            Retains: structure channels + NC position features
  no_structure           — zero patch channels [0..16] (nc_unp_u1..u16 +
                            struct_valid). Retains: alignment + positions.
  no_position            — zero features [6..10] (flank_start / flank_end /
                            boundary_dist_up/dn / target_side_up) and
                            patch channel [20] (paired_flank_pos_norm).
                            Retains: alignment, structure, NC-internal
                            positions.
  no_set_transformer     — bypass the SAB blocks in the tnp branch. PMA
                            gets raw per-site tokens with no cross-site
                            attention.
  permute_structure      — for each tnp, shuffle the structure channels
                            (0..16) of each candidate across sites of the
                            same tnp (site's alignment now sees a random
                            other site's structural context).

Test 9 (ground-truth candidate tracking) is computed for every ablation
so we can see which ablations wreck the model's ability to point at the
planted candidate.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import (
    TnpGroupedDataset,
    collate_tnp_batch,
    make_torch_tnp_dataset,
)
from training.metrics import (
    _auroc,
    _auprc,
    candidate_recall,
    nc_selection_accuracy,
    stratified_auroc,
    tnp_metrics,
)


# Channel indices in the (nc_unp_u1..u16, struct_valid, guide_mask, match_state_match,
# match_state_mismatch, paired_flank_pos_norm, align_position_in_guide) patch layout.
STRUCT_CHANNELS = slice(0, 17)     # nc_unp_u1..u16 + struct_valid
ALIGN_CHANNELS  = slice(17, 22)    # guide_mask ... align_position_in_guide
PAIRED_FLANK_CHANNEL = 20

# Feature indices (see model/v1.py::FEATURE_NAMES).
ALIGN_FEATURE_INDICES = list(range(0, 6))    # orient_{fwd,rc}, L, matches, mismatches, score
POS_FEATURE_INDICES   = list(range(6, 11))   # flank_start/end/boundary_up/dn/target_side
# nc_start_norm, nc_len_norm at 11, 12 are NOT masked (they're NC-internal)


def apply_ablation(batch: dict, mode: str, rng: np.random.Generator) -> dict:
    """Return a new batch dict with the requested ablation applied.

    Non-tensor entries (site_id, tnp_id, ...) are passed through by
    reference. Tensor entries that are modified are cloned first.
    """
    if mode == "none":
        return batch

    patches = batch["candidate_patches"]
    feats   = batch["candidate_features"]

    if mode == "no_alignment":
        patches = patches.clone()
        feats = feats.clone()
        patches[..., ALIGN_CHANNELS] = 0
        feats[..., ALIGN_FEATURE_INDICES] = 0
        return {**batch, "candidate_patches": patches, "candidate_features": feats}

    if mode == "no_structure":
        patches = patches.clone()
        patches[..., STRUCT_CHANNELS] = 0
        return {**batch, "candidate_patches": patches}

    if mode == "no_position":
        feats = feats.clone()
        patches = patches.clone()
        feats[..., POS_FEATURE_INDICES] = 0
        patches[..., PAIRED_FLANK_CHANNEL] = 0
        return {**batch, "candidate_features": feats, "candidate_patches": patches}

    if mode == "no_set_transformer":
        # Handled at forward time (model wrapper). No batch mutation.
        return batch

    if mode == "permute_structure":
        # For each tnp in batch, permute the structure channels across sites.
        # candidate_patches: (B, S, N, K, W, C)
        p = patches.clone()
        B, S = p.shape[:2]
        for b in range(B):
            perm = torch.from_numpy(rng.permutation(S))
            p[b, :, :, :, :, STRUCT_CHANNELS] = p[b, perm, :, :, :, STRUCT_CHANNELS]
        return {**batch, "candidate_patches": p}

    raise ValueError(f"unknown ablation mode {mode!r}")


def forward(model, batch, device, use_bf16=True, skip_set=False):
    """Run V1 forward, optionally skipping the set-transformer blocks."""
    if skip_set:
        saved = model.set_blocks
        model.set_blocks = torch.nn.ModuleList([])
    try:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=(use_bf16 and device.type == "cuda")):
            out = model(
                batch["candidate_patches"],
                batch["candidate_features"],
                batch["candidate_mask"],
                batch["nc_region_mask"],
            )
    finally:
        if skip_set:
            model.set_blocks = saved
    return out


def group_map(path: str) -> dict[str, str]:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r["transposase_id"]
            if t not in out:
                out[t] = "positive" if r["labels"].get("is_positive") else (
                    r["labels"].get("violation_profile") or "unknown"
                )
    return out


def gt_attention_stats(cand_attn_active: np.ndarray, true_slot: np.ndarray) -> dict:
    """Given (N_pos, K) attention weights at the active NC slot and (N_pos,)
    true slot indices, report:
      - mean_attn_true:   average attention on the true candidate
      - mean_attn_random: average attention over all other candidates
      - median_rank:      median rank of the true candidate (0-indexed)
      - top1_share:       mean(attn[true] > max(attn[other]))
    """
    A = cand_attn_active
    T = true_slot
    N, K = A.shape
    attn_true = A[np.arange(N), T]
    other_mask = np.ones_like(A, dtype=bool)
    other_mask[np.arange(N), T] = False
    attn_other = (A * other_mask).sum(-1) / max(1, K - 1)
    # rank of true (higher attention = better rank)
    order = np.argsort(-A, axis=-1, kind="stable")
    ranks = np.empty(N, dtype=np.int64)
    for i in range(N):
        ranks[i] = int(np.where(order[i] == T[i])[0][0])
    top1_share = float((ranks == 0).mean())
    return {
        "mean_attn_true": float(np.mean(attn_true)),
        "mean_attn_other": float(np.mean(attn_other)),
        "median_rank": float(np.median(ranks)),
        "top1_share": top1_share,
        "n": int(N),
    }


def evaluate_one_pass(
    model, dl, device, gmap, ablation_modes, rng, use_bf16=True, max_batches=None,
):
    """For each batch, run the model in EACH ablation mode. Collect
    per-tnp scores + labels, and per-positive-site aux info, per mode.
    """
    aggr: dict[str, dict] = {
        m: {
            "scores": [],
            "labels": [],
            "tnp_ids": [],
            "cand_active": [],  # (N_pos, K) attention at active NC slot
            "cand_active_raw": [],
            "true_slot": [],
            "active_nc": [],
            "nc_attn": [],
        }
        for m in ablation_modes
    }

    n_batches = 0
    t0 = time.time()
    for batch in dl:
        # Push tensors to device.
        batch = {
            k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        for mode in ablation_modes:
            skip_set = (mode == "no_set_transformer")
            b_ab = apply_ablation(batch, mode, rng)
            with torch.no_grad():
                out = forward(model, b_ab, device, use_bf16=use_bf16, skip_set=skip_set)
            aggr[mode]["scores"].append(
                torch.sigmoid(out["logit"]).float().cpu().numpy()
            )
            aggr[mode]["labels"].append(batch["is_positive"].cpu().numpy())
            aggr[mode]["tnp_ids"].extend(list(batch["tnp_id"]))

            cand_attn = out["cand_attn"].float().cpu().numpy()   # (B, S, N, K)
            cand_raw  = out["cand_raw"].float().cpu().numpy()
            nc_attn   = out["nc_attn"].float().cpu().numpy()      # (B, S, N)
            active    = batch["active_nc_index"].cpu().numpy()    # (B, S)
            true_slot = batch["true_slot_idx"].cpu().numpy()      # (B, S)
            B_, S_, N_, K_ = cand_attn.shape
            for bi in range(B_):
                for si in range(S_):
                    anc = int(active[bi, si])
                    ts = int(true_slot[bi, si])
                    if anc < 0 or ts < 0:
                        continue
                    aggr[mode]["cand_active"].append(cand_attn[bi, si, anc])
                    aggr[mode]["cand_active_raw"].append(cand_raw[bi, si, anc])
                    aggr[mode]["true_slot"].append(ts)
                    aggr[mode]["active_nc"].append(anc)
                    aggr[mode]["nc_attn"].append(nc_attn[bi, si])
        n_batches += 1
        if max_batches is not None and n_batches >= max_batches:
            break
    dt = time.time() - t0

    # Reduce.
    reports = {}
    for mode, buckets in aggr.items():
        scores = np.concatenate(buckets["scores"])
        labels = np.concatenate(buckets["labels"])
        tnp_ids = buckets["tnp_ids"]
        groups = np.asarray([gmap.get(t, "unknown") for t in tnp_ids])

        m = tnp_metrics(scores, labels)
        strat = stratified_auroc(scores, labels, groups)

        if buckets["cand_active"]:
            cand_attn_active = np.stack(buckets["cand_active"], 0)
            cand_raw_active  = np.stack(buckets["cand_active_raw"], 0)
            true_slots = np.asarray(buckets["true_slot"], dtype=np.int64)
            active_ncs = np.asarray(buckets["active_nc"], dtype=np.int64)
            nc_attns = np.stack(buckets["nc_attn"], 0)
            r = candidate_recall(cand_raw_active, true_slots, ks=(1, 5, 10))
            nc = nc_selection_accuracy(nc_attns, active_ncs)
            attn_stats = gt_attention_stats(cand_attn_active, true_slots)
        else:
            r = {f"recall@{k}": float("nan") for k in (1, 5, 10)} | {"n": 0}
            nc = {"nc_top1": float("nan"), "n": 0}
            attn_stats = {"mean_attn_true": float("nan"), "mean_attn_other": float("nan"),
                           "median_rank": float("nan"), "top1_share": float("nan"), "n": 0}

        reports[mode] = {
            "n_tnp_pos": m["n_pos"],
            "n_tnp_neg": m["n_neg"],
            "auroc": m["auroc"],
            "auprc": m["auprc"],
            **strat,
            **r,
            "nc_top1": nc["nc_top1"],
            "attn_true": attn_stats["mean_attn_true"],
            "attn_other": attn_stats["mean_attn_other"],
            "attn_median_rank": attn_stats["median_rank"],
            "attn_top1_share": attn_stats["top1_share"],
        }
    reports["_meta"] = {"n_batches": n_batches, "elapsed_s": dt}
    return reports


def format_report(reports: dict) -> str:
    """Compact matrix: rows = ablation mode, cols = metric."""
    modes = [m for m in reports if not m.startswith("_")]
    profiles = sorted({k[6:-1] for k in reports[modes[0]] if k.startswith("auroc[")})

    lines = []
    hdr = (
        f"{'ablation':<22} {'AUROC':>7} {'AUPRC':>7} {'R@1':>6} {'R@5':>6} "
        f"{'NC':>5} {'attnT':>7} {'attnR':>7} {'rank':>5} {'top1':>6}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for m in modes:
        r = reports[m]
        lines.append(
            f"{m:<22} {r['auroc']:>7.4f} {r['auprc']:>7.4f} "
            f"{r['recall@1']:>6.3f} {r['recall@5']:>6.3f} "
            f"{r['nc_top1']:>5.3f} {r['attn_true']:>7.4f} "
            f"{r['attn_other']:>7.4f} {r['attn_median_rank']:>5.1f} "
            f"{r['attn_top1_share']:>6.3f}"
        )
    lines.append("")
    lines.append("per-profile AUROC:")
    lines.append(f"{'ablation':<22} " + " ".join(f"{p[:10]:>12}" for p in profiles))
    lines.append("-" * (22 + 13 * len(profiles)))
    for m in modes:
        r = reports[m]
        vals = " ".join(
            f"{r.get(f'auroc[{p}]', float('nan')):>12.4f}" for p in profiles
        )
        lines.append(f"{m:<22} {vals}")
    lines.append("")
    lines.append(f"[meta] {reports['_meta']['n_batches']} batches in "
                  f"{reports['_meta']['elapsed_s']:.1f}s")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split-jsonl", required=True)
    p.add_argument("--structure-index", required=True)
    p.add_argument("--out-json", default=None)
    p.add_argument("--tnp-batch", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None,
                    help="cap batches for quick smoke")
    p.add_argument("--modes", default="none,no_alignment,no_structure,no_position,"
                                          "no_set_transformer,permute_structure")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt["cfg"])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[eval] loaded {args.ckpt} (epoch {ckpt['epoch']}, saved AUPRC={ckpt['auprc']:.4f})",
          flush=True)

    cache = StructureCache(args.structure_index)
    ds = TnpGroupedDataset(args.split_jsonl, cache, site_subsample_size=50, rng_seed=args.seed)
    print(f"[eval] {len(ds)} tnps", flush=True)
    dl = DataLoader(
        make_torch_tnp_dataset(ds),
        batch_size=args.tnp_batch, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
        persistent_workers=(args.num_workers > 0),
        pin_memory=True,
    )
    gmap = group_map(args.split_jsonl)
    modes = args.modes.split(",")
    rng = np.random.default_rng(args.seed)

    reports = evaluate_one_pass(
        model, dl, device, gmap, modes, rng,
        use_bf16=True, max_batches=args.max_batches,
    )
    text = format_report(reports)
    print()
    print(text)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(reports, f, indent=2)
        print(f"\n[eval] wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
