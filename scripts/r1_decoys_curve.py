"""R1 aggregation-collapse curve: AUROC(N_decoys).

For each cognate/shuffled pair, sweep N_decoys ∈ {0, 1, 3, 7, 19, 49, 95} and
measure how the pair scorer's AUROC evolves. Endpoints should match:

  N=0:   ~gold-only AUROC (previously 0.877 on this ckpt)
  N=95:  ~raw E2E AUROC

Intermediate N shows how quickly the gold's signal is drowned as more decoys
are added. Cognate bag: gold in slot 0 + N top decoys in slots 1..N.
Shuffled bag: top-(N+1) decoys in slots 0..N (no gold available).

All slots not in the active-N are masked out; each bag has exactly N+1
valid candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from model.v1 import V1Config, V1Model
from preprocess.candidates import (
    build_candidate_arrays, _fill_candidate_slot, Candidate,
    DEFAULT_L_MIN, DEFAULT_L_MAX, NUM_FEATURES, encode_dna,
)
from preprocess.site import DEFAULT_NC_MAX, DEFAULT_NUM_NC_SLOTS


def _find_gold_slot(feats, mask, cands, gold, overlap_frac=0.5):
    valid = np.where(mask)[0]
    best = -1; best_matches = -1
    for i in valid:
        c = cands[i]
        if c is None or c.orient != gold.orient: continue
        min_L = min(c.L, gold.L)
        ov_f = max(0, min(c.flank_start + c.L, gold.flank_start + gold.L) - max(c.flank_start, gold.flank_start))
        ov_n = max(0, min(c.nc_start + c.L, gold.nc_start + gold.L) - max(c.nc_start, gold.nc_start))
        thresh = overlap_frac * min_L
        if ov_f >= thresh and ov_n >= thresh and feats[i, 3] > best_matches:
            best = int(i); best_matches = float(feats[i, 3])
    return best


def build_bag_with_N_decoys(nc: str, flank: str, N: int,
                              gold: Candidate | None,
                              nc_slot: int,
                              num_nc_slots: int,
                              patch_channels: int, patch_width: int,
                              nc_max: int):
    """Return (patches, feats, mask, nc_region_mask) for a single-site bag with
    exactly N+1 valid candidates (gold + top-N decoys) if gold is provided,
    else (N+1 top decoys)."""
    prof = np.zeros((len(nc), 16), dtype=np.float32)
    val = np.zeros((len(nc), 16), dtype=bool)
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX,
    )
    K = patches.shape[0]
    W = patches.shape[1]; C = patches.shape[2]

    # Get sorted decoy slots (excluding gold-equivalent slot if gold given)
    gold_slot_in_pool = -1
    if gold is not None:
        gold_slot_in_pool = _find_gold_slot(feats, mask, cands, gold)
    matches = feats[:, 3]
    valid_slots = np.where(mask)[0]
    # Exclude gold if present
    decoy_pool = [i for i in valid_slots if i != gold_slot_in_pool]
    # Sort by matches descending
    decoy_pool.sort(key=lambda i: -matches[i])
    # Cognate: gold + N decoys = N+1 total; Shuffled: (N+1) top decoys = N+1 total.
    # Both bags always have the same number of valid candidates.
    n_decoys_to_place = N if gold is not None else (N + 1)
    top_decoys = decoy_pool[:n_decoys_to_place]

    # Construct padded arrays (num_nc_slots, K, W, C) with active slot at nc_slot
    p_out = np.zeros((num_nc_slots, K, W, C), dtype=np.float32)
    f_out = np.zeros((num_nc_slots, K, NUM_FEATURES), dtype=np.float32)
    m_out = np.zeros((num_nc_slots, K), dtype=bool)
    nc_region_mask = np.zeros(num_nc_slots, dtype=bool)
    nc_region_mask[nc_slot] = True

    # Copy top decoys into slots 1..N (or 0..N-1 if no gold)
    start_slot = 1 if gold is not None else 0
    for k, i in enumerate(top_decoys):
        s = start_slot + k
        if s >= K: break
        p_out[nc_slot, s] = patches[i]
        f_out[nc_slot, s] = feats[i]
        m_out[nc_slot, s] = True

    if gold is not None:
        # Fill gold at slot 0 via _fill_candidate_slot
        nc_codes = encode_dna(nc)
        flank_codes = encode_dna(flank)
        p_slot = p_out[nc_slot]
        f_slot = f_out[nc_slot]
        m_slot = m_out[nc_slot]
        _fill_candidate_slot(
            p_slot, f_slot, m_slot, 0,
            nc_codes, flank_codes, prof, val, gold, W, nc_max,
        )
        p_out[nc_slot] = p_slot
        f_out[nc_slot] = f_slot
        m_out[nc_slot] = m_slot

    return p_out, f_out, m_out, nc_region_mask


def infer_bag(model, patches_np, feats_np, mask_np, nc_region_mask_np, device,
                nc_slot: int = 0, primary_slot: int = 0):
    """Run model on bag and extract pair score + candidate attention stats.

    primary_slot is the slot that holds the reference candidate (gold for
    cognate, top-1 decoy for shuffled). Returns:
      s_pair, primary_attn_weight, primary_attn_rank, attn_entropy,
      top1_attn_mass, top3_attn_mass
    """
    patches = torch.from_numpy(patches_np[None, None, :, :, :, :]).to(device)
    feats   = torch.from_numpy(feats_np[None, None, :, :, :]).to(device)
    mask    = torch.from_numpy(mask_np[None, None, :, :]).to(device)
    nc_reg  = torch.from_numpy(nc_region_mask_np[None, None, :]).to(device)
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=(device.type == "cuda")):
            out = model(patches, feats, mask, nc_reg)
    s_pair = float(out["s_pair_aux"].detach().float().cpu().item())
    cand_attn = out["cand_attn"].detach().float().cpu().numpy()  # (1, 1, N_nc, K)
    attn_row = cand_attn[0, 0, nc_slot]                          # (K,)
    valid = mask_np[nc_slot]                                     # (K,)
    attn_valid = attn_row.copy()
    attn_valid[~valid] = 0.0
    s = attn_valid.sum()
    if s > 0:
        attn_valid = attn_valid / s
    else:
        attn_valid = np.zeros_like(attn_valid)
    n_valid = int(valid.sum())
    if n_valid == 0:
        return {
            "s_pair": s_pair, "primary_weight": None, "primary_rank": None,
            "entropy": None, "top1_mass": None, "top3_mass": None, "n_valid": 0,
        }
    primary_weight = float(attn_valid[primary_slot]) if valid[primary_slot] else 0.0
    # Rank of primary among valid slots (1-indexed, higher weight = better rank)
    valid_slots = np.where(valid)[0]
    w = attn_valid[valid_slots]
    primary_rank = int((w > primary_weight).sum() + 1) if valid[primary_slot] else None
    eps = 1e-12
    entropy = float(-(w * np.log(w + eps)).sum())
    sorted_w = np.sort(w)[::-1]
    top1_mass = float(sorted_w[:1].sum())
    top3_mass = float(sorted_w[:3].sum())
    return {
        "s_pair": s_pair, "primary_weight": primary_weight, "primary_rank": primary_rank,
        "entropy": entropy, "top1_mass": top1_mass, "top3_mass": top3_mass, "n_valid": n_valid,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cognate-jsonl", required=True)
    ap.add_argument("--shuffled-jsonl", required=True)
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--N-values", type=str, default="0,1,3,7,19,49,95")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    use_additive = "alpha_pair" in state or "alpha_geom" in state
    use_orient = any(k.startswith("orient_mlp.") or k.startswith("h_orient_aux.") for k in state.keys())
    v1_cfg = V1Config(use_multi_branch=True, use_explicit_geom_stats=True,
                       use_additive_fusion=use_additive, use_orient_branch=use_orient)
    model = V1Model(v1_cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    gold = {}
    with open(args.gold_jsonl) as f:
        for line in f:
            r = json.loads(line); gold[r["site_id"]] = r
    print(f"[gold] {len(gold)}", flush=True)

    N_values = [int(x) for x in args.N_values.split(",")]

    # Cache records
    def _load(path):
        out = []
        with open(path) as f:
            for line in f:
                r = json.loads(line); out.append(r)
        return out
    cog_records = _load(args.cognate_jsonl)
    shu_records = _load(args.shuffled_jsonl)

    # For each N, compute cognate and shuffled scores + AUROC + paired Δ
    from sklearn.metrics import roc_auc_score
    reports = {}
    for N in N_values:
        print(f"\n--- N_decoys = {N} ---", flush=True)
        cog_scores = []; cog_tnps = []
        shu_scores = []; shu_tnps = []
        cog_attn = []  # list of dicts: primary_weight, primary_rank, entropy, top1_mass, top3_mass
        shu_attn = []

        for r in cog_records:
            sid = r["site_id"]
            g = gold.get(sid)
            if g is None: continue
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]; flank = r["inputs"]["flank"]
            if not nc or not flank: continue
            gcand = Candidate(
                orient=g["target_flank_orientation"], L=g["target_binding_loop_length"],
                nc_start=g["guide_start_in_nc"], flank_start=g["target_flank_start"],
                matches=g["target_flank_matches"],
            )
            p, f, m, nrm = build_bag_with_N_decoys(
                nc, flank, N, gcand, active_nc, DEFAULT_NUM_NC_SLOTS,
                patch_channels=None, patch_width=None, nc_max=DEFAULT_NC_MAX,
            )
            info = infer_bag(model, p, f, m, nrm, device, nc_slot=active_nc, primary_slot=0)
            cog_scores.append(info["s_pair"]); cog_tnps.append(r["transposase_id"])
            cog_attn.append(info)

        for r in shu_records:
            sid = r["site_id"]; gold_sid = sid.replace("_shu_", "_cog_")
            g = gold.get(gold_sid)
            if g is None: continue
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]; flank = r["inputs"]["flank"]
            if not nc or not flank: continue
            p, f, m, nrm = build_bag_with_N_decoys(
                nc, flank, N, None, active_nc, DEFAULT_NUM_NC_SLOTS,
                patch_channels=None, patch_width=None, nc_max=DEFAULT_NC_MAX,
            )
            info = infer_bag(model, p, f, m, nrm, device, nc_slot=active_nc, primary_slot=0)
            shu_scores.append(info["s_pair"]); shu_tnps.append(r["transposase_id"])
            shu_attn.append(info)

        # AUROC (site-level pooled)
        scores = np.asarray(cog_scores + shu_scores, dtype=np.float32)
        labels = np.asarray([1] * len(cog_scores) + [0] * len(shu_scores))
        au = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else float("nan")

        # Paired Δ by parent tnp
        cog_by = defaultdict(list); shu_by = defaultdict(list)
        for t, s in zip(cog_tnps, cog_scores): cog_by[t].append(s)
        for t, s in zip(shu_tnps, shu_scores): shu_by[t].append(s)
        cog_avg = {t: float(np.mean(v)) for t, v in cog_by.items()}
        shu_avg = {t: float(np.mean(v)) for t, v in shu_by.items()}
        deltas = []
        for t, sc in cog_avg.items():
            neg = t.replace("_cog_", "_shu_")
            if neg in shu_avg:
                deltas.append(sc - shu_avg[neg])
        arr = np.asarray(deltas, dtype=np.float32)
        paired = {
            "n": int(len(arr)),
            "median": float(np.median(arr)) if len(arr) else None,
            "MAD":    float(np.median(np.abs(arr - np.median(arr)))) if len(arr) else None,
            "p_gt_0": float((arr > 0).mean()) if len(arr) else None,
        }
        # Attention summary (COGNATE — primary_slot is gold)
        def _attn_summary(rows, tag):
            wt = [r["primary_weight"] for r in rows if r["primary_weight"] is not None]
            rk = [r["primary_rank"] for r in rows if r["primary_rank"] is not None]
            en = [r["entropy"] for r in rows if r["entropy"] is not None]
            t1 = [r["top1_mass"] for r in rows if r["top1_mass"] is not None]
            t3 = [r["top3_mass"] for r in rows if r["top3_mass"] is not None]
            return {
                "median_primary_weight": float(np.median(wt)) if wt else None,
                "median_primary_rank":   float(np.median(rk)) if rk else None,
                "median_entropy":        float(np.median(en)) if en else None,
                "median_top1_mass":      float(np.median(t1)) if t1 else None,
                "median_top3_mass":      float(np.median(t3)) if t3 else None,
                "n":                     len(wt),
            }
        cog_attn_sum = _attn_summary(cog_attn, "cognate")
        shu_attn_sum = _attn_summary(shu_attn, "shuffled")
        reports[N] = {"AUROC": au, "paired": paired,
                       "cog_attn": cog_attn_sum, "shu_attn": shu_attn_sum}
        print(f"  N={N:>3}  AUROC={au:.4f}  paired: n={paired['n']}  median={paired['median']:+.3f}  P(Δ>0)={paired['p_gt_0']:.3f}", flush=True)
        pw = cog_attn_sum["median_primary_weight"]
        pr = cog_attn_sum["median_primary_rank"]
        en = cog_attn_sum["median_entropy"]
        t1m = cog_attn_sum["median_top1_mass"]
        if pw is not None:
            print(f"          COG attention: gold_weight={pw:.3f}  gold_rank={pr}  entropy={en:.3f}  top1_mass={t1m:.3f}", flush=True)
        else:
            print("          COG attention: (empty)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(reports, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
