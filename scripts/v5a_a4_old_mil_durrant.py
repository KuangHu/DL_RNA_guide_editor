"""V5A A4: Old MIL (48c1hA epoch_04) per-candidate attention on frozen Durrant.

Answers: does the old MIL's implicit selector (via GatedAttentionMIL attention
weights over candidates) beat raw_m 0.089 / length_pen 0.125 on Durrant?

Approach:
  1. Load 48c1hA epoch_04.pt (frozen).
  2. For each Durrant cognate Tnp, build the standard model input via
     TnpGroupedDataset + collate_tnp_batch (structure cache included).
  3. Forward pass returning cand_attn: (B, S, N_nc, K).
  4. Per site, extract cand_attn[b, s, active_nc, :]  =  K attention weights.
  5. Identify gold candidate slot from build_candidate_arrays + tolerant match.
     Cross-map between the model's K slot indices and the candidate slots.
  6. Rank the gold slot by attention weight (higher = more attended).
  7. Report pooled MRR + R@K with Tnp-clustered CI.

Non-blocking; nice-to-have to check whether the trained MIL captured signal on
real data that raw_m / length_pen do not.
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
from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX
from preprocess.tnp_dataset import (
    TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset,
)
from preprocess.site import StructureCache
from torch.utils.data import DataLoader


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _find_gold_slot(feats, mask, cands, orient, L, nc, fl, of=0.5):
    valid = np.where(mask)[0]
    if len(valid) == 0: return -1, 0.0
    matches = feats[:, 3]
    best = -1; best_m = -1.0
    for i in valid:
        c = cands[int(i)]
        if c.orient != orient: continue
        mn = min(c.L, L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc, nc + L)
        f_ov = _overlap(c.flank_start, c.flank_start + c.L, fl, fl + L)
        if nc_ov < of*mn or f_ov < of*mn: continue
        if matches[i] > best_m: best_m = float(matches[i]); best = int(i)
    return best, best_m


def _rank_stats(qs, cs_local, valid_mask=None, k_list=(1,4,8)):
    """Expected R@k + MRR under uniform tie-break, only among valid slots."""
    if valid_mask is not None:
        qs = qs[valid_mask]
        valid_idx = np.where(valid_mask)[0]
        # find where cs_local appears in the valid subset
        cs_pos_in_valid = int(np.where(valid_idx == cs_local)[0][0])
        cs_local = cs_pos_in_valid
    q_cs = qs[cs_local]
    other = np.delete(qs, cs_local)
    n_gt = int((other > q_cs).sum()); n_eq = int((other == q_cs).sum())
    tie = n_eq + 1
    rank_avg = n_gt + 1 + n_eq / 2.0
    R = {k: (0.0 if n_gt >= k else min(1.0, (k - n_gt) / tie)) for k in k_list}
    E_recip = float(np.mean(1.0 / (n_gt + 1 + np.arange(tie, dtype=np.float64))))
    return rank_avg, R, E_recip


def _bootstrap_delta_clustered(cluster_ids, a, b, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    uniq = np.unique(cluster_ids)
    idx_by = {c: np.where(cluster_ids == c)[0] for c in uniq}
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[c] for c in picks])
        deltas[i] = (a[rows] - b[rows]).mean()
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def _rebuild_gold_map(cog_path: str, gold_path: str) -> dict:
    """For each Durrant cognate site_id, return (gold_slot_in_pool, raw_m_rank, gold_matches, tnp_id).
    Uses the SAME candidate_proposer as the model receives — build_candidate_arrays.
    """
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    per_site = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"]);
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            prof = np.zeros((len(nc), 16), dtype=np.float32)
            val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)
            slot, gm = _find_gold_slot(feats, mask, cands,
                                          g["target_flank_orientation"],
                                          g["target_binding_loop_length"],
                                          g["guide_start_in_nc"],
                                          g["target_flank_start"])
            per_site[r["site_id"]] = {
                "tnp_id":       r["transposase_id"],
                "gold_slot":    int(slot),
                "gold_matches": float(gm) if slot >= 0 else float("nan"),
                "active_nc":    int(a),
                "pool_size":    int(mask.sum()),
                "raw_m":        feats[:, 3].copy(),
                "valid_mask":   mask.copy(),
            }
    return per_site


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--tnp-batch", type=int, default=4)
    ap.add_argument("--sites", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[dev] {device}", flush=True)

    print("[gold] building per-site gold slot map + raw_m ...", flush=True)
    per_site = _rebuild_gold_map(args.jsonl, args.gold_jsonl)
    print(f"  {len(per_site)} site_ids with gold annotation ({sum(1 for v in per_site.values() if v['gold_slot'] >= 0)} in pool)")

    # Load checkpoint
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    normalize_aux = any(k.startswith("bn_pair_aux.") or k.startswith("bn_geom_aux.") for k in state.keys())
    use_orient = any(k.startswith("orient_mlp.") or k.startswith("h_orient_aux.") for k in state.keys())
    use_additive = "alpha_pair" in state or "alpha_geom" in state
    cfg = V1Config(
        use_multi_branch=True,
        use_explicit_geom_stats=True,
        use_additive_fusion=use_additive,
        normalize_aux_logits=normalize_aux,
        use_and_fusion=True,   # 48c1hA is AND-fusion (48c1f family)
        use_orient_branch=use_orient,
    )
    print(f"[cfg] multi_branch=True and_fusion=True normalize={normalize_aux} orient={use_orient}",
          flush=True)
    model = V1Model(cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval()

    cache = StructureCache(args.cache)
    ds = TnpGroupedDataset(args.jsonl, cache, site_subsample_size=args.sites)
    print(f"[data] tnps={len(ds)}", flush=True)
    dl = DataLoader(
        make_torch_tnp_dataset(ds),
        batch_size=args.tnp_batch, shuffle=False, num_workers=2,
        collate_fn=lambda items: collate_tnp_batch(items, to_torch=True),
        pin_memory=(device.type == "cuda"),
    )

    ranks_mil = []; ranks_raw = []; cluster = []
    R1_mil = []; R4_mil = []; R8_mil = []
    R1_raw = []; R4_raw = []; R8_raw = []
    site_seen = 0
    with torch.no_grad():
        for batch in dl:
            batch_gpu = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                          for k, v in batch.items()}
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                  enabled=(device.type == "cuda")):
                out = model(
                    batch_gpu["candidate_patches"],
                    batch_gpu["candidate_features"],
                    batch_gpu["candidate_mask"],
                    batch_gpu["nc_region_mask"],
                )
            cand_attn = out["cand_attn"].detach().float().cpu().numpy()   # (B, S, N_nc, K)
            cand_mask = batch["candidate_mask"].numpy()                    # (B, S, N_nc, K) bool
            tnp_ids = list(batch["tnp_id"])
            site_ids = batch["site_ids"]                                    # list-of-lists (B, S)
            B, S, N_nc, K = cand_attn.shape
            for b in range(B):
                for s in range(S):
                    sid = site_ids[b][s] if s < len(site_ids[b]) else None
                    if sid is None or sid not in per_site: continue
                    ps = per_site[sid]
                    if ps["gold_slot"] < 0: continue
                    a_nc = ps["active_nc"]
                    if a_nc >= N_nc: a_nc = 0
                    scores = cand_attn[b, s, a_nc]                    # (K,)
                    valid = cand_mask[b, s, a_nc].astype(bool)         # (K,)
                    if ps["gold_slot"] >= K: continue
                    if not valid[ps["gold_slot"]]: continue
                    _, R_mil, MRR_mil = _rank_stats(scores, ps["gold_slot"], valid_mask=valid)
                    q_raw = ps["raw_m"]
                    # Filter raw_m to model's valid slots too (should match, but be safe)
                    if q_raw.shape[0] < K:
                        pad = np.full(K, -np.inf, dtype=np.float32)
                        pad[:q_raw.shape[0]] = q_raw
                        q_raw = pad
                    else:
                        q_raw = q_raw[:K]
                    _, R_raw, MRR_raw = _rank_stats(q_raw, ps["gold_slot"], valid_mask=valid)
                    ranks_mil.append(MRR_mil); ranks_raw.append(MRR_raw)
                    R1_mil.append(R_mil[1]); R4_mil.append(R_mil[4]); R8_mil.append(R_mil[8])
                    R1_raw.append(R_raw[1]); R4_raw.append(R_raw[4]); R8_raw.append(R_raw[8])
                    cluster.append(ps["tnp_id"])
                    site_seen += 1
    print(f"[done] scored sites: {site_seen}")

    if site_seen == 0:
        print("[fail] no sites scored — check dataset alignment")
        return

    mil_MRR = float(np.mean(ranks_mil))
    raw_MRR = float(np.mean(ranks_raw))
    lo, hi = _bootstrap_delta_clustered(cluster, ranks_mil, ranks_raw)
    print(f"\n=== A4 :: Old MIL (48c1hA epoch_04) vs raw_m on Durrant candidate ranking ===")
    print(f"  MIL attention  MRR = {mil_MRR:.4f}   R@1={np.mean(R1_mil):.4f}  R@4={np.mean(R4_mil):.4f}  R@8={np.mean(R8_mil):.4f}")
    print(f"  raw_m          MRR = {raw_MRR:.4f}   R@1={np.mean(R1_raw):.4f}  R@4={np.mean(R4_raw):.4f}  R@8={np.mean(R8_raw):.4f}")
    print(f"  Δ MRR (MIL − raw_m), Tnp-clustered 95% CI = [{lo:+.4f}, {hi:+.4f}]")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_sites":          site_seen,
            "MIL":              {"MRR": mil_MRR, "R@1": float(np.mean(R1_mil)),
                                    "R@4": float(np.mean(R4_mil)), "R@8": float(np.mean(R8_mil))},
            "raw_m":            {"MRR": raw_MRR, "R@1": float(np.mean(R1_raw)),
                                    "R@4": float(np.mean(R4_raw)), "R@8": float(np.mean(R8_raw))},
            "delta_MIL_raw_m_CI": [lo, hi],
        }, f, indent=2)
    print(f"[out] {args.out}")


if __name__ == "__main__":
    main()
