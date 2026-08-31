"""R1-B1a and R1-B2: gold-candidate injection into the model pipeline.

For each cognate POS bag, use the Durrant gold annotation to construct a
gold Candidate. Then evaluate the model under two candidate-pool modifications:

  B1a — Gold-present: guarantee the gold slot is filled with the real
        interaction (replacing the worst-matching slot if gold isn't
        already in the pool). Other decoys are preserved.
        Answers: "when the scorer can see the correct pairing, does it
        use it?"

  B2  — Gold-only: mask all candidate slots except the gold slot. The
        aggregator sees only the real pairing. This is the purest
        representation-transfer test.
        Answers: "does the learned pair encoder recognize real cognate
        pairing at all?"

Shuffled records are treated as-is (no gold to inject). For B2 they get
their top-1 candidate as the "sole" candidate (the best proposal-side
alignment they have), so the comparison is 'best-real-pairing' vs
'best-random-alignment'.

Outputs pair AUROC + paired Δ statistics for each mode.
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
    DEFAULT_L_MIN, DEFAULT_L_MAX, TOP_K_PER_COMBO_DEFAULT,
    PATCH_WIDTH_DEFAULT, PATCH_CHANNELS, NUM_FEATURES,
    encode_dna, dot_plot, windowed_matches,
)
from preprocess.site import DEFAULT_NC_MAX, DEFAULT_NUM_NC_SLOTS


def _matches(nc_codes: np.ndarray, flank_codes: np.ndarray,
              orient: str, L: int, nc_start: int, flank_start: int) -> int:
    """Count exact matching bases between nc[nc_start:nc_start+L] and the
    orient-aware flank window at flank_start."""
    from preprocess.candidates import _COMP_CODES
    if orient == "fwd":
        f = flank_codes[flank_start:flank_start + L]
    else:
        f = _COMP_CODES[flank_codes[flank_start:flank_start + L][::-1]]
    n = nc_codes[nc_start:nc_start + L]
    return int(np.sum(n == f))


def _gold_equivalent_slot(feats, mask, cands,
                            gold_orient, gold_L, gold_nc_start, gold_flank_start,
                            overlap_frac: float = 0.5) -> int:
    """Find slot index of the best gold-equivalent existing candidate; -1 if none."""
    valid = np.where(mask)[0]
    best = -1; best_matches = -1.0
    for i in valid:
        c = cands[i]
        if c is None or c.orient != gold_orient:
            continue
        min_L = min(c.L, gold_L)
        ov_f = max(0, min(c.flank_start + c.L, gold_flank_start + gold_L)
                       - max(c.flank_start, gold_flank_start))
        ov_n = max(0, min(c.nc_start + c.L, gold_nc_start + gold_L)
                       - max(c.nc_start, gold_nc_start))
        thresh = overlap_frac * min_L
        if ov_f >= thresh and ov_n >= thresh and feats[i, 3] > best_matches:
            best = int(i); best_matches = float(feats[i, 3])
    return best


def _worst_slot(feats, mask) -> int:
    """Return valid slot with lowest `matches` (feature index 3), for replacement."""
    valid = np.where(mask)[0]
    if len(valid) == 0:
        # No valid slot at all — need to allocate one. Return slot 0.
        return 0
    matches = feats[:, 3]
    subset = matches[valid]
    return int(valid[np.argmin(subset)])


def _fill_gold_at_slot(patches, feats, mask, cands, slot, gold_cand: Candidate,
                        nc_codes, flank_codes, structure_profile, structure_valid,
                        patch_width, nc_max):
    """Overwrite `slot` with the gold candidate (in-place). Marks slot valid."""
    _fill_candidate_slot(
        patches, feats, mask, slot,
        nc_codes, flank_codes,
        structure_profile, structure_valid,
        gold_cand, patch_width, nc_max,
    )
    cands[slot] = gold_cand
    mask[slot] = True


def infer_bag(model, patches_np, feats_np, mask_np, nc_region_mask_np, device):
    """Run one bag through the model (single site). Returns s_pair_aux float."""
    patches = torch.from_numpy(patches_np[None, None, :, :, :, :]).to(device)   # (1,1,N,K,W,C)
    feats   = torch.from_numpy(feats_np[None, None, :, :, :]).to(device)         # (1,1,N,K,F)
    mask    = torch.from_numpy(mask_np[None, None, :, :]).to(device)             # (1,1,N,K) bool
    nc_reg  = torch.from_numpy(nc_region_mask_np[None, None, :]).to(device)      # (1,1,N) bool
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=(device.type == "cuda")):
            out = model(patches, feats, mask, nc_reg)
    return float(out["s_pair_aux"].detach().float().cpu().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cognate-jsonl", required=True)
    ap.add_argument("--shuffled-jsonl", required=True)
    ap.add_argument("--gold-jsonl", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load ckpt
    obj = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = obj["model"] if isinstance(obj, dict) and "model" in obj else obj
    use_additive = "alpha_pair" in state or "alpha_geom" in state
    use_orient = any(k.startswith("orient_mlp.") or k.startswith("h_orient_aux.")
                       for k in state.keys())
    v1_cfg = V1Config(
        use_multi_branch=True,
        use_explicit_geom_stats=True,
        use_additive_fusion=use_additive,
        use_orient_branch=use_orient,
    )
    model = V1Model(v1_cfg).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()

    # Load gold annotation
    gold = {}
    with open(args.gold_jsonl) as f:
        for line in f:
            r = json.loads(line)
            gold[r["site_id"]] = r
    print(f"[gold] {len(gold)} annotated", flush=True)

    def _process(path, mode: str, is_cognate: bool):
        """mode ∈ {'raw', 'gold_present', 'gold_only'}."""
        scores = []
        tnps = []
        rank_before_stats = []
        rank_after_stats = []
        margins = []
        with open(path) as f:
            for i, line in enumerate(f):
                if args.limit is not None and i >= args.limit:
                    break
                r = json.loads(line)
                sid = r["site_id"]
                gold_sid = sid.replace("_shu_", "_cog_")
                g = gold.get(gold_sid)
                if g is None: continue
                active_nc = r["labels"].get("active_noncoding_index", 0) or 0
                ncs = r["inputs"]["noncoding_regions"]
                if active_nc >= len(ncs): active_nc = 0
                nc = ncs[active_nc]
                flank = r["inputs"]["flank"]
                if not nc or not flank: continue

                # Structure profile zero-filled (matches training conditions)
                nc_len = len(nc)
                prof = np.zeros((nc_len, 16), dtype=np.float32)
                val = np.zeros((nc_len, 16), dtype=bool)
                patches, feats, mask, cands = build_candidate_arrays(
                    nc, flank, prof, val,
                    L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX,
                )

                # Gold definition
                gold_cand = Candidate(
                    orient=g["target_flank_orientation"],
                    L=g["target_binding_loop_length"],
                    nc_start=g["guide_start_in_nc"],
                    flank_start=g["target_flank_start"],
                    matches=g["target_flank_matches"],
                )

                # Slot layout has to accommodate N_nc slots × K candidates. We are
                # dealing with a single-site record, so build a fake (num_nc_slots, K)
                # arrangement: put all real candidates in slot 0, then modify.
                num_nc_slots = DEFAULT_NUM_NC_SLOTS
                K = patches.shape[0]  # currently flat pool

                # Pad to (num_nc_slots, K, W, C) — real proposal in slot 0, others empty
                W = patches.shape[1]; C = patches.shape[2]
                p_out = np.zeros((num_nc_slots, K, W, C), dtype=np.float32)
                f_out = np.zeros((num_nc_slots, K, NUM_FEATURES), dtype=np.float32)
                m_out = np.zeros((num_nc_slots, K), dtype=bool)
                p_out[active_nc] = patches
                f_out[active_nc] = feats
                m_out[active_nc] = mask
                nc_region_mask = np.zeros(num_nc_slots, dtype=bool)
                nc_region_mask[active_nc] = True

                # Compute rank_before for cognate records
                rank_before = -1
                if is_cognate:
                    gold_slot = _gold_equivalent_slot(
                        feats, mask, cands,
                        gold_cand.orient, gold_cand.L,
                        gold_cand.nc_start, gold_cand.flank_start,
                    )
                    if gold_slot >= 0:
                        matches_arr = feats[:, 3]
                        rank_before = int((matches_arr[mask] > matches_arr[gold_slot]).sum() + 1)
                rank_before_stats.append(rank_before if rank_before > 0 else None)

                # Apply mode
                if mode == "raw":
                    pass
                elif mode == "gold_present" and is_cognate:
                    gold_slot = _gold_equivalent_slot(
                        feats, mask, cands,
                        gold_cand.orient, gold_cand.L,
                        gold_cand.nc_start, gold_cand.flank_start,
                    )
                    if gold_slot < 0:
                        # Not in pool: replace worst slot with gold
                        worst = _worst_slot(feats, mask)
                        # Rebuild that slot in the padded arrays
                        p_slot = p_out[active_nc]
                        f_slot = f_out[active_nc]
                        m_slot = m_out[active_nc]
                        nc_codes = encode_dna(nc)
                        flank_codes = encode_dna(flank)
                        _fill_gold_at_slot(
                            p_slot, f_slot, m_slot, cands,
                            worst, gold_cand, nc_codes, flank_codes,
                            prof, val, W, DEFAULT_NC_MAX,
                        )
                        p_out[active_nc] = p_slot
                        f_out[active_nc] = f_slot
                        m_out[active_nc] = m_slot
                elif mode == "gold_only":
                    if is_cognate:
                        # Mask out everything, then place gold at slot 0
                        p_out[active_nc, :] = 0
                        f_out[active_nc, :] = 0
                        m_out[active_nc, :] = False
                        p_slot = p_out[active_nc]
                        f_slot = f_out[active_nc]
                        m_slot = m_out[active_nc]
                        nc_codes = encode_dna(nc)
                        flank_codes = encode_dna(flank)
                        _fill_gold_at_slot(
                            p_slot, f_slot, m_slot, cands,
                            0, gold_cand, nc_codes, flank_codes,
                            prof, val, W, DEFAULT_NC_MAX,
                        )
                        p_out[active_nc] = p_slot
                        f_out[active_nc] = f_slot
                        m_out[active_nc] = m_slot
                    else:
                        # Shuffled: mask all except top-1 (best proposal-side alignment)
                        matches = f_out[active_nc, :, 3]
                        mask_a = m_out[active_nc]
                        if mask_a.any():
                            top1 = int(np.argmax(matches * mask_a))
                        else:
                            top1 = 0
                        keep = np.zeros_like(mask_a); keep[top1] = mask_a[top1]
                        m_out[active_nc] = keep

                # Compute rank_after
                rank_after = -1
                if is_cognate:
                    gold_slot = _gold_equivalent_slot(
                        f_out[active_nc], m_out[active_nc], cands,
                        gold_cand.orient, gold_cand.L,
                        gold_cand.nc_start, gold_cand.flank_start,
                    )
                    if gold_slot >= 0:
                        matches_arr = f_out[active_nc, :, 3]
                        m_arr = m_out[active_nc]
                        rank_after = int((matches_arr[m_arr] > matches_arr[gold_slot]).sum() + 1)
                        # Margin: gold matches vs best decoy matches
                        non_gold_idx = [j for j in np.where(m_arr)[0] if j != gold_slot]
                        if non_gold_idx:
                            best_dec = float(matches_arr[non_gold_idx].max())
                            margins.append(float(matches_arr[gold_slot]) - best_dec)
                rank_after_stats.append(rank_after if rank_after > 0 else None)

                s = infer_bag(model, p_out, f_out, m_out, nc_region_mask, device)
                scores.append(s)
                tnps.append(r["transposase_id"])

        return {
            "scores": scores, "tnps": tnps,
            "rank_before": rank_before_stats,
            "rank_after":  rank_after_stats,
            "margins":     margins,
        }

    # Run all modes
    modes = ["raw", "gold_present", "gold_only"]
    all_out = {}
    for mode in modes:
        print(f"\n--- mode: {mode} ---", flush=True)
        cog = _process(args.cognate_jsonl, mode, is_cognate=True)
        shu = _process(args.shuffled_jsonl, mode, is_cognate=False)
        # AUROC
        scores = np.asarray(cog["scores"] + shu["scores"], dtype=np.float32)
        labels = np.asarray([1] * len(cog["scores"]) + [0] * len(shu["scores"]))
        from sklearn.metrics import roc_auc_score
        au = float(roc_auc_score(labels, scores)) if len(set(labels.tolist())) == 2 else float("nan")
        # Paired Δ by parent (assume 5 records per tnp; average per tnp)
        cog_by_tnp = defaultdict(list); shu_by_tnp = defaultdict(list)
        for t, s in zip(cog["tnps"], cog["scores"]): cog_by_tnp[t].append(s)
        for t, s in zip(shu["tnps"], shu["scores"]): shu_by_tnp[t].append(s)
        def _key(tnp):
            # normalize cog / shu tokens to allow parent-matched pairing
            return tnp.replace("_cog_", "_").replace("_shu_", "_")
        cog_tnp_score = {t: float(np.mean(v)) for t, v in cog_by_tnp.items()}
        shu_tnp_score = {t: float(np.mean(v)) for t, v in shu_by_tnp.items()}
        deltas = []
        for tnp, s_cog in cog_tnp_score.items():
            neg = tnp.replace("_cog_", "_shu_")
            if neg in shu_tnp_score:
                deltas.append(s_cog - shu_tnp_score[neg])
        arr = np.asarray(deltas, dtype=np.float32)
        paired = {
            "n":      int(len(arr)),
            "median": float(np.median(arr)) if len(arr) else None,
            "MAD":    float(np.median(np.abs(arr - np.median(arr)))) if len(arr) else None,
            "p_gt_0": float((arr > 0).mean()) if len(arr) else None,
        }
        # Rank/margin summaries
        rb = [r for r in cog["rank_before"] if r is not None]
        ra = [r for r in cog["rank_after"] if r is not None]
        marg = np.asarray(cog["margins"], dtype=np.float32) if cog["margins"] else None
        all_out[mode] = {
            "AUROC":     au,
            "paired":    paired,
            "rank_before_median": float(np.median(rb)) if rb else None,
            "rank_after_median":  float(np.median(ra)) if ra else None,
            "n_rank_before_found": len(rb),
            "n_rank_after_found":  len(ra),
            "margin_median": float(np.median(marg)) if marg is not None else None,
            "margin_p_gt_0": float((marg > 0).mean()) if marg is not None else None,
        }
        print(f"  {mode:<14} AUROC={au:.4f}  paired: n={paired['n']}  median={paired['median']:+.3f}  P(Δ>0)={paired['p_gt_0']:.3f}",
              flush=True)
        if ra:
            print(f"                 rank_after_median={all_out[mode]['rank_after_median']:.1f}  "
                  f"(before={all_out[mode]['rank_before_median']:.1f})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
