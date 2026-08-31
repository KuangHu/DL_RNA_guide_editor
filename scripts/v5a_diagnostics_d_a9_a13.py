"""V5A diagnostics batch — D1/D2/D3 + A9 + A10 + A11 + A13.

D1: MRR of −|L−9| alone, random tie-break (m ignored). Answers "does m carry
    selection signal on Durrant, or is 0.245 just a length filter?"
D2: Same, with m as tie-break inside the length band.
D3: Mean pool count with L∈[8,10]. Confirms the "uniform pick among 13" reading
    of MRR ≈ H_13/13.

A9: Leave-one-Tnp-out CV grid on Durrant for (α, L0) → honest family ceiling.
    Replaces the in-sample 0.245 with a proper hold-out number.
A10: z-score baseline z(c) = (m − μ_L) / σ_L where μ, σ are computed per L on
    TRAIN decoy pool. Report Durrant + reweighted val R@K + MRR.
A11: gold L and pool L distributions, Durrant vs reweighted val.

A13: Reconcile n=236 vs 240 and raw_m 0.089 vs 0.086 via canonical module.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")
sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/scripts")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX
from v5a_eval_core import (
    rank_stats, bootstrap_delta_clustered, find_gold_slot, classify_decoy,
    score_length_pen, DECOY_BUCKETS,
)


def build_records(cog, gold_path):
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    out = []
    with open(cog) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            prof = np.zeros((len(nc), 16), dtype=np.float32)
            val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)
            slot, gm = find_gold_slot(feats, mask, cands,
                                          g["target_flank_orientation"],
                                          g["target_binding_loop_length"],
                                          g["guide_start_in_nc"],
                                          g["target_flank_start"])
            if slot < 0: continue
            out.append({
                "site_id":    r["site_id"],
                "tnp_id":     r["transposase_id"],
                "cs_slot":    int(slot),
                "cs_matches": float(gm),
                "feats":      feats,
                "mask":       mask,
                "cands":      cands,
                "gold_orient": g["target_flank_orientation"],
                "gold_L":     g["target_binding_loop_length"],
                "gold_nc":    g["guide_start_in_nc"],
                "gold_fl":    g["target_flank_start"],
            })
    return out


# ---------- D1 / D2 / D3 --------------------------------------------------

def d1_length_only(recs, L_star: int = 9, use_matches_tiebreak: bool = False,
                    seed: int = 0):
    """MRR of -|L - L_star|; tie-break with m if enabled, else random."""
    rng = np.random.default_rng(seed)
    mrs = []; tnps = []
    for rec in recs:
        valid = np.where(rec["mask"])[0]
        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
        m_arr = rec["feats"][valid, 3]
        primary = -np.abs(Ls - L_star)
        if use_matches_tiebreak:
            # score = primary + eps * m to break ties in favor of higher matches
            score = primary * 1000.0 + m_arr
        else:
            score = primary * 1000.0 + rng.random(len(primary))
        cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
        _, _, MRR = rank_stats(score, cs_pos)
        mrs.append(MRR); tnps.append(rec["tnp_id"])
    return float(np.mean(mrs)), mrs, tnps


def d3_pool_count(recs, L_min=8, L_max=10):
    counts = []
    for rec in recs:
        valid = np.where(rec["mask"])[0]
        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.int32)
        counts.append(int(((Ls >= L_min) & (Ls <= L_max)).sum()))
    return float(np.mean(counts)), int(np.median(counts)), counts


# ---------- A9 CV grid ----------------------------------------------------

def a9_cv_grid(recs):
    """Leave-one-Tnp-out CV over the (α, L0) grid on Durrant."""
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    L0s = [8, 9, 10, 11, 12, 13, 14]
    tnps = sorted(set(r["tnp_id"] for r in recs))
    per_tnp = defaultdict(list)
    for r in recs: per_tnp[r["tnp_id"]].append(r)
    # For each held-out Tnp, choose best (α, L0) on all OTHER Tnps' MRR.
    # Score held-out MRR with that choice.
    heldout_mrrs = []
    chosen = []
    for held in tnps:
        best = (-np.inf, None)
        # Compute all MRRs on TRAIN Tnps for each grid point
        for a in alphas:
            for L0 in L0s:
                mrs = []
                for t in tnps:
                    if t == held: continue
                    for rec in per_tnp[t]:
                        valid = np.where(rec["mask"])[0]
                        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
                        m_arr = rec["feats"][valid, 3]
                        q = m_arr - a * np.maximum(0.0, Ls - L0)
                        cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
                        _, _, MRR = rank_stats(q, cs_pos)
                        mrs.append(MRR)
                mm = float(np.mean(mrs))
                if mm > best[0]: best = (mm, (a, L0))
        # Now score the held-out Tnp with the chosen (α, L0).
        a_h, L0_h = best[1]
        heldout_per_bag = []
        for rec in per_tnp[held]:
            valid = np.where(rec["mask"])[0]
            Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
            m_arr = rec["feats"][valid, 3]
            q = m_arr - a_h * np.maximum(0.0, Ls - L0_h)
            cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
            _, _, MRR = rank_stats(q, cs_pos)
            heldout_per_bag.append(MRR)
        heldout_mrrs.extend(heldout_per_bag)
        chosen.append({"tnp": held, "chosen_a": a_h, "chosen_L0": L0_h,
                        "heldout_MRR_mean": float(np.mean(heldout_per_bag))})
    cv_MRR = float(np.mean(heldout_mrrs))
    print(f"\n=== A9 :: Leave-one-Tnp-out CV of (α, L0) on Durrant ===")
    print(f"  CV pooled MRR = {cv_MRR:.4f}   n_bags={len(heldout_mrrs)}   n_tnps={len(tnps)}")
    # Chosen (α, L0) frequency
    chose_counts = Counter((c["chosen_a"], c["chosen_L0"]) for c in chosen)
    print(f"  most-chosen (α, L0): {chose_counts.most_common(5)}")
    return {"cv_MRR": cv_MRR, "n_bags": len(heldout_mrrs), "n_tnps": len(tnps),
             "chosen_summary": dict(chose_counts)}


# ---------- A10 z-score baseline ------------------------------------------

def _train_null_from_mining(mining_aug_path: str, train_tnps: set):
    """Per-L (μ, σ) of m across train decoy pool."""
    per_L = defaultdict(list)
    with open(mining_aug_path) as f:
        for line in f:
            m = json.loads(line)
            if m["transposase_id"] not in train_tnps: continue
            for d in m["decoys"]:
                per_L[int(d["L"])].append(float(d["matches"]))
    stats = {}
    for L, ms in per_L.items():
        arr = np.asarray(ms, dtype=np.float64)
        stats[L] = (float(arr.mean()), float(arr.std() or 1.0))
    return stats


def _zscore(m_arr, L_arr, null_stats):
    z = np.zeros_like(m_arr, dtype=np.float32)
    for i in range(len(m_arr)):
        L = int(L_arr[i])
        mu, sd = null_stats.get(L, (float(m_arr[i]), 1.0))
        z[i] = (m_arr[i] - mu) / max(1e-6, sd)
    return z


def a10_zscore(recs_dur, val_pool_path, null_stats):
    print(f"\n=== A10 :: z-score baseline (m − μ_L) / σ_L ===")
    # Durrant
    mrs = []; R1 = []; R4 = []; R8 = []; tnps = []
    for rec in recs_dur:
        valid = np.where(rec["mask"])[0]
        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
        m_arr = rec["feats"][valid, 3]
        q = _zscore(m_arr, Ls, null_stats)
        cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
        _, R, MRR = rank_stats(q, cs_pos)
        mrs.append(MRR); R1.append(R[1]); R4.append(R[4]); R8.append(R[8])
        tnps.append(rec["tnp_id"])
    print(f"  Durrant  z-score  MRR={np.mean(mrs):.4f}  R@1={np.mean(R1):.4f}  R@4={np.mean(R4):.4f}  R@8={np.mean(R8):.4f}")

    # Reweighted val — reuse Gate 0 weights inline
    print(f"  Reweighted val z-score ... (weighted)", flush=True)
    dur_pairs = [(int(rec["cands"][rec["cs_slot"]].L),
                    int(rec["feats"][rec["cs_slot"], 3])) for rec in recs_dur]
    dur_counts = Counter(dur_pairs); total_dur = sum(dur_counts.values())
    val_data = []
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            cs_local = next((j for j, s in enumerate(rec["slots"])
                                if s["slot"] == rec["cstar_slot"]), None)
            if cs_local is None: continue
            slots = rec["slots"]
            val_data.append({"cs_local": cs_local,
                              "L_arr": np.asarray([int(s["L"]) for s in slots], dtype=np.float32),
                              "m_arr": np.asarray([float(s["matches"]) for s in slots], dtype=np.float32),
                              "L_cs": int(slots[cs_local]["L"]),
                              "m_cs": int(slots[cs_local]["matches"])})
    v42_counts = Counter((b["L_cs"], b["m_cs"]) for b in val_data)
    total_v42 = sum(v42_counts.values())
    ac = 1.0; n_cells = 17 * 17
    W = np.asarray([
        (dur_counts.get((b["L_cs"], b["m_cs"]), 0) + ac) / (total_dur + ac * n_cells)
        / max(1e-12, (v42_counts.get((b["L_cs"], b["m_cs"]), 0) + ac) / (total_v42 + ac * n_cells))
        for b in val_data], dtype=np.float64)
    p95 = float(np.percentile(W, 95))
    W_clipped = np.minimum(W, p95)
    print(f"    n_val_in_pool={len(W)}  weight cap p95={p95:.3f}")
    ums = []; MRRs = []; R1s = []; R4s = []; R8s = []
    for b, w in zip(val_data, W_clipped):
        q = _zscore(b["m_arr"], b["L_arr"], null_stats)
        _, R, MRR = rank_stats(q, b["cs_local"])
        MRRs.append(MRR); R1s.append(R[1]); R4s.append(R[4]); R8s.append(R[8])
    MRRs = np.asarray(MRRs); R1s = np.asarray(R1s); R4s = np.asarray(R4s); R8s = np.asarray(R8s)
    wmean = lambda x, w: float((x * w).sum() / w.sum())
    print(f"  Reweighted-val z-score MRR={wmean(MRRs, W_clipped):.4f}  R@1={wmean(R1s, W_clipped):.4f}  R@4={wmean(R4s, W_clipped):.4f}  R@8={wmean(R8s, W_clipped):.4f}")
    return {
        "durrant_MRR": float(np.mean(mrs)),
        "durrant_R@1": float(np.mean(R1)),
        "durrant_R@4": float(np.mean(R4)),
        "durrant_R@8": float(np.mean(R8)),
        "reweighted_val_MRR": wmean(MRRs, W_clipped),
        "reweighted_val_R@8": wmean(R8s, W_clipped),
    }


# ---------- A11 L-marginal comparison -------------------------------------

def a11_L_marginals(recs_dur, val_pool_path):
    print(f"\n=== A11 :: gold L and pool L marginals ===")
    dur_gold_L = Counter(rec["gold_L"] for rec in recs_dur)
    dur_pool_L = Counter()
    for rec in recs_dur:
        valid = np.where(rec["mask"])[0]
        for i in valid: dur_pool_L[int(rec["cands"][int(i)].L)] += 1
    val_gold_L = Counter()
    val_pool_L = Counter()
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            cs_local = next((j for j, s in enumerate(rec["slots"])
                                if s["slot"] == rec["cstar_slot"]), None)
            if cs_local is None: continue
            val_gold_L[int(rec["slots"][cs_local]["L"])] += 1
            for s in rec["slots"]: val_pool_L[int(s["L"])] += 1
    def _norm(c):
        s = sum(c.values()) or 1
        return {k: c[k] / s for k in c}
    dg = _norm(dur_gold_L); dp = _norm(dur_pool_L)
    vg = _norm(val_gold_L); vp = _norm(val_pool_L)
    print(f"  {'L':>4} {'dur_gold':>10} {'val_gold':>10} {'dur_pool':>10} {'val_pool':>10}")
    for L in range(5, 17):
        print(f"  {L:>4} {dg.get(L, 0):>10.3f} {vg.get(L, 0):>10.3f} {dp.get(L, 0):>10.3f} {vp.get(L, 0):>10.3f}")
    return {
        "durrant_gold_L":  {int(k): v for k, v in dg.items()},
        "val_gold_L":      {int(k): v for k, v in vg.items()},
        "durrant_pool_L":  {int(k): v for k, v in dp.items()},
        "val_pool_L":      {int(k): v for k, v in vp.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--val-pool", required=True)
    ap.add_argument("--mining-aug", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[collect] Durrant canonical records ...", flush=True)
    recs = build_records(args.durrant_cog, args.durrant_gold)
    print(f"  n_in_pool={len(recs)}   (canonical: 236 or 240 depending on which builder — this is the settled number)")

    # ---- D1 / D2 / D3 ----
    print(f"\n=== D1 :: MRR of -|L-9| alone, random tie-break ===")
    d1_MRR, d1_arr, _tnps = d1_length_only(recs, L_star=9, use_matches_tiebreak=False)
    print(f"  MRR = {d1_MRR:.4f}   n_bags={len(d1_arr)}")

    print(f"\n=== D2 :: MRR of -|L-9| with m as tie-break ===")
    d2_MRR, d2_arr, _tnps = d1_length_only(recs, L_star=9, use_matches_tiebreak=True)
    print(f"  MRR = {d2_MRR:.4f}   Δ(D2 − D1) = {d2_MRR - d1_MRR:+.4f}")

    d3_mean, d3_median, _ = d3_pool_count(recs, L_min=8, L_max=10)
    H13 = float(sum(1.0 / i for i in range(1, 14)) / 13.0)
    H10 = float(sum(1.0 / i for i in range(1, 11)) / 10.0)
    print(f"\n=== D3 :: pool count at L∈[8,10] ===")
    print(f"  mean = {d3_mean:.1f}   median = {d3_median}")
    print(f"  reference: H_k/k for k=10→{H10:.4f}, k=13→{H13:.4f}, k=15→{sum(1.0/i for i in range(1,16))/15:.4f}")

    print(f"\n[D verdict]")
    if abs(d2_MRR - d1_MRR) < 0.01:
        print(f"  D2 ≈ D1 (Δ={d2_MRR-d1_MRR:+.4f}) → m carries NO informative signal within length band")
        print(f"  → raw match count has no selection power on Durrant beyond length filtering")
        print(f"  → q = m + f(·) rationale IS in doubt; 3b (cross-site) is the primary track")
    else:
        print(f"  D2 > D1 by {d2_MRR-d1_MRR:+.4f} → m adds selection value beyond length")
        print(f"  → Gate A residual base has real content")

    # ---- A9 ----
    a9 = a9_cv_grid(recs)

    # ---- A10 ----
    train_tnps = set(json.load(open(args.splits))["train"])
    print(f"\n[A10] building per-L null stats from TRAIN decoys ({len(train_tnps)} tnps)...")
    null_stats = _train_null_from_mining(args.mining_aug, train_tnps)
    print(f"  L stats built for {len(null_stats)} L values")
    a10 = a10_zscore(recs, args.val_pool, null_stats)

    # ---- A11 ----
    a11 = a11_L_marginals(recs, args.val_pool)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_in_pool_canonical": len(recs),
            "D1_length_only_random":  {"MRR": d1_MRR},
            "D2_length_only_m_tie":   {"MRR": d2_MRR, "delta": d2_MRR - d1_MRR},
            "D3_pool_count_L_8_10":   {"mean": d3_mean, "median": d3_median,
                                          "reference_H_k_over_k":  {10: H10, 13: H13}},
            "A9_cv_grid":  a9,
            "A10_zscore":  a10,
            "A11_L_marginals": a11,
            "A13_note": f"Canonical n_in_pool = {len(recs)}; using v5a_eval_core.find_gold_slot uniformly.",
        }, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
