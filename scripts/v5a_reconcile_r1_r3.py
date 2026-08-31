"""V5A reconciliation R1 + R2 + R3.

R1: reconcile L=9 vs L=11.
    - Print gold-L histogram (Durrant + V4.2 val gold, in-pool).
    - Print exact length_pen(α=1.25, L0=9) penalty schedule vs. exact -|L-9|
      schedule at L=8..16.
    - Recompute D1' = "length_pen ONLY, m ignored, random tie-break" using the
      actual one-sided hinge, not the V-shape. Compare to old D1.
    - Recompute D2' = length_pen(1.25, 9) directly (m as canonical scorer),
      confirm it matches A6's in-sample 0.245.
    - Report the fraction of Durrant golds at L∈{8,9} (unpenalized by hinge)
      vs. L=11 (penalized 2.5).

R2: A10 z-score audit.
    - Report per-L (n_L, μ_L, σ_L) from the train-decoy null table used in A10.
      Especially L=11.
    - Recompute z-score using WITHIN-POOL per-L distribution (μ, σ from the
      SAME bag's candidates at that L, not the global train stat). Report
      Durrant + reweighted val MRR + R@8.
    - Compare within-pool z to global-null z; identify whether A10's failure
      was a mechanism failure or a data-coverage / definitional issue.

R3: Gate 0 re-validation with a power scorer.
    - Compute reweighted-val MRR under length_pen(1.25, 9) — the A9 CV winner.
    - Compare to Durrant length_pen(1.25, 9) CV = 0.230.
    - If reweighted-val length_pen ≈ 0.230, reweight is meaningful.
      If reweighted-val length_pen ≫ 0.230, transfer gap remains and the
      reweight's Gate 0 "sufficient" verdict rested on circular agreement at
      near-chance MRR.
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
                "cs_L":       int(cands[slot].L),
                "feats":      feats,
                "mask":       mask,
                "cands":      cands,
            })
    return out


# -------------------- R1 --------------------

def r1_penalty_schedule():
    print(f"\n=== R1.1 :: penalty schedules (α=1.25, L0=9) at each L ===")
    print(f"  {'L':>4} {'length_pen':>12} {'V shape -|L-9|':>16}")
    for L in range(5, 17):
        lp = 1.25 * max(0, L - 9)
        vs = abs(L - 9)
        print(f"  {L:>4} {lp:>12.3f} {vs:>16.3f}")
    print(f"  --> length_pen is ONE-SIDED (L<=9 unpenalized); -|L-9| is TWO-SIDED (V).")


def r1_gold_L_hist(recs_dur, val_pool_path):
    print(f"\n=== R1.2 :: gold candidate L histogram (canonical, in-pool) ===")
    dur = Counter(r["cs_L"] for r in recs_dur)
    val = Counter()
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            cs_local = next((j for j, s in enumerate(rec["slots"])
                                if s["slot"] == rec["cstar_slot"]), None)
            if cs_local is None: continue
            val[int(rec["slots"][cs_local]["L"])] += 1
    print(f"  {'L':>4} {'dur_gold':>10} {'val_gold':>10}")
    dur_tot = sum(dur.values()); val_tot = sum(val.values())
    for L in range(5, 17):
        print(f"  {L:>4} {dur.get(L, 0)/dur_tot:>10.3f} {val.get(L, 0)/val_tot:>10.3f}")
    dur_unpen = sum(dur.get(L, 0) for L in range(5, 10)) / dur_tot
    dur_L11   = dur.get(11, 0) / dur_tot
    print(f"  Durrant gold: {dur_unpen:.3%} at L<=9 (length_pen UNPENALIZED); "
          f"{dur_L11:.3%} at L=11 (penalty 2.5)")
    return {"durrant_gold_L": {int(k): v for k, v in dur.items()},
             "val_gold_L":     {int(k): v for k, v in val.items()}}


def r1_recompute_d1_d2(recs, alpha=1.25, L0=9, seed=0):
    """D1' = length_pen HINGE only, m ignored (random tiebreak).
    D2' = length_pen HINGE with m as tie-break within groups.
    D_hinge_pure = length_pen HINGE score with m contribution: identical to length_pen.
    """
    rng = np.random.default_rng(seed)
    d1p = []; d2p = []; d_true = []
    for rec in recs:
        valid = np.where(rec["mask"])[0]
        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
        m_arr = rec["feats"][valid, 3]
        cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
        primary = -alpha * np.maximum(0.0, Ls - L0)          # exact hinge, m ignored
        s1 = primary * 1000.0 + rng.random(len(primary))      # random tie-break
        s2 = primary * 1000.0 + m_arr                          # m tie-break
        s3 = m_arr - alpha * np.maximum(0.0, Ls - L0)         # canonical length_pen
        _, _, MRR1 = rank_stats(s1, cs_pos); d1p.append(MRR1)
        _, _, MRR2 = rank_stats(s2, cs_pos); d2p.append(MRR2)
        _, _, MRR3 = rank_stats(s3, cs_pos); d_true.append(MRR3)
    print(f"\n=== R1.3 :: D1' / D2' / length_pen(1.25, 9) — HINGE (correct) ===")
    print(f"  D1' (hinge, m ignored, random ties)    MRR = {np.mean(d1p):.4f}")
    print(f"  D2' (hinge, m as tie-break)             MRR = {np.mean(d2p):.4f}")
    print(f"  length_pen(1.25, 9) canonical           MRR = {np.mean(d_true):.4f}")
    print(f"  Δ (D2' − D1') = {np.mean(d2p) - np.mean(d1p):+.4f}   ← this is m's contribution beyond hinge")
    print(f"  For reference the old (buggy) D1 with -|L-9| gave 0.167; that was a V-shape not the hinge.")
    return {"d1_prime": float(np.mean(d1p)), "d2_prime": float(np.mean(d2p)),
             "length_pen": float(np.mean(d_true))}


# -------------------- R2 --------------------

def r2_null_coverage(mining_aug_path, train_tnps):
    print(f"\n=== R2.1 :: per-L null-table coverage (train decoys) ===")
    per_L = defaultdict(list)
    with open(mining_aug_path) as f:
        for line in f:
            m = json.loads(line)
            if m["transposase_id"] not in train_tnps: continue
            for d in m["decoys"]:
                per_L[int(d["L"])].append(float(d["matches"]))
    print(f"  {'L':>4} {'n_L':>10} {'μ_L':>7} {'σ_L':>7} {'m_min':>6} {'m_max':>6}")
    stats = {}
    for L in sorted(per_L):
        arr = np.asarray(per_L[L], dtype=np.float64)
        stats[L] = {"n": len(arr), "mu": float(arr.mean()), "sigma": float(arr.std() or 1.0),
                     "m_min": float(arr.min()), "m_max": float(arr.max())}
        print(f"  {L:>4} {len(arr):>10} {arr.mean():>7.2f} {arr.std():>7.2f} {arr.min():>6.0f} {arr.max():>6.0f}")
    return stats


def r2_zscore_variants(recs, val_pool_path, global_null_stats):
    """Global z (μ_L, σ_L from train) vs within-pool z (μ, σ from same bag's L=L candidates)."""
    print(f"\n=== R2.2 :: z-score on Durrant — GLOBAL vs WITHIN-POOL ===")

    def _within_pool_z(m_arr, L_arr):
        z = np.zeros_like(m_arr, dtype=np.float32)
        L_unique = np.unique(L_arr).astype(int)
        for L in L_unique:
            mask = (L_arr == L)
            if mask.sum() < 2: continue
            m_at_L = m_arr[mask]
            mu = float(m_at_L.mean()); sd = float(m_at_L.std() or 1.0)
            z[mask] = (m_at_L - mu) / max(1e-6, sd)
        return z

    def _global_z(m_arr, L_arr):
        z = np.zeros_like(m_arr, dtype=np.float32)
        for i in range(len(m_arr)):
            L = int(L_arr[i])
            st = global_null_stats.get(L)
            if st is None: z[i] = 0.0
            else: z[i] = (m_arr[i] - st["mu"]) / max(1e-6, st["sigma"])
        return z

    # Durrant
    for zvariant, fn in (("global (train per-L)", _global_z),
                          ("within-pool (same bag's L)", _within_pool_z)):
        mrs = []; R1 = []; R4 = []; R8 = []
        for rec in recs:
            valid = np.where(rec["mask"])[0]
            Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
            m_arr = rec["feats"][valid, 3]
            q = fn(m_arr, Ls)
            cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
            _, R, MRR = rank_stats(q, cs_pos)
            mrs.append(MRR); R1.append(R[1]); R4.append(R[4]); R8.append(R[8])
        print(f"  Durrant  z ({zvariant:<28})   MRR={np.mean(mrs):.4f}  "
              f"R@1={np.mean(R1):.4f}  R@4={np.mean(R4):.4f}  R@8={np.mean(R8):.4f}")
    return None


# -------------------- R3 --------------------

def r3_gate0_recheck(recs, val_pool_path):
    """Recompute reweighted-val MRR under length_pen(1.25, 9) and length_pen(0.5, 12).
    If reweighted-val length_pen(1.25, 9) ≈ Durrant CV 0.230, reweight has power.
    If reweighted-val length_pen(1.25, 9) is much higher than 0.230, transfer gap
    remains — Gate 0's "sufficient" verdict was circular."""
    print(f"\n=== R3 :: Gate 0 re-validation with power scorer ===")
    dur_pairs = [(int(rec["cands"][rec["cs_slot"]].L),
                    int(rec["feats"][rec["cs_slot"], 3])) for rec in recs]
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
    W_p95 = np.minimum(W, np.percentile(W, 95))
    W_uncap = W

    def _weighted_MRR(alpha, L0, W_):
        num = 0.0; den = 0.0
        for b, w in zip(val_data, W_):
            q = b["m_arr"] - alpha * np.maximum(0.0, b["L_arr"] - L0)
            _, _, MRR = rank_stats(q, b["cs_local"])
            num += w * MRR; den += w
        return num / max(den, 1e-12)

    # Durrant length_pen (in-sample) for both configs
    def _durrant_MRR(alpha, L0):
        mrs = []
        for rec in recs:
            valid = np.where(rec["mask"])[0]
            Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
            m_arr = rec["feats"][valid, 3]
            q = m_arr - alpha * np.maximum(0.0, Ls - L0)
            cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
            _, _, MRR = rank_stats(q, cs_pos)
            mrs.append(MRR)
        return float(np.mean(mrs))

    print(f"  scorer                    reweighted val (uncap) | reweighted val (p95) | Durrant in-sample | Durrant CV")
    for alpha, L0, tag in [(0.5, 12, "length_pen(0.5, 12) transfer"),
                              (1.25, 9, "length_pen(1.25, 9) A9 winner")]:
        rw_unc = _weighted_MRR(alpha, L0, W_uncap)
        rw_p95 = _weighted_MRR(alpha, L0, W_p95)
        dur_is = _durrant_MRR(alpha, L0)
        dur_cv = 0.230 if (alpha, L0) == (1.25, 9) else None
        cv_s = f"{dur_cv:.4f}" if dur_cv is not None else "n/a"
        print(f"  {tag:<28}   {rw_unc:>16.4f}  |   {rw_p95:>16.4f}  |   {dur_is:>13.4f}  | {cv_s:>10}")

    print(f"\n[R3 verdict]")
    print(f"  If reweighted-val length_pen(1.25, 9) ≈ 0.230 → reweight has power at the strong scorer, verdict holds")
    print(f"  If reweighted-val length_pen(1.25, 9) is far above 0.230 → transfer gap remains at the strong scorer")


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
    print(f"  n_in_pool={len(recs)}")

    r1_penalty_schedule()
    r1_hist = r1_gold_L_hist(recs, args.val_pool)
    r1_d = r1_recompute_d1_d2(recs)

    train_tnps = set(json.load(open(args.splits))["train"])
    null_stats = r2_null_coverage(args.mining_aug, train_tnps)
    r2_zscore_variants(recs, args.val_pool, null_stats)

    r3_gate0_recheck(recs, args.val_pool)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "n_in_pool_canonical": len(recs),
            "R1_gold_L_hist": r1_hist,
            "R1_D_recomputed": r1_d,
            "R2_null_coverage_per_L": {int(k): v for k, v in null_stats.items()},
        }, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
