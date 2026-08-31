"""V5A blocking batch S1 + S2 + S3 + S4.

S1: IoU-based correctness against the ANNOTATED TBL span.
    - A candidate is "correct" iff its orient matches AND its nc-span AND its
      flank-span each have IoU ≥ 0.5 with the annotated TBL span.
    - Multi-correct MRR: 1 / (rank of the first-ranked correct candidate).
    - R@k under IoU-correctness: fraction of records where ≥1 correct is in top k.
    - Report top-8 decoy fraction overlapping the annotated TBL span (i.e. how
      often the current classifier calls a biologically-plausible candidate
      "different_region" or "wrong_flank").

S2: within-pool z audit.
    - Per-bag distribution of σ_L (how many bags have σ=0 at some L, at L=9).
    - Rank-based variant: score = rank of m within same-L candidates in the
      same bag. No σ dependency.
    - Add length_pen(1.25, 9) R@k on Durrant so K=8 comparison is complete.

S3: RRF (reciprocal rank fusion) baseline.
    - rank_fuse(c) = 1/(k_rr + rank_z(c)) + 1/(k_rr + rank_len(c)), k_rr=60.
    - Full R@k / MRR, both slot-match and IoU-correct.

S4: Nested Tnp-clustered CV harness on Durrant.
    - 5-fold outer × inner (α, L0) grid search on train folds → held-out MRR / R@k.
    - Report length_pen family CV ceiling under the nested protocol.
    - This is the selection protocol from here on.

Slot-match (original) and IoU-correct metrics are reported side-by-side.
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
    score_length_pen, overlap, DECOY_BUCKETS,
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
                "TBL_orient": g["target_flank_orientation"],
                "TBL_L":      g["target_binding_loop_length"],
                "TBL_nc":     g["guide_start_in_nc"],
                "TBL_fl":     g["target_flank_start"],
            })
    return out


def _is_correct_iou(c, TBL_orient, TBL_L, TBL_nc, TBL_fl, iou_th=0.5):
    """Candidate 'correct' iff same orient AND IoU >= iou_th on BOTH nc + flank."""
    if c.orient != TBL_orient: return False
    def _iou(a0, a1, b0, b1):
        inter = overlap(a0, a1, b0, b1)
        union = (a1 - a0) + (b1 - b0) - inter
        return inter / max(1e-9, union)
    nc_iou = _iou(c.nc_start, c.nc_start + c.L, TBL_nc, TBL_nc + TBL_L)
    fl_iou = _iou(c.flank_start, c.flank_start + c.L, TBL_fl, TBL_fl + TBL_L)
    return nc_iou >= iou_th and fl_iou >= iou_th


def _first_correct_rank(qs, correct_mask):
    """Return (MRR contribution = 1/rank of first correct; R_k mask array)."""
    order = np.argsort(-qs, kind="stable")     # descending
    for r, i in enumerate(order, 1):
        if correct_mask[i]:
            return r, 1.0 / r
    return -1, 0.0


def _eval_scorer_iou(recs, score_fn):
    """Return dict with slot-match and IoU-correct R@k + MRR."""
    slot_R1 = []; slot_R4 = []; slot_R8 = []; slot_MRR = []
    iou_R1 = []; iou_R4 = []; iou_R8 = []; iou_MRR = []
    tnps = []; correct_counts = []
    for rec in recs:
        valid = np.where(rec["mask"])[0]
        # slot-match: use canonical rank_stats
        qs_all = np.full(len(rec["cands"]), -np.inf, dtype=np.float32)
        qs_all[valid] = score_fn(rec, valid)
        cs = rec["cs_slot"]
        _, R, MRR = rank_stats(qs_all[valid], int(np.where(valid == cs)[0][0]))
        slot_R1.append(R[1]); slot_R4.append(R[4]); slot_R8.append(R[8]); slot_MRR.append(MRR)
        # IoU-correct
        correct_mask = np.zeros(len(rec["cands"]), dtype=bool)
        for i in valid:
            if _is_correct_iou(rec["cands"][int(i)], rec["TBL_orient"], rec["TBL_L"],
                                  rec["TBL_nc"], rec["TBL_fl"]):
                correct_mask[i] = True
        correct_counts.append(int(correct_mask.sum()))
        if correct_mask.sum() == 0:
            iou_R1.append(0.0); iou_R4.append(0.0); iou_R8.append(0.0); iou_MRR.append(0.0)
        else:
            rk, mrr = _first_correct_rank(qs_all, correct_mask)
            iou_R1.append(1.0 if rk == 1 else 0.0)
            iou_R4.append(1.0 if 1 <= rk <= 4 else 0.0)
            iou_R8.append(1.0 if 1 <= rk <= 8 else 0.0)
            iou_MRR.append(mrr)
        tnps.append(rec["tnp_id"])
    return {
        "slot": {"R@1": float(np.mean(slot_R1)), "R@4": float(np.mean(slot_R4)),
                    "R@8": float(np.mean(slot_R8)), "MRR": float(np.mean(slot_MRR))},
        "iou":  {"R@1": float(np.mean(iou_R1)), "R@4": float(np.mean(iou_R4)),
                    "R@8": float(np.mean(iou_R8)), "MRR": float(np.mean(iou_MRR))},
        "correct_counts": correct_counts,
        "tnps": tnps,
        "slot_MRR_arr": slot_MRR, "iou_MRR_arr": iou_MRR,
        "slot_R8_arr": slot_R8, "iou_R8_arr": iou_R8,
    }


def _raw_m(rec, valid):     return rec["feats"][valid, 3]
def _length_pen(a, L0):
    def _fn(rec, valid):
        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
        m_arr = rec["feats"][valid, 3]
        return m_arr - a * np.maximum(0.0, Ls - L0)
    return _fn


def _within_pool_z(rec, valid):
    Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
    m_arr = rec["feats"][valid, 3]
    z = np.zeros_like(m_arr, dtype=np.float32)
    for L in np.unique(Ls).astype(int):
        mask = (Ls == L)
        if mask.sum() < 2: continue    # skip singletons
        m_at_L = m_arr[mask]
        mu = float(m_at_L.mean()); sd = float(m_at_L.std() or 1.0)
        if sd < 1e-6:
            z[mask] = 0.0     # all tied within L → no signal
        else:
            z[mask] = (m_at_L - mu) / sd
    return z


def _within_pool_z_rank(rec, valid):
    """Robust rank variant of within-pool z: rank of m within same-L group.
    Higher m → higher rank_score (normalized to [-1, +1] roughly)."""
    Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
    m_arr = rec["feats"][valid, 3]
    score = np.zeros_like(m_arr, dtype=np.float32)
    for L in np.unique(Ls).astype(int):
        mask = (Ls == L)
        n = int(mask.sum())
        if n == 0: continue
        m_at_L = m_arr[mask]
        # rank m descending within the L bucket
        order = np.argsort(-m_at_L, kind="stable")
        ranks = np.empty(n, dtype=np.float32)
        ranks[order] = np.arange(n)
        # Higher rank = better. Normalize to (n-1-rank)/(n-1) so top gets ~1.
        if n > 1:
            score[mask] = (n - 1 - ranks) / (n - 1)
        else:
            score[mask] = 0.5   # singleton fixed
    return score


def _rrf(rec, valid, k_rr=60):
    """Reciprocal-rank fusion of within-pool z + length_pen(1.25, 9)."""
    Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
    m_arr = rec["feats"][valid, 3]
    q_z  = _within_pool_z(rec, valid)
    q_lp = m_arr - 1.25 * np.maximum(0.0, Ls - 9.0)
    # ranks (1-indexed) by descending score
    def _ranks(v):
        order = np.argsort(-v, kind="stable")
        r = np.empty(len(v), dtype=np.float32); r[order] = np.arange(len(v)) + 1
        return r
    r_z = _ranks(q_z); r_lp = _ranks(q_lp)
    return 1.0 / (k_rr + r_z) + 1.0 / (k_rr + r_lp)


# ---------------- S1 (IoU) ----------------

def s1_scan(recs):
    print(f"\n=== S1 :: IoU correctness audit ===")
    scorers = [("raw_m", _raw_m),
                 ("length_pen(0.5, 12)", _length_pen(0.5, 12)),
                 ("length_pen(1.25, 9)", _length_pen(1.25, 9)),
                 ("within-pool z", _within_pool_z),
                 ("within-pool z_rank", _within_pool_z_rank),
                 ("RRF(z, lenpen(1.25,9))", _rrf)]
    reports = {}
    for name, fn in scorers:
        rep = _eval_scorer_iou(recs, fn)
        reports[name] = rep
        print(f"  {name:<28}   slot R@1={rep['slot']['R@1']:.4f}  R@8={rep['slot']['R@8']:.4f}  MRR={rep['slot']['MRR']:.4f}  |  "
              f"IoU R@1={rep['iou']['R@1']:.4f}  R@8={rep['iou']['R@8']:.4f}  MRR={rep['iou']['MRR']:.4f}")

    # Correct-count distribution
    counts = reports["raw_m"]["correct_counts"]
    print(f"\n  IoU-correct candidates per record: mean={np.mean(counts):.2f}  median={int(np.median(counts))}  "
          f"max={max(counts)}   fraction n>=2 = {(np.asarray(counts)>=2).mean():.3%}")

    # Top-8 decoy IoU-overlap fraction (raw_m ranking; the classic taxonomy setup)
    n_overlap_decoys = 0; n_decoy_slots = 0
    for rec in recs:
        valid = np.where(rec["mask"])[0]
        m_arr = rec["feats"][valid, 3]
        order = np.argsort(-m_arr, kind="stable")
        cs = rec["cs_slot"]
        cs_pos = int(np.where(valid == cs)[0][0])
        n_top = 0
        for idx in order:
            if idx == cs_pos: continue
            n_top += 1
            slot = int(valid[idx])
            if _is_correct_iou(rec["cands"][slot], rec["TBL_orient"], rec["TBL_L"],
                                  rec["TBL_nc"], rec["TBL_fl"]):
                n_overlap_decoys += 1
            n_decoy_slots += 1
            if n_top >= 8: break
    print(f"\n  Top-8 decoys under raw_m ranking that OVERLAP the annotated TBL: "
          f"{n_overlap_decoys}/{n_decoy_slots} = {n_overlap_decoys/max(1,n_decoy_slots):.2%}")
    print(f"  → These candidates are being called 'decoys' but are biologically plausible.")

    # Paired-bootstrap Tnp-clustered CIs for key comparisons
    print(f"\n  Tnp-clustered 95% CIs on Δ metrics vs length_pen(1.25, 9):")
    ref_slot_MRR = np.asarray(reports["length_pen(1.25, 9)"]["slot_MRR_arr"])
    ref_iou_MRR  = np.asarray(reports["length_pen(1.25, 9)"]["iou_MRR_arr"])
    ref_slot_R8  = np.asarray(reports["length_pen(1.25, 9)"]["slot_R8_arr"])
    ref_iou_R8   = np.asarray(reports["length_pen(1.25, 9)"]["iou_R8_arr"])
    tnps = reports["length_pen(1.25, 9)"]["tnps"]
    for name in ("within-pool z", "within-pool z_rank", "RRF(z, lenpen(1.25,9))"):
        for metric, arr_this, arr_ref in [
            ("slot MRR", np.asarray(reports[name]["slot_MRR_arr"]), ref_slot_MRR),
            ("IoU  MRR", np.asarray(reports[name]["iou_MRR_arr"]),  ref_iou_MRR),
            ("slot R@8", np.asarray(reports[name]["slot_R8_arr"]),  ref_slot_R8),
            ("IoU  R@8", np.asarray(reports[name]["iou_R8_arr"]),   ref_iou_R8),
        ]:
            lo, hi = bootstrap_delta_clustered(tnps, arr_this, arr_ref)
            print(f"    Δ {metric:<10} ({name:<28} − length_pen(1.25,9)) = [{lo:+.4f}, {hi:+.4f}]")

    return reports


# ---------------- S2 (z audit + length_pen R@8) --------

def s2_z_audit(recs):
    print(f"\n=== S2 :: within-pool z σ audit ===")
    # For each bag, tally per-L σ. What fraction of bags have σ=0 at L=9?
    sigma_by_L = defaultdict(list)     # L -> list of σ across bags
    n_at_L = defaultdict(list)          # L -> list of n across bags
    for rec in recs:
        valid = np.where(rec["mask"])[0]
        Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
        m_arr = rec["feats"][valid, 3]
        for L in np.unique(Ls).astype(int):
            mask = (Ls == L)
            n = int(mask.sum())
            n_at_L[L].append(n)
            if n < 2:
                sigma_by_L[L].append(-1.0)   # sentinel: no σ defined
                continue
            sigma_by_L[L].append(float(m_arr[mask].std() or 0.0))
    print(f"  {'L':>4} {'n_bags':>10} {'med(n_L)':>10} {'frac_σ0':>10} {'frac_singleton':>16}")
    for L in sorted(sigma_by_L):
        sigmas = np.asarray(sigma_by_L[L])
        ns = np.asarray(n_at_L[L])
        frac_sigma0 = float((sigmas == 0.0).mean())
        frac_singleton = float((ns < 2).mean())
        print(f"  {L:>4} {len(sigmas):>10} {int(np.median(ns)):>10} {frac_sigma0:>10.3f} {frac_singleton:>16.3f}")


# ---------------- S4 (nested CV) --------

def s4_nested_cv(recs, n_folds=5, alphas=(0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
                    L0s=(8, 9, 10, 11, 12, 13, 14)):
    print(f"\n=== S4 :: nested Tnp-clustered CV of length_pen (α, L0) ===")
    tnps = sorted(set(r["tnp_id"] for r in recs))
    rng = np.random.default_rng(0)
    rng.shuffle(tnps)
    fold_of = {t: i % n_folds for i, t in enumerate(tnps)}
    per_tnp = defaultdict(list)
    for r in recs: per_tnp[r["tnp_id"]].append(r)

    def _MRR_on(rec_list, a, L0):
        mrs = []
        for rec in rec_list:
            valid = np.where(rec["mask"])[0]
            Ls = np.asarray([rec["cands"][int(i)].L for i in valid], dtype=np.float32)
            m_arr = rec["feats"][valid, 3]
            q = m_arr - a * np.maximum(0.0, Ls - L0)
            cs_pos = int(np.where(valid == rec["cs_slot"])[0][0])
            _, _, MRR = rank_stats(q, cs_pos)
            mrs.append(MRR)
        return float(np.mean(mrs)) if mrs else 0.0

    heldout = []
    chosen_summary = Counter()
    for k in range(n_folds):
        train = [r for t in tnps for r in per_tnp[t] if fold_of[t] != k]
        test  = [r for t in tnps for r in per_tnp[t] if fold_of[t] == k]
        best = (-np.inf, None)
        for a in alphas:
            for L0 in L0s:
                v = _MRR_on(train, a, L0)
                if v > best[0]: best = (v, (a, L0))
        a_h, L0_h = best[1]
        MRR_h = _MRR_on(test, a_h, L0_h)
        heldout.append(MRR_h)
        chosen_summary[(a_h, L0_h)] += 1
        print(f"  fold {k}: chose (α={a_h}, L0={L0_h})  train_MRR={best[0]:.4f}  heldout_MRR={MRR_h:.4f}")
    print(f"  nested CV mean heldout MRR = {np.mean(heldout):.4f}  (family ceiling under proper selection)")
    print(f"  chosen (α, L0) frequency: {dict(chosen_summary)}")
    return {"cv_MRR": float(np.mean(heldout)), "chosen": {f"{k}": v for k, v in chosen_summary.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[collect] Durrant canonical records ...", flush=True)
    recs = build_records(args.durrant_cog, args.durrant_gold)
    print(f"  n_in_pool={len(recs)}")

    r_s1 = s1_scan(recs)
    s2_z_audit(recs)
    r_s4 = s4_nested_cv(recs)

    # Compact output — pull the aggregate numbers per method
    out = {"n_in_pool": len(recs),
             "S1_scorers": {name: {"slot": r["slot"], "iou": r["iou"]}
                                for name, r in r_s1.items()},
             "S4_nested_cv": r_s4}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
