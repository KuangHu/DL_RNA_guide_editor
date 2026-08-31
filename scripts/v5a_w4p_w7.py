"""W4+ (gapped SW recovery on all Durrant golds) and W7 (cross-site nc-position
consistency of gold vs top wrong_orient decoy).

W4+ decides whether gapped alignment is the missing input-representation
dimension. Test: for each of 325 Durrant records (both in-pool and not-in-pool),
run Smith-Waterman on the guide-region window vs. a ±20nt neighborhood of the
annotated flank position. Compare best gapped-SW match count to strict WC.
Report distribution of Δm and how it moves the E-value.

  Success criterion (from user): if gold median E under gapped drops from 2.02
  to < 0.5, gapped alignment IS the missing dimension — the "detection floor"
  problem is representation, not model.

W7 decides V5A-3b's last-standing rationale. Test: for each Tnp with ≥ 2 sites,
compute the cross-site variation in gold's nc_start (should be ~0 given shared
ncRNA per Durrant bag) and in the top-1 wrong_orient decoy's nc_start (may
vary because it depends on flank). If gold cross-site nc-consistency is
substantially higher than top wrong_orient decoy's, 3b has a real signal to
attack the 62% wrong_orient significance.

Also produces W6' metadata log: each metric records its (match rule, null,
tie-break, correctness criterion, denominator) tuple.
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
from scipy.stats import binom

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")
sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/scripts")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX
from preprocess.alignment import encode_dna, revcomp
from v5a_eval_core import overlap, find_gold_slot


_A, _C, _G, _T, _N = 0, 1, 2, 3, 4


# ---------------- Simple Smith-Waterman -------------------------------

def _sw(guide: str, target: str, match: int = 1, mismatch: int = 0, gap: int = -1):
    """Basic Smith-Waterman local alignment. Match/mismatch by base identity.
    Returns (best_score, aligned_matches).

    aligned_matches = number of match positions in the best local alignment.
    We keep score and matches separately so we can report match-count consistent
    with the ungapped m units.
    """
    m, n = len(guide), len(target)
    if m == 0 or n == 0: return 0, 0
    H = np.zeros((m + 1, n + 1), dtype=np.int32)
    B = np.zeros((m + 1, n + 1), dtype=np.int8)   # 0=stop, 1=diag, 2=up, 3=left
    for i in range(1, m + 1):
        gi = guide[i - 1]
        for j in range(1, n + 1):
            tj = target[j - 1]
            s = match if (gi == tj) else mismatch
            diag = H[i - 1, j - 1] + s
            up   = H[i - 1, j    ] + gap
            left = H[i    , j - 1] + gap
            best = max(0, diag, up, left)
            H[i, j] = best
            if best == 0: B[i, j] = 0
            elif best == diag: B[i, j] = 1
            elif best == up:   B[i, j] = 2
            else:              B[i, j] = 3
    i, j = np.unravel_index(int(H.argmax()), H.shape)
    best_score = int(H[i, j])
    matches = 0; L_align = 0
    while i > 0 and j > 0 and B[i, j] != 0:
        if B[i, j] == 1:
            if guide[i - 1] == target[j - 1]:
                matches += 1
            L_align += 1
            i -= 1; j -= 1
        elif B[i, j] == 2:
            L_align += 1; i -= 1
        else:
            L_align += 1; j -= 1
    return best_score, matches, L_align


def w4_plus(cog_path, gold_path):
    print(f"\n=== W4+ :: gapped SW recovery on ALL Durrant golds ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    strict_ms = []; sw_ms = []; sw_scores = []; L_alignes = []
    n_recovered_E = 0; n_gold_records = 0
    per_L = defaultdict(lambda: {"strict": [], "sw": []})
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            L = g["target_binding_loop_length"]
            orient = g["target_flank_orientation"]
            nc_start = g["guide_start_in_nc"]; fl_start = g["target_flank_start"]
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            if nc_start + L > len(nc): continue
            guide = nc[nc_start:nc_start + L]
            # Extract ±20nt window around flank position
            w = 20
            lo = max(0, fl_start - w); hi = min(len(flank), fl_start + L + w)
            target = flank[lo:hi]
            if orient == "rc":
                target = revcomp(target)
            # Strict WC ungapped match at annotated coords
            fw = flank[fl_start:fl_start + L]
            if orient == "rc": fw = revcomp(fw)
            ga = encode_dna(guide); ta = encode_dna(fw)
            valid = (ga < _N) & (ta < _N)
            m_strict = int(((ga == ta) & valid).sum())
            # Gapped SW
            sw_score, sw_matches, L_align = _sw(guide, target)
            strict_ms.append(m_strict); sw_ms.append(sw_matches); sw_scores.append(sw_score)
            L_alignes.append(L_align)
            per_L[L]["strict"].append(m_strict); per_L[L]["sw"].append(sw_matches)
            n_gold_records += 1

    strict_ms = np.asarray(strict_ms); sw_ms = np.asarray(sw_ms)
    print(f"  n_records = {n_gold_records}")
    print(f"  ungapped m median = {int(np.median(strict_ms))}   mean = {strict_ms.mean():.2f}")
    print(f"  SW-gapped m median = {int(np.median(sw_ms))}   mean = {sw_ms.mean():.2f}")
    print(f"  Δm distribution: median = {int(np.median(sw_ms - strict_ms))}   "
          f"mean = {(sw_ms - strict_ms).mean():.2f}   "
          f"max = {(sw_ms - strict_ms).max()}   "
          f"frac Δ>0 = {((sw_ms - strict_ms) > 0).mean():.2%}")

    # E-value under ungapped assumption at Durrant scale, using aligned length
    # for gapped case. Simplification: assume same N_windows(L_gold) either way.
    print(f"\n  E[chance ≥ m] at Durrant scale under ungapped (p=0.25) — gold gets:")
    def _E(m, L, p=0.25):
        nc_len = 177; flank_len = 120
        Nw = max(1, (nc_len - L + 1) * (flank_len - L + 1))
        return Nw * float(1.0 - binom.cdf(m - 1, L, p))
    E_ungapped = []; E_gapped = []
    for L in sorted(per_L):
        st = per_L[L]["strict"]; sw = per_L[L]["sw"]
        for m_s, m_w in zip(st, sw):
            E_ungapped.append(_E(m_s, L))
            E_gapped.append(_E(m_w, L))
    E_ungapped = np.asarray(E_ungapped); E_gapped = np.asarray(E_gapped)
    print(f"    ungapped: median E = {np.median(E_ungapped):.2f}   Q10 = {np.percentile(E_ungapped, 10):.2f}   Q90 = {np.percentile(E_ungapped, 90):.2f}")
    print(f"    SW-gapped: median E = {np.median(E_gapped):.2f}   Q10 = {np.percentile(E_gapped, 10):.2f}   Q90 = {np.percentile(E_gapped, 90):.2f}")
    print(f"    frac E<4  ungapped = {(E_ungapped < 4).mean():.2%}")
    print(f"    frac E<4  SW-gapped = {(E_gapped < 4).mean():.2%}")
    print(f"    frac E<0.5 SW-gapped = {(E_gapped < 0.5).mean():.2%}")
    print(f"\n  VERDICT criterion: if median E drops from 2.02 (ungapped) to < 0.5 (gapped),")
    print(f"  gap is the missing input dimension. Otherwise, ungapped WC is not what's holding us back.")
    return {"n": n_gold_records,
              "ungapped_m_median": int(np.median(strict_ms)),
              "gapped_m_median": int(np.median(sw_ms)),
              "delta_m_frac_gt_0": float(((sw_ms - strict_ms) > 0).mean()),
              "ungapped_E_median": float(np.median(E_ungapped)),
              "gapped_E_median": float(np.median(E_gapped)),
              "gapped_frac_E_lt_4": float((E_gapped < 4).mean()),
              "gapped_frac_E_lt_0p5": float((E_gapped < 0.5).mean())}


# ---------------- W7 cross-site consistency ------------------------

def w7_cross_site(cog_path, gold_path):
    print(f"\n=== W7 :: cross-site NC-position consistency (gold vs top wrong_orient decoy) ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    per_tnp_gold_nc = defaultdict(list)
    per_tnp_top_wo_nc = defaultdict(list)
    # For each in-pool record: (tnp_id, gold_nc_start, top_wrong_orient_nc_start)
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            prof = np.zeros((len(nc), 16), dtype=np.float32); val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)
            valid = np.where(mask)[0]
            if len(valid) == 0: continue
            TBL_orient = g["target_flank_orientation"]
            TBL_L = g["target_binding_loop_length"]
            TBL_nc = g["guide_start_in_nc"]; TBL_fl = g["target_flank_start"]

            gold_slot, _ = find_gold_slot(feats, mask, cands, TBL_orient, TBL_L, TBL_nc, TBL_fl)
            if gold_slot < 0: continue
            per_tnp_gold_nc[r["transposase_id"]].append(cands[gold_slot].nc_start)

            # Top-1 wrong_orient decoy (best raw_m among orient != TBL_orient)
            m_arr = feats[valid, 3]
            order = valid[np.argsort(-m_arr, kind="stable")]
            wo_top = -1
            for slot in order:
                if cands[int(slot)].orient != TBL_orient:
                    wo_top = int(slot); break
            if wo_top >= 0:
                per_tnp_top_wo_nc[r["transposase_id"]].append(cands[wo_top].nc_start)

    def _spread(vs):
        if len(vs) < 2: return None
        return int(max(vs) - min(vs))

    gold_spreads = []; wo_spreads = []
    for tnp in per_tnp_gold_nc:
        gs = _spread(per_tnp_gold_nc[tnp])
        if gs is not None: gold_spreads.append(gs)
        ws = _spread(per_tnp_top_wo_nc.get(tnp, []))
        if ws is not None: wo_spreads.append(ws)
    print(f"  n_tnps with >=2 sites: gold={len(gold_spreads)}  wrong_orient_top={len(wo_spreads)}")
    print(f"  gold nc_start SPREAD across sites of same Tnp:  median={int(np.median(gold_spreads))}  "
          f"Q75={int(np.percentile(gold_spreads, 75))}  max={max(gold_spreads)}")
    print(f"  top wrong_orient decoy nc_start SPREAD:         median={int(np.median(wo_spreads))}  "
          f"Q75={int(np.percentile(wo_spreads, 75))}  max={max(wo_spreads)}")
    # Also fraction of Tnps where gold spread is 0 vs wo spread is 0
    print(f"  fraction of Tnps with gold spread == 0:              {(np.asarray(gold_spreads) == 0).mean():.2%}")
    print(f"  fraction of Tnps with top wrong_orient spread == 0:  {(np.asarray(wo_spreads) == 0).mean():.2%}")
    from scipy.stats import mannwhitneyu
    if gold_spreads and wo_spreads:
        u, p = mannwhitneyu(gold_spreads, wo_spreads, alternative="less")
        print(f"  Mann-Whitney U test (gold < wo spread): p = {p:.4g}")
    print(f"\n  VERDICT:")
    print(f"    If gold spread is systematically ~0 and top wo spread is systematically nonzero,")
    print(f"    cross-site NC-position IS a discriminative signal for the 62% wrong_orient")
    print(f"    significant competitors → V5A-3b has a well-defined mechanism.")
    print(f"    If both spreads look similar, cross-site cannot separate them on this axis.")
    return {"gold_spread_median": int(np.median(gold_spreads)),
              "wo_top_spread_median": int(np.median(wo_spreads)),
              "gold_frac_zero_spread": float((np.asarray(gold_spreads) == 0).mean()),
              "wo_frac_zero_spread": float((np.asarray(wo_spreads) == 0).mean()) if wo_spreads else None,
              "mwu_p": float(p) if (gold_spreads and wo_spreads) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_w4 = w4_plus(args.durrant_cog, args.durrant_gold)
    r_w7 = w7_cross_site(args.durrant_cog, args.durrant_gold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"W4_plus": r_w4, "W7": r_w7,
                     "metric_metadata_stub": {
                         "note": "W6' — canonical eval module 5-tuple (match_rule, null_model, tie_break, correctness, denominator) not yet propagated; scripts/v5a_eval_core.py to be extended next",
                     }}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
