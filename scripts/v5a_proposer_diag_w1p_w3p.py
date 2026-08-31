"""W1' + W3' — corrections after W1 asymmetric-null error and W3 different_region rehabilitation.

W1': symmetric wobble test.
  - Compute gold m under STRICT (WC only) and WOBBLE (WC + G-T + T-G).
  - Compute Bernoulli null p_strict and p_wobble under TWO conventions:
      (a) fixed uniform p_strict = 0.25, p_wobble = 6/16 = 0.375
      (b) per-record p_hat estimated from that record's actual nc + flank
          base composition
  - Report:
      per-record ΔE = E_wobble − E_strict (using matched null in each case)
      records that move E<4 to E≥4 under wobble  (regressions)
      records that move E≥4 to E<4 under wobble  (rescues)
      net rescue count
  - Verdict: is wobble a net-positive addition to the proposer's match rule?

W3': significance profile of top-K wrong_orientation and different_region decoys.
  - For each Durrant record, compute E-value under STRICT p=0.25 (Bin(L, p̂))
    for every valid pool candidate. Also for the gold candidate.
  - Report distributions of E_gold vs. E for top-K wrong_orient decoys and
    top-K different_region decoys.
  - If top decoys have E >> 4 uniformly, they are noise the proposer never
    should have admitted. If they cluster near gold's E, they are real
    high-significance alternatives.
  - This decides whether V5A-3b's "different_region 55%" is a biological
    hard case OR sampling arithmetic from an uncalibrated proposer.
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


def _basewise_p(a_arr, b_arr, wobble: bool) -> float:
    """Empirical P(base_a matches base_b) under strict or wobble rule."""
    a = a_arr[a_arr < _N]; b = b_arr[b_arr < _N]
    if len(a) == 0 or len(b) == 0: return 0.25
    def _freq(arr):
        n = len(arr)
        return np.asarray([(arr == k).sum() / n for k in range(4)], dtype=np.float64)
    pa = _freq(a); pb = _freq(b)
    # strict: sum_k pa[k] * pb[k]
    p_strict = float(np.dot(pa, pb))
    if not wobble: return p_strict
    # wobble adds pa[G]*pb[T] + pa[T]*pb[G]
    p = p_strict + pa[_G] * pb[_T] + pa[_T] * pb[_G]
    return float(min(0.99, p))


def _gold_matches(nc: str, flank: str, orient: str, L: int, nc_start: int, fl_start: int,
                     wobble: bool) -> int:
    """Compute m at the annotated gold coords under strict or wobble."""
    if nc_start + L > len(nc) or fl_start + L > len(flank): return -1
    ga = encode_dna(nc[nc_start:nc_start + L])
    if orient == "fwd":
        ta = encode_dna(flank[fl_start:fl_start + L])
    else:
        ta = encode_dna(revcomp(flank[fl_start:fl_start + L]))
    valid = (ga < _N) & (ta < _N)
    m_strict = int(((ga == ta) & valid).sum())
    if not wobble: return m_strict
    wob = ((ga == _G) & (ta == _T)) | ((ga == _T) & (ta == _G))
    return m_strict + int((wob & valid).sum())


def w1_prime(cog_path, gold_path):
    print(f"\n=== W1' :: symmetric wobble under matched null ===")
    print(f"  Fixed nulls: p_strict = 0.25, p_wobble = 0.375")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    # Fixed uniform-p case
    E_strict_uni = []; E_wobble_uni = []
    # Per-record empirical-p case
    E_strict_emp = []; E_wobble_emp = []
    Ls = []
    net_rescue_uni = 0; net_regress_uni = 0
    net_rescue_emp = 0; net_regress_emp = 0
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
            m_s = _gold_matches(nc, flank, orient, L, nc_start, fl_start, wobble=False)
            m_w = _gold_matches(nc, flank, orient, L, nc_start, fl_start, wobble=True)
            if m_s < 0 or m_w < 0: continue
            Ls.append(L)
            # search-space size, per orient
            nc_len = len(nc); flank_len = len(flank)
            N_win = max(1, (nc_len - L + 1) * (flank_len - L + 1))
            # UNIFORM p null
            p_s_uni = 0.25; p_w_uni = 0.375
            E_s_uni = N_win * float(1.0 - binom.cdf(m_s - 1, L, p_s_uni))
            E_w_uni = N_win * float(1.0 - binom.cdf(m_w - 1, L, p_w_uni))
            E_strict_uni.append(E_s_uni); E_wobble_uni.append(E_w_uni)
            if E_s_uni >= 4 and E_w_uni < 4: net_rescue_uni += 1
            if E_s_uni < 4 and E_w_uni >= 4: net_regress_uni += 1
            # EMPIRICAL per-record p
            nc_arr = encode_dna(nc); flank_arr = encode_dna(flank)
            if orient == "rc":
                target_arr_full = encode_dna(revcomp(flank))
            else:
                target_arr_full = flank_arr
            p_s_emp = _basewise_p(nc_arr, target_arr_full, wobble=False)
            p_w_emp = _basewise_p(nc_arr, target_arr_full, wobble=True)
            E_s_emp = N_win * float(1.0 - binom.cdf(m_s - 1, L, p_s_emp))
            E_w_emp = N_win * float(1.0 - binom.cdf(m_w - 1, L, p_w_emp))
            E_strict_emp.append(E_s_emp); E_wobble_emp.append(E_w_emp)
            if E_s_emp >= 4 and E_w_emp < 4: net_rescue_emp += 1
            if E_s_emp < 4 and E_w_emp >= 4: net_regress_emp += 1

    n = len(E_strict_uni)
    print(f"  n_records = {n}")
    print(f"  UNIFORM null (p_strict=0.25, p_wobble=0.375):")
    print(f"    median E_strict = {np.median(E_strict_uni):.2f}   median E_wobble = {np.median(E_wobble_uni):.2f}")
    print(f"    E<4 count STRICT = {sum(1 for e in E_strict_uni if e < 4)}/{n}")
    print(f"    E<4 count WOBBLE = {sum(1 for e in E_wobble_uni if e < 4)}/{n}")
    print(f"    rescued  by wobble = {net_rescue_uni}")
    print(f"    regressed by wobble = {net_regress_uni}")
    print(f"    NET rescue = {net_rescue_uni - net_regress_uni}   ({(net_rescue_uni - net_regress_uni)/n:+.2%})")
    print()
    print(f"  EMPIRICAL per-record p (from actual nc + flank composition):")
    print(f"    median E_strict = {np.median(E_strict_emp):.2f}   median E_wobble = {np.median(E_wobble_emp):.2f}")
    print(f"    E<4 count STRICT = {sum(1 for e in E_strict_emp if e < 4)}/{n}")
    print(f"    E<4 count WOBBLE = {sum(1 for e in E_wobble_emp if e < 4)}/{n}")
    print(f"    rescued  by wobble = {net_rescue_emp}")
    print(f"    regressed by wobble = {net_regress_emp}")
    print(f"    NET rescue = {net_rescue_emp - net_regress_emp}   ({(net_rescue_emp - net_regress_emp)/n:+.2%})")
    print()
    print(f"  VERDICT: if NET rescue <= 0, wobble should NOT be a matching rule in proposer v2.")
    return {
        "n": n,
        "uniform_net_rescue": net_rescue_uni - net_regress_uni,
        "empirical_net_rescue": net_rescue_emp - net_regress_emp,
    }


def w3_prime(cog_path, gold_path):
    print(f"\n=== W3' :: E-value profile of wrong_orient + different_region top-K decoys ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    # For each Durrant record, compute E-value for every valid candidate under
    # strict WC null (p=0.25 uniform + p_hat empirical); collect distribution.
    all_gold_E = []
    per_bucket_E = defaultdict(list)   # bucket → list of E for top-8 decoys of that bucket
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
            TBL_orient = g["target_flank_orientation"]; TBL_L = g["target_binding_loop_length"]
            TBL_nc = g["guide_start_in_nc"]; TBL_fl = g["target_flank_start"]
            gold_slot, _ = find_gold_slot(feats, mask, cands, TBL_orient, TBL_L, TBL_nc, TBL_fl)
            if gold_slot < 0: continue

            # E-values under strict uniform-p=0.25
            valid = np.where(mask)[0]
            nc_len = len(nc); flank_len = len(flank)
            def _E(m, L):
                Nw = max(1, (nc_len - L + 1) * (flank_len - L + 1))
                return Nw * float(1.0 - binom.cdf(m - 1, L, 0.25))
            gold_c = cands[gold_slot]
            E_gold = _E(int(feats[gold_slot, 3]), int(gold_c.L))
            all_gold_E.append(E_gold)

            # Top-8 raw_m decoys and classify vs gold_slot
            order = valid[np.argsort(-feats[valid, 3], kind="stable")]
            decoys = order[order != gold_slot][:8]
            for slot in decoys:
                c = cands[int(slot)]
                if c.orient != TBL_orient: bucket = "wrong_orientation"
                else:
                    mn = min(c.L, gold_c.L)
                    nc_ov = overlap(c.nc_start, c.nc_start + c.L,
                                       gold_c.nc_start, gold_c.nc_start + gold_c.L)
                    f_ov = overlap(c.flank_start, c.flank_start + c.L,
                                       gold_c.flank_start, gold_c.flank_start + gold_c.L)
                    th = 0.5 * mn
                    if nc_ov < th: bucket = "different_region"
                    elif c.L > gold_c.L: bucket = "same_region_longer_L"
                    elif c.L < gold_c.L: bucket = "same_region_shorter_L"
                    elif f_ov < th: bucket = "same_region_same_L_wrong_flank"
                    else: bucket = "near_gold"
                E = _E(int(feats[int(slot), 3]), int(c.L))
                per_bucket_E[bucket].append(E)

    print(f"  gold E-value distribution (strict WC, p=0.25):")
    print(f"    n={len(all_gold_E)}  median={np.median(all_gold_E):.2f}  "
          f"Q10={np.percentile(all_gold_E, 10):.2f}  Q90={np.percentile(all_gold_E, 90):.2f}")
    print(f"    fraction of golds with E < 4  (significant vs. noise): "
          f"{(np.asarray(all_gold_E) < 4).mean():.2%}")
    print()
    print(f"  top-8 decoy E-value distribution by taxonomy bucket:")
    print(f"  {'bucket':<32} {'n':>7} {'median E':>10} {'frac E<4':>10} {'frac E<1':>10}  reading")
    for k in ("wrong_orientation", "different_region", "same_region_longer_L",
                "same_region_shorter_L", "same_region_same_L_wrong_flank", "near_gold"):
        Es = np.asarray(per_bucket_E.get(k, []), dtype=np.float64)
        if len(Es) == 0:
            print(f"  {k:<32} n=0"); continue
        frac4 = (Es < 4).mean(); frac1 = (Es < 1).mean()
        med = np.median(Es)
        # Reading
        if frac4 < 0.10:
            read = "NOISE — proposer artifact"
        elif frac4 > 0.5:
            read = "significant — real competitor"
        else:
            read = "mixed"
        print(f"  {k:<32} {len(Es):>7} {med:>10.2f} {frac4:>10.2%} {frac1:>10.2%}  {read}")
    print()
    print(f"  Verdict logic:")
    print(f"    If wrong_orientation + different_region are >90% E>=4 → they are the")
    print(f"    W2 statistical noise; V5A-3b's rationale (they are 91% of top-8 and")
    print(f"    raw_m is chance) is a proposer calibration failure, not biology.")
    print(f"    Fix by E-value ranking in proposer v2, not by cross-site model.")

    return {
        "gold_median_E": float(np.median(all_gold_E)),
        "gold_frac_E_lt_4": float((np.asarray(all_gold_E) < 4).mean()),
        "per_bucket": {k: {"n": len(v),
                              "median_E": float(np.median(v)) if v else None,
                              "frac_E_lt_4": float((np.asarray(v) < 4).mean()) if v else None,
                              "frac_E_lt_1": float((np.asarray(v) < 1).mean()) if v else None}
                          for k, v in per_bucket_E.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_w1 = w1_prime(args.durrant_cog, args.durrant_gold)
    r_w3 = w3_prime(args.durrant_cog, args.durrant_gold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"W1_prime": r_w1, "W3_prime": r_w3}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
