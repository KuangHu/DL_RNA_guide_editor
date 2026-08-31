"""V5A W1 + W2 + W3 proposer diagnostics.

W1: G-U wobble effect on Durrant gold match count and expected chance competitors.
    Compare STRICT WC dot_plot vs WOBBLE-extended (allow G-T and T-G pairs).
W2: Detection-limit landscape: E[chance ≥ m] in (L, m) plane for pool sizes
    Durrant nc=177, flank=120. Report E[chance]=4 boundary. Overlay Durrant
    and V4.2 gold (L, m) points.
W3: Overlap analysis. For every Durrant record:
    - fraction of pool candidates that overlap (IoU ≥ 0.5) the annotated TBL span
    - fraction of top-8 raw_m decoys overlapping TBL (already known ≈ 4%)
    - broken out by decoy taxonomy: how many "longer_L", "shorter_L", "near_gold",
      "same_region_same_L_wrong_flank" are the same biological site as gold
    - what fraction of top-8 decoys are same-length variants of the SAME site
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


# --- W1: wobble-aware dot plot -------------------------------------------

_A, _C, _G, _T, _N = 0, 1, 2, 3, 4
_COMP = np.array([_T, _G, _C, _A, _N], dtype=np.int8)


def _wobble_match_arr(nc_arr: np.ndarray, target_arr: np.ndarray) -> np.ndarray:
    """1 iff nc[i] and target[j] form a WC or wobble pair.
    WC: strict base equality within same strand (nc same as flank for fwd).
    Wobble: guide G ↔ target T (aka G-U in RNA-DNA hybrid); guide T ↔ target G.
    Both nc_arr and target_arr are int8-coded (A=0, C=1, G=2, T=3, N=4).
    target_arr is either flank_arr (fwd) or revcomp(flank)_arr (rc).
    """
    strict = (nc_arr[:, None] == target_arr[None, :])
    G_T = (nc_arr[:, None] == _G) & (target_arr[None, :] == _T)
    T_G = (nc_arr[:, None] == _T) & (target_arr[None, :] == _G)
    valid = (nc_arr[:, None] < _N) & (target_arr[None, :] < _N)
    return (strict | G_T | T_G) & valid


def _strict_match_arr(nc_arr, target_arr):
    valid = (nc_arr[:, None] < _N) & (target_arr[None, :] < _N)
    return (nc_arr[:, None] == target_arr[None, :]) & valid


def w1_wobble(cog_path, gold_path, n_positions=None):
    print(f"\n=== W1 :: G-U wobble effect ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    strict_ms = []; wobble_ms = []; L_vals = []; delta_m = []
    n_recovered = 0
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            L = g["target_binding_loop_length"]
            orient = g["target_flank_orientation"]
            nc_start = g["guide_start_in_nc"]
            fl_start = g["target_flank_start"]
            active_nc = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs): active_nc = 0
            nc = ncs[active_nc]; flank = r["inputs"]["flank"]
            if nc_start + L > len(nc): continue
            if fl_start + L > len(flank): continue
            guide_arr = encode_dna(nc[nc_start:nc_start + L])
            # flank window in the alignment's orientation
            if orient == "fwd":
                target_arr = encode_dna(flank[fl_start:fl_start + L])
            else:
                target_arr = encode_dna(revcomp(flank[fl_start:fl_start + L]))
            valid = (guide_arr < _N) & (target_arr < _N)
            m_strict = int(((guide_arr == target_arr) & valid).sum())
            wob = ((guide_arr == _G) & (target_arr == _T)) | \
                  ((guide_arr == _T) & (target_arr == _G))
            m_wobble = m_strict + int((wob & valid).sum())
            strict_ms.append(m_strict); wobble_ms.append(m_wobble); L_vals.append(L)
            delta_m.append(m_wobble - m_strict)
    print(f"  n_records = {len(strict_ms)}")
    print(f"  gold m (STRICT WC)  median={int(np.median(strict_ms))}  mean={np.mean(strict_ms):.2f}")
    print(f"  gold m (+wobble)    median={int(np.median(wobble_ms))}  mean={np.mean(wobble_ms):.2f}")
    print(f"  Δm from wobble       median={int(np.median(delta_m))}  mean={np.mean(delta_m):.2f}  "
          f"max={max(delta_m)}")
    # Detection-limit change
    print(f"\n  E[chance ≥ m] at Durrant scale (nc=177, flank=120, per orient):")
    print(f"    L    m_strict E[chance] | m_wobble E[chance]")
    for L, m_s, m_w in list(zip(L_vals, strict_ms, wobble_ms))[:0]: pass
    # aggregate by L
    per_L = defaultdict(lambda: {"strict": [], "wobble": []})
    for L, m_s, m_w in zip(L_vals, strict_ms, wobble_ms):
        per_L[L]["strict"].append(m_s); per_L[L]["wobble"].append(m_w)
    for L in sorted(per_L):
        ms = int(np.median(per_L[L]["strict"])); mw = int(np.median(per_L[L]["wobble"]))
        Nw = max(1, (177 - L + 1) * (120 - L + 1))
        # E[chance] = N_windows × P(Bin(L, 0.25) ≥ m) -- forward orientation only
        p_ge_strict = float(1.0 - binom.cdf(ms - 1, L, 0.25))
        p_ge_wobble = float(1.0 - binom.cdf(mw - 1, L, 0.25))
        E_strict = Nw * p_ge_strict; E_wobble = Nw * p_ge_wobble
        print(f"    L={L:<3} m={ms}→{mw}     E_strict={E_strict:>8.1f}  E_wobble={E_wobble:>8.1f}   n={len(per_L[L]['strict'])}")

    # How many "not in pool" would be rescued if wobble m were used?
    # Rough criterion: E_wobble < 4 → gets into top-4 per bucket
    n_rescue = 0; n_total = 0
    for L, m_s, m_w in zip(L_vals, strict_ms, wobble_ms):
        Nw = max(1, (177 - L + 1) * (120 - L + 1))
        p_s = float(1.0 - binom.cdf(m_s - 1, L, 0.25))
        p_w = float(1.0 - binom.cdf(m_w - 1, L, 0.25))
        if Nw * p_s >= 4 and Nw * p_w < 4:
            n_rescue += 1
        n_total += 1
    print(f"\n  Records that WOULD MOVE from 'above detection limit E>=4' to 'below E<4' with wobble:  {n_rescue}/{n_total} = {n_rescue/max(1,n_total):.2%}")
    return {"n_records": len(strict_ms),
              "gold_m_strict_median": int(np.median(strict_ms)),
              "gold_m_wobble_median": int(np.median(wobble_ms)),
              "delta_m_median": int(np.median(delta_m)),
              "delta_m_mean": float(np.mean(delta_m)),
              "delta_m_max": int(max(delta_m)),
              "rescue_frac_by_detection_limit": n_rescue / max(1, n_total)}


# --- W2: detection-limit landscape ---------------------------------------

def w2_landscape():
    print(f"\n=== W2 :: detection-limit landscape (E[chance] at Durrant scale) ===")
    nc_len = 177; flank_len = 120
    print(f"  E[chance ≥ m | L, p=0.25] = N_windows(L) × P(Bin(L, 0.25) ≥ m)")
    print(f"  N_windows(L) = (nc_len - L + 1) × (flank_len - L + 1)")
    print(f"  ~= per orient. K=4 per bucket → E ≥ 4 means gold cannot make top-4.")
    print()
    print(f"  {'L':>4} {'N_win':>7}  ", end="")
    for m_rel in ("m/L=1.0", "m/L=0.92", "m/L=0.83", "m/L=0.75", "m/L=0.67", "m/L=0.60"):
        print(f"{m_rel:>10}", end="")
    print()
    for L in range(5, 17):
        Nw = (nc_len - L + 1) * (flank_len - L + 1)
        row = f"  {L:>4} {Nw:>7}  "
        for f in (1.0, 0.92, 0.83, 0.75, 0.67, 0.60):
            m = int(round(f * L))
            p = float(1.0 - binom.cdf(m - 1, L, 0.25))
            E = Nw * p
            row += f"{E:>10.2f}"
        print(row)


# --- W3: overlap analysis ------------------------------------------------

def _iou(a0, a1, b0, b1):
    inter = overlap(a0, a1, b0, b1)
    union = (a1 - a0) + (b1 - b0) - inter
    return inter / max(1e-9, union)


def _same_site(c, TBL_orient, TBL_L, TBL_nc, TBL_fl, thresh=0.5):
    """A candidate is 'same biological site' if orient matches AND its span
    IoU-overlaps the TBL span in BOTH nc and flank."""
    if c.orient != TBL_orient: return False
    nc_iou = _iou(c.nc_start, c.nc_start + c.L, TBL_nc, TBL_nc + TBL_L)
    fl_iou = _iou(c.flank_start, c.flank_start + c.L, TBL_fl, TBL_fl + TBL_L)
    return nc_iou >= thresh and fl_iou >= thresh


def w3_overlap(cog, gold_path):
    print(f"\n=== W3 :: overlap analysis (same-biological-site duplication) ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    total_overlap_frac = []
    top8_overlap = 0; top8_total = 0
    same_site_by_L = Counter()
    top8_by_bucket = Counter()
    top8_same_site_by_bucket = Counter()
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
            TBL_orient = g["target_flank_orientation"]; TBL_L = g["target_binding_loop_length"]
            TBL_nc = g["guide_start_in_nc"]; TBL_fl = g["target_flank_start"]

            valid = np.where(mask)[0]
            n_pool = len(valid)
            n_overlap = 0
            per_L_overlap = Counter()
            for i in valid:
                c = cands[int(i)]
                if _same_site(c, TBL_orient, TBL_L, TBL_nc, TBL_fl):
                    n_overlap += 1
                    per_L_overlap[c.L] += 1
            if n_pool == 0: continue
            total_overlap_frac.append(n_overlap / n_pool)
            for L, k in per_L_overlap.items():
                same_site_by_L[L] += k

            # tolerant-matched gold slot for taxonomy classification
            gold_slot, _ = find_gold_slot(feats, mask, cands, TBL_orient, TBL_L, TBL_nc, TBL_fl)
            if gold_slot < 0: continue
            # top-8 decoys by raw m
            m_arr = feats[:, 3]
            order = valid[np.argsort(-m_arr[valid], kind="stable")]
            decoy_slots = order[order != gold_slot][:8]
            gold_c = cands[gold_slot]
            for slot in decoy_slots:
                c = cands[int(slot)]
                same = _same_site(c, TBL_orient, TBL_L, TBL_nc, TBL_fl)
                top8_total += 1
                if same: top8_overlap += 1
                # Classify decoy vs gold_slot span (existing taxonomy)
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
                top8_by_bucket[bucket] += 1
                if same: top8_same_site_by_bucket[bucket] += 1

    print(f"  pool candidates overlapping annotated TBL (IoU ≥ 0.5 on both): median frac = {np.median(total_overlap_frac):.3%}  "
          f"mean = {np.mean(total_overlap_frac):.3%}")
    print(f"  distribution of same-site pool candidates by L: {sorted(same_site_by_L.items())}")
    print(f"\n  top-8 raw_m decoys overlapping annotated TBL: {top8_overlap}/{top8_total} = {top8_overlap/max(1,top8_total):.3%}")
    print(f"\n  top-8 decoy taxonomy vs. how many are actually the same biological site:")
    print(f"  {'bucket':<32} {'n_decoys':>10} {'n_same_site':>13} {'frac_same_site':>16}")
    for k in ("wrong_orientation","different_region","same_region_longer_L",
                "same_region_shorter_L","same_region_same_L_wrong_flank","near_gold"):
        n = top8_by_bucket.get(k, 0)
        ss = top8_same_site_by_bucket.get(k, 0)
        frac = ss / max(1, n)
        print(f"  {k:<32} {n:>10} {ss:>13} {frac:>16.3%}")
    print(f"\n  Reading: any bucket with high 'frac_same_site' is the length-variant self-competition problem.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_w1 = w1_wobble(args.durrant_cog, args.durrant_gold)
    w2_landscape()
    w3_overlap(args.durrant_cog, args.durrant_gold)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"W1": r_w1}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
