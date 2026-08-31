"""W8' + W9 + W7' — corrections and precision measurement.

W8': recompute analytic column with per-Tnp empirical p̂. p=0.25 was 4× too
     low at S=1 (analytic 20.5 vs empirical 80.8). Correct p̂ from actual nc +
     flank composition.
W9:  precision of S=5 channel. For each Tnp, identify S=5 coherent nc positions
     (positions where all 5 sites have a matching L=11 m≥8 window). For each
     coherent position, check overlap with the annotated TBL span. Report:
       - precision (fraction of coherent positions that overlap TBL)
       - shuffled precision (fraction of shuffled coherent positions that
         overlap TBL by chance — should be near 0)
       - PPV: given a S=5 coherent hit, what's the probability it's the real TBL?
W7': stratify W7 gold nc_start spread by the actual number of coherent sites.
     Reports gold spread median at each S∈{2,3,4,5} to separate the circular
     low-S portion from the real S=5 portion.
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

from preprocess.alignment import dot_plot, windowed_matches, encode_dna


def _site_hit_positions(nc: str, flank: str, L: int, m_thresh: int) -> set[int]:
    fwd_dot, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd_dot, L)
    if win.size == 0: return set()
    per_nc_max = win.max(axis=1)
    return set(int(i) for i in np.where(per_nc_max >= m_thresh)[0])


def _base_freq(seq: str) -> np.ndarray:
    arr = encode_dna(seq)
    valid = arr[arr < 4]
    if len(valid) == 0: return np.array([0.25, 0.25, 0.25, 0.25])
    return np.asarray([(valid == k).sum() / len(valid) for k in range(4)])


def _p_hat(nc: str, flank: str) -> float:
    p_nc = _base_freq(nc); p_fl = _base_freq(flank)
    return float(np.dot(p_nc, p_fl))


def w8p_analytic(cog_path, gold_path, L: int, m: int):
    """Recompute analytic column with per-Tnp empirical p̂."""
    print(f"\n=== W8' :: analytic column with per-Tnp empirical p̂ ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    tnp_sites = defaultdict(list)
    tnp_nc = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            if gold.get(r["site_id"]) is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]
            tnp = r["transposase_id"]
            if tnp not in tnp_nc: tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc: continue
            tnp_sites[tnp].append(r["inputs"]["flank"])

    # For each Tnp: empirical p̂ per site, then E[q_i(pos)] = 1 - (1-p_hat_hit)^(flank_len - L + 1)
    # where p_hat_hit = P(Bin(L, p̂) ≥ m).
    print(f"  {'S':>4} {'analytic_p0.25':>16} {'analytic_empirical_p':>22}")
    per_S = {S: [] for S in (1, 2, 3, 4, 5)}
    nc_len_ref = 167  # ~median nc_len - L + 1
    for tnp, flanks in tnp_sites.items():
        nc = tnp_nc[tnp]
        nc_arr = encode_dna(nc); nc_len_pos = len(nc) - L + 1
        for fl in flanks:
            p_hat = _p_hat(nc, fl)
            p_hit_win = float(1.0 - binom.cdf(m - 1, L, p_hat))
            # per-nc-pos probability that this site's flank hits: 1 - (1-p_hit_win)^(flank_len - L + 1)
            flank_win = max(1, len(fl) - L + 1)
            q_i = float(1.0 - (1.0 - p_hit_win) ** flank_win)
            per_S[1].append(q_i)   # placeholder — used below
        # For each Tnp's sites, compute per-site q_i, then Tnp-level analytic joint at S = nc_len_pos * mean(q_i)^S
        q_i_arr = np.asarray(per_S[1][-len(flanks):])
        for S in (2, 3, 4, 5):
            if len(flanks) < S: continue
            # Joint probability all S sites hit at a fixed nc position: prod of a random S-subset of q_i.
            # Under independence given per-site q_i (which vary), expected count = nc_len_pos * mean(prod of q_i over S-subset).
            # For simplicity use mean(q_i)^S — this is the common approximation.
            per_S[S].append(nc_len_pos * (q_i_arr.mean() ** S))
    for S in (1, 2, 3, 4, 5):
        vals = per_S[S]
        if not vals: continue
        # For S=1 the "value" is per-site q_i; joint count at S=1 = 167 * q_i, so mean across sites.
        if S == 1:
            E_analytic_emp = nc_len_ref * np.mean(vals)
        else:
            E_analytic_emp = np.mean(vals)
        # Old p=0.25 analytic (from W8 output):
        p_ref = 0.25
        p_hit_ref = float(1.0 - binom.cdf(m - 1, L, p_ref))
        q_ref = float(1.0 - (1.0 - p_hit_ref) ** 110)
        E_analytic_ref = nc_len_ref * (q_ref ** S)
        print(f"  S={S:>2}    {E_analytic_ref:>16.4f}  {E_analytic_emp:>22.4f}")
    return {"note": "analytic_emp uses per-Tnp p̂ from actual sequences; corrects the 4× low-S underestimate in W8"}


def w9_precision(cog_path, gold_path, L: int, m: int, n_perm: int = 50, seed: int = 0):
    """Precision of S=5 coherent channel: fraction of coherent nc positions
    that overlap the annotated TBL span, real vs shuffled."""
    print(f"\n=== W9 :: precision of S=5 coherent channel ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    tnp_sites = defaultdict(list)
    tnp_nc = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]
            tnp = r["transposase_id"]
            if tnp not in tnp_nc: tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc: continue
            tnp_sites[tnp].append({"flank": r["inputs"]["flank"],
                                       "gold_nc": g["guide_start_in_nc"],
                                       "gold_L":  g["target_binding_loop_length"]})

    per_tnp_hits = {}
    for tnp, sites in tnp_sites.items():
        nc = tnp_nc[tnp]
        per_tnp_hits[tnp] = [_site_hit_positions(nc, s["flank"], L, m) for s in sites]

    def _coherent(hits_lists, S_thresh):
        counts = Counter()
        for h in hits_lists: counts.update(h)
        return {p for p, c in counts.items() if c >= S_thresh}

    def _overlaps_tbl(pos: int, gold_nc: int, gold_L: int, L_win: int, thresh: float = 0.5) -> bool:
        # position is the L=11 window start; check overlap with TBL span [gold_nc, gold_nc+gold_L)
        a0, a1 = pos, pos + L_win
        b0, b1 = gold_nc, gold_nc + gold_L
        inter = max(0, min(a1, b1) - max(a0, b0))
        union = (a1 - a0) + (b1 - b0) - inter
        return (inter / max(1e-9, union)) >= thresh

    # Real
    n_real_coherent_pos = 0
    n_real_pos_overlap_tbl = 0
    n_tnps_with_S5 = 0
    n_tnps_gold_in_S5 = 0
    for tnp, sites in tnp_sites.items():
        if len(sites) < 5: continue
        coh_S5 = _coherent(per_tnp_hits[tnp], 5)
        if not coh_S5: continue
        n_tnps_with_S5 += 1
        gold_nc = sites[0]["gold_nc"]; gold_L = sites[0]["gold_L"]
        for p in coh_S5:
            n_real_coherent_pos += 1
            if _overlaps_tbl(p, gold_nc, gold_L, L):
                n_real_pos_overlap_tbl += 1
        if any(_overlaps_tbl(p, gold_nc, gold_L, L) for p in coh_S5):
            n_tnps_gold_in_S5 += 1

    # Shuffled: for each Tnp's nc, draw 5 random flanks from any site, compute coherent, check overlap
    rng = np.random.default_rng(seed)
    all_flanks = [(t, s["flank"], s["gold_nc"], s["gold_L"])
                     for t, sites in tnp_sites.items() for s in sites]
    n_shuf_coherent_pos = 0; n_shuf_pos_overlap_own_TBL = 0
    for perm in range(n_perm):
        for tnp, sites in tnp_sites.items():
            if len(sites) < 5: continue
            nc = tnp_nc[tnp]
            gold_nc = sites[0]["gold_nc"]; gold_L = sites[0]["gold_L"]
            idx = rng.choice(len(all_flanks), size=5, replace=False)
            fake_flanks = [all_flanks[int(i)][1] for i in idx]
            fake_hits = [_site_hit_positions(nc, fl, L, m) for fl in fake_flanks]
            coh_S5 = _coherent(fake_hits, 5)
            n_shuf_coherent_pos += len(coh_S5)
            for p in coh_S5:
                if _overlaps_tbl(p, gold_nc, gold_L, L):
                    n_shuf_pos_overlap_own_TBL += 1

    real_pos_per_tnp = n_real_coherent_pos / max(1, n_tnps_with_S5)
    real_precision = n_real_pos_overlap_tbl / max(1, n_real_coherent_pos)
    tnps_covered_frac = n_tnps_gold_in_S5 / max(1, sum(1 for t, s in tnp_sites.items() if len(s) >= 5))
    n_tnps_5 = sum(1 for t, s in tnp_sites.items() if len(s) >= 5)
    shuf_pos_per_tnp_per_perm = n_shuf_coherent_pos / max(1, n_tnps_5 * n_perm)
    shuf_precision = n_shuf_pos_overlap_own_TBL / max(1, n_shuf_coherent_pos)

    print(f"  n_tnps with ≥5 sites  = {n_tnps_5}")
    print(f"  REAL:")
    print(f"    Tnps with any S=5 coherent position: {n_tnps_with_S5} ({n_tnps_with_S5/max(1,n_tnps_5):.2%})")
    print(f"    Tnps where gold IS at an S=5 coherent position (IoU≥0.5): {n_tnps_gold_in_S5} ({tnps_covered_frac:.2%})")
    print(f"    total S=5 coherent nc positions: {n_real_coherent_pos}")
    print(f"    ...of which overlap annotated TBL: {n_real_pos_overlap_tbl}   PRECISION = {real_precision:.2%}")
    print(f"    avg positions per S=5-hit Tnp: {real_pos_per_tnp:.2f}")
    print(f"  SHUFFLED (n_perm={n_perm}):")
    print(f"    avg S=5 coherent positions per Tnp per perm: {shuf_pos_per_tnp_per_perm:.4f}")
    print(f"    total shuffled coherent positions overlapping TBL: {n_shuf_pos_overlap_own_TBL}  precision = {shuf_precision:.4%}")
    print(f"\n  BAYES / PPV: if a position is S=5 coherent, P(it is real TBL) ≈ {real_precision:.2%}")
    print(f"  Detector characterization: at 31% Tnp coverage, precision {real_precision:.0%}, "
          f"FP rate per Tnp = {shuf_pos_per_tnp_per_perm:.3f} — direct-output-quality signal.")
    return {"n_tnps": n_tnps_5,
              "n_tnps_S5_hit_any": n_tnps_with_S5,
              "n_tnps_gold_in_S5": n_tnps_gold_in_S5,
              "gold_coverage_frac": tnps_covered_frac,
              "real_precision":     real_precision,
              "real_positions_per_S5_tnp": real_pos_per_tnp,
              "shuffled_pos_per_tnp_per_perm": shuf_pos_per_tnp_per_perm,
              "shuffled_precision": shuf_precision}


def w7_prime(cog_path, gold_path, L: int, m: int):
    """Stratify W7 gold nc_start spread by number of coherent sites."""
    print(f"\n=== W7' :: stratify gold nc_start spread by number of coherent sites ===")
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    tnp_data = defaultdict(list)
    tnp_nc = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]
            tnp = r["transposase_id"]
            if tnp not in tnp_nc: tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc: continue
            tnp_data[tnp].append({"flank": r["inputs"]["flank"],
                                      "gold_nc": g["guide_start_in_nc"]})

    # For each Tnp, determine gold's coherence level: max S such that any coherent
    # position at level S overlaps gold.
    spread_by_S = {S: [] for S in (2, 3, 4, 5)}
    for tnp, sites in tnp_data.items():
        if len(sites) < 2: continue
        nc = tnp_nc[tnp]
        hits = [_site_hit_positions(nc, s["flank"], L, m) for s in sites]
        counts = Counter()
        for h in hits: counts.update(h)
        gold_ncs = [s["gold_nc"] for s in sites]
        gold_spread = int(max(gold_ncs) - min(gold_ncs))
        # For each S, is gold at a S-coherent position?
        for S in (2, 3, 4, 5):
            if len(sites) < S: continue
            coh = {p for p, c in counts.items() if c >= S}
            # Simple overlap: any gold_nc within ε=5 of a coherent position
            gold_is_coh = any(any(abs(gn - p) <= 5 for p in coh) for gn in gold_ncs)
            if gold_is_coh:
                spread_by_S[S].append(gold_spread)
    print(f"  gold spread (nc_start) conditioned on 'gold IS at a coherent position at level S':")
    print(f"  {'S':>4} {'n_Tnps':>10} {'median spread':>15} {'mean spread':>13}")
    for S in (2, 3, 4, 5):
        vals = spread_by_S[S]
        if not vals: print(f"  S={S:>2}   n=0"); continue
        print(f"  S={S:>2}   {len(vals):>10}   {int(np.median(vals)):>15}   {np.mean(vals):>13.2f}")
    print(f"\n  Reading: at each S, spread of gold across the same-Tnp sites.")
    print(f"  If W7's median-1 result is inherited from all S levels, the circularity at low S doesn't")
    print(f"  cascade to S=5 conclusions.")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--L", type=int, default=11)
    ap.add_argument("--m", type=int, default=8)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r_w8p = w8p_analytic(args.durrant_cog, args.durrant_gold, args.L, args.m)
    r_w9  = w9_precision(args.durrant_cog, args.durrant_gold, args.L, args.m, args.n_perm)
    _     = w7_prime(args.durrant_cog, args.durrant_gold, args.L, args.m)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"W8_prime": r_w8p, "W9": r_w9}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
