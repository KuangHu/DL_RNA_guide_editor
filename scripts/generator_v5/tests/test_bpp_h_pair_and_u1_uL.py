"""Two quick checks flagged after 0a/0b commit:

Check 1 — H_pair distribution shape at gold vs competitor.
  On item 4.5's 265 pairs, is E_span<->H_pair r=-0.965 (gold) vs
  r=-0.069 (competitor) a real "decorrelation is the signal" or a
  value-domain collapse artefact (competitor H all near zero, no
  variance for correlation to see)?

  Report full quantiles of H_pair at gold vs competitor. If 80%+ of
  competitors sit in [0, 0.1], call it (b) collapse. If competitors
  spread across [0, ~1], call it (a) real decorrelation.

Check 2 — dG_open_u1 <-> dG_open_uL_pn correlation at gold and comp.
  Physically distinct (u=L involves cooperativity), but empirically
  is uL_pn = mean(u1)? If Spearman r > 0.98, u1 is redundant with
  uL_pn on window-averaged use. Kept as separate channel is worth
  the cost only if r drops.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from preprocess.features_structure_v2 import compute_features_v2
from scripts.v5a_framework.match_table import load as load_mt

MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
MAX_SITES_PER_TNP = 8
QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def top_competitor(m: np.ndarray, gold_pos: int) -> int | None:
    n = len(m)
    if n <= 1:
        return None
    masked = m.copy().astype(np.int32)
    if 0 <= gold_pos < n:
        masked[gold_pos] = -1
    return int(np.argmax(masked))


def main() -> int:
    mt = load_mt(MT_POS)
    print(f"[checks] {len(mt.tnp_ids)} Durrant Tnps")

    gold_H = []
    comp_H = []
    gold_u1 = []
    gold_uL = []
    comp_u1 = []
    comp_uL = []
    t0 = time.perf_counter()

    for ti, tnp_id in enumerate(mt.tnp_ids):
        rec = mt.tnps[tnp_id]
        nc = rec.nc.upper().replace("U", "T")
        gold_Ls = [s.gold_L for s in rec.sites if s.gold_L]
        if not gold_Ls:
            continue
        gL = int(np.median(gold_Ls))
        feats = compute_features_v2(nc, guide_length=gL)
        p_ss = feats.p_ss
        csum = np.concatenate(([0.0], np.cumsum(feats.dG_open_u1, dtype=np.float64)))
        # per-window mean of dG_open_u1 (for direct check vs dG_open_uL_pn)
        u1_win_mean = (csum[gL:] - csum[:-gL]) / gL
        n_win = len(feats.dG_open_uL_pn)

        used = 0
        for s in rec.sites:
            if used >= MAX_SITES_PER_TNP:
                break
            if s.gold_nc is None or s.gold_L is None:
                continue
            if not (0 <= s.gold_nc < n_win):
                continue
            try:
                m_f = mt.m_max(tnp_id, s.site_idx, "fwd", gL)
                m_r = mt.m_max(tnp_id, s.site_idx, "rc", gL)
            except KeyError:
                continue
            n_common = min(len(m_f), len(m_r), n_win)
            m_pooled = np.maximum(m_f[:n_common], m_r[:n_common])
            cp = top_competitor(m_pooled, s.gold_nc)
            if cp is None or cp == s.gold_nc or not (0 <= cp < n_win):
                continue

            for role, pos, H_arr, u1_arr, uL_arr in [
                ("gold", s.gold_nc, gold_H, gold_u1, gold_uL),
                ("comp", cp,        comp_H, comp_u1, comp_uL),
            ]:
                h = float(feats.H_pair_win[pos])
                if not np.isnan(h):
                    H_arr.append(h)
                    u1_arr.append(float(u1_win_mean[pos]))
                    uL_arr.append(float(feats.dG_open_uL_pn[pos]))
            used += 1
        mt.evict(tnp_id)

    print(f"[checks] extracted {len(gold_H)} gold + {len(comp_H)} comp values "
          f"in {time.perf_counter() - t0:.1f}s")

    print()
    print("=== Check 1: H_pair distribution ===")
    print(f"  {'quantile':<10s} {'gold':>10s} {'competitor':>12s}")
    gH = np.array(gold_H)
    cH = np.array(comp_H)
    for q in QUANTILES:
        print(f"  q{int(q*100):>02d}       {float(np.quantile(gH, q)):>10.3f} "
              f"{float(np.quantile(cH, q)):>12.3f}")
    print(f"  {'mean':<10s} {gH.mean():>10.3f} {cH.mean():>12.3f}")
    print(f"  {'std':<10s} {gH.std():>10.3f} {cH.std():>12.3f}")
    frac_low = float((cH <= 0.1).mean())
    print()
    print(f"  fraction of competitor H_pair <= 0.1: {frac_low * 100:.1f}%")
    if frac_low >= 0.80:
        verdict = "(b) VALUE-DOMAIN COLLAPSE: competitor H clustered near 0, r=0 is not signal"
    else:
        # Compute r via Spearman on this same data as a self-check
        from scipy.stats import spearmanr
        r_g, _ = spearmanr(gH, gold_u1)   # placeholder to reuse import
        r_g2 = float(spearmanr(gH, np.array(gold_u1)).correlation)
        # The correlation of interest is E_span vs H_pair; approximate here via
        # printing the spread and letting reader infer:
        verdict = "(a) REAL DECORRELATION: competitor H spans a substantial range"
    print(f"  verdict: {verdict}")

    print()
    print("=== Check 2: dG_open_u1 (window mean) vs dG_open_uL_pn ===")
    print("  physically distinct (u=L includes cooperativity beyond sum of u=1)")
    print("  but empirically may be near-duplicates on window average.")
    from scipy.stats import spearmanr
    r_gold, _ = spearmanr(gold_u1, gold_uL)
    r_comp, _ = spearmanr(comp_u1, comp_uL)
    print(f"  Spearman r(mean(u1), uL_pn) on gold        = {r_gold:+.4f}")
    print(f"  Spearman r(mean(u1), uL_pn) on competitor  = {r_comp:+.4f}")
    if abs(r_gold) > 0.98 and abs(r_comp) > 0.98:
        print("  verdict: NEAR-DUPLICATE on both. If keeping only one, keep uL_pn "
              "(more physically meaningful).")
    else:
        print("  verdict: distinct. Cooperativity is visible in the delta.")

    # Also report min/max of ratio uL_pn / mean(u1)
    r_ratio_gold = np.array(gold_uL) / np.maximum(np.array(gold_u1), 1e-6)
    r_ratio_comp = np.array(comp_uL) / np.maximum(np.array(comp_u1), 1e-6)
    print(f"  ratio (uL_pn / mean(u1)) on gold        : "
          f"median={float(np.median(r_ratio_gold)):.4f}, "
          f"p25={float(np.percentile(r_ratio_gold, 25)):.4f}, "
          f"p75={float(np.percentile(r_ratio_gold, 75)):.4f}")
    print(f"  ratio (uL_pn / mean(u1)) on competitor  : "
          f"median={float(np.median(r_ratio_comp)):.4f}, "
          f"p25={float(np.percentile(r_ratio_comp, 25)):.4f}, "
          f"p75={float(np.percentile(r_ratio_comp, 75)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
