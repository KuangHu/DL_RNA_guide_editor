"""Cooperativity | p_ss stratified check.

Cooperativity has a mathematical link to accessibility level:
  coop = (RT/L) * ln[P_joint / prod P_i]
When p_i are near 1, both P_joint and prod P_i are near 1 -> coop near 0.
When p_i are lower, the ratio has more room to move -> coop can be larger.

So the observed pattern (gold higher p_ss = 0.551, gold lower coop = 0.114;
competitor lower p_ss = 0.374, competitor higher coop = 0.970) MAY be a
monotone restatement of accessibility rather than an independent signal.

Test: within p_ss_window buckets (deciles of the paired mean or of the
gold/comp values separately), does cooperativity still discriminate?

Approaches:
  A. Pair-bucket by mean(p_ss_gold, p_ss_comp), report P(coop_gold_lower)
     within each bucket.
  B. Regress cooperativity on p_ss (all points, gold + comp pooled),
     take residuals, then compute P(coop_gold_lower_residual).

If A and B both stay >= 0.75 within buckets, cooperativity is independent.
If they collapse to ~0.5, coop is a restatement of accessibility.
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
N_BUCKETS = 5  # quintiles (safer than deciles for n=265)


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
    print(f"[coop|p_ss] {len(mt.tnp_ids)} Tnps")
    t0 = time.perf_counter()

    p_ss_gold = []; p_ss_comp = []
    coop_gold = []; coop_comp = []
    tnp_of_pair = []

    for tnp_id in mt.tnp_ids:
        rec = mt.tnps[tnp_id]
        nc = rec.nc.upper().replace("U", "T")
        gLs = [s.gold_L for s in rec.sites if s.gold_L]
        if not gLs:
            continue
        gL = int(np.median(gLs))
        feats = compute_features_v2(nc, guide_length=gL)
        p_ss = feats.p_ss
        csum = np.concatenate(([0.0], np.cumsum(p_ss, dtype=np.float64)))
        p_ss_win = (csum[gL:] - csum[:-gL]) / gL
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
            p_ss_gold.append(float(p_ss_win[s.gold_nc]))
            p_ss_comp.append(float(p_ss_win[cp]))
            coop_gold.append(float(feats.cooperativity_win_pn[s.gold_nc]))
            coop_comp.append(float(feats.cooperativity_win_pn[cp]))
            tnp_of_pair.append(tnp_id)
            used += 1
        mt.evict(tnp_id)

    p_ss_gold = np.array(p_ss_gold); p_ss_comp = np.array(p_ss_comp)
    coop_gold = np.array(coop_gold); coop_comp = np.array(coop_comp)
    n = len(p_ss_gold)
    print(f"[coop|p_ss] {n} pairs in {time.perf_counter()-t0:.1f}s")

    # Overall discrimination baseline (should match 4.5's 0.909)
    p_gold_lower = float((coop_gold < coop_comp).mean())
    print(f"\nOverall P(coop_gold < coop_comp) = {p_gold_lower:.3f}   "
          f"(4.5 reported P(gold_better) = 0.909)")

    # === Approach A: bucket by mean p_ss ===
    print()
    print("=== Approach A: bucket pairs by mean(p_ss_gold, p_ss_comp) ===")
    mean_p = 0.5 * (p_ss_gold + p_ss_comp)
    quantiles = np.linspace(0, 1, N_BUCKETS + 1)
    edges = np.quantile(mean_p, quantiles)
    print(f"  {'bucket':<20s} {'n':>4s} {'p_gold<comp':>12s} {'coop_g_med':>11s} {'coop_c_med':>11s}")
    for b in range(N_BUCKETS):
        lo, hi = edges[b], edges[b + 1]
        if b == N_BUCKETS - 1:
            mask = (mean_p >= lo) & (mean_p <= hi)
        else:
            mask = (mean_p >= lo) & (mean_p < hi)
        nb = int(mask.sum())
        if nb == 0:
            continue
        p_lower = float((coop_gold[mask] < coop_comp[mask]).mean())
        g_med = float(np.median(coop_gold[mask]))
        c_med = float(np.median(coop_comp[mask]))
        print(f"  [{lo:.3f}, {hi:.3f}]   {nb:>4d} {p_lower:>12.3f} {g_med:>11.3f} {c_med:>11.3f}")

    # === Approach B: regress cooperativity on p_ss (pooled), residuals ===
    print()
    print("=== Approach B: residualize cooperativity vs p_ss (pooled) ===")
    all_p = np.concatenate([p_ss_gold, p_ss_comp])
    all_coop = np.concatenate([coop_gold, coop_comp])
    # Fit a monotone-ish curve: log-linear regression coop ~ log(p_ss) since
    # coop ~ -log(P_joint) - -log(prod p) can grow as p drops.
    log_p = np.log(np.clip(all_p, 1e-6, 1.0))
    # Linear fit coop = a * log_p + b
    a, b = np.polyfit(log_p, all_coop, 1)
    print(f"  fit: cooperativity ~ {a:+.3f} * log(p_ss) + {b:+.3f}")
    resid_gold = coop_gold - (a * np.log(np.clip(p_ss_gold, 1e-6, 1.0)) + b)
    resid_comp = coop_comp - (a * np.log(np.clip(p_ss_comp, 1e-6, 1.0)) + b)
    p_lower_resid = float((resid_gold < resid_comp).mean())
    med_resid_delta = float(np.median(resid_gold - resid_comp))
    print(f"  residual P(gold < comp) = {p_lower_resid:.3f}")
    print(f"  median residual delta   = {med_resid_delta:+.3f} kcal/mol/nt")

    # Also do the same with a quadratic fit — captures more nonlinearity
    coefs = np.polyfit(log_p, all_coop, 2)
    def qpred(p): return np.polyval(coefs, np.log(np.clip(p, 1e-6, 1.0)))
    resid_gold_q = coop_gold - qpred(p_ss_gold)
    resid_comp_q = coop_comp - qpred(p_ss_comp)
    p_lower_q = float((resid_gold_q < resid_comp_q).mean())
    print(f"  residual (quadratic) P(gold < comp) = {p_lower_q:.3f}")

    print()
    print("=== Verdict ===")
    within_A_med = np.median([
        (coop_gold[(mean_p >= edges[b]) & (mean_p <= edges[b + 1])] <
         coop_comp[(mean_p >= edges[b]) & (mean_p <= edges[b + 1])]).mean()
        for b in range(N_BUCKETS)
    ])
    if within_A_med >= 0.75 and p_lower_resid >= 0.75:
        print(f"  INDEPENDENT: within-bucket median {within_A_med:.2f} and "
              f"residual {p_lower_resid:.2f} both >= 0.75. Cooperativity is not "
              f"a monotone restatement of p_ss.")
    elif within_A_med <= 0.55 and p_lower_resid <= 0.55:
        print(f"  RESTATEMENT: within-bucket median {within_A_med:.2f} and "
              f"residual {p_lower_resid:.2f} both near 0.5. Cooperativity is "
              f"largely a nonlinear restatement of p_ss.")
    else:
        print(f"  MIXED: within-bucket median {within_A_med:.2f}, residual "
              f"{p_lower_resid:.2f}. Cooperativity carries some independent "
              f"signal but is partly explained by p_ss.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
