"""Item 4.5 — Durrant gold vs top-competitor BPP channel separation.

Answers: do the new features_structure_v2 channels (dG_open_uL_pn,
E_span_win, H_pair_win) separate Durrant gold positions from their
top competitor positions?

48CS-A showed structure separates POS from *designed* WS. The Durrant
top-competitor population is NOT designed — it's random windows that
happen to hit high m. If channels don't separate this population, the
4 h cache rebuild investment shrinks or changes shape.

Method:
  For each Durrant Tnp t:
    - fold nc once with features_structure_v2 (guide_length = median gold_L
      across Tnp's sites; fall back to 11 if unavailable)
    - for each site s of t:
        gold_pos = s.gold_nc
        top_comp_pos = argmax_{p != gold_pos, p in valid_window_range}
                       m_max_pooled(fwd, rc) at L=gold_L for the site's flank
        record (gold_pos, top_comp_pos, feats at both)

  Aggregate paired deltas:
    for each channel c and each (gold, comp) pair:
      d_c = c(gold) - c(comp)
    Report P(d_c > 0), median, MAD, per Durrant + Tnp-clustered CI.

Threshold: any channel with |P(d>0) - 0.5| >= 0.15 (i.e. P >= 0.65 or
<= 0.35) is materially discriminative.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from preprocess.features_structure_v2 import (
    CHANNEL_GOLD_SIGN, compute_features_v2, p_gold_better,
)
from scripts.v5a_framework.match_table import load as load_mt

MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
MAX_TNPS = None       # None = all
MAX_SITES_PER_TNP = 8


def top_competitor(m_arr: np.ndarray, gold_pos: int) -> int | None:
    """Return the nc position with highest m_max other than gold_pos.
    Ties broken by first-index. Returns None if m_arr empty."""
    n = len(m_arr)
    if n <= 1:
        return None
    masked = m_arr.copy().astype(np.int32)
    if 0 <= gold_pos < n:
        masked[gold_pos] = -1
    return int(np.argmax(masked))


def main() -> int:
    mt = load_mt(MT_POS)
    tnp_ids = mt.tnp_ids
    if MAX_TNPS is not None:
        tnp_ids = tnp_ids[:MAX_TNPS]
    print(f"[4.5] {len(tnp_ids)} Durrant Tnps")

    # Aggregate records
    channels = ["dG_open_uL_pn", "E_span_win", "H_pair_win", "p_ss_window"]
    gold_vals: dict[str, list[float]] = {c: [] for c in channels}
    comp_vals: dict[str, list[float]] = {c: [] for c in channels}
    deltas: dict[str, list[float]] = {c: [] for c in channels}
    tnp_of_record: list[str] = []
    n_records = 0

    t0 = time.perf_counter()
    for ti, tnp_id in enumerate(tnp_ids):
        rec = mt.tnps[tnp_id]
        nc = rec.nc.upper().replace("U", "T")
        # Determine guide L from the sites' gold_L
        gold_Ls = [s.gold_L for s in rec.sites if s.gold_L]
        if not gold_Ls:
            continue
        # Use median L for the whole Tnp's fold, per-site pull the correct L
        gL_median = int(np.median(gold_Ls))
        # We fold once at L_median and read the L_median-window channels; per-site
        # gold_L may differ ±1-2 nt but the geometry channels change slowly.

        feats = compute_features_v2(nc, guide_length=gL_median)
        p_ss = feats.p_ss
        csum = np.concatenate(([0.0], np.cumsum(p_ss, dtype=np.float64)))
        p_ss_win = (csum[gL_median:] - csum[:-gL_median]) / gL_median

        n_win = len(feats.dG_open_uL_pn)

        used_sites = 0
        for s in rec.sites:
            if used_sites >= MAX_SITES_PER_TNP:
                break
            if s.gold_nc is None or s.gold_L is None:
                continue
            gold_pos = s.gold_nc
            if not (0 <= gold_pos < n_win):
                continue
            # Get this site's per-position m_max at L = gL_median, both orients
            try:
                m_fwd = mt.m_max(tnp_id, s.site_idx, "fwd", gL_median)
                m_rc = mt.m_max(tnp_id, s.site_idx, "rc", gL_median)
            except KeyError:
                continue
            n_common = min(len(m_fwd), len(m_rc), n_win)
            m_pooled = np.maximum(m_fwd[:n_common], m_rc[:n_common])
            comp_pos = top_competitor(m_pooled, gold_pos)
            if comp_pos is None or not (0 <= comp_pos < n_win):
                continue

            # Skip identical positions (would happen if gold_pos out of range)
            if comp_pos == gold_pos:
                continue

            def pull(idx: int) -> dict[str, float]:
                return {
                    "dG_open_uL_pn": float(feats.dG_open_uL_pn[idx]),
                    "E_span_win":    float(feats.E_span_win[idx]),
                    "H_pair_win":    float(feats.H_pair_win[idx]),
                    "p_ss_window":   float(p_ss_win[idx]),
                }
            gv = pull(gold_pos)
            cv = pull(comp_pos)
            for c in channels:
                if np.isnan(gv[c]) or np.isnan(cv[c]):
                    continue
                gold_vals[c].append(gv[c])
                comp_vals[c].append(cv[c])
                deltas[c].append(gv[c] - cv[c])
            tnp_of_record.append(tnp_id)
            used_sites += 1
            n_records += 1
        mt.evict(tnp_id)

        if (ti + 1) % 20 == 0:
            dt = time.perf_counter() - t0
            print(f"  [{ti+1}/{len(tnp_ids)}] {n_records} records, {dt:.1f}s")

    dt = time.perf_counter() - t0
    print(f"\n[4.5] done: {n_records} gold-vs-competitor pairs from "
          f"{len(tnp_ids)} Tnps in {dt:.1f}s")

    def summarize(c: str):
        d = np.array(deltas[c], dtype=np.float64)
        g = np.array(gold_vals[c], dtype=np.float64)
        v = np.array(comp_vals[c], dtype=np.float64)
        p_gt_0 = float((d > 0).mean()) if len(d) else float("nan")
        med_d = float(np.median(d)) if len(d) else float("nan")
        mad = float(np.median(np.abs(d - med_d))) if len(d) else float("nan")
        g_med = float(np.median(g)) if len(g) else float("nan")
        v_med = float(np.median(v)) if len(v) else float("nan")
        return {"n": len(d), "P(delta>0)": p_gt_0, "median_delta": med_d,
                "MAD": mad, "gold_med": g_med, "comp_med": v_med}

    def tnp_cluster_ci(c: str, B: int = 500) -> tuple[float, float]:
        """Tnp-clustered bootstrap 95% CI for P(delta>0)."""
        d = np.array(deltas[c], dtype=np.float64)
        tnp_arr = np.array(tnp_of_record)
        # Filter to same length (some channels drop records via NaN mask above)
        # Rebuild the paired arrays for this channel
        # For simplicity: reuse the deltas + tnp_of_record — assumes NaN filter
        # never dropped a record for one channel while keeping it for another.
        # (In our current setup NaN in gold or comp for E_span/H_pair would
        # skip that record's contribution to that channel's deltas, but
        # tnp_of_record still counted it. We use the safer per-channel n.)
        rng = np.random.default_rng(0)
        # Rebuild per-channel tnp list by re-indexing:
        # Instead use grouped bootstrap on unique Tnps.
        # For this diagnostic, cluster over unique Tnp IDs weighted by n:
        unique = np.unique(tnp_arr)
        boots = []
        # We need per-Tnp mean of (d>0)
        # Approximate: bootstrap by resampling Tnps
        by_tnp: dict[str, list[float]] = {t: [] for t in unique}
        # But d has the same length as records without NaN filtering — since
        # per channel it may drop records, we approximate by using ALL deltas
        # against ALL Tnp labels of same length. If lengths differ, fall back
        # to non-clustered CI.
        if len(d) != len(tnp_arr):
            # Non-clustered CI (record bootstrap)
            n = len(d)
            for _ in range(B):
                idx = rng.integers(0, n, size=n)
                boots.append(float((d[idx] > 0).mean()))
            return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        for tid, dv in zip(tnp_arr, d):
            by_tnp[tid].append(float(dv))
        # bootstrap Tnps
        tnps_arr = np.array(list(by_tnp.keys()))
        vals_by_tnp = [np.array(by_tnp[t]) for t in tnps_arr]
        for _ in range(B):
            idx = rng.integers(0, len(tnps_arr), size=len(tnps_arr))
            pooled = np.concatenate([vals_by_tnp[i] for i in idx])
            boots.append(float((pooled > 0).mean()))
        return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    print()
    print(f"  {'channel':<20s} {'n':>6s} {'sgn':>4s} {'gold_med':>10s} {'comp_med':>10s} "
          f"{'delta_med':>10s} {'P(d>0)':>8s} {'P(gold_better)':>15s} {'95% CI Tnp':>18s}")
    ordered = sorted(channels, key=lambda c: -abs(p_gold_better(summarize(c)['P(delta>0)'], c) - 0.5))
    for c in ordered:
        s = summarize(c)
        lo, hi = tnp_cluster_ci(c)
        pgb = p_gold_better(s['P(delta>0)'], c)
        sign = CHANNEL_GOLD_SIGN.get(c, 0)
        print(f"  {c:<20s} {s['n']:>6d} {sign:>+4d} {s['gold_med']:>10.3f} {s['comp_med']:>10.3f} "
              f"{s['median_delta']:>10.3f} {s['P(delta>0)']:>8.3f} {pgb:>15.3f} "
              f"[{lo:.3f}, {hi:.3f}]")

    print()
    print("  Verdict rule: P(gold_better) >= 0.65 = material discrimination")
    print("  (equivalent to |P(delta>0) - 0.5| >= 0.15 with expected direction accounted for)")

    # 0b: correlation matrix on the 4 channels, at gold and competitor positions
    from scipy.stats import spearmanr
    print()
    print("=== 0b: cross-channel Spearman r on gold + competitor positions ===")
    channels_ordered = ["dG_open_uL_pn", "E_span_win", "H_pair_win", "p_ss_window"]

    def stack_at(role: str) -> np.ndarray:
        arrs = []
        vals = gold_vals if role == "gold" else comp_vals
        for c in channels_ordered:
            arrs.append(np.array(vals[c], dtype=np.float64))
        # Common valid mask across channels
        stk = np.column_stack(arrs)
        mask = np.isfinite(stk).all(axis=1)
        return stk[mask]

    for role in ("gold", "competitor"):
        stk = stack_at(role)
        print(f"\n  {role.upper()} positions (n={len(stk)}):")
        r, _ = spearmanr(stk, axis=0)
        # r is (4, 4)
        print(f"    {'':<18s} " + " ".join(f"{c[:12]:>12s}" for c in channels_ordered))
        for i, c in enumerate(channels_ordered):
            row = " ".join(f"{r[i, j]:>+12.3f}" for j in range(4))
            print(f"    {c:<18s} {row}")

    # Report which channels are highly correlated on gold
    stk_g = stack_at("gold")
    r_g, _ = spearmanr(stk_g, axis=0)
    print()
    print("  Redundancy pairs on gold (|r| > 0.9):")
    any_red = False
    for i in range(len(channels_ordered)):
        for j in range(i + 1, len(channels_ordered)):
            if abs(r_g[i, j]) > 0.9:
                print(f"    {channels_ordered[i]}  <->  {channels_ordered[j]}   r = {r_g[i, j]:+.3f}")
                any_red = True
    if not any_red:
        print("    (none — all four channels carry distinct information)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
