"""D5 — position-concentration discriminator on Durrant + IS10-R.

Statistic per system (bag / Tnp):
  For each site, get best-hit nc position (argmax m across nc positions,
  both orients pooled per position).
  concentration = 1 / (1 + range_of_positions)   [higher = tighter cluster]

Null (within-system permutation):
  For each site: shuffle the m values across nc positions (preserving the
  site's m distribution as a multiset), take new argmax.
  Aggregate concentration under the shuffle. Repeat n_perm times.

Score:
  z = (observed - null_mean) / null_std
  p = fraction of null draws with concentration >= observed

Two outputs:
  1) Durrant 65 bags: does z separate detected variants from failed variants?
     Or, better, does every guided bag show high z regardless of coverage-
     threshold detection?
  2) IS10-R 27 Tnps: same score on the recompute. First AUROC readout of
     coherence discriminator vs non-guided.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from scripts.v5a_framework.match_table import load as load_mt


MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
IS10R = "/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/formatted/real_IS10-R_sites.jsonl"
L_DET = 11
SEED = 0
N_PERM = 500


def per_site_m_max(nc: str, flank: str, L: int) -> np.ndarray:
    """Per-position m_max at L, pooled max over both orientations."""
    fwd_dot, rc_dot = dot_plot(nc, flank)
    w_fwd = windowed_matches(fwd_dot, L)
    w_rc = windowed_matches(rc_dot, L)
    if w_fwd.size == 0 or w_rc.size == 0:
        return np.zeros(0, dtype=np.int32)
    m_fwd = w_fwd.max(axis=1)
    m_rc = w_rc.max(axis=1)
    n = min(len(m_fwd), len(m_rc))
    return np.maximum(m_fwd[:n], m_rc[:n])


def coherence_statistic(m_arrays: list[np.ndarray]) -> float:
    """max over nc positions p of sum_i m[i, p], where m[i, p] is site i's
    per-position max-match. Uses the full m array — no argmax tie-break
    fragility.
    """
    n = min(len(m) for m in m_arrays)
    stacked = np.stack([m[:n] for m in m_arrays])   # (5, n_pos)
    per_pos_sum = stacked.sum(axis=0)                # sum over sites
    return float(per_pos_sum.max())


def z_score_for_bag(m_arrays: list[np.ndarray], n_perm: int = N_PERM,
                     seed: int = SEED) -> dict:
    """Given 5 sites' m arrays, return the discriminator z on max-position-sum.

    Null: each site's m values are permuted across positions (preserves
    intensity distribution, randomizes position assignment). Coherent
    guiding creates high sum at a specific position; null spreads them out.
    """
    obs_stat = coherence_statistic(m_arrays)
    rng = np.random.default_rng(seed)
    null_stats = np.zeros(n_perm)
    for k in range(n_perm):
        shuffled = []
        for m in m_arrays:
            ms = m.copy()
            rng.shuffle(ms)
            shuffled.append(ms)
        null_stats[k] = coherence_statistic(shuffled)
    mu = null_stats.mean(); sd = null_stats.std() + 1e-12
    z = (obs_stat - mu) / sd
    p_val = float((null_stats >= obs_stat).mean())
    return {
        "obs_stat": obs_stat,
        "null_mean": mu,
        "null_std": sd,
        "z": z,
        "p_val": p_val,
    }


def _load_is10r(n_tnp=30, cap=10, seed=0):
    tnp_ins = defaultdict(dict)
    with open(IS10R) as f:
        for line in f:
            d = json.loads(line)
            m = d.get("generator_metadata", {})
            key = (m.get("insertion_start"), m.get("sample_id"))
            side = m.get("flank_side")
            fl = d.get("inputs", {}).get("flank")
            if side in ("upstream", "downstream") and fl:
                tnp_ins[d["transposase_id"]].setdefault(key, {})[side] = fl
    tnp_dn = {}
    for tnp, m in tnp_ins.items():
        both = [p["downstream"] for p in m.values() if "upstream" in p and "downstream" in p]
        if len(both) >= 5:
            tnp_dn[tnp] = both
    rng = np.random.default_rng(seed)
    picked = list(rng.choice(sorted(tnp_dn.keys()),
                                size=min(n_tnp, len(tnp_dn)), replace=False))
    # dedup + take 5 reps
    out = {}
    for tnp in picked:
        pool = tnp_dn[tnp]
        uniq = list(dict.fromkeys(pool))
        if len(uniq) >= 5:
            out[tnp] = uniq[:5]
    return out


def main() -> int:
    # ---------- Durrant 65 bags ----------
    print("[D5] loading Durrant MatchTable", flush=True)
    mt = load_mt(MT_POS)
    by_variant = defaultdict(list)
    for tnp_id in mt.tnp_ids:
        prefix = tnp_id.split("_paired_")[0]
        by_variant[prefix].append(tnp_id)

    # Compute per-bag z-score
    print(f"[D5] computing z on 65 Durrant bags with n_perm={N_PERM}...", flush=True)
    t0 = time.time()
    durrant_scores = {}
    for variant, tnp_ids in by_variant.items():
        for tnp_id in tnp_ids:
            nc = mt.tnps[tnp_id].nc
            m_arrays = [per_site_m_max(nc, s.flank, L_DET)
                          for s in mt.tnps[tnp_id].sites]
            r = z_score_for_bag(m_arrays, n_perm=N_PERM,
                                seed=hash((SEED, tnp_id)) & 0xFFFFFFFF)
            durrant_scores[tnp_id] = {**r, "variant": variant}
    print(f"[D5] durrant wall {time.time()-t0:.1f}s", flush=True)

    # Summarize per variant
    print()
    print(f"{'variant':<45s} {'n':>4} {'z_med':>7} {'z_min':>7} {'z_max':>7} {'p_med':>7}")
    for variant, tnp_ids in by_variant.items():
        zs = np.array([durrant_scores[t]["z"] for t in tnp_ids])
        ps = np.array([durrant_scores[t]["p_val"] for t in tnp_ids])
        print(f"  {variant:<45s} {len(tnp_ids):>4d} "
              f"{float(np.median(zs)):>7.2f} {zs.min():>7.2f} {zs.max():>7.2f} "
              f"{float(np.median(ps)):>7.4f}")

    # Threshold-detected (S=5) vs not: does z discriminate?
    # Load anchor detected list
    from scripts.v5a_framework.variant import spec_m_threshold_L11, run_variant
    peaks = run_variant(mt, spec_m_threshold_L11(m=8, tau=0, S=5))
    detected = {tnp_id for tnp_id, pks in peaks.items() if pks}
    z_detected = np.array([durrant_scores[t]["z"] for t in mt.tnp_ids if t in detected])
    z_failed = np.array([durrant_scores[t]["z"] for t in mt.tnp_ids if t not in detected])
    print()
    print(f"S=5 anchor-detected bags: n={len(z_detected)} z: mean={z_detected.mean():.2f}, med={float(np.median(z_detected)):.2f}, min={z_detected.min():.2f}")
    print(f"S=5 anchor-FAILED bags:   n={len(z_failed)} z: mean={z_failed.mean():.2f}, med={float(np.median(z_failed)):.2f}, min={z_failed.min():.2f}, max={z_failed.max():.2f}")

    # ---------- IS10-R 27 Tnps ----------
    print()
    print("[D5] loading IS10-R (30 Tnp / dedup keep 27)...", flush=True)
    is10r = _load_is10r(30, 10, 0)
    print(f"[D5] {len(is10r)} IS10-R Tnps qualify (5 unique flanks)", flush=True)

    # For fair comparison: each IS10-R Tnp × each Durrant nc pair is
    # a negative datapoint (27 × 65 = 1755 negatives). Durrant self bags
    # (65 positives, own nc). No max-over-ncs cherry-picking.
    print(f"[D5] computing per-(IS10-R Tnp, Durrant nc) z on 27 x 65 = 1755 pairs...", flush=True)
    t0 = time.time()
    is10r_pair_z = []
    for tnp, flanks in is10r.items():
        for nc_tnp_id in mt.tnp_ids:
            nc = mt.tnps[nc_tnp_id].nc
            m_arrays = [per_site_m_max(nc, fl, L_DET) for fl in flanks]
            r = z_score_for_bag(m_arrays, n_perm=100,
                                 seed=hash((SEED, tnp, nc_tnp_id)) & 0xFFFFFFFF)
            is10r_pair_z.append(r["z"])
    print(f"[D5] IS10-R wall {time.time()-t0:.1f}s", flush=True)
    z_neg = np.array(is10r_pair_z)
    print(f"IS10-R pairs (n=1755) z: mean={z_neg.mean():.2f} med={float(np.median(z_neg)):.2f} "
          f"p95={float(np.percentile(z_neg, 95)):.2f} max={z_neg.max():.2f}")

    z_pos = np.array([durrant_scores[t]["z"] for t in mt.tnp_ids])
    labels = np.concatenate([np.ones(len(z_pos)), np.zeros(len(z_neg))])
    scores = np.concatenate([z_pos, z_neg])
    order = np.argsort(-scores)
    n_pos = int(labels.sum()); n_neg = len(labels) - n_pos
    tp = 0; auc = 0.0
    for lbl in labels[order]:
        if lbl == 1: tp += 1
        else: auc += tp
    auc = auc / (n_pos * n_neg)
    print()
    print(f"[D5] AUROC (Durrant vs IS10-R per-pair): {auc:.4f}")
    print(f"     Durrant z:   mean={z_pos.mean():.2f}, med={float(np.median(z_pos)):.2f}")
    print(f"     IS10-R z:    mean={z_neg.mean():.2f}, med={float(np.median(z_neg)):.2f}")

    # Also report distribution: percentile of Durrant z among IS10-R pairs
    for pct in [50, 75, 90, 95, 99]:
        threshold = np.percentile(z_neg, pct)
        n_pos_above = int((z_pos >= threshold).sum())
        print(f"     IS10-R p{pct} = {threshold:.2f}   Durrant bags above = {n_pos_above}/65 = {n_pos_above/65:.2%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
