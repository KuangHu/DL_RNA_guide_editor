"""D5b — argmax-only discriminator, no intensity information.

Statistic:
  Per site: argmax nc position of per-position m_max (over both orients)
  Score: max_p (count of sites with argmax in [p-5, p+5])
         --- i.e. cross-site position concentration in a 11-wide window

Null:
  Per site: uniform random position on [0, n_positions)
  Same statistic on the sampled positions

z:
  (obs_stat - null_mean) / null_std

This strips out per-site intensity, addressing the T-WT saturation
diagnosed at D5 first draft:
  T-WT p_tight = 0.95 -> sum-based null was ~equal to obs -> z ≈ 0
  1_7bp p_tight = 0.80 -> sum-based null lower -> z high

Under argmax-only, the null is P(5 uniform positions concentrate within
11 nt window) = small. Under guided signal, obs concentrate at gold =
large. Insensitive to per-site m intensity.

Test: does the ordering under this statistic finally match the anchor
detection pattern? Specifically:
  - T-WT (34 bags): expected high z if coherence is real
  - RTG variants: expected varied z based on actual position concentration
  - Anchor-detected bags: expected significantly higher z than anchor-failed
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from scripts.v5a_framework.match_table import load as load_mt
from scripts.v5a_framework.variant import spec_m_threshold_L11, run_variant


MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
L_DET = 11
WINDOW = 5      # ±5 nt window for concentration
SEED = 0
N_PERM = 5000


def per_site_m_max(nc: str, flank: str, L: int) -> np.ndarray:
    fwd_dot, rc_dot = dot_plot(nc, flank)
    w_fwd = windowed_matches(fwd_dot, L)
    w_rc = windowed_matches(rc_dot, L)
    if w_fwd.size == 0 or w_rc.size == 0:
        return np.zeros(0, dtype=np.int32)
    m_fwd = w_fwd.max(axis=1)
    m_rc = w_rc.max(axis=1)
    n = min(len(m_fwd), len(m_rc))
    return np.maximum(m_fwd[:n], m_rc[:n])


def max_concentration(positions: list[int], n_pos: int, w: int = WINDOW) -> int:
    """max_p (count of positions in [p-w, p+w])"""
    if not positions:
        return 0
    positions_arr = np.array(positions)
    best = 0
    for p in range(n_pos):
        c = int(((positions_arr >= p - w) & (positions_arr <= p + w)).sum())
        if c > best:
            best = c
    return best


def argmax_z_for_bag(m_arrays: list[np.ndarray], n_perm: int = N_PERM,
                       seed: int = SEED) -> dict:
    n = min(len(m) for m in m_arrays)
    obs_positions = [int(np.argmax(m[:n])) for m in m_arrays]
    obs_stat = max_concentration(obs_positions, n_pos=n)

    rng = np.random.default_rng(seed)
    null_stats = np.zeros(n_perm, dtype=np.int32)
    for k in range(n_perm):
        perm_positions = list(rng.integers(0, n, size=len(m_arrays)).tolist())
        null_stats[k] = max_concentration(perm_positions, n_pos=n)
    mu = null_stats.mean(); sd = null_stats.std() + 1e-12
    z = (obs_stat - mu) / sd
    p_val = float((null_stats >= obs_stat).mean())
    # Also: obs positions and how many in gold vicinity
    gold_hits = sum(1 for p in obs_positions if 44 <= p <= 54)
    return {
        "obs_stat": int(obs_stat),
        "obs_positions": obs_positions,
        "gold_hits_45_54": gold_hits,
        "null_mean": mu,
        "null_std": sd,
        "z": float(z),
        "p_val": p_val,
    }


def main() -> int:
    print("[D5b] loading Durrant MatchTable")
    mt = load_mt(MT_POS)
    by_variant = defaultdict(list)
    for tnp_id in mt.tnp_ids:
        prefix = tnp_id.split("_paired_")[0]
        by_variant[prefix].append(tnp_id)

    # Compute z per bag
    import time; t0 = time.time()
    scores = {}
    for tnp_id in mt.tnp_ids:
        nc = mt.tnps[tnp_id].nc
        m_arrays = [per_site_m_max(nc, s.flank, L_DET) for s in mt.tnps[tnp_id].sites]
        r = argmax_z_for_bag(m_arrays, n_perm=N_PERM,
                              seed=hash((SEED, tnp_id)) & 0xFFFFFFFF)
        scores[tnp_id] = r
    print(f"[D5b] wall {time.time()-t0:.1f}s")

    # per-variant summary
    print()
    print(f"{'variant':<45s} {'n':>3} {'obs_med':>8} {'z_med':>7} {'z_min':>7} {'z_max':>7} {'p_med':>7} {'gold_hits_median':>17}")
    for variant, tnp_ids in by_variant.items():
        obss = np.array([scores[t]["obs_stat"] for t in tnp_ids])
        zs = np.array([scores[t]["z"] for t in tnp_ids])
        ps = np.array([scores[t]["p_val"] for t in tnp_ids])
        golds = np.array([scores[t]["gold_hits_45_54"] for t in tnp_ids])
        print(f"  {variant:<45s} {len(tnp_ids):>3d} {float(np.median(obss)):>8.1f} "
              f"{float(np.median(zs)):>7.2f} {zs.min():>7.2f} {zs.max():>7.2f} "
              f"{float(np.median(ps)):>7.4f} {float(np.median(golds)):>17.1f}")

    # Detected vs failed
    peaks = run_variant(mt, spec_m_threshold_L11(m=8, tau=0, S=5))
    detected = {t for t, p in peaks.items() if p}
    z_det = np.array([scores[t]["z"] for t in mt.tnp_ids if t in detected])
    z_fail = np.array([scores[t]["z"] for t in mt.tnp_ids if t not in detected])
    obs_det = np.array([scores[t]["obs_stat"] for t in mt.tnp_ids if t in detected])
    obs_fail = np.array([scores[t]["obs_stat"] for t in mt.tnp_ids if t not in detected])
    print()
    print(f"Anchor-detected (22): z mean={z_det.mean():.2f}, med={float(np.median(z_det)):.2f}, min={z_det.min():.2f}, obs_stat mean={obs_det.mean():.2f}, med={float(np.median(obs_det)):.1f}")
    print(f"Anchor-failed   (43): z mean={z_fail.mean():.2f}, med={float(np.median(z_fail)):.2f}, min={z_fail.min():.2f}, max={z_fail.max():.2f}, obs_stat mean={obs_fail.mean():.2f}, med={float(np.median(obs_fail)):.1f}")

    # Are the 22 detected significantly separated?
    # Simple within-Durrant AUROC (det vs failed): a real test since it's
    # positive vs "positive with weaker signal" — measures whether the
    # discriminator ranks better than the anchor threshold.
    from itertools import combinations
    n_ranked_correctly = 0
    n_pairs = 0
    for i in z_det:
        for j in z_fail:
            n_pairs += 1
            if i > j: n_ranked_correctly += 1
            elif i == j: n_ranked_correctly += 0.5
    auc_internal = n_ranked_correctly / n_pairs
    print(f"\nInternal AUROC (detected vs failed within Durrant): {auc_internal:.4f}")
    print(f"  If argmax z discriminates within Durrant, this beats 0.5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
