"""V5A canonical eval module.

Single source of truth for ranking evaluation across V5A. Every downstream
script imports from here — no per-script metric recomputation.

Conventions locked (2026-08-30):
  - Tie-break: expected R@k + expected MRR under uniform random tie-break
    within tied groups.
        rank_avg  = n_gt + 1 + n_eq/2
        R@k       = 0 if n_gt >= k else min(1, (k - n_gt) / (n_eq + 1))
        MRR       = mean_{i=0..n_eq}(1 / (n_gt + 1 + i))
  - "in pool" definition: tolerant match (same orient + overlap_frac ≥ 0.5 on
    both nc and flank spans) → highest-matches candidate in that neighborhood.
  - Denominator for Durrant / val pooled metrics: sites where gold_slot >= 0.
    Any pipeline that uses a different denominator must say so and NOT compare
    numbers with pipelines that use this one.
  - Bootstrap unit for real-data CIs: transposase_id (never bag).
"""
from __future__ import annotations

import numpy as np


def rank_stats(qs: np.ndarray, cs_local: int, valid_mask: np.ndarray | None = None,
                 k_list=(1, 4, 8)):
    """Return (rank_avg, R@k dict, MRR) with uniform-tie-break expectations.

    If `valid_mask` is provided, only masked-in candidates participate in the
    rank; cs_local must be a valid index BEFORE masking.
    """
    if valid_mask is not None:
        valid_idx = np.where(valid_mask)[0]
        cs_pos = int(np.where(valid_idx == cs_local)[0][0])
        qs = qs[valid_mask]
        cs_local = cs_pos
    q_cs = qs[cs_local]
    other = np.delete(qs, cs_local)
    n_gt = int((other > q_cs).sum())
    n_eq = int((other == q_cs).sum())
    tie = n_eq + 1
    rank_avg = n_gt + 1 + n_eq / 2.0
    R = {k: (0.0 if n_gt >= k else min(1.0, (k - n_gt) / tie)) for k in k_list}
    E_recip = float(np.mean(1.0 / (n_gt + 1 + np.arange(tie, dtype=np.float64))))
    return rank_avg, R, E_recip


def bootstrap_delta_clustered(cluster_ids, a, b, n_boot=5000, seed=0, weights=None):
    """Paired bootstrap on Δ mean(a - b), clustered by cluster_ids.

    Resamples clusters with replacement; within each cluster include all rows.
    `weights` optional per-row weights for weighted-mean acceptance.
    Returns (2.5-pct, 97.5-pct) of Δ.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    uniq = np.unique(cluster_ids)
    idx_by = {c: np.where(cluster_ids == c)[0] for c in uniq}
    if weights is None: weights = np.ones_like(a)
    else: weights = np.asarray(weights, dtype=np.float64)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        picks = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by[c] for c in picks])
        w = weights[rows]
        num = ((a[rows] - b[rows]) * w).sum()
        den = w.sum() or 1.0
        deltas[i] = num / den
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def find_gold_slot(feats, mask, cands, orient: str, L: int,
                    nc_start: int, flank_start: int, overlap_frac: float = 0.5):
    """Canonical tolerant match: same orient, overlap_frac >= 0.5 on both spans.
    Returns (slot_index_or_-1, gold_matches_at_that_slot)."""
    valid = np.where(mask)[0]
    if len(valid) == 0: return -1, 0.0
    matches = feats[:, 3]
    best_slot = -1; best_m = -1.0
    for i in valid:
        c = cands[int(i)]
        if c.orient != orient: continue
        mn = min(c.L, L)
        nc_ov = overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
        f_ov  = overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + L)
        th = overlap_frac * mn
        if nc_ov < th or f_ov < th: continue
        if matches[i] > best_m: best_m = float(matches[i]); best_slot = int(i)
    return best_slot, best_m


def classify_decoy(c, orient, L, nc_start, flank_start, overlap_frac=0.5):
    """Canonical decoy taxonomy relative to gold coords."""
    if c.orient != orient: return "wrong_orientation"
    mn = min(c.L, L)
    nc_ov = overlap(c.nc_start, c.nc_start + c.L, nc_start, nc_start + L)
    f_ov  = overlap(c.flank_start, c.flank_start + c.L, flank_start, flank_start + L)
    th = overlap_frac * mn
    if nc_ov < th: return "different_region"
    dL = c.L - L
    if dL > 0: return "same_region_longer_L"
    if dL < 0: return "same_region_shorter_L"
    if f_ov < th: return "same_region_same_L_wrong_flank"
    return "near_gold"


def score_length_pen(m: float, L: int, alpha: float, L0: int) -> float:
    return m - alpha * max(0.0, L - L0)


DECOY_BUCKETS = ("wrong_orientation", "different_region",
                   "same_region_longer_L", "same_region_shorter_L",
                   "same_region_same_L_wrong_flank", "near_gold")
