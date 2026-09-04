"""ncrna_sampler_v2 — BPP-based soft guide placement.

Replaces the strict-loop pick_guide_position in ncrna_sampler.py
(falsified 2026-08-31: T-WT's gold is at nc pos 49, not inside the
single MFE loop at 98-109; 4/6 real ncRNAs have zero >=11 nt loops).

New rule: sample guide_start with weight proportional to a Gaussian
around a target accessibility percentile in the BPP-derived
`p_ss_window_percentile` distribution.

Anchor calibration (n=2, biological ground truth):
  T-WT gold[49]        p_ss window %ile = 83.2%
  ISEc21 mature target p_ss window %ile = 85.2%
  → μ = 0.85

σ is set wider than the anchor spread (2 %ile) on purpose
(asymmetric-loss argument, user directive 2026-08-31):
  σ too narrow → generator places every guide at exactly 85%ile, a
                 pattern real ncRNAs do NOT satisfy (D5 shows top-10
                 windows on T-WT are at 94-102, and gold is at 49) →
                 model would learn a false strong prior.
  σ too wide  → guide occasionally lands in low-accessibility spots
                 → harder samples, no false pattern taught.
  Cost of narrow >> cost of wide, so default σ = 0.15.

Callers can override μ, σ, or supply a hard "must be above" floor via
`min_percentile` if a particular difficulty axis wants tighter placement.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

import numpy as np

from preprocess.features_structure_v2 import (
    StructureFeaturesV2, compute_features_v2,
)


DEFAULT_MU = 0.85
DEFAULT_SIGMA = 0.15
DEFAULT_GUIDE_LENGTH = 11


@dataclass(frozen=True)
class SampledPlacement:
    """A guide placement drawn from the soft distribution."""
    start: int
    percentile: float           # empirical rank of the chosen window
    p_ss_window_mean: float     # mean(P_ss) over the placed window
    weight: float               # Gaussian weight at the chosen percentile


def sample_guide_placement(
    feats: StructureFeaturesV2,
    rng: random.Random | np.random.Generator | None = None,
    mu: float = DEFAULT_MU,
    sigma: float = DEFAULT_SIGMA,
    min_percentile: float = 0.0,
) -> SampledPlacement | None:
    """Sample a guide start position with soft accessibility preference.

    Weight over window-start s:
        percentile(s) = rank of mean(P_ss over window s) / n_windows
        weight(s) = exp( -0.5 * ((percentile(s) - mu) / sigma)^2 )
        weight(s) *= 0 if percentile(s) < min_percentile

    Returns None if no window has non-zero weight (min_percentile too
    tight, or all windows have zero P_ss which cannot happen in practice).
    """
    if rng is None:
        rng = random.Random()

    L = feats.guide_length
    p_ss = feats.p_ss
    n_win = len(p_ss) - L + 1
    if n_win <= 0:
        return None

    csum = np.concatenate(([0.0], np.cumsum(p_ss, dtype=np.float64)))
    p_ss_window_mean = (csum[L:] - csum[:-L]) / L

    # Empirical percentile = fraction of windows strictly below this one.
    order = np.argsort(p_ss_window_mean, kind="stable")
    ranks = np.empty(n_win, dtype=np.float64)
    ranks[order] = np.arange(n_win)
    percentile = ranks / max(n_win - 1, 1)

    weight = np.exp(-0.5 * ((percentile - mu) / sigma) ** 2)
    if min_percentile > 0.0:
        weight = weight * (percentile >= min_percentile)
    total = weight.sum()
    if total <= 0.0:
        return None
    prob = weight / total

    # Draw a start index according to prob
    if isinstance(rng, random.Random):
        u = rng.random()
    else:
        u = float(rng.random())
    cum = np.cumsum(prob)
    start = int(np.searchsorted(cum, u))
    if start >= n_win:
        start = n_win - 1

    return SampledPlacement(
        start=start,
        percentile=float(percentile[start]),
        p_ss_window_mean=float(p_ss_window_mean[start]),
        weight=float(weight[start]),
    )


MATCH_CHANNELS_ALL: tuple[str, ...] = (
    "p_ss", "cooperativity_win_pn", "H_pair_win",
    "dG_open_uL_pn", "E_span_win",
)

# Stage 1d default match set. Item 4.5 measured per-channel P(gold better):
#   cooperativity 0.909  <-- most discriminative
#   dG_open_uL   0.853
#   H_pair       0.834
#   p_ss         0.781
#   E_span       0.630
# Matching only p_ss (0.781) leaves the strongest channel (cooperativity)
# free — the twin's provable-0.5 claim degrades to empirical, and A12 is
# likely to fail. Default match set uses the top-2 (cooperativity + p_ss)
# so twin sites are matched on the most discriminative channel by
# construction; A12 remains meaningful for the remaining 3 free channels.
MATCH_CHANNELS_DEFAULT: tuple[str, ...] = (
    "p_ss", "cooperativity_win_pn",
)


def _channel_percentiles(feats: StructureFeaturesV2, L: int,
                            channel: str) -> np.ndarray:
    """Return per-window-start empirical rank/(n_win-1) for the given
    channel. `p_ss` uses the cumsum window mean; others use direct
    per-window arrays where available."""
    if channel == "p_ss":
        p_ss = feats.p_ss
        n_win = len(p_ss) - L + 1
        if n_win <= 0:
            return np.zeros(0)
        csum = np.concatenate(([0.0], np.cumsum(p_ss, dtype=np.float64)))
        vals = (csum[L:] - csum[:-L]) / L
    else:
        arr = getattr(feats, channel, None)
        if arr is None:
            return np.zeros(0)
        vals = np.asarray(arr, dtype=np.float64)
        # Windowed channels already have shape (n_win,); non-windowed
        # (per-position) fall back to a sliding mean.
        exp_len = len(feats.p_ss) - L + 1
        if vals.shape[0] != exp_len:
            csum = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
            if csum.size < L + 1:
                return np.zeros(0)
            vals = (csum[L:] - csum[:-L]) / L
        vals = np.where(np.isnan(vals), np.nanmedian(vals), vals)
    n_win = vals.size
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(n_win, dtype=np.float64)
    ranks[order] = np.arange(n_win)
    return ranks / max(n_win - 1, 1)


def sample_percentile_matched_positions(
    feats: StructureFeaturesV2,
    target_percentile: float,
    n_distinct: int,
    exclude: set[int] | None = None,
    tol: float = 0.05,
    rng: random.Random | np.random.Generator | None = None,
    match_channels: tuple[str, ...] = ("p_ss",),
    per_channel_targets: dict[str, float] | None = None,
) -> list[int] | None:
    """1d twin-negative helper (added 2026-09-03).

    Return `n_distinct` DIFFERENT guide-start positions whose empirical
    percentile on EVERY channel in `match_channels` is within `tol` of
    that channel's target percentile, and that avoid every position in
    `exclude` (typically {p_true}).

    `per_channel_targets`: optional per-channel target percentiles. When
    None, `target_percentile` is used for every channel (backwards compat
    with the p_ss-only default). For a true twin negative, the caller
    passes the positive's actual per-channel percentiles here so all
    structure-derived features match.

    match_channels tuple selects which structure channels' percentile
    ranks the twin positions must match:
        ("p_ss",)                  = p_ss-only (default; loose)
        MATCH_CHANNELS_ALL         = all 5 channels (tight; may require
                                     widening tol or lowering n_distinct)

    NOTE — the "provably 0.5 structure-only AUROC" claim in the handoff
    spec is exact ONLY when the FULL structure channel vector is
    matched. Item 4.5 measured per-channel discriminative power
    (cooperativity 0.909, dG_open_uL 0.853, H_pair 0.834, p_ss 0.781,
    E_span 0.630). Matching p_ss alone leaves the other four channels
    free — the 0.5 claim degrades to empirical, tested by A12.

    Returns None if the intersection has fewer than `n_distinct`
    candidates. Caller must widen `tol`, drop a channel, or accept
    fewer twins.
    """
    if rng is None:
        rng = random.Random()

    L = feats.guide_length
    n_win = len(feats.p_ss) - L + 1
    if n_win <= 0:
        return None

    targets = dict(per_channel_targets or {})
    for ch in match_channels:
        targets.setdefault(ch, target_percentile)

    mask = np.ones(n_win, dtype=bool)
    for ch in match_channels:
        pct = _channel_percentiles(feats, L, ch)
        if pct.size != n_win:
            continue        # channel unavailable — skip silently
        t = targets[ch]
        mask &= (pct >= t - tol) & (pct <= t + tol)
    if exclude:
        for p in exclude:
            if 0 <= p < n_win:
                mask[p] = False
    candidates = np.where(mask)[0]
    if candidates.size < n_distinct:
        return None

    if isinstance(rng, random.Random):
        pool = list(int(c) for c in candidates)
        rng.shuffle(pool)
        return sorted(pool[:n_distinct])
    else:
        chosen = rng.choice(candidates, size=n_distinct, replace=False)
        return sorted(int(c) for c in chosen)


def sample_random_ncrna_and_placement(
    nc_length: int,
    guide_length: int,
    rng: random.Random | None = None,
    mu: float = DEFAULT_MU,
    sigma: float = DEFAULT_SIGMA,
    base_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25),
) -> tuple[str, StructureFeaturesV2, SampledPlacement] | None:
    """Draw one nc (uniform ACGT) at target length, fold + BPP, then
    sample a guide placement. Returns None only if the resulting fold
    is degenerate; retry policy is caller's problem."""
    if rng is None:
        rng = random.Random()
    seq = "".join(rng.choices("ACGT", weights=list(base_probs), k=nc_length))
    feats = compute_features_v2(seq, guide_length=guide_length)
    placement = sample_guide_placement(feats, rng=rng, mu=mu, sigma=sigma)
    if placement is None:
        return None
    return seq, feats, placement
