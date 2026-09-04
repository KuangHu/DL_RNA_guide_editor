"""Tensor layer v2 — feature-vector emitter (no patch tensor).

Follows `docs/tensor_layer_spec.md` after Items 3 + 3b measured Δ AUROC =
+0.030 / +0.031 (structure means vs scalars-only) on both the easy and
Channel-B-target slices of V5 L=11. The 22-channel × 64-nt patch is NOT
justified by that evidence, so this first cut emits only a feature
vector per candidate. The multi-scale patch geometry stays in the spec
as a future upgrade path, gated on Channel B demonstrating that patches
beat means.

For each candidate the tensor layer emits:

  scalars (16 features):
    orient_fwd, orient_rc                                              (2)
    matches, mismatches, identity = m/L                                (3)
    L_percentile_in_pool, L_delta_from_median                          (2)
    flank_start_norm, flank_end_norm                                   (2)
    boundary_dist_up, boundary_dist_dn                                 (2)
    target_side_up                                                     (1)
    nc_len_norm                                                        (1)
    mol_total_ss_frac                                                  (1)
    mol_helix_count                                                    (1)
    guide_window_accessibility_percentile                              (1)

  structure means (8 features — 4 channel means + 4 valid-fraction masks):
    dG_open_uL_pn_mean, dG_open_uL_pn_valid                            (2)
    H_pair_win_mean, H_pair_win_valid                                  (2)
    cooperativity_win_pn_mean, cooperativity_win_pn_valid              (2)
    E_span_win_mean, E_span_win_valid                                  (2)

Absolute-coordinate features REMOVED (from old `preprocess/candidates.py`):
  - `L` as absolute integer — family label; replaced by pool-relative
    L_percentile_in_pool + L_delta_from_median
  - `nc_start_norm` — T-WT's gold at nc=49/49 was a perfect learnable
    shortcut; dropped entirely, no replacement

No absolute NC-frame coordinates are emitted. The L-leak via
candidate-window shape is a data-vs-featurization concern the tensor
layer cannot solve; the generator's uniform L distribution
(`difficulty.py::sample_difficulty`) is the real defense there.

Consumers should tag their MetricConditions with
`emission_mode = EMISSION_MODE_CANDIDATE_LIST_ANCHOR` (or `_SPAN` if
they query by span coverage).
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

import numpy as np

from .candidates_v2 import CandidateV2


FEATURE_NAMES: list[str] = [
    "orient_fwd",
    "orient_rc",
    "matches",
    "mismatches",
    "identity",
    "L_percentile_in_pool",
    "L_delta_from_median",
    "flank_start_norm",
    "flank_end_norm",
    "boundary_dist_up",
    "boundary_dist_dn",
    "target_side_up",
    "nc_len_norm",
    "mol_total_ss_frac",
    "mol_helix_count",
    "guide_window_accessibility_percentile",
    "dG_open_uL_pn_mean",
    "dG_open_uL_pn_valid",
    "H_pair_win_mean",
    "H_pair_win_valid",
    "cooperativity_win_pn_mean",
    "cooperativity_win_pn_valid",
    "E_span_win_mean",
    "E_span_win_valid",
]
NUM_FEATURES = len(FEATURE_NAMES)


STRUCT_CHANNEL_KEYS = (
    "dG_open_uL_pn",
    "H_pair_win",
    "cooperativity_win_pn",
    "E_span_win",
)


NC_MAX_DEFAULT = 350


@dataclass(frozen=True)
class MolSummary:
    """Position-invariant per-molecule scalars, computed once per NC and
    broadcast to every candidate emitted from that NC. Position dependence
    within the molecule lives in the per-candidate features."""
    total_ss_frac: float           # fraction of nc positions with mean unpaired prob >= 0.5
    helix_count: int               # number of contiguous helix runs in the MFE dot-bracket
    accessibility_dist: np.ndarray  # per-position 1 - dG_open_uL_pn (float, len=nc_len)


def summarize_molecule(nc_channels: dict, structure: str | None = None) -> MolSummary:
    """Extract molecule-level summary. `nc_channels` is the per-NC struct
    blob from V5 JSONL (`labels.nc_channels[active_noncoding_index]`).
    `structure` is the optional dot-bracket string; if omitted, helix_count
    is 0 (a downstream consumer that wants it must fold the NC themselves
    and pass the dot-bracket).
    """
    dG = np.array(nc_channels.get("dG_open_uL_pn", []), dtype=np.float64)
    dg_max = dG.max() if dG.size else 1.0
    accessibility = 1.0 - dG / max(dg_max, 1e-9) if dG.size else np.zeros(0)
    total_ss_frac = float((accessibility >= 0.5).mean()) if accessibility.size else 0.0
    helix_count = 0
    if structure:
        # A "helix" here is a run of contiguous '(' or ')' characters.
        prev = None
        for ch in structure:
            if ch in "()" and ch != prev:
                helix_count += 1
            prev = ch
    return MolSummary(total_ss_frac=total_ss_frac, helix_count=helix_count,
                        accessibility_dist=accessibility)


def _window_mean_and_valid(arr: list | np.ndarray, start: int,
                             length: int) -> tuple[float, float]:
    """Mean over `arr[start:start+length]`, ignoring NaN/None; returns
    (mean, valid_fraction). Empty or all-NaN returns (0.0, 0.0)."""
    if arr is None:
        return 0.0, 0.0
    if isinstance(arr, np.ndarray):
        seg = arr[start:start + length]
        if seg.size == 0:
            return 0.0, 0.0
        valid = ~np.isnan(seg.astype(np.float64))
        if not valid.any():
            return 0.0, 0.0
        return float(seg[valid].astype(np.float64).mean()), float(valid.mean())
    seg = arr[start:start + length]
    if not seg:
        return 0.0, 0.0
    kept = [float(v) for v in seg
             if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not kept:
        return 0.0, 0.0
    return float(np.mean(kept)), len(kept) / len(seg)


def build_feature_matrix(
    cands: list[CandidateV2],
    flank_len: int,
    nc_len: int,
    nc_channels: dict,
    mol_summary: MolSummary | None = None,
    L_pool: tuple[int, ...] | None = None,
    nc_max: int = NC_MAX_DEFAULT,
) -> np.ndarray:
    """Emit the (n_cands, NUM_FEATURES) feature matrix.

    Arguments:
      cands:        the deduplicated candidate list from
                    `candidates_v2.enumerate_candidates_v2`.
      flank_len:    length of the flank sequence (for boundary norm).
      nc_len:       length of the NC (for nc_len_norm).
      nc_channels:  per-NC structure blob (V5 schema
                    `labels.nc_channels[active_noncoding_index]`).
      mol_summary:  precomputed molecule-level scalars. If None, computed
                    fresh from nc_channels.
      L_pool:       the L values seen in the pool for this record. If
                    None, taken from `cands`. Pool-relative L features
                    are computed against this distribution.
      nc_max:       normalization ceiling for nc_len_norm.

    Consumer of this matrix should tag their MetricConditions with
    `emission_mode = "candidate_list_anchor"` (or `_span`) — the strings
    are canonical constants in `candidates_v2`.
    """
    if mol_summary is None:
        mol_summary = summarize_molecule(nc_channels)
    if L_pool is None:
        L_pool = tuple(c.L for c in cands)
    if not L_pool:
        L_median = 11
        L_sorted = ()
    else:
        L_median = int(statistics.median(L_pool))
        L_sorted = tuple(sorted(L_pool))

    n = len(cands)
    out = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    if n == 0:
        return out

    accessibility = mol_summary.accessibility_dist

    for i, c in enumerate(cands):
        L = c.L
        # L pool-relative
        if L_sorted:
            L_pct = float(np.searchsorted(L_sorted, L, side="right") / len(L_sorted))
        else:
            L_pct = 0.0
        L_delta = L - L_median

        # Guide-window accessibility percentile
        if accessibility.size >= c.nc_start + L:
            win = accessibility[c.nc_start : c.nc_start + L]
            if win.size:
                win_mean = float(win.mean())
                acc_pct = float((accessibility < win_mean).mean())
            else:
                acc_pct = 0.0
        else:
            acc_pct = 0.0

        # Structure means
        struct_feats = []
        for ch in STRUCT_CHANNEL_KEYS:
            mean_v, valid_v = _window_mean_and_valid(nc_channels.get(ch, []),
                                                       c.nc_start, L)
            struct_feats.extend([mean_v, valid_v])

        flank_start = float(c.flank_start)
        flank_end = flank_start + L
        flank_center = flank_start + L / 2.0

        row = [
            1.0 if c.orient == "fwd" else 0.0,       # orient_fwd
            1.0 if c.orient == "rc" else 0.0,        # orient_rc
            float(c.matches),                         # matches
            float(L - c.matches),                     # mismatches
            c.matches / float(L),                     # identity
            L_pct,                                    # L_percentile_in_pool
            float(L_delta),                           # L_delta_from_median
            flank_start / float(flank_len),           # flank_start_norm
            flank_end / float(flank_len),             # flank_end_norm
            flank_start / float(flank_len),           # boundary_dist_up  (== flank_start_norm)
            (flank_len - flank_end) / float(flank_len),  # boundary_dist_dn
            1.0 if flank_center < flank_len / 2.0 else 0.0,  # target_side_up
            nc_len / float(max(1, nc_max)),           # nc_len_norm
            mol_summary.total_ss_frac,                # mol_total_ss_frac
            float(mol_summary.helix_count),           # mol_helix_count
            acc_pct,                                  # guide_window_accessibility_percentile
            *struct_feats,                            # 8 struct features
        ]
        assert len(row) == NUM_FEATURES, (len(row), NUM_FEATURES)
        out[i] = row

    return out
