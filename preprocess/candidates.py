"""Alignment-aware structure-patch candidate generator.

Given one (non-coding region, flank) pair and the NC's precomputed RNAplfold
per-nt unpaired-stretch profile, enumerate the top-K plausible short
ungapped alignments (per orientation × guide length L) and, for each
candidate, emit:

  1. A guide-centered structure PATCH of fixed width W (default 64) whose
     channels carry (a) RNAplfold accessibility around the guide and
     (b) alignment-specific overlays that tell the model exactly where the
     guide sits inside the patch, which of its positions are matches vs
     mismatches, and where each guide position pairs on the flank.

  2. A short SCALAR feature vector describing the candidate globally:
     orientation, length, mismatch count, target position on the flank
     (raw + boundary-relative), NC-relative guide position.

The output has fixed shape regardless of NC length — that is the whole
point of centering the patch on the guide. A downstream Set-Transformer
over candidates (per NC) → MIL over NCs (per site) → Set Transformer over
sites (per tnp) can consume this directly.

Candidate selection: top-K by match count PER (orient, L) combination.
Default K=4, orientations=('fwd','rc'), L in [5..16]  ->  2 * 12 * 4 = 96
candidates per NC region. K_max is always fixed to
`len(orientations) * (L_max - L_min + 1) * top_k_per_combo`; if a given
(orient, L) has fewer than K real candidates its slots are zero-filled and
`candidate_mask` is False for those slots.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .alignment import dot_plot, encode_dna, windowed_matches


PATCH_WIDTH_DEFAULT = 64
TOP_K_PER_COMBO_DEFAULT = 4
DEFAULT_L_MIN = 5
DEFAULT_L_MAX = 16
DEFAULT_ORIENTATIONS: tuple[str, ...] = ("fwd", "rc")


# Channel layout of the structure patch (see build_patch_and_features).
STRUCT_UNP_START = 0          # 16 channels: nc_unp_u1..u16
STRUCT_UNP_END = 16
STRUCT_VALID_CH = 16
GUIDE_MASK_CH = 17
MATCH_MATCH_CH = 18
MATCH_MISMATCH_CH = 19
PAIRED_FLANK_CH = 20
ALIGN_POS_CH = 21
PATCH_CHANNELS = 22

PATCH_CHANNEL_NAMES: list[str] = (
    [f"nc_unp_u{k + 1}" for k in range(16)]
    + [
        "struct_valid",
        "guide_mask",
        "match_state_match",
        "match_state_mismatch",
        "paired_flank_pos_norm",
        "align_position_in_guide",
    ]
)
assert len(PATCH_CHANNEL_NAMES) == PATCH_CHANNELS


FEATURE_NAMES: list[str] = [
    "orient_fwd",
    "orient_rc",
    "L",
    "matches",
    "mismatches",
    "score",
    "flank_start_norm",
    "flank_end_norm",
    "boundary_dist_up",
    "boundary_dist_dn",
    "target_side_up",
    "nc_start_norm",
    "nc_len_norm",
]
NUM_FEATURES = len(FEATURE_NAMES)


# WC complement lookup, keyed by our int codes (A=0, C=1, G=2, T=3, N=4).
_COMP_CODES = np.array([3, 2, 1, 0, 4], dtype=np.int8)


@dataclass(frozen=True)
class Candidate:
    """One ungapped alignment: guide at nc[nc_start:nc_start+L] pairs with
    the flank window starting at `flank_start` under `orient`.

    `matches` is the number of matching bases (integer in [0, L])."""
    orient: str          # "fwd" or "rc"
    L: int
    nc_start: int        # 0-indexed
    flank_start: int     # 0-indexed on the ORIGINAL flank (both orientations)
    matches: int


def _fill_candidate_slot(
    patches: np.ndarray,
    feats: np.ndarray,
    mask: np.ndarray,
    slot: int,
    nc_codes: np.ndarray,     # int8 (nc_len,)
    flank_codes: np.ndarray,  # int8 (flank_len,)
    structure_profile: np.ndarray,   # (nc_len, u_max) float
    structure_valid: np.ndarray,     # (nc_len, u_max) bool
    cand: Candidate,
    patch_width: int,
    nc_max: int,
) -> None:
    """Vectorized fill of one (patches[slot], feats[slot]) entry."""
    nc_len = nc_codes.shape[0]
    flank_len = flank_codes.shape[0]
    L = cand.L
    W = patch_width
    guide_center = cand.nc_start + L // 2
    patch_start = guide_center - W // 2
    patch_nc_pos = np.arange(W, dtype=np.int64) + patch_start   # (W,)

    # In-NC mask over patch positions.
    in_nc = (patch_nc_pos >= 0) & (patch_nc_pos < nc_len)

    # Structure channels: gather from structure_profile at valid patch positions.
    if np.any(in_nc):
        idx_nc = patch_nc_pos[in_nc]
        # unp_u1..u16
        u_max = structure_profile.shape[1]
        u_slice = min(u_max, STRUCT_UNP_END - STRUCT_UNP_START)
        patches[slot, in_nc, STRUCT_UNP_START:STRUCT_UNP_START + u_slice] = (
            structure_profile[idx_nc, :u_slice]
        )
        # struct_valid = True iff the l=1 unpaired probability is non-NA
        patches[slot, in_nc, STRUCT_VALID_CH] = structure_valid[idx_nc, 0].astype(np.float32)

    # Guide-mask + alignment-specific channels: only for patch positions
    # that lie inside the guide span (which is itself inside NC).
    in_guide = (
        in_nc
        & (patch_nc_pos >= cand.nc_start)
        & (patch_nc_pos < cand.nc_start + L)
    )
    if np.any(in_guide):
        guide_positions = np.where(in_guide)[0]
        guide_offsets = (patch_nc_pos[guide_positions] - cand.nc_start).astype(np.int64)
        patches[slot, guide_positions, GUIDE_MASK_CH] = 1.0

        nc_bases_guide = nc_codes[patch_nc_pos[guide_positions]]

        if cand.orient == "fwd":
            flank_positions = cand.flank_start + guide_offsets
        else:
            flank_positions = cand.flank_start + (L - 1) - guide_offsets

        # Bound-check flank positions (should be fine by construction).
        fp_ok = (flank_positions >= 0) & (flank_positions < flank_len)
        flank_bases = np.full_like(flank_positions, 4, dtype=np.int8)
        flank_bases[fp_ok] = flank_codes[flank_positions[fp_ok]]
        # For RC we compare against complement of the flank base.
        if cand.orient == "rc":
            flank_bases[fp_ok] = _COMP_CODES[flank_bases[fp_ok]]
        matches_here = (nc_bases_guide == flank_bases) & (nc_bases_guide < 4) & (flank_bases < 4)

        # Two channels: match / mismatch (both zero if we couldn't decide, e.g. N base)
        patches[slot, guide_positions[matches_here], MATCH_MATCH_CH] = 1.0
        # Mismatch = valid guide position but NOT a match.
        mm_here = ~matches_here
        patches[slot, guide_positions[mm_here], MATCH_MISMATCH_CH] = 1.0

        # Paired flank position (normalized) — only defined at guide positions.
        patches[slot, guide_positions, PAIRED_FLANK_CH] = (
            flank_positions.astype(np.float32) / float(flank_len)
        )
        # Align position within guide (0..L-1 mapped to [0,1]).
        denom = float(max(1, L - 1))
        patches[slot, guide_positions, ALIGN_POS_CH] = guide_offsets.astype(np.float32) / denom

    # Scalar feature vector.
    flank_start = float(cand.flank_start)
    flank_end = flank_start + L
    flank_center = flank_start + L / 2.0
    matches = float(cand.matches)
    mismatches = float(L - cand.matches)

    feats[slot, 0] = 1.0 if cand.orient == "fwd" else 0.0
    feats[slot, 1] = 1.0 if cand.orient == "rc" else 0.0
    feats[slot, 2] = float(L)
    feats[slot, 3] = matches
    feats[slot, 4] = mismatches
    feats[slot, 5] = matches / float(L)
    feats[slot, 6] = flank_start / float(flank_len)
    feats[slot, 7] = flank_end / float(flank_len)
    feats[slot, 8] = flank_center / float(flank_len)
    feats[slot, 9] = (flank_len - flank_center) / float(flank_len)
    feats[slot, 10] = 1.0 if flank_center < flank_len / 2.0 else 0.0
    feats[slot, 11] = cand.nc_start / float(max(1, nc_len))
    feats[slot, 12] = nc_len / float(max(1, nc_max))

    mask[slot] = True


def build_candidate_arrays(
    nc: str,
    flank: str,
    structure_profile: np.ndarray,
    structure_valid: np.ndarray,
    top_k_per_combo: int = TOP_K_PER_COMBO_DEFAULT,
    L_min: int = DEFAULT_L_MIN,
    L_max: int = DEFAULT_L_MAX,
    orientations: Sequence[str] = DEFAULT_ORIENTATIONS,
    patch_width: int = PATCH_WIDTH_DEFAULT,
    nc_max: int = 350,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Candidate]]:
    """Enumerate candidates for one (NC, flank) pair and build all patches + features.

    Slot ordering is deterministic:
        for orient in orientations:
            for L in range(L_min, L_max + 1):
                slots [ptr, ptr + top_k_per_combo) hold the top-K candidates
                for that (orient, L), sorted by matches descending.
                Empty (padded) slots are zero-filled and masked False.

    Returns:
      patches:    float32 (K_max, patch_width, PATCH_CHANNELS)
      features:   float32 (K_max, NUM_FEATURES)
      mask:       bool    (K_max,)                 True where a real cand exists
      candidates: list[Candidate | None] len K_max (None for padded slots)
    """
    nc_len = len(nc)
    flank_len = len(flank)
    if structure_profile.shape[0] != nc_len:
        raise ValueError(
            f"structure_profile length {structure_profile.shape[0]} != nc len {nc_len}"
        )
    if structure_valid.shape != structure_profile.shape:
        raise ValueError(
            f"structure_valid shape {structure_valid.shape} != profile shape {structure_profile.shape}"
        )

    n_orient = len(orientations)
    n_L = L_max - L_min + 1
    K_max = n_orient * n_L * top_k_per_combo

    patches = np.zeros((K_max, patch_width, PATCH_CHANNELS), dtype=np.float32)
    feats = np.zeros((K_max, NUM_FEATURES), dtype=np.float32)
    mask = np.zeros((K_max,), dtype=bool)
    cands: list[Candidate | None] = [None] * K_max

    fwd_dot, rc_dot = dot_plot(nc, flank)
    nc_codes = encode_dna(nc)
    flank_codes = encode_dna(flank)

    ptr = 0
    for orient in orientations:
        if orient not in ("fwd", "rc"):
            raise ValueError(f"unknown orientation {orient!r}")
        dot = fwd_dot if orient == "fwd" else rc_dot
        for L in range(L_min, L_max + 1):
            win = windowed_matches(dot, L)  # (nc_len - L + 1, W_win)
            if win.size == 0:
                ptr += top_k_per_combo
                continue
            k = min(top_k_per_combo, win.size)
            flat = win.flatten()
            top_idx = np.argpartition(-flat, k - 1)[:k]
            # Sort descending by score for canonical ordering within the combo.
            top_idx = top_idx[np.argsort(-flat[top_idx])]
            W_win = win.shape[1]
            for i in range(k):
                idx = int(top_idx[i])
                nc_start = idx // W_win
                col = idx % W_win
                if orient == "fwd":
                    flank_start = col
                else:
                    flank_start = flank_len - L - col
                matches = int(flat[idx])
                cand = Candidate(orient, L, nc_start, flank_start, matches)
                _fill_candidate_slot(
                    patches, feats, mask, ptr + i,
                    nc_codes, flank_codes,
                    structure_profile, structure_valid,
                    cand, patch_width, nc_max,
                )
                cands[ptr + i] = cand
            ptr += top_k_per_combo

    return patches, feats, mask, cands


def k_max(
    top_k_per_combo: int = TOP_K_PER_COMBO_DEFAULT,
    L_min: int = DEFAULT_L_MIN,
    L_max: int = DEFAULT_L_MAX,
    orientations: Sequence[str] = DEFAULT_ORIENTATIONS,
) -> int:
    """Total candidate slots per NC for the given hyperparameters."""
    return len(orientations) * (L_max - L_min + 1) * top_k_per_combo
