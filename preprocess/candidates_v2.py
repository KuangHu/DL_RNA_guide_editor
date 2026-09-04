"""Candidate layer v2 — nc-window-first enumeration with global E-value ranking.

Replaces `preprocess.candidates` (top-K per-(orient, L)) with a candidate
enumeration whose primary axis is the nc coordinate: for each (nc_position,
L, orient) triple, find the best flank alignment and emit it if its
E-value passes a global threshold. Then collapse the same-region peak
across neighboring L windows to the maximal-L extension.

Design targets set by `docs/candidate_layer_spec.md`:
  - **position index preserved**: every candidate names its nc coordinate
    directly; downstream (Channel A cross-site conjunction) can join by
    (Tnp, nc_position, L, orient) without going through a slot index
  - **global E-value ranking**: no per-(orient, L) top-K quota that
    wastes 16/96 slots on L≤6
  - **L range [7, 16]**: below L=7 the E-value density swamps signal
    (identity 1.0 gives E ≈ 0.01, so a "perfect" L=6 match is one
    expected background hit per NC)
  - **maximal-extension dedup**: a peak seen at L=9, 10, 11 collapses to
    the longest one that still passes the threshold, preserving one
    candidate per biological region

This module intentionally does NOT emit the per-candidate patch tensor
(module 4's job). It emits Candidate tuples with position + score fields;
the tensor layer attaches structural + alignment channels around each
candidate's `nc_start`.

Split-mode gapped candidates are a follow-up: this cut only emits
ungapped candidates. See TODO at bottom.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .alignment import dot_plot, windowed_matches


# Canonical emission-mode strings for downstream metric tagging. Any Metric
# constructed from a candidates_v2 output — position array or candidate list —
# MUST have its MetricCondition.emission_mode set to one of these constants,
# NOT the "not_applicable" default in v5a_eval_asserts (that default is a
# legacy back-compat only). Call `assert_v2_emission_mode(mode)` at the
# metric-recording site to raise if the caller forgot to override.
EMISSION_MODE_POSITION_ARRAY: str = "position_array"
EMISSION_MODE_CANDIDATE_LIST_ANCHOR: str = "candidate_list_anchor"
EMISSION_MODE_CANDIDATE_LIST_SPAN: str = "candidate_list_span"

V2_EMISSION_MODES: frozenset[str] = frozenset({
    EMISSION_MODE_POSITION_ARRAY,
    EMISSION_MODE_CANDIDATE_LIST_ANCHOR,
    EMISSION_MODE_CANDIDATE_LIST_SPAN,
})


def assert_v2_emission_mode(mode: str) -> str:
    """Assert `mode` is one of the v2-specific canonical emission modes,
    not the legacy default. Use at every site that produces a Metric
    from `enumerate_position_arrays` or `enumerate_candidates_v2`.

    Raises ValueError if `mode == "not_applicable"` (silent-forget failure
    mode caught 2026-09-02: dataclass default let new v2 consumers silently
    inherit legacy semantics, so cross-emission-mode ratios sneak through
    safe_ratio without a diff).
    """
    if mode == "not_applicable":
        raise ValueError(
            "emission_mode 'not_applicable' is the pre-2026-09-02 legacy default. "
            "Any Metric derived from candidates_v2 output must specify one of: "
            + ", ".join(sorted(V2_EMISSION_MODES))
        )
    if mode not in V2_EMISSION_MODES:
        raise ValueError(
            f"emission_mode {mode!r} not one of the v2-specific canonical modes: "
            + ", ".join(sorted(V2_EMISSION_MODES))
        )
    return mode


DEFAULT_L_MIN = 7
DEFAULT_L_MAX = 16
DEFAULT_E_THRESH = 1.0    # Threshold on the PER-(nc_start, L, orient) E-value
                          # — NOT the global-per-record E from the W2 table.
                          # For L=11 m=8, my E ≈ 0.24; W2's global E ≈ 21.8.
                          # Ratio ≈ n_nc_positions ≈ 190. See
                          # `_e_per_position_uniform` docstring for the exact
                          # relationship. This threshold controls admission
                          # at each individual (nc_start, L, orient) triple:
                          # E<=1 means "expected background hits <= 1 per
                          # triple". The pool ends up O(50) per record.
DEFAULT_ORIENTATIONS: tuple[str, ...] = ("fwd", "rc")


@dataclass(frozen=True, slots=True)
class CandidateV2:
    """One nc-window-first candidate at a specific nc position + length + orient.

    Position invariants:
      - `nc_start` is the canonical index on the NC (never a slot number)
      - `flank_start` is the index of the aligned flank window
      - `L` is the alignment length (nc window is nc[nc_start:nc_start+L])
    Score invariants:
      - `matches` in [0, L]
      - `e_value` is the expected # of same-or-better alignments across
        this record's (flank_positions × 2 orientations); global E-value
        assuming uniform ACGT background at the specified `bg_p` (default
        0.25 — matches historical Channel A calibration; per-record base
        composition override is left for a v2.1 pass)
    """
    orient: str
    L: int
    nc_start: int
    flank_start: int
    matches: int
    e_value: float


def _e_per_position_uniform(L: int, m: int, n_flank_starts: int,
                              n_orients: int = 2, bg_p: float = 0.25) -> float:
    """Per-(nc_start, L, orient) E-value — expected number of by-chance
    alignments of length L with >=m matches at ONE fixed nc_start,
    marginalized over flank position and orientation.

    Formula:
        E = P(m'>=m | L, uniform bg) * n_flank_starts * n_orients

    This is NOT the "global E" from the W2 diagnostic table. The W2
    table reports E = P * n_flank * n_nc * n_orients — the expected
    background count ACROSS the whole record. The two quantities differ
    by a factor of n_nc_starts (typically ~190):

        W2_global_E  ≈  this_per_position_E * n_nc_positions

    Example (L=11, m=8): per-position E ≈ 0.24; global E ≈ 45 (or ≈22
    with single-orient convention). Both are correct; they answer
    different questions. The candidate layer uses the per-position form
    because admission is decided per (nc_start, L, orient) triple —
    each such triple is judged on its own.

    Reference calibration: `difficulty.py`'s rate table also uses a
    per-position rate p (per (nc_start, L, orient), not global), so the
    two frameworks are consistent.
    """
    if m > L or m < 0:
        return math.inf
    p_ge_m = 0.0
    for k in range(m, L + 1):
        p_ge_m += math.comb(L, k) * (bg_p ** k) * ((1.0 - bg_p) ** (L - k))
    return p_ge_m * n_flank_starts * n_orients


def _e_global_uniform(L: int, m: int, n_flank_starts: int, n_nc_positions: int,
                        n_orients: int = 2, bg_p: float = 0.25) -> float:
    """Global (per-record, all-positions) E-value — expected number of
    by-chance alignments of length L with >=m matches ACROSS the whole
    record (all nc_start × flank_start × orient combinations).

    Formula:
        E_global = P(m'>=m | L, uniform bg) * n_flank * n_nc * n_orients
                 = _e_per_position_uniform(L, m, n_flank, n_orients) * n_nc

    This matches the W2 diagnostic table's E convention. It is NOT the
    threshold used for candidate admission — that's per-position. Use
    this when comparing to W2/diagnostic literature; use the per-position
    form when deciding whether an individual candidate at one (nc_start,
    L, orient) triple passes a threshold.

    Naming convention: functions ending in `_per_position_...` return
    the per-triple quantity; functions ending in `_global_...` return the
    per-record quantity. Two names for two different quantities makes
    silent conflation a compile-time / import-time impossibility rather
    than a docstring-only convention (2026-09-02 fix, 14th recurrence of
    "two-rules-share-a-name").
    """
    if m > L or m < 0:
        return math.inf
    per_pos = _e_per_position_uniform(L, m, n_flank_starts, n_orients=n_orients, bg_p=bg_p)
    return per_pos * n_nc_positions


def enumerate_candidates_v2(
    nc: str,
    flank: str,
    L_min: int = DEFAULT_L_MIN,
    L_max: int = DEFAULT_L_MAX,
    e_thresh: float = DEFAULT_E_THRESH,
    orientations: tuple[str, ...] = DEFAULT_ORIENTATIONS,
    bg_p: float = 0.25,
    dedup_maximal: bool = True,
) -> list[CandidateV2]:
    """Enumerate candidates nc-window-first.

    For each `(orient, L)` combination we compute the per-nc-start best
    match by taking the maximum along the flank axis of `windowed_matches`.
    That gives `best[nc_start] = max_flank_start matches[nc_start, flank_start]`
    and `argmax[nc_start] = flank_start*` for later reconstruction.

    A candidate is emitted at `(orient, L, nc_start)` if its E-value
    (computed at the record's flank axis size) is <= `e_thresh`.

    `dedup_maximal`: if True, collapse candidates at the same `orient` and
    overlapping `nc_start` (within one L of each other) to the one whose
    `L` is largest — the maximal-extension window for that region. If
    two overlapping candidates have equal `L`, keep the one with more
    matches; ties broken by lower `nc_start`.
    """
    nc_len = len(nc)
    flank_len = len(flank)
    fwd_dot, rc_dot = dot_plot(nc, flank)

    out: list[CandidateV2] = []
    for orient in orientations:
        if orient not in ("fwd", "rc"):
            raise ValueError(f"unknown orientation {orient!r}")
        dot = fwd_dot if orient == "fwd" else rc_dot
        for L in range(L_min, L_max + 1):
            if L > nc_len or L > flank_len:
                continue
            win = windowed_matches(dot, L)
            if win.size == 0:
                continue
            n_flank_starts = win.shape[1]
            # best per nc_start
            best = win.max(axis=1)
            best_flank_start = win.argmax(axis=1)
            for nc_start in range(win.shape[0]):
                m = int(best[nc_start])
                # Early E-value screen: if E for m matches passes, emit.
                e = _e_per_position_uniform(L, m, n_flank_starts, n_orients=len(orientations), bg_p=bg_p)
                if e > e_thresh:
                    continue
                flank_start = int(best_flank_start[nc_start])
                if orient == "rc":
                    # rc_dot's flank axis is right-anchored; convert to the
                    # canonical left-anchored index on the original flank.
                    flank_start = flank_len - L - flank_start
                out.append(CandidateV2(
                    orient=orient, L=L, nc_start=nc_start,
                    flank_start=flank_start, matches=m, e_value=e,
                ))

    if dedup_maximal:
        out = _dedup_maximal_extension(out)
    return out


def _dedup_maximal_extension(cands: list[CandidateV2]) -> list[CandidateV2]:
    """Same-peak dedup, but ONLY within the same L.

    Discovered 2026-09-02: cross-L dedup (previous behavior — collapse a
    planted L=11 candidate into an overlapping L=16 extension) destroys
    per-L identity. Channel A queries at fixed L (e.g. m>=8 at L=11)
    return 0 hits on such a dedup'd list because the L=11 candidate was
    absorbed into the L=16 winner. Any_L on the list still holds up
    because the L=16 winner covers the planted region under IoU 0.5, but
    that is only pool-recovery — not what Channel A does.

    The fix: dedup is now scoped to (orient, L). Two candidates collapse
    iff they have the same orient, the same L, and their nc + flank spans
    each have IoU >= 0.5 (so shifted-by-1 duplicates of the same peak at
    the same L collapse to one). Candidates at different L stay distinct
    even when their spans overlap; a downstream consumer that wants to
    reduce across-L to a single region representative can post-process,
    but the primary emission preserves per-L identity.

    Position-index preservation for cross-site aggregation is provided by
    `enumerate_position_arrays` — this list-based path is for tensor-attach
    consumers that want per-candidate features.
    """
    if not cands:
        return cands

    def _iou_1d(a0: int, a1: int, b0: int, b1: int) -> float:
        inter = max(0, min(a1, b1) - max(a0, b0))
        union = max(a1, b1) - min(a0, b0)
        return inter / union if union > 0 else 0.0

    def _same_peak_same_L(a: CandidateV2, b: CandidateV2, thresh: float = 0.5) -> bool:
        # Same-L requirement makes nc IoU + flank IoU strictly equivalent to
        # "shifted by <= L*(1-thresh)/(1+thresh)" on each axis, but we keep
        # the general check for robustness against L range changes.
        if a.L != b.L:
            return False
        nc_iou = _iou_1d(a.nc_start, a.nc_start + a.L,
                          b.nc_start, b.nc_start + b.L)
        if nc_iou < thresh:
            return False
        flank_iou = _iou_1d(a.flank_start, a.flank_start + a.L,
                             b.flank_start, b.flank_start + b.L)
        return flank_iou >= thresh

    # Group by (orient, L) then sorted-sweep within each group.
    by_key: dict[tuple[str, int], list[CandidateV2]] = {}
    for c in cands:
        by_key.setdefault((c.orient, c.L), []).append(c)

    out: list[CandidateV2] = []
    for (orient, L), xs in by_key.items():
        xs.sort(key=lambda c: (c.nc_start, -c.matches))
        open_reps: list[CandidateV2] = []
        for c in xs:
            keep: list[CandidateV2] = []
            for r in open_reps:
                if r.nc_start + r.L <= c.nc_start:
                    out.append(r)
                else:
                    keep.append(r)
            open_reps = keep
            merged = False
            for idx, r in enumerate(open_reps):
                if _same_peak_same_L(r, c):
                    # Keep the higher-matches candidate.
                    if (c.matches, -c.nc_start) > (r.matches, -r.nc_start):
                        open_reps[idx] = c
                    merged = True
                    break
            if not merged:
                open_reps.append(c)
        out.extend(open_reps)

    out.sort(key=lambda c: (c.orient, c.L, c.nc_start))
    return out


def enumerate_position_arrays(
    nc: str,
    flank: str,
    L_min: int = DEFAULT_L_MIN,
    L_max: int = DEFAULT_L_MAX,
    orientations: tuple[str, ...] = DEFAULT_ORIENTATIONS,
) -> dict[tuple[str, int], np.ndarray]:
    """Position-indexed emission — the shape that makes cross-site
    conjunction at nc position p an O(1) query instead of a list search.

    For each (orient, L) combination, return a 1-D int32 array `m_max`
    of length `nc_len - L + 1` such that

        m_max[p] = argmax over flank_start of matches(nc[p:p+L], flank[...])

    Indexing directly by nc position `p` is what preserves the shared
    coordinate frame across sites of a bag — Channel A's cross-site
    conjunction becomes a per-position stacking + count operation on
    this array, not a search through candidate lists.

    Companion to `enumerate_candidates_v2`: same underlying computation
    (windowed_matches per (orient, L)), but no E-value screen and no
    dedup — those steps discard the coordinate information that
    downstream position-indexed consumers need. Consumers that want the
    filtered list still call `enumerate_candidates_v2`; consumers that
    want the coordinate frame call this.

    The array is the same shape as `MatchArrays.m_max` from
    `scripts.v5a_framework.match_table` — i.e., a v2 candidate layer can
    populate a MatchTable directly from this call at bag-build time.
    """
    fwd_dot, rc_dot = dot_plot(nc, flank)
    out: dict[tuple[str, int], np.ndarray] = {}
    for orient in orientations:
        dot = fwd_dot if orient == "fwd" else rc_dot
        for L in range(L_min, L_max + 1):
            if L > len(nc) or L > len(flank):
                continue
            win = windowed_matches(dot, L)
            if win.size == 0:
                out[(orient, L)] = np.zeros(0, dtype=np.int32)
            else:
                out[(orient, L)] = win.max(axis=1).astype(np.int32)
    return out


# TODO(v2.1): gapped candidates for split-mode. Emit CandidateV2 with a
# nonzero gap field (add gap_position + L_A + L_B). Search on nc-anchored
# pairs of ungapped alignments constrained to have a gap <= 30 nt on the
# flank. Priority: OPTIONAL — the ungapped first cut hits split any_L 0.966,
# above the spec target 0.95.
