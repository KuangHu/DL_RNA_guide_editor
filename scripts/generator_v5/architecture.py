"""architecture.py — per-bag architecture axes for the V5 generator.

Every axis is randomized per bag to prevent the model from memorizing
IS110-family architecture. Axes:

  A/B guide decomposition (mandatory, contiguous target)
    L = A + B where A is LTG+core and B is RTG. Durrant Right Target Guide
    variants (RTG in {4bp, 7bp}) reveal LTG+core = 7 constant with RTG in
    {4, 7} giving L_gold in {11, 14}. Generic decomposition:
      A ~ U[max(5, L-8), min(8, L-3)]   B = L - A
    Constraints keep A in [5, 8] and B in [3, 8] as user directive.
    IMPORTANT: the target on the flank is CONTIGUOUS (one L-nt hit),
    not split across flank offsets. windowed_matches requires this;
    a split target would be undetectable by the very Channel A this
    generator supports. See is_split for the small-fraction axis.

  is_split (small fraction, ~15-20% of bags)
    When true: the guide is planted as two flank hits (A then B) with
    a 2-6 nt gap of intervening flank between them. Not the default; a
    marked difficulty axis to be reported separately in acceptance
    (detection rate on contiguous vs split subgroups). Spacing 2-6 nt
    matches core/stagger scale, NOT 8-50 nt (which was a spec placeholder
    that would render the sample undetectable).

  is_reversed_target
    fwd or rc target on the flank. Uniform is fine for a generic guide-
    RNA generator; family-specific orientation preference is not learned.

  N_nc in {1, 2, 3}, active_index uniform
    Bag emits N_nc ncRNAs; one carries the guide. INACTIVE ncRNAs must
    be folded — using an unfolded default would be a perfect label
    leak (structural channels zero on inactives). Length and composition
    also matched to active's distribution.

  TSD (target site duplication)
    tsd_width in {0, 2, 5, 8, 9, 12} and tsd_relation in {before, after,
    both_sides, none}. When tsd_width > 0, a stretch of the flank
    adjacent to the target is duplicated on the opposite side. Recorded
    in arch metadata; implementation is minimal for this scope.

  has_5p_stem_loop (probabilistic architecture metadata)
    After folding, check if the first 20 nt of the ncRNA contain a
    balanced stem-loop (a run of >=3 open brackets followed by a loop
    then >=3 close brackets). Reported in arch metadata.

  ncr_pos_rel_orf in {upstream, downstream, inline}
    Metadata only (this schema doesn't carry an ORF field), uniform.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal


DEFAULT_A_LO = 5
DEFAULT_A_HI = 8

# 2D mismatch geometry axis (added 2026-09-01 after T-WT diagnostic).
# T-WT gold sites cluster 89.5% of mismatches at positions {3, 9, 10} on
# L=11 — this leaves an L=9 subwindow with only 1 mismatch (m=8), which
# is what Mode 1 (E<4 admission) needs. V5's uniform placement gives
# essentially zero such subwindows. To let Channel A's Mode 1 have real
# signal to detect (or explicitly NOT), we randomize this per bag along
# two dimensions:
#
#   concentration ∈ {clustered, dispersed}
#     clustered  : places 2 mm at one end/middle, 1 mm elsewhere
#                   → guarantees SOME L=L-2 subwindow has 1 mm (m=L-3)
#     dispersed  : places 3 mm at spread positions
#                   → guarantees NO L=L-2 subwindow has < 2 mm
#
#   anchor ∈ {5p, 3p, mid}
#     which end/region hosts the cluster (for clustered) or spread center
#     (for dispersed).
#
# Six patterns total, uniform weight (per user directive — no family-
# specific bias like weighting toward T-WT-shape).
DEFAULT_MM_GEOM_CONCENTRATIONS = ("clustered", "dispersed")
# "mid" dropped: (clustered, mid) is impossible on L=11 (mid mm blocks
# every L=9 window); (dispersed, mid) with 3 mm at {0, L//2, L-1} leaves
# window [1:L-1] with 1 mm — that's actually clustered, not dispersed.
DEFAULT_MM_GEOM_ANCHORS = ("5p", "3p")
DEFAULT_B_LO = 3
DEFAULT_B_HI = 8
DEFAULT_SPLIT_PROB = 0.18                # 18% of bags planted as split targets
DEFAULT_SPLIT_GAP_LO = 2
DEFAULT_SPLIT_GAP_HI = 6
DEFAULT_REVERSED_PROB = 0.5              # fwd vs rc uniform
DEFAULT_N_NC_CHOICES = (1, 2, 3)         # uniform
DEFAULT_TSD_WIDTHS = (0, 2, 5, 8, 9, 12)  # uniform
DEFAULT_TSD_RELATIONS = ("before", "after", "both_sides", "none")
DEFAULT_NCR_POS = ("upstream", "downstream", "inline")

# flank_offset_mode — 2nd coherence axis (nc-side coherence was already the
# guide/nc/nc_position axes; flank-side coherence was missed until 2026-09-02).
# Mechanism: target position relative to the insertion junction is set by the
# transposase geometry — fixed within one system, arbitrary across systems.
# Durrant calibration: 73% of sites at offset=0 with bag-shared alignment
# → the calibration establishes THAT bag-internal coherence exists + sets the
#   jitter scale
# → NOT used to set the offset distribution (that is IS110-specific).
# The `inconsistent` value replays the pre-2026-09-02 behavior (per-site
# random) and is a new negative-mode form: nc-coherent but flank-incoherent,
# a real signature of non-guided systems with target preference.
DEFAULT_FLANK_OFFSET_MODES: tuple[str, ...] = ("consistent", "inconsistent")
DEFAULT_FLANK_JITTER: int = 2   # ±nt, prevents zero-variance signature

# ---- v6 axes (2026-09-03; wired in `docs/generator_v6_handoff.md`) ----
# Ranges locked by measurements A / B / verify_end_to_end / align_error.
# Do NOT narrow these to "safe" values — the hard cells are the point.
DEFAULT_NC_HOMOLOGY_RATES: tuple[float, ...] = (0.90, 0.95, 0.97, 0.99, 0.995)
DEFAULT_ACCESSIBILITY_TARGETS: tuple[float, float] = (0.40, 0.95)  # uniform-range
DEFAULT_GC_TARGET_RANGE: tuple[float, float] = (0.25, 0.65)         # uniform-range
DEFAULT_N_SITES_RANGE: tuple[int, int] = (3, 8)                     # inclusive


TsdRelation = Literal["before", "after", "both_sides", "none"]
NcrPos = Literal["upstream", "downstream", "inline"]


@dataclass(frozen=True)
class GuideComposition:
    """A + B decomposition of the guide length L."""
    L: int
    A: int    # LTG + core segment
    B: int    # RTG segment

    def __post_init__(self):
        if self.A + self.B != self.L:
            raise ValueError(f"A + B != L: {self.A} + {self.B} != {self.L}")


@dataclass(frozen=True)
class TSD:
    """Target site duplication metadata."""
    width: int          # nt
    relation: TsdRelation


@dataclass(frozen=True)
class Architecture:
    """All architecture axes for one bag."""
    guide: GuideComposition
    is_split: bool
    split_gap: int              # 0 if not split
    is_reversed_target: bool    # target on flank in rc orientation
    n_nc: int                    # number of noncoding regions
    active_nc_index: int         # which one carries the guide
    tsd: TSD
    ncr_pos_rel_orf: NcrPos
    mm_concentration: str        # {clustered, dispersed}
    mm_anchor: str               # {5p, 3p, mid}
    flank_offset_mode: str = "inconsistent"    # {consistent, inconsistent} — 2026-09-02
    flank_jitter: int = DEFAULT_FLANK_JITTER    # ±nt in consistent mode
    # v6 nc-homology axis (2026-09-03). Uniform per bag; default 1.0
    # preserves pre-v6 (site == canonical) behavior for compat when the
    # axis is turned off. See docs/generator_v6_handoff.md.
    nc_homology_rate: float = 1.0
    # v6 Stage 1e (2026-09-03): accessibility target percentile.
    # Replaces the fixed μ=0.85 that came from only 2 anchor points
    # (T-WT, ISEc21). Per-bag axis; the guide placement sampler uses
    # this as the target μ. Default 0.85 preserves pre-1e behavior;
    # a run that turns the axis on samples uniformly in [0.4, 0.95].
    accessibility_target_percentile: float = 0.85
    # v6 Stage 1f (2026-09-03): GC target for canonical nc sampling.
    # p(A)=p(T)=(1-gc)/2, p(G)=p(C)=gc/2. Default 0.5 preserves pre-1f
    # uniform-ACGT behavior BYTE-IDENTICALLY (rng.choices with equal
    # weights consumes the same RNG state as unweighted — verified
    # 2026-09-03). Under 1e×1f interaction: high GC × high mu can
    # exhaust candidate windows -> placement failure. Failure rate
    # per (gc, mu) 2D cell is a Stage 3 acceptance metric.
    gc_target: float = 0.5

    def to_metadata(self) -> dict:
        """JSONable representation for the bag record's arch{} field."""
        return {
            "L":                self.guide.L,
            "A":                self.guide.A,
            "B":                self.guide.B,
            "is_split":         self.is_split,
            "split_gap":        self.split_gap,
            "is_reversed_target": self.is_reversed_target,
            "n_nc":             self.n_nc,
            "active_nc_index":  self.active_nc_index,
            "tsd_width":        self.tsd.width,
            "tsd_relation":     self.tsd.relation,
            "ncr_pos_rel_orf":  self.ncr_pos_rel_orf,
            "mm_concentration": self.mm_concentration,
            "mm_anchor":        self.mm_anchor,
            "flank_offset_mode": self.flank_offset_mode,
            "flank_jitter":     self.flank_jitter,
            "nc_homology_rate": self.nc_homology_rate,
            "accessibility_target_percentile": self.accessibility_target_percentile,
            "gc_target": self.gc_target,
        }


def sample_flank_offset_mode(rng: random.Random,
                                choices: tuple[str, ...] = DEFAULT_FLANK_OFFSET_MODES
                                ) -> str:
    """Uniformly sample a flank_offset_mode. `consistent` = bag-shared
    plant_start with per-site ±jitter (matches Durrant's coherence axis).
    `inconsistent` = per-site random (V5 pre-2026-09-02 default; a distinct
    negative-shape when combined with nc-coherent placements)."""
    return rng.choice(choices)


def sample_guide_composition(rng: random.Random, L: int,
                                 a_lo: int = DEFAULT_A_LO,
                                 a_hi: int = DEFAULT_A_HI,
                                 b_lo: int = DEFAULT_B_LO,
                                 b_hi: int = DEFAULT_B_HI,
                                 ) -> GuideComposition:
    """Draw A ~ U[max(a_lo, L - b_hi), min(a_hi, L - b_lo)]; B = L - A.
    Guarantees A in [a_lo, a_hi] and B in [b_lo, b_hi]."""
    a_min = max(a_lo, L - b_hi)
    a_max = min(a_hi, L - b_lo)
    if a_min > a_max:
        raise ValueError(f"L={L} incompatible with A in [{a_lo},{a_hi}] "
                          f"and B in [{b_lo},{b_hi}]")
    A = rng.randint(a_min, a_max)
    return GuideComposition(L=L, A=A, B=L - A)


def sample_is_split(rng: random.Random,
                      p: float = DEFAULT_SPLIT_PROB) -> bool:
    return rng.random() < p


def sample_split_gap(rng: random.Random,
                       lo: int = DEFAULT_SPLIT_GAP_LO,
                       hi: int = DEFAULT_SPLIT_GAP_HI) -> int:
    return rng.randint(lo, hi)


def sample_is_reversed(rng: random.Random,
                         p: float = DEFAULT_REVERSED_PROB) -> bool:
    return rng.random() < p


def sample_n_nc(rng: random.Random,
                  choices: tuple[int, ...] = DEFAULT_N_NC_CHOICES) -> int:
    return rng.choice(choices)


def sample_active_index(rng: random.Random, n_nc: int) -> int:
    return rng.randrange(n_nc)


def sample_tsd(rng: random.Random,
                 widths: tuple[int, ...] = DEFAULT_TSD_WIDTHS,
                 relations: tuple[str, ...] = DEFAULT_TSD_RELATIONS) -> TSD:
    w = rng.choice(widths)
    if w == 0:
        r = "none"
    else:
        # If width > 0, drop "none" from the choice pool.
        r = rng.choice(tuple(x for x in relations if x != "none"))
    return TSD(width=w, relation=r)  # type: ignore[arg-type]


def sample_ncr_pos_rel_orf(rng: random.Random,
                              choices: tuple[str, ...] = DEFAULT_NCR_POS
                              ) -> NcrPos:
    return rng.choice(choices)  # type: ignore[return-value]


def sample_mm_geometry(rng: random.Random) -> tuple[str, str]:
    """Uniformly sample (mm_concentration, mm_anchor) from the 2D grid."""
    return (rng.choice(DEFAULT_MM_GEOM_CONCENTRATIONS),
              rng.choice(DEFAULT_MM_GEOM_ANCHORS))


def sample_mismatch_positions(
    rng: random.Random,
    L: int,
    n_mismatches: int,
    concentration: str,
    anchor: str,
) -> list[int]:
    """Return `n_mismatches` distinct positions in [0, L), sampled
    uniformly within the (visible | blind) equivalence class specified
    by `concentration`.

    Constrained-sampling implementation (2026-09-02) — the old hard-
    coded {0,1}/{L-2,L-1}/{2,4,6}/{L-3,L-5,L-7} positions were memorable
    signatures that Channel B could learn to identify without touching
    cross-site coherence. Replaced by class-uniform draws:

      concentration=clustered  → uniform over "visible" tuples (∃ Mode-1
                                 subwindow with <= 1 mm)
      concentration=dispersed  → uniform over "blind" tuples (∀ subwindow
                                 has >= 2 mm)

    The `anchor` parameter is retained in the signature for backward
    compat and downstream diagnostic stratification, but no longer
    influences position selection — 5p vs 3p is a mirror of the same
    class, and class-uniform sampling covers both symmetrically. If
    Channel A metrics stratify equally by mm_anchor after regen, this
    claim is confirmed.

    See `constrained_mm.py` for the enumeration + classifier. Falls back
    to uniform when the requested class is empty at (L, n_mm) — e.g.,
    L=11 n_mm=4 has no visible tuples (impossible to fit <= 1 mm in any
    9-nt subwindow), so "clustered" degrades gracefully to blind.
    """
    from scripts.generator_v5.constrained_mm import (
        sample_mismatch_positions_constrained,
    )
    return sample_mismatch_positions_constrained(
        rng, L, n_mismatches, concentration=concentration, anchor=anchor,
    )


def sample_nc_homology_rate(rng: random.Random,
                               choices: tuple[float, ...] = DEFAULT_NC_HOMOLOGY_RATES
                               ) -> float:
    """Uniformly sample a per-bag nc_homology_rate from the axis grid."""
    return rng.choice(choices)


def sample_accessibility_target_percentile(
    rng: random.Random,
    axis_range: tuple[float, float] = DEFAULT_ACCESSIBILITY_TARGETS,
) -> float:
    """Uniform-in-range per-bag accessibility target percentile (μ for
    the soft θ_pss %ile guide-placement sampler). Stage 1e (2026-09-03)."""
    lo, hi = axis_range
    return rng.uniform(lo, hi)


def sample_gc_target(
    rng: random.Random,
    axis_range: tuple[float, float] = DEFAULT_GC_TARGET_RANGE,
) -> float:
    """Uniform-in-range per-bag GC target for canonical nc sampling.
    Stage 1f (2026-09-03)."""
    lo, hi = axis_range
    return rng.uniform(lo, hi)


def sample_architecture(rng: random.Random, L: int,
                          nc_homology_rate_override: float | None = None,
                          flank_offset_mode_override: str | None = None,
                          accessibility_target_percentile_override: float | None = None,
                          gc_target_override: float | None = None,
                          ) -> Architecture:
    """Draw all architecture axes for a bag given its guide length L.

    `nc_homology_rate_override`: when supplied, use this exact value for
    the nc_homology_rate axis instead of sampling. A8a uses 1.0 (identity
    map anchor); A8a-1 uses 0.95 (cross-validation with verify script).
    `flank_offset_mode_override`: when supplied, pin flank_offset_mode to
    that value. A8a passes "inconsistent" so RNG state matches the 50K
    constrained pre-flank-axis anchor (0.4864/0.9681/0.4356).
    `accessibility_target_percentile_override`: pin the μ used by
    `ncrna_sampler_v2.sample_guide_placement`. Passing 0.85 preserves
    pre-1e behavior (fixed μ=0.85, no extra RNG consumed). None samples
    uniformly from DEFAULT_ACCESSIBILITY_TARGETS.
    """
    guide = sample_guide_composition(rng, L)
    is_split = sample_is_split(rng)
    split_gap = sample_split_gap(rng) if is_split else 0
    n_nc = sample_n_nc(rng)
    mm_conc, mm_anch = sample_mm_geometry(rng)
    if nc_homology_rate_override is not None:
        nc_hom = float(nc_homology_rate_override)
    else:
        nc_hom = sample_nc_homology_rate(rng)
    if accessibility_target_percentile_override is not None:
        access_pct = float(accessibility_target_percentile_override)
    else:
        access_pct = sample_accessibility_target_percentile(rng)
    if gc_target_override is not None:
        gc = float(gc_target_override)
    else:
        gc = sample_gc_target(rng)
    return Architecture(
        guide=guide,
        is_split=is_split,
        split_gap=split_gap,
        is_reversed_target=sample_is_reversed(rng),
        n_nc=n_nc,
        active_nc_index=sample_active_index(rng, n_nc),
        tsd=sample_tsd(rng),
        ncr_pos_rel_orf=sample_ncr_pos_rel_orf(rng),
        mm_concentration=mm_conc,
        mm_anchor=mm_anch,
        flank_offset_mode=(flank_offset_mode_override
                             if flank_offset_mode_override is not None
                             else sample_flank_offset_mode(rng)),
        flank_jitter=DEFAULT_FLANK_JITTER,
        nc_homology_rate=nc_hom,
        accessibility_target_percentile=access_pct,
        gc_target=gc,
    )


# ---------------- helpers used by bag.py ----------------

def check_5p_stem_loop(structure: str, window: int = 20, min_stem: int = 3
                         ) -> bool:
    """Return True if the first `window` nt of the dot-bracket structure
    contain a balanced stem-loop: >=min_stem open brackets, a loop of >=3
    dots, then >=min_stem close brackets, all inside the window."""
    if len(structure) < window:
        return False
    s = structure[:window]
    # Scan for a run of ( ... ) matching, with dots between and stem
    # length >= min_stem.
    for i in range(window):
        if s[i] != "(":
            continue
        depth = 0
        stem_open = 0
        stem_close = 0
        in_stem_close = False
        for j in range(i, window):
            c = s[j]
            if c == "(":
                if in_stem_close:
                    break
                depth += 1
                stem_open += 1
            elif c == ")":
                if not in_stem_close and stem_open >= min_stem:
                    in_stem_close = True
                depth -= 1
                if in_stem_close:
                    stem_close += 1
                if depth == 0:
                    if stem_open >= min_stem and stem_close >= min_stem:
                        return True
                    else:
                        break
        # Try next starting position
    return False
