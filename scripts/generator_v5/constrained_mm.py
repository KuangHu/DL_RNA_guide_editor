"""Constrained mismatch-position sampling — replaces the hard-coded
{0,1}/{L-2,L-1}/{2,4,6}/{L-3,L-5,L-7} signature with class-uniform
sampling.

Design (2026-09-02, from the user's spec):

  For each (L, n_mm), enumerate ALL C(L, n_mm) position tuples in [0, L).
  Categorize each tuple into:

    mode2_visible  =  ∃ L' ∈ detector_Ls ∩ [1, L], ∃ offset ∈ [0, L - L']
                       : the mm tuple has <= mm_ceiling mismatches in
                       [offset, offset + L'). With `mm_ceiling=1` this
                       matches Channel A Mode 2's admission (m >= 8 at
                       L' = 9 -> at most 1 mm).
    mode2_blind    =  ∀ L' ∈ detector_Ls ∩ [1, L], ∀ offset ∈ [0, L - L']
                       : the mm tuple has > mm_ceiling mismatches
                       (at least 2 with the default).

  mode2_visible ⨄ mode2_blind = the whole enumeration.

  IMPORTANT — the names encode the RELATIVE detector. "mode2_blind"
  does NOT mean "no detector could pick this up". Mode 1's admission
  (E<4 at L' in {9..12}, m >= L'-3, i.e. mm <= 3) is more lenient, so a
  mode2_blind tuple can still admit Mode 1 at its planted position.
  Measured post-constrained-regen: 32% of dispersed L=11 Mode 1 fires
  hit exact planted (`finding_constrained_sampling_verdict.md`). If a
  future caller wants "no detector picks this up", they need a
  mode1_blind class (mm_ceiling=3), which at L=11 n_mm=3 is empty by
  arithmetic (any 3-mm arrangement admits Mode 1 somewhere) — so that
  variant would collapse the dispersed axis at its most common cell.

  Naming rule (14th recurrence of "one name, two meanings" fixed the
  same way as `_e_per_position_uniform` / `_e_global_uniform`): the
  mode index is in the name. `visible`/`blind` (no prefix) is deprecated
  as ambiguous. Callers get one of two return values, both explicit
  about their reference detector. If a mode1-scoped class ever ships,
  it lives in its own function; the two names never collide.

  When bag_v2 asks for a mismatch layout with mm_concentration ∈
  {clustered, dispersed}, we uniformly sample from
  {mode2_visible, mode2_blind} respectively. The mm_anchor axis
  (5p/3p) is dropped — 5p/3p is a mirror of the same class, so
  uniform-within-class covers both. The arg is kept in the signature
  for backward compat and ignored.

Detector L range (`SCAN_LS`):
  Mode 1 scans L' ∈ {9, 10, 11, 12}. That's the L range visited by our
  detector variants; using it here means "Mode 2's admission threshold
  (mm_ceiling=1) applied across the L set Mode 1 also scans". Mode 2's
  actual admission is at L'=11 alone, but a stricter check over more L
  values is safe (mode2_visible-under-this-scan ⇒ mode2_visible-at-L'=11).

Enumeration cost:
  Worst case at (L=14, n_mm=7): C(14, 7) = 3432 tuples. Total across
  all (L, n_mm) in the V5 range: ~30k tuples. Enumerated once at module
  load, cached by (L, n_mm).

Falsifiable predictions after regen (verified 2026-09-02, both PASS —
see FROZEN.md and `finding_constrained_sampling_verdict.md`):
  Mode 2 clustered L=11 exact:    0.386 -> 0.4356 (+5.0pp, > +2.4pp predicted)
  Mode 1 clustered L=11 coverage: 0.491 -> 0.4652 (stayed, -2.6pp)
"""
from __future__ import annotations

import random
from functools import lru_cache
from itertools import combinations


SCAN_LS: tuple[int, ...] = (9, 10, 11, 12)
"""Detector scan L' values. Mode 1's admission range."""


def _classify_tuple(mm_positions: tuple[int, ...], L: int,
                     scan_Ls: tuple[int, ...] = SCAN_LS,
                     mm_ceiling: int = 1) -> str:
    """Return 'mode2_visible' if any (L', offset) subwindow has <= mm_ceiling
    mismatches (default mm_ceiling=1 = Mode 2 admission at L'=9 m>=8).
    Otherwise 'mode2_blind'. See module docstring for why the mode-scoped
    name matters."""
    mm_set = set(mm_positions)
    scan_here = tuple(Lp for Lp in scan_Ls if 1 <= Lp <= L)
    if not scan_here:
        # No usable subwindow (L below Mode 1's floor). Visible iff whole
        # guide has <= mm_ceiling mismatches.
        return "mode2_visible" if len(mm_set) <= mm_ceiling else "mode2_blind"
    for Lp in scan_here:
        for off in range(L - Lp + 1):
            hits = sum(1 for p in mm_set if off <= p < off + Lp)
            if hits <= mm_ceiling:
                return "mode2_visible"
    return "mode2_blind"


@lru_cache(maxsize=None)
def _enumerate_classes(L: int, n_mm: int,
                        scan_Ls: tuple[int, ...] = SCAN_LS,
                        mm_ceiling: int = 1
                        ) -> tuple[tuple[tuple[int, ...], ...],
                                    tuple[tuple[int, ...], ...]]:
    """Enumerate all C(L, n_mm) position tuples and partition by class.
    Cached: computed once per (L, n_mm)."""
    if n_mm > L or n_mm < 0:
        return (), ()
    mode2_visible: list[tuple[int, ...]] = []
    mode2_blind: list[tuple[int, ...]] = []
    for combo in combinations(range(L), n_mm):
        cls = _classify_tuple(combo, L, scan_Ls, mm_ceiling)
        (mode2_visible if cls == "mode2_visible" else mode2_blind).append(combo)
    return tuple(mode2_visible), tuple(mode2_blind)


def enumerate_classes(L: int, n_mm: int
                       ) -> tuple[tuple[tuple[int, ...], ...],
                                   tuple[tuple[int, ...], ...]]:
    """Public accessor. Returns (mode2_visible_tuples, mode2_blind_tuples)."""
    return _enumerate_classes(L, n_mm)


def class_counts(L: int, n_mm: int) -> tuple[int, int]:
    """Return (|mode2_visible|, |mode2_blind|) — useful for reporting balance."""
    v, b = _enumerate_classes(L, n_mm)
    return len(v), len(b)


def sample_mismatch_positions_constrained(
    rng: random.Random,
    L: int,
    n_mismatches: int,
    concentration: str,
    anchor: str = "5p",     # ignored — mirror axis absorbed into class-uniform
) -> list[int]:
    """Sample n_mismatches positions in [0, L) uniformly from the
    (concentration ∈ {clustered, dispersed}) equivalence class.

    concentration:
      "clustered" -> uniform over mode2_visible tuples
      "dispersed" -> uniform over mode2_blind tuples

    Fallbacks:
      - If n_mismatches < 2 or L < 3: uniform sample (geometry is
        meaningless below 2 mm).
      - If the requested class is empty for this (L, n_mm): fall back to
        the other class if it is non-empty, else uniform sample. This
        handles edge cases like L=11 n_mm=4 where |mode2_visible|=0 by
        arithmetic (any 4 mm in 11 positions leaves >=1 mm in every
        9-nt subwindow), so "clustered" degrades to mode2_blind.
    """
    if n_mismatches < 2 or L < 3:
        return sorted(rng.sample(range(L), n_mismatches))

    mode2_visible, mode2_blind = _enumerate_classes(L, n_mismatches)
    if concentration == "clustered":
        pool = mode2_visible or mode2_blind
    elif concentration == "dispersed":
        pool = mode2_blind or mode2_visible
    else:
        raise ValueError(f"concentration must be 'clustered' or 'dispersed', got {concentration!r}")

    if not pool:
        return sorted(rng.sample(range(L), n_mismatches))
    return sorted(rng.choice(pool))


def _self_test() -> None:
    """Sanity checks used by the module test."""
    # At L=11 n_mm=3:
    v, b = enumerate_classes(11, 3)
    total = len(v) + len(b)
    assert total == 165, f"expected C(11,3)=165, got {total}"

    # Old hard-coded clustered_5p was [0, 1, x] — all such tuples must be in mode2_visible
    for x in range(2, 11):
        assert (0, 1, x) in v, f"expected clustered {(0, 1, x)} in mode2_visible"

    # Old hard-coded dispersed_5p was {2, 4, 6} — must be in blind
    assert (2, 4, 6) in b, "expected dispersed {2,4,6} in blind"

    # Class sizes should be non-degenerate at this (L, n_mm)
    assert len(v) > 0 and len(b) > 0, (len(v), len(b))

    # Uniform sampling should produce different tuples across draws
    rng = random.Random(0)
    draws_v = {tuple(sample_mismatch_positions_constrained(rng, 11, 3, "clustered"))
                for _ in range(50)}
    draws_b = {tuple(sample_mismatch_positions_constrained(rng, 11, 3, "dispersed"))
                for _ in range(50)}
    assert len(draws_v) >= 20, f"clustered draws not diverse: {len(draws_v)}"
    assert len(draws_b) >= 5, f"dispersed draws not diverse: {len(draws_b)}"

    # Every visible draw must have some Mode-1 subwindow with <=1 mm
    for tup in draws_v:
        assert _classify_tuple(tup, 11) == "mode2_visible", tup
    for tup in draws_b:
        assert _classify_tuple(tup, 11) == "mode2_blind", tup

    print(f"[constrained_mm] self-test PASS "
          f"(L=11 n_mm=3: {len(v)} mode2_visible, {len(b)} mode2_blind)")


if __name__ == "__main__":
    _self_test()
