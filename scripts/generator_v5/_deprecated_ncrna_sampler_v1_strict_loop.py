"""Per-bag ncRNA sampler with fold-based loop mask extraction.

Each bag draws:
  1. length uniform on [ncrna_len_lo, ncrna_len_hi]
  2. sequence uniform on ACGT (composition-matching = A3 axis, not here)
  3. RNAfold MFE structure
  4. loop mask: positions where dot-bracket == '.'
  5. loop windows: contiguous single-stranded runs of length >= guide_length

Callers use loop windows to place the guide (bag.py). No persistent pool;
each bag folds once. Per A1 benchmark: ~75 ms/bag mean; per-bag folding
across 50K bags = ~1 h one-time.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

import RNA


@dataclass(frozen=True)
class SampledNcRNA:
    """One sampled + folded ncRNA."""
    sequence: str            # DNA alphabet (ACGT), length = len
    structure: str           # dot-bracket, same length
    length: int
    loop_mask: np.ndarray    # bool[length]: True = single-stranded
    loop_windows: list[tuple[int, int]]   # (start, length) for contiguous loops


def sample_ncrna(length: int, rng: random.Random | None = None,
                 base_probs: tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
                 ) -> SampledNcRNA:
    """Sample a random ncRNA of the given length, fold, extract loop windows."""
    if rng is None:
        rng = random.Random()
    seq = "".join(rng.choices("ACGT", weights=base_probs, k=length))
    # RNAfold uses U for RNA. Convert to RNA for fold, keep DNA representation.
    seq_rna = seq.replace("T", "U")
    fc = RNA.fold_compound(seq_rna)
    structure, _energy = fc.mfe()
    mask = np.array([c == "." for c in structure], dtype=bool)
    windows = _extract_loop_windows(mask)
    return SampledNcRNA(sequence=seq, structure=structure, length=length,
                          loop_mask=mask, loop_windows=windows)


def _extract_loop_windows(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, length) for each maximal contiguous run of True in mask."""
    windows = []
    n = len(mask)
    i = 0
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            windows.append((i, j - i))
            i = j
        else:
            i += 1
    return windows


def qualifying_windows(nc: SampledNcRNA, min_length: int) -> list[tuple[int, int]]:
    """Windows in the loop mask of at least `min_length` nt.
    Returns (start_of_run, run_length) pairs. Caller draws a placement
    inside each run by choosing a start position in [run_start,
    run_start + run_length - min_length].
    """
    return [(s, L) for s, L in nc.loop_windows if L >= min_length]


def pick_guide_position(nc: SampledNcRNA, guide_length: int,
                         rng: random.Random | None = None) -> int | None:
    """Uniformly pick a start position for a guide of the given length,
    such that the ENTIRE guide fits inside a single loop run.
    Returns None if no run is long enough.
    """
    if rng is None:
        rng = random.Random()
    runs = qualifying_windows(nc, guide_length)
    if not runs:
        return None
    total_placements = sum(L - guide_length + 1 for _, L in runs)
    k = rng.randint(0, total_placements - 1)
    for start, L in runs:
        n_here = L - guide_length + 1
        if k < n_here:
            return start + k
        k -= n_here
    raise RuntimeError("pick_guide_position: unreachable")
