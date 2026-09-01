"""bag_v2.py — V5 generator bag assembler with full architecture axes.

Supersedes bag.py (v1 kept as historical baseline; that scaffold used
the strict-loop pick_guide_position that D1-D5 falsified).

Bag structure (per bag):

  Architecture axes (bag-level, sampled once):
    guide length L, A/B split, is_split, split_gap, is_reversed_target,
    N_nc, active_nc_index, TSD, ncr_pos_rel_orf.
    All from difficulty.py + architecture.py.

  ncRNAs:
    N_nc ncRNAs sampled at the SAME target length. All folded via
    features_structure_v2. Active one carries the guide; inactives are
    structurally-plausible distractors (same length distribution, same
    ACGT composition) — asserted at construction time so an inactive-
    fold-shortcut can't be introduced silently.

  Guide placement on active nc:
    ncrna_sampler_v2 (soft theta_pss %ile sampler, mu=0.85 sigma=0.15).

  Sites (5 per bag by default):
    Distinct real bacterial 120-nt flanks; the target (A + [gap] + B) is
    planted at a uniform position within each flank. When is_split, the
    gap positions retain the flank's original bases (NOT new random
    ACGT) so no fresh randomness is introduced and gap-position
    accidental matches are honest.

  Per-site m_at_planted is measured after planting and stored, so
  downstream acceptance can distinguish "degraded but detectable"
  (m >= 6) from "below detection limit" (m < 6) for split targets.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from preprocess.features_structure_v2 import (
    StructureFeaturesV2, compute_features_v2,
)
from scripts.generator_v5.architecture import (
    Architecture, sample_architecture, check_5p_stem_loop,
)
from scripts.generator_v5.difficulty import (
    Difficulty, RateTable, sample_difficulty, load_or_build_rate_table,
)
from scripts.generator_v5.ncrna_sampler_v2 import sample_guide_placement


DEFAULT_FLANK_LEN = 120
DEFAULT_N_SITES = 5


@dataclass(frozen=True)
class Site:
    site_idx: int
    flank: str
    planted_start_on_flank: int                     # A block start on flank
    planted_A_end_on_flank: int                     # A block end (== A block start + A)
    planted_B_start_on_flank: int                   # B block start (== A end + gap when split, == A end when contig)
    planted_B_end_on_flank: int
    mutated_target: str                              # actually planted concatenation A_mut + gap_original + B_mut (for split) or A_mut+B_mut (contig)
    mismatch_positions: list[int]                    # 0-indexed in the guide (concatenated A+B), not counting the gap
    all_matching_positions_on_nc: list[int]          # nc positions where m_max >= threshold
    m_at_planted: int                                # m measured at the planted nc position with an L-window


@dataclass(frozen=True)
class Bag:
    bag_id: str
    difficulty: Difficulty
    architecture: Architecture
    active_nc_index: int
    ncrna_sequences: list[str]                       # length == n_nc
    ncrna_features: list[StructureFeaturesV2]        # length == n_nc, per-nc BPP fold
    guide_sequence: str                              # A + B (length L)
    planted_start_on_nc: int                         # position on active nc; shared across sites
    n_mismatches: int
    has_5p_stem_loop_per_nc: list[bool]
    sites: list[Site] = field(default_factory=list)

    def to_v42_jsonl(self, m_threshold: int = 8) -> list[dict]:
        recs = []
        for s in self.sites:
            recs.append({
                "site_id": f"{self.bag_id}_site_{s.site_idx:04d}",
                "transposase_id": self.bag_id,
                "ncrna_id": f"{self.bag_id}_ncrna",
                "inputs": {
                    "flank": s.flank,
                    "noncoding_regions": list(self.ncrna_sequences),
                },
                "labels": {
                    "is_positive": True,
                    "target_position_in_flank": [s.planted_start_on_flank,
                                                   s.planted_B_end_on_flank],
                    "target_dna": s.mutated_target,
                    "guide_dna": s.mutated_target,
                    "perfect_guide_dna": self.guide_sequence,
                    "guide_length": self.difficulty.L,
                    "n_mismatches": self.n_mismatches,
                    "mismatch_positions": s.mismatch_positions,
                    "active_noncoding_index": self.active_nc_index,
                    "num_noncoding_regions": len(self.ncrna_sequences),
                    "guide_span_in_active_noncoding": [
                        self.planted_start_on_nc,
                        self.planted_start_on_nc + self.difficulty.L,
                    ],
                    "ncrna_length": len(self.ncrna_sequences[self.active_nc_index]),
                    "arch": self.architecture.to_metadata()
                        | {"target_m_at_planted": s.m_at_planted,
                              "has_5p_stem_loop_active": self.has_5p_stem_loop_per_nc[self.active_nc_index]},
                    "all_matching_positions_on_nc": s.all_matching_positions_on_nc,
                },
            })
        return recs


# ---------------- helpers ----------------

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(s: str) -> str:
    return s.translate(_COMPLEMENT)[::-1]


def _plant_mismatches(guide: str, n_mismatches: int, rng: random.Random
                        ) -> tuple[str, list[int]]:
    if n_mismatches == 0:
        return guide, []
    L = len(guide)
    if n_mismatches > L:
        raise ValueError(f"n_mismatches ({n_mismatches}) > L ({L})")
    positions = rng.sample(range(L), n_mismatches)
    positions.sort()
    chars = list(guide)
    for p in positions:
        original = chars[p]
        alt = [b for b in "ACGT" if b != original]
        chars[p] = rng.choice(alt)
    return "".join(chars), positions


def _plant_target_on_flank(
    flank: str,
    guide_A: str,
    guide_B: str,
    is_split: bool,
    split_gap: int,
    is_reversed: bool,
    n_mismatches: int,
    rng: random.Random,
) -> tuple[str, int, int, int, int, str, list[int]]:
    """Plant target on flank.

    Layout (before any reversal):
        flank[start          : start+A]        = mut_guide[0:A]
        flank[start+A        : start+A+gap]    = unchanged (original flank; gap=0 if !split)
        flank[start+A+gap    : start+A+gap+B]  = mut_guide[A:A+B]

    If is_reversed, the entire *planted region* is replaced by its reverse
    complement. The split gap swaps to the opposite side geometrically: on
    the flank, the block containing rc(B) sits at [start, start+B) and
    rc(A) at [start+B+gap, start+B+gap+A). That's just the mirror image
    of the forward planting.

    Returns:
        flank_final, plant_start, A_end_slot, B_start_slot, plant_end,
        mutated_concat, mm_pos.
        A_end_slot / B_start_slot delimit the gap; plant_end = plant_start +
        A + gap + B. Sizes on either side reflect the (possibly reversed)
        orientation.
    """
    A_len = len(guide_A)
    B_len = len(guide_B)
    total_width = A_len + (split_gap if is_split else 0) + B_len

    mut_full, mm_pos = _plant_mismatches(guide_A + guide_B, n_mismatches, rng)
    mut_A = mut_full[:A_len]
    mut_B = mut_full[A_len:]

    max_start = len(flank) - total_width
    if max_start <= 0:
        raise ValueError(f"target width {total_width} > flank {len(flank)}")
    plant_start = rng.randint(0, max_start)
    plant_end = plant_start + total_width

    if is_reversed:
        # Reversed layout: first block is rc(B) (length B), then gap
        # (original flank in-place), then rc(A) (length A).
        left_block = _reverse_complement(mut_B)
        right_block = _reverse_complement(mut_A)
        left_len, right_len = B_len, A_len
    else:
        left_block, right_block = mut_A, mut_B
        left_len, right_len = A_len, B_len

    A_end_slot = plant_start + left_len
    B_start_slot = A_end_slot + (split_gap if is_split else 0)
    # sanity: B_start_slot + right_len == plant_end

    flank_final = (
        flank[:plant_start]
        + left_block
        + flank[A_end_slot : B_start_slot]   # gap: original flank bases untouched
        + right_block
        + flank[plant_end:]
    )
    assert len(flank_final) == len(flank), (len(flank_final), len(flank))
    mutated_concat = flank_final[plant_start:plant_end]
    return (flank_final, plant_start, A_end_slot, B_start_slot,
            plant_end, mutated_concat, mm_pos)


def _pos_m_max_L(nc: str, flank: str, L: int) -> np.ndarray:
    fwd, rc = dot_plot(nc, flank)
    w_f = windowed_matches(fwd, L)
    w_r = windowed_matches(rc, L)
    if w_f.size == 0 or w_r.size == 0:
        return np.zeros(0, dtype=np.int32)
    n = min(w_f.shape[0], w_r.shape[0])
    return np.maximum(w_f.max(axis=1)[:n], w_r.max(axis=1)[:n])


def _assert_inactive_distribution_matched(ncs: list[str], active_idx: int):
    """Static assertion: inactive ncRNAs must have the SAME length as active
    and ACGT-only composition (no leakage via length or unusual bases)."""
    active_len = len(ncs[active_idx])
    for i, nc in enumerate(ncs):
        if i == active_idx:
            continue
        assert len(nc) == active_len, \
            f"inactive nc {i} length {len(nc)} != active {active_len}"
        for ch in nc:
            assert ch in "ACGT", f"inactive nc {i} has non-ACGT: {ch!r}"


# ---------------- flank pool loader ----------------

def load_flank_pool() -> list[str]:
    pool = []
    from scripts.generator_v5.difficulty import (
        REAL_FLANK_POOL_FAMILIES, REAL_FLANK_POOL_BASEDIR, DEFAULT_FLANK_LEN as FL,
    )
    for fam in REAL_FLANK_POOL_FAMILIES:
        p = f"{REAL_FLANK_POOL_BASEDIR}/real_{fam}_sites.jsonl"
        try:
            with open(p) as f:
                for line in f:
                    d = json.loads(line)
                    m = d.get("generator_metadata", {})
                    if m.get("flank_side") != "downstream":
                        continue
                    fl = d.get("inputs", {}).get("flank")
                    if fl and len(fl) == FL:
                        pool.append(fl.upper())
        except FileNotFoundError:
            continue
    return pool


# ---------------- top-level ----------------

def sample_ncrna(rng: random.Random, nc_len: int) -> str:
    return "".join(rng.choices("ACGT", k=nc_len))


def build_bag(
    bag_id: str,
    rng: random.Random,
    flank_pool: list[str],
    rate_table: RateTable,
    n_sites: int = DEFAULT_N_SITES,
    m_threshold_for_all_matching: int = 8,
) -> Bag | None:
    """Assemble one bag end-to-end. Returns None if the θ-sampler failed
    on the active fold (extremely rare with the soft rule)."""
    diff = sample_difficulty(rng, rate_table)
    arch = sample_architecture(rng, diff.L)

    # ncRNA generation + fold. All same nc_len (from difficulty).
    ncs = [sample_ncrna(rng, diff.nc_len) for _ in range(arch.n_nc)]
    feats = [compute_features_v2(nc, guide_length=diff.L) for nc in ncs]
    _assert_inactive_distribution_matched(ncs, arch.active_nc_index)

    # Guide placement on active nc
    placement = sample_guide_placement(feats[arch.active_nc_index], rng=rng)
    if placement is None:
        return None
    planted_start_on_nc = placement.start
    active_nc = ncs[arch.active_nc_index]
    guide = active_nc[planted_start_on_nc : planted_start_on_nc + diff.L]
    if len(guide) != diff.L:
        return None
    guide_A = guide[: arch.guide.A]
    guide_B = guide[arch.guide.A :]

    n_mismatches = max(0, diff.L - diff.planted_m)

    # 5' stem-loop flags: dot-bracket MFE per nc.
    import RNA
    sl_flags: list[bool] = []
    for nc in ncs:
        fc = RNA.fold_compound(nc.replace("T", "U"))
        structure, _ = fc.mfe()
        sl_flags.append(check_5p_stem_loop(structure))

    # Sites
    sites: list[Site] = []
    if len(flank_pool) < n_sites:
        raise RuntimeError(f"flank pool size {len(flank_pool)} < n_sites {n_sites}")
    fl_idx = rng.sample(range(len(flank_pool)), n_sites)
    for i, k in enumerate(fl_idx):
        base_flank = flank_pool[k]
        (flank_final, A_start, A_end, B_start, B_end,
         mutated_concat, mm_pos) = _plant_target_on_flank(
            base_flank, guide_A, guide_B,
            arch.is_split, arch.split_gap,
            arch.is_reversed_target, n_mismatches, rng,
        )
        # A_start=plant_start; A_end=slot boundary (left block end);
        # B_start=slot boundary (right block start); B_end=plant_end.
        m_arr = _pos_m_max_L(active_nc, flank_final, diff.L)
        matching = [int(p) for p in np.where(m_arr >= m_threshold_for_all_matching)[0]]
        if planted_start_on_nc < len(m_arr):
            m_at_planted = int(m_arr[planted_start_on_nc])
        else:
            m_at_planted = 0
        sites.append(Site(
            site_idx=i,
            flank=flank_final,
            planted_start_on_flank=A_start,
            planted_A_end_on_flank=A_end,
            planted_B_start_on_flank=B_start,
            planted_B_end_on_flank=B_end,
            mutated_target=mutated_concat,
            mismatch_positions=mm_pos,
            all_matching_positions_on_nc=matching,
            m_at_planted=m_at_planted,
        ))

    return Bag(
        bag_id=bag_id,
        difficulty=diff,
        architecture=arch,
        active_nc_index=arch.active_nc_index,
        ncrna_sequences=ncs,
        ncrna_features=feats,
        guide_sequence=guide,
        planted_start_on_nc=planted_start_on_nc,
        n_mismatches=n_mismatches,
        has_5p_stem_loop_per_nc=sl_flags,
        sites=sites,
    )
