"""Bag scaffold — the V4.2 gap fix.

A bag contains N_bag sites that share:
  - one ncRNA
  - one guide sequence (drawn from a loop region of the ncRNA)
  - one planted_start_on_nc (position where the guide sits on nc)
  - one intended target sequence (= reverse-complement of guide)

Each site of the bag differs in:
  - its 120-nt real bacterial flank (drawn from the 2,763-flank pool)
  - the planted position of the target within that flank
  - the specific mismatches introduced (m of them, uniform position)

The schema carries `all_matching_positions_on_nc` per site so
downstream evaluation is not tricked by real-background matches
elsewhere on the flank producing coherent coincident hits at
nc positions other than the planted one.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from scripts.generator_v5.ncrna_sampler import SampledNcRNA, sample_ncrna, pick_guide_position


@dataclass(frozen=True)
class Site:
    """One insertion site in a bag."""
    site_idx: int
    flank: str                                # 120 nt, target planted somewhere inside
    planted_start_on_flank: int
    mutated_target: str                        # what was actually planted (guide with m mismatches)
    mismatch_positions: list[int]              # positions within target where mismatches sit (0-indexed)
    all_matching_positions_on_nc: list[int]    # nc positions where m_max on this flank >= m_threshold


@dataclass(frozen=True)
class Bag:
    """One bag = one Tnp x one ncRNA x one guide x N sites."""
    bag_id: str
    ncrna_sequence: str                        # DNA alphabet
    ncrna_structure: str                       # dot-bracket
    guide_sequence: str                        # the L-nt guide segment (DNA)
    guide_length: int                          # L
    planted_start_on_nc: int                   # where the guide sits on nc; SAME across all sites
    n_mismatches: int                          # m mismatches per site
    sites: list[Site] = field(default_factory=list)

    def to_v42_jsonl(self) -> list[dict]:
        """Emit each site as a JSONL record matching V4.2 positives_v42.jsonl schema."""
        out = []
        for s in self.sites:
            out.append({
                "site_id": f"{self.bag_id}_site_{s.site_idx:04d}",
                "transposase_id": self.bag_id,
                "ncrna_id": f"{self.bag_id}_ncrna",
                "inputs": {
                    "flank": s.flank,
                    "noncoding_regions": [self.ncrna_sequence],
                },
                "labels": {
                    "is_positive": True,
                    "target_position_in_flank": [s.planted_start_on_flank,
                                                  s.planted_start_on_flank + self.guide_length],
                    "target_dna": s.mutated_target,
                    "guide_dna": s.mutated_target,   # observed target = guide with mismatches
                    "perfect_guide_dna": self.guide_sequence,
                    "guide_length": self.guide_length,
                    "n_mismatches": self.n_mismatches,
                    "mismatch_positions": s.mismatch_positions,
                    "active_noncoding_index": 0,
                    "num_noncoding_regions": 1,
                    "guide_span_in_active_noncoding": [self.planted_start_on_nc,
                                                        self.planted_start_on_nc + self.guide_length],
                    "ncrna_length": len(self.ncrna_sequence),
                    "arch": {
                        # Populated by architecture axes in A3.
                        # Reserved keys per docs/generator_spec.md.
                    },
                    # Bag-scaffold-specific: for downstream evaluation.
                    "all_matching_positions_on_nc": s.all_matching_positions_on_nc,
                },
            })
        return out


def _plant_mismatches(guide: str, n_mismatches: int, rng: random.Random
                      ) -> tuple[str, list[int]]:
    """Return (mutated_target, mismatch_positions).
    Mismatch positions uniform over [0, len(guide)); the mutated base is
    drawn uniform from the 3 non-original bases.
    """
    if n_mismatches == 0:
        return guide, []
    L = len(guide)
    if n_mismatches > L:
        raise ValueError(f"n_mismatches ({n_mismatches}) > L ({L})")
    positions = rng.sample(range(L), n_mismatches)
    positions.sort()
    seq = list(guide)
    for p in positions:
        original = seq[p]
        alternatives = [b for b in "ACGT" if b != original]
        seq[p] = rng.choice(alternatives)
    return "".join(seq), positions


def _competitors_on_nc(nc: str, flank: str, L: int, m_threshold: int
                       ) -> list[int]:
    """Return nc positions where m_max(p, flank) >= m_threshold, pooled
    over both orientations. This is what Channel A would 'see' as
    admissible positions for this site.
    """
    fwd_dot, rc_dot = dot_plot(nc, flank)
    w_f = windowed_matches(fwd_dot, L)
    w_r = windowed_matches(rc_dot, L)
    if w_f.size == 0 or w_r.size == 0:
        return []
    n = min(w_f.shape[0], w_r.shape[0])
    m_max = np.maximum(w_f.max(axis=1)[:n], w_r.max(axis=1)[:n])
    return [int(p) for p in np.where(m_max >= m_threshold)[0]]


def _insert_target_into_flank(flank: str, target: str, position: int) -> str:
    """Overwrite flank[position:position+len(target)] with target."""
    L = len(target)
    if position < 0 or position + L > len(flank):
        raise ValueError(f"target [{position}:{position + L}] outside flank of length {len(flank)}")
    return flank[:position] + target + flank[position + L:]


def build_bag(bag_id: str,
              nc: SampledNcRNA,
              guide_length: int,
              planted_start_on_nc: int,
              flanks: list[str],
              n_mismatches: int,
              rng: random.Random | None = None,
              m_threshold_for_competitors: int = 8,
              ) -> Bag:
    """Assemble a bag from a pre-sampled ncRNA, chosen guide position, and
    N pre-drawn flanks.

    Each site plants the mutated target at a uniform random position within
    its flank. The GUIDE sequence and PLANTED nc position are shared across
    sites (V4.2 gap fix). Mismatch positions are drawn per site (independent
    for each site's target realization).
    """
    if rng is None:
        rng = random.Random()

    guide = nc.sequence[planted_start_on_nc : planted_start_on_nc + guide_length]
    if len(guide) != guide_length:
        raise ValueError(f"guide extraction failed at planted_start_on_nc={planted_start_on_nc}")

    sites: list[Site] = []
    flank_len = len(flanks[0]) if flanks else 120
    max_flank_start = flank_len - guide_length
    for i, flank_template in enumerate(flanks):
        # Plant target (guide with m mismatches) at uniform position within flank.
        mutated_target, mismatch_positions = _plant_mismatches(guide, n_mismatches, rng)
        planted_start_on_flank = rng.randint(0, max_flank_start)
        flank_final = _insert_target_into_flank(flank_template, mutated_target, planted_start_on_flank)
        # Compute all nc positions where this flank's m_max >= m_threshold.
        # This captures both the planted position AND any real-background
        # accidental matches on the flank template.
        matching = _competitors_on_nc(nc.sequence, flank_final, guide_length,
                                        m_threshold=m_threshold_for_competitors)
        sites.append(Site(
            site_idx=i,
            flank=flank_final,
            planted_start_on_flank=planted_start_on_flank,
            mutated_target=mutated_target,
            mismatch_positions=mismatch_positions,
            all_matching_positions_on_nc=matching,
        ))
    return Bag(
        bag_id=bag_id,
        ncrna_sequence=nc.sequence,
        ncrna_structure=nc.structure,
        guide_sequence=guide,
        guide_length=guide_length,
        planted_start_on_nc=planted_start_on_nc,
        n_mismatches=n_mismatches,
        sites=sites,
    )


def load_flank_pool() -> list[str]:
    """Load the 2,763-family negative-family downstream flanks (same as A+)."""
    pool = []
    for fam in ("IS10-R", "IS30", "IS903", "ISAjo2", "ISLdl1"):
        p = f"/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/formatted/real_{fam}_sites.jsonl"
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                m = d.get("generator_metadata", {})
                if m.get("flank_side") != "downstream":
                    continue
                fl = d.get("inputs", {}).get("flank")
                if fl and len(fl) == 120:
                    pool.append(fl.upper())
    return pool


def generate_bag_end_to_end(
    bag_id: str,
    ncrna_length: int,
    guide_length: int,
    n_mismatches: int,
    n_bag: int,
    flank_pool: list[str],
    rng: random.Random | None = None,
) -> Bag | None:
    """End-to-end: sample nc, place guide in loop, draw N flanks, assemble.
    Returns None if no loop window is long enough for the guide (very rare).
    """
    if rng is None:
        rng = random.Random()
    nc = sample_ncrna(ncrna_length, rng)
    planted_start = pick_guide_position(nc, guide_length, rng)
    if planted_start is None:
        return None
    flank_indices = rng.sample(range(len(flank_pool)), n_bag)
    flanks = [flank_pool[i] for i in flank_indices]
    return build_bag(bag_id, nc, guide_length, planted_start, flanks,
                       n_mismatches, rng)
