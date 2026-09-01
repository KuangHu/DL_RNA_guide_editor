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
    all_matching_positions_on_nc: list[int]          # nc positions where m_max >= m_threshold_for_all_matching (fixed 8; downstream MIL candidates)
    m_at_planted: int                                # m measured at the planted nc position with an L-window
    competitor_count_at_planted_m: int                # nc positions where m_max >= this SITE's planted_m
    planted_m: int                                    # this site's planted m (drawn per-site around bag target_m)
    n_mismatches: int                                 # this site's n_mismatches = L - planted_m


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
    has_5p_stem_loop_per_nc: list[bool]
    sites: list[Site] = field(default_factory=list)

    def _nc_channels(self, nc_idx: int) -> dict:
        """Emit the 5 structure channels + mask for one nc as JSON-safe lists."""
        f = self.ncrna_features[nc_idx]
        # NaN in float arrays -> None (JSON-safe) so the mask channel becomes
        # the source of truth for validity.
        def _nan_to_none(a: np.ndarray) -> list:
            out = []
            for v in a.tolist():
                out.append(None if (v != v) else float(v))
            return out
        return {
            "dG_open_u1":            [float(v) for v in f.dG_open_u1.tolist()],
            "dG_open_uL_pn":         [float(v) for v in f.dG_open_uL_pn.tolist()],
            "cooperativity_win_pn":  [float(v) for v in f.cooperativity_win_pn.tolist()],
            "E_span_win":            _nan_to_none(f.E_span_win),
            "H_pair_win":            _nan_to_none(f.H_pair_win),
            "windowed_valid":        [bool(v) for v in f.windowed_valid.tolist()],
        }

    def to_v42_jsonl(self, m_threshold: int = 8,
                       include_structure_channels: bool = True) -> list[dict]:
        """Emit one JSONL record per site. Schema (frozen 2026-08-31):

        Per site:
          site_id, transposase_id, ncrna_id
          inputs.flank, inputs.noncoding_regions
          labels.is_positive
          labels.target_position_in_flank  (plant_start, plant_end) — full
            plant width including split gap; use planted_start / planted_end
            below for the block-level breakdown.
          labels.planted_start (=block-A start on flank)
          labels.planted_A_end, planted_B_start, planted_B_end
          labels.planted_m       (bag-level; used for competitor-count def)
          labels.perfect_guide_dna, guide_dna (=mutated_target incl gap for split)
          labels.guide_length (=L), n_mismatches, mismatch_positions
          labels.active_noncoding_index, num_noncoding_regions
          labels.guide_span_in_active_noncoding (=[planted_start_on_nc,
            planted_start_on_nc + L])
          labels.ncrna_length
          labels.arch{}  (all architecture axes + segment_count + orient)
          labels.all_matching_positions_on_nc  (at fixed m_threshold=8)
          labels.competitor_count_at_planted_m (integer scalar)
          labels.m_at_planted    (integer scalar, per-site)

        Per nc (in nc_channels, list of length num_noncoding_regions):
          role  ("active" / "inactive")
          dG_open_u1, dG_open_uL_pn, cooperativity_win_pn, E_span_win,
          H_pair_win, windowed_valid   (per-position or per-window arrays)
          has_5p_stem_loop
        """
        # Precompute all nc channel blobs once per bag (shared across sites).
        if include_structure_channels:
            nc_channels_per_bag = []
            for i, _ in enumerate(self.ncrna_sequences):
                blob = self._nc_channels(i)
                blob["role"] = "active" if i == self.active_nc_index else "inactive"
                blob["has_5p_stem_loop"] = self.has_5p_stem_loop_per_nc[i]
                nc_channels_per_bag.append(blob)
        else:
            nc_channels_per_bag = None

        recs = []
        arch_meta = self.architecture.to_metadata()
        # Denormalized architecture fields per user schema spec.
        arch_meta["segment_count"] = 2 if self.architecture.is_split else 1
        arch_meta["orient"] = ("rev" if self.architecture.is_reversed_target
                                else "fwd")
        for s in self.sites:
            arch_site = dict(arch_meta)
            arch_site["target_m_at_planted"] = s.m_at_planted
            arch_site["has_5p_stem_loop_active"] = self.has_5p_stem_loop_per_nc[self.active_nc_index]
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
                    "planted_start":  s.planted_start_on_flank,
                    "planted_A_end":  s.planted_A_end_on_flank,
                    "planted_B_start": s.planted_B_start_on_flank,
                    "planted_B_end":  s.planted_B_end_on_flank,
                    "planted_m":      s.planted_m,
                    "bag_target_m":   self.difficulty.target_m,
                    "target_dna":     s.mutated_target,
                    "guide_dna":      s.mutated_target,
                    "perfect_guide_dna": self.guide_sequence,
                    "guide_length":   self.difficulty.L,
                    "n_mismatches":   s.n_mismatches,
                    "mismatch_positions": s.mismatch_positions,
                    "active_noncoding_index": self.active_nc_index,
                    "num_noncoding_regions": len(self.ncrna_sequences),
                    "guide_span_in_active_noncoding": [
                        self.planted_start_on_nc,
                        self.planted_start_on_nc + self.difficulty.L,
                    ],
                    "ncrna_length":   len(self.ncrna_sequences[self.active_nc_index]),
                    "arch":           arch_site,
                    "all_matching_positions_on_nc": s.all_matching_positions_on_nc,
                    "competitor_count_at_planted_m": s.competitor_count_at_planted_m,
                    "m_at_planted":   s.m_at_planted,
                    "nc_channels":    nc_channels_per_bag,
                },
            })
        return recs


# ---------------- helpers ----------------

_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def _reverse_complement(s: str) -> str:
    return s.translate(_COMPLEMENT)[::-1]


def _plant_mismatches(guide: str, n_mismatches: int, rng: random.Random,
                        mm_concentration: str | None = None,
                        mm_anchor: str | None = None) -> tuple[str, list[int]]:
    """Return (mutated_target, mm_positions).

    If mm_concentration+mm_anchor supplied (from the architecture axis),
    positions follow that 2D geometry pattern (see
    scripts.generator_v5.architecture.sample_mismatch_positions).
    Otherwise falls back to uniform-random placement.
    """
    if n_mismatches == 0:
        return guide, []
    L = len(guide)
    if n_mismatches > L:
        raise ValueError(f"n_mismatches ({n_mismatches}) > L ({L})")
    if mm_concentration is not None and mm_anchor is not None:
        from scripts.generator_v5.architecture import sample_mismatch_positions
        positions = sample_mismatch_positions(rng, L, n_mismatches,
                                                 mm_concentration, mm_anchor)
    else:
        positions = sorted(rng.sample(range(L), n_mismatches))
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
    mm_concentration: str | None = None,
    mm_anchor: str | None = None,
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

    mut_full, mm_pos = _plant_mismatches(guide_A + guide_B, n_mismatches, rng,
                                             mm_concentration=mm_concentration,
                                             mm_anchor=mm_anchor)
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


def build_negative_bag(
    bag_id: str,
    rng: random.Random,
    flank_pool: list[str],
    rate_table: RateTable,
    n_sites: int = DEFAULT_N_SITES,
) -> "NegativeBag | None":
    """Assemble one NEGATIVE bag (no guide planted).

    Structurally identical to positive bags on the parts Channel A sees:
    shared nc (random ACGT + fold + structure channels), 5 distinct real
    bacterial flanks, N_nc >= 1 ncRNAs. NO target planted. Any peak
    Channel A emits on such a bag is a false positive.

    Sampled axes (kept for stratified FP-rate analysis):
      L (nominal, drives structure L for guide_length field)
      nc_len (from same U[70,300] distribution as positives)
      n_nc, active_nc_index (uniform)
      has_5p_stem_loop_active (post-fold check)
      ncr_pos_rel_orf (metadata)

    Not sampled (irrelevant without a target):
      is_split, split_gap, is_reversed_target, tsd_*, A/B decomposition
    """
    diff = sample_difficulty(rng, rate_table)
    from scripts.generator_v5.architecture import (
        DEFAULT_N_NC_CHOICES, DEFAULT_NCR_POS,
    )
    n_nc = rng.choice(DEFAULT_N_NC_CHOICES)
    active_idx = rng.randrange(n_nc)
    ncr_pos = rng.choice(DEFAULT_NCR_POS)

    ncs = [sample_ncrna(rng, diff.nc_len) for _ in range(n_nc)]
    feats = [compute_features_v2(nc, guide_length=diff.L) for nc in ncs]
    _assert_inactive_distribution_matched(ncs, active_idx)

    # 5' stem-loop flags
    import RNA
    sl_flags: list[bool] = []
    for nc in ncs:
        fc = RNA.fold_compound(nc.replace("T", "U"))
        structure, _ = fc.mfe()
        sl_flags.append(check_5p_stem_loop(structure))

    # Sample flanks; NO planting — pool flanks stay untouched.
    if len(flank_pool) < n_sites:
        raise RuntimeError(f"flank pool size {len(flank_pool)} < n_sites {n_sites}")
    fl_idx = rng.sample(range(len(flank_pool)), n_sites)
    flanks = [flank_pool[k] for k in fl_idx]

    return NegativeBag(
        bag_id=bag_id,
        L=diff.L,
        nc_len=diff.nc_len,
        n_nc=n_nc,
        active_nc_index=active_idx,
        ncr_pos_rel_orf=ncr_pos,
        ncrna_sequences=ncs,
        ncrna_features=feats,
        has_5p_stem_loop_per_nc=sl_flags,
        flanks=flanks,
    )


@dataclass(frozen=True)
class NegativeBag:
    """A no-plant bag. 5 flanks, one shared nc, no target."""
    bag_id: str
    L: int
    nc_len: int
    n_nc: int
    active_nc_index: int
    ncr_pos_rel_orf: str
    ncrna_sequences: list[str]
    ncrna_features: list[StructureFeaturesV2]
    has_5p_stem_loop_per_nc: list[bool]
    flanks: list[str]

    def to_jsonl(self) -> list[dict]:
        """Emit per-site records. Uses the same schema shape as positives
        so downstream MatchTable build can construct SiteRecords; gold
        fields set to sentinel -1 (channel_a_v5 will treat these as
        negatives via is_positive=False)."""
        active_nc = self.ncrna_sequences[self.active_nc_index]
        # Precompute per-nc structure channels (shared across sites)
        def _nan_to_none(a):
            out = []
            for v in a.tolist():
                out.append(None if (v != v) else float(v))
            return out
        nc_channels = []
        for i, (nc, f) in enumerate(zip(self.ncrna_sequences, self.ncrna_features)):
            nc_channels.append({
                "role": "active" if i == self.active_nc_index else "inactive",
                "has_5p_stem_loop": self.has_5p_stem_loop_per_nc[i],
                "dG_open_u1": [float(v) for v in f.dG_open_u1.tolist()],
                "dG_open_uL_pn": [float(v) for v in f.dG_open_uL_pn.tolist()],
                "cooperativity_win_pn": [float(v) for v in f.cooperativity_win_pn.tolist()],
                "E_span_win": _nan_to_none(f.E_span_win),
                "H_pair_win": _nan_to_none(f.H_pair_win),
                "windowed_valid": [bool(v) for v in f.windowed_valid.tolist()],
            })

        arch = {
            "L": self.L,
            "n_nc": self.n_nc,
            "active_nc_index": self.active_nc_index,
            "ncr_pos_rel_orf": self.ncr_pos_rel_orf,
            "is_negative": True,
            "has_5p_stem_loop_active": self.has_5p_stem_loop_per_nc[self.active_nc_index],
        }
        recs = []
        for i, fl in enumerate(self.flanks):
            recs.append({
                "site_id": f"{self.bag_id}_site_{i:04d}",
                "transposase_id": self.bag_id,
                "ncrna_id": f"{self.bag_id}_ncrna",
                "inputs": {
                    "flank": fl,
                    "noncoding_regions": list(self.ncrna_sequences),
                },
                "labels": {
                    "is_positive": False,
                    "guide_length": self.L,
                    "active_noncoding_index": self.active_nc_index,
                    "num_noncoding_regions": self.n_nc,
                    "guide_span_in_active_noncoding": [-1, -1],
                    "ncrna_length": len(active_nc),
                    "arch": arch,
                    "nc_channels": nc_channels,
                },
            })
        return recs


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

    # PER-SITE planted_m (2026-08-31 fix): the SAME 86/10/4 tail sampled
    # INDEPENDENTLY per site around the bag's target_m. Bag-level shared m
    # made Channel A's S=5 conjunction trivial: 55% of bags had range=0,
    # 80% had all-5-hits by construction. Per-site draws restore the
    # probabilistic-hit premise Channel A operates on.
    from scripts.generator_v5.difficulty import sample_planted_m
    per_site_planted_m = [sample_planted_m(rng, diff.target_m) for _ in range(n_sites)]
    per_site_n_mismatches = [max(0, diff.L - m) for m in per_site_planted_m]

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
        site_planted_m = per_site_planted_m[i]
        site_n_mismatches = per_site_n_mismatches[i]
        (flank_final, A_start, A_end, B_start, B_end,
         mutated_concat, mm_pos) = _plant_target_on_flank(
            base_flank, guide_A, guide_B,
            arch.is_split, arch.split_gap,
            arch.is_reversed_target, site_n_mismatches, rng,
            mm_concentration=arch.mm_concentration,
            mm_anchor=arch.mm_anchor,
        )
        # A_start=plant_start; A_end=slot boundary (left block end);
        # B_start=slot boundary (right block start); B_end=plant_end.
        m_arr = _pos_m_max_L(active_nc, flank_final, diff.L)
        matching = [int(p) for p in np.where(m_arr >= m_threshold_for_all_matching)[0]]
        if planted_start_on_nc < len(m_arr):
            m_at_planted = int(m_arr[planted_start_on_nc])
        else:
            m_at_planted = 0
        # competitor_count_at_planted_m: positions with m_max >= this site's
        # planted_m. Test 1 semantics: per-site.
        competitor_count_at_planted_m = int((m_arr >= site_planted_m).sum())
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
            competitor_count_at_planted_m=competitor_count_at_planted_m,
            planted_m=site_planted_m,
            n_mismatches=site_n_mismatches,
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
        has_5p_stem_loop_per_nc=sl_flags,
        sites=sites,
    )
