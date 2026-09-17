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
    competitor_count_at_site_planted_m: int                # nc positions where m_max >= this SITE's planted_m
    planted_m: int                                    # this site's planted m (drawn per-site around bag target_m)
    n_mismatches: int                                 # this site's n_mismatches = L - planted_m


@dataclass(frozen=True)
class Bag:
    bag_id: str
    difficulty: Difficulty
    architecture: Architecture
    active_nc_index: int
    ncrna_sequences: list[str]                       # length == n_nc  (bag-level reference; canonical for the active slot)
    ncrna_features: list[StructureFeaturesV2]        # length == n_nc, per-nc BPP fold (over canonical for the active slot)
    guide_sequence: str                              # A + B (length L); "" for scattered
    planted_start_on_nc: int                         # position on active nc; shared across sites; -1 for scattered
    has_5p_stem_loop_per_nc: list[bool]
    sites: list[Site] = field(default_factory=list)
    negative_mode: str = "none"                      # 'none' | 'scattered' | 'partial'
    per_site_nc_start: list[int] = field(default_factory=list)   # per-site nc_start (scattered) or repeat of bag-level
    per_site_is_planted: list[bool] = field(default_factory=list)  # partial: False for unplanted sites
    # v7 axis 6 — per-site orient (V7_SPEC §2.4). When empty list, bag
    # follows legacy bag-level `arch.is_reversed_target` behavior for
    # every site (v6r2 semantic). When populated, each site's orient
    # overrides the bag-level; length must equal n_sites.
    per_site_is_reversed: list[bool] = field(default_factory=list)
    # v7 — recorded PROV for the sampled `p_same` (the bag's "follow
    # bag_orient" rate). Only meaningful when per_site_is_reversed is
    # populated. Default 1.0 = "all sites follow bag orient" = legacy.
    orient_p_same: float = 1.0
    # v7 — was this bag built with v7_mode=True? Used at emit time to
    # decide output_format default. Explicit rather than inferred so a
    # v7 bag with junction_motif_length=0 is still emitted as v7.
    v7_mode: bool = False
    # v7 axis 8 — junction motif (V7_SPEC §2.6). PROV only. Not model input.
    junction_motif_length: int = 0
    junction_motif_consistent: bool = False
    # v6 canonical + homology (2026-09-03).
    canonical_nc: str = ""                                    # bag-level canonical active-slot nc (== ncrna_sequences[active] when homology_rate=1.0)
    canonical_fold: str = ""                                  # dot-bracket MFE of canonical_nc; "" for legacy bags
    per_site_site_nc: list[str] = field(default_factory=list) # per-site mutated active-slot nc (== canonical_nc under homology=1.0)
    per_site_site_to_canonical_map: list[list[int]] = field(default_factory=list)  # per-site site→canonical map (from PairwiseAligner — the SAME function that runs at deploy)
    per_site_oracle_map: list[list[int]] = field(default_factory=list)  # per-site ground-truth map from the mutation model (diagnostic ONLY; NEVER used as model input — see feedback_conjunction_train_deploy_gap)
    per_site_epsilon_align: list[float] = field(default_factory=list)   # per-site alignment error rate = disagreement between aligner map and oracle map, normalized by site_nc length
    # v6 Stage 1d twin negative diagnostics.
    twin_tol_used: float = 0.0            # tolerance actually used for percentile matching; 0.0 for non-twin bags
    twin_positions: list[int] = field(default_factory=list)  # per-site canonical positions (twin bags only)

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
                       include_structure_channels: bool = True,
                       output_format: str | None = None,
                       generator_version_or_commit: str = "unknown",
                       build_date: str = "",
                       flank_pool_source: str | None = None,
                       ) -> list[dict]:
        # v7-mode bags default to v7 output; legacy defaults to v6r2. Callers
        # can still override output_format explicitly (mostly for test).
        if output_format is None:
            output_format = "v7" if self.v7_mode else "v6r2"
        if flank_pool_source is None:
            flank_pool_source = "synthetic_random" if self.v7_mode else "is_sites"
        """Emit one JSONL record per site.

        `output_format`:
          - "v6r2" (default): frozen v6r2 schema, byte-compatible with existing shards.
            Emits multi-region noncoding_regions + labels.canonical_nc /
            canonical_fold / site_to_canonical_map / oracle_map / epsilon_align.
          - "v7": per V7_SPEC. Emits ONLY 1 nc region (the active), NO
            canonical_nc / canonical_fold / site_to_canonical_map / oracle_map
            / epsilon_align in labels (they were the source of the 2026-09-10
            SUBSTANTIVE_INPUT_MISMATCH retraction). Adds `generator_metadata`
            with `data_source` / `build_date` / `generator_version_or_commit` /
            `flank_pool_source` per CANONICAL_BAG_SPEC §2.6.

        `generator_version_or_commit`, `build_date`, `flank_pool_source`:
          only used when output_format="v7"; ignored under "v6r2".

        Schema (v6r2 default, frozen 2026-08-31):
          Per site: site_id, transposase_id, ncrna_id;
                    inputs.flank, inputs.noncoding_regions;
                    labels.is_positive / target_position_in_flank /
                      planted_start / planted_A_end / planted_B_start /
                      planted_B_end / planted_m / perfect_guide_dna /
                      guide_dna / guide_length / n_mismatches /
                      mismatch_positions / active_noncoding_index /
                      num_noncoding_regions / guide_span_in_active_noncoding /
                      ncrna_length / arch / all_matching_positions_on_nc /
                      competitor_count_at_site_planted_m / m_at_planted /
                      nc_channels
                    v6r2 also: canonical_nc / canonical_fold /
                      site_nc_sequence / site_to_canonical_map /
                      oracle_map / epsilon_align (all DROPPED under v7).
          Per nc (nc_channels list): role, dG_open_u1, dG_open_uL_pn,
                                     cooperativity_win_pn, E_span_win,
                                     H_pair_win, windowed_valid,
                                     has_5p_stem_loop.
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
        # Stage 1c: bag-level n_sites (uniform on DEFAULT_N_SITES_RANGE
        # when the run doesn't pin it). Consumed by Channel A's theta =
        # S/n_sites threshold and by the data checker's Layer 1
        # (bag_target_diversity) + Layer 3 stratification.
        arch_meta["n_sites"] = len(self.sites)
        if self.negative_mode == "twin":
            arch_meta["twin_tol_used"] = self.twin_tol_used
            arch_meta["twin_positions"] = list(self.twin_positions)
            # p_true: the positive's canonical planted position. Recorded
            # so a downstream A12 pass can compare gold-vs-twin AUROC on
            # each of the 5 structure channels directly, without needing
            # to reconstruct p_true from the accessibility target.
            arch_meta["twin_p_true"] = self.planted_start_on_nc
        # Schema convention: emit the pool-side name so downstream (preprocess/
        # candidates.py, MatchTable) can use arch.orient directly without a
        # naming shim. 'rc' = reverse-complement (was 'rev' pre-2026-09-01).
        # See docs/v5_schema.md for the mapping. Pool_recovery + Channel A
        # accept both 'rev' (frozen 50K batch) and 'rc' (new batches).
        arch_meta["orient"] = ("rc" if self.architecture.is_reversed_target
                                else "fwd")
        # v7 (V7_SPEC §2.4): emit per-site orient list + p_same as PROV when
        # per-site orient is populated. Preserves scalar `arch.orient` above
        # (legacy readers still see it as bag-majority). Loader (data.py)
        # after 2026-09-10 rev no longer consumes either — both are PROV.
        if self.per_site_is_reversed:
            arch_meta["orient_per_site"] = [
                ("rc" if r else "fwd") for r in self.per_site_is_reversed
            ]
            arch_meta["orient_p_same"] = float(self.orient_p_same)
        # v7 (V7_SPEC §2.6): junction motif PROV.
        if self.v7_mode:
            arch_meta["junction_motif_length"] = int(self.junction_motif_length)
            arch_meta["junction_motif_consistent"] = bool(self.junction_motif_consistent)
        is_positive_bag = (self.negative_mode == "none")
        # Canonical present iff v6 emission path ran (build_bag). Legacy
        # NegativeBag / older code paths that construct Bag without v6
        # fields emit no canonical block, and MatchTable falls back to
        # the pre-v6 code path (site_nc == canonical).
        has_canonical = bool(self.canonical_nc)
        for s in self.sites:
            arch_site = dict(arch_meta)
            arch_site["target_m_at_planted"] = s.m_at_planted
            arch_site["has_5p_stem_loop_active"] = self.has_5p_stem_loop_per_nc[self.active_nc_index]
            # Per-site nc_start (scattered varies per site; positive + partial
            # share the bag-level start). For unplanted sites in partial mode
            # the "guide_span" is still meaningful (it points to the bag's
            # shared guide window on nc); the label is_planted flags what
            # actually happened on the flank.
            site_nc_start = (self.per_site_nc_start[s.site_idx]
                              if self.per_site_nc_start
                              else self.planted_start_on_nc)
            is_planted = (self.per_site_is_planted[s.site_idx]
                          if self.per_site_is_planted else True)
            # Substitute site's mutated active-slot nc into the emitted
            # noncoding_regions when v6 canonical is on. Inactive slots
            # still carry the bag-level random ncs (their content doesn't
            # affect Channel A; leakage guarded by matched distribution).
            if has_canonical and self.per_site_site_nc:
                site_ncs = list(self.ncrna_sequences)
                site_ncs[self.active_nc_index] = self.per_site_site_nc[s.site_idx]
            else:
                site_ncs = list(self.ncrna_sequences)
            # Per-site perfect guide: for scattered/twin the "perfect guide"
            # is the canonical bytes at THIS site's nc_start, not the
            # bag-level guide_sequence (which is p_true's guide for
            # twin, undefined for scattered).
            if self.negative_mode in ("scattered", "twin"):
                p_i = self.per_site_nc_start[s.site_idx]
                if 0 <= p_i and p_i + self.difficulty.L <= len(self.canonical_nc):
                    perfect_guide = self.canonical_nc[p_i : p_i + self.difficulty.L]
                else:
                    perfect_guide = ""
            else:
                perfect_guide = self.guide_sequence
            labels = {
                "is_positive": is_positive_bag,
                "negative_mode": self.negative_mode,
                "is_planted":    is_planted,
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
                "perfect_guide_dna": perfect_guide,
                "guide_length":   self.difficulty.L,
                "n_mismatches":   s.n_mismatches,
                "mismatch_positions": s.mismatch_positions,
                "active_noncoding_index": self.active_nc_index,
                "num_noncoding_regions": len(self.ncrna_sequences),
                "guide_span_in_active_noncoding": [
                    site_nc_start,
                    site_nc_start + self.difficulty.L,
                ],
                "ncrna_length":   len(site_ncs[self.active_nc_index]),
                "arch":           arch_site,
                "all_matching_positions_on_nc": s.all_matching_positions_on_nc,
                "competitor_count_at_site_planted_m": s.competitor_count_at_site_planted_m,
                # A4-comparable rate at fixed m>=8 threshold, computed in
                # the CANONICAL nc coordinate space (all_matching_positions_on_nc
                # is measured on active_nc == canonical_nc in build_bag).
                # Denominator = canonical_nc_len - L + 1.
                # Explicitly named to prevent the "same word, two quantities"
                # error that produced the 0.245 vs 0.229 confusion
                # (2026-09-03): "rate" pre-rename could mean either
                # competitor_count_at_site_planted_m/n_pos (per-site variable
                # threshold, 86/10/4 mix) OR len(all_matching_positions_on_nc)
                # /n_pos (fixed m>=8, the A4 anchor). Use rate_at_m8_fixed
                # for A4 comparisons; use competitor_count_at_site_planted_m
                # for Test 1 semantics. Under homology<1.0 the site nc has
                # ~5% substitutions vs canonical, so a site-space rate would
                # differ slightly (bounded by n_mm/n_pos). Keeping this in
                # canonical space matches A4's baseline exactly.
                "rate_at_m8_fixed": (len(s.all_matching_positions_on_nc)
                                       / max(1, (len(self.canonical_nc) if self.canonical_nc
                                                    else len(site_ncs[self.active_nc_index]))
                                                    - self.difficulty.L + 1)),
                "m_at_planted":   s.m_at_planted,
                "nc_channels":    nc_channels_per_bag,
            }
            if has_canonical and output_format == "v6r2":
                labels["canonical_nc"] = self.canonical_nc
                labels["canonical_fold"] = self.canonical_fold
                labels["site_nc_sequence"] = self.per_site_site_nc[s.site_idx]
                labels["site_to_canonical_map"] = list(
                    self.per_site_site_to_canonical_map[s.site_idx])
                # Oracle map + ε_align: diagnostic-only fields for ε_align
                # stratification during Channel B eval. NEVER an input.
                if s.site_idx < len(self.per_site_oracle_map):
                    labels["oracle_map"] = list(
                        self.per_site_oracle_map[s.site_idx])
                if s.site_idx < len(self.per_site_epsilon_align):
                    labels["epsilon_align"] = float(
                        self.per_site_epsilon_align[s.site_idx])
            # v7 output: skip canonical/site_to_canonical/oracle/epsilon fields
            # entirely per V7_SPEC §1.1 (removed from INPUT_TENSOR_LABEL_WHITELIST).
            # active_noncoding_index becomes 0 (only 1 region emitted).
            if output_format == "v7":
                labels["active_noncoding_index"] = 0
                labels["num_noncoding_regions"] = 1
                # Also emit ONLY the active region — v7 spec: nc_region_count = 1.
                site_ncs_emit = [site_ncs[self.active_nc_index]]
            elif output_format == "v6r2":
                site_ncs_emit = site_ncs
            else:
                raise ValueError(f"unknown output_format: {output_format!r}")
            rec = {
                "site_id": f"{self.bag_id}_site_{s.site_idx:04d}",
                "transposase_id": self.bag_id,
                "ncrna_id": f"{self.bag_id}_ncrna",
                "inputs": {
                    "flank": s.flank,
                    "noncoding_regions": site_ncs_emit,
                },
                "labels": labels,
            }
            # v7 generator_metadata per CANONICAL_BAG_SPEC §2.6 (3 required
            # fields) + flank_pool_source per V7_SPEC §2.2.
            if output_format == "v7":
                rec["generator_metadata"] = {
                    "data_source": f"v7_{self.negative_mode or 'positive'}",
                    "build_date": build_date,
                    "generator_version_or_commit": generator_version_or_commit,
                    "flank_pool_source": flank_pool_source,
                }
            recs.append(rec)
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
    plant_start: int | None = None,
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
    if plant_start is None:
        plant_start = rng.randint(0, max_start)
    else:
        plant_start = max(0, min(int(plant_start), max_start))
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


def _fake_plant_on_flank(
    flank: str,
    guide_A: str,
    guide_B: str,
    is_split: bool,
    split_gap: int,
    is_reversed: bool,
    rng: random.Random,
    plant_start: int,
) -> tuple[str, int, int, int, int]:
    """Insert a COMPOSITION-MATCHED SHUFFLE of the guide bases at
    `plant_start` on `flank`. Used for negative_mode='partial' un-planted
    sites and for `negative_mode='none'` all-sites. This makes the
    "insertion happened" signal (nucleotide + dinuc composition of the
    inserted 11 nt) IDENTICAL across positive and negative examples;
    only the ordering (and thus cross-site aggregatable signal) differs.

    Discovered 2026-09-02 via a flank-only shortcut probe: without
    fake-plant, a logistic regression on 22 flank-level features gave
    AUROC 0.55 (edited vs raw). Fix: match composition on unplanted
    sites by shuffling the guide.

    Returns (flank_final, plant_start, A_end_slot, B_start_slot, plant_end).
    No mismatches (we don't need to track them — this signal is
    orthogonal), no mutated_concat (nothing meaningful to store).
    """
    A_len = len(guide_A)
    B_len = len(guide_B)
    total_width = A_len + (split_gap if is_split else 0) + B_len

    # Shuffle the concatenated guide bases (composition-preserving)
    chars = list(guide_A + guide_B)
    rng.shuffle(chars)
    shuf_full = "".join(chars)
    shuf_A = shuf_full[:A_len]
    shuf_B = shuf_full[A_len:]

    max_start = len(flank) - total_width
    if max_start <= 0:
        raise ValueError(f"target width {total_width} > flank {len(flank)}")
    plant_start = max(0, min(int(plant_start), max_start))
    plant_end = plant_start + total_width

    if is_reversed:
        left_block = _reverse_complement(shuf_B)
        right_block = _reverse_complement(shuf_A)
        left_len, right_len = B_len, A_len
    else:
        left_block, right_block = shuf_A, shuf_B
        left_len, right_len = A_len, B_len

    A_end_slot = plant_start + left_len
    B_start_slot = A_end_slot + (split_gap if is_split else 0)

    flank_final = (
        flank[:plant_start]
        + left_block
        + flank[A_end_slot : B_start_slot]
        + right_block
        + flank[plant_end:]
    )
    assert len(flank_final) == len(flank), (len(flank_final), len(flank))
    return flank_final, plant_start, A_end_slot, B_start_slot, plant_end


def _mutate_preserving_guide(canonical: str, mutation_rate: float, rng: random.Random,
                                 guide_span: tuple[int, int],
                                 ) -> tuple[str, list[int]]:
    """Return (site_nc, site_to_canonical_map). Same mutation model as
    scripts/measure_align_error.py: per-position event probability =
    `mutation_rate`, with sub 0.7, ins 0.15, del 0.15 given an event.

    IMPORTANT — `mutation_rate` is the MUTATION probability, NOT the
    homology rate. When routing an `nc_homology_rate` arch value, the
    caller MUST pass `1.0 - nc_homology_rate`. An earlier caller passed
    the homology_rate directly and produced a corpus at ~5% homology
    instead of 95% — crashed A8a-1 coverage from ~0.44 to ~0.37 (see
    2026-09-03). Parameter renamed from `rate` to `mutation_rate` to
    make that mistake syntactically obvious.

    The `guide_span` canonical positions [gs, gs+L) are copied verbatim —
    no substitutions, no indels inside that range. Guarantees the
    planted-target coordinate system remains meaningful (m_at_planted_start
    == L survives homology).

    site_to_canonical_map[k] = canonical pos that maps to site pos k (or
    -1 if the site position is an inserted base). Length == len(site_nc).
    Bases FROM canonical, no ACGT filtering (canonical is ACGT-only).
    """
    if not 0.0 <= mutation_rate <= 1.0:
        raise ValueError(f"mutation_rate {mutation_rate} out of [0, 1]")
    gs, ge = guide_span
    site_chars: list[str] = []
    site_to_canonical: list[int] = []
    for i, c in enumerate(canonical):
        if gs <= i < ge:
            # Guide window: copy verbatim, no event
            site_to_canonical.append(i)
            site_chars.append(c)
            continue
        u = rng.random()
        if u >= mutation_rate:
            site_to_canonical.append(i)
            site_chars.append(c)
        else:
            u2 = rng.random()
            if u2 < 0.7:
                # Substitution
                site_to_canonical.append(i)
                site_chars.append(rng.choice([b for b in "ACGT" if b != c]))
            elif u2 < 0.85:
                # Insertion BEFORE this canonical position
                site_to_canonical.append(-1)          # inserted (site-only)
                site_chars.append(rng.choice("ACGT"))
                site_to_canonical.append(i)
                site_chars.append(c)
            else:
                # Deletion: canonical position has no site counterpart.
                pass
    return "".join(site_chars), site_to_canonical


_ALIGNER_SINGLETON = None


def _get_aligner():
    """Lazy Bio.Align.PairwiseAligner with the parameters locked in
    docs/generator_v6_handoff.md (match=+2, mismatch=-1, open=-2, ext=-1).
    Reused across sites to amortize aligner construction."""
    global _ALIGNER_SINGLETON
    if _ALIGNER_SINGLETON is None:
        from Bio.Align import PairwiseAligner
        aln = PairwiseAligner()
        aln.mode = "global"
        aln.match_score = 2
        aln.mismatch_score = -1
        aln.open_gap_score = -2
        aln.extend_gap_score = -1
        _ALIGNER_SINGLETON = aln
    return _ALIGNER_SINGLETON


def _pairwise_align_site_to_canonical(canonical: str, site: str
                                          ) -> list[int]:
    """Run pairwise global alignment and return site_to_canonical_map.

    Entry k = canonical pos that aligns to site pos k, or -1 if the site
    pos is an inserted base (gap in canonical). Length == len(site).

    Perf note: under homology=1.0 the caller should short-circuit BEFORE
    invoking this — canonical == site so the map is arange(len).
    """
    aligner = _get_aligner()
    aln = aligner.align(canonical, site)[0]
    a_can, a_site = str(aln[0]), str(aln[1])
    mapping = [-1] * len(site)
    can_i = 0
    site_i = 0
    for c_char, s_char in zip(a_can, a_site):
        if c_char == "-":
            # gap in canonical → this site pos is inserted
            mapping[site_i] = -1
            site_i += 1
        elif s_char == "-":
            # gap in site → canonical position with no site counterpart
            can_i += 1
        else:
            mapping[site_i] = can_i
            can_i += 1
            site_i += 1
    return mapping


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


# ---------------- flank sources ----------------

def load_flank_pool_from_is_sites() -> list[str]:
    """v6r2 legacy flank pool: real downstream 120bp flanks from the 5 DDE
    IS families in REAL_FLANK_POOL_FAMILIES. Kept for v6r2 reproducibility.
    v7 does NOT use this — see build_random_flank + V7_SPEC §2.1. Never
    delete; a v6r2 rerun depends on it byte-for-byte."""
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


# Backward-compat alias for v6r2 callers (run_generator.py, run_negatives.py,
# and any legacy scripts). Do not remove without an audit of every caller.
load_flank_pool = load_flank_pool_from_is_sites


def _gc_weighted_bases(rng: random.Random, gc: float, length: int) -> str:
    """ACGT string of given length with GC-weighted composition.
        p(A) = p(T) = (1 - gc) / 2
        p(G) = p(C) = gc / 2
    Note: NO special-case for gc==0.5 (unlike sample_ncrna in bag_v2.py:649).
    Uniform weights path and no-weights path can consume different RNG
    state on some Python versions, so we always take the weighted path for
    determinism across arbitrarily-close gc values."""
    if length == 0:
        return ""
    p_at = (1.0 - gc) / 2.0
    p_gc = gc / 2.0
    return "".join(rng.choices("ACGT",
                                  weights=[p_at, p_gc, p_gc, p_at],
                                  k=length))


def build_random_flank(rng: random.Random, gc: float,
                          length: int = DEFAULT_FLANK_LEN) -> str:
    """Synthetic random flank per V7_SPEC §2.1. GC-weighted convention.
    `gc` is BAG-LEVEL — the caller passes the same `gc` for every site in
    a bag. This function does NOT sample `gc`; do not add it to the
    signature."""
    return _gc_weighted_bases(rng, gc, length)


# ---------------- junction motif (V7 axis 8) ----------------

# Per V7_SPEC §2.6: 0 (no motif) 50%, each nonzero length 10%.
#
# 2026-09-13 v7-real refactor: RETIRED for the v7-real generation path.
# Weights collapsed to {0: 1.0} — every bag samples motif_length=0.
# Retirement rationale (spec §2.6 amendment):
#   - The axis was a statistical distractor (proxy for cross-site sequence
#     sharing near junction), not a TSD biology model. Its value is now
#     dominated by the harder real-flank + multi-region-nc difficulty
#     introduced in the v7-real refactor.
#   - Junction position needs to migrate from flank[0] (v7 legacy) to
#     flank[60] (v7-real, mid-flank). Maintaining a position-aware motif
#     planter isn't worth the cost given (a).
# Reversibility: restore the original weights list to reactivate the axis
# without any other code change. Sampler and planter functions are kept.
_JUNCTION_MOTIF_LENGTHS = [0]
_JUNCTION_MOTIF_WEIGHTS = [1.0]
# Original weights preserved for audit / potential reactivation:
# _JUNCTION_MOTIF_LENGTHS_LEGACY = [0,   4,   6,   8,   9,   10]
# _JUNCTION_MOTIF_WEIGHTS_LEGACY = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]


def sample_junction_motif(rng: random.Random) -> tuple[int, bool]:
    """Bag-level sample of (junction_motif_length, junction_motif_consistent).
    Called ONCE per bag. Per V7_SPEC §2.6."""
    length = rng.choices(_JUNCTION_MOTIF_LENGTHS,
                             weights=_JUNCTION_MOTIF_WEIGHTS, k=1)[0]
    consistent = rng.random() < 0.5
    return length, consistent


def build_motif_bases(rng: random.Random, gc: float, length: int) -> str:
    """Random ACGT motif string of given length, GC-weighted to match the
    bag's flank composition (so motif region isn't a composition outlier).
    Per V7_SPEC §2.6."""
    return _gc_weighted_bases(rng, gc, length)


def sample_site_orients(rng: random.Random, n_sites: int
                            ) -> tuple[list[str], float]:
    """v7 per-site orient sampler per V7_SPEC §2.4 (rev5 simplified).

    Returns (orients, p_same) where:
      - orients: list of length n_sites, each element in {'fwd', 'rc'}
      - p_same: the sampled per-site "follow bag orient" probability,
        drawn uniformly from [0.5, 1.0]. Recorded as PROV for stratification.

    Sampling: p_same ~ U(0.5, 1.0); one bag_orient ~ {fwd, rc}; each site
    independently takes bag_orient with prob p_same else the opposite.
    Range starts at 0.5 (not 0) because p_same < 0.5 is semantically
    equivalent to flipping bag_orient — no coverage loss.

    No numpy dep, single random.Random source, p_same directly interpretable
    (expected per-site same-as-bag-orient rate = p_same exactly)."""
    p_same = rng.uniform(0.5, 1.0)
    bag_orient = rng.choice(["fwd", "rc"])
    other = "rc" if bag_orient == "fwd" else "fwd"
    orients = [bag_orient if rng.random() < p_same else other
                   for _ in range(n_sites)]
    return orients, p_same


def sample_planted_m_uniform(rng: random.Random, L: int,
                                m_range: tuple[int, int] = (8, 11)) -> int:
    """v7 per-site planted_m sampler per V7_SPEC §2.5 (rev after Step 6b).

    Range NARROWED from U{5..11} to U{8..11} because the low end (5,6,7)
    is unreachable in practice: at L=11-14 the background m_max at any nc
    position saturates near 8 (100% of planted_m=5 sites had m_at_planted
    ≥ 8; verified 2026-09-11 Step 6). Low planted_m only dilutes gold
    signal (S = |{sites: m ≥ 8}| loses sites whose m sits below threshold
    by chance despite plant).

    The `m_range` param is kept for testing (call with (5, 11) to reproduce
    the pre-Step-6b sampler). Default is (8, 11).

    Absolute match count. Upper bound clamped to L; if `m_range[0] > L`
    this raises."""
    lo, hi = m_range
    if hi > L:
        raise AssertionError(
            f"sample_planted_m_uniform requires L >= m_range[1]={hi}, got L={L}. "
            f"If L range is being extended below {hi}, re-review V7_SPEC §2.5.")
    if lo > hi:
        raise AssertionError(f"m_range malformed: {m_range}")
    return rng.randint(lo, hi)


def plant_junction_motif(flank: str, motif: str) -> str:
    """Overwrite flank[0:len(motif)] with motif, verbatim. Preserves the
    flank's total length. Per V7_SPEC §2.6.1: motif is planted AS-IS in the
    recorded flank string, NO reverse-complement based on site orient."""
    if not motif:
        return flank
    if len(motif) > len(flank):
        raise ValueError(f"motif ({len(motif)}) longer than flank ({len(flank)})")
    return motif + flank[len(motif):]


# ---------------- top-level ----------------

def sample_ncrna(rng: random.Random, nc_len: int, gc: float = 0.5) -> str:
    """Stage 1f: canonical nc bases with GC target.
    p(A)=p(T)=(1-gc)/2, p(G)=p(C)=gc/2. At gc=0.5 the weighted path is
    RNG-byte-identical to the unweighted path (verified 2026-09-03 —
    Python random.choices with equal weights consumes the same
    RNG state as the None-weights branch), so pre-1f byte compat
    holds when gc_target axis is pinned to 0.5."""
    if gc == 0.5:
        return "".join(rng.choices("ACGT", k=nc_len))
    p_at = (1.0 - gc) / 2.0
    p_gc = gc / 2.0
    return "".join(rng.choices("ACGT",
                                  weights=[p_at, p_gc, p_gc, p_at],
                                  k=nc_len))


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


VALID_NEGATIVE_MODES = ("none", "scattered", "partial", "twin")


def build_bag(
    bag_id: str,
    rng: random.Random,
    flank_pool: list[str],
    rate_table: RateTable,
    n_sites: int | None = None,
    m_threshold_for_all_matching: int = 8,
    negative_mode: str = "none",
    nc_homology_rate: float | None = None,
    flank_offset_mode: str | None = None,
    accessibility_target_percentile: float | None = None,
    gc_target: float | None = None,
    v7_mode: bool = False,
) -> Bag | None:
    # v7 SINGLE-FLAG SEMANTIC (2026-09-10 rev): `v7_mode=True` activates ALL
    # v7 axes together — flank source (synthetic random via
    # build_random_flank), junction motif (§2.6 planted verbatim), planted_m
    # sampler (uniform U{5..min(11, L)} via sample_planted_m_uniform), per-site
    # orient (§2.4 via sample_site_orients), and v7 output format (drops
    # canonical_nc / canonical_fold / site_to_canonical_map, single nc
    # region, generator_metadata with 3 required fields). Half-modes (e.g.
    # v7 orient but legacy planted_m) are BANNED — no spec covers them.
    # `v7_mode=False` = v6r2 legacy semantic, byte-compatible.
    """`n_sites`: when None, drawn uniformly from DEFAULT_N_SITES_RANGE
    inside sample_difficulty (Stage 1c). When an int, PIN it (adds no
    extra RNG consumption vs pre-1c code — required for A8a byte
    reproduction where we want n_sites=5).

    `accessibility_target_percentile`: pin the μ used by
    sample_guide_placement. Pass 0.85 to preserve pre-1e behavior and
    A8a byte-identity. None samples uniformly in [0.4, 0.95]."""
    """Assemble one bag end-to-end.

    negative_mode:
      "none"       -- POSITIVE: all sites share the same guide (bag-shared
                      nc_start_on_active) and every site is planted.
      "scattered"  -- HARD NEGATIVE (per-site nc_start): each site samples
                      its own guide placement from the active nc, so 5 sites
                      have 5 different guides at 5 different nc positions.
                      Every site still receives a plant of ITS OWN guide;
                      cross-site NC-position coherence is destroyed.
      "partial"    -- HARD NEGATIVE (partial coherence): the bag samples a
                      single shared guide (like a positive), but only
                      n_planted ∈ {1, 2, 3} sites receive the plant. The
                      remaining sites keep their raw flank.

    Returns None if the θ-sampler failed on the active fold (extremely rare
    with the soft rule).
    """
    if negative_mode not in VALID_NEGATIVE_MODES:
        raise ValueError(f"negative_mode {negative_mode!r} not in {VALID_NEGATIVE_MODES}")

    # Stage 1f (2026-09-03): sample gc_target FIRST so target_m can be
    # conditioned on it. Under gc_target=0.5 pinned (or None with a
    # sample landing at 0.5), the conditioned lookup returns the same
    # numbers as the uniform-ACGT baseline — byte-identity preserved.
    # RNG consumption: when gc_target is pinned, sample_gc_target is NOT
    # called (no extra draw); when None, one uniform draw here shifts
    # every subsequent draw relative to pre-1f. Pinned-run byte-identity
    # is the intended contract.
    from scripts.generator_v5.architecture import sample_gc_target
    if gc_target is None:
        gc_for_diff = sample_gc_target(rng)
    else:
        gc_for_diff = float(gc_target)
    diff = sample_difficulty(rng, rate_table, n_sites_override=n_sites,
                                gc=gc_for_diff)
    arch = sample_architecture(rng, diff.L,
                                 nc_homology_rate_override=nc_homology_rate,
                                 flank_offset_mode_override=flank_offset_mode,
                                 accessibility_target_percentile_override=accessibility_target_percentile,
                                 gc_target_override=gc_for_diff)
    n_sites = diff.n_sites

    # ncRNA generation + fold. All same nc_len (from difficulty).
    # The active-slot nc IS the canonical_nc (bag-level reference). Per-
    # site mutations are applied later, preserving the guide window.
    # Stage 1f: nc bases sampled at arch.gc_target composition. All ncs
    # in a bag share the same gc target (inactive matching preserves the
    # matched-composition invariant _assert_inactive_distribution_matched
    # checks, since active and inactive are drawn from the same
    # distribution).
    ncs = [sample_ncrna(rng, diff.nc_len, gc=arch.gc_target)
             for _ in range(arch.n_nc)]
    feats = [compute_features_v2(nc, guide_length=diff.L) for nc in ncs]
    _assert_inactive_distribution_matched(ncs, arch.active_nc_index)
    canonical_nc = ncs[arch.active_nc_index]

    # Guide placement on active nc.
    # For "scattered", we sample a fresh placement per site inside the site
    # loop below. For "none" and "partial" the bag shares a single placement.
    active_nc = ncs[arch.active_nc_index]
    if negative_mode == "scattered":
        planted_start_on_nc = -1     # sentinel: no bag-level placement
        guide = ""
        guide_A = guide_B = ""
    else:
        placement = sample_guide_placement(feats[arch.active_nc_index], rng=rng,
                                             mu=arch.accessibility_target_percentile)
        if placement is None:
            return None
        planted_start_on_nc = placement.start
        guide = active_nc[planted_start_on_nc : planted_start_on_nc + diff.L]
        if len(guide) != diff.L:
            return None
        guide_A = guide[: arch.guide.A]
        guide_B = guide[arch.guide.A :]

    # v6 Stage 1d — twin negative sampling.
    # Same canonical_nc, same fold, same per-channel percentile MU as
    # positive; each site plants a guide at a DIFFERENT canonical
    # position p_i whose (p_ss, cooperativity) percentile matches the
    # positive's p_true within tol. A12's structure-only AUROC should
    # be ~= 0.50 under this construction. If matching fails at tol=0.30
    # the bag is dropped.
    #
    # SCOPE OF "SHARED WITH POSITIVE" (verified 2026-09-03 by same-seed
    # byte-identity test):
    #   BYTE-IDENTICAL (both bags at same seed): canonical_nc,
    #     canonical_fold, all 5 structure feature channels on active nc,
    #     positive p_true.
    #   DIVERGES:          site flanks (5 per bag). Reason: this twin-
    #     specific block consumes RNG via sample_percentile_matched_
    #     positions BEFORE fl_idx = rng.sample(...), so the positive
    #     and twin see different fl_idx draws.
    #
    # Consequences:
    #   ✓ Structure-only classifier (bag-level nc_channels + any feature
    #     of active nc): provably 0.5 AUROC — same bytes go in.
    #   ✗ Flank-conditional pairing (e.g. "the same 5 flanks with vs
    #     without the true guide"): does NOT hold. Any A12 variant that
    #     assumes shared flanks needs the twin-vs-positive corpus to be
    #     built by an outer pairing loop, not by same-seed alone.
    twin_positions: list[int] = []
    twin_tol_used: float = 0.0
    if negative_mode == "twin":
        if planted_start_on_nc < 0:
            return None
        from scripts.generator_v5.ncrna_sampler_v2 import (
            sample_percentile_matched_positions,
            MATCH_CHANNELS_DEFAULT,
            _channel_percentiles,
        )
        # Positive's per-channel percentiles at p_true.
        active_feats = feats[arch.active_nc_index]
        per_channel_targets = {}
        for ch in MATCH_CHANNELS_DEFAULT:
            pct = _channel_percentiles(active_feats, diff.L, ch)
            if pct.size == 0 or planted_start_on_nc >= pct.size:
                per_channel_targets[ch] = 0.5
            else:
                per_channel_targets[ch] = float(pct[planted_start_on_nc])
        # Widening tol retry until n_sites twins found or 0.30 cap.
        picks = None
        for tol_try in (0.05, 0.10, 0.15, 0.20, 0.30):
            picks = sample_percentile_matched_positions(
                active_feats,
                target_percentile=per_channel_targets["p_ss"],
                n_distinct=n_sites,
                exclude={planted_start_on_nc},
                tol=tol_try,
                rng=rng,
                match_channels=MATCH_CHANNELS_DEFAULT,
                per_channel_targets=per_channel_targets,
            )
            if picks is not None:
                twin_tol_used = float(tol_try)
                break
        if picks is None:
            return None
        twin_positions = list(picks)

    # Choose which sites are planted (partial: 1..n_sites-1 uniformly).
    # n_planted=4 is the deployment-relevant cell: it measures the S=5 rule's
    # discrimination boundary where one noise site sits alongside four planted
    # ones. Excluding n=4 makes the sharpness of the n=3→n=5 transition
    # unobservable, so it's included. n_planted=n_sites would be a positive.
    if negative_mode == "partial":
        n_planted = rng.randint(1, n_sites - 1)
        planted_indices = set(rng.sample(range(n_sites), n_planted))
    else:
        planted_indices = set(range(n_sites))    # all planted

    # PER-SITE planted_m (2026-08-31 fix): the SAME 86/10/4 tail sampled
    # INDEPENDENTLY per site around the bag's target_m. Bag-level shared m
    # made Channel A's S=5 conjunction trivial: 55% of bags had range=0,
    # 80% had all-5-hits by construction. Per-site draws restore the
    # probabilistic-hit premise Channel A operates on.
    # v7 mode (V7_SPEC §2.5): uniform U{5..min(11, L)} instead of tail-around-target_m.
    from scripts.generator_v5.difficulty import sample_planted_m
    if v7_mode:
        per_site_planted_m = [sample_planted_m_uniform(rng, diff.L) for _ in range(n_sites)]
    else:
        per_site_planted_m = [sample_planted_m(rng, diff.target_m) for _ in range(n_sites)]
    per_site_n_mismatches = [max(0, diff.L - m) for m in per_site_planted_m]

    # v7 axis 6 (V7_SPEC §2.4). Under `v7_mode=True`, sample
    # per-site orient list + p_same. Under `False` (v6r2 default), all
    # sites inherit arch.is_reversed_target (bit-exact to legacy).
    if v7_mode:
        _orients_str, orient_p_same_val = sample_site_orients(rng, n_sites)
        per_site_is_reversed_list = [(s == "rc") for s in _orients_str]
    else:
        per_site_is_reversed_list = [arch.is_reversed_target] * n_sites
        orient_p_same_val = 1.0

    # 5' stem-loop flags: dot-bracket MFE per nc. Also capture the
    # canonical (active-slot) dot-bracket structure — v6 canonical_fold.
    import RNA
    sl_flags: list[bool] = []
    canonical_fold = ""
    for i, nc in enumerate(ncs):
        fc = RNA.fold_compound(nc.replace("T", "U"))
        structure, _ = fc.mfe()
        sl_flags.append(check_5p_stem_loop(structure))
        if i == arch.active_nc_index:
            canonical_fold = structure

    # Sites
    sites: list[Site] = []
    per_site_nc_start: list[int] = []          # populated for scattered mode
    per_site_is_planted: list[bool] = []       # True for planted, False for partial-unplanted
    per_site_site_nc: list[str] = []
    per_site_site_to_canonical_map: list[list[int]] = []
    per_site_oracle_map: list[list[int]] = []
    per_site_epsilon_align: list[float] = []
    # v7 mode (V7_SPEC §2.1 + §2.6): replace flank pool with n_sites freshly-
    # sampled synthetic random flanks + per-bag junction motif planted at
    # flank[0:motif_len]. Legacy: consume the passed-in flank_pool.
    if v7_mode:
        motif_len, motif_consistent = sample_junction_motif(rng)
        if motif_consistent and motif_len > 0:
            _shared_motif = build_motif_bases(rng, arch.gc_target, motif_len)
        else:
            _shared_motif = None
        _v7_flanks = []
        for _s_idx in range(n_sites):
            fl = build_random_flank(rng, arch.gc_target)
            if motif_len > 0:
                if _shared_motif is not None:
                    site_motif = _shared_motif
                else:
                    site_motif = build_motif_bases(rng, arch.gc_target, motif_len)
                fl = plant_junction_motif(fl, site_motif)
            _v7_flanks.append(fl)
        # Under v7 the "pool" IS exactly the n_sites flanks we just built.
        flank_pool = _v7_flanks
        # Ordered use (no re-sample); RNG consumption diverges from v6r2
        # (v6r2 uses rng.sample) — intentional, v7 has no byte-compat contract.
        fl_idx = list(range(n_sites))
        _v7_junction_motif_length = int(motif_len)
        _v7_junction_motif_consistent = bool(motif_consistent)
    else:
        if len(flank_pool) < n_sites:
            raise RuntimeError(f"flank pool size {len(flank_pool)} < n_sites {n_sites}")
        fl_idx = rng.sample(range(len(flank_pool)), n_sites)
        _v7_junction_motif_length = 0
        _v7_junction_motif_consistent = False

    # Bag-shared flank offset (2026-09-02 — 2nd coherence axis). Under
    # arch.flank_offset_mode == "consistent" all 5 sites plant at
    # bag_flank_base + ±jitter. Under "inconsistent" each site draws a
    # fresh plant_start (V5 pre-2026-09-02 behavior; a distinct
    # negative-shape when combined with nc-coherent placements).
    #
    # IMPORTANT: `bag_flank_base = rng.randint(...)` MUST run only when
    # the arch is 'consistent'. Drawing it unconditionally shifts every
    # subsequent RNG state relative to the pre-flank-axis code, which
    # broke A8a byte-identity (0.4510 vs anchor 0.4864 at 5K after RNG
    # divergence). See 2026-09-03 diagnosis.
    total_width = arch.guide.A + (arch.split_gap if arch.is_split else 0) + arch.guide.B
    min_max_start = min(len(flank_pool[k]) - total_width for k in fl_idx)
    if min_max_start <= 0:
        raise RuntimeError(f"flank pool has flank too short for target width {total_width}")
    if arch.flank_offset_mode == "consistent":
        bag_flank_base = rng.randint(0, min_max_start)
    else:
        bag_flank_base = 0    # unused under inconsistent
    for i, k in enumerate(fl_idx):
        base_flank = flank_pool[k]
        site_planted_m = per_site_planted_m[i]
        site_n_mismatches = per_site_n_mismatches[i]
        is_planted = i in planted_indices

        # Determine THIS site's guide + nc_start.
        if negative_mode == "scattered":
            # Per-site placement + guide.
            placement_i = sample_guide_placement(feats[arch.active_nc_index], rng=rng,
                                             mu=arch.accessibility_target_percentile)
            if placement_i is None:
                # Placement failure on active nc — skip this bag.
                return None
            site_nc_start = placement_i.start
            site_guide = active_nc[site_nc_start : site_nc_start + diff.L]
            if len(site_guide) != diff.L:
                return None
            site_guide_A = site_guide[: arch.guide.A]
            site_guide_B = site_guide[arch.guide.A :]
        elif negative_mode == "twin":
            # Twin: each site uses its own accessibility-matched
            # canonical position, distinct from the positive's p_true
            # and distinct across sites.
            site_nc_start = twin_positions[i]
            site_guide = active_nc[site_nc_start : site_nc_start + diff.L]
            if len(site_guide) != diff.L:
                return None
            site_guide_A = site_guide[: arch.guide.A]
            site_guide_B = site_guide[arch.guide.A :]
        else:
            # Bag-shared guide (positive or partial).
            site_nc_start = planted_start_on_nc
            site_guide_A, site_guide_B = guide_A, guide_B
        per_site_nc_start.append(site_nc_start)
        per_site_is_planted.append(is_planted)

        # v6: per-site mutated active-slot nc (canonical + homology_rate).
        # Guide window on canonical is preserved (no substitutions/indels
        # inside [site_nc_start, site_nc_start+L)) so m_at_planted keeps
        # its meaning across homology levels. Under rate=1.0 we
        # short-circuit to identity to avoid the pairwise-alignment cost
        # and keep byte-identical semantics with pre-v6.
        if arch.nc_homology_rate >= 1.0 or site_nc_start < 0:
            # scattered's site_nc_start=-1 sentinel: no guide window
            # to preserve on canonical; treat as identity for now
            # (scattered's cross-site coherence is destroyed elsewhere).
            site_nc_seq = canonical_nc
            s2c_map = list(range(len(canonical_nc)))
            oracle_map = list(range(len(canonical_nc)))
        else:
            # nc_homology_rate = fraction of positions kept identical to
            # canonical. Mutation rate is 1 - homology. See _mutate_preserving_guide
            # docstring: an earlier version passed nc_homology_rate directly
            # as mutation_rate, producing a corpus at ~5% homology at
            # nc_homology_rate=0.95.
            mutation_rate = 1.0 - arch.nc_homology_rate
            site_nc_seq, oracle_map = _mutate_preserving_guide(
                canonical_nc, mutation_rate, rng,
                guide_span=(site_nc_start, site_nc_start + diff.L))
            s2c_map = _pairwise_align_site_to_canonical(canonical_nc, site_nc_seq)
        per_site_site_nc.append(site_nc_seq)
        per_site_site_to_canonical_map.append(s2c_map)
        per_site_oracle_map.append(oracle_map)
        # ε_align = position-wise disagreement rate between aligner and
        # oracle maps, normalized by site_nc length. Both maps have length
        # len(site_nc). A mismatch at position k means aligner assigned a
        # different canonical position than the mutation model's truth.
        # -1 vs -1 counts as agreement; any other divergence counts as
        # disagreement. Diagnostic-only field for ε_align stratification.
        if len(s2c_map) == len(oracle_map) and len(s2c_map) > 0:
            n_diff = sum(1 for a, b in zip(s2c_map, oracle_map) if a != b)
            per_site_epsilon_align.append(n_diff / len(s2c_map))
        else:
            per_site_epsilon_align.append(0.0)

        # Choose plant_start per the flank_offset_mode axis.
        if arch.flank_offset_mode == "consistent":
            site_plant_start = bag_flank_base + rng.randint(-arch.flank_jitter, arch.flank_jitter)
        else:
            site_plant_start = None    # let _plant_target_on_flank draw uniformly

        # v7: per-site orient. Under legacy (v7_mode=False)
        # this is bit-exact to `arch.is_reversed_target` since all list
        # elements equal it.
        site_is_reversed = per_site_is_reversed_list[i]
        if is_planted:
            (flank_final, A_start, A_end, B_start, B_end,
             mutated_concat, mm_pos) = _plant_target_on_flank(
                base_flank, site_guide_A, site_guide_B,
                arch.is_split, arch.split_gap,
                site_is_reversed, site_n_mismatches, rng,
                mm_concentration=arch.mm_concentration,
                mm_anchor=arch.mm_anchor,
                plant_start=site_plant_start,
            )
        else:
            # NEG-mode UN-PLANTED site: composition-matched fake-plant of a
            # shuffled guide at the same flank position the planted sites use.
            # See _fake_plant_on_flank docstring. Prevents the flank-level
            # composition shortcut Check 2 measured at AUROC 0.55.
            if site_plant_start is None:
                site_plant_start = rng.randint(0, len(base_flank) - total_width)
            (flank_final, A_start, A_end, B_start, B_end) = _fake_plant_on_flank(
                base_flank, site_guide_A, site_guide_B,
                arch.is_split, arch.split_gap,
                site_is_reversed, rng,
                plant_start=site_plant_start,
            )
            mutated_concat = ""
            mm_pos = []

        # A_start=plant_start; A_end=slot boundary (left block end);
        # B_start=slot boundary (right block start); B_end=plant_end.
        # For scattered mode m_arr is measured against THIS site's nc_start,
        # so m_at_planted here is the m at THIS site's guide window on THIS
        # site's flank — meaningful. For unplanted-in-partial, m_at_planted
        # is measured against the (bag-shared) guide window on the raw flank
        # — should be low (random chance).
        m_arr = _pos_m_max_L(active_nc, flank_final, diff.L)
        matching = [int(p) for p in np.where(m_arr >= m_threshold_for_all_matching)[0]]
        if site_nc_start < len(m_arr):
            m_at_planted = int(m_arr[site_nc_start])
        else:
            m_at_planted = 0
        # competitor_count_at_site_planted_m: positions with m_max >= this site's
        # planted_m. Test 1 semantics: per-site.
        competitor_count_at_site_planted_m = int((m_arr >= site_planted_m).sum())
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
            competitor_count_at_site_planted_m=competitor_count_at_site_planted_m,
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
        negative_mode=negative_mode,
        per_site_nc_start=per_site_nc_start,
        per_site_is_planted=per_site_is_planted,
        canonical_nc=canonical_nc,
        canonical_fold=canonical_fold,
        per_site_site_nc=per_site_site_nc,
        per_site_site_to_canonical_map=per_site_site_to_canonical_map,
        per_site_oracle_map=per_site_oracle_map,
        per_site_epsilon_align=per_site_epsilon_align,
        twin_tol_used=twin_tol_used,
        twin_positions=list(twin_positions),
        # v7 axis 6 (V7_SPEC §2.4). Only populated when v7_mode=True;
        # empty list under v6r2 default (all sites inherit arch.is_reversed_target
        # via per_site_is_reversed_list construction above).
        per_site_is_reversed=(list(per_site_is_reversed_list)
                                  if v7_mode else []),
        orient_p_same=(orient_p_same_val if v7_mode else 1.0),
        v7_mode=v7_mode,
        junction_motif_length=_v7_junction_motif_length,
        junction_motif_consistent=_v7_junction_motif_consistent,
    )
