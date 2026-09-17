"""v7-real generator — reversed flow with real bacterial flanks + multi-region nc.

Direction: real 120bp flank → read target region → minimally mutate flank
at target region to match a shared bag guide with planted_m — → synthesize
two nc regions (one contains the guide, one is pure noise) — → shuffle
noncoding_regions order; active_noncoding_index is GOLD.

Locked design (2026-09-13 user directive):
  - flank source: 50-genome pool via RealFlankPool (see real_flank_pool.py)
  - target center offset from junction (position 60): U[-40, +40], clipped
  - target_L: U{9..14} (v7 wide distribution, NOT tuned to Durrant)
  - planted_m: U{8..11} (v7 wide, NOT tuned to Durrant)
  - nc region len: U[80, 250] each (covers IS621's 193/107 with margin)
  - guide plant in nc: avoids each end by MAX_L bases (no N-spacer boundary
    interaction under the loader's concat_with_N_spacer path)
  - junction_motif: retired (weights collapsed to {0: 1.0} in bag_v2.py)

Negative modes:
  - twin:      same K flanks + same target rewriting toward G_A;
               nc_planted contains an INDEPENDENT G_B instead. Sites'
               targets do not match nc's guide.
  - partial:   n_planted ∈ {1..K-1} sites get target rewriting toward G;
               remaining sites keep real flank (random low match to G).
  - scattered: K independent guides G_1..G_K; each site's target rewritten
               to match its own G_i; nc_planted contains ALL K guides at
               different positions (avoiding edges, and each other).
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from scripts.generator_v5.bag_v2 import (
    sample_ncrna, sample_planted_m_uniform, sample_site_orients,
    _gc_weighted_bases, Bag,
)
from scripts.generator_v5.real_flank_pool import RealFlankPool


# ---- constants ----
FLANK_LEN = 120
JUNCTION_POS = 60          # target center reference point
TARGET_L_MIN = 9
TARGET_L_MAX = 14
CENTER_OFFSET_MIN = -40    # target center offset from junction (min)
CENTER_OFFSET_MAX =  40    # target center offset from junction (max)
NC_LEN_MIN = 100     # raised from 80 (2026-09-13): eliminates ~10bp systematic
NC_LEN_MAX = 250     # lift on `scattered` (which resamples nc_len when
                     # min_required=90 doesn't fit). U[100,250] still covers
                     # IS621's 107/193 with margin; kills the label proxy.
MULTI_REGION_SCORING = "concat_with_N_spacer"

# MAX_L for nc-edge avoidance AND between-guide gap — must match
# model/channel_b/constants.MAX_L so no search window can straddle
# either the concat N-spacer boundary or two consecutive planted
# guides in nc (would create spurious cross-guide matches).
try:
    from model.channel_b.constants import MAX_L as _MAX_L  # 12 under current
except Exception:
    _MAX_L = 12
_NC_EDGE_AVOID = _MAX_L
_MIN_GUIDE_GAP = _MAX_L      # MAX_L bases between two planted guides in nc

# scattered mode: fixed number of nc guides regardless of n_sites.
# Sites randomly matched to one of these — cross-site coherence at any
# single nc position is at most n_sites/N_SCATTERED_GUIDES, not zero.
# 3 chosen so nc requirement stays within U[100, 250] even at L=14:
#   min_required = 2*edge_avoid + 3*L + 2*min_gap
#                = 2*12 + 3*14 + 2*12 = 90 ≤ NC_LEN_MIN=100  → OK, no rejection needed
#   With NC_LEN_MIN=100 the scattered mode no longer skews the nc_len
#   distribution (was ~+10bp lift when NC_LEN_MIN=80).
N_SCATTERED_GUIDES = 3


VALID_V7_REAL_NEGATIVE_MODES = ("none", "twin", "partial", "scattered")


# ---- helpers ----

def sample_bag_guide(rng: random.Random, L: int, gc: float = 0.5) -> str:
    """Sample a canonical guide sequence (length L) for one bag.
    GC-weighted to match the bag's gc target."""
    return _gc_weighted_bases(rng, gc, L)


def sample_target_position(rng: random.Random, L: int,
                                flank_len: int = FLANK_LEN,
                                junction: int = JUNCTION_POS) -> int:
    """Sample the target START position in the flank.
    Target center offset ~ U[CENTER_OFFSET_MIN, CENTER_OFFSET_MAX] relative
    to junction; clipped so [start, start+L) fits in [0, flank_len).
    Returns the 0-indexed start position of the target L-mer in the flank.
    """
    center_off = rng.uniform(CENTER_OFFSET_MIN, CENTER_OFFSET_MAX)
    center = junction + center_off
    start = int(round(center - L / 2))
    start = max(0, min(flank_len - L, start))
    return start


def sample_target_L(rng: random.Random,
                        lo: int = TARGET_L_MIN, hi: int = TARGET_L_MAX) -> int:
    """v7-real target length ~ U{lo..hi}. Wide range (Durrant fixed L=11
    is one point inside this)."""
    return rng.randint(lo, hi)


def sample_nc_len(rng: random.Random) -> int:
    return rng.randint(NC_LEN_MIN, NC_LEN_MAX)


def mutate_target_to_match(rng: random.Random, target_seq: str, guide: str,
                                target_m: int) -> str:
    """Minimum-edit mutation of `target_seq` so that its Hamming-match count
    against `guide` equals `target_m`.

    Both sequences must have the same length L. Edits ONLY the positions
    needed to hit target_m: if natural matches < target_m, flip mismatched
    positions to match; if natural matches > target_m, flip matched
    positions to non-match. Chooses positions to flip UNIFORMLY at random.

    Returns the mutated target string (same length as target_seq).
    """
    L = len(target_seq)
    if len(guide) != L:
        raise ValueError(f"guide len {len(guide)} != target_seq len {L}")
    if not (0 <= target_m <= L):
        raise ValueError(f"target_m {target_m} out of [0, {L}]")

    match_positions    = [i for i in range(L) if target_seq[i] == guide[i]]
    mismatch_positions = [i for i in range(L) if target_seq[i] != guide[i]]
    n_natural = len(match_positions)

    out = list(target_seq)
    if n_natural == target_m:
        return target_seq
    elif n_natural < target_m:
        # Need to add (target_m - n_natural) matches.
        # Flip that many mismatched positions to match guide.
        need = target_m - n_natural
        to_flip = rng.sample(mismatch_positions, need)
        for p in to_flip:
            out[p] = guide[p]
    else:
        # Need to remove (n_natural - target_m) matches.
        # Flip that many matched positions to something ≠ guide[p].
        need = n_natural - target_m
        to_flip = rng.sample(match_positions, need)
        for p in to_flip:
            g = guide[p]
            # Pick a base different from g (equal weights over the other 3)
            others = [b for b in "ACGT" if b != g]
            out[p] = rng.choice(others)
    return "".join(out)


def plant_guide_in_nc(rng: random.Random, nc: str, guide: str,
                          edge_avoid: int = _NC_EDGE_AVOID) -> tuple[str, int]:
    """Insert `guide` at a random position in `nc`, avoiding both ends by
    `edge_avoid` bases. Overwrites nc at [pos:pos+L]; nc length preserved.

    Returns (new_nc, planted_start). Raises ValueError if nc is too short.
    """
    L = len(guide)
    lo = edge_avoid
    hi = len(nc) - edge_avoid - L
    if hi < lo:
        raise ValueError(
            f"nc too short (len={len(nc)}) for guide len={L} + "
            f"2×{edge_avoid} edge avoidance")
    pos = rng.randint(lo, hi)
    new_nc = nc[:pos] + guide + nc[pos + L:]
    return new_nc, pos


def plant_multiple_guides_in_nc(rng: random.Random, nc: str,
                                     guides: Sequence[str],
                                     edge_avoid: int = _NC_EDGE_AVOID,
                                     min_gap: int = 0) -> tuple[str, list[int]]:
    """Insert each guide at a random position in nc, non-overlapping, in
    the provided order (positions monotonically increase). Guides are
    separated by at least min_gap bases; both ends avoid `edge_avoid`.

    Uses random-composition placement (not rejection sampling) so the call
    always succeeds when the arithmetic fits — throws immediately if it
    doesn't. Slack (nc space not used by guides + gaps + edges) is
    distributed uniformly across (n_guides + 1) gap slots.

    Returns (new_nc, planted_starts_list).
    """
    L_total = sum(len(g) for g in guides) + max(0, len(guides) - 1) * min_gap
    slack = len(nc) - 2 * edge_avoid - L_total
    if slack < 0:
        raise ValueError(
            f"plant_multiple_guides_in_nc: nc too short — need at least "
            f"{2*edge_avoid + L_total} bp, got {len(nc)} "
            f"(edge_avoid={edge_avoid}, min_gap={min_gap}, "
            f"n_guides={len(guides)}, guide_lens={[len(g) for g in guides]}).")
    # Split slack across n_guides+1 slots uniformly (stars-and-bars via
    # sorted-breakpoints). n_slots = n_guides + 1 (before first, between
    # consecutive, after last).
    n_slots = len(guides) + 1
    if n_slots == 1:
        slots = [slack]
    else:
        breaks = sorted(rng.randint(0, slack) for _ in range(n_slots - 1))
        breaks = [0] + breaks + [slack]
        slots = [breaks[i + 1] - breaks[i] for i in range(n_slots)]

    new_nc_chars = list(nc)
    cursor = edge_avoid + slots[0]
    starts = []
    for i, g in enumerate(guides):
        L = len(g)
        for j, base in enumerate(g):
            new_nc_chars[cursor + j] = base
        starts.append(cursor)
        cursor += L + min_gap + slots[i + 1]
    return "".join(new_nc_chars), starts


def _rc(s: str) -> str:
    comp = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(comp[b] for b in reversed(s))


# ---- entry point ----

@dataclass
class V7RealBagRecord:
    """Per-bag output of build_bag_v7_real. Consumed by the JSONL emit path."""
    bag_id: str
    n_sites: int
    negative_mode: str
    gc: float
    # bag-shared guide (canonical, no mutations). For "scattered" this is
    # the guide of site 0; each other site has its own guide (see per_site).
    bag_guide: str
    bag_guide_L: int
    # nc regions (unshuffled reporting fields; the emit path handles shuffle)
    nc_planted: str            # region that contains bag_guide (or twin's G_B)
    nc_noise:   str
    active_index_in_output: int    # 0 or 1 in the emitted noncoding_regions
    nc_planted_positions: list[int]  # positions of the guide(s) inside nc_planted
    # per-site data (length n_sites)
    per_site_flank: list[str]
    per_site_target_start: list[int]
    per_site_target_L: list[int]
    per_site_planted_m: list[int]
    per_site_is_planted: list[bool]   # for partial; True unless un-planted
    per_site_guide: list[str]         # each site's guide (same as bag_guide except twin/scattered)
    per_site_nc_planted_pos: list[int | None]   # nc position of each site's guide;
                                                  # None if site is un-planted (partial)
    per_site_orient: list[str]
    orient_p_same: float


def build_bag_v7_real(bag_id: str, rng: random.Random,
                          real_flank_pool: RealFlankPool,
                          n_sites: int | None = None,
                          negative_mode: str = "none",
                          gc: float = 0.5,
                          at_max: float = 0.70) -> V7RealBagRecord | None:
    """Build one v7-real bag under the reversed-flow design.

    Args:
      bag_id: unique per bag
      rng: random source
      real_flank_pool: 50-genome pool sampler
      n_sites: if None, sampled uniformly from {3..8}
      negative_mode: one of VALID_V7_REAL_NEGATIVE_MODES
      gc: bag-level gc for synthetic nc + guide composition
      at_max: flank AT-content filter threshold

    Returns None only under an unrecoverable sampling failure.
    """
    if negative_mode not in VALID_V7_REAL_NEGATIVE_MODES:
        raise ValueError(f"negative_mode {negative_mode!r} not in {VALID_V7_REAL_NEGATIVE_MODES}")

    if n_sites is None:
        n_sites = rng.randint(3, 8)

    # Bag-level params
    bag_guide_L = sample_target_L(rng)
    bag_guide = sample_bag_guide(rng, bag_guide_L, gc=gc)

    # Sample K real flanks + per-site target positions + planted_m
    per_site_flank_raw = []
    per_site_target_start = []
    per_site_target_L = []
    per_site_planted_m = []
    for _ in range(n_sites):
        fl = real_flank_pool.sample_flank_120(rng, at_max=at_max)
        per_site_flank_raw.append(fl)
        target_start = sample_target_position(rng, bag_guide_L)
        per_site_target_start.append(target_start)
        per_site_target_L.append(bag_guide_L)  # all sites in bag use same L (=bag_guide_L)
        pm = sample_planted_m_uniform(rng, bag_guide_L, m_range=(8, min(11, bag_guide_L)))
        per_site_planted_m.append(pm)

    # Per-site orients
    orients, orient_p_same = sample_site_orients(rng, n_sites)

    # Per-site guide — three modes:
    #   none / partial: all sites share bag_guide (cross-site coherence)
    #   twin: each site uses its own independent guide → no shared target
    #         sequence across flanks; nc holds a DIFFERENT unrelated guide
    #         so per-site m at any nc position is background-level
    #   scattered: N_SCATTERED_GUIDES independent guides in nc; each site
    #         randomly assigned to one → partial cross-site coherence
    #         (up to n_sites/N sites at any single nc position)
    if negative_mode == "twin":
        # Each site has its own independent guide — breaks cross-site
        # coherence in the flank rewriting step
        per_site_guide = [sample_bag_guide(rng, bag_guide_L, gc=gc)
                             for _ in range(n_sites)]
    elif negative_mode == "scattered":
        # N_SCATTERED_GUIDES candidates; each site assigned to one uniformly
        scattered_candidates = [sample_bag_guide(rng, bag_guide_L, gc=gc)
                                    for _ in range(N_SCATTERED_GUIDES)]
        per_site_guide = [rng.choice(scattered_candidates) for _ in range(n_sites)]
    else:
        per_site_guide = [bag_guide] * n_sites

    # Determine which sites are planted (partial only; others = all planted)
    if negative_mode == "partial":
        n_planted = rng.randint(1, max(1, n_sites - 1))
        planted_indices = set(rng.sample(range(n_sites), n_planted))
        per_site_is_planted = [i in planted_indices for i in range(n_sites)]
    else:
        per_site_is_planted = [True] * n_sites

    # Rewrite flank at target region for planted sites
    per_site_flank_final = []
    for i in range(n_sites):
        fl = per_site_flank_raw[i]
        ts = per_site_target_start[i]
        L = per_site_target_L[i]
        target_seq = fl[ts:ts + L]
        if per_site_is_planted[i]:
            mut_target = mutate_target_to_match(
                rng, target_seq, per_site_guide[i], per_site_planted_m[i])
        else:
            mut_target = target_seq   # untouched real flank
        fl_final = fl[:ts] + mut_target + fl[ts + L:]
        assert len(fl_final) == FLANK_LEN
        per_site_flank_final.append(fl_final)

    # Synthesize nc regions.
    # For scattered mode: nc_planted must hold N_SCATTERED_GUIDES guides
    # spaced by _MIN_GUIDE_GAP with _NC_EDGE_AVOID on both ends. With
    # L≤14 and N=3 and gap=12 that's 2·12 + 3·14 + 2·12 = 90 bp min.
    # Rejection-resample nc_planted_len until it fits (usually one draw).
    if negative_mode == "scattered":
        min_required = (2 * _NC_EDGE_AVOID
                              + N_SCATTERED_GUIDES * bag_guide_L
                              + (N_SCATTERED_GUIDES - 1) * _MIN_GUIDE_GAP)
        for _ in range(20):
            nc_planted_len = sample_nc_len(rng)
            if nc_planted_len >= min_required:
                break
        else:
            # Extremely unlikely — min_required=90 usually, NC_LEN_MAX=250
            raise RuntimeError(
                f"[v7-real:scattered] {bag_id}: failed to sample nc_len ≥ "
                f"{min_required} in 20 tries. NC_LEN_MIN={NC_LEN_MIN} may be too low.")
    else:
        nc_planted_len = sample_nc_len(rng)
    nc_noise_len = sample_nc_len(rng)
    nc_planted_base = sample_ncrna(rng, nc_planted_len, gc=gc)
    nc_noise        = sample_ncrna(rng, nc_noise_len,   gc=gc)

    # Populate nc_planted with the correct guide(s) per negative_mode
    nc_planted_positions: list[int] = []
    if negative_mode == "scattered":
        # nc contains the N_SCATTERED_GUIDES candidates each site's flank
        # was rewritten toward; positions spaced by _MIN_GUIDE_GAP so
        # window-length search cannot straddle two guides.
        nc_planted, nc_planted_positions = plant_multiple_guides_in_nc(
            rng, nc_planted_base, scattered_candidates,
            edge_avoid=_NC_EDGE_AVOID, min_gap=_MIN_GUIDE_GAP)
    elif negative_mode == "twin":
        # Each site's flank was rewritten toward its OWN per_site_guide[i];
        # nc holds ONE unrelated guide → no nc position matches any site's
        # rewritten target region → per-site m at any nc pos is background.
        # For random-ACGT sampling, expected Hamming distance to any of
        # the K per-site guides is 3L/4 ≈ 8-10 — already "unrelated" without
        # explicit rejection sampling.
        g_unrelated = sample_bag_guide(rng, bag_guide_L, gc=gc)
        nc_planted, planted_pos = plant_guide_in_nc(
            rng, nc_planted_base, g_unrelated, edge_avoid=_NC_EDGE_AVOID)
        nc_planted_positions = [planted_pos]
    else:
        # positive OR partial: nc contains bag_guide (which every site was
        # rewritten toward). All sites' rewritten targets align at the
        # single nc position holding bag_guide → cross-site coherence.
        nc_planted, planted_pos = plant_guide_in_nc(
            rng, nc_planted_base, bag_guide, edge_avoid=_NC_EDGE_AVOID)
        nc_planted_positions = [planted_pos]

    # Build per-site nc_planted_pos mapping:
    #   none / partial: all planted sites → position 0 of nc_planted_positions
    #                   (the single bag_guide position); un-planted → None
    #   twin: sites' guides don't appear in nc → per_site_nc_planted_pos = None
    #         (marks "no site-matching position in nc")
    #   scattered: each site's guide is one of scattered_candidates; its
    #              nc position is nc_planted_positions[index_of_that_candidate]
    per_site_nc_planted_pos: list[int | None] = []
    if negative_mode == "none" or negative_mode == "partial":
        for i in range(n_sites):
            per_site_nc_planted_pos.append(
                nc_planted_positions[0] if per_site_is_planted[i] else None)
    elif negative_mode == "twin":
        per_site_nc_planted_pos = [None] * n_sites
    elif negative_mode == "scattered":
        # per_site_guide[i] is one of scattered_candidates — find which
        cand_to_pos = {id(g): pos for g, pos in
                          zip(scattered_candidates, nc_planted_positions)}
        for i in range(n_sites):
            per_site_nc_planted_pos.append(cand_to_pos[id(per_site_guide[i])])

    # Shuffle regions; record active_index (index of nc_planted after shuffle)
    order = [0, 1]
    rng.shuffle(order)
    active_index_in_output = order.index(0)   # 0 = nc_planted, 1 = nc_noise

    return V7RealBagRecord(
        bag_id=bag_id,
        n_sites=n_sites,
        negative_mode=negative_mode,
        gc=gc,
        bag_guide=bag_guide,
        bag_guide_L=bag_guide_L,
        nc_planted=nc_planted,
        nc_noise=nc_noise,
        active_index_in_output=active_index_in_output,
        nc_planted_positions=nc_planted_positions,
        per_site_flank=per_site_flank_final,
        per_site_target_start=per_site_target_start,
        per_site_target_L=per_site_target_L,
        per_site_planted_m=per_site_planted_m,
        per_site_is_planted=per_site_is_planted,
        per_site_guide=per_site_guide,
        per_site_nc_planted_pos=per_site_nc_planted_pos,
        per_site_orient=orients,
        orient_p_same=orient_p_same,
    )


def v7_real_to_jsonl_records(bag: V7RealBagRecord) -> list[dict]:
    """Emit v7-compatible JSONL records — one per site — from a V7RealBagRecord.
    Loader groups by transposase_id; all n_sites records share one.
    """
    # Compose noncoding_regions in the shuffled order recorded in bag
    if bag.active_index_in_output == 0:
        noncoding_regions = [bag.nc_planted, bag.nc_noise]
    else:
        noncoding_regions = [bag.nc_noise, bag.nc_planted]
    active_nc_index = bag.active_index_in_output

    is_positive = (bag.negative_mode == "none")

    # Precompute the concat-coordinate offset for the active region.
    # `_build_target` in model/channel_b/data.py reads
    # labels.guide_span_in_active_noncoding[0] as a position in the loader's
    # CONCATENATED coordinate space (post concat_with_N_spacer). For v7
    # single-region bags this equals the active-local position, but v7-real
    # is multi-region: if active_noncoding_index==1, positions must be
    # shifted by len(regions[0]) + spacer.
    _spacer_len = _MAX_L - 1
    if active_nc_index == 0:
        _concat_offset = 0
    else:
        # active is at index >= 1; each earlier region contributes its length
        # + one spacer.
        _concat_offset = sum(len(noncoding_regions[k]) + _spacer_len
                                    for k in range(active_nc_index))

    records = []
    for i in range(bag.n_sites):
        target_start = bag.per_site_target_start[i]
        target_L = bag.per_site_target_L[i]
        # guide_span_in_active_noncoding (CONCAT-coord) — training target
        # depends on this. None for un-planted (partial) and for twin
        # (whose per_site_nc_planted_pos is None by construction — nc holds
        # an unrelated guide, no site's own guide is in nc).
        _local_pos = bag.per_site_nc_planted_pos[i]
        if _local_pos is None:
            _guide_span = None
        else:
            _concat_pos = _local_pos + _concat_offset
            _guide_span = [_concat_pos, _concat_pos + target_L]
        rec = {
            "site_id":       f"{bag.bag_id}_site_{i:04d}",
            "transposase_id": bag.bag_id,
            "ncrna_id":       f"{bag.bag_id}_ncrna",
            "inputs": {
                "flank":              bag.per_site_flank[i],
                "noncoding_regions":  noncoding_regions,
            },
            "labels": {
                "is_positive":                 is_positive,
                "is_planted":                  bag.per_site_is_planted[i],
                "negative_mode":               bag.negative_mode,
                # Target-in-flank (GOLD — where guide was READ from)
                "target_position_in_flank":    [target_start, target_start + target_L],
                # planted_start = nc position of THIS site's guide in the
                # ACTIVE-region local coord (GOLD). None for un-planted
                # (partial) or when nc doesn't hold a guide this site's
                # flank matches (twin).
                "planted_start":               bag.per_site_nc_planted_pos[i],
                # guide_span_in_active_noncoding = [start, end] in CONCAT
                # coord (loader's post-concat_with_N_spacer coord). Read by
                # model/channel_b/data._build_target. None → this site
                # contributes 0 to y (correct semantics for un-planted).
                "guide_span_in_active_noncoding": _guide_span,
                "guide_length":                target_L,
                "planted_m":                   bag.per_site_planted_m[i],
                "m_at_planted":                bag.per_site_planted_m[i],
                "n_mismatches":                target_L - bag.per_site_planted_m[i],
                "num_noncoding_regions":       len(noncoding_regions),
                "active_noncoding_index":      active_nc_index,     # GOLD
                "ncrna_length":                sum(len(x) for x in noncoding_regions),
                "guide_dna":                   bag.per_site_guide[i],
                "arch": {
                    "n_sites":                     bag.n_sites,
                    "nc_multi_region_scoring":     MULTI_REGION_SCORING,
                    "nc_homology_rate":            1.0,
                    "gc_target":                   bag.gc,
                    "orient":                      bag.per_site_orient[i],
                    "orient_p_same":               bag.orient_p_same,
                    "is_reversed":                 bag.per_site_orient[i] == "rc",
                    "target_m_at_planted":         bag.per_site_planted_m[i],
                },
            },
            "generator_metadata": {
                "data_source":                  "v7_real",
                "build_date":                    "runtime",
                "generator_version_or_commit":   "v7_real_2026-09-13",
                "flank_pool_source":             "50-bacterial-genome pool (NCBI RefSeq)",
                "nc_multi_region_scoring":       MULTI_REGION_SCORING,
                "reversed_flow":                 True,
                # nc positions of ALL guides planted in nc_planted (in the
                # active noncoding region). Purely diagnostic — twin has
                # one unrelated guide; scattered has N_SCATTERED_GUIDES; none
                # and partial have one. `planted_start` (per-site) is None
                # for twin because no site's OWN guide is in nc; this
                # field lets diagnostics still locate the planted guide.
                "nc_planted_positions":          list(bag.nc_planted_positions),
            },
        }
        records.append(rec)
    return records
