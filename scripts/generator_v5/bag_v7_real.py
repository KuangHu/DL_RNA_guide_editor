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
import gzip
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from scripts.generator_v5.bag_v2 import (
    sample_ncrna, sample_planted_m_uniform, sample_site_orients,
    _gc_weighted_bases, Bag,
)
from scripts.generator_v5.real_flank_pool import RealFlankPool


# ---- V8.2 (2026-09-27) Rfam bacterial ncRNA pool ----
# Loaded lazily on first bracket draw. Downloaded gz FASTA per family
# cached on scratch. 8 bacterial families in the 100-250bp range so
# every bracket window (max = RNA_CONSERVED_LEN_MAX × 2 + TARGET_L_MAX
# = 35 + 14 + 35 = 84 bp) fits inside a family seed sequence.
# See finding_v81_rfam_scaffold_visibility for the source rationale.
_RFAM_CACHE_DIR = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                          "v81_rfam_cache")
_RFAM_FAMILIES = ("RF00013", "RF00050", "RF00080", "RF00114",
                    "RF00174", "RF00234", "RF00504", "RF01055")
_RFAM_URL_TEMPLATE = ("https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/"
                        "fasta_files/{fam}.fa.gz")
_RFAM_MIN_SEQ_LEN = 100
_RFAM_MAX_SEQ_LEN = 250
_RFAM_POOL: list[str] | None = None
_RFAM_POOL_BY_FAMILY: dict[str, list[str]] | None = None


def _load_rfam_pool() -> list[str]:
    """Load bacterial Rfam seed sequences from disk cache, downloading
    families that aren't cached yet. Filters to [MIN, MAX] length and
    ACGT-only (U→T). Returns FLAT list (audits still use this); the
    per-family index is populated in _RFAM_POOL_BY_FAMILY at the same
    time so `_sample_rfam_window` can do family-balanced sampling."""
    global _RFAM_POOL, _RFAM_POOL_BY_FAMILY
    if _RFAM_POOL is not None:
        return _RFAM_POOL
    _RFAM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    by_family: dict[str, list[str]] = {}
    pool: list[str] = []
    for fam in _RFAM_FAMILIES:
        p = _RFAM_CACHE_DIR / f"{fam}.fa.gz"
        if not p.exists():
            url = _RFAM_URL_TEMPLATE.format(fam=fam)
            with urllib.request.urlopen(url, timeout=60) as r:
                p.write_bytes(r.read())
        text = gzip.decompress(p.read_bytes()).decode("utf-8",
                                                            errors="replace")
        fam_seqs: list[str] = []
        header = None
        parts: list[str] = []

        def _maybe_keep(bases_parts: list[str]) -> None:
            s = ("".join(bases_parts).upper().replace("U", "T")
                    .replace(".", "").replace("-", ""))
            if (_RFAM_MIN_SEQ_LEN <= len(s) <= _RFAM_MAX_SEQ_LEN
                    and all(c in "ACGT" for c in s)):
                fam_seqs.append(s)

        for line in text.splitlines():
            if line.startswith(">"):
                if header is not None:
                    _maybe_keep(parts)
                header = line[1:]
                parts = []
            else:
                parts.append(line.strip())
        if header is not None:
            _maybe_keep(parts)

        by_family[fam] = fam_seqs
        pool.extend(fam_seqs)

    empty_families = [f for f, xs in by_family.items() if not xs]
    if empty_families:
        raise RuntimeError(
            f"Rfam pool empty for families {empty_families} — check "
            f"{_RFAM_CACHE_DIR} and family list {_RFAM_FAMILIES}")
    _RFAM_POOL = pool
    _RFAM_POOL_BY_FAMILY = by_family
    return pool


def _sample_rfam_window(rng: random.Random, target_len: int) -> str:
    """Return a contiguous target_len-bp window from a family-BALANCED
    draw of the bacterial Rfam pool.

    Two-step sampling: (1) pick family uniformly at random; (2) pick a
    sequence within that family that is ≥ target_len bp. Prevents the
    dominant family (RF00174, ~50% of the flat pool) from monopolizing
    training and biasing the model toward one family's sequence
    features. Consumes a bounded number of rng draws (~3 max).
    """
    _load_rfam_pool()   # populates _RFAM_POOL_BY_FAMILY
    fam = rng.choice(_RFAM_FAMILIES)
    fam_seqs = _RFAM_POOL_BY_FAMILY[fam]
    ok = [s for s in fam_seqs if len(s) >= target_len]
    if not ok:
        # Fallback: some families' longest seq may be < target_len.
        # Try the next family in a deterministic-rng way to keep
        # alignment stable. In practice at target_len ≤ 84, every family
        # here has sequences long enough (min family max_len is ~156).
        for alt in _RFAM_FAMILIES:
            if alt == fam:
                continue
            alt_seqs = _RFAM_POOL_BY_FAMILY[alt]
            ok = [s for s in alt_seqs if len(s) >= target_len]
            if ok:
                fam = alt
                break
        if not ok:
            raise RuntimeError(
                f"No Rfam family has a sequence ≥ {target_len} bp "
                f"(max seen across all families = "
                f"{max(max(len(s) for s in xs) for xs in _RFAM_POOL_BY_FAMILY.values())}).")
    seq = rng.choice(ok)
    start = rng.randint(0, len(seq) - target_len)
    return seq[start:start + target_len]


# ---- constants ----
FLANK_LEN = 120
JUNCTION_POS = 60          # target center reference point
TARGET_L_MIN = 9
TARGET_L_MAX = 14
CENTER_OFFSET_MIN = -15    # target center offset from junction (min).
CENTER_OFFSET_MAX =  15    # V8 (2026-09-23) narrowed from ±40 to ±15 to
                           # keep the L-mer + left/right conserved regions
                           # inside the 120-bp flank. Worst-case footprint:
                           # RNA_CONSERVED_LEN_MAX(=35) + TARGET_L_MAX(=14)
                           # + RNA_CONSERVED_LEN_MAX(=35) = 84 bp.
                           # ts ∈ [conserved_left, 120 - L - conserved_right].
                           # NB: this is a real reduction in inter-bag position
                           # variability (was U[-40,+40]); trade-off accepted
                           # to accommodate V8 conserved-region design.
TARGET_START_JITTER = 2    # v8: per-site jitter around bag-level target
                           # center, U{-JITTER..+JITTER}. See §8.4 scope
                           # table in V7_SPEC.md and FROZEN.md entry
                           # "V8 target_start scope fix" (2026-09-23).

# --- V8 RNA conserved regions (2026-09-23) ---
# Left + right conserved regions bracketing the guide, encoded into both
# nc and the site flanks. Represent "conserved sequence around the guide"
# — the RNA_conserved_region — real biology of most RNA-guided elements
# has such regions. Length per side drawn independently to avoid
# symmetry becoming a trivial feature.
RNA_CONSERVED_LEN_MIN = 15
RNA_CONSERVED_LEN_MAX = 35
CONSERVED_MATCH_FRAC_MIN = 0.55   # fraction of conserved-region positions
CONSERVED_MATCH_FRAC_MAX = 0.75   # matched between flank and bag conserved.
                                   # Deliberately looser than guide's m/L
                                   # (~0.73-0.92) because biology of
                                   # conserved regions is structural/
                                   # recognition, not direct base-pairing.

NC_LEN_MIN = 120     # V8: raised from 100 (2026-09-23) to fit conserved-
                     # region+guide footprint (max 84 bp) plus 2×edge_avoid
                     # (=24 bp) = 108 bp min; 120 gives 12 bp headroom.
NC_LEN_MAX = 250
MULTI_REGION_SCORING = "concat_with_N_spacer"

# V8.3 context-cluster mechanism RETIRED 2026-09-27.
# Flank-side clusters were removed under V8.4 Step 2 (no biological
# basis; V8.1 flank-scope invariant restored). Nc-side clusters had
# no signal path without their flank counterparts — a nc-only write
# doesn't affect m_max unless flank has the same 3bp template to
# match it — so retiring them too. See V8.4 Step 3 directive.

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


VALID_V7_REAL_NEGATIVE_MODES = ("none", "partial", "scattered",
                                 "flank_scattered",         # V8 mode 1
                                 "no_alignment",             # V8 mode 3
                                 "repeat_flank",             # V8 mode 4
                                 "tsd_negative")             # V8 mode 5
                                                             # (2026-09-25)
# V8.4 (2026-09-27): mode 2 "unstructured_nc_full" REMOVED. At 60 bp
# scale, dinuc-shuffled Rfam is not distinguishable from real Rfam via
# any of the 4 structure channels (ch4-7 Cohen's d < 0.05 across all
# channels, KS < 0.06). Without a channel to detect it, the mode was
# a noise negative under any construction. Retired in V8.4 rebuild.

# V8.1 invariant exemptions (2026-09-27). Modes that DELIBERATELY violate
# a bag-level structural invariant. Adding a new negative mode → default
# behavior is that ALL invariants apply, and if the new mode is meant to
# violate one it must be added here explicitly. Fails closed.
#
# _FLANK_SCOPE_EXEMPT: modes that write to flank OUTSIDE the target
# window [ts, ts+L). repeat_flank replicates site 0's flank to sites
# 1..K-1 wholesale with U[0, 0.05] mutation — this is its whole point.
_FLANK_SCOPE_EXEMPT = frozenset({"repeat_flank"})
#
# _TS_SPAN_EXEMPT: modes that intentionally scatter per_site_target_start
# across the flank instead of using a bag-shared center. flank_scattered
# is the only such mode.
_TS_SPAN_EXEMPT = frozenset({"flank_scattered"})
_TS_SPAN_MAX = 2 * TARGET_START_JITTER   # non-scattered bags: span ≤ 4


def _assert_flank_scope(negative_mode: str,
                          per_site_flank_raw: list[str],
                          per_site_flank_final: list[str],
                          per_site_target_start: list[int],
                          per_site_target_L: list[int]) -> None:
    """Invariant 1: for all modes not in _FLANK_SCOPE_EXEMPT, the final
    120-bp flank differs from the raw pool flank ONLY inside the target
    window [ts, ts+L). Any modification outside that window means a
    mode is silently reintroducing off-target flank rewriting (like the
    V8.0 flank-cons bug or the V8.3 flank-cluster experiment, both
    since retracted). Fails loudly at bag construction time.

    STRICT V8.1 semantics restored 2026-09-27 as part of the V8.4
    rebuild — the V8.3 widened signature (which accepted a
    context_clusters kwarg and allowed writes at cluster spans) is
    deleted. Under V8.4 flanks are 120-bp real bacterial DNA with ONLY
    the target region [ts, ts+L) mutated toward the bag guide."""
    if negative_mode in _FLANK_SCOPE_EXEMPT:
        return
    for i in range(len(per_site_flank_raw)):
        raw = per_site_flank_raw[i]
        final = per_site_flank_final[i]
        ts = per_site_target_start[i]
        L = per_site_target_L[i]
        outside_raw   = raw[:ts]   + raw[ts + L:]
        outside_final = final[:ts] + final[ts + L:]
        if outside_raw != outside_final:
            for j in range(len(outside_raw)):
                if outside_raw[j] != outside_final[j]:
                    global_j = j if j < ts else j + L
                    raise AssertionError(
                        f"[V8.4 flank-scope invariant] mode={negative_mode} "
                        f"site={i}: flank modified outside target window "
                        f"[{ts}, {ts + L}). First diff at pos {global_j}: "
                        f"raw[{global_j}]={raw[global_j]!r} "
                        f"final[{global_j}]={final[global_j]!r}. "
                        f"If this mode intentionally writes off-target, "
                        f"add it to _FLANK_SCOPE_EXEMPT.")
            raise AssertionError(
                f"[V8.4 flank-scope invariant] mode={negative_mode} "
                f"site={i}: outside-window difference detected but "
                f"position search failed.")


def _assert_ts_span(negative_mode: str,
                     per_site_target_start: list[int]) -> None:
    """Invariant 2: for all modes not in _TS_SPAN_EXEMPT, the per-bag
    target_start span must be ≤ 2 × TARGET_START_JITTER. This catches
    the pre-V8 scope-drift bug (per-site independent centers over ±40
    bp instead of a bag-shared center + small jitter)."""
    if negative_mode in _TS_SPAN_EXEMPT:
        return
    if len(per_site_target_start) < 2:
        return
    span = max(per_site_target_start) - min(per_site_target_start)
    if span > _TS_SPAN_MAX:
        raise AssertionError(
            f"[V8.1 ts-span invariant] mode={negative_mode}: "
            f"target_start span {span} > 2×JITTER {_TS_SPAN_MAX}. "
            f"per_site_target_start={per_site_target_start}. "
            f"If this mode intentionally scatters ts, add it to "
            f"_TS_SPAN_EXEMPT.")

# tsd_negative (V8 mode 5, 2026-09-25): DDE-mimicking negative that is
# ORTHOGONAL to repeat_flank. Per-site flanks are drawn INDEPENDENTLY
# from the real pool (so flank_bg_identity ≈ 0.26 — this negative
# won't be caught by the flank_bg channel). Each site's flank has a
# TSD (target site duplication) sequence inserted at the target_start
# region. nc plants a mock "insert body" of [TSD + random_middle + TSD]
# (mimics TIR-TIR annotation of real DDE elements). The bag-level TSD
# is drawn from one of 6 DDE-family TIR signatures (extracted from
# the 2026-09-25 DDE audit: IS1/IS3/IS6/IS66/IS256/ISL3 head sequences).
# K ~ U{7, 8, 9}. Model discrimination path: peak m at flank↔nc TSD
# match is ~K (7-9) not full L; peak neighbors don't show conserved-
# region template. If the model can't separate this from positive,
# adding a "peak_context_conservation" channel becomes justified.

# 6 DDE-family TIR head signatures, from scripts/audit_dde_tsd_visual.py
# (2026-09-25). First 9 bp of each family's insert body — the model-
# recognizable "family signature" that appears at every insertion of
# that element. Used as the TSD source pool for tsd_negative mode.
DDE_TSD_SIGNATURES: tuple[str, ...] = (
    "GGTAATGAC",   # IS1
    "ACTGTACTG",   # IS3
    "GGCACTGTT",   # IS6
    "GTAAGCGTA",   # IS66
    "GAGCCTGTA",   # IS256
    "GGGTCTTCC",   # ISL3
)
# repeat_flank (V8 mode 4, 2026-09-25) added after V8 DDE eval showed
# real transposon multi-copy elements (DDE families) exhibit near-
# identical flanks across sites — coherent flank_dev signal that fools
# the model into scoring them as positive (DDE p50 = 6.65 vs pos 5.02
# on v8_main/best.pt). This mode plants a designed guide in nc (same
# as positive) but reuses ONE base flank across all K sites, with
# per-site mutation rate U[0, 0.05] to cover the "identical to
# slightly-diverged copies" spectrum. Together with a planned
# `flank_bg_identity` input channel (per-bag cross-site pairwise flank
# similarity outside the alignment window), teaches the model to
# require additional evidence beyond mere flank_dev coherence.
# unstructured_nc_half was drafted but RETIRED 2026-09-23: (a) the extra
# bag-level rng.choice draw caused a 4bp nc_len drift vs other modes
# (label proxy risk); (b) unstructured_nc_full already covers the
# "no conserved context" negative direction; the half variant added
# an intermediate difficulty band that wasn't clearly worth its cost
# in this iteration. May revisit in a later V.
#
# twin was in v7-real but REMOVED from V8 (2026-09-23): the "each site
# has its own independent guide" negative was subsumed by scattered (which
# reduces cross-site guide coherence via a small candidate set) plus
# flank_scattered (which breaks position coherence). Twin implementation
# preserved at git tag `v7-real-frozen` for historical reproduction.


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
    # V8.1 diagnostic fields (2026-09-27) — expose bag-level Rfam
    # bracket templates so audits can verify nc content.
    left_conserved: str = ""
    right_conserved: str = ""
    left_conserved_len: int = 0
    right_conserved_len: int = 0


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

    # V8.4 (2026-09-27) — nc bracket regions come from ONE contiguous
    # window of a real bacterial Rfam ncRNA. Left + right lengths drawn
    # independently per bag; total window = left_len + L + right_len.
    # The middle L bp of the window are subsequently overwritten by the
    # planted guide, but left+right sides of the window are byte-identical
    # to positions [i, i+left_len) and [i+left_len+L, i+left_len+L+right_len)
    # of the same source Rfam sequence — so they retain the "these two
    # segments naturally sit next to each other in a real folded ncRNA"
    # property (with an L-bp gap where the guide lands). All modes
    # uniformly: no mode-specific bracket-source selection (V8.2's
    # dinuc-shuffle for unstructured_nc_full is retired along with the
    # mode itself). No cluster mechanism (V8.3 flank-side + nc-side
    # retired in Step 2 + Step 3 respectively).
    left_conserved_len = rng.randint(RNA_CONSERVED_LEN_MIN, RNA_CONSERVED_LEN_MAX)
    right_conserved_len = rng.randint(RNA_CONSERVED_LEN_MIN, RNA_CONSERVED_LEN_MAX)
    _bracket_window_len = left_conserved_len + bag_guide_L + right_conserved_len
    _rfam_bracket = _sample_rfam_window(rng, _bracket_window_len)
    left_conserved = _rfam_bracket[:left_conserved_len]
    right_conserved = _rfam_bracket[left_conserved_len + bag_guide_L:]

    # V8 (2026-09-23) — RNG-alignment fix: draw nc lengths + bases HERE,
    # BEFORE any mode-specific rng consumption (per-site guides, partial's
    # is_planted set, conserved-region rewriting). This eliminates the
    # ~7bp nc_len drift across modes observed in the initial V8 smoke.
    # nc_len bounds already accommodate the max planted-block footprint
    # under NC_LEN_MIN=120 (worst-case block = 2·edge + left_max + L_max +
    # right_max = 24 + 35 + 14 + 35 = 108), so no rejection sampling
    # needed even for scattered mode (min_required for scattered=90).
    nc_planted_len = sample_nc_len(rng)
    nc_noise_len   = sample_nc_len(rng)
    nc_planted_base = sample_ncrna(rng, nc_planted_len, gc=gc)
    nc_noise        = sample_ncrna(rng, nc_noise_len,   gc=gc)

    # Sample K real flanks + per-site target positions + planted_m
    # V8 target_start scope fix (2026-09-23): bag-shared target center
    # + per-site jitter U{-2..+2} → per-bag span ≤ 4. Applied uniformly
    # to all modes EXCEPT flank_scattered (V8 negative mode 1), which
    # deliberately restores the pre-V8 per-site independent sampling
    # to expose scattered flank alignment as a training negative.
    # bag_center_off is drawn unconditionally so rng consumption for
    # downstream draws (nc, planted_m, orient) is aligned across modes.
    # CENTER_OFFSET range narrowed to ±15 (V8) to fit conserved-region
    # footprint within 120 bp flank.
    bag_center_off = rng.uniform(CENTER_OFFSET_MIN, CENTER_OFFSET_MAX)
    per_site_flank_raw = []
    per_site_target_start = []
    per_site_target_L = []
    per_site_planted_m = []
    # target_start range must leave room for left_conserved (before ts) and
    # right_conserved (after ts+L) inside the 120 bp flank.
    ts_lo_bound = left_conserved_len
    ts_hi_bound = FLANK_LEN - bag_guide_L - right_conserved_len
    for _ in range(n_sites):
        fl = real_flank_pool.sample_flank_120(rng, at_max=at_max)
        per_site_flank_raw.append(fl)
        # V8 rng-alignment (2026-09-23 fix): both branches draw the SAME
        # 2 rng samples per site (uniform center_off + randint jitter) so
        # downstream planted_m sampling is byte-identical across all 6
        # modes. flank_scattered uses the per-site center_off; other
        # modes draw it but ignore it (use bag_center_off instead).
        per_site_center_off = rng.uniform(CENTER_OFFSET_MIN, CENTER_OFFSET_MAX)
        jitter = rng.randint(-TARGET_START_JITTER, TARGET_START_JITTER)
        if negative_mode == "flank_scattered":
            # V8 negative mode 1: per-site independent target center →
            # scattered target locations across sites in the bag.
            center = JUNCTION_POS + per_site_center_off + jitter
        else:
            # V8 default: bag-shared center + small per-site jitter → per-
            # bag target-start span ≤ 4 (coherent across sites).
            center = JUNCTION_POS + bag_center_off + jitter
        target_start = max(ts_lo_bound, min(ts_hi_bound,
                                            int(round(center - bag_guide_L / 2))))
        per_site_target_start.append(target_start)
        per_site_target_L.append(bag_guide_L)  # all sites in bag use same L (=bag_guide_L)
        pm = sample_planted_m_uniform(rng, bag_guide_L, m_range=(8, min(11, bag_guide_L)))
        per_site_planted_m.append(pm)

    # Per-site orients
    orients, orient_p_same = sample_site_orients(rng, n_sites)

    # Per-site guide:
    #   none / partial / flank_scattered / unstructured_nc_full:
    #         all sites share bag_guide (cross-site coherence)
    #   scattered: N_SCATTERED_GUIDES independent guides in nc; each site
    #         randomly assigned to one → partial cross-site coherence
    #         (up to n_sites/N sites at any single nc position)
    if negative_mode == "scattered":
        # N_SCATTERED_GUIDES candidates; each site assigned to one uniformly
        scattered_candidates = [sample_bag_guide(rng, bag_guide_L, gc=gc)
                                    for _ in range(N_SCATTERED_GUIDES)]
        per_site_guide = [rng.choice(scattered_candidates) for _ in range(n_sites)]
    else:
        per_site_guide = [bag_guide] * n_sites

    # Determine which sites are planted:
    #   partial: 1..n_sites-1 sites planted (random subset)
    #   no_alignment (V8 mode 3): NO sites planted — flank stays raw
    #   tsd_negative (V8 mode 5): NO guide-based plant; flank rewriting
    #     handled by a dedicated per-site TSD-insert block below.
    #   all others: all sites planted
    if negative_mode == "partial":
        # V8.4 (2026-09-27): match fraction must be ≤ 30% so partial
        # is unambiguously negative (no "good alignment by chance"
        # where enough sites plant to look positive). n_planted upper
        # bound = max(1, floor(0.3 * n_sites)):
        #   K=3: [1, 1]   K=4-5: [1, 1]   K=6: [1, 1]
        #   K=7: [1, 2]   K=8: [1, 2]
        # match fraction range 12.5%-33% (K=3 edge case is 33% since
        # 0.3 × 3 = 0.9 → 1 plant → 1/3 ≈ 33%; accepted for K=3).
        max_planted = max(1, int(0.3 * n_sites))
        n_planted = rng.randint(1, max_planted)
        planted_indices = set(rng.sample(range(n_sites), n_planted))
        per_site_is_planted = [i in planted_indices for i in range(n_sites)]
    elif negative_mode in ("no_alignment", "tsd_negative"):
        per_site_is_planted = [False] * n_sites
    else:
        per_site_is_planted = [True] * n_sites

    # V8.1 (2026-09-27): flank-side conserved-region rewriting REMOVED —
    # was a bug (Durrant WT genomic flanks have no synthetic 15-35bp
    # cons-template match, so v8_main_v3 suppressed real IS110 flanks
    # to score 0.24). Conserved regions now live on nc side ONLY (see
    # nc synthesis branch below). Flank is: [real bacterial DNA] with
    # only the target region [ts:ts+L] mutated to match bag_guide at
    # per_site_planted_m fidelity.
    # ts bounds (ts_lo_bound / ts_hi_bound) intentionally KEPT as-is so
    # the target_start distribution is byte-identical to V8.0 — cons
    # removal changes flank *content*, not target position.
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
        # V8.4 (2026-09-27) — flank-side context-cluster writes removed.
        # Under V8.4 the 120-bp flank is real bacterial DNA with ONLY
        # the target region mutated. Any bag-level cluster templates
        # apply on nc only (see nc-synthesis block below), not on flank.
        assert len(fl_final) == FLANK_LEN
        per_site_flank_final.append(fl_final)

    # V8 mode 4: repeat_flank — AFTER all per-site rewrites, copy site 0's
    # FINAL flank to sites 1..K-1 with per-site mutation rate U[0, 0.05].
    # This is done post-rewrite so the target-region and conserved-region
    # patterns are IDENTICAL across sites (the per-site variance from
    # per_site_planted_m + per-site match_frac + per-site position picks
    # would otherwise reduce cross-site flank similarity from ~1.0 to
    # ~0.7 — which is what the 2026-09-25 first smoke exposed). This
    # ordering makes the negative match its design intent: near-identical
    # final flanks (DDE-like), planted-nc plus conserved wrap on nc side
    # (positive-like) → discriminated only by cross-site flank identity.
    if negative_mode == "repeat_flank":
        base_final = per_site_flank_final[0]
        for i in range(1, n_sites):
            per_site_rate = rng.uniform(0.0, 0.05)
            n_mut = int(round(per_site_rate * len(base_final)))
            if n_mut > 0:
                positions = rng.sample(range(len(base_final)), n_mut)
                fl_list = list(base_final)
                for p in positions:
                    orig = fl_list[p]
                    fl_list[p] = rng.choice([b for b in "ACGT" if b != orig])
                per_site_flank_final[i] = "".join(fl_list)
            else:
                per_site_flank_final[i] = base_final

    # V8 mode 5: tsd_negative — INDEPENDENT per-site flanks (like positive)
    # but each has the bag-level TSD (a DDE-family TIR signature, length
    # K ~ U{7,8,9}) inserted at target_start. flank_bg_identity ≈ 0.26
    # (positive-like) so the new channel doesn't fire on this negative.
    # Nc-side plant handled in the nc-synthesis block below.
    # Bag-level draws (family + K + bag_tsd) also feed the nc plant.
    bag_tsd: str | None = None
    if negative_mode == "tsd_negative":
        _fam_idx = rng.randint(0, len(DDE_TSD_SIGNATURES) - 1)
        _K = rng.randint(7, 9)
        bag_tsd = DDE_TSD_SIGNATURES[_fam_idx][:_K]
        # Overwrite per_site_flank_final with independent raw flanks that
        # have the TSD inserted at target_start. V8.4 (2026-09-27): no
        # flank cluster application — TSD insert stays inside the
        # target window [ts, ts+K_i) ⊆ [ts, ts+L), so flank-scope
        # invariant is satisfied without any cluster carve-out.
        rewritten = []
        for i in range(n_sites):
            fl = per_site_flank_raw[i]
            ts = per_site_target_start[i]
            K_i = min(_K, FLANK_LEN - ts)
            fl_new = fl[:ts] + bag_tsd[:K_i] + fl[ts + K_i:]
            assert len(fl_new) == FLANK_LEN
            rewritten.append(fl_new)
        per_site_flank_final = rewritten

    # NC lengths + bases were drawn EARLIER (RNG-alignment fix, V8 2026-
    # 09-23). Verify the scattered mode's minimum length constraint is
    # satisfied by the current NC_LEN_MIN — this is a static check now,
    # not a rejection loop.
    if negative_mode == "scattered":
        _min_required_scatter = (2 * _NC_EDGE_AVOID
                                 + N_SCATTERED_GUIDES * bag_guide_L
                                 + (N_SCATTERED_GUIDES - 1) * _MIN_GUIDE_GAP)
        if nc_planted_len < _min_required_scatter:
            # With NC_LEN_MIN=120 and max scattered requirement=90 this
            # should never trigger. Kept as a hard fail-fast so any future
            # constant change gets caught here rather than silently.
            raise RuntimeError(
                f"[v7-real:scattered] {bag_id}: nc_planted_len={nc_planted_len} "
                f"< min_required={_min_required_scatter}. Either raise NC_LEN_MIN "
                f"or reduce N_SCATTERED_GUIDES / bag_guide_L.")

    # Populate nc_planted with the correct guide(s) per negative_mode.
    # V8 (2026-09-23): single-guide modes plant the full block
    #   [left_conserved + middle_guide + right_conserved]
    # into nc so the whole conserved-region + guide context appears
    # contiguously in nc. `nc_planted_positions` records the GUIDE's
    # position (not the block start), keeping it consistent with pre-V8
    # semantics of "where the guide sits in nc" for training-target y.
    nc_planted_positions: list[int] = []
    if negative_mode == "no_alignment":
        # V8 mode 3: pure random nc, nothing planted. bag_guide/left_conserved/
        # right_conserved were drawn (bag-level rng-alignment) but discarded.
        nc_planted = nc_planted_base
        # nc_planted_positions stays [] — no plant anywhere.
    elif negative_mode == "tsd_negative":
        # V8 mode 5: nc plants a mock DDE insert body = TSD + random_middle +
        # TSD. Middle length ~ U[10, 20]; base composition matches gc target.
        # This mimics the standard IS annotation ([TIR-INSERT_BODY-TIR]).
        # No bag_guide, no conserved-region wrapping (both discarded like
        # unstructured_nc_full).
        assert bag_tsd is not None
        _middle_len = rng.randint(10, 20)
        _middle = _gc_weighted_bases(rng, gc, _middle_len)
        _insert_body = bag_tsd + _middle + bag_tsd
        nc_planted, _block_start = plant_guide_in_nc(
            rng, nc_planted_base, _insert_body, edge_avoid=_NC_EDGE_AVOID)
        # nc_planted_positions kept empty → y = 0 for all sites (negative
        # label; the "insert body" is diagnostic only, not a training target).
    elif negative_mode == "scattered":
        # scattered keeps its 3-guide layout without conserved-region
        # wrapping (would exceed NC_LEN_MAX). Known asymmetry vs other
        # modes on nc composition — noted in spec §8.4.1.
        nc_planted, nc_planted_positions = plant_multiple_guides_in_nc(
            rng, nc_planted_base, scattered_candidates,
            edge_avoid=_NC_EDGE_AVOID, min_gap=_MIN_GUIDE_GAP)
    else:
        # V8.4 (2026-09-27): positive OR partial OR repeat_flank OR
        # flank_scattered — plant
        # `[left_rfam_bracket + bag_guide + right_rfam_bracket]` as ONE
        # contiguous block into the random nc_planted_base. Left+right
        # bracket bytes are BOTH slices of the SAME Rfam window with an
        # L-bp gap in the middle (where the guide goes), so together
        # with the guide they occupy the same physical contiguous span
        # they'd have in the source Rfam sequence.
        full_planted_seq = left_conserved + bag_guide + right_conserved
        nc_planted, block_start = plant_guide_in_nc(
            rng, nc_planted_base, full_planted_seq, edge_avoid=_NC_EDGE_AVOID)
        planted_pos = block_start + left_conserved_len   # guide's own pos
        nc_planted_positions = [planted_pos]

    # Build per-site nc_planted_pos mapping:
    #   none / partial: all planted sites → position 0 of nc_planted_positions
    #                   (the single bag_guide position); un-planted → None
    #   scattered: each site's guide is one of scattered_candidates; its
    #              nc position is nc_planted_positions[index_of_that_candidate]
    #   flank_scattered (V8 mode 1): bag_guide IS in nc at nc_planted_positions[0],
    #         but the training label is y=zeros so the loss teaches the model
    #         to reject scattered flank patterns despite the nc-side match.
    #         Semantic vs training-signal split is intentional; see
    #         V7_SPEC.md §8.6 (to be added).
    #   unstructured_nc_full: nc holds bag_guide but WITHOUT conserved-region
    #         wrapping → no cross-site conserved context around the guide;
    #         y=zeros so the model must reject.
    per_site_nc_planted_pos: list[int | None] = []
    if negative_mode == "none" or negative_mode == "partial":
        for i in range(n_sites):
            per_site_nc_planted_pos.append(
                nc_planted_positions[0] if per_site_is_planted[i] else None)
    elif negative_mode in ("flank_scattered",
                            "no_alignment", "repeat_flank", "tsd_negative"):
        # y = zeros — all these modes plant like positive on nc/flank but
        # are LABELED negative. Model must reject on some other feature
        # (position scatter, missing plant, repeated flank across sites,
        # or short-TSD-only plant).
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

    # V8.1 structural invariants (2026-09-27). These run on EVERY bag,
    # fail loudly. They are cheap (string compare + arithmetic).
    _assert_flank_scope(negative_mode, per_site_flank_raw,
                          per_site_flank_final,
                          per_site_target_start, per_site_target_L)
    _assert_ts_span(negative_mode, per_site_target_start)

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
        left_conserved=left_conserved,
        right_conserved=right_conserved,
        left_conserved_len=left_conserved_len,
        right_conserved_len=right_conserved_len,
    )


# V8.1 rng-alignment validator (2026-09-27). Cross-mode invariant:
# for a fixed seed, all bag-level draws (before mode-specific rng
# consumption diverges) must be byte-identical across all 8 modes.
# This catches the "rng consumption drift" class of bug that has bitten
# twice already (nc_len drift ~7bp before the V8 fix, and any future
# addition/removal of an rng call inside a mode-specific branch that
# is not counter-balanced across the other branches).
#
# Fields compared across modes for each seed:
#   ALWAYS (bag-level, drawn before any mode-specific branch):
#     bag_guide_L, bag_guide, left_conserved_len, right_conserved_len,
#     left_conserved, right_conserved, nc_planted length, nc_noise
#     length, first 32 chars of nc_planted (pre-plant), first 32 chars
#     of nc_noise.
#   NON-SCATTERED modes only (share bag_center_off):
#     per_site_target_start (list).
#
# NOT compared (mode-specific by construction):
#   per_site_flank_final (rewrite differs across modes), nc_planted
#   (plant content differs), per_site_planted_m (rng consumed after
#   per_site_flank_raw sampling, which itself consumes different rng),
#   orient list.

def _bag_signature_shared(bag: "V7RealBagRecord") -> tuple:
    """Bag-level draws that are made BEFORE any mode-specific rng
    branch (bag_guide, cons lens, nc lengths + bases, orient list).
    Under per-bag seeding, these must be byte-identical across every
    mode for a given per-bag seed.

    V8.2 note: left_conserved / right_conserved are NOT in the shared
    signature because their content depends on negative_mode
    (unstructured_nc_full → random alt, others → Rfam window).
    Byte-identical bracket LENGTHS still hold because they are drawn
    upstream; and the two SOURCE candidates (rfam_window,
    random_alt) are BOTH drawn in every mode with aligned rng —
    only the selection differs. See _assert_bracket_alignment for the
    dedicated check on those two candidates."""
    return (
        bag.bag_guide_L,
        bag.bag_guide,
        bag.left_conserved_len,
        bag.right_conserved_len,
        len(bag.nc_planted),
        len(bag.nc_noise),
        # per_site_orient is drawn from `sample_site_orients` which runs
        # BEFORE the scattered/partial/tsd_negative branches, so it must
        # align across all modes under per-bag seeding.
        tuple(bag.per_site_orient),
        bag.orient_p_same,
    )


def validate_rng_alignment(seed: int = 0, n_bags: int = 100,
                              real_flank_pool: "RealFlankPool | None" = None,
                              modes: tuple[str, ...] = VALID_V7_REAL_NEGATIVE_MODES,
                              ) -> None:
    """V8.1 Invariant 3 (2026-09-27) — cross-mode rng consumption alignment.

    Uses PER-BAG seeding, exactly matching run_generator.py's contract:
    ``master_rng = Random(seed); per_bag_seeds = [master_rng.randrange(...) for _ in range(n_bags)]``.
    For each per-bag seed, generates one bag under EVERY mode and asserts
    that bag-level draws (before any mode-specific branch diverges) are
    byte-identical across every mode. Non-scattered modes additionally
    must produce identical `per_site_target_start` (bag-shared center +
    integer jitter is a pure function of the bag-shared rng state).

    A byte-mismatch on _bag_signature_shared means some mode-specific
    branch consumed rng BEFORE the shared draws it should have preceded
    (bug), or added a new branch that got its rng ordering wrong
    (regression). A per_site_target_start mismatch on two non-scattered
    modes means a mode-specific extra draw was inserted between
    bag_center_off and the ts loop.

    Raises AssertionError on the first misalignment. O(n_bags · |modes|)
    time; ~seconds at n_bags=100.
    """
    if real_flank_pool is None:
        real_flank_pool = RealFlankPool.load_default()

    master_rng = random.Random(seed)
    per_bag_seeds = [master_rng.randrange(0, 2**31 - 1)
                        for _ in range(n_bags)]
    reference_mode = modes[0]

    for i, s in enumerate(per_bag_seeds):
        ref_bag = build_bag_v7_real(
            bag_id=f"__rng_align_ref_{i:06d}",
            rng=random.Random(s), real_flank_pool=real_flank_pool,
            negative_mode=reference_mode,
        )
        if ref_bag is None:
            raise AssertionError(
                f"[V8.1 rng-alignment] reference mode={reference_mode} "
                f"bag={i}: build_bag_v7_real returned None.")
        rs = _bag_signature_shared(ref_bag)

        for mode in modes[1:]:
            other = build_bag_v7_real(
                bag_id=f"__rng_align_{mode}_{i:06d}",
                rng=random.Random(s), real_flank_pool=real_flank_pool,
                negative_mode=mode,
            )
            if other is None:
                raise AssertionError(
                    f"[V8.1 rng-alignment] mode={mode} bag={i}: "
                    f"build_bag_v7_real returned None.")
            os_ = _bag_signature_shared(other)
            if rs != os_:
                # Find first field that differs for a compact error.
                field_names = ("bag_guide_L", "bag_guide",
                                "left_conserved_len", "right_conserved_len",
                                "len(nc_planted)", "len(nc_noise)",
                                "per_site_orient", "orient_p_same")
                diffs = [(n, a, b) for n, a, b in zip(field_names, rs, os_)
                              if a != b]
                raise AssertionError(
                    f"[V8.1 rng-alignment] bag-shared signature mismatch: "
                    f"seed={seed} per_bag_seed={s} bag={i} "
                    f"ref_mode={reference_mode} other_mode={mode}\n"
                    f"  first mismatched fields: {diffs[:3]}")
            if (reference_mode not in _TS_SPAN_EXEMPT
                    and mode not in _TS_SPAN_EXEMPT):
                if ref_bag.per_site_target_start != other.per_site_target_start:
                    raise AssertionError(
                        f"[V8.1 rng-alignment] per_site_target_start "
                        f"mismatch: seed={seed} per_bag_seed={s} bag={i} "
                        f"ref_mode={reference_mode} other_mode={mode}\n"
                        f"  reference: {ref_bag.per_site_target_start}\n"
                        f"  other:     {other.per_site_target_start}")
            # V8.4 bracket alignment: EVERY mode now uses the same Rfam
            # window (unstructured_nc_full and its shuffle branch are
            # retired), so `left_conserved` and `right_conserved` must
            # be byte-identical across every mode for a given per-bag
            # seed.
            if (ref_bag.left_conserved != other.left_conserved
                    or ref_bag.right_conserved != other.right_conserved):
                raise AssertionError(
                    f"[V8.4 bracket-alignment] Rfam bracket content "
                    f"mismatch across modes: "
                    f"seed={seed} per_bag_seed={s} bag={i} "
                    f"ref_mode={reference_mode} other_mode={mode}\n"
                    f"  reference L: {ref_bag.left_conserved[:24]!r}\n"
                    f"  other     L: {other.left_conserved[:24]!r}")


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
                "generator_version_or_commit":   "v7_real_2026-09-23_v8",
                "flank_pool_source":             "50-bacterial-genome pool (NCBI RefSeq)",
                "nc_multi_region_scoring":       MULTI_REGION_SCORING,
                "reversed_flow":                 True,
                # V8 (2026-09-23) — coordinate convention serialized WITH
                # the record. Previously the loader derived spacer_len from
                # its own MAX_L constant, and when MAX_L changed (v7-real
                # 12 → V8 14) old corpora ended up with a 2-bp shift in
                # `guide_span_in_active_noncoding` at active_index=1.
                # Loader MUST read these; do not derive from load-time
                # MAX_L. See FROZEN "coord convention serialized with
                # corpus" rule.
                "concat_spacer_len":             _MAX_L - 1,
                "max_l_at_generation":           _MAX_L,
                # nc positions of ALL guides planted in nc_planted (in the
                # active noncoding region). Purely diagnostic —
                # scattered has N_SCATTERED_GUIDES; none / partial have one;
                # flank_scattered / unstructured_nc_full / no_alignment
                # per_site_nc_planted_pos is None so no site-level y placed.
                "nc_planted_positions":          list(bag.nc_planted_positions),
            },
        }
        records.append(rec)
    return records
