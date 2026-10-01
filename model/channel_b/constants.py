"""Channel B constants — locked before writing data.py or model.py."""
from __future__ import annotations

# --- L axis (guide-window sizes) --------------------------------------
# V8 (2026-09-23): widened to L=9..14 to match V8 generator's target_L
# U{9..14}. Previous v7-real range was L=9..12 (frozen at
# `v7-real-frozen`). Widening triggers: N_CHANNELS 15→19 (add
# m_max_L13/L14 + flank_dev_L13/L14), _MULTI_REGION_SPACER_LEN 11→13
# (in data.py, computed as MAX_L-1). Structure window in data.py
# also moves from L=12 to L=14 as a follow-on.

Ls: tuple[int, ...] = (9, 10, 11, 12, 13, 14)
MAX_L: int = max(Ls)

# --- site axis --------------------------------------------------------
# n_sites at generation time is in DEFAULT_N_SITES_RANGE (arch.py) —
# currently [3, 8]. Pad to MAX_N_SITES; site_mask carries validity.

MAX_N_SITES: int = 8

# --- input channels (15 total per (site, position) cell) --------------
# Deploy-computable, per Channel B spec §1 and finding_train_deploy_gap_fields.
# Any change here requires:
#   1. an update to the whitelist below
#   2. a fresh (gc × nc_len) leakage audit of the added feature
#   3. re-running C6 LDA to confirm the channel doesn't inflate structure leakage

CHANNELS: tuple[str, ...] = (
    # 6 per-site per-position m_max channels, one per L (V8: L=9..14)
    "m_max_L9", "m_max_L10", "m_max_L11", "m_max_L12", "m_max_L13", "m_max_L14",
    # 4 per-position structure channels — broadcast across sites in a bag
    "dG_open_uL_pn", "H_pair_win", "cooperativity_win_pn", "E_span_win",
    # 1 per-position structure validity mask (0 where E_span/H_pair NaN)
    "structure_valid",
    # 6 per-site per-position flank_argmax channels — relative to bag-median
    # (centred coherence signal, one per L)
    "flank_dev_L9", "flank_dev_L10", "flank_dev_L11", "flank_dev_L12",
    "flank_dev_L13", "flank_dev_L14",
    # 2 per-site orientation one-hots — LEGACY ZERO SLOTS (2026-09-10 rev).
    # Retained for architectural continuity; always zero at load time.
    "orient_fwd", "orient_rc",
    # 1 bag-level scalar broadcast to all (site, position) cells:
    #   flank_bg_identity = mean pairwise Hamming similarity across sites'
    #   flanks, computed on flank[:FLANK_BG_EXCL_LO] + flank[FLANK_BG_EXCL_HI:]
    #   (excludes the target-mutation + conserved-region rewrite envelope).
    # V8 mode 4 (repeat_flank) exercises this at ~0.96; positive at ~0.26.
    # See FROZEN "V8 flank_bg_identity channel" for rationale + window
    # dependency on CENTER_OFFSET + RNA_CONSERVED_LEN_MAX.
    "flank_bg_identity",
)
assert len(CHANNELS) == 20, f"V8 channel count changed: {len(CHANNELS)}"
N_CHANNELS: int = len(CHANNELS)

# --- flank_bg_identity window ------------------------------------------
# Fixed exclusion window covering the target-mutation + conserved-region
# rewrite envelope. Derivation:
#   target_start ∈ [ts_lo_bound, ts_hi_bound] ≈ [1..36, 71..96]
#     (from JUNCTION_POS=60 ± CENTER_OFFSET=15 ± TARGET_START_JITTER=2)
#   left_conserved_len ∈ U[15, RNA_CONSERVED_LEN_MAX=35]
#   right_conserved_len ∈ U[15, 35]
#   → rewrite span ≈ [ts − left_cons, ts + L + right_cons] ⊂ [~1, ~120]
#   typical rewrite envelope ⊂ [30, 90]
# Empirical validation (v8_bgchk job 26429414):
#   fixed [30,90] gives 0.258 (positive) vs 0.962 (repeat_flank) — clean.
#
# **Coupling warning:** if CENTER_OFFSET, RNA_CONSERVED_LEN_MAX,
# JUNCTION_POS, or TARGET_START_JITTER change in bag_v7_real.py, this
# window must move. Not derived automatically — declared here.
FLANK_BG_EXCL_LO: int = 30
FLANK_BG_EXCL_HI: int = 90


# --- label whitelist / blacklist --------------------------------------
# EVERY read of a `labels` field during input-tensor construction must
# come from INPUT_TENSOR_LABEL_WHITELIST. `data.py` runs a runtime assert
# via `_read_input_label`. Target construction (`_build_target`) uses a
# SEPARATE read path via `labels.get(...)` directly and is permitted to
# consume fields in TARGET_ONLY_LABEL_KEYS (below) — target consumption of
# ground truth is what supervised learning is.

# Fields the loader is permitted to read for MODEL INPUT tensor construction.
# These are all deploy-computable — nothing here depends on training-time
# oracle information.
INPUT_TENSOR_LABEL_WHITELIST: frozenset[str] = frozenset({
    # NOTE 2026-09-10: `canonical_nc`, `canonical_fold`, `site_to_canonical_map`
    # REMOVED. Loader now reads nc directly from `inputs.noncoding_regions[0]`
    # and recomputes the fold via ViennaRNA at load time. See V7_SPEC §1.1.
    "guide_length",             # L, at deploy passed to compute_features_v2
    "arch",                     # arch dict — subset of keys allowed, see ARCH_ALLOWED
})

# Fields the TARGET builder (`_build_target`) is permitted to read. These
# ARE GOLD; consuming them for the training target `y` is legitimate
# supervised learning. They MUST NOT appear in INPUT_TENSOR_LABEL_WHITELIST
# and MUST NOT be read via `_read_input_label`.
TARGET_ONLY_LABEL_KEYS: frozenset[str] = frozenset({
    "is_planted",                        # per-site planting bool
    "guide_span_in_active_noncoding",    # gold guide position in nc — that's the y
})

# Fields inside `arch` that are deploy-derivable and legal to read for
# INPUT construction. Note: `orient` was REMOVED 2026-09-10 — reading it
# to pick which shard-orient's m_max to consume was a GOLD leak. The
# loader now reads both fwd and rc shard arrays and combines them
# per-position (max over orient); no orient label is consumed.
ARCH_ALLOWED_KEYS: frozenset[str] = frozenset({
    "n_sites",                  # number of proposer sites, always known
    "flank_offset_mode",        # NOT a model input — kept for stratification only
    "nc_homology_rate",         # NOT a model input — stratification only
    "nc_multi_region_scoring",  # 2026-09-13 — multi-region policy declaration
                                # per CANONICAL_BAG_SPEC §3.2. Currently only
                                # "concat_with_N_spacer" is implemented; loader
                                # raises on any other value. v7 records don't
                                # set this (single-region); DDE adapter sets it.
})

# Fields that MUST NEVER enter the input tensor. Any read of these
# during model input construction is a training/deploy gap.
TRAIN_ONLY_LABEL_KEYS: frozenset[str] = frozenset({
    # Ground-truth planting information
    "planted_start", "m_at_planted", "mismatch_positions", "n_mismatches",
    "bag_target_m", "planted_A_end", "planted_B_end", "planted_B_start",
    "perfect_guide_dna",
    # Oracle alignment map (diagnostic only)
    "oracle_map", "epsilon_align",
    # Twin corpus construction
    "twin_positions", "twin_tol_used", "twin_p_true",
    # Composition targets that only exist as labels at training time
    "gc_target",
    # is_planted / is_positive are LABELS not inputs
    "is_planted", "is_positive",
    # Competitor bookkeeping (label-side statistic)
    "competitor_count_at_site_planted_m",
    "all_matching_positions_on_nc",
    # nc_channels blob (raw ViennaRNA outputs — we build our own subset)
    "nc_channels",
    # Diagnostic
    "site_nc_sequence",
    # Architecture axes that are training-time only
    "mm_concentration", "mm_anchor", "accessibility_target_percentile",
})

# --- flank_argmax normalization --------------------------------------
# Deviation-from-bag-median encoding; divide by this fixed scale to bring
# typical |dev| into O(1). Choice: 10.0 — flank_jitter_max in the
# generator is small integer nt (single digits), so 10× covers the tail.

FLANK_DEV_SCALE: float = 10.0


# --- per-channel fixed scales (applied at load time) -------------------
# BatchNorm rejected because it (a) breaks bit-exact site-permutation
# equivariance via batch-statistic coupling, (b) creates train/deploy
# running-stats divergence, and (c) can't distinguish real values from
# padding zeros. LayerNorm sidesteps (a) but still adds a learnable
# scale/shift the model can drift on. Fixed divisors keep the input
# distribution reproducible bit-for-bit at deploy, match the flank_dev
# principle, and let permutation-equivariance stay exact.
#
# Divisors chosen from the empirical range in one bag's dump (2026-09-04)
# with generous headroom. If a channel's range shifts materially in a
# corpus refresh, revisit — but do NOT let the divisor become a per-
# dataset stat file (that would recreate the train/deploy gap).

CHANNEL_SCALES: dict[str, float] = {
    # m_max / MAX_L: bounded values, [0, L] → [0, L/MAX_L]. Divisor uses
    # MAX_L (upper bound) so per-L channels stay on the same numeric
    # scale. V8 (2026-09-23): MAX_L 12→14 → divisor 12.0→14.0 for all
    # m_max channels; new m_max_L13/L14 use the same 14.0.
    "m_max_L9":  14.0, "m_max_L10": 14.0, "m_max_L11": 14.0,
    "m_max_L12": 14.0, "m_max_L13": 14.0, "m_max_L14": 14.0,
    # structure channels
    "dG_open_uL_pn":        0.2,   # kcal/mol; typical 0.05-0.5
    "H_pair_win":           1.0,   # nats; typical 0.3-0.7
    "cooperativity_win_pn": 0.3,   # unitless; typical 0.1-0.3
    "E_span_win":          20.0,   # nt; typical 15-20, spec §Item 4.5 marks this as the weakest channel
    "structure_valid":      1.0,   # binary
    # flank_dev — already divided by FLANK_DEV_SCALE inside data.py; no further scale
    "flank_dev_L9":  1.0, "flank_dev_L10": 1.0, "flank_dev_L11": 1.0,
    "flank_dev_L12": 1.0, "flank_dev_L13": 1.0, "flank_dev_L14": 1.0,
    # orient one-hots — already [0, 1]
    "orient_fwd": 1.0, "orient_rc": 1.0,
    # flank_bg_identity is already in [0, 1] by construction (mean Hamming
    # fraction); no additional scaling.
    "flank_bg_identity": 1.0,
}
assert set(CHANNEL_SCALES) == set(CHANNELS), (
    f"CHANNEL_SCALES keys mismatch CHANNELS: "
    f"missing={set(CHANNELS) - set(CHANNEL_SCALES)}, "
    f"extra={set(CHANNEL_SCALES) - set(CHANNELS)}"
)
