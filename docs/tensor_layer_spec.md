# Tensor layer spec (Module 4 rebuild)

Written 2026-09-02, closing Module 3's last piece. Follows the two v2
candidate-layer emission modes (position arrays for cross-site
aggregation, dedup'd candidate list for per-candidate feature attach) —
this spec describes what the LIST-side consumer (the tensor layer) emits.

## Consumer-first measurement (2026-09-02) — the patch does NOT justify itself

Before implementing, a discrimination measurement was run on V5 L=11 bags
where Channel A Mode 2 fires: gold = candidate at planted position;
competitor = top-m candidate at another nc_start on the same site. 5-fold
CV logistic regression. See `scripts/generator_v5/item3_structure_vs_scalar.py`.

  scalar-only (m, m/L, m − bag_median, m / (flank_len/L)):
      **AUROC 0.834 ± 0.018**
  scalar + 4 structure means (dG_open_uL_pn, H_pair_win,
      cooperativity_win_pn, E_span_win) with valid-fraction masks:
      **AUROC 0.863 ± 0.014**
  Δ AUROC = **+0.030** (marginal band, not the ≥ 0.05 patch-justification bar)

**Scope of this result** — narrowed 2026-09-02 to what was actually
measured:
- The task is 1-vs-1 per-candidate discrimination on Channel-A-detected
  L=11 bags. It is NOT ranking across the ~38 competitors per site that
  the real deployment sees; the real R@1 would be substantially lower
  than 0.834.
- The scalar baseline already includes cross-candidate normalization
  (`m − bag_median`, `m / (flank_len/L)`) — weak significance features
  in disguise. So the 0.834 baseline is not "naked m"; it already
  carries ~1/2 of what an E-value column would provide.
- Channel-A-detected bags are the EASIEST slice: 5-of-5 site coherence
  at planted position. This is the group Channel B doesn't need to
  handle. So Δ AUROC here understates what structure could contribute
  where the aggregator is under-informed (S=3 or 4). Item 3b below
  measures that regime directly.

**Consequence**: the spec's original design (22-channel × 144-nt patch,
multi-scale ±k·L masks) is DEFERRED. The tensor layer's first cut emits
a feature vector only, no patch tensor. Structure appears as 4 mean
scalars + 4 valid-fraction masks. The multi-scale patch geometry stays
in the spec below as a future upgrade path, gated on Channel B (or its
Item 3b consumer-view) demonstrating that patches beat means. Consumer-
first principle: don't build the tensor before the consumer that would
justify it.

## Item 3b (2026-09-02) — same measurement on the actual Channel B consumers

Item 3 tested on Channel A-detected bags — the wrong consumer. Channel B's
target is the S=3 or S=4 bags: 3 or 4 sites have m>=8 at planted, but
5-of-5 coherence fails and Channel A does not fire. Structure means may
contribute more where m is less decisive. See
`scripts/generator_v5/item3b_structure_on_partial_coherence.py`.

**Result** (V5 5K L=11 bags, S ∈ {3, 4}, n_rows=15046):
  scalar-only:          **AUROC 0.8293 ± 0.008**
  scalar + structure:   **AUROC 0.8599 ± 0.005**
  Δ AUROC = **+0.031** — **identical to Item 3's +0.030 on the S=5 easy slice**

Two independent measurements agree: structure lift is ~3pp on both the
easy (S=5) and Channel-B-target (S=3/4) slices. The multi-scale patch
geometry is NOT justified in either regime.

**S distribution over 5000 L=11 bags** (bonus signal):
| S  | bags | fraction |
|----|------|----------|
| 0  | 250  | 5% |
| 1  | 355  | 7% |
| 2  | 239  | 5% |
| 3  | 513  | 10% |
| 4  | 1496 | 30% |
| 5  | 2147 | 43% |

Channel B's addressable market is the **40% of bags in S=3/4** — Channel A
misses 57% total, but 17% (S≤2) is likely unrecoverable by single-molecule
per-site views. The remaining 40% is where Channel B needs to shine, and
structure gives it +3pp of lift over scalars — real but modest.

## Why the old (`preprocess/candidates.py::PATCH_CHANNELS`) is replaced

Three structural defects:

1. **`patch_width = 64` fixed.** D3 measured max pair span at 76–93% of
   molecule length; a 64-nt window centered on the guide truncates most
   long-range structural context. On a 200-nt NC that's covered fold, the
   patch sees at best 30% of the molecule; hemi-truncation depends on
   guide position, so the crop shape becomes a family-specific signature.

2. **`feats[2] = L` (absolute).** T-WT has L=11 uniformly; V4.2 has L=15
   for 84% of sites; families use L differently. An absolute L in the
   feature vector is a family label. `feats[2]` was frozen by the pair
   encoder in 48B and downstream; every fusion head has already seen it.
   Removing it here does not un-train the pair encoder, but it prevents
   the geometry branch from picking L back up.

3. **`feats[11] = nc_start_norm` (absolute nc coord).** T-WT's gold
   window sits at nc=49 uniformly (`range=[49,49]` per FROZEN.md). Any
   feature that carries `49/nc_len` for T-WT positives and something
   else for negatives is a perfect shortcut. This was one of the
   48C1a diagnostics that motivated the geometry-bypass audit.

## What replaces them

### Multi-scale patches (±k·L, k ∈ {0, 1, 2, 4})

The patch axis is centered on each candidate's `nc_start + L/2` (guide
center on the NC). Four nested windows are emitted:

  - k=0: the guide itself, width L
  - k=1: guide + ±L flank, width 3L
  - k=2: guide + ±2L flank, width 5L
  - k=4: guide + ±4L flank, width 9L

All four are stored inside ONE fixed-width patch of width
`W = 2 * 4 * L_max + L_max = 144` (with `L_max = 16`), positions outside
the ±4L window masked with `struct_valid=False`. Four scale-mask channels
`mask_k0..mask_k4` mark which positions belong to each scale.

This is L-relative by construction: the model sees the same effective
neighborhood (±1L, ±2L, ±4L) regardless of L, so a family-specific L
never becomes a family-specific patch shape. The absolute width W is a
storage-shape convenience, not a scale.

### Companion mask for E_span_win NaN

`features_structure_v2::E_span_win` returns NaN at windows with no
predicted pairing. The old tensor path silently converted these to 0,
which the model may have learned as "structure asserts unpaired". The
new tensor layer:

  - Fills NaN with 0 in the numeric channel
  - Emits a companion channel `E_span_valid` = `not isnan(E_span_win)`
  - Downstream loss should mask the E_span contribution where
    `E_span_valid = False`; a masked-mean over the guide window is the
    default reduction

This convention is fixed before the first training run to avoid the
silent-zero-gradient failure mode.

### Position-invariant molecule summary

Three global (per-molecule, per-Tnp) scalars, replicated across every
candidate emitted from that molecule:

  - `mol_total_ss_frac`: fraction of nc positions with `mean unpaired`
    probability >= 0.5 (from RNAplfold u_L / L). Molecule-level, position-
    invariant.
  - `mol_helix_count`: number of contiguous helix runs in the dot-bracket
    MFE structure. Molecule-level scalar.
  - `guide_window_accessibility_percentile`: percentile rank of the
    candidate window's mean `1 - dG_open_uL_pn` versus the whole molecule.
    This IS position-dependent within a molecule, but scale-invariant
    across molecules — the percentile is [0, 1] regardless of nc_len.

### Pool-relative L features

`L` as an absolute integer is dropped from `feats`. Replaced by:

  - `L_percentile_in_pool`: rank of this candidate's L among the pool's
    L distribution for this record, in [0, 1]. If the pool has L ∈
    {9, 11, 13, 16}, an L=11 candidate scores 0.5.
  - `L_delta_from_median`: L − median(L in pool), an integer in
    [−(L_max − L_min), (L_max − L_min)].

Both are pool-relative, so a family whose pool skews to long L doesn't
telegraph L via these features.

## Final feature vector

```
FEATURE_NAMES = [
  "orient_fwd",                  # 0
  "orient_rc",                   # 1
  "matches",                     # 2 (integer count, [0, L_max])
  "mismatches",                  # 3
  "identity",                    # 4 (m / L, [0, 1])
  "L_percentile_in_pool",        # 5  ← replaces old feats[2]
  "L_delta_from_median",         # 6  ← replaces old feats[2]
  "flank_start_norm",            # 7  (start / flank_len)
  "flank_end_norm",              # 8
  "boundary_dist_up",            # 9
  "boundary_dist_dn",            # 10
  "target_side_up",              # 11 (was 10; L=10 dropped)
  "nc_len_norm",                 # 12
  "mol_total_ss_frac",           # 13 ← new
  "mol_helix_count",             # 14 ← new
  "guide_window_accessibility_percentile",   # 15 ← new
]
NUM_FEATURES = 16
```

`nc_start_norm` (old feats[11]) is dropped. `L` (old feats[2]) is dropped.

## Patch channels

```
STRUCT_UNP_START = 0        # 16 channels: nc_unp_u1..u16 (RNAplfold accessibility)
STRUCT_UNP_END   = 16
STRUCT_VALID_CH  = 16       # ← True where the L-window RNAplfold call was valid
E_SPAN_CH        = 17       # ← new, formerly implicit / silently zero-filled
E_SPAN_VALID_CH  = 18       # ← new, companion mask for NaN handling
GUIDE_MASK_CH    = 19
MATCH_MATCH_CH   = 20
MATCH_MISMATCH_CH = 21
PAIRED_FLANK_CH  = 22
ALIGN_POS_CH     = 23
MASK_K0_CH       = 24       # ← guide-only span (width L)
MASK_K1_CH       = 25       # ← ±1L
MASK_K2_CH       = 26       # ← ±2L
MASK_K4_CH       = 27       # ← ±4L
PATCH_CHANNELS   = 28
```

## The L-leak that features can't prevent

The candidate-centered RNA window itself has length L (via the guide-mask
channel and, indirectly, the `windowed_valid` shape). A patch model can
recover `L` from channel geometry alone even after we drop `L` from
`feats`. This is a data-vs-featurization concern the tensor layer cannot
solve.

The real defense is the generator: `difficulty.py::sample_difficulty`
draws L uniformly over [L_min, L_max]. As long as the generator holds
that invariant, "family knows L" is prevented at the data level, and
the feature vector's job is just not to make it trivially easy for
gradient descent. Document this here so a future reader does not
mistake the feature vector for a full defense.

## Anchors

The tensor layer must pass a smoke test that:
  - `NUM_FEATURES == 16` (not 13)
  - `PATCH_CHANNELS == 28` (not 22)
  - No candidate's `feats` vector contains `L` as an absolute
  - No candidate's `feats` vector contains `nc_start / nc_len`
  - Every candidate's `patches[:, :, E_SPAN_VALID_CH]` is defined and
    aligns with NaN presence in `E_span_win`

Wire these into `scripts/generator_v5/tests/test_tensor_layer_v2.py`
alongside the candidate-layer anchors. Add to `run_all_anchors.py` once
the module lands.

## What's not in this spec

- Actual training-time consumption of these tensors (that's the model side).
- Structure module (`preprocess/features_structure_v2.py`) is upstream and
  unchanged; the tensor layer just reads its arrays.
- The BPP-vs-RNAplfold channel choice (Item 4.5 already validated the
  4 BPP channels; the tensor layer emits both RNAplfold `unp_uL` and
  BPP-derived `dG_open_uL_pn` / `E_span_win` etc.).
