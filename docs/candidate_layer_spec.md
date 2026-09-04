# Candidate layer spec (Module 3 rebuild)

Written 2026-09-01, after the pool-recovery measurement on the V5 50K
corpus (250K sites) settled which arguments for the rebuild survive
ground-truth verification.

## Why the old 96-slot layer is being replaced

The old preprocessor (`preprocess/candidates.py`) enumerates
`2 orient × 12 L × 4 top_k = 96 slots` per NC region and admits the
top-K by match count per `(orient, L)`. Three structural defects — none of
which are debatable from measurement, and none of which any refit of the
same architecture can address:

1. **Position index destroyed.**
   The 96 slots are a per-`(orient, L)` selection with no association back
   to a shared nc-coordinate frame. A downstream module that needs to ask
   "did site A's peak at nc position p overlap site B's peak at nc position
   p?" cannot ask it — the pool's answer is "here are 96 candidates,
   sorted by match count within (orient, L)", full stop. Channel A's cross-
   site conjunction is the primary consumer of that question and is
   structurally uncomputable on this layer.

2. **Wasted expressive capacity in L=5..6.**
   The layer allocates 4 slots × 2 orient × 2 L = 16 of 96 slots (17%) to
   L=5 and L=6. Identity 1.0 at L=5 gives E-value ≈ 19.6 on an NC of length
   200 and a flank of length 110 — the pool floods with random hits that
   carry no signal. See `preprocess/alignment.py::perfect_seed_density` for
   the density computation; the L≤6 slots do not lift on any correctness
   axis measured in this project.

3. **Ungapped-only architecture.**
   Split-mode planted sites (18% of the V5 50K corpus) have strict recovery
   0.018 vs 0.082 on non-split (4.5× drop). No ungapped candidate can
   represent an A-gap-B guide as a single alignment. The measured any_L
   still lands at 0.836 for split sites because IoU 0.5 lets a candidate
   covering one half count as a hit — but "covering one half" is exactly the
   derivative-label problem in item 4 below.

## What we're NOT claiming any more (retracted 2026-09-01)

These were on an earlier draft. Ground-truth verification retired them.

- ~~"57% of the corpus is lost."~~
  After the orient-naming bug fix, any_L = 0.864 → **14% loss**. Marginal
  as a headline argument.

- ~~"Strict recovery ≥ 0.80 must be achievable."~~
  In-principle unreachable, and the pool-recovery data now proves it. In
  clustered_5p configuration, the mismatch pattern places mm at `{0,1,X}`
  so guide[2:11] is clean. The frame-shift window (nc_start+2, flank_start+2,
  L=11) then carries 8+ matches out of 9 clean positions plus ~1 by
  chance = expected m ≈ 8.5 — HIGHER than the planted window's m = 8. Any
  proposer that ranks by match count (or E-value at fixed L, monotone in m)
  prefers the frame-shift. This is the same phenomenon as D5b: the true
  guide is not the strongest single-site match. Durrant's tolerant matcher
  hits it too, lifting 40% of gold to L=9 subwindows. The frame-shift is
  fidelity, not a defect — a fix that eliminates it removes biology, not
  bias.

- ~~"Labels are derivative therefore broken."~~
  92% derivative is intrinsic to any single-site scorer working on a
  frame-shift-permitting biological ground truth. The rebuild handles this
  via (a) explicit overlap-based correctness (already adopted in
  pool_recovery.py) and (b) cross-site consensus (already Channel A's
  mechanism). No architectural changes needed to accommodate — the fix is
  not "prevent derivative labels" but "score correctness by overlap on the
  planted region, not identity of a single candidate slot".

## Correctness definition (structural, non-negotiable)

A candidate `c = (orient, L, nc_start)` **covers the planted region** for a
site with planted `(orient_p, L_p, nc_start_p)` when:

  1. `c.orient == pool_orient(orient_p)` where `pool_orient` maps the
     generator's arch orient to the pool's orient (`{'fwd','rev'} → {'fwd','rc'}`).
     See "Naming convention" below.
  2. `c.nc_start` overlaps `nc_start_p .. nc_start_p + L_p` with `IoU ≥ 0.5`
     on nc coordinates.
  3. `c` was emitted with position-preserving indexing that lets Channel A's
     cross-site aggregation locate `c` by nc position, not by slot number.

`strict` recovery (identity of `(orient, L, nc_start)`) is measured only as
a diagnostic — never as a target. `strict` is bounded above by the
frame-shift steal rate that the mm_geometry axis (correctly) produces.

## Acceptance targets

| target             | old-layer measured | rebuild bar | rationale                       |
|--------------------|--------------------|-------------|---------------------------------|
| any_L (IoU ≥ 0.5)  | 0.864              | **≥ 0.98**  | 14% loss → ≤ 2% loss ceiling    |
| position-preserved | no                 | **yes**     | structural, decisive            |
| gapped support     | no                 | **yes**     | split-mode any_L → ≥ 0.95       |
| L<7 slots          | 16/96 wasted       | **0**       | E-value density argument        |
| strict identity    | 0.070              | not target  | frame-shift is fidelity         |
| exact position     | via cross-site     | via cross-site | Durrant hits 87% single-nt   |

## Dual emission (position arrays + candidate list)

The candidate layer emits TWO complementary views of the same underlying
`windowed_matches` computation. They answer different consumer questions
and MUST NOT be conflated:

**Position arrays** (`preprocess.candidates_v2.enumerate_position_arrays`):
  `dict[(orient, L)] → np.ndarray[nc_len - L + 1]`, dense, unfiltered,
  un-deduplicated. `arr[nc_start] = max over flank_start of matches`. This
  is what preserves the shared coordinate frame across sites of a bag —
  the array shape lets Channel A's cross-site conjunction at nc position
  `p` become an O(1) per-site index + count, not a list search. It is
  the same shape as `MatchArrays.m_max` from `v5a_framework.match_table`,
  so a v2 MatchTable populated by this call reproduces current Channel A
  numbers byte-identically. This is the "position index preserved" claim.

**Candidate list** (`preprocess.candidates_v2.enumerate_candidates_v2`):
  `list[CandidateV2]`, one per (nc_start, orient, L) that passes the
  E-value threshold, then deduplicated within the same L to one
  representative per biological peak (nc IoU >= 0.5 AND flank IoU >= 0.5,
  higher matches wins). This is the input for tensor-attach / feature
  extraction consumers that want one candidate per biological region.
  Pool-recovery `any_L` is a list-side metric — it measures whether some
  candidate in the list covers the planted region under IoU tolerance.
  Anchor-only cross-site queries on this list DO NOT work (dedup
  collapses shifted duplicates of the same peak), so Channel A must not
  read from this list; it reads from position arrays.

Attempting to run Channel A on the list gives 0.10 coverage at L=11 m>=8
S=5, vs 0.65 on the arrays (measured 2026-09-02, 500-bag L=11 subset).
The 0.55pp gap IS the dedup cost — recovered by using position arrays.
The list is a compressed downstream-friendly emission; the arrays are the
canonical cross-site aggregation input. Retain both.

## Architecture (nc-window-first)

- **Enumeration**: for each nc position `p ∈ [0, nc_len - L_min)`, compute
  the max-m alignment on the flank (both orientations) for each
  `L ∈ [L_min, L_max]`. Emit one candidate per (p, L) pair whose E-value
  passes a global threshold — not per-(orient, L) top-K. Position arrays
  emit the full computation without the E-value screen or dedup;
  candidate list applies both.

- **L range**: `L_min = 7, L_max = 16`. Below L=7 the E-value density
  swamps signal; above L=16 the guide-length distribution has < 0.1%
  mass.

- **Ranking**: E-value at the **per-(nc_start, L, orient)** grain,
  marginalized over L via min-E. No per-(orient, L) quota. The E-value
  here is `P(m'>=m | L, uniform bg) * n_flank * n_orients` — the
  expected background count at ONE nc_start over the flank axis and
  both orientations. NOT the W2 diagnostic table's "global E" (which
  additionally multiplies by n_nc_positions ≈ 190). Threshold `E<=1`
  admits candidates whose expected by-chance count at their own
  position is at most 1. See `preprocess.candidates_v2._e_value_uniform`
  docstring for the exact relationship. If the pool must be capped for
  tensor size, cap by top-K per nc position (not per L), keeping the
  position-preservation invariant.

- **Deduplication**: candidates that fall within one L of each other on
  the nc axis at the same orient — the same "peak" seen at neighboring
  windows — collapse to the one with the maximal L extension. Not the
  most-significant; the most-extensive. This preserves the "which nc
  region hit" answer without inflating pool size.

- **Split-mode**: support a `gap ≤ 30 nt` gapped alignment as a first-
  class candidate type. Emit `(nc_start, gap_position, L_A, L_B)` with
  the same nc-coordinate frame. The pool's shape becomes
  `(candidates × [ungapped fields, gap fields])` with a gap-length
  channel; gap-length = 0 flags an ungapped candidate.

## Naming convention (unified 2026-09-01)

Generator arch metadata and pool candidates now use the SAME orient
labels: `{'fwd', 'rc'}` on both sides.

- **Pre-2026-09-01 V5 batches** (the frozen 50K + neg 10K) emit
  `arch.orient ∈ {'fwd', 'rev'}`. All consumers must accept BOTH.
  Canonical mapping: `pool_orient(x) = 'rc' if x == 'rev' else x`.
- **New V5 batches** (post-mm_geometry regen + this fix) emit
  `arch.orient ∈ {'fwd', 'rc'}` natively. No mapping needed.
- Enforcement: `scripts/v5a_framework/metrics.py::assert_non_degenerate_stratification`
  refuses to publish any report where a stratum with n ≥ 50 has all rate
  metrics == 0.0 or all == 1.0. Wired into
  `scripts/preprocess_pool_recovery/stratify.py` and expected in every
  stratifier from here on.

## What Channel A gains from the rebuild

- **Cross-site conjunction becomes computable**: with position-preserved
  candidates the aggregator can ask "does nc position p carry a same-L,
  same-orient candidate on ≥ S sites of the same Tnp?" — the S=5 rule
  in `channel_a_v5.py` currently computes this by rebuilding position
  coverage from raw dot-plots (bypassing the pool). After the rebuild
  Channel A reads directly from the pool.

- **Split-mode Channel A stops undercounting split sites**: no more
  half-covers as first-class candidates that then fail the S=5 gate for
  the wrong reason.

- **Anchor pass-through unchanged**: the S=5 rule + Mode 1/Mode 2 spec
  are unaffected. Pool-recovery numbers change; conjunction gate does
  not.

## Not in scope

- Structure channels (nc_channels are computed elsewhere; the candidate
  layer emits (nc_start, orient, L, gap-fields) tuples; the tensor layer
  attaches the per-nt structural context around each candidate).
- Feature vector redesign (dropping absolute coordinates from `feats[11]`
  and moving to relative-position-only features is Module 4's job).
- Constrained mismatch sampling (Channel B blocker — separate track).

## What Channel A is (and is not) detecting

Because `n_planted` is a variable axis of the V5 generator, the same object
can be labeled either way depending on how you define "guided". A Tnp with
four truly guide-mediated sites plus one random insertion has
`n_planted=4` and is a positive if you define guided as "any evidence of
guide-mediated targeting" and a negative if you define it as "≥5 sites share
one guide region". This ambiguity is intrinsic to biological RNA-guided
transposition, not a data defect.

Channel A tests the SECOND definition. Its output should be read as
"≥ S sites of this Tnp share a common guide-mediated nc-region signature",
NOT "this element is RNA-guided". A downstream consumer that wants the
first definition has to relax S below 5 (and pay the corresponding FPR
cost, which the hard-negative sweep quantifies).

Quantitatively: the probability that Channel A fires at the planted
position for a Tnp with n_planted guide-mediated observable sites (of 5)
decomposes as

  P(fire) = p^n × q^(5 − n)

with p = 0.86 (per-planted-site detection rate at the Channel A spec
of m ≥ 8, S = 5, L = 11) and q = 0.21 (background per-position rate,
externally calibrated — the same q reproduces the raw-noise FPR floor
0.033 = q^5 · #positions). Sensitivity at n < 5 falls as p^n, so losing
one observable site drops sensitivity from 0.47 to 0.11 (~5×), and the
likelihood ratio contributed by each additional confirming site is
p / q ≈ 4.1. The S-threshold tradeoff on that surface is now analytic;
scanning the S dimension is no longer necessary once (p, q) are fitted.

The n_planted axis is a stratification dimension in BOTH the positive
scoring (how much coherence a real element carries) and the negative FPR
sweep (how much partial coherence the S=5 rule tolerates). The scores at
matched n_planted quantify the safety margin at each definition; they are
not comparable across definitions.
