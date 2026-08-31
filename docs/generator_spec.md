# Generator specification — architecture randomization × difficulty matching

Companion to `docs/channel_a.md`. Requirement 0 v3 as defined there is
the target this generator satisfies. This document is the design
contract; the implementation task is downstream.

## Why this generator exists

Real IS110 / IS1111 data has architecture baked into every example
(guide position, ncRNA length, match-segment order, NCR position
relative to ORF, TSD relation to guide target). A model trained on
any single natural family learns that family's architecture as if it
were the task and fails on any different-architecture holdout. V4.2's
downstream failure and V5A-3a's collapse to chance are both instances
of this — captured in `docs/channel_a.md` under D5b.

The generator's role is therefore not "make more samples" but "make
a training environment where architecture cannot be learned as a
shortcut," while keeping difficulty matched to real observed data.
Difficulty is controlled by two parameters (D5e / A+ chain, calibrated
in the doc): `planted_m` distribution and `ncRNA_length` distribution.
Architecture is the set of everything else, and all of it must be
uniformly randomized per-example within physically valid ranges.

## Difficulty side (matched to real data)

Values are frozen against the natural calibration base.

### `guide_length L`

Discrete uniform over `{11, 12, 13, 14}` per example.

Calibration base: T-WT native L=11 (170 sites, primary evidence for
IS110); ISEc11 native L=13 (`GTGAAAATACTGT`, seekRNA Supp Dataset 1);
ISEc21 native L=14 (`CACAGCGATCAGGG`, seekRNA Supp Dataset 1).
Additional length data available from the Durrant `2_7bp_RTG` and
`3_7bp_RTG` variants (L=14 engineered) as consistency checks against
the extremes but not as sample-generation targets.

Rationale: L varies across natural families; hardcoding L=11 would
reinforce the shortcut that failed the RTG variants. The `{11..14}`
range covers the observed natural spread. Wider extension (L ∈ {9,
10} on the low side, L ≥ 15 on the high side) requires per-example
justification and is not part of the base spec.

### `planted_m` distribution at chosen L

For each site's target insertion into the flank, the target sequence
is planted with a match count `m` drawn from a distribution centered
at the T-WT operating point:

- Mode at `m = 8`, ~86% mass concentrated in `[7, 9]`.
- Zero mass at `m ≥ 10`.
- Small tail at `m ∈ {5, 6}` reflecting the failing 4bp_RTG variants'
  observed rate (they exist in nature, they just do not detect at
  the S=5 threshold).

Calibration base: Durrant T-WT L=11 at nc position 49 —
`d5e_competitor_count.py` measures per-site m distribution with n=170.
The zero mass at `m ≥ 10` is the concrete diagnostic (V4.2 currently
has 45% at m≥9, 20% at m≥10 — the concrete synthetic→real gap).

Extension to other L: assume similar per-L operating shape unless a
natural counter-example emerges. Currently the only L=14 systems in
hand are RTG engineered variants with mixed detection outcomes and
insufficient calibration material.

### `ncRNA_length`

Continuous uniform over `[177, 281]` nt.

Calibration base: n=6 natural ncRNAs measured in
`aplus_calibration.py` — Durrant T-WT (177), and five seekRNA systems
ISEc11 (273), ISKpn4 (261), ISPst6 (255), ISPa11 (275), ISEc21 (281).

Rationale: absolute `competitor_count = rate × length`, and rate is
essentially the analytic constant `2 × N_starts × P(Bin(L, 0.25) ≥ m)`
(A+ measured 0.204–0.238 per nt against the analytic 0.262, difference
= composition skew). At matched planted_m, difficulty scales linearly
with ncRNA length; the range [177, 281] gives a 1.6× spread in
absolute competitor count, forcing the model to see that difficulty
is not a fixed number.

Extension to shorter/longer ncRNAs requires per-length competitor
curve re-measurement (trivial with the A+ machinery) and per-length
planted_m calibration (not trivial — no natural anchor outside T-WT).

### Background composition

Real bacterial from the 2,763-flank negative-family pool
(`real_data/formatted/real_*_sites.jsonl`). Composition-preserving
shuffling for negative controls but never as training positives.

## Architecture side (randomized per example to destroy shortcuts)

Every axis below draws uniformly (or with the stated discrete
distribution) *per example*. Do not correlate axes across a batch.
Do not sample from a per-family joint that mirrors any real family.

### `guide_position_in_ncRNA` (most important)

**Uniform within the RNAfold-predicted single-stranded (loop) region
of the ncRNA.** Not uniform over the whole length; that would place
guides inside stems, which does not exist mechanistically.

Implementation: generate the ncRNA, fold it with RNAfold, mask the
positions where paired probability > 0.5 (or similar), draw guide
start position uniformly from the remaining single-stranded windows
that are long enough to accommodate the guide.

Rationale: T-WT places all 170 sites at nc position 49 (`range=[49,
49]` in supp gold). If the generator places guides at any consistent
relative position, the model learns "look there." Uniform over
loops removes the positional shortcut without violating the
biophysical constraint that guides sit in single-stranded regions.

This is a mechanism-based constraint, not a family-specific prior.
No natural family provides the target distribution.

### `segment_order` (family-agnostic axis)

Discrete uniform over `{forward, reversed}` per example.

Calibration base: Seekhi et al., Nat Commun 15:5235 (2024), main text
plus main Fig 7a — IS110 has the top-strand match 3' of the
bottom-strand match on the ncRNA; IS1111 has the reverse order.
Empirical binary axis, no ambiguity.

Rationale: this is the single-most-cited family-distinguishing
architectural feature. A model that reads segment order as a family
signal cannot generalize across families. Randomizing to 50/50
removes this signal.

### `segment_count`

Discrete `{1, 2, 3}` with default weights `{0.2, 0.6, 0.2}`.

Rationale: canonical IS110/IS1111 systems have two match segments
(one for each target strand); single-segment (short seekRNA variants)
and three-segment cases (some engineered constructs) exist. Weights
reflect approximate natural prevalence but the tails are important
to include.

### `segment_spacing`

Discrete uniform over `[8, 50]` nt spacing between the two match
segments on the ncRNA (when `segment_count = 2`).

**Marked as "wide range, calibration pending."** ISEc21's
experimental seekRNA construction (seekRNA Supp Dataset 1's pSFA
constructs) implicitly gives some spacing information but the
extraction is deferred. The wide range is intentionally over-inclusive
so training does not exclude the natural range.

### `NCR_position_relative_to_ORF`

Discrete uniform over `{upstream, downstream}` per example.

Calibration base: IS110 NCR is upstream of tnp ORF; IS1111 NCR is
downstream. Randomizing removes this family signal.

### `N_nc` (number of non-coding regions per site)

Discrete `{1, 2, 3}` with weights `{1/3, 1/3, 1/3}`.

Rationale: V4.2 mining showed 33/33/34% distribution across N_nc
1/2/3 on real IS elements; Durrant's own corpus is 100% N_nc=1 (which
is why the specificity claim can't be measured there). Uniform
weights over `{1, 2, 3}` covers the observed spread.

### `TSD_width × spatial_relation_to_guide_target`

Two-axis joint. First axis (TSD width): discrete `{0, 2, 8, 9}` nt
with weights `{0.4, 0.15, 0.2, 0.25}` reflecting IS110 (no TSD, 40%)
and the four measured non-guided family widths (IS30 = 2, ISLdl1 = 8,
IS10-R and IS903 = 9). Second axis (spatial relation): discrete
`{overlap, adjacent, separated_40_60bp}` uniform.

Rationale: the overlap case is what makes junction-masking dangerous
(D5b prep discussion). The separated case is CAST/Cas12k-Tn7 style.
Both cases exist; forcing the model to see both is the only way it
learns to handle TSD without hardcoding a family-specific relation.

### `5_prime_stem_loop_present`

Discrete `{True, False}` uniform.

Rationale: ISPpu10 lacks the 5' stem-loop the other IS110 systems
have (Durrant Supp Table 5). Presence/absence is a binary architectural
axis that must not be a family signal.

## Cross-site structure (what V4.2 was missing)

The V4.2 corpus generates each site independently with a fresh planted
guide sequence. This breaks the coherence signal that the real task
depends on: at each Tnp, all sites share the same guide, and the
guide-target consistency across sites is what the S=k conjunction
detects.

The generator must produce sites in bags:

- Draw one guide sequence per Tnp bag.
- Generate all N_bag sites (N_bag = 5 for the baseline calibration
  match) as insertions of that same guide's target into different
  flanks.
- Position spread across sites: draw from `Normal(0, 1)` nt centered
  on the planted position (T-WT range=[49, 49] gives ~1 nt spread as
  observed).

Detection tests will run on the generated bags exactly as on the
Durrant corpus.

## Acceptance tests

The generator's output is acceptable iff it passes both of these
tests against the exact same measurement code that produced the
Channel A doc's numbers.

### Test 1 — `competitor_count` distribution matches T-WT

Reuse `d5e_competitor_count.py`. On 2,000 generated positive sites at
L=11 (or the site's own L, tabulated separately):

- `fraction(competitor_count ≤ 10) < 5%` at L=11 (T-WT baseline
  2.94%).
- `median(competitor_count / ncRNA_length) at planted_m=8` within
  `[0.19, 0.24]` (T-WT gives 0.215, 6-system range 0.204–0.238).
- Zero mass at `planted_m ≥ 10` — the specific V4.2 gap.

### Test 2 — planted guide is NOT per-site argmax

Reuse `d5b_argmax_discriminator.py` logic. On 2,000 generated positive
sites at L=11:

- `fraction(planted position IS argmax on nc)` should sit around
  T-WT's 15.3% (a strict 100% ties, and delta=0 pattern) — not V4.2's
  30.7%.
- `median(competitor_count at planted position) ≥ 2` — planted must
  always be tied by at least one other position, matching T-WT's
  0/170 sole-max structure.

The second bullet is the D5b structural property: guided signal
is only visible through cross-site conjunction, not per-site
extremum. Generator failing test 2 means it produced an easier
problem than reality regardless of what test 1 says.

## Output format

JSONL matching V4.2's `positives_v42.jsonl` schema so downstream
tools (MatchTable builder, framework tests) accept the output without
changes:

```json
{
  "site_id": "...",
  "transposase_id": "...",   // shared across N_bag sites of the bag
  "ncrna_id": "...",         // one per bag
  "inputs": {
    "flank": "<120 nt>",
    "noncoding_regions": ["<177-281 nt>", ...]
  },
  "labels": {
    "is_positive": true,
    "target_position_in_flank": [start, end],
    "target_dna": "<12-16 nt>",
    "guide_dna": "<12-16 nt, may differ from target by planted mismatches>",
    "perfect_guide_dna": "<12-16 nt matching bridge RNA guide>",
    "guide_length": 12,
    "n_mismatches": 2,
    "active_noncoding_index": 0,
    "num_noncoding_regions": 1,
    "guide_span_in_active_noncoding": [start, end],
    "ncrna_length": 177,

    // NEW fields for architecture axes (backwards-compatible additions)
    "arch": {
      "segment_order": "forward",
      "segment_count": 2,
      "segment_spacing": 22,
      "ncr_position_relative_to_orf": "upstream",
      "tsd_width": 0,
      "tsd_spatial_relation": "overlap",
      "five_prime_stem_loop_present": true,
      "planted_m_at_L": 8,
      "planted_L": 11
    }
  }
}
```

## Backlog (do NOT block on these)

1. **seekRNA architecture axes extraction from Supp Table 1 colored
   bases.** Two extractions needed: per-system normalized guide
   position (verify cross-family conservation ~27–28% of ncRNA length,
   as T-WT 27.7% and ISEc21 5'-third suggest), and segment spacing
   ranges. Delivers only calibration refinements; the generator's
   randomization schemes (uniform in loops for position, wide range
   [8, 50] for spacing) already cover them.
2. **IUPAC consensus sampling from Durrant Supp Table 5.** Would
   confirm intra-architecture competitor rate variance. Value:
   tighten Requirement 0 range if variance is small. Value: analytic
   rate `≈ 0.21` is already known, and sampling would reproduce it
   modulo composition skew. Deferred as lowest priority.
3. **seekRNA `AtaideLab/Targets` pipeline execution.** Delivers
   real observed insertion sites per IS1111 system for the held-out
   generalization test. Runtime: days. Blocked on nothing at
   generator-spec level; runs after generator is producing samples.
4. **Durrant `BridgeRNA2024` pipeline execution.** Delivers larger-N
   IS110 material. Runtime: weeks. Lowest priority — IS110 is n=1
   for architecture and does not address generalization.

## Provenance chain

- `docs/channel_a.md` — parent document; Requirement 0 v3 lives there.
- `scripts/v5a_framework/tests/d5b_argmax_discriminator.py` — Test 2's
  D5b structural property (guided signal not per-site extremum).
- `scripts/v5a_framework/tests/d5e_competitor_count.py` — Test 1's
  competitor_count measurement code.
- `scripts/v5a_framework/tests/aplus_calibration.py` — A+ ncRNA length
  range and analytic rate verification.
- Data sources: Durrant Nature 630:984 (2024), Siddiquee et al.
  Nat Commun 15:5235 (2024).
