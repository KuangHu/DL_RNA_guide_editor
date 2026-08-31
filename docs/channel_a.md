# Channel A — Cross-Site Coherence Detector for RNA-Guided Editors

## Why the method exists

In real IS110 target-site data, the guide–target match is **not the
strongest match** at any individual insertion site. It becomes visible
only through **cross-site consistency**: the same guide-adjacent nc
position produces above-threshold matches across multiple observed
insertions of the same transposase, even though at every one of those
sites individually there is some *other* nc position with a stronger
match. **Any statistic that first collapses each site to a single best
candidate — including but not limited to argmax, top-K, ranked
per-site models, and per-site regression on match features — is
bounded below the signal by construction.** Channel A is a
conjunction over thresholded per-site events, and this is why it works
where the earlier per-site selectors in this project (`raw_m`,
`length_pen`, within-pool z, V5A-3a's learned selector) all plateaued
between MRR 0.086 and 0.150.

The direct evidence is in the framework tests:

- On the sole natural wild-type bridge RNA in the Durrant benchmark
  (34 evaluation bags of `T-WT_D-WT`), Channel A detects 20/34 bags
  (58.8%) with the sensitivity numbers below. Yet the per-site
  argmax on nc lands in the gold neighborhood [45, 54] with a
  **median of 0 sites per bag** (`test_tau0_anchor.py`,
  `d5b_argmax_discriminator.py`). Detection comes from the S=5
  conjunction, not from any per-site extremum.
- Per-site `m` at the gold position on T-WT (L=11) has **mean 7.94**
  and is *tied* by other nc positions **100% of the time** (0/170
  sites have the planted position as sole max;
  `d5e_competitor_count.py`). The signal is only extractable by
  requiring 5 sites to agree above threshold at the same nc
  coordinate.

Everything below is derived from this structural fact, or from what
the corpus can and cannot say given it.

## Method

**Mode 2 — m-threshold, fixed L = 11, m ≥ 8, S = 5.** For each
transposase with 5 candidate insertion sites sharing an ncRNA:

1. For each nc position p, for each site s_i, take
   `m_max(p, s_i) = max over flank offset f of matches between nc[p:p+11]
   and s_i.flank[f:f+11]`, pooled over both orientations.
2. `site_hit(p, s_i) = 1` iff `m_max(p, s_i) ≥ 8`.
3. `S(p) = Σ_i site_hit(p, s_i)`.
4. Optionally apply Gaussian kernel over `S` with τ ∈ [0, 5].
5. Emit peaks where `S(p) ≥ 5` and `p` is locally maximal within ±5 nt.

**Mode 1 (previously "recommended default", now retracted).** Mode 1
used `min-E` marginalization over `L ∈ {9, 10, 11, 12}` with
E-threshold admission. It is retracted because at least one Durrant
variant class (`7bp_RTG`) has target-loop length 14 and cannot be
detected at any `L ∈ {9..12}`. Mode 1 remains a valid variant when
the target-loop length is known to sit within its `L` range; for
general use, `L` must span the operating regime of the target task,
which the Durrant corpus alone cannot determine.

**Peak-finding convention.** The local-max rule is *"i is a peak iff
no neighbor j in [i-min_dist, i+min_dist] has S[j] > S[i]"*, with
`min_dist = 5`. This is equivalent to `S[i] ≥ S[j]` for all j in the
window, so plateau ties emit every position on the plateau. `min_dist`
is a radius for the local-max check, not a merge distance for the
output. At τ=0 with integer-valued S, plateau duplication is a
positive-measure event; at τ>0 with Gaussian-smoothed S it is
measure-zero. All τ=0 metrics report both peak-level and Tnp-level
denominators.

**Provenance.** `scripts/v5a_framework/variant.py::spec_m_threshold_L11`
implements Mode 2. `scripts/v5a_framework/tests/test_tau0_anchor.py`
freezes the historical baseline as a regression: on 65 Durrant
`_paired_bag*` records, exactly 22 covered, 23 peaks emitted, 22
IoU-correct, 21 with correct-peak per Tnp, 21 within centroid
distance ≤ 1 nt of gold.

## Localization when discriminated positive

On the 22 detected bags, the τ=0 baseline reports:

| statistic | value | denominator |
|---|---:|---|
| Coverage (any peak emitted) | 33.85% (22/65) | N_bag_total |
| Plateau width (median) | 1 nt | N_detected_bags |
| Plateau width (mean) | 1.05 nt | N_detected_bags |
| `contains_gold_frac` | 90.9% (20/22) | N_detected_bags |
| `centroid_dist` to gold (median) | 0.0 nt | N_detected_bags |
| `centroid_dist` to gold (mean) | 1.02 nt | N_detected_bags (dominated by one 21-nt outlier) |

*In 19 of 22 detections the detector produces a 1-nt-wide primary
peak exactly at annotated `gold_nc`.* In one detection the plateau is
2-wide centered on gold (bag010, positions 49 and 50, gold at 49). In
one detection the peak is at gold + 1 (bag001 of an RTG variant with
1-nt biological offset). In one detection the peak is 21 nt from
gold. This is the honest form of the "single-nucleotide precision"
claim.

**Three correctness criteria are reported side by side. They differ
by one bag each and are NOT interchangeable.**

| criterion | value | what it tests |
|---|---:|---|
| Tnp-level PPV (IoU ≥ 0.5) | 95.5% (21/22), CP CI [0.772, 0.999] | window overlap |
| `contains_gold_frac` | 90.9% (20/22) | plateau strictly contains gold_nc |
| centroid_dist ≤ 1 | 95.5% (21/22) | plateau centroid within 1 nt of gold |
| centroid_dist == 0 | 86.4% (19/22) | plateau centroid exactly at gold |

The 20/22 vs 21/22 vs 19/22 counts come from `bag001` (plateau {50},
gold 49: IoU pass, contains_gold fail, `≤1` pass, `==0` fail);
`bag000` (plateau {70}, gold 49: all fail); and `bag010` (plateau
{49, 50}, gold 49: all pass except centroid `==0`). Historical
"87% single-nucleotide precision" used a gold-aware "closest to gold"
tie-break which peeks at the answer. Under the gold-blind centroid
rule the same 22 detections give 86.4% at `centroid_dist == 0`.

**PPV per what?** The 21/22 CP CI assumes 22 independent trials. But
20 of the 22 successes come from bags of the same bridge RNA
(`T-WT_D-WT`, 34 bags of one nc sequence with disjoint 5-site
subsamples, see D1 in `a2_probe_endflank_motif.py`). Two claims
must be reported separately:

- **Per-detection localization precision (denominator = 22
  detections):** plateau width median 1 nt, `contains_gold`
  90.9%, `centroid_dist ≤ 1` 95.5%. Clopper-Pearson CIs on 22
  are meaningful for this claim; each detection is a
  point-emission event, and 22 emissions have been made.
- **Per-natural-system detection rate (denominator = 1 natural
  bridge RNA):** T-WT succeeded, 1_7bp_RTG succeeded, and the
  other 7 engineered variants did not. This is n = 9 as observed,
  n = 2 as successes, with the caveat that eight of the nine are
  engineered perturbations of one wild type (see Section 4).

The doc reports both. Confusing them was one recurring source of
retractions in the diagnostic phase.

## Sensitivity, variant-stratified

The 65 Durrant `_paired_bag*` identifiers do not represent 65
independent systems. They cover **9 unique ncRNA sequences**
(committed as `38626da`): one natural wild-type bridge RNA plus
eight engineered target-guide (RTG) variants. Detection rate by
variant, with target-loop length (TBL) from `durrant_gold_v1.jsonl`:

| variant | # bags | TBL | detected | rate |
|---|---:|---:|---:|---:|
| **`T-WT_D-WT`** (natural WT) | 34 | 11 | 20 | 58.8% |
| **`1_7bp_RTG_D-WT`** | 2 | 14 | 2 | 100% |
| `2_7bp_RTG_D-WT` | 5 | 14 | 0 | 0% |
| `3_7bp_RTG_D-WT` | 2 | 14 | 0 | 0% |
| `4_7bp_RTG_D-WT` | 3 | 14 | 0 | 0% |
| `4_4bp_RTG_D-WT` | 3 | 11 | 0 | 0% |
| `1-4bp-RTG_D-WT` | 2 | 11 | 0 | 0% |
| `2_4bp_RTG_D-WT` | 9 | 11 | 0 | 0% |
| `3_4bp_RTG_D-WT` | 5 | 11 | 0 | 0% |

**Two-of-nine bridge RNAs succeed under Mode 2 (fixed L=11, m≥8, S=5)**:
the natural wild type and one 7bp-RTG variant. The other 7 fail
completely.

Coverage failure is *not* qualitatively bimodal ("some variants work,
others don't for a special reason"). It is a monotone function of two
site-level quantities that Channel A jointly requires: per-site
P(m ≥ 8 at gold vicinity), and cross-site position coincidence. The
`4bp_RTG` group has median m ≈ 7 at L=11 at gold vicinity
(`d5b_argmax_discriminator.py`), which sits just under the m ≥ 8
threshold, and its site-level detection collapses. The `7bp_RTG`
group has target-loop length 14, so the L=11 detector window can
only capture a subset of matches; only the highest-fidelity of that
group (1_7bp with `target_flank_matches` median 11 out of 14) has
enough L=11 matches to pass. **The historical "p^5" attribution is
retracted:** it fit the T-WT number within 30% and predicted
0.4–1.2 detections on failing variants (against 0 observed),
consistent with either the p^5 model or normal small-N noise, but
`c5d0a75` showed the actual controlling quantity is
`competitor_count = #{positions : m ≥ m_planted}`, and on Durrant
T-WT that count is always ≥ 1 (planted position is never sole max).

**Statistical-power caveat.** The claim "V4.2's failing planted_m
distribution matches T-WT's difficulty at the operating point" is
based on the shared-support stratum `planted_m = 8` (T-WT n = 147,
V4.2 n = 600). Other strata are underpowered on the T-WT side
(m=7 has n=17, m=9 has n=6, and m∈{10,11} have n=0). The stratum
that dominates T-WT (m=8, 86% of sites) is the same stratum where
both corpora have sufficient sample; on that stratum V4.2's
competitor-count median (53) is close to T-WT's (38), differing by
the extra 74 nc positions V4.2 offers (nc length 251 vs 177).

## Specificity — not measured on this corpus, and why

The current corpus provides **no measurable specificity signal for
Channel A**, for two independent structural reasons.

**Reason 1 — negative flanks × Durrant nc is double-null.** Under
"Option E" evaluation (scoring negative-family flanks against the 65
Durrant ncs), both sides are architecturally random with respect to
Durrant's guide targets. The 5 non-guided families (IS10-R, IS30,
IS903, ISAjo2, ISLdl1) target their own genomic contexts, not any
Durrant bridge RNA position; and no coherence is expected between
different insertions of a non-guided family scored against Durrant's
target-position 49. The measured cross-family AUROC on the
sum-based coherence discriminator is **0.514** (`860adfd`,
`b478f66`), which is not evidence that Channel A generalizes to
non-guided families — it is evidence that the negative construction
gives an identically-random distribution on both sides.

**Reason 2 — Durrant self-cross gives 9 unique nc sequences.** The
paired shuffling protocol used in prior versions (Durrant flank ×
another Durrant bag's nc) draws from 9 unique bridge RNA sequences,
34 of which are the same T-WT variant. The historical
`shuffled null = 0.0226 per bag` was measured under this
duplication, and it *understates* the null by an unknown factor
because "shuffled" 5-flank groups often draw multiple flanks that
target the same T-WT bridge RNA position 49 by construction. That
inflation propagates into the historical `real / shuffled = 15.5×`
figure, which is a lower bound rather than a point estimate.

Specificity evaluation therefore requires **independent guided
systems with architecture variability**. The current corpus has
neither.

### Real data pipeline (calibration + holdout)

Two identified real-data sources fill part of the gap:

- **~10–11 ncRNA sequences from published supplementary tables.**
  Durrant Supp Table 5 provides consensus bridge RNA sequences with
  structure for 6 IS110/IS1111 family members (`ISPpu10`, `ISAar29`,
  `ISHne5`, and 3 others). seekRNA Supp Table 1 (Siddiquee et al.,
  Nat Commun 2024) provides 5 wet-lab-characterized systems (4
  IS1111 + 1 IS110): `ISEc11`, `ISKpn4`, `ISPst6`, `ISPa11`,
  `ISEc21`, NCR lengths 74–96 nt. Some overlap likely. These do not
  include per-recombinase multi-site flanks, but they carry the
  architecture axes needed for the calibration approach below.
- **Durrant's 1,054-recombinase supplementary table is
  phylogenetic metadata only** (6 columns: hashed Protein ID,
  ISfinder ID, Kingdom, Phylum, RuvC 80-aa alignment fragment,
  Tnp 80-aa alignment fragment). It contains no host genome
  accession, no chromosomal coordinates, no per-element bridge RNA
  sequence, no LTG/RTG. This pool is **not** a calibration set as
  such — per-recombinase data (bridge RNA, target site, multi-site
  flanks) requires re-running Durrant's extraction pipeline
  (`hsulab-arc/BridgeRNA2024`) against NCBI + metagenomics reference
  databases, a weeks-scale engineering commitment that produces
  IS110 material only and therefore does not address architecture
  generalization.
- **seekRNA architecture axes are usable now.** The 5 seekRNA
  systems span testable architectural differences relative to IS110:
  reversed order of upstream vs downstream match segments on the
  ncRNA (main Fig 7a; the binary architectural fingerprint the
  detector must not depend on); NCR downstream of the ORF rather
  than upstream; presence of sTIR; ncRNA lengths 74–96 nt against
  IS110 bridge RNA lengths ~150–250 nt — a 2–3× search-space spread
  that lets us test whether `competitor_count` really is
  architecture-invariant. The seekRNA `AtaideLab/Targets` pipeline
  (GPL-3.0) can regenerate per-system genomic insertion sites in
  days, providing the held-out generalization test on IS1111.

**Background competitor-density curves — the immediate calibration
approach.** `competitor_count` factors:
```
competitor_count = f(planted_strength, background_match_density)
```
Only `planted_strength` requires observed real insertion sites (which
the ~11 supplementary ncRNAs do not carry). `background_match_density`
does not: it is the number of positions on a given ncRNA where a
random L-mer from a real bacterial flank hits `m ≥ k`. This is
computable directly from any random real bacterial sequence, and
this project already has 2,763 negative-family flanks (`IS10-R`,
`IS30`, `IS903`, `ISAjo2`, `ISLdl1`), all real bacterial sequences,
loaded on disk. Pairing each of the ~11 published ncRNAs with random
draws from the 2,763-flank pool yields, per ncRNA, a background
competitor-density curve `competitor_count(m)` for `m ∈ {6..12}`.
That is the difficulty-dominating factor in Requirement 0.

The "real planted_m" factor remains n = 1 — Durrant T-WT's
experimentally validated wild-type operating point (median m = 8 at
L = 11, 86% concentrated there). Combined with 11 architecture-
diverse background curves, the calibration base becomes:
"11 ncRNAs' background competitor density × 1 validated wild-type
operating point," with a built-in cross-architecture invariance
test (2–3× ncRNA length spread, order-inversion between families):
if two systems 3× apart in length show equivalent detector
performance at matched competitor count, competitor count is
architecture-invariant, as needed. Cost: several hours.

Real data provides **calibration and holdout**, not training and
selection. Training on any single real family alone would learn its
family-specific architecture (position, length, orientation, N_nc,
match-segment order) as if that architecture were the task, and fail
the held-out family. That is the class of failure V4.2 downstream
selectors repeatedly reproduced on Durrant.

### Generator role (training + selection)

Training and selection require a synthetic generator whose role is
**architecture randomization × difficulty matching**, orthogonal:

- **Architecture axes randomized** (per-example uniform draws where
  applicable): guide position in ncRNA (most important); guide
  length `L`; ncRNA length; number and order of guide match
  segments (upstream first, downstream first, single segment, three
  segments); NCR position relative to ORF (upstream / downstream);
  orientation convention; N_nc (1, 2, 3); TSD presence × width ×
  spatial relation to guide target; secondary-structure context
  (with / without 5' stem-loop).
- **Difficulty axes matched** to the calibration distribution
  from the background-curve method above (each of the ~11
  supplementary ncRNAs paired with random real bacterial flanks
  from the 2,763-flank negative-family corpus, giving
  `competitor_count(m)` per ncRNA for `m ∈ {6..12}`), pinned at
  the T-WT wild-type operating point (median m = 8 at L = 11).

The orthogonality is what makes this useful. Difficulty controlled
by competitor-count is family-agnostic; every architecture axis can
vary freely without changing the detection burden. Any statistic the
detector learns that depends on architecture will fail on IS1111's
flipped-order holdout. Any statistic that depends only on
cross-site coherence — which is the only architecture-invariant
signal — will generalize.

**Requirement 0 for the generator, decomposed into two constraints.**

Under the analytic form `E = N_windows · P(Bin(L, 0.25) ≥ m_planted)`,
per-nc-position competitor rate is essentially a mathematical constant
determined by `(L, m_planted, flank_length)` alone. At L=11, m=8, 120-nt
flank pooled over 2 orientations:
```
110 flank starts × 2 orient × P(Bin(11, 0.25) ≥ 8) = 0.262 competitors / nc position
```
A+ measurement on 6 clean natural ncRNAs (Durrant T-WT plus 5 seekRNA
systems, 177–281 nt, spanning IS110 and IS1111) gave observed rates
0.204–0.238 — the shortfall from 0.262 is exactly the composition-skew
correction. **Because the rate is analytically pinned, "match the
observed rate" is a nearly-vacuous constraint that any reasonable
generator satisfies automatically.** The A+ finding confirms
architecture invariance holds — that was needed for the generator's
orthogonality argument — but does not by itself constrain the
generator.

Absolute competitor count, which is what enters the S=k conjunction
probability (`q^S`, q ≈ absolute count / nc positions), is
determined by `rate × ncRNA_length`. Same rate 0.21 with ncRNA lengths
177 vs 281 gives 38 vs 65 competitors — a 1.7× difficulty spread. So
Requirement 0 needs two independent parameters, not one:

- **`planted_m` and `L` distribution.** Match the T-WT operating point:
  L = 11, mode `planted_m = 8`, ~86% concentrated at m=8 within [7, 9],
  zero mass at `m ≥ 10`. V4.2 currently has 45% at m ≥ 9 and 20% at
  m ≥ 10 (D5d). This is the gap.
- **ncRNA length distribution.** Sampled from the 6-system natural
  range: 177–281 nt (T-WT + 5 seekRNA). V4.2's 251 nt sits inside
  this range, so V4.2's length distribution is not the problem. If a
  novel family with substantially different ncRNA length is targeted,
  the range extends accordingly (calibration base grows with n).

The A+ finding pins the rate parameter (analytically, always). The
D5d finding pins the `planted_m` distribution parameter (T-WT
experimentally). The ncRNA length parameter is now calibrated to 6
natural systems' range, up from 1.

**Requirement 0's calibration base:** n = 6 clean ncRNAs for the
length parameter (T-WT plus seekRNA ISEc11, ISKpn4, ISPst6, ISPa11,
ISEc21); n = 1 experimentally validated for `planted_m` distribution
(T-WT operating point). Durrant Supp Table 5's 6 IUPAC consensus
sequences add another 6 architectures via consensus sampling
(instantiate each consensus 100× using the position-wise IUPAC codes;
each instance gives a competitor curve). This 6-consensus expansion
is a follow-up step measurable in hours; combined with the seekRNA
architecture axes (guide position within NCR, match-segment
order/count/spacing, family sTIR presence), it completes the
architecture side.

Multi-site flanks from seekRNA's `AtaideLab/Targets` pipeline (days)
and Durrant's `BridgeRNA2024` pipeline (weeks; IS110-only, low
priority because IS110 is n=1 for architecture) remain deferred; the
6-system length + 1-system operating-point calibration is a
substantially stronger base than the earlier n=1, and it is what the
generator will train against. Building against a single anchor
without cross-family length calibration hardcodes IS110's specific
characteristics into the "reality" definition and reintroduces
exactly the class of overfitting that has been retracted 16 times
in this project.

### Diagnostic byproduct: architecture-axis stratification

Architecture randomization gives one measurement that is impossible
on any single-family real dataset: **detector performance stratified
by architecture axis.** For example, in a generator run:

- guide-position-in-ncRNA random vs fixed at 40% of length: if
  the detector shows different accuracy on the two, it is
  depending on a positional prior it should not have.
- number of guide match segments 1 vs 2 vs 3: if performance
  collapses when the segment count differs from IS110's canonical
  2, the model has memorized segment count.
- NCR upstream vs downstream of ORF: same test.

These stratifications become the pre-holdout diagnostics before the
IS1111 test in seekRNA. The IS1111 comparison then confirms whether
the synthetic diagnostic transfers.

## Retractions from prior versions

- **Coverage `33.85%` as a headline.** Replaced with variant-stratified
  breakdown: 20/34 on T-WT, 2/2 on 1_7bp_RTG, 0/31 on other 7 variants.
- **"87% single-nucleotide precision" quoted with peak-level denominator
  (23 peaks) and gold-aware tie-break.** Under gold-blind centroid the
  same 22 detections give 86.4% at `centroid_dist == 0` and 95.5% at
  `centroid_dist ≤ 1` — both numbers ship together, with the plateau
  three-tuple as the primary description.
- **PPV = 0.955 (21/22).** Correct at the Tnp-level, meaningful for
  per-detection localization; not meaningful as a "how many bridge
  RNAs work" number, for which the denominator is 2/9.
- **Mode 1 (min-E over L ∈ {9..12}) as recommended default.** Cannot
  reach L=14 for `7bp_RTG` variants. Retained only when the target
  task's L range is known to sit within {9..12}.
- **"TSD-partition (`flank_mask = "partition"`) available."** The
  Durrant corpus has no characteristic TSD (median pairwise
  max-match at ±5 nt around target boundary = 3.0 nt, at chance for
  p = 0.25); IS110 mechanism does not produce one. The 5 non-guided
  families in the corpus have TSDs but score against Durrant nc
  under Option E as double-null, so partition efficacy cannot be
  measured here. `flank_mask` remains a documented framework axis,
  but its practical value is not testable on this corpus.
- **p^5 quantitative attribution.** Fitted T-WT within 30% (predicted
  25.9, observed 20) but attribution to independence-across-sites is
  broken by the tie structure (competitor count ≥ 1 always on T-WT).
  Replaced by competitor-count formulation.
- **Requirement 0 in `≤ 10% within argmax ±5` form.** Tie-blind and
  gameable by lowering fidelity. Replaced by `fraction(competitor
  count ≤ 10) < 5%`, then further replaced by the two-parameter
  form below because the single-threshold form was still T-WT-
  derived and not architecture-aware.
- **Requirement 0 as `competitor_count / L ∈ [0.18, 0.25]`.** The A+
  cross-architecture measurement showed this rate is essentially a
  mathematical constant (analytically `≈ 0.262` from `2 × N_starts
  × P(Bin(L, 0.25) ≥ m)`), so requiring a generator to match it is
  vacuous. Absolute competitor count is the operative quantity for
  the S=k conjunction, and it varies with ncRNA length at fixed
  rate. Replaced by two independent constraints: `planted_m/L`
  distribution (T-WT-calibrated, n=1 experimental anchor) and ncRNA
  length distribution (6-clean-system-calibrated, 177–281 nt).
- **The 5-family FP measurement plan** (IS10-R, IS30, IS903, ISAjo2,
  ISLdl1 × Durrant nc): retired as double-null. Any per-family FP
  number derived from this construction has no meaning as a
  specificity statistic.
- **`shuffled null = 0.0226 per bag` as an absolute baseline.**
  Understates the null due to the 9-unique-nc duplication in
  Durrant. Historical `real / shuffled = 15.5×` becomes a lower
  bound rather than a point estimate.

## Provenance

- **Framework code:** `scripts/v5a_framework/{match_table,variant,
  metrics,cv,layers,flank_coherence}.py`,
  `scripts/v5a_framework/e_match_table.py`.
- **Regression test:** `scripts/v5a_framework/tests/test_tau0_anchor.py`
  (asserts 6 discrete counts on the historical baseline).
- **W9 recomputation:** `scripts/v5a_framework/tests/recompute_w9.py`.
- **Diagnostic phase evidence** (all in `scripts/v5a_framework/tests/`):
    - `a1_probe_is10r.py`, `a1p_probe_is10r_dedup.py` — Option E
      probe on IS10-R, duplicate-flank contamination and geometry
      diagnostics.
    - `a2_probe_endflank_motif.py` — the 9-unique-bridge-RNA
      finding.
    - `d5_discriminator_probe.py`, `d5b_argmax_discriminator.py` —
      discriminator reframe attempts, extremum-statistic failure
      diagnosis.
    - `d5c_v42_argmax_check.py`, `d5d_l_consistent.py` — V4.2
      argmax analysis at L=guide-length (D5c) corrected to L=11
      (D5d).
    - `d5e_competitor_count.py` — tie-robust competitor-count
      metric and Requirement 0.

Doc lives at `docs/channel_a.md`; render on demand.
