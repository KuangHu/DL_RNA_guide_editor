# Channel A — Joint-Significance Coherence Detector

A no-training, no-family-specific-tuning coordinate detector for
RNA-guided editors observed as ≥5 candidate insertion sites. Ships as a
stand-alone diagnostic; does not require the downstream V5A model.

## Method

**Mode 1 — min-E (recommended default).** For each transposase T with S
sites sharing one non-coding region (ncRNA):

1. For each nc position p and each L ∈ {9, 10, 11, 12}:
   - `hits(p, L, s_i) = m_max`, the max match count across all flank
     positions for that L-window starting at p in site i's flank.
   - `E(p, L, s_i) = N_windows(L) · P(Bin(L, 0.25) ≥ m_max)`.
2. `min-E(p, s_i) = min` over L of E(p, L, s_i).
3. `site_hit(p, s_i) = 1` iff `min-E(p, s_i) < 4.0`.
4. `Coherence(p) = Σ_i site_hit(p, s_i)`; optionally apply a Gaussian
   kernel over p with τ ∈ [0, 5].
5. Output = positions where `Coherence(p) = S` (all sites agree,
   kernel-smoothed).

**Mode 2 — m-threshold (higher-coverage alternative).** Fixed L = 11,
`m ≥ 8`, kernel τ = 1.

Both modes: no training, no family-specific hyperparameters, no length
threshold gate, no orientation prior. Peak-finding uses a local-max
rule (no strictly-greater neighbor within ±5 nt), which allows plateau
ties at τ=0 to emit multiple positions; τ=0 metrics report both peak
and plateau structure.

## Result on the Durrant benchmark (65 Tnps × 5 sites = 325 records)

### Detection resolution — the headline

For the historical baseline configuration
(`fixed_L11_m8`, τ=0, S=5, `tsd_handling=off`), on the 22 detected Tnps:

| statistic | value | denominator | correctness criterion |
|---|---:|---|---|
| Coverage | 33.85% (22/65) | N_tnp_total | detection emitted |
| Plateau width (median) | 1 nt | detected Tnps | — (descriptive) |
| Plateau width (mean) | 1.05 nt | detected Tnps | — (descriptive) |
| centroid_dist to gold (median) | 0.0 nt | detected Tnps | — (descriptive) |
| centroid_dist to gold (mean) | 1.02 nt | detected Tnps | — (descriptive; dominated by one 21-nt outlier) |

The plateau structure is the honest form of the "single-nucleotide
precision" claim. On 19 of 22 detections the detector produces a
1-nt-wide primary peak exactly at the annotated `gold_nc`; on one
detection it is a 2-wide plateau centered on gold; on one detection
the peak is at gold + 1 (biologically real 1-nt offset in an
RTG-variant Tnp); on one detection the peak is 21 nt from gold.

### Three correctness criteria, three counts on the same 22 detections

The numbers below use three DIFFERENT correctness criteria on the same
run and are NOT interchangeable — a reader tempted to reconcile them
arithmetically will get a wrong answer.

| number | criterion | value | denominator |
|---|---|---:|---|
| Tnp-level PPV | IoU([peak, peak+L), [gold_nc, gold_nc+L)) ≥ 0.5 | **95.5% (21/22)** | detected Tnps |
| contains_gold_frac | `gold_nc ∈ plateau_positions` (position-strict) | **90.9% (20/22)** | detected Tnps |
| exact_le_1 | `\|plateau_centroid − gold_nc\| ≤ 1` | **95.5% (21/22)** | detected Tnps |
| exact_eq_0 | `\|plateau_centroid − gold_nc\| == 0` | **86.4% (19/22)** | detected Tnps |

The three "detection is correct" counts (95.5% / 90.9% / 95.5%) differ
by one Tnp each, on the same 22 detections, because the criteria
answer different questions:

- **IoU ≥ 0.5** — is the emitted window's overlap with the annotated
  target above half. Coarse, standard for slot-match tasks.
- **contains_gold** — does the primary plateau STRICTLY include the
  annotated gold position. Fails when the plateau is one nt off, even
  if that one nt would be perfectly acceptable under IoU.
- **exact_le_1** — is the plateau centroid within 1 nt of gold. Under
  gold-blind centroid this is stricter than IoU on plateau-off-by-one
  cases and less strict than contains_gold on plateau-adjacent cases.

For the run at hand, one Tnp (`bag001`) has a width-1 plateau at
position 50 with gold at 49: IoU passes (window overlap 10/11),
contains_gold fails (49 ∉ {50}), exact_le_1 passes (dist = 1). One
Tnp (`bag000`) has a width-1 plateau at position 70 with gold at 49:
all three criteria fail. One Tnp (`bag010`) has a width-2 plateau at
{49, 50} with gold at 49: all three pass, but the centroid 49.5 fails
exact_eq_0.

The 95.5% (Tnp-level PPV) and 95.5% (exact_le_1) coincide accidentally
on this run — Tnp-level PPV counts bag001 and bag010 as correct, fails
bag000; exact_le_1 counts bag001 and bag010 as correct, fails bag000.
The three-way split becomes visible at any τ > 0 or any different
peak_min_dist.

### Exact-hit rate — four numbers, one table

| tolerance | detected-Tnp denominator | all-Tnp denominator |
|---|---:|---:|
| centroid_dist == 0 | **86.4% (19/22)** | 29.2% (19/65) |
| centroid_dist ≤ 1 | **95.5% (21/22)** | 32.3% (21/65) |

Under the older gold-aware "closest to gold" tie-break the same 22
detections gave 87% at exact position (20/23 peaks — plateau ties
emitted more than one peak). The 0.6 pp shift from 87% to 86.4%
is fully explained: one Tnp (bag010) has a plateau at nc positions
49 and 50, gold at 49; the gold-blind centroid is 49.5, so under the
strict tolerance == 0 that Tnp is no longer "exact." Under the ≤1
tolerance, it is. Both numbers ship together.

### Precision (IoU ≥ 0.5) on the same 22-Tnp detected set

| convention | value | 95% CI |
|---|---:|---|
| Tnp-level PPV (21 / 22) | 0.9545 | Clopper-Pearson [0.7716, 0.9989] |
| Peak-level PPV (22 / 23) | 0.9565 | Clopper-Pearson [0.7846, 0.9989] |

Tnp-level is the canonical convention for cross-τ comparison because
peak-level's denominator inflates at τ=0 (plateau duplication) and
deflates at τ>0 (kernel merge).

### Signal-to-noise vs. shuffled null

| null | rate per Tnp | ratio |
|---|---:|---:|
| Real Tnps (real flanks) | 0.354 | 1× |
| Shuffled Tnps (random Durrant flanks from other Tnps) | 0.0244 | 14.5× |
| Contamination-corrected (V1.d) | 0.0226 | 15.5× |

Under a matched analytic null (Bin(N=5, q) with q from the empirical
hit rate at random positions), the S=5 rate ratio is ≈ 84×.

### Confounder controls

- **V1' (external flanks):** mean-m at S=5 positions against 50
  dinuc-shuffled unrelated sequences = 6.57; at random control
  positions against the same external sequences = 6.66; Δ = −0.09.
  S=5 positions carry no intrinsic stickiness against non-related
  targets.
- **V1.c (composition):** Shannon entropy of the L=11 window at each
  S=5 position = 1.87 (median); ncRNA average = 1.80. S=5 positions
  are not low-complexity, not repeats, not compositionally biased.
- **V1''' (family background):** with target_flank_start-preserved
  and only non-target background dinuc-shuffled, ratio real / shuffled
  = 1.00×. Family-specific flank background contributes essentially
  zero to S=5 detection rate. The cross-family transfer claim is
  intact — but this null preserves the target segment, so it cannot
  rule out TSD confound directly; the TSD scope caveat below covers
  that channel.

## Applicability

**Required inputs:**

- ≥ 5 candidate insertion sites for the target transposase
- ncRNA (bridge RNA) shared across the transposase's sites — this is a
  biological requirement of the guide-target pairing mechanism, not a
  data convention. Verified on Durrant: 100% shared ncRNA across the
  5 sites of each of the 65 Tnps.
- Per-site target DNA (flank)

**Ship-ready domain:** IS110 relatives with the Durrant benchmark's
data structure — a serine-recombinase-like transposase family whose
element-carried bridge RNA specifies all insertion sites. Documented
performance stands.

**Cross-family transfer:** family-specific flank background contributes
zero to S=5 detection rate (V1''' 1.00× ratio). Channel A should
transfer 1:1 to any novel family with the shared-ncRNA architecture,
as long as `targeting_intact=True`.

**Boundary property:** the detector's power scales with the number of
observed sites; multi-site clustering must precede Channel A.

**Untested regimes (P1 / P2 in framework rebuild):**

- N_nc > 1 (Durrant is 100% N_nc=1; real IS elements are N_nc ≥ 2 in
  ~65% of V4.2 mining records). Same-region-across-sites constraint
  and per-nc search space scale in opposite directions; net effect
  unmeasured on Durrant.
- Real non-guided multi-site systems as negative controls. A 5-family
  pool (IS10-R, IS30, IS903, ISAjo2, ISLdl1; 2,763 non-guided Tnps
  with ≥5 physical insertions) is staged in
  `real_data/formatted/real_{fam}_sites.jsonl`. Under the framework's
  Option E measurement (`negative flanks × Durrant nc substrate`),
  per-family FP profiles will be reported alongside a `predicted vs.
  observed fp_hazard` scatter that extrapolates hazard to novel
  families. Until P2 completes, specificity is bounded by
  target-destroyed nulls only.

## Failure modes and scope caveats

- **Coverage bound:** ~68% of Durrant Tnps do not yield S=5 coherence
  under Mode 2 baseline. Those fall to Channel B (S=4 with local
  corroboration) or Channel C (per-site ranking with local model).
- **Coordinate assumption:** the current implementation uses exact
  absolute nc coordinates. For deployment on real guided families
  where the shared ncRNA is 95%+ homologous but not identical, use
  normalized-position or sequence-alignment coordinates (ε_norm ≈
  0.02 is the V4.2-calibrated tolerance).
- **TSD channel:** IS110 does not characteristically produce TSDs
  (composite RuvC + serine mechanism; strand exchange rather than
  staggered cut). Verified in the Durrant data: median pairwise
  max-match at ±5 nt around the target boundary = 3.0 nt, at chance
  for p=0.25. TSD confounds do not apply to this specific family, but
  a `flank_mask="partition"` variant is available for families that
  do produce TSDs (produces both an `S_all` and an
  `S_outside_TSD` score for each detection; never a filter).

## Deliberately excluded from Channel A itself

- Model training and family-specific tuning
- The taxonomy (`wrong_orient`, `different_region`, etc.) — Channel A
  produces coordinates, not slot-match against a decoy pool
- Length threshold gating — Channel A operates at a single specified L
  per run (multiple-L runs combine via Mode 1 min-E)
- Local-alignment ranking of within-region candidates — Channel C's
  responsibility

## Provenance

- Method: `scripts/v5a_framework/variant.py::spec_m_threshold_L11` (Mode
  2 baseline) and `spec_min_E_9_12` (Mode 1).
- Regression test: `scripts/v5a_framework/tests/test_tau0_anchor.py`
  (asserts 6 exact discrete counts on 65 / 22 / 23 / 22 / 21 / 21).
- W9 recomputation script:
  `scripts/v5a_framework/tests/recompute_w9.py`.
- Confounder-control diagnostics: `scripts/v5a_w8p_w9_w7p.py`,
  `scripts/v5a_v1_promiscuity.py`, `scripts/v5a_v1ppp.py`,
  `scripts/v5a_w10p_v1pp.py`.

## Retractions from prior versions

- "87% single-nucleotide precision" quoted with peak-level denominator
  (23 peaks): the number was under gold-aware "closest to gold"
  tie-break, which peeks at the answer to pick the peak on plateaus.
  Under the gold-blind centroid rule the same 22 detections give
  86.4% at centroid_dist == 0 (Tnp-level denominator) and 95.5% at
  centroid_dist ≤ 1. Both numbers ship together with the 3-tuple
  resolution.
- "PPV = 0.955" was Tnp-level; "PPV = 0.957" was peak-level. Both are
  correct measurements on the same run; different denominators.
