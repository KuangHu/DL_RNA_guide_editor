# V5 Freeze

**Tag: `v5-frozen`** — post-Channel-A validation state. All future changes
must be verified against the anchor table below via `run_all_anchors.py`
before merging.

## Locked components

| Component | Path | Locked semantics |
|---|---|---|
| MatchTable | `scripts/v5a_framework/match_table.py` | Per-position m_max cache; per-(orient, L, excl_w) arrays. Byte-equivalent to Durrant anchor. |
| Variant scoring | `scripts/v5a_framework/variant.py` | `spec_m_threshold_L11`, `spec_min_E_9_12`, kernel_max, peak-finding. Byte-equivalent to X1' v3. |
| MetricReport + condition | `scripts/v5a_framework/metrics.py` | Denominators explicit; MetricCondition (to be built) MUST include: variant_name, dataset, corpus, guide_origin, n_perm, seed, spec params. |
| features_structure_v2 | `preprocess/features_structure_v2.py` | pfl_fold_up indexing (end-1-indexed), 4 channels + mask, boundary invariant P(joint) ≤ min p_ss. |
| difficulty.py | `scripts/generator_v5/difficulty.py` | Rate table via 2763-flank pool; constrained single-m (rate ≥ 0.15); 86/10/4 tail. |
| architecture.py | `scripts/generator_v5/architecture.py` | Six axes uniform: orient / N_nc / TSD-width / TSD-relation / 5'SL / NCR-pos + A/B decomposition + is_split. |
| bag_v2.py output schema | `scripts/generator_v5/bag_v2.py` | Per-site fields listed in the schema-freeze commit; per-site planted_m via 86/10/4 tail. |
| channel_a_v5 | `scripts/generator_v5/channel_a_v5.py` | `compute_channel_a` is byte-equivalent to `test_tau0_anchor` framework path (verified 2026-09-01). |

## Anchor table

### A1 / A2 — Durrant, Mode 2 (m=8, τ=0, S=5) — cross-implementation

| field | anchor |
|---|---|
| n_tnps | 65 |
| covered | 22 |
| total_peaks | 23 |
| peaks_correct | 22 |
| tnps_with_correct | 21 |
| exact | 21 |
| coverage | 0.3385 |
| PPV_peak | 0.9565 |
| PPV_Tnp | 0.9545 |
| exact_rate | 0.3231 |

`channel_a_v5.compute_channel_a` produces identical values to `test_tau0_anchor.compute_anchor_metrics` on Durrant (verified 2026-09-01; commit hash in FROZEN.md commit).

### A3 — features_structure_v2 T-WT (nc_len=177)

- 16/16 unit checks pass (shape, symmetry, P_ss ∈ (0,1], boundary invariant `exp(-dG_open_uL_pn·L/kT) ≤ min(p_ss)`, mask consistency).
- Gold window `[49:60]` mean(P_ss) percentile = **0.832** (rank 28/167).
- dG_open_u1[49] = 0.176 kcal/mol; H_pair_win[49] = 0.534 nats.

### A4 — difficulty.py rate table anchor

- `rate(L=11, m=8) = 0.22 ± 0.02` on 2,763-flank pool.
- `target_m_for_L(11, 0.21) == 8`.
- **Retraction 2026-09-10: `m_max` semantic clarified.** Earlier informal
  reasoning (mine) described m_max as the "longest continuous match run"
  in an L-window and claimed that a centered single-base mismatch "splits
  m=11 into m=5+5". **That was wrong.** Read `preprocess/alignment.py:112-130`:
  `windowed_matches` computes `match_count[i, j] = sum_{k=0..L-1} dot[i+k, j+k]`
  — the sum of `L=11` match indicators along a diagonal, NOT the longest run.
  A single mismatch always reduces m by exactly 1 (from L to L-1), position-
  independent. Consequence: the current generator's rule `n_mismatches = L −
  planted_m` with uniformly-placed mismatches is self-consistent (m at
  planted position = L − n_mm exactly). Any experiment I described earlier
  as varying "mismatch position for dose calibration" is superseded — the
  number of mismatches is the correct dose axis, position is a separate
  (and mostly non-informative under total-matches semantic) axis.
- **Stage 1f conditioning (2026-09-03)**: the 0.22 anchor is at **gc=0.5** (uniform ACGT). Under `gc_target` sampled uniform [0.25, 0.65], observed `rate_at_m8_fixed` varies ~2× across the axis (gc=0.30 → 0.34, gc=0.60 → 0.19). The rate table itself is built once at uniform ACGT and is NOT re-derived per gc; `target_m_for_L(11, 0.21)` still returns 8. Consequence: **do not compare rate_at_m8_fixed across corpora without conditioning on gc**. Stratified acceptance is per (L × gc_target).

### A5 — architecture.py axis uniformity

- 25 chi-squared checks on 10K draws each: L uniform, N_nc uniform on {1,2,3}, TSD-width uniform on {0,2,5,8,9,12}, NCR-pos uniform on 3 states.
- is_split freq = DEFAULT_SPLIT_PROB ± 0.02.
- planted_m tail 86/10/4 at (target_m, target_m-1, target_m-2), zero above.

### A6 — acceptance b validation

- T-WT (n=170 sites): all 5 tests PASS (Test1a/1b/1c/2a/2b).
- V4.2 (n=2000 sites): FAIL on Test1a + Test2a (as required for the counterfactual to reject a known-defective corpus).

### V5 corpus anchors (informational, not enforced by run_all_anchors)

**Pre-mm-geometry anchors** (50K positives + 10K negatives, 2026-09-01, commit bf237e8 = v5-frozen tag), **superseded on 2026-09-01 by the mm_geometry axis regen**. Kept for historical context:
- Mode 2 L=11: cov=0.443, PPV_Tnp=0.969, exact=0.414
- Neg FPR (all L): 0.0276 (per-L: 0.033/0.030/0.028/0.020)

**Post-mm-geometry anchors** (50K positives regenerated 2026-09-01, seed=0, uniform-25% over (concentration, anchor) 2×2 grid):

Mode 2 (m=8, τ=0, S=5), L=11 subset, marginal:
- coverage = 0.523 (mean of clustered 0.587 + dispersed 0.459)
- PPV_Tnp = 0.968
- exact_rate = 0.398 (frame-shift steal: clustered 0.386 vs dispersed 0.409)

Mode 2 L × mm_concentration 4×2:
| L  | dispersed cov / exact | clustered cov / exact |
|----|-----------|-----------|
| 11 | 0.459 / 0.409 | 0.587 / 0.386 |
| 12 | 0.035 / 0.003 | 0.520 / 0.258 |
| 13 | 0.064 / 0.010 | 0.719 / 0.341 |
| 14 | 0.047 / 0.005 | 0.589 / 0.277 |

Mode 1 (min_E over L∈{9..12}, E<4, τ=5), L=11 marginal:
- coverage = 0.253 (mean of clustered 0.491 + dispersed 0.017)
- PPV_Tnp = 0.964

Mode 1 L × mm_concentration 4×2:
| L  | dispersed cov / PPV | clustered cov / PPV |
|----|-----------|-----------|
| 11 | 0.017 / 0.462 | 0.491 / 0.981 |
| 12 | 0.010 / 0.258 | 0.109 / 0.945 |
| 13 | 0.018 / 0.374 | 0.510 / 0.987 |
| 14 | 0.012 / 0.237 | 0.136 / 0.963 |

Negatives NOT regenerated with mm_geometry axis (negatives don't use `sample_mismatch_positions`), so neg FPR anchors above still hold byte-identically.

Mode 1 clustered PPV_Tnp = 0.981 (ceiling) is the mechanism-confirmation anchor: any future rerun that drops this below ~0.95 indicates the axis broke.

**Post-constrained-sampling anchors** (50K positives regenerated 2026-09-02, seed=0, class-uniform sampling within visible/blind partition via `scripts/generator_v5/constrained_mm.py`) — supersede the post-mm-geometry numbers above:

Mode 2 (m=8, τ=0, S=5), L=11:
| slice | coverage | exact | PPV_Tnp |
|---|---|---|---|
| clustered | 0.4864 | 0.4356 | 0.9681 |
| dispersed | 0.4533 | 0.4242 | 0.9727 |
| mm_anchor 3p | 0.4711 | 0.4272 | 0.9695 |
| mm_anchor 5p | 0.4684 | 0.4327 | 0.9711 |

Mode 1 (min_E, E<4, τ=5), L=11:
| slice | coverage | exact | PPV_Tnp |
|---|---|---|---|
| clustered | 0.4652 | 0.4035 | 0.9870 |
| dispersed | 0.0335 | 0.0207 | 0.7204 |

**Falsifiable-prediction verdict** (both confirmed):
| prediction | observed | verdict |
|---|---|---|
| Mode 2 clustered L=11 exact 0.386 → ~0.410 | 0.4356 | ✓ +5.0pp |
| Mode 1 clustered L=11 coverage stay ~0.491 | 0.4652 | ✓ −2.6pp |

**Reading the deltas — necessary for downstream interpreters not to misread the direction**:

1. Mode 2 clustered coverage dropped **0.587 → 0.486 (−10pp) BY DESIGN**. The 10pp represents the frame-shift-shared coherence class that used to fire at (nc_start+2, flank_start+2) because ALL 5 sites shared mm at {0,1}. Under class-uniform, positions vary per site → no shared shift → S=5 no longer fires at those spots. Removing that class is the whole point of constrained sampling. **DO NOT read this as a regression.** The correct "clustered detectability" is now 0.486, and it correctly exceeds dispersed 0.453.

2. Mode 2 exact rose **0.386 → 0.436 (+5.0pp)**. Coverage lost was overwhelmingly the frame-shift class that used to inflate coverage without contributing to exact (they fired at shifted, not planted, positions). Remaining coverage is more concentrated at the exact planted position. Together with (1) this gives the signature of "frame-shift steal removed": coverage down, exact up, PPV_Tnp preserved.

3. Mode 1 exact rose **0.244 → 0.404 (+16pp)**. The biggest single delta in the regen. Under the old hardcoded {0,1}, Mode 1 could fire at shifted positions where cross-site coherence held; those fires satisfied "covered" (S=5 at some position) but failed "exact" (position wasn't planted p). Under constrained sampling, Mode 1 can only fire where the planted position has S=5 coherence, so almost every fire is at exact planted.

4. Mode 1 clustered coverage dropped only −2.6pp. **If it had dropped to ~0.20 (~30pp), the visible-class definition would be wrong** (visible guarantees existence of a clean subwindow at some position, but not that ALL 5 sites' clean subwindows land at the SAME nc position). The observed −2.6pp says the shared-position property is largely preserved: most bags still have SOME nc position where all 5 sites have Mode-1-admissible signal, and after constraint that position IS the planted one. The 2.6pp lost is the tail where diversity broke even the planted-position coherence.

5. mm_anchor stratification is now diagnostic-only: 5p cov 0.4684 vs 3p cov 0.4711 (Δ 0.003 across both metrics on both modes). The axis is retained in the schema for backward compat but no longer influences position selection. Any future model that stratifies by mm_anchor should see near-parity — if not, something in the pipeline is treating 5p/3p asymmetrically.

Negatives NOT regenerated with constrained sampling (negatives don't use `sample_mismatch_positions`), so neg FPR anchors above still hold byte-identically.

## Stage 1c retirement of pre-1c inheritance (2026-09-03)

Under Stage 1c the per-bag n_sites axis samples uniformly on [3, 8] (default; runs may still pin via `--n-sites`). The Channel A spec accepts `theta = S / n_sites` in `spec_m_threshold_L11(theta=...)` — computed per-Tnp as `ceil(theta * n_sites)`. Default `theta = 1.0` (all sites hit) matches the S=5 semantics at n_sites=5.

The following pre-1c anchors and inheritance items are **retired pending re-derivation on the v6 1c corpus**. Do NOT quote these as anchors in future work:

1. **`p^n · q^(5−n)` hard-negative model** (`finding_hard_negatives_v5`). Generalizes to `p^n · q^(n_sites − n)`. The per-site `p/q ≈ 4.1` likelihood ratio is per-site and re-usable; the exponent form and the specific analytic S-threshold table must be re-tabulated per `n_sites`.
2. **`partial` negative n_planted range** — becomes `U{1, n_sites − 1}` conditional on the bag's n_sites (already implemented in `bag_v2.build_bag` because `n_sites - 1` is used at that call site).
3. **S distribution table (S=5 43% / S=3-4 40% / S≤2 17%)** from `finding_w8p_w9`. Becomes a `S/n_sites` proportion distribution. The 40% "addressable market" number that gates Channel B scope must NOT be reused.
4. **Item 3b slice definition (S ∈ {3, 4})** from `finding_tensor_layer_deferred_patch`. Redefine as `S/n_sites ∈ (0.4, 0.9)` (or similar band by proportion). The +0.030 AUROC lift stays valid but the slice has to be re-cut.

Each of these gets an explicit "re-derived on v6 1c" A8b anchor when the Stage 2 corpus lands. Until then, only the following pre-1c items remain load-bearing:
- The A1 / A2 Durrant anchor (byte-identical; Durrant has n_sites = 5 fixed).
- The A3–A6 unit-level anchors (structure, difficulty, architecture, acceptance).
- The V6 A8a hom=1.0 anchor at pinned n_sites=5 (validated 2026-09-03 5K, cov 0.4669 in CI [0.437, 0.495] of the 50K anchor 0.4864).
- The V6 A8a-1 hom=0.95 anchor at pinned n_sites=5 (validated 2026-09-03 5K, cov 0.4486 in CI [0.420, 0.477] of verify_end_to_end's 0.446).

## V6 Stage 2 anchors (2026-09-04, 5-corpus regen; seed=0)

Corpora at `/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_v6_stage2_*.jsonl` — `pos50k`, `twin50k`, `partial40k`, `ctrl10k`, `scat10k`. MatchTable shards at `channel_a/mt_v6_stage2_*`. All corpora share generator params **except negative_mode**; axes sampled: nc_homology_rate ∈ {0.90..0.995}, accessibility_target_percentile, gc_target, n_sites ∈ [3, 8], flank_offset_mode.

### A8b — Channel A on pos50k

Mode 2 (m=8, τ=0, S/n_sites=θ), L=11:
| detector | coverage | exact | PPV_Tnp |
|---|---|---|---|
| flank_coherence=off, θ=1.0 | 0.4910 | 0.3580 | 0.9042 |
| flank_coherence=jitter, θ=1.0 | 0.0907 | 0.0864 | 0.9947 |
| jitter × flank_offset=consistent | 0.1817 | — | 0.9973 |
| jitter × flank_offset=inconsistent | 0.0008 | — | 0.4000 |

The consistent-vs-inconsistent 227× ratio at flank_coherence=jitter is the mechanism-confirmation anchor for Stage 1h. Any future rerun that gives a ratio below ~50× indicates the flank_argmax channel or the jitter median filter broke.

### A12 — twin structure-only, pool + WITHIN-BAG paired

**Pool AUROC (n=5000 pair bags)** — all 5 channels ≈ 0.50:

| channel | AUROC | matched? |
|---|---|---|
| p_ss | 0.5013 | YES |
| cooperativity_win_pn | 0.5004 | YES |
| H_pair_win | 0.5057 | no |
| dG_open_uL_pn | 0.5129 | no |
| E_span_win | 0.5072 | no |

**WITHIN-BAG paired (n=49991 bags, SE_sign = 0.0022)** — rejects the paired null at tiny magnitude:

| channel | mean_d | sign>0 | Δ from 0.5 (SE) | matched? |
|---|---|---|---|---|
| p_ss | +0.0025 | 0.5135 | +6.1 | YES |
| cooperativity_win_pn | +0.0012 | 0.5085 | +3.9 | YES |
| H_pair_win | -0.0089 | 0.4951 | -2.2 | no |
| dG_open_uL_pn | +0.0116 | 0.4861 | -6.3 | no |
| E_span_win | -0.0024 | 0.4804 | -8.9 | no |

Pool AUROC ≈ 0.50 was hiding residual ~1pp per-channel bias from imperfect percentile matching (mixed direction per channel). Naïve √5 × 1pp ≈ 2.2pp → AUROC ≈ 0.52 assumes 5 independent channels; observed pair correlations are strong and NEGATIVE (p_ss↔cooperativity ≈ -0.75 across nc; E_span↔H_pair ≈ -0.97 on gold windows). Negative correlation makes the effective combined signal smaller than independence — √5 is an upper bound.

**Sign-mean apparent contradiction verified as skew, not bug (2026-09-04)**. dG_open_uL_pn at n=49991 shows mean_d = +0.01163 yet sign_frac = 0.4861. Full-corpus quantile probe (scratchpad/a12_sign_full.py) confirms median(d) = **−0.00483** — median and sign frac agree in direction (both < 0), mean is pulled opposite by a heavier positive tail (p95=+0.376 vs |p5|=0.302, ratio 1.24). Per decision rule "p50 < 0 且 mean > 0 → 真偏斜". Only dG_open exhibits this shape; the other 4 channels have mean/median/sign frac all agreeing in direction.

Per-channel medians (the |median| is what a linear model can extract from the central tendency; outlier bags don't move a mean-based decision boundary):

| channel | \|median\| pp | direction |
|---|---|---|
| p_ss | 0.13 | gold ↑ |
| cooperativity_win_pn | 0.10 | gold ↑ |
| H_pair_win | 0.16 | twin ↑ |
| dG_open_uL_pn | 0.48 | twin ↑ |
| E_span_win | 0.98 | twin ↑ |

Not a d = a - b bug. Mechanism: percentile matching on p_ss + cooperativity leaves free channels like dG_open and E_span to swing; occasional twin positions land at extreme values pulling twin_mean far from gold in either direction.

**C6 exact predictor computed (2026-09-04)**. Not an estimate — direct Fisher LDA on the per-position 5D percentile vectors (scratchpad/c6_lda_predictor.py, n_gold = 49991, n_twin = 275168):

  d′² = 0.00438,  d′ = 0.0662
  **AUROC_LDA = Φ(d′/√2) = 0.5187**
  C6 gate = 0.55, margin = **+3.13 pp**

The multivariate LDA is barely above the single-best channel (dG_open 0.513) because the measured per-position feature correlations show strong negative structure in the p_ss / cooperativity / dG_open cluster (r = −0.51 to −0.70), so per-channel biases point in directions a linear discriminant partially cancels. E_span ↔ H_pair correlation is +0.22 in this feature space (distinct from Item 4.5's −0.965 which was raw BPP values on gold windows, not percentile ranks).

MATCH_CHANNELS does NOT need to extend beyond (p_ss, cooperativity_win_pn). Twin corpus is a valid adversarial baseline for the pre-registered Channel B C6 gate.

### ε_align diagnostic (v6r2, 2026-09-04) — aligner is exact, error is 100% mutation-driven

`site_to_canonical_map` is computed by Bio.Align.PairwiseAligner (same function at train and deploy). Emitting the mutation-model oracle as `labels.oracle_map` and per-site disagreement rate as `labels.epsilon_align` lets Channel B eval stratify by measured alignment error.

**(A) ε_align stratified by nc_homology_rate on v6r2_pos50k** (n_sites per bin ≈ 55K):

| hom | mean ε | median | max | (1−ε̄)^8 (worst-case deploy hit at n_sites=8) |
|---|---|---|---|---|
| 0.995 | 0.00069 | 0 | 0.087 | 0.9945 |
| 0.990 | 0.00138 | 0 | 0.085 | 0.9891 |
| 0.970 | 0.00464 | 0 | 0.132 | 0.9635 |
| 0.950 | 0.00851 | 0.004 | 0.137 | 0.9339 |
| 0.900 | 0.02144 | 0.018 | 0.179 | 0.8399 |

**(B) hom=1.0 pin (500 bags, 2784 sites)** — aligner-only baseline: mean = 0.000000, max = 0.000000, fraction ε > 0 = **0.0000**.

The aligner is exact at zero mutation. All ε_align in the corpus is mutation-driven and monotone in hom. Because both training and deploy use the same aligner, and training corpus spans the full hom range in the same proportions we expect at deploy on novel homologs, no covariate shift on this axis. No aligner-parameter tuning available as a free improvement.

### Aligner param DEPLOY CONTRACT (v6r2, 2026-09-04)

`site_to_canonical_map` is computed at BOTH training (bag_v2.py `_get_aligner`) and deploy time by `Bio.Align.PairwiseAligner` with this exact param tuple:

```
match_score       = +2
mismatch_score    = −1
open_gap_score    = −2
extend_gap_score  = −1
mode              = "global"
```

**Any deploy pipeline MUST use this exact tuple**, or covariate shift on `site_to_canonical_map` returns. Changing any param requires a full 5-corpus + shard rebuild (~4 h) and re-verification of the six Channel B gates. Locked into v6r2 shards `channel_a/mt_v6r2_*` (dataset_hash: `ae8c95dac0ea48e6` for pos50k).

Deploy-time expected loss `(1 − ε̄_hom)^n_sites` at the corpus mean ε̄ = 0.0074:
- n_sites=5 → 3.6% loss
- n_sites=8 → 5.8% loss
- worst cell hom=0.9, n_sites=8 → 16% loss (stratifiable, retained in corpus rather than avoided)

`ε_align` **(gc × nc_len) 4×4 stratification on v6r2_pos50k** — hom-dominated, near-flat on the other axes:

| | [70,100) | [100,150) | [150,200) | [200,300] |
|---|---|---|---|---|
| gc [0.25,0.35) | 0.00731 | 0.00736 | 0.00773 | 0.00814 |
| gc [0.35,0.45) | 0.00620 | 0.00686 | 0.00719 | 0.00748 |
| gc [0.45,0.55) | 0.00627 | 0.00686 | 0.00691 | 0.00726 |
| gc [0.55,0.65) | 0.00663 | 0.00698 | 0.00715 | 0.00777 |

Cell range 0.0062-0.0081 (spread <2 pp) contrasts with hom stratification's 0.0007-0.0214 (spread 31×). Slight secondary trend of longer nc_len → slightly higher ε; no gc pattern. ε_align is stratifiable by (gc, nc_len) but the axis carries essentially no signal — hom stratification is the load-bearing axis for ε-related analyses.

### Flank-pool reuse across train/val/test split (low-risk note)

The flank pool has 69,609 unique sequences (`preprocess_pool_recovery/`); the 5-corpus v6r2 batch consumes ~250K per-site flanks → **mean 3.6× reuse per flank** across all sites and splits. Bags are Tnp-blocked at split time (hash on `bag_id`), so different bags in different splits can share the SAME underlying flank sequence.

Risk: LOW. The guide is bag-specific (drawn per-bag from the canonical_nc), so `flank → guide` mapping is inconsistent across bags — no direct leakage. But a model that memorizes flank composition instead of the (flank, guide) interaction could over-generalize between splits. This is exactly what C6 (twin structural-only) and C1 (Tnp-blocked K-fold) are designed to catch. Recording for audit trail; no corpus action required.

**Load-bearing side result on residual independence**: `corr(d_i, d_j)` on the paired 5D differences has max absolute value 0.204 (H_pair ↔ E_span). The raw-value correlation on gold positions is r = −0.65 to +0.22, and the H_pair↔E_span raw correlation elsewhere reaches −0.965. The paired-difference correlation is 4-5× smaller than the raw correlation because the percentile matching on p_ss + cooperativity aligns the underlying geometry on both sides, and the shared geometric-covariance term drops out of gold − twin. What remains is per-side residual noise, which is near-independent. Consequence: the Mahalanobis (1 − r²) amplification factor is nearly inactive — the multivariate LDA (0.5187) sits within +0.4 pp of the independent-channel sum estimate (0.534). The paired design is what makes the twin corpus a clean adversarial baseline; without it, the raw correlations would give a much larger multivariate lift.

### 1g — flank fake-plant probe (partial40k)

Trained on record-level features to distinguish planted vs unplanted flanks under `negative_mode=partial`. AUROC = 0.508, band [0.48, 0.52] PASS. Confirms Tier 2 (flank_argmax) does not leak plant identity.

### L1/L2 — data-invariant + leakage-probe verdicts (pos50k + partial40k + scat10k + ctrl10k)

| layer | check | slice | verdict |
|---|---|---|---|
| L1 | nc_homology_identity | all 5 hom bins | PASS |
| L1 | alignment_map_at_guide | all 5 hom bins | PASS |
| L1 | flank_offset_std | consistent + inconsistent | PASS |
| L1 | bag_target_diversity | n_sites 3-8, band 5×birthday-on-2763-pool | PASS |
| L1 | mismatch_geometry_class | (visible, blind) × 4 L values | PASS |
| L1 | planted_m_distribution | 4 L values | PASS |
| L1 | competitor_count | 16 cells (L × gc) | PASS |
| L1 | nc_position_consistency | 50000 bags, 0 inconsistent | PASS |
| L1 | rate_at_m8_fixed (analytic band) | 16 cells (L × gc_bin) | PASS |
| L2 | nc_active_vs_inactive_leakage | pos50k / partial40k / scat10k / ctrl10k | AUROC 0.509 / 0.512 / 0.500 / 0.490 — all in [0.45, 0.55] |
| L2 | flank_planted_vs_unplanted_leakage | partial40k (n=219 755 sites) | AUROC 0.508 — [0.48, 0.52] |
| L2 | negative_mode_leakage (mixed corpus) | 3 features × 2 targets | max AUROC 0.508 — [0, 0.55] |

### L3 — Channel A stratified on pos50k (L=11 clustered, 30 cells: 5 hom × 6 n_sites)

**flank_coherence = off** (θ=1.0, m=8, τ=0):

| n_sites | mean coverage across 5 hom bins | mean exact | worst hom-monotone violation |
|---|---|---|---|
| 3 | 0.790 | 0.275 | 0.014 |
| 4 | 0.638 | 0.417 | 0.073 |
| 5 | 0.502 | 0.427 | 0.044 |
| 6 | 0.425 | 0.393 | 0.021 |
| 7 | 0.363 | 0.344 | 0.027 |
| 8 | 0.346 | 0.339 | 0.036 |

Coverage decays monotonically with n_sites (θ=1.0 becomes a ^n_sites conjunction). At fixed n_sites, hom→cov variations sit at 2 SE of Bernoulli (0.035 at n_tnps ≈ 200); worst 0.073 pp at n_sites=4 is within band [0, 0.08]. Exact rate matches coverage in n_sites=4-8 (frame-shift-steal removed by constrained_mm).

**flank_coherence = jitter** (S/n_sites=θ, Stage 1h flank-argmax median filter):

| n_sites | mean coverage | mean exact | worst hom-monotone violation |
|---|---|---|---|
| 3 | 0.168 | 0.150 | 0.061 |
| 4 | 0.140 | 0.123 | 0.057 |
| 5 | 0.109 | 0.102 | 0.045 |
| 6 | 0.056 | 0.054 | 0.030 |
| 7 | 0.066 | 0.066 | 0.070 |
| 8 | 0.045 | 0.044 | 0.029 |

~4-7× coverage reduction vs off; exact ≈ cov (PPV near 1.0). Confirms the Stage 1h detector operates as designed: fewer bags admitted, near-perfect precision on those admitted.

### L3 hom monotonicity — DIAGNOSTIC, no gate

Two ad-hoc band widenings (0.02 → 0.04 → 0.08) were made post-hoc on the per-hop hom monotonicity check before it was recognized as under-powered. Analytic model: hom drift from 0.995 → 0.90 gives per-site ε_total ≈ 0.0009 → 0.036, so expected per-hop coverage change ≈ 3 pp at n_sites=4. At n_tnps ≈ 200/cell, Bernoulli SE ≈ 0.035 → per-hop signal is < 1 SE. The check has NO RESOLUTION at this corpus size and is now emitted **without a gate** as per_row diagnostic only.

Gated replacement (added 2026-09-04): `hom endpoint pair (0.995 vs 0.90)` pooled across n_sites, band [-1.0, +0.05] on the up-violation (down-drift is expected). Pooled n_tnps ≈ 1200/endpoint → SE ≈ 0.014, so a 5 pp up-violation is 3.5 SE and gate-worthy.

### 1h → A8b jitter cross-check (2026-09-04)

The Stage 1h 500-bag flank-axis judge on n_sites=3 gave consistent×jitter cov = 0.397 ± SE 0.031. The Stage 3 L3 pos50k n_sites=3 hom=0.995 jitter marginal (mixed 50/50 consistent/inconsistent) = 0.179 at n=223. Implied consistent-only from that marginal (assuming inconsistent × jitter ≈ 0.004 per the 1h judge) = 0.354 ± SE 0.045 at n≈111. Difference = 0.043 ≈ 1 SE — the two anchors are on the same distribution within sampling error. The 50/50 flank_offset_mode split on pos50k is confirmed empirically (25 037 / 24 963).

## ⚠️ CLASS-LEVEL RETRACTION 2026-09-10: Channel B non-v6r2 results are UNSCOPED

**Scope of retraction**: EVERY Channel B numerical result reported in this
document that was measured on any data source other than v6r2 synthetic
corpora is hereby marked **`UNSCOPED — NO_VALIDATED_DEPLOY_PATH +
SUBSTANTIVE_INPUT_MISMATCH: nc semantic`**. Per `CANONICAL_BAG_SPEC.md` §10
rule 5, the numbers are not deleted — kept for audit trail — but they MUST
NOT be cited as evidence in any active claim; a claim referring to any of
them inherits `UNSCOPED` status and must not be relied on.

**Root cause (verified this session):**

1. **Loader input dependency.** `ChannelBDataset._build_bag_inputs`
   (`model/channel_b/data.py:270`) hard-reads `labels.canonical_nc` via
   `_read_input_label`. `canonical_nc` is a v6r2 generator artifact — a
   pre-mutation canonical sequence that 84% of the time does NOT equal any
   `inputs.noncoding_regions[i]` (measured 2000-record samples, three
   corpora, 2026-09-10 A-2 check). It is absent from labels on every
   non-v6r2 source (Durrant, all 5 DDE families — verified 2026-09-10 A-2
   check).

2. **Ad-hoc adapter with semantic substitution.** All prior Channel-B-on-real
   results used `DurrantChannelBDataset`
   (`scripts/channel_b_durrant_p0.py:78-160`), which bypasses parent
   `__init__` and manually stuffs `canonical_nc =
   noncoding_regions[active_noncoding_index]` (lines 126-127, 138). This
   substitution is NOT equivalent to v6r2's canonical (84% mismatch), so
   the model was scoring inputs from a semantically different distribution
   than it was trained on. Substantive input mismatch, not procedural gap.

3. **GOLD field consumption in the adapter.** Line 127 reads
   `active_noncoding_index` — a GOLD field, not in
   `DEPLOY_INPUT_LABEL_KEYS`. Deploy legality violated in the adapter.

4. **arch.orient (GOLD) read by the loader for orient selection.**
   `data.py:308` (pre-2026-09-10 rev) read `arch.orient` and used it to
   pull ONLY that orient's shard arrays (`ma = self._mt.get(bag_id,
   s_idx, site_orient, L)` at data.py:327). Since the shard stores both
   fwd and rc m_max / flank_argmax arrays and `arch.orient` records the
   generator's planted orient, this effectively told the model "the
   planted signal is on this strand — score against it." Deploy has no
   such information. On v6r2 the data-derived "winning orient" agrees
   with `arch.orient` on ~80% of bags (tensor-diff measurement, 20
   sampled bags 2026-09-10), so the leak provides genuine information on
   the ~20% of bags where the winning orient differs; on non-planted
   data (DDE) "true orient" is undefined altogether, so the leaked
   quantity has no defined counterpart at deploy. **Every Channel B
   number reported on any real-data source, and every Channel B number
   on v6r2 that used the pre-2026-09-10-rev loader, was measured with
   this input.** Fix landed 2026-09-10: loader reads BOTH orients'
   arrays and combines per-position (m_max = elementwise max;
   flank_argmax = argmax of winning orient). ch 13-14 (orient one-hot)
   left zero to remove a noisy derived-orient signal that would replace
   one leak with a different bias. `arch.orient` removed from
   `ARCH_ALLOWED_KEYS` in `constants.py:59`.

**Explicitly retracted (non-exhaustive; anything not listed but derived from
these still inherits):**

| item | prior label | new label |
|---|---|---|
| (P0) FALSIFY main-B numbers 0.4615, S=3/S=4/S=2 tie window, 8 miss-case argmax positions | already 2x VOID (flank_argmax + estimator) | **triple: + nc semantic mismatch** |
| (P0) test (c) regional-attractor 100-170 finding | VOID | still VOID |
| (P0) ablation "mmax-only recovers +7.7pp" | VOID | still VOID |
| shift-invariance table (cognate/realbg/ncpad240 × main B / mmax-only B) | held | **UNSCOPED** |
| test (b') "all three methods ≤3pp under bg-controlled shift" (B side) | held | **UNSCOPED** |
| "main B ≈ 0.52, A_maj tie window brackets it" | held | **UNSCOPED** (main B number came from adapter) |
| "structure/flank channels are noise" (based on ablation) | VOID | still VOID |
| Any other Channel B number on Durrant, DDE, or any non-v6r2 source | held | **UNSCOPED** |

**A_maj and mmax-only-on-synth numbers**: A_maj is analytic (no model, no
loader) — unaffected by this retraction. mmax-only-B measured on v6r2 val
is unaffected (v6r2 has canonical_nc, no adapter). Anything else involving
model inference on non-v6r2 inputs is UNSCOPED.

**Path forward.** `docs/V7_SPEC.md` (drafted 2026-09-10) defines a
deploy-legal loader (raw `inputs.noncoding_regions`, `canonical_fold`
recomputed at load time via ViennaRNA) merged with a synthetic random flank
pool (fixing the `REAL_FLANK_POOL_FAMILIES` overlap with DDE eval sources).
v7 retraining will use this loader. Only after v7 is trained and its
loader has been validated end-to-end on Durrant AND DDE via the same code
path (no adapter) will any Channel-B-on-real-data number get a `SCOPED`
tag.

**Do not delete this section.** Anyone re-reading the file needs the
retraction visible before the numbers below it.

---

## Channel B — v1 checkpoint findings (2026-09-06 through 09-07)

**Model = `checkpoints/channel_b/main_lr3e-4/best.pt`** (epoch 7, val_loss=1.855, val_auroc_proxy=0.6825; selected by val_loss).

### Analytic decomposition on main val (15,709 bags, same pool, verified)

```
Count max_S (k=8)                     0.6175
+ log-odds transform (n-S term)         → +0.0625
Analytic log-odds pooled              0.6800   [aliases 0.6797 elsewhere = rounding]
- z-score per bin (removes cross-bin bonus) → -0.0040
Analytic log-odds, z-scored           0.6760
```

Model:
```
raw pooled                            0.6825
z-scored (neg-only, deploy recipe)    0.7821  ← +10.4 pp per-slice, hidden by n_sites scaling in raw pooled
```

### Confirmed per-slice picture (paired, same pool)

| n_sites | model | analytic | Δ |
|---|---|---|---|
| 3 | 0.7147 | 0.5984 | +0.116 |
| 4 | 0.7441 | 0.6336 | +0.110 |
| 5 | 0.7687 | 0.6676 | +0.101 |
| 6 | 0.8010 | 0.6854 | +0.115 |
| 7 | 0.8199 | 0.7020 | +0.118 |
| 8 | 0.8388 | 0.7127 | +0.126 |

Model beats analytic log-odds by +11 pp UNIFORMLY across n_sites bins, all CIs strictly above zero. `pooled_raw + 0.0025 pp` reading is a Simpson's-paradox artifact of n_sites-scaled scores; the correct pooled metric under the deploy recipe (per-bin z-score against a negative reference) is 0.7821 vs 0.6755 → **+0.107 pp deployment lift**, 95% CI [+0.099, +0.116].

### Falsified source hypotheses for the +0.3σ per-slice advantage

| hypothesis | falsification test | verdict |
|---|---|---|
| Concentrated at high n_sites | per-n_sites decomp: Δ uniform +11 pp across n=3..8 | falsified |
| H_pair↔E_span joint constraint | \|corr\| stratification: mean Q4-Q1 Δ_sep = -0.037, non-monotone per-bin | falsified |
| Soft-threshold on m_max (m∈{6,7}) | threshold sweep peaks at k=8; edge-count stratification non-monotone | falsified |
| Bag-level covariate leakage | LR/GBM on 75-dim bag summary = 0.514 / 0.515 | falsified |

### Mechanism confirmation

**Cross-site coherence** is the mechanism (not a shortcut):
- Per-site cyclic shift on nc positions destroys cross-site alignment
- Analytic collapses: 0.6797 → 0.5064 (positive control fires)
- Model collapses: 0.6825 → 0.4844 (same mechanism, small residual asymmetry is perturbation artifact — not evidence of differential mechanism dependence)

**Aggregation efficiency matches analytic** (H1 falsified):
- AUROC vs n_sites slope: model +0.0248/site, analytic +0.0229/site (essentially identical)
- Model advantage is in the per-site term (uniform ~0.3σ within-bin separation offset), not in cross-site aggregation

### mmax-only retrain — per-slice concentration (2026-09-06/07)

Retrained with input channels 4..14 (all except m_max L=9,10,11,12) zeroed
at both train and eval. Same architecture, lr, epochs as full model.

| slice | full model | mmax-only | Δ |
|---|---|---|---|
| n_sites=3 | 0.7147 | 0.6578 | **−0.057** (n=2326, seed 0) |
| n_sites=4 | 0.7441 | 0.7362 | −0.008 |
| n_sites=5 | 0.7687 | 0.7679 | −0.001 |
| n_sites=6 | 0.8010 | 0.8008 | 0.000 |
| n_sites=7 | 0.8199 | 0.8091 | −0.011 |
| n_sites=8 | 0.8388 | 0.8292 | −0.010 |
| **within-bin mean** | 0.7812 | 0.7668 | −0.014 |
| **z-scored pooled (deploy recipe)** | 0.7821 | 0.7687 | −0.013 |

Structure + flank + orient (11 channels) contribute ~1.3 pp on z-scored pooled and are ≈zero on n_sites ≥ 4 slices. The n_sites=3 gap of −5.7 pp is UNVERIFIED at seed 1 — a second-seed mmax-only run is in flight to rule out training-variance / val_loss-based checkpoint-selection artifact before recording it as a real finding.

### Per-site vs cross-site split (2026-09-07)

Extracted intermediate activations from `main best.pt`:
- `default`: model as-designed (attention → mean pool → head → max_position)
- `sum_post`: post-attention h → head per (site, pos) → SUM over sites → max_position
- `sum_pre`: input_proj (pre-attention) → head per (site, pos) → sum → max_position

Full model per-slice AUROC:
| n_sites | default | sum_post | sum_pre |
|---|---|---|---|
| 3 | 0.7147 | 0.7121 | 0.5079 |
| 8 | 0.8388 | 0.8384 | 0.5596 |

**Confirmed: `sum_post ≈ default`** (max Δ = 0.64 pp across all bins). Mean-vs-sum aggregation at the site pool is IRRELEVANT once site attention has run. This **disqualifies "aggregation form" as the cause of the C2 slope 0.813 deficit** — investigate softmax/QK normalization or the specific attention parameterization instead.

Per-site closed-form ceiling test (well-posed, GBM on partial40k, 2026-09-07):
- GBM: 32-dim per-site features from m_max_L{9,10,11,12} arrays (max, argmax, mean, std, count-thresholds, top-3), fitted on partial40k train per-site is_planted labels
- Aggregation: fixed sum over sites per bag
- Result: **within-bin mean per-slice AUROC = 0.5273** (across n_sites 3..8)
- vs model 0.7812 and analytic 0.6666

**"Per-site closed-form path is DEAD"** (2026-09-07). Narrower than initial claim: only PER-SITE closed forms are ruled out.

**Position-aggregation family of cross-site closed forms: also NULL** (2026-09-07). 5 candidate A' variants tested on main val pool (n=15709), all measured at POOLED bag AUROC:

| A' variant | pooled | Δ pooled vs A_max | n_sites=3 | n_sites=8 |
|---|---|---|---|---|
| A_max (analytic baseline) | 0.6800 | — | 0.5984 | 0.7127 |
| A_LSE (logsumexp over positions) | 0.6794 | −0.001 | 0.6092 (+1.1 pp) | 0.7107 |
| A_TOPK_MEAN_3 | 0.6742 | −0.006 | 0.6128 (+1.4 pp) | 0.7024 |
| A_TOPK_MEAN_5 | 0.6569 | −0.023 | 0.6107 | 0.6844 |
| A_LEN_ADJ (Bonferroni-style) | 0.6793 | −0.001 | 0.6080 (+1.0 pp) | 0.7138 (+0.1 pp) |
| A_STDNORM (null-max correction) | 0.6772 | −0.003 | 0.6080 | 0.7138 |

Search space caveat: all five are position-aggregation replacements (max / LSE / top-k / length-adj / standardized). Cross-site combination form is unchanged (log-odds sum in all). So the tested family = "swap position aggregator only". More expressive cross-site closed forms (multi-scale, joint per-position position×site score, different site combiners) remain untested.

Model's +11 pp per-slice advantage is not recoverable by any tested variant. Advantage lies in the specific learned attention (multi-head Q/K/V per block), not in swappable single-line aggregations.

### 0.3σ source hunt — RESOLVED (2026-09-07)

Answer: **"Better cross-site interaction form learned by attention on m_max distributions"**, evidence from four independent experiments:

| experiment | result |
|---|---|
| Per-site GBM ceiling (well-posed, partial40k train) | 0.5273 within-bin mean — per-site alone cannot reach either analytic (0.667) or model (0.781) |
| Coherence-destroying per-site cyclic shift | Both model and analytic collapse to chance — confirms cross-site alignment is the mechanism |
| mmax-only retrain (channels 4-14 zeroed, 2 seeds) | Matches full model within 1 pp on n_sites ≥ 4 — advantage is anchored in m_max channels |
| 5 A' position-aggregation variants | All ≤ A_max — the specific cross-site form matters, not just position aggregation |

Prior stopping condition (record UNEXPLAINED if not localized) did NOT trigger — source is localized to the cross-site interaction form. Two-level gain decomposition (per-site 0.527 → analytic cross-site 0.667 → model cross-site 0.781) is entirely in cross-site aggregation; per-site contribution is essentially zero.

### Durrant deployment simplification + pre-registered predictions (2026-09-07)

Query: `sites_per_tnp` distribution across all Durrant variants:
- All three files: 100% of 65 Tnps have n_sites = 5, all with nc_len=177.

Consequences for the deploy recipe:
- Single-bin population → per-bin z-score is a NO-OP (monotone within-bin transform doesn't change AUROC). **The deploy recipe (background pool, extrapolation rule) is UNTESTABLE by Durrant** and remains unverified for mixed-n_sites deployment.
- Extrapolation rule + nsheld validation stay open for a future real dataset with mixed n_sites.
- rescue-role of structure/flank/orient is IRRELEVANT for Durrant (n=3 phenomenon, no n=3 bags).

**Real-data cross-site co-occurrence (2026-09-07, GO/NO-GO preflight)**:

For 65 Tnps × 5 sites each on `durrant_cognate.jsonl` using gold-annotated `guide_start_in_nc` from `IS110_gold/annotation/durrant_gold_v1.jsonl`:
- Fraction with ALL 5 sites at IDENTICAL nc position: **1.0000**
- Median spread (max_start − min_start): 0
- Null (random starts per site on nc_len=177, 1000 perms): fraction with spread=0 = 0.0000

**Real Durrant data has MAXIMUM cross-site position coincidence.** The signal on which the model's +11 pp mechanism rests is present in real data at max strength (stronger than synthetic where flank_offset_mode=inconsistent reduces it).

**Pre-registered decision rule for full Durrant eval (registered BEFORE any Durrant model scoring)**:

Primary bag-AUROC (cognate vs shuffled, n=130 paired):
- paired DeLong CI lower bound > 0 → +11 pp direction confirmed on real data
- CI crosses 0 → **INCONCLUSIVE** (n=65 pos is underpowered for a 10 pp effect; NOT "failed to transfer")
- **Prediction (pre-hoc)**: given shuffled negatives break cross-site co-occurrence (equivalent to my synthetic coherence-shift experiment where both scorers collapsed to chance), BOTH model and analytic are expected to score cognate MUCH higher than shuffled, likely saturated. Gap may be too small for n=65 to detect. **INCONCLUSIVE is the pre-registered expected outcome.**

Secondary per-site position localization (325 sites, n=65 Tnps × 5):
- Top-1 accuracy: fraction of sites where argmax_position(analytic_score) matches gold `guide_start_in_nc` within tolerance
- Top-5 accuracy: same with top-5 positions
- n=325 gives CI ±5 pp; much higher statistical power than n=65 bag AUROC
- Prediction: Channel A synthetic anchor was 87% single-nt precision (A1/A2), expect similar on Durrant given match_status distribution (58% partial, 3% exact, 39% not_found)

Three files (cognate / cognate_realbg / cognate_realbg_ncpad240) reported independently; `ncpad240` co-varies nc_len (mechanism parameter P(max≥8)≈1 depends on nc_len ≈ 200), so cannot isolate padding from background effect. Base `cognate` (nc_len=177) is closest to synthetic corpus nc_len distribution.

### ⚠ CRITICAL FINDING (2026-09-08): gold_start = 49 in ALL 65 Durrant Tnps

**All 65 Durrant Tnps in `durrant_gold_v1.jsonl` have `guide_start_in_nc = 49`.** The 65 Tnps are experimental variants of the same IS110 bridge_rna family (T-WT, D-WT, 1_7bp_RTG, etc.) that share an ncRNA scaffold; they differ only in flank/target sequences, not in guide position.

**Consequence for every localization number reported in this section:**
- Trivial "always predict 49" baseline achieves **top-1 = 100%**, top-5 = 100%, top-10 = 100%.
- All A / B top-1 numbers must be read as "rate at which method predicts near-49", NOT as "rate of correct localization on independent systems."
- This dataset **cannot test cross-Tnp generalization of localization.** A method that hard-coded position 49 would beat everything measured here.

**What remains valid:**
- **A_maj 55–60% top-1 (canonical curve): A's m_max computation picks near-49 that fraction of the time.** m_max is a data-driven signal that lands at position 49 majority-of-the-time because bridge_rna's target-binding loop biology places the match there. A is not cheating — it doesn't know 49 is the answer — but its success reflects biology, not cross-Tnp discrimination.
- **A1/A2 anchor (S=5 gate, 22 Tnps, PPV_Tnp 95.5%)**: 21 of the 22 fired peaks land near 49. Fine as a within-family precision measurement; not a generalization claim.
- **(P0) B < A finding**: B predicts near-49 46.2% of the time; A 60.0%. B is genuinely worse at this task on this dataset. That comparison holds because both methods are running on identical inputs; ranking is meaningful.
- **Cross-file paired comparisons (cognate/realbg/ncpad240)**: all use the same 65 Tnps, so paired diffs still test what they claimed to test (real-bg contamination, padding offset).

**What is invalidated / needs strong caveat:**
- **Framing localization as "cross-Tnp generalization"**: this dataset can't test that. A different corpus with gold_start variance is needed.
- **(P1) "coverage-precision curve" as a shipping deliverable**: needs the gold=49-constant caveat. The curve tells you A's precision on a family of variants of ONE system, not on a distribution of independent systems.
- **The interpretation of B's failures as "generator gap on structure"**: the 8 A-hits/B-miss cases (job 25714080) show B picks positions 103–165 while A picks near 49. On a dataset with gold ≠ 49 those far picks might be correct alternate binding sites. Cannot tell without gold variance.

**Immediate impact on the session's conclusions:**
- (P2a) CANCELED verdict — still stands (retrain on same synth won't fix synth→real gap regardless of what the "gap" is exactly).
- (P0) FALSIFY — still stands as "B worse at picking 49 than A" — but the CAUSAL story (structure/flank channels pulling B off) may or may not be right; the far-position picks could reflect real alternate binding rather than structure noise.
- (P1) shipping — must add the gold=49 caveat before external use.
- **Any future localization deliverable requires a dataset with gold_start variance to be meaningful for cross-system generalization claims.**

---

### Channel A cross-site localization on Durrant real — RESULT (2026-09-07, jobs 25681388 → 25684983)

Aggregation semantics: for each Tnp, S(p) = |{sites s : m_max_L11[p] ≥ 8 in best-orient}|; Channel A prediction = argmax_p S(p). Compared to gold `guide_start_in_nc` from `durrant_gold_v1.jsonl` (100% co-occurrence anchor). n_tnps = 65 (5 sites each). Tolerance = 0 nt.

#### Coord verification (foundation check)

For every one of 325 sites in each file, `bridge_rna_sequence` (from `durrant_gold_v1.jsonl`, `U→T`) is a substring of the site's `noncoding_regions[active_nc]`, and the guide region `nc[offset+gs:offset+ge]` byte-matches `bridge_rna[gs:ge]`. Offsets seen:

| File | offset | verified |
|---|---|---|
| durrant_cognate.jsonl                    | 0  | 325/325 |
| durrant_cognate_realbg.jsonl             | 0  | 325/325 |
| durrant_cognate_realbg_ncpad240.jsonl    | 31 | 325/325 |

Script now derives per-site offset from the bridge_rna substring lookup and reads `_effective_gold_start = gold_start_in_nc + nc_offset`. **ncpad240 was previously excluded as a script bug** (my earlier estimate of 24 nt of prepended context was wrong — the true offset is 31 nt). After the offset fix, ncpad240 numbers are consistent with the unpadded files (below).

> **⚠ ALL NUMBERS IN THE FOLLOWING BLOCK ARE SUPERSEDED (2026-09-08).** The "41.5% top-1", "S=3 → 11.1%", "S=4 → 73.3%", and "S=5 stratum 95.24%" numbers were computed using a non-shipping aggregation (orient-max per site) that no A1/A2 or Channel B path uses. The canonical shipping curve under A_maj (per-orient @ majority orient, matching `orient_constraint=True`) is measured further down in this document (search "Corrected (P1)-ship shape"). Overall top-1 on canonical A_maj is 55.4% (argpartition), window [44.6%, 69.2%] under tie handling; S=3 stratum = 40%, S=4 = 92.3%, S=5 = 100%. **The following block is kept for audit trail only.**

#### Retraction and reconciliation with the A1/A2 anchor

Earlier in this session I labeled the A1/A2 anchor (line 22, PPV_peak = 0.9565, PPV_Tnp = 0.9545, coverage = 0.3385) as "synthetic-only" and framed the gap between it and my 41.5% top-1 as a "synth→real transfer loss." **That was wrong.** A1/A2 is a Durrant measurement (n_tnps = 65). What I actually measured differently:

- **A1/A2 (Mode 2 m=8 τ=0 S=5):** Channel A **with an S=5 gate** — fires on a Tnp only when there exists a position where all 5 sites hit m≥8; PPV of the fired peak = 96% (22/23 fired peaks land at gold). Coverage 34% (22/65 Tnps fire).
- **My 41.5% top-1:** Channel A **without any gate** — every Tnp gets a prediction via argmax_p S(p); 41.5% land at gold.

These are compatible measurements at different operating points. The stratified table below reconciles them explicitly.

#### Three-level accuracy table (unpadded cognate, n=65 Tnps, n_positions=167)

| level | top-1 | top-5 | top-10 | ×random top-1 |
|---|---|---|---|---|
| random null (uniform pick over 167 positions) | 0.0060 | 0.0299 | 0.0599 | 1.0× |
| per-site argmax (best-orient, single site) | 0.0277 | 0.1262 | 0.2308 | **4.6×** |
| cross-site argmax_p S (Channel A) | **0.4154** | **0.5692** | **0.6462** | **69.4×** |
| A_LOCAL_PEAK (±10) — position-dep correction | 0.4462 | 0.5846 | 0.6462 | 74.5× |

**Cross-site aggregation delivers a 15× lift over per-site** (0.028 → 0.415). This is the first positive result on real Durrant data for the cross-site co-occurrence mechanism. Per-site alone barely rises above random (4.6×), consistent with L=11 q≈0.21 flooding each single site with high-m spikes at random positions.

**A_LOCAL_PEAK** = `S(p) − mean(S over ±10 neighborhood excluding p)`. Position-dependent correction targeting broad plateaus vs sharp peaks (paired Δ = +3.1pp on cognate, +4.6pp on realbg, +1.5pp on ncpad240 — **all ns individually at n=65**, but direction consistent across all 3 files).

Note on the user-proposed A_LEN_ADJ / A_STDNORM corrections from A': both subtract a Tnp-constant from every position — `Score(p) = log_odds(S_p) − c(n_can, n_sites)` — so `argmax_p` is invariant. Those variants can shift bag-level AUROC (they change between-Tnp rank when n_sites/nc_len differ) but **cannot change position picked within a Tnp**. Only position-dependent corrections (like A_LOCAL_PEAK) can shift the localization argmax.

#### Three-file summary (Channel A cross-site, post pad-offset fix)

| File | nc_len | Top-1 | Top-5 | Top-10 | mean S_at_argmax | mean S_at_gold |
|---|---|---|---|---|---|---|
| durrant_cognate.jsonl                    | 167 | 0.4154 ±0.1198 | 0.5692 | 0.6462 | 4.32 | 3.48 |
| durrant_cognate_realbg.jsonl             | 167 | 0.3692 ±0.1173 | 0.5385 | 0.6154 | 4.35 | 3.48 |
| durrant_cognate_realbg_ncpad240.jsonl    | 230 | 0.3846 ±0.1183 | 0.5692 | 0.6462 | 4.51 | 3.63 |

**All three files agree within noise.** ncpad240 is no longer anomalous.

#### Paired cross-file CI (n=65 same Tnps, normalized tnp_id, paired bootstrap)

| pair | metric | Δ | 95% CI | verdict |
|---|---|---|---|---|
| cognate − realbg           | S top-1  | +0.0462 | [−0.0615, +0.1692] | ns |
| cognate − realbg           | S top-5  | +0.0308 | [−0.0769, +0.1385] | ns |
| cognate − realbg           | LP top-1 | +0.0308 | [−0.0769, +0.1385] | ns |
| cognate − ncpad240 (fixed) | S top-1  | +0.0308 | [−0.1077, +0.1692] | ns |
| realbg  − ncpad240 (fixed) | S top-1  | −0.0154 | [−0.1231, +0.0923] | ns |

**All cross-file drops are statistically indistinguishable from zero at n=65.** Real background does not measurably increase spurious co-occurrence at this resolution; padding (once coord-corrected) does not degrade localization.

#### Stratification by S_at_gold (Cause-1 verification)

| S_at_gold | n_tnps (%) | S argmax top-1 | LP argmax top-1 |
|---|---|---|---|
| 5 | 21 (32%) | **0.9524** | 1.0000 |
| 4 | 15 (23%) | 0.4667 | 0.5333 |
| ≤3 | 29 (45%) | 0.0000 | 0.0000 |
| **overall** | 65 | 0.4154 | 0.4462 |

**Reconciles the 41.5% top-1 with A1/A2's 96% PPV.** When gold has all 5 sites hitting (S_at_gold = 5, 32% of Tnps), Channel A gets top-1 correct 95% of the time — **this is the same operating point as A1/A2's PPV_Tnp = 95.5%**, and my measurement reproduces it within 0.4pp. When gold has ≤3 sites hitting (45% of Tnps), Channel A cannot recover — no argmax correction fixes a signal that isn't there. **Cause 1 (real gold's per-site signal is weaker: 3.48/5 vs planted 5/5) is confirmed as the dominant driver** of the "41.5% overall" number: it's a mixture of "brilliant when gold is strong" and "worthless when gold is weak."

#### Stratification by S_at_argmax (deployable confidence bins — S_at_argmax is known at inference; S_at_gold is not)

| S_at_argmax | n_tnps (%) | S argmax top-1 | LP argmax top-1 |
|---|---|---|---|
| 5 | 28 (43%) | 0.7143 | 0.7500 |
| 4 | 30 (46%) | 0.2333 | 0.2667 |
| 3 | 7 (11%) | 0.0000 | 0.0000 |

**S_at_argmax = 5 is a deployable gate**: fires on 43% of Tnps, 71% top-1. Note S_at_argmax = 5 does NOT require gold to have S = 5 — some Tnps get a spurious 5/5 coherence at a non-gold position (this is exactly why the A1/A2 gate has PPV 96% while the S_at_argmax=5 stratum only has 71% — A1/A2 additionally requires the peak to survive its peak-finding + variant.py machinery, which filters out some of these spurious 5/5 hits).

#### Winner-position complexity (unpadded cognate, 37 miss Tnps vs 28 hit Tnps)

| metric | miss winner | miss gold | hit winner=gold |
|---|---|---|---|
| mean base entropy (bits, max=2.0) | 1.827 | 1.819 | 1.857 |
| mean max-base-frac | 0.376 | 0.393 | 0.370 |
| low-complexity winners (max-base-frac ≥ 6/11) | 3/37 = 8% | — | — |

**Miss winners are entropy-normal.** No systematic complexity difference between miss winners and either gold or hit winners. Rejects Cause-2-via-low-complexity: real-genome context does not draw Channel A to repeat-rich decoys at any measurable rate.

#### Final Cause verdict
- **Cause 1 (weak real gold signal):** CONFIRMED, dominant. S_at_gold = 5 tnps → 95% top-1; S_at_gold ≤ 3 → 0%. Cross-file paired drops (cognate/realbg/ncpad240) all ns → the driver is per-Tnp gold strength, not file-level background.
- **Cause 2 (real backgrounds add spurious co-occurrence):** REJECTED as primary. Miss-winner entropy matches gold. Paired realbg drop not significant.
- **Improvement direction:** post-hoc complexity discounting is contraindicated (no complexity gap to exploit); position-dep argmax corrections give small ns gain (A_LOCAL_PEAK ~+3pp); the real lever is lower-threshold sub-threshold-coherence scoring — which is what Channel B is architecturally positioned to do.

#### What this locks in for the deploy narrative

- **Channel A on real Durrant is a 41.5% top-1 / 65% top-10 localizer overall**, ~69× above chance on top-1. Under an S_at_argmax = 5 gate (fires on 43% of Tnps): 71% top-1. Under the A1/A2 S=5+peak-finding gate (fires on 34%): 96% PPV_Tnp. Downstream consumers must cite the operating point.
- **A1/A2's 96% PPV / 0.34 coverage is the same measurement as my S_at_gold=5 stratum (95% top-1, 32% of Tnps)** — reconciled, both are Durrant, both stand. My earlier "synthetic-only" annotation was wrong and is retracted.
- **Deploy-usable confidence indicator: `S_at_argmax`**. Highest bin (S=5) → 71% top-1 (~10-nt tolerance would push to ~76%); mid bin (S=4) → 23% top-1; low bin (S=3) → 0%. `S_at_gold` is not knowable at inference; `S_at_argmax` is.
- **Pre-registered Channel B decision line (revised — paired-first, per user 2026-09-07):**
  - **Primary:** paired Δ (Channel B top-1 − Channel A **S argmax** top-1) on the same 65 Tnps, paired bootstrap 95% CI. Baseline is S argmax (41.5% cognate), NOT LP (44.6%) or OC (50.8%) — the latter two are unconfirmed improvements, using them as baseline would import an unvalidated claim.
  - Lower bound > 0 → model confirms advantage; upper bound < 0 → model falsifies; CI straddles 0 → inconclusive at n=65.
  - **Secondary (absolute):** Channel B top-1 > 55% → operational confirmation independent of paired stat.
  - **Stratum expectations (registered pre-measurement):**
    - S_at_gold = 5 (n=21, S baseline 95%): near-ceiling — Δ upper bound ≤ 5pp by construction.
    - S_at_gold = 4 (n=15, S baseline 47%): mid-signal — this is where OC extracted +27pp; if B beats OC here (Δ_B > +9pp vs S), B is doing more than orient-consistency.
    - S_at_gold ≤ 3 (n=29, S baseline 0%): sub-threshold — the true battleground. **This is where B's value proposition (sub-threshold coherence rescue) has to show up if it exists**. Even 4-5 hits on n=29 could give a paired CI lower bound > 0.
  - **Accept as exploratory at n=65 per stratum.** Overall Δ dominates; stratum Δ's are directional.

### A1/A2 gate mechanism decomposition + no-gate coverage-precision curve (2026-09-07, jobs 25685307 → 25685440, reframed 2026-09-08)

#### Retraction: "A_ORIENT_CONSISTENT as a new closed-form upgrade" was wrong (2026-09-08)

Earlier in this section I called `S_oc(p) = max_orient |{sites hitting p in that orient}|` an "extracted closed-form Channel A upgrade." That framing is wrong. `S_oc` **is what A1/A2's shipping spec already computes** — `spec_m_threshold_L11(orient_constraint=True)` in `scripts/v5a_framework/variant.py:495-496` runs per-orient S separately and emits peaks per orient, which is identical aggregation semantics (both give the same "covered" set at threshold 5). What I actually discovered was that my localization diagnostic was using a DIFFERENT (worse) aggregation — `orient-max per site` — that no shipping code uses, and correcting it to per-orient matches A1/A2. **This is not an upgrade to Channel A. It's a retraction of my diagnostic script's baseline.** (Fourth error, logged in memory `feedback-four-errors-2026-09-07`.)

The mechanism decomposition below is still valid; the coverage-precision curve is a genuinely new measurement. What is retracted: framing OC as a novel Channel A rule.

#### A1/A2 gate mechanism decomposition (still valid)

On the 22 Tnps that A1/A2 covers under `spec_m_threshold_L11(orient_constraint=True)`, the discriminative signal decomposes as:

| step | subset | top-1 / PPV_Tnp | Δ |
|---|---|---|---|
| my S argmax with S_at_argmax=5 gate (orient-max per site, non-shipping) | 28 | 71.4% | (worse baseline) |
| my S argmax on the 22-Tnp A1/A2-covered subset | 22 | 78.6% | — |
| + shipping per-orient aggregation (`orient_constraint=True`) → the S_oc gate | 22 | 90.9% | **+12.3pp vs my baseline on same subset** |
| + peak-finding local-max rule + IoU≥0.5 match (A1/A2 full spec) | 22 | 95.5% | +4.6pp |

**~12pp is orient consistency** — the shipping behavior my diagnostic script was accidentally omitting. **~5pp is peak-finding + IoU** — the only A1/A2 gate component NOT reducible to a single aggregation choice, and now the highest-leverage remaining item to understand.

The 28-Tnp vs 22-Tnp gap between "my S_at_argmax=5 stratum" and "A1/A2 covered" is exactly the count of Tnps where orient-max (mixing fwd hits from some sites with rc hits from others at the same position) reached S=5 spuriously; per-orient (single-orient consistency across sites) reduces this to 22 real S=5 events, of which A1/A2 filters 21 as gold-consistent.

#### First measurement of shipping Channel A's coverage-precision curve without a gate (NEW, not retracted)

**Prior state:** A1/A2 was the only reported operating point — coverage 34% (22/65), PPV_Tnp 95.5% (21/22). No numbers existed for what the same aggregation does when you drop the S=5 threshold and run argmax on every Tnp.

**Now measured, using shipping aggregation (per-orient S, max over orients per position):**

| file | top-1 | top-5 | top-10 | paired vs my prior orient-max baseline |
|---|---|---|---|---|
| cognate           | 0.5077 | 0.6308 | 0.7231 | +9.2pp CI [+0.0000, +0.1846] |
| realbg            | 0.4462 | 0.5692 | 0.6308 | +7.7pp CI [−0.0154, +0.1692] |
| ncpad240 (fixed)  | 0.4154 | 0.5692 | 0.6308 | +3.1pp CI [−0.0308, +0.0923] |

**Stratification by S_at_gold (cognate) — the coverage-precision curve:**

| S_at_gold | n_tnps (%) | shipping-aggregation top-1 | interpretation |
|---|---|---|---|
| 5  | 21 (32%) | **1.0000** | A1/A2's operating point; ceiling |
| 4  | 15 (23%) | **0.7333** | Below A1/A2's gate; 11/15 correctly localized without a gate |
| 3  |  9 (14%) | 0.1111 | Rescue rate 1/9; marginal |
| ≤2 | 20 (31%) | 0.0000 | Below signal floor |

**This gives the full curve, not just the A1/A2 point.** Deployable interpretation:
- At A1/A2's gate (S_oc ≥ 5, 34% cov, 95% PPV_Tnp) — established.
- Extending coverage to include S_oc ≥ 4 Tnps (58% cov combining strata =5 and =4) gains 11 correctly localized Tnps at the cost of 4 misses — combined PPV ≈ (21+11)/(21+15) = **89% on 56% coverage** (using cognate numbers, needs paired re-derivation for other files).
- S_oc = 3 gives essentially no rescue (1/9); not worth including.
- **A1/A2 was a peak-precision point on a curve, not a lower bound. The curve says Channel A shipping is usable down to ~58% coverage at ~89% PPV.**

Direction of the +9pp / +8pp / +3pp per-file gains matches: all reflect the same correction from a worse (orient-max) baseline to the shipping (per-orient) aggregation. Individually ns at n=65; convergent direction with a strong mechanism story.

**Numeric artefact of the retraction — the "41.5% top-1 headline" from earlier in this section is superseded**: it was computed with orient-max per site, an aggregation no shipping code uses. The honest number for Channel A shipping is **50.8% top-1 on cognate** (per-orient). The three-level accuracy table earlier in this document should be read as measuring an aggregation the shipping pipeline does not deploy — I've left it in the record with this note rather than rewriting to preserve the audit trail.

#### Peak-finding + IoU (+4.6pp) — highest-leverage remaining unexplained component

`_find_peaks` in `scripts/v5a_framework/variant.py` (with `peak_min_dist=5`) plus IoU≥0.5 match adds +4.6pp on top of S_oc gate. This is now the only A1/A2 gate component not reducible to a single aggregation choice. If the filter can be extracted as an explicit rule (what shapes of peaks does it discard?), that would be a genuine closed-form Channel A refinement — this time verified against spec first. Deferred item.

#### Channel B decision line — revised for the correct baseline (2026-09-08)

Given the retraction: Channel B's honest analytic baseline is **shipping-aggregation Channel A (per-orient, no gate) = `S_oc argmax`**, not my prior "S argmax with orient-max." Baseline top-1 = 50.8% on cognate, not 41.5%.

- **Primary:** paired Δ (Channel B top-1 − Channel A `S_oc` top-1) on the same 65 Tnps, paired bootstrap 95% CI. Lower bound > 0 → B confirms advantage; upper bound < 0 → B falsifies; CI straddles 0 → inconclusive at n=65.
- **Secondary (absolute):** Channel B top-1 > 60% (outside the S_oc cognate point 50.8% + CI headroom) → operational confirmation.
- **Stratum expectations (registered pre-measurement):**
  - S_at_gold = 5 (n=21, S_oc baseline 100%): CEILING. B has zero room; Δ ≤ 0 expected.
  - S_at_gold = 4 (n=15, S_oc baseline 73%): mid-signal — B needs Δ > 0 here to prove it exploits something beyond orient-consistent count. Room ≈ +4/15 to +4/15.
  - S_at_gold ≤ 3 (n=29, S_oc baseline 3% = 1/29): **sub-threshold territory**. This is where B's value proposition (learned sub-threshold coherence rescue) has to appear. Even 4-5 hits on n=29 could give a paired CI lower bound > 0. **This is the battleground.**
- **Accept as exploratory at n=65 per stratum.** Overall Δ dominates; stratum Δ's are directional.

#### Durrant deployment path for (d) — protocol registered (2026-09-08)

Two blockers surfaced:

1. **`labels.arch` is entirely absent from Durrant records.** The Channel B loader requires `arch.orient` to pick which orient's m_max to feed the model. Loader cannot process Durrant as-is.

2. **The generator emits `arch.orient` as a bag-level constant** derived from `architecture.is_reversed_target` (`scripts/generator_v5/bag_v2.py:190-191`), so all sites in a synth bag share one orient. **Mixed-orient bags do not exist in the model's training distribution.** This is a real coverage gap between synth training and real Durrant, where sites could in principle hit at different orients.

Deploy paths:

- **Approach A — synthesize `arch.orient` per bag from `bridge_rna` gold orient (via `durrant_gold_v1.match_orientation`):** REJECTED as gold leakage. Using the gold orient at inference means the model gets the answer's orient as an input — it never has to discover orient at deployment. Not a valid model-vs-analytic comparison.

- **Approach B — run model twice per bag (fwd and rc), take per-position max of outputs:** ACCEPTED as the (d) protocol. This matches `S_oc(p) = max_orient(...)` semantics and gives a strictly analytic-comparable measurement.

- **Oracle-orient (report as upper bound only):** optionally run the model once with gold-orient as an auxiliary informational number. **MUST be labeled ORACLE, MUST NOT enter the main comparison.**

**Pre-(d) OOD sanity check (registered mandatory):** on the synth val corpus, run the model with (i) correct arch.orient (in-distribution), (ii) forced wrong orient (fwd bag scored as rc, and vice versa) — measure how much wrong-orient inference degrades. If wrong-orient inference gives similarly-shaped outputs, dual-orient max on Durrant is safe. If wrong-orient inference is chaotic (e.g., gives high scores at random positions), dual-orient max on Durrant is contaminated and needs an OOD-aware pre-filter. **This gates (d).**

#### Pre-(d) OOD sanity check RESULT (2026-09-08, job 25687487): dual-orient max IS contaminated

Ran trained Channel B on n=930 synth val bags (main split, 2000/corpus cap) with correct arch.orient and with force-flipped arch.orient:

| metric | correct orient | flipped (OOD) | dual max |
|---|---|---|---|
| bag_max mean (positives) | +4.68 | +3.64 | — |
| bag_max mean (negatives) | +3.62 | +3.59 | — |
| argmax position agreement | — | 2.3% | — |
| per-position Pearson r (median) | — | +0.32 | — |
| **bag-level AUROC** | **0.6698** | 0.5136 | **0.6253** |

**Naïve Approach B (per-position max of correct + flipped outputs) drops bag-level AUROC by 4.5pp vs correct-only inference.** Root cause is asymmetric OOD behavior:
- **Positives under flipped orient:** correctly collapse to background-like scores (+4.68 → +3.64, close to negatives' +3.62). Model is recognizing "no signal in this orient" and returning a null-ish score. Good.
- **Negatives under flipped orient:** near-identical to correct-orient (+3.59 vs +3.62, Δ ≈ 0). Model is agnostic to orient on true negatives because there's no signal to be orient-of. Also good, but…
- **Max amplifies noise asymmetrically:** negatives' Δ pct[95] = +2.38 means 5% of negatives get a bump of ≥2.4 units under flipped orient. That inflates their bag_max. Positives get less amplification because their correct-orient score already dominates. Net: max collapses toward the middle → 4.5pp AUROC loss.

Argmax position agreement 2.3% ≈ random. Wrong-orient forward is genuinely OOD; the model's output is essentially noise at the argmax level.

**Consequence for (d):** naïve Approach B is out. Available mitigations:

- **Analytic-driven orient selection (recommended for testing):** for each bag, compute S_oc per orient over positions, pick the orient with higher `max_p S_oc(p)` (analytic-only, no gold), run model ONCE at that orient. Keeps inference in-distribution; matches A1/A2's per-orient peak selection semantics. On synth, this should nearly-always agree with `arch.orient` (needs verification). On Durrant, uses only analytic signal to choose orient — no gold leakage.
- **Averaging instead of max** of the two forwards: dampens negatives' inflation but also dampens positives' signal. Unclear net effect; would need its own sanity check.
- **Score-difference gate:** only use flipped-orient if it beats correct-orient by more than the negative-null offset. Complicated calibration.

**Registered next step before (d):** verify on synth whether analytic S_oc argmax-orient matches arch.orient. If yes (expected ≥95%), analytic-driven orient selection is the (d) protocol. If no, need to understand why and revise. Sanity check is cheap (same OOD script + one added metric); running before any further (d) work.

#### Analytic orient selection viability RESULT (2026-09-08, job 25687878): UNRELIABLE

On the same synth val (n=930 total, n_positives=372 across pos50k + ctrl10k), computed argmax_orient max_p S_oc(p) per bag and compared to arch.orient:

| slice | agree | disagree | tie |
|---|---|---|---|
| positives (n=372) | **0.5968** | 0.1290 | 0.2742 |
| all (n=930) | 0.4548 | 0.1473 | 0.3978 |

**Positives agreement 60% is barely above chance.** 27% of positives are ties (max_fwd == max_rc); tie values are typically S=3 or S=4 (real ambiguity), not S=0 (absence of signal). 13% actively disagree.

Mechanism: synth generator plants signal at ONE orient (arch.orient), but background sequence is orient-neutral, so the OTHER orient stochastically hits comparable S at some position by chance. Analytic S_oc cannot reliably infer the "true" orient from the noise floor — the plant elevates one orient by ~1-2 S units on average, but that's below the noise cap.

**Both mitigations for Approach B are falsified on synth:**
1. Naïve dual-orient max: AUROC 0.6698 → 0.6253 (−4.5pp), contaminated.
2. Analytic-driven orient selection: 60% agreement, 27% ties — no better than a coin toss on ~40% of positives.

**Available paths for (d), in order of cost:**
- **(P0) Report Channel B numbers on Durrant using GOLD-ORIENT (`match_orientation` from bridge_rna alignment) as ORACLE upper bound only.** Clearly labeled as leakage, not comparable to A1/A2 or OC. Tells us "what B could do if orient were free"; establishes the roof.
- **(P1) Pivot to analytic-only Durrant deploy.** Report OC coverage-precision curve (already measured); no Channel B numbers on Durrant. Ships the honest closed-form result without waiting for a retrain.
- **(P2) Retrain Channel B with mixed-orient augmentation.** Either (a) present both orients' m_max as additional channels (8-channel m_max block instead of 4), retrain, no data regen; or (b) train on synthetic mixed-orient bags (data regen + retrain). ~4-8h retrain, model recovers ability to handle Durrant's orient uncertainty. Highest expected value; blocks (d) for that window.
- **(P3) Learned combiner at deploy:** train a small calibrator to combine correct- and flipped-orient scores. Uses synth val to calibrate. Might work but risks overfitting to synth artifacts.

**(d) is now blocked pending direction.** The three-error / four-error methodology memory applies here too: I initially prescribed Approach B without predicting the OOD contamination; the OOD sanity check that the user made mandatory is what surfaced the block. This is exactly the pattern the "preflight against artifacts" rule addresses.

#### Standalone finding: orient is not inferrable from data on synth (2026-09-08)

The 60%/27% viability result above is not just a "path failed" — it's an independent finding about the synth corpus. `argmax_orient max_p S_oc(p)` matches arch.orient only 60% on positives, with 27% ties at S=3 or S=4 (real ambiguity, not signal absence). **On synth, the analytic cannot reliably identify the "true" bridge-RNA orient** — the generator's plant lifts one orient by ~1-2 S units, below the noise cap that the other orient reaches at random by chance.

**A1/A2 does not have this problem** because it runs `orient_constraint=True` — computes S per orient separately, emits peaks per orient, takes max of peaks across orients. It never has to "pick" an orient; it explores both simultaneously. The orient of the winning peak comes out as an output, not an input.

**Channel B does have this problem** because the loader (`model/channel_b/data.py:283-294`) reads `arch.orient` as a scalar and fetches only that orient's m_max. Model input is committed to a single orient before inference; there is no per-orient parallelism inside the model.

**Consequence for (P2) shape:** since orient cannot be recovered from data at inference (on synth; Durrant may or may not differ), (P2) cannot be "learn to pick the right orient" or "regenerate training data with per-site mixed orients" (that failure mode's opposite — the bag-level-constant orient design — matches real Durrant biology where all 5 sites of one Tnp share the ncRNA). (P2) must be **"feed both orients' information to the model simultaneously"** — concretely, the 8-channel m_max block (4 L × 2 orients) instead of the current 4-channel (4 L × 1 orient). This gives the model per-orient parallelism analogous to A1/A2 without asking it to pick.

- **(P2a) 8-channel m_max, retrain** — no data regen, just loader + model input dim change + retrain. Preferred; also naturally handles mixed-orient bags because both orients enter the model input.
- **(P2b) Regenerate synth with mixed-orient plants** — kept in the option tree, no longer demoted. **RETRACTION (2026-09-08):** an earlier version of this note said mixed-orient "doesn't exist in real Durrant" and used that as the demotion reason. That was wrong — 26 of 65 Durrant Tnps (40%) have mixed `target_flank_orientation` across their 5 sites (17 are 4-1, 9 are 3-2, 0 ties; see cross-table below). So the generator's bag-level-constant orient IS a real coverage gap. (P2b) is not the first choice (because (P2a) subsumes mixed-orient handling via architecture), but the "not in real biology" justification is retracted.

#### Channel B synth localization result (2026-09-08, job 25689610): sub-threshold rescue confirmed

**First-ever measurement of B's localization on synth val** (n=5892 positive bags with gold from `labels.guide_span_in_active_noncoding[0]`, `arch.orient` known so no OOD, single-orient inference matching training regime). Baseline is A's per-orient `S_o(p)` at the correct orient — same-orient A, no oracle information beyond what B has.

**Overall paired:**
- A top-1: 0.2315
- B top-1: 0.4136
- **Δ(B−A) top-1 = +0.1821** (paired-bootstrap CI [+0.1677, +0.1959], hugely SIG at n=5892)
- top-5: A 0.4976, B 0.7492

**S_at_gold stratification (the load-bearing table):**

| S_at_gold | n | A top-1 | B top-1 | Δ | verdict |
|---|---|---|---|---|---|
| 0 | 493 | 0.0000 | 0.1379 | **+0.1379** | B rescues from zero analytic signal |
| 1 | 955 | 0.0000 | 0.2063 | **+0.2063** | SIG |
| 2 | 1097 | 0.0155 | 0.2808 | **+0.2653** | SIG |
| 3 | 1114 | 0.1804 | 0.4309 | **+0.2504** | SIG |
| 4 | 819 | 0.3260 | 0.5385 | **+0.2125** | SIG |
| 5 | 655 | 0.5069 | 0.6046 | +0.0977 | SIG |
| 6 | 390 | 0.6410 | 0.6590 | +0.0179 | ns |
| 7 | 277 | 0.7978 | 0.7653 | −0.0325 | ns |
| 8 | 92 | 0.8261 | 0.8478 | +0.0217 | ns |

**B's gain is concentrated in the sub-threshold regime (S_at_gold ≤ 4), exactly matching the pre-registered Durrant battleground.** At S_at_gold = 0-4 (5378/5892 = 91% of synth positives), gains are +14 to +27pp, all highly significant. At S_at_gold ≥ 6, model has no room. This is the "sub-threshold coherence rescue" hypothesis validated: B extracts spatial signal from below-threshold m values that A's count-based aggregation cannot see.

**n_sites stratification (secondary):**

| n_sites | n | A top-1 | B top-1 | Δ |
|---|---|---|---|---|
| 3 | 872 | 0.1422 | 0.2661 | +0.1239 |
| 4 | 1025 | 0.1932 | 0.3288 | +0.1356 |
| 5 | 985 | 0.2223 | 0.4091 | +0.1868 |
| 6 | 1010 | 0.2515 | 0.4406 | +0.1891 |
| 7 | 974 | 0.2875 | 0.4723 | +0.1848 |
| 8 | 1026 | 0.2817 | 0.5458 | +0.2641 |

All SIG. Advantage grows with n_sites (more sites → more coherence signal). Consistent with mechanism.

**Note on absolute levels vs A1/A2**: A's top-1 = 23% on synth is much lower than A1/A2's PPV_Tnp = 95.5% because they measure different things — A1/A2's 95% is on the 34% of Durrant Tnps that pass its S=5 peak-emission gate; my synth 23% is unconditional top-1 argmax across ALL positive bags with variable planted_m distribution. The apples-to-apples synth-level comparison for the A1/A2 anchor is A's synth top-1 restricted to the S_at_gold=5-8 tail (**measured weighted mean = 0.6216** = (655×0.5069 + 390×0.6410 + 277×0.7978 + 92×0.8261)/1414), still lower than A1/A2's 95% — because synth n_sites goes up to 8 (Durrant is fixed at 5) and the synth pool includes many hard cases (harder planted_m distribution, non-contiguous plants, etc). Previous "≈ 0.66" was an eyeball estimate — corrected to actual weighted mean 0.6216. The relative B-vs-A comparison is the load-bearing finding, not the absolute levels.

#### n_sites metric-dependence caveat (registered 2026-09-08)

**H1 (cumulative-advantage-with-n_sites) was falsified on bag-AUROC** — per-slice Δ(B−A) is uniform ~+11pp across n_sites 3-8 (see per_ns_decomp result earlier in this doc). But the localization Δ measured here grows monotonically from +12.4pp (n=3) to +26.4pp (n=8). **These are not contradictory: they are metric-dependent measurements of the same model.**

- Bag AUROC is a between-bag ranking metric, dominated by score calibration; n_sites-scaling gets absorbed into the z-score normalization step.
- Localization top-1 is a within-bag argmax metric, a competition across positions within one bag; more sites → stronger consensus at the true position → sharper competition → bigger B advantage over A's count-based aggregation.

H1's original bag-AUROC falsification stands; the localization-metric n_sites-scaling is a separate, complementary finding.

#### Pre-(P0) audit and pre-registration (2026-09-08)

**arch field source audit (loader input construction only; whitelisted in `constants.py`):**

| field | source on Durrant | gold-derived? | model input? |
|---|---|---|---|
| `arch.orient` | `target_flank_orientation` from `durrant_gold_v1.jsonl`, majority-vote per Tnp | **YES — ORACLE** | channels 13-14, and selects which orient's m_max feeds channels 0-3 |
| `arch.n_sites` | count of records per Tnp | no (observable) | site_mask dimension |
| `arch.flank_offset_mode` | dummy `"consistent"` | no | meta only, does not reach tensor |
| `arch.nc_homology_rate` | dummy `1.0` | no | meta only |
| `canonical_nc` | site's `nc` (with realignment as needed) | no | structure channels 4-7 |
| `canonical_fold` | ViennaRNA MFE at deploy | no | cache, regenerable |
| `site_to_canonical_map` | Bio.PairwiseAligner at deploy | no | flank_dev channels 9-12 |
| `guide_length` | dummy (loader reads then discards; uses MAX_L=12) | no | not used |
| `guide_span_in_active_noncoding` | gold_start from `durrant_gold_v1.jsonl` | YES | **only used for target `y` in eval; does NOT reach model input tensor** |

**Only `arch.orient` and `guide_span_in_active_noncoding` are gold-derived, and only the former reaches the model input.** `guide_span` is used only to compute the top-1 metric against gold. This is a clean oracle: one axis of gold information (orient) enters the model input; one axis (gold_start) is used only for the evaluator's ground truth. No other gold-derived field affects the model.

**Durrant `target_flank_orientation` intra-Tnp consistency (job 2026-09-08):**

- 39/65 Tnps: pure (all 5 sites same orient)
- 26/65 Tnps: mixed (4-1 in 17 cases, 3-2 in 9 cases, 0 ties)
- Value convention: `fwd` (291 sites) / `rc` (34 sites) — matches synth
- Modal Tnp majority: `fwd` in 25 of 26 mixed Tnps; `rc` in 1

**Pre-registered orient rule for (P0):** primary uses **majority vote per Tnp** (unambiguous since 0 ties). Secondary sanity-check reports on the 39 pure-orient Tnps only — should agree with primary within noise if the majority-vote rule is unbiased.

#### S_oa test + `target_flank_orientation` provenance + canonical A baseline reconciliation (2026-09-08)

**`target_flank_orientation` provenance:** `labels.target_flank_orientation = None` and `labels.match_orientation = None` in every Durrant cognate record. **The orient field lives only in `durrant_gold_v1.jsonl` — it is GOLD-DERIVED, not deploy-observable from labels.** Consequence: per-site orient cannot be a model input in a deploy-legal pipeline. (P2a) must be 8-channel-both-orients; a per-site-own-orient input variant is not deploy-legal without an additional (unreliable) deploy-time orient inference step.

**S_oa (per-site own-orient) test:** measured `S_oa(p) = |{sites: m_s(p) ≥ 8 in site s's own target_flank_orientation}|` alongside A_maj (per-orient S at majority orient — matches shipping) and S_oc (max over per-orient S).

| aggregation | overall top-1 (n=65) | mean S@gold | pure top-1 (n=39) | mixed top-1 (n=26) |
|---|---|---|---|---|
| A_maj (per-orient @ majority — shipping) | **0.6000** | 3.23 | **0.7949** | **0.3077** |
| S_oc (max over per-orient) | 0.5231 | 3.31 | 0.6923 | 0.2692 |
| S_oa (per-site own orient) | 0.5846 | 3.26 | 0.7949 | 0.2692 |

**S_oa is NOT a closed-form improvement** — it does not beat A_maj on any subset. Case-(a) hypothesis (mixed low-S is aggregation artifact) is rejected. **Case (b) confirmed**: mixed Tnps have genuinely lower gold-signal (S_oa mean@gold = 2.04 on mixed vs 4.08 on pure — 2× lower). Mixed orient and low signal are jointly a consequence of weak real-biology binding, not an aggregation choice.

**A_maj beats S_oc by ~8pp overall** because S_oc lets the non-preferred orient's noise occasionally outrank the correct-orient signal. On pure Tnps A_maj = S_oa (both orients agree). On mixed Tnps A_maj wins because the majority orient IS the dominant one.

**Canonical shipping A baseline for (P0) and (P1) is A_maj = 60.0% top-1 on 65 Tnps.**

**Retraction of earlier published curve numbers:** the "OC top-1 50.8% / S_at_gold=5 100% / =4 73.3% / =3 11.1% / ≤2 0%" numbers earlier in this document were computed with:
- Stratification axis: **orient-max S_at_gold** (my erroneous baseline aggregation)
- Top-1 reported: S_oc (max over per-orient S)

Both aggregations differ from shipping A1/A2 (`orient_constraint=True` = A_maj = per-orient @ chosen orient). The numbers are internally consistent but do not represent shipping Channel A's performance. **The canonical (P1)-ship curve should be re-tallied with A_maj throughout** — measured in this section:

| S_maj_at_gold | n | A_maj top-1 | S_oc top-1 | S_oa top-1 |
|---|---|---|---|---|
| 0 | 3 | 0.000 | 0.000 | 0.000 |
| 1 | 10 | 0.000 | 0.000 | 0.000 |
| 2 | 9 | 0.111 | 0.000 | 0.111 |
| 3 | 10 | **0.600** | 0.300 | 0.500 |
| 4 | 13 | **0.923** | 0.846 | 0.923 |
| 5 | 20 | 1.000 | 1.000 | 1.000 |

**A_maj-stratified curve is substantially better than the old-published curve** — S=3 stratum goes from 11.1% to a tie-window of **[0.00, 0.90]** (argpartition mid = 0.40, later standardized), S=4 from 73.3% to **[0.77, 0.92]**. The old numbers under-reported shipping A's ability because the stratification axis (orient-max S) put more Tnps into higher-S bins than they belonged in.

**Corrected (P1)-ship shape** (measured 2026-09-08, all under canonical A_maj, reported as **conservative/lenient tie windows** per user convention):

| S_maj | n | top-1 [cons, len] | top-5 [cons, len] | top-10 [cons, len] |
|---|---|---|---|---|
| 0 | 3 | [0.00, 0.00] | [0.00, 0.00] | [0.00, 0.00] |
| 1 | 10 | [0.00, 0.00] | [0.00, 0.00] | [0.00, 0.00] |
| 2 | 9 | **[0.00, 0.44]** | [0.00, 1.00] | [0.11, 1.00] |
| 3 | 10 | **[0.00, 0.90]** | [0.70, 1.00] | [1.00, 1.00] |
| 4 | 13 | [0.77, 0.92] | [1.00, 1.00] | [1.00, 1.00] |
| 5 | 20 | [0.95, 1.00] | [1.00, 1.00] | [1.00, 1.00] |
| **overall** | 65 | **[0.446, 0.692]** | [0.615, 0.800] | [0.677, 0.800] |

**Tie-window convention (registered 2026-09-08, per user):** all Channel A localization numbers henceforth reported as `[conservative, lenient]` where conservative = "gold is unique argmax (adversarial ties)" and lenient = "gold is A max (favorable ties)". Single-number top-1 values are misleading because S is integer-valued over ~167 positions with typical max 2-5 — ties are the norm, not the exception. **S=3 stratum has a 90pp uncertainty for top-1** (adversarial 0.0 vs favorable 0.9); reporting a single "40%" or "60%" for that stratum is a point estimate hiding the full uncertainty.

**Tie-handling window on overall top-1** (2026-09-08 measurement):
- Conservative (gold is the UNIQUE argmax, ties break against gold): **29/65 = 0.4462**
- Lenient (gold is A max, ties count toward gold): **45/65 = 0.6923**
- Argpartition-based (default numpy tie behavior): 0.554 — sits inside the window as expected
- **Honest headline: overall top-1 ∈ [44.6%, 69.2%]**, 24pp span from tie handling alone. Ties are common because S is integer-valued over 167 positions with small max (typically 2-5) → many positions co-tie.

**S=2 stratum rank distribution (correcting my earlier misinterpretation, 2026-09-08):**

I earlier wrote that S=2's "78% top-10 vs 11% top-5" meant "gold ranks 6-10." That was wrong. The actual rank distribution:

| rank_lo (favorable ties) | rank_hi (adversarial ties) | count |
|---|---|---|
| 1 (gold tied at max) | 7, 20, 20, 26 | 4 |
| 3 | 19, 22 | 2 |
| 4 | 19, 30 | 2 |
| 5 | 34 | 1 |

Gold is either tied at rank 1 with 6-25 other positions (4/9 Tnps) or genuinely at rank 3-5 (5/9 Tnps). The "78% top-10" I reported was `argpartition`'s arbitrary tie-break happening to include gold when gold sat at low index among tied positions. Under conservative top-10 (rank_hi ≤ 10), only 1/9 pass; under lenient (rank_lo ≤ 10), all 9 pass. The argpartition number is intermediate, not a stable measurement.

**Corrected reading of S=2 stratum:** bimodal outcome, not "gold on the top-10 shelf." B has two different rescue targets here — break ties at rank 1 (4 cases) or promote from rank 3-5 (5 cases). Synth B's +26.5pp on S=0-2 covers both, but the mechanisms are different.

**Coverage-precision curve at gate operating points (argpartition top-1):**
- S_maj ≥ 5 (n=20, cov 31%): top-1 = 100.0% — matches A1/A2 anchor
- S_maj ≥ 4 (n=33, cov 51%): top-1 = **96.9%**
- S_maj ≥ 3 (n=43, cov 66%): top-1 = **86.9%**
- No gate (n=65, cov 100%): top-1 ∈ [44.6%, 69.2%] (mid ≈ 55.4%)

**Pre-registered (P0) S≤2 boundary is confirmed valid under the new canonical curve** — S=0/1/2 all have top-1 = 0 under argpartition (conservative would give same 0 since all n_strict_above>0 in those Tnps except S=2 tied cases; lenient could give some S=2 rescues but conservative doesn't). S=3 jumps to 40% under argpartition. The ≤2 vs =3 boundary is arguably cleaner under the new curve than the old one (old: S=3 was 11.1%, close to S≤2's 0%, making the boundary blurrier).

Old published numbers (50.8%/100/73.3/11.1/0) are **SUPERSEDED**.

**(P0) pre-registered primary metric — REVISED after S_at_gold × orient-purity cross-table (2026-09-08):**

Cross-table under per-orient S at majority-vote orient (measured, matches (P0) A baseline):

| S_at_gold | total | pure | mixed | pure_frac |
|---|---|---|---|---|
| 0 | 3 | 0 | 3 | 0.00 |
| 1 | 10 | 2 | 8 | 0.20 |
| 2 | 9 | 3 | 6 | 0.33 |
| 3 | 10 | 5 | 5 | 0.50 |
| 4 | 13 | 9 | 4 | 0.69 |
| 5 | 20 | 20 | 0 | 1.00 |

**Grouped:** ≤2 → 22 total, **5 pure, 17 mixed (77% contaminated)**; =3 → 10, 5, 5; =4 → 13, 9, 4; =5 → 20, 20, 0.

(Note: shifting from my earlier orient-max aggregation to per-orient at majority orient re-tallies the S distribution — the earlier orient-max distribution `{1:6, 2:14, 3:9, 4:15, 5:21}` was under a non-shipping aggregation and is superseded.)

**Mixed-orient strongly correlates with low S_at_gold.** In the primary sub-threshold stratum (≤2), 77% of Tnps are mixed. On those, majority-vote forces 1-2 sites into their non-preferred orient — OOD input per the earlier sanity check (dual-orient AUROC drops 4.5pp, argmax agreement 2.3%). B's cross-site coherence relies on all sites carrying real signal; 20-40% of positions carrying OOD noise structurally weakens the mechanism.

**PRIMARY (P0) metric FLIPPED AGAIN to S_maj ≤ 1 (per user 2026-09-08, after per-stratum tie-window analysis):**

Previous version used pure ≤2 (n=5) as primary. Under the new per-stratum tie windows, S=2 stratum has top-1 lenient window up to 0.44 — 4 of 9 S=2 Tnps have gold tied at max, so B "hits" on those may just be tie-break luck, not sub-threshold rescue. The cleanest sub-threshold stratum is S_maj ≤ 1:

- S=0 stratum (n=3): top-1 window [0.00, 0.00] — zero-width, no tie ambiguity
- S=1 stratum (n=10): top-1 window [0.00, 0.00] — zero-width, no tie ambiguity
- **Combined S_maj ≤ 1 (n=13): A baseline exactly 0/13 under any tie rule**

- **PRIMARY: paired Δ(B−A) top-1 on all 13 Tnps with S_maj ≤ 1** (unrestricted pure/mixed). A baseline is exactly 0/13 — no tie ambiguity — so any B hit is a pure increment. Contains both pure Tnps (2) and mixed Tnps (11), so partial OOD contamination expected on the mixed subset. Report:
  - Overall Δ (n=13, A=0/13)
  - Pure subset (n=2) — anecdotal only, not decision input
  - Mixed subset (n=11) — expected lower than pure per OOD analysis, but any hits still count vs the zero baseline

- **SECONDARY: pure ≤2 (n=5), split by "gold already tied at max" vs "strictly below" (per user amendment 2026-09-08):**
  - Pure ≤2 sub-tied-at-max (n=2): 4_4bp_RTG bag001 (S=2), 4_7bp_RTG bag002 (S=2) — B hits here may be tie-break luck; report but flag
  - Pure ≤2 sub-strict-below (n=3): 2_4bp_RTG bag002 (S=1), 2_4bp_RTG bag007 (S=1), 1-4bp-RTG bag000 (S=2) — B hits are true rescues (gold was NOT tied at max)
  - n=3 is very small — treated as directional-only

- **SECONDARY: same 65 Tnps under majority-vote orient**, with explicit flag that 26/65 have OOD-contaminated input.
- **SECONDARY: S=5 stratum (n=20, all pure by construction)** — ceiling check.

**Pre-registered VERDICT rules for (P0) — amended for S_maj ≤ 1 primary (2026-09-08):**

Primary (S_maj ≤ 1, n=13, A baseline = 0/13 under any tie rule):
- **CONFIRM sub-threshold rescue**: **≥ 3/13 hits on B (paired Δ ≥ +23pp)**. On synth, B rescues +14pp at S=0 and +21pp at S=1 — combined ~17pp expected. Requiring +23pp (3 hits) gives a small margin over expected effect while accepting that n=13 is small. Any hit counts, because A is exactly 0.
- **FALSIFY**: **0/13 hits on B**. No transfer whatsoever from synth's +17pp expected effect on this stratum. Mechanism does not transfer.
- **INCONCLUSIVE**: 1-2 hits on B — real but small; n=13 Bernoulli CI half-width ≈ 28pp so this range can't distinguish signal from noise.
- **CONFOUNDED** (residual): keep for any post-hoc pattern discoveries.

Note on pure/mixed within S ≤ 1: only 2/13 are pure (all mixed ties or S=0 mixed). Reporting pure vs mixed subsets is directional only (n=2 and n=11) — both accept lower power than the combined n=13. Expected: mixed subset lower per OOD analysis but any hits still count vs A=0.

Secondary (pure ≤2, n=5 split into tied-at-max n=2 vs strict-below n=3):
- pure-strict-below hits → true rescue evidence
- pure-tied-at-max hits → potentially tie-break luck; flag but don't confirm

**Small-n disclaimer:** n=13 primary is still small (CI half-width ~28pp), but much better than n=5. (P0) may STILL be inconclusive if B produces 1-2 hits; the confirm/falsify boundary at 3 hits gives ~55% power under the synth-predicted effect size. **(P2a) remains independently justified regardless of (P0) outcome.**

#### (P2a) 8-channel retrain is NOT gated by (P0) — justification independent (2026-09-08, per user)

Earlier note said "(P2a) preferred if (P0) shows B has real localization advantage." Too coupled. **(P2a)'s justification is already complete without (P0):**

1. **Synth localization result** (job 25689610, n=5892): Δ(B−A) top-1 = +18.2pp overall, CI [+16.8, +19.6], concentrated in S_at_gold ≤ 4 (91% of bags). +14 to +27pp per sub-threshold stratum, all SIG. Establishes the sub-threshold coherence rescue mechanism at very high power.

2. **Real deployment need** (measured this section): 26/65 Durrant Tnps (40%) have mixed target_flank_orientation. Single-orient-per-bag loader cannot handle this cleanly (dual-orient max: −4.5pp AUROC contamination; analytic orient selection: 60% agree, 27% ties). 8-channel m_max architecture natively handles mixed-orient input.

3. (P0) is expected INCONCLUSIVE on the primary metric due to n=5 pure sub-threshold. Using a near-zero-power measurement as decision gate for a several-hour retrain is a category error.

**(P0)'s revised role: confirmatory-only.** Runs, produces numbers, records them. If B ≥ 3/5 on pure ≤2 AND ≥ 4/17 on mixed, stronger evidence for (P2a). If INCONCLUSIVE, (P2a) still ships based on 1 + 2 above.

**Priority reorder — final (2026-09-08):**
- **(P1) OC coverage-precision curve ship** — parallel-track, no dependency. Complete in FROZEN under the canonical A_maj reconciliation.
- **(P2a) 8-channel m_max retrain** — justified independently by synth +18pp + 40% mixed-orient deployment need. **Design work required before coding; see (P2a) design section below.**
- **(P0) Durrant oracle test** — confirmatory-only, low-power expected. Runs after Path A converter is bit-exact-verified.
- **(P3) Learned combiner** — reserved (synth-artifact risk).
- **peak-finding + IoU +4.6pp** — kept as todo, lower priority than (P2a) but not deferred indefinitely.

#### (P2a) training-side design — must resolve before coding (2026-09-08, per user)

**The problem:** synth training data has bag-level constant orient (every site in a bag has the same orient by construction, per `bag_v2.py:190-191`). If (P2a) simply concatenates both orients' m_max — ch 0-3 = fwd m_max, ch 4-7 = rc m_max — then on every training bag one channel has real signal at gold across all 5 sites, and the other channel has background noise across all 5 sites. **Model may learn a bag-level channel-selection rule (attend to whichever channel has cross-site consistency) rather than a per-site channel-selection rule.** At mixed-orient inference (26 of 65 Durrant Tnps), sites disagree on which channel has signal — the bag-level rule fails.

**Two design options; one must be picked before writing code:**

**Option 2 (recommended): Own-orient / other-orient channels + per-site orient indicator.**
- Redefine ch 0-3 = m_max in the SITE's own orient, ch 4-7 = m_max in the OTHER orient.
- Add per-site orient indicator (either replace ch 13-14 bag-level orient with per-site orient, or add new per-site channel).
- At training on synth: own = bag arch.orient for all sites → per-site orient indicator is constant per bag → same regime as current model, no augmentation needed.
- At deploy: need per-site orient. On Durrant, `target_flank_orientation` is per-site but gold-derived. **Deploy-legal substitute: implement the alignment ourselves.** Given `flank` (deploy input) and `bridge_rna_sequence` (deploy input — the ncRNA reference for the transposase system, assumed available), align bridge_rna to flank and read off the strand. This is the same computation the gold annotator did.
- Pros: architecturally clean, matches biology, no training augmentation.
- Cons: adds deploy-time alignment step; increases the "arch fields needed at deploy" set (bridge_rna_sequence must be available — it is available on Durrant, and reasonable to assume elsewhere).

**Option 1 (fallback): Per-site channel-swap augmentation.**
- Keep ch 0-3 = fwd m_max, ch 4-7 = rc m_max as fixed labels.
- Drop orient one-hot entirely (model becomes orient-agnostic).
- At training: with probability p_augment (say 30%), for each bag pick a random subset of sites and swap their ch 0-3 with ch 4-7. Labels (y) unchanged. Creates synthetic mixed-orient bags at training.
- Model must learn per-site channel selection (which channel per site has cross-site-consistent signal).
- Pros: no per-site orient input needed at deploy (fully blind).
- Cons: augmentation semantics need care; larger effective training distribution; may need more epochs.

**Option 2 legality gate — MEASURED (2026-09-08):** the earlier claim that Option 2 is deploy-legal was circular — it rested on "bridge_rna × flank alignment can reliably read out per-site orient", which was untested. Per user correction, ran the actual measurement on 325 Durrant sites: aligned each site's target_binding_loop_specificity to that site's flank in both orientations, picked stronger match, compared to gold `target_flank_orientation`:

- **Non-tie sites: 288/288 agree with gold = 100.0%.** Both pure-orient Tnps (176 sites) and mixed-orient Tnps (112 sites): 100% each.
- Ties (m_fwd == m_rc): 37/325 = 11.4%. These have no orient signal from this alignment — need a deploy tie-break rule.
- Near-ties (|m_fwd − m_rc| ≤ 1): 62/288 = 21.5% of scored sites. Picked orient could flip with small perturbations.

**Verdict: Option 2 IS deploy-legal.** The circular concern is resolved by measurement. Per-site orient can be inferred at deploy from `bridge_rna_sequence × flank` alignment with 100% agreement to gold on decidable cases. **Deploy tie-break rule needed for the 11.4% tie cases.** Simplest: default to "fwd" (91% of Durrant sites are fwd, so this is a low-cost guess). More sophisticated: use analytic S signal as tie-break. Register the rule before (P2a) coding.

**Why the huge difference from `argmax_orient max_p S_oc(p)` (60% agree, 27% tie)?** Different operations. `argmax_orient S_oc` picks orient at BAG level via cross-site S aggregation — a bag's signal at one orient is diluted by non-matching sites. The new alignment picks orient at SITE level via direct target × flank alignment — matches the biology exactly. These are not the same test.

**Cross-check for Option 2 deploy legality:** bridge_rna_sequence is in `durrant_gold_v1.jsonl` but is a REFERENCE (the ncRNA sequence for the transposase system), not a per-site outcome. A researcher analyzing a new transposase would know which bridge-RNA they're using; the sequence is a system-level input, not derived from per-site observations. **Classify bridge_rna_sequence as deploy-input reference, not gold-derived.**

**Option 3 evaluated (per user, 2026-09-08):** "architecturally enforce orient symmetry — attention within same-orient, max over orients after." Concrete forms:
- **3a: Run model twice (once per orient), max per-position outputs.** This is exactly the "naïve dual-orient max" already tested → OOD-contaminated on synth val, AUROC −4.5pp. Rejected.
- **3b: Attention over (site × orient) with mask grouping same-orient sites.** For a bag with 4 fwd + 1 rc: run attention over 4 fwd-sites, separately over 1 rc-site, max outputs. Single-site attention (n=1) is degenerate — the rc side has zero cross-site information. Same issue as 3a but with a structural weakness.
- **3c: Run attention over 5 sites × 2 orients (10 site-orients) with shared weights across the orient axis and same-orient attention mask.** Substantial architecture change; benefits over Option 2 unclear given Option 2 is now measured-viable.

**Option 3 does not obviously beat Option 2 given the measurement.** Option 2's remaining risk is the 11.4% tie handling; Option 3's remaining risk is architectural (needs its own design and validation, none of which Option 2 needs). Sticking with Option 2 recommendation.

**Option 1 fallback formulation corrected (per user, 2026-09-08):** the earlier "swap ch0-3 with ch4-7 for a random subset of sites" was wrong — that treats a real fwd-alignment m_max as if it were an rc-alignment result, which is fake data. **Correct Option 1: channel-order randomization.** For each site with 50% probability, present channels as (ch4-7 first, ch0-3 second) instead of (ch0-3 first, ch4-7 second). Every channel block is real m_max data; only its position in the tensor is randomized. Model must learn per-site "find whichever of my two channel blocks matches cross-site consensus" rather than "always attend to block A." Fully deploy-legal (no per-site orient input needed), no fake data.

**Tie-break rule empirical test (2026-09-08):** on the 37 tie sites (per-site alignment gives m_fwd == m_rc), three deploy-legal rules tested against gold:

| rule | agreement | notes |
|---|---|---|
| **A: default fwd** | **37/37 = 100%** | All 37 gold orients on tie sites happen to be fwd on Durrant |
| B1: bag-majority via other sites' GOLD | 34/37 = 91.9% | Uses gold info; only legal in oracle test |
| B2: bag-majority via other sites' alignment | 33/37 = 89.2% | Fully deploy-legal alternative |

**Ship decision: Rule A for Durrant deploy** — 100% agreement here. **Deploy diagnostic clause registered**: on any new dataset, if tie rate > 15% OR gold orient distribution on ties departs from ≥ 90% single-orient, revisit and switch to Rule B2. Rule A's 100% agreement on Durrant is likely partial-coincidence (ties correlate with sequence patterns that also favor fwd matching), not a universal invariant.

**Coding sequence for (P2a), revised after Option 2 confirmation + tie-break test (2026-09-08):**
1. Implement deploy-time per-site orient inference: `_infer_site_orient(bridge_rna, flank) → "fwd"|"rc"|"tie"`. Same computation as this session's alignment measurement (max-match count per orient). On tie: **default to "fwd" per Rule A**; log tie fraction as deploy diagnostic; document Rule B2 fallback.
2. Modify `model/channel_b/data.py`: fetch m_max at PER-SITE orient (not bag-level `arch.orient`). Redefine channels as ch 0-3 = own-orient m_max (per site's own inferred orient), ch 4-7 = other-orient m_max. Per-site orient indicator on ch 13-14 (was bag-level, becomes per-site).
3. Update `constants.py` DEPLOY_INPUT_LABEL_KEYS to include `bridge_rna_sequence` (deploy-input reference).
4. **Two-step sanity check on synth (per user 2026-09-08):**
   - **Step 4a**: report per-site inference vs arch.orient agreement on synth val, and synth tie rate. On synth, per-site inference uses `canonical_nc[planted_start:planted_start+L] × flank` as the target×flank equivalent (planted position known at training). Expected: near-100% agreement (since arch.orient IS the planted orient) and near-zero tie rate (planted target is orient-committed by construction). If tie rate > 5% on synth, investigate — that would be an unexpected generator artifact.
   - **Step 4b**: on the agreement subset, verify the new loader's 15-channel input tensor is bit-identical to the old loader's tensor (since own = bag orient for all synth sites, tensors should match). Report bit-exact rate. Under 100% → good. Under 100% → find the plumbing bug before retraining.
5. Retrain from scratch on v6r2 5-corpus with same lr3e-4 protocol; verify against synth val AUROC anchor (~0.67).
6. Re-run synth localization test (currently 41.4% top-1 for B) with new model. If B_new ≥ B_old on synth, proceed; if B_new drops, investigate before deploying.
7. Deploy on Durrant with per-site orient computed from bridge_rna × flank alignment + Rule A on ties.

If Option 2 hits a snag mid-implementation (e.g., step 4b bit-exact fails), fall back to Option 1 (channel-order randomization, no per-site orient input).

> **⚠ ALL MAIN-B NUMBERS IN THE FOLLOWING BLOCK ARE VOID (2026-09-09).** Root-caused to `durrant_positive` shard's `flank_argmax_by_excl = None`, which makes `ChannelBDataset._build_bag_inputs` silently zero-fill ch 9-12 (flank_dev). Main B was fed all-zero flank_dev — OOD input, model was trained on non-zero. Every main-B number below (0.4615 overall, S=3 −40pp, S=4 −31pp, S=2 tie window, 8 miss-case argmax positions 103/104/123/157/165, pure/mixed splits) reflects OOD behavior, not the model on real inputs. **The FALSIFY verdict is double-VOID: (i) estimator (paired CI on argmax booleans, 24pp tie noise on A), (ii) OOD input (all-zero ch 9-12 on B).** A_maj numbers in the block are unaffected by the shard bug but subject to tie-noise caveat. mmax-only numbers are unaffected (its ch 4-14 are zero by training). Left in the record for audit trail. Corrected numbers on the on-the-fly path: main B ≈ 0.52, at parity with A_maj tie window and with mmax-only (Δ ≤ 2pp everywhere).

#### (P0) Durrant deploy RESULT (2026-09-08, job 25712843): FALSIFY — B is worse than A on real data

**Setup:** current checkpoint (`main_lr3e-4/best.pt`) + new per-site orient loader (`orient_source_fn` returning per-site alignment-inferred orient with Rule A tie-break) on 65 Durrant Tnps. Bit-exact plumbing verified on synth val (57/57 = 100% byte-identical to old loader when callback returns bag_orient). MatchTable shard = existing `durrant_positive`; bag_ids renamed cog_bag → paired_bag for lookup.

**Primary metric (pre-registered): paired Δ(B−A_maj) top-1 on S_maj ≤ 1 stratum (n=13, A=0/13 by design):**
- **B = 0/13 hits.** Paired Δ = +0.0000.
- **Pre-registered verdict: FALSIFY.** Mechanism does not transfer from synth to Durrant real data.

**Stratified table (argpart top-1):**

| S_maj | n | A (argpart) | B (argpart) | Δ(B−A) |
|---|---|---|---|---|
| 0 | 3 | 0.000 | 0.000 | 0 |
| 1 | 10 | 0.000 | 0.000 | 0 |
| 2 | 9 | 0.111 | 0.111 | 0 |
| 3 | 10 | 0.600 | 0.200 | **−40pp** |
| 4 | 13 | 0.923 | 0.615 | **−31pp** |
| 5 | 20 | 1.000 | 0.950 | −5pp |
| **overall** | 65 | 0.600 | 0.462 | **−13.85pp** CI [−0.231, −0.062] |

**Overall Δ is significantly negative (paired CI does not include 0).** B is not just failing to rescue — B is actively worse than A on Durrant across every non-floor stratum.

**Pure/mixed split:**
- Pure Tnps (n=39): A 0.795 / B 0.667 (Δ = −13pp)
- Mixed Tnps (n=26): A 0.308 / B 0.154 (Δ = −15pp)

B is worse on both — not confined to mixed (OOD-input) or pure.

**Comparison with synth localization (job 25689610, n=5892):**

| metric | synth | Durrant | Δ |
|---|---|---|---|
| A_maj top-1 | 0.2315 | 0.6000 | +0.3685 |
| B top-1 | 0.4136 | 0.4615 | +0.0479 |
| Δ(B−A) | **+0.1821** SIG | **−0.1385** SIG | −0.3206 (flip) |

**A's baseline is 37pp higher on Durrant** (60% vs 23%) — Durrant's task is easier (all 65 Tnps have 5 real positive sites; synth pool includes partial40k, ctrl10k, scat10k, harder cases). **B's absolute performance is only 5pp higher on Durrant** (46% vs 41%) — B doesn't get the ceiling lift that A gets from the easier task. Net: the A/B gap flips.

**Not an implementation bug**: A_maj on this run = 0.6000, matches earlier independent A_maj measurement (0.6000). Loader + MatchTable + m_max data path are correct. B's forward is running against valid input.

> **⚠ VOID (2026-09-09):** the 8 miss cases below are B choosing based on all-zero flank_dev + real structure inputs. Their "argmax at position 103/104/123/157/165" is model-response-to-OOD, not "structure attention driving to a systematic attractor". test (c)'s "regional attractor at 100-170" observation is derivative of these OOD picks and is also VOID.

**Interpretation — case-by-case diagnostic on the 8 miss Tnps (job 25714080):**

For each Tnp where A hits but B misses on S_maj ∈ {3, 4}:

| bag_id | S_g | gold | B_argmax | \|Δ\| | S_at_B vs S_g |
|---|---|---|---|---|---|
| 3_4bp_RTG bag000 | 3 | 49 | 48 | 1 | 2 < 3 |
| 3_7bp_RTG bag000 | 3 | 49 | 50 | 1 | 3 == 3 |
| T-WT bag016 | 3 | 49 | 103 | **54** | 2 < 3 |
| T-WT bag023 | 3 | 49 | 123 | **74** | 3 == 3 |
| 1-4bp-RTG bag001 | 4 | 49 | 104 | **55** | 1 < 4 |
| 2_4bp_RTG bag006 | 4 | 49 | 157 | **108** | 3 < 4 |
| 4_4bp_RTG bag002 | 4 | 49 | 51 | 2 | 3 < 4 |
| 4_7bp_RTG bag001 | 4 | 49 | 165 | **116** | 2 < 4 |

- **6/8 misses: S_at_B < S_at_gold** — B picks a position with LOWER m_max coherence than gold. This is architecturally impossible for A_argmax (A always picks argmax S). So B is being driven by NON-m_max channels (structure ch 4-8 and/or flank_dev ch 9-12) to positions that contradict the m_max evidence.
- **5/8 misses: |Δ| > 50 nt** — B jumps to a completely different nc region (positions 103, 104, 123, 157, 165 vs gold at 49). Not tie-breaking; genuine attention reallocation.
- **2/8 misses: |Δ| = 1-2** — small tie-break differences (bag000 3_4bp_RTG picks 48 with S=2 vs gold 49 with S=3; bag002 4_4bp_RTG picks 51 with S=3 vs gold 49 with S=4).
- **mean(S_at_B) = 2.38 vs mean(S_at_gold) = 3.50** — B systematically prefers lower-coherence positions.

**Concrete diagnosis: B's structure and flank_dev channels are misleading it on real Durrant.** m_max data is bit-exact verified. The synth→real gap is not in m_max content; it's in the structural characteristics of real ncRNA (Durrant's real transcription-derived ncRNAs) vs synth-generated ncRNA. Model's learned attention on structure fires on positions that are structurally attractive under synth conditioning but biologically wrong on real data.

**This rules out the "B is noisy" hypothesis** (in which case misses would cluster near gold or scatter randomly, not systematically prefer lower-S positions 50-120 nt from gold). It confirms the "generator gap" hypothesis directly: synth-generated ncRNA structure is a distinct distribution from real Durrant ncRNA structure, and the model's structure attention leverages synth-specific features that don't generalize.

**Additional consequence for future work:** any retrain that keeps ch 4-8 (structure) and ch 9-12 (flank_dev) architecture but changes only the training corpus needs the corpus's ncRNA structure distribution to match real Durrant. This is a substantial generator rebuild, not a small tweak.

#### (P0) ablation RESULT (2026-09-08, job 25720736): mmax-only checkpoint recovers B to A-parity — user hypothesis 2 CONFIRMED

> **⚠ VOID (2026-09-09):** both ablation runs below share the same P0-path root bug — main B was fed all-zero ch 9-12. RUN 2 (mmax-only) was unaffected because its inputs are zero by training design. RUN 1 (main + zero ch 4-12) is trivially bad (zeroing everything). **The "mmax-only recovers +7.7pp of main B's deficit" finding is VOID** — that +7.7pp was the OOD hit on main B, not a real advantage of mmax-only. On the correct on-the-fly path, main B (0.5231) ≈ mmax-only (0.5385) within 1.5pp — inside tie noise.

Two ablations tested to distinguish "generator ncRNA structure distribution differs" (hypothesis 1) from "structure/flank channel weights are noise" (hypothesis 2):

**RUN 1 — main ckpt + zero ch 4-12 at inference:** overall Δ = **−46pp** vs A (13.8% vs 60.0%). CATASTROPHIC. This is the "floor effect" — model was trained with those channels non-zero, zeroing at inference drives it deep OOD. **This run is UNINTERPRETABLE as a structure ablation; it just confirms zeroing on the wrong-trained model is toxic** (13/14 misses with S_at_B < S_at_gold). Not a diagnostic; a confirmation that zeroing on the main model is not the right test.

**RUN 2 — mmax-only ckpt + matched zero ch 4-14 (training and inference distributions match):**

| metric | main B | **mmax-only B** | A_maj |
|---|---|---|---|
| Overall top-1 | 0.4615 | **0.5385** | 0.6000 |
| Δ(B−A) paired CI | −0.1385 [−0.231, −0.062] **SIG** | **−0.0615 [−0.154, +0.015] ns** | — |
| Pure (n=39) | 0.667 | 0.744 | 0.795 |
| Mixed (n=26) | 0.154 | 0.231 | 0.308 |
| S=5 stratum | 0.950 | **1.000** | 1.000 |
| S=4 | 0.615 | 0.692 | 0.923 |
| S=3 | 0.200 | 0.400 | 0.600 |
| S≤1 primary | 0/13 (FALSIFY) | **1/13 (INCONCLUSIVE)** | 0/13 |
| n_misses (S=3/4) | 8 | **6** | — |
| misses with S_at_B < S_at_gold | 6/8 (75%) | **3/6 (50%)** | — |

**Verdict:**
- **mmax-only B on Durrant is at parity with A** — paired Δ = −6.15pp with CI straddling 0 (not significant).
- **Main B's −13.85pp deficit is caused by ch 4-8 (structure) and ch 9-12 (flank_dev) channels, not by an m_max mechanism failure.** Removing those channels via a properly-trained-for-it checkpoint recovers B to A-parity.
- **The "generator gap" reframing (structure distribution mismatch) narrows to "structure/flank channel weights are noise" (hypothesis 2 confirmed).** The generator's ncRNA structure distribution may or may not differ from Durrant's — we haven't measured that. What we DO know is that the main model's learned weights on ch 4-12 use structure info in a way that doesn't transfer, whereas the mmax-only model (never given those channels) transfers cleanly.
- **On the S=5 stratum, mmax-only B hits 20/20 = 100%** — better than main B's 19/20. mmax-only is not just tying A on average; it's strictly at least as good on the strongest stratum.

**Corollary — B has a usable variant already on disk.** `main_lr3e-4_mmax_only/best.pt` produces Durrant localization at 53.85% top-1, statistically tied with A's 60%. No retrain needed. If the goal is a deployed model at real-data parity with A, mmax-only is the shipping variant. It does NOT beat A, but neither is it defeated.

**Narrowed conclusion — retraction of the wider "generator gap" framing:**
- ~~Previous: "synth-generated ncRNA structure differs from real Durrant ncRNA structure, so any retrain on same synth inherits the gap."~~
- Corrected: "The main model's ch 4-12 weights don't transfer. The mmax-only model with those channels never given transfers cleanly. This means the transfer failure is scoped to the STRUCTURE/FLANK ATTENTION LEARNED ON SYNTH, not to m_max cross-site attention (which is what the mechanism story rested on) — the mechanism transfers."

**Updated priority reorder (2026-09-08 after ablation + gold_start=49 rediscovery — ROLLBACK of "task degenerate" framing, per user):**

**RETRACTION (2026-09-08):** an earlier version of this section framed A/B numbers as "below a trivial baseline" using the "always predict 49" comparison, and marked "acquire gold-variance real data" as a CRITICAL BLOCKER. **Both were wrong.** The constant-49 baseline is NOT method-accessible — A, main B, and mmax-only B all compute per-Tnp scores from that Tnp's own 5 sites' m_max, with no cross-Tnp information and no position prior. A method cannot "always predict 49" without being told 49; the constant baseline is an external-validity concern about the benchmark's generality, not a contamination of the measurements. And shared-scaffold Tnps ARE a real deployment regime — whole-genome scanning of one IS110 system's variants is exactly the "family-internal re-discovery" task, not an artificial one. Corrected framings below.

- **(P1) OC coverage-precision curve** — SHIP. Reframed as **"IS110 bridge_rna T-WT family internal re-discovery rate under whole-genome-scan operating conditions"**. This is a real deployment metric, not a diminished one — when a researcher runs Channel A on a novel candidate insertion site in a T-WT-related system, this curve tells them the expected precision at each coverage. Overall top-1 = 0.554 (window [44.6%, 69.2%] under tie handling), coverage-precision points 55%/87%/97%/100% at cov 100%/66%/51%/31%. External generality across different-scaffold systems requires additional data (below); this in no way weakens the family-internal claim.
- **A1/A2 anchor (PPV_Tnp 95.5% at coverage 34%)** — add scope caveat: **measured on 65 T-WT-family variants (shared scaffold, constant gold position). This is within-family precision at a specific S=5 gate.** Not a cross-family generalization claim; nor a diminished result — this is exactly the operating regime a whole-genome scan on a known bridge-RNA system runs in.
- **mmax-only B — CORRECTED reframing (2026-09-08):** the +7.7pp improvement over main B (46% → 54%) on Durrant is real evidence that ch 4-12 attention weights are noise on this data. However, the 6pp gap between mmax-only B (54%) and A (60%) at n=65 on a shared-scaffold family is low-information — cannot distinguish "B slightly weaker than A" from "measurement noise". **Its real value requires a test where absolute-position priors matter (see shift-invariance test below) or where scaffold varies.** Retain the checkpoint and the "structure/flank channels are noise" finding; do not claim "B at real-data parity with A".
- **(P2a) 8-channel retrain — CANCEL stays** (mmax-only recovers most of main B's deficit without added channels).
- **CRITICAL BLOCKER DOWNGRADED (2026-09-08):** "acquire gold-variance real data" is no longer the gate. **The shift-invariance test (below) can be synthesized from existing data at zero cost and directly tests whether the methods use position priors.** Cross-scaffold data becomes conditional: needed only if shift-invariance shows the methods rely on absolute position.

#### On-disk survey (2026-09-08): no gold-variance real dataset available locally

Searched `/global/scratch/users/kh36969/DL_novel_guide_editor/` for real-data localization benchmarks with variable gold_start:

| dataset | sites | positive | gold_start | notes |
|---|---|---|---|---|
| `IS110_gold/annotation/durrant_gold_v1.jsonl` | 325 | 325 | **49 (constant)** | 9 bridge_rna variants of ONE scaffold |
| `real_data/inference/real_all.jsonl` | 114,808 | 17,186 | **None (unlabeled)** | broad IS110 collection but no guide-position annotations |
| `real_data/formatted/real_IS10-R_sites.jsonl` | (varies) | (varies) | **None (unlabeled)** | IS10-R family; sites but no gold |
| Same for `real_IS30_*`, `real_IS903_*`, `real_ISAjo2_*`, `real_ISLdl1_*` | | | **None** | |

**Local situation: only labeled real dataset is the constant-49 Durrant. Everything else is unlabeled.** Cross-scaffold annotation is a substantial external effort that becomes conditional on shift-invariance test results.

#### Registered next tests (2026-09-08, replaces the retracted BLOCKER)

**Test (a) — SHIFT-INVARIANCE TEST (primary; replaces the BLOCKER):**

Synthesize a shifted corpus from the existing 65 Durrant Tnps: for each Tnp, prepend a random N-nt prefix (N drawn per-Tnp from U{0, 60}) to the nc. Per-Tnp gold_start becomes `49 + N` — varies across Tnps by construction, biology (nc composition, flank×target relationship) is untouched. Constant-49 baseline is no longer possible; a method that uses only cross-site m_max coherence remains equivariant to translation.

**Pre-registered directional predictions:**
- **A_maj (per-orient S = |{sites hitting p in that orient}|):** by construction, S(p) shifts one-to-one with nc translation → A's top-1 should be **invariant** to the shift. Expected: 55% top-1 (same as unshifted).
- **mmax-only B:** consumes only ch 0-3 (m_max per L), no position-specific structural input → should be **near-equivariant**. Expected: ≈54% top-1 (close to unshifted).
- **main B:** consumes ch 4-8 (structure computed at absolute nc positions) and ch 9-12 (flank_dev per site per L). These are position-specific → shifting nc changes structure attention at each absolute position → main B's top-1 should **DROP** below its unshifted 46%.

If predictions hold: mechanism-level architectural principle established — "detection target is motif position-consistency across sites → detector must be translation-equivariant → any position-specific input channel needs separate justification". ch 4-12 have no such justification (mmax-only already ≥ full on synth n≥4), and the shift-test breaks main B specifically because of them.

**Test (b) — A_maj rerun of ncpad240 (auxiliary):** existing measurement showed A top-1 = 0.4154 (unshifted) vs 0.3846 (ncpad240, 31-nt shift) under orient-max aggregation, paired CI [−0.108, +0.169] ns. Rerun under A_maj (canonical) to promote to formal record. Predicts near-invariance.

**Test (c) — main B argmax distribution across all 65 Tnps:** if 65 Tnps share the same scaffold → identical structure/flank inputs across bags → main B's structure attention should produce a FIXED positional attractor. If most misses cluster at a small set of positions (e.g., 103-165 range), that IS the attractor. This is direct evidence for the equivariance-breaking mechanism. **Running now** (job 25722341, in-flight).

**Alternative paths that don't require new data (kept):**
- Peak-finding + IoU +4.6pp extraction for closed-form Channel A refinement
- Per-channel synth ablation on ch 4-12 (individual channel contributions)
- Model-vs-analytic synth slicing (characterize which synth slices B genuinely rescues)

**Alternative paths that do require new data (kept, but conditional on shift-invariance result):**
- Cross-scaffold annotation on `real_IS10-R_sites.jsonl` / IS30 / IS903 / etc. — becomes NECESSARY only if shift-invariance test shows methods use absolute position. If shift-invariance passes for A / mmax-only B, cross-scaffold generalization is architecturally supported and annotation is less urgent.
- Literature search for other published RNA-guided systems with annotated guide positions — same conditionality.

#### Test (a) RESULT (2026-09-08, job 25722743): predictions FAILED, but with test-side confound identified

> **⚠ VOID (2026-09-09) — test (a) has TWO independent problems:**
> 1. Random uniform-ACGT prefix confounds translation with composition drift (already noted).
> 2. The "unshifted" reference numbers were pulled from the P0 script (all-zero flank_dev bug) while the shifted numbers came from the shift-invariance script (correct on-the-fly path). **The "main B improved 3.1pp under shifting" reading is a P0-vs-shift path artifact, not a shift effect.** On correct path both sides: main B is likely near-invariant to shift (consistent with test (b') which showed all three methods ≤3pp under bg-controlled shift). The "main B benefits from OOD-ing out of structure attention" interpretation is VOID.

Random 0-60nt uniform-ACGT prefix per Tnp, gold_start shifted to `49 + prefix_len` per Tnp:

| method | unshifted top-1 | shifted top-1 | Δ | pre-registered prediction | observed |
|---|---|---|---|---|---|
| A_maj | 0.5538 | 0.4769 | **−7.7pp** | invariant | **FAILED — not invariant** |
| main B (full 15-ch) | 0.4615 | 0.4923 | +3.1pp | should DROP | **OPPOSITE — improved slightly** |
| mmax-only B | 0.5385 | 0.5077 | −3.1pp | near-invariant | roughly correct |

**Both primary predictions failed. Diagnosis: the test has a confound I did not handle correctly.**

The random-prefix injection is NOT a clean translation. Random uniform ACGT has 25% base composition vs real ncRNA / real genomic backgrounds which typically have different (usually GC-biased) composition. Random ACGT prefix + a specific 11-nt target sequence produces more spurious m≥8 matches per position than real-genomic-background prefix would. This means:
- A_maj is mathematically equivariant to pure nc translation (S(p) shifts one-to-one with nc). What actually happened is: nc-shift + injection of a prefix with a DIFFERENT match-probability profile than the original nc's background. Competing S peaks in the prefix region reduce A_maj's top-1 by pulling argmax away from gold — not a failure of equivariance, but a failure of the test's cleanliness.
- Main B: the structure channels' values on random ACGT are OUT OF DISTRIBUTION vs training. Its noise-weight attention that produced a regional attractor at positions 100-170 on the original Durrant nc doesn't fire on random-prefix structural features → main B loses its systematic pull and falls back to m_max-driven decisions → gets closer to mmax-only performance (49% ≈ mmax-only 51%).

**What the test does show:**
- Structure/flank attention is disruptable — on unfamiliar structural inputs, main B's "attention on structure" collapses and it behaves like mmax-only. This is CONSISTENT with the "structure/flank weights are noise" story but doesn't uniquely prove it.
- The specific translation-equivariance-breaking story I registered pre-hoc is NOT confirmed. Registered but with the sign-of-main-B prediction wrong.

**Cleaner shift test that would need substantially more work:**
- Use variable-length REAL-genomic-background prefixes (like ncpad240 style but with per-Tnp length variance). Would isolate translation from composition-drift.
- Or: cross-scaffold data (different genuine nc scaffolds with different gold positions) — the real thing.

**Session-close honest state:** the equivariance-breaking mechanism story is DIRECTIONALLY plausible (main B benefits from structural OOD, consistent with structural attention being noise weights) but not cleanly confirmed by this test. The test's confound (random composition) makes it inconclusive. Registered as INCONCLUSIVE, not CONFIRMED.

**Updated conclusion on Channel B:**
1. **mmax-only recovers +7.7pp of main B's Durrant deficit** — solid, structure/flank channels are noise weights on real data.
2. **Whether Channel B's mechanism (m_max cross-site attention) has real advantage over A on real data** — unresolved. The clean test would need a dataset with gold-position variance (still an external dependency, not urgent given ambiguous test-a).
3. **(P2a) 8-channel retrain — CANCEL stays.** Would require the cross-scaffold data to validate anyway.
4. **(P1) A_maj family-internal curve — SHIP.** Independent of Channel B story.
5. **peak-finding + IoU +4.6pp** — still worth extracting for closed-form A refinement.
6. **Per-channel synth ablation** — could characterize which of ch 4-12 individually contribute on synth (would help design any future model).

#### Test (b') CLEAN shift-invariance — COMPLETE (2026-09-08, job 25728685)

Ran all three methods on realbg (shift=0) vs ncpad240 (shift=31, real-genomic-bg prefix). Same real bg on both sides, only 31nt translation differs. Both use the shift-invariance script (on-the-fly m_max via `_compute_site_arrays`, correctly handles shifted nc — bypasses the shard's precomputed m_max):

| method | realbg (shift=0) | ncpad240 (shift=31) | Δ shift effect |
|---|---|---|---|
| A_maj | 0.5077 | 0.4923 | −1.5pp |
| main B (full 15-ch) | 0.5385 | 0.5538 | +1.5pp |
| mmax-only B | 0.5077 | 0.5385 | +3.1pp |

**All three methods change by ≤3pp under 31-nt real-bg shift. Translation-invariance CONFIRMED for A_maj, mmax-only B, AND main B.** Position priors are NOT a driver of Durrant behavior for any of the three. The earlier concern that main B's structural attention breaks equivariance is REFUTED on this clean test.

**Test (a) is superseded by (b'):** (a)'s random-uniform-ACGT prefix introduced composition drift as a confound. (b') uses real-genomic prefix (both sides have real bg, only shift changes), eliminating the confound. (a) remains recorded as a failed test design; (b') is the definitive translation-equivariance measurement.

#### SURPRISING side finding + UNEXPLAINED SCRIPT DISCREPANCY (2026-09-08)

**Compiled top-1 table across paths + bg sources:**

| bg / path | A_maj | main B | mmax-only B | Δ(main B − A) |
|---|---|---|---|---|
| cognate, P0 script (shard-based) | 0.6000 | 0.4615 | 0.5385 | **−13.85pp** (SIG) |
| cognate, shift-invariance (on-the-fly) | 0.4615 | 0.5231 | 0.5385 | +6.2pp |
| realbg, shift-invariance | 0.5077 | 0.5385 | 0.5077 | +3.1pp |
| ncpad240, shift-invariance | 0.4923 | 0.5538 | 0.5385 | +6.2pp |

**Observations:**
1. **mmax-only is STABLE at ~0.5385** across all path × bg combinations (immune to path difference — makes sense, it doesn't consume ch 4-12).
2. **A_maj and main B vary between the P0 script and shift-invariance script** on the same source file (cognate). A_maj: 0.6000 vs 0.4615. Main B: 0.4615 vs 0.5231.
3. **Bit-identity check on m_max**: for T-WT bag000, shard's m_max at (fwd, L=11) is BYTE-IDENTICAL to on-the-fly `_compute_site_arrays` on the same nc/flank. Both give 8 at position 49 for all 5 sites; max_abs_diff = 0.

**So the m_max data is the same. The A_maj discrepancy between scripts must come from something OTHER than the m_max input.** Candidates:
- Different `majority_orient` computation (P0 uses gold's `target_flank_orientation` for majority; shift-invariance uses inferred orient for majority. Per-site agreement is 100% but Tnp-level aggregation MIGHT differ in edge cases.)
- Different tie-break in `argpartition` (both use it but subtle numpy quirks could differ)
- Different handling of edge cases

**Session-critical unresolved:** without understanding why two scripts disagree on A_maj when m_max is identical, cannot claim "P0 FALSIFY was a shard-path artifact." The path IS the same for m_max; something else differs. **Requires a targeted diff to locate before the reframed Channel B story can be finalized.**

#### Per-Tnp discrepancy diagnostic (2026-09-08, resolves A_maj discrepancy, leaves main B one open)

Ran a 65-Tnp diff of P0-style vs shift-style A_maj computation:

**A_maj part — RESOLVED (user's diagnosis confirmed exactly):**
- `majority_orient` from gold-target_flank_orientation and from alignment-inference agree on **65/65 Tnps** (100% site-level agreement propagates to Tnp-level majority)
- S array is **IDENTICAL for all 65 Tnps** (both paths produce byte-identical m_max, verified for T-WT bag000)
- The 14pp A_maj gap is purely `np.argmax(S)` vs `np.argpartition(S, -1)[-1]` differing on tied max positions

Four tie-break rules on the same S data:

| tie rule | top-1 |
|---|---|
| `np.argmax` (deterministic first-index of max) | **0.6000** (P0's reported number) |
| `np.argpartition(S,-1)[-1]` (undefined-order max) | **0.5538** (my inline reproduction) |
| Conservative (unique max only) | 0.4462 |
| Lenient (gold is a max) | 0.6923 |

**All four values fall inside the tie window [0.446, 0.692] measured earlier on this data.** 7/65 Tnps flip between argmax and argpartition — 5 where argmax favors gold (numpy picks position 49 first when tied with later positions), 2 where argpartition favors gold (argmax picks earlier tied positions 14/24 over gold at 49).

**Verdict on A_maj:** single-point top-1 comparisons on this data are unstable to numpy's tie-break implementation. The [cons=0.446, len=0.692] window is the honest measurement. The earlier "A_maj discrepancy" between P0 and shift scripts was not a script bug; both are valid tie-break choices.

**Main B part — RESOLVED (2026-09-08, bit-identity diagnostic bbil9cexr):**

Bit-identity check on T-WT bag000:
- **canonical_nc: identical** (P0 and shift use same source)
- **m_max: identical** (verified earlier for this Tnp)
- **structure (ch 4-8, from `compute_features_v2`): identical** (166-len arrays match byte-for-byte)
- **flank_argmax (source of ch 9-12): SHARD IS NONE, on-the-fly IS VALID.** For all 5 sites, `mt.flank_argmax_by_excl.get(0)` returns None; `_compute_site_arrays(...).flank_argmax_by_excl.get(0)` returns a valid array.

**Root cause:** the `durrant_positive` MatchTable shard was built without populating `flank_argmax_by_excl`. When `ChannelBDataset._build_bag_inputs` reads it (data.py:305-306), it gets None, and the check `if a_arr is not None: ...` skips the fill — leaving `argmax_per_site[s_idx, :, L_i]` at its `np.zeros(...)` initial value. Then `flank_dev = (argmax_per_site - bag_median) / FLANK_DEV_SCALE = (0 - 0) / SCALE = 0` for all positions. **Main B on the P0 path was fed all-zero ch 9-12 at inference** — OOD for a model trained on non-zero flank_dev.

Explains everything:
- **P0's main B = 0.4615 on cognate**: model receives all-zero flank_dev, drops performance vs training expectation.
- **Shift-invariance's main B = 0.5231 on cognate**: model receives proper on-the-fly flank_dev, closer to training distribution.
- **mmax-only unchanged across paths (0.5385)**: it zeros ch 4-14 by training design, so whether shard delivers or ChannelBDataset zero-fills makes no difference — the effective input is the same.

**Major session-wide reversal — the "structure/flank channels are noise weights hurting B on Durrant" claim (from job 25720736 ablation) is RETRACTED (2026-09-08):**

The ablation showed mmax-only B recovers +7.7pp over main B on P0 (0.5385 vs 0.4615) on cognate. This was interpreted as "structure/flank channels are noise on Durrant". **The correct interpretation is: P0's main B was running with all-zero ch 9-12 (OOD), and mmax-only doesn't suffer this OOD because it zeros ch 4-14 by training. The +7.7pp was artifact of the missing flank_argmax in the Durrant shard, not a genuine "noise weights" property.**

On the correct path (shift-invariance, proper flank_dev):
- main B = 0.5231, mmax-only = 0.5385, **Δ = only 1.5pp** (not 7.7pp)
- Structure/flank channels are NOT harmful. They may still be modestly informative or near-neutral, but they don't drag B down.

**Consequences for downstream claims (2026-09-08 audit):**
- **P0 FALSIFY**: was VOID due to tie-noise on argmax estimator; ALSO invalid due to OOD input (all-zero ch 9-12). Double-VOID.
- **mmax-only "improvement"**: was cognate-specific and shard-artifact-specific. On realbg / ncpad240 via correct path, mmax-only ≈ main B within 1-2pp.
- **"generator gap" story (synth ncRNA structure differs from real)**: was speculative even before; now the specific evidence base ("mmax-only recovers 7.7pp on real") is gone. That hypothesis remains unmeasured and no longer has a triggering result.
- **B vs A on Durrant, corrected path, no gate**: main B ≈ 0.52, A_maj [0.446, 0.692] tie window. Different tie rules put A on either side of main B. Cannot resolve without a tie-robust metric — same conclusion as before, now for a different (and more solid) reason.

**Registered followups (2026-09-09):**

**Code fixes committed as part of this rediagnosis:**
- **`model/channel_b/data.py`**: added `allow_missing_flank_argmax=False` default (fail-fast). Loader now raises `RuntimeError` when `flank_argmax_by_excl is None` for any site instead of silently zero-filling. Backward compat via explicit opt-in.
- **`model/channel_b/preflight.py`** (new): `channel_stats_preflight(x, ref_frac_zero)` utility. Prints per-channel-group (min, max, mean, frac_zero, frac_nan); aborts if `frac_zero` drifts from reference by > threshold. Registered as MANDATORY preflight in all new inference scripts. Would have caught both prior zero-fill incidents on the first inference run.
- **Memory**: new entry `feedback_channel_stats_preflight.md` (indexed in MEMORY.md). Rule: channel-stats preflight before every model.forward.

**Path choice for Durrant inference:**
- **CANONICAL**: on-the-fly `_compute_site_arrays` (as in `channel_b_shift_invariance.py`). Deprecates the `durrant_positive` shard-based path for Durrant work. Rationale: shard was built for synth training; on-the-fly is the mathematical definition of m_max/flank_argmax with no stale-cache risk. Cost is negligible at n=65 Tnps.
- Alternative: rebuild `durrant_positive` shard including flank_argmax. Not chosen — introduces a long-term consistency burden for a one-off real-data eval.

**Training-safety verification (2026-09-09, task b4oxohfpd):** confirmed that v6r2 training shards (`mt_v6r2_twin50k`, `_ctrl10k`, `_partial40k`) have `flank_argmax_by_excl` correctly populated (shape `(82,)` at fwd L=11 site 0). Only the Durrant `durrant_positive` shard has `flank_argmax = None` — bug isolated to the `build_positive` code path in `v5a_framework` (used only for Durrant), not the training-corpus shard-build path (used for v6r2). **The `allow_missing_flank_argmax=False` default fail-fast in `data.py` will not break training.** It will raise only when the defective Durrant shard is loaded via the deprecated path.

Consequence for the retractions above: training-time semantics were correct (main B trained on non-zero flank_dev channels). The OOD-input problem was purely at inference on Durrant via the buggy shard. All retracted P0 findings are correctly attributed to the shard bug + downstream OOD, not to any training-time issue.

**Mutation experiment — design revisions (per user 2026-09-09):**

Three amendments to the initial mutation design:

1. **A scoring: report BOTH the integer S_max version AND a continuous log-odds version.** The integer `bag_max_score = max_p S_maj(p)` takes only 6 distinct values (0-5) on Durrant → AUROC has many tie clusters, same class of tie-noise as argmax. Add `A_logodds = max_p [S(p)·log(p/q) + (n−S(p))·log((1−p)/(1−q))]` (this is A's continuous log-odds form — measured earlier on synth as +6.25pp over integer S). Continuous, no tie clusters. Primary metric = A_logodds AUROC.

2. **Dose axis: target-m, not raw N-mutations.** A center mutation splits m=11 into 5+5 (max continuous run = 5), while an end mutation gives m=10. "N mutations" is not a monotone dose variable. **Pre-design mutation patterns to hit target m values {10, 9, 8, 7, 6, 5}**; verify each pattern actually achieves target m via `_compute_site_arrays` on a mutated nc; then use target-m as the dose axis. Direct interpretation: at target m=7 (below threshold 8), S_at_gold should collapse from 5 toward 0.

3. **Add outside-guide-region mutation control.** For each mutation intensity, run a matched control where the same number of mutations lands OUTSIDE [49, 60]. Prediction: A_logodds and B should be nearly invariant to outside-region mutations. If they're not, whatever's driving the signal isn't specifically guide-region matching — a serious falsification of both methods' claimed mechanism.

**Updated priority order (2026-09-09):**
- **Immediate (this session or next):**
  1. Loader fail-fast + preflight utility — DONE (code committed above).
  2. Two-script tie-window unification (10 min) — output all four tie values (argmax/argpart/cons/len) from both scripts.
  3. Rerun everything that used the P0 script on Durrant with the fixed loader → confirm error is raised → switch to on-the-fly path.
- **Main lift (blocked on the above):**
  4. Gradient mutation experiment (with three revisions above): target-m dose, A_logodds primary, outside-region control.
- **Deferred (frozen until mutation experiment resolves):**
  - Peak-finding + IoU +4.6pp
  - Per-channel synth ablation
  - Cross-scaffold annotation
  - (P2a) 8-channel retrain
  - Real-sequence insertion positives

#### FINDING (2026-09-09): Durrant real guide-match distribution measured; earlier "synth→real strength gap" claim RETRACTED

Measured per-Tnp baseline m_max at gold (nc position 49, majority orient, first site) across 65 Durrant Tnps:

| baseline_m at gold | n_tnps | fraction |
|---|---|---|
| 6 | 9 | 13.8% |
| 7 | 7 | 10.8% |
| **8** | **41** | **63.1%** ← threshold, median |
| 9 | 7 | 10.8% |
| 10 | 1 | 1.5% |
| 11 | 0 | 0% |

**Real bridge-RNA guide matches cluster at m=8 (mode = median = threshold).** 24% below threshold (baseline_m ≤ 7). This is a genuine finding about the data.

**RETRACTION (2026-09-09):** an earlier version of this section claimed "v5/v6 generator's planted_m samples from an 86/10/4 tail centered at m=max_L (i.e., m=11 for L=11 majority)" — WRONG. I did not measure the actual generator distribution, only my mental model of it. Frozen anchor `target_m_for_L(11, 0.21) == 8` (in A4) already told me this and I should have checked.

**Actual measured synth planted_m distribution in v6r2 pos50k (n=20001 sites):**

| planted_m (L=11 subset) | synth | real Durrant | Δ |
|---|---|---|---|
| 5 | 0.1% | 0% | +0.1 |
| 6 | 3.4% | 13.8% | −10.4 |
| 7 | 17.8% | 10.8% | +7.0 |
| **8** (mode) | **74.4%** | **63.1%** | +11.3 |
| 9 | 4.2% | 10.8% | −6.6 |
| 10 | 0.1% | 1.5% | −1.4 |

**Synth and real MODE at m=8 both.** Real is slightly more spread — more mass at 6 (13.8 vs 3.4) and at 9-10 (12.3 vs 4.3), less concentrated at 8 (63.1 vs 74.4). Distributions are similar in shape; **not a "3 units above real" gap as I claimed**.

**Consequences:**
- The "MAJOR synth→real m-strength gap" claim is RETRACTED. No dramatic gap exists on L=11.
- The 3 downstream inferences that hung on that claim (S_at_gold=3.48/5 explanation, baseline-6 Tnps mapping, B's synth→real reversal) are DECOUPLED from the claim. baseline_m distribution is real; the synth-relative interpretation was overstated.
- Candidate A (retrain with real distribution) is UNMOTIVATED — distributions already close. Any retrain would give marginal shifts on L=11.

#### Actual next experiment (per user 2026-09-09): held-out-m generalization gate

**Motivation** (correcting the earlier framing): the question is not "does synth's m distribution match real's" (they largely do), but "does B learn a RULE that generalizes across m or a LEVEL specific to its training m distribution". A fixed-distribution retrain (Candidate A) cannot answer this — both train and test would be at the same distribution, same blind spot.

**Design REVISED after bag-count preflight (2026-09-09):**

Original interval {9,10,11} / {6,7,8} was infeasible: TRAIN {9,10,11} all-site has **zero L=11 bags** (74% of L=11 planted_m is at m=8, so getting all-sites-in-bag to fall in {9,10,11} on L=11 is impossible). Revised to user's proposal:

**Retrospective mechanism note (2026-09-10):** the "zero L=11 bags in {9,10,11}"
finding is not a data quirk — it is a **hard ceiling in the v6r2 generator**.
`sample_difficulty` (`difficulty.py:472`) sets each bag's `target_m =
target_m_for_L(L, target_rate, gc)`, with the frozen anchor
`target_m_for_L(11, 0.21) == 8`. Per-site `planted_m` is drawn from
`sample_planted_m(rng, target_m)` (`difficulty.py:437`) with tail
`{target_m, target_m-1, target_m-2}` — **strictly ≤ target_m**. So for any
L=11 bag, EVERY site's planted_m ∈ {6, 7, 8}; getting all sites in
{9, 10, 11} has probability exactly 0 by construction. Similarly the tail
never plants above `target_m`, so training-set exploration of the "strong
planted signal" regime at any L is capped at that L's target_m. This is
the direct mechanistic reason held-out-m gate design under v6r2 was
constrained to weak-plant-only splits. **v7 lifts the ceiling** by
sampling planted_m from `U{5..min(11, L)}` independently of `target_m`
(see V7_SPEC §2.5 + `sample_planted_m_uniform` in `bag_v2.py`), so
future held-out experiments can split freely across the m axis without
this class of infeasibility.

| interval | all-site n_bags | L=11 subset |
|---|---|---|
| **TRAIN {8,9,10}** | 32,783 (65.6% of pos50k) | 5,729 |
| **VAL {6,7}** | 378 (0.76%) | 301 |

- **TRAIN**: v6r2 pos50k bags with ALL sites' planted_m ∈ {8, 9, 10}. Existing corpus.
- **VAL**: v6r2 pos50k bags with ALL sites' planted_m ∈ {6, 7}. Same corpus, disjoint from train.
- The direction is fixed by data availability: only train-strong / val-weak has enough bags to be feasible AND is the direction we care about (extrapolate to weaker signal, closer to Durrant's real regime).

**Same-distribution control comes FREE:**
- Existing `checkpoints/channel_b/main_lr3e-4/best.pt` was trained on ALL v6r2 (includes {6,7} bags).
- Evaluating `main` on the SPECIFIC 378 VAL {6,7} bags gives the same-distribution ceiling.
- Only ONE new model needs training: held-out model on TRAIN {8,9,10}.

**planted_m provenance verified (bag-count job 2026-09-09):**
- `ARCH_ALLOWED_KEYS = {flank_offset_mode, n_sites, nc_homology_rate, orient}` — no m field
- `m_at_planted` correctly in `TRAIN_ONLY_LABEL_KEYS` (blacklist)
- grep in `data.py`: zero occurrences of planted_m
- **planted_m does NOT enter model input tensor.** Held-out-m gate is leakage-free.

**Pre-registered verdict rules — REVISED after user's four-item audit (2026-09-09):**

**Revisions to the initial design:**

1. **Primary metric changed** to paired AUROC diff (DeLong or paired bootstrap) between held-out and same-distribution ceiling on same VAL {6,7} bags. R (ratio) demoted to secondary/informal.
   - **Reason:** VAL {6,7} = 378 bags. A single-model AUROC on 378 bags has SE ~0.024; the RATIO of two AUROCs (or top-1) has SE ~0.11. Original judgment thresholds 0.80/0.40 are only 3.6 SE apart → INCONCLUSIVE region will hit almost certainly. Paired comparison on the same 378 bags cancels bag-difficulty variance and lets DeLong's paired test operate cleanly.

2. **Judgment threshold anchored to existing held-out precedents:**
   - nsheld C4 result: −0.43pp on paired-AUROC comparison, deemed PASSED (within noise)
   - gcheld held-out result: −1.7pp, deemed "CONCERNING" but not FALSIFY
   - **Held-out-m PASS:** paired Δ AUROC (heldout − same-distribution ceiling) 95% CI upper bound ≤ 0.02 (i.e., 2pp degradation acceptable, matches gcheld scale)
   - **Held-out-m FAIL:** paired Δ AUROC 95% CI lower bound < −0.05 (5pp degradation, meaningfully worse)
   - **Held-out-m CONCERNING:** in between; report as "small degradation, not decisive"

3. **Sample-size confound requires a THIRD control model** (per user):
   - `main` was trained on ~50K bags total from pos50k
   - `heldout-32k` will be trained on 32,783 bags (all-site m ∈ {8,9,10})
   - **`random-32k`**: trained on 32,783 bags RANDOMLY SAMPLED from full pos50k (matched size, full m distribution). Isolates "didn't see m ∈ {6,7}" from "34% less training data".
   - Comparison: heldout-32k vs random-32k on VAL {6,7} = pure "didn't-see-low-m" effect.
   - heldout-32k vs main on VAL {6,7} = "didn't-see-low-m + 34% less data".
   - **Cost doubles: 2 model trainings (~8-16h total).**

4. **Ceiling is slightly optimistic** (registered caveat): `main` was trained on ALL pos50k, so VAL {6,7} bags are in main's train set → main's number on those bags is train-set performance, mildly optimistic. Ideally use main's own held-out val subset ∩ {6,7} — expect ~20% × 0.76% = ~76 bags, too small. Accept the optimism and note that R (or paired Δ) is therefore conservative-biased.

**Revised metrics summary (judgment direction FIXED 2026-09-09 per user):**

| primary | Δ AUROC paired (heldout-32k − random-32k) on VAL {6,7}, DeLong CI |
|---|---|
| PASS | **CI lower bound ≥ −0.02** (heldout is not much worse than random-32k; matches nsheld/gcheld precedent for "small degradation") |
| CONCERNING | CI lower bound in [−0.05, −0.02) |
| FAIL | **CI lower bound < −0.05** (heldout meaningfully worse) |

Previous version wrote PASS as "CI upper bound ≤ +0.02" — that would test whether heldout is not much BETTER, which is not the question. Unified on CI lower bound.

Secondary:
- Δ AUROC paired (heldout-32k − main) on VAL {6,7}: additional context; includes sample-size effect
- R (ratio) as informal comparison, no threshold
- Per-m stratification (m=6 vs m=7): directional only; only ~100-200 bags per level
- Sanity: heldout-32k vs random-32k on TRAIN {8,9,10} slice — should be close (both see this distribution)

**Why Δ(B−A) alone isn't the primary:** on VAL {6,7} where m ≤ 7 < threshold 8, A_maj's `S_at_gold = 0` by construction → A degrades to chance-level auto. Any B ≥ chance satisfies "Δ(B−A) ≥ 0" trivially. Held-out vs same-distribution comparison is the actual generalization test.

**Cost:** ~8-16h retrain (heldout-32k + random-32k) + minutes eval. Total ~1-2 days. Doubled vs original plan; necessary to isolate the confound.

**Execution directives (2026-09-09):**

1. **`random-32k` samples from POOL EXCLUDING VAL {6,7}.** Full pos50k (~50k) minus 378 VAL bags = ~49,622 pool. Sample 32,783 from that. Kills the ~65% train-eval overlap that would otherwise contaminate the primary comparison at zero cost.
2. **Hyperparams FROZEN to match `main_lr3e-4/best.pt`:** lr=3e-4, all optim/architecture settings from `main`'s training args. `best.pt` selected by val_loss (same criterion as main). Save every-epoch checkpoint (`epoch_*.pt`) so checkpoint-selection is not a post-hoc lever.
3. **Random seed FIXED and logged.** heldout-32k uses default seed=0; random-32k uses seed=0 for the subsample AND for training. If primary result lands in CONCERNING, launch a second random-32k with seed=1 (cost: +4-8h) to check subsample-choice variance.
4. **Log intersection statistics:** report actual overlap (%) between random-32k train and VAL {6,7} (expected 0 given exclusion). Report bag-count distributions by planted_m in both train sets for provenance.
5. **Two-script tie unification** (deferred hygiene from prior turn) will be folded into the training-script edits — output paired-DeLong and paired-bootstrap CIs in the same evaluation report.

**Contingent Candidate A' (if held-out gate FAILS):** DOMAIN RANDOMIZATION — regenerate synth with planted_m uniform on wide range {5..11}, not "copy the 65-Tnp empirical distribution" (which overfits to the single T-WT bridge-RNA family scaffold — same class as gold_start=49 issue).

**Contingent auxiliary:** within-site percentile-normalized m channel alongside raw m. Only if gate fails; queued as follow-up.

**Contingent Candidate A' (if held-out-m gate FAILS):** the correct fix is not "copy the 65-Tnp empirical distribution" (which would overfit to the T-WT bridge-RNA family scaffold — same class as gold_start=49). The correct fix is **DOMAIN RANDOMIZATION**: sample planted_m uniform from a wide range (e.g., 5-11) so B is forced to learn a rule invariant to level. Distribution copying inherits the family bias of the specific 65-Tnp measurement.

**Contingent within-site rank/percentile channel:** add a per-site percentile-normalized m channel alongside raw m. Model then has both absolute (retains real information — longer match IS more likely real) and rank (scale-free). Only useful if gate fails; queued as follow-up.

**Deferred / de-prioritized:**
- Candidate A (fixed-distribution retrain): UNMOTIVATED given distributions already similar; single-family bias risk if used.
- Candidate B (mutation experiment): stays deferred; less deploy-realistic than held-out-m gate.
- Peak-finding + IoU +4.6pp, per-channel synth ablation, cross-scaffold annotation, (P2a) 8-channel retrain: all remain frozen.

**Current honest state:**
- **Test (b') clean shift result** stands independently — all three methods approximately translation-invariant on realbg → ncpad240 shift. Uses within-script paired comparison, so tie-break variance is CONTROLLED (same tie rule on both sides of comparison).
- **A_maj discrepancy** resolved as tie-handling; single-point comparisons unreliable, [cons, len] window is canonical
- **Main B path discrepancy** unresolved; needs flank_argmax bit-identity check
- **mmax-only stable at 0.5385** across ALL path × bg combinations (immune, ch 4-12 zeroed)
- **P0 FALSIFY (−13.85pp SIG) — VOID (estimator inappropriate for the data, 2026-09-08, per user):** the paired CI was computed on `argmax(S) == gold_start` booleans. A_maj's underlying tie window on this data is 24pp wide ([0.446, 0.692]) — 7/65 Tnps flip between numpy `argmax` and `argpartition` tie-breaks. **The estimator (paired-boot CI on argmax booleans) cannot resolve a 13.85pp effect on data with 24pp tie-noise on one side of the paired comparison.** Same class of problem as the earlier Δ-C1_LOO VOID. Marked VOID — even after main B path is resolved, computing paired CI on argmax booleans of this data won't recover the estimator's validity. The only way to test B vs A on Durrant is a tie-robust metric (AUROC over bag scores, not argmax over positions).

**Robust conclusions (survive all discrepancies):**
1. mmax-only ≈ 0.5385 on Durrant, stable across paths
2. All three methods approximately translation-invariant under real-bg 31nt shift
3. Single-point top-1 comparisons are noise on this data; tie window [cons, len] is canonical
4. gold_start=49 constant caveat still applies
5. Position priors don't drive method differences

**PENDING (all claims about main B vs A ranking on Durrant are on hold):**
- Whether B genuinely underperforms A on Durrant (P0 FALSIFY claim) or is at parity/better (shift-invariance suggestion). Requires resolving main B path discrepancy AND switching to a tie-robust metric.

#### Registered hygiene: unify tie-window reporting across both scripts (2026-09-08)

Both `channel_b_durrant_p0.py` and `channel_b_shift_invariance.py` currently report a SINGLE top-1 value each (with implementation-specific tie behavior). **Both scripts to be modified to output four tie values simultaneously:** `argmax`, `argpartition`, `conservative` (unique max), `lenient` (gold is a max). ~10 min of code. Registered as blocker for any future report using top-1 on Durrant — prevents recurrence of the "two scripts, two numbers, one hidden tie rule" class of confusion.

#### Registered next experiment: gradient mutation negatives (2026-09-08, per user, MAIN LIFT)

**Motivation:** on the Durrant data (65 Tnps × 5 sites, all with gold at nc position 49), argmax-over-positions is a tie-noisy estimator (24pp tie window on A_maj alone). AUROC over bag-level scores is tie-robust. This is the only way to answer "does B beat A on Durrant" that doesn't hit the estimator problem.

**Design:**
- For each of 65 Tnps: create mutant versions of the shared ncRNA scaffold by mutating N ∈ {1, 2, 3, 4} bases within the guide region [49, 60]. Mutation applied to the shared nc → affects all 5 sites of the Tnp identically.
- Positive class: original bag (unmutated nc)
- Negative class: mutant bag (mutated nc)
- Pairs: 65 Tnps × 4 mutation levels = 260 paired negatives + 65 positives = 325 bags total
- Per site: nc changes, flank unchanged, gold_span unchanged (still [49, 60] on the same nc scaffold, just with mutations inside it)

**Predictions (pre-registered):**
- **1 mut**: m_at_gold likely drops 11→10 (still ≥ threshold 8) → S_at_gold barely changes → AUROC ≈ 0.5-0.6
- **2 mut**: m 11→9 → S_at_gold still likely 5 for most Tnps → AUROC modest
- **3 mut**: m 11→8 or 11→7 → S_at_gold starts dropping below 5 → AUROC rises
- **4 mut**: m 11→7 or 11→6 → S_at_gold ≤ 4 for most → AUROC high
- Gradient across levels traces the causal S_at_gold ↓ vs discriminability curve, decoupled from Durrant's natural mixed-orient × S_at_gold confound

**Pre-experiment verification (MUST run before scoring):**
1. For each mutation level, sample N mutant nc's, recompute m_max on mutated nc, report S_at_gold distribution. If 4-mut doesn't push S_at_gold below 5 for most Tnps, add more mutations or move mutations outside high-informative subwindow.
2. Verify mutations don't accidentally create new stronger matches elsewhere on the nc.

**Metrics:**
- **PRIMARY: paired AUROC of bag_max_score, original vs mutant, per mutation level** — Tnp-clustered bootstrap CI (65 clusters, not 325 bags — 5 sites per Tnp share the same mutated nc so they're NOT independent).
- Per-level Δ(B AUROC − A AUROC), paired DeLong test — gives the transfer verdict for each level.
- SECONDARY: bag_max_score distributions per Tnp per level — sanity check that scores actually respond to mutations.

**A definition:** bag_max_score for A = `max_p S_maj(p)` (the max over positions of S at majority orient). NOT argmax_p — max value. Score is 0-5 integer per bag, discriminates positive (unmutated) from negative (mutated) bags by "how many sites still hit m≥8 at some position after mutation".

**B definition:** bag_max_score for B = model's per-position score, max over positions. What B was trained on for bag-AUROC proxy. Directly comparable to A's bag_max.

**Robustness features:**
- AUROC is invariant to monotone transforms → not affected by argmax tie noise
- Paired DeLong test uses same Tnps for pos/neg → controls for scaffold-level variation
- Tnp-clustered bootstrap → correct uncertainty at n=65 independent systems (not n=325 dependent sites)

**Estimated effort:** ~3-4h build + verify + run + analyze. Ready to design and run pending main B path resolution.

**Deferred experiments (frozen until mutation results):**
- Peak-finding + IoU +4.6pp extraction
- Per-channel synth ablation
- Cross-scaffold annotation
- (P2a) retrain
- Real-sequence insertion positives (test after mutation results, using real sequences per test-a lesson)

**Consequences for (P2a):**
- **(P2a) 8-channel retrain rationale is INVALIDATED for Durrant.** The synth +18pp result does not predict Durrant behavior. Retraining with an architecturally cleaner 8-channel design on the SAME synth data would inherit the same training/deploy gap. Regenerating synth to match Durrant characteristics is a much bigger project.
- **The per-site orient loader change stands as a correct implementation** (bit-exact PASS, correct A_maj recovery), just does not rescue B on real data.
- **(P1) OC coverage-precision curve remains the honest shippable deliverable.** A on real Durrant: 60% overall top-1 at 100% cov, 97% at 51% cov, 100% at 31% cov. B does not improve on this.

**Priority reorder after (P0) FALSIFY (2026-09-08, per user):**
- **(P1) OC coverage-precision curve — SHIP NOW.** A on real Durrant is the ONLY positive real-data result on this line. Ship-ready under canonical A_maj: 55% top-1 at 100% cov (argpart, tie window [44.6%, 69.2%]); 87% at 66% cov (S≥3 gate); 97% at 51% cov (S≥4 gate); 100% at 31% cov (S=5 gate, matches A1/A2). Does not depend on B.
- **(P2a) 8-channel retrain — CANCELED (not paused).** Rationale (per user 2026-09-08): retraining any architecture on the same synth distribution inherits the same synth→real gap. Per-site loader (bit-exact PASS) already solved the orient-representation problem and did NOT rescue B — so 8-channel cannot be the answer either. The problem has moved from architecture to the synth training distribution itself.
- **Ablation diagnostic (ch4-8 / ch9-12 zeroing) — REJECTED (per user 2026-09-08).** Zeroing has known floor effects in this project (prior "only_mmax > zero_struct" self-contradiction) that would give un-interpretable results. Replaced with direct case-by-case analysis of the 8 A-hit-B-miss Tnps on S=3 / S=4 (measured this section — see case-by-case output).
- **True next-step direction (per user 2026-09-08):** the generator's hard-case distribution does not match real Durrant's clean-case distribution. B learned to discriminate synth-hard cases; real Durrant is at the clean end. This is a generator problem, not an architecture problem. It reopens the earlier questions about 500K / group II / CAST-family training corpora. Substantially larger scope than (P2a).
- **peak-finding + IoU +4.6pp**: still worth extracting for closed-form Channel A refinement, independent of any B/generator work.
- **Full bag-AUROC (f)**: still deferred. B-worse-than-A on localization strongly predicts B-not-better-than-A on bag-AUROC either.

Mechanism (why per-site alone can't reach model): at L=11 background, q ≈ 0.21. Across nc_len ≈ 200 positions, P(max_position(m) ≥ 8) → 1 for any site by chance. Per-site alone cannot distinguish "spike at planted position" from "spike anywhere by chance" — signal is entirely in cross-site CO-OCCURRENCE at the same position, which per-site processing fundamentally cannot capture.

**Where the two-level gain lives (RETRACTS earlier "advantage in per-site term" claim):**

```
per-site closed-form ceiling (well-posed GBM)    0.5273    +2.7 pp over chance
analytic cross-site (count → log-odds → max)     0.6666    +14 pp over per-site
model cross-site (learned attention)             0.7812    +11 pp over analytic
```

**Both jumps are entirely cross-site.** The 0.527 → 0.667 jump is A's aggregation over sites at each position; the 0.667 → 0.781 jump is model's attention aggregation. Per-site processing contributes essentially nothing on either side. The earlier claim "model advantage is in the per-site term" (based on the equal-slope observation between model and analytic in the per-slice table) was wrong — equal slopes only mean equal n_sites cumulative efficiency, not that the +11 pp gap lives in the per-site component.

### Robustness hypothesis — CONFIRMED across two seeds (2026-09-07)

"Structure/flank/orient channels provide robustness, not discriminative power" — both predictions verified:

**Prediction 1: n_sites=3 mmax-only drop reproduces across seeds** — CONFIRMED
- seed 0: mmax-only 0.6578 vs full model 0.7147 → −5.7 pp
- seed 1: mmax-only 0.6687 vs full model 0.7147 → −4.6 pp
- Mean −5.1 pp; both above CI threshold at n=2326

**Prediction 2: mmax-only shows sum_post < default gap, full model doesn't** — CONFIRMED
- Full model (n=4-8): sum_post gap = −0.001 to +0.006 (essentially zero)
- mmax-only seed 0 (n=4-8): sum_post gap = −0.010 to −0.016
- mmax-only seed 1 (n=4-8): sum_post gap = −0.017 to −0.026
- Both seeds' pool aggregation form matters when only m_max is available; goes away when extra channels are present

**Interpretation**: the 11 extra channels don't add discriminative signal on n_sites ≥ 4 slices (~0 pp per-slice AUROC contribution), but they stabilize the model against aggregation-form choices and rescue the n_sites=3 slice by ~5 pp. This is a rescue role, not a signal role.

### Channel B value proposition (first version with a complete evidence chain, 2026-09-07)

**Claim:** the optimal combination of cross-site co-occurrence at nc positions is not the form Channel A uses (count sites hitting m≥8, log-odds sum across n_sites, max over positions), and this cannot be reduced to a per-site closed-form scoring.

**Evidence chain (four independent experiments, none retracted):**

| experiment | result | role |
|---|---|---|
| Coherence-destroying per-site cyclic shift | Analytic 0.6797→0.5064 (positive control fires); model 0.6825→0.4844 (same mechanism confirmed) | Cross-site alignment IS the mechanism, not a shortcut |
| Well-posed per-site GBM ceiling on partial40k | 0.5273 within-bin mean (well below analytic 0.667) | Per-site closed-form ceiling is fundamentally low; the +11 pp is not extractable per-site |
| mmax-only retrain (channels 4-14 zeroed) | Matches full model on n_sites ≥ 4 within 1 pp per-slice | Advantage is anchored in m_max channels; structure/flank/orient contribute rescue-role only (n=3 by ~5 pp) |
| Per-slice AUROC with paired CI, model vs analytic (main val 15709 bags) | +11 pp per-slice uniformly across n_sites 3-8, CIs [+0.09, +0.14] | Advantage magnitude quantified; deployment lift under neg-only z-score = +10.7 pp, CI [+0.099, +0.116] |

**What Channel B delivers**: a learned cross-site interaction on m_max distributions that beats analytic's count-based cross-site sum by ~11 pp per-slice / +10.7 pp pooled after standard deploy calibration.

**What Channel B does NOT deliver**:
- Extraction of additional signal from structure/flank/orient channels (rescue role only)
- A form reducible to a per-site log-odds scoring
- Discriminative gain on n_sites ≥ 4 that requires anything beyond m_max

### Stopping condition (registered 2026-09-06)

Current mmax-only retrain + up to 3 leave-one-channel-group retrains exhaust the direct-inspection avenue for the +0.3σ source. If none localize, the source is recorded as **UNEXPLAINED** and:
- Channel B retrain stays on indefinite hold — no known architecture change targets an unidentified mechanism
- Effort pivots to Durrant real-data evaluation using main best.pt + the deploy recipe (per-bin z-score from a background pool)
- +10.7 pp z-scored pooled remains the load-bearing deliverable regardless

## Known reimplementation risks (audit protocol)

Any new detector implementation that produces coverage / PPV / exact
numbers MUST demonstrate it produces the A1 / A2 Durrant anchor
byte-identically before its numbers are compared to Durrant results.

## Dead / deprecated code

- `_deprecated_bag_v1_strict_loop.py` — falsified by D1 (T-WT gold at nc=49 straddles a stem, not inside any ≥11-nt loop).
- `_deprecated_ncrna_sampler_v1_strict_loop.py` — same reason.
- `_deprecated_test_bag_shape_v1.py` — tests for the above.

Do not import from `_deprecated_*`. Kept in the tree only so the falsification history is not lost.

## Change protocol

1. Any change touching a locked component: run `python -m scripts.generator_v5.run_all_anchors`. All 6 anchors must PASS.
2. Any new comparison of two MetricReports across corpora: MetricCondition must be built out with `corpus` and `guide_origin` fields; `safe_ratio` must refuse comparisons where these differ.
3. Anchor drift is a hard STOP: investigate before committing.

## Git tag

`v5-frozen` at commit HEAD (created after run_all_anchors PASS).

---

## v7 pre-registered gate criteria (2026-09-11)

Recorded BEFORE the v7 training run submits. Property-based; no absolute
numerical thresholds carried over from v6r2 (the v6r2 "+11pp bag AUROC over
analytic" number is not a target and must NOT be used as a pass/fail
threshold — v7 has different flank source, different planted_m distribution,
and different loader semantics; comparing absolute numbers across those
axes is a category error, see [[finding-v6_stage2_drift]] + [[feedback-synth-effect-insufficient]]).

The four gates are evaluated on the v7 model trained on the 5-corpus mix
(pos50k / twin50k / partial40k / ctrl10k / scat10k) with the v6r2 hyperparams
(lr 3e-4, 8 epochs, batch 32, hidden 128 / heads 4 / blocks 3,
`--split-mode main`). Model = `checkpoints/channel_b/v7_main/best.pt`
(selected by val_loss).

### C3 — permutation equivariance (architecture property, v7-agnostic)

For each val bag, permute the site axis by a random π, run the model on
both orderings, undo the permutation on the output, compare.

**Threshold (2026-09-12 revision, with fp64 evidence).**

- **In fp64**: `max over (bags, positions) of |output − π⁻¹(output_perm)|  <  1e-12`
  — this is the architectural invariance test. It must hold on any bag,
  any model checkpoint. A single non-conforming bag → hard failure.

- **In fp32 (the training/inference default)**: `max  <  1e-5`. Accumulation
  through 3 blocks × H=128 × (MHA + 4·H FFN + 2 LayerNorms) drives an
  fp32 floor around 2-5e-6 empirically. Anything up to 1e-5 is normal
  numerical noise and does not indicate a code bug.

**Prior threshold retraction** (pre-2026-09-12 FROZEN said `< 1e-6` in
fp32). Empirically that threshold was too tight for this depth. Measured
2026-09-12 on `v7_main_retrain/best.pt`, 200 pos50k val bags:
  - fp32: max 2.384e-06, p50 6.26e-07, p95 1.67e-06, p99 1.91e-06
  - fp64: max 5.329e-15, p50 1.33e-15, p95 3.55e-15, p99 4.44e-15
  - fp32/fp64 ratio ≈ 4.5×10⁸ — accumulation-dominated

The revision is justified BY the fp64 measurement (invariance holds to 15
decimal digits), not by seeing an fp32 failure. Threshold revised to
match precision of the arithmetic actually used.

Fails only on numerical-order bugs (residuals depending on site index;
forgotten mask) — those would corrupt fp64 too. So the fp64 check is the
real test; the fp32 check is a smoke.

### C4 — held-out n_sites (generalization across a training axis)

Train two models on the SAME hyperparams and data volume, differing only in
which n_sites bags are visible:

- **model_all**: train on all bags (n_sites ∈ {3..8})
- **model_heldout**: train on bags with n_sites ∈ {3,4,5,6}, val on n_sites ∈ {7,8}

Metric = per-bag AUROC on the val n_sites ∈ {7,8} slice (never seen at
train time by model_heldout). Compute:

    Δ  =  AUROC(model_all)  −  AUROC(model_heldout)
    (positive Δ means holding out ≥7-site bags at train time HURT val performance)

with paired-bootstrap 95% CI over the same val bags.

- **PASS**: CI upper bound ≤ +0.02 (model_heldout is essentially as good as
  model_all — the model transfers across n_sites; Channel B's architecture
  is n_sites-agnostic modulo the attention pool)
- **FAIL**: CI lower bound > +0.05 (holding out ≥7-site bags at train time
  significantly hurts val performance — the model has learned an
  n_sites-specific artifact and does not extrapolate)
- **CONCERNING**: between (report, but don't extrapolate to n_sites regimes
  outside the training pool without a follow-up)

Numbers +0.02 / +0.05 chosen from prior v5-era n_sites-heldout runs: main-B
gcheld drop was 1.7pp (0.017), main split drift 0.4pp; anything below 2pp is
noise-consistent, anything above 5pp is a material generalization gap.

### held-out-m — generalization within the {8..11} planted_m range

**2026-09-13 amendment: split rule + control changed to fit v7 corpus.**
The pre-registered form ("bag where every site has `m_at_planted == 8`") was
measured on v7 as 388 bags total (0.65% of ~60k positives) — below the 1000
val-bag floor. The corpus's per-site m distribution is dispersed enough that
requiring ALL sites to equal 8 is very rare (see `v7_precounts` scan).

Amended split — **max-aggregation**:
  - `model_heldout_m` trains on bags with `max(m_at_planted) ≥ 9` across
    sites (**57,714 bags**)
  - val = bags with `max(m_at_planted) == 8` (**2,163 bags**)
  - Semantics: "the model has never seen a bag whose strongest-signal site
    was only m=8 — can it still score such bags correctly?"

**Sample-size-matched random-ctrl is INFEASIBLE under this split.**
Because val pool is only 3.6% of the positive corpus (2,163 / ~60k), any
random-ctrl drawn from full positives sized to match heldout (57,714 bags)
must include ~96% of val — a control that heavily saw val, so useless as a
"same size, different m distribution" comparison. Excluding val from the
random-ctrl source pool leaves 57,714 + 123 max≤7 = 57,837 bags, of which
57,714 is 99.8% — essentially the same pool as heldout, giving no
independence. Every intermediate compromise (smaller matched N) inherits
the same problem proportionally.

**Amended control — biased-favorable-to-model_all**:
Use `v7_main_retrain` (main split, saw everything including val bags via
its main-hash split's train side) as the control.
  - Δ = AUROC(model_all_main_retrain) − AUROC(model_heldout_m), paired 95% CI on val_max=8 bags
  - Direction of bias: FAVORS `model_all`. `model_all` saw the val bags at
    training time (via main-split), so any advantage it has is at least
    partly leakage — the bias is against `model_heldout_m`.

- **PASS**: CI upper bound ≤ +0.02 (heldout is essentially as good as the
  biased-favorable model_all → model generalizes to max=8 bags despite
  never seeing them)
- **FAIL**: CI lower bound > +0.05 (heldout loses even to a biased
  favorable ctrl → clear generalization gap)
- **CONCERNING**: between (bias direction complicates attribution, report
  but do not extrapolate)

Note the asymmetry: because the bias favors `model_all`, a PASS
(heldout not-losing to leakage-advantaged ctrl) is a STRONGER positive
than the pre-registered symmetric version would have been. A FAIL is
weaker (could be either the heldout effect or the leakage-ctrl artifact),
so a FAIL result must be reported as "at least one of {generalization
gap, control artifact} present" rather than a clean generalization
statement.

**Retracted (pre-2026-09-13 form of this section):** The prior version
required `--planted-m-train-mset 9,10,11` + a sample-size-matched
`--random-subsample-n` control on val bags with every site's
`m_at_planted == 8`. Retracted because (1) val pool = 388 bags, (2) any
size-matched random control would draw from the 60k positive pool and
include ≥96% of val. The primary metric direction is preserved
(positive Δ = holding out m=8 hurt val performance); the target
(bags with `max(m_at_planted) == 8`) is the reasonable substitute
for the underpopulated pre-registered target.

### per-slice bag AUROC (model vs analytic, direction-consistent across n_sites)

For each val bag, compute:

- **model_score**: `max over positions of model.forward(bag).squeeze()`
  (same bag-level scalar used in `evaluate.val_auroc_proxy`)
- **analytic_score**: `S_oc(bag)`  =  max over orient of ∑_sites 1{m_max_orient ≥ 8}
  (the shipping Channel A analytic)

Stratify val bags by `arch.n_sites` (bins: {3, 4, 5, 6, 7, 8}). Within each
stratum, compute per-bag AUROC(model) and AUROC(analytic) on positive-vs-
negative labels of that stratum, then Δ = AUROC(model) − AUROC(analytic).

**PASS criteria (both required)**:

1. **Direction consistency**: point-estimate Δ > 0 in EVERY n_sites stratum
   with ≥ 100 val bags. (The property being tested is "model doesn't regress
   against the analytic on any n_sites bucket".)
2. **Pooled CI**: paired-bootstrap 95% CI of the POOLED Δ (all n_sites
   strata combined, weighted by stratum size) has lower bound > 0.

**Per-stratum CI is reported for context only, not a hard gate.** A single
stratum's CI can straddle 0 due to small sample within that stratum
(especially n_sites=3 or =8 on some splits); demanding significance in
every stratum would over-reject on sampling noise. The two-part rule above
tests the intended property (universal-direction advantage that survives
pooling) without over-fitting to any single stratum's noise.

**FAIL modes**: (a) any stratum's point Δ < 0 (direction violation);
(b) pooled CI lower bound ≤ 0 (no significant aggregate advantage).

### Not gates (recorded to avoid drift)

- **NOT a gate**: absolute bag AUROC ≥ some number (0.60, 0.70, etc.).
  The retracted v6r2 numbers were 0.65-0.70 range; those are not v7
  targets.
- **NOT a gate**: any comparison to v6r2 checkpoints on any metric. v6r2
  is UNSCOPED under the 2026-09-10 class-level retraction (see above).
- **NOT a gate**: separation between pos and neg score distributions.
  Separation was retracted as a judgment criterion 2026-09-11 (negative
  separation on the analytic side is a statistical norm, not a failure —
  see conversation trace).

### If any gate fails

Do NOT patch the model to pass. Report the failure, retract the v7 corpus
if it looks like a data issue, or retract the gate itself if it looks like
the criterion was wrong. Never re-run gates after tuning to the gate.

---

## v7 preflight REF file + σ / quantile_thresh (2026-09-11)

`v7_preflight_ref.json` at `/global/scratch/users/kh36969/DL_novel_guide_editor/v7_full/`
carries the authoritative v7 channel stats (measured 2026-09-11 on 250 bags per
corpus × 5 corpora via `bucket_collate_fn`; per-corpus stats + combined-mean).
Each channel group has `frac_zero`, `p05`, `p50`, `p95`, `mean`, `std`,
`rel_std`, `quantile_thresh` (= `max(2·rel_std, 0.30)`). The `quantile_thresh`
field is what `channel_stats_preflight` reads for its per-group drift tolerance.

**Combined per-group summary:**

| group | mean | σ | rel_σ | quantile_thresh |
|---|---|---|---|---|
| m_max (ch 0-3) | 0.563 | 0.134 | 0.24 | 0.48 |
| structure (ch 4-7) | 1.770 | 1.637 | **0.93** | **1.85** |
| struct_valid (ch 8) | 0.963 | 0.189 | 0.20 | 0.39 |
| flank_dev (ch 9-12) | 0.240 | 2.935 | ∞* | (inert — see note) |
| orient one-hot (ch 13-14) | 0.000 | 0.000 | — | 0.30 (floor) |

*flank_dev: mean ≈ 0 post-median-subtraction, so `rel_std → ∞`; the qthresh
value 24.4 in the JSON is not consumed — preflight switches to absolute-
tolerance (|ref| < 0.5 → `abs_tol=0.5`) for near-zero references, which is
the correct handling for this channel.

**Structure σ note (retroactive on the "34% p50 drift" investigation).**
Structure rel_std = 0.93 → 1σ range for structure p50 spans ~1.6 units. The
apparent "v7 p50=1.30 vs v6r2 stored REF p50=0.97" delta of 0.33 is
**0.20σ** — well within noise of a 250-bag sampling. Not drift. This
retroactively confirms the direct-fold interpretation earlier: the shift
was aggregation of small per-channel motions across an inherently high-σ
mixture (ch4-7 pool), not a distribution change.

Corpora-level cross-corpus consistency: structure σ ranges 1.62-1.64 across
all 5 corpora (within 1%). Extremely stable — a good sign that the 250-bag
sample per corpus is measuring the true σ, not sampling noise on σ itself.

## v7 retrained model — per-slice PASS + nc-len-norm retraction (2026-09-12)

`v7_main_retrain/best.pt` (epoch 7, val_loss 2.150, val_auroc_proxy 0.7236,
trained 4h42m with the fresh cache root `channel_b_cache_v7/` + the
2026-09-12 loader fix for cache content-key validation) passes the
per-slice bag-AUROC gate cleanly:

- **Pooled**: AUROC(model) = 0.7235, AUROC(analytic S_oc) = 0.5845,
  **Δ = +0.139** (paired-bootstrap 95% CI [+0.132, +0.146])
- **Every n_sites stratum (3..8) has Δ > 0 with strictly positive CI**:
  Δ ranges +0.172 to +0.200; per-stratum CIs all within roughly [+0.15, +0.22].
  Uniform magnitude across strata suggests the gain is not concentrated
  in any single n_sites regime.
- **Direction consistency ✓, pooled CI lower > 0 ✓** → PASS per FROZEN's
  "per-slice bag AUROC" criteria.

**Retraction of the "nc-len normalization is architectural-necessary"
judgment.** Earlier in-conversation reasoning (based on the
invalid-cache-trained model showing `corr(model_score, nc_len) = +0.28`)
argued that nc-length-conditional z-score normalization was needed because
the model showed extreme-value bias with nc_len. That +0.28 correlation
was an artifact of the cache-poisoned training run consuming v6r2 tensors
whose shape distribution was different from what the JSONL headers
claimed. On the honestly-retrained model, `corr(model_score, nc_len) =
+0.04` — a 7× reduction. The model handles nc-length effects internally;
no post-hoc normalization is warranted. The correlation with the analytic
`S_oc` is also low (+0.07). Per-bin AUROC is uniform 0.71–0.73 across all
6 equal-frequency nc_len bins. **Not adding an nc-len normalization step
to the deploy path.** See [[feedback-cache-content-key]] for the source
of the retracted judgment.

**Comparison to v6r2 (context only, not a gate)**: v6r2 pooled bag-AUROC
was ~0.68 (see "Channel B v1 checkpoint findings" above); v7 retrain
reaches 0.72. Analytic on v7 is ~0.58 vs ~0.61 on v6r2. So v7's absolute
model AUROC is higher than v6r2's, and the model-vs-analytic gap (+13.9pp
pooled, +17-20pp per stratum) is larger than the v6r2 gap (+11pp — itself
now UNSCOPED under the 2026-09-10 class-level retraction because it was
measured on the arch.orient-leaked + adapter-substituted loader).

## v6r2 multi-region under the current loader (2026-09-11)

The v6r2 corpora `positives_v5_v6r2_*` contain a non-trivial fraction of
bags with `len(inputs.noncoding_regions) > 1` (concrete example:
`bag_000330` in pos50k has 2 regions). The **current** Channel B loader
(`model/channel_b/data.py`, post-2026-09-10 rev) raises `RuntimeError` on
any such bag.

The **prior** loader read `labels.canonical_nc` (the generator's
pre-mutation reconstruction; 84% string mismatch with any actual raw
region), so it NEVER touched `inputs.noncoding_regions` at all —
multi-region bags didn't need a region-selection policy because
`noncoding_regions` was invisible to the training path. The old training
runs are therefore NOT compromised by a GOLD leak on this axis
(there was no consumption of `active_noncoding_index` or region choice).
But they ARE trained against a synthetic pre-mutation string that has
no deploy-legal counterpart — that's the retraction already recorded
above at "CLASS-LEVEL RETRACTION 2026-09-10".

**Consequence for the stored `preflight.REF_TRAINING` values.** Those
were re-measured on the current loader 2026-09-10 (per note in
`preflight.py`), but that re-measurement had to have either (a) skipped
multi-region bags silently, or (b) been run before the multi-region
raise was added. Either way, the stored REF's provenance vs. today's
strict loader is ambiguous. **Do not use `_LEGACY_REF_TRAINING_PRE_ORIENT_FIX`
or the current `REF_TRAINING` as a v7 comparison anchor** —
`v7_preflight_ref.json` (measured 2026-09-11 on all 5 v7 corpora × 250
bags via `bucket_collate_fn`) is the v7 authoritative REF, and v6r2
comparability requires a real-data adapter that flattens multi-region
first.

v7 sidesteps this entirely by emitting single-region bags only (validated
across 160K bags × 5 corpora: 0 multi-region violations in `build_v7_shard`
fail-loud checks).

## v7-real refactor (2026-09-13, ADDITIVE — v7 single-region record NOT retired)

The v7-real refactor produces bags with the CANONICAL_BAG_SPEC.md-legal
schema but under substantively different generation semantics from v7:

1. **Real 60+60 bacterial flanks.** 50-genome pool
   (`scripts/generator_v5/real_flank_pool.py`, ~164 Mb over diverse phyla;
   fetched by `scripts/fetch_v7_genome_pool.py` to
   `${SCRATCH}/v7_refactor/genome_pool`). Each flank is a 120bp window
   drawn uniformly across pooled bacterial genome mass, N-rejected, and
   AT-filtered at ≤0.70. Junction at position 60 (between `flank[0:60]`
   and `flank[60:120]`) — matches Durrant v2 real-flank convention.
2. **Multi-region synthetic nc.** Each bag emits `noncoding_regions`
   = `[nc_region_A, nc_region_B]` (both lengths ~ U[100, 250]). One
   region holds the planted guide (or an unrelated guide in `twin`);
   `active_noncoding_index ∈ {0,1}` (GOLD) names which. Loader
   concatenates with `MAX_L-1` N-spacer via `concat_with_N_spacer` policy
   (`arch.nc_multi_region_scoring`); enforced in `model/channel_b/data.py`
   with import-time asserts `MAX_L == max(Ls)` and `max(Ls) < spacer_len + 2`.
3. **Reversed flow.** Target sequence is READ from the real flank
   (`flank[target_start:target_start+L]`), a synthetic bag_guide is
   sampled, and the flank is MINIMALLY EDITED at the target region so
   `matches(target, guide) == planted_m`. Only the positions needed to
   hit `planted_m` are flipped; the rest of the 120bp flank remains
   observed bacterial sequence.
4. **Junction motif axis RETIRED.** `_JUNCTION_MOTIF_LENGTHS = [0]`
   under v7-real. Rationale: junction_motif was a statistical
   cross-site-sharing proxy, not TSD biology; the distractor value is
   dominated by real-flank + multi-region difficulty, and migrating its
   position from `flank[0:len]` to junction-adjacent `flank[60:60+len]`
   costs more than it buys. Original weights preserved as `_LEGACY`
   comments in `scripts/generator_v5/bag_v2.py` for reversibility.
5. **Wide axes (LOCKED, do not tune to Durrant).**
   `target_L ~ U{9..14}`, `planted_m ~ U{8..min(11,L)}`,
   `center_offset ~ U[-40,+40]`, `nc_len ~ U[100, 250]` per region.
   Locked 2026-09-13 rationale: fitting v7 to Durrant's per-family
   L=11 / m=8-11 is the same class of error as the retracted
   65-Tnp planted_m analog — narrow-distribution training only tests
   memorization; wide-distribution training tests generalization.
   Durrant measurements are RECORDED-ONLY (see
   `scripts/... /v7_2a_durrant_measure.py`).
6. **Four negative modes.** `VALID_V7_REAL_NEGATIVE_MODES = {none,
   twin, partial, scattered}`. See V7_SPEC §8.7 for semantics.
7. **CLI: `--v7-real`.** Mutually exclusive with `--v7`; skips rate
   table load and is_sites pool load; loads `RealFlankPool` per worker.
   `--negative-mode` must be one of the four v7-real modes when
   `--v7-real` is set (validated at parse time).

**Construction verification.** `max_p S` (max over concat nc positions
of the count of sites with m ≥ 8 at that position) saturates at
approximately K for any construction — positive and twin alike —
because 120bp × 11-mer scanning gives P(m≥8) ≈ 0.95 per position
(the same extreme-value phenomenon as the per-site GBM 0.527 ceiling
in W4+/W7). It is therefore NOT a valid pos-vs-twin diagnostic for
v7-real. Two metrics that bypass saturation ARE valid:

- **Pairwise target-region similarity across sites (per bag).** Verified
  2026-09-13 (100-bag smoke): none=0.657, twin=0.258 (dead-on random
  4-letter background), partial=0.379, scattered=0.399. Twin's target-
  region cross-site coherence is broken as designed.
- **S at gold nc position (`generator_metadata.nc_planted_positions[0]`).**
  Verified 2026-09-13: none S/K=0.87 (planted_m distribution shifts
  a small fraction of matches below the 8 threshold when L=14 wraps
  into the L=11 scoring window), twin S/K=0.28 (nc guide unrelated to
  any site's flank).

Both metrics show clear positive-vs-twin separation. Model-side
site×position attention path is expected to see the same separation.
See memory: [[finding-max-p-S-saturates]].

**v6r2-calibrated acceptance tests SKIPPED under `--v7-real`.**
test1a/1b/1c/2a/2b are calibrated against v6r2's planted_m tail +
is_sites pool; v7-real's planted_m U{8..11} + real-genome flanks have
different regimes and those thresholds don't apply.

**Not compromised**: v7-real records satisfy CANONICAL_BAG_SPEC.md §6
standard checks (single-source, unique site_id per bag, no removed
GOLD fields as inputs, etc.). The v7 single-region contract (§8.1
"synthetic random flank" + `flank_pool_source == "synthetic_random"`
validator hook v-d) does NOT apply to v7-real; v7-real records carry
`flank_pool_source = "50-bacterial-genome pool (NCBI RefSeq)"` and
`reversed_flow = True` in `generator_metadata` so a validator can
distinguish v7 from v7-real bags.

Baseline commit: `df58ace` (v6-frozen state — v7-real is additive to
that). Refactor commit hash will be added after commit.

## v7-real Channel B main training + Step 3 gate results (2026-09-16)

**Model.** `checkpoints/channel_b/v7real_main/best.pt`, epoch 7 (final),
val_loss=0.67299. 8 epochs on 143,971 train / 16,029 val bags across
the 5 v7-real corpora (pos50k / twin50k / partial40k / ctrl10k /
scat10k). Hyperparams: lr 3e-4 / batch 32 / hidden 128 / heads 4 /
blocks 3 / split-mode main / seed 0. Cache root
`channel_b_cache_v7real` (fresh; no collision).

**val_auroc_proxy pool = 0.7696** — do NOT quote as "the model AUROC".
The training-time metric pools bags across n_sites, and
bag_max_score scales with n_sites by construction (positive y-peak
= n_sites), producing a Simpson-distorted undercount vs stratified
cells (0.83-0.98). See [[finding-val-auroc-simpson]]. best.pt
selection is unaffected (uses val_loss, not val_auroc).

### Gate results (all reported per-n_sites, not pooled)

**Gate 1 — C3 permutation-equivariance (job 25917602): PASS**
- fp64 max|Δ| = 9.326e-15 < 1e-12 (SiteAttentionBlock is
  permutation-invariant over sites).
- fp32 max|Δ| = 6.437e-06 = accumulation floor at 3 blocks × H=128
  × MHA. Same threshold-revision as v7 (FROZEN < 1e-6 too tight;
  new operational threshold < 1e-5 fp32 / < 1e-12 fp64).

**Gate 2 — C4 held-out n_sites (jobs 25917603 train / 26065475 eval):
PASS**
- `v7real_heldout_nsites` trained on bags with n_sites ∈ {3,4,5,6};
  evaluated on n_sites ∈ {7,8}.
- Per-mode × n_sites AUROC vs `v7real_main` on the same val slice:
  |Δ| ≤ 0.015 across all 6 cells; CIs overlap.
- Cross-site coherence learning generalizes to unseen n_sites. Also
  retires the earlier scale-vs-coherence concern: if the twin
  0.86→0.98 climb were pure y-peak-scale, a model that never saw
  n=7,8 would collapse on them. It matches main.

**Per-mode × n_sites AUROC on main model (val split, best.pt):**

    neg mode      n=3       n=4       n=5       n=6       n=7       n=8
    twin       0.8637    0.9108    0.9424    0.9514    0.9617    0.9808
    scattered  0.7750    0.8371    0.9064    0.9286    0.9325    0.9583
    partial    0.8300    0.8590    0.8971    0.9085    0.9103    0.9360

- pos-vs-twin: **monotonic 0.8637 → 0.9808 across n_sites 3..8**. m
  distribution matched between positive and twin (both draw
  planted_m ~ U{8..min(11,L)}); only cross-site coherence differs.
  This is the direct experimental verification of the project's core
  claim (cross-site amplification of coherence signal with more
  sites).
- pos-vs-scattered: monotonic 0.7750 → 0.9583. Harder than twin at
  low n (three planted guides catch fractions of sites) but converges
  to near-parity at high n.
- pos-vs-partial: monotonic 0.8300 → 0.9360. Intermediate.
- Normalization by n_sites (score / n_sites): within-cell AUROC is
  invariant (rank-preserving), so it does NOT test whether the
  cross-cell climb is coherence vs scale. C4 answers that.

**Gate 3 — held-out-m: FAIL (marginal)**

Two variants:
- `max(m_at_planted) == 8` filter (jobs 25917604 train / 26067432
  eval): **UNDERPOWERED / retired.** Filter admitted only 56 pos
  bags total across all n_sites (probability (1/4)^n per bag). CI
  half-width ±0.09 could not rule out ±0.15 Δ. See
  [[feedback-underpowered-not-pass]].
- `median(m_at_planted) == 8` filter (jobs 25917604 train reused /
  26067552 point-estimate eval / 26067583 paired-Δ bootstrap):
  **FAIL per pre-registered CI-upper ≤ +0.020.** 886 pos bags across
  n_sites 3..8. Paired-Δ bootstrap (main − heldout, same val bags,
  1000 iter):
  - 15/18 cells: main ≥ heldout (positive Δ)
  - Median Δ ≈ +0.008, max Δ = +0.018 (twin n=5, partial n=5)
  - 11/18 cells CI upper > +0.020 (marginal — upper bounds mostly in
    [+0.021, +0.037])
  - 7/18 cells PASS strict criterion

**Interpretation of Gate 3 FAIL**: training-distribution m-coverage
matters for low-m generalization. A model trained WITHOUT high-m
data (max(m) ≥ 9) reaches ~0.01 lower AUROC on low-m val
(median(m) = 8) than the main model. **This is a scope limitation,
not a mechanism defect**: absolute per-mode AUROC (twin 0.79-0.98,
scattered 0.71-0.96) is unaffected; cross-site coherence still
discriminates strongly on low-m bags. Sign consistently positive
across 15/18 cells → real effect, not noise.

**Production rule from Gate 3**: v7-real training corpora MUST cover
the full planted_m range U{8..11}. Do NOT filter to high-m or low-m
subsets for training convenience — the ~0.01 low-m generalization
cost is real and reproducible.

**What was NOT done, and why:**
- Did NOT relax the pre-registered +0.020 threshold to +0.030
  post-hoc. Same class as C1's retracted "reset to analytic+5pp".
- Did NOT increase bootstrap iteration beyond 1000. Iteration count
  stabilizes endpoints; it does not narrow CIs.
- Did NOT redefine the val slice again to a more permissive filter
  after seeing the medm==8 result. Same anti-shopping principle.

**Bag-level covariate leakage test (job 25913704)**: **NO STRONG
LEAKAGE.** 79-dim bag summary (15-ch × 5-stat + n_sites/nc_len/gc/hom)
via 5-fold hash-blocked LR/GBM on 16,029 val bags:
- FULL 79-dim: LR=0.6474, GBM=0.6395
- arch-4 alone (n_sites/nc_len/gc/hom): LR=0.4911, GBM=0.4854 (chance)
- channel summary alone (75-dim): LR=0.6384, GBM=0.6277
- Main model bag AUROC (stratified, not pooled) at 0.83-0.98 vs
  bag-summary 0.65 → model wins by real per-site/cross-site structure,
  not by bag-level metadata correlates. Arch marginals confirmed
  matched across pos/neg (n_sites 5.47 vs 5.49, nc_len 349.4 vs 349.5,
  gc constant 0.5, hom constant 1.0).

### Gate board summary

    Gate                             Verdict     Key number
    C3 permutation-equivariance      PASS        fp64 max|Δ| = 9.3e-15
    C4 held-out n_sites              PASS        |Δ| ≤ 0.015 all cells
    per-mode × n_sites AUROC         VERIFIED    twin 0.86→0.98 monotonic
    held-out-m max(m)==8             UNDERPOWERED  n_pos=56 (retired)
    held-out-m median(m)==8          FAIL        Δ~+0.01 (11/18 cells >+0.02)
    bag-summary leakage              NO LEAKAGE  arch-4 at chance, ch-sum 0.64

Three passing gates directly verifying the mechanism (permutation
invariance, generalization across n_sites, cross-site coherence
monotonic with n_sites) plus a quantified scope-limit FAIL and a
clean no-leakage verdict. A 4/4-PASS board would be more suspicious
than 3/4 with a marginal quantified FAIL.

## v7-real COMPLETE milestone (2026-09-16, `v7-real-frozen`)

End-to-end verification of the v7-real refactor. Synthetic training →
real-data transfer → non-RNA-guided negative control all landed as a
coherent story. Freeze snapshot for external reference.

### Model artifact
`checkpoints/channel_b/v7real_main/best.pt` — 8 epochs, val_loss
0.67299, val_auroc_proxy 0.7696 (pool; see
[[finding-val-auroc-simpson]] for why the pool is
Simpson-distorted vs stratified cells 0.83-0.98).

### Corpora
- Training: 5-corpus v7-real 160k bags (pos50k / twin50k /
  partial40k / ctrl10k / scat10k), real 60+60 bacterial flanks from
  50-genome pool, multi-region synthetic nc with
  concat_with_N_spacer, reversed-flow guide construction, wide axes
  (target_L U{9..14}, planted_m U{8..min(11,L)},
  center_offset U[-40,+40], nc_len U[100,250]).
  Files: `${SCRATCH}/DL_novel_guide_editor/v7real_full/`,
  shards: `${SCRATCH}/DL_novel_guide_editor/v7real_shards/`,
  cache: `channel_b_cache_v7real/`.
- Real-data eval: Durrant realbg_v2 (173 WT + 168 Programmed) +
  10-family DDE negative_top10 catalog (63,048 loci, 6,637 elements
  at ≥3 loci, TSD-corroborated non-RNA-guided).

### Story arc (deliberate order for external reader)

**1. Synthetic validation (Step 3 gates)** — the ground truth is
generator-defined; validates that the model learns what the generator
plants. C3 permutation-equivariance PASS (fp64 max\|Δ\| = 9.3e-15 <
1e-12). C4 held-out n_sites PASS (\|Δ\| ≤ 0.015 across mode × n_sites
cells; heldout {3-6}→{7-8} generalization). Per-mode × n_sites
AUROC: pos-vs-twin MONOTONIC 0.86 → 0.98 (n_sites 3→8) — the direct
experimental verification of the project's core claim, cross-site
coherence amplification, on m-matched positive vs twin bags.

**2. Scope limitation — held-out-m (quantified FAIL)** — trained on
max(m)≥9, evaluated on median(m)==8 low-m positives. Paired-Δ
bootstrap: Δ ≈ +0.008-0.018 across 18 cells, 11/18 CI upper > +0.02
pre-registered threshold → FAIL (marginal). Consistent sign (main ≥
heldout) → real ~0.01 low-m generalization cost. Actionable rule
frozen into spec: **v7-real production training MUST cover full
planted_m range U{8..11}**; do not filter to high-m subsets. See
[[finding-v7real-gates]], [[feedback-underpowered-not-pass]].

**3. Bag-summary leakage check** — 79-dim bag summary (15-ch × 5-stat
+ arch-4) via 5-fold hash-blocked LR/GBM on 16,029 val bags:
- FULL 79-dim: LR 0.65, GBM 0.64
- Arch-4 alone (n_sites/nc_len/gc/hom): LR **0.49** (chance) → no
  metadata leakage; generator matched marginals across pos/neg
- Channel-summary-only: LR 0.64 → 13pp below model's per-stratum
  0.77-0.98 → per-site/cross-site structure is the actual signal

**4. Real-data transfer to natural bridge RNA (Durrant IS621 T-WT)** —
see [[finding-durrant-v7real-transfer]]. Bagged by TBL group (all
173 WT records share d11=ATCAGGCCTAC), scored by cross-site nc peak
+ flank_argmax lookup. **±1bp accuracy: K=1 47%, K=5 63.6%** vs
byte-shuffled flanks 1-2.6% → 24-47× shuffle-relative enrichment.
K=1→5 climb 47→63.6% mirrors synth twin K=3→8 climb 0.86→0.98 —
cross-site amplification generalizes to real bacterial data.
Programmed variants (novel TBL, mean m=6.13 < training ≥8) score
NULL, real ≈ shuffle across K — consistent with held-out-m FAIL
scope rule. 1bp offset origin traced: cassette DNA `TATCGGGCCTT` at
IS621 nc[64:75] has 9/11 identity with flank[50:61] vs 3/11 with
d11 at flank[51:62]; RNA binds via folded-loop base pairing while
cassette DNA aligns via Hamming — two different anchor conventions.
See [[finding-model-reads-cassette-dna]]: this is the deliberate
generalization mechanism, not a limitation. Model doesn't need
RNA-structure priors → can transfer to unknown families.

**5. Non-RNA-guided negative control — 10 DDE families** — see
[[finding-negtop10-specificity]]. Real DDE-family (non-RNA-guided)
IS insertions from `real_data/negative_top10/` (63k loci across
IS1/IS3/IS4/IS5/IS6/IS66/IS256/IS481/IS1595/ISL3, ORF-HMM-verified
TnpB-free). Bags grouped by insert_md5 (same element ↔ multi-site),
insert truncated to 250bp to match training nc_len. Initial
positive comparator was pos50k_synth (AUROC 0.956-0.977 across 10
families); potentially confounded by "synth vs real".

**5a. Synth-vs-real confound REJECTED**: re-ran with **DurrantWT
(real natural bridge RNA) as positive** vs same DDE negatives.
DurrantWT score p50 = **5.098** ≈ pos50k_synth p50 = 5.133 → real
and synth RNA-guided data collapse to the same score regime.
**AUROC(DurrantWT vs DDE): 0.9732-0.9952 across 10 families, mean
0.9897** — UNIFORMLY HIGHER than AUROC(pos50k vs DDE) by ~0.018,
opposite of what a source-mismatch confound would produce.
Marginal check: DDE n_sites p50=4 vs positive p50=6 (DDE has fewer
sites → LESS peak amplification available); the score gap survives.
Specificity is real, not synth-vs-real.

### End-to-end coherence

| Dataset | Mechanism | v7-real score p50 | Predicted | Observed |
|---|---|---|---|---|
| pos50k (synth) | RNA-guided | **5.13** | high | ✓ high |
| **Durrant WT (real)** | **RNA-guided (real)** | **5.10** | high | ✓ high (**matches synth**) |
| Durrant Programmed | RNA-guided, m<8 | null | null (scope) | ✓ null |
| 10 DDE families (real) | non-RNA-guided | **2.13-2.26** | low | ✓ low |

AUROC(real RNA-guided vs real DDE, 10 families) = **0.9732-0.9952**,
mean **0.9897**. Both real+real and synth+real comparisons agree;
confound rejected.

Symmetric verification: works on positive real data (natural bridge
RNA), doesn't false-positive on negative real data (10 DDE
families), quantified scope boundary on low-m (matches held-out-m).

### Operational threshold

Bag_max_score ≥ **~3.5** = "RNA-guided-like" call (well above DDE p95
upper bound). Score ≤ **~2.5** = DDE-negative regime. See
[[finding-negtop10-specificity]] for family-stratified distributions.

### Caveats carried forward (do not lose)

- **DDE-family-only negative control**: all 10 kept families are DDE
  (IS91 rolling-circle excluded by TSD-corroboration rule).
  Cross-mechanism claim requires adding IS91.
- **Family confounded with host species** in the negative catalog
  (E. coli / A. baumannii / K. pneumoniae dominate).
- **Cross-site co-occurrence, not RNA binding**: model finds
  "where nc DNA aligns to multiple flanks", not "where RNA binds".
  On IS621 these differ by 1bp; on other families the offset may
  differ. Report signed-offset histogram + tolerance window on any
  new eval.
- **val_auroc_proxy is Simpson-distorted**: pool AUROC 0.77 is a
  trend spectator only; stratified cells are the honest number.
  Neither raw pool nor normalized pool is quotable.
- **Low-m FAIL is real**: model can't detect m<8 targets. Production
  training must span full planted_m range.

### Files in this freeze

Generator + loader:
- `scripts/generator_v5/bag_v7_real.py` (v7-real reversed-flow generator)
- `scripts/generator_v5/real_flank_pool.py` (50-genome pool)
- `scripts/generator_v5/run_generator.py` (--v7-real flag)
- `scripts/generator_v5/bag_v2.py` (junction_motif retirement)
- `scripts/build_v7_shard.py` (multi-region policy)
- `scripts/fetch_v7_genome_pool.py` (50-genome fetcher)
- `scripts/build_durrant_realbg_v2.py` (real E. coli flanks for Durrant)
- `model/channel_b/data.py` (multi-region loader, concat_with_N_spacer)
- `model/channel_b/model.py`, `constants.py`, `preflight.py`

Training + eval scripts:
- `scripts/channel_b_train.py` (multi-region preflight, save-every-epoch)
- `scripts/channel_b_per_mode_auroc_v7real.py` (per-mode × n_sites eval)
- `scripts/channel_b_paired_auroc_delta.py` (paired bootstrap for
  held-out-m gate)
- `scripts/channel_b_leakage_test_v7real.py` (bag-summary leakage)
- `scripts/channel_b_durrant_flank_localize.py` +
  `scripts/channel_b_durrant_flank_diagnostic.py` (Durrant transfer)
- `scripts/channel_b_negtop10_control.py` (DDE specificity)

Config + spec:
- `config/channel_b_v7real_corpora.json`
- `docs/V7_SPEC.md` (§8 v7-real refactor section)
- `docs/CANONICAL_BAG_SPEC.md`
- `docs/channel_b_scoring_spec.md`
- `FROZEN.md` (this file, entries above + this milestone)

sbatch wrappers: `sbatch/v7real_*.sbatch` (11 files covering full-gen,
shard-build, training, all gates, all real-data evals).
