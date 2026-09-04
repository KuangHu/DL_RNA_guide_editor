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

Pool AUROC ≈ 0.50 was hiding residual ~1pp per-channel bias from imperfect percentile matching (mixed direction per channel). Combined 5-channel signal ≈ √5 × 1pp ≈ 2.2pp → AUROC ≈ 0.52 under gaussian ROC. Below the pre-registered Channel B C6 gate (0.55) by 3pp margin: twin corpus is a valid adversarial baseline, but the paired test is the load-bearing measurement, not the pool AUROC.

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
