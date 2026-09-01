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

### A5 — architecture.py axis uniformity

- 25 chi-squared checks on 10K draws each: L uniform, N_nc uniform on {1,2,3}, TSD-width uniform on {0,2,5,8,9,12}, NCR-pos uniform on 3 states.
- is_split freq = DEFAULT_SPLIT_PROB ± 0.02.
- planted_m tail 86/10/4 at (target_m, target_m-1, target_m-2), zero above.

### A6 — acceptance b validation

- T-WT (n=170 sites): all 5 tests PASS (Test1a/1b/1c/2a/2b).
- V4.2 (n=2000 sites): FAIL on Test1a + Test2a (as required for the counterfactual to reject a known-defective corpus).

### V5 corpus anchors (informational, not enforced by run_all_anchors)

Recorded from the 50K positives + 10K negatives generated 2026-09-01:

**Positives**, Mode 2 (m=8, τ=0, S=5), L=11 subset:
- coverage = 0.443 (analytic prediction 0.86^5 = 0.470)
- PPV_Tnp = 0.969
- exact_rate = 0.414

**Negatives**, same spec, all L:
- FPR = 0.0276 (Durrant shuffled null comparable: 0.021)
- Per-L FPR (L=11/12/13/14): 0.033 / 0.030 / 0.028 / 0.020

Analytic q^5 = 0.0327 / n_positions_L11 → q ≈ 0.18, matches the calibrated 0.21.

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
