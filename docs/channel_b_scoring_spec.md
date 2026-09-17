# Channel B — frozen scoring spec (v1, 2026-09-06)

Locks the evaluation pipeline for every downstream comparison
(AttrIndex, oracle sweeps, slice gates, cross-corpus deploy). Nothing
in this file gates model quality — it only pins the metric definitions
so subsequent measurements are commensurable.

**Applies to**: current main best.pt (epoch 7, val_auroc_proxy=0.6825)
and any future Channel B checkpoint.

---

## Bag score — deployment recipe

The model output is `pred[bag, position]`, shape `(nc_len_eff,)`. The
deployment bag score is:

```
bag_max = max_position pred[bag, position]  over real (unpadded) positions
```

Raw `bag_max` is NOT calibrated across n_sites (the model's per-position
scale grows sublinearly in n_sites relative to the ideal 1.41 log-odds
slope; C2 gate FAIL, slope 0.813). Use the deployment recipe below
for cross-n_sites comparability.

### Deployment recipe (neg-only calibration, label-free)

**Inputs**:
- `bag_max` for each bag under evaluation
- A **background reference pool** — any set of ≥ 100 bags per n_sites
  bin drawn from the target deployment distribution or from the
  training corpus's twin/scattered/none bags. Used only for its
  score distribution; labels are not required.

**Per-bin statistics**:
For each n_sites bin `b`, compute on the background pool only:
```
μ_b = mean(bag_max)      over background bags with n_sites = b
σ_b = std(bag_max)       over background bags with n_sites = b
```

**Score transform**:
```
bag_calibrated[k] = (bag_max[k] - μ_{n_sites(k)}) / σ_{n_sites(k)}
```

Threshold `bag_calibrated` on a fixed cutoff chosen per deploy target
(e.g. `bag_calibrated ≥ 1.5` for a specific FPR).

### Extrapolation rule for unseen n_sites

Fit a monotone function to `(n_sites, μ_b)` and `(n_sites, σ_b)` from
the training/background pool:

```
μ(n) = a₀ + a₁ · n     (linear fit; correct sign per C2 mechanism)
σ(n) = b₀ + b₁ · √n    (√n scaling from central-limit intuition)
```

For any bag with n_sites outside the fitted range, use the fitted
`μ(n), σ(n)` values.

**Verification requirement**: the extrapolation must be validated on
held-out n_sites before use. The nsheld corpus (trained on n_sites
∈ {3-6}, val on n_sites ∈ {7, 8}) is the standing test set. Pooled
AUROC on nsheld val after z-scoring with extrapolated (μ, σ) must
be within 0.02 of the same-n_sites pooled AUROC using empirical
(μ, σ) from the val set itself.

---

## Metric definitions (immovable across future measurements)

| metric | definition | pool | notes |
|---|---|---|---|
| `bag_max_raw` | max_position(pred) | any | pre-calibration, use only for backwards-compat |
| `bag_max_calibrated` | (bag_max − μ_bin) / σ_bin | any + bg ref | deploy recipe |
| **`per_slice_AUROC[n_sites=b]`** | roc_auc_score(y, bag_max) restricted to bag n_sites=b | main val or nsheld val | **mechanism metric** — measures within-slice separation |
| **`pooled_AUROC`** | roc_auc_score(y, bag_max_calibrated) on all bins pooled | main val | **deployment metric** — cross-bag comparability |
| `within_bin_mean_AUROC` | mean over bins of per_slice_AUROC[b] | main val | reference — should ≈ pooled_AUROC after calibration |

### Current recorded values (main best.pt, epoch 7)

| slice | model raw | model calibrated | analytic raw |
|---|---|---|---|
| n_sites=3 | 0.7147 | — | 0.5984 |
| n_sites=4 | 0.7441 | — | 0.6336 |
| n_sites=5 | 0.7687 | — | 0.6676 |
| n_sites=6 | 0.8010 | — | 0.6854 |
| n_sites=7 | 0.8199 | — | 0.7020 |
| n_sites=8 | 0.8388 | — | 0.7127 |
| **within-bin mean** | **0.7812** | — | 0.6666 |
| **pooled (raw)** | **0.6825** | — | **0.6800** |
| **pooled (z-score all, label-informed)** | 0.7827 | — | 0.6760 |
| **pooled (z-score neg-only, deploy recipe)** | **0.7821** | — | 0.6755 |
| paired Δ (model − analytic) after neg-only z | | | +0.1066, 95% CI [+0.099, +0.116] |

### Recorded quantities from same-pool paired analyses

**C2 slope diagnostic** (channel_b_c2_dose_response.py, main best.pt):
- Pooled OLS: `score_at_planted ≈ 1.358 + (−0.224) × target_m + 0.813 × n_sites`
- The n_sites coefficient +0.813 is under the spec ideal 1.41 by a factor of 0.577
- This is the same defect as `pooled_raw − within_bin_mean = −0.099` (both measure the model's under-scaling on n_sites)

**Coherence-shift diagnostic** (channel_b_coherence_shift.py, same pool):
- model: 0.6825 → 0.4844 (drop 0.198)
- analytic: 0.6797 → 0.5064 (drop 0.173)
- Both collapse to chance under per-site position shuffling → confirms model uses cross-site position-alignment mechanism, not a shortcut. Shift-end residual asymmetry (±1–2 pp) is a perturbation artifact, not evidence of differential mechanism dependence.

**H1 verdict — falsified** (channel_b_per_ns_decomp.py, paired same pool):
- Per-slice Δ (model − analytic) is **uniform** at +11-13 pp across n_sites 3-8
- Slopes of AUROC vs n_sites are essentially identical between model and analytic (~+2.3 pp / n_sites unit)
- Model's advantage is in the **per-site** term, not in cross-site aggregation

**H3 verdict — falsified** (same experiment):
- Negative-mode composition (none/partial/scat/twin) is uniform across n_sites bins
- The +11 pp advantage is not a negative-class geometry artifact

---

## Change protocol

Any change to the pool, background reference, extrapolation function,
or metric definitions requires updating this file with a dated note
and re-computing all pinned values in the current-values table.

## Stopping condition for the +0.3σ / +11 pp source hunt

Three hypotheses on where the model's uniform per-slice +11 pp
advantage over analytic log-odds comes from have been falsified:

| hypothesis | verdict | test |
|---|---|---|
| Concentrated on high n_sites | falsified | per_ns_decomp: Δ is uniform +11 pp across n=3..8 |
| H_pair↔E_span joint constraint | falsified | corr_stratify: Δ_sep is not monotone in \|ρ\| |
| Soft-threshold on m_max (m∈{6,7} edge evidence) | falsified | soft_threshold: analytic AUROC peaks at k=8, not k<8 |

**Registered stopping condition (2026-09-06):**

The current experiment (m_max-only retrain: does the model reach the
same 0.78 per-slice using only m_max channels?) plus up to three
leave-one-channel-group retrains (structure / flank / orient) exhaust
the direct-inspection avenue. If none of these localize the source,
the +0.3σ is recorded as **UNEXPLAINED** and:

- Channel B (B) retrain stays on **indefinite hold** — no known
  architecture change targets an unidentified mechanism
- Effort pivots to Durrant real-data evaluation using the current
  main best.pt + the deploy recipe defined above (per-bin z-score
  from a background pool)
- 0.3σ becomes a known-quantity anomaly, not a bug

The +10.2 pp pooled AUROC after neg-only z-score calibration remains
the load-bearing deliverable regardless of source attribution.

## Not covered here

- AttrIndex (Δ-C1_LOO based) — separate spec, will be registered before
  oracle runs
- Slice gates (Channel A failure axes) — spec cleared; three hypothesized
  failure axes were falsified above. Any future slice gate must have
  a mechanism-first justification, not "candidate axis to check".
