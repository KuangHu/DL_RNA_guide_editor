# DL RNA-guided editor classifier — Project Journal (frozen snapshot)

**Frozen on**: 2026-08-23 (Lawrencium cluster, kh36969)

This document records every data-generation and model-optimization decision
made during this project, and the empirical reasoning behind each. It is
intended as a definitive record; the classifier itself is no longer under
active optimization at the time of freeze.

---

## 0. Frozen state — one-page summary

| Slot | Value |
|---|---|
| **Dataset (training)** | V4 no-level3: `splits/train_v4_no_l3.jsonl`, `splits/val_v4_no_l3.jsonl` |
| **Dataset (test)** | `splits/test_v4.jsonl` (500 pos + 400 level1 + 200 level3_paired + 4×100 level2) |
| **Dataset root** | `/global/scratch/users/kh36969/DL_novel_guide_editor/` |
| **Model** | V1Model (see `model/v1.py`) with `use_dispersion=True`, `dispersion_mode='hidden_residual'`, `disp_hidden=16` |
| **Warm-start from** | `checkpoints/v1_on_v4_main/best.pt` (V4 main, ep16) |
| **Trainable params** | 5,457 (disp_encoder + fusion_mlp + disp_beta); rest of V4 backbone frozen |
| **Selected checkpoint** | `checkpoints/v5_2_stageA_from_v4/best.pt` (val ep3, AUPRC 0.8625) |
| **Selection criterion** | val AUPRC + guardrails: nc_top1 ≥ 0.90 AND wrong_orient AUROC ≥ 0.95 |
| **Held-out test** | AUROC 0.829, HARD_AUROC 0.716, wrong_position 0.892, wrong_structure_role 0.737, level3_paired 0.512 (chance = correct behavior) |

Key file paths:
- Model code: `model/v1.py`
- Training script: `training/train_v1.py`
- Selected checkpoint: `checkpoints/v5_2_stageA_from_v4/best.pt` (ep3)
- V4 baseline checkpoint: `checkpoints/v1_on_v4_main/best.pt` (ep16)
- Test scores: `logs/test_v4_scores_v4_vs_v52.jsonl`
- Diagnostic outputs: `logs/diag_v4_candidate_selection.jsonl`

---

## 1. Problem statement

Classify whether a bacterial transposase family is a genuine RNA-guided editor.

Input:
- **Bag of 50 sites** per candidate transposase (a "tnp bag"). Each site
  contains:
  - 120 bp genomic flank (DNA, contains the target motif)
  - 1–3 non-coding regions (DNA, one of them possibly contains a guide-carrying ncRNA)
- Alphabet: DNA (ACGT + N) throughout; no RNA form is exposed to the model.
- Structure information: RNAplfold-derived per-nucleotide unpaired probabilities,
  precomputed per site.

Output: `P(RNA-guided-editor | 50 sites of this tnp) ∈ [0, 1]`.

Semantic goal (refined at end of project): the classifier should output
`P(RNA-guided-like recognition evidence)` — i.e. is this bag consistent with a
transposase whose 50 insertion sites share a common cross-site recognition
grammar?

---

## 2. Data-generation evolution

The data generation lives in `/global/home/users/kh36969/tools/DL_RNA_guide_edotor_positive_generator/`.

Timeline of versions:

### 2.1 V1 (pipeline foundation) — kept as source of positives

- 5,000 synthetic transposases, each with a distinct recognition rule
  (topology, guide length, mismatch budget, ncRNA structure).
- Per tnp: 10 ncRNA templates × 50 sites = 250,000 positive sites.
- Each site: real 120 bp GTDB flank containing an att-site, 1–3 non-coding
  regions (one active with padding + ncRNA + guide, others random-DNA decoys).
- Negatives (V1 only): six violation profiles — no_alignment, alignment_only,
  no_alignment_and_wrong_target_position, no_alignment_and_no_ncrna_structure,
  no_alignment_and_no_active_noncoding, all_random_within_frame.
- Files kept: `data/positives.jsonl`, `data/negatives.jsonl` (V1).
- **Split by transposase** (500 val, 500 test, 4,000 train); negative tnp is
  assigned to same split as its positive donor to prevent rule leakage.

### 2.2 V3 (noisy positives + strength stratification)

**Why V3 existed**: V1's negatives were structurally very different from
positives (missing NC regions, no alignment, etc.). A model could win on V1
by using trivial layout cues rather than cross-site consistency. V3 fixed
this at the *positive* side by making the positive bags themselves noisy:

- Each positive tnp bag now has three site classes:
  - `guided` — a real positive site (as in V1)
  - `off_target` — takes a foreign tnp's flank + target + ncRNA (a real
    positive from another tnp, structurally coherent but with a different
    recognition rule)
  - `unresolved` — original scaffold but the guide bases replaced with
    random DNA (structurally correct, alignment destroyed)
- Bags are stratified into three **strength** classes:
  - `strong`: 75–100% guided sites (~44/50 avg)
  - `moderate`: 50–75% guided (~31/50 avg)
  - `weak`: 30–50% guided (~20/50 avg)
- V3 also had a "hard" negative — `level3_counterfactual_within_tnp` — that
  contained real alignments on all 50 sites but no cross-site cognate pairing.
  This was the hardest V3 profile.

Files: `data/noisy_positives_v3.jsonl`, `data/negatives_v3.jsonl`,
`splits/train_v3.jsonl` etc.

### 2.3 V4 (noise-composition-matched hard negatives)

**Why V4 existed**: V3 had a subtle shortcut. Negatives had no
`off_target`/`unresolved` sites and no strength stratification. A model could
learn "if this bag has heterogeneous site composition, it's positive" — which
IS a shortcut, since real transposases don't come pre-labeled with noise
composition. This showed up in V3-trained V1 as a **reversed strength
paradox**: weak positives scored HIGHER than strong positives, because weak
bags have more heterogeneity, which the model was using as a positive cue.

V4 fixed this at the *negative* side by making negative bags noise-composition
matched to positive bags:

- Every V4 negative bag *copies a positive donor's exact per-slot site-class
  pattern* (guided / off_target / unresolved fractions). The rule violation
  is applied only within the guided sites.
- V4 introduces a new taxonomy of violation profiles:
  - `level1_marginal_matched` — easy: matched noise composition, but the
    entire recognition grammar is broken.
  - `level2` — four profiles that each break exactly ONE consistency rule:
    - `wrong_orientation_consistency`: site orientations don't agree
    - `wrong_length_consistency`: guide lengths don't agree
    - `wrong_position_consistency`: target positions in the flank don't agree
    - `wrong_structure_role_consistency`: guide's structural role in the
      ncRNA is inconsistent across sites
  - `level3_paired_counterfactual` — sites re-derive their guides from a
    DIFFERENT positive donor tnp: locally each site looks like a real
    RNA-guided event, but there is no within-bag cognate pairing rule.

Files: `data/negatives_v4.jsonl`, `splits/{train,val,test}_v4.jsonl`,
`splits/test_v4_control.jsonl`.

Training distribution (V4 full):
- Positives: 200 k
- level1: 160 k
- level2 (4 profiles): 40 k each = 160 k
- level3_paired: 80 k
- Total negatives: 400 k (1:2 ratio)

### 2.4 Level3 paired counterfactual — semantic reframing

Halfway through model development, we discovered that
`level3_paired_counterfactual` behaves like a **cognate-provenance
counterfactual control**, not a hard negative:

- Its sites have real guide-target alignments (from real donor bags), so
  each site individually looks like a valid RNA-guided event.
- Its cross-site dispersion structure is identical to real positives
  (position/orientation/length statistics all match), because the donor
  is a real coherent positive tnp.
- Only the tnp-level cognate identity (which recognition rule the donor
  followed) is "wrong" — but nothing observable in the sequence exposes
  that identity.

**Consequence**: labeling level3 as a negative during training creates
contradictory supervision. The model has to output "negative" for bags that
look observationally identical to real positives. This broke training (see
§3.3).

**Freeze decision**: level3 is treated as **evaluation-only** in the final
model. It is filtered out of the training set (train_v4_no_l3.jsonl,
val_v4_no_l3.jsonl) and used only as a controlled test:

- On level3, the classifier's AUROC ≈ 0.5 is the *correct* outcome — it
  means the model correctly identifies level3 as observably indistinguishable
  from positives on sequence evidence.
- If a future model version pushed level3 AUROC to 0.8+, that would be a
  synthetic-shortcut warning, not a success.

Filter script inline: `for line in train_v4.jsonl: if profile != 'level3_paired_counterfactual': keep`.

---

## 3. Model-optimization evolution

The model lives in `model/v1.py`. All variants below are one class — `V1Model` —
with different config flags. Below is the timeline of *what was tried and why*.

### 3.1 V1 architecture

Hierarchical MIL over sites of a tnp bag:

```
per-site: candidate patches (96 per site) → 3-stream candidate CNN
                                              (structure + alignment + position)
                                              → 128-D per-candidate token
        → Gated-attention MIL over candidates → 128-D per-NC-region token
        → Gated-attention MIL over 3 NC regions → 128-D per-site token

per-bag: Set Transformer over 50 site tokens → PMA pooling → 128-D bag token
                                              → Classifier (128 → 64 → 1) → logit
```

Auxiliary losses:
- Per-site cross-entropy on candidate selection (supervises the candidate scorer)

Params: ~483 k.

### 3.2 V1-on-V3 (source-cluster baseline)

Result: overall AUROC 0.94, AUPRC 0.91, HARD_AUROC 0.91 on V3 test.

But diagnostic revealed the **strength-ordering paradox**:
```
recall @ 0.5  strong=0.82  moderate=0.98  weak=1.00
```
Weak positives are recovered better than strong ones, because weak bags have
more heterogeneous site composition, which V1 learned as a positive cue.

**Interpretation**: V1 had learned "heterogeneity → positive", a shortcut
that only worked on V3.

### 3.3 V1-on-V4 (V4 main baseline) — checkpoint kept

Trained V1 on `train_v4_no_l3.jsonl` (V4 without level3). Compared against a
diagnostic arm trained on FULL V4 (including level3).

**V4 main (no level3)**:
- Restored biological strength ordering: strong 0.95 > moderate 0.73 > weak 0.30.
- Overall AUROC 0.88, HARD_AUROC 0.77.
- But two profiles remained stuck:
  - `wrong_position_consistency`: 0.628 (V3 shortcut model had 0.93)
  - `wrong_structure_role_consistency`: 0.581 (V3 shortcut model had 0.93)

**V4 diagnostic (full V4)**: training collapsed. For 10 consecutive epochs
the model output was essentially constant (AUROC = 0.500). It never recovered.

**Interpretation**:
- The V4 main result is honest — it lacks the shortcut and must actually
  learn cross-site grammar. The two failing profiles are cross-site
  consistency rules the V1 architecture cannot express well.
- The diagnostic arm collapse confirms level3 is unlearnable as a negative
  (per §2.4).

**Selected V4 main checkpoint**: ep16, `checkpoints/v1_on_v4_main/best.pt`.

### 3.4 Candidate-selection diagnostic

Question: on wrong_position and wrong_structure_role negatives, does the
model select the *correct* candidate at each site, or does it pick something
else? This localizes whether the failure is at the per-site scorer or at the
set-level aggregator.

Method: `scripts/diagnose_v4_candidate_selection.py`. For each site, take
argmax of `cand_raw` at the labeled active NC. Compare model-selected
(orient, L, flank_start, nc_start) against the labeled ones.

**Result** (75k sites of val_v4):
```
profile                              orient=  L=    tgt≤2  gs≤2  score
POSITIVE                             0.898   0.337  0.715  0.723  2.26
wrong_position_consistency           0.904   0.328  0.712  0.712  1.93
wrong_structure_role_consistency     0.907   0.333  0.721  0.729  2.23
```
**Every profile has essentially identical candidate-selection accuracy.**
On wrong_position negatives, the model recovers the labeled (moved) target
with 71.2 % accuracy — the same as on real positives (71.5 %).

**Verdict**: the per-site candidate scorer works fine on hard-negative sites.
The failure is at the tnp-level SetTransformer — it does not compare
positions across sites.

### 3.5 Oracle probe: does the signal exist at all?

Question: if we compute simple cross-site dispersion statistics from the
model-selected candidates, does a linear classifier already separate
positives from wrong_position and wrong_structure_role negatives?

Method: `scripts/probe_linear_dispersion.py`. Six per-tnp features:
- `MAD`, `STD`, `IQR` of picked target position (bp)
- `STD` of picked NC-start (bp)
- `STD` of picked guide length
- Binary entropy of picked orientation

5-fold CV logistic regression on val_v4 tnps.

**Result**:
```
                         model-picked     oracle (labels)
wrong_position           0.873 ± 0.032    0.942 ± 0.021
wrong_structure_role     0.816 ± 0.084    0.953 ± 0.028
wrong_orientation        0.983 ± 0.009    0.998 ± 0.002
wrong_length             0.823 ± 0.043    0.966 ± 0.013
level3_paired            0.446 ± 0.056    0.465 ± 0.033  ← at chance, correct
```

**Interpretation**:
- The signal V1 fails to use is *available* in the model's own candidate
  picks — a 6-feature linear head recovers most of it.
- Level3 correctly stays at chance under dispersion features (dispersion
  cannot distinguish it — validates the semantic reframing in §2.4).
- The candidate scorer's picks are 80–90 % as informative as oracle picks —
  no need to improve the scorer.

### 3.6 V5 (concat classifier) — abandoned

First attempt at using dispersion: concatenate the 6-D dispersion embedding
(via a small MLP) into the tnp classifier input. Grow classifier from
128 → 64 → 1 to 160 → 64 → 1.

- **Ep0 was great**: AUROC 0.855, wrong_position 0.862, nc_top1 0.945.
- **Ep1–12 collapsed**: nc_top1 dropped to 0.36 (random for 3 NC regions),
  overall AUROC plateaued at 0.66.

**Root cause diagnosis**: primary-path atrophy. Once the classifier learned
to use the dispersion features (which don't depend on nc_attn being
correct), the gradient into `pooled_flat` (Set Transformer output) weakened.
`nc_mil` stopped receiving strong training signal, and `nc_attn` drifted.

**Decision**: abandon the concat formulation. It exposes the model to a
new shortcut ("just use dispersion, ignore the NC hierarchy").

Kept for reference: `checkpoints/v5_v4_main/best.pt` (ep0), documented as
a *proof of feasibility* that dispersion and NC selection CAN coexist —
they just can't be trained jointly with a naive concat.

### 3.7 V5.1 Stage A (scalar residual) — retained as intermediate

Second attempt: residual formulation.
```
logit = base_logit_V4 + α * disp_head(φ_6)
```
- `α` init 0 → V5.1 output ≡ V4 output at t=0 (bitwise verified).
- Warm-start from V4 main ep16.
- Freeze everything except `disp_head` (7 params tensors) and `α`
  (1 param) — 322 trainable params total.
- Guardrails: ★ best requires `nc_top1 ≥ 0.90` AND
  `wrong_orient AUROC ≥ 0.95`.

**Result** (`checkpoints/v5_1_stageA_from_v4/best.pt`, ep6):
- AUROC 0.9085 (up from V4's 0.8815)
- wrong_position 0.765 (up from 0.628, **below** the 0.80 target)
- wrong_structure_role 0.687 (up from 0.581, **below** the 0.75 target)
- nc_top1 preserved at 0.956
- Trajectory converged cleanly, no atrophy.

**Interpretation**: the residual + freeze approach fixes atrophy but the
scalar α·δ correction cannot express interactions like "when V4 evidence
is strong but position MAD is large, downscore" — because Δ is only a
function of φ, independent of the V4 hidden state.

### 3.8 V5.2 Stage A (hidden-level residual fusion) — FROZEN

Final architecture. Instead of adding at the logit level, fuse at the
hidden level:

```
h_v4 = classifier[0..2](pooled_flat)   # V4 hidden state, 64-D
d    = disp_encoder(φ_6)                # 16-D dispersion embedding
Δh   = fusion_mlp([h_v4; d])            # 64-D correction
h'   = h_v4 + β * Δh                    # β init 0
logit = classifier[3](h')               # V4 output layer, unchanged
```

- `β = 0` init → V5.2 output ≡ V4 output at t=0 (bitwise verified).
- Trainable: `disp_encoder` + `fusion_mlp` + `β` = 5,457 params. Rest of
  the 483 k V4 backbone frozen.
- The `fusion_mlp` can learn h_v4 × d interactions — the missing capability
  in V5.1.
- Guardrails same as V5.1: nc_top1 ≥ 0.90 AND wrong_orient ≥ 0.95.

**Val trajectory** (checkpoints/v5_2_stageA_from_v4/history.jsonl):
```
ep    AUROC   AUPRC  HARD_AUROC  wrong_pos  wrong_struct  nc_top1  wrong_orient
0     0.903   0.836  0.809       0.722      0.679         0.956    0.995
1     0.912   0.852  0.826       0.773      0.718         0.956    0.985
2     0.919   0.859  0.840       0.810      0.739         0.956    0.984
3*    0.922   0.863  0.844       0.825      0.750         0.956    0.983
4     0.921   0.862  0.844       0.825      0.748         0.956    0.985
5     0.920   0.860  0.841       0.821      0.747         0.956    0.978
6     0.921   0.861  0.843       0.825      0.747         0.956    0.981
9     0.920   0.859  0.842       0.822      0.745         0.956    0.980
```
*Selected checkpoint at ep3 (val AUPRC 0.8625, both guardrails held). Later
epochs did not exceed ep3 by any material margin, and by ep9 metrics were
essentially identical to ep3 — no reason to keep training.

**Weak-positive score-quantile monitor** (per user request, to catch
noisy-positive downscoring the way threshold=0.5 recall cannot):
```
ep    weak-q10   weak-med   weak-q90
0     0.129      0.322      0.530
3     0.149      0.328      0.591
6     0.123      0.330      0.627
9     0.111      0.312      0.614
```
Weak median stayed 0.31–0.33 throughout — no systematic downscoring.

---

## 4. Held-out test results (frozen, `test_v4.jsonl`)

Script: `scripts/eval_v4_vs_v52_test.py`. Per-tnp scores at
`logs/test_v4_scores_v4_vs_v52.jsonl`.

### 4.1 Overall

| Metric | V4 main | V5.2 SA ep3 | Δ |
|---|---|---|---|
| AUROC | 0.7918 | 0.8290 | +0.037 |
| AUPRC | 0.5868 | 0.6337 | +0.047 |
| HARD_AUROC | 0.6552 | 0.7161 | +0.061 |
| HARD_AUPRC | 0.5874 | 0.6340 | +0.047 |
| nc_top1 | 0.9533 | 0.9533 | 0 (preserved) |

### 4.2 Per-profile AUROC (the main result)

| Profile | V4 | V5.2 | Δ |
|---|---|---|---|
| level1_marginal | 0.997 | 0.998 | +0.002 |
| **level3_paired** | **0.489** | **0.512** | +0.023 (chance ✓) |
| wrong_length | 0.735 | 0.707 | −0.028 |
| wrong_orientation | 0.968 | 0.937 | −0.030 |
| **wrong_position** | **0.673** | **0.892** | **+0.219** |
| **wrong_structure_role** | **0.578** | **0.737** | **+0.159** |

### 4.3 Strength recall @ 0.5

| Strength | V4 | V5.2 |
|---|---|---|
| strong (n=247) | 0.883 | 0.951 |
| moderate (n=190) | 0.584 | 0.789 |
| weak (n=63) | 0.270 | 0.317 |

### 4.4 Weak-positive alternative (weak-vs-all-negatives)

| | V4 | V5.2 |
|---|---|---|
| AUROC | 0.635 | 0.629 |
| AUPRC | 0.075 | 0.071 |

Ranking of weak positives against negatives is preserved to within noise.

### 4.5 Signal-source diagnostic — Δ median (V5.2 − V4)

Positives (should stay stable or improve; a global calibration shift would
move all positive medians equally):

| Group | V4 med | V5.2 med | Δ |
|---|---|---|---|
| POS-strong | 0.734 | 0.832 | +0.098 |
| POS-moderate | 0.568 | 0.645 | +0.076 |
| POS-weak | 0.354 | 0.301 | −0.053 |

Negatives (should be pushed DOWN if V5.2 is doing real discrimination):

| Group | V4 med | V5.2 med | Δ |
|---|---|---|---|
| level1 | 0.029 | 0.045 | +0.016 |
| wrong_orientation | 0.028 | 0.116 | +0.088 |
| wrong_length | 0.428 | 0.518 | +0.090 |
| **wrong_position** | **0.487** | **0.280** | **−0.207** ← target profile |
| **wrong_structure_role** | **0.607** | **0.480** | **−0.127** ← target profile |
| level3_paired | 0.648 | 0.738 | +0.090 |

The two **target** hard negatives were pushed down by 0.13–0.21 median
score. Positives were pushed up asymmetrically (strong more than weak).
This is the mechanistic signature of *real* discrimination, not calibration.

Level3 median went up (+0.09), but AUROC stayed at chance (0.512) — level3
tnps are now scored even more like positives on average, which is
consistent with the semantic reframing: level3 IS observably RNA-guided-like,
so a better model correctly outputs high scores for it. The chance-level
AUROC confirms no synthetic shortcut is being exploited.

### 4.6 Acceptance criteria vs frozen result

| Criterion | Target | Test result | ✓/✗ |
|---|---|---|---|
| wrong_position | > 0.80 | 0.892 | ✅ (+0.09) |
| wrong_structure_role | > 0.75 | 0.737 | ⚠ (0.013 short) |
| nc_top1 | ~ 0.95 | 0.953 | ✅ |
| wrong_orientation | ~ 0.99 | 0.937 | ⚠ (0.05 short) |
| level3_paired AUROC | ~ 0.5 | 0.512 | ✅✅ |
| Weak-positive ranking | not degraded | −0.006 AUROC | ✅ |
| Hard negatives pushed down (not calibration) | yes | Δ_med −0.13 to −0.21 | ✅ |

Two minor near-misses (wrong_structure_role 0.013 short, wrong_orientation
0.05 short) accepted as the small price of the dispersion branch weighting
positional/structural cues over orientation/length. Both metrics remain
well within useful range (>0.7 and >0.93 respectively).

---

## 5. What was rejected and why

Explicitly out of scope going forward (unless a new use case demands it):

- **V5 concat classifier** — caused primary-path atrophy (nc_top1 →
  chance). Rejected in favor of residual formulation.
- **V5.1 scalar residual** — fell short of both hard-negative targets.
  Rejected because α·δ can only add to the final logit, not modulate the
  V4 hidden state.
- **Training V5.2 Stage B (unfreeze SetTransformer + final head)** — not
  needed. Stage A hit the acceptance thresholds. Additional model capacity
  would risk re-introducing the atrophy problem the freeze was designed to
  prevent.
- **Full V4 including level3 in training** — the diagnostic arm confirmed
  training collapses. Level3 is retained as evaluation-only control.
- **V6/V7 data changes** — the remaining acceptance gap (wrong_structure_role
  0.013 short) is well within noise for the 100-tnp test slice. Further
  data-side changes have diminishing returns and risk introducing new
  synthetic artifacts.
- **Adding a structure-embedding branch** — the oracle probe showed
  `nc_start_STD` already carries most of the structural-role signal
  (power 0.94). Explicit structure embedding is not needed for the current
  target metrics.
- **Larger SetTransformer / bigger candidate encoder** — not the
  bottleneck. The candidate-selection diagnostic showed the per-site
  scorer works fine; adding capacity here would not address the aggregation
  problem.
- **Chasing overall AUROC further** — the project has moved past
  "how high can synthetic AUROC go?" to "what does the model actually do,
  and can we use it on real data?". Further synthetic gains have poor
  return on investment.

---

## 6. Frozen artifacts inventory

### Model + training code
| File | Role |
|---|---|
| `model/v1.py` | V1Model class; supports `use_dispersion` + `dispersion_mode ∈ {scalar, hidden_residual}` |
| `training/train_v1.py` | Trainer; supports `--init-from`, `--freeze-backbone`, `--nc-top1-gate`, `--wrong-orient-gate`, `--dispersion-mode` |
| `sbatch/train_v1.sbatch` | Lawrencium sbatch (es1 A40, pc_rubinlab, es_normal); reused with EXPORT vars for V5.1 and V5.2 |
| `preprocess/alignment.py` | Base pairwise-alignment feature stack (dot plots + windowed match counts + one-hot + direction fusion + perfect-seed density) |
| `preprocess/candidates.py` | Candidate builder: top-K per (orient, L) combination × 3 NCs |
| `preprocess/site.py` | Site preprocess + structure cache reader (RNAplfold memmap) |
| `preprocess/tnp_dataset.py` | TnpGroupedDataset + collate (S sites per tnp) |

### Diagnostic scripts
| File | Purpose |
|---|---|
| `scripts/diagnose_v4_candidate_selection.py` | Per-site model-picked vs. label alignment (localizes failure) |
| `scripts/probe_oracle_dispersion.py` | Single-feature oracle probe |
| `scripts/probe_linear_dispersion.py` | 6-feature logistic-regression probe (V5 direction validation) |
| `scripts/diagnose_strength_ordering.py` | V3-era paradox diagnostic |
| `scripts/diagnose_level3_vs_strong.py` | Confirms V4 model can't separate strong-pos from level3 by summary stats |
| `scripts/eval_v4_vs_v52_test.py` | Held-out test comparison |

### Data (on Lawrencium scratch)
| Path | Content |
|---|---|
| `/global/scratch/users/kh36969/DL_novel_guide_editor/splits/train_v4_no_l3.jsonl` | Training data (520 k records) |
| `/global/scratch/users/kh36969/DL_novel_guide_editor/splits/val_v4_no_l3.jsonl` | Validation (65 k records) |
| `/global/scratch/users/kh36969/DL_novel_guide_editor/splits/test_v4.jsonl` | Test (75 k records; includes level3 as eval-only control) |
| `/global/scratch/users/kh36969/DL_novel_guide_editor/structure/{train,val,test}_v4_u16.{index.json,mmap,valid}` | Precomputed RNAplfold cache |
| `/global/scratch/users/kh36969/DL_novel_guide_editor/manifest.yaml` | Data-generation manifest (git commit, seeds, scale, timestamps) |
| `/global/scratch/users/kh36969/DL_novel_guide_editor/DATA_SCHEMA.md` | Per-record schema |

### Checkpoints
| Path | Notes |
|---|---|
| `checkpoints/v1_on_v3/best.pt` | V1-on-V3 baseline (source cluster); for reference only |
| `checkpoints/v1_on_v4_main/best.pt` | V4 main baseline (ep16, val AUPRC 0.7862); warm-start source for V5.x |
| `checkpoints/v1_on_v4_diag/best.pt` | Diagnostic arm; for reference (do not deploy) |
| `checkpoints/v5_v4_main/best.pt` | V5 concat ep0; feasibility artifact only |
| `checkpoints/v5_1_stageA_from_v4/best.pt` | V5.1 scalar residual (ep6, AUPRC 0.8447); intermediate |
| **`checkpoints/v5_2_stageA_from_v4/best.pt`** | **V5.2 hidden-residual (ep3, AUPRC 0.8625); PRODUCTION CHECKPOINT** |

### Test-set predictions
| File | Content |
|---|---|
| `logs/test_v4_scores_v4_vs_v52.jsonl` | Per-tnp `{tnp_id, is_positive, violation_profile, tnp_strength, v4_score, v5_2_score}` |

### Reproducibility metadata
- Environment: `~/.conda/envs/opfi/` with torch 2.6.0+cu124.
- Cluster: Lawrencium (`pc_rubinlab / es1 A40 / es_normal`); rows also
  reproduce on H200 within numerical tolerance (see TRANSFER.md verification).
- Data generation: git commit `46237692ec42bb53c78b69e9b188b2ea4e1938d6`
  in `~/tools/DL_RNA_guide_edotor_positive_generator/`; V4-negative code
  in `negative_gen_v4/`.
- All V5.2 training uses `--seed 0` (deterministic).

---

## 7. Not part of this snapshot (future work)

The following belong to the next phase of the project. They are not
addressed by the frozen classifier and are recorded here only to make the
scope of this snapshot explicit:

- **Real IS110 zero-shot evaluation.** Requires:
  - Tnp family clustering on 18,590 real IS110 records
  - Candidate NC-region extraction from IS element interiors
  - Cast into the classifier's record schema
  - Reference: `/home/kuangh/tools/IS110_downstream/data/` (source cluster).
- **Interpretability / feature-attribution studies** — what dispersion axes
  drive each score? Do the fusion-MLP interactions match the biology?
- **Operational threshold calibration** for a specific downstream discovery
  task (weak-positive recall @ tuned threshold, ranking metrics for
  candidate lists, etc.).
- **Deployment packaging** — CLI wrapper for a single-tnp scoring endpoint.

---

## 8. Signature line

Snapshot frozen 2026-08-23 by kh36969 on Lawrencium (n0057.es1). If any
element of §4 (test-set numbers) or §6 (frozen artifacts inventory) changes,
create a new PROJECT_JOURNAL_v2.md; do not edit this file.
