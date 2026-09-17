# Channel B — spec v1 (2026-09-02)

Channel A handles the 43% of bags where 5-of-5 site coherence at planted
position gives S=5 under Mode 2 (m ≥ 8 at L=11). Channel B's job is the
40% S=3/4 slice where per-site evidence is under-informed and Channel A
does not fire. The 17% S ≤ 2 tail is out of scope — single-molecule
per-site evidence cannot recover it.

## Three design decisions locked before writing model code

### 1. Input: raw position arrays, NOT discrete S counts

The failure the (τ, S) manual sweep hit — Pareto-optimal at S=5 τ=0 with
no grid point dominating — is a symptom of thresholding too early. Any
Channel B version that reads `S = #{site : m ≥ 8}` performs per-site
pooling BEFORE cross-site aggregation, violating the D5b invariant that
gave every V5A-3 variant a chance-level ceiling. Instead:

  input tensor shape = (batch, S=5 sites, nc_len, C) where C =
    m_max[L'=9], m_max[10], m_max[11], m_max[12]   (4 continuous m axes)
    dG_open_uL_pn, H_pair_win, cooperativity_win_pn, E_span_win + valid  (5)
    flank_argmax[L'=9..12]                          (4 — the second consistency axis)
    orient_fwd, orient_rc                           (2)

The 4 m_max channels are v2's `enumerate_position_arrays` output (byte-
identical to MatchTable m_max_by_excl[0], A7/A8 anchored). Structure
channels are the 4 that Items 3+3b showed contribute Δ AUROC +3pp.
`flank_argmax[L']` gives per-position the flank offset that achieved the
best m — cross-site conjunction over shared flank offset is a second
consistency signal orthogonal to the nc-position signal Channel A uses.

Total C = 15 channels per (site, nc position).

### 2. Aggregation: site-symmetric, applied BEFORE per-site collapse

The site axis is a permutation-equivariant set of size 5. Channel B's
model must be symmetric in site order (a Set-Transformer or per-position
MLP → attention pool over sites, applied identically per nc position).
Critically: **the cross-site attention pool operates on the raw
(m_max[L'=9..12], structure, flank_argmax) tensor per nc position — no
per-site pooling before this step.**

This is the exact opposite of the V5A-3 pattern that trained a per-site
MLP → mean/max pool over sites → classifier. That pattern collapsed
each site to a scalar (or short vector) before aggregation and lost the
signal in "1 strong + 4 weak" vs "5 medium" — two configurations that
p^n · q^(5-n) analytics show are behaviorally distinct.

Practically: at each nc position p, the model attends across the 5
sites' 15-channel feature vectors and emits a scalar log-odds per (p).
Site count is not locked at 5 — the same head runs on n_sites ∈ {1..8}
without retraining (permutation-equivariant transformer).

### 3. Training target: per-position ORDINAL label `n_planted_at_position`

Refinement over "per-position binary" (2026-09-02, from the user's spec
review): the label is not `is_planted[p] ∈ {0, 1}` but the **integer
count** of how many of the bag's `n_sites` sites have their planted
guide at nc position `p`:

  n_planted_at_position[p] = |{ site : is_planted(site) ∧
                                  guide_span_start(site) == p }|

Values under each negative_mode:
  none:      one position with count = 5 (the shared bag_nc_start), else 0
  scattered: 5 positions with count = 1 each (each site plants its own), else 0
  partial:   one position with count = n_planted ∈ {1..4}, else 0
  positive:  same as `none` (count = 5 at bag_nc_start)

Why ordinal, not binary:

- The user's ambiguity flag: partial n=3/4 is Channel B's TARGET
  slice, not an unambiguous negative. Forcing it into `y ∈ {0, 1}`
  discards exactly the count the model needs to see.
- Distinguishes "S=3 by threshold" (all 5 sites carry real guide,
  n_planted_at_position = 5, but 2 sites have m<8 by rate-table tail)
  from "S=3 by design" (only 3 sites planted, n_planted_at_position = 3).
  These have different signal structure — same input might legitimately
  score differently under each — and the ordinal label lets the model
  learn both regimes without a hard decision.
- Extends the "no early thresholding" principle from input to label.
  Whether a bag is called guided at inference is a threshold on the
  regression head's output, chosen at deployment. Not baked into
  training.

Loss: ordinal regression on `n_planted_at_position` — implementation
choice between (a) squared error on the integer, (b) K-1 binary heads
for the cumulative CDF (proportional-odds), (c) Poisson NLL. Start with
(a) as the simplest; compare on val AUROC at the S ≥ 5 threshold.

Label positives dominate loss ratio ~1 position per bag (all others = 0),
so weight positive positions by nc_len (~190) to balance.

The p^n · q^(5-n) analytic model still holds: at inference, positional
logits above threshold at count ≥ 5 give exactly the Channel A analog.

## Data recipe

Positives: the V5 50K constrained corpus (2026-09-02 regen, this batch).
Negatives: `scripts/generator_v5/bag_v2.py::build_bag(negative_mode=...)`
supports three:

  `negative_mode="none"`      — positive control
  `negative_mode="scattered"` — cross-site coherence destroyed
  `negative_mode="partial"`   — n_planted ∈ {1..4} sites planted
                                (Channel B's real target: n=3, n=4)

The partial n=3 and n=4 slices are precisely where Channel A's Mode 2 FPR
was measured at 0.039 and 0.107 (finding_hard_negatives_v5). Channel B
must beat that FPR at matched recall.

Training corpus proposal:
- 50K positives (constrained)
- 10K negatives at `negative_mode=none`
- 10K negatives at `negative_mode=scattered`
- **40K negatives at `negative_mode=partial`** — uniform `n_planted ∈ {1..4}`
  gives ~10K bags per n. The n=3 and n=4 slices at 10K each are the
  minimum bench size for Channel B to have gradient on the boundary
  the S=3/4-by-threshold positives share with n=3/4-by-design negatives.
  At 5K partial (the prior batch), each n-slice had ~1.3K bags — too few
  to reliably fit the n=3 vs n=5 boundary. 40K partial is the fix.
- Split by transposase_id, 90/10 train/val, no Tnp overlap

## Anchors that must hold before Channel B is trusted

A9. **Recall at matched FPR** (2026-09-02 fix — the earlier
    "beat Channel A's recall" was circular because Channel A's recall is
    defined under S=5 by Mode 2, which Channel B's ordinal output does
    not natively produce):

    Common scale = FPR on the `none` negatives (raw noise). Channel A
    Mode 2 sits at 0.033 on this set. Channel B must:
      recall on V5 val positives at FPR ≤ 0.033 on `none` > 0.44
      (Channel A's pooled coverage on the constrained corpus).

    Also report the same comparison at the S=3/4 slice — that's where
    the addressable market is quantified (finding_hard_negatives_v5).
    If Channel B doesn't beat Channel A THERE at matched FPR, the model
    is not solving its stated problem.

A10. Channel B on Durrant, at Channel A's operating point, must give
     numbers in the same neighborhood as A2's coverage=0.34, ppv=0.95.
     Otherwise the domain shift from V5 to real data breaks the model.

A11. Ablation: replace the site-attention pool with a symmetric mean pool
     of per-site MLP outputs (the V5A-3 pattern). Expected outcome:
     accuracy drops materially. If it doesn't, the site-attention was
     not doing what the D5b argument said it should — investigate.

## Pre-registered acceptance (2026-09-04, BEFORE any training)

These six gates are pre-registered on the v6-frozen Stage 2 corpus and
must be met on the held-out V5 val split (10% of pos50k, blocked by
Tnp) before Channel B is called Complete. Post-hoc gate movement is
banned — if a gate fails, the failure is the anchor, not a re-tune of
the gate.

| # | Gate | Metric | Threshold | Rationale |
|---|---|---|---|---|
| C1 | Tnp-blocked K-fold AUROC | mean over 10 Tnp-blocked folds on the val split (~5000 bags / ~5000 Tnps, one Tnp per bag) | ≥ 0.85 | If site-symmetric aggregation actually learns the D5b invariant, held-out-Tnp generalization stays high; ≥ 0.85 is 6 SE above the pooled-mean baseline (0.70) at n_val ≈ 5000. Note: Durrant has n=65 Tnps and used LOO; v6 val at 5000 Tnps uses Tnp-blocked K-fold instead. |
| C2 | Planted-m dose-response slope | logit-P(positive) vs planted_m (linear fit across m ∈ {6, 7, 8, 9, 10}) | slope = 1.41 ± 0.30 | Biology says P(binding | m) ∝ exp(m · ΔΔG/kT), ΔΔG≈0.7 kcal/mol/mismatch. Model must recover this slope within 2 SE (0.15 per fit) OR it's memorizing corpus artifacts, not sequence coherence. |
| C3 | Site-order shuffle | AUROC after permuting site axis independently per bag | exactly 0.500 (SE 0.007 at n_val = 5000, so \|Δ\| ≤ 0.014) | The site axis IS a permutation-invariant set; a nonzero shuffled AUROC proves the attention pool is order-dependent (leaky). Set-Transformer per §2 makes this exactly 0.500 by construction — this gate is a construction verifier. |
| C4 | Held-out n_sites drop | AUROC on val bags with n_sites ∈ {3, 8} (extremes not in train) minus AUROC on train-mode n_sites ∈ {4-7} | ≤ 5 pp | Cross-site aggregation must generalize across bag size. A large drop = model overfit to n=5 shape. |
| C5 | Trap corpus (all-decoy, no plant) | mean predicted P(positive) on Item-4 trap bags (see finding_bpp_gold_vs_competitor for construction) | ≤ 3 pp above chance (0.53) | Structure-only detectors give 0.50-0.51 on these; Channel B may exceed by up to 3 pp for genuine cross-site trap discrimination, but a larger lift = shortcut. |
| C6 | Twin structural-only ablation (FIXED LINEAR PROBE) | AUROC of a Fisher LDA / L2-regularized logistic regression on the 5-D structure percentile vector, gold positions vs twin positions | ≤ 0.55 | A12 within-bag confirms fold byte-shared pos↔twin; the residual bias per channel is small (max median 1 pp on E_span). Direct Fisher LDA prediction on the corpus gives AUROC = 0.5187 (margin +3.1 pp vs 0.55). **The probe is fixed as LINEAR** — a nonlinear probe (GBM, MLP) could exceed 0.55 by picking up (a) the dG_open positive-skew tail, (b) inter-channel interactions, (c) non-monotone patterns not captured by LDA. If it did, we could not distinguish "twin match incomplete" from "probe too strong". Pre-registration discipline requires the gate to be immovable, so the probe class is what gets fixed instead. |

**Reporting protocol.**

Emit all six numbers on the val split (not the test set — test lands
only after these six pass). Each accompanied by:
  - n (bags used for that gate)
  - SE (Bernoulli or bootstrap as appropriate)
  - the specific train-time epoch the checkpoint was drawn from
  - the operating point (threshold) where the metric was computed

Gate C2 also emits the full (planted_m, mean_logit_P) table so slope-fit
residuals can be audited.

**Backup discipline.** If a gate fails, revert to the last checkpoint
that met the same gate on a proxy at 10% corpus scale (partial40k
subset) rather than retrain-and-hope. Anchoring on a partial-scale proxy
prevents the "train more epochs and see" loop that adds risk of leakage
without addressing the failure mechanism.

## What's not in this spec

- Model hyperparameters (width, depth, attention heads) — first cut is a
  small transformer; grid on val.
- The residual 3.35% Mode 1 dispersed classifier mismatch — recorded in
  finding_constrained_sampling_verdict.md backlog. Channel B doesn't need
  it fixed; the mismatch is at a fringe cell.
- Multi-scale ±k·L structural patches (deferred to Module 4 v2 upgrade
  path, gated on Channel B first showing that patches beat means).

## Estimated cost

Model: ~2M parameters at (hidden=128, layers=3, heads=4). Training on
50K positives + 30K negatives ~ 1 day at A40. Anchors A9-A11 land within
that same window.

## Pre-training data-side checks (2026-09-02)

Two diagnostics ran before any model code; both changed what the spec
implies about Channel B's ceiling and cleanness.

### Check 1 — boundary separability in `m`

  weak-planted (positive corpus, m<8, true guide):     mean m = 6.90
  unplanted   (partial corpus, is_planted=False):       mean m = 7.66
  P(weak_planted_m > unplanted_m) = **0.415** (< 0.5 = wrong direction)

Interpretation: at the "S=3-by-threshold positive vs partial n=3
negative" boundary — the slice Channel B was scoped to attack — the m
signal points AWAY from the correct label. The negative's raw flank
noise max over 100 flank positions has mean 7.66 (matches the calibrated
per-position q ≈ 0.21 and P(max ≥ 8) ≈ 0.23 background), while the
weak-planted site has m capped at 7 by its own mismatch tail. On m
alone the boundary is unlearnable.

Structure means give +3pp AUROC over scalars (Items 3 + 3b), but that's
from a scalar-baseline of 0.83 on the S=5 easy slice — not from 0.5 on
this hard boundary. The residual signal from structure at partial
coherence is bounded.

**Consequence for the spec**: the "40% addressable market of S=3/4" claim
is optimistic. A more honest read is that Channel B's ceiling on that
slice is limited by the information available in the input — structure
+ flank_argmax cross-site conjunction may recover a chunk, but not
close to the way S=5 anchors Channel A.

An alternative scope worth naming: attack the **L≠11 and is_split
slices** where Channel A structurally misses (mm_geometry L=13-14
coverage was 0.66-0.72 pre-constraint; split coverage is architecturally
limited). Those are hard-Channel-A cases that are not information-
limited in the same way.

### Check 2 — flank-level shortcut

  Flank-only logistic regression (composition + dinuc + entropy):
    AUROC edited-vs-raw = **0.5504 ± 0.010**

Just above the 0.55 threshold — a real but modest shortcut. A Channel B
model can partially guess "did this flank get planted?" from static
composition, then bypass cross-site aggregation for that fraction.

**Fix**: for `negative_mode='partial'` non-planted sites and for
`negative_mode='none'` sites, insert a SHUFFLED-composition-matched
version of the bag's guide sequence at `bag_nc_start` (not the real
mutated guide, just its shuffle). This makes composition and edit-
existence uniform across positive and negative flanks; only the
ordering (and thus cross-site aggregatable signal) differs. Cost: one
regen of partial + none negatives (~ 1 hr).

Preferred order:
  1. Fix Check 2 first (fake-shuffled-plant regen) — cheap, un-ambiguous.
  2. Re-run Check 1 post-fix — confirm the m-based unlearnability holds
     independently of composition artifact (should be unchanged; if it
     changed, the composition was part of the boundary signal, which is
     itself a red flag).
  3. Then decide: attack S=3/4 with acknowledgment of the ceiling, or
     re-scope to L≠11 / is_split.
