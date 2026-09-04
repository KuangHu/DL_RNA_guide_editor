# Generator v6 rebuild — hand-off after pre-checks locked

Written 2026-09-03. All architectural decisions below are locked by
measurement, not by argument. Read `docs/generator_spec.md` for the
inherited contract; this doc lists the deltas.

## Decisions locked by measurement

### Coordinate scheme — **aligned (canonical), NOT absolute**

Measurement A (2026-09-02, `scripts/generator_v5/measure_A_drift.sbatch`
+ v2 variant): applied cumulative-indel drift to per-site m_max arrays,
ran Channel A serially on 10K Tnps. Results:

  rate | homology | coverage | exact
  0.000 | 100%    | 0.466    | 0.423
  0.005 | 99.5%   | 0.327    | 0.281   ← −30% coverage at 0.6 nt shift std
  0.050 | 95%     | 0.153    | 0.103   ← catastrophic

The 5th-power argument (`(1-ε_site)^5`) explains the sensitivity to
sub-nt shifts. τ tolerance cannot rescue this — τ=1 lifts FPR from
0.033 to 0.083 without recovering coverage.

End-to-end verification (2026-09-03, `scripts/verify_end_to_end.sbatch`):
applied real drift + Bio.Align.PairwiseAligner backward-mapping to
canonical coordinates. At rate=0.05:

  cov=0.446, exact=0.403, PPV_Tnp=0.964

Alignment recovers 96% of the drift-lost coverage. **Absolute coord
fails, canonical + pairwise alignment works.**

### Alignment tool — **`Bio.Align.PairwiseAligner`, global mode**

Parameters: `match=+2`, `mismatch=-1`, `open_gap=-2`, `extend_gap=-1`.
These reproduce the alignment error measurement (rates 0.005–0.10
mapped to ε_align 0–0.013). Store tool+params in the manifest.

### `nc_homology_rate` axis range — `{0.90, 0.95, 0.97, 0.99, 0.995}`

Uniform per bag. Rationale: real IS-family element copies span 0.85–
0.999 depending on element age. Restricting to ≥ 0.97 would avoid
a real region. The 0.90 cell is expected to show ~10pp coverage loss
after alignment; that degradation is a finding to report, not a
value to hide (same principle as `n_planted=4`).

Alignment error measurement (2026-09-03,
`scripts/measure_align_error.py`): survival (1-ε_total)^5 at guide
window:

  rate 0.005 → 0.996 | rate 0.05 → 0.923 | rate 0.10 → 0.831

### `same_region` for N_nc ∈ {2, 3} — **deployment gap, model handles it**

Measurement B (2026-09-02, `scripts/measure_B_sameregion.py`):
structural inspection revealed the current MatchTable uses ground-
truth `active_noncoding_index` to select ONE region per Tnp. Same-
region constraint vacuously satisfied on training data.

At inference time the ground-truth active_idx is unavailable — the
model must handle all N_nc regions. Fix in v6: **MatchTable builds
per-Tnp arrays that stack all N_nc regions with a `region_index`
channel**. The cross-site attention in Channel B learns the same-region
preference.

For Channel A, this changes the position axis shape from `nc_len` to
`sum_over_regions(nc_len_r)` per Tnp. Channel A code needs no change
if the position axis is treated opaquely (it is).

## Prior-work state — what's already in the tree

Three tiers. Do NOT collapse them: tier 2 must go through Stage 3
acceptance just like new code, tier 1 must not.

**Tier 1 — validated (do NOT re-do, do NOT re-test)**

- **`constrained_mm.py`**: renamed classes to `mode2_visible` /
  `mode2_blind` with full docstring rationale; self-test PASS.
- **`docs/candidate_layer_spec.md`**: consumer-first decision on the
  22-channel patch (deferred; feature-vector only).
- **`preprocess/candidates_v2.py`**: nc-window-first candidate layer +
  `enumerate_position_arrays` (byte-identity anchored to MatchTable
  via A7/A8).
- **`preprocess/tensor_layer_v2.py`**: 24-feature emitter, no patch
  tensor.
- **A7 + A8**: end-to-end Durrant + V5 anchors on candidates_v2. Both
  PASS pre-v6.

**Tier 2 — written, never corpus-verified (MUST go through Stage 3)**

The four flank-fix-only regens submitted 2026-09-02 were cancelled
before any completed. Consequently the following code has been merged
but no downstream corpus has ever been generated with it. Treat these
as new code for verification purposes:

- **`architecture.py`**: `flank_offset_mode` axis
  (`consistent` / `inconsistent`), `flank_jitter=2` field on
  `Architecture`.
- **`bag_v2.py`**: `_fake_plant_on_flank` helper (composition-matched
  shuffle for un-planted sites); `plant_start` parameter on
  `_plant_target_on_flank`; bag-shared `bag_flank_base` in the site
  loop; fake-plant applied for `is_planted=False` sites.

Verify under Stage 3:
  1b  consistent flank_start std ≈ 1.2, inconsistent ≈ 27
  1g  flank-only shortcut AUROC ≤ 0.52 (Check 2 fake-plant)

If either 1b or 1g fails on the v6 corpus, treat as a bug in this
merged code, not a v6 regression.

**Tier 3 — stubs already in `architecture.py`, no logic**

- `DEFAULT_NC_HOMOLOGY_RATES`, `DEFAULT_ACCESSIBILITY_TARGETS`,
  `DEFAULT_GC_TARGET_RANGE`, `DEFAULT_N_SITES_RANGE` — axis constants
  present; samplers + downstream wiring pending (Stage 1a/1c/1e/1f).

Any v6 code MUST preserve Tier 1. Tier 2 is on probation until Stage 3
confirms.

## Stage 1 code work — remaining

### 1a. Canonical nc + homology + per-site alignment map

Insert BEFORE the current per-site `sample_ncrna` calls:

```python
# In build_bag, before ncs loop:
canonical_nc = sample_ncrna(rng, diff.nc_len)
canonical_fold = fold(canonical_nc)
homology_rate = arch.nc_homology_rate    # from architecture axis
per_site_ncs: list[str] = []
per_site_align_maps: list[list[int]] = []  # canonical_pos -> site_pos, or -1
for i in range(n_sites):
    site_nc = mutate_preserving_guide(
        canonical_nc, homology_rate, rng,
        guide_span=(planted_start_on_nc, planted_start_on_nc + diff.L))
    align_map = pairwise_align_map(canonical_nc, site_nc)
    per_site_ncs.append(site_nc)
    per_site_align_maps.append(align_map)
```

`mutate_preserving_guide`: same as the mutation model in
`measure_align_error.py` but the guide window (canonical positions
`[gs, gs+L)`) is copied verbatim — no substitutions, no indels
inside that range. This ensures the planted target's canonical
coordinates remain meaningful and the planted alignment recovers m
at the expected canonical position.

`pairwise_align_map`: same call as `_pw_align_map` in
`measure_align_error.py`.

Emit to JSONL:
  `labels.canonical_nc` (per-bag)
  `labels.canonical_fold` (per-bag, dot-bracket)
  `labels.arch.nc_homology_rate` (per-bag)
  Per site under `labels`:
    `site_nc_sequence` — the actually-mutated NC bases for this site
    `site_to_canonical_map` — list of len `site_nc_len`; entry k =
      canonical pos that aligns to site pos k, or -1

Manifest additions:
  alignment_tool_version (Biopython version)
  alignment_params dict

### 1b. Flank offset consistency — DONE from earlier work

Verify the `arch.flank_offset_mode` axis + jitter is still present in
`architecture.py` after v6 rewrite of same file; smoke test as in
prior session (`flank_smoke` script). No new code required.

### 1c. `n_sites` variable + S/n_sites threshold

`difficulty.py::sample_difficulty` adds:

```python
n_sites = rng.randint(3, 8)
```

Channel A spec migrates: instead of `S=5`, use `theta = S/n_sites`
threshold. Recommended default: `theta = 1.0` (all sites must hit)
for the deployable-strict mode, matching the current S=5 semantics
at n_sites=5. Add `theta` as a spec field in
`scripts/v5a_framework/variant.py::spec_m_threshold_L11`.

`(τ, θ)` surface has to be re-swept once. The old (τ, S) result
(S=5, τ=0 Pareto-optimal) does NOT transfer — a new anchor is
needed.

`labels.n_planted_at_position` range extends from `[0, 5]` to
`[0, n_sites]`.

**Inheritance that 1c INVALIDATES — must be re-derived before use**

The following current findings/scores are baked to `n_sites=5`. Every
one of them becomes stale under 1c. Any Stage 5 or downstream planning
that quotes an inheritance number from below must first re-derive it
on the v6 corpus:

- **`p^n · q^(5−n)` hard-negative model** (`finding_hard_negatives_v5`).
  Generalises to `p^n · q^(n_sites − n)`. The likelihood-ratio `p/q≈4.1`
  is still per-site and re-usable; the exponent form is not. The
  analytic S-threshold tradeoff table is quoted at `n_sites=5` and
  must be re-tabulated.
- **Partial-corpus `n_planted` range**. Currently `U{1..4}` because
  Channel A fires at S=5; needs `U{1..n_sites−1}`. Partial regen at
  Stage 2 must sample n_planted CONDITIONAL on the bag's n_sites.
- **S distribution table (S=5 43% / S=3-4 40% / S≤2 17%)**
  from `finding_w8p_w9`. This becomes an n-sites-mixed proportion
  distribution `S/n_sites`. **The 40% "addressable market" number
  that gates Channel B's Stage 5 scope decision comes from this
  table — do NOT re-use the 40% figure to scope Channel B**; re-derive
  from the v6 corpus after 1c lands.
- **Item 3b slice definition (S ∈ {3,4})**
  from `finding_tensor_layer_deferred_patch`. Redefine as
  `S/n_sites ∈ (0.4, 0.9)` (or similar band by proportion), not
  absolute S. The +0.030 AUROC lift stays valid but the slice it was
  measured on has to be re-cut.

Every one of these gets an explicit "re-derived on v6" note in
`FROZEN.md` alongside its A8b measurement, so future readers do not
carry the `n_sites=5`-conditioned numbers forward.

### 1d. Twin negatives — new `negative_mode="twin"`

Same canonical_nc, same canonical fold, same flanks as the positive.
Each site plants a guide at a position `p_i` chosen such that:
  `accessibility_percentile(p_i)` == `accessibility_percentile(p_true)`

Where `p_true` is the true planted position for the positive.

Result: structure channels between positive and twin are byte-identical
(same fold). The ONLY difference is the guide-region identity: positive
has 5 sites at the same canonical position; twin has 5 sites at
different-but-equally-accessible positions.

Prediction: structure-only AUROC (positive vs twin) → ~0.50, provable
by construction.

### 1e. `accessibility_target_percentile` architecture axis

Uniform `[0.4, 0.95]` per bag. Replaces the current fixed `μ=0.85`
that came from only 2 anchor points (T-WT, ISEc21). Per-bag axis; the
guide placement sampler uses this as the target.

### 1f. `gc_target` architecture axis

Uniform `[0.25, 0.65]` per bag. Applied at nc sampling: canonical bases
drawn from a biased distribution `p(A)=p(T)=(1-gc)/2`,
`p(G)=p(C)=gc/2`.

Stage 3 acceptance test: hold out `gc ∈ [0.55, 0.65]` for testing;
train on `gc ∈ [0.25, 0.45]`. Verify Channel A performance is
maintained on held-out GC.

**1e × 1f INTERACTION (2026-09-03)** — GC content determines fold
density, and fold density determines the accessibility distribution's
shape. At GC=0.65 the "85th percentile" window is a much less
accessible window than the 85th percentile at GC=0.25 (denser folding,
lower absolute p_ss everywhere). The two axes SEEM uniform marginally
but their joint is NOT rectangular:

  - High GC × high accessibility percentile: few candidate windows
    with high p_ss exist → `sample_guide_placement` returns None
    frequently → those bags are DROPPED.
  - Low GC × low accessibility percentile: rarely fails but the
    absolute p_ss floor is high → guide falls in unusual position.

Bag dropping is NON-RANDOM: it correlates with the (GC, accessibility)
cell. Under 1f Stage 3 acceptance, we MUST report `placement_failure_
rate` stratified by `(gc_target, accessibility_target_percentile)`
2D. Cells with high failure rate need one of:
  (a) widening that cell's tolerance
  (b) explicit sparse-cell tag in the checker report
  (c) rejection sampling with a per-cell retry budget

Do NOT quote joint-axis Channel A numbers without checking placement
failure rate first — sparse cells give unreliable statistics.

### 1g. Check 2 fake-plant — DONE from earlier work

Verify: run the flank-only probe on the v6 corpus. Expect AUROC ≤ 0.52.

### 1h. Detector support for the second-coherence (flank) axis — REQUIRED for Stage 3

Uncovered 2026-09-03 during θ scan analysis. The generator has emitted
`arch.flank_offset_mode` since 2026-09-02 and Stage 1a wired it, but
the detector currently CANNOT read it: MatchTable keeps
`m_max = max over flank offset` per nc position and throws away
`flank_argmax`. Stage 3's 1b acceptance test (consistent std ≈ 1.2)
and — more importantly — the flank-axis validation slice at
n_sites=3 (the ONLY slice where nc-axis alone provably fails,
E[FP]/bag = 1.565, and flank-added provably succeeds, E[FP] = 0.006,
250× drop; see finding_theta_scan_1c_v6.md) both need a detector
that can require cross-site flank_argmax agreement within ±jitter.

Concrete tasks:
  MatchTable extension: alongside `m_max_by_excl[w][p]` store
  `flank_argmax_by_excl[w][p]` — the flank offset f that achieved
  the max at nc position p. Same shape / dtype (int16 or int8).
  Serialize under a new npz key `{site_idx}|{orient}|{L}|argmax{w}`.

  VariantSpec extension: add `flank_coherence: {"off","jitter"}` and
  `flank_jitter: int = 2` fields (frozen). Under "jitter", the S-count
  at nc position p only admits sites whose flank_argmax(p) sits within
  ±jitter of the bag's median flank_argmax at p. Under "off", current
  behavior (max over flank offset per site independently).

  Falsifiable prediction — 2×2 grid on the n_sites=3 slice (NOT two
  rows on a single detector; that would evaluate wrong):

                          nc-only detector          flank-coherent detector
    consistent bag       cov ~= 0.79 exact ~= 0.30  cov ~= 0.86^3 = 0.636, exact ~= cov
    inconsistent bag     cov ~= 0.79 exact ~= 0.30  cov ~= 0 (planted flank offsets
                                                    don't agree across sites)

  Numbers updated 2026-09-03 by n=1000 measurement (n_sites=3):

                     nc-only              flank-coherent
    inconsistent    0.774 / 0.299         0.004
    consistent      0.786 / 0.329         0.397 / 0.376

  LEFT column ≈ 1 − e^(−E[FP]) = 1 − e^(−1.565) = 0.79 (Poisson
  correction to the earlier "≈1.0" background-saturation estimate;
  E[FP]=1.565 is the mean, not the P(≥1) which caps at 1 with a
  Poisson tail). RIGHT top 0.40 vs analytic 0.636 is a jitter=±2
  admission-rate loss (per-site rate ≈ 0.40^(1/3) = 0.735, ~15% below
  per-site 0.86 because background occasionally captures argmax
  outside the jitter window). Both signatures preserved: RIGHT
  column shows the 100× ratio between bag types, LEFT is flat, and
  (consistent × coherent) has cov ≈ exact.

  The JUDGE has three parts, all of which must hold:
    (a) LEFT column is flat: nc-only detector cannot distinguish
        consistent from inconsistent bags. This is the negative
        control — it proves any signal in the right column comes
        from flank coherence itself, not some correlated axis.
    (b) RIGHT column has the big drop: consistent 0.636 vs
        inconsistent ~= 0. Neither single cell nor either column
        alone is the judge — the pattern is.
    (c) cov ~= exact in the (consistent, flank-coherent) cell
        confirms real detection at the planted position, not
        background saturation.

  A single-cell reading can misinterpret bg saturation as detection
  or vice versa. Report all four cells or none.

  Priority: implement 1h BEFORE Stage 3 acceptance, otherwise 1b's
  falsifiable prediction (consistent std ≈ 1.2) is measurable
  in the corpus but the acceptance criteria "detector-picks-up-the-
  axis" cannot be measured at all.

## Stage 2 regen — job sequence

  Positives 50K       sbatch/gen_v5_batch.sbatch NEG_MODE=none TAG=v6_50k
  Twin negatives 50K  sbatch/gen_v5_batch.sbatch NEG_MODE=twin  TAG=v6_twin50k
  None      10K       sbatch/gen_v5_batch.sbatch NEG_MODE=none  TAG=v6_neg10k_none  (still positive control)
  Scattered 10K       sbatch/gen_v5_batch.sbatch NEG_MODE=scattered TAG=v6_neg10k_scat
  Partial   40K       sbatch/gen_v5_batch.sbatch NEG_MODE=partial TAG=v6_neg40k_partial

Twin regen expected ~5% overhead beyond positive (shared fold cached).

**ORDER GATE (2026-09-03)**: 1h (MatchTable.flank_argmax +
VariantSpec.flank_coherence) MUST land BEFORE Stage 2 regen. 1h
changes MatchTable's stored per-shard array shape (new argmax channel);
if Stage 2 runs first, ALL 5 shard sets (50K positives + 50K twin +
10K none + 10K scattered + 40K partial) have to be rebuilt (~4 hrs
per set at 32 workers). 1h before Stage 2 keeps the shard build to
one pass. Correct order: 1f → 1g → 1h → Stage 2.

## Stage 3 stratified acceptance

Every check has a falsifiable prediction.

  1a  homology=0 sanity: alignment map is identity; coverage matches
      the current 0.4864 anchor byte-exactly. Stratified: 0.995 near
      baseline; 0.90 ≈ 10pp coverage drop.
  1b  consistent flank_start std ≈ 1.2; inconsistent ≈ 27 (was smoke-
      confirmed earlier — will hold under v6 regen too).
  1c  performance vs n_sites at fixed theta: roughly flat.
  1d  positive-vs-twin AUROC using ONLY structure channels: ≈ 0.50.
  1e  accessibility percentile marginal is uniform; performance
      stratified by percentile reported.
  1f  held-out GC test: train [0.25, 0.45], test [0.55, 0.65].
  1g  flank-only shortcut AUROC ≤ 0.52.
  Check 1 redo: weak-planted vs unplanted separability. Under flank-
      offset consistency, expect the model to distinguish these because
      the weak-planted site's `argmax_flank` matches the bag-shared
      offset while the unplanted site's is random.

## Stage 4 anchor rebuild

  A1 A2 A7 — must be unchanged (Durrant is N_nc=1 byte-identical; no
             homology, no flank consistency, no alignment invocation).

  **A8a runs in three separated sub-checks after 1a. Do NOT collapse
  them — they diagnose disjoint failure modes.**

  A8a-0 — Identity assertion. homology_rate=1.0 must produce the
           identity alignment map: for every canonical position k,
           `site_to_canonical_map[k] == k`. Assert element-wise on a
           handful of bags. Seconds; no Channel A needed. Failure =
           alignment tool wiring or coordinate convention bug. This
           check is cheap and MUST pass before A8a runs.

  A8a   — Baseline reproduction. Same axis config as A8a-0 (homology
           =1.0, n_sites=5, all other v6 axes at DEFAULT_INCONSISTENT).
           Generate 500 bags → run Channel A → must reproduce
           0.4864 / 0.9681 / 0.4356 byte-exactly against the current
           V5 anchor. A8a-0 pass but A8a fail = MatchTable canonical-
           coord write-back logic bug (map is right, downstream reads
           it wrong).

  A8a-1 — Cross-validation with the end-to-end verify implementation.
           homology_rate=0.95, n_sites=5, all other axes at
           DEFAULT_INCONSISTENT. Generate 500 bags → run Channel A →
           coverage must be 0.446 ± 0.01 (from `verify_end_to_end.py`
           at rate=0.05). This confirms the offline alignment map
           (v6 generator) is equivalent to the online alignment
           (verify script) — two independent implementations of "the
           same quantity" — a class of unchecked equivalence that
           has cost this project real time before. A8a-0 + A8a pass
           but A8a-1 fail = the offline mutation model diverges from
           the online one (mutation rates, seed handling, or
           canonical-vs-site direction), not an alignment bug.

  Run order: A8a-0 → A8a → A8a-1. All three must pass BEFORE 1c
  begins — 1c changes the S-threshold definition, so A8a's anchor
  values stop being valid and the diagnostic value is lost.

  A8b     — v6 defaults on → new baseline. Record in FROZEN.md with
             the code commit hash + git tag `v6-frozen`.
  A12 (new) — twin AUROC on structure-only ≈ 0.50 ± 0.02
  A13 (new) — held-out GC coverage delta ≤ 0.03

## Stage 5 Channel B — scoped after Stage 3

Check 1 redo drives the decision:
  - Under v6 flank consistency, weak-planted vs unplanted separable →
    Channel B attacks S=3/4 (40%) + architectural misses.
  - Still not separable → Channel B attacks only architectural misses
    (L≠11, is_split).

Channel A's `(τ, θ)` Pareto surface must be re-swept under v6 before
Channel B's A9 anchor gates on it.

## Files to touch

  scripts/generator_v5/architecture.py
    + nc_homology_rate axis + sampler
    + accessibility_target_percentile axis + sampler
    + gc_target axis + sampler
    + n_sites variable (or move to difficulty.py)
  scripts/generator_v5/difficulty.py
    + n_sites sampler (if not in architecture)
  scripts/generator_v5/bag_v2.py
    + canonical_nc + mutate_preserving_guide + pairwise_align_map calls
    + per-site alignment map storage in JSONL emission
    + twin negative_mode branch (per-site accessibility-matched p_i)
    + accessibility target routing to sample_guide_placement
    + gc_target routing to sample_ncrna
  scripts/v5a_framework/match_table.py
    + apply site→canonical alignment map when computing m_max
    + stack N_nc regions with region_index channel
  scripts/v5a_framework/variant.py
    + theta parameter (replaces or complements S)
  scripts/generator_v5/channel_a_v5.py
    + read theta from spec; region_index-aware conjunction if desired
  docs/generator_spec.md
    + record all v6 axes + fifth coherence-axis rule
  FROZEN.md
    + A8a/A8b split; A12/A13 new anchors; v6-frozen tag

## Ordering constraints

  1a must land before Stage 2 (regen needs canonical + maps in JSONL).
  1c must land before Channel A spec update (theta replaces S).
  1d must land after 1e (twin uses accessibility target).
  MatchTable canonical-coord change (in match_table.py) is atomic with
    1a — either both land or neither.

  **A8a runs IMMEDIATELY after 1a lands, not at Stage 4.**
  A8a is the identity-alignment regression check: with
  `nc_homology_rate=1.0`, the pairwise alignment map must be the
  identity, and Channel A must reproduce 0.4864 / 0.9681 / 0.4356
  byte-exactly against the current V5 anchor. If it does not, the
  alignment layer has a bug and everything from 1c onward is wasted
  work. Do NOT wait for the full Stage 4 sweep; run A8a as soon as
  1a compiles and MatchTable canonical-coord land.

  A8a must pass before A8b, and A8b before A12/A13.
