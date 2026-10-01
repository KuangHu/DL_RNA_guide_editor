# V7_SPEC (draft, 2026-09-10)

**Status:** DRAFT — user review before V1 implementation.

**Purpose.** Freeze the v7 generator + Channel B loader spec, motivated by
two findings this session:

1. **Deploy-illegality on nc.** Current Channel B reads `labels.canonical_nc`
   (a generator-derived pre-mutation sequence that 84% mismatches any raw
   `noncoding_regions[i]`); this field is absent on all non-v6r2 sources.
   Past `ChannelBDataset`-on-real-data results used the `DurrantChannelBDataset`
   adapter that substituted `noncoding_regions[active_noncoding_index]` for
   canonical_nc — a substantive input mismatch. See FROZEN retraction section
   `NO_VALIDATED_DEPLOY_PATH` / `SUBSTANTIVE_INPUT_MISMATCH`.
2. **Non-independent flank pool.** `REAL_FLANK_POOL_FAMILIES` in
   `scripts/generator_v5/difficulty.py:54` = all 5 DDE families, meaning
   any B-on-DDE eval is trained-set contaminated on the flank side.

v7 merges the deploy-legal input interface (`B0 (ii)`) with the flank-pool
independence fix into ONE retraining. v7 also elevates TSD from an inherited
uncontrolled property of real flanks to an explicit generator axis, so
"detecting the rule despite TSD" is measurable.

This spec MUST conform to `docs/CANONICAL_BAG_SPEC.md`. Every departure MUST
be labeled and justified in the conformance report §7.

---

## 0. Scope and non-goals

**In scope for v7:**
- Deploy-legal input interface (raw nc, no canonical_nc dependence)
- Synthetic random flank pool (independent of any real IS family)
- TSD as an explicit randomized axis
- Wide planted_m distribution (5-11 uniform, not narrow)
- Per-site orient (not bag-level constant)

**Out of scope for v7:**
- Multi-region nc — v7 emits ONE region per bag (nc_region_count = 1)
- Any real-data flank ingestion
- Any use of `canonical_nc` / `canonical_fold` / `site_to_canonical_map` as
  labels (these are REMOVED from `DEPLOY_INPUT_LABEL_KEYS`)
- Durrant / DDE evaluation design (V7_SPEC defines training language only;
  eval design lives in separate downstream specs)

---

## 1. Input interface (loader changes)

### 1.1 Fields the loader reads

**Renamed constants (done 2026-09-10 rev3):**
- `DEPLOY_INPUT_LABEL_KEYS` → `INPUT_TENSOR_LABEL_WHITELIST` (accurate name;
  the field is the input-tensor path's whitelist, not "everything deploy
  needs")
- New: `TARGET_ONLY_LABEL_KEYS` — the GOLD fields the target builder
  legitimately consumes (`is_planted`, `guide_span_in_active_noncoding`)

**REMOVE from `INPUT_TENSOR_LABEL_WHITELIST`** (`model/channel_b/constants.py:46`):
- `canonical_nc` — replaced by direct read of `inputs.noncoding_regions`
- `canonical_fold` — recomputed at load time via ViennaRNA (see §1.3)
- `site_to_canonical_map` — no longer needed (no canonical space)

**KEEP in `INPUT_TENSOR_LABEL_WHITELIST`:**
- `guide_length` — still deploy-known (L)
- `arch` — subset per `ARCH_ALLOWED_KEYS` still applies

**ADD:**
- direct read of `inputs.noncoding_regions[0]` as the nc string (v7 emits 1
  region per bag; loader asserts `len(regions) == 1` and fails otherwise)

**Target vs input paths are separate (existing design, now named).** The
loader has two read paths:
- `_read_input_label` (input tensor path): whitelist-checked against
  `INPUT_TENSOR_LABEL_WHITELIST`. Any GOLD leak raises immediately.
- `_build_target` (target `y` construction path): reads GOLD directly via
  `labels.get(...)`. May consume fields in `TARGET_ONLY_LABEL_KEYS` —
  target consumption of ground truth is what supervised learning is. This
  path CANNOT leak into the input tensor because target and input arrays
  are constructed separately.

`guide_span_in_active_noncoding` moved out of the input whitelist (it was
never actually consumed as input — only by `_build_target` — but its
presence in the input whitelist was misleading; verified 2026-09-10 V0.5
item 1). Target path (`data.py:108`) reads it via `labels.get(...)` — that
path is unchanged.

The runtime assert in `_read_input_label` (`data.py:60-68`) MUST reject
any load-time read of the three removed fields.

### 1.2 Multi-region policy

v7 emits `nc_region_count = 1` uniformly. Loader asserts this. **Multi-region
scoring policy per `CANONICAL_BAG_SPEC.md §3.2` is NOT needed for v7**;
it becomes needed when real-data (DDE, DDE with 2 regions) is later fed
through the same loader. That policy will be declared as part of a real-data
adapter spec, not this doc.

### 1.3 canonical_fold: recomputed at load time

v6r2 stored `canonical_fold` (ViennaRNA MFE dot-bracket) as a label. v7 does
NOT store it. Instead the loader recomputes it via ViennaRNA at load time.

**ViennaRNA version pinning:** pinned to **2.7.2** (currently installed in
`opfi` conda env at `/global/home/users/kh36969/.conda/envs/opfi/`; verified
2026-09-10 via `RNA.__version__` and `RNAfold --version`). Any environment
upgrade that ships a different ViennaRNA MUST re-run the §1.3 equality
test before v7 loader is trusted on new hardware; a version mismatch flags
the loader as unverified and blocks its use for eval-of-record work. The
loader writes `viennarna_version` into every cache-dir header alongside
`params`.

**Parameters:** temperature = 37.0°C, dangle-model = 2, no probing constraints,
no G-quadruplex, no partition-function mode (MFE only). Any deviation from
these defaults MUST be noted in the loader header.

**Comparability with v6r2:** the generator's `canonical_fold` was computed
under some (currently unrecorded) ViennaRNA settings. To make v6r2 and v7
comparable on the fold-derived channels (ch 4-7 structure, ch 8 struct_valid),
we MUST measure: pick 200 v6r2 records, recompute `canonical_fold` under the
v7 loader settings, and check byte-level equality with the stored value. If
they diverge, structure-channel numbers between v6r2 and v7 are not directly
comparable and any comparison MUST normalize per-source.

### 1.4 Removed adapter

`DurrantChannelBDataset` in `scripts/channel_b_durrant_p0.py` (lines 78-100+)
is DEPRECATED. Durrant records under the v7 loader go through the same code
path as v7 records: raw nc via `inputs.noncoding_regions[0]`, fold recomputed.
The adapter's `canonical_nc = noncoding_regions[active_noncoding_index]`
substitution (lines 126-138) — the source of two prior retractions — is
removed.

**Migration**: `channel_b_durrant_p0.py` MUST be deleted or moved to a
`legacy/` subdirectory once v7 loader lands; do not maintain both paths.

---

## 2. Generator changes

### 2.1 Flank source: synthetic random

**REPLACE** `load_flank_pool()` in `scripts/generator_v5/bag_v2.py:625-644`
(which reads `real_{fam}_sites.jsonl` for all 5 DDE families) with a
synthetic random flank generator:

```
For each site:
    flank = "".join(rng.choices("ACGT", weights=..., k=120))
    where weights reflect the bag's `gc_target` axis:
        p(A) = p(T) = (1 - gc)/2
        p(G) = p(C) = gc/2
```

Same GC-target convention as `sample_ncrna()` in `bag_v2.py:649-662`.

**PRESERVE** the old function under a new name:
```
load_flank_pool_from_is_sites()  # renamed old load_flank_pool
build_random_flank(rng, gc)      # new synthetic path
```

The old function is kept for reproducibility of v6r2 (do NOT delete). A
top-level config flag or CLI arg (`--flank-source={random,is_sites}`)
selects between them. Default for v7 = `random`.

### 2.2 `generator_metadata` per record

Every emitted site record MUST carry the three fields required by
`CANONICAL_BAG_SPEC.md §2.6`:
```
generator_metadata = {
    "data_source": f"v7_{corpus_tag}",         # e.g. "v7_pos50k"
    "build_date":  ISO-8601 datetime,
    "generator_version_or_commit": git sha of scripts/generator_v5/,
    "flank_pool_source": "synthetic_random"    # or "is_sites" for legacy runs
}
```

`flank_pool_source` is v7-specific; validator uses it to distinguish v7 bags
from any accidentally regenerated v6r2-style bag.

### 2.3 Randomization axes

v7 has **9 randomization axes**. Six inherited from v6r2, three new/modified:

| axis | v6r2 | v7 | change |
|---|---|---|---|
| 1. `n_sites`             | 3-8 uniform | 3-8 uniform | unchanged |
| 2. `nc_homology_rate`    | 0.90-0.995 | 0.90-0.995 | unchanged |
| 3. `gc_target`           | 0.30-0.70 | 0.30-0.70 | unchanged |
| 4. `nc_len`              | 70-300 | 70-300 | unchanged |
| 5. `flank_offset_mode`   | {consistent, inconsistent} | {consistent, inconsistent} | unchanged |
| 6. `orient`              | bag-level constant | **per-site sampled** (see §2.4) | changed |
| 7. `planted_m`           | narrow (per-L target table) | **uniform {5..11}** (see §2.5) | new |
| 8. `junction_motif_length` | (inherited from real flank, uncontrolled) | **{0, 4, 6, 8, 9, 10} weighted (0=50%) + bag-consistency flag** (see §2.6) | new; renamed from `tsd_length` — it is a **statistical proxy**, not TSD biology |
| 9. `flank_pool_source`   | is_sites (fixed) | synthetic_random (fixed for v7 corpus) | new |

### 2.4 axis 6 — per-site orient

v6r2 used a single bag-level `orient` ∈ {fwd, rc}, so all sites in a bag
targeted the same strand. Durrant real data shows ~40% mixed-orient bags,
per session's earlier observation.

**v7 sampling rule (rev5 simplification, no Beta / no numpy dep):**
```
p_same = rng.uniform(0.5, 1.0)              # bag-level, PROV
bag_orient = rng.choice({fwd, rc})
other = "rc" if bag_orient == "fwd" else "fwd"
site_orients = [bag_orient if rng.random() < p_same else other
                for _ in range(n_sites)]
```

Implemented as `sample_site_orients(rng, n_sites) -> (list[str], float)`
in `bag_v2.py`. `p_same` is directly interpretable as the expected
per-site rate of matching bag_orient (unlike the earlier Beta+two-layer
scheme which had `p_agree + (1-p_agree)/2` as the actual same-rate — an
indirection that invites misreading). Range starts at 0.5 (not 0)
because `p_same < 0.5` is semantically identical to flipping `bag_orient`
— no coverage loss.

Under n_sites=5, this yields **~33% "pure" bags** (all sites same orient),
in the same order of magnitude as Durrant's 60% pure but not calibrated
to it — Durrant is one family; v7 should span a range wider than any
one observation.

Record `arch.orient` as a per-site LIST of length `n_sites`, not a scalar.
Also record `arch.orient_p_same` (float) as PROV for stratification.

**Loader semantics: `arch.orient` is PROV-only, NOT a model input.** The
loader MUST NOT consult `arch.orient` to select an orientation for feature
extraction; it computes `m_max` and `flank_argmax` for BOTH orientations
independently and lets the per-position best-orient pooling in the downstream
Channel A / Channel B path pick the winner (this is exactly the `S_oc` /
"per-position best orient" logic in `layers.py:115`, `orient_constraint=True`).
`arch.orient` values are recorded so downstream analysis can stratify by
per-site orient (e.g. "does per-site mixed-orient bag score differently
than uniform-orient bag?"), but they never enter feature construction.

Rationale: `orient` is GOLD (see `CANONICAL_BAG_SPEC.md §2.4`). Deploy code
has no gold orient. Both-orient scan matches the deploy path, and matches
the `A_maj_heuristic` semantics from `CANONICAL_BAG_SPEC.md §3.3`.

### 2.5 axis 7 — planted_m wide uniform

v6r2 sampled planted_m from a per-L target table (`target_m_for_L`), which
concentrated on m=8 for L=11 and topped out at `target_m(L)` per bag.
Held-out-m gate showed this narrow, L-tied distribution makes "rule vs
level" hard to distinguish (the `{9,10,11}` holdout was empty on L=11 —
see FROZEN retrospective note dated 2026-09-10).

**v7 sampling rule (rev after Step 6b, 2026-09-11):**
```
per-site planted_m ~ U{8..min(11, L)}     # integer uniform, inclusive
```

**Range NARROWED from U{5..11} to U{8..11}** after Step 6b empirical
finding: at L=11-14 the background max m per nc position saturates near
8 (100% of planted_m=5 sites had m_at_planted ≥ 8; see below). Low
planted_m (5-7) only dilutes gold signal — `S = |{sites: m ≥ 8}|` loses
sites whose m sits at 7 by chance despite plant. The gold_S dropped to
2.40 (v6r2 was 2.98) and signal-background separation went from −0.76 to
−1.52, empirically confirming the low end is harmful, not just unreachable.

Under `sample_L` = U{11, 12, 13, 14}, `min(11, L) == 11` always, so the
sampler collapses to `U{8..11}` and code implements it as
`rng.randint(8, 11)` with `assert L >= 11`. The conditional form in the spec text is prescriptive protection
against future L-range changes that could drop below 11 — if any future
change does, the assert will fire and force a re-review here.

**Absolute-cap rationale (cap at 11, NOT at L).** `planted_m` is the
ABSOLUTE match count between the planted guide window and the flank
L-mer, NOT a fraction of L. The upper bound is fixed at 11 because
Channel A's threshold `m ≥ 8` is an absolute value; keeping `planted_m`
on the same absolute scale makes cross-L slicing (`m=8` in L=11 vs
`m=8` in L=14) semantically identical. The cost is that L=14 bags have
at least `n_mismatches = 14 - 11 = 3` (never perfect-match); this is
deliberate, not an oversight. If a future experiment needs "perfect-match
bags at every L," it's a separate axis, not this one.

**Stratification rule (mandatory for held-out-m analyses).** When splitting
by planted_m for held-out or generalization experiments, the split MUST
be reported per L (or the analysis MUST justify pooling). Under the current
L range this is a no-op (planted_m is L-independent within U{5..11}), but
the rule blocks silent breakage the moment L is extended.

**Sanity:** `sample_planted_m_uniform(rng, L)` in `bag_v2.py` is the v7
sampler; it lives alongside the legacy `sample_planted_m(rng, target_m)`
in `difficulty.py` (kept untouched for v6r2 reproducibility). Do not
merge; two names, two semantics.

Bag-level `bag_target_m` (per-site max, min, mean) recorded as PROV for
stratification but NOT input.

**Held-out splits MUST use `labels.m_at_planted`, NOT `labels.planted_m`.**
`planted_m` is the SAMPLED target used to derive `n_mismatches = L −
planted_m`; `m_at_planted` is what the generator actually MEASURED at
`nc[gold]` after planting (uses `_pos_m_max_L` = both-orient max, so random
background can add matches). The gap `m_at_planted − planted_m` is empirically
1–2 in a nontrivial fraction of bags (verified 2026-09-10 Step 4b-1 test:
legacy seed=0 site 0 had planted=9 but measured=11). Splitting by
`planted_m` would misclassify those bags. Splitting by `m_at_planted`
gives the truthful signal-level split.

Small-batch generation MUST report the `m_at_planted − planted_m`
distribution stratified by `planted_m` — if the offset is severe at low
`planted_m`, consider tightening the sampler (e.g. reject-resample bags
where the gap exceeds 2).

Rationale: with a wide flat distribution, any experiment holding out a
subset of m values leaves a substantial fraction of the training distribution
intact, so held-out-vs-random has statistical power without the "did the
model see this exact level" confound v6r2's narrow distribution created.

### 2.6 axis 8 — junction motif (statistical proxy for TSD, NOT TSD itself)

**Naming.** Renamed from `tsd_length` / `tsd_consistent` (rev1) to
`junction_motif_length` / `junction_motif_consistent`. **This is a
STATISTICAL PROXY for the effect a real TSD would have on the training
task, not the biological TSD structure.** Rationale:

- v7's flank is single-side downstream 120bp. A real TSD (target-site
  duplication) is generated by the transposition mechanism as TWO copies —
  one on each side of the insertion. v7 only sees one side and so cannot
  represent the two-copy structure.
- What v7 does model instead: "junction-proximal shared short motif across
  sites of the same bag." This captures the training-relevant statistical
  effect (sites in the same bag share a short sequence near the junction,
  creating spurious cross-site coincidence pressure) without claiming to
  model TSD biology.
- Any downstream claim about "the model does / does not use TSD" must
  therefore be worded as "the model does / does not use the junction-motif
  proxy"; a real-biology TSD claim would require v7' or a real-data
  benchmark.

**v7 sampling rule:**
```
LENGTH_WEIGHTS = {0: 0.50, 4: 0.10, 6: 0.10, 8: 0.10, 9: 0.10, 10: 0.10}
                 # 0 (no motif) = 50%; each nonzero length ~10%
junction_motif_length      = weighted rng.choice(LENGTH_WEIGHTS)
junction_motif_consistent  = rng.random() < 0.5   # True/False per bag
if junction_motif_length > 0:
    if junction_motif_consistent:
        motif_seq = random ACGT string of length junction_motif_length  # bag-shared
    for each site:
        if junction_motif_consistent:
            site_motif = motif_seq
        else:
            site_motif = random ACGT string of length junction_motif_length
        # Plant site_motif AS-IS at flank[0:junction_motif_length]
        # NO reverse-complement based on site orient (see §2.6.1)
        site.flank = site_motif + rng_bases(120 - junction_motif_length)
```

Position is fixed at `flank[0:junction_motif_length]` (junction-adjacent).
Length weighted so "no motif" is the majority (50%) — no-motif is the
default, presence is a stress test.

`arch.junction_motif_length` and `arch.junction_motif_consistent` recorded
as PROV.

**Test for "does the model use the junction-motif proxy?"** — trained model
should score `junction_motif_consistent=True` bags no higher than
`junction_motif_consistent=False` bags of matched planted_m and n_sites.
If it does score them higher, the model is exploiting the proxy as a
positivity cue, and its cross-site coherence signal is partially proxy-driven,
not m-based. Run in V4.

### 2.6.1 junction_motif × site_orient interaction — defined explicitly

`arch.orient` is a per-site list (§2.4). `site.flank` is the DNA sequence
as recorded — whichever strand the record represents; the generator does
not re-store it based on `orient`. Therefore:

**RULE**: the junction motif is planted AT `flank[0:junction_motif_length]`
in the RECORDED flank string, verbatim, for every site regardless of
`site_orient`. It is NOT reverse-complemented when `site_orient == rc`.

Rationale for this choice (vs the "biologically-motivated RC" alternative):

- It matches the recorded-sequence view the loader consumes: `flank` in →
  motif at `flank[0:len]`, no orient-dependent transform.
- It keeps "junction-motif consistency across sites" a well-defined property
  measurable directly on the flank strings, without needing orient
  metadata.
- The alternative (RC the motif on rc sites) would make a
  `consistent=True` bag with mixed orient produce "half same-motif, half
  RC-motif" flanks — an unintended interaction between two independent
  axes that muddies both.

Loader implication: `flank_argmax` (ch 9-12) at the motif region is a
function of `flank[0:len]` alone; orient does not need to be consulted at
that position.

Alternative "biologically-motivated RC" is not implemented in v7. If a
downstream experiment needs it, it will be a v7' extension with an explicit
opt-in flag.

### 2.7 Corpora composition

v7 emits 5 corpora matching v6r2 tag scheme:
- `v7_pos50k`  — all positives
- `v7_twin50k` — twins of pos50k (negative label, structural pair)
- `v7_partial40k` — partial-planting mixed positives
- `v7_ctrl10k` — control positives
- `v7_scat10k` — scattered negatives

Each corpus samples axes 1-8 independently (unlike v6r2's ctrl10k which had
`target_position_in_flank` distribution byte-identical to pos50k — noted in
`v6r2` conformance report as a non-independence quirk).

**v7 rule:** each corpus MUST use a distinct RNG seed for target-position and
planting sampling. Validator (§6 check 5) MUST verify per-corpus
`target_position_in_flank` distributions are NOT byte-identical across
corpora.

---

## 3. Deploy-time input path

At deploy, given a bag with `inputs.noncoding_regions` (a list of 1+ ncRNA
strings) and `inputs.flank` (per site):

```
1. Assert len(noncoding_regions) == 1 for v7 loader                            # v7 doesn't handle multi-region
2. nc = noncoding_regions[0]
3. nc_len = len(nc)
4. canonical_fold = ViennaRNA(nc, temp=37.0, dangles=2, ...)                   # per §1.3
5. feats = compute_features_v2(nc, guide_length=L)                             # unchanged
6. For each site: read flank from inputs.flank, orient from arch.orient[i]
7. Build 15-channel tensor as in v6r2, but nc source is raw (not canonical)
```

No GOLD field consumed. No adapter needed. Same code path for v7 training,
Durrant inference, DDE inference (when Durrant/DDE are wrapped to satisfy
the `len(regions) == 1` assert — for DDE with 2 regions, a real-data
adapter with declared multi-region policy per `CANONICAL_BAG_SPEC.md §3.2`
must be built as a separate task).

---

## 4. Conformance report (v7-specific fields)

Beyond the standard fields in `CANONICAL_BAG_SPEC.md §7`, v7's report MUST
also list:

- Per-axis realized distribution (measured, not spec-claimed).
- ViennaRNA version + parameters used for `canonical_fold`.
- byte-equality result for the "recompute v6r2 canonical_fold under v7
  settings" test in §1.3.
- Cross-corpus `target_position_in_flank` independence check.
- Sanity: no `canonical_nc` / `canonical_fold` / `site_to_canonical_map`
  keys appear anywhere in emitted labels.

---

## 5. Validator hooks

Validator (from `CANONICAL_BAG_SPEC.md §6`) runs the following v7-specific
checks in addition to the eight standard ones:

- **v-a**: assert no removed-field keys are present in any label. Fail if
  `canonical_nc` / `canonical_fold` / `site_to_canonical_map` appear.
- **v-b**: assert `len(noncoding_regions) == 1` on every record.
- **v-c**: assert `arch.orient` is a list of length `n_sites`, not a scalar.
- **v-d**: assert `flank_pool_source == "synthetic_random"` in
  `generator_metadata`.
- **v-e**: assert per-corpus target-position independence (§2.7).

Fail-loud on any.

---

## 6. What v7 does NOT solve

- Real-data adapter for DDE/Durrant (still needs canonical_fold recomputed
  under matching ViennaRNA settings; multi-region policy for DDE still needs
  declaration).
- Flank-prefix baseline on Durrant (Durrant flank[0:14] = target motif, 65/65
  confirmed — Step 2 downstream must include prefix-only or prefix-masked
  control).
- Model architecture changes (v7 uses same 15-channel architecture; only
  input source changes).
- Any change to Channel A (analytic, unaffected).
- **Bilateral TSD (direct / inverted repeat) is NOT modeled.** v7's
  junction-motif axis (§2.6) is single-side only; real TSD leaves two
  copies (one on each side of the insertion), and real DDE data records
  those two copies as TWO sites of the same insertion event (upstream +
  downstream flank, each with one copy). v7 has no such two-site-per-event
  structure. Consequence: models trained on v7 will treat "same-event
  upstream/downstream flanks share a copy of a short motif" as OOD when
  they encounter real DDE data downstream. Any downstream Channel B
  evaluation on DDE MUST expect this OOD region and either document it
  as a scope limitation or add a v7' extension that models bilateral TSD
  before drawing conclusions about "TSD generalization." (`v7'` design
  deferred; noted here for tracking.)

---

## 7. Open decisions — RESOLVED (2026-09-10 rev2)

- **ViennaRNA version pin**: **2.7.2** (opfi conda env; verified installed).
  Cache header records `viennarna_version=2.7.2`.
- **`nc_len` distribution**: **keep U[70, 300]**. Rationale: narrowing to
  Durrant-matched 177 would remove nc_len as a randomization axis, but
  nc_len is the primary driver of background match rate — losing it would
  make v7 useless for testing "does the model depend on nc length."
- **`n_sites` distribution**: **keep U[3, 8]**. Rationale: fixing at 5 for
  Durrant comparability would lose the C4 held-out-n_sites gate, the
  strongest generalization evidence line to date.
- **junction-motif position rule**: **junction only, fixed at
  `flank[0:junction_motif_length]`**. Randomizing position would
  cross-contaminate with the "flank_argmax location" statistic and prevent
  clean interpretation of that channel.
- **`junction_motif_length` distribution**: **weighted, 0 = 50%**, each of
  {4, 6, 8, 9, 10} = 10%. Rationale: "no motif" is the default; presence
  is a stress test of cross-site coherence detection in the face of a
  shared-motif confounder.

---

## 8. v7-real refactor (2026-09-13, additive to v7)

**Status:** ADDITIVE. v7-real replaces the v7 synthetic-flank +
single-region-nc path with a REAL-genomic-flank + multi-region-nc path
under a REVERSED-FLOW guide-target relationship. The v7 loader spec (§1)
is unchanged; v7-real emits records that satisfy it. §2.1, §2.6, §2.7,
§3 have companion clauses below that supersede the v7 defaults WHEN
`--v7-real` is passed to `run_generator.py`. All v7 semantics that this
section does not touch (§2.2, §2.3, §2.4, §2.5) hold as-is for v7-real.

### 8.1 Flank source: REAL bacterial genome (supersedes §2.1)

`scripts/generator_v5/real_flank_pool.RealFlankPool` loads 50 diverse
RefSeq assemblies (~164 Mb total; fetched by
`scripts/fetch_v7_genome_pool.py` to `${SCRATCH}/v7_refactor/genome_pool`).
Each site's flank is a 120bp window drawn UNIFORMLY across pooled genome
mass, weighted by per-genome length. Filters: reject if any `N` in the
window; reject if AT-fraction > 0.70 (default). Junction is at position
60 (between `flank[0:60]` and `flank[60:120]`).

### 8.2 Reversed flow (guide is a READ, not a WRITE, on the flank)

v7 wrote a synthetic guide into a synthetic flank at the planted
target. v7-real is REVERSED:

1. Sample a real 60+60 flank.
2. Sample target start position (center offset ∈ U[-40, +40] from
   junction; clipped to fit).
3. READ the target subsequence from `flank[target_start:target_start+L]`.
4. Sample a bag_guide (synthetic, GC-weighted).
5. Compute natural matches between target and bag_guide.
6. MINIMALLY EDIT the flank at the target region so `matches(target, guide)`
   equals `planted_m`. Only positions needed to hit `planted_m` are
   flipped; the rest of the 120bp flank is untouched.
7. Plant the bag_guide (or unrelated guide, per negative mode) into one
   of the synthetic multi-region nc strings.

Rationale: real flanks must stay real. Editing minimally at a small
target region keeps 106+ bp of the 120bp flank as observed bacterial
sequence.

### 8.3 Junction motif axis RETIRED (supersedes §2.6)

`_JUNCTION_MOTIF_LENGTHS = [0]`, `_JUNCTION_MOTIF_WEIGHTS = [1.0]`.
Rationale (locked 2026-09-13): junction_motif is a statistical proxy
for cross-site sequence sharing, not a TSD biology model. The v7-real
refactor introduces two independent difficulty sources (real 60+60 flanks +
multi-region nc) that dominate the distractor value the junction-motif
axis was intended to provide. Position must also migrate from
`flank[0:len]` to a junction-adjacent window at `flank[60:60+len]` under
v7-real's mid-flank junction convention; the maintenance cost of that
migration exceeds the remaining distractor value.

The original weights are preserved as `_LEGACY` comments in
`scripts/generator_v5/bag_v2.py` for reversibility.

### 8.4 Wide distributions — preserved regardless of Durrant measurement

- `target_L ~ U{9..14}`
- `planted_m ~ U{8..min(11, L)}` (unchanged from §2.5)
- `center_offset ~ U[-40, +40]` (target center around junction=60)
- `nc_len ~ U[100, 250]` per region (raised from U[80,250] on 2026-09-13:
  eliminates a ~10bp systematic lift on `scattered` mode from rejection
  resampling when `min_required = 90` didn't fit an 80-floor.)

**Do NOT tune these to Durrant's per-family L=11 / m=8-11.** Locked
2026-09-13 rationale: fitting v7 to a single family's L/m distribution
is the same class of error as the retracted 65-Tnp planted_m analog —
narrow-distribution training only tests memorization; wide-distribution
training tests generalization. Durrant measurements are RECORDED ONLY
(see `scripts/... /v7_2a_durrant_measure.py`).

#### 8.4.1 Sampling scope table (added 2026-09-23 for V8)

Wide-distribution axes above describe **between-bag** variability. Each
axis has an explicit scope:

| axis | scope | drawn |
|---|---|---|
| `target_L` (=bag_guide_L) | BAG-level | one per bag |
| `bag_guide` (identity + length) | BAG-level | one per bag |
| `planted_pos` in nc | BAG-level | one per bag |
| `center_offset` (bag target center around junction) | **BAG-level** (V8 fix) | one per bag |
| `target_start` per-site jitter | site-level | `U{−TARGET_START_JITTER..+TARGET_START_JITTER}` per site around bag center |
| `planted_m` | site-level | one per site |
| `orient` | site-level | per bag orient dist + per-site draw (§2.4) |
| `nc_len` (per region) | BAG-level (each region) | one per region |

**V8 fix (2026-09-23):** `target_start` is BAG-shared with small
per-site jitter (`TARGET_START_JITTER = 2` → per-site jitter
`U{−2..+2}`, per-bag span ≤ 4 bp). This aligns synthetic positives
with the real-biology observation that IS110/TnpB target sites
cluster near the transposon junction (Durrant WT median
`flank_argmax_std_L11 = 1.32 bp` from the 2026-09-23 reference
measurement).

**Interpretation of "wide distribution":** wide across bags. Within
one bag, sites converge on the same target-position window (mimicking
real target-site duplication biology). Pre-V8 code drew per-site
`target_start` independently from `U[−40, +40]` → 40–80 bp per-bag
span → mismatched biology.

**Applies to all `negative_mode` values** (twin/partial/scattered
inherit the same bag-shared + jitter target_start scope so they
remain distributional twins of positive on this axis).

### 8.5 Multi-region nc

Each bag emits `noncoding_regions = [nc_region_A, nc_region_B]` (list
of two synthetic strings, each with independently sampled length in
U[100, 250]). ONE region contains the planted guide (or, in negative
modes, an unrelated guide); the other is pure noise.
`active_noncoding_index ∈ {0, 1}` (GOLD) names the guide-containing
region. Loader concatenates with an `N * (MAX_L - 1)` spacer via the
`concat_with_N_spacer` policy declared in
`arch.nc_multi_region_scoring` and enforced in
`model/channel_b/data.py`. Import-time asserts guarantee `MAX_L ==
max(Ls)` and `max(Ls) < spacer_len + 2` so no search window can straddle
the spacer.

Rationale: matches Durrant real-flank v2's `[nc_region_1 (193bp),
nc_region_2 (107bp)]` from IS621 element decomposition; makes v7-real
training-time distribution match real-data deploy-time distribution.

### 8.6 Deploy-time input path (supersedes §3 for v7-real records)

The `len(noncoding_regions) == 1` assert of §3 does NOT hold for
v7-real. Loader consumes multi-region nc via the `concat_with_N_spacer`
policy. All other steps in §3 unchanged; no GOLD field is consumed
(active_noncoding_index remains a label, not an input).

### 8.7 Negative modes (v7-real specific — supersedes v7 conventions)

Set of valid modes: `{none, twin, partial, scattered}` (declared in
`VALID_V7_REAL_NEGATIVE_MODES`; CLI validates on `--negative-mode`).

- **none** (positive): all K sites share `bag_guide`; each site's flank
  is edited toward it at `planted_m` matches; `nc_planted` holds
  `bag_guide` at one position.
- **twin** (negative — breaks cross-site coherence at target region):
  each of K sites uses its OWN independent guide (list of length K,
  independently sampled). Each site's flank is edited toward its own
  guide. `nc_planted` holds ONE unrelated guide (also independently
  sampled). No site's flank matches the nc-planted guide beyond
  background.
- **partial** (mixed — some sites are un-planted): a subset of sites
  has `per_site_is_planted = False`; their flanks are NOT edited
  (raw real background); the remaining sites are planted as in `none`.
- **scattered** (bag-wide dilution): `N_SCATTERED_GUIDES = 3`
  independent guides are planted in `nc_planted` at distinct positions;
  each site is randomly assigned to ONE of them; site's flank is edited
  toward its assigned guide. Cross-site coherence at any nc position is
  ~K/3, not K.

### 8.8 max_p S saturates — NOT a valid pos-vs-twin diagnostic

On 120bp × 11-mer scanning, P(m≥8 at any position) ≈ 0.95, so
`max_p S` (max over concat nc positions of the count of sites with
m≥8 at that position) approaches ~K on any construction — positive and
twin alike. This is the same extreme-value phenomenon as the per-site
GBM 0.527 ceiling and the m_max saturation cited in W4+/W7.

Construction verification for v7-real MUST use two metrics that bypass
saturation:

- **Pairwise target-region similarity across sites (per bag).**
  Positive ≈ p_match (0.65-0.73); twin ≈ 0.25 (random-4-letter
  background). Verified 2026-09-13 (100-bag smoke, `NC_LEN_MIN=100`,
  `nc_planted_positions` metadata):
  none=0.657, twin=0.258 (dead-on background), partial=0.379,
  scattered=0.399.
- **S at gold nc position (not max_p).** Read S only at the position
  where the planted guide was actually placed
  (`generator_metadata.nc_planted_positions[0]`). Positive S/K ≈ 0.87
  (planted_m distribution shifted a small fraction below m=8 threshold);
  twin S/K ≈ 0.28 (nc guide is unrelated). Verified 2026-09-13:
  none=4.52 (S/K=0.87), twin=1.36 (S/K=0.28).

Both metrics show clear positive-vs-twin separation. The model's
site×position attention path is expected to see the same separation.

### 8.10 Conserved regions — nc-side ONLY (V8.1, 2026-09-27)

V8.0 (2026-09-23) introduced bag-level RNA conserved-region templates
(`left_conserved`, `right_conserved`; lengths `U{15..35}`, GC-weighted
random ACGT) and rewrote BOTH nc and flank so that
`nc = [random][left_cons][bag_guide][right_cons][random]` AND
`flank[ts-left_len : ts]` and `flank[ts+L : ts+L+right_len]` matched
the same templates at fraction `U[0.55, 0.75]`.

V8.1 (2026-09-27) REMOVES the flank-side rewriting. Real Durrant WT
genomic flanks are bacterial DNA and do NOT carry a synthetic 15-35bp
cons-template match beside the guide binding site. Training v8_main_v3
on flank-side cons made that synthetic pattern a *label proxy*, and
Durrant WT scored 0.24 (vs 5.16 on synth positive) despite being real
IS110 events.

**V8.1 rules:**

- **flank** is real bacterial DNA (`RealFlankPool`) with ONLY the
  target region `flank[ts : ts+L]` mutated to match `bag_guide` at
  `per_site_planted_m` matches. No cons-template rewrite anywhere else
  on the flank.
- **nc** carries the conserved-region wrap:
  `nc_planted = [random ACGT] + [left_conserved + bag_guide + right_conserved] + [random ACGT]`,
  with the plant placed inside via `plant_guide_in_nc`. This is the
  only place where the bag's cons templates appear.
- **ts bounds unchanged.** `ts_lo_bound = left_conserved_len`,
  `ts_hi_bound = FLANK_LEN - L - right_conserved_len` remain in place
  so target-start distribution is byte-identical to V8.0. Only the
  flank *content* around the target changes; positions do not.

**Negative-mode implications** — cons wrap is now the primary bit that
distinguishes nc-side variants:

| mode                   | flank rewrite? | nc plant                              |
|------------------------|----------------|---------------------------------------|
| none (positive)        | target only    | `[cons + guide + cons]` (1 block)     |
| partial                | target on planted sites | `[cons + guide + cons]`      |
| flank_scattered        | target only, per-site independent center | `[cons + guide + cons]` |
| repeat_flank           | target only, site 0's flank copied to 1..K-1 with U[0, 0.05] mut | `[cons + guide + cons]` |
| unstructured_nc_full   | target only    | `bag_guide` alone, NO cons wrap       |
| scattered              | target only, 3 independent guides | 3 independent guides, NO cons wrap |
| no_alignment           | none           | pure random, nothing planted          |
| tsd_negative           | none           | `[TSD + random_middle + TSD]`, no guide |

`unstructured_nc_full` is what teaches the model "positive requires
cons wrap around the guide in nc" — plant-position is present, guide
is present, but the cons context is missing. Distinct from
`no_alignment` (nothing planted) and from `scattered` (multiple guides
diluting cross-site coherence).

The `rewrite_left` / `rewrite_right` switches from V8.0 are deleted
(no callers).

### 8.11 V8.4 rebuild — nc = Rfam scaffold, cluster mechanism retired (2026-09-27)

Supersedes §8.10. V8.4 rebuild after two intermediate iterations (V8.2
Rfam bracket + V8.3 context clusters) that were both retracted:

**Nc construction (all 7 modes):**

`nc_planted = [random ACGT prefix pad] + [rfam_left + bag_guide + rfam_right] + [random ACGT suffix pad]`

- `rfam_left` and `rfam_right` are BYTE-IDENTICAL slices of the SAME
  contiguous window of a bacterial Rfam ncRNA:
  `rfam_seq[i : i + left_conserved_len]` and
  `rfam_seq[i + left_conserved_len + bag_guide_L :
             i + left_conserved_len + bag_guide_L + right_conserved_len]`.
  The middle `bag_guide_L` bp of the source window are DISCARDED
  (replaced by `bag_guide`). The two flanking segments therefore sit
  at their natural neighboring positions with an L-bp gap where the
  guide lands — so together they retain the "these two segments
  originally fold together" property.
- `nc_planted_base` (the random surround) has length
  `nc_planted_len = U[NC_LEN_MIN=120, NC_LEN_MAX=250]` (bag-level).
- Family-BALANCED sampling: each `_sample_rfam_window` call picks a
  Rfam family uniformly (from the 8 bacterial families), then a
  sequence within that family. This prevents the dominant family
  RF00174 (~50% of the flat pool) from monopolizing training and
  biasing the model toward one family's sequence features.

**Flank (all 7 modes):**

Flank is 120-bp real bacterial DNA from the 50-genome
`RealFlankPool`, with ONLY the target region `flank[ts : ts + L]`
mutated toward `bag_guide` at `per_site_planted_m` matches. NO
off-target rewrite — invariant `_assert_flank_scope` restored to
strict V8.1 semantics after the V8.3 flank-cluster experiment was
retracted. (`repeat_flank` remains the sole exempt mode, per its
whole-flank copy design.)

**Rfam pool (bacterial only):**

8 families, seed FASTAs downloaded once from Rfam CURRENT release
and cached under `_RFAM_CACHE_DIR`. Post length+ACGT filter
[100, 250] bp:
| family | short | ~kept |
|---|---|---|
| RF00013 | 6S RNA               | 5150 |
| RF00050 | FMN riboswitch       | 6353 |
| RF00080 | yybP-ykoY (Mn2+ SW)  | 1107 |
| RF00114 | S15 leader           | 796  |
| RF00174 | Cobalamin (B12 SW)   | 19670 |
| RF00234 | glmS ribozyme        | 1286 |
| RF00504 | Glycine riboswitch   | 3969 |
| RF01055 | MOCO_RNA             | 1406 |

**Negative modes (7 total, V8.4):**

- `none` (positive), `partial`, `scattered`, `flank_scattered`,
  `no_alignment`, `repeat_flank`, `tsd_negative`.
- `unstructured_nc_full` REMOVED. At 60-bp bracket scale, a
  dinuc-shuffled Rfam window is indistinguishable from real Rfam
  via any candidate structure channel already computed by
  `compute_features_v2` (ch4-7 max |d| = 0.048; p_ss / dG_open_u1 /
  max_bpp / pair_entropy all |d| < 0.04, KS < 0.05). Without a
  channel path to detect it, the mode was a noise negative under any
  bracket construction.

**Cluster mechanism (V8.3, both flank and nc): RETIRED.**

- Flank-side clusters: rewrote 3-bp templates at bag-drawn positions
  outside the target window. Removed in V8.4 Step 2. Restored the
  strict flank-scope invariant.
- Nc-side clusters: rewrote same templates at guide-anchored nc
  positions. Without their flank counterparts, they carried no signal
  path (m_max needs flank↔nc matching bp to elevate). Removed in
  V8.4 Step 3.

**Structural invariants (unchanged from V8.1):**

- `_assert_flank_scope` (strict V8.1): flank differs from raw pool
  flank ONLY inside `[ts, ts + L)`. Exempt: `repeat_flank`.
- `_assert_ts_span`: per-bag `target_start` span ≤ 2 × JITTER = 4.
  Exempt: `flank_scattered`.
- `validate_rng_alignment`: per-bag seed, bag-shared signature
  (`bag_guide_L`, `bag_guide`, cons lengths, nc lengths, orient list,
  `orient_p_same`) byte-identical across all 7 modes; `left_conserved`
  and `right_conserved` also byte-identical (V8.4 uses same Rfam
  window in every mode).

### 8.12 V8.4 mode-design clarifications (2026-09-27 revisions)

Four clarifications to the V8.4 negative-mode design, added after the
final code review of the 7-mode enum:

1. **`flank_scattered` is a distributional-single-variable negative.**
   Its nc plant uses the SAME code path as positive (same
   `[random pad][left_conserved + bag_guide + right_conserved][random pad]`
   block, planted via the same `plant_guide_in_nc` call, same bag-shared
   Rfam window). The ONLY intended difference is per-site `ts` scoping:
   positive uses bag-shared `center = JUNCTION_POS + bag_center_off + jitter`
   (span ≤ 4), flank_scattered uses per-site independent
   `center = JUNCTION_POS + per_site_center_off + jitter` (span up to
   ~30 bp).

   **Byte-equality caveat:** the nc_planted CONTENT is NOT strictly
   byte-identical between positive and flank_scattered at the same
   per-bag seed (~24% coincidence rate). `mutate_target_to_match`
   consumes `rng.sample(mismatch_positions, k)` where
   `k = target_m − natural_matches`; `natural_matches` depends on
   `target_seq = flank[ts : ts+L]` and `ts` differs between the two
   modes → different `k` → different rng consumption → drifted rng
   state at the subsequent `plant_guide_in_nc` call → different plant
   position → different nc_planted. Drift is small (a few draws per
   site) but the plant position is sensitive over ~150 possible slots.
   The DISTRIBUTION of nc plant properties (block width, cons lengths,
   nc lengths, plant-block position within nc) is essentially identical
   across the two modes; only the per-bag content differs.

   The "single-variable" claim is a distributional one, not a byte-
   equality one. Fixing to strict byte-equality would require
   decoupling `mutate_target_to_match`'s rng consumption from its
   input content — deferred, not required for the training goal.

2. **`scattered` has a known nc-composition asymmetry vs positive.**
   Nc plants 3 independent guides at 3 positions in `nc_planted_base`
   WITHOUT the Rfam bracket wrap, because the length budget
   (`3 × 84 = 252 bp` > `NC_LEN_MAX = 250 bp`) forces omission.
   Consequence: `scattered` differs from positive on TWO axes —
   per-site guide identity (3 candidates) AND absence of Rfam context
   around each plant. This is an accepted design limitation, not a
   bug. If tighter comparability is needed, either shrink the per-guide
   bracket budget or raise `NC_LEN_MAX`.

3. **`partial` uses a strict ≤30% match ceiling.**
   `n_planted ~ U{1, max(1, ⌊0.3·K⌋)}`. Concretely for K=3-6:
   n_planted = 1 (13-33%); for K=7-8: n_planted ∈ {1, 2} (14-29%).
   Rationale: if 60-70%+ of sites co-plant the same guide, that IS
   the target system by any reasonable definition, not a negative.
   The strict cap avoids "borderline positives happening by chance"
   which would introduce label noise.

4. **Bag-level nc is assumed to be one scaffold.** In real
   RNA-guided element data, homologous copies of a transposase within
   a single cluster may each carry their own noncoding region
   (depending on the encoding operon layout). V8.4 assumes ONE
   `bag_ncrna_id` per bag (all K sites share nc_planted + nc_noise).
   The mismatch impact on model generalization is NOT MEASURED and
   is a known simplification.

### 8.13 K ≥ 3 scope declaration (V8.4, 2026-09-28)

**Channel B REQUIRES bags with K ≥ 3 sites.** Single-copy elements
(K = 1) are NOT in scope for Channel B and must be routed to Channel A.

**Rationale:**
- Project rationale from inception: single-site signal is too weak,
  cross-site aggregation is what gives Channel B its edge over
  Channel A.
- `flank_dev_L*` channels (6 of the 20 input channels) are computed
  as per-site deviation from bag-median flank_argmax. For K = 1, a
  single site's argmax is trivially the median → deviation = 0 →
  those 6 channels carry no information. Empirically (job 26517666),
  zeroing flank_dev collapses v84_none scores from 4.98 to 0.005 —
  meaning flank_dev is ~100% of the score's dynamic range.
- `flank_bg_identity` (ch19) is defined as mean pairwise Hamming
  similarity across a bag's flanks. For K < 2, no pairs exist;
  data.py hardcodes 0.0 as a neutral default. That default is 0.25
  below the training-positive distribution (~0.26), pushing K = 1
  bags out of distribution on that channel too.
- Result on real Durrant WT (job 26518880):
  * K1: v84 p50 = −0.004 (correct — no signal to work with)
  * K3: v84 p50 = 2.31 (borderline)
  * K5: v84 p50 = 4.35 (near in-distribution)
  * K8: v84 p50 = 7.54 (strong positive)
- v8_main_v3 was monotone-DECREASING (3.5 → 0.24) — that model's K=1
  score of 3.5 was an uncalibrated response to no-signal defaults,
  NOT a real single-site detection. v84 correctly floors on K=1.

**Deployment path:** two-channel system.
- K = 1 (single-copy elements) → Channel A (closed-form m ≥ threshold
  test, ~96% PPV per [[channel-a-documentation]]).
- K ≥ 3 → Channel B (this model). K = 2 acceptable at reduced
  confidence; report explicitly.

**Bags with K < 3 must be filtered upstream** or routed to Channel A.
Do not attempt to interpret Channel B scores on K = 1 bags as either
positive or negative — the model has no valid input to work with.

### 8.9 CLI + generator metadata

`python -m scripts.generator_v5.run_generator --v7-real --negative-mode
{none|twin|partial|scattered} ...`

- Mutually exclusive with `--v7`.
- Skips rate table load and is_sites pool load; loads
  `RealFlankPool.load_default()` (per-worker, via `_worker_init(v7_real_mode=True)`).
- `generator_metadata` per record includes `data_source="v7_real"`,
  `flank_pool_source="50-bacterial-genome pool (NCBI RefSeq)"`,
  `nc_multi_region_scoring="concat_with_N_spacer"`, `reversed_flow=True`,
  and `nc_planted_positions=[...]` (bag-level; positions of ALL guides
  planted in `nc_planted`; enables construction-verification diagnostics
  even for twin where `labels.planted_start` is None).
- v6r2-calibrated acceptance tests (test1a/1b/1c/2a/2b) SKIPPED — those
  thresholds are v6r2-specific. v7-real acceptance is via the
  construction-verification path (§8.8) plus the standard
  CANONICAL_BAG_SPEC.md §6 checks.

---

## CHANGELOG

- 2026-09-27: **V8.4 — Rfam bacterial ncRNA nc scaffold; V8.3 cluster
  mechanism (flank + nc) retired; `unstructured_nc_full` retired** (§8.11).
  Nc `[random pad][rfam_left + guide + rfam_right][random pad]` with
  left+right from ONE contiguous window of a family-BALANCED-sampled
  Rfam sequence (8 bacterial families; RF00174 no longer 50%-dominant
  by construction). Flank restored to strict "target-only rewrite"
  under the strict V8.1 flank-scope invariant. 7 negative modes total.
  Superseded intermediate V8.2 (Rfam bracket, no shuffle control) and
  V8.3 (position-specific context clusters, flank + nc, which turned
  out to be dominated by mono/di composition confounds under the
  dinuc-shuffle control gate — see FROZEN V8.4 entry).
- 2026-09-27: **V8.1 — flank-side conserved-region rewriting REMOVED**
  (§8.10 added). Flank-side cons was a label proxy — Durrant WT
  genomic flanks do not carry synthetic 15-35bp cons templates, so
  v8_main_v3 suppressed Durrant to 0.24 vs 5.16 on synth positive.
  Cons regions now live on the nc side only. `unstructured_nc_full`
  updated to plant `bag_guide` alone in nc (no cons wrap), so it
  remains distinct from positive after the flank change.
  `rewrite_left` / `rewrite_right` switches deleted. `ts` bounds
  intentionally kept, so target-start distribution is unchanged.
  Requires cache rebuild + fresh training (new source hash → new
  cache dir; SCHEMA_KEY unchanged since channel list didn't move).
- 2026-09-13: §8 v7-real refactor block added. Real 60+60 genomic
  flanks (RealFlankPool, 50-genome ~164 Mb pool), REVERSED FLOW
  (guide READ from flank, then flank minimally edited toward it),
  multi-region nc with concat_with_N_spacer, junction motif retired,
  NC_LEN_MIN raised to 100 (eliminates scattered's ~10bp lift). Locked
  wide distributions: target_L U{9..14}, planted_m U{8..min(11,L)},
  center_offset U[-40,+40], nc_len U[100,250]. Negative modes:
  {none, twin, partial, scattered}. New CLI flag `--v7-real`,
  mutually exclusive with `--v7`. Construction verified via pairwise
  target similarity + S_at_gold (§8.8) since max_p S saturates and
  cannot separate pos from twin.
- 2026-09-10 rev5: §2.4 sampler simplified. Replaced Beta(2,2) two-layer
  scheme with direct `p_same ~ U(0.5, 1.0)` one-layer sampler. Reasons:
  no numpy/scipy dependency, `p_same` is directly interpretable as the
  same-as-bag-orient rate (Beta version had `p_agree + (1-p_agree)/2` as
  actual rate — indirection that invites misreading), and `p_same < 0.5`
  is redundant with `bag_orient` flip so no coverage loss. Implemented as
  `sample_site_orients(rng, n_sites) -> (list[str], float)` in `bag_v2.py`.
  Validated 5-item test: length/charset/correlation/pure-bag rate matches
  analytic 33.3%/determinism.
- 2026-09-10 rev4: §2.5 rewritten. `planted_m` sampler:
  `U{5..min(11, L)}` (prescriptive form); code implements `rng.randint(5, 11)`
  + `assert L >= 11` (current L range makes the two equivalent; assert is
  future-proofing). Explicit rationale that cap is at 11 not L
  (planted_m = absolute match count, matches Channel A's absolute-8
  threshold semantic). Stratify-by-L rule mandated for held-out-m
  analyses. `sample_planted_m_uniform(rng, L)` added to `bag_v2.py`;
  legacy `sample_planted_m` in `difficulty.py` untouched (v6r2 repro).
  Passed 4-item test: distribution flat, L=14 still returns U{5..11},
  L=10 fires assert, determinism holds.
- 2026-09-10 rev3: user review pass 3. §1.1 constants renamed:
  `DEPLOY_INPUT_LABEL_KEYS` → `INPUT_TENSOR_LABEL_WHITELIST`; added
  `TARGET_ONLY_LABEL_KEYS = {is_planted, guide_span_in_active_noncoding}`.
  Moved `guide_span_in_active_noncoding` OUT of input whitelist — verified
  V0.5 item 1 that it was never read by the input path, only by
  `_build_target`. Target-vs-input separation now explicit in spec + code
  + tests. §2.4 added "loader does not consult arch.orient" clause — both
  orients scanned, `arch.orient` PROV-only. §1.3 v6r2 fold equivalence
  test RUN: 200/200 records byte-equal under ViennaRNA 2.7.2 +
  temp=37/dangles=2 — structure channels v6r2↔v7 comparable, V4 gate can
  use v6r2 old values as reference. Code changes applied: `constants.py`,
  `data.py`, `tests/test_whitelist.py`; all 8 whitelist tests pass.
- 2026-09-10 rev2: user review pass. §2.6 axis renamed `tsd_*` →
  `junction_motif_*` (explicit statistical proxy, not TSD biology); length
  weighted (0=50%); added §2.6.1 defining `junction_motif × site_orient`
  interaction as "planted verbatim in recorded flank, no RC transform"
  with rationale vs the RC alternative. §6 added a bilateral-TSD-not-modeled
  clause tagging real DDE two-site-per-event structure as OOD for v7. §1.3
  ViennaRNA pinned to 2.7.2 (verified installed). §7 open decisions resolved.
- 2026-09-10 rev1: initial draft. Motivated by (a) A-2 discovery that Channel
  B loader reads `labels.canonical_nc` which is absent on all non-v6r2 sources
  and semantically differs (84% mismatch with any raw region), and (b) A-1
  discovery that `REAL_FLANK_POOL_FAMILIES` reuses all 5 DDE families as v6r2
  training flank pool. v7 fixes both in a single retraining pass while
  elevating TSD to an explicit generator axis.
