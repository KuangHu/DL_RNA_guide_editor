# CANONICAL_BAG_SPEC (draft, 2026-09-10)

**Status:** DRAFT — user review before committing.

**Purpose.** Every data source that enters the Channel A / Channel B pipeline
must produce records conforming to this spec, and produce a **conformance
report** stating any allowed departures. Whenever a conclusion cites a number
from any source, the record of that conclusion must attach the source's
conformance report (or a pointer to it) as scope.

This spec is the ONE place where field semantics, coordinate origins, and
biological-unit definitions live. All other docs (`channel_a.md`,
`channel_b_spec.md`, `generator_spec.md`) MUST defer to this doc on the
following questions and MUST NOT re-define them locally.

**When to update this spec:** any time a data source is added, a field's
meaning changes, or a validator check is added or loosened. Update the
`CHANGELOG` at the bottom with a one-line entry.

---

## 0. Historical motivation (why this exists)

Five bug classes from this project trace to two data sources disagreeing on a
schema convention that was never written down. Any one of them would have
been caught by a validator on load:

1. **Durrant `flank_argmax_by_excl = None`** → `ChannelBDataset` silently
   zero-filled channels 9–12. Four rounds of P0 experiments analyzed OOD input.
2. **Durrant `guide_start_in_nc = 49` for all 65 Tnps** → localization work
   was below the trivial "always predict 49" baseline; only caught after 8
   miss cases happened to share the value.
3. **`ChannelBDataset` per-site orient vs bag-level orient** — loader assumed
   one, Durrant provided the other.
4. **Per-site `m_max` misread as A's `max_p S`** — inference from mechanism
   rather than reading `_aggregated_S` in `layers.py:115`.
5. **DDE n≥4 Tnps use `recs[0].noncoding_regions` for all sites** — different
   insertion events on the same Tnp have different nc regions; the analysis
   script silently used the first event's nc for every site.

Rule (from `feedback_three_errors_2026_09_07.md`): **read the spec or measure
before citing existing system behavior.** This document is the "spec" side of
that rule.

---

## 1. Biological units and the bag definition

A **bag** is the smallest unit that Channel A / Channel B scores as a single
positive/negative decision.

**A bag holds:**
1. A single **ncRNA context** — one nc sequence (possibly a set of nc regions,
   see §3.2) that ALL sites in the bag share.
2. One or more **sites** — each an independent flank observation associated
   with the same ncRNA context.
3. A single **bag label** (`is_positive: bool`) — the training target.

**A bag does NOT necessarily correspond to one Tnp.** A Tnp (transposase
cluster) may contribute multiple bags if its insertion events have different
nc contexts. Conversely one bag may be the entire Tnp when all sites truly
share nc (IS110-Durrant).

**Required per-bag identity:** every bag has a unique `bag_id`. Two bags with
the same `bag_id` are the same bag. Bags across different **corpora** MAY
share `bag_id` strings; downstream code MUST key by `(corpus, bag_id)`, not
`bag_id` alone. (This bit the Channel B eval on 2026-09-09.)

Each bag also carries a `transposase_id` (the phylogenetic cluster the Tnp
protein belongs to). Multiple bags may share a `transposase_id`; cluster-
level bootstrap MUST resample by `transposase_id`, not `bag_id`, when the
statistic is "how does the method perform on new Tnp clusters."

**Bag-level nc invariant (must-hold):** all sites in a bag share IDENTICAL
`noncoding_regions`. If sites disagree, they belong to DIFFERENT bags — split
before it enters the pipeline. The validator MUST enforce this.

**Cross-bag nc sharing:** MULTIPLE bags MAY share the same nc sequence (e.g.
Durrant's 65 Tnps use a shared bridge-RNA scaffold; this is the mechanism
behind the `guide_start_in_nc = 49` degeneracy). This is legitimate but MUST
be visible. Every site record MUST carry `nc_sequence_hash` = a stable hash
(e.g. SHA-1 of the JSON-serialized `noncoding_regions` list). The validator
(§6 check 8) reports the number of distinct `nc_sequence_hash` values vs the
number of bags; a large `n_bags / n_unique_nc_hashes` ratio flags a scaffold-
sharing corpus and its downstream localization statistics MUST be interpreted
under that scope.

---

## 2. Site required fields (canonical schema)

Every site record MUST carry the following. Fields marked **DEPLOY** are
available at inference time on unseen data (may be model input). Fields marked
**GOLD** are available only when the record is labeled; models MUST NOT
consume them as input. Fields marked **PROV** are provenance metadata (never
model input; used for validators, stratification, and error attribution).

### 2.1 Identity

| field | type | class | notes |
|---|---|---|---|
| `site_id`         | str | PROV   | unique per site; MUST prefix its bag |
| `bag_id`          | str | PROV   | see §1 |
| `transposase_id`  | str | PROV   | Tnp cluster id; may be shared across bags |
| `ncrna_id`        | str | PROV   | ncRNA context id; MUST equal across sites of one bag |
| `corpus`          | str | PROV   | source tag; needed to disambiguate `bag_id` collisions |

### 2.2 Flank (§3.1 for construction rules)

| field | type | class | notes |
|---|---|---|---|
| `inputs.flank`    | str  | DEPLOY | ACGT/N sequence; length + side per §3.1 |
| `flank_len`       | int  | PROV   | must equal `len(inputs.flank)` |
| `flank_side`      | enum | PROV   | one of `{upstream, downstream, joined, other}` |
| `flank_source`    | enum | PROV   | how it was carved (§3.1); e.g. `real_genomic`, `synthetic_designed`, `durrant_designed` |

### 2.3 Non-coding context (§3.2)

| field | type | class | notes |
|---|---|---|---|
| `inputs.noncoding_regions` | list[str] | DEPLOY | ≥1 region, ACGT/N |
| `nc_region_count`    | int   | PROV   | == len(noncoding_regions) |
| `nc_total_len`       | int   | PROV   | Σ len over regions (excl. spacer) |
| `active_noncoding_index` | int | GOLD | which region contains the gold guide target; DEPLOY MUST NOT read (see §5) |
| `nc_padding_scheme`  | enum  | PROV   | `{none, right_padded_to_K, ncpad240, ncpad350, ...}` |
| `nc_padding_offset`  | int   | PROV   | number of padding bases inserted before position 0 |
| `nc_sequence_hash`   | str   | PROV   | stable hash (e.g. SHA-1 hex) of the JSON-serialized `noncoding_regions` list; used for §6 check 8 (scaffold-sharing detection) |
| `nc_multi_region_scoring` | enum | PROV | REQUIRED when `nc_region_count > 1`. See §3.2. |

### 2.4 Orient (§3.3)

| field | type | class | notes |
|---|---|---|---|
| `orient_granularity` | enum | PROV | `{per_site, per_bag}` — the semantic under which this record's `orient` field must be read |
| `orient`             | enum | GOLD | `{fwd, rc, unknown}`; may be per-site or per-bag per `orient_granularity`; MUST NOT be a model input (see §5) |
| `orient_source`      | enum | PROV | `{observed_gold, generator_planted, unknown, inferred}` |

### 2.5 Labels (bag-level and site-level)

| field | type | class | notes |
|---|---|---|---|
| `labels.is_positive`    | bool  | GOLD (bag-level) | classification target |
| `labels.is_planted`     | bool  | GOLD (per-site)  | whether this site carries a gold planted match |
| `labels.guide_dna`      | str \| null | GOLD | gold guide DNA (if any); null on real neg |
| `labels.guide_length`   | int \| null | GOLD |
| `labels.planted_start`  | int \| null | GOLD | gold guide start position in nc, per §4 origin |
| `labels.planted_m`      | int \| null | GOLD | planted match count (positive corpora only) |
| `labels.m_at_planted`   | int \| null | GOLD | mirror of `arch.target_m_at_planted` for legacy |
| `labels.epsilon_align`  | float \| null | GOLD | generator-computed alignment noise; MUST NOT be a model input (see [[finding-train-deploy-gap-fields]]) |

### 2.6 Generator metadata (source of truth for provenance)

MUST include AT MINIMUM three fields (validator hard-checks these):

| field | type | notes |
|---|---|---|
| `data_source`               | str | stable string identifying the source (e.g. `"v6r2_pos50k"`, `"durrant_cognate"`, `"real_IS10-R_collection"`) |
| `build_date`                | str | ISO-8601 date (or datetime) when the file was produced |
| `generator_version_or_commit` | str | git commit sha or explicit version tag of the code that produced this record |

Rationale: the `flank_argmax`-missing shard bug (§0 item 1) happened because
`build_positive` walked a different code path than expected; a
`generator_version_or_commit` field on every record would have surfaced the
inconsistency at load time.

Any additional fields are free-form and MAY carry generator/collection
parameters that affect distribution of DEPLOY fields (family, organism,
insertion_start, flank_side, etc.).

---

## 3. Construction rules per axis

### 3.1 Flank construction

A `flank` is the DNA immediately surrounding an insertion site. Each source
declares its `flank_source` and MUST document:

- **How many sides** the flank includes (`upstream`, `downstream`, or
  `joined` = concatenation of both sides with an explicit spacer).
- **Length per side** and total length.
- **Orientation** of the sequence as stored (5'→3' of which strand).

A `joined` flank of the form "upstream ⊕ downstream" MUST specify the spacer
(empty or `N`-run of declared length). Sites in a bag with `joined` flanks
MUST NOT be split back into two `upstream`/`downstream` records — the model
sees the joined string.

**No source may silently mix `flank_side` values within a bag** unless the
bag's declared semantic is "each side is an independent site of the same
insertion event" — in which case those must be TWO bags, one per event, per
§1's nc-invariant rule. (This is the DDE case: each insertion event = one
bag with 2 sites (up + down), both scanning the same nc.)

### 3.2 Non-coding context construction

`noncoding_regions` is an ordered list of one or more sequences. The
convention across sources so far:

- **v6r2 synthetic**: single nc region, variable length (~70–300 bp).
- **IS110-Durrant**: single nc region, length 177 bp (padded to fixed width).
- **DDE real**: TWO regions (`nc_lens = [left, right]`), where `left` and
  `right` are the genomic non-coding sequences flanking the insertion, both
  passed as separate regions.

Because `active_noncoding_index` is GOLD (§2.3), DEPLOY code MUST score
across ALL regions and pool by position.

**Multi-region scoring policy (must be declared, no default).** When
`nc_region_count > 1`, the source MUST set `nc_multi_region_scoring` on every
site record to one of:

| value | semantic |
|---|---|
| `per_region_max`      | scan each region independently, return max score per region; positions are region-local (`nc_raw` with region index) |
| `concat_with_N_spacer` | concatenate regions with a spacer of length ≥ L-1 N bases (which cannot match anything under the aligner's N-handling), then scan as one string |
| `longest_only`        | scan only the longest region, discard others (state the ties rule) |
| `custom:<name>`       | any other scheme; MUST attach a spec-linked description in the conformance report |

A default value is BANNED: any source with `nc_region_count > 1` that lacks
`nc_multi_region_scoring` FAILS the validator and CANNOT enter the pipeline.

Rationale: DDE `nc_lens = [350, 31]` — the 31 bp region admits only 21 L=11
windows. Silent concatenation across the boundary produces synthetic
match-space that spans a nonsense junction; silent "use region 0 only" hides
the 31 bp region's contribution entirely. Both are common failure modes and
both are indistinguishable from correct behavior in logs unless the policy is
declared upfront.

This axis is also the prerequisite for the currently-unknown
`active_noncoding_index` rule on DDE (see §9); the validator MUST verify
that the declared `nc_multi_region_scoring` matches the loader's actual
behavior.

**Declared-vs-actual test (validator, concrete):** for each source with
`nc_region_count > 1`, the validator MUST run this active test:

1. Take a sampled bag from the source.
2. Replace ALL positions in region 0 with a base that cannot match anything
   at ≥ 8 hits (e.g. `"N" * len(region_0)`), leaving region 1 unchanged.
3. Plant an L-mer match into region 1 by synthesizing a flank that equals
   the reverse of region 1's first L bases (so a genuine ≥ L hit exists
   only in region 1).
4. Score the bag through the loader's actual scoring path.
5. Assert the observed score-per-region matches the declared policy:
   - `per_region_max` → region 1 score is non-zero (the planted match)
   - `longest_only` → if region 0 was longer, score is zero (region 1 ignored)
   - `concat_with_N_spacer` → the region 1 match still registers
   - `custom:*` → conformance report describes the expected outcome

Any mismatch = source FAILS validation, regardless of what its declaration
says. Symmetrically, swap regions and repeat to catch policies that
implicitly favor region 0.

**Padding**: any padding bases (e.g. `ncpad240` inserts up to 240 bp of pad
on the right) MUST be recorded in `nc_padding_scheme` + `nc_padding_offset`,
and MUST be excluded from position-based statistics (unless explicitly
opting in with rationale). Coordinate systems (§4) MUST state whether they
are pre- or post-padding.

**Per-site nc variants and the "first-site nc = bag nc" rule (2026-09-11
addition).** In some single-region sources — v7 in particular, when
`arch.nc_homology_rate < 1.0` — each site record may carry its own
mutated variant of the bag's underlying nc in `inputs.noncoding_regions[0]`.
These variants are legitimate site-level data (they reflect real
sequence divergence between homologous non-coding contexts) but the
bag-level pipeline needs a single canonical string to key its per-position
tensors against. The convention:

- **The bag's canonical nc = the FIRST site record's `noncoding_regions[0]`**
  in JSONL order. The loader (`model/channel_b/data.py`) reads it as
  `first["inputs"]["noncoding_regions"][0]` unconditionally, and both the
  shard builder (`scripts/build_v7_shard.py`) and every downstream
  per-position quantity are keyed against THIS nc for every site of the
  bag.
- **Per-site variant nc strings are ACCEPTED, not rejected.** The shard
  builder counts them (`per-site nc variants under hom<1.0=<N>`) but does
  not filter them out. Match arrays for those sites are computed against
  the canonical (first-site) nc — the mutations show up as reduced match
  scores at the mutated positions, which is the correct deploy-legal
  behavior (at deploy, the model sees one candidate ncRNA and scores
  every flank against it, whichever real nc the flank is genomically
  paired with).
- **Rationale.** At deploy, the classifier is handed exactly one nc
  candidate + one set of flanks; it has no oracle for "the truer nc" per
  site. The training pipeline MUST make the same choice. Using
  `noncoding_regions[0]` (a specific, deploy-observable, JSONL-order
  choice) is more principled than any "consensus" or "majority-vote"
  string, which would be a train-only construct that has no deploy
  counterpart.
- **What this replaces.** The old `labels.canonical_nc` field was the
  generator's pre-mutation reconstruction, which had 84% string mismatch
  against every actual raw region and was deploy-illegal. The 2026-09-10
  loader revision dropped it. This rule pins the successor semantics
  explicitly so future adapters (and any Channel A/B re-implementations)
  cannot silently re-introduce a train-only "canonical" nc.
- **Fail-loud requirement.** Any tool that materializes a per-bag nc
  (shard builders, in-memory MatchTable shims, per-bag caches) MUST NOT
  silently drop per-site variant nc records — the class of bug that
  produces (a) `KeyError: (site_idx, orient, L)` at training time or
  (b) missing sites in the tensor. `scripts/build_v7_shard.py` raises
  `RuntimeError` if the collected per-bag site count doesn't match the
  record's `arch.n_sites`; other builders should follow the same
  contract.

### 3.3 Orient handling

Two granularities:

- **per_bag**: one orient value applies to all sites in the bag. This is the
  synthetic v6r2 convention. Loaders assuming this convention will misread
  per-site sources.
- **per_site**: each site declares its own orient. This is the IS110-Durrant
  convention. Loaders assuming per-bag will silently drop site-level orient
  variance.

Every source MUST declare `orient_granularity` at the SOURCE level (in its
conformance report, §7), and every site record MAY carry `orient` accordingly.

`orient` is GOLD — it is the strand on which the planted (or observed) target
lies. Models MUST NOT consume `orient` as input.

**Two Channel A variants exist with DIFFERENT deploy legality:**

- **`S_oc` (per-position best orient)** — implemented as
  `S_pooled = S_stack.max(axis=0)` in `layers.py:115` under
  `orient_constraint=True`. For each nc position, pick whichever orient gives
  more site hits at that position. This variant **does NOT consume `orient`
  as an input** — it is a max over both orients per position. It is
  **DEPLOY-LEGAL** and MAY be used on unseen data.
- **`A_maj` (bag-level majority orient)** — picks a single orient per bag
  (either the true orient, if known, or a heuristic argmax over aggregate
  hit counts). When the orient choice uses the **gold** orient, this variant
  is **NOT DEPLOY-LEGAL** — using it as a real-data benchmark misuses
  ground-truth information. When it uses a heuristic (e.g. argmax of total
  hit count over orients), the heuristic MUST be declared in the conformance
  report and the variant MUST be tagged `A_maj_heuristic` to distinguish it
  from the gold-consuming form.

The FROZEN entry citing an A_maj number MUST state which variant was used.
Deploy-legal comparisons (e.g. cross-family AUROC on unseen Tnps) MUST use
`S_oc` OR `A_maj_heuristic`, never `A_maj_gold`.

---

## 4. Coordinate systems (origins, explicit)

Every position-valued field MUST attach to a named coordinate system. The
systems recognized by this spec:

| system | origin | scale | notes |
|---|---|---|---|
| **nc_raw**         | position 0 of the raw nc string as stored in `inputs.noncoding_regions[k]` for region `k` | 1 bp | per-region — must state the region index `k` |
| **nc_pad_pre**     | position 0 of the padded nc (padding included) | 1 bp | ambiguous unless padding scheme named |
| **nc_pad_post**    | position 0 of the padded nc, after right-padding | 1 bp | for `ncpad240`/`ncpad350` sources; MUST also state the padding scheme |
| **flank_local**    | position 0 of the flank sequence | 1 bp | side-specific; `upstream` and `downstream` are separate spaces |
| **insertion_genomic** | genome coordinate of `insertion_start` (from `generator_metadata`) | 1 bp | real-genomic sources only |

Any field storing a position (e.g. `planted_start`, `guide_start_in_nc`,
`gold_nc`, `active_noncoding_index`, `flank_argmax`) MUST have its coordinate
system stated in the spec entry for that field (§2.3, §2.5). Sources that
store positions in an ambiguous coordinate system MUST declare which system
they use in their conformance report.

**The rule (from `feedback_three_errors_2026_09_07.md` #2):** before
comparing two positions, verify they are in the same coordinate system by
substring-check, not by-eye-estimation. The validator (§6) will do this
automatically for gold-vs-nc pairings.

---

## 5. Deploy-observable vs gold-derived (input allow-list)

The training-time loader MUST enforce a **deploy-observable allow-list**: any
field read by feature construction is either in `DEPLOY` (§2.2, §2.3) OR
explicitly whitelisted as an intentional exception with rationale in this
spec.

Model INPUT (`ChannelBDataset._build_bag_inputs` and equivalents) MAY consume
ONLY these fields (see `feedback_train_deploy_gap_fields.md`):

- `inputs.flank`
- `inputs.noncoding_regions`
- Derived per-position statistics computed from the above (m_max, structure,
  flank_dev, orient one-hot — all functions of DEPLOY fields)
- Bag-level architecture that is deploy-recoverable: `n_sites`,
  `nc_region_count`, `nc_total_len`, `flank_len`

Model input MUST NOT consume: `is_planted`, `planted_start`, `planted_m`,
`m_at_planted`, `guide_dna`, `active_noncoding_index`, `orient` (as a value),
`guide_span_in_active_noncoding`, `gold_nc`, `epsilon_align`, anything under
`arch.target_*`, or any field whose value depends on knowing the ground truth.

Training TARGETS (`_build_target`) may consume GOLD fields; those are used to
build the loss, not fed into the encoder. The validator (§6) MUST verify no
GOLD field name leaks into the feature-building path.

---

## 6. Validator

The validator (`scripts/canonical_bag_validator.py`, to be written in Step S2)
runs on any data source and produces a **conformance report**. It performs at
minimum:

1. **Field existence** — every §2 required field present on every record.
2. **Type check** — string, int, list, etc.
3. **Bag nc-invariant** — all sites of the same `bag_id` share IDENTICAL
   `noncoding_regions`. Failure = must-split-into-multiple-bags error.
4. **Coordinate self-check** — for every record with a GOLD position, verify
   the position resolves to a substring the spec expects (e.g., `guide_dna`
   appears at `planted_start` in the named coordinate system). See
   `feedback_three_errors_2026_09_07.md` #2 for method.
5. **Value distributions** — report per-source distributions of `flank_len`,
   `n_sites`, `nc_total_len`, `nc_region_count`, and the frequency table of
   any GOLD position field. **Any DEGENERATE distribution (single value on
   >95% of records) MUST be flagged** — this catches `guide_start_in_nc = 49`
   before it becomes a benchmark.
6. **Field taxonomy** — enforce the §5 allow-list against the source's declared
   input pipeline (grep-based check on the loader).
7. **Semantic consistency** — check declared vs actual `orient_granularity`,
   `nc_padding_scheme`, `flank_side` semantics against a sample of records.
8. **Scaffold-sharing (nc uniqueness)** — always report `n_bags`,
   `n_unique_nc_hashes`, `ratio = n_bags / n_unique_nc_hashes`, and the
   bag-count of the top-5 most-common `nc_sequence_hash` values. NO
   threshold flag: `ratio` values speak for themselves (v6r2 expected ≈ 1.0,
   independent nc per bag; Durrant 65.0, one scaffold; DDE ≈ 1.0 per event).
   Any ratio > 1.0 means SOME scaffold sharing exists; the top-5 counts show
   whether it's uniform (all near ratio) or concentrated (one dominant
   scaffold like Durrant). Downstream conclusions on any corpus with
   ratio > 1.0 MUST cite the ratio in their scope declaration.

The validator MUST fail-loud on any check failure. Silent zero-fill,
default-substitution, or "warn and continue" are BANNED (see
`feedback_channel_stats_preflight.md` — the exact bug that led to the P0
retraction).

---

## 7. Conformance report per source (scope declaration)

Every data source contributes a `conformance/<source>.report.md` with these
sections:

1. **Source identity**: manifest path, build date, generator/collection version.
2. **Bag definition on this source**: what biological unit constitutes a bag
   (Tnp? insertion event? per-record?).
3. **§2 field coverage**: which required fields are present, which are missing,
   which have a source-specific meaning.
4. **§3 construction axes**: `flank_side`, `flank_len`, `nc_region_count`,
   `nc_padding_scheme`, `nc_padding_offset`, `orient_granularity`.
5. **§4 coordinate systems** used by each position field.
6. **Known GOLD-position degenerate distributions** (from validator check 5)
   — e.g. "Durrant: `guide_start_in_nc = 49` for 65/65 Tnps."
7. **Known departures from spec**: any allowed non-conformance, with rationale.
8. **Sample-size and unit counts**: n_bags, n_sites, n_tnps, n_events (per §1).

Any conclusion drawn on a source's data MUST be recorded in FROZEN with a
pointer to that source's report. If the report changes (validator re-run),
downstream conclusions are automatically flagged as needing re-audit.

---

## 8. Comparability contracts (cross-source)

Two sources are **comparable at level X** iff their conformance reports agree
on:

- **level=input** — flank_len, flank_side semantics, nc coordinate system,
  nc_padding, orient granularity all identical. Bag structure may differ.
- **level=bag** — level=input AND bag-definition matches (same biological
  unit).
- **level=metric** — level=input AND the metric is invariant to bag structure
  (e.g. per-site scoring, or saturation ratio S/n_sites).
- **level=null** — the METRIC's null distribution is comparable across
  sources. Necessary for comparing ABSOLUTE metric values (not just ordering
  under a paired test on a shared pool). Equivalent to: both sources'
  scanning spaces (nc position count × orient × candidate windows) have the
  same expected chance-hit distribution. When null distributions differ,
  absolute metric values MUST be normalized against each source's own null
  before comparison (e.g. excess-over-null, E-value normalization, or z-score
  against a per-source shuffled background). Without this normalization,
  differences in nc length, flank count, and background composition ARE the
  measurement.

Any cross-source comparison MUST declare its comparability level, and the
metric used MUST match that level. Example, corrected for this session's
Step 1.5b lesson:

> **IS110-Durrant (Tnp-bag) vs DDE (event-bag)** are NOT comparable at
> level=bag because bag definitions differ. Saturation ratio S/n_sites is
> comparable at level=metric ONLY when the scanning space is matched — and
> here it is NOT (Durrant nc = 177, DDE nc = 350+31, so more nc positions
> for DDE means higher chance of ≥8 hits per site independent of biology).
> The valid comparison is therefore at **level=null**: normalize S against
> each source's own null (e.g. excess-over-null, or E-value on this source's
> position count). Absolute max_p S and even S/n_sites are NOT interpretable
> across these two sources without that normalization.

---

## 9. Known departures (current sources — non-binding until validator runs)

Below is the current best guess from this session's audits; **not authoritative**
until the validator produces conformance reports in Step S2.

### v6r2 synthetic (`positives_v5_v6r2_*`)
- Bag = Tnp. n_sites ∈ [3, 8].
- flank_len = 120, flank_side = downstream (single-sided). Source: flank pool
  loaded from `real_{family}_sites.jsonl` via
  `scripts/generator_v5/bag_v2.py:625-644`, filtered to `flank_side ==
  "downstream"`. — **CONFIRMED 2026-09-10 Check A.**
- **Falsifies prior guess "flank_side likely joined"** — v6r2 uses single-side
  downstream flanks (real DDE downstream flanks, reused as pool). No
  flank-structure train/deploy gap; three sources (v6r2, Durrant, DDE) all
  ship 120bp single-side downstream flanks.
- **nc_region_count ∈ {1, 2, 3}, distributed ~1/3 each** (measured on 10K
  records/corpus, 2026-09-10 Check A). Prior "= 1" guess is WRONG. Multi-region
  scoring policy (§3.2) MUST be declared for v6r2 — currently undeclared,
  validator will fail on v6r2 as written.
- planted_m per-site; orient granularity = per_bag (VERIFY).
- Flank-pool independence risk (VERIFY A-1): if `REAL_FLANK_POOL_FAMILIES`
  overlaps with the DDE families being used as evaluation negatives, the
  cross-mechanism experiment loses independence on Channel B (the model saw
  those flank sequences at train time, with synthetic targets planted).
  Channel A unaffected (no training).

### IS110-Durrant (`durrant_cognate.jsonl` etc.)
- Bag = Tnp. n_sites = 5 (all 65 Tnps).
- nc_region_count = 1. nc_total_len = 177 (with padding — scheme name TBD).
- flank_len = 120. flank_side = `downstream` (single-sided).
- Orient granularity = per_site (verified this session).
- **GOLD-position degenerate**: `guide_start_in_nc = 49` for all 65 Tnps
  (see `feedback_check_labels_first.md`).
- `flank_argmax_by_excl` missing in some shard variants → loader must fail-fast
  (already fixed in `data.py`).

### DDE real (`real_{IS10-R,IS30,IS903,ISAjo2,ISLdl1}_sites_bagdedup.jsonl`)
- Bag = insertion event (2 sites per event; ~75% of Tnps have 1 event).
- nc_region_count = 2 (`[left_nc, right_nc]`, lengths e.g. `[350, 31]`).
- flank_len = 120. flank_side ∈ {upstream, downstream}, one per site.
- Each event's TWO sites share IDENTICAL nc (verified). Different events on
  same Tnp have DIFFERENT nc — MUST be split into different bags.
- `active_noncoding_index`: rule unknown (VERIFY — how is one of the 2 regions
  chosen as the "active" one, or is scoring pooled across both?).
- No planted_m (real negatives). Bag `is_positive = False` uniformly.
- ISAjo2 is ISNCY/IS1202 (not-classified family); scope declaration MUST
  flag it as unverified non-IS110.

---

## 10. Enforcement lifecycle

1. **On every new source**: author writes conformance report per §7 before
   the source enters the pipeline.
2. **On every loader change**: re-run validator on all sources.
3. **On every experimental conclusion**: the FROZEN entry MUST cite the
   conformance reports of the sources involved and any comparability level
   assumed.
4. **On validator failure**: the source is REMOVED from the pipeline until
   the report is updated or the loader is fixed. No "warn and continue."
5. **Retrospective annotation (one-time, on first validator run).** After the
   validator's initial run on all existing sources, FROZEN MUST be walked
   end-to-end and every conclusion citing a source get one of:
   - **`SCOPED [source_a=report_v, source_b=report_v, level=X]`** — the
     conformance level is verifiable from the reports and the conclusion
     is compatible with that level. Keep the conclusion.
   - **`RESCOPED [was: ..., now: ...]`** — the conclusion held under a
     stricter interpretation than reports support; rewrite the claim.
   - **`UNSCOPED`** — the source's conformance cannot be reconstructed from
     current artifacts. Mark the number as suspended pending re-audit; do
     NOT delete (audit trail).
   Any conclusion referenced from later work MUST inherit its parent's scope
   tag; a fresh claim built on an `UNSCOPED` number is itself `UNSCOPED`.

---

## CHANGELOG

- 2026-09-11 rev4: §3.2 gained the "first-site nc = bag nc" rule. v7 under
  homology<1.0 emits per-site variant nc strings in `noncoding_regions[0]`;
  the bag's canonical nc is pinned to the FIRST site's version and every
  shard/loader materialization keys against it. Prompted by a v7 training-smoke
  KeyError (shard builder was silently dropping per-site nc variants);
  paired with a fail-loud rewrite of `scripts/build_v7_shard.py` that raises
  on any per-bag site-count mismatch with `arch.n_sites`.
- 2026-09-10 rev1: initial draft, motivated by Step 1.5b bag-definition
  mismatch between DDE real (event-bag) and IS110-Durrant (Tnp-bag). Author:
  user directive after Step 1.5b-fix report.
- 2026-09-10 rev3: user review pass 2. §6 check 8 threshold removed —
  always report `ratio` + top-5 scaffold hash counts (v6r2 ≈ 1.0, Durrant
  65.0, no interpolation). §3.2 declared-vs-actual test written concretely
  (N-blank one region, plant match in the other, assert loader behavior
  matches declared policy) — closes the "declaration and implementation may
  drift silently" gap.
- 2026-09-10 rev2: user review pass. Removed `canonical_nc` coordinate system
  (was undefined). Added `nc_sequence_hash` (§2.3) + §6 check 8 for scaffold-
  sharing detection (guards against `guide_start_in_nc = 49`-class failures).
  Added `nc_multi_region_scoring` (§2.3, §3.2) with declared-only-no-default
  policy. Added `epsilon_align` to §2.5. Tightened `generator_metadata` to
  three required subfields (`data_source`, `build_date`,
  `generator_version_or_commit`). Split §3.3 Channel A variants into
  `S_oc` (deploy-legal), `A_maj_gold` (not deploy-legal), `A_maj_heuristic`
  (deploy-legal with declared heuristic). Added §8 level=null comparability
  tier and corrected the DDE-vs-IS110 example to require null normalization.
  Added §10 rule 5 (retrospective annotation of FROZEN with `SCOPED` /
  `RESCOPED` / `UNSCOPED` tags after first validator run).
