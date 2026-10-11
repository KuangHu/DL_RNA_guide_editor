# Model-free scorer v3 — spec and progress log

Date: 2026-10-10

## Problem

v1 / v2 of the model-free scorer (`scripts/model_free_scorer.py`,
`scripts/model_free_scorer_v2.py`) had five correctness or completeness
issues that let low-complexity bags, RC-frame-mismatched hits, and
chain-linked clonal backgrounds inflate the top of the ranking:

1. **RC frame mixing.** `np.where(m_f >= m_r, a_f, a_r)` mixed fwd-flank
   coordinates with rc-flank coordinates on the same axis. The identity
   `a_r_in_fwd = flank_len − a_r − L` from `preprocess/alignment.py:228`
   was not applied, so `position_consistency` and target-substring
   extraction were silently wrong for any site that scored higher on RC.
2. **Argmax-only retention.** `_mt.get(...).m_max_by_excl.get(0)` is
   already reduced to the single best flank offset per `(site, nc_pos, L)`.
   A cross-site cluster whose shared hit is a per-site SECOND-best match
   cannot be recovered — the argmax reduction has thrown it away.
3. **No complexity gate.** `ATATATATAT`, `ACGACGACG` and
   `ACGTACGTACGT` with perfect per-site match pass `m_consistency = 1`,
   `position_consistency = 1`, and only soft penalties through
   `flank_diversity`. The last one is the critical case: it has
   `H_base = 2.00`, `AT_frac = 0.5`, `longest_run = 1`, and only
   period-4 detection catches it.
4. **Wobble×2 doubling** multiplied the final score when
   `std_a < 5 AND unique_targets > K/2`. That rule confirms target
   variation but gives no evidence of guide variation; a 2× multiplier
   on that signal has no justification.
5. **Flank diversity as a soft multiplier.** A clonal-flank bag (all
   sites > 90% identical outside the target) was penalized by a lower
   `flank_diversity`, not rejected. Chain-linked local backgrounds
   bridged dissimilar pairs via a transitive partner, inflating
   apparent independence.

A sixth issue, reported mid-review: `m_max_by_excl.get(0)` was described
as "no mismatch tolerance band." It is actually the **TSD exclusion
width = 0** key (no TSD exclusion). The 0 is correct for IS110 /
Durrant positives but wrong default for IS10-R / IS903 which should
use 9. v3 inherits the key choice; family-specific `excl_w` is a
follow-up.

## v3 architecture

Six phases. Each phase's gates are explicit knobs, every gate rejection
is logged with a reason, nothing is scored and quietly penalized.

```
sites  ──Phase A──▶  BagRetention(K_valid, hits per (site, nc_pos, L))
                      │
                      ├──Phase B──▶ gates on guide AND target per hit
                      │              drops hits, K_valid frozen upstream
                      │
                      ├──Phase C──▶ exhaustive window enumeration
                      │              yields list[Cluster] per bag
                      │
                      └──Phase D──▶ bg grouping (left/right separate),
                                     n_local_bg_groups ≥ 3 hard gate
                                     yields (passing, failing) with BgReport

                     Phase G (parallel, calibration-only):
                     real + 4 null constructions → report distributions,
                     set m_floor_report(L) and ranking thresholds.
```

---

## Phase A — multi-hit retention

**Module:** `scripts/v5a_framework/multi_hit_retention.py`

- `retain_hits(nc, flank, Ls, m_floor_fn) → list[Hit]` recomputes
  `windowed_matches(dot_plot(nc, flank), L)` ourselves rather than
  reading `_mt`; every flank offset with `m ≥ m_floor_retain(L)` is
  kept, with orient and fwd-flank coord already resolved.
- RC hits are converted to fwd-flank coord at retention time via
  `rc_flank_pos_to_flank_pos(j_rc, L, flank_len)`.
- Windows that straddle the 13-N `nc_concat` spacer are dropped before
  any match counting.
- `K_valid` frozen by `compute_K_valid(sites, min_nc_len, min_flank_len)`
  BEFORE any filtering — a site that fails Phase B motif gates still
  counts in the denominator.
- `m_floor_retain_default(L) = ⌈0.6 · L⌉`. This is a loose RECALL knob,
  explicitly NOT a credibility threshold; the credibility floor
  `m_floor_report(L)` comes from Phase G.

**Tests (`scripts/tests/test_phase_a_multi_hit.py`):**

| # | case | what it verifies |
|---|---|---|
| 1 | Each site's best hit at scattered positions (10, 25, 40, 55, 65) + shared m=L hit at 80 | both best AND non-best cross-site hits retained; argmax-reduction regression impossible |
| 10 | Build positive bag, revcomp both nc and flank per site | retention surfaces hits at coord-transformed `(nc_len − nc_pos − L, flank_len − flank_pos − L)`; `orient` may differ, coord equivalence must not |
| 11 | 8 input sites: 4 valid, 4 malformed (empty nc list, nc < min_len, flank < min_len, missing flank key) | `K_valid = 4`; downstream filtering never shrinks it |

All three PASS.

---

## Phase B — bilateral motif gates

**Module:** `scripts/v5a_framework/motif_gates.py`

Applied per hit on BOTH guide and target. Four separately-tunable gates
(each with its own knob, no composite score):

```
GateConfig(homopolymer_frac=0.5,
            period_frac=0.80, periods_to_check=(1,2,3,4),
            at_frac_max=0.75,
            h_base_min=1.3)
```

| gate | reject if | catches |
|---|---|---|
| homopolymer | `longest_same_base_run / L ≥ homopolymer_frac` | poly-A/T/G/C |
| short-period | `max_p∈{1,2,3,4} periodicity_fraction(seq, p) ≥ period_frac` where `periodicity_fraction = (# i with seq[i]==seq[i-p]) / (L-p)` | ATAT, ACGACG, **ACGTACGT** (the case the earlier `distinct_dinuc` / `distinct_trinuc` plan missed) |
| at_frac | `(A+T)/L > at_frac_max` | AT-rich non-repetitive |
| h_base | `shannon_base(seq) < h_base_min` bits | overall low complexity |

`apply_phase_b(bag, sites, cfg) → (bag_filtered, RejectionLog)` runs the
gate on every retained hit's guide (from site-specific `nc_concat`) and
target (`flank[flank_pos:flank_pos+L]`). A hit is dropped iff EITHER
side fails. `RejectionLog` tabulates `(side, reason) → count` for
diagnostic reports.

**Tests (`scripts/tests/test_phase_b_motif_gates.py`):**

| case | what passes/fails | critical property |
|---|---|---|
| primitives | `longest_run`, `periodicity_fraction`, `at_fraction`, `shannon_base` match hand-computed values | — |
| 2 | poly-A `AAAAAAAAAAA` | all 7 gates fire (expected multi-reason reject) |
| 3 | `ATATATATATA` | period2 + period4 + at_frac + h_base all fire |
| 4 | `ACGTACGTACGT` | **ONLY period4 fires** — H_base=2.00, at_frac=0.5, longest_run=1 all pass. The gate that the plan's earlier dinuc/trinuc check missed. |
| 5 | `AATTAACGTTAA` | **ONLY at_frac fires** — H_base=1.63, no period ≥ 0.80, longest_run=2. Proves AT-rich and complexity are independent knobs. |
| E2E | Case-4 bag through Phase A → B → C | 2029 clusters before Phase B; after, **0 clusters at the guide location**. 1,932 target + 1,462 guide period-4 hits removed. |

All PASS.

---

## Phase C — exhaustive window enumeration

**Module:** `scripts/v5a_framework/cluster_enumeration.py`

For each `(nc_pos, L)` where any site has a hit:

1. Candidate window centers = every observed `flank_pos` (deduped).
2. Window = `[center − max_offset, center + max_offset]`.
3. Each site contributes at most ONE hit per window — the one with
   smallest `|flank_pos − center|`, tie-broken by `-m` then `flank_pos`.
4. Keep if `n_support / K_valid ≥ min_frac` (default 0.5).
5. Dedup bleed-through from partial-match offsets: if two candidate
   clusters share `≥ dedup_jaccard_min` of supporting sites AND their
   medians are within `max_offset`, keep the one with higher
   `mean_m_frac` (ties → tighter MAD).

Each `Cluster` carries full per-site provenance:
`supporting_sites[].rank_within_site`, `offset_from_median`,
`chosen_hit`, `n_hits_this_site_at_nc_pos_L` (ties-for-best count,
NOT raw hit count — partial-match offsets no longer inflate it).
`mean_hit_multiplicity_at_nc_pos_L` reports the mean of
ties-for-best per supporting site.

**Tests (`scripts/tests/test_phase_c_clustering.py`):**

| # | case | outcome |
|---|---|---|
| 6 | Each site has the same guide at flank positions 60, 80, 100 | 3 separate clusters at median 60/80/100, each `n_support=5`, `mean_hit_multiplicity=3.0`. Partial m=7 hits inflating a position-cluster is prevented. |
| 9 | Two shared guides at different `(nc_pos, L)`: `(100, 11, 80)` and `(250, 12, 40)` | Both emitted as independent cluster rows. |
| 12 | 3-site chain-linked background construction A~B=0.922, B~C=0.922, A~C=0.845 | Phase D (partial, needed to test this) reports `n_local_bg_groups=1`, `chain_linked=True`. |

All PASS.

### Correctness fixes landed vs first draft

- `mean_hit_multiplicity` = ties-for-best per site (not raw hit count);
  partial-match offsets no longer inflate it.
- Dedup distance widened from `max_offset/2` to `max_offset`.
- Dedup sort key puts `mean_m_frac` ahead of MAD so the strongest
  cluster wins the merge, not the first-seen.

---

## Phase D — background grouping, left/right separate, hard gate

**Module:** `scripts/v5a_framework/background_grouping.py`

```
BgConfig(margin=3,
          clonal_bg_threshold=0.85,
          min_bg_bp=20,
          min_bg_groups=3,
          require_both_sides=False)   # permissive default
```

Per supporting site:

- Mask `[flank_pos − margin, flank_pos + L + margin]`.
- `n_bp_left_available = lo`, `n_bp_right_available = len(flank) − hi`.
- `edge_insufficient_left / right / both` flags if `< min_bg_bp`.

Per pair of sites that both have ANY masked bg:

- Pairwise identity computed BOTH on each side independently (reported
  as `pairwise_left_identity`, `pairwise_right_identity`) AND on
  concatenated bg (used for grouping). The common chunk is trimmed
  adjacent to each site's own mask.
- `median_left_identity`, `median_right_identity` summarize per-side
  pair identities (omitted when either side lacks `min_bg_bp` on either
  partner).

Grouping:

- Edge between two sites iff concat identity > `clonal_bg_threshold`.
- Connected components = **local background groups**.
- `chain_linked` flag set when any within-component pair falls at or
  below threshold — honest signal that transitivity bridged them.

Hard gate:

- `passes_bg_gate = (n_local_bg_groups ≥ min_bg_groups)`.
- Strict mode (`require_both_sides=True`): additionally fails if ANY
  supporting site has `edge_insufficient_left` or
  `edge_insufficient_right`.

`apply_phase_d_gate(clusters, flanks_per_site, cfg)` returns
`(passing, failing)` where each is `list[(Cluster, BgReport)]`.
Neither list drops clusters — both carry the `BgReport` so a failing
cluster's `rejection_reason` is inspectable.

**Caveat (part of module contract, cite in output):**
`n_local_bg_groups` is a LOCAL 120 bp observation, NOT a genome-level
independence claim. Chain-linked components may bridge dissimilar pairs
through a transitive partner. For genome-level distinctness,
cross-reference `carriers_derived` / `pre_event_genome` in
`top4_candidates/*/provenance.tsv`. Distinct assembly accessions do
not themselves prove sequence context differs; the local bg check is
the only sequence-level signal here.

**Tests (`scripts/tests/test_phase_d_background.py`):**

| # | case | outcome |
|---|---|---|
| 7 | 5 clonal flanks (~95% pairwise identity on masked flank), shared motif | `n_local_bg_groups=1`, `rejection_reason='insufficient_bg_groups(1<3)'` |
| 8 (permissive) | guide at `flank_pos=109, L=11, flank_len=120` | `edge_insufficient_right={0..4}`, left bg 106 bp per site, random: `n_local_bg_groups=5`, PASSES |
| 8 (strict) | same but `require_both_sides=True` | REJECTED with `rejection_reason='edge_insufficient_sites([0,1,2,3,4])'` |
| mixed | bag A clean, bag B clonal, bag C edge | A: pass=1, B: fail=1, C: pass=1 |

All PASS. Case 12 (chain-linked) from Phase C still passes under the
expanded `BgReport` schema (regression check).

---

## Phase G — negative-control calibration

**Modules:**
- `scripts/v5a_framework/null_generators.py` — four null constructors
- `scripts/v5a_framework/pipeline.py` — unified Phase A→D runner
- `scripts/calibrate_v3_nulls.py` — driver
- `scripts/analyze_v3_calib.py` — post-processor

### Nulls

| id | construction | tests |
|---|---|---|
| `null1_dinuc` | per-flank dinucleotide-shuffle (preserves dinuc composition, destroys higher-period) | baseline null for repeat structure |
| `null2_4mer` | per-flank 4-mer-shuffle (preserves 3-mer composition) | tighter null, closer to real repeat statistics |
| `null3_unpaired` | this bag's flanks replaced by K flanks sampled from a different donor bag | preserves per-flank composition exactly; breaks nc↔flank pairing. CAVEAT: donor bag may itself carry real signal → read as plausibility bound, not strict null |
| `null4_rc_shuf` | per-flank: reverse-complement then dinuc-shuffle | isolates RC-strand leakage from the retention machinery |

Shuffle is a random-walk approximation on the k-mer transition graph
(preserves conditional composition in expectation, not exact k-mer
counts). Good enough as a null that scales to tens of thousands of
flanks; swap in Altschul-Erickson Eulerian walk for publication-grade
exact preservation.

### Smoke run (20 bags, local, ~3 min)

Confirmed the pipeline runs end-to-end. Qualitative signal already
visible:

| condition | p99 max_mean_m | p99 max_n_support | median n_passing |
|---|---|---|---|
| real | **0.938** | 8 | 219 |
| null1 dinuc-shuffle | 0.704 | 8 | 292 |
| null2 4-mer shuffle | 0.741 | 8 | 483 |
| null3 unpaired bag | 0.885 | 8 | 192 |
| null4 RC+shuffle | 0.700 | 8 | 374 |

- `max_mean_m` cleanly separates real from true nulls; null-3 is
  tighter (0.885) as expected.
- `max_n_support` is uninformative — saturates at K=8 for every
  condition.
- `n_passing` is **200–500 per bag across ALL conditions** → the loose
  retention floor + permissive gates produce massive cluster slates.
  The report threshold Phase G will pick drives this to single digits
  per bag on nulls.
- Analyzer picks `min_max_mean_m ≈ 0.78` where every null is ≤ 5%
  pass-rate while real recall is 20% — too strict for a 20-bag sample.
  Awaiting the 500-bag run.

### 500-bag calibration

Submitted as Slurm job 26836963 on lr7 (`sbatch/v3_calib_nulls.sbatch`,
2 h time limit, 4 CPUs, 500 bags from
`fna_ins_discovery_v5_md_seed2/adapted.jsonl`). Expected runtime
~75 min. On completion, run:

```
python scripts/analyze_v3_calib.py \
  --details logs/v3_calib_500_details.tsv \
  --target-null-pass 0.01
```

Output picks:
- `min_max_mean_m` such that every null's pass-rate ≤ 1%
- `min_max_n_support` under the same constraint
- Suggested `m_floor_report(L)` from REAL p25 of `best_m_at_top / L`
  per `L`.

---

## Status

| step | phase | tests | status |
|---|---|---|---|
| 1 | A multi-hit retention | 1, 10, 11 | DONE |
| 2 | C cluster enumeration + partial D | 6, 9, 12 | DONE |
| 3 | B motif gates + E2E | primitives, 2, 3, 4, 5, E2E | DONE |
| 4 | D full + apply_phase_d_gate | 7, 8 permissive, 8 strict, mixed | DONE |
| 5 | G 4-null calibration | smoke 20-bag + 500-bag Slurm (in flight) | 500-bag running |
| 6 | ranking + final output schema (`candidates_v3.tsv`, `rejections_v3.tsv`) | — | pending |

### What's not yet done (step 6)

Per the plan `## v3 revised plan` from the design review:

- `candidates_v3.tsv` — one row per surviving `(bag_id, nc_pos, L, cluster_id)` with full provenance:
  `nc_pos, L, flank_pos_median, flank_pos_mad, K_valid, n_support_raw, support_frac_raw, n_local_bg_groups, support_frac_bg_unique, mean_m_frac, min_m_frac, n_sites_cluster_hit_is_best, n_sites_cluster_hit_rank_1_or_2, guide_at_cluster, mean_guide_H_base, mean_guide_H_dinuc, mean_guide_at_frac, guide_longest_run, mean_target_H_base, mean_hit_multiplicity_in_window, left_median_identity, right_median_identity, n_sites_edge_left, n_sites_edge_right, pass_phase_B, pass_phase_C_support, pass_phase_D_bg`
- `rejections_v3.tsv` — one row per rejected candidate with phase + reason + per-site contributions
- Three orthogonal ranking columns (no composite): `rank_by_bg_groups`, `rank_by_mean_m`, `rank_by_tightness`
- Legacy single-score column dropped; joins to v1 / model use `bag_id`
- Only after Phase G thresholds are set: define a single publication threshold.

## Files

```
scripts/v5a_framework/
    multi_hit_retention.py        # Phase A
    cluster_enumeration.py        # Phase C
    background_grouping.py        # Phase D
    motif_gates.py                # Phase B
    null_generators.py            # Phase G nulls
    pipeline.py                   # unified A→D runner
scripts/
    calibrate_v3_nulls.py         # Phase G driver
    analyze_v3_calib.py           # Phase G post-processor
    model_free_scorer.py          # v1 (unchanged, reference)
    model_free_scorer_v2.py       # v2 (unmerged partial fix, reference only)
scripts/tests/
    test_phase_a_multi_hit.py
    test_phase_b_motif_gates.py
    test_phase_c_clustering.py
    test_phase_d_background.py
sbatch/
    v3_calib_nulls.sbatch
docs/
    model_free_scorer_v3.md       # this file
```

Run all tests:

```
for t in scripts/tests/test_phase_?_*.py; do
    python "$t" || break
done
```
