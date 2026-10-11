# Model-free scorer v3 — spec and progress log

Date: 2026-10-10 (updated 2026-10-11)

## Retractions (2026-10-11)

Four claims in earlier revisions of this doc are retracted. Reader should
ignore them wherever they appear below until the sections are rewritten
in a follow-up pass.

### R1 — "0.89 publication threshold strictly controls null pass-rate ≤ 1 %"

Not supported by the calibration code. `scripts/calibrate_v3_nulls.py`
(line 117) records the mean_m of each bag's **max-support cluster**;
the publication filter in `scripts/run_v3_scorer.py` accepts
**any** cluster whose mean_m ≥ threshold. These are different
statistics. A bag with cluster A (support=8, mean_m=0.80) and cluster
B (support=4, mean_m=1.00) is recorded as 0.80 in the calibration
but passes the filter via cluster B at 0.89. The reported
"null frac_pass = 0.0 %" is therefore an undercount of the actual
null pass-rate under the deployed filter rule.

Calibrator and reporter must share a single bag-decision function.
500-bag Phase G must be re-run after fixing BOTH the metric and the
upstream clustering before any claim about `min_max_mean_m` is
honored.

### R2 — "Y1_Tnp hit = positive-control recovery of IS200/IS605 RNA-guided"

Family-level error. Y1_Tnp (Pfam PF01797) is a **HUH / single-tyrosine
transposase** (TnpA-like) — not the RNA-guided nuclease. In IS200/605
the RNA-guided protein is **TnpB** (Pfam `OrfB_IS605`,
`HTH_OrfB_IS605`, `Cas12f1-like_TNB`, `Transposase_28/31`). A Y1_Tnp
hit identifies a transposase of a family that *contains* an RNA-guided
nuclease elsewhere in its loci; it is not itself the RG protein and
does not validate v3's RG discovery.

(Note: an earlier retraction draft miscalled Y1_Tnp a serine
recombinase. Y1_Tnp is HUH / single-tyrosine, not serine. HUH and
serine are distinct transposase folds.)

### R3 — Actual TnpB-family recovery is 0 of 10 at 0.89

`all_candidates/manifest.tsv` contains **10 bags** whose dominant ORF
Pfam at E ≤ 1e-5 is a TnpB-family hit (`OrfB_IS605`,
`HTH_OrfB_IS605`, `Cas12f1-like_TNB`, individually or in
combination):

| CID | strict Pfam | n_pass_D | n_pub_thr | best mean_m | bg | sup |
|---|---|---:|---:|---:|---:|---:|
| CDS03228 | Cas12f1-like_TNB, HTH_OrfB_IS605, OrfB_IS605 | 34 | 0 | 0.733 | 3 | 5 |
| CDS02499 | OrfB_IS605, Cas12f1-like_TNB | 94 | 0 | 0.725 | 3 | 4 |
| CDS03385 | OrfB_IS605, Cas12f1-like_TNB | 82 | 0 | 0.750 | 4 | 4 |
| CDS03362 | Cas12f1-like_TNB | 70 | 0 | 0.733 | 3 | 5 |
| CDS03474 | HTH_OrfB_IS605 | 334 | 0 | 0.833 | 4 | 4 |
| CDS03281 | OrfB_IS605, Cas12f1-like_TNB | 23 | 0 | 0.682 | 4 | 4 |
| CDS02404 | OrfB_IS605, Cas12f1-like_TNB | 111 | 0 | 0.727 | 4 | 4 |
| CDS02902 | OrfB_IS605, Cas12f1-like_TNB | 568 | 0 | 0.750 | 4 | 4 |
| CDS03319 | OrfB_IS605, Cas12f1-like_TNB | 55 | 0 | 0.722 | 4 | 4 |
| CDS03026 | OrfB_IS605, Cas12f1-like_TNB | 206 | 0 | 0.727 | 4 | 4 |

**0 of 10 TnpB bags cross the 0.89 bar.** Every TnpB bag produces
strong cross-site evidence (23–568 Phase-D-passing clusters, 3–4
local bg groups, 4–5 supporting sites) but at mean_m 0.68–0.83 —
consistent with TnpB's wobble-tolerant guide biology but below the
calibrated null bar. The strict threshold excludes the biological
positive control it was supposed to validate.

Equivalent-threshold IS110 recovery is also mostly misses — 2 of 28 pass.
Previous attribution of the 26 IS110 misses to "wobble" was presented
too strongly; per-case failure analysis (coord sanity, max per-site m,
flank-position spread, Phase B rejection reasons) is pending before
any attribution.

### R4 — "CDS03597 validated as novel RNA-guided"

Overclaim. CDS03597 has: strong cross-site mean_m=1.000 target
coherence, 4 local bg groups, 4/8 sites supporting, dominant ORF
239 aa, strict Pfam = none, sensitive hmmscan hits `DUF1672`
(E=3e-3, 17.6 bits — weak, uncharacterized) and `DUF6803`
(E=5e-3, 17.7 bits — weak, uncharacterized). Correct description:
"strong cross-site target-recognition signal; no reliable functional
assignment at either strict or sensitive Pfam." That is consistent
with — but does not establish — a novel RG system.

Separately, the "novel" tag applied by `annotate_v3_top_hits.py`
to any bag with no significant Pfam hit (strict and sensitive) is
rephrased as "no reliable Pfam annotation". A biology claim of
"novel RNA-guided system" requires more than Pfam-blindness.

## Discovery pipeline is not model/Pfam-blind

The v3 **scorer** does not use the v4/v5 model or Pfam annotation. The
1173-bag **input** to the sweep is `all_candidates/`, which was
filtered by `max(v4_bag_max, v5_bag_max) ≥ 1.5` AND dominant Pfam
not in a known-non-RG set (DDE, rve/TnsB, phage_integrase, IS-support
HTH). Any claim about "discovery" refers to v3's blind scoring **on
top of** a model- and Pfam-filtered input set, not to a from-scratch
Pfam-blind pipeline.

## Known residual bugs (listed for the fix queue)

1. **Calibration/filter statistic mismatch** (`calibrate_v3_nulls.py:117`).
   See R1. Must be fixed before any `min_max_mean_m` threshold is
   re-published.
2. **Phase C window enumeration** (`cluster_enumeration.py:103`).
   Candidate centers are observed positions only, not integer sweep;
   no re-check that supporting-site offsets from the **median** are
   within `max_offset`. Example `[0,0,10,10]` misses the 4-support
   cluster at center 5; example `[0,5,10,10,10]` returns a cluster
   with offset_from_median = −10.
3. **`support_frac_bg_unique`** (`run_v3_scorer.py:153`). Computed as
   `n_support / n_local_bg_groups` — numerator in sites, denominator
   in bg groups; can exceed 1; misleadingly named. Being dropped
   rather than renamed; a true bg-dedup fraction requires a
   consistent assessable-background set over ALL valid sites, not
   just supporters.
4. **ncRNA cross-site coord assumption**. Phase C groups hits by raw
   `nc_pos` across sites. For v5-reconstructed bags where sites' nc
   may differ by indel or reorder, same `nc_pos` is not the same
   physical position. Audit required before any further quantitative
   claim.
5. **Indel-aware bg identity**. Char-by-char `_pair_identity` can
   miss indel-shifted clonal backgrounds. Separate from
   `require_both_sides=True` (which only guarantees enough bp per
   side, not that both sides differ).

## Fix queue (agreed order)

```
1. Doc retractions (this section)                       — in this commit
2. Coord + bg audit on 1173 inputs                      — upstream data
3. Phase C fix (window + half-integer median +
   per-site re-select-then-remove) + exhaustive
   small-input reference tests                           — code
4. Stat-field cleanup (drop support_frac_bg_unique;
   keep two separate columns; defer bg-dedup frac)      — code
5. Shared bag-decision function used by BOTH
   calibrate_v3_nulls.py and run_v3_scorer.py            — code
6. Regression (15 existing + 2 counterexamples +
   exhaustive small-input test)                          — gate
7. Freeze; calibrate (500-bag); report whatever null
   distribution says, including "no threshold ≤ 1.0
   meets target" as an allowed outcome                   — stats
8. Re-run 1173 sweep; per-case IS110 (28) + TnpB (10)
   failure analysis                                      — final
```

Everything below this line was written before the retractions and
is NOT yet rewritten to reflect them. Specific claims about the 0.89
threshold's strict null control, Y1_Tnp as positive-control recovery,
and CDS03597 as "validated" are superseded by R1–R4. Section-by-section
rewrite follows the fix queue.

---

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

Confirmed the pipeline runs end-to-end on real data. Numbers superseded
by the 500-bag calibration below.

### 500-bag calibration

Slurm job 26836963 on lr7 (`sbatch/v3_calib_nulls.sbatch`,
4 CPUs). Completed 2026-10-10 23:23, elapsed **36:36**, exit 0.
Input: 500 bags sampled from
`fna_ins_discovery_v5_md_seed2/adapted.jsonl` with seed 2026.
Each bag runs through Phase A→D under real + 4 null conditions
(2,500 pipeline runs).

Outputs:
- `logs/v3_calib_500_summary.tsv` — one row per condition
- `logs/v3_calib_500_details.tsv` — one row per (bag, condition)

**max_mean_m distribution** (best cluster per bag per condition):

| condition | p25 | p50 | p75 | p90 | p99 | p99.9 | max |
|---|---|---|---|---|---|---|---|
| real              | 0.659 | 0.716 | 0.778 | 0.875 | **1.000** | 1.000 | 1.000 |
| null1_dinuc       | 0.625 | 0.643 | 0.675 | 0.704 | 0.778 | 0.815 | 0.815 |
| null2_4mer        | 0.629 | 0.657 | 0.683 | 0.727 | 0.822 | 0.879 | 0.879 |
| null3_unpaired    | 0.638 | 0.675 | 0.741 | 0.787 | 0.889 | 1.000 | **1.000** |
| null4_rc_shuf     | 0.629 | 0.643 | 0.675 | 0.711 | 0.778 | 0.815 | 0.815 |

Reads:
- real has a long upper tail — p99 reaches perfect m. Nulls 1/2/4 top
  out at ~0.82.
- null3 (unpaired flanks from a real donor bag) extends to max=1.000.
  This is the expected caveat: a donor bag is itself a real bag and can
  carry real signal — null3 is a plausibility bound, NOT a strict null.
- max_n_support saturates at K=8 for every condition and does not
  discriminate real from null. **It is dropped as a ranking signal.**

### Chosen thresholds

Analyzer target: null pass-rate ≤ 1 % for every strict null (null1,
null2, null4). null3 reported separately with its caveat.

| threshold | value | real recall | null1 | null2 | null3 | null4 |
|---|---|---|---|---|---|---|
| `min_max_mean_m` | **0.89** | **6.2 %** (31/500) | 0.0 % | 0.0 % | 0.20 % | 0.0 % |
| `min_max_n_support` | — | — (saturates at K; drop as signal) | — | — | — | — |

**`m_floor_report(L)` per L** (from real p25 of `best_m_at_top/L`):

| L | n_real_bags | p25 | suggested `m_floor_report(L)` | fraction |
|---|---|---|---|---|
| 9  | 176 | 0.778 | 7 | 0.78 · L |
| 10 | 236 | 0.600 | 6 | 0.60 · L |
| 11 | 55  | 0.636 | 7 | 0.64 · L |
| 12 | 11  | 0.667 | 8 | 0.67 · L |
| 13 | 6   | 0.615 | 8 | 0.62 · L |

Same as or slightly tighter than the current retention floor
`⌈0.6 · L⌉`. The retention floor stays as-is (recall knob);
`m_floor_report(L)` applies at the final ranking step.

### Interpretation

6.2 % of random v5 bags clear the 0.89 `max_mean_m` bar with every
strict null below 1 %. The bars match the biology: most v5 bags are
DDE/other transposases where the model-free scorer should NOT find a
strong cross-site motif. Only a small tail — RNA-guided candidates
plus whatever other systems have genuinely coherent target recognition
— survives.

null3's one max=1.000 case is the predicted leak: a donor bag that
itself carried a real cluster, which the shuffled target bag's nc
happened to also hit. This is not a correctness failure of v3 —
it is why the plan treats null3 as a plausibility bound and why
`occupied_carriers` / `pre_event_genome` from
`top4_candidates/*/provenance.tsv` is the primary genome-level
independence signal, not a 120-bp local background.

---

## Acceptance against the four criteria

Mapping to the review criteria from 2026-10-10:

| # | criterion (zh) | coverage | status |
|---|---|---|---|
| 1 | 最佳命中分散时，能找回共同的次佳位置簇 | `test_phase_a` case 1 verifies retention preserves non-best; `test_phase_c` **case 1-strict** (added 2026-10-10) verifies full A→C surfaces the shared pos=80 cluster when each site's STRICT best is a different scattered position. The surviving cluster carries `n_sites_cluster_hit_is_best = 0` as the diagnostic. | **PASS** |
| 2 | 实际 guide 和 target 都经过复杂度及 AT-rich 硬过滤 | `apply_phase_b` evaluates both sides on every hit. Cases 2/3/4/5 verify each gate on individual sequences; the E2E case verifies the ACGT-period-4 bag has **zero clusters** at the guide location after Phase B. AT_frac is a SEPARATE knob from complexity (case 5). | **PASS** |
| 3 | 支持比例分母固定，重复背景不增加独立支持 | `compute_K_valid` freezes the denominator against site input adequacy only — downstream filtering never shrinks it (`test_phase_a` case 11). Chain-linked clonal flanks collapse to one component (`test_phase_c` case 12, `test_phase_d` case 7). `support_frac_bg_unique = n_support / n_local_bg_groups` as its own ranking column is specified for step 6; current `BgReport` has the primitives for it. | **partial** (support ratio frozen; bg-unique ranking column pending step 6) |
| 4 | 完整搜索流程经过负对照校准，而不只是 smoke 测试 | 500-bag calibration run (job 26836963, 36:36 wall, 2500 pipeline executions on 4 nulls + real) with thresholds reported above. `min_max_mean_m = 0.89` gives real recall 6.2 % while every strict null's pass-rate is ≤ 0.0 % (null3 at 0.20 % with the documented caveat). | **PASS** (thresholds set; applying them in `candidates_v3.tsv` output is step 6) |

## Step 6 — output schema + applied to top-4

**Driver:** `scripts/run_v3_scorer.py`

Reads a sites JSONL + optional bag-id filter, runs the full Phase A→D
pipeline per bag, writes two TSVs:

- `candidates_v3.tsv` — one row per `(bag_id, cluster_id)` with all 38
  schema columns (`nc_pos, L, flank_pos_median, flank_pos_mad, …,
  guide_at_cluster, mean_guide_H_base, mean_guide_H_dinuc,
  mean_guide_at_frac, mean_target_H_base, mean_hit_multiplicity_at_nc_pos_L,
  left_median_identity, right_median_identity, n_sites_edge_*,
  passes_phase_D, passes_publication_threshold, rank_by_bg_groups,
  rank_by_mean_m, rank_by_tightness`). Phase-D-failing clusters stay in
  the file with `passes_phase_D = False` so the full enumeration is
  auditable; three orthogonal rank columns are populated only for
  passing rows. **No composite score column** per the plan.
- `rejections_v3.tsv` — one row per Phase-D-failing cluster with
  `phase + rejection_reason`.

`support_frac_bg_unique = n_support / n_local_bg_groups` is included,
closing the remaining gap flagged against criterion 3.

Default publication threshold: `min_max_mean_m = 0.89` from the Phase G
500-bag calibration.

### Applied to the top-4 banked candidates

Input: 4 bags from `fna_ins_discovery_v5_md_seed2/adapted.jsonl` (K=8 each).

| bag | n_pass_D | n_pub_thr | best mean_m | best cluster (by bg groups) |
|---|---:|---:|---:|---|
| **CDS03597** | 807 | **185** | **1.000** | nc=1, L=9, fp=50, mean_m=1.000, 6 bg groups |
| CDS04337 | 404 | 0 | 0.806 | nc=42, L=9, fp=9, 8 bg groups (below 0.89 bar) |
| CDS00113 | 71 | 0 | 0.705 | nc=96, L=11, fp=7, 6 bg groups |
| CDS02973 | 29 | 0 | 0.650 | nc=155, L=10, fp=71, 5 bg groups |

**CDS03597 is the only one that clears the strict v3 threshold.**
Its 185 passing clusters are offset-variants of a single conserved
GC-rich motif:

```
guide_at_cluster  n_pos  flank_pos  L  mean_m  n_support  n_bg
----------------  -----  ---------  --  ------  ---------  ----
CCGCGCGCA         9      42         9  1.000   4          4
CCGCGCGCAGAGCC    9      37        14  1.000   4          4
CCGCGCGCAGAGC     9      38        13  1.000   4          4
… (all 185 are L-slide / nc-slide neighbours of the same CCGCGCGCAGAGCC core)
```

A second strong signal shows up at `rank_by_bg_groups = 1`:
`TCATTGTTCCGGC` at nc_pos=27, flank_pos=20, mean_m=0.879 (just below
the 0.89 bar), **7/8 sites support, 6 local bg groups, 6/7 supporting
sites in RC orientation** — a candidate cross-strand recognition
motif that would deserve manual inspection.

The other three banked candidates — CDS04337, CDS02973, CDS00113 —
have top mean_m in the 0.65–0.81 range, inside or just above the strict-null
upper tail (null1 p99 = 0.778, null2 p99 = 0.822). v3 does not confirm
them as cross-site target-recognizers at the 1 %-null-pass bar. This
is a sharper read than v1/v2 produced; it matches the biology that
not every bag with model signal is RNA-guided.

### Resolved in step 7: cross-L / cross-nc_pos cluster merging

Phase C deduplicates within the SAME `(nc_pos, L)` only. Step 7 adds a
greedy single-anchor merger
(`scripts/v5a_framework/cluster_merging.py`) that collapses clusters
representing the same physical motif viewed at different L or offset
by a few bp.

Merge rule: two clusters merge iff
- `|nc_pos_i - nc_pos_j| ≤ cfg.nc_offset` (default 5)
- `|flank_pos_median_i - flank_pos_median_j| ≤ cfg.flank_offset` (5)
- `Jaccard(supporting_sites_i, supporting_sites_j) ≥ cfg.support_jaccard` (0.80)

**Greedy single-anchor**, not union-find — the first draft used
union-find and produced one 210-cluster group spanning nc_pos 0–34,
flank_pos 13–50 because of transitive chain-linking across unrelated
motifs. Greedy anchors at the best (max mean_m, max L) and absorbs
only clusters within `±nc_offset` of the anchor, bounding group
diameter at `2·nc_offset`.

Representative = highest `mean_m_frac`, ties → largest `L`.
`MergedRow` carries `n_merged, nc_pos_range, L_range,
flank_pos_range, support_union`.

**Three unit tests** (`scripts/tests/test_cluster_merging.py`):

| # | case | outcome |
|---|---|---|
| 1 | 5 sliding-window variants (nc_pos 10–12, L 9–14, same support set, same flank_pos ±2) | 1 group; rep is L=14 (longest high-m) |
| 2 | 5 chain-adjacent clusters at nc_pos 0, 5, 10, 15, 20 (same support) | **3 groups** (diameter ≤ 2·offset verified) — proves chain-linking is bounded |
| 3 | 2 clusters with disjoint support sets at same nc_pos/flank_pos/L | 2 groups kept separate |

### Step-6 top-4 results re-expressed under step-7 merging

CDS03597's 185 publication-threshold-passing unmerged rows collapse to
**10 distinct merged clusters** above the 0.89 bar. Top rows:

```
rank  rep nc  L   fp   mean_m  support  bg  n_merged  guide
----  ------  --  ---  ------  -------  --  --------  ------------------
 1       7   14   39   1.000     4      4     33     AACCGCGCGCAGAG
 2      30   12   18   1.000     4      4     29     GAGTCCCTGTTT
 3      24   14   22   1.000     4      4     51     ACGATAGAGTCCCT
 4       1   14   45   1.000     4      4     37     TCGTCTAACCGCGC
 5      13   14   33   0.982     4      4     30     GCGCAGAGCCGACG
 6      19   13   28   0.981     4      4     10     AGCCGACGATAGA
 7      44   12    4   0.938     4      4      1     GCGCCCGGCTGC
 8      21    9   30   0.917     4      4      2     CCGACGATA
 9      14    9   37   0.917     4      4      3     CGCAGAGCC
10      12    9   39   0.917     4      4      5     CGCGCAGAG
```

Reading the pattern: nc_pos monotonically increases (1 → 44) while
flank_pos monotonically decreases (45 → 4). This is characteristic of
**one long contiguous RC pairing** between nc≈1–50 and flank≈0–50,
not ten independent motifs. The merger produces the right-granularity
report (per-subwindow evidence); interpreting them as one physical
pairing is a biological read, not a dedup task.

CDS04337 / CDS02973 / CDS00113: 209 / 24 / 50 merged clusters
respectively, **zero** above the 0.89 publication threshold.

### Step 7b — full 1173-bag sweep + Pfam annotation

Slurm job 26837369 completed 2026-10-11 00:34, elapsed **00:55:22**, exit 0.
Input: all 1173 CDS IDs in `all_candidates/` (intersection with
v5 adapted jsonl = 100%). Output:

- `candidates_v3.tsv` (194 MB, 1,157,555 rows unmerged)
- `candidates_v3_merged.tsv` (65 MB, 342,797 rows merged)
- `rejections_v3.tsv` (16 MB, 236,607 rows)

Of 342,797 merged rows:
- **1,304** cross the 0.89 publication threshold
- across **230 unique bags** (**19.6 %** of the input 1173)

Post-sweep annotation via `scripts/annotate_v3_top_hits.py`:
- Joined with `all_candidates/manifest.tsv` → species, dominant_orf_aa,
  tier, strict Pfam at E ≤ 1e-5
- Joined with v4 / v5 model `bag_max` scores
- Sensitive hmmscan at E ≤ 10 on the dominant ORF of each top-50 bag

### Top-50 by mean_m — 30 ANNOTATED, 20 NOVEL

Positive-control recovery (two Y1_Tnp hits — IS200/IS605 family, known RNA-guided):

| rank | bag_id | mean_m | sup/K | bg | strict Pfam | sensitive Pfam | species |
|---:|---|---:|---:|---:|---|---|---|
| 1  | **CDS01672** | 1.000 | 5/8 | 5 | — | **Y1_Tnp** (E=3e-5) | ecoli |
| 45 | CDS01048 | 1.000 | 4/8 | 3 | — | Y1_Tnp (E=7e-3, weak) | ecoli, kpneu |

Novel candidates (perfect mean_m, 4–5 local bg groups, no Pfam at either strict or meaningful sensitive):

| rank | bag_id | sup/K | bg | sens Pfam | species |
|---:|---|---:|---:|---|---|
|  5 | **CDS03548** | 5/8 | 5 | NO HIT  | hpylori |
|  8 | CDS06773 | 5/8 | 5 | DUF4508 (E=0.1, weak) | sa |
| 13 | CDS00990 | 7/8 | 4 | DUF4834 (E=3, noise) | ecoli |
| 15 | CDS01051 | 5/8 | 4 | NO HIT | ecoli |
| 17 | **CDS03597** | 4/8 | 4 | DUF1672 (E=3e-3, 17.6b) | kpneu |
| 19 | CDS03929 | 5/8 | 4 | NO HIT | kpneu |
| 20 | CDS03991 | 4/8 | 4 | NO HIT | kpneu |
| 21 | **CDS04073** | 7/8 | 4 | NO HIT | kpneu |
| 25 | CDS05802 | 5/8 | 4 | NO HIT | ngonorrhoeae |
| 29 | CDS08625 | 4/8 | 4 | NO HIT | spneumoniae |
| 33 | CDS02146 | 4/8 | 4 | NO HIT | ecoli |
| 34 | CDS01144 | 4/8 | 4 | NO HIT | ecoli |
| 36 | CDS05857 | 4/8 | 4 | NO HIT | ngonorrhoeae |
| 37 | CDS06377 | 5/8 | 4 | NO HIT | paeruginosa |
| 40 | CDS04209 | 4/8 | 4 | NO HIT | kpneu |
| 44 | CDS00986 | 6/8 | 3 | NO HIT | ecoli |

(Rows 22, 38, 48 are weaker novel; omitted for brevity. CDS03597,
previously the top-4 banked candidate, lands at rank 17 — one of
~15 equally-strong novels at perfect mean_m with 4+ bg groups.)

### Species distribution in top-50 novel

```
kpneu           7
ecoli           6
sa              2
ngonorrhoeae    2
hpylori, mtuberculosis, paeruginosa, spneumoniae, efaecalis   1 each
```

Spread across 9 species — the strict v3 signal is not a kpneu artifact.

### Known residual

- `rank_by_*` columns in `candidates_v3_merged.tsv` were empty on the
  sweep output (bug in `run_v3_scorer.py`: ranks were computed on the
  unmerged list only; fixed in commit after this run).
  `scripts/annotate_v3_top_hits.py` computes ranks locally from the
  raw columns so the top-50 ranking above is correct.

## Status

| step | phase | tests + run | status |
|---|---|---|---|
| 1 | A multi-hit retention | 1, 10, 11 | DONE |
| 2 | C cluster enumeration + partial D | 6, 9, 12 | DONE |
| 3 | B motif gates + E2E | primitives, 2, 3, 4, 5, E2E | DONE |
| 4 | D full + apply_phase_d_gate | 7, 8 permissive, 8 strict, mixed | DONE |
| 5 | G 4-null calibration + case 1-strict | 500-bag calibration + non-best cluster recovery through A→C | DONE |
| 6 | ranking + final output schema | `scripts/run_v3_scorer.py` + applied to top-4 | DONE |
| 7a | multi-L / multi-nc_pos merge + 3 unit tests + applied to top-4 | DONE |
| 7b | apply v3 + Pfam annotation to all 1173 `all_candidates/` bags (Slurm 26837369) | DONE |

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
    cluster_merging.py            # step 7 greedy single-anchor merger
scripts/
    run_v3_scorer.py              # step 6+7 output driver
    calibrate_v3_nulls.py         # Phase G driver
    analyze_v3_calib.py           # Phase G post-processor
    model_free_scorer.py          # v1 (unchanged, reference)
    model_free_scorer_v2.py       # v2 (unmerged partial fix, reference only)
scripts/tests/
    test_phase_a_multi_hit.py
    test_phase_b_motif_gates.py
    test_phase_c_clustering.py    # cases 1-strict, 6, 9, 12
    test_phase_d_background.py
    test_cluster_merging.py       # merge cases 1, 2, 3
sbatch/
    v3_calib_nulls.sbatch
    v3_sweep_all_candidates.sbatch
config/
    v3_sweep_cids.txt             # 1173 all_candidates CIDs
docs/
    model_free_scorer_v3.md       # this file
```

Run all tests:

```
for t in scripts/tests/test_phase_?_*.py; do
    python "$t" || break
done
```

Score the top-4 banked candidates:

```
python scripts/run_v3_scorer.py \
    --jsonl /scratch/.../fna_ins_discovery_v5_md_seed2/adapted.jsonl \
    --bag-ids CDS03597,CDS04337,CDS02973,CDS00113 \
    --outdir logs/v3_top4
```
