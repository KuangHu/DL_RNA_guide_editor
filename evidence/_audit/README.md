# Step 1 — raw corpus audit

Per-site data-source availability for the v3 model-free scorer corpus.
Explicit `(bag_id, site_id)` join. Six independent axes per site; nothing
is collapsed into a single "resolved" column.

Scope reported twice:
- **1173-bag sweep scope** = `config/v3_sweep_cids.txt`
- **4380-bag upstream** = all distinct `transposase_id` in `adapted.jsonl`

## Retraction / correction history

The audit went through **four code bugs** that each shifted numbers
materially. All reported here so the record is honest; the current numbers
below are post-all-fixes.

1. **db_id parser** stripped only `.site`, not `.copyNNNN` — the 19131
   "fail:no_event_in_insert_cds" in the first draft was 100% a parser
   artifact. Fixed.
2. **Allele key split** was `split("|", 1)[1]` — kept the trailing
   `|len=…|n=…` suffix so `longest_allele = "A"` never matched the stored
   key `"A|len=2229|n=6"`. First draft reported `sequence_recovered` = 0 ok
   everywhere. Fixed.
3. **Flank-recovery check** tested the joined flank against the
   **occupied** allele — biologically impossible (the insert sits between
   the two halves). The flank must match the **empty** allele. First-draft
   message said "CDS03597 5/8 ready" based on the buggy pass; corrected
   audit shows **7/8 strict ready**. Fixed.
4. **Species→genome-dir hardcoded map** was wrong for most species
   (genomes live in `genomes_<species>/` not `genomes/<species>/`, and
   some species tags don't match dir names — `sa_k24` → `genomes_saureus`,
   not `genomes/staphaureus`). Replaced with a filesystem-wide
   `rglob("*.fna")` scan over every `genomes*/` dir (and nested dirs
   inside sibling species-subdirs). Fixed.

Separately, a labelling fix came from reviewer feedback: single-cutpoint
fallback (`junction_L_abs - start`) was initially relabelled as
`ok(fallback:...)`. It is NOT equivalent to the full `[lo, hi)`
equivalent-decomposition set the scorer needs, so it's now a distinct
tier `partial:single_cutpoint_only`, which is **not** counted as `ok`.

## Deliverables

| File | What |
|---|---|
| `per_site_audit.final.tsv` | 30891 rows; 6 axes per site. |
| `per_bag_summary.tsv` | 4380 rows; per-axis counts + `n_raw_source_ready_strict` and `n_raw_source_ready_with_partial_bndry`. |
| `pilot_candidate_bags_sweep_strict.txt` | 144 bag IDs in sweep scope with ≥4 STRICT-ready sites. |
| `pilot_candidate_bags_sweep_with_partial.txt` | 370 bag IDs (as above, counting partial boundaries too). |
| `pilot_candidate_bags_upstream_strict.txt` | 277 bags across the full 4380, ≥4 strict. |
| `pilot_candidate_bags_upstream_with_partial.txt` | 769 bags across the full 4380, ≥4 with partial. |
| `audit.log` | Run trace. |

## Join chain (verified end-to-end)

```
adapted.jsonl
  bag_id   = transposase_id
  site_id  = "<species>.<EXXXXX>.site"   OR
             "<species>.<EXXXXX>.copyNNNN"
  db_id    = site_id with (.site | .copyNNNN) stripped
     │  (bag_id, site_id)  — row order used only as a consistency check
     ▼
fna_ins_discovery/cds_v3/insert_cds.tsv            (cds_cluster_id → db_id)
     │  (bag_id, db_id)
     ▼
fna_ins_discovery/database_v3/insertions.tsv        (db_id → locus_id
                                                        = "<panel>/<genome_dir>|<short_locus>")
     │  locus_id
     ▼
<panel>/l2/<genome_dir>/loci.tsv                    (short_locus → longest_allele,
                                                        shortest_allele, junction_*,
                                                        pre_event_*, insert_start_in_long_bp, …)
     │  longest_allele, shortest_allele
     ▼
<panel>/l2/<genome_dir>/alleles.fna                 (header = "short_locus|allele_id|len=…|n=…")

Separately, seed-genome FASTA for pre_event sanity:
     fna_ins_discovery/{genomes,genomes_<species>,<any>/<genomes*>}/**/*.fna
     (indexed at audit startup; 27 579 assemblies on disk 2026-10-11)
```

## Axis definitions (kept separate)

| Axis | What is tested | Not tested |
|---|---|---|
| `joined` | `(bag_id, db_id)` in `insert_cds.tsv`, `db_id` in `insertions.tsv`, short locus in panel loci.tsv. | — |
| `raw_files_exist` | panel loci.tsv + alleles.fna + seed-genome FASTA (via filesystem-wide assembly index). Fail buckets: `loci_tsv_missing`, `alleles_fna_missing`, `seed_genome_missing`. | — |
| `sequence_recovered` | `flank` exact match in **empty** (1 copy, any strand) AND each `nc fragment` exact match in **occupied** (1 copy each). Ambiguity (>1 hit) reported separately. | Does NOT verify flank orientation/strand convention upstream used. Does NOT verify nc fragment order. |
| `boundary_set_complete` | **STRICT** = `insert_start_in_long_bp` is numeric AND junction_L_abs + junction_R_abs + n_equivalent_decompositions are present. **partial:single_cutpoint_only** = only junction_L_abs + start are numeric (the v2-builder fallback). | Does NOT enumerate the equivalent decomposition `[lo, hi)` intervals (that's a Step-2 responsibility). |
| `orf_msa_ready` | PENDING — Step 2 deliverable. | n/a |
| `target_coord_ready` | PENDING — Step 2 deliverable. | n/a |

A site passing all four tested axes means **the raw inputs are present and
self-consistent**, not that it will survive Step 2's stability / support /
background-grouping gates. "Raw-source-ready" is deliberate; it is not
"evidence-ready".

## Coverage — 1173-bag sweep scope (8697 sites)

| Axis | ok | fail breakdown |
|---|---:|---|
| joined | 8697 | — |
| raw_files_exist | 6309 | seed_genome_missing: 2388 |
| sequence_recovered | 3196 | raw_files_not_ok: 2388; flank_not_in_empty: 2873; ambiguous: 186; nc_frag_not_in_occupied: 54 |
| boundary_set_complete | 3843 strict | partial:single_cutpoint_only: 4854 |

### Raw-source-ready per-bag histogram — 1173 sweep

| threshold | strict | +partial |
|---:|---:|---:|
| ≥8 | 9 | 105 |
| ≥7 | 36 | 170 |
| ≥6 | 59 | 218 |
| ≥5 | 105 | 272 |
| ≥4 | **144** | **370** |
| ≥3 | 209 | 507 |
| ≥2 | 309 | 680 |
| ≥1 | 437 | 874 |

## Coverage — 4380-bag upstream (30891 sites)

| Axis | ok | fail breakdown |
|---|---:|---|
| joined | 30891 | — |
| raw_files_exist | 22713 | seed_genome_missing: 8178 |
| sequence_recovered | 8429 | raw_files_not_ok: 8178; flank_not_in_empty: 13772; ambiguous: 371; nc_frag_not_in_occupied: 161 |
| boundary_set_complete | 12599 strict | partial:single_cutpoint_only: 18292 |

### Raw-source-ready per-bag histogram — 4380 upstream

| threshold | strict | +partial |
|---:|---:|---:|
| ≥8 | 15 | 249 |
| ≥7 | 56 | 360 |
| ≥6 | 105 | 457 |
| ≥5 | 174 | 573 |
| ≥4 | **277** | **769** |
| ≥3 | 405 | 1074 |
| ≥2 | 642 | 1690 |
| ≥1 | 1286 | 3257 |

## Headline findings (post all four fixes)

1. **100% join**. Upstream pipeline retains ALL provenance — rebuild is NOT needed.
2. **Residual `seed_genome_missing` is 2388 sweep / 8178 upstream**. These are
   genuinely missing — assembly ID in loci.tsv's `seed_genome` field doesn't
   exist anywhere on the filesystem (not even in `__done_` archive dirs).
   The upstream fna_ins_discovery pipeline references assemblies no longer
   present locally. Could be rescued by re-downloading from NCBI; not a
   code-side issue.
3. **`flank_not_in_empty` is the next real residual** (2873 sweep / 13772
   upstream). The reviewer explicitly deferred fuzzy rescue; Step 2 needs
   to trace the deterministic flank-generation rule and apply the inverse
   transform.
4. **Realistic calibration universe is ~144 bags (sweep, strict) / 277
   (upstream, strict)**. With partial boundaries counted: 370 / 769. The
   partial tier should NOT be used until Step 2 proves the full
   equivalent-decomposition set can be reconstructed from a single cutpoint
   fallback.
5. **CDS03597 verified**: 7/8 strict ready (sites 0–6 all axes ok), site 7
   missing `insert_start_in_long_bp` → partial boundary.
6. **Pilot pool for Step 2** = `pilot_candidate_bags_sweep_strict.txt`.
   CDS00064, CDS00113, CDS00249, CDS00253, CDS02566, CDS02674, CDS02750,
   CDS02832, CDS02856, CDS02894 are the first 10 "all 8 sites ready"
   candidates. CDS03597 is in the list.

## What this audit does NOT establish

- Equivalent-decomposition `[lo, hi)` intervals — only the single
  insert_start_in_long_bp cutpoint. The partial tier flags where even that
  single cutpoint falls back to `junction_L_abs - start`.
- ORF orientation / homology grouping / context MSA — Step 2.
- Target coordinate projection from empty allele → raw flank origin — Step 2.
- Per-site support-fraction, shared-frame or background-group integrity —
  pipeline-level, Step 2/3.
- Why upstream `flank` sometimes diverges from `empty[flo:fhi]` — Step 2
  investigation of the flank-generation rule.
