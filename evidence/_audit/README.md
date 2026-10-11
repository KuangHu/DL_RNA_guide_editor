# Step 1 — raw corpus audit

Per-site data-source availability for the v3 model-free scorer corpus.
Explicit `(bag_id, site_id)` join. Six independent axes per site; nothing
is collapsed into a single "resolved" column.

Scope reported twice:
- **1173-bag sweep scope** = `config/v3_sweep_cids.txt`
- **4380-bag upstream** = all distinct `transposase_id` in `adapted.jsonl`

## Retraction / correction history

The first-pass audit had **three bugs** that materially moved every number:

1. **db_id parser** stripped only `.site`, not `.copyNNNN` — the 19131
   "fail:no_event_in_insert_cds" reported in the first draft was 100% a
   parser artifact. All 30891 sites actually resolve.
2. **Allele key split** was `split("|", 1)[1]` — kept the trailing
   `|len=…|n=…` suffix on the allele key so `longest_allele = "A"` never
   matched the stored key `"A|len=2229|n=6"`. First draft reported
   `sequence_recovered` = 0 ok everywhere. Fixed to `split("|")[1]`.
3. **Flank-recovery check** tested the joined flank against the
   **occupied** allele. That's biologically impossible — the insert sits
   between the two halves. The flank must match the **empty** allele.
   First-draft draft message said "CDS03597 5/8 ready" based on the buggy
   pass; corrected audit shows **7/8 strict ready**.

A fourth labelling fix came from reviewer feedback: single-cutpoint fallback
(`junction_L_abs - start`) was initially relabelled as `ok(fallback:...)`.
It is NOT equivalent to the full `[lo, hi)` equivalent-decomposition set
the scorer needs, so it's now a distinct tier
`partial:single_cutpoint_only`, which is **not** counted as `ok`.

## Deliverables

| File | What |
|---|---|
| `per_site_audit.final.tsv` | 30891 rows; 6 axes per site. |
| `per_bag_summary.tsv` | 4380 rows; per-axis counts + `n_raw_source_ready_strict` and `n_raw_source_ready_with_partial_bndry`. |
| `pilot_candidate_bags_sweep_strict.txt` | 70 bag IDs in sweep scope with ≥4 STRICT-ready sites. |
| `pilot_candidate_bags_sweep_with_partial.txt` | 128 bag IDs (as above, counting partial boundaries too). |
| `pilot_candidate_bags_upstream_strict.txt` | 148 bags across the full 4380, ≥4 strict. |
| `pilot_candidate_bags_upstream_with_partial.txt` | 302 bags across the full 4380, ≥4 with partial. |
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
```

## Axis definitions (kept separate)

| Axis | What is tested | Not tested |
|---|---|---|
| `joined` | `(bag_id, db_id)` in `insert_cds.tsv`, `db_id` in `insertions.tsv`, short locus in panel loci.tsv. | — |
| `raw_files_exist` | panel loci.tsv + alleles.fna + seed-genome FASTA (species-dir map). Separate fail buckets: `loci_tsv_missing`, `alleles_fna_missing`, `seed_genome_missing`. | — |
| `sequence_recovered` | `flank` exact match in **empty** (1 copy, any strand) AND each `nc fragment` exact match in **occupied** (1 copy each). Ambiguity (>1 hit) reported separately. | Does NOT verify the flank orientation/strand convention upstream used. Does NOT verify nc fragment order. |
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
| raw_files_exist | 2631 | seed_genome_missing: 6066 |
| sequence_recovered | 1261 | raw_files_not_ok: 6066; flank_not_in_empty: 1289; ambiguous: 77; nc_frag_not_in_occupied: 4 |
| boundary_set_complete | 3843 strict | partial:single_cutpoint_only: 4854 |

### Raw-source-ready per-bag histogram — 1173 sweep

| threshold | strict | +partial |
|---:|---:|---:|
| ≥8 | 3 | 34 |
| ≥7 | 13 | 54 |
| ≥6 | 24 | 70 |
| ≥5 | 49 | 89 |
| ≥4 | **70** | **128** |
| ≥3 | 114 | 203 |
| ≥2 | 186 | 291 |
| ≥1 | 277 | 392 |

## Coverage — 4380-bag upstream (30891 sites)

| Axis | ok | fail breakdown |
|---|---:|---|
| joined | 30891 | — |
| raw_files_exist | 9111 | seed_genome_missing: 21780 |
| sequence_recovered | 3375 | raw_files_not_ok: 21780; flank_not_in_empty: 5589; ambiguous: 138; nc_frag_not_in_occupied: 8 |
| boundary_set_complete | 12599 strict | partial:single_cutpoint_only: 18292 |

### Raw-source-ready per-bag histogram — 4380 upstream

| threshold | strict | +partial |
|---:|---:|---:|
| ≥8 | 4 | 92 |
| ≥7 | 19 | 126 |
| ≥6 | 42 | 161 |
| ≥5 | 81 | 214 |
| ≥4 | **148** | **302** |
| ≥3 | 237 | 437 |
| ≥2 | 407 | 685 |
| ≥1 | 869 | 1358 |

## Headline findings (post-correction)

1. **100% join**. All 30891 sites resolve through `insert_cds.tsv →
   insertions.tsv → panel loci.tsv`. The earlier "needs upstream rebuild"
   conclusion was wrong — upstream provenance IS retained.
2. **Dominant failure is `seed_genome_missing`** (70.5% of upstream, 69.7%
   of sweep). The locus's panel data is present but our species-genome dir
   map doesn't have a FASTA for that assembly. Root cause not yet
   investigated per-species — a running probe is open (background task
   `bcd1h5cie`); whether this is a path-mapping error or genuinely missing
   FASTAs is still TBD.
3. **Secondary failure is `flank_not_in_empty`** (~15% of sites that pass
   raw_files_exist). CDS03597 is NOT in this bucket under the corrected
   check. For sites that ARE, the next question is whether upstream has a
   deterministic flank-generation rule (reverse-complement / N-pad / trim)
   that would make the flank exactly recoverable with a known transform —
   NOT fuzzy matching. Deferred to Step 2 per reviewer directive.
4. **Realistic calibration universe is ~70 bags (sweep scope, strict
   boundary) / ~148 bags (upstream)**. If partial boundaries are later
   proven sufficient for Step 2's reconstruction, the pool grows to
   128/302.
5. **CDS03597 verified**: 7/8 strict ready (all axes `ok` on sites 0–6);
   site 7 only misses `insert_start_in_long_bp` → partial boundary. Earlier
   "5/8 ready" message was from the buggy pass and has been retracted here.
6. **Pilot pool for Step 2/3** = `pilot_candidate_bags_sweep_strict.txt`
   (strict first) with fallback to `..._with_partial.txt`. CDS03597 is in
   both lists; CDS00113 (reviewer-named control) is also in both.

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
