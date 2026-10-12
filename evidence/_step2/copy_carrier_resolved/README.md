> **Audit correction (2026-10-11):** this is the original cluster export,
> not an approved builder input. Two CDS04925 rows joined the wrong carriers;
> see `../copy_carrier_bundle_audit/join_issues.tsv`. There are 4,126 verified
> metadata joins and 2,814 verified slice joins. Neither count verifies nc
> extraction, insertion boundaries or event independence. Metadata MD5 equality
> compares inherited source IDs; only 176/2,816 actual intervals match the source
> sequence on either strand. All raw export rows are retained for audit.

# Copy-carrier evidence resolved from cluster — 2026-10-11

Resolves the 4,128 `.copyNNNN` sites that `scripts/build_v3_corpus_evidence.py`
quarantined with `copy_requires_actual_carrier_alleles_and_nc`. All carrier
coordinates come from the on-cluster `fna_ins_discovery/copysites/` tables
(not the HF v5 provenance), and all carrier contig slices come from
on-cluster assembly FASTAs. **No field is substituted from the source event.**

## Numbers

| field | value |
|---|---:|
| manifest sites (quarantined copies) | 4,128 |
| coords resolved from cluster copysites tables | **4,128 / 4,128 (100%)** |
| carrier contig slice extracted (`[elem_start - 200, elem_end + 200]`) | 2,816 |
| slice NOT extracted (carrier FASTA not on cluster, all ecoli) | 1,312 |
| `insert_md5` match: carrier copysites row vs source event inherited | 4,128 / 4,128 |
| spot check — slice reproduces source `insert_md5` when `pct_ident == 1.0 ∧ elem_len == inherited_len` | 176 / 177 (99.4%) |

The 2,639 slices where `pct_ident < 1.0` or `elem_len ≠ inherited_len` do
**not** reproduce the source `insert_md5` on purpose — those carriers hold
mutated or indel-containing copies of the insert, which is the exact
divergence the quarantine was designed to preserve.

The 1,312 unresolvable sites are all ecoli; this is the same cluster-side
assembly gap that Step 1 flagged as `seed_genome_missing` and that was
documented as off-cluster (not uploaded) in the HF dataset README. The
coords are still exposed in `copy_carrier_resolved.tsv`; downstream
reconstruction needs those assembly FASTAs re-downloaded from NCBI.

## Deliverables

| file | size | contents |
|---|---:|---|
| `copy_carrier_resolved.tsv` | 2.0 MB | 4,128 rows. Columns: `bag_id`, `site_id`, `source_event`, `species`, `carrier_assembly`, `contig`, `elem_start_0b`, `elem_end_0b_exclusive`, `elem_len_from_copysites`, `tsd_len`, `pct_ident`, `q_cov`, `orient`, `empty_site_source`, `flank_120`, `flank_fwd_120`, `copysites_insert_md5`, `inherited_insert_md5`, `md5_match`, `carrier_fasta_relpath`, `contig_len_bp`, `slice_lo_0b`, `slice_hi_0b_exclusive`, `slice_pad_bp`, `copysites_source_file`. |
| `copy_carrier_slices.fna` | 7.0 MB | 2,816 FASTA records. Header `>bag_id|site_id|carrier|contig|slice_lo|slice_hi|orient_from_copysites`. Each sequence is `carrier_contig[max(0, elem_start-200) : min(len, elem_end+200)]`, forward strand (no RC applied). |
| `copy_carrier_unresolved.tsv` | 114 KB | 1,312 rows + 0 missing-coord rows. All ecoli; `reason = carrier_fasta_missing`. |
| `summary.json` | <2 KB | Per-species counts, source-file provenance, builder integration note. |

## Resolver script

`scripts/resolve_copy_carriers.py` — reads the inventory manifest, iterates
every row of every candidate `copysites/*_sites.tsv` and `copysites/shards/*.s##_sites.tsv`
for the relevant species, matches on `db_id`, extracts carrier contig
slices from the Step-1 filesystem-wide assembly index (same rglob rule as
`scripts/step1_corpus_audit.py`). Nothing is defaulted or substituted.

## Schema caveat and corrected integration rules

Copysites `orient` encodes flank canonicalization, not element/ORF orientation.
Legacy missing values remain unknown; do not default them to biological +.

The earlier builder integration recommendation is withdrawn. Do not inherit
source-event `insert_start_in_long_bp`, decomposition quality/confidence,
`n_equivalent_decompositions`, length or sequence hash as copy-locus evidence.
An actual copy may contain substitutions and indels. Reconstructed empty flanks
must remain explicitly reconstructed; they are not observed empty alleles.

Use the audit's verified slice keys and retain quarantined/missing rows as
unknown. Derive actual nc fragments and orientation from actual carrier DNA;
validate copy-specific boundaries separately. The main builder still correctly
quarantines copies pending those steps.

The resolver now requires site ID, requested carrier, source event, source hash
and requested flank together, and refuses distinct ambiguous coordinates.
Rerun it on cluster to recover the two wrong joins; correct replacement
coordinates cannot be inferred from this exported bundle alone.
