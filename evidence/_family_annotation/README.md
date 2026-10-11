# Independent source-element family annotation (2026-10-11)

All 6139 bags were annotated without reading signal scores, using 98604 source
insertion records. Every source DNA passed length and MD5 verification. The
upstream database_v3/insertions.tsv actually contains full insert_seq; the HF
README claim that insert sequences are absent does not apply to these source
events. Actual carrier-copy sequences/coordinates remain a separate missing
source and are not validated by this annotation.

83015 distinct insert sequences yielded 153508 predicted ORFs and 42993 distinct
proteins (Pyrodigal 3.7.1). All were searched against 14 Pfam models using curated
GA cutoffs and 266 multi-member ISEScan profiles, including competing families.
The singleton ISEScan file is preserved in the model snapshot but was not used
in phmmer. Eight Rfam group-II covariance models were searched over ALL source
inserts on both strands (Infernal 1.1.5, Rfam GA). Reviewed UniProt protein
homology was searched for profile-positive proteins (BLAST+ 2.16.0).

## Conservative supported labels

| Dominant family/role | Bags |
|---|---:|
| IS110 | 23 |
| Group II intron: RT/maturase + same-strand downstream D5 | 8 |
| IS200/IS605 TnpA | 4 |
| TnpB-like | 5 |
| IS1 | 5 |
| IS3 | 53 |
| IS5 | 11 |

Of these controls, only IS1=3, IS3=35, IS5=8 meet the stricter clean-control
criteria (46 total). Every source member must have the control label, and any
possible guided-family Pfam GA hit, even a partial domain, or group-II RNA
hit excludes the bag from this clean subset. CDS08259 was additionally excluded
by this partial-domain check.

6030 bags do not receive one of the supported target labels. This includes
other families, partial genes, insufficient profile/domain support, or mixed
source contexts. They are UNKNOWN for this panel, not nonguided negatives.
144 potentially relevant unresolved bags have a manual review queue.

## Accuracy safeguards and limits

- Dominant ORF selection uses longest nucleotide span, matching upstream
  111_cds_cluster.py. For all 35733 source events in labeled bags, translated
  dominant length differs from the upstream value only by zero or one terminal
  stop character; no larger differences were seen.
- IS110 requires ordered DEDD and Transposase_20 domains plus a competitive
  IS110-family profile. An element's numerical name is never its family label.
- IS200/IS605 TnpA requires Y1 and a short family-specific profile, with RAYT
  reference competition retained. IS200 versus IS605 subtype remains unresolved.
- TnpB-like requires ordered HTH_OrfB and OrfB domains. It is a role-family
  annotation, not a functional RNA-guidance assay; distant Cas12/IscB relatives
  may require more specific reference work.
- Group II requires RT plus specific maturase on one complete ORF and a Rfam
  D5 hit downstream on the same strand. D1-D4 support is retained independently.
  Generic RT alone is insufficient. Full-length LtrA reference checks justified
  a 60% coverage floor for RVT_1/Domain_X, rather than the usual 70%; GA and RNA
  requirements remain.
- IS1/3/5 require specific family-profile competition and corroborating Pfam
  domains, a complete dominant ORF, >=90% unique-source-member consistency, and
  no competing supported family label. Coencoded roles remain visible.
- Seven counterexample tests passed. Eleven reviewed reference proteins were
  checked: nine expected roles were supported, two were left unresolved. These
  checks include shared/possibly training reference sequences and cannot
  estimate real precision/recall. No numerical accuracy claim is made.

## Files

- bag_family_annotations.tsv: all 6139 rows, family, candidate, source-member
  coverage, competing profiles, coencoded roles, and clean-control eligibility.
- supported_target_and_control_bags.tsv: 40 guided-family/role supported bags
  plus 46 strict family controls, for later selection/review.
- representative_annotation_evidence.tsv: 109 labeled bags, deterministic
  median-length source representative, protein/Pfam/ISEScan/UniProt evidence,
  and RNA coordinates. This is representative evidence; all member coverage
  remains in the full annotation tables.
- protein_role_annotations.tsv: all 42993 protein rows, including no-hit unknowns.
- profile_hits.tsv and rna_hits.tsv: every reported raw model hit.
- source_events.tsv, gene_origins.tsv, insert_family_annotations.tsv: exact
  source joins, coordinates within insert sequences, and member-level labels.
- manual_review_queue.tsv: potentially relevant unresolved bags.
- reference_validation/: reference sanity checks, with synthetic coordinates
  only for exercising the reporter; these are not biological insert alleles.

The earlier 1173-bag scan contains only 3 clean controls (2 IS3, 1 IS5), alongside
18 IS110, 3 group-II, 2 TnpA and 4 TnpB-like bags. It is a selected cohort and
must not be used as a representative full-family discrimination benchmark.

## Reproduction

Model URLs and decoded SHA256 are pinned in config/family_annotation_models.json.
Run scripts/fetch_family_annotation_models.py to verify/recover the snapshot;
a changed current API model fails its hash check rather than silently upgrading.
Runtime: Biopython, Pyrodigal, PyHMMER 0.12.3, BLAST+, and Infernal 1.1.5.

1. build_family_annotation_inputs.py --provenance-root PROVENANCE --bags-tsv
   insertions_bags.tsv --outdir OUTPUT
2. scan_family_models.py --models MODELS --inputs OUTPUT --cpus 8
3. scan_groupII_rna.py --models MODELS --inputs OUTPUT --cmsearch cmsearch
   (on Windows, the local run used --wsl-distro Ubuntu).
4. Build the reviewed UniProt FASTA BLAST database, then blast profile-positive
   proteins using E<=1e-10 and the documented 12-column alignment format.
5. classify_bag_families.py --inputs OUTPUT --models MODELS --bags-tsv
   insertions_bags.tsv

Primary model sources:
https://github.com/xiezhq/ISEScan
https://www.ebi.ac.uk/interpro/api/
https://rfam.org/family/RF00029
https://rest.uniprot.org/
