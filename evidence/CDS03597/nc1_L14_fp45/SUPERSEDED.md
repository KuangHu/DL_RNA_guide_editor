# This folder has been superseded (2026-10-11)

The v1 evidence folder for CDS03597 is kept for diff/history, but its
`inserts/site_*.gbk` falsely annotated the whole 1824 bp insert as a
single CDS, and the extended-pairing diagrams were presented as TIR
confirmation. The TIR verdict (commit `6744793`) is **retracted** — see
R5 in `docs/model_free_scorer_v3.md`.

Use `evidence/CDS03597/nc1_L14_fp45_v2/` instead:

- real-genome coord-level fields per site (`loci_evidence.tsv`)
- occupied / empty allele FASTAs
- insert carved from allele with provenance
- proper multi-CDS Prodigal GBK (3 real ORFs per insert, 97-100% id
  cross-site)
- blastn insert vs occupied + insert vs empty, exposing 61-180 bp of
  decomposition-boundary host overlap
- pairwise ORF homology confirms a conserved 3-gene mobile element
