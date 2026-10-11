# CDS03597 — real-genome evidence v2

- cluster: nc_pos=1  L=14  flank_pos=45
- supporting sites: [0, 4, 6, 7]

## Per-site deliverables (in each site_<i>/ folder)
- `loci_evidence.tsv` — all coord-level fields from loci.tsv
- `occupied.fna`, `empty.fna` — alleles containing / lacking the insert
- `insert_from_allele.fna` — insert carved from occupied allele
- `insert.gbk` — Prodigal (meta) multi-CDS GBK, with locus/coord
  provenance in annotations/comment. **NO whole-insert fake CDS.**
- `insert_vs_occupied.blast` / `insert_vs_empty.blast` — blastn tables
- `locus_region.fna` — ±1 kb of locus in the seed genome contig
- `md5_check.txt` — md5(insert_from_allele) == loci_tsv.inserted_md5 ?
- `insert_s<i>.prodigal.{gff,faa}` — raw Prodigal outputs

## Cross-site deliverables (at parent dir)
- `inserts_pairwise.blast` — all-vs-all blastn across supporting sites
- `summary.tsv` — one row per supporting site with key fields

## Scope
This v2 package replaces the v1 inflated evidence folder (nc1_L14_fp45) whose `insert.gbk` incorrectly labeled the entire 1824 bp as one CDS and whose pairing diagrams were presented as TIR confirmation. v2 presents raw genomic evidence only — no TIR verdict. Interpretation is for the reviewer.
