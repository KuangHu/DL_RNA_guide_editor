# CDS03597 — cluster evidence package

- cluster: nc_pos=1  L=14  flank_pos=45
- supporting sites: [0, 4, 6, 7]

## Files

- `provenance.tsv` — per-site db_id, locus, md5, carriers
- `cluster.tsv` — cluster metadata
- `nc_context.fna` / `nc_msa.fna` — per-site nc ±context + MAFFT MSA
- `target_context.fna` / `target_msa.fna` — per-site target ±context + MAFFT MSA
- `inserts/site_<i>_insert.{fna,gbk}` — full 1824-bp inserted element (annotated with dominant ORF)
- `targets/site_<i>_target.{fna,gbk}` — full 400-bp pre-insertion target (insertion point annotated)
- `pairings/site_<i>_pairing.txt` — nc 5′→3′ aligned to RC(flank) ±20 nt around the window, per-base match line

## Read the evidence

1. **Provenance** — check that supporting sites are in distinct genomes with distinct insert_md5. If they share md5 or carriers, suspect clonal duplication.
2. **Target MSA** — do the supporting sites' targets align to the same orthologous host locus? If yes, same insertion site in different genomes.
3. **nc MSA** — do the supporting sites' nc contexts align? If yes, same ncRNA family. CRITICAL: check whether raw `nc_pos` maps to the SAME alignment column across sites. If not, the Phase-C cluster at that raw index is not at a shared scaffold position.
4. **Pairings** — the nc 5′→3′ vs RC(flank) match lines show whether complementarity extends beyond the 14-mer. A long contiguous match suggests either real extended guide-target base pairing or an extraction/sequence overlap.
5. **GBK files** — load inserts/*.gbk and targets/*.gbk in a genome browser (Artemis, SnapGene, Benchling) to see dominant ORF and insertion point in context.
