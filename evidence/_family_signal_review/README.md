# Human review of overlapping family signals (2026-10-11)

This is a selected-anchor diagnostic review, not a new classifier or accuracy
estimate. Selected each family's highest/lowest dinucleotide excess and highest
raw score: 17 distinct bags, 73 supporting alignments. Selection was automatic
from the completed 73-bag scan; all selection roles are disclosed.

## What was actually checked

- Re-read the existing nc/target core and +/-30 bp alignment, with the selected
  anchor fixed; no replacement best local alignment was chosen.
- Looked up actual occupied/empty event alleles. For copy sites, used actual
  carrier slices only and excluded the two quarantined joins. Never substituted
  parent event DNA for missing copy DNA. 23 selected support copies lack slices.
- Full-fragment unique mapping resolves the nc core for 47/73 supporters (44
  event sites and 3 copies); three other actual-copy fragments do not fully map.
- Recomputed complete optimal deletion boundaries when supported. For omitted
  stored cutpoints, display the leftmost actual-source-hash-matching cutpoint,
  explicitly a display convention. Copy aligner intervals and unsupported indel
  boundaries are not biologically verified insertion endpoints.
- Compared first/last 200 bp of each displayed insert interval for exact direct
  and inverted local matches >=10 bp, preserving actual coordinate/strand
  provenance. No exact match does not exclude imperfect or longer repeats.
- Compared all five scanned group-II bags' input fragments with independently
  annotated source Rfam RNA intervals. This is interval coverage, not EBS
  assignment or actual-copy RNA validation.

## Concrete findings

1. 55/73 best candidates are L=9; none is L=14. 12/299 best-candidate supporting
   cores are exact. Approximate pairing is not intrinsically wrong, but high
   mean match from many short windows is not specific evidence of a guide.
2. 55/73 bags have mixed raw orientations at the best candidate. Biological
   orientation is unresolved, so this is a diagnostic warning rather than a
   contradiction. Raw concat position equality is not conserved RNA/MSA position.
3. IS5 CDS00149 / abaum.E003897.copy25557: nc core overlaps 7 bp of an 11 bp exact
   inverted terminal-window match. Actual copy nc maps uniquely; endpoints are
   aligner intervals. IS1 CDS00712 / ecoli.E024252.site: core overlaps 3 bp of a
   13 bp inverted terminal-window match; there are 10 best deletion boundaries.
   Both demonstrate that terminal-repeat-related core sequence can enter the
   current candidates. Neither proves the whole bag signal is caused by TIR.
4. IS110 CDS07967: all four best-cluster supporters have exact 9/9 cores and
   10 bp contiguous matches. All cores remain inside every best boundary (three
   per site). No >=10 bp exact direct/inverted end-window match was detected.
   This deserves further scaffold/orientation/target validation; it is not a
   validated guide. The target spans the extraction midpoint, which must not be
   automatically masked as a supposed artefact.
5. CDS02002: all three input sets omit the independently detected source
   group-II D1-D4 RNA-model interval (154 bp). Both fragments map uniquely onto
   the source insert; this is a real input-coverage gap, not failed projection.
   The actual EBS positions remain unknown. CDS00871 includes its corresponding
   source-model interval, so truncation is not a universal explanation.
6. Current L=9..14 excludes known short DNA-guide anchors. Merely lowering L
   is insufficient: the current gate rejects TTAC / GTAA by homopolymer fraction
   and TTGAT / ATCAA by AT content plus period-4 based on one comparison.
   These are published IS608 / ISDra2 target examples and reverse complements,
   not asserted motifs of our bags. Gate behavior is in review_summary.json.

## TSD/TIR protection actually in the scanned path

The completed cohort used coordinate_mode=legacy. Phase B rejects poly-base,
short periodicity, AT fraction and low base entropy. Phase D groups masked local
backgrounds. Neither identifies TSD or TIR sequence origin. Dinucleotide shuffle
controls remove physical positional structure and thus cannot rule out TIR/TSD
sources; real donor controls also do not identify those sources.

The older flank_coherence.py has a short exact within-site TSD diagnostic and
variant.py an optional partition; tsd_handling defaults off. These are not called
by this Phase A-D scan. There is no dedicated TIR-source partition in its path.
Verified-coordinate mode checks unique fragments, stable boundaries and shared
MSA/ORF frames, which helps with boundary artefacts; it is not a comprehensive
TSD/TIR recognizer and was not enabled in this exploratory cohort scan.

A symmetric exterior direct-repeat diagnostic is included per displayed cutpoint;
it is NOT a full TSD detector. Deletion intervals can include one duplicated copy,
so zero exterior repeat is not absence of TSD. TSD assessment needs both occupied
junctions and the observed empty allele over all supported boundaries.

## Repair order justified by the review

1. Derive actual-copy nc and complete RNA/scaffold context. Stop first-two-
   fragments/250 bp truncation; keep missing carrier evidence unknown.
2. Use actual ORF/scaffold alignment and target origin to unify strand and
   position. Keep orthologous-locus and carrier grouping separate from labels.
3. Add continuous origin features: TSD-copy overlap, inverted/direct-repeat
   overlap, host-homology extent, boundary sensitivity and pairing inside/outside
   those regions. Keep unknowns distinct from zero overlap. Report all-source
   and repeat-independent evidence separately, without terminal blanket masking.
4. Add short DNA-guide and multi-segment RNA-anchor candidates with full context,
   spacing/orientation support and length-aware complexity/background models.
   Do not reuse long-window ratio gates for 4/5 bp anchors.
5. Score evidence across supporters, guide/scaffold positions, target context and
   family-appropriate pairing geometry, rather than only maximum mean identity.
   A TnpB role label is not proof that this bag's insertion process is RNA-guided.
6. Recalibrate both real and background through the same changed path; compare
   held-out bags/families and sequence-independent loci. No improvement claim
   should be based on this selected high/low case series.

## Viewing and reproducing

human_review.html contains 17 bags with 73 expandable sequence/track panels.
selected_alignment_architecture.png/.svg shows four actual examples. Track
scales are independent. Copy target flanks are reconstructed input, not observed
empty alleles. Red/green terminal matches are diagnostics, not assigned mechanisms.

Run from repo root (external local data/output paths shown):

```
python scripts/review_family_signal_context.py   --scan ../../output/family_signal_scan   --audit evidence/_audit/per_site_audit.final.tsv   --provenance-root ../../data/fna_ins_discovery_v5/provenance_unpacked   --copy-bundle evidence/_step2/copy_carrier_resolved   --annotation-root ../../output/family_annotation_all6139   --outdir ../../output/family_signal_human_review
```

Three diagnostic tests check inverted/direct coordinate reconstruction and exact
junction anchoring; these test diagnostic implementation, not biological TSD/TIR
classification accuracy.

Mechanism references:
- IS608 DNA guide: https://pubmed.ncbi.nlm.nih.gov/19524540/
- IS608/ISDra2 tetranucleotide/pentanucleotide targets:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC2989099/
- Group-II separated EBS1/EBS2: https://pmc.ncbi.nlm.nih.gov/articles/PMC8643678/
- IS110 bridge RNA: https://doi.org/10.1038/s41586-024-07552-4
- TnpB/IsrB inheritance mechanisms: https://pmc.ncbi.nlm.nih.gov/articles/PMC10055086/
