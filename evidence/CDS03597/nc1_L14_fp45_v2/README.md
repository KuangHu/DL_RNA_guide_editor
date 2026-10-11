# CDS03597 — real-genome evidence v2 (2026-10-11)

Supersedes `../nc1_L14_fp45/` (v1 TIR verdict RETRACTED).

- **cluster**: nc_pos=1, L=14, flank_pos=45
- **supporting sites**: 0, 4, 6, 7

## What this folder is

Raw genomic evidence pulled from `deep_kpneu/l2/` + `full_kpneu/l2/`,
with no interpretation layered on top. Each supporting site has its
own subfolder (`site_<i>/`) containing:

| file | what it is |
|---|---|
| `loci_evidence.tsv` | Every coord-level field from `loci.tsv` (contig, start, end, junction_L_abs, junction_R_abs, junction_ambiguity_bp, pre_event_genome, strand, inserted_len, inserted_md5, decomposition_method/quality/confidence, n_equivalent_decompositions, homologous_identity/coverage, insert_start_in_long_bp). |
| `occupied.fna` | The LONGEST allele of this locus — the "with-insert" allele. |
| `empty.fna` | The SHORTEST allele — the pre-insertion / empty host target. |
| `insert_from_allele.fna` | 1824 bp insert carved from the occupied allele at `insert_start_in_long_bp` (or `junction_L_abs - start` fallback for `full_kpneu`). |
| `insert.gbk` | Proper multi-CDS GenBank from Prodigal `-p meta`. **NO whole-insert fake CDS.** Includes locus/coord provenance in annotations. |
| `insert_s<i>.prodigal.{gff,faa}` | Raw Prodigal outputs (ORF coords + translated proteins). |
| `insert_vs_occupied.blast` | Blastn of insert against its occupied allele (short-word, no dust, max-hsps=20). The primary hit is the 100% self-placement; secondary hits expose boundary ambiguity. |
| `insert_vs_empty.blast` | Blastn of insert against the EMPTY allele — the critical check for decomposition-boundary host overlap. |
| `locus_region.fna` | ±1 kb of the locus in the seed genome contig (for coord sanity). |
| `md5_check.txt` | MD5(insert_from_allele) vs loci.tsv `inserted_md5` vs per-site JSONL `insert_md5`. |

Cross-site deliverables at this level:

| file | what it is |
|---|---|
| `summary.tsv` | one row per supporting site with the key coord/decomposition fields. |
| `_all_inserts.fna` | 4 inserts concatenated for pairwise alignment. |
| `inserts_pairwise.blast` | all-vs-all blastn of the 4 inserts. |
| `_all_orfs.faa` | all Prodigal-called proteins across sites, renamed `s<i>_orf<k>`. |
| `orfs_pairwise.blast` | all-vs-all blastp of the 4 sites' ORFs. |
| `boundary_overlap.tsv` | per-site edge hits (5p_edge / 3p_edge / middle) extracted from the insert_vs_empty.blast tables. |

## Factual findings (not verdicts)

### 1. The insert IS a mobile element with 3 conserved ORFs
Blastp all-vs-all (`orfs_pairwise.blast`) shows three ORF families, each
present once per insert at 97-100% identity across all 4 sites:

| ORF family | length | sites | min cross-site identity |
|---|---|---|---|
| A | 239 aa | all 4 | 97.1% |
| B | 161 aa | all 4 | 98.1% |
| C |  65 aa | 0, 4, 6 (not 7) | 100% |

So the insert IS a conserved ~1.8 kb mobile element. CDS03597 maps to
one of these ORFs. Pfam / RNA-guided classification is a separate
step not performed here.

### 2. Decomposition is ambiguous at 60-180 bp scale
`loci_evidence.tsv` for each site reports `n_equivalent_decompositions`
in **{66, 108, 131, 155}** and `junction_ambiguity_bp` in **{65, 107,
130, 154}**. The chosen "exact" or "tolerant_alignment" boundary is
ONE of 66-155 equally supported picks; the actual boundary could lie
anywhere in a 65-154 bp window.

For sites 4 and 7 — **same contig CP055119.1, same empty-locus coords
(2599952-2600352), same pre-event genome GCA_022354725.1** — the two
chosen junctions sit 42 bp apart (2600049 vs 2600091). This is well
inside each site's own ambiguity zone (65-107 bp). Sites 4 and 7
are plausibly **two decompositions of the same orthologous insertion**,
not two independent events.

### 3. "Insert edges" overlap host (empty) allele
`boundary_overlap.tsv` and `site_<i>/insert_vs_empty.blast`: for every
site, the extracted insert's first ~60-130 bp AND last ~130-180 bp
match the EMPTY allele at 92-98% identity.

Example — site 6:

- insert[1646:1824] (**last 179 bp**) → empty[71:249] at 98.3%
- insert[2:62]       (first 61 bp)   → empty[251:310] at 91.8%

Because the hits are present in the **empty** allele too, these are
real host-flanking sequences. Interpretations (not decided here):

- **Boundary mis-call**: the decomposition pipeline chose boundaries
  60-180 bp outside the TRUE element edge, pulling host sequence into
  the "insert". This is consistent with n_equivalent_decompositions in
  the hundreds and with pre_event_confidence being low (0.043 for
  site 4, 0.015 for site 7, 0.250 for site 0).
- **Composite / replicative element**: the mobile element genuinely
  carries ~100-180 bp of host-sequence-like repeat at each end.

Deciding between these requires: aligning the full insert against the
EMPTY-ALLELE / seed-genome contig flanks of all four sites, looking at
whether the ORFs' coding boundaries sit comfortably inside the "trimmed"
insert that would remain after removing the host-matching edges, and
examining whether the host-matching edges carry element-family
signatures (TIRs with mobile-element characteristics vs plain host DNA).

### 4. Impact on the v3 Phase C "shared guide" signal
The v3 cluster at `nc_pos=1, L=14, flank_pos=45` extracted guide
`nc[1:15] = TTTAGCGCAGGGAA` and matched it to `RC(flank[45:59])` at
all 4 sites. The nc comes from the **extracted** mobile-element
noncoding region (near CDS03597); the flank comes from the **extracted**
host context after the chosen insert boundary. The v1 pairings showed
this 14-mer match extends 40-55 bp through contiguous Watson-Crick
pairing on 3 of 4 sites.

Because the chosen boundary is in a host-matching zone (finding 3),
the FIRST ~100-180 bp of the extracted flank are not pure host — they
are the host-matching edge of the "insert" placed on the flank side of
the chosen boundary. If the TRUE boundary is 100+ bp inward, those bp
belong to the ELEMENT, not the host. In that case the 40-55 bp
"guide-target pairing" is intra-element inverted-repeat content
(classical TIR), not RNA-guided target recognition — but equally, if
the boundary is correct, the 40-55 bp stretch is a real cross-strand
match within the extracted window.

Nothing above decides between TIR / boundary-artefact / neither. That
is the explicit outcome of this evidence pull: it is not a decision.

## What is NOT in this folder (and is required to decide)

- Full six-frame or Prodigal annotation of the EMPTY allele to confirm
  whether the "host-matching edge" of the insert really is host CDS or
  a decoy / insertion-site repeat.
- Pfam/HMMER profile scan of the three conserved ORFs.
- Multiple alignment of the four inserts anchored on ORF coordinates
  (rather than on the ambiguous boundaries), to re-call the TRUE
  element edge and the TRUE guide context.
- Comparison with the top candidate's homologous_identity /
  homologous_coverage sister bags.

These are step-4 and step-5 of the agreed order; not in this commit.
