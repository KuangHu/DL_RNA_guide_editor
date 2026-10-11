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

### 2. Decomposition boundary is highly uncertain
`loci_evidence.tsv`:
- `n_equivalent_decompositions` ∈ **{66, 108, 131, 155}**.
- `junction_ambiguity_bp` ∈ **{65, 107, 130, 154}**.
- `decomposition_confidence` ∈ **{0.008, 0.009, 0.006, 0.015}** —
  LOW at all 4 sites.
- `decomposition_quality` is 0.985 (site 4) and 0.988 (site 7) for
  the two `tolerant_alignment` sites; NA for the two `exact` sites
  (0, 6). High quality here means the allele-vs-allele alignment
  itself is clean (high host-background similarity around the
  insertion site) — it does NOT lower boundary uncertainty.

For sites 4 and 7 — same contig CP055119.1, same empty-locus coords
2599952–2600352, same pre-event genome GCA_022354725.1 — the empty
allele is identical, and the occupied alleles align **same direction**
at 99.236 % (2208 matches / 2225 bp). Chosen junctions sit 42 bp
apart (2600049 vs 2600091), inside each site's own 65–107 bp
ambiguity window. **Treat sites 4 and 7 as one background group**;
whether they are two independent insertions or two assembly-level
decompositions of one event is NOT settled by loci.tsv alone and
needs carrier-level evidence (strain phylogeny, assembly dating,
flanking-marker comparison).

### 3. "Insert edges" overlap host (empty) allele
`boundary_overlap.tsv` and `site_<i>/insert_vs_empty.blast`: for every
site, the extracted insert's first ~60-130 bp AND last ~130-180 bp
match the EMPTY allele at 92-98% identity.

Example — site 6:

- insert[1646:1824] (**last 179 bp**) → empty[71:249] at 98.3%
- insert[2:62]       (first 61 bp)   → empty[251:310] at 91.8%

Because the hits are present in the **empty** allele too, these are
real host-flanking sequences. Deciding between a composite /
replicative element vs. a boundary mis-call requires: aligning the
full insert against the EMPTY-ALLELE / seed-genome contig flanks of
all four sites, checking whether the ORFs' coding boundaries sit
comfortably inside the "trimmed" insert, and examining whether the
host-matching edges carry element-family signatures.

### 4. Occupied-frame measurement: 52 / 107 / 41 / 65 bp direct repeat
`occupied_core_stability.tsv` (from `scripts/boundary_stability.py`)
localises the two 14-mer cores in each site's occupied allele:

| site | repeat_type | core_a_pos | core_b_pos | insert | DIRECT-repeat total |
|---|---|---|---|---|---|
| 0 | direct | 221 | 2045 | [220..2044) | **52 bp**  (L=14, left=0,  right=38) |
| 4 | direct |  98 | 1922 | [97..1921)  | **107 bp** (L=14, left=1,  right=92) |
| 6 | direct | 250 | 2074 | [249..2073) | **41 bp**  (L=14, left=0,  right=27) |
| 7 | direct | 140 | 1965 | [139..1964) | **65 bp**  (L=14, left=1,  right=50) |

For every site: `guide_core` is found 2× on the occupied forward
strand (NOT `revcomp(guide_core)`). The chosen boundary places
core_a 1 bp inside the insert and core_b 1 bp outside — right
between the two direct-repeat copies.

**This is a direct repeat on the occupied forward strand, not an
inverted repeat / TIR.** The v1 pairing display (nc vs RC(flank))
made it look TIR-like by comparing a sequence to its reverse
complement; on genomic forward-strand terms, the two cores are the
same 14-mer in the same orientation, bracketing the insertion.

The lengths differ from the v1 nc-vs-RC(flank) measurements
(52, 55, 41, 23) and from the empty-target overlap lengths
(61–127 at 5′ edge + 138–179 at 3′ edge). Any future figure must
label which frame the number comes from (occupied / empty /
extracted window / pairings).

### 5. Impact on the v3 Phase C "shared guide" signal
Not fully decided here. Per reviewer inspection, sites 0 and 6
preserve the inside/outside assignment of the two cores at ALL best
boundary picks; sites 4 and 7 flip across boundary choices. So the
same-direction repeat at sites 0 and 6 is NOT a pure
boundary-decomposition artefact.

Most-reliable current description: a real same-direction (direct)
repeat spanning the insertion exists at all 4 sites, AND the
insertion boundary is highly uncertain (65–154 bp ambiguity window,
66–155 equivalent decompositions, decomposition_confidence
0.006–0.015 everywhere). The signal is worth continuing to
investigate; it is NOT yet classifiable as TIR, DNA-guided, or
extraction artefact.

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
