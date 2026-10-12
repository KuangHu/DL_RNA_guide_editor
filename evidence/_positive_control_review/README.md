# Positive-control status correction (2026-10-11)

## What the current 40 target-family bags establish

All 40 annotated target-family/role bags explicitly carry
mechanism_status=not_experimentally_verified. They are family comparators, not
experimentally validated guide-target pairing controls. The eleven earlier
reviewed protein checks validate role annotation, not guide or target detection.
The family AUCs remain descriptive associations, not detector sensitivity or
specificity against experimentally validated insertion-guidance ground truth.
No family annotation or numerical score was changed in this correction.

## CDS00730 actual context

All eight sampled source inserts have independently supported TnpA annotations.
The representative protein matched reviewed Salmonella IS200 P59696 at 94.079%
identity and full query coverage; the family annotation does not resolve
IS200-versus-IS605 subtype or establish DNA-guide contact positions.

The pictured ecoli.E027450.site selected core is only 6/9. Its target is empty
allele [239,248), while the complete best deletion cutpoints are 204,205,206:
the core is at least 33 bp from those cutpoints. Its nc core is source-insert
[120,129). This is a global-search selected approximate motif, not verified
junction recognition. It must not be displayed as a validated guide alignment.

The eight source ORF orientations are five + and three -, and raw fragment
lengths/fragment counts vary. ORF-canonical full inserts, actual occupied/empty
sequences and ends are exported for inspection. Seven source boundary sets can
be recomputed; one indel case remains unresolved. The displayed source interval
is the leftmost source-hash matching interval, not biological boundary truth.
The adjacent 4/5-mers are candidates only, not assigned target motifs.

## A real published mechanism falsifies the simplified alignment premise

The IS608 structural/functional study identifies left guide G_L=AAAG and target
C_T=TTAC (both written 5'->3'). These are not simple consecutively indexed
reverse-complement fourmers. Current index-wise identity is 1/4 on both target
orientations, below ceil(0.6*4)=3; L4 is not currently searched anyway.
The existing guide gate also rejects AAAG for its run and entropy, and the
existing target gate rejects TTAC for its run. Lowering L without changing
contact geometry and short-guide complexity treatment cannot solve this.

This is a published motif-level sanity case, NOT a completed end-to-end scorer
control: full element/target/context sequences and experimentally assigned
contact indices must be imported and verified before reporting recovery.

DNA guide in IS608 depends on subterminal imperfect-palindrome structure and
canonical/noncanonical ordered contacts, including additional hairpin-associated
bases. A gapped linear alignment alone should not be assumed sufficient.
The IS608 case is not asserted to be the mechanism/contact map of CDS00730.

Primary sources:
- Targeting IS608 transposon integration to highly specific sequences by
  structure-based transposon engineering (2018), Figure 1 / structural results:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC5934647/
- Reconstitution of a functional IS608 single-strand transpososome: role of
  non-canonical base pairing (2011):
  https://pmc.ncbi.nlm.nih.gov/articles/PMC3201880/

## Required control hierarchy before the next architecture claim

1. Experimental mechanism controls: authentic full published sequences and
   assigned guide/target contacts (IS608/ISDra2, IS621, a group-II system, TnpB).
   Guide/target-dead mutants or shuffled-contact controls where supplied.
2. Homologous mechanism candidates: verified scaffold/contact-region homology
   to those references, with element/target positions and uncertainty retained.
3. Family comparators: the current 40 annotated bags, which need independent
   validation before promotion to pairing positives.

TnpB nuclease guidance and RNA/DNA-guided insertion are different observables;
its positive assay must be specified rather than inferred from its family role.
Structural-contact mechanisms should be assessed separately from contiguous and
split Watson-Crick RNA pairing. A terminal hairpin should not be erased as TIR
without evidence of how the region participates in recognition.

Files: family_positive_status.tsv covers all 40 bags; CDS00730_actual_context.tsv
and .fna provide all eight actual contexts; IS608_published_motif_sanity.json
records the authentic motif-level counterexample and current gate responses.

Reproduction from repository root:

```
python scripts/audit_positive_control_status.py   --annotation evidence/_family_annotation/bag_family_annotations.tsv   --scan evidence/_family_signal/family_signal_by_bag.tsv   --outdir ../../output/positive_control_review   --inputs evidence/_family_signal/cohort_inputs.jsonl   --audit evidence/_audit/per_site_audit.final.tsv   --provenance-root ../../data/fna_ins_discovery_v5/provenance_unpacked   --annotation-root ../../output/family_annotation_all6139
```
