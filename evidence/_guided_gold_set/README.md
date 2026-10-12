# Guided-mechanism gold set and annotated GenBank

This bundle turns the independently annotated 86-bag cohort into an explicit,
reviewable mechanism-family gold set and exports annotated GenBank for every
source insertion record in those bags.

## Contents

- `mechanism_gold_set.tsv`: one row per bag, including annotation strength,
  representative source event, reviewed protein reference, scan coverage,
  score, and separate `event_level_guide_target_verified` / `event_status`
  fields.
- `representative_family_mechanisms.gbk`: one annotated, full-length source
  insertion per bag for quick visual review.
- `all_member_events.gbk.gz`: all 7,462 source insertion records in the 86
  bags, GenBank annotated with CDSs, protein-family/profile hits, and every
  detected Rfam RNA model interval. Decompress before opening in a genome
  browser.
- `all_member_events.tsv`: index of the 7,462 GenBank records. It retains both
  the bag family and each insert's own member-level family label; mixed or
  unknown member annotations are not silently overwritten.
- `manifest.json`: counts and scope warning.

## Gold labels

| Label | Count | Interpretation |
|---|---:|---|
| `GUIDED_INSERTION_RNA` | 31 | 23 IS110 plus 8 group-II-intron bags; bridge-RNA recombination or EBS/IBS retrohoming is an established family mechanism. |
| `GUIDED_INSERTION_DNA` | 4 | IS200/IS605-TnpA bags; target choice uses transposon-encoded DNA guide/structured end recognition. |
| `GUIDED_NUCLEASE_NOT_INSERTION` | 5 | TnpB-like bags; RNA-guided cleavage is established, but that alone does not establish RNA-guided insertion. |
| `NON_GUIDED_IS_CONTROL` | 46 | Strict IS1/IS3/IS5 family controls. |

These are **family-mechanism labels**, grounded in independently supported
sequence-family annotation and known molecular mechanisms. They are useful
gold labels for asking whether a detector distinguishes mechanism classes.
They are not event-level biochemical labels. For the natural insertion record
in any one bag, this project has not yet verified the actual guide molecule,
its contact bases, and the observed genomic target/junction together. Thus the
event-level field currently reads `unverified_not_negative`; no weak scan score
turns it into a biological negative, and no high score turns it into a
validated positive.

The annotation source covers inserted-element sequences. It does not contain
the complete carrier chromosome, both occupied junctions, or empty allele for
all members. IS110 bridge-RNA loops are not yet independently called by a
structure-aware model. Group-II Rfam ncRNA intervals are included where found,
but EBS/IBS contacts are not assigned. TnpA target-contact geometry and
TnpB's role in the insertion process are not inferred from a family label.

## Primary mechanism references

- IS110 bridge RNA: [Durrant et al., Nature (2024)](https://www.nature.com/articles/s41586-024-07552-4).
- Group II intron EBS/IBS and retrohoming: [Group II Introns: Flexibility and Repurposing (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9294222/).
- IS200/IS605 TnpA target recognition: [Ronning et al. (2009)](https://pmc.ncbi.nlm.nih.gov/articles/PMC2680152/).
- TnpB RNA-guided cleavage: [Deltcheva et al., Nature (2021)](https://www.nature.com/articles/s41586-021-04058-1).

## Rebuild

From the repository worktree, with the full local annotation run available at
`../../output/family_annotation_all6139`:

```powershell
python scripts/build_mechanism_gold_set_gbk.py
```
