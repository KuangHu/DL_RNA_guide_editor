# Independently annotated family signal scan (2026-10-11)

Cohort: 40 supported guided-family/role bags + 46 clean IS1/IS3/IS5 controls,
selected by source-element annotation without looking at signal ranks.
73/86 have historical adapted inputs with K>=3: 35 target-family/role bags
and 38 controls. Thirteen K<3 raw bags remain unscored/unknown; no negative
score is assigned. n_sites_in_raw_bag and K used are separate coverage fields.

Foreground: 30 exact-input previous scans reused, 43 new bags scanned. The
foreground and both background conditions have identical config fingerprints.
All 73 foreground bags have a current Phase D candidate. The raw mode preserves
the old adapted first-two-fragment/250-bp and max-K=8 geometry.

## Backgrounds and continuous reporting

16 repeats per bag per condition, 2336 distinct run/condition/replicate rows:
- exact_dinuc: random Euler trails preserve exact dinucleotide counts, base
  counts, sequence length and endpoints of each flank. Identical flanks within
  a bag reuse an identical shuffle. Random trails are not uniform samples of
  all admissible sequences.
- unpaired_same_K: intact donor bags from the 4380-bag historical input pool,
  same K and a distinct recipient ID. Donor flank order is randomized while
  the recipient nc is retained. Donors can contain real signals; this is a
  pairing control, not a guaranteed signal-free null. GC is not separately
  matched to recipients under this condition.

The statistic is the same maximum mean match fraction among current Phase D
passing clusters in every condition. A background with no cluster has statistic
zero; this convention does not classify any real bag biologically.
Continuous outputs include foreground-minus-background mean, SD-based z where
SD is nonzero, and a finite Monte Carlo rank. No score cutoff is used.
Background perturbations can alter local background grouping; this is not a
fixed-group conditional test.

## Main descriptive results vs pooled strict controls

| Family/role | Scanned | Raw AUC | Exact-dinuc excess AUC | Unpaired excess AUC |
|---|---:|---:|---:|---:|
| IS110 | 22 | 0.773 | 0.694 | 0.579 |
| Group II intron | 5 | 0.716 | 0.621 | 0.679 |
| IS200/IS605-TnpA | 3 | 0.513 | 0.333 | 0.289 |
| TnpB-like | 5 | 0.429 | 0.426 | 0.474 |

AUC is the probability a random family-A bag has a larger continuous score than
family B (ties count half). It is not a trained classifier probability.
Pairwise AUCs, same-K pair-weighted AUC, K=8-only AUC, and equal-K-stratum macro
AUC are retained. Macro AUC can be dominated by tiny strata; it must not be
reported alone. Bootstrap intervals resample bags within this fixed labeled
subset, not independent insertion events; this is descriptive sampling
uncertainty, not a generalization claim. Small families have broad uncertainty.

Current results do not establish universal guided/non-guided separation.
Single-segment 9--14 bp, raw nc coordinates and a noncoding-only input can miss
short DNA guide segments, separate RNA recognition segments, or regions excluded
by extraction. Actual copy nc is still inherited for copy records. Weak score
under this architecture does not revise independently supported family labels.

## Integrity and the interrupted attempt

The first background attempt aborted with an unexpected tuple hit AttributeError
in Phase B. Source inspection did not establish a persistent hit cache as the
cause, and no core scorer fix was claimed. It was superseded by a complete
rerun with failure-sample capture and a separate full repeat. Both complete
runs produced identical fields for all 2336 keys, with no captured failure.
The root cause of the initial isolated failure remains unresolved. No partial
rows from the interrupted run are included. The repeat is validation, not an
additional 16 distinct random seeds per condition.

Exact shuffle invariants, reproducibility, and AUC direction/ties passed three
checks. The earlier seven family-annotation counterexample checks also pass.
No calibration/publication threshold is created.

## Review files

- family_signal_by_bag.tsv: all 86 cohort rows, coverage, match/entropy/position/
  backgrounds/extension features, two background-adjusted scores, copy fraction,
  distinct nc sequences, and source status. Missing foreground scores stay blank.
- family_best_alignments.tsv: per-supporter sequence context at each scored bag's
  strongest candidate; inherited-copy nc is explicitly marked.
- family_distributions.tsv: coverage and feature distributions by family.
- pairwise_auc.tsv: continuous ranking comparisons and uncertainty.
- family_signal_overview.png/.svg: distributions and ranking AUC matrix.
- background_scores.tsv / background_manifest.json: all background run results.
- background_reproducibility.json: full two-run identity check.
- comparison_summary.json: numeric summary.

Full new foreground candidate tables are under foreground_new/. Reused source
results are under output/signal_exploratory1173_full/; the bag ID lists record
which output supplies each foreground result.
