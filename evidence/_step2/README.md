# Step 2 real-corpus run — 2026-10-11

All 1173 sweep bags / 8697 selected sites were checked. The builder emitted
1249 site evidence records, of which 1201 have unique nc fragment sources.
The scanner completed all bags: 8 `ready`, 1165 `needs_evidence`.
No calibrated publication threshold or positive call was produced.

Files:

- `scan_review.tsv`: all bags, original site denominator, evidence coverage,
  passing-cluster maxima and per-site issues. Complete bags sort first.
- `failure_summary.tsv`: site failure counts, including 4128 copy records
  quarantined because their nc was inherited from the parent source event.
- `builder_parameters.json`: exact inputs' hashes, method parameters and
  dataset revision. Source sequences and large candidate/context tables
  are retained in the local workspace, not included here.

Complete evidence bags and maximum Phase-D-passing mean match:

| bag | max mean match |
|---|---:|
| CDS02442 | 0.866667 |
| CDS03127 | 0.818182 |
| CDS03597 | 0.777778 |
| CDS02674 | 0.722222 |
| CDS03835 | 0.711111 |
| CDS02856 | 0.688889 |
| CDS05060 | 0.666667 |
| CDS00770 | 0.659091 |

These numbers are uncalibrated evidence statistics, not biological
confidence probabilities. Eight complete bags cannot satisfy the default
calibration sample-size requirements. Incomplete bags' maxima remain
diagnostic and must not be ranked as formally eligible discoveries.

The source-side audit's saved selected IDs were used to reconstruct the
legacy adapted input from HF v5. This preserves first-two-fragment /
250-bp truncation and ambiguity sanitization. Randomly resampling the raw
file does not reproduce the historical selected-site list.

The context MSA uses ORF-split global star alignment. It does not propagate
all alternative optimal gap placements into motif-column uncertainty.
Target coordinates refer to the ORF-oriented extraction midpoint, not a
resolved biological junction. Indel-aware boundary reconstruction remains
unsupported; equal-length deletion solutions must reproduce the upstream
equivalent-decomposition count and insert MD5.

The full scanner ran in 376 seconds. Following the copy-source correction,
only CDS04209 and CDS05682 had changed evidence and were rescored. Every
other bag's evidence was compared exactly and its output reused. Counts:
23660 candidate rows (including Phase-D failures), 15447 rejection rows,
3600 merged rows. No old numeric publication threshold was used.

Reproduction commands and input requirements are in
`docs/model_free_scorer_v3.md` and `scripts/build_v3_corpus_evidence.py`.
