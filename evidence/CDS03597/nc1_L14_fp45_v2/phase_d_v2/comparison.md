# Phase D v2 (ortho-aware) vs current Phase D for CDS03597

## Setup
- cluster: nc_pos=1 L=14 flank_pos_median=45  n_support=4
- margin=3  max_offset=5  clonal_threshold=0.85

## Current Phase D output
- n_local_bg_groups = 4
- groups: [0], [4], [6], [7]
- passes_bg_gate = True
- rejection_reason = None

## P2c Phase D v2 output
- ortho_locus_groups (from loci.tsv): [[0], [4, 7], [6]]
- n_local_bg_groups_v2 (ortho + seq merge): 3
- groups: [[0], [4, 7], [6]]

## Per-pair repeat source (gapped identity, R3)
| si | sj | same_ortho | left_id | left_cov | left_indels | right_id | right_cov | right_indels | repeat_source | src_id | src_cov |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | False | 0.489 | 47 | 10 | 0.508 | 63 | 10 | right | 0.508 | 63 |
| 0 | 6 | False | 0.400 | 45 | 6 | 0.470 | 66 | 16 | right | 0.470 | 66 |
| 0 | 7 | False | 0.532 | 47 | 10 | 0.477 | 65 | 14 | left | 0.532 | 47 |
| 4 | 6 | False | 0.396 | 48 | 12 | 0.444 | 63 | 10 | right | 0.444 | 63 |
| 4 | 7 | True | 0.458 | 48 | 12 | 0.443 | 61 | 6 | left | 0.458 | 48 |
| 6 | 7 | False | 0.404 | 47 | 10 | 0.533 | 60 | 4 | right | 0.533 | 60 |
