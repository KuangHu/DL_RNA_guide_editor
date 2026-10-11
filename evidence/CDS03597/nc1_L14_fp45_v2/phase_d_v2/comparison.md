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

## Per-pair repeat source
| si | sj | same_ortho | left_id | left_cov | right_id | right_cov | concat_id | concat_cov | repeat_source |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | False | 0.487 | 39 | 0.357 | 56 | 0.340 | 97 | left |
| 0 | 6 | False | 0.359 | 39 | 0.415 | 53 | 0.337 | 95 | right |
| 0 | 7 | False | 0.405 | 37 | 0.426 | 54 | 0.370 | 100 | right |
| 4 | 6 | False | 0.282 | 39 | 0.375 | 56 | 0.295 | 95 | right |
| 4 | 7 | True | 0.368 | 38 | 0.351 | 57 | 0.323 | 99 | left |
| 6 | 7 | False | 0.316 | 38 | 0.439 | 57 | 0.330 | 100 | right |
