# Phase D v2 (ortho-aware) vs current Phase D for CDS03597

## Setup
- cluster: nc_pos=1 L=14 flank_pos_median=45.0  n_support=4
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

## Per-pair side selection (R3-fix: coverage-first, then identity)
| si | sj | same_ortho | left (id/paired_bp/indels) | right (id/paired_bp/indels) | chosen | chosen id | chosen paired_bp |
|---|---|---|---|---|---|---|---|
| 0 | 4 | False | 0.622/37/10 | 0.604/53/10 | left | 0.622 | 37 |
| 0 | 6 | False | 0.462/39/6 | 0.620/50/16 | right | 0.620 | 50 |
| 0 | 7 | False | 0.676/37/10 | 0.608/51/14 | left | 0.676 | 37 |
| 4 | 6 | False | 0.528/36/12 | 0.547/53/10 | right | 0.547 | 53 |
| 4 | 7 | True | 0.611/36/12 | 0.491/55/6 | left | 0.611 | 36 |
| 6 | 7 | False | 0.556/36/12 | 0.571/56/4 | right | 0.571 | 56 |
