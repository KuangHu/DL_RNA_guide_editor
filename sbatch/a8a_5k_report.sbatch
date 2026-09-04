#!/bin/bash
#SBATCH --job-name=a8a_5k_rep
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=logs/a8a_5k_report_%j.out
#SBATCH --error=logs/a8a_5k_report_%j.err

set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen
CFGS="hom=1.0|${OUTBASE}/large_batch/positives_v5_v6_5k_hom10_incons.jsonl|${OUTBASE}/channel_a/mt_v6_5k_hom10_incons"
CFGS="${CFGS};hom=0.99|${OUTBASE}/large_batch/positives_v5_v6_5k_hom99_incons.jsonl|${OUTBASE}/channel_a/mt_v6_5k_hom99_incons"
CFGS="${CFGS};hom=0.95|${OUTBASE}/large_batch/positives_v5_v6_5k_hom95_incons.jsonl|${OUTBASE}/channel_a/mt_v6_5k_hom95_incons"
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u "$SCRATCH/a8a_5k_report.py" --configs "$CFGS"
