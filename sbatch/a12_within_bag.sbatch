#!/bin/bash
#SBATCH --job-name=a12_within
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=1:30:00
#SBATCH --output=logs/a12_within_%j.out
#SBATCH --error=logs/a12_within_%j.err
set -uo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad
BATCH=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch

# Full 50K bags — n_pairs → SE_sign ≈ 0.00224 (2 SE ≈ 0.0045);
# so a 1pp shift from 0.5 is detectable, and 0.5 within 2SE is the acceptance
$PY $SCRATCH/a12_within_bag.py \
    --pos $BATCH/positives_v5_v6_stage2_pos50k.jsonl \
    --twin $BATCH/positives_v5_v6_stage2_twin50k.jsonl \
    --out $SCRATCH/a12_within_report.json \
    --workers 32 --max-bags 50000
