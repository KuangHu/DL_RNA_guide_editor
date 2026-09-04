#!/bin/bash
#SBATCH --job-name=a12_5k
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=logs/a12_5k_%j.out
#SBATCH --error=logs/a12_5k_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u "$SCRATCH/a12_paired.py" \
    --pos "$OUTBASE/large_batch/positives_v5_v6_stage2_pos50k.jsonl" \
    --twin "$OUTBASE/large_batch/positives_v5_v6_stage2_twin50k.jsonl" \
    --out "$SCRATCH/a12_report_5k.json" \
    --workers 32 --max-bags 5000
