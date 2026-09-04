#!/bin/bash
#SBATCH --job-name=theta_scan
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=logs/theta_scan_%j.out
#SBATCH --error=logs/theta_scan_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen
JSONL=$OUTBASE/large_batch/positives_v5_v6_5k_hom10_1c_sampled.jsonl
SHARD=$OUTBASE/channel_a/mt_v6_5k_hom10_1c_sampled
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u "$SCRATCH/theta_scan.py" \
    --jsonl "$JSONL" --shard-dir "$SHARD" \
    --thetas 0.5,0.6,0.7,0.8,0.9,1.0 --workers 32 \
    --out "$SCRATCH/theta_scan_report.json"
