#!/bin/bash
#SBATCH --job-name=chk_1c
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:10:00
#SBATCH --output=logs/chk_1c_%j.out
#SBATCH --error=logs/chk_1c_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad
JSONL="$SCRATCH/v6_1c_smoke.jsonl"
SHARD="$SCRATCH/mt_1c_smoke"
rm -rf "$SHARD"
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u -m scripts.generator_v5.generator_data_checker \
    --jsonl "$JSONL" --shard-dir "$SHARD" --layer 2,3 --workers 16 --max-bags 200
