#!/bin/bash
#SBATCH --job-name=chk_L3
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=logs/chk_L3_%j.out
#SBATCH --error=logs/chk_L3_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
: "${JSONL:?}"
: "${SHARD:?}"
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u -m scripts.generator_v5.generator_data_checker \
    --jsonl "$JSONL" --shard-dir "$SHARD" --layer 3 --workers 32
