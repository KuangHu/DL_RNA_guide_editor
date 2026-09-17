#!/bin/bash
#SBATCH --job-name=v7real_c3_fp64
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=logs/v7real_c3_fp64_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad

$PY -u "$SCRATCH/v7_c3_fp64.py" \
    --cfg config/channel_b_v7real_corpora.json \
    --ckpt checkpoints/channel_b/v7real_main/best.pt \
    --n-bags 200 --seed 0
echo "[v7real-c3-fp64] DONE"
