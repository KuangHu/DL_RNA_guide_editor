#!/bin/bash
#SBATCH --job-name=v7real_c4_heldout
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/v7real_c4_heldout_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# C4 gate for v7-real: model_heldout — train on bags with n_sites ∈ {3..6},
# val on {7,8}. Same hyperparams as v7real_main, DEDICATED fresh cache root
# (no reuse of channel_b_cache_v7real to prevent tensor-collision).
# 24h wall (fold epoch 0 is ~7h given train slice is smaller than main).
$PY -u -m scripts.channel_b_train \
    --run-name v7real_heldout_nsites \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real_heldout_nsites \
    --split-mode n_sites_heldout \
    --lr 3e-4 --wd 1e-2 \
    --epochs 8 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v7real-c4-train] DONE"
