#!/bin/bash
#SBATCH --job-name=v8_train
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/v8_train_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# ============================================================================
# V8 Channel B main training (2026-09-23).
#
# Corpora: 6 V8 modes (config/channel_b_v8_corpora.json) — 1 positive
# (none) + 5 negatives (partial, scattered, flank_scattered,
# unstructured_nc_full, no_alignment). All at 50K bags / 274,704 sites each.
#
# Model changes vs v7-real: Ls widened to (9..14), MAX_L=14, N_CHANNELS
# 15→19. New checkpoint architecture; NOT compatible with v7-real
# checkpoint. Same hyperparams as v7-real main so v7-real ↔ v8
# comparison is controlled at training-side.
#
# Cache-collision guard: --cache-root points at a FRESH v8-specific dir.
# channel_b_train.py:325-333 will print pre-existing .pt count per corpus
# and flag PRE-EXISTING if non-zero.
#
# Speed tripwire: epoch 0 MUST be visibly slower than epochs 1+ (loader
# recomputes ViennaRNA fold in epoch 0 and writes cache; subsequent
# epochs load fp32 tensors from disk, ~10× faster). If all epochs run at
# similar speed, cache is being consumed under wrong keys — STOP.
# ============================================================================
$PY -u -m scripts.channel_b_train \
    --run-name v8_main \
    --corpora-config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --split-mode main \
    --lr 3e-4 --wd 1e-2 \
    --epochs 8 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v8-train] DONE"
