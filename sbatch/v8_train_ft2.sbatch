#!/bin/bash
#SBATCH --job-name=v8_ft2
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/v8_ft2_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# ============================================================================
# V8 Channel B fine-tune (2026-09-25). Continue from v8_main/best.pt
# (epoch 5, val_loss=0.284, val_auroc_proxy=0.935) for 2 more epochs.
#
# --resume-from loads MODEL WEIGHTS ONLY; optimizer + schedule are fresh.
# Fine-tune LR chosen as 5e-5 (upper end of the LR range v8_main epochs
# 5-6 were seeing under the 8-epoch cosine to 3e-4). Cosine over 2
# epochs → 5e-5 → 0.
#
# Cache is fully warm from the prior run — expect ~40 min per epoch,
# ~80 min total.
#
# Separate run-name (v8_main_ft2) so we don't clobber v8_main's history:
# best.pt / epoch_0..6.pt / log.jsonl in v8_main/ are preserved.
# ============================================================================
$PY -u -m scripts.channel_b_train \
    --run-name v8_main_ft2 \
    --corpora-config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --resume-from checkpoints/channel_b/v8_main/best.pt \
    --split-mode main \
    --lr 5e-5 --wd 1e-2 \
    --epochs 2 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v8-ft2] DONE"
