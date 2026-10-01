#!/bin/bash
#SBATCH --job-name=v8_train3
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/v8_train3_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# V8 main training v3 (2026-09-26):
#   - 8 corpora (adds tsd_negative — the DDE-mimicking negative that
#     shares insert-body TIR signature across sites but has INDEPENDENT
#     per-site flanks, orthogonal to repeat_flank)
#   - SAME model architecture as v2 (Ls=9..14, N_CHANNELS=20 with
#     flank_bg_identity); no schema change
#   - SAME cache root as v2; SCHEMA_KEY unchanged so 7 modes' cache
#     entries are reused directly. Only v8_tsd_negative needs cold-warm
#     (handled by dependency prewarm sbatch v8_wm_v8_tsd_negative).
#   - v8_main_v2 killed at epoch 2 step ~1800 to make room; its
#     checkpoints preserved in v8_main_v2/ (epoch 0, epoch 1).
#   - Wall 48h same as v2 (defensive)
$PY -u -m scripts.channel_b_train \
    --run-name v8_main_v3 \
    --corpora-config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --split-mode main \
    --lr 3e-4 --wd 1e-2 \
    --epochs 8 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v8-train3] DONE"
