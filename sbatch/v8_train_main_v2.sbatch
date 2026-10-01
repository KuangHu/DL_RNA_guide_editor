#!/bin/bash
#SBATCH --job-name=v8_train2
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/v8_train2_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# V8 main training v2 (2026-09-25):
#   - 7 corpora (adds repeat_flank)
#   - N_CHANNELS = 20 (adds flank_bg_identity)
#   - SCHEMA_KEY-tagged cache (fresh; independent of prior 19-ch cache)
#   - Fresh run-name v8_main_v2 so v8_main/ history is preserved
#   - Wall 48h (learned from v8_main's 24h wall running epoch 0 in ~22h;
#     pre-warm cuts that but keep margin for repeat_flank cold compute)
$PY -u -m scripts.channel_b_train \
    --run-name v8_main_v2 \
    --corpora-config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --split-mode main \
    --lr 3e-4 --wd 1e-2 \
    --epochs 8 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v8-train2] DONE"
