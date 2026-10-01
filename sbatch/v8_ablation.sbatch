#!/bin/bash
#SBATCH --job-name=v8_abl
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/v8_abl_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

$PY -u -m scripts.channel_b_v8_channel_ablation \
    --ckpt checkpoints/channel_b/v8_main/best.pt \
    --config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --seed 0 --batch-size 32 --workers 4
echo "[v8_abl] DONE"
