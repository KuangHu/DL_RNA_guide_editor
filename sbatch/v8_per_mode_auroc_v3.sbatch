#!/bin/bash
#SBATCH --job-name=v3_pma
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/v3_pma_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

$PY -u -m scripts.channel_b_v8_per_mode_auroc \
    --ckpt checkpoints/channel_b/v8_main_v3/best.pt \
    --config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --seed 0 --batch-size 32 --workers 4
echo "[v3_pma] DONE"
