#!/bin/bash
#SBATCH --job-name=v8_wm
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=3:00:00
#SBATCH --output=logs/v8_wm_%x_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

: "${MODE:?MODE env var required (--export=MODE=v8_<name>)}"

CACHE_ROOT=/global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8
WORKERS=24   # cpus - 4 for headroom

echo "[v8_wm] mode=$MODE  start at $(date)"
$PY -u -m scripts.prewarm_one_mode \
    --mode "$MODE" \
    --config config/channel_b_v8_corpora.json \
    --cache-root "$CACHE_ROOT" \
    --workers "$WORKERS"
echo "[v8_wm] mode=$MODE DONE at $(date)"
