#!/bin/bash
#SBATCH --job-name=v84_wm
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --time=3:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/v84_wm_%A_%a.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

MODES=(v84_none v84_partial v84_scattered v84_flank_scattered
        v84_no_alignment v84_repeat_flank v84_tsd_negative)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

CACHE_ROOT=/global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v84
WORKERS=24   # cpus - 4 for headroom

echo "[v84_wm] mode=$MODE  start at $(date)"
$PY -u -m scripts.prewarm_one_mode \
    --mode "$MODE" \
    --config config/channel_b_v8_corpora.json \
    --cache-root "$CACHE_ROOT" \
    --workers "$WORKERS"
echo "[v84_wm] mode=$MODE DONE at $(date)"
