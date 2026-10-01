#!/bin/bash
#SBATCH --job-name=v84_50k
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/v84_50k_%A_%a.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

MODES=(none partial scattered flank_scattered no_alignment repeat_flank tsd_negative)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

OUT_DIR=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_generation/50k
mkdir -p "$OUT_DIR"

echo "==== V8.4 50K $MODE gen (array task $SLURM_ARRAY_TASK_ID) ===="
date
$PY -u -m scripts.generator_v5.run_generator \
    --v7-real \
    --negative-mode "$MODE" \
    --n-bags 50000 \
    --seed 0 \
    --workers 16 \
    --progress-every 5000 \
    --out         "$OUT_DIR/v84_${MODE}.jsonl" \
    --stats-out   "$OUT_DIR/v84_${MODE}.stats.json" \
    --manifest-out "$OUT_DIR/v84_${MODE}.manifest.json"

N=$(wc -l < "$OUT_DIR/v84_${MODE}.jsonl")
echo "[v84_50k_$MODE] DONE  n_records=$N"
date
