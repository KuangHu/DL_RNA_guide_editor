#!/bin/bash
#SBATCH --job-name=v8_50k_rf
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=1:00:00
#SBATCH --output=logs/v8_50k_rf_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

OUT_DIR=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_generation/50k
MODE=repeat_flank

echo "==== V8 50K repeat_flank gen ===="
date
$PY -u -m scripts.generator_v5.run_generator \
    --v7-real \
    --negative-mode "$MODE" \
    --n-bags 50000 \
    --seed 0 \
    --workers 16 \
    --progress-every 5000 \
    --out "$OUT_DIR/v8_${MODE}.jsonl" \
    --stats-out "$OUT_DIR/v8_${MODE}.stats.json" \
    --manifest-out "$OUT_DIR/v8_${MODE}.manifest.json"

N=$(wc -l < "$OUT_DIR/v8_${MODE}.jsonl")
echo "[v8_50k_rf] DONE  n_sites=$N"
