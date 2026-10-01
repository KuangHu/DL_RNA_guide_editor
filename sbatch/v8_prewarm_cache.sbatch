#!/bin/bash
#SBATCH --job-name=v8_warm
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=3:00:00
#SBATCH --output=logs/v8_warm_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Cache dir MUST match the training job's --cache-root exactly, or
# training's DataLoader won't see the warmed .pt files.
CACHE_ROOT=/global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8

# workers = cpus - 4 (headroom for OS + main proc)
WORKERS=52

echo "[v8_warm] start at $(date)"
$PY -u -m scripts.prewarm_channel_b_cache \
    --config config/channel_b_v8_corpora.json \
    --cache-root "$CACHE_ROOT" \
    --workers "$WORKERS"

echo "[v8_warm] cache dir summary:"
for D in "$CACHE_ROOT"/*/; do
    N=$(find "$D" -name "*.pt" 2>/dev/null | wc -l)
    printf "  %-32s  n_pt=%s\n" "$(basename $D)" "$N"
done

echo "[v8_warm] DONE at $(date)"
