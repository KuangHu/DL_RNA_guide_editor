#!/bin/bash
#SBATCH --job-name=v84_sh
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/v84_sh_%A_%a.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

MODES=(none partial scattered flank_scattered no_alignment repeat_flank tsd_negative)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}

BATCH=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_generation/50k
SHARD_ROOT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_shards
mkdir -p "$SHARD_ROOT"

JSONL="$BATCH/v84_${MODE}.jsonl"
SHARD="$SHARD_ROOT/v84_${MODE}_shard"

if [ ! -f "$JSONL" ]; then
    echo "ERROR: JSONL not found: $JSONL"
    exit 1
fi

if [ -d "$SHARD" ] && [ -f "$SHARD/_index.json" ]; then
    echo "SKIP: shard already exists at $SHARD (delete to rebuild)"
    exit 0
fi

rm -rf "$SHARD"

echo "==== V8.4 shard: mode=$MODE  → $SHARD ===="
date
$PY -u -m scripts.build_v7_shard \
    --jsonl "$JSONL" \
    --shard-dir "$SHARD" \
    --cap-sites 8 \
    --family-label v84

echo
N_NPZ=$(find "$SHARD" -maxdepth 1 -name "*.npz" | wc -l)
SIZE=$(du -sh "$SHARD" | awk '{print $1}')
echo "==== V8.4 shard DONE: mode=$MODE  n_npz=$N_NPZ  size=$SIZE ===="
date
