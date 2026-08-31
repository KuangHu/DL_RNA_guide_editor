#!/bin/bash
# Slurm array job for per-nt RNAplfold structure precompute.
#
# Submit with:
#   sbatch --export=SPLIT=val,NUM_SHARDS=4  --array=0-3%4  sbatch/precompute_structure_array.sbatch
#   sbatch --export=SPLIT=test,NUM_SHARDS=4 --array=0-3%4  sbatch/precompute_structure_array.sbatch
#   sbatch --export=SPLIT=train,NUM_SHARDS=24 --array=0-23%12 sbatch/precompute_structure_array.sbatch
#
# Each array task processes every NUM_SHARDS-th record of the split and
# writes its own memmap + index to <OUTBASE>/<SPLIT>_u16_shard<II>.{mmap,valid,index.json}.
# After all shards finish, merge with:
#   python -m scripts.merge_structure_shards \
#       --in-glob "<OUTBASE>/<SPLIT>_u16_shard*.index.json" \
#       --out    "<OUTBASE>/<SPLIT>_u16"

#SBATCH --job-name=precomp_str
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=6:00:00
#SBATCH --output=logs/precomp_%x_%A_%a.out
#SBATCH --error=logs/precomp_%x_%A_%a.err

set -euo pipefail
# ViennaRNA CLI (RNAplfold) is not a cluster module here; it lives in its
# own conda env so it cannot perturb the torch env.
export PATH=/global/home/users/kh36969/.conda/envs/vrna/bin:$PATH

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/structure
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${SPLIT:?SPLIT env var required (train|val|test)}"
: "${NUM_SHARDS:?NUM_SHARDS env var required}"
: "${TAG:=}"       # optional suffix in output filename, e.g. TAG=v2 -> val_v2_u16_*

SPLIT_PATH=/global/scratch/users/kh36969/DL_novel_guide_editor/splits/${SPLIT}.jsonl
if [[ ! -f "$SPLIT_PATH" ]]; then
    echo "split file not found: $SPLIT_PATH" >&2
    exit 1
fi

PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python

SHARD_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
if [[ -n "$TAG" ]]; then
    OUT_PATH=${OUTBASE}/${SPLIT}_${TAG}_u16_shard${SHARD_ID}
else
    OUT_PATH=${OUTBASE}/${SPLIT}_u16_shard${SHARD_ID}
fi

echo "[$(date -Iseconds)] shard $SHARD_ID of $NUM_SHARDS for split=$SPLIT"
echo "  split path: $SPLIT_PATH"
echo "  out base:   $OUT_PATH"
echo "  RNAplfold:  $(which RNAplfold)"
echo "  python:     $PYTHON"

cd "$CODEDIR"
$PYTHON -m scripts.precompute_structure \
    --split "$SPLIT_PATH" \
    --out "$OUT_PATH" \
    --u-max 16 --W 120 --L 60 \
    --batch-size 500 \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --shard-count "$NUM_SHARDS" \
    --progress-every 5000

echo "[$(date -Iseconds)] shard $SHARD_ID done"
