#!/bin/bash
# Shortcut feature extraction — Slurm array (one shard per array task).
#
# Submit e.g.:
#   sbatch --export=SPLIT=val,NUM_SHARDS=4 --array=0-3%4 sbatch/extract_shortcut_array.sbatch
#   sbatch --export=SPLIT=test,NUM_SHARDS=4 --array=0-3%4 sbatch/extract_shortcut_array.sbatch
#   sbatch --export=SPLIT=train,NUM_SHARDS=20 --array=0-19%12 sbatch/extract_shortcut_array.sbatch
#
# Per-shard writes <OUTBASE>/<SPLIT>_shortcut_shard<II>.npz

#SBATCH --job-name=shortcut
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/shortcut_%x_%A_%a.out
#SBATCH --error=logs/shortcut_%x_%A_%a.err

set -euo pipefail

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
DATA=/global/scratch/users/kh36969/DL_novel_guide_editor
OUTBASE=${CODEDIR}/shortcut_features
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${SPLIT:?SPLIT env var required (train|val|test)}"
: "${NUM_SHARDS:?NUM_SHARDS env var required}"
: "${MAX_TNPS:=0}"      # 0 = no cap
# Optional overrides for V2 or custom paths (fall back to <SPLIT>-derived defaults):
: "${SPLIT_JSONL:=$DATA/splits/${SPLIT}.jsonl}"
: "${STRUCTURE_INDEX:=$DATA/structure/${SPLIT}_u16.index.json}"

SHARD_ID=$(printf '%02d' "$SLURM_ARRAY_TASK_ID")
OUT_PATH=${OUTBASE}/${SPLIT}_shortcut_shard${SHARD_ID}.npz

echo "[$(date -Iseconds)] shard $SHARD_ID / $NUM_SHARDS for split=$SPLIT"

EXTRA=""
if [[ "$MAX_TNPS" != "0" ]]; then
  EXTRA="--max-tnps $MAX_TNPS"
fi

cd "$CODEDIR"
/global/home/users/kh36969/.conda/envs/opfi/bin/python -m scripts.extract_shortcut_features \
    --split-jsonl "$SPLIT_JSONL" \
    --structure-index "$STRUCTURE_INDEX" \
    --out "$OUT_PATH" \
    --shard-index "$SLURM_ARRAY_TASK_ID" \
    --shard-count "$NUM_SHARDS" \
    $EXTRA

echo "[$(date -Iseconds)] shard $SHARD_ID done"
