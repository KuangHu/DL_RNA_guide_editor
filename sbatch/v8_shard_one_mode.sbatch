#!/bin/bash
#SBATCH --job-name=v8_sh
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=4:00:00
#SBATCH --output=logs/v8_sh_%x_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Mode passed via --export=MODE=<name>. Fail loud if unset.
: "${MODE:?MODE env var must be set (--export=MODE=<name>)}"

BATCH=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_generation/50k
SHARD_ROOT=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_shards
mkdir -p "$SHARD_ROOT"

JSONL="$BATCH/v8_${MODE}.jsonl"
SHARD="$SHARD_ROOT/v8_${MODE}_shard"

if [ ! -f "$JSONL" ]; then
    echo "ERROR: JSONL not found: $JSONL"
    exit 1
fi

if [ -d "$SHARD" ] && [ -f "$SHARD/_index.json" ]; then
    echo "SKIP: shard already exists at $SHARD (delete to rebuild)"
    exit 0
fi

# Fresh build: remove any partial dir
rm -rf "$SHARD"

echo "==== V8 shard: mode=$MODE  → $SHARD ===="
date
$PY -u -m scripts.build_v7_shard \
    --jsonl "$JSONL" \
    --shard-dir "$SHARD" \
    --cap-sites 8 \
    --family-label v8

echo
N_NPZ=$(find "$SHARD" -maxdepth 1 -name "*.npz" | wc -l)
SIZE=$(du -sh "$SHARD" | awk '{print $1}')
echo "==== V8 shard DONE: mode=$MODE  n_npz=$N_NPZ  size=$SIZE ===="
date
