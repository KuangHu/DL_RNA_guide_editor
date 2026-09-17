#!/bin/bash
#SBATCH --job-name=v7real_shard_build
#SBATCH --partition=lr4
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --output=logs/v7real_shard_build_%x_%j.out
# Note (2026-09-13): moved from es1/A40 → lr3 (CPU). shard-build is
# CPU-only work (ViennaRNA fold + match-table numpy ops); GPU sat idle
# on the earlier 5-parallel run. Kept 8 CPUs; walltime unchanged.
set -euo pipefail    # -e added 2026-09-13 (same rationale as v7real_train_main.sbatch)
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

CORPUS=${CORPUS:?"set CORPUS via --export=CORPUS=<name>"}
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v7real_full
SHARD_ROOT=/global/scratch/users/kh36969/DL_novel_guide_editor/v7real_shards
mkdir -p "$SHARD_ROOT" logs

JSONL="$OUT/v7real_${CORPUS}.jsonl"
SHARD="$SHARD_ROOT/v7real_${CORPUS}_shard"
if [[ -d "$SHARD" ]]; then rm -rf "$SHARD"; fi

echo "[v7real-shard-build] corpus=$CORPUS  jsonl=$JSONL  shard=$SHARD"
$PY -u -m scripts.build_v7_shard \
    --jsonl "$JSONL" \
    --shard-dir "$SHARD" \
    --min-sites 1 \
    --family-label "v7real_${CORPUS}"
echo "[v7real-shard-build] $CORPUS DONE"
