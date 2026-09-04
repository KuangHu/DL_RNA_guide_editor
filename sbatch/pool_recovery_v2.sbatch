#!/bin/bash
# Pool-recovery on the v2 candidate layer (nc-window-first + global E-value
# + dedup-by-same-peak). Compare against v1 (top-K per-(orient,L)) with the
# same downstream stratifier.

#SBATCH --job-name=pool_recov_v2
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=logs/pool_recov_v2_%x_%j.out
#SBATCH --error=logs/pool_recov_v2_%x_%j.err

set -euo pipefail

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/pool_recovery
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${V5_JSONL:?V5_JSONL env var required}"
: "${TAG:=v5}"
: "${WORKERS:=$(nproc)}"

ROWS=${OUTBASE}/rows_v2_${TAG}.jsonl
REPORT=${OUTBASE}/stratified_v2_${TAG}.txt

echo "[$(date -Iseconds)] v2 pool-recovery on $V5_JSONL"

cd "$CODEDIR"
$PYTHON -m scripts.preprocess_pool_recovery.pool_recovery_v2 \
    --v5-jsonl "$V5_JSONL" \
    --out "$ROWS" \
    --workers "$WORKERS"

$PYTHON -m scripts.preprocess_pool_recovery.stratify --rows "$ROWS" | tee "$REPORT"

echo "[$(date -Iseconds)] done"
