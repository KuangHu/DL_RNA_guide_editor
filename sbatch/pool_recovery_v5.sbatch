#!/bin/bash
# Pool-recovery measurement on V5 corpus: run the old 96-slot preprocessor
# per site, check whether gold planted position enters the pool, stratify
# by (L, planted_m, mm_concentration, is_split).
#
# Submit:
#   sbatch --export=V5_JSONL=/.../positives_v5_50k.jsonl,TAG=50k sbatch/pool_recovery_v5.sbatch

#SBATCH --job-name=pool_recov
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=logs/pool_recov_%x_%j.out
#SBATCH --error=logs/pool_recov_%x_%j.err

set -euo pipefail

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/pool_recovery
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${V5_JSONL:?V5_JSONL env var required}"
: "${TAG:=v5}"
: "${WORKERS:=$(nproc)}"

ROWS=${OUTBASE}/rows_${TAG}.jsonl
REPORT=${OUTBASE}/stratified_${TAG}.txt

echo "[$(date -Iseconds)] V5 pool-recovery on $V5_JSONL"
echo "  host:    $(hostname)"
echo "  cores:   $(nproc)"
echo "  workers: $WORKERS"
echo "  rows:    $ROWS"
echo "  report:  $REPORT"

cd "$CODEDIR"
$PYTHON -m scripts.preprocess_pool_recovery.pool_recovery \
    --v5-jsonl "$V5_JSONL" \
    --out "$ROWS" \
    --workers "$WORKERS"

$PYTHON -m scripts.preprocess_pool_recovery.stratify --rows "$ROWS" | tee "$REPORT"

echo "[$(date -Iseconds)] done"
