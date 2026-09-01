#!/bin/bash
# Run Channel A on V5 generator JSONL, stratified by architecture axes.
#
# Submit:
#   sbatch --export=V5_JSONL=/.../positives_v5_50k.jsonl,TAG=50k \
#          sbatch/channel_a_v5.sbatch

#SBATCH --job-name=chA_v5
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=logs/chA_v5_%x_%j.out
#SBATCH --error=logs/chA_v5_%x_%j.err

set -euo pipefail

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${V5_JSONL:?V5_JSONL env var required}"
: "${TAG:=v5}"

SHARD_DIR=${OUTBASE}/mt_${TAG}
REPORT=${OUTBASE}/channel_a_${TAG}.json

echo "[$(date -Iseconds)] Channel A on V5"
echo "  host:      $(hostname)"
echo "  cores:     $(nproc)"
echo "  V5 JSONL:  $V5_JSONL"
echo "  shard dir: $SHARD_DIR"
echo "  report:    $REPORT"
echo "  python:    $PYTHON"

cd "$CODEDIR"
: "${WORKERS:=$(nproc)}"
$PYTHON -m scripts.generator_v5.channel_a_v5 \
    --v5-jsonl "$V5_JSONL" \
    --shard-dir "$SHARD_DIR" \
    --report-out "$REPORT" \
    --workers "$WORKERS"

echo "[$(date -Iseconds)] done"
