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
: "${SPEC:=m8}"
REPORT=${OUTBASE}/channel_a_${TAG}_${SPEC}.json

echo "[$(date -Iseconds)] Channel A on V5"
echo "  host:      $(hostname)"
echo "  cores:     $(nproc)"
echo "  V5 JSONL:  $V5_JSONL"
echo "  shard dir: $SHARD_DIR"
echo "  report:    $REPORT"
echo "  python:    $PYTHON"

cd "$CODEDIR"
: "${WORKERS:=$(nproc)}"
: "${SPEC:=m8}"
EXTRA_ARGS=""
if [ "${NEGATIVES:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --treat-as-negatives"
fi
if [ "${GRID:-0}" = "1" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --grid"
fi
if [ -n "${TAU:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --tau $TAU"
fi
if [ -n "${S_VAL:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --S $S_VAL"
fi
if [ -n "${M_VAL:-}" ]; then
    EXTRA_ARGS="$EXTRA_ARGS --m $M_VAL"
fi
$PYTHON -m scripts.generator_v5.channel_a_v5 \
    --v5-jsonl "$V5_JSONL" \
    --shard-dir "$SHARD_DIR" \
    --report-out "$REPORT" \
    --workers "$WORKERS" \
    --spec "$SPEC" $EXTRA_ARGS

echo "[$(date -Iseconds)] done"
