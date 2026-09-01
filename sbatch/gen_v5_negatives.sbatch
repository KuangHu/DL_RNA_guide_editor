#!/bin/bash
# V5 negative-bag generator.
# Submit:
#   sbatch --export=N_BAGS=10000,SEED=100,TAG=neg10k sbatch/gen_v5_negatives.sbatch

#SBATCH --job-name=gen_v5_neg
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:00:00
#SBATCH --output=logs/gen_v5_%x_%j.out
#SBATCH --error=logs/gen_v5_%x_%j.err

set -euo pipefail
CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${N_BAGS:?N_BAGS required}"
: "${SEED:=100}"
: "${TAG:=neg${N_BAGS}}"
: "${WORKERS:=$(nproc)}"

OUT_JSONL=${OUTBASE}/negatives_v5_${TAG}.jsonl
OUT_MANIFEST=${OUTBASE}/manifest_${TAG}.json

echo "[$(date -Iseconds)] V5 negative-bag generator"
echo "  host:   $(hostname)   cores: $(nproc)"
echo "  N_BAGS: $N_BAGS   SEED: $SEED   WORKERS: $WORKERS"
echo "  out:    $OUT_JSONL"

cd "$CODEDIR"
$PYTHON -m scripts.generator_v5.run_negatives \
    --n-bags "$N_BAGS" --seed "$SEED" --workers "$WORKERS" \
    --out "$OUT_JSONL" --manifest-out "$OUT_MANIFEST" \
    --progress-every 500

echo "[$(date -Iseconds)] done"
