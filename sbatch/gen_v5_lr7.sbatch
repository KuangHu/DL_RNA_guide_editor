#!/bin/bash
#SBATCH --job-name=gen_v5
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=56
#SBATCH --time=1:00:00
#SBATCH --output=logs/gen_v5_%x_%j.out
#SBATCH --error=logs/gen_v5_%x_%j.err

set -euo pipefail
CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${N_BAGS:?N_BAGS env var required}"
: "${SEED:=0}"
: "${TAG:=${N_BAGS}}"
: "${WORKERS:=56}"

OUT_JSONL=${OUTBASE}/positives_v5_${TAG}.jsonl
OUT_STATS=${OUTBASE}/acceptance_${TAG}.json
OUT_MANIFEST=${OUTBASE}/manifest_${TAG}.json

: "${NEG_MODE:=none}"
: "${NC_HOMOLOGY_RATE:=}"
: "${FLANK_OFFSET_MODE:=}"

HOM_ARG=()
if [[ -n "$NC_HOMOLOGY_RATE" ]]; then HOM_ARG=(--nc-homology-rate "$NC_HOMOLOGY_RATE"); fi
FOM_ARG=()
if [[ -n "$FLANK_OFFSET_MODE" ]]; then FOM_ARG=(--flank-offset-mode "$FLANK_OFFSET_MODE"); fi

cd "$CODEDIR"
$PYTHON -m scripts.generator_v5.run_generator \
    --n-bags "$N_BAGS" --seed "$SEED" --workers "$WORKERS" \
    --out "$OUT_JSONL" --stats-out "$OUT_STATS" --manifest-out "$OUT_MANIFEST" \
    --negative-mode "$NEG_MODE" "${HOM_ARG[@]}" "${FOM_ARG[@]}" \
    --progress-every 500
