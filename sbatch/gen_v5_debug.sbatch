#!/bin/bash
#SBATCH --job-name=gen_v5
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=logs/gen_v5_%x_%j.out
#SBATCH --error=logs/gen_v5_%x_%j.err
set -euo pipefail
CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch
mkdir -p "$OUTBASE" "$CODEDIR/logs"
: "${N_BAGS:?}"
: "${SEED:=0}"
: "${TAG:=${N_BAGS}}"
: "${WORKERS:=32}"
OUT_JSONL=${OUTBASE}/positives_v5_${TAG}.jsonl
OUT_STATS=${OUTBASE}/acceptance_${TAG}.json
OUT_MANIFEST=${OUTBASE}/manifest_${TAG}.json
: "${NEG_MODE:=none}"
: "${NC_HOMOLOGY_RATE:=}"
: "${FLANK_OFFSET_MODE:=}"
: "${N_SITES:=}"
: "${ACCESSIBILITY_TARGET:=}"
: "${GC_TARGET:=}"
HOM_ARG=()
if [[ -n "$NC_HOMOLOGY_RATE" ]]; then HOM_ARG=(--nc-homology-rate "$NC_HOMOLOGY_RATE"); fi
FOM_ARG=()
if [[ -n "$FLANK_OFFSET_MODE" ]]; then FOM_ARG=(--flank-offset-mode "$FLANK_OFFSET_MODE"); fi
NS_ARG=()
if [[ -n "$N_SITES" ]]; then NS_ARG=(--n-sites "$N_SITES"); fi
AT_ARG=()
if [[ -n "$ACCESSIBILITY_TARGET" ]]; then AT_ARG=(--accessibility-target-percentile "$ACCESSIBILITY_TARGET"); fi
GC_ARG=()
if [[ -n "$GC_TARGET" ]]; then GC_ARG=(--gc-target "$GC_TARGET"); fi
cd "$CODEDIR"
$PYTHON -m scripts.generator_v5.run_generator \
    --n-bags "$N_BAGS" --seed "$SEED" --workers "$WORKERS" \
    --out "$OUT_JSONL" --stats-out "$OUT_STATS" --manifest-out "$OUT_MANIFEST" \
    --negative-mode "$NEG_MODE" "${HOM_ARG[@]}" "${FOM_ARG[@]}" "${NS_ARG[@]}" "${AT_ARG[@]}" "${GC_ARG[@]}" \
    --progress-every 500
