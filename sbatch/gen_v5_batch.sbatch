#!/bin/bash
# V5 generator batch — 5K validation or 50K production.
#
# The generator computes per-bag ViennaRNA folds (pf() + bpp() + pfl_fold_up)
# which is CPU-bound. cf1 gives exclusive nodes (~64 cores); we saturate
# with multiprocessing.Pool inside a single process. No Slurm array
# needed for N <= 50K (one node finishes in <1 h).
#
# Submit examples:
#   sbatch --export=N_BAGS=5000,SEED=0,TAG=5k  sbatch/gen_v5_batch.sbatch
#   sbatch --export=N_BAGS=50000,SEED=0,TAG=50k sbatch/gen_v5_batch.sbatch
#   # override worker count (default = nproc):
#   sbatch --export=N_BAGS=5000,SEED=0,TAG=5k,WORKERS=32 sbatch/gen_v5_batch.sbatch
#
# Time budget: measured 35 ms/bag/worker at 16 workers on 40-core node
# (near-linear). At 64 workers: 5K ~ 4 s launch + 8 s compute + I/O.
# 50K ~ 80 s compute. Add manifest overhead + safety = 20 min budget.

#SBATCH --job-name=gen_v5
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=logs/gen_v5_%x_%j.out
#SBATCH --error=logs/gen_v5_%x_%j.err

set -euo pipefail

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch
mkdir -p "$OUTBASE" "$CODEDIR/logs"

: "${N_BAGS:?N_BAGS env var required, e.g. 5000 or 50000}"
: "${SEED:=0}"
: "${TAG:=${N_BAGS}}"
: "${WORKERS:=$(nproc)}"

OUT_JSONL=${OUTBASE}/positives_v5_${TAG}.jsonl
OUT_STATS=${OUTBASE}/acceptance_${TAG}.json
OUT_MANIFEST=${OUTBASE}/manifest_${TAG}.json

echo "[$(date -Iseconds)] V5 generator batch"
echo "  host:      $(hostname)"
echo "  cores:     $(nproc)"
echo "  N_BAGS:    $N_BAGS"
echo "  SEED:      $SEED"
echo "  WORKERS:   $WORKERS"
echo "  out jsonl: $OUT_JSONL"
echo "  stats:     $OUT_STATS"
echo "  manifest:  $OUT_MANIFEST"
echo "  python:    $PYTHON"

: "${NEG_MODE:=none}"     # none | scattered | partial
: "${NC_HOMOLOGY_RATE:=}"  # empty = sample uniformly; a float pins the axis
: "${FLANK_OFFSET_MODE:=}"  # empty = sample uniformly; 'consistent' or 'inconsistent' pins
: "${N_SITES:=}"           # empty = sample uniformly [3, 8]; int pins (e.g., 5 for A8a compat)
: "${ACCESSIBILITY_TARGET:=}"  # empty = sample uniformly [0.4, 0.95]; float pins (e.g., 0.85 for pre-1e compat)
: "${GC_TARGET:=}"             # empty = sample uniformly [0.25, 0.65]; float pins (e.g., 0.5 for pre-1f compat)

HOM_ARG=()
if [[ -n "$NC_HOMOLOGY_RATE" ]]; then
    HOM_ARG=(--nc-homology-rate "$NC_HOMOLOGY_RATE")
    echo "  NC_HOMOLOGY_RATE (override): $NC_HOMOLOGY_RATE"
fi
FOM_ARG=()
if [[ -n "$FLANK_OFFSET_MODE" ]]; then
    FOM_ARG=(--flank-offset-mode "$FLANK_OFFSET_MODE")
    echo "  FLANK_OFFSET_MODE (override): $FLANK_OFFSET_MODE"
fi
NS_ARG=()
if [[ -n "$N_SITES" ]]; then
    NS_ARG=(--n-sites "$N_SITES")
    echo "  N_SITES (override): $N_SITES"
fi
AT_ARG=()
if [[ -n "$ACCESSIBILITY_TARGET" ]]; then
    AT_ARG=(--accessibility-target-percentile "$ACCESSIBILITY_TARGET")
    echo "  ACCESSIBILITY_TARGET (override): $ACCESSIBILITY_TARGET"
fi
GC_ARG=()
if [[ -n "$GC_TARGET" ]]; then
    GC_ARG=(--gc-target "$GC_TARGET")
    echo "  GC_TARGET (override): $GC_TARGET"
fi

cd "$CODEDIR"
$PYTHON -m scripts.generator_v5.run_generator \
    --n-bags "$N_BAGS" \
    --seed "$SEED" \
    --workers "$WORKERS" \
    --out "$OUT_JSONL" \
    --stats-out "$OUT_STATS" \
    --manifest-out "$OUT_MANIFEST" \
    --negative-mode "$NEG_MODE" \
    "${HOM_ARG[@]}" \
    "${FOM_ARG[@]}" \
    "${NS_ARG[@]}" \
    "${AT_ARG[@]}" \
    "${GC_ARG[@]}" \
    --progress-every 500

echo "[$(date -Iseconds)] done"
