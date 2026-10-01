#!/bin/bash
#SBATCH --job-name=v8_rf_smk
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=logs/v8_rf_smk_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

OUT_DIR=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_generation/smoke
mkdir -p "$OUT_DIR"

# Generate 1K bags of repeat_flank (new mode 4). Positive smoke already
# exists from earlier V8 smokes; keep as reference for axis parity.
MODE=repeat_flank
OUT="$OUT_DIR/v8_smoke_${MODE}.jsonl"
STATS="$OUT_DIR/v8_smoke_${MODE}.stats.json"

echo "==== V8 repeat_flank gen: 1000 bags → $OUT ===="
$PY -u -m scripts.generator_v5.run_generator \
    --v7-real \
    --negative-mode "$MODE" \
    --n-bags 1000 \
    --seed 0 \
    --workers 8 \
    --progress-every 200 \
    --out "$OUT" \
    --stats-out "$STATS"

echo
# If v8_smoke_none.jsonl doesn't exist, regenerate that too
if [ ! -f "$OUT_DIR/v8_smoke_none.jsonl" ]; then
    echo "==== regenerating v8_smoke_none for axis-parity reference ===="
    $PY -u -m scripts.generator_v5.run_generator \
        --v7-real --negative-mode none --n-bags 1000 --seed 0 --workers 8 \
        --progress-every 200 \
        --out "$OUT_DIR/v8_smoke_none.jsonl" \
        --stats-out "$OUT_DIR/v8_smoke_none.stats.json"
fi

echo
echo "==== analysis ===="
$PY -u -m scripts.audit_v8_repeat_flank_smoke --smoke-dir "$OUT_DIR"
echo "[v8_rf_smk] DONE"
