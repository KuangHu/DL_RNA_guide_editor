#!/bin/bash
#SBATCH --job-name=v8_gen1k
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/v8_gen1k_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

OUT_DIR=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_generation/smoke
mkdir -p "$OUT_DIR"

MODES=(none partial scattered flank_scattered unstructured_nc_full no_alignment)
N_BAGS=1000

for MODE in "${MODES[@]}"; do
    OUT="$OUT_DIR/v8_smoke_${MODE}.jsonl"
    STATS="$OUT_DIR/v8_smoke_${MODE}.stats.json"
    echo "==== V8 gen: mode=$MODE  n=$N_BAGS  → $OUT ===="
    $PY -u -m scripts.generator_v5.run_generator \
        --v7-real \
        --negative-mode "$MODE" \
        --n-bags "$N_BAGS" \
        --seed 0 \
        --workers 8 \
        --progress-every 200 \
        --out "$OUT" \
        --stats-out "$STATS"
done

echo
echo "==== V8 smoke SUMMARY ===="
for MODE in "${MODES[@]}"; do
    OUT="$OUT_DIR/v8_smoke_${MODE}.jsonl"
    SITES=$(wc -l < "$OUT")
    BAGS=$(python3 -c "
import json
seen = set()
for line in open('$OUT'):
    seen.add(json.loads(line)['transposase_id'])
print(len(seen))
")
    printf "  %-24s  sites=%6s  bags=%6s  file=%s\n" "$MODE" "$SITES" "$BAGS" "$(du -h $OUT | cut -f1)"
done

echo "[v8_gen1k] DONE"
