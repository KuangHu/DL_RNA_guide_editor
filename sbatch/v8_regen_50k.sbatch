#!/bin/bash
#SBATCH --job-name=v8_regen
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=logs/v8_regen_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

OUT_DIR=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_generation/50k
mkdir -p "$OUT_DIR"

MODES=(none partial scattered flank_scattered unstructured_nc_full no_alignment)
N_BAGS=50000

for M in "${MODES[@]}"; do
    OUT="$OUT_DIR/v8_${M}.jsonl"
    STATS="$OUT_DIR/v8_${M}.stats.json"
    MANIF="$OUT_DIR/v8_${M}.manifest.json"
    if [ -f "$OUT" ]; then
        echo "SKIP: $OUT exists"
        continue
    fi
    echo
    echo "==== V8 regen: mode=$M  n=$N_BAGS  → $OUT ===="
    date
    $PY -u -m scripts.generator_v5.run_generator \
        --v7-real \
        --negative-mode "$M" \
        --n-bags "$N_BAGS" \
        --seed 0 \
        --workers 16 \
        --progress-every 5000 \
        --out "$OUT" \
        --stats-out "$STATS" \
        --manifest-out "$MANIF"
done

echo
echo "==== V8 regen SUMMARY (with concat_spacer_len in generator_metadata) ===="
for M in "${MODES[@]}"; do
    OUT="$OUT_DIR/v8_${M}.jsonl"
    SITES=$(wc -l < "$OUT")
    BAGS=$(python3 -c "
import json
seen=set(); csl=None
for line in open('$OUT'):
    r=json.loads(line); seen.add(r['transposase_id'])
    if csl is None: csl = r['generator_metadata'].get('concat_spacer_len')
print(f'{len(seen)}  concat_spacer_len={csl}')
")
    printf "  %-24s  sites=%9s  bags/spacer=%s  file=%s\n" "$M" "$SITES" "$BAGS" "$(ls -lh $OUT | awk '{print \$5}')"
done
echo "[v8_regen] DONE"
