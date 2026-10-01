#!/bin/bash
#SBATCH --job-name=v84p1
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=logs/v84p1_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_generation/smoke/v84_none_pilot.jsonl
mkdir -p "$(dirname "$OUT")"
$PY -u -m scripts.generator_v5.run_generator \
    --v7-real \
    --negative-mode none \
    --n-bags 100 \
    --out "$OUT" \
    --seed 0 \
    --workers 2
echo "[v84p1] pipeline OK"
wc -l "$OUT"
head -1 "$OUT" | $PY -c "import json, sys; r=json.loads(sys.stdin.read()); print('site_id:', r['site_id']); print('gen_meta.data_source:', r['generator_metadata']['data_source']); print('gen_meta.generator_version:', r['generator_metadata']['generator_version_or_commit']); print('inputs.keys:', sorted(r['inputs'].keys())); print('flank_len:', len(r['inputs']['flank'])); print('nc_regions_lens:', [len(x) for x in r['inputs']['noncoding_regions']])"
echo "[v84p1] DONE"
