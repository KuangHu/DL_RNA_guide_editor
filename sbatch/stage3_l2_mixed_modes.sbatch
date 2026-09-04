#!/bin/bash
#SBATCH --job-name=stage3_l2_mixed
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=logs/stage3_l2_mixed_%j.out
#SBATCH --error=logs/stage3_l2_mixed_%j.err
set -uo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad

echo "=== mode composition of merged corpus ==="
$PY -c "
import json
from collections import Counter
c = Counter()
for line in open('$SCRATCH/mixed_modes.jsonl'):
    c[json.loads(line)['labels'].get('negative_mode','?')] += 1
print(c)"

echo "=== L2 negative_mode probe on mixed corpus ==="
$PY -m scripts.generator_v5.generator_data_checker \
    --jsonl $SCRATCH/mixed_modes.jsonl \
    --layer 2 --max-bags 12000 \
    --report-out $SCRATCH/stage3_l2_mixed_modes.json 2>&1 | tail -40

echo "=== DONE ==="
