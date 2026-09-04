#!/bin/bash
#SBATCH --job-name=rate_tbl
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/rate_tbl_%j.out
#SBATCH --error=logs/rate_tbl_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
/global/home/users/kh36969/.conda/envs/opfi/bin/python -c "
import sys
sys.path.insert(0, '.')
from scripts.generator_v5.difficulty import load_or_build_rate_table
tbl = load_or_build_rate_table(rebuild=True)
print(f'built rate table with gc_bins={tbl.gc_bins}')
for gc in tbl.gc_bins:
    r = tbl.rate_by_gc[gc].get((11, 8), float('nan'))
    print(f'  gc={gc:.2f}  rate(L=11, m=8) = {r:.4f}')
print(f'baseline (self.rate) rate(L=11, m=8) = {tbl.rate.get((11, 8), float(\"nan\")):.4f}')
"
