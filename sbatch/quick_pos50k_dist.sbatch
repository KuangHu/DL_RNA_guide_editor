#!/bin/bash
#SBATCH --job-name=pos50k_dist
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=logs/pos50k_dist_%j.out
set -uo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
/global/home/users/kh36969/.conda/envs/opfi/bin/python -c "
import json
from collections import Counter
BATCH='/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch'
c_flank = Counter(); c_ns = Counter(); c_both = Counter()
seen = set()
for line in open(f'{BATCH}/positives_v5_v6_stage2_pos50k.jsonl'):
    r = json.loads(line)
    t = r['transposase_id']
    if t in seen: continue
    seen.add(t)
    a = r['labels']['arch']
    f = a.get('flank_offset_mode', '?')
    n = a.get('n_sites', -1)
    c_flank[f] += 1
    c_ns[n] += 1
    c_both[(n, f)] += 1
print('flank_offset_mode:', dict(c_flank))
print('n_sites:', dict(sorted(c_ns.items())))
print('cells n_sites=3:')
for k,v in sorted(c_both.items()):
    if k[0]==3:
        print(' ', k, v)
"
