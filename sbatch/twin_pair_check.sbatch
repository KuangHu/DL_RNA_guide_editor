#!/bin/bash
#SBATCH --job-name=twinpair
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=logs/twinpair_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u <<'PY'
import json
from collections import Counter

# Just grab bag_ids and axis stats — one pass each
def scan(path, target_ids=None):
    ids = set()
    axis = {'gc': [], 'mu': [], 'nc_len': [], 'nc_hom': [], 'ns': []}
    n_records = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            t = r['transposase_id']
            first = t not in ids
            ids.add(t)
            n_records += 1
            if target_ids is not None and t not in target_ids:
                continue
            if first:
                a = r['labels']['arch']
                axis['gc'].append(a.get('gc_target'))
                axis['mu'].append(a.get('accessibility_target_percentile'))
                axis['nc_hom'].append(a.get('nc_homology_rate'))
                axis['ns'].append(a.get('n_sites'))
                axis['nc_len'].append(r['labels'].get('ncrna_length'))
    return ids, axis, n_records

print("Loading pos bag_ids...", flush=True)
pos_ids, _, _ = scan('/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_v6_stage2_pos50k.jsonl')
print(f"pos: {len(pos_ids)} bag_ids", flush=True)
print("Loading twin bag_ids...", flush=True)
twin_ids, _, _ = scan('/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_v6_stage2_twin50k.jsonl')
print(f"twin: {len(twin_ids)} bag_ids", flush=True)

overlap = pos_ids & twin_ids
print(f"overlap: {len(overlap)}  ({100*len(overlap)/len(pos_ids):.2f}% of pos)", flush=True)
dropped = pos_ids - twin_ids
print(f"dropped (pos has, twin doesn't): {len(dropped)}", flush=True)

if dropped:
    print(f"Sample: {sorted(dropped)[:9]}", flush=True)
    _, axis, _ = scan('/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_v6_stage2_pos50k.jsonl', target_ids=dropped)
    print(f"\nDropped bags axes (n={len(axis['gc'])}):", flush=True)
    import statistics
    for k, vs in axis.items():
        if vs:
            print(f"  {k}: min={min(vs):.3f} med={statistics.median(vs):.3f} max={max(vs):.3f}  unique={len(set(vs))}", flush=True)
    # Distribution by (gc, mu) bucket
    print(f"\nDropped bags per gc bin:", flush=True)
    from collections import Counter
    gc_ct = Counter()
    for g in axis['gc']:
        bin = f'[{int(g*10)/10:.1f},{int(g*10+1)/10:.1f})'
        gc_ct[bin] += 1
    print(f"  {dict(sorted(gc_ct.items()))}", flush=True)
    mu_ct = Counter()
    for m in axis['mu']:
        bin = f'[{int(m*10)/10:.1f},{int(m*10+1)/10:.1f})'
        mu_ct[bin] += 1
    print(f"per mu bin:  {dict(sorted(mu_ct.items()))}", flush=True)
PY
