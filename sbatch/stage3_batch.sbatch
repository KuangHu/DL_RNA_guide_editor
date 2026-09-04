#!/bin/bash
#SBATCH --job-name=stage3
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=logs/stage3_%x_%j.out
#SBATCH --error=logs/stage3_%x_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
: "${CHECK:?}"
SCRATCH=/global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad
OUTBASE=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen

case "$CHECK" in
    a12)
        /global/home/users/kh36969/.conda/envs/opfi/bin/python -u "$SCRATCH/a12_paired.py" \
            --pos "$OUTBASE/large_batch/positives_v5_v6_stage2_pos50k.jsonl" \
            --twin "$OUTBASE/large_batch/positives_v5_v6_stage2_twin50k.jsonl" \
            --out "$SCRATCH/a12_report.json"
        ;;
    a8b)
        /global/home/users/kh36969/.conda/envs/opfi/bin/python -u <<'PY'
import sys
sys.path.insert(0, '.')
from scripts.v5a_framework.match_table import load as load_mt
from scripts.generator_v5.channel_a_v5 import compute_channel_a, _parallel_run_variant
from scripts.v5a_framework.variant import spec_m_threshold_L11
import json

SHARD = '/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_v6_stage2_pos50k'
JSONL = '/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_v6_stage2_pos50k.jsonl'

mt = load_mt(SHARD)
tnp_arch = {}
with open(JSONL) as f:
    for line in f:
        r = json.loads(line)
        t = r['transposase_id']
        if t in tnp_arch: continue
        tnp_arch[t] = dict(r['labels'].get('arch', {}))
        n_len = int(r['labels'].get('ncrna_length', 0))
        if n_len < 120: tnp_arch[t]['nc_len_bucket']='070-119'
        elif n_len < 180: tnp_arch[t]['nc_len_bucket']='120-179'
        elif n_len < 240: tnp_arch[t]['nc_len_bucket']='180-239'
        else: tnp_arch[t]['nc_len_bucket']='240-300'
        tnp_arch[t]['nc_len'] = n_len

for coh in ("off", "jitter"):
    spec = spec_m_threshold_L11(m=8, tau=0, theta=1.0, flank_coherence=coh, flank_jitter=2)
    peaks = _parallel_run_variant(mt, spec, SHARD, 32)
    for restrict in ({'L': 11}, {'L': 11, 'flank_offset_mode': 'consistent'},
                       {'L': 11, 'flank_offset_mode': 'inconsistent'}):
        res = compute_channel_a(mt, peaks, tnp_arch, restrict_to=restrict)
        r = res.get('all', {})
        print(f"[a8b {coh} {restrict}] n={r.get('n_tnps',0)} cov={r.get('coverage',0):.4f} exact={r.get('exact_rate',0):.4f} ppv_tnp={r.get('ppv_tnp',0):.4f}", flush=True)
PY
        ;;
    fake_plant)
        /global/home/users/kh36969/.conda/envs/opfi/bin/python -u -m scripts.generator_v5.generator_data_checker \
            --jsonl "$OUTBASE/large_batch/positives_v5_v6_stage2_partial40k.jsonl" \
            --layer 2 --max-bags 40000
        ;;
    *)  echo "unknown CHECK=$CHECK"; exit 1 ;;
esac
