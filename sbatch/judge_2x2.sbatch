#!/bin/bash
#SBATCH --job-name=judge_2x2
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --output=logs/judge_2x2_%j.out
#SBATCH --error=logs/judge_2x2_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u <<'PY'
import shutil, os, sys
sys.path.insert(0, '.')
from scripts.generator_v5.channel_a_v5 import build_v5_positive, compute_channel_a, _parallel_run_variant
from scripts.v5a_framework.variant import spec_m_threshold_L11

OUT = '/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen'
def run_cell(jsonl, shard, coherence):
    if not os.path.exists(shard + "/_index.json"):
        if os.path.exists(shard):
            shutil.rmtree(shard)
        mt, tnp_arch = build_v5_positive(jsonl, shard, workers=16)
    else:
        from scripts.v5a_framework.match_table import load as load_mt
        mt = load_mt(shard); tnp_arch = {}
        import json
        with open(jsonl) as f:
            for line in f:
                r = json.loads(line); t = r['transposase_id']
                if t in tnp_arch: continue
                tnp_arch[t] = dict(r['labels'].get('arch', {}))
    spec = spec_m_threshold_L11(m=8, tau=0, S=3, theta=None,
                                    flank_coherence=coherence, flank_jitter=2)
    peaks = _parallel_run_variant(mt, spec, shard, 16)
    res = compute_channel_a(mt, peaks, tnp_arch, restrict_to={'L': 11})
    return res.get('all', {})

corpora = {
    'inconsistent': OUT + '/large_batch/positives_v5_v6_1k_ns3_incons.jsonl',
    'consistent':   OUT + '/large_batch/positives_v5_v6_1k_ns3_cons.jsonl',
}
shards = {
    'inconsistent': OUT + '/channel_a/mt_1k_ns3_incons',
    'consistent':   OUT + '/channel_a/mt_1k_ns3_cons',
}

print("2x2 flank-coherence judge (n_sites=3, L=11, m=8, tau=0, S=3):", flush=True)
print(f"  {'row':<14s} {'nc-only cov / exact':<28s} {'flank-coherent cov / exact':<28s}", flush=True)
for row, jsonl in corpora.items():
    off = run_cell(jsonl, shards[row], 'off')
    jit = run_cell(jsonl, shards[row], 'jitter')
    print(f"  {row:<14s} "
          f"cov={off.get('coverage',0):.4f} exact={off.get('exact_rate',0):.4f}   "
          f"cov={jit.get('coverage',0):.4f} exact={jit.get('exact_rate',0):.4f}",
          flush=True)
print()
print("Predictions (finding_theta_scan_1c_v6):", flush=True)
print("  inconsistent  nc-only: cov high, exact ~0.05 (bg sat)  coherent: cov ~0 (verified)", flush=True)
print("  consistent    nc-only: cov high, exact ~0.05 (bg sat)  coherent: cov ~0.86^3=0.636, exact ~= cov", flush=True)
PY
