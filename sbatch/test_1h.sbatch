#!/bin/bash
#SBATCH --job-name=test_1h
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --output=logs/test_1h_%j.out
#SBATCH --error=logs/test_1h_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u <<'PY'
import shutil, sys, os
sys.path.insert(0, '.')
from scripts.generator_v5.channel_a_v5 import build_v5_positive, compute_channel_a, _parallel_run_variant
from scripts.v5a_framework.variant import spec_m_threshold_L11, run_variant

JSONL = '/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_v6_5k_1f_gc_conditioned_pin.jsonl'
SHARD = '/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_1h_pin_all'
if os.path.exists(SHARD):
    shutil.rmtree(SHARD)
print("Building MatchTable with argmax...", flush=True)
mt, tnp_arch = build_v5_positive(JSONL, SHARD, workers=32)

# Byte-identity: flank_coherence="off"
spec_off = spec_m_threshold_L11(m=8, tau=0, S=5, flank_coherence="off")
peaks_off = _parallel_run_variant(mt, spec_off, SHARD, 32)
res_off = compute_channel_a(mt, peaks_off, tnp_arch, restrict_to={'L': 11})
print(f"[off L=11] cov={res_off['all']['coverage']:.4f} ppv_tnp={res_off['all']['ppv_tnp']:.4f} exact={res_off['all']['exact_rate']:.4f}", flush=True)

# Coherent
spec_coh = spec_m_threshold_L11(m=8, tau=0, S=5, flank_coherence="jitter", flank_jitter=2)
peaks_coh = _parallel_run_variant(mt, spec_coh, SHARD, 32)
res_coh = compute_channel_a(mt, peaks_coh, tnp_arch, restrict_to={'L': 11})
print(f"[jitter=2 L=11] cov={res_coh['all']['coverage']:.4f} ppv_tnp={res_coh['all']['ppv_tnp']:.4f} exact={res_coh['all']['exact_rate']:.4f}", flush=True)
print(f"[jitter=2 L=11 n_tnps]={res_coh['all']['n_tnps']}", flush=True)
PY
