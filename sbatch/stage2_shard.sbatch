#!/bin/bash
#SBATCH --job-name=stage2_shard
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=logs/stage2_shard_%x_%j.out
#SBATCH --error=logs/stage2_shard_%x_%j.err
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
: "${TAG:?}"
JSONL=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_${TAG}.jsonl
SHARD=/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_${TAG}
rm -rf "$SHARD"
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u <<'PY'
import os, sys
sys.path.insert(0, '.')
from scripts.generator_v5.channel_a_v5 import build_v5_positive
tag = os.environ['TAG']
JSONL = f'/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_{tag}.jsonl'
SHARD = f'/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_{tag}'
mt, tnp_arch = build_v5_positive(JSONL, SHARD, workers=64)
print(f'done. n_tnps={len(mt.tnp_ids)}')
PY
