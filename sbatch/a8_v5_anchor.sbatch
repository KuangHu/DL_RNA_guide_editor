#!/bin/bash
# A8 — Channel A on v2 position arrays over V5 50K, assert L=11 anchor.

#SBATCH --job-name=a8_v5
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=1:30:00
#SBATCH --output=logs/a8_v5_%x_%j.out
#SBATCH --error=logs/a8_v5_%x_%j.err

set -euo pipefail
CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python

echo "[$(date -Iseconds)] A8: v2 arrays → Channel A → V5 50K → L=11 anchor"
cd "$CODEDIR"
$PYTHON -m scripts.generator_v5.tests.test_candidates_v2_channel_a_v5_anchor --workers 60
echo "[$(date -Iseconds)] done"
