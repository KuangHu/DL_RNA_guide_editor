#!/bin/bash
#SBATCH --job-name=A_drift
#SBATCH --partition=cf1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=cf_normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=2:00:00
#SBATCH --output=logs/A_drift_%j.out
#SBATCH --error=logs/A_drift_%j.err

set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
/global/home/users/kh36969/.conda/envs/opfi/bin/python -u \
    /global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/7fc4fa63-4f23-4274-bd5f-c26a03b323fb/scratchpad/measure_A_drift.py
