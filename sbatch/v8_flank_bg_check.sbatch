#!/bin/bash
#SBATCH --job-name=v8_bgchk
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/v8_bgchk_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

$PY -u -m scripts.audit_v8_flank_bg_exclusion_check \
    --smoke-dir /global/scratch/users/kh36969/DL_novel_guide_editor/v8_generation/smoke \
    --max-bags 200 --L 11 --n-sample-positions 5 --fixed-window 30,90
echo "[v8_bgchk] DONE"
