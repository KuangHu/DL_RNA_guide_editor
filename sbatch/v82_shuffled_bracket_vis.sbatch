#!/bin/bash
#SBATCH --job-name=v82_shvis
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/v82_shvis_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
$PY -u -m scripts.audit_v82_shuffled_bracket_visibility
echo "[v82_shvis] DONE"
