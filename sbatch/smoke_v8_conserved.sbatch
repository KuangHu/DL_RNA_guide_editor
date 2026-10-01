#!/bin/bash
#SBATCH --job-name=v8_cons_sm
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:30:00
#SBATCH --output=logs/v8_cons_sm_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

$PY -u -m scripts.smoke_v8_conserved_regions
echo "[v8_cons_sm] DONE"
