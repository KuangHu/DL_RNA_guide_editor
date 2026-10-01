#!/bin/bash
#SBATCH --job-name=v8_pre
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --output=logs/v8_pre_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

$PY -u -m scripts.audit_v8_50k_preshard --n-scan 10000
echo "[v8_pre] DONE"
