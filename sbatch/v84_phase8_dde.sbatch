#!/bin/bash
#SBATCH --job-name=v84p8
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/v84p8_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
$PY -u -m scripts.audit_v84_phase8_dde
echo "[v84p8] DONE"
