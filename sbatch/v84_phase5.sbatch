#!/bin/bash
#SBATCH --job-name=v84p5
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/v84p5_%j.out
#SBATCH --mem=32G
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
$PY -u -m scripts.audit_v84_phase5_loader_smoke
echo "[v84p5] DONE"
