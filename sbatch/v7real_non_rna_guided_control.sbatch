#!/bin/bash
#SBATCH --job-name=v7real_nrg_ctrl
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/v7real_nrg_ctrl_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Specificity control: v7real_main on real DDE-family (non-RNA-guided) IS
# insertions (IS10-R, IS903) vs real IS110-family (RNA-guided).
# Expected: is110 scores HIGH, IS10-R/IS903 scores LOW → AUROC ≥ 0.80.

$PY -u -m scripts.channel_b_non_rna_guided_control \
    --n-bags-per-family 200 --seed 0
echo "[nrg-ctrl] DONE"
