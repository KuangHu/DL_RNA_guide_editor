#!/bin/bash
#SBATCH --job-name=v7real_negtop10_realpos
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/v7real_negtop10_realpos_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Sanity extension: Durrant WT (REAL RNA-guided) vs 10 DDE families.
# Also reports n_sites + nc_len marginals to expose synth-vs-real confound
# in the earlier pos50k-vs-DDE 0.956-0.977 result.

$PY -u -m scripts.channel_b_negtop10_realpos \
    --n-bags-per-family 50 --n-bags-pos50k 200 --n-bags-durrant 50 \
    --durrant-n-sites-range 3,8 --seed 0
echo "[nrg-realpos] DONE"
