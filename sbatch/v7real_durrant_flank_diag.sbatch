#!/bin/bash
#SBATCH --job-name=v7real_durrant_diag
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/v7real_durrant_diag_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Diagnostic post-retraction of ±3bp accuracy claim:
# (1) Prediction-position histogram — is model output constant (~52) or varying?
# (2) Shuffle control — real vs byte-permuted flank; if model output changes on
#     shuffle, content-driven signal exists.
# Baseline = "always predict pred_dist mode" (NOT uniform).

$PY -u -m scripts.channel_b_durrant_flank_diagnostic \
    --n-bags-per-cell 100 --seed 0
echo "[durrant-diag] DONE"
