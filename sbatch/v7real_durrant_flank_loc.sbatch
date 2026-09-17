#!/bin/bash
#SBATCH --job-name=v7real_durrant_flank
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/v7real_durrant_flank_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Durrant flank-side target localization test.
# For each (cohort, n_sites) cell, sample 100 K-site bags from a TBL group.
# Metric: at model's peak nc position, does flank_argmax lookup match gold
# target_position_in_flank? Random baseline: 0.9% exact, 6.4% ±3bp.
# If accuracy monotonic with n_sites, cross-site amplification generalizes
# to real bacterial data.

$PY -u -m scripts.channel_b_durrant_flank_localize \
    --ckpt checkpoints/channel_b/v7real_main/best.pt \
    --n-sites 1,3,5,8 \
    --n-bags-per-cell 100 \
    --seed 0
echo "[v7real-durrant-flank] DONE"
