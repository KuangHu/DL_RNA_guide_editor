#!/bin/bash
#SBATCH --job-name=v7real_paired_lowm
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/v7real_paired_lowm_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Paired-Δ bootstrap for low-m gate (median(m)==8 filter).
# A = main (biased-favorable control); B = heldout-m (trained on max(m)>=9).
# PASS if paired-Δ 95% CI upper ≤ +0.02 in EVERY (neg_mode, n_sites) cell.

$PY -u -m scripts.channel_b_paired_auroc_delta \
    --ckpt-a checkpoints/channel_b/v7real_main/best.pt \
    --ckpt-b checkpoints/channel_b/v7real_heldout_m/best.pt \
    --label-a main --label-b heldout \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --seed 0 \
    --filter-kind medm --filter-val 8 \
    --n-boot 1000 \
    --pass-threshold 0.02
echo "[v7real-paired-lowm] DONE"
