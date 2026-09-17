#!/bin/bash
#SBATCH --job-name=v7real_c4_eval
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:45:00
#SBATCH --output=logs/v7real_c4_eval_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# C4 gate: per-mode AUROC × n_sites for {heldout, main} on val split
# restricted to n_sites ∈ {7, 8}.
# Heldout model NEVER saw {7,8} during training. Main model saw them all.
# PASS: paired Δ CI upper ≤ +0.02 → heldout matches main.

echo "==================== HELDOUT model (trained on n_sites {3-6}) ===================="
$PY -u -m scripts.channel_b_per_mode_auroc_v7real \
    --ckpt checkpoints/channel_b/v7real_heldout_nsites/best.pt \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real_heldout_nsites \
    --seed 0 \
    --n-sites-filter 7,8

echo ""
echo "==================== MAIN model (reference) — same val slice ===================="
$PY -u -m scripts.channel_b_per_mode_auroc_v7real \
    --ckpt checkpoints/channel_b/v7real_main/best.pt \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --seed 0 \
    --n-sites-filter 7,8
echo "[v7real-c4-eval] DONE"
