#!/bin/bash
#SBATCH --job-name=v7real_lowm_medm
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/v7real_lowm_medm_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Low-m gate (REPLACES underpowered held-out-m gate 2026-09-16):
# Filter: median(m_at_planted across sites) == 8 (bag's typical site is low-m).
# Same held-out-m checkpoint (trained on max(m)>=9) — no retraining.
# medm==8 admits orders more val bags than maxm==8 (which was ~56 total).

echo "==================== HELDOUT-M model (trained on max(m)>=9) ===================="
$PY -u -m scripts.channel_b_per_mode_auroc_v7real \
    --ckpt checkpoints/channel_b/v7real_heldout_m/best.pt \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --seed 0 \
    --medm-filter-eq 8

echo ""
echo "==================== MAIN model (biased-favorable control) — same slice ===================="
$PY -u -m scripts.channel_b_per_mode_auroc_v7real \
    --ckpt checkpoints/channel_b/v7real_main/best.pt \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --seed 0 \
    --medm-filter-eq 8
echo "[v7real-lowm-medm] DONE"
