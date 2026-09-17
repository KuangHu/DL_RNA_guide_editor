#!/bin/bash
#SBATCH --job-name=v7real_heldout_m_eval
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00     # bumped from 0:45; first run hit wall
#SBATCH --output=logs/v7real_heldout_m_eval_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# held-out-m gate: per-mode AUROC × n_sites for {heldout-m, main} on val
# slice restricted to POSITIVE bags with max(m_at_planted) == 8.
# Heldout model trained on max(m) >= 9 (never saw val positives).
# Main model saw the whole distribution (biased favorable — this is the
# control).

# Both runs use the warm MAIN cache — tensors are keyed by
# (jsonl_hash, bag_id), independent of which model checkpoint is loaded.
# Held-out-m's own cache only has training-set bags (positives filtered
# to max(m)>=9), so val bags would need fold recomputation from scratch
# (5000 bags × 100ms × 5 corpora = 40+ min; first run at 45min wall
# hit timeout mid-twin50k). Main cache has all 143k train + all 16k val.

echo "==================== HELDOUT-M model (trained on max(m)>=9) ===================="
$PY -u -m scripts.channel_b_per_mode_auroc_v7real \
    --ckpt checkpoints/channel_b/v7real_heldout_m/best.pt \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --seed 0 \
    --maxm-filter-eq 8

echo ""
echo "==================== MAIN model (biased-favorable control) — same slice ===================="
$PY -u -m scripts.channel_b_per_mode_auroc_v7real \
    --ckpt checkpoints/channel_b/v7real_main/best.pt \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --seed 0 \
    --maxm-filter-eq 8
echo "[v7real-heldout-m-eval] DONE"
