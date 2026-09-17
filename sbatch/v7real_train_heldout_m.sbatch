#!/bin/bash
#SBATCH --job-name=v7real_heldout_m
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=logs/v7real_heldout_m_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# held-out-m gate for v7-real — max-aggregation variant.
# train: POSITIVE bags with max(m_at_planted across sites) >= 9
# val:   POSITIVE bags with max(m_at_planted) == 8
# Control = v7real_main (biased-favorable — saw val bags via main split).
# Same hyperparams as v7real_main, fresh dedicated cache root.
$PY -u -m scripts.channel_b_train \
    --run-name v7real_heldout_m \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real_heldout_m \
    --split-mode main \
    --maxm-train-min 9 \
    --maxm-val-eq 8 \
    --lr 3e-4 --wd 1e-2 \
    --epochs 8 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v7real-heldout-m-train] DONE"
