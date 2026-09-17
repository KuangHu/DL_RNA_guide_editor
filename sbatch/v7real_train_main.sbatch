#!/bin/bash
#SBATCH --job-name=v7real_train
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00     # 2026-09-14: raised from 12h. Epoch 0 (cold
                              # fold + cache write) was 9.1h on v7-real;
                              # epochs 1-7 hit cache but still ~30-60 min
                              # each. 8 * ~1h + 9h = ~17h → 24h has margin.
#SBATCH --output=logs/v7real_train_%j.out
set -euo pipefail    # -e added 2026-09-13: earlier v7real_train 25858771
                      # crashed on missing shard but the trailing `echo DONE`
                      # made Slurm mark it COMPLETED (misleading). Fail-loud now.
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# ============================================================================
# v7-real Channel B main training. Hyperparams identical to v7 main so the
# v7 ↔ v7-real comparison is controlled at the training-side; the only
# variables are the corpora (v7 synthetic random flanks + junction-motif +
# single-region nc  vs  v7-real real bacterial flanks + junction-motif-retired
# + multi-region nc + reversed-flow) and the cache root.
#
# Cache-collision guard: --cache-root points at a FRESH v7real-specific dir.
# channel_b_train.py:325-333 will print pre-existing .pt count per corpus and
# flag PRE-EXISTING if non-zero — this is the primary tripwire.
#
# Secondary tripwire (per user 2026-09-13): epoch 0 MUST be visibly slower
# than epochs 1+ (loader recomputes ViennaRNA fold in epoch 0 and writes cache;
# subsequent epochs load fp32 tensors from disk, ~10× faster). If all epochs
# run at similar speed, cache is being consumed under wrong keys — STOP.
# ============================================================================
$PY -u -m scripts.channel_b_train \
    --run-name v7real_main \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --split-mode main \
    --lr 3e-4 --wd 1e-2 \
    --epochs 8 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v7real-train] DONE"
