#!/bin/bash
#SBATCH --job-name=v84_train
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=logs/v84_train_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# V8.4 main training (2026-09-27):
#   - 7 corpora (unstructured_nc_full retired; see FROZEN V8.4 block)
#   - nc scaffold = [random][rfam_left+guide+rfam_right][random] with
#     family-balanced Rfam sampling (8 bacterial families)
#   - flank_scattered nc identical to positive shape (single-variable
#     ts-scoping negative)
#   - partial n_planted ≤ ⌊0.3·K⌋
#   - flank = 120bp real bacterial DNA, strict flank-scope invariant
#   - Cluster mechanism retired (V8.3, both flank + nc)
#   - NEW cache root — must NOT collide with v8_main_v3's cache
#     (source_hash differs because JSONL content differs; also
#     explicitly rerouted to a new path for defense).
#   - Same architecture as v3 (Ls=9..14, N_CHANNELS=20)
#   - Same hyperparams as v3 (lr=3e-4, wd=1e-2, 8 epochs, batch=32,
#     hidden=128, heads=4, blocks=3, seed=0)
$PY -u -m scripts.channel_b_train \
    --run-name v84_main \
    --corpora-config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v84 \
    --split-mode main \
    --lr 3e-4 --wd 1e-2 \
    --epochs 8 --batch-size 32 \
    --hidden 128 --n-heads 4 --n-blocks 3 \
    --seed 0 --workers 4 \
    --eval-every 1 \
    --save-every-epoch
echo "[v84-train] DONE"
