#!/bin/bash
#SBATCH --job-name=v7real_leaktest
#SBATCH --partition=lr4
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/v7real_leaktest_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# CPU-only: LR + GBM + 5-fold CV on 79-dim bag summary. Uses the
# already-warm channel_b_cache_v7real (all 143k+ train tensors cached
# during the epoch-0 fold of 25894759), so no fold happens here.
$PY -u -m scripts.channel_b_leakage_test_v7real \
    --corpora-config config/channel_b_v7real_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v7real \
    --split-mode main --split-side val \
    --seed 0 --n-folds 5
echo "[v7real-leaktest] DONE"
