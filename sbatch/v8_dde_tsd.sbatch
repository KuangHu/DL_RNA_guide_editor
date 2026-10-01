#!/bin/bash
#SBATCH --job-name=v8_tsd
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/v8_tsd_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

$PY -u -m scripts.channel_b_v8_dde_tsd_diagnostic \
    --ckpt checkpoints/channel_b/v8_main/best.pt \
    --v8-config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --dde-eval-dir /global/scratch/users/kh36969/DL_novel_guide_editor/v8_evals/dde_eval_v1 \
    --n-bags-per-fam 20 --n-sample-bags-print 3 --n-unstruct-bags 200 --seed 0
echo "[v8_tsd] DONE"
