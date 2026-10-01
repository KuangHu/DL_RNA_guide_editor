#!/bin/bash
#SBATCH --job-name=v8_dde
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/v8_dde_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_evals/dde_eval_v1
mkdir -p "$OUT"

$PY -u -m scripts.channel_b_v8_dde_eval \
    --ckpt checkpoints/channel_b/v8_main/best.pt \
    --v8-config config/channel_b_v8_corpora.json \
    --cache-root /global/scratch/users/kh36969/DL_novel_guide_editor/channel_b_cache_v8 \
    --out-root "$OUT" \
    --n-dde-per-fam 200 --n-durrant 173 --seed 0
echo "[v8_dde] DONE"
