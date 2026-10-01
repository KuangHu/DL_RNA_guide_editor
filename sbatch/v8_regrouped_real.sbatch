#!/bin/bash
#SBATCH --job-name=v8_regrp
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/v8_regrp_%j.out
#SBATCH --exclude=n0005.es1,n0066.es1
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v8_evals/regrouped_v3
mkdir -p "$OUT"

$PY -u -m scripts.audit_v8_regrouped_real \
    --ckpt checkpoints/channel_b/v8_main_v3/best.pt \
    --out-root "$OUT" \
    --Ks 1 3 5 8 --seed 0 --n-dde-bags-target 200
echo "[v8_regrp] DONE"
