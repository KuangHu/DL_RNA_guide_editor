#!/bin/bash
# V1 training on the H200 GPU.
#
# Submit:
#   sbatch sbatch/train_v1.sbatch
#
# Or with a custom out-dir:
#   sbatch --export=OUT_DIR=./checkpoints/v1_run3,EPOCHS=30 sbatch/train_v1.sbatch

#SBATCH --job-name=v5_train
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_v5_%j.out
#SBATCH --error=logs/train_v5_%j.err

set -euo pipefail

CODEDIR=/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
DATA=/global/scratch/users/kh36969/DL_novel_guide_editor
: "${OUT_DIR:=${CODEDIR}/checkpoints/v5_${SLURM_JOB_ID:-manual}}"
: "${EPOCHS:=20}"
: "${TNP_BATCH:=8}"
: "${SITES_TRAIN:=16}"
: "${SITES_VAL:=50}"
: "${NUM_WORKERS:=8}"
: "${LR:=3e-4}"
: "${TRAIN_JSONL:=$DATA/splits/train.jsonl}"
: "${TRAIN_CACHE:=$DATA/structure/train_u16.index.json}"
: "${VAL_JSONL:=$DATA/splits/val.jsonl}"
: "${VAL_CACHE:=$DATA/structure/val_u16.index.json}"
: "${EXTRA_ARGS:=--use-dispersion}"

mkdir -p "$OUT_DIR" "${CODEDIR}/logs"

echo "[env] host: $(hostname)"
echo "[env] date: $(date -Iseconds)"
echo "[env] job:  ${SLURM_JOB_ID:-manual}"
echo "[env] out:  $OUT_DIR"
nvidia-smi | head -20 || true

# Torch env: opfi conda env has torch 2.6.0+cu124
PYTHON=/global/home/users/kh36969/.conda/envs/opfi/bin/python

# Prefer TF32 matmul (H200 supports; harmless for BF16-autocast paths too).
export TORCH_ALLOW_TF32=1
export NVIDIA_TF32_OVERRIDE=1

cd "$CODEDIR"
"$PYTHON" -m training.train_v1 \
    --train-jsonl "$TRAIN_JSONL" \
    --train-cache "$TRAIN_CACHE" \
    --val-jsonl   "$VAL_JSONL" \
    --val-cache   "$VAL_CACHE" \
    --out-dir     "$OUT_DIR" \
    --epochs      "$EPOCHS" \
    --tnp-batch   "$TNP_BATCH" \
    --sites-train "$SITES_TRAIN" \
    --sites-val   "$SITES_VAL" \
    --num-workers "$NUM_WORKERS" \
    --lr          "$LR" \
    $EXTRA_ARGS

echo "[done] $(date -Iseconds)"
