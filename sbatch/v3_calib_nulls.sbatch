#!/bin/bash
#SBATCH --job-name=v3_calib
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/v3_calib_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
mkdir -p logs

JSONL=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery_v5_md_seed2/adapted.jsonl
: "${EVIDENCE_MANIFEST:?Set EVIDENCE_MANIFEST to the verified per-site JSONL sidecar}"

$PY -u scripts/calibrate_v3_nulls.py \
    --jsonl "$JSONL" \
    --evidence-manifest "$EVIDENCE_MANIFEST" \
    --n-bags 500 \
    --seed 2026 \
    --out logs/v3_calib_500_summary.tsv \
    --details logs/v3_calib_500_details.tsv

$PY -u scripts/analyze_v3_calib.py \
    --details logs/v3_calib_500_details.tsv \
    --out logs/v3_calibration_verified.json

echo "[v3_calib] DONE"
