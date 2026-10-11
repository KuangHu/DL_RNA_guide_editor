#!/bin/bash
#SBATCH --job-name=v3_sweep
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:30:00
#SBATCH --output=logs/v3_sweep_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
mkdir -p logs

JSONL=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery_v5_md_seed2/adapted.jsonl
: "${EVIDENCE_MANIFEST:?Set EVIDENCE_MANIFEST to the verified per-site JSONL sidecar}"
: "${CALIBRATION_JSON:?Set CALIBRATION_JSON to the validated calibration artifact}"
BAG_IDS=$(cat config/v3_sweep_cids.txt | paste -sd,)

$PY -u scripts/run_v3_scorer.py \
    --jsonl "$JSONL" \
    --evidence-manifest "$EVIDENCE_MANIFEST" \
    --calibration "$CALIBRATION_JSON" \
    --bag-ids "$BAG_IDS" \
    --outdir logs/v3_all_candidates

echo "[v3_sweep] DONE"
