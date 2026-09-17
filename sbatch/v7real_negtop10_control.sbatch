#!/bin/bash
#SBATCH --job-name=v7real_negtop10
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --output=logs/v7real_negtop10_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Negative control on real_data/negative_top10 (10 DDE families,
# 63k loci, insert_md5-grouped bags). Score v7real_main; expected LOW
# vs high on v7real pos50k positives. AUROC(pos vs each fam) reports
# specificity per family.

$PY -u -m scripts.channel_b_negtop10_control \
    --n-bags-per-family 50 --n-bags-positive 200 --seed 0
echo "[negtop10] DONE"
