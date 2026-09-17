#!/bin/bash
#SBATCH --job-name=dur_boot_K50
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:05:00
#SBATCH --output=logs/dur_boot_K50_%j.out
set -uo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
$PY -u -m scripts.build_durrant_bootstrap_bags --K 50 --seed 0 --n-min 3 --n-max 8
