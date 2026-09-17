#!/bin/bash
#SBATCH --job-name=v7real_pretrainck
#SBATCH --partition=lr4
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/v7real_pretrain_ck_%j.out
# CPU partition (2026-09-13): 200-bag ViennaRNA fold + numpy preflight
# is CPU work; no GPU needed. Earlier A40 reservation idle.
set -uo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

$PY -u /global/scratch/users/kh36969/tmp/claude-50246/-global-home-users-kh36969-tools-DL-RNA-guide-edotor-classifer/0ccc8f3b-56a6-4130-9f44-1c40a872fe3f/scratchpad/v7real_pre_training_verify.py
