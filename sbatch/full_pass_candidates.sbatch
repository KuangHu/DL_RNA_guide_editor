#!/bin/bash
#SBATCH --job-name=fullpass
#SBATCH --partition=es1
#SBATCH --account=pc_rubinlab
#SBATCH --qos=es_normal
#SBATCH --gres=gpu:A40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/fullpass_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

for f in \
    /global/scratch/users/kh36969/fna_ins_discovery/v7real_scored_v1/all_scores.tsv \
    /global/scratch/users/kh36969/tmp/fna_ins_discovery_score/fna_ins_discovery.jsonl \
    /global/scratch/users/kh36969/fna_ins_discovery/bags/insertions_sites.jsonl \
    checkpoints/channel_b/v7real_main/best.pt; do
    [[ -e "$f" ]] || { echo "[fp] MISSING $f"; exit 2; }
done

$PY -u -m scripts.full_pass_candidates
echo "[fp] DONE"
