#!/bin/bash
#SBATCH --job-name=annot_top100
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/annot_top100_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

# Preflight — verify inputs exist before the script does its own longer work
for f in \
    /global/scratch/users/kh36969/fna_ins_discovery/v7real_scored_v1/top_100_candidates.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/bags/insertions_sites.jsonl \
    /global/scratch/users/kh36969/fna_ins_discovery/database/insertions_inserts.fna \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/insert_cds.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/clu_cluster.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/clu_rep_seq.fasta \
    /global/scratch/users/kh36969/is110_refs/is110_reps_95.fa \
    /global/scratch/users/kh36969/funcannot_dbs/pfam/Pfam-A.hmm; do
    [[ -e "$f" ]] || { echo "[annot] MISSING $f"; exit 2; }
done
echo "[annot] preflight ok"

$PY -u -m scripts.annotate_top100_is110 --threads 8
echo "[annot] DONE"
