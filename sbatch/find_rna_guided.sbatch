#!/bin/bash
#SBATCH --job-name=find_rg
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0:30:00
#SBATCH --output=logs/find_rg_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

for f in \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/clu_rep_seq.fasta \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/clu_cluster.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/insert_cds.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/v7real_scored_v1/all_scores.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/v7real_scored_v1/all_scored_final_candidates.tsv \
    /global/scratch/users/kh36969/funcannot_dbs/pfam/Pfam-A.hmm; do
    [[ -e "$f" ]] || { echo "[fr] MISSING $f"; exit 2; }
done

$PY -u -m scripts.find_rna_guided_bags --threads 8
echo "[fr] DONE"
