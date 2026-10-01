#!/bin/bash
#SBATCH --job-name=annot_cand
#SBATCH --partition=lr6
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1:00:00
#SBATCH --output=logs/annot_cand_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

IN=/global/scratch/users/kh36969/fna_ins_discovery/v7real_scored_v1/all_scored_final_candidates.tsv
for f in "$IN" \
    /global/scratch/users/kh36969/fna_ins_discovery/bags/insertions_sites.jsonl \
    /global/scratch/users/kh36969/fna_ins_discovery/database/insertions_inserts.fna \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/insert_cds.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/clu_cluster.tsv \
    /global/scratch/users/kh36969/fna_ins_discovery/cds/clu_rep_seq.fasta \
    /global/scratch/users/kh36969/is110_refs/is110_reps_95.fa \
    /global/scratch/users/kh36969/funcannot_dbs/pfam/Pfam-A.hmm; do
    [[ -e "$f" ]] || { echo "[ac] MISSING $f"; exit 2; }
done
echo "[ac] preflight ok"

# Annotate ALL 72 passing bags (TSD-CLUSTERED + MOTIF-CLUSTERED)
$PY -u -m scripts.annotate_candidates \
    --in-tsv "$IN" \
    --out-basename all_passing_annotation \
    --threads 8

echo "[ac] DONE"
