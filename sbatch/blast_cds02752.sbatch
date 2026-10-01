#!/bin/bash
#SBATCH --job-name=v84_bl752
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1:00:00
#SBATCH --output=logs/v84_bl752_%j.out
set +e
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
CONS=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds02752_conserved.faa
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds02752_nr.blastp.tsv
echo "# remote nr blastp on CDS02752 conserved 112aa protein"
echo "# start $(date +%H:%M:%S)"
timeout 3000 $BLASTP -query "$CONS" -db nr -remote   -outfmt "6 qseqid sseqid pident length evalue bitscore stitle"   -max_target_seqs 30 -evalue 1   > "$OUT" 2>&1
echo "# done $(date +%H:%M:%S)  hits=$(wc -l < $OUT)"
if [ -s "$OUT" ]; then
  head -15 "$OUT" | awk -F'\t' '{printf "  %.1f%%  len=%s  E=%.1e  bit=%s  %s\n", $3, $4, $5, $6, $7}'
fi
