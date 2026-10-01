#!/bin/bash
#SBATCH --job-name=v84_pfam2
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=logs/v84_pfam2_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
HMMSCAN=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan
PFAM=/global/scratch/users/kh36969/pfam_db/Pfam-A.hmm
FAA=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds08046_extended_orfs.faa
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds08046_pfam.domtbl.tsv

echo "# Pfam-A scan on 6-frame extended ORFs of CDS08046 events"
echo "# HMM: $PFAM"
ls -la $PFAM $PFAM.h3* 2>&1 | head -4
echo ""

$HMMSCAN --cpu 4 -E 1e-3 --domtblout "$OUT" "$PFAM" "$FAA" > /dev/null

echo "# ==== domtbl hits (E<1e-3) ===="
awk 'NR<=3 || !/^#/{
  if ($0 ~ /^#/) next
  printf "  %-25s  %-40s  E=%-9s  bit=%-7s  ali[%s..%s]\n", $4, $1, $7, $8, $18, $19
}' "$OUT" | head -50

echo ""
echo "[v84_pfam2] DONE"
