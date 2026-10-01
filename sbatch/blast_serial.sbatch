#!/bin/bash
#SBATCH --job-name=v84_blast
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00:00
#SBATCH --output=logs/v84_blast_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast

# fresh empty results
for cid in CDS02973 CDS03228 CDS07465 CDS08046; do
  : > "$OUT/$cid.blastp.tsv"
  : > "$OUT/$cid.blastp.err"
done

# start novel unknowns FIRST (highest priority questions), then controls
for cid in CDS02973 CDS08046 CDS03228 CDS07465; do
  q="$OUT/split/$cid.faa"
  echo "[start $cid] $(date +%H:%M:%S)"
  timeout 2400 $BLASTP -query "$q" -db nr -remote \
    -outfmt "6 qseqid sseqid pident length evalue bitscore stitle" \
    -max_target_seqs 20 -evalue 1e-3 \
    > "$OUT/$cid.blastp.tsv" 2> "$OUT/$cid.blastp.err" || echo "[timeout/fail $cid]"
  hits=$(wc -l < "$OUT/$cid.blastp.tsv")
  echo "[done  $cid] $(date +%H:%M:%S)  hits=$hits"
  if [ "$hits" -gt 0 ]; then
    echo "  top-5:"
    head -5 "$OUT/$cid.blastp.tsv" | awk -F'\t' '{printf "    %.1f%%  E=%.1e  bit=%s  %s\n", $3, $5, $6, $7}'
  else
    echo "  err:"; head -3 "$OUT/$cid.blastp.err" | sed 's/^/    /'
  fi
  echo ""
done

echo "[v84_blast] DONE"
