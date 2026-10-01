#!/bin/bash
#SBATCH --job-name=v84_blast
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1:30:00
#SBATCH --output=logs/v84_blast_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
FAA=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/candidates.faa
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast

echo "# candidates:"
grep ">" "$FAA"
echo ""

# Split into single-query files, run in parallel
mkdir -p "$OUT/split"
python3 -c "
from pathlib import Path
faa = Path('$FAA').read_text().split('>')[1:]
for rec in faa:
    cid = rec.split('_')[0]
    (Path('$OUT/split') / f'{cid}.faa').write_text('>' + rec)
"

run_blast() {
  local q=$1
  local cid=$(basename "$q" .faa)
  echo "[start $cid] $(date +%H:%M:%S)"
  $BLASTP -query "$q" -db nr -remote \
    -outfmt "6 qseqid sseqid pident length evalue bitscore stitle" \
    -max_target_seqs 20 -evalue 1e-3 \
    > "$OUT/$cid.blastp.tsv" 2> "$OUT/$cid.blastp.err" || {
      echo "[FAIL $cid]"; cat "$OUT/$cid.blastp.err" | head -5
      return
  }
  echo "[done $cid] $(date +%H:%M:%S) hits=$(wc -l < $OUT/$cid.blastp.tsv)"
}

for q in "$OUT"/split/*.faa; do
  run_blast "$q" &
done
wait

echo ""
echo "# ==== TOP HITS per candidate ===="
for f in "$OUT"/*.blastp.tsv; do
  cid=$(basename "$f" .blastp.tsv)
  echo "## $cid ($(wc -l < $f) hits)"
  head -10 "$f" | awk -F'\t' '{printf "  %.1f%%  E=%.1e  bit=%s  %s\n", $3, $5, $6, $7}'
  echo ""
done
echo "[v84_blast] DONE"
