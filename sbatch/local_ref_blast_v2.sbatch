#!/bin/bash
#SBATCH --job-name=v84_lref2
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=logs/v84_lref2_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
MAKEDB=/global/home/users/kh36969/.conda/envs/opfi/bin/makeblastdb
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast
D=$OUT/refs2

# 1. download UniProt refs
bash scripts/build_local_ref.sh

COMBINED=$D/combined_refs.fa
n_total=$(grep -c ">" "$COMBINED" || echo 0)
echo ""
if [ "$n_total" -lt 20 ]; then
  echo "[FAIL] combined ref too small ($n_total); check downloads"
  head -20 "$COMBINED"
  exit 1
fi

# dedup and build DB
python3 - <<PY
from pathlib import Path
combined = Path("$COMBINED")
seen = set(); out = []
cur_h = None; cur_s = []
def flush():
    if cur_h:
        seq = "".join(cur_s)
        key = cur_h.split("|")[1] if "|" in cur_h else cur_h[:40]
        if key not in seen:
            seen.add(key)
            out.append((cur_h, seq))
for line in combined.read_text().splitlines():
    if line.startswith(">"):
        flush()
        cur_h = line; cur_s = []
    else:
        cur_s.append(line)
flush()
dedup = Path("$D/combined_dedup.fa")
with dedup.open("w") as f:
    for h, s in out:
        f.write(f"{h}\n{s}\n")
print(f"# dedup: {len(seen)} unique sequences")
PY

$MAKEDB -in "$D/combined_dedup.fa" -dbtype prot -out "$D/local_ref" -title "curated_rna_guided_refs" 2>&1 | tail -3

# query each candidate
CAND_FA=$OUT/candidates.faa
LOCAL_TSV=$OUT/local_blast_results.tsv
echo -e "qseqid\tsseqid\tpident\tlength\tevalue\tbitscore\tstitle" > "$LOCAL_TSV"

$BLASTP -query "$CAND_FA" -db "$D/local_ref" \
  -outfmt "6 qseqid sseqid pident length evalue bitscore stitle" \
  -max_target_seqs 10 -evalue 10 -num_threads 4 \
  >> "$LOCAL_TSV" 2>&1

echo ""
echo "# ==== LOCAL BLAST vs curated Cas12f/TnpB/Y1_Tnp/CRISPR-Cas ref (dedup) ===="
for cid in CDS02973 CDS03228 CDS07465 CDS08046; do
  echo ""
  echo "## $cid:"
  awk -F'\t' -v c="$cid" 'NR>1 && $1 ~ c {printf "  %.1f%%  len=%s  E=%.1e  bit=%s  %s\n", $3, $4, $5, $6, $7}' "$LOCAL_TSV" | head -8
  n=$(awk -F'\t' -v c="$cid" 'NR>1 && $1 ~ c' "$LOCAL_TSV" | wc -l)
  if [ "$n" -eq 0 ]; then
    echo "  (no hit vs any curated RNA-guided/IS-family ref)"
  fi
done
echo ""
echo "[v84_lref2] DONE"
