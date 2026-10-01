#!/bin/bash
#SBATCH --job-name=v84_lref
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:30:00
#SBATCH --output=logs/v84_lref_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
MAKEDB=/global/home/users/kh36969/.conda/envs/opfi/bin/makeblastdb
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast
REFDIR=$OUT/refs
mkdir -p "$REFDIR"

# Curated Pfam families → download seed alignment sequences via InterPro API
# Families covering RNA-guided nucleases + IS200/IS605 + integrases
declare -a PFAM_IDS=(
  "PF19147"  # Cas12f1-like_TNB
  "PF07282"  # OrfB_IS605  (this IS TnpB itself)
  "PF12762"  # HTH_OrfB_IS605
  "PF01797"  # Y1_Tnp (IS200/IS605 TnpA)
  "PF00589"  # rve (retroviral-type integrase)
  "PF13358"  # rve_3 (integrase)
  "PF13276"  # HTH_38
  "PF00665"  # DDE_Tnp_1 (DDE transposase family 1)
  "PF13276"  # HTH_38
  "PF14319"  # DDE_Tnp_1_6
  "PF01609"  # DDE_Tnp_1_7 (Fer4-like)
  "PF13843"  # DDE_Tnp_1_5
  "PF03050"  # DDE_Tnp_ISL3
)

# Also add TnpB and Cas12f reference proteins from UniProt via UniProt REST
echo "# downloading Pfam family seed sequences..."
for pf in "${PFAM_IDS[@]}"; do
  url="https://www.ebi.ac.uk/interpro/api/entry/pfam/${pf}/?extra_fields=short_name"
  # Get pfam family full-fasta seed from InterPro API (may not exist)
  # Simpler: fetch the pfam-A seed via ftp/http
  out="$REFDIR/${pf}.fa"
  if [ ! -s "$out" ]; then
    curl -s -o "$out" "https://www.ebi.ac.uk/interpro/api/protein/reviewed/entry/pfam/${pf}/?extra_fields=sequence&page_size=200&format=fasta" || true
    n=$(grep -c ">" "$out" 2>/dev/null || echo 0)
    echo "  $pf → $n sequences"
  fi
done

# Also fetch specific known-Cas12f + TnpB proteins by UniProt ID
declare -a UNIPROT_IDS=(
  "Q7X6X8"  # Cas12f1 (Un1Cas12f1 / GeoCas12f)
  "A0A249KZC0"  # Cas12f
  "A0A2H4YRV3"  # TnpB Deinococcus
  "P25811"  # ISDra2 TnpB
  "Q9WYY0"  # AAP7 Cas12a-like
)
UNIP_FA=$REFDIR/uniprot_refs.fa
: > "$UNIP_FA"
for u in "${UNIPROT_IDS[@]}"; do
  curl -s "https://rest.uniprot.org/uniprotkb/${u}.fasta" >> "$UNIP_FA" || true
done
echo "  uniprot direct → $(grep -c '>' $UNIP_FA) sequences"

# Combine all
COMBINED=$REFDIR/combined_refs.fa
cat "$REFDIR"/*.fa > "$COMBINED"
n_total=$(grep -c ">" "$COMBINED" || echo 0)
echo "# combined reference: $n_total sequences"
head -2 "$COMBINED"

if [ "$n_total" -lt 5 ]; then
  echo "[FAIL] combined ref too small; check downloads"
  exit 1
fi

# Build blast DB
$MAKEDB -in "$COMBINED" -dbtype prot -out "$REFDIR/local_ref" -title "curated_rna_guided_refs" 2>&1 | tail -3

# Query each candidate
CAND_FA=$OUT/candidates.faa
LOCAL_TSV=$OUT/local_blast_results.tsv
echo -e "qseqid\tsseqid\tpident\tlength\tevalue\tbitscore\tstitle" > "$LOCAL_TSV"

$BLASTP -query "$CAND_FA" -db "$REFDIR/local_ref" \
  -outfmt "6 qseqid sseqid pident length evalue bitscore stitle" \
  -max_target_seqs 10 -evalue 10 -num_threads 4 \
  >> "$LOCAL_TSV" 2>&1

echo ""
echo "# ==== LOCAL BLAST RESULTS (curated Cas12f/TnpB/Y1_Tnp/DDE ref) ===="
for cid in CDS02973 CDS03228 CDS07465 CDS08046; do
  echo "## $cid:"
  awk -F'\t' -v c="$cid" 'NR>1 && $1 ~ c {printf "  %.1f%%  len=%s  E=%.1e  bit=%s  %s\n", $3, $4, $5, $6, $7}' "$LOCAL_TSV" | head -8
  echo ""
done

echo "[v84_lref] DONE"
