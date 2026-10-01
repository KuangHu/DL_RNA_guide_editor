#!/bin/bash
# Download curated Cas12f/TnpB/Y1_Tnp/CRISPR-Cas references from UniProt REST
# and build a local blastp database.
set +e
D=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/refs2
mkdir -p "$D"
echo "# UniProt REST search — RNA-guided nuclease + IS200/IS605 references"
declare -a QUERIES=(
  "cas12f"
  "cas12"
  "cas12a"
  "cas9"
  "cas14"
  "tnpb+is605"
  "tnpb"
  "tnpa+is605"
  "tnpa+is200"
  "orfb+is605"
  "orfa+is605"
)
for q in "${QUERIES[@]}"; do
  out=$D/${q//+/_}.fa
  # try reviewed first
  curl -s "https://rest.uniprot.org/uniprotkb/search?query=${q}+AND+reviewed:true&format=fasta&size=100" > "$out"
  n=$(grep -c ">" "$out" 2>/dev/null || echo 0)
  if [ "$n" -lt 3 ]; then
    curl -s "https://rest.uniprot.org/uniprotkb/search?query=${q}&format=fasta&size=100" > "$out"
    n=$(grep -c ">" "$out" 2>/dev/null || echo 0)
  fi
  echo "  $q -> $n sequences"
done
COMBINED=$D/combined_refs.fa
cat "$D"/*.fa > "$COMBINED"
n_total=$(grep -c ">" "$COMBINED" || echo 0)
echo "# TOTAL: $n_total sequences (may include duplicates)"
