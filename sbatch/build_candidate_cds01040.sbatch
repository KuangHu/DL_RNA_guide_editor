#!/bin/bash
#SBATCH --job-name=v84_pack
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=logs/v84_pack_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
HMMSCAN=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
PFAM=/global/scratch/users/kh36969/pfam_db/Pfam-A.hmm
REFDB=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/refs2/local_ref

CID="${CID:-CDS01040}"
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate/$CID

# 1. Initial build (creates orfs.faa)
$PY -u scripts/build_candidate_pack.py --cds "$CID"

# 2. Pfam scan on all this CDS's ORFs
FAA=$OUT/orfs.faa
DOMTBL=$OUT/pfam_all_orfs.domtbl.tsv
echo ""
echo "# Pfam-A scan on $CID ORFs..."
$HMMSCAN --cpu 4 -E 1e-3 --domtblout "$DOMTBL" "$PFAM" "$FAA" > /dev/null
echo "# Pfam hits summary:"
awk '!/^#/ {printf "  %-25s  %-40s  E=%-9s  bit=%s\n", $4, $1, $7, $8}' "$DOMTBL" | head -30

# 3. Local BLAST vs curated ref
BOUT=$OUT/local_blast.tsv
echo ""
echo "# Local BLAST vs Cas12f/TnpB/Y1_Tnp ref..."
echo -e "qseqid\tsseqid\tpident\tlength\tevalue\tbitscore\tstitle" > "$BOUT"
$BLASTP -query "$FAA" -db "$REFDB" \
  -outfmt "6 qseqid sseqid pident length evalue bitscore stitle" \
  -max_target_seqs 3 -evalue 1 -num_threads 4 \
  >> "$BOUT" 2>&1

# 4. rebuild pack now that we have Pfam hits
# Update the script to use the local domtbl if present
$PY -u - <<PY
import sys
from pathlib import Path
sys.path.insert(0, "$(pwd)")
# monkey-patch to use the local Pfam domtbl
import scripts.build_candidate_pack as bcp
bcp.PFAM_DOMTBL_TOP = Path("$DOMTBL")
sys.argv = ["build_candidate_pack.py", "--cds", "$CID"]
bcp.main()
PY

echo ""
ls -la "$OUT"
echo ""
echo "# ==== README.md ===="
cat "$OUT/README.md"
echo ""
echo "[v84_pack] DONE"
