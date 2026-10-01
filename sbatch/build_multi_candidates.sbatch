#!/bin/bash
#SBATCH --job-name=v84_multi
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=logs/v84_multi_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
HMMSCAN=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan
PFAM=/global/scratch/users/kh36969/pfam_db/Pfam-A.hmm

echo "# ==== SHORTLIST OF NOVEL-RNA-GUIDED CANDIDATES ===="
$PY -u scripts/shortlist_novel_candidates.py

echo ""
echo "# ==== BUILDING PACKS FOR CDS02973 + top 5 novel candidates ===="

# Always include CDS02973
CIDS="CDS02973"
# Get top 5 novel by novelty_score
TOP5=$($PY -c "
import sys
sys.path.insert(0, '.')
from scripts.shortlist_novel_candidates import parse_realness, parse_pfam, is_known_rna_guided, get_species_of_cid, BATCH_LOG, PFAM_TSV
realness = parse_realness(BATCH_LOG)
pfam = parse_pfam(PFAM_TSV)
cands = []
for cid, r in realness.items():
    pf = pfam.get(cid, r['pfam_short'])
    if is_known_rna_guided(pf): continue
    sp = get_species_of_cid(cid)
    ns = r['realness'] + 0.3*len(sp)
    cands.append((cid, ns))
cands.sort(key=lambda x: -x[1])
print(' '.join(c[0] for c in cands[:6]))
")
CIDS="$CIDS $TOP5"
echo "# CIDs to pack: $CIDS"
echo ""

for CID in $CIDS; do
  OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate/$CID
  mkdir -p "$OUT"
  echo "## Building pack for $CID..."
  # 1. initial build (creates orfs.faa)
  $PY -u scripts/build_candidate_pack.py --cds "$CID" 2>&1 | tail -4

  # 2. Pfam scan
  FAA=$OUT/orfs.faa
  DOMTBL=$OUT/pfam_all_orfs.domtbl.tsv
  $HMMSCAN --cpu 4 -E 1e-3 --domtblout "$DOMTBL" "$PFAM" "$FAA" > /dev/null

  # 3. rebuild pack with domtbl
  $PY -u - <<PY 2>&1 | tail -4
import sys
sys.path.insert(0, ".")
from pathlib import Path
import scripts.build_candidate_pack as bcp
bcp.PFAM_DOMTBL_TOP = Path("$DOMTBL")
sys.argv = ["build_candidate_pack.py", "--cds", "$CID"]
bcp.main()
PY
  echo "  → $(ls -la $OUT | wc -l) files in $OUT"
  echo ""
done

echo ""
echo "# ==== Final layout ===="
ls /global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate/
echo ""
echo "[v84_multi] DONE"
