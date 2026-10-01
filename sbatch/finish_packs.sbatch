#!/bin/bash
#SBATCH --job-name=v84_pack2
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=logs/v84_pack2_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
HMMSCAN=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan
PFAM=/global/scratch/users/kh36969/pfam_db/Pfam-A.hmm

# Finish CDS00985 (missing Pfam), build CDS01484 fresh
# Also build additional interesting novel candidates:
#   CDS02752 (4 species!), CDS00724 (rve 240 events), CDS02601 (rve 9 events)
for CID in CDS00985 CDS01484 CDS02752 CDS00724 CDS02601; do
  OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate/$CID
  mkdir -p "$OUT"
  echo "## Building pack for $CID..."
  $PY -u scripts/build_candidate_pack.py --cds "$CID" 2>&1 | tail -3

  FAA=$OUT/orfs.faa
  DOMTBL=$OUT/pfam_all_orfs.domtbl.tsv
  $HMMSCAN --cpu 4 -E 1e-3 --domtblout "$DOMTBL" "$PFAM" "$FAA" > /dev/null

  $PY -u - <<PY 2>&1 | tail -3
import sys
sys.path.insert(0, ".")
from pathlib import Path
import scripts.build_candidate_pack as bcp
bcp.PFAM_DOMTBL_TOP = Path("$DOMTBL")
sys.argv = ["build_candidate_pack.py", "--cds", "$CID"]
bcp.main()
PY
  echo ""
done

echo ""
echo "# ==== Final layout ===="
for cid in CDS01040 CDS02973 CDS05957 CDS03122 CDS00343 CDS02370 CDS00985 CDS01484 CDS02752 CDS00724 CDS02601; do
  d=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate/$cid
  if [ -d "$d" ]; then
    n_events=$(grep -c "^LOCUS" "$d/events.gbk" 2>/dev/null || echo 0)
    n_pfam=$(wc -l < "$d/pfam_hits.tsv" 2>/dev/null || echo 0)
    echo "  $cid: $n_events events, $n_pfam pfam-lines"
  fi
done
echo ""
echo "[v84_pack2] DONE"
