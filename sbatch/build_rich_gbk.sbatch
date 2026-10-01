#!/bin/bash
#SBATCH --job-name=v84_rgbk
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=logs/v84_rgbk_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

echo "# Building rich GBK (CDS + flank + nc + v84 anchor + Vienna stems) for all candidates"
$PY -u scripts/build_rich_gbk.py --cds dummy --all 2>&1
echo ""
echo "# ==== event counts + feature counts ===="
for d in /global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate/CDS*/; do
  cid=$(basename "$d")
  n_rec=$(grep -c "^LOCUS" "$d/events_rich.gbk" 2>/dev/null || echo 0)
  n_feat=$(grep -c "^     [a-z]" "$d/events_rich.gbk" 2>/dev/null || echo 0)
  n_stem=$(grep -c "^     stem_loop" "$d/events_rich.gbk" 2>/dev/null || echo 0)
  n_misc=$(grep -c "^     misc_feature" "$d/events_rich.gbk" 2>/dev/null || echo 0)
  n_cds=$(grep -c "^     CDS " "$d/events_rich.gbk" 2>/dev/null || echo 0)
  echo "  $cid  events=$n_rec  CDS=$n_cds  misc_feature=$n_misc  stem_loop=$n_stem"
done
echo ""
echo "[v84_rgbk] DONE"
