#!/bin/bash
#SBATCH --job-name=v84_tmot
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:20:00
#SBATCH --output=logs/v84_tmot_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3

echo "# computing per-site TARGET MOTIFS in flanks (flank_argmax at v84 top_p)"
$PY -u scripts/compute_target_motifs.py --all
echo ""

# Show one example per candidate
echo "# ==== per-candidate target_motifs.tsv (L=14 column, head) ===="
for d in /global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate/CDS*/; do
  cid=$(basename "$d")
  tsv="$d/target_motifs.tsv"
  if [ -f "$tsv" ]; then
    n=$(( $(wc -l < "$tsv") - 1 ))
    echo ""
    echo "## $cid ($n sites)"
    awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="L14_argmax") a=i; else if($i=="L14_m") b=i; else if($i=="L14_orient") c=i; else if($i=="L14_seq") d=i; else if($i=="site_id") e=i; print "  site_id  L14_argmax  L14_m  L14_orient  L14_seq"} NR>1 && NR<=5 {printf "  %s  %s  %s  %s  %s\n", $e, $a, $b, $c, $d}' "$tsv"
  fi
done

echo ""
echo "[v84_tmot] DONE"
