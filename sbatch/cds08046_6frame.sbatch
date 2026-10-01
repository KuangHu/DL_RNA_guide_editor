#!/bin/bash
#SBATCH --job-name=v84_6f
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=0:10:00
#SBATCH --output=logs/v84_6f_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
HMMSCAN=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan

# 1. run 6-frame + frameshift analysis
$PY -u scripts/cds08046_6frame.py

FAA=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds08046_extended_orfs.faa

# 2. blast against local curated ref
REFDB=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/refs2/local_ref
OUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds08046_6frame_blast.tsv
echo ""
echo "# ==== BLAST extended ORFs vs curated Cas12f/TnpB/Y1_Tnp ref ===="
$BLASTP -query "$FAA" -db "$REFDB" \
  -outfmt "6 qseqid sseqid pident length evalue bitscore stitle" \
  -max_target_seqs 5 -evalue 10 -num_threads 2 \
  > "$OUT" 2>&1
n=$(wc -l < "$OUT")
echo "  $n hit lines total"

# report top hit per query
$PY -c "
from collections import defaultdict
best = defaultdict(list)
for line in open('$OUT'):
    parts = line.rstrip().split('\t')
    if len(parts) < 7: continue
    q, s, pid, ln, e, bit, title = parts[:7]
    best[q].append((float(pid), int(ln), float(e), float(bit), title))
for q in sorted(best):
    best[q].sort(key=lambda x: -x[3])
    print(f'## {q}:')
    for pid, ln, e, bit, title in best[q][:3]:
        print(f'  {pid:.1f}%  len={ln}  E={e:.1e}  bit={bit:.1f}  {title[:80]}')
    print()
"

# 3. Pfam scan
PFAMDB=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/Pfam-A.hmm
if [ -f "$PFAMDB" ]; then
  echo "# ==== Pfam scan on extended ORFs ===="
  $HMMSCAN --cpu 2 -E 1e-3 --domtblout /tmp/cds08046_6f.pfam.domtbl.tsv "$PFAMDB" "$FAA" | tail -20
  echo ""
  echo "# domtbl hits:"
  grep -v "^#" /tmp/cds08046_6f.pfam.domtbl.tsv | awk '{printf "  %-25s  %-40s  E=%s  bit=%s\n", $4, $1, $7, $8}'
else
  echo "# Pfam-A.hmm not found at $PFAMDB"
fi

echo ""
echo "[v84_6f] DONE"
