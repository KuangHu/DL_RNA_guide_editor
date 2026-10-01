#!/bin/bash
#SBATCH --job-name=v84_c2973
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:15:00
#SBATCH --output=logs/v84_c2973_%j.out
set -euo pipefail
cd /global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer
PY=/global/home/users/kh36969/.conda/envs/opfi/bin/python3
HMMSCAN=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan
BLASTP=/global/home/users/kh36969/.conda/envs/opfi/bin/blastp
PFAM=/global/scratch/users/kh36969/pfam_db/Pfam-A.hmm
REFDB=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/refs2/local_ref

# 1. Run analysis: dumps all ORFs to cds02973_all_orfs.faa
$PY -u scripts/cds02973_analysis.py

FAA=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds02973_all_orfs.faa

# 2. Pfam scan
DOMTBL=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds02973_pfam.domtbl.tsv
echo ""
echo "# ==== Pfam-A scan on CDS02973 ORFs ===="
$HMMSCAN --cpu 4 -E 1e-3 --domtblout "$DOMTBL" "$PFAM" "$FAA" > /dev/null
awk '!/^#/ {printf "  %-30s  %-40s  E=%-9s  bit=%-7s  ali[%s..%s]\n", $4, $1, $7, $8, $18, $19}' "$DOMTBL" | head -40

# 3. Local BLAST vs Cas12f/TnpB/Y1_Tnp ref
BOUT=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/blast/cds02973_localblast.tsv
echo ""
echo "# ==== Local BLAST vs Cas12f/TnpB/Y1_Tnp ref ===="
$BLASTP -query "$FAA" -db "$REFDB" \
  -outfmt "6 qseqid sseqid pident length evalue bitscore stitle" \
  -max_target_seqs 3 -evalue 1 -num_threads 4 \
  > "$BOUT" 2>&1
$PY -c "
from collections import defaultdict
best = defaultdict(list)
for line in open('$BOUT'):
    parts = line.rstrip().split('\t')
    if len(parts) < 7: continue
    q, s, pid, ln, e, bit, title = parts[:7]
    best[q].append((float(pid), int(ln), float(e), float(bit), title))
for q in sorted(best):
    if not best[q]: continue
    best[q].sort(key=lambda x: -x[3])
    print(f'## {q}:')
    for pid, ln, e, bit, title in best[q][:3]:
        print(f'  {pid:.1f}%  len={ln}  E={e:.1e}  bit={bit:.1f}  {title[:80]}')
"

echo ""
echo "[v84_c2973] DONE"
