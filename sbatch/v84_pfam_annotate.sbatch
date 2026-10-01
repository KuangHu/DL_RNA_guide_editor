#!/bin/bash
#SBATCH --job-name=v84_pfam
#SBATCH --partition=lr7
#SBATCH --account=pc_rubinlab
#SBATCH --qos=lr_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=3:00:00
#SBATCH --output=logs/v84_pfam_%j.out
#SBATCH --mem=16G
set -euo pipefail

PFAM_DIR=/global/scratch/users/kh36969/pfam_db
FAA=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/top_k5_proteins.faa
OUT_DIR=/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery
HMMSCAN=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan
HMMPRESS=/global/home/users/kh36969/.conda/envs/opfi/bin/hmmpress

mkdir -p "$PFAM_DIR"

# 1) Download Pfam-A.hmm.gz if not cached
HMM="$PFAM_DIR/Pfam-A.hmm"
if [ ! -f "$HMM" ]; then
    echo "[pfam] downloading Pfam-A.hmm.gz ..."
    cd "$PFAM_DIR"
    curl -sS -O "https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
    echo "[pfam] gunzipping ..."
    gunzip Pfam-A.hmm.gz
fi
echo "[pfam] HMM at $HMM  size=$(du -sh $HMM | awk '{print $1}')"

# 2) hmmpress if indexes missing
if [ ! -f "${HMM}.h3f" ]; then
    echo "[pfam] hmmpress ..."
    $HMMPRESS "$HMM"
fi
echo "[pfam] indexes ready"

# 3) hmmscan on top_k5_proteins.faa
OUT_TSV="$OUT_DIR/pfam_domtbl.tsv"
OUT_LOG="$OUT_DIR/pfam_scan.log"
echo "[pfam] hmmscan starting on $FAA with 16 cpus"
date
$HMMSCAN --cpu 16 -E 1e-5 --domE 1e-5 \
    --domtblout "$OUT_TSV" \
    "$HMM" "$FAA" > "$OUT_LOG"
date

# 4) Summarize hits per CDS cluster
echo "[pfam] hmmscan done, summarizing ..."
python3 <<PY
from collections import defaultdict
hits = defaultdict(list)
with open("$OUT_TSV") as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split()
        # domtblout: target_name accession tlen query_name accession qlen E-value ...
        if len(parts) < 12:
            continue
        pfam_name = parts[0]
        pfam_acc = parts[1]
        query = parts[3]
        evalue = float(parts[6])
        cds_id = query.split("__")[0] if "__" in query else query
        hits[cds_id].append((pfam_name, pfam_acc, evalue))

# Load scores.tsv for bag_max annotation
scores = {}
with open("$OUT_DIR/scores.tsv") as f:
    for i, line in enumerate(f):
        if i == 0: continue
        bid, n_sites, top_p, bmax = line.strip().split("\t")
        scores[bid] = (int(n_sites), int(top_p), float(bmax))

# Emit per-cluster annotation
out_summary = "$OUT_DIR/pfam_by_cluster.tsv"
with open(out_summary, "w") as out:
    out.write("cds_cluster\tn_sites\ttop_p\tbag_max\tn_pfam_hits\tpfam_hits\n")
    # Sort by bag_max
    sorted_cids = sorted(scores.keys(),
                          key=lambda c: -scores[c][2])
    for cid in sorted_cids:
        ns, tp, bmax = scores[cid]
        if ns < 5:
            continue
        h = hits.get(cid, [])
        # Dedupe pfam names (keep best evalue per pfam)
        by_name = {}
        for name, acc, ev in h:
            if name not in by_name or ev < by_name[name][1]:
                by_name[name] = (acc, ev)
        hit_str = ";".join(f"{name}({acc},{ev:.1e})"
                              for name, (acc, ev) in
                              sorted(by_name.items(), key=lambda x: x[1][1]))
        out.write(f"{cid}\t{ns}\t{tp}\t{bmax:.3f}\t{len(by_name)}\t{hit_str}\n")
print(f"[pfam] wrote {out_summary}")

# Top-20 summary
print()
print("## TOP 20 K>=5 CANDIDATES WITH PFAM ANNOTATIONS")
n = 0
for cid in sorted_cids:
    if scores[cid][0] < 5:
        continue
    ns, tp, bmax = scores[cid]
    h = hits.get(cid, [])
    by_name = {}
    for name, acc, ev in h:
        if name not in by_name or ev < by_name[name][1]:
            by_name[name] = (acc, ev)
    hit_names = ", ".join(name for name, _ in sorted(by_name.items(), key=lambda x: x[1][1])[:5])
    if not hit_names:
        hit_names = "(no Pfam hit at E<=1e-5)"
    print(f"  #{n+1:2d}  {cid}  K={ns}  bag_max={bmax:.3f}  top_p={tp}  pfam: {hit_names}")
    n += 1
    if n >= 20:
        break
PY

echo "[v84_pfam] DONE"
