"""Per-site boundary-overlap summary — parses insert_vs_empty.blast and
insert_vs_occupied.blast to flag edge regions of the "insert" that match
the empty-allele host sequence.

A long (>20 bp) edge hit means the loci-decomposition boundary was called
INSIDE a region that is also found in the empty flank: either the element
has host-flanking repeats, or the decomposition is off by that many bp.

Output: boundary_overlap.tsv at the parent evidence dir.
Also runs tblastn / blastp of per-site Prodigal ORFs cross-site to confirm
whether the inserts share the same ORF content.
"""
from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict


def parse_blast(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < len(header): continue
            row = dict(zip(header, parts))
            for k in ("pident", "evalue", "bitscore"):
                row[k] = float(row[k])
            for k in ("length", "mismatch", "gapopen", "qstart", "qend",
                      "sstart", "send", "qlen", "slen"):
                row[k] = int(row[k])
            rows.append(row)
    return rows


def classify_edge_hit(r: Dict, qlen: int) -> str:
    """Classify a secondary blast hit by whether it covers the query's 5'
    edge, 3' edge, or middle."""
    qs, qe = r["qstart"], r["qend"]
    if qs <= 10 and qe < qlen * 0.3:
        return "5p_edge"
    if qe >= qlen - 10 and qs > qlen * 0.7:
        return "3p_edge"
    return "middle"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", type=Path, required=True,
                     help="e.g. evidence/CDS03597/nc1_L14_fp45_v2")
    args = ap.parse_args()

    sites = sorted(p.name.split("_")[1] for p in args.evidence_dir.glob("site_*")
                    if p.is_dir())
    out = args.evidence_dir / "boundary_overlap.tsv"
    with out.open("w") as f:
        f.write("site_idx\tcompare_to\tedge\thit_pident\thit_length\t"
                "insert_qstart_qend\tsubj_sstart_send\n")
        for sid in sites:
            site_dir = args.evidence_dir / f"site_{sid}"
            for cmp in ("empty", "occupied"):
                blast = site_dir / f"insert_vs_{cmp}.blast"
                if not blast.exists():
                    continue
                rows = parse_blast(blast)
                for r in rows:
                    # Skip the primary 100% self-hit (insert → occupied, full-length)
                    if (cmp == "occupied" and r["pident"] >= 99.9 and
                            r["length"] >= 0.95 * r["qlen"]):
                        continue
                    edge = classify_edge_hit(r, r["qlen"])
                    f.write(f"{sid}\t{cmp}\t{edge}\t"
                             f"{r['pident']:.2f}\t{r['length']}\t"
                             f"{r['qstart']}-{r['qend']}\t"
                             f"{r['sstart']}-{r['send']}\n")

    print(f"wrote {out}")

    # Cross-site ORF homology
    all_faa = args.evidence_dir / "_all_orfs.faa"
    with all_faa.open("w") as fw:
        for sid in sites:
            site_dir = args.evidence_dir / f"site_{sid}"
            faa = site_dir / f"insert_s{sid}.prodigal.faa"
            if not faa.exists(): continue
            orf_idx = 0
            with faa.open() as f:
                for line in f:
                    if line.startswith(">"):
                        orf_idx += 1
                        fw.write(f">s{sid}_orf{orf_idx}\n")
                    else:
                        fw.write(line)

    # All-vs-all blastp
    print("running blastp all-vs-all ORFs...")
    r = subprocess.run(
        ["blastp", "-query", str(all_faa), "-subject", str(all_faa),
         "-outfmt", "6 qseqid sseqid pident length evalue bitscore qlen slen",
         "-evalue", "1e-5", "-max_hsps", "1"],
        capture_output=True, text=True, check=True)
    pw = args.evidence_dir / "orfs_pairwise.blast"
    with pw.open("w") as f:
        f.write("qseqid\tsseqid\tpident\tlength\tevalue\tbitscore\tqlen\tslen\n")
        f.write(r.stdout)
    print(f"wrote {pw}")


if __name__ == "__main__":
    main()
