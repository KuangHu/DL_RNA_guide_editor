"""Rank top-100 v84 candidates that are candidates for NOVEL RNA-guided systems.

Exclusion criteria (known RNA-guided families to skip):
  - IS110 (DEDD_Tnp_IS110, Transposase_20)
  - IS200/IS605 with Y1_Tnp
  - Group II intron (GIIM)
  - CRISPR-Cas (Cas12f, Cas12m, Cas12a, Cas9, Cas14)
  - Tn7-CAST (TnsA, TnsB paired with Cas)

Everything else — including rve/TnsB-family, DDE_Tnp, MULE, HTH-only, or
completely un-annotated — is a CANDIDATE for novel RNA-guided biology if
v84 flagged it.

Ranking: composite of realness-heuristic score + multi-species bonus.
"""
from __future__ import annotations
import csv
import re
import statistics
from pathlib import Path

BATCH_LOG = Path("/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/v84_pbatch_26551628.out")
PFAM_TSV = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/pfam_by_cluster.tsv")
INSERT_TSV = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/insert_cds.tsv")

KNOWN_RNA_GUIDED_PFAMS = {
    "DEDD_Tnp_IS110", "Transposase_20",  # IS110
    "Y1_Tnp",  # IS200/IS605 TnpA
    "GIIM",  # Group II intron
    "Cas12f1-like_TNB", "OrfB_IS605", "HTH_OrfB_IS605",  # IS605 TnpB / Cas12f
    "Cas12a", "Cpf1", "Cas9", "Cas14", "Cas12m",
    "TnsA", "TnsB", "TnsC", "TnsE",  # Tn7-CAST
    "Cas12f1", "TnsB_C",  # already-flagged
}


def parse_realness(log_path: Path) -> dict[str, dict]:
    """Parse the batch probe output and return CID -> realness metadata."""
    out = {}
    with log_path.open() as f:
        for line in f:
            line = line.rstrip()
            m = re.match(r"\s*#\s*(\d+)\s+(CDS\d+)\s+([-\d.]+)\s+(\d+)\s+(\d+)"
                              r"\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)"
                              r"\s+([-\d.]+)\s+(\d+)bp\s+(\d+)aa\s+(\d+)\s+(\d+)"
                              r"\s+([-\d.]+)\s+(.+?)$", line)
            if not m:
                continue
            g = m.groups()
            out[g[1]] = {
                "rank": int(g[0]), "bag_max": float(g[2]), "K": int(g[3]),
                "top_p": int(g[4]), "peak_ratio": float(g[5]),
                "m_min": float(g[6]), "m_mean": float(g[7]),
                "struct_drop": float(g[8]), "loc_frac": float(g[9]),
                "size_bp": int(g[10]), "orf_aa": int(g[11]),
                "n_species": int(g[12]), "n_events": int(g[13]),
                "realness": float(g[14]), "pfam_short": g[15].strip(),
            }
    return out


def parse_pfam(tsv_path: Path) -> dict[str, str]:
    """CID -> full Pfam string (as originally in the summary tsv)."""
    d = {}
    with tsv_path.open() as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip().split("\t")
            cid = parts[0]
            if len(parts) >= 6:
                d[cid] = parts[5]
            else:
                d[cid] = ""
    return d


def is_known_rna_guided(pfam_str: str) -> bool:
    hits = set(re.findall(r"([A-Za-z0-9_\-]+)\(", pfam_str)) if pfam_str else set()
    return bool(hits & KNOWN_RNA_GUIDED_PFAMS)


def get_species_of_cid(cid: str) -> list[str]:
    species = set()
    with INSERT_TSV.open() as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip().split("\t")
            if parts[8] == cid:
                species.add(parts[1])
    return sorted(species)


def main():
    realness = parse_realness(BATCH_LOG)
    pfam = parse_pfam(PFAM_TSV)
    print(f"# parsed {len(realness)} candidates from batch probe log")

    candidates = []
    for cid, r in realness.items():
        pf_full = pfam.get(cid, r["pfam_short"])
        if is_known_rna_guided(pf_full):
            continue
        species = get_species_of_cid(cid)
        r["species_list"] = species
        r["pfam_full"] = pf_full
        r["novelty_score"] = r["realness"] + (0.3 * len(species))
        candidates.append((cid, r))

    candidates.sort(key=lambda x: -x[1]["novelty_score"])

    print(f"# {len(candidates)} candidates after excluding known-RNA-guided families\n")
    print(f"# ==== TOP 30 NOVEL-RNA-GUIDED CANDIDATES ====")
    print(f"  {'rank':>4s}  {'cds':<10s}  {'RS':>5s}  {'NS':>4s}  "
              f"{'K':>2s}  {'peak':>5s}  {'m_min':>5s}  {'strD':>5s}  "
              f"{'loc':>5s}  {'size':>6s}  {'orf':>4s}  {'spc':>3s}  "
              f"{'evt':>4s}  {'pfam':<40s}")
    for i, (cid, r) in enumerate(candidates[:30], 1):
        print(f"  {i:>3d}   {cid:<10s}  {r['realness']:>5.2f}  "
                  f"{r['novelty_score']:>4.2f}  {r['K']:>2d}  "
                  f"{r['peak_ratio']:>5.2f}  {r['m_min']:>5.2f}  "
                  f"{r['struct_drop']:>5.2f}  {r['loc_frac']:>5.2f}  "
                  f"{r['size_bp']:>4d}bp  {r['orf_aa']:>3d}aa  "
                  f"{r['n_species']:>3d}  {r['n_events']:>4d}  "
                  f"{r['pfam_short'][:40]:<40s}")


if __name__ == "__main__":
    main()
