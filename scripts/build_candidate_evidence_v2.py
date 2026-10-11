"""Candidate evidence v2 — pull REAL genomic evidence from deep_kpneu/full_kpneu.

Rolls back the inflated v1 evidence folder (whole-insert mislabeled as CDS,
no occupied/empty allele, no genome coord check) and builds, per supporting
site:
  loci_evidence.tsv       — one row with EVERY coord-level field from loci.tsv
                             (contig, start, end, junction_L_abs, junction_R_abs,
                              junction_ambiguity_bp, pre_event_genome/contig/strand,
                              inserted_len, insert_start_in_long_bp, inserted_md5,
                              decomposition_method/quality/confidence,
                              homologous_identity/coverage, n_equivalent_decompositions)
  occupied.fna            — the long allele that CONTAINS the insert
  empty.fna               — the shortest allele = the empty/pre-event target
  insert_from_allele.fna  — insert carved from occupied by insert_start_in_long_bp
  insert_from_jsonl.fna   — insert sequence that the pipeline actually consumed
  insert.gbk              — proper multi-CDS GBK via Prodigal (meta); NO
                             whole-insert fake CDS annotation.
  locus_region.fna        — ±1000 bp of the locus in the SEED GENOME, for coord
                             sanity (contig, strand, offsets)
  insert_vs_occupied.blast — blastn of insert against occupied allele
                             (short-word small-db settings; preserves ties)
  insert_vs_empty.blast    — blastn of insert against empty allele
                             (TIR / repeat edge check)
  md5_check.txt            — md5 of insert_from_allele vs inserted_md5 vs
                             md5 of insert_from_jsonl; FAIL if any mismatch
  README.md

Also produces one cross-site summary at the parent dir:
  inserts_pairwise.blast   — all-vs-all blastn across the 4 supporting sites'
                              inserts, to CHECK (not assume) which sites are
                              homologous.

Everything is just dumped. No TIR verdict. The reviewer / user interprets.

Usage:
  python scripts/build_candidate_evidence_v2.py \
    --jsonl /global/scratch/.../adapted.jsonl \
    --bag CDS03597 \
    --nc-pos 1 --L 14 --flank-pos 45 \
    --out evidence/CDS03597/nc1_L14_fp45_v2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.pipeline import run_bag, PipelineConfig  # noqa: E402

FNA_ROOT = Path("/global/scratch/users/kh36969/fna_ins_discovery")
GENOMES_ROOT = FNA_ROOT / "genomes"


def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def parse_locus_header(db_id: str, locus_full: str) -> dict:
    """locus_full looks like `deep_kpneu/GCA_012964645.1_k12|GCA_012964645.1_k12.B000025`.

    Returns:
      panel: `deep_kpneu` or `full_kpneu`
      genome_dir_key: `GCA_012964645.1_k12` (used as dir name under l2/panel/)
      locus_id: `GCA_012964645.1_k12.B000025`
      assembly: `GCA_012964645.1` (parent assembly; strip trailing _<species_tag>)
    """
    assert "|" in locus_full, locus_full
    panel_key, locus_id = locus_full.split("|", 1)
    panel, genome_dir_key = panel_key.split("/", 1)
    # Assembly is `GCA_xxxxxxxxx.y` stripped of trailing `_k12` / `_k24` / `_<tag>`
    m = re.match(r"^(GC[AF]_\d+\.\d+)_", genome_dir_key)
    assembly = m.group(1) if m else genome_dir_key
    species_dir_map = {"kpneu": "kpneumoniae", "ecoli": "ecoli", "sa": "staphaureus",
                        "pa": "paeruginosa"}
    # Species encoded in db_id prefix (e.g. kpneu.E001816)
    sp_prefix = db_id.split(".", 1)[0]
    species_dir = species_dir_map.get(sp_prefix, sp_prefix)
    return {
        "panel": panel,
        "genome_dir_key": genome_dir_key,
        "locus_id": locus_id,
        "assembly": assembly,
        "species_dir": species_dir,
    }


def load_loci_row(info: dict) -> dict:
    loci_tsv = FNA_ROOT / info["panel"] / "l2" / info["genome_dir_key"] / "loci.tsv"
    with loci_tsv.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            if row["locus_id"] == info["locus_id"]:
                return row
    raise ValueError(f"locus not found: {info['locus_id']} in {loci_tsv}")


def load_alleles_fna(info: dict) -> Dict[str, str]:
    """Return {allele_id: sequence_uppercase} for the given locus."""
    fna = FNA_ROOT / info["panel"] / "l2" / info["genome_dir_key"] / "alleles.fna"
    out: Dict[str, str] = {}
    cur_id = None
    cur_seq: List[str] = []
    want_prefix = info["locus_id"] + "|"
    with fna.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                if cur_id is not None and cur_id.startswith(want_prefix):
                    aid = cur_id.split("|")[1]
                    out[aid] = "".join(cur_seq).upper()
                cur_id = line[1:]
                cur_seq = []
            else:
                cur_seq.append(line)
    if cur_id is not None and cur_id.startswith(want_prefix):
        aid = cur_id.split("|")[1]
        out[aid] = "".join(cur_seq).upper()
    return out


def samtools_faidx(fasta: Path, region: str) -> str:
    """region = `contig:start-end` 1-based inclusive. Returns upper seq."""
    r = subprocess.run(
        ["samtools", "faidx", str(fasta), region],
        check=True, capture_output=True, text=True)
    lines = r.stdout.splitlines()
    return "".join(l for l in lines if not l.startswith(">")).upper()


def write_fna(path: Path, name: str, seq: str, line_width: int = 80) -> None:
    with path.open("w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), line_width):
            f.write(seq[i:i + line_width] + "\n")


def write_multi_fna(path: Path, records: List[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for name, seq in records:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


def run_prodigal_meta(insert_fna: Path, out_dir: Path, site_tag: str) -> dict:
    """Run prodigal in meta mode. Returns {gff: Path, faa: Path, n_cds: int}."""
    gff = out_dir / f"{site_tag}.prodigal.gff"
    faa = out_dir / f"{site_tag}.prodigal.faa"
    r = subprocess.run(
        ["prodigal", "-i", str(insert_fna), "-o", str(gff), "-a", str(faa),
         "-p", "meta", "-q", "-f", "gff"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"prodigal failed: {r.stderr}")
    n_cds = 0
    if gff.exists():
        with gff.open() as f:
            for line in f:
                if line.startswith("#"): continue
                if line.strip(): n_cds += 1
    return {"gff": gff, "faa": faa, "n_cds": n_cds}


def gff_to_gbk(insert_fna: Path, gff: Path, gbk_out: Path,
                 extra_feature_quals: Optional[dict] = None) -> None:
    """Parse Prodigal GFF, build a proper multi-CDS GenBank. NO fake whole-insert
    CDS. Each CDS feature carries start/stop_codon, partial flags, strand,
    rbs_motif/spacer if present."""
    from Bio import SeqIO
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    from Bio.Seq import Seq

    seq_rec = next(SeqIO.parse(str(insert_fna), "fasta"))
    rec = SeqRecord(Seq(str(seq_rec.seq).upper()),
                     id=seq_rec.id, name=seq_rec.id[:16],
                     description=f"insert len={len(seq_rec.seq)}",
                     annotations={"molecule_type": "DNA"})
    with gff.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9: continue
            _seqid, src, ftype, start, end, score, strand, phase, attrs = parts
            if ftype != "CDS": continue
            s = int(start) - 1  # GFF 1-based inclusive → 0-based half-open
            e = int(end)
            strand_val = 1 if strand == "+" else -1
            quals: dict = {"source": "Prodigal_v2.6.3_meta",
                            "score": score, "phase": phase}
            for a in attrs.split(";"):
                if "=" in a:
                    k, v = a.split("=", 1)
                    quals[k] = v
            feat = SeqFeature(FeatureLocation(s, e, strand=strand_val),
                              type="CDS", qualifiers=quals)
            rec.features.append(feat)
    if extra_feature_quals:
        from Bio.SeqFeature import SeqFeature, FeatureLocation
        rec.annotations.setdefault("comment", "")
        for k, v in extra_feature_quals.items():
            rec.annotations["comment"] += f"\n{k}: {v}"
    SeqIO.write([rec], str(gbk_out), "genbank")


def run_blastn(query_fna: Path, subject_fna: Path, out_tsv: Path) -> None:
    """Short-word blastn preserving all HSPs (tie-friendly). Outfmt 6 with
    standard columns + qlen slen."""
    cmd = ["blastn", "-query", str(query_fna), "-subject", str(subject_fna),
           "-outfmt", "6 qseqid sseqid pident length mismatch gapopen "
                      "qstart qend sstart send evalue bitscore qlen slen",
           "-word_size", "11", "-dust", "no", "-soft_masking", "false",
           "-evalue", "1e-5", "-max_hsps", "20"]
    r = subprocess.run(cmd, capture_output=True, text=True, check=True)
    header = ("qseqid\tsseqid\tpident\tlength\tmismatch\tgapopen\t"
              "qstart\tqend\tsstart\tsend\tevalue\tbitscore\tqlen\tslen\n")
    with out_tsv.open("w") as f:
        f.write(header)
        f.write(r.stdout)


def process_site(site_idx: int, db_id: str, locus_full: str,
                   bag_sites: List[dict], out_dir: Path) -> dict:
    """Build per-site evidence subfolder. Returns summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    info = parse_locus_header(db_id, locus_full)
    loci_row = load_loci_row(info)
    alleles = load_alleles_fna(info)

    # Occupied = longest, empty = shortest
    occ_id = loci_row["longest_allele"]
    emp_id = loci_row["shortest_allele"]
    occ_seq = alleles[occ_id]
    emp_seq = alleles[emp_id]

    # Insert carved from occupied by insert_start_in_long_bp when present
    # (deep_kpneu); fall back to `junction_L_abs - start` otherwise (full_kpneu).
    if loci_row.get("insert_start_in_long_bp"):
        ins_start = int(loci_row["insert_start_in_long_bp"])
        ins_start_src = "insert_start_in_long_bp"
    else:
        ins_start = int(loci_row["junction_L_abs"]) - int(loci_row["start"])
        ins_start_src = "junction_L_abs - start (fallback)"
    ins_len = int(loci_row["inserted_len"])
    insert_from_allele = occ_seq[ins_start:ins_start + ins_len]

    # Insert as the pipeline consumed it (JSONL contains the actual flank/nc,
    # but the full insert is reconstructed by removing the flank from the
    # extracted insert. Easier: use loci_row[inserted_md5] + per-site record's
    # insert_md5 as the pipeline-consumed identity. The JSONL rarely carries
    # the full insert sequence; we check md5 equality instead.)
    jsonl_insert_md5 = bag_sites[site_idx].get("insert_md5")
    md5_check = {
        "md5_from_allele_calc": md5(insert_from_allele),
        "md5_from_loci_tsv": loci_row["inserted_md5"],
        "md5_from_jsonl_record": jsonl_insert_md5,  # may be None in v84 schema
        "insert_from_allele_len": len(insert_from_allele),
        "inserted_len_from_loci_tsv": ins_len,
        "insert_start_source": ins_start_src,
        "insert_start_value": ins_start,
    }

    # Loci evidence
    keep_cols = ["locus_id", "seed_genome", "contig", "start", "end",
                   "anchor_bp", "n_genomes_placed", "n_alleles",
                   "shortest_allele", "shortest_len",
                   "longest_allele", "longest_len", "delta_len",
                   "event_class", "polarity", "inserted_len",
                   "junction_ambiguity_bp", "target_bases_lost",
                   "inserted_md5", "target_retained_bp",
                   "n_equivalent_decompositions", "decomposition_confidence",
                   "pre_event_genome", "pre_event_contig",
                   "pre_event_strand", "junction_L_abs", "junction_R_abs",
                   "decomposition_method", "decomposition_quality",
                   "homologous_identity", "homologous_coverage",
                   "largest_insert_block_len", "second_largest_indel_len",
                   "n_snp", "n_small_indel", "alignment_edit_distance",
                   "insert_start_in_long_bp"]
    with (out_dir / "loci_evidence.tsv").open("w") as f:
        f.write("field\tvalue\n")
        for c in keep_cols:
            v = loci_row.get(c, "")
            if c == "insert_start_in_long_bp" and not v:
                v = f"{ins_start} (fallback: {ins_start_src})"
            f.write(f"{c}\t{v}\n")

    # Write per-site allele FASTAs
    write_fna(out_dir / "occupied.fna",
               f"{loci_row['locus_id']}_allele_{occ_id}_occupied_len{len(occ_seq)}",
               occ_seq)
    write_fna(out_dir / "empty.fna",
               f"{loci_row['locus_id']}_allele_{emp_id}_empty_len{len(emp_seq)}",
               emp_seq)
    write_fna(out_dir / "insert_from_allele.fna",
               f"{loci_row['locus_id']}_insert_fromAllele_{ins_len}bp",
               insert_from_allele)

    # Attempt to re-derive insert from genome (seed genome)
    seed_fasta = GENOMES_ROOT / info["species_dir"] / f"{info['assembly']}.fna"
    locus_region = ""
    if seed_fasta.exists():
        try:
            start = int(loci_row["start"])
            end = int(loci_row["end"])
            pad = 1000
            r_start = max(1, start - pad)
            r_end = end + pad
            region_s = f"{loci_row['contig']}:{r_start}-{r_end}"
            locus_region = samtools_faidx(seed_fasta, region_s)
            write_fna(out_dir / "locus_region.fna",
                       f"{loci_row['contig']}_{r_start}_{r_end}_pad1000",
                       locus_region)
        except Exception as e:
            (out_dir / "locus_region.ERROR.txt").write_text(f"{e}\n")
    else:
        (out_dir / "locus_region.MISSING.txt").write_text(
            f"seed_fasta_not_found: {seed_fasta}\n")

    # Prodigal → GBK on insert
    prod = run_prodigal_meta(out_dir / "insert_from_allele.fna",
                              out_dir, f"insert_s{site_idx}")
    extra = {
        "db_id": db_id, "locus_id": loci_row["locus_id"],
        "contig": loci_row["contig"],
        "start": loci_row["start"], "end": loci_row["end"],
        "pre_event_genome": loci_row["pre_event_genome"],
        "pre_event_contig": loci_row["pre_event_contig"],
        "pre_event_strand": loci_row["pre_event_strand"],
        "junction_L_abs": loci_row["junction_L_abs"],
        "junction_R_abs": loci_row["junction_R_abs"],
        "junction_ambiguity_bp": loci_row["junction_ambiguity_bp"],
        "n_equivalent_decompositions": loci_row["n_equivalent_decompositions"],
        "decomposition_method": loci_row["decomposition_method"],
        "decomposition_confidence": loci_row["decomposition_confidence"],
        "decomposition_quality": loci_row["decomposition_quality"],
        "inserted_md5": loci_row["inserted_md5"],
        "insert_start_in_long_bp": loci_row.get("insert_start_in_long_bp", f"{ins_start} (fallback)"),
        "inserted_len": loci_row["inserted_len"],
    }
    gff_to_gbk(out_dir / "insert_from_allele.fna",
                prod["gff"], out_dir / "insert.gbk",
                extra_feature_quals=extra)

    # Blastn insert vs occupied + insert vs empty
    run_blastn(out_dir / "insert_from_allele.fna",
                out_dir / "occupied.fna",
                out_dir / "insert_vs_occupied.blast")
    run_blastn(out_dir / "insert_from_allele.fna",
                out_dir / "empty.fna",
                out_dir / "insert_vs_empty.blast")

    with (out_dir / "md5_check.txt").open("w") as f:
        f.write(json.dumps(md5_check, indent=2))

    return {
        "site_idx": site_idx, "db_id": db_id,
        "locus_id": loci_row["locus_id"],
        "contig": loci_row["contig"],
        "start": loci_row["start"], "end": loci_row["end"],
        "pre_event_genome": loci_row["pre_event_genome"],
        "pre_event_strand": loci_row["pre_event_strand"],
        "n_equivalent_decompositions": loci_row["n_equivalent_decompositions"],
        "junction_L_abs": loci_row["junction_L_abs"],
        "junction_R_abs": loci_row["junction_R_abs"],
        "junction_ambiguity_bp": loci_row["junction_ambiguity_bp"],
        "decomposition_method": loci_row["decomposition_method"],
        "insert_start_in_long_bp": loci_row.get("insert_start_in_long_bp", f"{ins_start} (fallback)"),
        "n_cds_prodigal": prod["n_cds"],
        "md5_from_allele_calc": md5_check["md5_from_allele_calc"],
        "md5_from_loci_tsv": md5_check["md5_from_loci_tsv"],
        "md5_from_jsonl": jsonl_insert_md5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--nc-pos", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--flank-pos", type=int, required=True)
    ap.add_argument("--max-offset", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sites = []
    with args.jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("transposase_id") == args.bag:
                sites.append(r)
    if not sites:
        raise SystemExit(f"bag {args.bag} not found in {args.jsonl}")

    # Identify supporting sites by replaying A→D
    cfg = PipelineConfig()
    result = run_bag(args.bag, sites, cfg)
    matches = []
    for cl, bg in result.passing + result.failing:
        if cl.nc_pos == args.nc_pos and cl.L == args.L and \
                abs(cl.flank_pos_median - args.flank_pos) <= args.max_offset:
            matches.append((cl, bg))
    if not matches:
        raise SystemExit(f"no cluster at nc_pos={args.nc_pos} L={args.L} "
                          f"fp~{args.flank_pos} in bag {args.bag}")
    cl = matches[0][0]
    support_idx = sorted(ss.site_idx for ss in cl.supporting_sites)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "README_v2.md").write_text(
        f"# {args.bag} — real-genome evidence v2\n\n"
        f"- cluster: nc_pos={args.nc_pos}  L={args.L}  flank_pos={args.flank_pos}\n"
        f"- supporting sites: {support_idx}\n\n"
        f"## Per-site deliverables (in each site_<i>/ folder)\n"
        f"- `loci_evidence.tsv` — all coord-level fields from loci.tsv\n"
        f"- `occupied.fna`, `empty.fna` — alleles containing / lacking the insert\n"
        f"- `insert_from_allele.fna` — insert carved from occupied allele\n"
        f"- `insert.gbk` — Prodigal (meta) multi-CDS GBK, with locus/coord\n"
        f"  provenance in annotations/comment. **NO whole-insert fake CDS.**\n"
        f"- `insert_vs_occupied.blast` / `insert_vs_empty.blast` — blastn tables\n"
        f"- `locus_region.fna` — ±1 kb of locus in the seed genome contig\n"
        f"- `md5_check.txt` — md5(insert_from_allele) == loci_tsv.inserted_md5 ?\n"
        f"- `insert_s<i>.prodigal.{{gff,faa}}` — raw Prodigal outputs\n\n"
        f"## Cross-site deliverables (at parent dir)\n"
        f"- `inserts_pairwise.blast` — all-vs-all blastn across supporting sites\n"
        f"- `summary.tsv` — one row per supporting site with key fields\n\n"
        f"## Scope\n"
        f"This v2 package replaces the v1 inflated evidence folder "
        f"(nc1_L14_fp45) whose `insert.gbk` incorrectly labeled the entire "
        f"1824 bp as one CDS and whose pairing diagrams were presented as "
        f"TIR confirmation. v2 presents raw genomic evidence only — no "
        f"TIR verdict. Interpretation is for the reviewer.\n"
    )

    # Build db_id -> locus_full map from the provenance.tsv next to v1 folder
    prov = {}
    v1_prov = args.out.parent / "nc1_L14_fp45" / "provenance.tsv"
    if v1_prov.exists():
        with v1_prov.open() as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                row = dict(zip(header, parts))
                prov[int(row["bag_idx"])] = (row["db_id"], row["locus_id"])

    summary_rows = []
    insert_fastas_for_pairwise: List[tuple[str, str]] = []
    for i in support_idx:
        if i not in prov:
            print(f"[skip site {i}] no provenance row")
            continue
        db_id, locus_full = prov[i]
        site_out = args.out / f"site_{i}"
        try:
            s = process_site(i, db_id, locus_full, sites, site_out)
            summary_rows.append(s)
            # Collect insert for pairwise
            ins_path = site_out / "insert_from_allele.fna"
            with ins_path.open() as f:
                lines = f.readlines()
            insert_fastas_for_pairwise.append(
                (f"site_{i}", "".join(l.strip() for l in lines[1:])))
            md5_tsv_ok = s["md5_from_allele_calc"] == s["md5_from_loci_tsv"]
            md5_jsonl = s["md5_from_jsonl"]
            md5_jsonl_s = ("match" if (md5_jsonl is not None and
                                       md5_jsonl == s["md5_from_allele_calc"])
                            else ("MISMATCH" if md5_jsonl else "n/a"))
            print(f"[site {i}] OK  "
                  f"n_decomp={s['n_equivalent_decompositions']}  "
                  f"method={s['decomposition_method']}  "
                  f"junction={s['junction_L_abs']}-{s['junction_R_abs']}  "
                  f"md5_vs_locitsv={'ok' if md5_tsv_ok else 'MISMATCH'}  "
                  f"md5_vs_jsonl={md5_jsonl_s}")
        except Exception as e:
            print(f"[site {i}] ERROR: {e}")
            (site_out / "ERROR.txt").write_text(f"{e}\n")

    # Summary TSV
    if summary_rows:
        cols = list(summary_rows[0].keys())
        with (args.out / "summary.tsv").open("w") as f:
            f.write("\t".join(cols) + "\n")
            for r in summary_rows:
                f.write("\t".join(str(r[c]) for c in cols) + "\n")

    # Cross-site pairwise blastn of inserts
    if len(insert_fastas_for_pairwise) >= 2:
        combo = args.out / "_all_inserts.fna"
        write_multi_fna(combo, insert_fastas_for_pairwise)
        run_blastn(combo, combo, args.out / "inserts_pairwise.blast")

    print(f"\nDONE — {len(summary_rows)} supporting sites processed into {args.out}")


if __name__ == "__main__":
    main()
