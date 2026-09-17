"""Build v2 Durrant real-genomic-flank + IS621-nc JSONLs.

Produces:
  durrant_wt_realbg_v2.jsonl        (173 records — WT insertion sites, SuppTable3)
  durrant_programmed_realbg_v2.jsonl (168 records — Programmed insertion sites, SuppTable3)

Each record — v7-compatible schema:
  - inputs.flank         : 120bp real E. coli BL21(DE3) target-site sequence
                            (60bp upstream + 60bp downstream around insertion,
                             reverse-complemented if strand=='-')
  - inputs.noncoding_regions:
        [nc_region_1 (193bp, 5' of tnpA — contains bridge RNA),
         nc_region_2 (107bp, 3' of tnpA)]
  - labels.is_positive   : True (all Durrant insertions are positive events)
  - labels.arch.nc_multi_region_scoring: "concat_with_N_spacer"
  - labels.arch.n_sites  : 1 (one target site per bag)
  - generator_metadata.data_source : "durrant_wt_realbg_v2" / "durrant_programmed_realbg_v2"

CONSTRUCTION VERIFIED (2026-09-13):
  1. 60+60 flank extraction: verified on 5 sites; Durrant's 11bp AND 14bp
     recognition sequences appear at asm[51:62] / asm[51:65] of the 120bp
     assembled flank (junction at position 60). 173/173 WT flanks match ref.
  2. IS621 CDS localization: element frame +1, aa 64→390 (326 aa),
     cds_start=193, cds_end=1174 (incl stop TAA). Non-coding split: 193bp
     5' region contains the 177bp bridge RNA; 107bp 3' region.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

import openpyxl
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq


REPO = Path("/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")
DATA_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor")
CONTIG = DATA_ROOT / "IS110_gold/annotation/NZ_CP053602.1.fna"
SUPP3 = DATA_ROOT / "IS110_gold/annotation/2023-09-16026B-s3/2023-09-16026B-SupplementaryTable3.xlsx"
CDS_SPLIT_JSON = DATA_ROOT / "IS110_gold/annotation/is621_cds_split.json"

OUT_DIR = DATA_ROOT / "IS110_gold/inference"
OUT_WT = OUT_DIR / "durrant_wt_realbg_v2.jsonl"
OUT_PROG = OUT_DIR / "durrant_programmed_realbg_v2.jsonl"

GIT_COMMIT = "post-2026-09-13-v2"   # updated once we commit
BUILD_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_contig():
    return str(next(SeqIO.parse(CONTIG, "fasta")).seq).upper()


def load_is621_nc():
    d = json.loads(CDS_SPLIT_JSON.read_text())
    return d["nc_region_1"], d["nc_region_2"]


def extract_120bp_flank(contig: str, cs_1idx: int, ce_1idx: int, strand: str) -> tuple[str, int]:
    """Return (assembled_120bp, junction_pos_in_asm=60).
    See dde_multi_region_verify + smoke test — 60bp upstream + 60bp downstream
    around core midpoint, RC if strand=='-'."""
    mid_1idx = (cs_1idx + ce_1idx) // 2
    mid_0idx = mid_1idx - 1
    plus_120 = contig[mid_0idx - 60 : mid_0idx + 60]
    if strand == "-":
        asm = str(Seq(plus_120).reverse_complement())
    else:
        asm = plus_120
    return asm, 60


def make_record(row, contig, nc1, nc2, family_tag: str) -> dict:
    cs = int(row["Insertion Core Start"])
    ce = int(row["Insertion Core End"])
    strand = row["Insertion Strand"]
    d11 = str(row["Genome Insertion Site Sequence 11bp"])
    d14 = str(row["Genome Insertion Site Sequence 14bp"])
    tbl11 = str(row["TBL Target 11bp"])
    dbl11 = str(row["DBL Target 11bp"])
    brna = str(row["Bridge RNA ID"])
    contig_id = str(row["Contig ID"])

    flank_120, junction_pos = extract_120bp_flank(contig, cs, ce, strand)
    assert len(flank_120) == 120

    # Sanity: Durrant's 11bp SHOULD be present at asm[51:62]
    d11_pos = flank_120.find(d11)
    if d11_pos != 51:
        # Rare Ns or edge cases — do not skip, just record
        pass

    site_id = f"durrant_{family_tag}_{brna}_{contig_id}_{cs}_{ce}_{strand}"
    return {
        "site_id":         site_id,
        "transposase_id": site_id,           # one bag per site (n_sites=1)
        "ncrna_id":        f"{site_id}_ncrna",
        "inputs": {
            "flank":                flank_120,
            "noncoding_regions":    [nc1, nc2],   # v7 multi-region
        },
        "labels": {
            "is_positive":                    True,
            "target_position_in_flank":       [d11_pos, d11_pos + len(d11)] if d11_pos != -1 else None,
            "target_dna":                     d11,
            "guide_length":                   11,
            "n_mismatches":                    None,
            "mismatch_positions":              None,
            "active_noncoding_index":          0,       # legacy field; multi-region
            "num_noncoding_regions":           2,
            "guide_span_in_active_noncoding": None,
            "ncrna_length":                    len(nc1) + len(nc2),
            "match_orientation":               None,
            "site_class":                      f"durrant_{family_tag}",
            "arch": {
                "n_sites":                       1,
                "nc_multi_region_scoring":       "concat_with_N_spacer",
                "nc_homology_rate":              1.0,
            },
            # Provenance (train-only labels; ignored by loader whitelist)
            "durrant_bridge_rna_id":           brna,
            "durrant_contig_id":               contig_id,
            "durrant_insertion_core_start":    cs,
            "durrant_insertion_core_end":      ce,
            "durrant_insertion_strand":        strand,
            "durrant_genome_insertion_site_11bp": d11,
            "durrant_genome_insertion_site_14bp": d14,
            "durrant_tbl_target_11bp":         tbl11,
            "durrant_dbl_target_11bp":         dbl11,
            "junction_position_in_flank":      junction_pos,
        },
        "generator_metadata": {
            "data_source":                    f"durrant_{family_tag}_realbg_v2",
            "build_date":                     BUILD_DATE,
            "generator_version_or_commit":    GIT_COMMIT,
            "flank_pool_source":              "NZ_CP053602.1 (E. coli BL21(DE3))",
            "flank_side":                     "target_site_before_insertion",
            "nc_source":                      "durrant_supp1_IS621_full_element_cds_split",
            "nc_split_summary":                f"nc_region_1 {len(nc1)}bp (5' of tnpA, contains bridge RNA), nc_region_2 {len(nc2)}bp (3' of tnpA)",
        },
    }


def build_sheet(sheet_name: str, family_tag: str, out_path: Path,
                    contig: str, nc1: str, nc2: str):
    df = pd.read_excel(SUPP3, sheet_name=sheet_name)
    # keep rows with Contig ID
    df = df[df["Contig ID"].notna()].reset_index(drop=True)
    print(f"[build] {sheet_name}: {len(df)} rows")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_11bp_at_51 = 0
    n_11bp_missing = 0
    with open(out_path, "w") as f:
        for _, row in df.iterrows():
            rec = make_record(row, contig, nc1, nc2, family_tag)
            f.write(json.dumps(rec) + "\n")
            n_written += 1
            pos = rec["labels"].get("target_position_in_flank")
            if pos and pos[0] == 51:
                n_11bp_at_51 += 1
            elif pos is None or pos == [None, None]:
                n_11bp_missing += 1
    print(f"[build]   wrote {n_written} records to {out_path}")
    print(f"[build]   Durrant 11bp at asm pos 51 (expected): {n_11bp_at_51}/{n_written}")
    print(f"[build]   Durrant 11bp not found in asm:        {n_11bp_missing}/{n_written}")
    return n_written


def main():
    print(f"[main] loading contig + cds split ...")
    contig = load_contig()
    nc1, nc2 = load_is621_nc()
    print(f"[main] contig len {len(contig):,}  nc1 {len(nc1)}bp  nc2 {len(nc2)}bp")

    n_wt = build_sheet("WT Genomic Insertion Sites",  "wt",   OUT_WT, contig, nc1, nc2)
    n_pr = build_sheet("Programmed Genomic Ins. Sites","programmed", OUT_PROG, contig, nc1, nc2)
    print()
    print(f"[main] DONE — {n_wt + n_pr} records total")
    print(f"[main]   WT:         {OUT_WT}  ({n_wt})")
    print(f"[main]   Programmed: {OUT_PROG} ({n_pr})")


if __name__ == "__main__":
    main()
