"""Build a candidate package under a per-CDS subfolder.

Usage:
  python3 scripts/build_candidate_pack.py --cds CDS01040 --out-root /path/to/candidate

Output tree:
  <out-root>/<CDS>/
    README.md              — summary of candidate: family, events, v84 signal
    events.fna             — nucleotide sequences of all events
    orfs.faa               — protein sequences of all ORFs (from Prodigal)
    orfs.tsv               — ORF metadata table (coords/strand/partial/pfam)
    intergenic.fna         — intergenic regions (candidate guide RNA loci)
    pfam_hits.tsv          — Pfam hits per ORF (from prior hmmscan)
    v84_probe.txt          — copy of interpretability probe if present
    events.gbk             — one GenBank record per event, ORFs annotated
"""
from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation, CompoundLocation

INSERT_TSV = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/insert_cds.tsv")
GFF = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/orfs.gff")
INSERTS_FNA_LIST = [
    Path("/global/scratch/users/kh36969/fna_ins_discovery/database_v3/insertions_inserts.fna"),
    Path("/global/scratch/users/kh36969/fna_ins_discovery/database/insertions_inserts.fna"),
]
ORFS_FAA = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/orfs.faa")
PFAM_DOMTBL_TOP = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/pfam_domtbl.tsv")


def get_events(cid: str) -> list[dict]:
    events = []
    with INSERT_TSV.open() as f:
        header = f.readline().rstrip().split("\t")
        for line in f:
            parts = line.rstrip().split("\t")
            row = dict(zip(header, parts))
            if row["cds_cluster_id"] == cid:
                events.append(row)
    return events


def get_orfs_for_event(event_id: str) -> list[dict]:
    orfs = []
    with GFF.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if parts[0] != event_id or parts[2] != "CDS":
                continue
            attrs = {}
            for kv in parts[8].split(";"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    attrs[k] = v
            orfs.append({
                "seqid": parts[0],
                "start": int(parts[3]),
                "end": int(parts[4]),
                "strand": parts[6],
                "id": attrs.get("ID", "?"),
                "partial": attrs.get("partial", "?"),
                "start_type": attrs.get("start_type", "?"),
                "conf": attrs.get("conf", "?"),
                "score": attrs.get("score", "?"),
            })
    return sorted(orfs, key=lambda o: o["start"])


def load_fasta(path: Path, keys: set[str]) -> dict[str, str]:
    d = {}
    cur = None; buf = []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur is not None and cur in keys:
                    d[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None and cur in keys:
            d[cur] = "".join(buf)
    return d


def load_orf_faa(event_ids: list[str]) -> dict[str, str]:
    """Return orf_id (e.g., "ecoli.E004469_1") -> protein sequence."""
    d = {}
    cur = None; buf = []
    with ORFS_FAA.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur is not None and any(cur.startswith(e + "_") for e in event_ids):
                    d[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur is not None and any(cur.startswith(e + "_") for e in event_ids):
            d[cur] = "".join(buf)
    return d


def load_pfam_domtbl(event_ids: list[str], eval_thresh: float = 1e-3) -> dict[str, list[tuple[str, float, float]]]:
    """orf_id -> [(pfam_name, evalue, bitscore), ...]"""
    d: dict[str, list[tuple[str, float, float]]] = {}
    if not PFAM_DOMTBL_TOP.exists():
        return d
    with PFAM_DOMTBL_TOP.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 20:
                continue
            pfam = p[0]
            qname = p[3]
            evalue = float(p[6])
            bit = float(p[7])
            if evalue > eval_thresh:
                continue
            if any(qname.startswith(e + "_") or qname.startswith(e + "__") for e in event_ids):
                d.setdefault(qname, []).append((pfam, evalue, bit))
    return d


def intergenics_for_event(orfs: list[dict], seq_len: int, min_len: int = 20) -> list[dict]:
    ig = []
    if not orfs:
        return [{"start": 1, "end": seq_len, "label": "all_intergenic"}]
    prev_end = 0
    for i, o in enumerate(orfs):
        gap_start = prev_end + 1
        gap_end = o["start"] - 1
        if gap_end - gap_start + 1 >= min_len:
            label = f"before_ORF{i+1}" if i == 0 else f"between_ORF{i}_ORF{i+1}"
            ig.append({"start": gap_start, "end": gap_end, "label": label})
        prev_end = max(prev_end, o["end"])
    if seq_len - prev_end >= min_len:
        ig.append({"start": prev_end + 1, "end": seq_len, "label": "after_last_ORF"})
    return ig


def build_gbk_record(event_row: dict, nt: str, orfs: list[dict], orf_seqs: dict[str, str],
                             pfam_hits: dict[str, list[tuple[str, float, float]]],
                             intergenic: list[dict], cid: str) -> SeqRecord:
    """Assemble one GenBank record with source + CDS + misc_feature features."""
    seq = Seq(nt)
    rec = SeqRecord(
        seq,
        id=event_row["db_id"],
        name=event_row["db_id"][:15],
        description=f"{cid} candidate insertion — {event_row['species']} ({event_row['phylum']}), {event_row['inserted_len']}bp, dominant_orf={event_row['dominant_orf']}",
        annotations={"molecule_type": "DNA", "organism": event_row["species"],
                             "topology": "linear"},
    )
    rec.features.append(SeqFeature(
        FeatureLocation(0, len(nt)),
        type="source",
        qualifiers={"organism": [event_row["species"]],
                        "mol_type": ["genomic DNA"],
                        "note": [f"cds_cluster={cid}; inserted_len={event_row['inserted_len']}"]},
    ))

    for i, o in enumerate(orfs, 1):
        strand = 1 if o["strand"] == "+" else -1
        loc = FeatureLocation(o["start"] - 1, o["end"], strand=strand)
        orf_id = f"{event_row['db_id']}_{i}"
        aa_seq = orf_seqs.get(orf_id, "").rstrip("*")
        pfams = pfam_hits.get(orf_id, [])
        product = "hypothetical protein"
        if pfams:
            top = min(pfams, key=lambda x: x[1])
            product = f"{top[0]} (Pfam E={top[1]:.1e})"
        qualifiers = {
            "locus_tag": [orf_id],
            "product": [product],
            "note": [f"ORF{i}; partial={o['partial']}; start_type={o['start_type']}; conf={o['conf']}"],
        }
        if aa_seq:
            qualifiers["translation"] = [aa_seq]
        if pfams:
            qualifiers["db_xref"] = [f"Pfam:{p[0]}" for p in pfams]
            qualifiers["inference"] = [f"protein motif:Pfam:{p[0]} E={p[1]:.1e} bit={p[2]}" for p in pfams]
        rec.features.append(SeqFeature(loc, type="CDS", qualifiers=qualifiers))

    for ig in intergenic:
        loc = FeatureLocation(ig["start"] - 1, ig["end"])
        rec.features.append(SeqFeature(
            loc, type="misc_feature",
            qualifiers={"note": [f"{ig['label']} — candidate guide-RNA / ncRNA locus"]},
        ))
    return rec


def write_readme(root: Path, cid: str, events: list[dict], pfam_hits: dict, orfs_by_event: dict) -> None:
    lines = [f"# {cid} — v84 candidate\n"]
    lines.append("## v84 discovery-run metadata\n")
    lines.append(f"- CDS cluster ID: `{cid}`")
    lines.append(f"- Number of events: {len(events)}")
    species = sorted({e["species"] for e in events})
    lines.append(f"- Species: {', '.join(species)}")
    phyla = sorted({e["phylum"] for e in events})
    lines.append(f"- Phyla: {', '.join(phyla)}")
    lens = sorted(int(e["inserted_len"]) for e in events)
    lines.append(f"- Insertion length: median={lens[len(lens)//2]}bp, range {lens[0]}-{lens[-1]}bp")

    n_orfs = [int(e["n_orfs"]) for e in events]
    lines.append(f"- ORFs per event: {sorted(set(n_orfs))} (modal architecture)")

    lines.append("\n## ORF Pfam summary (from pfam_domtbl.tsv, E ≤ 1e-3)\n")
    seen_pfams: dict[str, list[str]] = {}
    for event in events:
        for i, o in enumerate(orfs_by_event.get(event["db_id"], []), 1):
            orf_id = f"{event['db_id']}_{i}"
            for pf, ev, bit in pfam_hits.get(orf_id, []):
                seen_pfams.setdefault(pf, []).append(f"{orf_id} (E={ev:.1e})")
    if seen_pfams:
        for pf, hits in sorted(seen_pfams.items()):
            lines.append(f"- **{pf}** — {len(hits)} hits across events")
            for h in hits[:5]:
                lines.append(f"  - {h}")
    else:
        lines.append("- (no Pfam hits at E ≤ 1e-3)")

    lines.append("\n## Files in this folder\n")
    lines.append("- `events.fna` — 6bp-line nucleotide fasta of all events")
    lines.append("- `orfs.faa` — protein sequences of all Prodigal ORFs across events")
    lines.append("- `orfs.tsv` — per-ORF metadata (coords, strand, Pfam)")
    lines.append("- `intergenic.fna` — intergenic regions per event (candidate guide-RNA loci)")
    lines.append("- `events.gbk` — GenBank records with CDS + misc_feature annotations")
    lines.append("- `pfam_hits.tsv` — Pfam hits per ORF")
    lines.append("- `v84_probe.txt` — v84 interpretability probe (if available)")

    (root / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cds", required=True, help="CDS cluster ID (e.g., CDS01040)")
    ap.add_argument("--out-root", default="/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery/candidate")
    args = ap.parse_args()

    cid = args.cds
    out = Path(args.out_root) / cid
    out.mkdir(parents=True, exist_ok=True)
    print(f"# candidate pack for {cid} → {out}")

    events = get_events(cid)
    if not events:
        print(f"[FAIL] no events found for {cid}")
        sys.exit(1)
    print(f"  {len(events)} events")

    event_ids = [e["db_id"] for e in events]
    nt_by_event = {}
    for src in INSERTS_FNA_LIST:
        if not src.exists():
            continue
        found = load_fasta(src, set(event_ids) - set(nt_by_event.keys()))
        nt_by_event.update(found)
        print(f"  loaded {len(found)} nt seqs from {src.name} ({src.parent.name})")
    print(f"  total: {len(nt_by_event)} event nt sequences")

    orfs_by_event = {e: get_orfs_for_event(e) for e in event_ids}
    total_orfs = sum(len(v) for v in orfs_by_event.values())
    print(f"  loaded {total_orfs} Prodigal ORFs across events")

    orf_faa = load_orf_faa(event_ids)
    print(f"  loaded {len(orf_faa)} ORF protein sequences")

    pfam_hits = load_pfam_domtbl(event_ids)
    print(f"  loaded pfam hits for {len(pfam_hits)} ORFs")

    # events.fna
    with (out / "events.fna").open("w") as f:
        for e in events:
            nt = nt_by_event.get(e["db_id"], "")
            if not nt:
                continue
            f.write(f">{e['db_id']}  species={e['species']}  cluster={cid}  len={len(nt)}\n")
            for i in range(0, len(nt), 60):
                f.write(nt[i:i+60] + "\n")

    # orfs.faa
    with (out / "orfs.faa").open("w") as f:
        for event_id in event_ids:
            for i, o in enumerate(orfs_by_event[event_id], 1):
                orf_id = f"{event_id}_{i}"
                seq = orf_faa.get(orf_id, "").rstrip("*")
                if not seq:
                    continue
                pfam_str = ""
                if orf_id in pfam_hits:
                    pfam_str = "  pfam=" + ",".join(f"{p[0]}(E={p[1]:.1e})" for p in pfam_hits[orf_id])
                f.write(f">{orf_id}  strand={o['strand']}  nt[{o['start']}..{o['end']}]  partial={o['partial']}{pfam_str}\n")
                for j in range(0, len(seq), 60):
                    f.write(seq[j:j+60] + "\n")

    # orfs.tsv
    with (out / "orfs.tsv").open("w") as f:
        f.write("orf_id\tevent\tstart\tend\tstrand\taa_len\tpartial\tstart_type\tconf\tpfam_hits\n")
        for event_id in event_ids:
            for i, o in enumerate(orfs_by_event[event_id], 1):
                orf_id = f"{event_id}_{i}"
                aa = len(orf_faa.get(orf_id, "").rstrip("*"))
                pf = ";".join(f"{p[0]}:E={p[1]:.1e}:bit={p[2]}" for p in pfam_hits.get(orf_id, []))
                f.write(f"{orf_id}\t{event_id}\t{o['start']}\t{o['end']}\t{o['strand']}\t{aa}\t{o['partial']}\t{o['start_type']}\t{o['conf']}\t{pf}\n")

    # intergenic.fna
    with (out / "intergenic.fna").open("w") as f:
        for e in events:
            nt = nt_by_event.get(e["db_id"], "")
            if not nt:
                continue
            igs = intergenics_for_event(orfs_by_event[e["db_id"]], len(nt))
            for ig in igs:
                sub = nt[ig["start"]-1:ig["end"]]
                f.write(f">{e['db_id']}__{ig['label']}  nt[{ig['start']}..{ig['end']}]  len={len(sub)}\n")
                for i in range(0, len(sub), 60):
                    f.write(sub[i:i+60] + "\n")

    # pfam_hits.tsv
    with (out / "pfam_hits.tsv").open("w") as f:
        f.write("orf_id\tpfam\tevalue\tbitscore\n")
        for orf_id, hits in sorted(pfam_hits.items()):
            for p, ev, bit in hits:
                f.write(f"{orf_id}\t{p}\t{ev}\t{bit}\n")

    # GBK
    records = []
    for e in events:
        nt = nt_by_event.get(e["db_id"], "")
        if not nt:
            continue
        orfs = orfs_by_event[e["db_id"]]
        igs = intergenics_for_event(orfs, len(nt))
        rec = build_gbk_record(e, nt, orfs, orf_faa, pfam_hits, igs, cid)
        records.append(rec)
    with (out / "events.gbk").open("w") as f:
        SeqIO.write(records, f, "genbank")
    print(f"  wrote {len(records)} GenBank records")

    # copy v84 probe if available
    probe_src = Path(f"/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/v84_prb_26551309.out")
    if cid == "CDS01040" and probe_src.exists():
        (out / "v84_probe.txt").write_text(probe_src.read_text())
        print(f"  copied v84 probe: {probe_src.name}")

    write_readme(out, cid, events, pfam_hits, orfs_by_event)
    print(f"# done: {out}")


if __name__ == "__main__":
    main()
