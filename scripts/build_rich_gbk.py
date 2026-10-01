"""Build an enriched GenBank per-event for a candidate.

Annotations added beyond the plain pack:
  - CDS features for Prodigal ORFs with Pfam qualifiers (reuses pfam_hits.tsv)
  - misc_feature `flank_left` / `flank_right` — the 120bp joined flank halves,
    labelled with the insertion_point_in_flank from the site record
  - misc_feature `nc_region` — the full non-coding region per site
  - misc_feature `v84_guide_anchor` — model top_p ± 14 bp (the L14 guide candidate)
    *only on events that match the probed bag*
  - stem_loop — ViennaRNA MFE-structure stems in the nc region (consecutive paired
    bases, min stem length 3)

Reads v84 top_p from `probe_summary.md` if present (parsed), else re-runs inference
once per bag (not per event — this is a bag-level property).

Usage:
  python3 scripts/build_rich_gbk.py --cds CDS01040 --out events_rich.gbk
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqFeature import SeqFeature, FeatureLocation

try:
    import RNA  # ViennaRNA
    HAS_RNA = True
except ImportError:
    HAS_RNA = False

CAND_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                        "fna_ins_discovery/candidate")
INSERT_TSV = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/insert_cds.tsv")
GFF = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/orfs.gff")
INSERTS_FNA_LIST = [
    Path("/global/scratch/users/kh36969/fna_ins_discovery/database_v3/insertions_inserts.fna"),
    Path("/global/scratch/users/kh36969/fna_ins_discovery/database/insertions_inserts.fna"),
]
ORFS_FAA = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/orfs.faa")
SITES_JSONL = Path("/global/scratch/users/kh36969/fna_ins_discovery/bags_v4/insertions_sites.jsonl")

MIN_STEM_LEN = 3


def get_events(cid: str) -> list[dict]:
    rows = []
    with INSERT_TSV.open() as f:
        header = f.readline().rstrip().split("\t")
        for line in f:
            parts = line.rstrip().split("\t")
            d = dict(zip(header, parts))
            if d["cds_cluster_id"] == cid:
                rows.append(d)
    return rows


def get_orfs_for_event(event_id: str) -> list[dict]:
    orfs = []
    with GFF.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if p[0] != event_id or p[2] != "CDS":
                continue
            attrs = {k: v for k, v in (kv.split("=", 1) for kv in p[8].split(";") if "=" in kv)}
            orfs.append({
                "start": int(p[3]), "end": int(p[4]), "strand": p[6],
                "id": attrs.get("ID", "?"), "partial": attrs.get("partial", "?"),
                "start_type": attrs.get("start_type", "?"),
                "conf": attrs.get("conf", "?"),
            })
    return sorted(orfs, key=lambda o: o["start"])


def load_fasta(path: Path, keys: set[str]) -> dict[str, str]:
    d = {}
    cur = None; buf = []
    with path.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur in keys:
                    d[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur in keys:
            d[cur] = "".join(buf)
    return d


def load_orf_faa(event_ids: list[str]) -> dict[str, str]:
    d = {}
    cur = None; buf = []
    with ORFS_FAA.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur and any(cur.startswith(e + "_") for e in event_ids):
                    d[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur and any(cur.startswith(e + "_") for e in event_ids):
            d[cur] = "".join(buf)
    return d


def load_pfam_hits(cand_dir: Path) -> dict[str, list[tuple[str, float, float]]]:
    """From the candidate's pfam_hits.tsv."""
    d: dict[str, list] = {}
    pf = cand_dir / "pfam_hits.tsv"
    if not pf.exists():
        return d
    with pf.open() as f:
        header = f.readline()
        for line in f:
            parts = line.rstrip().split("\t")
            if len(parts) < 4:
                continue
            orf_id, pfam, ev, bit = parts[0], parts[1], float(parts[2]), float(parts[3])
            d.setdefault(orf_id, []).append((pfam, ev, bit))
    return d


def load_site_records(event_ids: set[str]) -> dict[str, dict]:
    """From insertions_sites.jsonl, keyed by site_id prefix (= db_id)."""
    d = {}
    with SITES_JSONL.open() as f:
        for line in f:
            rec = json.loads(line)
            sid = rec.get("site_id", "")
            # site_id looks like "ecoli.E004469.site"; event_id = "ecoli.E004469"
            event = sid.split(".site")[0] if ".site" in sid else sid
            if event in event_ids:
                d[event] = rec
    return d


def load_target_motifs(cand_dir: Path) -> dict[str, dict]:
    """target_motifs.tsv → {event_id: {L14_argmax, L14_m, L14_orient, L14_seq}}"""
    tsv = cand_dir / "target_motifs.tsv"
    if not tsv.exists():
        return {}
    import csv
    d = {}
    with tsv.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row.get("site_id", "")
            event = sid.split(".site")[0] if ".site" in sid else sid
            if not event or not row.get("L14_argmax"):
                continue
            try:
                d[event] = {
                    "argmax": int(row["L14_argmax"]),
                    "m": int(row["L14_m"]) if row.get("L14_m") else 0,
                    "orient": row.get("L14_orient", ""),
                    "seq": row.get("L14_seq", ""),
                }
            except (ValueError, TypeError):
                continue
    return d


def parse_probe_top_p(cand_dir: Path) -> dict:
    """Parse probe_summary.md if present; else fall back to v84_probe.txt.
    Returns {top_p, top_v, guide_L14}."""
    out = {}
    ps = cand_dir / "probe_summary.md"
    if ps.exists():
        txt = ps.read_text()
        m = re.search(r"Top position: nc\[(\d+)\], score = \*\*([-\d.]+)\*\*", txt)
        if m:
            out["top_p"] = int(m.group(1))
            out["top_v"] = float(m.group(2))
        m = re.search(r"Guide candidate at top_p \(L=14\): `([^`]+)`", txt)
        if m:
            out["guide_L14"] = m.group(1)
        if out:
            return out
    pt = cand_dir / "v84_probe.txt"
    if pt.exists():
        txt = pt.read_text()
        m = re.search(r"^# TOP: pos=(\d+)\s+score=([-\d.]+)", txt, re.M)
        if m:
            out["top_p"] = int(m.group(1))
            out["top_v"] = float(m.group(2))
        m = re.search(r"guide-candidate @ top_p \(L=14\):\s*'([^']+)'", txt)
        if m:
            out["guide_L14"] = m.group(1)
    return out


def mfe_stems(seq: str) -> list[tuple[int, int, int]]:
    """Return list of (stem_5p_start, stem_5p_end, stem_3p_start) for ViennaRNA MFE stems.

    All coordinates are 1-based inclusive. A stem is a run of ≥ MIN_STEM_LEN
    consecutive paired bases.
    """
    if not HAS_RNA or len(seq) < 20:
        return []
    try:
        structure, _mfe = RNA.fold(seq)
    except Exception:
        return []
    # Walk the dot-bracket string, pair positions via stack
    pairs = []
    stack = []
    for i, c in enumerate(structure):
        if c == "(":
            stack.append(i)
        elif c == ")":
            if stack:
                j = stack.pop()
                pairs.append((j + 1, i + 1))  # 1-based
    pairs.sort()
    # Group consecutive pairs into stems
    stems = []
    if not pairs:
        return stems
    cur = [pairs[0]]
    for a, b in pairs[1:]:
        prev_a, prev_b = cur[-1]
        if a == prev_a + 1 and b == prev_b - 1:
            cur.append((a, b))
        else:
            if len(cur) >= MIN_STEM_LEN:
                stems.append((cur[0][0], cur[-1][0], cur[-1][1]))
            cur = [(a, b)]
    if len(cur) >= MIN_STEM_LEN:
        stems.append((cur[0][0], cur[-1][0], cur[-1][1]))
    return stems


def build_record(event: dict, nt: str, orfs: list[dict],
                        orf_seqs: dict[str, str],
                        pfam_hits: dict[str, list],
                        site_rec: dict | None,
                        probe_top_p: int | None, cid: str,
                        target_motif: dict | None = None) -> SeqRecord:
    rec = SeqRecord(
        Seq(nt), id=event["db_id"], name=event["db_id"][:15],
        description=f"{cid} candidate — {event['species']} ({event['phylum']}), {event['inserted_len']}bp",
        annotations={"molecule_type": "DNA", "organism": event["species"], "topology": "linear"},
    )
    rec.features.append(SeqFeature(
        FeatureLocation(0, len(nt)), type="source",
        qualifiers={"organism": [event["species"]],
                        "mol_type": ["genomic DNA"],
                        "note": [f"cds_cluster={cid}; inserted_len={event['inserted_len']}"]},
    ))

    # ORFs
    for i, o in enumerate(orfs, 1):
        strand = 1 if o["strand"] == "+" else -1
        loc = FeatureLocation(o["start"] - 1, o["end"], strand=strand)
        orf_id = f"{event['db_id']}_{i}"
        aa = orf_seqs.get(orf_id, "").rstrip("*")
        pfams = pfam_hits.get(orf_id, [])
        product = "hypothetical protein"
        if pfams:
            top = min(pfams, key=lambda x: x[1])
            product = f"{top[0]} (Pfam E={top[1]:.1e})"
        q = {"locus_tag": [orf_id], "product": [product],
                 "note": [f"ORF{i}; partial={o['partial']}; start_type={o['start_type']}; conf={o['conf']}"]}
        if aa:
            q["translation"] = [aa]
        if pfams:
            q["db_xref"] = [f"Pfam:{p[0]}" for p in pfams]
            q["inference"] = [f"protein motif:Pfam:{p[0]} E={p[1]:.1e} bit={p[2]}" for p in pfams]
        rec.features.append(SeqFeature(loc, type="CDS", qualifiers=q))

    # Flank motif annotation — from site record
    if site_rec:
        flank = site_rec.get("flank", "")
        ip = site_rec.get("insertion_point_in_flank")
        side = site_rec.get("flank_side", "")
        if flank and ip is not None:
            # The joined flank is NOT part of the insertion nt — annotate as a separate
            # free-text misc_feature noting it (coords outside the insertion). Use
            # position 0..0 with a note.
            q = {
                "note": [f"PRE_INSERTION_FLANK: len={len(flank)} bp, side={side}, "
                             f"insertion_point_in_flank={ip} — see /flank_left and /flank_right"],
                "flank_left": [flank[:ip]],
                "flank_right": [flank[ip:]],
            }
            if target_motif:
                a = target_motif["argmax"]
                L = len(target_motif.get("seq", "")) or 14
                rel = a - ip
                q["note"].append(
                    f"V84_TARGET_ANCHOR: pos_in_flank={a} ({rel:+d} bp from junction), "
                    f"L={L}, m_matches={target_motif['m']}/{L}, orient={target_motif['orient']}, "
                    f"seq='{target_motif['seq']}' — model's predicted target site in flank")
                q["target_anchor_pos"] = [str(a)]
                q["target_anchor_rel_junction"] = [str(rel)]
                q["target_anchor_seq"] = [target_motif["seq"]]
                q["target_anchor_m"] = [str(target_motif["m"])]
                q["target_anchor_orient"] = [target_motif["orient"]]
            rec.features.append(SeqFeature(
                FeatureLocation(0, 0), type="misc_feature", qualifiers=q,
            ))

        # nc region(s) — subrange of the full insertion
        # The nc region is stored in the site record as noncoding_regions = [{sequence, ...}, ...]
        # Find its start in the insertion by substring match
        for ri, reg in enumerate(site_rec.get("noncoding_regions", []), 1):
            seq_nc = reg.get("sequence", "") if isinstance(reg, dict) else str(reg)
            if not seq_nc:
                continue
            idx = nt.find(seq_nc)
            if idx < 0:
                idx = nt.find(seq_nc[:40])  # fuzzy — first 40bp
            if idx >= 0:
                nc_start, nc_end = idx, idx + len(seq_nc)
                rec.features.append(SeqFeature(
                    FeatureLocation(nc_start, nc_end), type="misc_feature",
                    qualifiers={"note": [f"NON_CODING_REGION_{ri}; len={len(seq_nc)}bp — "
                                                 f"candidate guide-RNA / scaffold locus"],
                                    "function": ["noncoding"]},
                ))

                # v84 top_p anchor (if we have one from the probe)
                if probe_top_p is not None and 0 <= probe_top_p < len(seq_nc):
                    anchor_start = nc_start + probe_top_p
                    anchor_end = anchor_start + 14  # L14 window
                    if anchor_end <= nc_end:
                        rec.features.append(SeqFeature(
                            FeatureLocation(anchor_start, anchor_end), type="misc_feature",
                            qualifiers={"note": [f"V84_GUIDE_ANCHOR at nc[{probe_top_p}] "
                                                         f"— model's top-ranked guide candidate (L=14)"],
                                            "function": ["putative RNA-guide anchor"]},
                        ))

                # ViennaRNA stems
                if HAS_RNA:
                    stems = mfe_stems(seq_nc)
                    for (s5_start, s5_end, s3_end) in stems:
                        # 5' arm
                        a = nc_start + s5_start - 1
                        b = nc_start + s5_end
                        rec.features.append(SeqFeature(
                            FeatureLocation(a, b), type="stem_loop",
                            qualifiers={"note": [f"MFE stem 5'-arm; paired to nt {nc_start + s3_end}-"
                                                         f"{nc_start + s3_end - (s5_end - s5_start)}"]},
                        ))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cds", required=True)
    ap.add_argument("--out", default="events_rich.gbk",
                        help="Output filename inside the candidate folder")
    ap.add_argument("--all", action="store_true",
                        help="Process every candidate folder under CAND_ROOT")
    args = ap.parse_args()

    cids = []
    if args.all:
        for d in sorted(CAND_ROOT.iterdir()):
            if d.is_dir() and d.name.startswith("CDS"):
                cids.append(d.name)
    else:
        cids = [args.cds]

    for cid in cids:
        folder = CAND_ROOT / cid
        if not folder.exists():
            print(f"[SKIP {cid}] no folder")
            continue

        events = get_events(cid)
        event_ids = [e["db_id"] for e in events]
        print(f"\n## {cid}: {len(events)} events")

        # Load nt
        nt_by_event = {}
        for src in INSERTS_FNA_LIST:
            if not src.exists():
                continue
            found = load_fasta(src, set(event_ids) - set(nt_by_event))
            nt_by_event.update(found)

        orfs_by_event = {e: get_orfs_for_event(e) for e in event_ids}
        orf_faa = load_orf_faa(event_ids)
        pfam_hits = load_pfam_hits(folder)
        site_records = load_site_records(set(event_ids))

        probe = parse_probe_top_p(folder)
        top_p = probe.get("top_p")
        if top_p is not None:
            print(f"  v84 probe top_p = {top_p}")

        target_motifs = load_target_motifs(folder)
        if target_motifs:
            print(f"  target motifs loaded for {len(target_motifs)} sites")

        records = []
        for e in events:
            nt = nt_by_event.get(e["db_id"], "")
            if not nt:
                continue
            rec = build_record(e, nt, orfs_by_event[e["db_id"]], orf_faa,
                                       pfam_hits, site_records.get(e["db_id"]),
                                       top_p, cid,
                                       target_motif=target_motifs.get(e["db_id"]))
            records.append(rec)

        out_path = folder / args.out
        with out_path.open("w") as f:
            SeqIO.write(records, f, "genbank")
        n_feats = sum(len(r.features) for r in records)
        print(f"  wrote {len(records)} records, {n_feats} features total → {out_path}")


if __name__ == "__main__":
    main()
