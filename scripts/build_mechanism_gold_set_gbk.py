#!/usr/bin/env python3
"""Build a reviewable, family-mechanism gold set and annotated representative GBKs.

This encodes experimentally established *family mechanisms* as labels. It does
not promote a particular natural insertion to an experimentally validated
guide/target pair; that requires event-level guide and junction evidence.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord


MECHANISMS = {
    "IS110": ("GUIDED_INSERTION_RNA", "bridge-RNA-guided recombination/insertion"),
    "group_II_intron": ("GUIDED_INSERTION_RNA", "EBS/IBS RNA-guided retrohoming"),
    "IS200/IS605-TnpA": ("GUIDED_INSERTION_DNA", "TnpA DNA-guide/subterminal-hairpin target recognition"),
    "TnpB-like": ("GUIDED_NUCLEASE_NOT_INSERTION", "RNA-guided DNA cleavage; insertion guidance not established"),
    "IS1": ("NON_GUIDED_IS_CONTROL", "canonical IS1 transposition control"),
    "IS3": ("NON_GUIDED_IS_CONTROL", "canonical IS3 transposition control"),
    "IS5": ("NON_GUIDED_IS_CONTROL", "canonical IS5 transposition control"),
}


def read_tsv(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t")


def write_tsv(path: Path, rows: list[dict], columns: list[str]):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def safe_int(x, default=0):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def mechanism_label(family: str):
    return MECHANISMS.get(family, ("UNRESOLVED", "no gold mechanism label"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotation-root", type=Path, default=Path("../../output/family_annotation_all6139"))
    ap.add_argument("--family-evidence", type=Path, default=Path("evidence/_family_annotation/supported_target_and_control_bags.tsv"))
    ap.add_argument("--scan-status", type=Path, default=Path("evidence/_positive_control_review/family_positive_status.tsv"))
    ap.add_argument("--outdir", type=Path, default=Path("evidence/_guided_gold_set"))
    args = ap.parse_args()
    root, out = args.annotation_root, args.outdir
    out.mkdir(parents=True, exist_ok=True)

    bag_rows = {r["bag_id"]: r for r in read_tsv(args.family_evidence)}
    scan_rows = {r["bag_id"]: r for r in read_tsv(args.scan_status)}
    representatives = [r for r in read_tsv(root / "representative_annotation_evidence.tsv") if r["bag_id"] in bag_rows]
    rep_by_bag = {r["bag_id"]: r for r in representatives}
    missing = sorted(set(bag_rows) - set(rep_by_bag))
    if missing:
        raise SystemExit(f"Missing representative annotations for: {missing}")

    seqs = {r.id: str(r.seq).upper() for r in SeqIO.parse(root / "insert_sequences.fna", "fasta")}
    source = {r["db_id"]: r for r in read_tsv(root / "source_events.tsv")}
    genes = defaultdict(list)
    for r in read_tsv(root / "gene_origins.tsv"):
        genes[r["insert_id"]].append(r)
    profiles = defaultdict(list)
    for r in read_tsv(root / "profile_hits.tsv"):
        profiles[r["protein_id"]].append(r)
    roles = {r["protein_id"]: r for r in read_tsv(root / "protein_role_annotations.tsv")}
    rna_hits = defaultdict(list)
    for r in read_tsv(root / "rna_hits.tsv"):
        rna_hits[r["insert_id"]].append(r)
    member_annotations = {r["insert_id"]: r for r in read_tsv(root / "insert_family_annotations.tsv")}

    def annotated_record(event, bag_id, family, label, nt, scope):
        insert_id = event["insert_id"]
        member_label = member_annotations.get(insert_id, {}).get("label", "UNKNOWN")
        rec = SeqRecord(Seq(nt), id=event["db_id"], name=event["db_id"][:16],
                        description=f"{bag_id} {scope}; {family}; member={member_label}; {label}")
        rec.annotations.update({"molecule_type": "DNA", "organism": event["species"], "topology": "linear"})
        rec.features.append(SeqFeature(FeatureLocation(0, len(nt)), type="source", qualifiers={
            "organism": [event["species"]], "mol_type": ["genomic DNA"],
            "note": [f"bag={bag_id}; insert_id={insert_id}; bag_family={family}; insert_member_label={member_label}; mechanism_gold={label}; scope={scope}"]}))
        rec.features.append(SeqFeature(FeatureLocation(0, len(nt)), type="misc_feature", qualifiers={
            "note": [f"MECHANISM GOLD LABEL: {label}; {mechanism_label(family)[1]}",
                     "Family/member annotation is not an experimentally validated guide-target pair for this natural insertion."]}))
        for g in sorted(genes[insert_id], key=lambda x: safe_int(x["lo"])):
            lo, hi = safe_int(g["lo"]), safe_int(g["hi"])
            strand = 1 if g["strand"] in ("1", "+") else -1
            pid = g["protein_id"]
            role = roles.get(pid, {}).get("role", "unannotated")
            phits = profiles.get(pid, [])
            models = sorted({f"{h['library']}:{h['model']}[{h['accession']}]" for h in phits})
            q = {"locus_tag": [pid], "product": [role], "note": [f"gene_idx={g['gene_idx']}; partial_begin={g['partial_begin']}; partial_end={g['partial_end']}; models={';'.join(models) or 'none'}"]}
            if pid in roles:
                q["inference"] = [f"family/profile annotation: {roles[pid].get('reason','')}".strip()]
            rec.features.append(SeqFeature(FeatureLocation(lo, hi, strand=strand), type="CDS", qualifiers=q))
            for h in phits:
                aa_lo = max(0, safe_int(h.get("ali_from")) - 1)
                aa_hi = max(aa_lo + 1, safe_int(h.get("ali_to")))
                if strand == 1:
                    d_lo, d_hi = lo + 3 * aa_lo, min(hi, lo + 3 * aa_hi)
                else:
                    d_lo, d_hi = max(lo, hi - 3 * aa_hi), hi - 3 * aa_lo
                if d_lo < d_hi:
                    rec.features.append(SeqFeature(FeatureLocation(d_lo, d_hi, strand=strand), type="misc_feature", qualifiers={
                        "note": [f"protein profile {h['library']}:{h['model']} {h['accession']}; E={h['domain_i_evalue']}; aa={h['ali_from']}..{h['ali_to']}"]}))
        for h in rna_hits[insert_id]:
            a, b = safe_int(h["seq_from"]), safe_int(h["seq_to"])
            if not a or not b:
                continue
            lo, hi = min(a, b) - 1, max(a, b)
            strand = 1 if h["strand"] == "+" else -1
            rec.features.append(SeqFeature(FeatureLocation(lo, hi, strand=strand), type="ncRNA", qualifiers={
                "product": [h["model"]], "note": [f"Rfam {h['rfam_accession']}; score={h['score']}; E={h['evalue']}; model_coords={h['model_from']}..{h['model_to']}; truncation={h['truncation']}"]}))
        return rec

    gold_rows = []
    records = []
    for bag_id in sorted(bag_rows):
        fam = bag_rows[bag_id]["family_label"]
        label, mechanism = mechanism_label(fam)
        rep = rep_by_bag[bag_id]
        event = source.get(rep["source_event"])
        if not event or event["insert_id"] != rep["insert_id"]:
            raise SystemExit(f"Representative/source join failed for {bag_id}: {rep['source_event']}")
        insert_id = event["insert_id"]
        nt = seqs.get(insert_id)
        if nt is None or len(nt) != safe_int(event["length"]):
            raise SystemExit(f"Missing/length-mismatched sequence for {bag_id} / {insert_id}")
        digest_prefix = hashlib.md5(nt.encode()).hexdigest()[:len(event["insert_md5"])]
        if digest_prefix != event["insert_md5"]:
            raise SystemExit(f"Sequence hash mismatch for {bag_id} / {insert_id}")

        records.append(annotated_record(event, bag_id, fam, label, nt, "bag_representative"))
        status = scan_rows.get(bag_id, {})
        gold_rows.append({
            "bag_id": bag_id, "family": fam, "mechanism_gold_label": label,
            "mechanism_description": mechanism, "gold_scope": "independently_supported_family_mechanism",
            "event_level_guide_target_verified": "no", "event_status": "unverified_not_negative",
            "annotation_status": bag_rows[bag_id].get("annotation_status", ""),
            "member_label_fraction": bag_rows[bag_id].get("label_member_fraction", ""),
            "representative_event": event["db_id"], "insert_id": insert_id,
            "insert_length": len(nt), "dominant_role": rep.get("dominant_role", ""),
            "dominant_protein_id": rep.get("protein_id", ""),
            "reviewed_reference": rep.get("reviewed_reference", ""),
            "reviewed_identity": rep.get("reviewed_identity", ""),
            "reviewed_query_coverage": rep.get("reviewed_query_coverage", ""),
            "rna_models": rep.get("rna_structure_note", "") or rep.get("rna_hits_json", ""),
            "scan_status": status.get("signal_input_status", ""),
            "candidate_scan_score": status.get("raw_score", ""),
            "scan_score_is_ground_truth": "no",
        })

    SeqIO.write(records, out / "representative_family_mechanisms.gbk", "genbank")

    cohort_events = [r for r in read_tsv(root / "source_events.tsv") if r["bag_id"] in bag_rows]
    member_rows = []
    member_gbk_path = out / "all_member_events.gbk.gz"
    with gzip.open(member_gbk_path, "wt", encoding="utf-8", newline="") as fh:
        for event in cohort_events:
            bag_id = event["bag_id"]
            family = bag_rows[bag_id]["family_label"]
            bag_label, _ = mechanism_label(family)
            insert_id = event["insert_id"]
            nt = seqs.get(insert_id)
            if nt is None or len(nt) != safe_int(event["length"]):
                raise SystemExit(f"Missing/length-mismatched member sequence for {event['db_id']}")
            digest_prefix = hashlib.md5(nt.encode()).hexdigest()[:len(event["insert_md5"])]
            if digest_prefix != event["insert_md5"]:
                raise SystemExit(f"Member sequence hash mismatch for {event['db_id']}")
            member = member_annotations.get(insert_id, {})
            member_family = member.get("label", "UNKNOWN")
            member_label, _ = mechanism_label(member_family)
            member_rows.append({
                "event_id": event["db_id"], "bag_id": bag_id, "insert_id": insert_id,
                "species": event["species"], "insert_length": len(nt),
                "bag_family": family, "bag_mechanism_label": bag_label,
                "member_family_label": member_family, "member_mechanism_label": member_label,
                "member_label_matches_bag": str(member_family == family),
                "event_level_guide_target_verified": "no",
                "event_status": "unverified_not_negative",
            })
            member_mechanism_family = member_family if member_family in MECHANISMS else "UNRESOLVED"
            SeqIO.write(annotated_record(event, bag_id, member_mechanism_family, member_label, nt, "all_member_events"), fh, "genbank")
    member_cols = list(member_rows[0]) if member_rows else []
    write_tsv(out / "all_member_events.tsv", member_rows, member_cols)
    cols = list(gold_rows[0])
    write_tsv(out / "mechanism_gold_set.tsv", gold_rows, cols)
    counts = defaultdict(int)
    for r in gold_rows:
        counts[r["mechanism_gold_label"]] += 1
    manifest = {
        "n_bags": len(gold_rows), "n_representative_gbk_records": len(records),
        "n_all_member_event_records": len(member_rows),
        "all_member_genbank": member_gbk_path.name,
        "label_counts": dict(sorted(counts.items())),
        "n_event_level_guide_target_verified": 0,
        "n_event_level_unknown_not_negative": len(gold_rows),
        "input_annotation_root": str(root),
        "labels_are_family_mechanism_gold_only": True,
        "caveat": "These are experimentally grounded mechanism-family labels joined to independently annotated source elements. No specific natural event has verified guide-target pairing in this dataset; TnpB labels guided cleavage, not guided insertion.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
