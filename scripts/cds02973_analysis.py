"""Analyze CDS02973 events (n=29): 6-frame + Pfam + find ORF2 partner cluster.

For each event:
  - print all Prodigal ORFs from orfs.gff (dominant + ORF2 partner)
  - run 6-frame translation to see if any hidden ORF beats Prodigal
  - dump ORF1 + ORF2 protein sequences

For ORF2 partner:
  - look up in clu_cluster.tsv to find the MMseqs cluster it belongs to
  - report if that cluster is another top-100 v84 candidate
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.cds08046_6frame import all_frames, translate  # noqa

INSERTS = Path("/global/scratch/users/kh36969/fna_ins_discovery/database/insertions_inserts.fna")
GFF = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/orfs.gff")
CLU = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/clu_cluster.tsv")
ORFS_FAA = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/orfs.faa")
INSERT_TSV = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/insert_cds.tsv")

# CDS02973 events
def get_cds_events(cid: str) -> list[str]:
    events = []
    for line in INSERT_TSV.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if parts[8] == cid:
            events.append(parts[0])
    return events


def get_orfs_for_event(event: str) -> list[dict]:
    """Return list of Prodigal ORF records for `event`."""
    orfs = []
    for line in GFF.read_text().splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        if parts[0] != event or parts[2] != "CDS":
            continue
        start, end = int(parts[3]), int(parts[4])
        strand = parts[6]
        attrs = {}
        for kv in parts[8].split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                attrs[k] = v
        orfs.append({
            "start": start, "end": end, "strand": strand,
            "partial": attrs.get("partial", "?"),
            "start_type": attrs.get("start_type", "?"),
            "conf": attrs.get("conf", "?"),
            "orf_id": attrs.get("ID", "?"),
        })
    return orfs


def load_fasta_index(faa: Path) -> dict[str, str]:
    """Load orfs.faa — key: ORF header, value: sequence."""
    d = {}
    cur = None; buf = []
    for line in faa.read_text().splitlines():
        if line.startswith(">"):
            if cur:
                d[cur] = "".join(buf)
            cur = line[1:].split()[0]
            buf = []
        else:
            buf.append(line)
    if cur:
        d[cur] = "".join(buf)
    return d


def load_cluster(clu: Path) -> dict[str, str]:
    """clu_cluster.tsv format: rep\tmember. Return member → rep."""
    d = {}
    for line in clu.read_text().splitlines():
        p = line.split("\t")
        if len(p) >= 2:
            d[p[1]] = p[0]
    return d


def load_insert_nt(inserts: Path, events: set[str]) -> dict[str, str]:
    d = {}
    cur = None; buf = []
    with inserts.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur and cur in events:
                    d[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur and cur in events:
            d[cur] = "".join(buf)
    return d


def main():
    cds = "CDS02973"
    events = get_cds_events(cds)
    print(f"# {cds} events: {len(events)}")

    orfs_faa = load_fasta_index(ORFS_FAA)
    cluster_map = load_cluster(CLU)
    insert_nt = load_insert_nt(INSERTS, set(events))
    print(f"# loaded {len(orfs_faa)} orf sequences, {len(cluster_map)} cluster assignments")

    # Represent each event by its ORF layout + members' cluster assignments
    orf1_clusters = set()
    orf2_clusters = set()
    orf2_orfs = []
    all_orfs_to_dump = []

    print(f"\n## Per-event ORF layout (from orfs.gff)")
    for event in events[:8]:  # print first 8 for brevity
        orfs = get_orfs_for_event(event)
        print(f"### {event}  (insert len={len(insert_nt.get(event, ''))})")
        for i, o in enumerate(orfs, 1):
            orf_id = f"{event}_{i}"
            seq = orfs_faa.get(orf_id, "")
            clu_rep = cluster_map.get(orf_id, "?")
            aa_len = len(seq.rstrip("*"))
            print(f"  ORF{i}  nt[{o['start']:4d}..{o['end']:4d}]  {o['strand']}  "
                       f"partial={o['partial']}  {o['start_type']:5s}  conf={o['conf']}  "
                       f"aa_len={aa_len}  cluster_rep={clu_rep}")

    # Collect all ORFs across events for Pfam scan
    for event in events:
        orfs = get_orfs_for_event(event)
        for i, o in enumerate(orfs, 1):
            orf_id = f"{event}_{i}"
            seq = orfs_faa.get(orf_id, "")
            if not seq:
                continue
            clu_rep = cluster_map.get(orf_id, "?")
            if i == 1:
                orf1_clusters.add(clu_rep)
            else:
                orf2_clusters.add(clu_rep)
                orf2_orfs.append((orf_id, seq, clu_rep))
            all_orfs_to_dump.append((f"{event}_orf{i}_c={clu_rep}", seq))

    print(f"\n## Cluster inventory across all {len(events)} events:")
    print(f"  ORF1 clusters (dominant): {orf1_clusters}")
    print(f"  ORF2 clusters (partner):  {orf2_clusters}")

    # 6-frame on a few representative event nt
    print(f"\n## 6-frame check on representative events (see if any hidden ORF beats Prodigal)")
    for event in events[:5]:
        nt = insert_nt.get(event, "")
        if len(nt) < 100:
            continue
        prod_orfs = get_orfs_for_event(event)
        prod_best_len = max((o["end"] - o["start"] + 1) // 3 for o in prod_orfs) if prod_orfs else 0
        six = all_frames(nt.strip("N"), min_len=30)
        best6 = six[0] if six else None
        print(f"  {event}: Prodigal best={prod_best_len}aa  vs  6-frame best="
                   f"{best6['aa_len'] if best6 else 0}aa "
                   f"({best6['strand']}f{best6['frame']})" if best6 else "")

    # Dump all ORFs for Pfam scan
    out_faa = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                        "fna_ins_discovery/blast/cds02973_all_orfs.faa")
    out_faa.parent.mkdir(exist_ok=True, parents=True)
    with out_faa.open("w") as f:
        # Deduplicate by sequence
        seen = set()
        for name, seq in all_orfs_to_dump:
            key = seq
            if key in seen:
                continue
            seen.add(key)
            f.write(f">{name}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")
    print(f"\n# wrote {len(seen) if 'seen' in dir() else 0} unique ORF sequences to {out_faa}")


if __name__ == "__main__":
    main()
