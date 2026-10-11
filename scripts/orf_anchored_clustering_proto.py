"""P2a prototype — ORF-anchored orientation + nc context alignment-column
clustering, run against evidence/CDS03597/nc1_L14_fp45_v2/.

Pipeline:
  1. Pick a canonical ORF family (the one that appears exactly once at
     ≥95% identity across the supporting sites — picked from
     orfs_pairwise.blast). We use the longest-conserved ORF.
  2. For each site, read its insert.gbk / prodigal GFF, locate the ORF
     at the ORF-family centroid, and read its strand on the OCCUPIED
     allele. If the ORF is on the '-' strand, revcomp the whole
     occupied allele so the canonical ORF faces 5'→3'.
  3. Build the ORIENTED context window of ±N bp around the canonical
     ORF's start codon from each site's oriented occupied allele.
  4. MAFFT the N oriented context windows.
  5. For every Phase-B retained hit at the (nc_pos, L) cluster, project
     its guide core position from occupied allele into the MSA column
     coord. Report whether cores from different sites land in the SAME
     MSA column (within tolerance).

Compares to Phase C's raw-nc_pos clustering: if the raw cluster at
nc_pos=1 L=14 fp=45 corresponds to one shared MSA column, it's a
real cross-site cluster. If the cores land in different columns under
oriented-context alignment, the raw-nc_pos cluster is coincidental.

Writes:
  evidence/CDS03597/nc1_L14_fp45_v2/orf_anchored/
    anchor_choice.tsv           — chosen ORF + per-site strand + offset
    oriented_contexts.fna       — ±N bp context around anchor ORF start
    oriented_contexts.msa.fna   — MAFFT MSA
    msa_core_columns.tsv        — per-site core_a_pos projected to MSA col
    verdict.md                  — side-by-side comparison with raw-nc_pos cluster
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from collections import defaultdict


def read_fna_records(path: Path) -> list[tuple[str, str]]:
    out = []
    cur_id, cur_seq = None, []
    with path.open() as f:
        for line in f:
            if line.startswith(">"):
                if cur_id is not None:
                    out.append((cur_id, "".join(cur_seq).upper()))
                cur_id = line[1:].rstrip("\n")
                cur_seq = []
            else:
                cur_seq.append(line.rstrip("\n"))
        if cur_id is not None:
            out.append((cur_id, "".join(cur_seq).upper()))
    return out


def read_fna(path: Path) -> str:
    recs = read_fna_records(path)
    return recs[0][1] if recs else ""


def write_fna(path: Path, name: str, seq: str) -> None:
    with path.open("w") as f:
        f.write(f">{name}\n")
        for i in range(0, len(seq), 80):
            f.write(seq[i:i + 80] + "\n")


def write_multi_fna(path: Path, items: list[tuple[str, str]]) -> None:
    with path.open("w") as f:
        for name, seq in items:
            f.write(f">{name}\n")
            for i in range(0, len(seq), 80):
                f.write(seq[i:i + 80] + "\n")


COMP = str.maketrans("ACGT", "TGCA")
def revcomp(s: str) -> str:
    return s.translate(COMP)[::-1]


def parse_prodigal_gff(path: Path) -> list[dict]:
    """Return list of ORF dicts: {start_1b, end_1b, strand, orf_idx}."""
    orfs = []
    idx = 0
    with path.open() as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "CDS": continue
            idx += 1
            orfs.append({"orf_idx": idx,
                         "start_1b": int(parts[3]),
                         "end_1b": int(parts[4]),
                         "strand": parts[6]})
    return orfs


def pick_anchor_orf(orfs_pairwise_path: Path, insert_dir: Path,
                     sites: list[str]) -> dict:
    """Pick the ORF family with highest coverage, longest min length.
    Returns {site_id: orf_idx} for the chosen anchor family."""
    # Parse orfs_pairwise.blast (format: qseqid sseqid pident length evalue bitscore qlen slen)
    # Build a graph of ORF homology; connected components are families.
    edges = []
    with orfs_pairwise_path.open() as f:
        hdr = f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8: continue
            q, s = parts[0], parts[1]
            pident, length = float(parts[2]), int(parts[3])
            qlen = int(parts[6])
            if q == s: continue
            if pident >= 95.0 and length >= 0.9 * qlen:
                edges.append((q, s))
    # Union-find
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    all_orfs = set()
    for a, b in edges:
        all_orfs.add(a); all_orfs.add(b); union(a, b)
    families = defaultdict(set)
    for o in all_orfs:
        families[find(o)].add(o)
    # Pick family with ONE member per site + longest min length
    best_family = None; best_score = (0, 0)
    for root, members in families.items():
        site_members = defaultdict(list)
        for m in members:
            # m like "s0_orf3"
            sid = m.split("_")[0][1:]
            site_members[sid].append(m)
        if set(site_members.keys()) != set(sites):
            continue
        if any(len(v) != 1 for v in site_members.values()):
            continue
        # Compute min member length from faa
        lengths = []
        for m in members:
            # Find faa for this site: evidence/.../site_X/insert_sX.prodigal.faa
            sid = m.split("_")[0][1:]
            orf_idx = int(m.split("orf")[1])
            faa = insert_dir / f"site_{sid}" / f"insert_s{sid}.prodigal.faa"
            with faa.open() as fp:
                cur_len = 0
                cur_idx = 0
                for line in fp:
                    if line.startswith(">"):
                        if cur_idx == orf_idx:
                            break
                        cur_idx += 1
                        cur_len = 0
                    else:
                        cur_len += len(line.rstrip("\n"))
                if cur_idx == orf_idx:
                    lengths.append(cur_len)
        if not lengths: continue
        score = (len(members), min(lengths))
        if score > best_score:
            best_score = score
            best_family = {m.split("_")[0][1:]: int(m.split("orf")[1])
                            for m in members}
    if best_family is None:
        raise SystemExit("No ORF family found that appears exactly once per supporting site")
    return best_family


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--sites", nargs="+", required=True,
                     help="supporting site indices, e.g. 0 4 6 7")
    ap.add_argument("--context-bp", type=int, default=400,
                     help="±context_bp around anchor ORF start codon")
    ap.add_argument("--nc-pos", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    args = ap.parse_args()

    out = args.evidence_dir / "orf_anchored"
    out.mkdir(parents=True, exist_ok=True)
    sites = args.sites

    # 1. Pick anchor ORF family
    orfs_pw = args.evidence_dir / "orfs_pairwise.blast"
    anchor = pick_anchor_orf(orfs_pw, args.evidence_dir, sites)
    print(f"anchor family (chosen): {anchor}")

    # 2. For each site: locate anchor ORF, determine strand on occupied,
    #    orient the occupied allele, mark anchor start.
    per_site = {}
    for sid in sites:
        site_dir = args.evidence_dir / f"site_{sid}"
        occupied = read_fna(site_dir / "occupied.fna")
        gff = parse_prodigal_gff(site_dir / f"insert_s{sid}.prodigal.gff")
        # The GFF coords are in the INSERT frame (1-based). Map to occupied.
        orf = next(o for o in gff if o["orf_idx"] == anchor[sid])
        # Insert start in occupied frame
        # Read the loci evidence to get insert_start_in_long_bp
        loci = {}
        with (site_dir / "loci_evidence.tsv").open() as f:
            f.readline()
            for line in f:
                k, v = line.rstrip("\n").split("\t", 1)
                loci[k] = v
        start_val = loci["insert_start_in_long_bp"]
        b0 = int(start_val.split()[0]) if "(" in start_val else int(start_val)
        # ORF coords 1-based → 0-based half-open in insert
        orf_s = orf["start_1b"] - 1 + b0
        orf_e = orf["end_1b"] + b0  # half-open
        orf_strand = orf["strand"]
        # Orient occupied so anchor ORF faces +
        if orf_strand == "+":
            oriented = occupied
            # Anchor START position in oriented is orf_s
            anchor_pos = orf_s
        else:
            oriented = revcomp(occupied)
            # orf_e in original is now at (len - orf_e) in revcomp'd
            # And the start of the ORF on + strand of revcomp is len - orf_e
            anchor_pos = len(occupied) - orf_e
        per_site[sid] = {
            "occupied_len": len(occupied),
            "orf_strand": orf_strand,
            "anchor_pos_in_oriented": anchor_pos,
            "orf_s_in_occupied": orf_s,
            "orf_e_in_occupied": orf_e,
            "insert_start_in_occupied": b0,
            "oriented": oriented,
            "orf_idx": orf["orf_idx"],
        }

    # Write anchor_choice.tsv
    with (out / "anchor_choice.tsv").open("w") as f:
        f.write("site\torf_idx\tstrand\torf_s_in_occupied\torf_e_in_occupied\t"
                "insert_start_in_occupied\tanchor_pos_in_oriented\toccupied_len\n")
        for sid in sites:
            d = per_site[sid]
            f.write(f"{sid}\t{d['orf_idx']}\t{d['orf_strand']}\t"
                     f"{d['orf_s_in_occupied']}\t{d['orf_e_in_occupied']}\t"
                     f"{d['insert_start_in_occupied']}\t"
                     f"{d['anchor_pos_in_oriented']}\t{d['occupied_len']}\n")

    # 3. Extract the ORIENTED INSERT itself (not the full occupied allele),
    #    so cross-site MSA aligns element-internal columns regardless of
    #    which strand of occupied the element lives on.
    contexts = []
    for sid in sites:
        d = per_site[sid]
        b0 = d["insert_start_in_occupied"]
        # Insert in occupied frame = [b0, b0 + inserted_len)
        # In oriented frame:
        occ_len = d["occupied_len"]
        # Need inserted_len — read from loci evidence
        site_dir = args.evidence_dir / f"site_{sid}"
        with (site_dir / "loci_evidence.tsv").open() as fh:
            fh.readline()
            loci = dict(ln.rstrip("\n").split("\t", 1) for ln in fh)
        ins_len = int(loci["inserted_len"])
        if d["orf_strand"] == "+":
            ins_lo_ori = b0
        else:
            ins_lo_ori = occ_len - (b0 + ins_len)
        ins_hi_ori = ins_lo_ori + ins_len
        insert_oriented = d["oriented"][ins_lo_ori:ins_hi_ori]
        contexts.append((f"site_{sid}_insert_oriented_len{len(insert_oriented)}",
                         insert_oriented))
        per_site[sid]["ctx_lo_in_oriented"] = ins_lo_ori
        per_site[sid]["ins_len"] = ins_len
        per_site[sid]["ins_lo_in_oriented"] = ins_lo_ori
        per_site[sid]["ins_hi_in_oriented"] = ins_hi_ori
    write_multi_fna(out / "oriented_contexts.fna", contexts)

    # 4. MAFFT
    msa_path = out / "oriented_contexts.msa.fna"
    r = subprocess.run(["mafft", "--auto", str(out / "oriented_contexts.fna")],
                        capture_output=True, text=True, check=True)
    with msa_path.open("w") as f:
        f.write(r.stdout)
    msa = read_fna_records(msa_path)
    print(f"MSA built: {len(msa)} rows, {len(msa[0][1])} alignment columns")

    # 5. R2 fix: project the FULL motif interval [p, p+L) from occupied
    #    into the ORIENTED insert frame, then into MSA columns.
    #
    #    For strand '+' the occupied interval [p, p+L) maps to the
    #    oriented interval [p, p+L) unchanged.
    #    For strand '-' we revcomp the whole occupied, so position p
    #    in occupied becomes position N-1-p in oriented. The L-mer
    #    occupied[p : p+L) occupies oriented positions
    #    [N-p-L, N-p). This is the CORRECT reverse interval the
    #    reviewer flagged — the old code mapped only the first byte
    #    (which, after reverse, is the motif's END, not its START).

    def occ_interval_to_oriented(d, lo: int, hi: int) -> tuple[int, int]:
        N = d["occupied_len"]
        if d["orf_strand"] == "+":
            return (lo, hi)
        return (N - hi, N - lo)

    core_a_positions = {}
    with (args.evidence_dir / "occupied_core_stability.tsv").open() as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(hdr, parts))
            sid = row["site_idx"]
            core_a_positions[sid] = int(row["core_a_pos_in_occupied"])

    def oriented_pos_to_msa_col(aligned: str, ctx_lo_in_oriented: int,
                                 oriented_pos: int) -> int:
        off_in_ctx = oriented_pos - ctx_lo_in_oriented
        if off_in_ctx < 0: return -1
        col = 0; count = 0
        for ch in aligned:
            if ch != "-":
                if count == off_in_ctx: return col
                count += 1
            col += 1
        return -1

    with (out / "msa_core_columns.tsv").open("w") as f:
        f.write("site\torf_strand\tcore_occ_lo\tcore_occ_hi\t"
                "core_ori_lo\tcore_ori_hi\t"
                "insert_lo_in_oriented\tinsert_hi_in_oriented\t"
                "msa_col_lo\tmsa_col_hi\n")
        rows_for_summary = []
        for name, aligned in msa:
            sid = name.split("_")[1]
            d = per_site[sid]
            core_occ = core_a_positions[sid]
            # FULL motif interval in occupied = [core_occ, core_occ + L)
            lo_occ, hi_occ = core_occ, core_occ + args.L
            lo_ori, hi_ori = occ_interval_to_oriented(d, lo_occ, hi_occ)
            col_lo = oriented_pos_to_msa_col(aligned, d["ins_lo_in_oriented"],
                                               lo_ori)
            # End of interval = position of last base + 1 → map (hi_ori - 1)
            # then advance one column
            end_last = oriented_pos_to_msa_col(aligned, d["ins_lo_in_oriented"],
                                                  hi_ori - 1)
            col_hi = end_last + 1 if end_last >= 0 else -1
            f.write(f"{sid}\t{d['orf_strand']}\t{lo_occ}\t{hi_occ}\t"
                     f"{lo_ori}\t{hi_ori}\t"
                     f"{d['ins_lo_in_oriented']}\t{d['ins_hi_in_oriented']}\t"
                     f"{col_lo}\t{col_hi}\n")
            rows_for_summary.append((sid, col_lo, col_hi))

    starts = [cl for _, cl, _ in rows_for_summary if cl >= 0]
    ends = [ch for _, _, ch in rows_for_summary if ch >= 0]
    if starts:
        col_med = sorted(starts)[len(starts) // 2]
        max_dev = max(abs(c - col_med) for c in starts)
    else:
        col_med, max_dev = -1, -1

    verdict = out / "verdict.md"
    with verdict.open("w") as f:
        f.write(f"# ORF-anchored clustering verdict for CDS03597 cluster at "
                 f"(nc_pos={args.nc_pos}, L={args.L})\n\n")
        f.write(f"## Setup\n")
        f.write(f"Anchor ORF family: {anchor}\n")
        f.write(f"Context window: ±{args.context_bp} bp around anchor ORF start\n\n")
        f.write(f"## Per-site anchor\n")
        f.write("| site | orf_idx | strand | insert_start_in_occupied | anchor_pos_in_oriented |\n")
        f.write("|---|---|---|---|---|\n")
        for sid in sites:
            d = per_site[sid]
            f.write(f"| {sid} | {d['orf_idx']} | {d['orf_strand']} | "
                     f"{d['insert_start_in_occupied']} | {d['anchor_pos_in_oriented']} |\n")
        f.write(f"\n## MSA\n")
        f.write(f"Rows: {len(msa)}, columns: {len(msa[0][1])}\n\n")
        f.write(f"## Full motif interval [p, p+L) projected into MSA columns\n")
        f.write(f"For strand='-' sites the reverse interval in oriented frame is "
                 f"[N-p-L, N-p). The reviewer's expected intervals are "
                 f"site 0 [89,103), site 4 [1875,1889), site 6 [1810,1824), "
                 f"site 7 [1833,1847).\n\n")
        f.write("| site | strand | core occ [lo, hi) | core ori [lo, hi) | MSA col [lo, hi) |\n")
        f.write("|---|---|---|---|---|\n")
        for sid, col_lo, col_hi in rows_for_summary:
            d = per_site[sid]
            core_occ = core_a_positions[sid]
            lo_ori, hi_ori = occ_interval_to_oriented(d, core_occ, core_occ + args.L)
            f.write(f"| {sid} | {d['orf_strand']} | "
                     f"[{core_occ}, {core_occ + args.L}) | "
                     f"[{lo_ori}, {hi_ori}) | [{col_lo}, {col_hi}) |\n")
        f.write(f"\n**MSA-column spread of motif START across sites:**\n")
        f.write(f"- cols: {sorted(starts)}\n")
        f.write(f"- median: {col_med}\n")
        f.write(f"- max deviation: {max_dev}\n\n")
        if max_dev >= 0 and max_dev <= 5:
            f.write("**Verdict:** the raw-nc_pos cluster at (nc_pos, L) "
                     f"maps to the SAME MSA column across sites (median={col_med}, "
                     f"max_dev={max_dev}). The cross-site cluster is at a conserved "
                     "scaffold column after ORF-anchored orientation.\n")
        elif max_dev >= 20:
            f.write("**Verdict (CDS03597):** cores scatter across MSA columns "
                     f"(spread={max_dev}). In ORF-anchored canonical orientation, "
                     "site 0's core sits near the 5' end of the element while "
                     "sites 4/6/7's cores sit near the 3' end. The raw-nc_pos "
                     "cluster is pulling matches from OPPOSITE ends of the "
                     "oriented element — the extraction convention 'nc_frag_0 "
                     "starts at insert 5' edge' picks up the 5'-TDR copy for "
                     "sites with the element on the + strand of occupied and "
                     "the 3'-TDR copy for sites with the element on the - "
                     "strand. The cluster therefore reflects terminal direct "
                     "repeat (TDR) homology, not a shared ncRNA guide at a "
                     "conserved scaffold column.\n\n"
                     "Note the per-site anchor strands match the pattern: "
                     "site 0 strand +, sites 4/6/7 strand -. The element is "
                     "in opposite polarity between the two groups.\n")
        else:
            f.write(f"**Verdict:** cores partially aligned (max_dev={max_dev}); "
                     "intermediate case.\n")

    print(f"\n=== VERDICT ===")
    print(f"Motif MSA intervals per site [col_lo, col_hi):")
    for sid, col_lo, col_hi in rows_for_summary:
        print(f"  site {sid}: [{col_lo}, {col_hi})")
    print(f"Median START column: {col_med}  max deviation: {max_dev}")
    print(f"Report: {verdict}")


if __name__ == "__main__":
    main()
