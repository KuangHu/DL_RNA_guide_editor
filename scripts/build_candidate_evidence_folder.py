"""Build a per-candidate evidence folder with everything a human reviewer
needs to inspect a v3 cluster, including annotated GenBank files for the
inserts and target loci of each supporting site.

Writes to  evidence/<CID>/<cluster_key>/  with:

  README.md                           - short summary + what each file is
  provenance.tsv                      - per-site db_id, locus, md5, carriers
  cluster.tsv                         - cluster metadata (nc_pos, L, fp, mean_m, ...)
  nc_context.fna / nc_msa.fna         - per-site nc +/- CTX nt + MAFFT MSA
  target_context.fna / target_msa.fna - per-site target +/- CTX nt + MAFFT MSA
  inserts/site_<i>_insert.fna         - full 1824-bp inserted element per site
  inserts/site_<i>_insert.gbk         - annotated GenBank (dominant ORF marked)
  targets/site_<i>_target.fna         - full 400-bp pre-insertion target per site
  targets/site_<i>_target.gbk         - annotated GenBank (insertion point marked)
  pairings/site_<i>_pairing.txt       - nc vs target per-base pairing with context

Does NOT run v3 to re-derive clusters -- it takes the (nc_pos, L, flank_pos)
on the command line.  The intent is "give me evidence for THIS cluster",
not "re-score".

Usage:
  python scripts/build_candidate_evidence_folder.py \
    --jsonl .../adapted.jsonl \
    --bag CDS03597 --nc-pos 1 --L 14 --flank-pos 45 \
    --supporting 0,4,6,7 \
    --outdir evidence/CDS03597/nc1_L14_fp45 \
    --context 100
"""
from __future__ import annotations
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.alignment import revcomp
from scripts.v5a_framework.multi_hit_retention import SPACER_N

MAFFT = "/global/home/users/kh36969/.conda/envs/opfi/bin/mafft"
INSERTS_FNA = Path("/global/scratch/users/kh36969/fna_ins_discovery/database_v3/insertions_inserts.fna")
INSERTIONS_TSV = Path("/global/scratch/users/kh36969/fna_ins_discovery/database_v3/insertions.tsv")
INSERT_CDS = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/insert_cds.tsv")

try:
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
    from Bio.SeqFeature import SeqFeature, FeatureLocation
    from Bio import SeqIO
    HAVE_BIO = True
except ImportError:
    HAVE_BIO = False
    print("WARNING: Biopython not available — GBK files will be skipped.", file=sys.stderr)


def load_bag(jsonl: Path, bag_id: str):
    sites = []
    with jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r["transposase_id"] == bag_id:
                sites.append(r)
    return sites


def load_insertions_for(db_ids: set[str]) -> dict:
    out = {}
    with INSERTIONS_TSV.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["db_id"] in db_ids:
                out[row["db_id"]] = row
    return out


def load_dominant_orf(bag_id: str) -> dict:
    """Return {db_id -> dominant_orf_name, dominant_orf_aa_len} for the bag's events."""
    out = {}
    with INSERT_CDS.open() as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            if row["cds_cluster_id"] == bag_id:
                out[row["db_id"]] = {"orf": row["dominant_orf"],
                                      "aa": row["dominant_orf_len_aa"]}
    return out


def load_insert_seqs(db_ids: set[str]) -> dict:
    out = {}
    cur = None; buf = []
    with INSERTS_FNA.open() as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if cur and cur in db_ids:
                    out[cur] = "".join(buf)
                cur = line[1:].split()[0]
                buf = []
            else:
                buf.append(line)
        if cur and cur in db_ids:
            out[cur] = "".join(buf)
    return out


def _concat_nc(nc_list: list[str]) -> str:
    out = nc_list[0]
    for extra in nc_list[1:]:
        out = out + "N" * SPACER_N + extra
    return out


def _match_line(a: str, b: str) -> str:
    o = []
    for x, y in zip(a, b):
        if x == "N" or y == "N": o.append("N")
        elif x == y: o.append("|")
        else: o.append(".")
    return "".join(o)


def write_insert_gbk(path: Path, db_id: str, seq: str, orf_name: str,
                      orf_aa: str, bag_id: str):
    if not HAVE_BIO:
        return
    rec = SeqRecord(Seq(seq), id=db_id, name=db_id.replace(".", "_")[:16],
                     description=f"{bag_id} insert sequence (1824 bp family)")
    rec.annotations["molecule_type"] = "DNA"
    # Add dominant ORF as a feature if we know its length (approx)
    try:
        aa_len = int(orf_aa)
        # No exact coord — add as a comment feature covering the full span
        orf_feat = SeqFeature(
            FeatureLocation(0, len(seq)),
            type="CDS",
            qualifiers={
                "gene": [orf_name],
                "note": [f"dominant_orf {orf_name}; aa_len approximately {aa_len}; "
                          f"bag_id {bag_id}; exact nucleotide coord not stored in insert file"],
            },
        )
        rec.features.append(orf_feat)
    except (ValueError, TypeError):
        pass
    rec.annotations["comment"] = (f"bag_id: {bag_id}\n"
                                    f"db_id: {db_id}\n"
                                    f"dominant_orf: {orf_name} ({orf_aa} aa)")
    with path.open("w") as f:
        SeqIO.write([rec], f, "genbank")


def write_target_gbk(path: Path, db_id: str, target_seq: str,
                      insertion_point: int, bag_id: str, locus_id: str,
                      carriers_empty: str, pre_event_genome: str = "",
                      pre_event_contig: str = ""):
    if not HAVE_BIO:
        return
    rec = SeqRecord(Seq(target_seq), id=db_id,
                     name=(db_id + "_tgt").replace(".", "_")[:16],
                     description=f"{bag_id} pre-insertion target (empty) locus")
    rec.annotations["molecule_type"] = "DNA"
    rec.annotations["comment"] = (f"bag_id: {bag_id}\n"
                                    f"db_id: {db_id}\n"
                                    f"locus_id: {locus_id}\n"
                                    f"pre_event_genome: {pre_event_genome}\n"
                                    f"pre_event_contig: {pre_event_contig}\n"
                                    f"carriers_empty: {carriers_empty}")
    # Insertion-point feature (join between pre- and post-insertion halves)
    rec.features.append(SeqFeature(
        FeatureLocation(insertion_point, insertion_point + 1),
        type="misc_feature",
        qualifiers={
            "note": [f"insertion_point_offset={insertion_point} "
                      f"(pre-insertion target; element would insert here)"],
        }))
    with path.open("w") as f:
        SeqIO.write([rec], f, "genbank")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--nc-pos", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--flank-pos", type=int, required=True)
    ap.add_argument("--supporting", required=True,
                      help="comma-separated bag-site indices that support this cluster")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--context", type=int, default=100)
    ap.add_argument("--all-sites", action="store_true",
                      help="also include non-supporting sites in evidence outputs")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    support = [int(x) for x in args.supporting.split(",")]
    sites = load_bag(args.jsonl, args.bag)
    print(f"# bag {args.bag}  K={len(sites)}  supporting={support}")

    db_ids = [s["site_id"].replace(".site", "") for s in sites]
    ins_rows = load_insertions_for(set(db_ids))
    orf_info = load_dominant_orf(args.bag)
    insert_seqs = load_insert_seqs(set(db_ids))

    # provenance.tsv
    with (args.outdir / "provenance.tsv").open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["bag_idx","db_id","is_support","locus_id","inserted_len",
                    "insert_md5","target_len","insertion_point_offset",
                    "n_carriers_derived","n_carriers_empty","decomposition_method"])
        for i, (s, d) in enumerate(zip(sites, db_ids)):
            r = ins_rows.get(d, {})
            w.writerow([i, d, i in support,
                        r.get("locus_id",""),
                        r.get("inserted_len",""),
                        r.get("insert_md5",""),
                        r.get("target_len",""),
                        r.get("insertion_point_offset",""),
                        len(r.get("carriers_derived","").split(",")) if r.get("carriers_derived") else 0,
                        len(r.get("carriers_empty","").split(",")) if r.get("carriers_empty") else 0,
                        r.get("decomposition_method","")])
    print(f"# wrote {args.outdir}/provenance.tsv")

    # cluster.tsv
    with (args.outdir / "cluster.tsv").open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["field","value"])
        for k, v in [("bag_id", args.bag), ("nc_pos", args.nc_pos),
                      ("L", args.L), ("flank_pos", args.flank_pos),
                      ("supporting_sites", ",".join(map(str, support)))]:
            w.writerow([k, v])

    # ±ctx contexts for nc + target
    sites_to_dump = support if not args.all_sites else list(range(len(sites)))
    with (args.outdir / "nc_context.fna").open("w") as fw:
        for i in sites_to_dump:
            nc = _concat_nc(sites[i]["inputs"]["noncoding_regions"])
            lo = max(0, args.nc_pos - args.context)
            hi = min(len(nc), args.nc_pos + args.L + args.context)
            fw.write(f">site{i}_nc  support={i in support}  "
                     f"nc_len={len(nc)}  slice=[{lo}:{hi}]  "
                     f"guide_rel_pos={args.nc_pos - lo}\n{nc[lo:hi]}\n")
    with (args.outdir / "target_context.fna").open("w") as fw:
        for i in sites_to_dump:
            fl = sites[i]["inputs"]["flank"]
            lo = max(0, args.flank_pos - args.context)
            hi = min(len(fl), args.flank_pos + args.L + args.context)
            fw.write(f">site{i}_target  support={i in support}  "
                     f"flank_len={len(fl)}  slice=[{lo}:{hi}]  "
                     f"target_rel_pos={args.flank_pos - lo}\n{fl[lo:hi]}\n")
    print(f"# wrote contexts")

    # MAFFT
    for base in ("nc_context", "target_context"):
        inp = args.outdir / f"{base}.fna"
        outp = args.outdir / f"{base.replace('context','msa')}.fna"
        r = subprocess.run([MAFFT, "--auto", "--quiet", str(inp)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"MAFFT failed on {base}: {r.stderr[:200]}")
        else:
            outp.write_text(r.stdout)
            print(f"# wrote {outp}")

    # inserts + targets as FASTA + GBK
    (args.outdir / "inserts").mkdir(exist_ok=True)
    (args.outdir / "targets").mkdir(exist_ok=True)
    (args.outdir / "pairings").mkdir(exist_ok=True)
    for i in sites_to_dump:
        d = db_ids[i]
        row = ins_rows.get(d, {})
        # insert fna + gbk
        iseq = insert_seqs.get(d, "")
        if iseq:
            (args.outdir / "inserts" / f"site_{i}_insert.fna").write_text(
                f">{d}  {args.bag}  inserted_len={row.get('inserted_len','')}\n{iseq}\n")
            orf = orf_info.get(d, {})
            write_insert_gbk(args.outdir / "inserts" / f"site_{i}_insert.gbk",
                              d, iseq, orf.get("orf",""), orf.get("aa",""), args.bag)
        # target fna + gbk
        tseq = row.get("target_seq","")
        if tseq:
            try: ipt = int(row.get("insertion_point_offset", 0))
            except ValueError: ipt = 0
            (args.outdir / "targets" / f"site_{i}_target.fna").write_text(
                f">{d}  {args.bag}  target_len={row.get('target_len','')} "
                f"insertion_point={ipt}\n{tseq}\n")
            write_target_gbk(args.outdir / "targets" / f"site_{i}_target.gbk",
                              d, tseq, ipt, args.bag, row.get("locus_id",""),
                              row.get("carriers_empty","")[:80])

        # pairing diagram: extended match beyond the L-mer, properly aligned.
        # Pair nc[nc_pos - PAD : nc_pos + L + PAD] with
        #     revcomp(flank[flank_pos - PAD : flank_pos + L + PAD]).
        # These are the same length, so aligning them position-by-position makes
        # the central L-mer line up, and any extended complementarity shows up
        # on either side of the guide window.
        if i in support:
            nc = _concat_nc(sites[i]["inputs"]["noncoding_regions"])
            fl = sites[i]["inputs"]["flank"]
            PAD = 40
            nc_lo = max(0, args.nc_pos - PAD)
            nc_hi = min(len(nc), args.nc_pos + args.L + PAD)
            fp_lo = max(0, args.flank_pos - PAD)
            fp_hi = min(len(fl), args.flank_pos + args.L + PAD)
            nc_slice = nc[nc_lo:nc_hi]
            fl_slice = fl[fp_lo:fp_hi]
            # Compute RC on the flank slice.  Note on frame:
            # if nc[nc_pos:nc_pos+L] matches RC(flank[flank_pos:flank_pos+L]),
            # then shifting both windows outwards by +k on each side must align
            # nc[nc_pos+k] with RC(flank[flank_pos+L+PAD-1 - k - L])
            # = flank[flank_pos+L-1+PAD-k] complement reversed.
            # Simpler: build the RC of the WHOLE flank slice, then line up by
            # the central L-mer.  Both are length PAD + L + PAD (apart from edges).
            tgt_rc = revcomp(fl_slice)
            # Alignment: both slices should have the matching L-mer in the middle
            # (assuming no edge truncation).  Compute center offsets.
            left_nc  = args.nc_pos - nc_lo
            right_nc = nc_hi - (args.nc_pos + args.L)
            left_fl  = args.flank_pos - fp_lo
            right_fl = fp_hi - (args.flank_pos + args.L)
            # In RC(flank), the matching window is at offset (right_fl) from the start
            # (because reverse flipped sides).  So to align:
            left_rc = right_fl       # bp of RC(flank) to the LEFT of the match window
            right_rc = left_fl       # bp to the RIGHT
            # Build aligned display
            total_left  = max(left_nc, left_rc)
            total_right = max(right_nc, right_rc)
            nc_display = ("-" * (total_left - left_nc)) + nc_slice + ("-" * (total_right - right_nc))
            rc_display = ("-" * (total_left - left_rc)) + tgt_rc + ("-" * (total_right - right_rc))
            mline = _match_line(nc_display.upper(), rc_display.upper())
            # Count extended match
            core_match = sum(1 for a, b in zip(
                nc_display[total_left:total_left+args.L].upper(),
                rc_display[total_left:total_left+args.L].upper()) if a == b and a != "-")
            # Match extended beyond the L-mer core, in both directions
            left_ext = 0
            for k in range(1, total_left + 1):
                a = nc_display[total_left - k].upper()
                b = rc_display[total_left - k].upper()
                if a == b and a in "ACGT": left_ext += 1
                else: break
            right_ext = 0
            for k in range(args.L):
                pass  # ensure we start from after the L-mer
            end_core = total_left + args.L
            for k in range(len(nc_display) - end_core):
                a = nc_display[end_core + k].upper()
                b = rc_display[end_core + k].upper()
                if a == b and a in "ACGT": right_ext += 1
                else: break
            with (args.outdir / "pairings" / f"site_{i}_pairing.txt").open("w") as fw:
                fw.write(f"# site {i}  db_id={d}  bag={args.bag}\n"
                         f"# nc[{nc_lo}:{nc_hi}] aligned to revcomp(flank[{fp_lo}:{fp_hi}])\n")
                fw.write(f"# central L-mer (guide-target core): nc_pos={args.nc_pos} L={args.L}\n")
                fw.write(f"# core match: {core_match}/{args.L}  "
                         f"left_extension: {left_ext}  right_extension: {right_ext}  "
                         f"total contiguous match through core: {left_ext + core_match + right_ext}\n")
                fw.write(f"\n")
                # Mark the central L-mer with ^ below the match line
                mark_line = (" " * total_left) + ("^" * args.L) + (" " * (len(nc_display) - total_left - args.L))
                fw.write(f"nc  5'->3':   {nc_display}\n")
                fw.write(f"              {mline}\n")
                fw.write(f"RC(flank):    {rc_display}\n")
                fw.write(f"              {mark_line}\n")
                fw.write(f"\nguide    = nc[{args.nc_pos}:{args.nc_pos+args.L}]  = {nc[args.nc_pos:args.nc_pos+args.L]}\n")
                fw.write(f"target   = flank[{args.flank_pos}:{args.flank_pos+args.L}] = {fl[args.flank_pos:args.flank_pos+args.L]}\n")
                fw.write(f"RC(tgt)  = {revcomp(fl[args.flank_pos:args.flank_pos+args.L])}\n")

    # README
    with (args.outdir / "README.md").open("w") as fw:
        fw.write(f"# {args.bag} — cluster evidence package\n\n")
        fw.write(f"- cluster: nc_pos={args.nc_pos}  L={args.L}  flank_pos={args.flank_pos}\n")
        fw.write(f"- supporting sites: {sorted(support)}\n\n")
        fw.write("## Files\n\n")
        fw.write("- `provenance.tsv` — per-site db_id, locus, md5, carriers\n")
        fw.write("- `cluster.tsv` — cluster metadata\n")
        fw.write("- `nc_context.fna` / `nc_msa.fna` — per-site nc ±context + MAFFT MSA\n")
        fw.write("- `target_context.fna` / `target_msa.fna` — per-site target ±context + MAFFT MSA\n")
        fw.write("- `inserts/site_<i>_insert.{fna,gbk}` — full 1824-bp inserted element (annotated with dominant ORF)\n")
        fw.write("- `targets/site_<i>_target.{fna,gbk}` — full 400-bp pre-insertion target (insertion point annotated)\n")
        fw.write("- `pairings/site_<i>_pairing.txt` — nc 5′→3′ aligned to RC(flank) ±20 nt around the window, per-base match line\n\n")
        fw.write(f"## Read the evidence\n\n")
        fw.write("1. **Provenance** — check that supporting sites are in distinct genomes with "
                  "distinct insert_md5. If they share md5 or carriers, suspect clonal duplication.\n")
        fw.write("2. **Target MSA** — do the supporting sites' targets align to the same orthologous "
                  "host locus? If yes, same insertion site in different genomes.\n")
        fw.write("3. **nc MSA** — do the supporting sites' nc contexts align? If yes, same ncRNA "
                  "family. CRITICAL: check whether raw `nc_pos` maps to the SAME alignment column "
                  "across sites. If not, the Phase-C cluster at that raw index is not at a shared "
                  "scaffold position.\n")
        fw.write("4. **Pairings** — the nc 5′→3′ vs RC(flank) match lines show whether complementarity "
                  "extends beyond the 14-mer. A long contiguous match suggests either real extended "
                  "guide-target base pairing or an extraction/sequence overlap.\n")
        fw.write("5. **GBK files** — load inserts/*.gbk and targets/*.gbk in a genome browser "
                  "(Artemis, SnapGene, Benchling) to see dominant ORF and insertion point in context.\n")


if __name__ == "__main__":
    main()
