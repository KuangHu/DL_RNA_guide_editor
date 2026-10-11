"""Post-sweep annotation: join v3 candidates_v3_merged.tsv with per-bag
species/Pfam metadata and run sensitive hmmscan on any top hit whose
dominant ORF has not yet been Pfam-scanned.

Pulls Pfam from all_candidates/manifest.tsv (precomputed sweep union of
8+ domtbl files, E ≤ 1e-5).  For bags with no strict Pfam hit AND top
v3 rank, runs hmmscan at E ≤ 10 on the dominant ORF so the "truly novel"
claim is backed by an actual sensitive scan.

Writes:
  candidates_v3_merged_annotated.tsv  (all merged rows + species/Pfam/tier/
                                        v4+v5 model scores)
  top_v3_hits_pfam_sensitive.tsv       (sensitive Pfam scan for top-N hits)
"""
from __future__ import annotations
import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


MANIFEST = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                "fna_ins_discovery/all_candidates/manifest.tsv")
INSERT_CDS = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/insert_cds.tsv")
ORFS_FAA   = Path("/global/scratch/users/kh36969/fna_ins_discovery/cds_v3/orfs.faa")
V4_SCORES  = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                   "fna_ins_discovery/scores.tsv")
V5_SCORES  = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/"
                   "fna_ins_discovery_v5/scores.tsv")

HMMSCAN = "/global/home/users/kh36969/.conda/envs/opfi/bin/hmmscan"
PFAM_HMM = Path("/global/scratch/users/kh36969/pfam_db/Pfam-A.hmm")


def load_manifest():
    m = {}
    with MANIFEST.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            m[r["cid"]] = r
    return m


def load_dominant_orf():
    out = {}
    with INSERT_CDS.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            cid = r["cds_cluster_id"]
            # First row per CID has the dominant ORF info
            if cid not in out:
                out[cid] = {"dominant_orf": r["dominant_orf"],
                             "dominant_orf_aa": r["dominant_orf_len_aa"],
                             "species_set": set()}
            out[cid]["species_set"].add(r["species"])
    for cid in out:
        out[cid]["species"] = ",".join(sorted(out[cid]["species_set"]))
    return out


def load_scores(path):
    s = {}
    with path.open() as f:
        next(f)
        for line in f:
            p = line.rstrip().split("\t")
            s[p[0]] = float(p[3])
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--top-n-hmmscan", type=int, default=50,
                      help="run sensitive hmmscan on this many top hits")
    ap.add_argument("--rank-column", default="rank_by_bg_groups",
                      help="which rank column to prioritize (default rank_by_bg_groups)")
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    orf_info = load_dominant_orf()
    v4 = load_scores(V4_SCORES)
    v5 = load_scores(V5_SCORES)

    # Load merged candidates and augment
    rows = list(csv.DictReader(args.merged.open(), delimiter="\t"))
    print(f"# loaded {len(rows)} merged rows")

    # Augment columns
    for r in rows:
        cid = r["bag_id"]
        m = manifest.get(cid, {})
        o = orf_info.get(cid, {})
        r["species"] = o.get("species", "")
        r["dominant_orf_aa"] = o.get("dominant_orf_aa", "")
        r["v4_bag_max"] = f"{v4.get(cid, float('nan')):.3f}" if cid in v4 else ""
        r["v5_bag_max"] = f"{v5.get(cid, float('nan')):.3f}" if cid in v5 else ""
        r["tier"] = m.get("tier", "")
        r["pfam_strict_E1e-5"] = m.get("pfam", "")
        r["pfam_reason"] = m.get("reason", "")

    # Write annotated TSV
    extra_cols = ["species", "dominant_orf_aa", "v4_bag_max", "v5_bag_max",
                   "tier", "pfam_strict_E1e-5", "pfam_reason"]
    with args.merged.open() as f:
        hdr = next(csv.reader(f, delimiter="\t"))
    out_cols = hdr + extra_cols
    ann_path = args.outdir / "candidates_v3_merged_annotated.tsv"
    with ann_path.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=out_cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in out_cols})
    print(f"# wrote {ann_path}")

    # Rank by publication-passing rows.  NOTE: the sweep's rank_by_* columns
    # were empty due to a bug in run_v3_scorer.py (ranks were filled on the
    # unmerged list, not the merged list).  We compute ranks here directly
    # from the raw columns so this annotator is self-sufficient.
    pub_rows = [r for r in rows
                if r.get("passes_publication_threshold", "False") == "True"]
    print(f"# {len(pub_rows)} merged rows above publication threshold")

    # Fill ranks per-bag over publication rows.
    from collections import defaultdict
    by_bag_pub = defaultdict(list)
    for r in pub_rows:
        by_bag_pub[r["bag_id"]].append(r)
    for bag, rr in by_bag_pub.items():
        for i, r in enumerate(sorted(rr, key=lambda r: (
                -int(r["n_local_bg_groups"]),
                -float(r["mean_m_frac"]),
                float(r["flank_pos_mad"]))), 1):
            r["_rank_by_bg_groups"] = i
        for i, r in enumerate(sorted(rr, key=lambda r: (
                -float(r["mean_m_frac"]),
                -int(r["n_local_bg_groups"]),
                float(r["flank_pos_mad"]))), 1):
            r["_rank_by_mean_m"] = i
        for i, r in enumerate(sorted(rr, key=lambda r: (
                float(r["flank_pos_mad"]),
                -int(r["n_local_bg_groups"]),
                -float(r["mean_m_frac"]))), 1):
            r["_rank_by_tightness"] = i

    rank_key = f"_{args.rank_column}"

    # Each bag's BEST cluster per the chosen column (rank=1 within bag)
    bag_best: dict[str, dict] = {}
    for r in pub_rows:
        bid = r["bag_id"]
        if bid not in bag_best or r[rank_key] < bag_best[bid][rank_key]:
            bag_best[bid] = r

    # Sort bags GLOBALLY by their best cluster's raw metric
    if args.rank_column == "rank_by_mean_m":
        bag_sort_key = lambda r: (-float(r["mean_m_frac"]),
                                     -int(r["n_local_bg_groups"]),
                                     float(r["flank_pos_mad"]))
    elif args.rank_column == "rank_by_bg_groups":
        bag_sort_key = lambda r: (-int(r["n_local_bg_groups"]),
                                     -float(r["mean_m_frac"]),
                                     float(r["flank_pos_mad"]))
    else:   # rank_by_tightness
        bag_sort_key = lambda r: (float(r["flank_pos_mad"]),
                                     -int(r["n_local_bg_groups"]),
                                     -float(r["mean_m_frac"]))
    seen_cids = [r["bag_id"] for r in sorted(bag_best.values(), key=bag_sort_key)]

    print(f"# {len(seen_cids)} unique bags above publication threshold")
    hmmscan_targets = seen_cids[: args.top_n_hmmscan]
    print(f"# will sensitive-hmmscan top {len(hmmscan_targets)} bags")

    # Extract dominant-ORF proteins for all targets
    want_orfs = {orf_info[c]["dominant_orf"]: c for c in hmmscan_targets
                 if c in orf_info}
    faa_path = args.outdir / "top_v3_dominant_orfs.faa"
    with faa_path.open("w") as fw, ORFS_FAA.open() as fr:
        keep = False
        for line in fr:
            if line.startswith(">"):
                name = line[1:].split()[0]
                keep = name in want_orfs
                if keep:
                    cid = want_orfs[name]
                    fw.write(f">{cid}__{name}\n")
            elif keep:
                fw.write(line)
    print(f"# wrote {faa_path} ({len(want_orfs)} proteins)")

    # Sensitive hmmscan
    domtbl = args.outdir / "top_v3_pfam_sensitive.domtbl"
    print(f"# running hmmscan -E 10 (sensitive) on {len(want_orfs)} proteins...")
    r = subprocess.run([HMMSCAN, "--cpu", "4", "-E", "10",
                         "--domtblout", str(domtbl), str(PFAM_HMM), str(faa_path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(f"hmmscan FAILED: {r.stderr[:500]}")
        return
    # Parse domtbl and summarize per CID
    hits_per_cid: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    with domtbl.open() as f:
        for line in f:
            if line.startswith("#"): continue
            p = line.split()
            if len(p) < 20: continue
            tgt = p[3]
            cid = tgt.split("__")[0]
            try:
                e = float(p[6]); bits = float(p[7])
            except ValueError: continue
            hits_per_cid[cid].append((p[0], e, bits))
    for c in hits_per_cid:
        hits_per_cid[c].sort(key=lambda h: h[1])

    out_tsv = args.outdir / "top_v3_hits_pfam_sensitive.tsv"
    with out_tsv.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["rank", "bag_id", "mean_m_frac", "n_support_raw",
                    "support_frac_raw", "n_local_bg_groups",
                    "guide_at_cluster", "v4_bag_max", "v5_bag_max", "tier",
                    "pfam_strict_E1e-5", "pfam_sensitive_E10",
                    "novelty", "species", "dominant_orf_aa"])
        for i, cid in enumerate(hmmscan_targets, 1):
            pub_r = bag_best[cid]
            hits = hits_per_cid.get(cid, [])
            sens = ",".join(f"{n}(E={e:.0e},{b:.1f}b)" for n, e, b in hits[:4]) or "-"
            pfam_strict = pub_r["pfam_strict_E1e-5"]
            novelty = "NOVEL" if (pfam_strict == "-" or not pfam_strict) and \
                                 (not hits or hits[0][1] > 1e-3) else "ANNOTATED"
            w.writerow([i, cid, pub_r["mean_m_frac"], pub_r["n_support_raw"],
                        pub_r["support_frac_raw"], pub_r["n_local_bg_groups"],
                        pub_r["guide_at_cluster"],
                        pub_r["v4_bag_max"], pub_r["v5_bag_max"],
                        pub_r["tier"], pfam_strict, sens, novelty,
                        pub_r["species"], pub_r["dominant_orf_aa"]])
    print(f"# wrote {out_tsv}")

    # Short textual summary
    print(f"\n## top-20 v3 hits (by {args.rank_column})")
    for i, cid in enumerate(hmmscan_targets[:20], 1):
        pub_r = bag_best[cid]
        hits = hits_per_cid.get(cid, [])
        pfam_strict = pub_r["pfam_strict_E1e-5"]
        pfam_sens = ",".join(f"{n}({e:.0e})" for n,e,_ in hits[:2]) or "NO HIT"
        tag = "NOVEL" if (pfam_strict == "-" or not pfam_strict) and \
                        (not hits or hits[0][1] > 1e-3) else "ANNOTATED"
        print(f"  {i:>2d}  {cid:<10s}  mm={pub_r['mean_m_frac']:>5s}  "
              f"sup={pub_r['n_support_raw']:>2s}/{pub_r['K_valid']:>1s}  "
              f"bg={pub_r['n_local_bg_groups']:>1s}  "
              f"v4={pub_r['v4_bag_max'] or '-':>6s}  "
              f"v5={pub_r['v5_bag_max'] or '-':>6s}  "
              f"strict=[{pfam_strict[:30]}]  "
              f"sens=[{pfam_sens[:35]}]  "
              f"{tag}  "
              f"({pub_r['species']})")


if __name__ == "__main__":
    main()
