"""Find all bags in fna_ins_discovery that carry IS200/605-family or
Group II intron Pfam signatures, then check whether v7-real scored them,
whether they passed the site-independence + TSD-clustering filters.

Target Pfam markers:
  IS200/605-family (RNA-guided TnpB carriers):
    - PF01385  OrfB_IS605      (direct TnpB / IS605 marker)
    - PF07282  Cas12f1-like_TNB (broader TnpB/Cas12f family)
  Group II intron (RNA-guided LtrA carriers):
    - PF08388  GIIM             (Group II Intron Maturase — diagnostic)
    - PF00078  RVT_1            (broad reverse-transcriptase — reported as
                                 supporting evidence; RVT_1 alone can be
                                 retron, DGR, Abi RT, or Group II intron)

Method:
  1. Extract the 4 target HMMs from Pfam-A via hmmfetch
  2. hmmsearch each against cds/clu_rep_seq.fasta (6095 cluster reps)
     with --cut_ga
  3. Map hit cluster reps → bags via cds/clu_cluster.tsv + cds/insert_cds.tsv
  4. For each hit bag, look up v7 score (all_scores.tsv) + pass status
     (all_scored_final_candidates.tsv) → classify as SCORED_PASS,
     SCORED_FAIL, UNSCORED (n_sites < 3)
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from collections import defaultdict, Counter
from pathlib import Path


CONDA = "/global/home/users/kh36969/.conda/envs/opfi/bin"
D = Path("/global/scratch/users/kh36969/fna_ins_discovery")
V = D / "v7real_scored_v1"

TARGET_HMMS = {
    "PF01385": "OrfB_IS605",
    "PF07282": "Cas12f1-like_TNB",
    "PF08388": "GIIM",
    "PF00078": "RVT_1",
}
IS200_605_MARKERS = {"PF01385", "PF07282"}
GROUP_II_INTRON_MARKER = "PF08388"
RVT_SUPPORT_MARKER = "PF00078"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=V)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    work = args.out_dir / ".find_rna_guided_work"; work.mkdir(exist_ok=True)

    # Step 1: extract target HMMs. hmmfetch keys by NAME (OrfB_IS605), not
    # accession (PF01385). Values of TARGET_HMMS ARE the names.
    mini_hmm = work / "target.hmm"
    print(f"[find] extracting {len(TARGET_HMMS)} HMMs from Pfam-A...")
    with open(mini_hmm, "w") as outh:
        for pf_acc, pf_name in TARGET_HMMS.items():
            res = subprocess.run(
                [f"{CONDA}/hmmfetch",
                 str(D.parent / "funcannot_dbs" / "pfam" / "Pfam-A.hmm"), pf_name],
                capture_output=True, text=True)
            if res.returncode != 0:
                print(f"[find] WARN: hmmfetch failed for {pf_name} ({pf_acc}): {res.stderr}")
                continue
            outh.write(res.stdout)
    # hmmpress the mini HMM
    for ext in (".h3f", ".h3i", ".h3m", ".h3p"):
        p = Path(str(mini_hmm) + ext)
        if p.exists(): p.unlink()
    subprocess.run([f"{CONDA}/hmmpress", str(mini_hmm)], check=True)

    # Step 2: hmmsearch each HMM against clu_rep_seq.fasta
    faa = D / "cds" / "clu_rep_seq.fasta"
    print(f"[find] hmmsearch {faa}  ({sum(1 for l in open(faa) if l.startswith('>'))} proteins)")
    per_pf_hits: dict[str, dict[str, dict]] = defaultdict(dict)
    for pf_acc, pf_name in TARGET_HMMS.items():
        # Fetch the model as a standalone file and hmmsearch it. Use NAME.
        one = work / f"{pf_acc}.hmm"
        subprocess.run([f"{CONDA}/hmmfetch",
                        str(D.parent / "funcannot_dbs" / "pfam" / "Pfam-A.hmm"), pf_name],
                        stdout=open(one, "w"), check=True)
        tbl = work / f"{pf_acc}.tbl"
        subprocess.run([f"{CONDA}/hmmsearch", "--cpu", str(args.threads), "--cut_ga",
                        "--tblout", str(tbl), "-o", str(work / f"{pf_acc}.stdout"),
                        str(one), str(faa)], check=True)
        # Parse tblout: cols are target_name accession query_name qaccession
        with open(tbl) as f:
            for line in f:
                if line.startswith("#"): continue
                parts = line.split()
                if len(parts) < 5: continue
                rep_prot = parts[0]
                try: evalue = float(parts[4]); score = float(parts[5])
                except ValueError: continue
                per_pf_hits[pf_acc][rep_prot] = {"evalue": evalue, "score": score}
        print(f"  {pf_acc} ({pf_name}): {len(per_pf_hits[pf_acc])} rep proteins hit")

    # Step 3: map rep protein → cluster members → bags via insert_cds.tsv
    # clu_cluster.tsv: rep\tmember
    print("[find] mapping rep proteins → cluster members → bags...")
    rep_to_members: dict[str, list[str]] = defaultdict(list)
    with open(D / "cds" / "clu_cluster.tsv") as f:
        for line in f:
            rep, mem = line.rstrip().split("\t")
            rep_to_members[rep].append(mem)
    # insert_cds.tsv: db_id | ... | dominant_orf | ... | transposase_id | ...
    orf_to_bag: dict[str, str] = {}
    bag_to_orfs: dict[str, list[str]] = defaultdict(list)
    with open(D / "cds" / "insert_cds.tsv") as f:
        hdr = next(f).rstrip().split("\t")
        i_orf = hdr.index("dominant_orf"); i_bag = hdr.index("transposase_id")
        for line in f:
            parts = line.rstrip().split("\t")
            orf_to_bag[parts[i_orf]] = parts[i_bag]
            bag_to_orfs[parts[i_bag]].append(parts[i_orf])
    # For each hit rep_protein, walk to bags
    def hit_reps_to_bags(rep_prots: set[str]) -> set[str]:
        bags = set()
        for rep in rep_prots:
            for mem in rep_to_members.get(rep, [rep]):
                b = orf_to_bag.get(mem)
                if b: bags.add(b)
        return bags

    is200_bags = set()
    for pf in IS200_605_MARKERS:
        is200_bags |= hit_reps_to_bags(set(per_pf_hits[pf].keys()))
    group2_bags = hit_reps_to_bags(set(per_pf_hits[GROUP_II_INTRON_MARKER].keys()))
    rvt_bags   = hit_reps_to_bags(set(per_pf_hits[RVT_SUPPORT_MARKER].keys()))
    print(f"[find] IS200/605-family bags: {len(is200_bags)}")
    print(f"[find] Group II intron bags (GIIM):        {len(group2_bags)}")
    print(f"[find] RVT_1 bags (broad retroelement):    {len(rvt_bags)}")
    print(f"[find] RVT_1 minus GIIM (non-intron RT):   {len(rvt_bags - group2_bags)}")

    # Step 4: cross-reference with scored bags + passing bags
    scored_bags: dict[str, dict] = {}
    with open(V / "all_scores.tsv") as f:
        hdr = next(f).rstrip().split("\t")
        for line in f:
            row = dict(zip(hdr, line.rstrip().split("\t")))
            scored_bags[row["bag_id"]] = {
                "species": row["species"], "phylum": row["phylum"],
                "score": float(row["score"]),
                "n_sites_scored": int(row["n_sites_scored"]),
            }
    print(f"[find] scored bags: {len(scored_bags)}")

    passer_bags: dict[str, dict] = {}
    with open(V / "all_scored_final_candidates.tsv") as f:
        hdr = next(f).rstrip().split("\t")
        for line in f:
            row = dict(zip(hdr, line.rstrip().split("\t")))
            passer_bags[row["bag_id"]] = {
                "tsd_verdict": row["tsd_verdict"],
                "off_median": row["off_median"],
                "off_std": row["off_std"],
                "score": float(row["score"]),
            }
    print(f"[find] passer bags: {len(passer_bags)}")

    def classify_bag(b: str) -> tuple[str, str]:
        s = scored_bags.get(b); p = passer_bags.get(b)
        if p: return ("SCORED_PASS", f"score={s['score']:+.3f} {p['tsd_verdict']}")
        if s: return ("SCORED_FAIL", f"score={s['score']:+.3f}")
        return ("UNSCORED", "n_sites<3, not eligible")

    # Emit joint TSV
    out = args.out_dir / "rna_guided_hits_vs_model.tsv"
    with open(out, "w") as f:
        f.write("bag_id\tspecies\tphylum\tfamily\tn_sites_scored\tv7_score\tmodel_pass_status\ttsd_verdict\toff_median\toff_std\n")
        for family, bags in (("IS200/605-family", is200_bags),
                              ("Group_II_intron", group2_bags)):
            for b in sorted(bags):
                s = scored_bags.get(b)
                p = passer_bags.get(b)
                sp = s["species"] if s else "-"
                ph = s["phylum"] if s else "-"
                ns = s["n_sites_scored"] if s else "-"
                sc = f"{s['score']:.4f}" if s else "-"
                if p:
                    st = "SCORED_PASS"; tv = p["tsd_verdict"]
                    om = p["off_median"]; os_ = p["off_std"]
                elif s:
                    st = "SCORED_FAIL"; tv = "-"; om = "-"; os_ = "-"
                else:
                    st = "UNSCORED"; tv = "-"; om = "-"; os_ = "-"
                f.write(f"{b}\t{sp}\t{ph}\t{family}\t{ns}\t{sc}\t{st}\t{tv}\t{om}\t{os_}\n")

    print(f"\n[find] wrote {out}")

    # Summary tables
    for family, bags in (("IS200/605-family", is200_bags),
                          ("Group II intron (GIIM+)", group2_bags)):
        print(f"\n=== {family}: {len(bags)} bags total ===")
        c = Counter()
        for b in bags:
            st, _ = classify_bag(b); c[st] += 1
        print(f"  SCORED_PASS : {c['SCORED_PASS']:>3d}  ({100*c['SCORED_PASS']/max(len(bags),1):5.1f}%)")
        print(f"  SCORED_FAIL : {c['SCORED_FAIL']:>3d}  ({100*c['SCORED_FAIL']/max(len(bags),1):5.1f}%)")
        print(f"  UNSCORED    : {c['UNSCORED']:>3d}  ({100*c['UNSCORED']/max(len(bags),1):5.1f}%)")
        # List the passers
        passers = [b for b in bags if b in passer_bags]
        if passers:
            print(f"  --- PASSERS ---")
            for b in sorted(passers, key=lambda b: -passer_bags[b]["score"]):
                s = scored_bags[b]; p = passer_bags[b]
                print(f"    {b:<10s} {s['species']:<14s} score={p['score']:+.3f} "
                      f"{p['tsd_verdict']:<16s} off={p['off_median']}±{p['off_std']}")
        # List the scored-fails at high score (potentially missed hits)
        high_scored_fails = [
            (b, scored_bags[b]["score"], scored_bags[b]["species"])
            for b in bags if b in scored_bags and b not in passer_bags
            and scored_bags[b]["score"] >= 3.0
        ]
        if high_scored_fails:
            high_scored_fails.sort(key=lambda t: -t[1])
            print(f"  --- SCORED_FAIL with score>=3.0 (missed by filters) ---")
            for b, sc, sp in high_scored_fails[:15]:
                print(f"    {b:<10s} {sp:<14s} v7 score={sc:+.3f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
