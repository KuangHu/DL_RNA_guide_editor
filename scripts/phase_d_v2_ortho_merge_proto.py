"""P2c prototype — Phase D with ortho-locus merging + offset-tolerant
background comparison + coverage reporting.

Runs on CDS03597's known cluster at (nc_pos=1, L=14, flank_pos~45) and
compares to the current Phase D output.

Changes over scripts/v5a_framework/background_grouping.py:

  1. Explicit ortho-locus grouping: sites sharing (pre_event_genome,
     pre_event_contig, overlapping locus coords) are declared one
     background group BEFORE any sequence comparison, taken from
     loci.tsv provenance. This promotes the audit observation
     "sites 4 and 7 are one orthologous locus" into scoring logic
     instead of relying on the sequence-identity threshold to catch
     it.

  2. Offset-tolerant pairwise identity: slide one side by up to
     ±max_offset bp and keep the best alignment. Resists small
     indels / strand-length mismatches without needing a full
     Needleman-Wunsch.

  3. Both consistency rate AND coverage length are reported in the
     output.

  4. Per-pair match tags which side (left / right / concat) contributed
     the identity (= "repeat source").

Writes evidence/CDS03597/nc1_L14_fp45_v2/phase_d_v2/
  comparison.md     — side-by-side with current Phase D output
  per_site.tsv      — per-site locus_info + edge availability
  pairwise.tsv      — per-pair identity, coverage length, side source
  ortho_merge.tsv   — ortho groupings
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.pipeline import run_bag, PipelineConfig
from scripts.v5a_framework.background_grouping import compute_bg_groups


def read_loci_evidence(site_dir: Path) -> dict:
    out = {}
    with (site_dir / "loci_evidence.tsv").open() as f:
        f.readline()
        for line in f:
            k, v = line.rstrip("\n").split("\t", 1)
            out[k] = v
    return out


def ortho_groups(site_locus_info: dict[int, dict],
                   overlap_bp: int = 100,
                   ) -> list[list[int]]:
    """Group sites by (pre_event_genome, pre_event_contig) + locus coord
    overlap >= overlap_bp. Returns list of groups (each a list of site_idx).
    """
    sites = sorted(site_locus_info)
    parent = {s: s for s in sites}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    for i, si in enumerate(sites):
        for sj in sites[i + 1:]:
            ai, aj = site_locus_info[si], site_locus_info[sj]
            if ai["pre_event_genome"] != aj["pre_event_genome"]: continue
            if ai["pre_event_contig"] != aj["pre_event_contig"]: continue
            lo = max(ai["start"], aj["start"])
            hi = min(ai["end"], aj["end"])
            if hi - lo >= overlap_bp:
                union(si, sj)
    gmap: dict[int, list[int]] = {}
    for s in sites:
        r = find(s)
        gmap.setdefault(r, []).append(s)
    return sorted(gmap.values(), key=lambda g: g[0])


def offset_tolerant_identity(a: str, b: str, max_offset: int = 5,
                               ) -> tuple[float, int, int]:
    """Return (best_identity, coverage_length_bp, best_offset).
    Slides a vs b by [-max_offset..+max_offset], keeps best identity."""
    if not a or not b: return (0.0, 0, 0)
    best = (-1.0, 0, 0)
    for off in range(-max_offset, max_offset + 1):
        if off >= 0:
            a_al, b_al = a[:len(b) - off], b[off:off + len(a)]
        else:
            a_al, b_al = a[-off:-off + len(b)], b[:len(a) + off]
        n = min(len(a_al), len(b_al))
        if n < 10: continue
        m = sum(1 for x, y in zip(a_al[:n], b_al[:n]) if x == y)
        ident = m / n
        if ident > best[0] or (ident == best[0] and n > best[1]):
            best = (ident, n, off)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--evidence-dir", type=Path, required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--nc-pos", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--flank-pos", type=int, required=True)
    ap.add_argument("--max-offset", type=int, default=5)
    ap.add_argument("--margin", type=int, default=3)
    ap.add_argument("--clonal-threshold", type=float, default=0.85)
    args = ap.parse_args()

    out = args.evidence_dir / "phase_d_v2"
    out.mkdir(parents=True, exist_ok=True)

    # 1. Load bag; identify cluster
    sites = []
    with args.jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("transposase_id") == args.bag:
                sites.append(r)
    cfg = PipelineConfig()
    result = run_bag(args.bag, sites, cfg)
    target = None
    for cl, bg in result.passing + result.failing:
        if cl.nc_pos == args.nc_pos and cl.L == args.L and \
                abs(cl.flank_pos_median - args.flank_pos) <= 5:
            target = (cl, bg); break
    if target is None:
        raise SystemExit("cluster not found")
    cluster, orig_bg = target

    # 2. Build per-site locus_info from loci_evidence.tsv
    site_locus_info = {}
    for ss in cluster.supporting_sites:
        s = ss.site_idx
        site_dir = args.evidence_dir / f"site_{s}"
        if not site_dir.exists():
            print(f"[warn] no evidence folder for site {s}")
            continue
        loci = read_loci_evidence(site_dir)
        site_locus_info[s] = {
            "pre_event_genome": loci["pre_event_genome"],
            "pre_event_contig": loci["pre_event_contig"],
            "start": int(loci["start"]),
            "end": int(loci["end"]),
            "junction_L_abs": int(loci["junction_L_abs"]),
            "junction_R_abs": int(loci["junction_R_abs"]),
            "junction_ambiguity_bp": int(loci["junction_ambiguity_bp"]),
            "decomposition_confidence": float(loci["decomposition_confidence"]),
            "pre_event_strand": loci["pre_event_strand"],
        }

    # 3. Ortho-locus groups
    ortho = ortho_groups(site_locus_info, overlap_bp=100)
    print(f"ortho_locus_groups: {ortho}")

    # 4. Flanks per site
    flanks_per_site = {ss.site_idx: sites[ss.site_idx]["inputs"]["flank"]
                        for ss in cluster.supporting_sites}

    # 5. Offset-tolerant pairwise identity + repeat source
    pair_rows = []
    for i, ssi in enumerate(cluster.supporting_sites):
        for ssj in cluster.supporting_sites[i + 1:]:
            si, sj = ssi.site_idx, ssj.site_idx
            fA, fB = flanks_per_site[si], flanks_per_site[sj]
            # Mask + extract left/right flanks of each
            def masked(f: str, fp: int) -> tuple[str, str]:
                lo = max(0, fp - args.margin)
                hi = min(len(f), fp + args.L + args.margin)
                return f[:lo], f[hi:]
            L_A, R_A = masked(fA, ssi.chosen_hit.flank_pos)
            L_B, R_B = masked(fB, ssj.chosen_hit.flank_pos)
            left = offset_tolerant_identity(L_A, L_B, args.max_offset)
            right = offset_tolerant_identity(R_A, R_B, args.max_offset)
            concat_A = L_A + R_A
            concat_B = L_B + R_B
            concat = offset_tolerant_identity(concat_A, concat_B, args.max_offset)
            # Determine repeat_source
            best_side = max([("left", left), ("right", right), ("concat", concat)],
                             key=lambda kv: kv[1][0])
            pair_rows.append({
                "si": si, "sj": sj,
                "left_identity": left[0], "left_cov": left[1], "left_offset": left[2],
                "right_identity": right[0], "right_cov": right[1], "right_offset": right[2],
                "concat_identity": concat[0], "concat_cov": concat[1], "concat_offset": concat[2],
                "repeat_source": best_side[0], "repeat_source_identity": best_side[1][0],
                "repeat_source_cov": best_side[1][1],
                "same_ortho_group": any(si in g and sj in g for g in ortho),
            })

    # 6. Build adjacency using ortho merge first, then concat identity threshold
    sites_in_cluster = [ss.site_idx for ss in cluster.supporting_sites]
    parent = {s: s for s in sites_in_cluster}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(a, b):
        parent[find(a)] = find(b)
    # Pre-merge via ortho
    for g in ortho:
        for s in g[1:]:
            if s in parent and g[0] in parent:
                union(g[0], s)
    # Then merge by sequence identity (same logic as current Phase D)
    for r in pair_rows:
        if r["concat_identity"] > args.clonal_threshold and r["concat_cov"] >= 20:
            if r["si"] in parent and r["sj"] in parent:
                union(r["si"], r["sj"])
    merged = {}
    for s in sites_in_cluster:
        merged.setdefault(find(s), []).append(s)
    merged_groups = sorted(merged.values(), key=lambda g: g[0])
    n_groups_v2 = len(merged_groups)

    # 7. Write outputs
    with (out / "per_site.tsv").open("w") as f:
        f.write("site\tpre_event_genome\tpre_event_contig\tstart\tend\t"
                "pre_event_strand\tjunction_ambiguity_bp\tdecomp_confidence\n")
        for s, info in sorted(site_locus_info.items()):
            f.write(f"{s}\t{info['pre_event_genome']}\t{info['pre_event_contig']}\t"
                     f"{info['start']}\t{info['end']}\t{info['pre_event_strand']}\t"
                     f"{info['junction_ambiguity_bp']}\t{info['decomposition_confidence']}\n")

    with (out / "ortho_merge.tsv").open("w") as f:
        f.write("group_rep\tmembers\n")
        for g in ortho:
            f.write(f"{g[0]}\t{','.join(str(x) for x in g)}\n")

    with (out / "pairwise.tsv").open("w") as f:
        f.write("si\tsj\tsame_ortho\tleft_id\tleft_cov\tleft_off\t"
                "right_id\tright_cov\tright_off\tconcat_id\tconcat_cov\t"
                "concat_off\trepeat_source\trepeat_src_id\trepeat_src_cov\n")
        for r in pair_rows:
            f.write(f"{r['si']}\t{r['sj']}\t{r['same_ortho_group']}\t"
                     f"{r['left_identity']:.3f}\t{r['left_cov']}\t{r['left_offset']}\t"
                     f"{r['right_identity']:.3f}\t{r['right_cov']}\t{r['right_offset']}\t"
                     f"{r['concat_identity']:.3f}\t{r['concat_cov']}\t{r['concat_offset']}\t"
                     f"{r['repeat_source']}\t{r['repeat_source_identity']:.3f}\t"
                     f"{r['repeat_source_cov']}\n")

    # 8. Compare to current Phase D
    with (out / "comparison.md").open("w") as f:
        f.write("# Phase D v2 (ortho-aware) vs current Phase D for CDS03597\n\n")
        f.write(f"## Setup\n")
        f.write(f"- cluster: nc_pos={cluster.nc_pos} L={cluster.L} "
                 f"flank_pos_median={cluster.flank_pos_median}  n_support={cluster.n_support}\n")
        f.write(f"- margin={args.margin}  max_offset={args.max_offset}  "
                 f"clonal_threshold={args.clonal_threshold}\n\n")
        f.write(f"## Current Phase D output\n")
        f.write(f"- n_local_bg_groups = {orig_bg.n_local_bg_groups}\n")
        f.write(f"- groups: ")
        f.write(", ".join(str(g.members) for g in orig_bg.groups) + "\n")
        f.write(f"- passes_bg_gate = {orig_bg.passes_bg_gate}\n")
        f.write(f"- rejection_reason = {orig_bg.rejection_reason}\n\n")
        f.write(f"## P2c Phase D v2 output\n")
        f.write(f"- ortho_locus_groups (from loci.tsv): {ortho}\n")
        f.write(f"- n_local_bg_groups_v2 (ortho + seq merge): {n_groups_v2}\n")
        f.write(f"- groups: {merged_groups}\n\n")
        f.write(f"## Per-pair repeat source\n")
        f.write("| si | sj | same_ortho | left_id | left_cov | right_id | right_cov | "
                 "concat_id | concat_cov | repeat_source |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in pair_rows:
            f.write(f"| {r['si']} | {r['sj']} | {r['same_ortho_group']} | "
                     f"{r['left_identity']:.3f} | {r['left_cov']} | "
                     f"{r['right_identity']:.3f} | {r['right_cov']} | "
                     f"{r['concat_identity']:.3f} | {r['concat_cov']} | "
                     f"{r['repeat_source']} |\n")

    print(f"\nCurrent Phase D: n_local_bg_groups={orig_bg.n_local_bg_groups}  "
          f"groups={[g.members for g in orig_bg.groups]}  passes={orig_bg.passes_bg_gate}")
    print(f"P2c Phase D v2: n_local_bg_groups={n_groups_v2}  "
          f"groups={merged_groups}  ortho={ortho}")


if __name__ == "__main__":
    main()
