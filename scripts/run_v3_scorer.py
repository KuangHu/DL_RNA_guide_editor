"""v3 scorer driver — step 6 of model_free_scorer_v3.md.

Runs Phase A→D on a specified set of bags from an adapted.jsonl and emits:

  candidates_v3.tsv : one row per surviving (bag_id, nc_pos, L, cluster_id)
                      with every per-cluster field from the output schema;
                      three orthogonal ranking columns (no composite score)
  rejections_v3.tsv : one row per (bag_id, nc_pos, L, cluster_id) that
                      failed any gate, with the reason

A publication threshold `min_max_mean_m` (default 0.89 from Phase G 500-bag
calibration at target_null_pass=0.01) is applied as a column
`passes_publication_threshold` — rows that fail stay in `candidates_v3.tsv`
with that column = False so the full enumeration is auditable.

Usage:
  python scripts/run_v3_scorer.py \
    --jsonl /scratch/.../fna_ins_discovery_v5_md_seed2/adapted.jsonl \
    --bag-ids CDS03597,CDS04337,CDS02973,CDS00113 \
    --outdir logs/v3_top4
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median, mean

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.pipeline import run_bag, PipelineConfig
from scripts.v5a_framework.motif_gates import (
    shannon_base, at_fraction, longest_run, periodicity_fraction,
)
from scripts.v5a_framework.cluster_merging import merge_clusters, MergeConfig

# dinuc entropy helper (not exported from motif_gates; defined inline for the
# candidates_v3.tsv `mean_guide_H_dinuc` column)
def shannon_dinuc(seq: str) -> float:
    import math
    from collections import Counter
    dinucs = [seq[i:i+2] for i in range(len(seq)-1)
              if seq[i] in "ACGT" and seq[i+1] in "ACGT"]
    if len(dinucs) < 2: return 0.0
    n = len(dinucs); c = Counter(dinucs)
    H = 0.0
    for v in c.values():
        p = v / n
        H -= p * math.log2(p)
    return H
from scripts.v5a_framework.multi_hit_retention import SPACER_N


PUBLICATION_THRESHOLD_DEFAULT = 0.89   # Phase G 500-bag at null_pass ≤ 0.01


CANDIDATE_COLS = [
    "bag_id", "cluster_id",
    "nc_pos", "L",
    "flank_pos_median", "flank_pos_mad",
    "flank_pos_min", "flank_pos_max",
    "K_valid",
    "n_support_raw", "support_frac_raw",
    "n_local_bg_groups",
    "support_frac_bg_unique",       # n_support / n_local_bg_groups  (per plan)
    "mean_m_frac", "min_m_frac",
    "n_sites_cluster_hit_is_best",
    "n_sites_cluster_hit_rank_1_or_2",
    "guide_at_cluster",
    "mean_guide_H_base", "mean_guide_H_dinuc", "mean_guide_at_frac", "mean_guide_longest_run",
    "mean_target_H_base", "mean_target_at_frac",
    "mean_hit_multiplicity_at_nc_pos_L",
    "left_median_identity", "right_median_identity",
    "n_sites_edge_left", "n_sites_edge_right", "n_sites_edge_both",
    "any_chain_linked",
    "orient_fwd", "orient_rc",
    "passes_phase_D",
    "passes_publication_threshold",
    "rank_by_bg_groups", "rank_by_mean_m", "rank_by_tightness",
]

REJECTION_COLS = [
    "bag_id", "cluster_id",
    "nc_pos", "L",
    "flank_pos_median",
    "K_valid", "n_support_raw", "n_local_bg_groups",
    "mean_m_frac",
    "phase", "rejection_reason",
]


def _bags_from_jsonl(path: Path, bag_ids: set[str] | None):
    bags: dict[str, list[dict]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            bid = r["transposase_id"]
            if bag_ids is None or bid in bag_ids:
                bags[bid].append(r)
    return bags


def _reconstruct_nc(sites: list[dict], site_idx: int) -> str:
    nc_list = sites[site_idx]["inputs"]["noncoding_regions"]
    nc = nc_list[0]
    for extra in nc_list[1:]:
        nc = nc + "N" * SPACER_N + extra
    return nc


def _row_for_cluster(bag_id: str, cluster_id: int,
                      cluster, bg_report,
                      sites: list[dict],
                      passes_phase_D: bool,
                      publication_threshold: float):
    nc_pos, L = cluster.nc_pos, cluster.L
    # per-site guide/target sequences at this cluster
    guide_strs = []
    target_strs = []
    for ss in cluster.supporting_sites:
        nc = _reconstruct_nc(sites, ss.site_idx)
        f  = sites[ss.site_idx]["inputs"]["flank"]
        g  = nc[nc_pos:nc_pos + L] if nc_pos + L <= len(nc) else ""
        t  = f[ss.chosen_hit.flank_pos:ss.chosen_hit.flank_pos + L] \
             if ss.chosen_hit.flank_pos + L <= len(f) else ""
        if g: guide_strs.append(g)
        if t: target_strs.append(t)

    # guide_at_cluster — most common guide across sites (v5 reconstruction may
    # mutate nc per site; pick modal to represent)
    if guide_strs:
        from collections import Counter
        guide_mode = Counter(guide_strs).most_common(1)[0][0]
    else:
        guide_mode = ""

    def _mean(xs): return mean(xs) if xs else 0.0

    mean_guide_H_base   = _mean([shannon_base(g)  for g in guide_strs])
    mean_guide_H_dinuc  = _mean([shannon_dinuc(g) for g in guide_strs])
    mean_guide_at_frac  = _mean([at_fraction(g)   for g in guide_strs])
    mean_guide_run      = _mean([longest_run(g)   for g in guide_strs])
    mean_target_H_base  = _mean([shannon_base(t)  for t in target_strs])
    mean_target_at_frac = _mean([at_fraction(t)   for t in target_strs])

    # rank-1-or-2 count
    n_rank12 = sum(1 for ss in cluster.supporting_sites
                     if ss.rank_within_site <= 2)

    n_groups = bg_report.n_local_bg_groups
    support_frac_bg_unique = (cluster.n_support / n_groups) if n_groups else 0.0

    return {
        "bag_id": bag_id, "cluster_id": cluster_id,
        "nc_pos": nc_pos, "L": L,
        "flank_pos_median": cluster.flank_pos_median,
        "flank_pos_mad":    f"{cluster.flank_pos_mad:.2f}",
        "flank_pos_min":    cluster.flank_pos_min,
        "flank_pos_max":    cluster.flank_pos_max,
        "K_valid":          cluster.K_valid,
        "n_support_raw":    cluster.n_support,
        "support_frac_raw": f"{cluster.support_frac:.3f}",
        "n_local_bg_groups": n_groups,
        "support_frac_bg_unique": f"{support_frac_bg_unique:.3f}",
        "mean_m_frac":       f"{cluster.mean_m_frac:.3f}",
        "min_m_frac":        f"{cluster.min_m_frac:.3f}",
        "n_sites_cluster_hit_is_best":    cluster.n_sites_cluster_hit_is_best,
        "n_sites_cluster_hit_rank_1_or_2": n_rank12,
        "guide_at_cluster":  guide_mode,
        "mean_guide_H_base":  f"{mean_guide_H_base:.3f}",
        "mean_guide_H_dinuc": f"{mean_guide_H_dinuc:.3f}",
        "mean_guide_at_frac": f"{mean_guide_at_frac:.3f}",
        "mean_guide_longest_run": f"{mean_guide_run:.2f}",
        "mean_target_H_base":  f"{mean_target_H_base:.3f}",
        "mean_target_at_frac": f"{mean_target_at_frac:.3f}",
        "mean_hit_multiplicity_at_nc_pos_L":
            f"{cluster.mean_hit_multiplicity_at_nc_pos_L:.2f}",
        "left_median_identity":
            f"{bg_report.median_left_identity:.3f}"  if bg_report.median_left_identity  is not None else "",
        "right_median_identity":
            f"{bg_report.median_right_identity:.3f}" if bg_report.median_right_identity is not None else "",
        "n_sites_edge_left":  len(bg_report.edge_insufficient_left),
        "n_sites_edge_right": len(bg_report.edge_insufficient_right),
        "n_sites_edge_both":  len(bg_report.edge_insufficient_both),
        "any_chain_linked":   bg_report.any_chain_linked,
        "orient_fwd": cluster.orient_counts.get("fwd", 0),
        "orient_rc":  cluster.orient_counts.get("rc",  0),
        "passes_phase_D":    passes_phase_D,
        "passes_publication_threshold": (passes_phase_D and
                                           cluster.mean_m_frac >= publication_threshold),
        # ranks filled after all rows gathered
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--bag-ids", type=str, default="",
                      help="comma-separated bag ids; empty = all bags")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--publication-threshold", type=float,
                      default=PUBLICATION_THRESHOLD_DEFAULT)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bag_ids = set(args.bag_ids.split(",")) if args.bag_ids else None
    print(f"# loading bags from {args.jsonl}  "
          f"(filter: {len(bag_ids) if bag_ids else 'ALL'})")
    bags = _bags_from_jsonl(args.jsonl, bag_ids)
    print(f"# {len(bags)} bags to score")

    cfg = PipelineConfig()
    all_cand_rows: list[dict] = []
    all_rej_rows:  list[dict] = []
    t0 = time.time()
    for idx, (bag_id, sites) in enumerate(sorted(bags.items()), 1):
        result = run_bag(bag_id, sites, cfg)
        # candidates: all passing + all failing Phase D, kept with pass flag
        all_clusters = result.passing + result.failing
        for cid, (cluster, bg_report) in enumerate(all_clusters, 1):
            row = _row_for_cluster(bag_id, cid, cluster, bg_report,
                                     sites,
                                     passes_phase_D=(cluster, bg_report) in result.passing,
                                     publication_threshold=args.publication_threshold)
            all_cand_rows.append(row)
        for cid, (cluster, bg_report) in enumerate(result.failing,
                                                       1 + len(result.passing)):
            all_rej_rows.append({
                "bag_id": bag_id, "cluster_id": cid,
                "nc_pos": cluster.nc_pos, "L": cluster.L,
                "flank_pos_median": cluster.flank_pos_median,
                "K_valid": cluster.K_valid,
                "n_support_raw": cluster.n_support,
                "n_local_bg_groups": bg_report.n_local_bg_groups,
                "mean_m_frac": f"{cluster.mean_m_frac:.3f}",
                "phase": "D",
                "rejection_reason": bg_report.rejection_reason or "",
            })
        if idx % 25 == 0:
            dt = time.time() - t0
            print(f"  [{idx}/{len(bags)}] {dt:.1f}s  "
                  f"candidates so far: {len(all_cand_rows)}, "
                  f"rejections: {len(all_rej_rows)}",
                  flush=True)

    # Fill the three orthogonal rank columns per bag
    by_bag_pass = defaultdict(list)
    for r in all_cand_rows:
        if r["passes_phase_D"]:
            by_bag_pass[r["bag_id"]].append(r)
    for bag_id, rows in by_bag_pass.items():
        # rank_by_bg_groups: higher n_local_bg_groups is better
        for i, r in enumerate(sorted(rows, key=lambda r: (-r["n_local_bg_groups"],
                                                             -float(r["mean_m_frac"]))), 1):
            r["rank_by_bg_groups"] = i
        # rank_by_mean_m: higher mean_m_frac is better
        for i, r in enumerate(sorted(rows, key=lambda r: -float(r["mean_m_frac"])), 1):
            r["rank_by_mean_m"] = i
        # rank_by_tightness: lower MAD is better
        for i, r in enumerate(sorted(rows, key=lambda r: float(r["flank_pos_mad"])), 1):
            r["rank_by_tightness"] = i
    # Default rank columns to empty when the row didn't pass Phase D
    for r in all_cand_rows:
        for k in ("rank_by_bg_groups", "rank_by_mean_m", "rank_by_tightness"):
            r.setdefault(k, "")

    cand_path = args.outdir / "candidates_v3.tsv"
    rej_path  = args.outdir / "rejections_v3.tsv"
    with cand_path.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=CANDIDATE_COLS)
        w.writeheader()
        for r in all_cand_rows:
            w.writerow({k: r.get(k, "") for k in CANDIDATE_COLS})
    with rej_path.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=REJECTION_COLS)
        w.writeheader()
        for r in all_rej_rows:
            w.writerow({k: r.get(k, "") for k in REJECTION_COLS})
    print(f"\n# wrote {cand_path} ({len(all_cand_rows)} rows)")
    print(f"# wrote {rej_path} ({len(all_rej_rows)} rows)")

    # ---------- step 7: cross-L / cross-nc_pos merging ----------
    # Merge only clusters that passed Phase D (publication-worthy).  The
    # un-merged file above remains for audit.
    merge_cfg = MergeConfig()
    merged_rows_out: list[dict] = []
    for bag_id, sites in sorted(bags.items()):
        # recollect this bag's Phase-D-passing clusters by replaying run_bag
        # (cheap: one pass per bag), then run merger
        result = run_bag(bag_id, sites, cfg)
        if not result.passing:
            continue
        merged = merge_clusters(result.passing, merge_cfg)
        for rank_in_bag, m in enumerate(merged, 1):
            rep_cluster, rep_bg = m.representative
            # synthesize a row from the representative, add merge columns
            row = _row_for_cluster(
                bag_id, rank_in_bag, rep_cluster, rep_bg, sites,
                passes_phase_D=True,
                publication_threshold=args.publication_threshold,
            )
            row.update({
                "n_merged": m.n_merged,
                "nc_pos_range": f"{m.nc_pos_min}-{m.nc_pos_max}",
                "L_range":      f"{m.L_min}-{m.L_max}",
                "flank_pos_range": f"{m.flank_pos_median_min}-{m.flank_pos_median_max}",
                "support_union": ",".join(str(s) for s in m.support_union),
            })
            merged_rows_out.append(row)

    merged_path = args.outdir / "candidates_v3_merged.tsv"
    MERGED_COLS = (
        CANDIDATE_COLS
        + ["n_merged", "nc_pos_range", "L_range",
            "flank_pos_range", "support_union"]
    )
    with merged_path.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=MERGED_COLS)
        w.writeheader()
        for r in merged_rows_out:
            w.writerow({k: r.get(k, "") for k in MERGED_COLS})
    print(f"# wrote {merged_path} ({len(merged_rows_out)} merged rows)")

    # Per-bag publication-threshold summary
    print(f"\n## per-bag publication summary (threshold={args.publication_threshold})")
    print(f"  {'bag_id':<12s}  {'K':>2s}  {'n_pass_D':>8s}  "
          f"{'n_pub_thr':>9s}  {'best_mean_m':>11s}  "
          f"{'best_bg_grps':>12s}  {'best_cluster':>12s}")
    for bag_id in sorted(bags.keys()):
        rows_bag = [r for r in all_cand_rows if r["bag_id"] == bag_id]
        rows_pass_D = [r for r in rows_bag if r["passes_phase_D"]]
        rows_pub = [r for r in rows_pass_D if r["passes_publication_threshold"]]
        if rows_pass_D:
            best = max(rows_pass_D, key=lambda r: float(r["mean_m_frac"]))
            best_desc = (f"nc={best['nc_pos']} L={best['L']} fp={best['flank_pos_median']} "
                         f"mm={best['mean_m_frac']}")
        else:
            best = None; best_desc = "-"
        best_bg = max(rows_pass_D, key=lambda r: r["n_local_bg_groups"]) if rows_pass_D else None
        print(f"  {bag_id:<12s}  "
              f"{rows_bag[0]['K_valid'] if rows_bag else 0:>2d}  "
              f"{len(rows_pass_D):>8d}  "
              f"{len(rows_pub):>9d}  "
              f"{best['mean_m_frac'] if best else '-':>11s}  "
              f"{best_bg['n_local_bg_groups'] if best_bg else 0:>12d}  "
              f"{best_desc:>12s}")


if __name__ == "__main__":
    main()
