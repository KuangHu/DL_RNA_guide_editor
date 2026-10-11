"""v3 scorer driver — step 6 of model_free_scorer_v3.md.

Runs Phase A→D on a specified set of bags from an adapted.jsonl and emits:

  candidates_v3.tsv : one row per surviving (bag_id, nc_pos, L, cluster_id)
                      with every per-cluster field from the output schema;
                      three orthogonal ranking columns (no composite score)
  rejections_v3.tsv : one row per (bag_id, nc_pos, L, cluster_id) that
                      failed any gate, with the reason

Verified coordinates are the default. Use --evidence-manifest for a
sequence-bound sidecar, or inline scorer_evidence in each JSONL record.
Missing evidence is reported in bag_summary.jsonl, never as a negative.
There is no default publication threshold; --calibration consumes a
matching held-out calibration artifact. An explicit numeric threshold is
exploratory only. --coordinate-mode legacy is available for old audits.

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
from scripts.v5a_framework.verified_coordinates import attach_manifest
from scripts.v5a_framework.reporting_statistic import bag_statistics, config_fingerprint, read_calibration, STATISTIC_VERSION, effective_configuration
from preprocess.alignment import revcomp
from scripts.nc_coord_mapping import concat_pos_to_frag
from dataclasses import asdict

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


PUBLICATION_THRESHOLD_DEFAULT = None  # old 0.89 calibration is invalid for this pipeline


CANDIDATE_COLS = [
    "bag_id", "cluster_id",
    "nc_pos", "L",
    "flank_pos_median", "flank_pos_mad",
    "flank_pos_min", "flank_pos_max",
    "K_valid",
    "n_support_raw", "support_frac_raw",
    "n_local_bg_groups",
    "sites_per_bg_group",
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
    "bag_status", "coordinate_frame", "nc_msa_columns", "support_sites_json",
    "background_pairs_json", "background_merge_sources_json",
    "passes_exploratory_threshold", "calibration_status",
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


def _local_anchor_alignment(site, hit, context_bp=30):
    """Display the selected ungapped anchor with context, never a new best hit.

    Context cannot cross an nc-fragment boundary or spacer. Exact contiguous
    extension is diagnostic sequence evidence, not an extra score/TIR claim.
    """
    fragments=site['inputs']['noncoding_regions']
    idx,k,spacer=concat_pos_to_frag(fragments,hit.source_nc_pos())
    if spacer:
        return None
    guide=fragments[idx].upper()
    flank=site['inputs']['flank'].upper()
    fp=hit.source_flank_pos()
    if (hit.raw_orient or hit.orient)=='rc':
        flank=revcomp(flank)
        fp=len(flank)-fp-hit.L
    left=min(k,fp,context_bp)
    right=min(len(guide)-k-hit.L,len(flank)-fp-hit.L,context_bp)
    a=guide[k-left:k+hit.L+right]
    b=flank[fp-left:fp+hit.L+right]
    matches=''.join('|' if x==y and x in 'ACGT' else '.' for x,y in zip(a,b))
    exact=guide[k:k+hit.L]==flank[fp:fp+hit.L]
    ext_left=ext_right=0
    if exact:
        while k-ext_left-1>=0 and fp-ext_left-1>=0:
            x,y=guide[k-ext_left-1],flank[fp-ext_left-1]
            if x!=y or x not in 'ACGT':
                break
            ext_left+=1
        while k+hit.L+ext_right<len(guide) and fp+hit.L+ext_right<len(flank):
            x,y=guide[k+hit.L+ext_right],flank[fp+hit.L+ext_right]
            if x!=y or x not in 'ACGT':
                break
            ext_right+=1
    return dict(guide=a,match_line=matches,target_in_guide_orientation=b,
                core_start=left,core_length=hit.L,core_is_exact=exact,
                exact_extension_left=ext_left if exact else None,
                exact_extension_right=ext_right if exact else None,
                total_exact_contiguous_bp=hit.L+ext_left+ext_right if exact else None)


def _row_for_cluster(bag_id: str, cluster_id: int,
                      cluster, bg_report,
                      sites: list[dict],
                      passes_phase_D: bool,
                      publication_threshold: float | None,
                      bag_status: str = "ready",
                      calibration_validated: bool = False):
    nc_pos, L = cluster.nc_pos, cluster.L
    # per-site guide/target sequences at this cluster
    guide_strs = []
    target_strs = []
    evidence_rows = []
    for ss in cluster.supporting_sites:
        nc = _reconstruct_nc(sites, ss.site_idx)
        f  = sites[ss.site_idx]["inputs"]["flank"]
        h = ss.chosen_hit
        raw_np, raw_fp = h.source_nc_pos(), h.source_flank_pos()
        g = nc[raw_np:raw_np + L]
        t = f[raw_fp:raw_fp + L]
        evidence_rows.append(dict(site_idx=ss.site_idx, raw_nc_pos=raw_np,
                                  raw_flank_pos=raw_fp, raw_orient=h.raw_orient or h.orient,
                                  mapped_flank_pos=h.flank_pos, mapped_nc_columns=h.nc_columns,
                                  guide_raw=g, target_raw=t,
                                  target_in_guide_orientation=revcomp(t) if (h.raw_orient or h.orient) == 'rc' else t,
                                  nc_context=nc[max(0, raw_np-30):raw_np+L+30],
                                  nc_context_start=max(0, raw_np-30),
                                  target_context=f[max(0, raw_fp-30):raw_fp+L+30],
                                  target_context_start=max(0, raw_fp-30),
                                  match_count=h.m, rank_within_site=ss.rank_within_site,
                                  local_alignment=_local_anchor_alignment(sites[ss.site_idx], h)))
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
    sites_per_bg_group = (cluster.n_support / n_groups) if n_groups else 0.0

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
        "sites_per_bg_group": f"{sites_per_bg_group:.3f}",
        "mean_m_frac":       cluster.mean_m_frac,
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
                                           cluster.mean_m_frac >= publication_threshold)
                                           if calibration_validated and publication_threshold is not None and bag_status == "ready" else "",
        "passes_exploratory_threshold": (passes_phase_D and cluster.mean_m_frac >= publication_threshold)
                                           if publication_threshold is not None else "",
        "bag_status": bag_status,
        "coordinate_frame": cluster.coordinate_frame,
        "nc_msa_columns": json.dumps(cluster.nc_columns),
        "support_sites_json": json.dumps(evidence_rows),
        "background_pairs_json": json.dumps({f"{i},{j}": v for (i,j),v in bg_report.pairwise_alignment.items()}),
        "background_merge_sources_json": json.dumps({f"{i},{j}": v for (i,j),v in bg_report.merge_sources.items()}),
        "calibration_status": "empirically_validated" if calibration_validated else "uncalibrated",
        # ranks filled after all rows gathered
    }


def _rank_rows(rows):
    passing = [r for r in rows if r['passes_phase_D']]
    for name, key in [
        ('rank_by_bg_groups', lambda r: (-r['n_local_bg_groups'], -r['mean_m_frac'])),
        ('rank_by_mean_m', lambda r: -r['mean_m_frac']),
        ('rank_by_tightness', lambda r: float(r['flank_pos_mad'])),
    ]:
        for rank, row in enumerate(sorted(passing, key=key), 1):
            row[name] = rank
        for row in rows:
            row.setdefault(name, '')


def main():
    from contextlib import ExitStack
    from scripts.v5a_framework.background_alignment import gapped_identity
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--bag-ids', default='', help='comma-separated IDs; empty means all bags')
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--evidence-manifest', type=Path)
    ap.add_argument('--coordinate-mode', choices=('verified','legacy'), default='verified')
    ap.add_argument('--calibration', type=Path)
    ap.add_argument('--publication-threshold', type=float, default=None,
                    help='exploratory threshold only; formal flags require --calibration')
    args = ap.parse_args()
    if args.publication_threshold is not None and not 0 <= args.publication_threshold <= 1:
        ap.error('exploratory threshold must be between zero and one')
    if args.calibration and args.publication_threshold is not None:
        ap.error('choose either a calibration artifact or an exploratory threshold')
    bag_ids = {s.strip() for s in args.bag_ids.split(',') if s.strip()} or None
    bags = _bags_from_jsonl(args.jsonl, bag_ids)
    if not bags:
        ap.error('no bags found in corpus')
    if bag_ids and bag_ids - set(bags):
        ap.error(f'requested bags missing from corpus: {sorted(bag_ids-set(bags))}')
    cfg = PipelineConfig(coordinate_mode=args.coordinate_mode)
    attach_manifest(bags, args.evidence_manifest)
    calibration = read_calibration(args.calibration, cfg) if args.calibration else None
    threshold = calibration['threshold'] if calibration else args.publication_threshold
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = dict(config=effective_configuration(cfg), config_fingerprint=config_fingerprint(cfg),
                    statistic_version=STATISTIC_VERSION, input_jsonl=str(args.jsonl),
                    evidence_manifest=str(args.evidence_manifest) if args.evidence_manifest else None,
                    calibration=calibration, exploratory_threshold=threshold if not calibration else None,
                    complete=False)
    manifest_path = args.outdir/'run_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    merged_cols = CANDIDATE_COLS + ['n_merged','nc_pos_range','L_range','flank_pos_range','support_union']
    totals = dict(candidates=0,rejections=0,merged=0,ready=0,needs_evidence=0)
    t0 = time.time()
    print(f'# Scoring {len(bags)} bags; mode={args.coordinate_mode}; calibrated={bool(calibration)}',flush=True)
    with ExitStack() as stack:
        streams, writers = {}, {}
        for key, filename, columns in [
            ('candidates','candidates_v3.tsv',CANDIDATE_COLS),
            ('rejections','rejections_v3.tsv',REJECTION_COLS),
            ('merged','candidates_v3_merged.tsv',merged_cols),
        ]:
            fh=stack.enter_context((args.outdir/filename).open('w',newline='',encoding='utf-8'))
            streams[key]=fh
            writers[key]=csv.DictWriter(fh,delimiter='\t',fieldnames=columns,extrasaction='ignore')
            writers[key].writeheader()
        summary_fh=stack.enter_context((args.outdir/'bag_summary.jsonl').open('w',encoding='utf-8'))
        for number,(bid,sites) in enumerate(sorted(bags.items()),1):
            gapped_identity.cache_clear()
            result=run_bag(bid,sites,cfg)
            stat=bag_statistics(result)
            rows=[]
            rejected=[]
            for cid,(cluster,bg) in enumerate(result.passing+result.failing,1):
                passes=cid <= len(result.passing)
                rows.append(_row_for_cluster(bid,cid,cluster,bg,sites,passes,threshold,
                                             result.status,bool(calibration)))
                if not passes:
                    rejected.append(dict(bag_id=bid,cluster_id=cid,nc_pos=cluster.nc_pos,L=cluster.L,
                                         flank_pos_median=cluster.flank_pos_median,K_valid=cluster.K_valid,
                                         n_support_raw=cluster.n_support,n_local_bg_groups=bg.n_local_bg_groups,
                                         mean_m_frac=cluster.mean_m_frac,phase='D',
                                         rejection_reason=bg.rejection_reason or ''))
            _rank_rows(rows)
            merged_rows=[]
            for rank,m in enumerate(merge_clusters(result.passing,MergeConfig()),1):
                cluster,bg=m.representative
                row=_row_for_cluster(bid,rank,cluster,bg,sites,True,threshold,result.status,bool(calibration))
                row.update(n_merged=m.n_merged,nc_pos_range=f'{m.nc_pos_min}-{m.nc_pos_max}',
                           L_range=f'{m.L_min}-{m.L_max}',
                           flank_pos_range=f'{m.flank_pos_median_min}-{m.flank_pos_median_max}',
                           support_union=','.join(map(str,m.support_union)))
                merged_rows.append(row)
            _rank_rows(merged_rows)
            # Each bag is written once; no global candidate/hit cache is held.
            for key,rs in [('candidates',rows),('rejections',rejected),('merged',merged_rows)]:
                writers[key].writerows(rs)
                streams[key].flush()
                totals[key]+=len(rs)
            summary=dict(bag_id=bid,K_valid=result.K_valid,**stat,
                         coordinate_report=asdict(result.coordinate_report) if result.coordinate_report else None,
                         n_candidates=len(rows),n_merged=len(merged_rows),
                         calibration_status='empirically_validated' if calibration else 'uncalibrated')
            summary_fh.write(json.dumps(summary)+'\n')
            summary_fh.flush()
            if result.status in totals:
                totals[result.status]+=1
            if number % 25 == 0 or number == len(bags):
                print(f'  [{number}/{len(bags)}] {time.time()-t0:.1f}s {totals}',flush=True)
    manifest.update(complete=True,n_bags=len(bags),totals=totals,elapsed_seconds=time.time()-t0)
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('# Wrote candidate, rejection, merged and per-bag evidence tables to',args.outdir)



if __name__ == "__main__":
    main()
