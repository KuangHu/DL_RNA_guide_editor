"""Step-7 cluster-merging unit tests.

Covers:
  merge_case_1 : sliding-window variants of one physical motif (nc_pos
                  differs by 1, L differs by 1, same support set, same
                  flank_pos within ±2) all collapse to one group.
  merge_case_2 : chain-link bound.  Clusters at nc_pos 0, 5, 10, 15, 20
                  share supporting sites but are spaced 5 apart each.
                  Greedy single-anchor keeps the diameter bounded; two
                  distinct 'anchors' (0 and 10, or similar) should emerge
                  rather than one chain-linked group of 5.
  merge_case_3 : different supporting-site sets do NOT merge even when
                  nc_pos and flank_pos agree.
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v5a_framework.cluster_enumeration import Cluster, SupportingSite
from scripts.v5a_framework.multi_hit_retention import Hit
from scripts.v5a_framework.background_grouping import BgReport, BgGroup
from scripts.v5a_framework.cluster_merging import merge_clusters, MergeConfig


def _mk_cluster(nc_pos, L, flank_pos_median, mean_m, support_ids, K_valid=8):
    """Build a Cluster + stub BgReport for merge testing."""
    hits = [Hit(nc_pos=nc_pos, L=L, flank_pos=flank_pos_median,
                 m=int(round(mean_m * L)), orient="fwd") for _ in support_ids]
    supporting = [SupportingSite(site_idx=sid, chosen_hit=h,
                                   rank_within_site=1, offset_from_median=0,
                                   n_hits_this_site_at_nc_pos_L=1)
                  for sid, h in zip(support_ids, hits)]
    c = Cluster(
        nc_pos=nc_pos, L=L,
        flank_pos_median=flank_pos_median, flank_pos_mad=0.0,
        flank_pos_min=flank_pos_median, flank_pos_max=flank_pos_median,
        n_support=len(support_ids), support_frac=len(support_ids) / K_valid,
        K_valid=K_valid, supporting_sites=supporting,
        mean_m_frac=mean_m, min_m_frac=mean_m,
        n_sites_cluster_hit_is_best=len(support_ids),
        mean_hit_multiplicity_at_nc_pos_L=1.0,
        orient_counts={"fwd": len(support_ids), "rc": 0},
    )
    bg = BgReport(
        cluster_key=c.key(),
        n_local_bg_groups=len(support_ids),
        groups=[BgGroup(members=[sid], chain_linked=False,
                         min_pair_identity=1.0, max_pair_identity=1.0)
                for sid in support_ids],
        pairwise_identities={}, masked_flank_per_site={},
        any_chain_linked=False,
        n_bp_left_available={}, n_bp_right_available={},
        edge_insufficient_left=set(), edge_insufficient_right=set(),
        edge_insufficient_both=set(),
        pairwise_left_identity={}, pairwise_right_identity={},
        median_left_identity=None, median_right_identity=None,
        passes_bg_gate=True, rejection_reason=None,
    )
    return (c, bg)


def test_merge_case_1_sliding_variants():
    print("\n=== merge_case_1: sliding-window variants collapse ===")
    # Same support set {0,1,2,3}, same flank_pos (±2), nc_pos +/-1, L varies
    items = [
        _mk_cluster(nc_pos=10, L=9,  flank_pos_median=40, mean_m=1.0, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=10, L=11, flank_pos_median=40, mean_m=1.0, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=11, L=10, flank_pos_median=41, mean_m=1.0, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=12, L=12, flank_pos_median=39, mean_m=1.0, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=10, L=14, flank_pos_median=40, mean_m=1.0, support_ids=[0,1,2,3]),
    ]
    merged = merge_clusters(items, MergeConfig())
    print(f"  5 input clusters → {len(merged)} merged groups")
    for m in merged:
        rep, _ = m.representative
        print(f"    rep nc={rep.nc_pos} L={rep.L} fp={rep.flank_pos_median}  "
              f"n_merged={m.n_merged}  nc_range={m.nc_pos_min}-{m.nc_pos_max}  "
              f"L_range={m.L_min}-{m.L_max}")
    assert len(merged) == 1, f"expected 1 merged group, got {len(merged)}"
    assert merged[0].n_merged == 5
    # Representative should be the largest L (tie-break on mean_m)
    assert merged[0].representative[0].L == 14
    print("  PASS")


def test_merge_case_2_chain_link_bound():
    print("\n=== merge_case_2: chain-link bound (greedy single-anchor) ===")
    # Clusters at nc_pos 0, 5, 10, 15, 20 all with the same support set.
    # Pairwise adjacency within nc_offset=5 forms a chain under union-find
    # (0-5-10-15-20).  Under greedy single-anchor, the anchor (whichever
    # comes first after sorting by mean_m) absorbs up to ±5; the rest start
    # new groups.  Expect at least 2 groups (diameter bounded).
    items = [
        _mk_cluster(nc_pos= 0, L=11, flank_pos_median=50, mean_m=0.95, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos= 5, L=11, flank_pos_median=50, mean_m=0.95, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=10, L=11, flank_pos_median=50, mean_m=0.95, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=15, L=11, flank_pos_median=50, mean_m=0.95, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=20, L=11, flank_pos_median=50, mean_m=0.95, support_ids=[0,1,2,3]),
    ]
    merged = merge_clusters(items, MergeConfig())
    print(f"  5 chain-adjacent clusters (nc 0..20) → {len(merged)} groups")
    for m in merged:
        print(f"    anchor_nc={m.representative[0].nc_pos}  "
              f"n_merged={m.n_merged}  nc_range={m.nc_pos_min}-{m.nc_pos_max}")
    assert len(merged) >= 2, (
        f"chain of 5 at step=5 should NOT collapse to one group under "
        f"greedy single-anchor; got {len(merged)}")
    # Diameter bound: each group's nc_pos range ≤ 2 * nc_offset
    for m in merged:
        assert m.nc_pos_max - m.nc_pos_min <= 2 * MergeConfig().nc_offset
    print("  PASS — group diameter bounded, chain-link avoided")


def test_merge_case_3_distinct_support_sets():
    print("\n=== merge_case_3: distinct support sets do NOT merge ===")
    items = [
        # Same nc_pos / flank_pos / L, but completely different supporting sites
        _mk_cluster(nc_pos=50, L=11, flank_pos_median=30, mean_m=0.9, support_ids=[0,1,2,3]),
        _mk_cluster(nc_pos=50, L=11, flank_pos_median=30, mean_m=0.9, support_ids=[4,5,6,7]),
    ]
    merged = merge_clusters(items, MergeConfig())
    print(f"  2 disjoint-support clusters → {len(merged)} groups")
    for m in merged:
        print(f"    rep support={sorted([ss.site_idx for ss in m.representative[0].supporting_sites])}  "
              f"n_merged={m.n_merged}")
    assert len(merged) == 2
    print("  PASS — disjoint support kept separate")


def main():
    test_merge_case_1_sliding_variants()
    test_merge_case_2_chain_link_bound()
    test_merge_case_3_distinct_support_sets()
    print("\n[all 3 cluster-merging tests PASSED]")


if __name__ == "__main__":
    main()
