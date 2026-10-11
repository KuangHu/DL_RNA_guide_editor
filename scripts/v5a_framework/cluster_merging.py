"""Step 7 — cross-L / cross-nc_pos cluster merging.

Phase C deduplicates within same (nc_pos, L).  This module collapses
clusters that are the same PHYSICAL motif viewed at different L or
offset by a few bp of nc_pos / flank_pos.

Merge rule: two clusters merge iff
  - same bag
  - |nc_pos_i - nc_pos_j| ≤ cfg.nc_offset
  - |flank_pos_median_i - flank_pos_median_j| ≤ cfg.flank_offset
  - Jaccard(supporting_sites_i, supporting_sites_j) ≥ cfg.support_jaccard

Union-find yields groups; representative within each group =
max(mean_m_frac, L) (longest high-m motif wins).

Merged metadata on the representative row:
  n_merged, nc_pos_min, nc_pos_max, L_min, L_max,
  flank_pos_median_min, flank_pos_median_max,
  support_union  (sorted site_idx list, as JSON)
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v5a_framework.cluster_enumeration import Cluster
from scripts.v5a_framework.background_grouping import BgReport


@dataclass(frozen=True)
class MergeConfig:
    nc_offset: int = 5
    flank_offset: int = 5
    support_jaccard: float = 0.80


@dataclass
class MergedRow:
    representative: tuple[Cluster, BgReport]
    n_merged: int
    nc_pos_min: int
    nc_pos_max: int
    L_min: int
    L_max: int
    flank_pos_median_min: int
    flank_pos_median_max: int
    support_union: list[int]
    members: list[tuple[Cluster, BgReport]]


def _support_set(cluster: Cluster) -> set[int]:
    return {ss.site_idx for ss in cluster.supporting_sites}


def _rep_sort_key(c: Cluster):
    # higher mean_m, then larger L, then tighter MAD
    return (-c.mean_m_frac, -c.L, c.flank_pos_mad)


def merge_clusters(items: list[tuple[Cluster, BgReport]],
                    cfg: MergeConfig = MergeConfig(),
                    ) -> list[MergedRow]:
    """Greedy single-anchor merge.  Clusters sorted best-first; each absorbed
    into an existing group only if it is within merge-distance of that group's
    anchor (the first-added, i.e. best, member).  This bounds group diameter
    at 2 * cfg.nc_offset in nc and 2 * cfg.flank_offset in flank, avoiding
    transitive chain-linking across unrelated motifs."""
    if not items: return []
    sets = {id(c): _support_set(c) for (c, _) in items}
    items_sorted = sorted(items, key=lambda cb: _rep_sort_key(cb[0]))

    groups: list[list[tuple[Cluster, BgReport]]] = []
    for item in items_sorted:
        cluster, _ = item
        absorbed = False
        for grp in groups:
            anchor_cluster, _ = grp[0]
            if abs(cluster.nc_pos - anchor_cluster.nc_pos) > cfg.nc_offset:
                continue
            if abs(cluster.flank_pos_median - anchor_cluster.flank_pos_median) > cfg.flank_offset:
                continue
            inter = len(sets[id(cluster)] & sets[id(anchor_cluster)])
            uni   = len(sets[id(cluster)] | sets[id(anchor_cluster)])
            if uni == 0: continue
            if (inter / uni) >= cfg.support_jaccard:
                grp.append(item)
                absorbed = True
                break
        if not absorbed:
            groups.append([item])

    out: list[MergedRow] = []
    for grp in groups:
        rep = grp[0]  # anchor = best
        all_c = [cb[0] for cb in grp]
        union_sites: set[int] = set()
        for c in all_c:
            union_sites |= sets[id(c)]
        out.append(MergedRow(
            representative=rep,
            n_merged=len(all_c),
            nc_pos_min=min(c.nc_pos for c in all_c),
            nc_pos_max=max(c.nc_pos for c in all_c),
            L_min=min(c.L for c in all_c),
            L_max=max(c.L for c in all_c),
            flank_pos_median_min=min(c.flank_pos_median for c in all_c),
            flank_pos_median_max=max(c.flank_pos_median for c in all_c),
            support_union=sorted(union_sites),
            members=grp,
        ))
    out.sort(key=lambda r: (-r.representative[0].mean_m_frac,
                              -r.n_merged,
                              r.representative[0].flank_pos_mad))
    return out
