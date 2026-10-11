"""Phase C of model-free scorer v3 — exhaustive position-cluster enumeration.

Consumes BagRetention from Phase A and finds across-site flank_pos clusters
by enumerating a window around EVERY observed hit position (not iterating
from a single seed median).  Each site contributes at most one hit per
window — the one closest to the window center.

Correctness points vs any seed-median approach:
  - Every observed flank_pos is considered as a window center → recovery of
    a non-best cross-site cluster is independent of which hit happens to
    appear first.
  - Each site contributes at most one hit per window; the chosen hit's
    rank_within_site is recorded.
  - Support fraction uses the FROZEN K_valid from Phase A as denominator,
    not the per-cluster site count.
  - Overlapping windows are deduplicated by Jaccard overlap of supporting
    sites + distance between medians (keep the tighter MAD).

Public API:
  enumerate_clusters(bag, max_offset=5, min_frac=0.5) -> list[Cluster]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys
from statistics import median
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v5a_framework.multi_hit_retention import BagRetention, Hit


@dataclass
class SupportingSite:
    """One site's contribution to one cluster."""
    site_idx: int
    chosen_hit: Hit
    rank_within_site: int    # 1 = this site's best hit at (nc_pos, L)
    offset_from_median: int  # chosen_hit.flank_pos - cluster.flank_pos_median
    n_hits_this_site_at_nc_pos_L: int   # multiplicity signal (ambiguity)


@dataclass
class Cluster:
    nc_pos: int
    L: int
    flank_pos_median: int
    flank_pos_mad: float          # median absolute deviation
    flank_pos_min: int
    flank_pos_max: int
    n_support: int                # == len(supporting_sites)
    support_frac: float           # n_support / K_valid
    K_valid: int                  # denominator (passthrough)
    supporting_sites: list[SupportingSite]
    mean_m_frac: float            # mean(m)/L over supporting sites
    min_m_frac: float
    n_sites_cluster_hit_is_best: int    # count where rank_within_site == 1
    mean_hit_multiplicity_at_nc_pos_L: float
    orient_counts: dict            # {'fwd': int, 'rc': int}

    def key(self) -> tuple[int, int, int]:
        return (self.nc_pos, self.L, self.flank_pos_median)


def _mad(xs: list[int]) -> float:
    if not xs:
        return 0.0
    m = median(xs)
    return float(median(abs(x - m) for x in xs))


def _per_site_hit_by_nc_pos_L(bag: BagRetention) -> dict[tuple[int, int], dict[int, list[Hit]]]:
    """Reindex: (nc_pos, L) -> {site_idx -> hits_sorted_best_first}."""
    out: dict[tuple[int, int], dict[int, list[Hit]]] = defaultdict(dict)
    for sr in bag.retained:
        for (nc_pos, L), hits in sr.hits_by_nc_pos_L.items():
            # already sorted by (-m, flank_pos) at retention
            out[(nc_pos, L)][sr.site_idx] = hits
    return out


def _ranked_hits(site_hits: list[Hit]) -> list[tuple[int, Hit]]:
    """Return [(rank, hit), ...] where rank 1 = best (highest m, tie-break by flank_pos)."""
    # Already sorted best-first by (-m, flank_pos) in multi_hit_retention._group_by_nc_pos_L
    return [(r + 1, h) for r, h in enumerate(site_hits)]


def enumerate_clusters(bag: BagRetention,
                        max_offset: int = 5,
                        min_frac: float = 0.5,
                        dedup_jaccard_min: float = 0.80,
                        ) -> list[Cluster]:
    """Enumerate all (nc_pos, L) windows whose support_frac >= min_frac."""
    K_valid = bag.K_valid
    if K_valid == 0:
        return []
    min_support = max(1, int(min_frac * K_valid + 0.9999))  # ceil
    by_nc_pos_L = _per_site_hit_by_nc_pos_L(bag)

    all_clusters: list[Cluster] = []
    for (nc_pos, L), per_site in by_nc_pos_L.items():
        # candidate window centers = every observed flank_pos (deduped)
        centers = sorted({h.flank_pos for site_hits in per_site.values() for h in site_hits})
        if not centers:
            continue

        # per-site hit lists with ranks
        ranked_by_site: dict[int, list[tuple[int, Hit]]] = {
            s: _ranked_hits(hits) for s, hits in per_site.items()
        }
        # multiplicity = ties-for-best per site at this (nc_pos, L).  Partial
        # matches at nearby offsets inflate a raw hit count but are not real
        # ambiguity — only hits that tie the per-site best m are counted.
        ties_for_best_by_site: dict[int, int] = {}
        for s, hits in per_site.items():
            if not hits:
                ties_for_best_by_site[s] = 0; continue
            best_m = max(h.m for h in hits)
            ties_for_best_by_site[s] = sum(1 for h in hits if h.m == best_m)

        candidate_clusters: list[Cluster] = []
        for center in centers:
            lo, hi = center - max_offset, center + max_offset
            supporting: list[SupportingSite] = []
            # for each site, choose the hit in [lo, hi] closest to center;
            # if no hit in window, site does not support this cluster
            for s, ranked in ranked_by_site.items():
                in_win = [(r, h) for r, h in ranked if lo <= h.flank_pos <= hi]
                if not in_win:
                    continue
                in_win.sort(key=lambda rh: (abs(rh[1].flank_pos - center),
                                             -rh[1].m, rh[1].flank_pos))
                r, h = in_win[0]
                supporting.append(SupportingSite(
                    site_idx=s, chosen_hit=h, rank_within_site=r,
                    offset_from_median=0,  # fill after we know the median
                    n_hits_this_site_at_nc_pos_L=ties_for_best_by_site[s],
                ))
            if len(supporting) < min_support:
                continue

            positions = [ss.chosen_hit.flank_pos for ss in supporting]
            fpm = int(median(positions))
            fp_mad = _mad(positions)
            # refill offset_from_median
            for ss in supporting:
                ss.offset_from_median = ss.chosen_hit.flank_pos - fpm
            m_fracs = [ss.chosen_hit.m / L for ss in supporting]
            orient_counts = {"fwd": 0, "rc": 0}
            for ss in supporting:
                orient_counts[ss.chosen_hit.orient] += 1
            mean_mult = sum(ties_for_best_by_site[ss.site_idx]
                              for ss in supporting) / len(supporting)
            n_best = sum(1 for ss in supporting if ss.rank_within_site == 1)
            candidate_clusters.append(Cluster(
                nc_pos=nc_pos, L=L,
                flank_pos_median=fpm, flank_pos_mad=fp_mad,
                flank_pos_min=min(positions), flank_pos_max=max(positions),
                n_support=len(supporting),
                support_frac=len(supporting) / K_valid,
                K_valid=K_valid,
                supporting_sites=supporting,
                mean_m_frac=sum(m_fracs) / len(m_fracs),
                min_m_frac=min(m_fracs),
                n_sites_cluster_hit_is_best=n_best,
                mean_hit_multiplicity_at_nc_pos_L=mean_mult,
                orient_counts=orient_counts,
            ))

        # Dedup bleed-through: candidates at nearby offsets that come from
        # partial matches of the same underlying motif copy share supporting
        # sites.  Rule: if two candidates share >= dedup_jaccard_min of
        # supporting sites AND medians are within `max_offset` of each other
        # (full window width, not half), keep the one with higher mean_m_frac,
        # breaking ties by tighter MAD.
        candidate_clusters.sort(key=lambda c: (-c.n_support,
                                                 -c.mean_m_frac,
                                                 c.flank_pos_mad))
        kept: list[Cluster] = []
        for cand in candidate_clusters:
            cand_sites = {ss.site_idx for ss in cand.supporting_sites}
            is_dup = False
            for k in kept:
                k_sites = {ss.site_idx for ss in k.supporting_sites}
                jaccard = (len(cand_sites & k_sites)
                           / len(cand_sites | k_sites))
                close = abs(cand.flank_pos_median - k.flank_pos_median) <= max_offset
                if jaccard >= dedup_jaccard_min and close:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(cand)
        all_clusters.extend(kept)

    all_clusters.sort(key=lambda c: (-c.n_support, c.flank_pos_mad, c.nc_pos))
    return all_clusters
