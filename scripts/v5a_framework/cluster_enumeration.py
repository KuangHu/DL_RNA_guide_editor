"""Phase C of model-free scorer v3 — exhaustive position-cluster enumeration.

Consumes BagRetention from Phase A and finds across-site flank_pos clusters.

**P2b fixes vs the original (2026-10-11):**

  bug 1  Original enumerated candidate window centers at `{h.flank_pos for h
         in all_hits}` ONLY. For positions [0, 0, 10, 10] with max_offset=5
         this yields centers {0, 10} and misses the 4-support cluster whose
         natural center is 5. Fixed: candidate centers = every integer in
         [min_pos, max_pos], so EVERY possible 2*max_offset+1 window is
         enumerated.
  bug 2  After supports were picked from the window, the original computed
         the cluster median but never re-checked that each supporting hit was
         still within max_offset of the median. Fixed: iterative trim —
         compute median → remove any supporter beyond max_offset of median
         → recompute → repeat until stable OR n_support < min_support.

Public API: `enumerate_clusters(bag, max_offset=5, min_frac=0.5)`.
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
    rank_within_site: int       # 1 = this site's best hit at (nc_pos, L)
    offset_from_median: float   # R4: float (chosen_hit.flank_pos - cluster.flank_pos_median)
    n_hits_this_site_at_nc_pos_L: int   # multiplicity signal (ambiguity)


@dataclass
class Cluster:
    nc_pos: int
    L: int
    flank_pos_median: float       # R4: float (half-bp median preserved)
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
    coordinate_frame: str = "legacy_raw"
    nc_columns: tuple[int, ...] = ()

    def key(self) -> tuple[int, int, float]:
        return (self.nc_pos, self.L, self.flank_pos_median)


def _mad(xs: list[int]) -> float:
    if not xs:
        return 0.0
    m = median(xs)
    return float(median(abs(x - m) for x in xs))


def _per_site_hit_by_nc_pos_L(bag: BagRetention) -> dict[tuple, dict[int, list[Hit]]]:
    """Reindex: (nc_pos, L) -> {site_idx -> hits_sorted_best_first}."""
    out: dict[tuple, dict[int, list[Hit]]] = defaultdict(dict)
    for sr in bag.retained:
        for key, hits in sr.hits_by_nc_pos_L.items():
            out[key][sr.site_idx] = hits
    return out


def _ranked_hits(site_hits: list[Hit]) -> list[tuple[int, Hit]]:
    """[(rank, hit), ...] where rank 1 = best. Already sorted best-first."""
    return [(r + 1, h) for r, h in enumerate(site_hits)]


def _select_supporters_at_center(center: float,
                                   ranked_by_site: dict[int, list[tuple[int, Hit]]],
                                   ties_for_best_by_site: dict[int, int],
                                   max_offset: int) -> list[SupportingSite]:
    """For each site pick the hit in [center-max_offset, center+max_offset]
    closest to center. Returns [] when no site qualifies."""
    lo, hi = center - max_offset, center + max_offset
    out: list[SupportingSite] = []
    for s, ranked in ranked_by_site.items():
        in_win = [(r, h) for r, h in ranked if lo <= h.flank_pos <= hi]
        if not in_win:
            continue
        in_win.sort(key=lambda rh: (abs(rh[1].flank_pos - center),
                                     -rh[1].m, rh[1].flank_pos))
        r, h = in_win[0]
        out.append(SupportingSite(
            site_idx=s, chosen_hit=h, rank_within_site=r,
            offset_from_median=0,
            n_hits_this_site_at_nc_pos_L=ties_for_best_by_site[s],
        ))
    return out


def _select_supporters_at_center_float(center: float,
                                          ranked_by_site: dict[int, list[tuple[int, Hit]]],
                                          ties_for_best_by_site: dict[int, int],
                                          max_offset: int) -> list[SupportingSite]:
    """Like _select_supporters_at_center but center may be fractional."""
    lo, hi = center - max_offset, center + max_offset
    out: list[SupportingSite] = []
    for s, ranked in ranked_by_site.items():
        in_win = [(r, h) for r, h in ranked if lo <= h.flank_pos <= hi]
        if not in_win:
            continue
        in_win.sort(key=lambda rh: (abs(rh[1].flank_pos - center),
                                     -rh[1].m, rh[1].flank_pos))
        r, h = in_win[0]
        out.append(SupportingSite(
            site_idx=s, chosen_hit=h, rank_within_site=r,
            offset_from_median=0.0,
            n_hits_this_site_at_nc_pos_L=ties_for_best_by_site[s],
        ))
    return out


def _iterative_trim_to_median_window(supporters: list[SupportingSite],
                                       max_offset: int,
                                       min_support: int,
                                       ranked_by_site: dict[int, list[tuple[int, Hit]]],
                                       ties_for_best_by_site: dict[int, int],
                                       max_iter: int = 20,
                                       ) -> tuple[list[SupportingSite], float]:
    """R4: iterate using FLOAT median. For [0,1,10,10] the true median is
    5.5 — truncating to 5 kept a supporter 5.5 bp from the true median
    inside a max_offset=5 window. Float median tightens the window to
    [0.5, 10.5] and correctly drops the supporter at 0.

    Returns (final_supporters, final_median_or_-1.0_if_empty).
    """
    prev_sig: tuple | None = None
    for _ in range(max_iter):
        if len(supporters) < min_support:
            return [], -1.0
        positions = [ss.chosen_hit.flank_pos for ss in supporters]
        fpm = float(median(positions))
        rebuilt = _select_supporters_at_center_float(
            fpm, ranked_by_site, ties_for_best_by_site, max_offset)
        if len(rebuilt) < min_support:
            return [], -1.0
        new_positions = [ss.chosen_hit.flank_pos for ss in rebuilt]
        new_fpm = float(median(new_positions))
        new_sig = (new_fpm, tuple(sorted((ss.site_idx, ss.chosen_hit.flank_pos)
                                           for ss in rebuilt)))
        if new_sig == prev_sig:
            final = [ss for ss in rebuilt
                      if abs(ss.chosen_hit.flank_pos - new_fpm) <= max_offset]
            if len(final) < min_support:
                return [], -1.0
            return final, new_fpm
        prev_sig = new_sig
        supporters = rebuilt
    # A final trim can change the median again. An unconverged window has
    # not met the public invariant and must not be emitted as a cluster.
    return [], -1.0


def enumerate_clusters(bag: BagRetention,
                        max_offset: int = 5,
                        min_frac: float = 0.5,
                        dedup_jaccard_min: float = 0.80,
                        ) -> list[Cluster]:
    """Enumerate all (nc_pos, L) windows whose support_frac >= min_frac.

    P2b correctness invariants:
      - Centers = every integer in [min_observed, max_observed] per (nc_pos, L).
      - After supports selected, iterative trim ensures every supporting hit
        is within max_offset of the cluster median.
      - Each site contributes at most one hit per cluster.
      - support_frac denominator = frozen K_valid (not per-cluster site count).
    """
    K_valid = bag.K_valid
    if K_valid == 0:
        return []
    min_support = max(1, int(min_frac * K_valid + 0.9999))
    by_nc_pos_L = _per_site_hit_by_nc_pos_L(bag)

    all_clusters: list[Cluster] = []
    for group_key, per_site in by_nc_pos_L.items():
        nc_pos, L = group_key[:2]
        observed = sorted({h.flank_pos for site_hits in per_site.values()
                            for h in site_hits})
        if not observed:
            continue
        ranked_by_site = {s: _ranked_hits(hits) for s, hits in per_site.items()}
        ties_for_best_by_site: dict[int, int] = {}
        for s, hits in per_site.items():
            if not hits:
                ties_for_best_by_site[s] = 0; continue
            best_m = max(h.m for h in hits)
            ties_for_best_by_site[s] = sum(1 for h in hits if h.m == best_m)

        # P2b bug-1 fix: enumerate EVERY integer center in [min, max].
        centers = range(observed[0], observed[-1] + 1)
        candidate_clusters: list[Cluster] = []
        seen_centers: set[tuple] = set()
        for center in centers:
            supporters = _select_supporters_at_center(
                center, ranked_by_site, ties_for_best_by_site, max_offset)
            if len(supporters) < min_support:
                continue
            # P2b bug-2 fix: iterative trim to median window
            supporters, fpm = _iterative_trim_to_median_window(
                supporters, max_offset, min_support,
                ranked_by_site, ties_for_best_by_site)
            if len(supporters) < min_support:
                continue
            # Dedup identical cluster signatures by (median, support set)
            key = (fpm, frozenset(ss.site_idx for ss in supporters))
            if key in seen_centers:
                continue
            seen_centers.add(key)

            positions = [ss.chosen_hit.flank_pos for ss in supporters]
            fp_mad = _mad(positions)
            for ss in supporters:
                ss.offset_from_median = float(ss.chosen_hit.flank_pos) - fpm
            m_fracs = [ss.chosen_hit.m / L for ss in supporters]
            orient_counts = {"fwd": 0, "rc": 0}
            for ss in supporters:
                orient_counts[ss.chosen_hit.orient] += 1
            mean_mult = sum(ties_for_best_by_site[ss.site_idx]
                              for ss in supporters) / len(supporters)
            n_best = sum(1 for ss in supporters if ss.rank_within_site == 1)
            candidate_clusters.append(Cluster(
                nc_pos=nc_pos, L=L,
                flank_pos_median=fpm, flank_pos_mad=fp_mad,
                flank_pos_min=min(positions), flank_pos_max=max(positions),
                n_support=len(supporters),
                support_frac=len(supporters) / K_valid,
                K_valid=K_valid,
                supporting_sites=supporters,
                mean_m_frac=sum(m_fracs) / len(m_fracs),
                min_m_frac=min(m_fracs),
                n_sites_cluster_hit_is_best=n_best,
                mean_hit_multiplicity_at_nc_pos_L=mean_mult,
                orient_counts=orient_counts,
                coordinate_frame=supporters[0].chosen_hit.coordinate_frame,
                nc_columns=supporters[0].chosen_hit.nc_columns,
            ))

        # Dedup bleed-through: candidates at nearby offsets sharing supports
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
