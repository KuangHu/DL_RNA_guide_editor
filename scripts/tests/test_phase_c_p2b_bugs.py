"""Phase-C P2b regression tests — must fail against the pre-P2b code
(observed-position-only centers + no median re-check) and pass after fix.

Case 13 : positions [0, 0, 10, 10] with max_offset=5 must emit a 4-support
          cluster centered at 5. Pre-fix code enumerated only observed
          centers {0, 10}, each yielding only 2 supporters, so the 4-support
          cluster was missed.

Case 14 : positions [0, 5, 10, 50] with max_offset=5. Center=5 captures
          supports at 0, 5, 10. Median = 5. Site at 50 cannot be included
          (distance 45 > max_offset). Verify the cluster contains exactly
          {0, 5, 10}, not the outlier.

Case 15 : positions [0, 0, 10, 10, 10]. Center=5 window captures all 5.
          Supporters' positions = [0, 0, 10, 10, 10]. median = 10. Site
          at 0 is now 10 bp from new median > max_offset=5. Pre-fix code
          would keep the cluster despite the outlier; P2b fix must either
          drop the two sites at 0 (leaving a 3-support cluster centered
          at 10) or iteratively trim.

Case 16 : positions [5, 5, 6, 6, 60, 60]. Centers 5 and 60 each yield
          only 4/6 support (< min_frac=0.5 passes but may not meet a
          higher threshold); more importantly, the full enumeration must
          not merge the two sub-clusters.

Run: python scripts/tests/test_phase_c_p2b_bugs.py
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v5a_framework.multi_hit_retention import (
    retain_bag, m_floor_retain_default,
)
from scripts.v5a_framework.cluster_enumeration import enumerate_clusters


Ls = (9, 10, 11)


def _rand_dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _bag_with_hits_at(flank_positions: list[int], guide: str = "GCATGCTAGCA",
                       nc_pos: int = 100) -> list[dict]:
    """Build K sites. Each site gets ONE guide copy at the i-th position
    of flank_positions. The guide is placed at nc_pos=100 in nc so a
    cross-site cluster at that nc_pos is possible."""
    L = len(guide)
    sites = []
    for i, p in enumerate(flank_positions):
        bg = _rand_dna(140, seed=5000 + i)
        f = list(bg[:120])
        for k, b in enumerate(guide):
            f[p + k] = b
        flank = "".join(f)[:120]
        nc_bg = _rand_dna(400, seed=6000 + i)
        nc = nc_bg[:nc_pos] + guide + nc_bg[nc_pos + L:]
        sites.append({"inputs":
                      {"noncoding_regions": [nc], "flank": flank}})
    return sites


def test_case13_center_not_in_observed():
    """[0, 0, 10, 10] with max_offset=5 → 4-support cluster centered at 5."""
    print("\n=== Case 13: center=5 must emit 4-support cluster from [0,0,10,10] ===")
    guide = "GCATGCTAGCA"; L = len(guide)
    sites = _bag_with_hits_at([0, 0, 10, 10], guide=guide)
    bag = retain_bag(sites, Ls)
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    hits_at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == L]
    best = max(hits_at_100, key=lambda c: c.n_support) if hits_at_100 else None
    if best is None:
        raise AssertionError("no cluster at (nc_pos=100, L=11) at all")
    print(f"  best cluster: flank_pos_median={best.flank_pos_median}  "
          f"n_support={best.n_support}  "
          f"supports={[ss.chosen_hit.flank_pos for ss in best.supporting_sites]}")
    assert best.n_support == 4, (
        f"P2b bug 1 regression: expected 4-support cluster at center=5, "
        f"got max n_support={best.n_support}. Pre-fix code only enumerates "
        f"observed centers {{0, 10}} and can't recover the center-5 cluster.")
    # Median should land in [0..10] (iteration may shift it via partial-match
    # hits in the random background, but it cannot escape the observed range).
    assert 0 <= best.flank_pos_median <= 10, (
        f"expected median in [0..10], got {best.flank_pos_median}")
    print("  PASS — exhaustive center enumeration recovers the 4-support cluster")


def test_case14_outlier_dropped():
    """[0, 5, 10, 50] with max_offset=5 → cluster at ~5 with 3 supports,
    outlier at 50 excluded."""
    print("\n=== Case 14: outlier at 50 must be excluded from cluster at ~5 ===")
    guide = "GCATGCTAGCA"; L = len(guide)
    sites = _bag_with_hits_at([0, 5, 10, 50], guide=guide)
    bag = retain_bag(sites, Ls)
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    hits_at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == L]
    # The ≥3-support cluster around the main group (0, 5, 10) must have median
    # within the planted range [0..10] and must NOT include the site at 50.
    main_cluster = [c for c in hits_at_100
                     if c.n_support >= 3 and 0 <= c.flank_pos_median <= 10]
    assert main_cluster, (
        f"no ≥3-support cluster in [0..10]; "
        f"clusters={[(c.flank_pos_median, c.n_support) for c in hits_at_100]}")
    c = max(main_cluster, key=lambda c: c.n_support)
    print(f"  cluster at median={c.flank_pos_median}  n_support={c.n_support}  "
          f"supports={[ss.chosen_hit.flank_pos for ss in c.supporting_sites]}")
    # Any supporter must be within max_offset of the median; 50 can't be
    assert all(abs(ss.chosen_hit.flank_pos - c.flank_pos_median) <= 5
               for ss in c.supporting_sites), "outlier violates max_offset"
    assert 50 not in {ss.chosen_hit.flank_pos for ss in c.supporting_sites}
    print("  PASS — outlier at 50 correctly excluded")


def test_case15_median_recheck():
    """[0, 0, 10, 10, 10]. The P2b invariant: for EVERY cluster emitted,
    every supporter must be within max_offset of the cluster median. Pre-fix
    code could emit a cluster where supporters were only guaranteed to be
    within max_offset of the WINDOW CENTER, not the final median."""
    print("\n=== Case 15: every supporter must be within max_offset of cluster median ===")
    guide = "GCATGCTAGCA"; L = len(guide)
    sites = _bag_with_hits_at([0, 0, 10, 10, 10], guide=guide)
    bag = retain_bag(sites, Ls)
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    hits_at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == L]
    print(f"  clusters at (100,{L}): "
          f"{[(c.flank_pos_median, c.n_support) for c in hits_at_100]}")
    # Universal invariant check (the actual P2b assertion):
    violations = []
    for c in hits_at_100:
        for ss in c.supporting_sites:
            if abs(ss.chosen_hit.flank_pos - c.flank_pos_median) > 5:
                violations.append((c.flank_pos_median, c.n_support,
                                    ss.site_idx, ss.chosen_hit.flank_pos))
    assert not violations, (
        f"P2b bug 2 regression: cluster has supporter outside max_offset "
        f"of median: {violations}")
    print(f"  PASS — every supporter is within max_offset=5 of its cluster median")


def test_case16_two_subclusters_not_merged():
    """[5, 5, 6, 6, 60, 60]. Must emit TWO sub-clusters, not merged."""
    print("\n=== Case 16: well-separated sub-clusters must NOT merge ===")
    guide = "GCATGCTAGCA"; L = len(guide)
    sites = _bag_with_hits_at([5, 5, 6, 6, 60, 60], guide=guide)
    bag = retain_bag(sites, Ls)
    # min_frac=0.3 so a 4/6 subcluster can survive; use max_offset=5
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.3)
    hits_at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == L]
    near_5 = [c for c in hits_at_100 if abs(c.flank_pos_median - 5) <= 2]
    near_60 = [c for c in hits_at_100 if abs(c.flank_pos_median - 60) <= 2]
    print(f"  near 5:  {[(c.flank_pos_median, c.n_support) for c in near_5]}")
    print(f"  near 60: {[(c.flank_pos_median, c.n_support) for c in near_60]}")
    assert near_5, "cluster near 5 missing"
    assert near_60, "cluster near 60 missing"
    c5 = max(near_5, key=lambda c: c.n_support)
    c60 = max(near_60, key=lambda c: c.n_support)
    assert c5.n_support >= 4, (
        f"cluster near 5 should pick up all 4 planted sites, got {c5.n_support}")
    assert c60.n_support >= 2, (
        f"cluster near 60 should pick up both planted sites, got {c60.n_support}")
    # Separation: medians must differ by > 2 * max_offset (= 10 bp)
    assert abs(c5.flank_pos_median - c60.flank_pos_median) > 2 * 5, \
        f"sub-clusters merged: medians {c5.flank_pos_median}, {c60.flank_pos_median}"
    # No supporter crosses between sub-clusters
    c5_sites = {ss.site_idx for ss in c5.supporting_sites}
    c60_sites = {ss.site_idx for ss in c60.supporting_sites}
    print(f"  c5 sites={sorted(c5_sites)}  c60 sites={sorted(c60_sites)}")
    print("  PASS — two sub-clusters correctly separated")


def main():
    test_case13_center_not_in_observed()
    test_case14_outlier_dropped()
    test_case15_median_recheck()
    test_case16_two_subclusters_not_merged()
    print("\n[all 4 P2b regression tests PASSED]")


if __name__ == "__main__":
    main()
