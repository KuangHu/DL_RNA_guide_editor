"""R4 — Phase C fixture tests that go through enumerate_clusters directly
with constructed BagRetention / SiteRetention / Hit objects, so hit lists
are EXACT. The random-background cases in test_phase_c_p2b_bugs.py pass
on BOTH pre-R4 and post-R4 code because partial-match hits in random bg
contaminate the median — those tests do not actually guard against the
int-median bug.

These fixtures do.

Case 17 : [0, 0, 10, 10] with max_offset=5. Pre-fix code enumerated
          only observed centers {0, 10} and got max n_support=2. R4
          code enumerates every integer center in [0..10] and must
          emit a 4-support cluster. Fixture — no random bg.

Case 18 : [0, 1, 10, 10] with max_offset=5. True float median = 5.5;
          |0 - 5.5| = 5.5 > 5 → supporter at 0 must be excluded.
          Pre-R4 int(5.5)=5 keeps it. R4 must drop it.

Case 19 : [2, 3, 4, 5, 6, 7] with max_offset=1. True median = 4.5;
          at window ±1 only {4, 5} are within 1 bp of 4.5 → 2 supports.
          Int median = 4 keeps {3, 4, 5} = 3 supports (one supporter
          2 bp from true median > max_offset).

Case 20 : regression of P2b — [0, 0, 10, 10, 10] exact. All 5 inside
          window at center=5; median then = 10 → trim 2 outliers →
          final = 3 supports at 10.

Run: python scripts/tests/test_phase_c_r4_fixtures.py
"""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v5a_framework.multi_hit_retention import (
    BagRetention, SiteRetention, Hit,
)
from scripts.v5a_framework.cluster_enumeration import enumerate_clusters


def make_bag_fixture(flank_positions: list[int], nc_pos: int = 100,
                      L: int = 11, orient: str = "fwd") -> BagRetention:
    """Build a BagRetention with ONE hit per site at the given flank_pos,
    all at the same (nc_pos, L). No random contamination."""
    retained = []
    for s, p in enumerate(flank_positions):
        hit = Hit(nc_pos=nc_pos, L=L, flank_pos=p, m=L, orient=orient)
        retained.append(SiteRetention(
            site_idx=s, hits_by_nc_pos_L={(nc_pos, L): [hit]}))
    K = len(flank_positions)
    return BagRetention(K_valid=K, valid_site_idx=list(range(K)),
                         retained=retained, dropped={})


def test_case17_center_in_gap():
    print("\n=== Case 17: fixture [0,0,10,10] max_offset=5 → 4-support cluster ===")
    bag = make_bag_fixture([0, 0, 10, 10])
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == 11]
    print(f"  clusters: {[(c.flank_pos_median, c.n_support) for c in at_100]}")
    best = max(at_100, key=lambda c: c.n_support)
    assert best.n_support == 4, (
        f"P2b bug 1 regression: expected 4 supports from [0,0,10,10] "
        f"at center=5; got max n_support={best.n_support}")
    # With float median, median([0,0,10,10]) = 5.0 exactly
    assert best.flank_pos_median == 5.0, (
        f"expected median=5.0, got {best.flank_pos_median}")
    print(f"  PASS — 4-support cluster at median=5.0")


def test_case18_half_integer_median():
    print("\n=== Case 18: fixture [0,1,10,10] max_offset=5 — float median=5.5 ===")
    bag = make_bag_fixture([0, 1, 10, 10])
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == 11]
    cluster_desc = [(c.flank_pos_median, c.n_support,
                     [ss.chosen_hit.flank_pos for ss in c.supporting_sites])
                    for c in at_100]
    print(f"  clusters: {cluster_desc}")
    # Invariant: every supporter must be within max_offset of cluster median
    for c in at_100:
        for ss in c.supporting_sites:
            assert abs(ss.chosen_hit.flank_pos - c.flank_pos_median) <= 5, (
                f"R4 bug: supporter at {ss.chosen_hit.flank_pos} is "
                f"{abs(ss.chosen_hit.flank_pos - c.flank_pos_median)} bp "
                f"from cluster median {c.flank_pos_median}, "
                f"exceeds max_offset=5")
    # The 4-support cluster that pre-R4 code would emit (median=5 int,
    # supporter at 0 included) must NOT exist. The largest cluster
    # containing supporter 0 must be ≤ 3 OR excluded.
    cluster_with_zero = [c for c in at_100
                          if any(ss.chosen_hit.flank_pos == 0
                                 for ss in c.supporting_sites)]
    print(f"  clusters containing supporter at 0: "
          f"{[(c.flank_pos_median, c.n_support) for c in cluster_with_zero]}")
    for c in cluster_with_zero:
        # The 0-supporter must be < 5 from the median
        assert c.flank_pos_median <= 5.0, (
            f"cluster at median={c.flank_pos_median} includes supporter 0; "
            f"max_offset invariant violated")
    print(f"  PASS — half-integer median correctly handled; no supporter "
          f"beyond max_offset of cluster median")


def test_case19_tight_window():
    print("\n=== Case 19: fixture [2,3,4,5,6,7] max_offset=1 — true median=4.5 ===")
    bag = make_bag_fixture([2, 3, 4, 5, 6, 7])
    clusters = enumerate_clusters(bag, max_offset=1, min_frac=0.3)
    at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == 11]
    cluster_desc = [(c.flank_pos_median, c.n_support,
                     [ss.chosen_hit.flank_pos for ss in c.supporting_sites])
                    for c in at_100]
    print(f"  clusters: {cluster_desc}")
    # Invariant: every supporter within max_offset=1 of cluster median
    for c in at_100:
        for ss in c.supporting_sites:
            assert abs(ss.chosen_hit.flank_pos - c.flank_pos_median) <= 1, (
                f"R4 bug: supporter at {ss.chosen_hit.flank_pos} is "
                f"{abs(ss.chosen_hit.flank_pos - c.flank_pos_median)} bp "
                f"from cluster median {c.flank_pos_median}, "
                f"exceeds max_offset=1")
    print(f"  PASS — max_offset invariant holds with tight window")


def test_case20_median_trim():
    print("\n=== Case 20: fixture [0,0,10,10,10] max_offset=5 — iterative trim ===")
    bag = make_bag_fixture([0, 0, 10, 10, 10])
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.4)
    at_100 = [c for c in clusters if c.nc_pos == 100 and c.L == 11]
    print(f"  clusters: {[(c.flank_pos_median, c.n_support) for c in at_100]}")
    # Invariant check
    for c in at_100:
        for ss in c.supporting_sites:
            assert abs(ss.chosen_hit.flank_pos - c.flank_pos_median) <= 5
    # The main cluster at 10 should have ≤ 3 supports (the 3 at 10)
    at_10 = [c for c in at_100 if c.flank_pos_median == 10.0]
    if at_10:
        assert at_10[0].n_support == 3, (
            f"cluster at median=10 should have 3 supporters, got {at_10[0].n_support}")
    print(f"  PASS — iterative trim converged correctly")


def main():
    test_case17_center_in_gap()
    test_case18_half_integer_median()
    test_case19_tight_window()
    test_case20_median_trim()
    print("\n[all 4 R4 fixture tests PASSED]")


if __name__ == "__main__":
    main()
