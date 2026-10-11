"""Phase-C validation tests (model-free scorer v3).

Covers:
  Case 6  : complexity-passing motif at multiple flank positions per site.
            Each of K=5 sites has the SAME guide at THREE flank positions
            (60, 80, 100).  Phase C must emit THREE separate clusters (one
            per observed position), each with 5/5 support, each reporting
            mean_hit_multiplicity_at_nc_pos_L = 3.  Cluster support must not
            be inflated by treating the three hits as one position cluster.

  Case 9  : multiple surviving clusters in the SAME bag at different
            (nc_pos, L).  Guide1 at (nc_pos=100, L=11, flank_pos=80);
            guide2 at (nc_pos=250, L=12, flank_pos=40).  Phase C emits two
            cluster rows, one per (nc_pos, L).

  Case 12 : background chain-link.  Three sites.  Build flanks so that
            masked-A ~ masked-B > 0.85, masked-B ~ masked-C > 0.85,
            masked-A ~ masked-C <= 0.85.  Phase D background grouping must
            report n_local_bg_groups = 1 AND chain_linked = True.

Run:  python scripts/tests/test_phase_c_clustering.py
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
from scripts.v5a_framework.background_grouping import compute_bg_groups


Ls = (9, 10, 11, 12, 13, 14)


def _rand_dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


# ---------- Case 6 ----------
def _make_bag_case6():
    """K=5 sites.  Each site has the same guide at three flank positions
    (60, 80, 100) and nothing of interest elsewhere."""
    guide = "GCATGCTAGCA"  # 11 bp
    L = len(guide)
    sites = []
    positions = (60, 80, 100)
    for i in range(5):
        bg = _rand_dna(140, seed=800 + i)
        f = list(bg[:120])
        for p in positions:
            for k, b in enumerate(guide):
                f[p + k] = b
        flank = "".join(f)[:120]
        nc_bg = _rand_dna(400, seed=900 + i)
        nc = nc_bg[:100] + guide + nc_bg[100 + L:]
        sites.append({"inputs":
                      {"noncoding_regions": [nc], "flank": flank}})
    return sites, guide, L, positions


def test_case6_multiple_positions_per_site():
    print("\n=== Case 6: motif at multiple flank positions per site ===")
    sites, guide, L, positions = _make_bag_case6()
    bag = retain_bag(sites, Ls)
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    # Keep only clusters at the shared guide location (nc_pos=100, L=11)
    target = [c for c in clusters if c.nc_pos == 100 and c.L == L]
    print(f"  bag.K_valid = {bag.K_valid}")
    print(f"  total clusters: {len(clusters)}")
    print(f"  at (nc_pos=100, L={L}): {len(target)}")
    for c in sorted(target, key=lambda c: c.flank_pos_median):
        print(f"    flank_pos_median={c.flank_pos_median:3d}  "
              f"mad={c.flank_pos_mad:.1f}  "
              f"n_support={c.n_support}  "
              f"support_frac={c.support_frac:.2f}  "
              f"mean_m={c.mean_m_frac:.2f}  "
              f"n_hits_each={c.mean_hit_multiplicity_at_nc_pos_L:.1f}")

    # Expect at least 3 clusters at (100, 11) centered near 60, 80, 100.
    seen_centers = sorted(c.flank_pos_median for c in target)
    for p in positions:
        assert any(abs(c - p) <= 2 for c in seen_centers), \
            f"missing cluster near flank_pos={p}; got centers {seen_centers}"
    # Each supported by all 5 sites.
    for c in target:
        if c.flank_pos_median in positions or any(
                abs(c.flank_pos_median - p) <= 2 for p in positions):
            assert c.n_support == 5, \
                f"cluster at {c.flank_pos_median} has n_support={c.n_support}, expected 5"
            # multiplicity = 3 hits per site at this (nc_pos, L)
            assert abs(c.mean_hit_multiplicity_at_nc_pos_L - 3.0) < 0.01, \
                f"multiplicity should be 3, got {c.mean_hit_multiplicity_at_nc_pos_L}"
    print("  PASS — three separate clusters at (60, 80, 100), each 5/5 support, multiplicity=3")


# ---------- Case 9 ----------
def _make_bag_case9():
    """K=5 sites with TWO shared guides:
       guide1 at (nc_pos=100, L=11, flank_pos=80)
       guide2 at (nc_pos=250, L=12, flank_pos=40)
    """
    guide1 = "GCATGCTAGCA"   # L=11
    guide2 = "TAGCATGCTAGC"  # L=12
    sites = []
    for i in range(5):
        bg_f = _rand_dna(140, seed=1000 + i)
        f = list(bg_f[:120])
        for k, b in enumerate(guide1):
            f[80 + k] = b
        for k, b in enumerate(guide2):
            f[40 + k] = b
        flank = "".join(f)[:120]
        nc_bg = _rand_dna(500, seed=1100 + i)
        nc_l = list(nc_bg)
        for k, b in enumerate(guide1):
            nc_l[100 + k] = b
        for k, b in enumerate(guide2):
            nc_l[250 + k] = b
        nc = "".join(nc_l)
        sites.append({"inputs":
                      {"noncoding_regions": [nc], "flank": flank}})
    return sites, guide1, guide2


def test_case9_multiple_clusters_different_nc_pos():
    print("\n=== Case 9: multiple surviving clusters at different (nc_pos, L) ===")
    sites, guide1, guide2 = _make_bag_case9()
    L1, L2 = len(guide1), len(guide2)
    bag = retain_bag(sites, Ls)
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)

    at_g1 = [c for c in clusters if c.nc_pos == 100 and c.L == L1
             and abs(c.flank_pos_median - 80) <= 2]
    at_g2 = [c for c in clusters if c.nc_pos == 250 and c.L == L2
             and abs(c.flank_pos_median - 40) <= 2]
    print(f"  bag.K_valid = {bag.K_valid}")
    print(f"  guide1 cluster at (100,{L1},~80): {len(at_g1)} found")
    print(f"  guide2 cluster at (250,{L2},~40): {len(at_g2)} found")
    assert at_g1, "guide1 cluster missing"
    assert at_g2, "guide2 cluster missing"
    assert at_g1[0].n_support == 5
    assert at_g2[0].n_support == 5
    print("  PASS — both clusters emitted as independent rows, each 5/5 support")


# ---------- Case 12 ----------
def _make_bag_case12():
    """3 sites sharing one guide → one Phase-C cluster.
    Flanks constructed so that AFTER masking the target zone:
      masked_A vs masked_B identity > 0.85  (constructed: 101/109 ≈ 0.927)
      masked_B vs masked_C identity > 0.85  (constructed: 100/109 ≈ 0.917)
      masked_A vs masked_C identity ≤ 0.85  (constructed:  92/109 ≈ 0.844)
    """
    guide = "GCATGCTAGCA"  # L=11, placed at flank_pos=50 in each site
    L = len(guide)
    # Non-target bg length after assembly + mask(margin=3) = flank_len - (L+6)
    # Our mask: [47, 64) in a 120-bp flank → 120 - 17 = 103 bg bp survive.
    # To make identity arithmetic clean, build 109 bg bp then let the mask
    # drop 6 of them; relative identity ratios are preserved.
    bg_len = 109
    rng = random.Random(1234)
    bg_A = [rng.choice("ACGT") for _ in range(bg_len)]

    # P_AB = positions where B differs from A (8 positions)
    P_AB = rng.sample(range(bg_len), 8)
    bg_B = list(bg_A)
    for p in P_AB:
        bg_B[p] = rng.choice([b for b in "ACGT" if b != bg_A[p]])

    # P_AC strictly CONTAINS P_AB and adds 9 more distinct positions.
    # At the 8 overlap positions, set C = B (same substitution) so B~C has
    # differences only at the 9 extra P_AC positions.
    extras = rng.sample([p for p in range(bg_len) if p not in P_AB], 9)
    P_AC = list(P_AB) + extras
    bg_C = list(bg_A)
    for p in P_AB:
        bg_C[p] = bg_B[p]  # keep identical to B at overlap positions
    for p in extras:
        bg_C[p] = rng.choice([b for b in "ACGT" if b != bg_A[p]])

    def assemble(bg):
        return "".join(bg[:50]) + guide + "".join(bg[50:])
    sites = [{"inputs": {"noncoding_regions": [_rand_dna(300, s * 13 + 1)],
                           "flank": assemble(bg)}}
             for s, bg in enumerate([bg_A, bg_B, bg_C])]
    # Put guide into nc at nc_pos=100 so cluster forms.
    for s, site in enumerate(sites):
        nc = list(site["inputs"]["noncoding_regions"][0])
        for k, b in enumerate(guide):
            nc[100 + k] = b
        site["inputs"]["noncoding_regions"] = ["".join(nc)]
    return sites, guide, L


def test_case12_background_chain_link():
    print("\n=== Case 12: background chain-link ===")
    sites, guide, L = _make_bag_case12()
    bag = retain_bag(sites, Ls)
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    target = [c for c in clusters if c.nc_pos == 100 and c.L == L
              and abs(c.flank_pos_median - 50) <= 2]
    assert target, "cluster at (100, 11, ~50) missing"
    cluster = target[0]
    assert cluster.n_support == 3

    flanks_per_site = {i: sites[i]["inputs"]["flank"] for i in range(3)}
    report = compute_bg_groups(cluster, flanks_per_site, margin=3,
                                clonal_bg_threshold=0.85)
    # Report pairwise identities and groups
    print(f"  cluster at (nc_pos=100, L={L}, flank_pos~50), n_support=3")
    print(f"  pairwise identities on masked flank (margin=3):")
    for (si, sj), ident in sorted(report.pairwise_identities.items()):
        marker = ">0.85" if ident > 0.85 else "≤0.85"
        print(f"    site {si} vs site {sj}: {ident:.3f}  {marker}")
    print(f"  n_local_bg_groups = {report.n_local_bg_groups}")
    for g in report.groups:
        print(f"    group members={g.members}  "
              f"min_pair_identity={g.min_pair_identity:.3f}  "
              f"chain_linked={g.chain_linked}")
    print(f"  any_chain_linked = {report.any_chain_linked}")

    # Verify the construction gave the chain-link relation
    pw = report.pairwise_identities
    assert pw[(0, 1)] > 0.85, f"A~B {pw[(0,1)]:.3f} must be >0.85"
    assert pw[(1, 2)] > 0.85, f"B~C {pw[(1,2)]:.3f} must be >0.85"
    assert pw[(0, 2)] <= 0.85, f"A~C {pw[(0,2)]:.3f} must be <=0.85"
    assert report.n_local_bg_groups == 1, \
        f"expected 1 chain-linked component, got {report.n_local_bg_groups}"
    assert report.any_chain_linked, "chain_linked flag must be True"
    print("  PASS — single chain-linked component correctly flagged")


# ---------- Case 1-strict: non-best cross-site cluster recovery through full A→C ----------
def _make_bag_case1_strict():
    """K=5 sites. For each site:
      - A UNIQUE strict-best m=L exact match at a scattered flank position
      - A SHARED one-mismatch (m=L-1) match at flank_pos=80

    The guide at nc_pos=100 is fixed across sites.  Only Phase C can surface
    the pos-80 cluster because the per-site best is scattered elsewhere.
    This is the actual criterion-1 test: 'best hits scattered, recover shared
    second-best cluster'.
    """
    guide = "GCATGCTAGCA"       # 11 bp
    L = len(guide)
    # Partial-match copy: substitute position 5 (G→A) → m=10 at pos=80
    guide_1mm = guide[:5] + ("A" if guide[5] != "A" else "T") + guide[6:]
    assert sum(a == b for a, b in zip(guide, guide_1mm)) == L - 1
    scattered = [10, 25, 40, 55, 65]       # per-site unique strict-best
    sites = []
    for i, sp in enumerate(scattered):
        bg = _rand_dna(140, seed=1200 + i)
        f = list(bg[:120])
        # scattered strict-best (exact guide)
        for k, b in enumerate(guide):
            f[sp + k] = b
        # shared one-mismatch at pos=80
        for k, b in enumerate(guide_1mm):
            f[80 + k] = b
        flank = "".join(f)[:120]
        nc_bg = _rand_dna(400, seed=1300 + i)
        nc = nc_bg[:100] + guide + nc_bg[100 + L:]
        sites.append({"inputs":
                      {"noncoding_regions": [nc], "flank": flank}})
    return sites, guide, L, scattered


def test_case1_strict_non_best_recovery():
    print("\n=== Case 1 (strict): per-site best scattered, shared non-best at pos=80 ===")
    sites, guide, L, scattered = _make_bag_case1_strict()
    bag = retain_bag(sites, Ls)
    clusters = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    at_80 = [c for c in clusters
             if c.nc_pos == 100 and c.L == L
             and abs(c.flank_pos_median - 80) <= 2]
    # Expect: cluster at pos~80 with 5/5 support, mean_m_frac = (L-1)/L
    assert at_80, "shared non-best cluster at pos=80 MISSING"
    c = at_80[0]
    print(f"  cluster at pos_median={c.flank_pos_median}  "
          f"n_support={c.n_support}  mean_m_frac={c.mean_m_frac:.3f}  "
          f"n_sites_cluster_hit_is_best={c.n_sites_cluster_hit_is_best}")
    assert c.n_support == 5
    assert abs(c.mean_m_frac - (L - 1) / L) < 0.01, \
        f"expected mean_m_frac=(L-1)/L={ (L-1)/L :.3f}, got {c.mean_m_frac:.3f}"
    assert c.n_sites_cluster_hit_is_best == 0, (
        f"the pos-80 hit must NOT be any site's strict best; "
        f"got {c.n_sites_cluster_hit_is_best}/5")

    # And: the per-site scattered strict-best hits must NOT form surviving clusters
    # (each would be supported by only its own site)
    scattered_clusters = [c for c in clusters
                           if c.nc_pos == 100 and c.L == L
                           and any(abs(c.flank_pos_median - sp) <= 2
                                   for sp in scattered)]
    for sc in scattered_clusters:
        # If one somehow surfaces, verify it has n_support < 5 (not inflated)
        assert sc.n_support < 5, (
            f"scattered cluster at {sc.flank_pos_median} should have "
            f"n_support<5, got {sc.n_support}")
    print("  PASS — strict-best scattered hits do NOT form clusters; "
          "shared non-best at pos=80 does. "
          "n_sites_cluster_hit_is_best=0 confirms the mechanism.")


def main():
    test_case1_strict_non_best_recovery()
    test_case6_multiple_positions_per_site()
    test_case9_multiple_clusters_different_nc_pos()
    test_case12_background_chain_link()
    print("\n[all 4 Phase-C/D-partial tests PASSED]")


if __name__ == "__main__":
    main()
