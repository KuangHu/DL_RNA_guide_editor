"""Phase-D validation tests (model-free scorer v3).

Covers:
  Case 7 : clonal flanks  — K=5 sites with highly similar flanks (>0.95
                            identity everywhere, mask or not).  A shared
                            motif gives one Phase-C cluster.  Phase D must
                            reject with n_local_bg_groups < 3.

  Case 8 : target near flank edge — K=5 sites with the guide placed at
                                     flank_pos=109 (L=11, flank_len=120)
                                     so right-side bg = 0 bp.
    Default (permissive) config  → cluster PASSES, with
                                    edge_insufficient_right = {0..4}.
    Strict config (require_both_sides=True) → cluster REJECTED by
                                    edge_insufficient_sites reason.

Also exercises apply_phase_d_gate end-to-end on a 3-cluster bag:
  cluster A (clean indep flanks)        → PASS
  cluster B (clonal flanks)             → FAIL
  cluster C (edge-adjacent, permissive) → PASS with edge flags
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v5a_framework.multi_hit_retention import retain_bag
from scripts.v5a_framework.cluster_enumeration import enumerate_clusters
from scripts.v5a_framework.motif_gates import apply_phase_b, GateConfig
from scripts.v5a_framework.background_grouping import (
    compute_bg_groups, apply_phase_d_gate, BgConfig,
)


Ls = (9, 10, 11, 12, 13, 14)
GUIDE = "GCATGCTAGCA"           # 11 bp, non-repetitive, complexity passes
L = len(GUIDE)


def _rand_dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _place_guide(flank_bg: list[str], flank_pos: int, guide: str = GUIDE) -> str:
    out = list(flank_bg)
    for k, b in enumerate(guide):
        out[flank_pos + k] = b
    return "".join(out)


# ----------------- CASE 7 -----------------
def _make_bag_case7(flank_pos: int = 50, K: int = 5):
    """K sites with highly similar flanks.  Start from one random flank, derive
    the other K-1 by mutating ~2% of positions each; all should land in one
    chain-connected background component after masking."""
    rng = random.Random(707)
    base = [rng.choice("ACGT") for _ in range(120)]
    sites = []
    nc_bg_base = _rand_dna(400, seed=701)
    for i in range(K):
        bg = list(base)
        # ~2% mutations
        for p in rng.sample(range(120), 3):
            bg[p] = rng.choice([b for b in "ACGT" if b != base[p]])
        flank = _place_guide(bg, flank_pos)
        nc_bg = list(nc_bg_base)
        for k, b in enumerate(GUIDE):
            nc_bg[100 + k] = b
        nc = "".join(nc_bg)
        sites.append({"inputs": {"noncoding_regions": [nc], "flank": flank}})
    return sites


def test_case7_clonal_flanks():
    print("\n=== Case 7: clonal flanks rejected by bg-groups gate ===")
    sites = _make_bag_case7()
    bag = retain_bag(sites, Ls)
    bag_f, _ = apply_phase_b(bag, sites, GateConfig())
    clusters = enumerate_clusters(bag_f, max_offset=5, min_frac=0.5)
    cl = [c for c in clusters
          if c.nc_pos == 100 and c.L == L and abs(c.flank_pos_median - 50) <= 2]
    assert cl, "setup check: shared-guide cluster expected"
    cluster = cl[0]
    flanks_per_site = {i: sites[i]["inputs"]["flank"] for i in range(len(sites))}
    report = compute_bg_groups(cluster, flanks_per_site,
                                margin=3, clonal_bg_threshold=0.85,
                                min_bg_bp=20, min_bg_groups=3)
    print(f"  cluster at (nc_pos=100, L={L}, flank_pos_median={cluster.flank_pos_median})")
    print(f"  pairwise concat-bg identities (first few):")
    for (si, sj), ident in list(sorted(report.pairwise_identities.items()))[:4]:
        print(f"    site {si} vs {sj}: {ident:.3f}")
    print(f"  median_left={report.median_left_identity:.3f}  "
          f"median_right={report.median_right_identity:.3f}")
    print(f"  n_local_bg_groups = {report.n_local_bg_groups}  "
          f"passes_bg_gate = {report.passes_bg_gate}  "
          f"rejection_reason = {report.rejection_reason!r}")
    assert report.n_local_bg_groups == 1, \
        f"expected 1 clonal component, got {report.n_local_bg_groups}"
    assert not report.passes_bg_gate
    assert report.rejection_reason and "insufficient_bg_groups" in report.rejection_reason
    print("  PASS — clonal cluster rejected by n_local_bg_groups<3")


# ----------------- CASE 8 -----------------
def _make_bag_case8(flank_pos: int = 109, K: int = 5):
    """Each site has an independent random flank + guide at flank_pos=109
    (window=[109, 120)).  Right-side bg = 0 bp."""
    rng = random.Random(808)
    sites = []
    for i in range(K):
        bg = [rng.choice("ACGT") for _ in range(120)]
        flank = _place_guide(bg, flank_pos)
        nc_bg = [rng.choice("ACGT") for _ in range(400)]
        for k, b in enumerate(GUIDE):
            nc_bg[100 + k] = b
        nc = "".join(nc_bg)
        sites.append({"inputs": {"noncoding_regions": [nc], "flank": flank}})
    return sites


def test_case8_edge_adjacent():
    print("\n=== Case 8: target near edge — permissive vs strict ===")
    sites = _make_bag_case8()
    bag = retain_bag(sites, Ls)
    bag_f, _ = apply_phase_b(bag, sites, GateConfig())
    clusters = enumerate_clusters(bag_f, max_offset=5, min_frac=0.5)
    cl = [c for c in clusters
          if c.nc_pos == 100 and c.L == L and abs(c.flank_pos_median - 109) <= 2]
    assert cl, "setup check: edge-adjacent cluster expected"
    cluster = cl[0]
    flanks_per_site = {i: sites[i]["inputs"]["flank"] for i in range(len(sites))}

    # --- permissive (default) ---
    permissive = compute_bg_groups(cluster, flanks_per_site,
                                     margin=3, clonal_bg_threshold=0.85,
                                     min_bg_bp=20, min_bg_groups=3,
                                     require_both_sides=False)
    print(f"  permissive:")
    print(f"    n_bp_left_available  = {dict(sorted(permissive.n_bp_left_available.items()))}")
    print(f"    n_bp_right_available = {dict(sorted(permissive.n_bp_right_available.items()))}")
    print(f"    edge_insufficient_right = {sorted(permissive.edge_insufficient_right)}")
    print(f"    edge_insufficient_left  = {sorted(permissive.edge_insufficient_left)}")
    print(f"    edge_insufficient_both  = {sorted(permissive.edge_insufficient_both)}")
    print(f"    median_left={permissive.median_left_identity:.3f}  "
          f"median_right={permissive.median_right_identity}")
    print(f"    n_local_bg_groups = {permissive.n_local_bg_groups}  "
          f"passes = {permissive.passes_bg_gate}")
    assert permissive.edge_insufficient_right == set(range(5)), \
        f"expected all 5 sites edge_insufficient_right, got {permissive.edge_insufficient_right}"
    assert not permissive.edge_insufficient_left, \
        f"left bg has 106 bp; should be adequate for all sites"
    assert permissive.median_right_identity is None, \
        "no right pair should be comparable at min_bg_bp=20"
    assert permissive.passes_bg_gate, \
        f"independent random left bgs should give enough groups to pass; " \
        f"got n_groups={permissive.n_local_bg_groups}"
    print("  permissive PASS — edge_insufficient_right flagged on all 5 sites, "
          "but n_local_bg_groups from left-only bg carries the gate")

    # --- strict (require both sides) ---
    strict = compute_bg_groups(cluster, flanks_per_site,
                                margin=3, clonal_bg_threshold=0.85,
                                min_bg_bp=20, min_bg_groups=3,
                                require_both_sides=True)
    print(f"  strict:")
    print(f"    passes = {strict.passes_bg_gate}  "
          f"rejection_reason = {strict.rejection_reason!r}")
    assert not strict.passes_bg_gate
    assert strict.rejection_reason and "edge_insufficient_sites" in strict.rejection_reason
    print("  strict PASS — edge-adjacent sites force a reject under require_both_sides")


# ----------------- end-to-end apply_phase_d_gate -----------------
def test_apply_phase_d_gate_mixed():
    print("\n=== apply_phase_d_gate: mixed-cluster bag ===")
    # Build three bags (A clean, B clonal, C edge) and treat the union of
    # their clusters as a mock pipeline input.
    sites_A = _make_bag_case8(flank_pos=40, K=5)   # independent flanks, central target
    sites_B = _make_bag_case7(flank_pos=40, K=5)   # clonal flanks, central target
    sites_C = _make_bag_case8(flank_pos=109, K=5)  # independent flanks, edge target

    def pipeline(sites):
        bag = retain_bag(sites, Ls)
        bag_f, _ = apply_phase_b(bag, sites, GateConfig())
        clusters = enumerate_clusters(bag_f, max_offset=5, min_frac=0.5)
        return clusters

    clusters_A = [c for c in pipeline(sites_A)
                   if c.nc_pos == 100 and c.L == L and abs(c.flank_pos_median - 40) <= 2]
    clusters_B = [c for c in pipeline(sites_B)
                   if c.nc_pos == 100 and c.L == L and abs(c.flank_pos_median - 40) <= 2]
    clusters_C = [c for c in pipeline(sites_C)
                   if c.nc_pos == 100 and c.L == L and abs(c.flank_pos_median - 109) <= 2]
    assert clusters_A and clusters_B and clusters_C

    cfg = BgConfig()  # defaults

    pA, fA = apply_phase_d_gate(clusters_A,
                                 {i: sites_A[i]["inputs"]["flank"] for i in range(5)},
                                 cfg)
    pB, fB = apply_phase_d_gate(clusters_B,
                                 {i: sites_B[i]["inputs"]["flank"] for i in range(5)},
                                 cfg)
    pC, fC = apply_phase_d_gate(clusters_C,
                                 {i: sites_C[i]["inputs"]["flank"] for i in range(5)},
                                 cfg)
    print(f"  bag A (clean):  pass={len(pA)}  fail={len(fA)}")
    print(f"  bag B (clonal): pass={len(pB)}  fail={len(fB)}")
    print(f"  bag C (edge):   pass={len(pC)}  fail={len(fC)}")
    assert pA and not fA, "clean clusters should all pass"
    assert fB and not pB, "clonal clusters should all fail"
    assert pC and not fC, "edge-adjacent passes under permissive default"
    print("  PASS — gate assigns each bag to the expected outcome")


def main():
    test_case7_clonal_flanks()
    test_case8_edge_adjacent()
    test_apply_phase_d_gate_mixed()
    print("\n[all Phase-D tests PASSED]")


if __name__ == "__main__":
    main()
