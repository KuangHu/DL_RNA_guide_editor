"""Phase-B validation tests (model-free scorer v3).

Covers:
  Case 2 : poly-A guide                → REJECTED by homopolymer gate
  Case 3 : ATATATATATA guide           → REJECTED by period-2 gate
  Case 4 : ACGTACGTACGT guide          → REJECTED by period-4 gate (the one
                                          my earlier distinct-dinuc/trinuc
                                          gate missed; H_base at 2.0 bits
                                          and AT_frac at 0.5 both pass)
  Case 5 : AATTAACGTTAA guide          → REJECTED by at_frac gate ONLY
                                          (H_base >=1.3, no period≥0.80,
                                          no homopolymer ≥ L/2)

Plus a per-primitive sanity check on the gate functions themselves and an
end-to-end pipeline check (Phase A → Phase B → Phase C): case 4 bag must
produce ZERO clusters after Phase B strips all guide/target hits.
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v5a_framework.multi_hit_retention import retain_bag
from scripts.v5a_framework.cluster_enumeration import enumerate_clusters
from scripts.v5a_framework.motif_gates import (
    GateConfig, evaluate_motif, apply_phase_b,
    longest_run, periodicity_fraction, at_fraction, shannon_base,
)


Ls = (9, 10, 11, 12, 13, 14)


def test_primitives():
    print("\n=== primitives sanity ===")
    # longest_run
    assert longest_run("AAAAAAA") == 7
    assert longest_run("ATATAT") == 1
    assert longest_run("ACGTACGT") == 1
    assert longest_run("AATTAACG") == 2
    # periodicity
    assert periodicity_fraction("ATATATATATA", 2) == 1.0   # 9/9 all match
    assert abs(periodicity_fraction("ACGTACGTACGT", 4) - 1.0) < 1e-6  # 8/8
    assert periodicity_fraction("ACGTACGTACGT", 1) < 0.1
    assert periodicity_fraction("AAAAAAAAAAA", 1) == 1.0
    # at_fraction
    assert at_fraction("AAAATTTT") == 1.0
    assert at_fraction("ACGTACGT") == 0.5
    assert at_fraction("AATTAACGTTAA") == 10 / 12
    # shannon_base
    assert shannon_base("AAAAAA") == 0.0
    assert abs(shannon_base("ACGT") - 2.0) < 1e-6
    assert abs(shannon_base("AT") - 1.0) < 1e-6
    print("  PASS")


def _check_rejection(seq, cfg, expected_reason_prefix, forbidden_reason_prefix=()):
    """Assert seq is rejected and the FIRST reason matching expected_reason_prefix
    is present; optionally assert that no reason in forbidden_reason_prefix is present."""
    r = evaluate_motif(seq, cfg)
    assert not r.passed, f"expected REJECT for {seq!r}, got PASS with reasons {r.reasons}"
    matches = [reason for reason in r.reasons
               if any(reason.startswith(e) for e in [expected_reason_prefix])]
    assert matches, (f"expected reason starting with {expected_reason_prefix!r} "
                     f"in {r.reasons}")
    for forbidden in forbidden_reason_prefix:
        bad = [reason for reason in r.reasons if reason.startswith(forbidden)]
        assert not bad, (f"forbidden reason {forbidden!r} found in rejection "
                         f"reasons for {seq!r}: {r.reasons}")
    return r


def _check_pass(seq, cfg):
    r = evaluate_motif(seq, cfg)
    assert r.passed, f"expected PASS for {seq!r}, got REJECT with reasons {r.reasons}"
    return r


def test_case2_polyA():
    print("\n=== Case 2: poly-A guide ===")
    cfg = GateConfig()
    r = _check_rejection("AAAAAAAAAAA", cfg, "homopolymer_run")
    print(f"  AAAAAAAAAAA → REJECT: reasons={r.reasons}")
    print("  PASS")


def test_case3_period2():
    print("\n=== Case 3: ATATATATATA guide ===")
    cfg = GateConfig()
    r = _check_rejection("ATATATATATA", cfg, "period2")
    print(f"  ATATATATATA → REJECT: reasons={r.reasons}")
    print("  PASS")


def test_case4_period4_acgt():
    print("\n=== Case 4: ACGTACGTACGT guide (critical — period gate only) ===")
    cfg = GateConfig()
    # must fail period4, but H_base, homopolymer, at_frac must all PASS
    r = _check_rejection("ACGTACGTACGT", cfg, "period4",
                           forbidden_reason_prefix=["h_base", "homopolymer_run", "at_frac"])
    print(f"  ACGTACGTACGT → REJECT: reasons={r.reasons}")
    print(f"    period_fracs: {r.period_fracs}  h_base={r.h_base:.2f}  "
          f"at_frac={r.at_frac:.2f}  longest_run={r.longest_run}")
    print("  PASS — only period-4 gate triggered")


def test_case5_at_rich_non_repetitive():
    print("\n=== Case 5: AT-rich non-repetitive guide (at_frac gate only) ===")
    cfg = GateConfig()
    # AATTAACGTTAA: AT 10/12 = 0.833 > 0.75; H_base ~ 1.62; no period >= 0.80;
    # longest_run = 2 (< 6).  Only at_frac should trigger.
    r = _check_rejection("AATTAACGTTAA", cfg, "at_frac",
                           forbidden_reason_prefix=["h_base", "homopolymer_run",
                                                      "period1", "period2",
                                                      "period3", "period4"])
    print(f"  AATTAACGTTAA → REJECT: reasons={r.reasons}")
    print(f"    at_frac={r.at_frac:.3f}  h_base={r.h_base:.3f}  "
          f"period_fracs={r.period_fracs}  longest_run={r.longest_run}")
    print("  PASS — only at_frac gate triggered")


def test_case4_e2e_pipeline():
    """End-to-end: a bag whose ONLY cross-site signal is ACGT-period-4 must
    produce zero Phase-C clusters after Phase B."""
    print("\n=== Case 4 end-to-end: Phase A → Phase B → Phase C ===")
    guide = "ACGTACGTACGT"   # L=12
    L = len(guide)
    rng = random.Random(2026)
    sites = []
    for i in range(5):
        bg = [rng.choice("ACGT") for _ in range(120)]
        # place guide at flank_pos=60
        for k, b in enumerate(guide):
            bg[60 + k] = b
        flank = "".join(bg)
        nc_bg = [rng.choice("ACGT") for _ in range(400)]
        for k, b in enumerate(guide):
            nc_bg[100 + k] = b
        nc = "".join(nc_bg)
        sites.append({"inputs": {"noncoding_regions": [nc], "flank": flank}})

    bag = retain_bag(sites, Ls)
    clusters_before = enumerate_clusters(bag, max_offset=5, min_frac=0.5)
    cluster_at_guide_before = [c for c in clusters_before
                                 if c.nc_pos == 100 and c.L == L
                                 and abs(c.flank_pos_median - 60) <= 2]
    print(f"  before Phase B: {len(clusters_before)} total clusters, "
          f"{len(cluster_at_guide_before)} at the guide location")
    assert cluster_at_guide_before, \
        "setup check: ACGT guide must produce a Phase-A/C cluster before gating"

    bag_filtered, log = apply_phase_b(bag, sites, GateConfig())
    clusters_after = enumerate_clusters(bag_filtered, max_offset=5, min_frac=0.5)
    cluster_at_guide_after = [c for c in clusters_after
                                if c.nc_pos == 100 and c.L == L
                                and abs(c.flank_pos_median - 60) <= 2]
    print(f"  rejection log:")
    for line in log.summary():
        print(f"    {line}")
    print(f"  after Phase B: {len(clusters_after)} total clusters, "
          f"{len(cluster_at_guide_after)} at the guide location")
    assert not cluster_at_guide_after, (
        f"ACGT guide should NOT survive Phase B; got clusters at guide: "
        f"{[(c.flank_pos_median, c.n_support) for c in cluster_at_guide_after]}")
    print("  PASS — ACGT-period-4 motif removed before clustering")


def main():
    test_primitives()
    test_case2_polyA()
    test_case3_period2()
    test_case4_period4_acgt()
    test_case5_at_rich_non_repetitive()
    test_case4_e2e_pipeline()
    print("\n[all Phase-B tests PASSED]")


if __name__ == "__main__":
    main()
