"""Phase-A validation tests (model-free scorer v3).

Covers the three unit cases from the v3 plan:

  Case 1  : non-best cross-site cluster recovery
            Each site's single-site best hit is at a DIFFERENT flank_pos
            (scattered across 10..100).  All sites also carry an m=11 hit at
            the shared flank_pos=80.  Phase A must retain BOTH hits per site —
            if argmax reduction had survived, the pos=80 hit would be lost
            on every site.

  Case 10 : RC frame round-trip.
            Build a positive bag with FWD hits at (nc_pos, flank_pos).
            Reverse-complement every site's nc AND flank.
            Phase A on the RC bag must find hits at the coordinate-transformed
            positions with the same m value.  The hit's `orient` is allowed
            to differ (symmetry of dot_plot) but the flank_pos must be in
            FWD-flank coord of the (already revcomped) sequence.

  Case 11 : support denominator frozen.
            8 sites: 4 have usable nc+flank, 4 are too short / malformed.
            compute_K_valid returns 4.  retain_bag reports K_valid=4 even
            when only 2 of the 4 valid sites actually produce any hit.

Run:  python scripts/tests/test_phase_a_multi_hit.py
"""
from __future__ import annotations
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import revcomp
from scripts.v5a_framework.multi_hit_retention import (
    retain_hits, retain_bag, compute_K_valid, Hit,
)


Ls = (9, 10, 11, 12, 13, 14)


def _rand_dna(n: int, seed: int) -> str:
    rng = random.Random(seed)
    return "".join(rng.choice("ACGT") for _ in range(n))


def _mutate(seq: str, n_mut: int, seed: int) -> str:
    """Return a copy of seq with exactly n_mut positions substituted."""
    rng = random.Random(seed)
    s = list(seq)
    positions = rng.sample(range(len(s)), n_mut)
    for p in positions:
        new = rng.choice([b for b in "ACGT" if b != s[p]])
        s[p] = new
    return "".join(s)


def _make_bag_case1():
    """K=5 sites.  Each site has:
       - guide 'GCATGCTAGCA' (11 bp) at nc_pos=100
       - a BETTER per-site hit (variant of guide with mutation 0 .. so m=11)
         at a DIFFERENT flank_pos per site
       - an m=11 shared hit at flank_pos=80

    Wait — if the per-site 'best' is also m=11 (same as the shared pos-80 hit),
    there is no 'better' — they tie.  To make it a strict 'non-best' case we
    give the site-specific hit a perfect m=11 at the scattered position AND a
    perfect m=11 at pos=80 (two tied hits per site at scattered + shared).
    Phase A must retain BOTH — downstream Phase C will discover that pos=80
    is the one that is consistent across sites.
    """
    guide = "GCATGCTAGCA"   # 11bp, no obvious repeat
    L = len(guide)
    # scattered positions must be far from shared_pos=80 so the two insertions
    # do not overlap in the flank (scatter + L <= 80 - 1).
    scattered = [10, 25, 40, 55, 65]
    sites = []
    for i, scatter_pos in enumerate(scattered):
        # Build a flank of length 120 that has guide at BOTH scatter_pos AND 80.
        bg = _rand_dna(120, seed=100 + i)
        flank = (bg[:scatter_pos] + guide
                 + bg[scatter_pos + L:80]
                 + guide
                 + bg[80 + L:])
        # Make sure flank length stayed at 120
        flank = flank[:120]
        # Build nc with guide at nc_pos=100
        nc_bg = _rand_dna(400, seed=200 + i)
        nc = nc_bg[:100] + guide + nc_bg[100 + L:]
        sites.append({"inputs":
                      {"noncoding_regions": [nc],
                       "flank": flank}})
    return sites, guide, scattered, 80


def test_case1_non_best_cluster_retained():
    print("\n=== Case 1: non-best cluster recovery ===")
    sites, guide, scattered, shared_pos = _make_bag_case1()
    L = len(guide)
    bag = retain_bag(sites, Ls)
    assert bag.K_valid == 5, f"K_valid should be 5, got {bag.K_valid}"

    for sr, scatter_pos in zip(bag.retained, scattered):
        key = (100, L)
        hits = sr.hits_by_nc_pos_L.get(key, [])
        # m-at-top for each site: must have >=2 hits with m>=0.6*L; both shared and scattered
        m_at_L_full = [h for h in hits if h.m == L]
        flank_positions_full = sorted({h.flank_pos for h in m_at_L_full})
        print(f"  site {sr.site_idx}: scattered_pos={scatter_pos}  m={L} hits at flank_pos={flank_positions_full}")
        assert shared_pos in flank_positions_full, \
            f"site {sr.site_idx}: shared pos {shared_pos} MISSING from retained m={L} hits {flank_positions_full}"
        assert scatter_pos in flank_positions_full, \
            f"site {sr.site_idx}: scattered pos {scatter_pos} MISSING from retained m={L} hits {flank_positions_full}"
    print("  PASS — every site retains BOTH the scattered best and the shared pos=80 hit")


def _make_bag_case10():
    """3 sites.  Each has a FWD-matching guide at known coord."""
    guide = "GCATGCTAGCA"
    L = len(guide)
    sites = []
    nc_pos = 150
    flank_pos = 70
    for i in range(3):
        bg_fl = _rand_dna(120, seed=300 + i)
        flank = bg_fl[:flank_pos] + guide + bg_fl[flank_pos + L:]
        flank = flank[:120]
        bg_nc = _rand_dna(400, seed=400 + i)
        nc = bg_nc[:nc_pos] + guide + bg_nc[nc_pos + L:]
        sites.append({"inputs":
                      {"noncoding_regions": [nc],
                       "flank": flank}})
    return sites, guide, nc_pos, flank_pos


def test_case10_rc_round_trip():
    print("\n=== Case 10: RC round-trip ===")
    sites, guide, nc_pos, flank_pos = _make_bag_case10()
    L = len(guide)
    bag_orig = retain_bag(sites, Ls)

    # Collect per-site: for each site, find any m==L hit at (nc_pos=nc_pos, L=L, flank_pos=flank_pos)
    for sr in bag_orig.retained:
        hits = sr.hits_by_nc_pos_L.get((nc_pos, L), [])
        m_full = [h for h in hits if h.m == L and h.flank_pos == flank_pos]
        assert m_full, f"orig site {sr.site_idx}: missing m=L hit at ({nc_pos},{flank_pos})"
    print(f"  orig bag: all sites have m={L} FWD hit at (nc_pos={nc_pos}, flank_pos={flank_pos})")

    # Build RC'd bag
    rc_sites = []
    nc_len = len(sites[0]["inputs"]["noncoding_regions"][0])
    flank_len = len(sites[0]["inputs"]["flank"])
    for s in sites:
        nc_r = revcomp(s["inputs"]["noncoding_regions"][0])
        fl_r = revcomp(s["inputs"]["flank"])
        rc_sites.append({"inputs":
                          {"noncoding_regions": [nc_r],
                           "flank": fl_r}})

    # Expected transformed coord: in the revcomped nc, the guide revcomp lives
    # at nc_pos' = nc_len - nc_pos - L.  The guide-revcomp matches flank-revcomp
    # at flank_pos' = flank_len - flank_pos - L  (FWD-frame on the new flank).
    # Alternatively, in the revcomped nc, the ORIGINAL guide sequence still
    # appears somewhere — actually revcomp(nc) does NOT contain the original
    # guide, only its reverse complement.  So in the RC'd bag the retention
    # should surface:
    #   nc_pos'    = nc_len - nc_pos - L
    #   flank_pos' = flank_len - flank_pos - L
    #   orient     = 'fwd' (guide-revcomp matches flank-revcomp in fwd dot plot)
    exp_nc_pos    = nc_len   - nc_pos    - L
    exp_flank_pos = flank_len - flank_pos - L

    bag_rc = retain_bag(rc_sites, Ls)
    for sr in bag_rc.retained:
        hits = sr.hits_by_nc_pos_L.get((exp_nc_pos, L), [])
        m_full = [h for h in hits if h.m == L and h.flank_pos == exp_flank_pos]
        print(f"  rc site {sr.site_idx}: expect ({exp_nc_pos},{exp_flank_pos})  "
              f"got {[(h.flank_pos, h.orient) for h in m_full]}")
        assert m_full, \
            f"rc site {sr.site_idx}: NO m=L hit at coord-transformed ({exp_nc_pos},{exp_flank_pos})"
    print(f"  PASS — RC round-trip preserves hits at coord-transformed (nc_pos'={exp_nc_pos}, flank_pos'={exp_flank_pos})")


def _make_bag_case11():
    """8 sites: 4 have usable nc+flank, 4 are too short or malformed."""
    sites = []
    # 4 valid sites
    for i in range(4):
        bg_nc = _rand_dna(300, seed=500 + i)
        bg_fl = _rand_dna(120, seed=600 + i)
        sites.append({"inputs":
                      {"noncoding_regions": [bg_nc], "flank": bg_fl}})
    # 4 invalid sites
    sites.append({"inputs":
                  {"noncoding_regions": [], "flank": _rand_dna(120, 700)}})  # empty nc list
    sites.append({"inputs":
                  {"noncoding_regions": ["ACG"], "flank": _rand_dna(120, 701)}})  # nc too short
    sites.append({"inputs":
                  {"noncoding_regions": [_rand_dna(300, 702)], "flank": "AC"}})  # flank too short
    sites.append({"inputs":
                  {"noncoding_regions": [_rand_dna(300, 703)]}})  # missing flank
    return sites


def test_case11_support_denominator_frozen():
    print("\n=== Case 11: K_valid frozen before filtering ===")
    sites = _make_bag_case11()
    K, valid, dropped = compute_K_valid(sites, min_nc_len=max(Ls),
                                              min_flank_len=max(Ls))
    print(f"  K_valid = {K}  valid_idx = {valid}")
    for i, r in sorted(dropped.items()):
        print(f"  dropped site {i}: {r}")
    assert K == 4, f"expected K_valid=4, got {K}"
    assert set(valid) == {0, 1, 2, 3}, f"unexpected valid_idx={valid}"
    assert len(dropped) == 4, f"expected 4 dropped, got {len(dropped)}"

    # Now run full retain_bag — K_valid must stay 4 even if some valid sites
    # produce zero hits (random flank + random nc → high chance 0 hits at m>=ceil(0.6*L))
    bag = retain_bag(sites, Ls)
    assert bag.K_valid == 4, f"retain_bag K_valid should be 4, got {bag.K_valid}"
    assert len(bag.retained) == 4
    # Report how many of the 4 produced any hit at all
    n_with_hits = sum(1 for sr in bag.retained if sr.hits_by_nc_pos_L)
    print(f"  of 4 valid sites, {n_with_hits} produced ≥1 hit at retention floor")
    print(f"  PASS — K_valid frozen at 4 regardless of per-site hit outcome")


def main():
    test_case1_non_best_cluster_retained()
    test_case10_rc_round_trip()
    test_case11_support_denominator_frozen()
    print("\n[all 3 Phase-A tests PASSED]")


if __name__ == "__main__":
    main()
