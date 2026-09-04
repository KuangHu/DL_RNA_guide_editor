"""A7 — v2 candidate layer position-array → Channel A end-to-end anchor.

Replaces the current Durrant MatchTable's per-site arrays with arrays
computed by `preprocess.candidates_v2.enumerate_position_arrays`, then
runs the SAME `run_variant` + `compute_channel_a` code path that A1
locks. Asserts the Durrant anchor holds unchanged.

Why not just compare arrays byte-for-byte? Because "compute the same
windowed_matches" is an inference about the shape of two functions, not
a measurement of what compute_channel_a does. This project has 12 prior
same-type inferences that turned out wrong at end-to-end; the most
recent was the channel_a_v5 cross-implementation anchor (A1). A7 turns
"same computation, same output" from inference into anchored regression
by literally running Channel A on v2-driven arrays.

Anchor (per test_tau0_anchor / A1):
    n_tnps=65, covered=22, coverage=0.3385, ppv_peak=0.9565,
    ppv_tnp=0.9545, exact_rate=0.3231
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.v5a_framework.match_table import (
    load as load_mt, MatchArrays, EXCL_WIDTHS, _windowed_max_by_excl,
)
from scripts.v5a_framework.variant import spec_m_threshold_L11, run_variant
from scripts.generator_v5.channel_a_v5 import compute_channel_a
from preprocess.candidates_v2 import enumerate_position_arrays
from preprocess.alignment import dot_plot, windowed_matches


DURRANT_SHARD = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"

ANCHOR = {
    "n_tnps":            65,
    "covered":           22,
    "total_peaks":       23,
    "peaks_correct":     22,
    "tnps_with_correct": 21,
    "exact":             21,
    "coverage":          22 / 65,
    "ppv_peak":          22 / 23,
    "ppv_tnp":           21 / 22,
    "exact_rate":        21 / 65,
}


def _populate_cache_from_v2(mt) -> None:
    """Overwrite mt._cache with per-(site, orient, L) MatchArrays whose
    m_max_by_excl[0] comes from `enumerate_position_arrays`, and higher
    excl widths from the standard `_windowed_max_by_excl` on the same
    `windowed_matches` output. Channel A reads only m_max_by_excl[0] for
    Mode 2 L=11 tau=0; higher-w slots are populated for correctness.
    """
    for tnp_id in mt.tnp_ids:
        tnp = mt.tnps[tnp_id]
        entry: dict = {}
        for site in tnp.sites:
            # v2 pass — position arrays at all (orient, L)
            v2_arrays = enumerate_position_arrays(
                tnp.nc, site.flank,
                L_min=min(mt.Ls), L_max=max(mt.Ls),
                orientations=mt.orients,
            )
            # For higher-w exclusion slots we need the full win output.
            # Recompute per (orient, L) — same windowed_matches call v2 uses.
            fwd_dot, rc_dot = dot_plot(tnp.nc, site.flank)
            for orient in mt.orients:
                dot = fwd_dot if orient == "fwd" else rc_dot
                for L in mt.Ls:
                    win = windowed_matches(dot, L)
                    all_w = _windowed_max_by_excl(win, EXCL_WIDTHS)
                    # Overwrite w=0 with the v2 array to force the arrays
                    # to come from the v2 path — must equal the mt-native
                    # w=0 if the parity claim holds.
                    v2_arr = v2_arrays.get((orient, L))
                    if v2_arr is not None and v2_arr.size > 0:
                        all_w[0] = v2_arr.astype(np.int8)
                    entry[(site.site_idx, orient, L)] = MatchArrays(m_max_by_excl=all_w)
        mt._cache[tnp_id] = entry


def main() -> int:
    print(f"[A7] loading Durrant MatchTable from {DURRANT_SHARD}")
    mt = load_mt(DURRANT_SHARD)
    print(f"[A7] populating _cache from candidates_v2.enumerate_position_arrays")
    _populate_cache_from_v2(mt)

    print(f"[A7] running Mode 2 spec (fixed L=11, m>=8, tau=0, S=5)")
    spec = spec_m_threshold_L11(m=8, tau=0, S=5)
    peaks = run_variant(mt, spec)

    empty_arch: dict = {t: {} for t in mt.tnp_ids}
    result = compute_channel_a(mt, peaks, empty_arch, stratify_by=None)
    got = result["all"]

    print()
    print(f"  {'field':<20s} {'anchor':>8s} {'got':>8s}  {'match':>5s}")
    all_ok = True
    for field, want in ANCHOR.items():
        v = got.get(field, float("nan"))
        if isinstance(want, float):
            ok = abs(float(v) - want) < 1e-4
            print(f"  {field:<20s} {want:>8.4f} {float(v):>8.4f}  {'✓' if ok else '✗'}")
        else:
            ok = int(v) == int(want)
            print(f"  {field:<20s} {want:>8d} {int(v):>8d}  {'✓' if ok else '✗'}")
        if not ok:
            all_ok = False

    print()
    print("[A7] " + (
        "PASS: Channel A on candidates_v2 position arrays reproduces the "
        "Durrant anchor end-to-end. The v2 → Channel A path is safe."
        if all_ok else
        "FAIL: Channel A on v2 arrays diverges from the Durrant anchor. "
        "v2 must be audited before ANY Channel-A-driven claim on v2 arrays."
    ))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
