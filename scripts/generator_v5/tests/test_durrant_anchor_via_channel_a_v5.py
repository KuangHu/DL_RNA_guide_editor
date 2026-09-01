"""Cross-implementation anchor check.

Runs channel_a_v5.compute_channel_a on the frozen Durrant MatchTable
and verifies it produces the same numbers as tests/test_tau0_anchor.py.

If they match: channel_a_v5 is equivalent to the framework path, and
the V5 44.3%/96.9% vs Durrant 33.8%/95.5% comparison is valid across
implementations.
If they don't: V5 numbers must be recomputed with the framework code
before any V5↔Durrant claim.

Anchor (Channel A doc / test_tau0_anchor):
    covered=22   peaks=23   correct=22
    tnps_with_correct=21   exact=21
    coverage=0.3385  PPV_peak=0.9565  PPV_Tnp=0.9545  exact_rate=0.3231
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.v5a_framework.match_table import load as load_mt
from scripts.v5a_framework.variant import spec_m_threshold_L11, run_variant
from scripts.generator_v5.channel_a_v5 import compute_channel_a


DURRANT_SHARD = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"

# Anchor values (per test_tau0_anchor.py — locked)
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


def main() -> int:
    print(f"[anchor] loading Durrant MatchTable from {DURRANT_SHARD}")
    mt = load_mt(DURRANT_SHARD)
    print(f"[anchor] loaded {len(mt.tnp_ids)} Tnps")

    print(f"[anchor] running Mode 2 spec (fixed L=11, m>=8, tau=0, S=5)")
    spec = spec_m_threshold_L11(m=8, tau=0, S=5)
    peaks = run_variant(mt, spec)

    print(f"[anchor] computing metrics via channel_a_v5.compute_channel_a")
    # Empty arch dict — Durrant Tnps have no V5-style arch metadata; we
    # only care about the overall (unstratified) result.
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
    print("[anchor] " + ("PASS: channel_a_v5 and framework path produce identical Durrant numbers"
                          if all_ok else
                          "FAIL: channel_a_v5 diverges from framework — V5↔Durrant comparison invalid"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
