"""Do Mode 1 and Mode 2 detect the same Durrant Tnps?

Both give coverage = 22/65 = 0.338 on Durrant but under DIFFERENT
mechanisms (Mode 2 = L=11 direct m>=8, Mode 1 = L=9 subwindow E<4).
If the detected Tnp sets are identical, Mode 1's PPV 1.000 vs Mode 2's
0.9565 is just "same 22 Tnps, Mode 1 drops one Mode 2 false positive"
— L-marginalization adds nothing on Durrant.

If they differ, Mode 1 admits some Tnps Mode 2 misses (real
L-marginalization value) and vice versa.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.v5a_framework.match_table import load as load_mt
from scripts.v5a_framework.variant import (
    spec_m_threshold_L11, spec_min_E_9_12, run_variant,
)


DURRANT_SHARD = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"


def main() -> int:
    mt = load_mt(DURRANT_SHARD)
    all_tnps = set(mt.tnp_ids)

    print(f"[overlap] loaded {len(all_tnps)} Durrant Tnps")
    print(f"[overlap] Mode 2 (m8, tau=0, S=5)")
    peaks_m2 = run_variant(mt, spec_m_threshold_L11(m=8, tau=0, S=5))
    detected_m2 = {t for t, p in peaks_m2.items() if p}

    print(f"[overlap] Mode 1 (min_E, tau=5, S=5)")
    peaks_m1 = run_variant(mt, spec_min_E_9_12(E=4.0, tau=5, S=5))
    detected_m1 = {t for t, p in peaks_m1.items() if p}

    print()
    print(f"=== Detected Tnps ===")
    print(f"  Mode 2:              {len(detected_m2)}")
    print(f"  Mode 1:              {len(detected_m1)}")
    print(f"  Intersection:        {len(detected_m2 & detected_m1)}")
    print(f"  Mode 2 only:         {len(detected_m2 - detected_m1)}")
    print(f"  Mode 1 only:         {len(detected_m1 - detected_m2)}")
    print(f"  Union:               {len(detected_m2 | detected_m1)}")

    # Identical set?
    if detected_m2 == detected_m1:
        print(f"\n  IDENTICAL sets — L-marginalization adds ZERO new Tnps on Durrant.")
        print(f"  Mode 1's PPV improvement is just dropping one Mode 2 FP.")
    else:
        print(f"\n  DIFFERENT sets — Mode 1 admits {len(detected_m1 - detected_m2)} Tnps Mode 2 misses.")

    # Also check overlap of correctly-classified Tnps (has a peak passing IoU)
    def _correct(mt, peaks):
        correct = set()
        for tnp_id, pks in peaks.items():
            if not pks:
                continue
            tnp = mt.tnps[tnp_id]
            gold_nc = tnp.sites[0].gold_nc
            gold_L = tnp.sites[0].gold_L
            for pk in pks:
                a0, a1 = pk.position, pk.position + pk.L_at_peak
                b0, b1 = gold_nc, gold_nc + gold_L
                inter = max(0, min(a1, b1) - max(a0, b0))
                union = (a1 - a0) + (b1 - b0) - inter
                if union > 0 and inter / union >= 0.5:
                    correct.add(tnp_id)
                    break
        return correct

    correct_m2 = _correct(mt, peaks_m2)
    correct_m1 = _correct(mt, peaks_m1)
    print()
    print(f"=== Correctly-classified Tnps (peak passes IoU >= 0.5 with gold) ===")
    print(f"  Mode 2 correct:     {len(correct_m2)}")
    print(f"  Mode 1 correct:     {len(correct_m1)}")
    print(f"  Both correct:       {len(correct_m2 & correct_m1)}")
    print(f"  Only Mode 2:        {len(correct_m2 - correct_m1)}   -> {sorted(correct_m2 - correct_m1)[:5]}")
    print(f"  Only Mode 1:        {len(correct_m1 - correct_m2)}   -> {sorted(correct_m1 - correct_m2)[:5]}")

    # Union verdict
    if correct_m1 - correct_m2:
        print(f"\n  Mode 1 correctly detects {len(correct_m1 - correct_m2)} Tnps Mode 2 does NOT.")
        print(f"  L-marginalization has REAL rescue value on Durrant.")
    elif correct_m2 - correct_m1:
        print(f"\n  Mode 1 MISSES {len(correct_m2 - correct_m1)} correct-Tnp detections Mode 2 makes.")
        print(f"  L-marginalization is a STRICT LOSS on Durrant.")
    else:
        print(f"\n  Correct sets are IDENTICAL — L-marginalization adds zero rescue on Durrant.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
