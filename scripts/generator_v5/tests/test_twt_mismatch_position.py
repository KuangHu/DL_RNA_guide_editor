"""Measure real mismatch position distribution on T-WT gold sites.

For each of the 170 T-WT gold sites in the Durrant MatchTable:
  guide  = nc[gold_nc : gold_nc + gold_L]
  target = flank[target_flank_start : target_flank_start + gold_L]
Compare position-by-position, record mismatch positions in the L=11 window.

If mismatches cluster at one end, uniform-random mismatch placement in
V5 is a generator defect that explains Mode 1's failure (E<4 at L=9
subwindows requires clustered mismatches).

Also reports the L=9 subwindow m distribution — the direct predictor
of whether Mode 1 sees the site.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.v5a_framework.match_table import load as load_mt


DURRANT_SHARD = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"

_COMPLEMENT = str.maketrans("ACGT", "TGCA")
def _rc(s: str) -> str: return s.translate(_COMPLEMENT)[::-1]


def main() -> int:
    mt = load_mt(DURRANT_SHARD)
    twt_ids = [t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT")]
    print(f"[twt-mm] T-WT variants: {len(twt_ids)} Tnps")

    mm_positions_fwd = []           # each is list[int] of mismatch positions per site
    mm_positions_rc  = []
    m_at_L9_fwd = []
    m_at_L9_rc = []

    for tnp_id in twt_ids:
        tnp = mt.tnps[tnp_id]
        nc = tnp.nc.upper().replace("U", "T")
        for s in tnp.sites:
            if s.gold_nc is None or s.gold_L is None or s.target_flank_start is None:
                continue
            L = s.gold_L
            guide = nc[s.gold_nc : s.gold_nc + L]
            target = s.flank[s.target_flank_start : s.target_flank_start + L]
            if len(guide) != L or len(target) != L:
                continue
            # fwd alignment
            fwd_mm = [i for i in range(L) if guide[i] != target[i]]
            mm_positions_fwd.append(fwd_mm)
            # rc alignment (Durrant convention: target is planted on the flank
            # such that rc(guide) matches target; try both)
            rc_guide = _rc(guide)
            rc_mm = [i for i in range(L) if rc_guide[i] != target[i]]
            mm_positions_rc.append(rc_mm)

            # L=9 subwindow m: best over 3 subwindows [0:9], [1:10], [2:11]
            best_L9_fwd = 0
            best_L9_rc = 0
            for off in range(L - 9 + 1):
                m_f = sum(1 for i in range(9) if guide[off + i] == target[off + i])
                m_r = sum(1 for i in range(9) if rc_guide[off + i] == target[off + i])
                if m_f > best_L9_fwd: best_L9_fwd = m_f
                if m_r > best_L9_rc: best_L9_rc = m_r
            m_at_L9_fwd.append(best_L9_fwd)
            m_at_L9_rc.append(best_L9_rc)

    # Determine orientation: whichever gives fewer mean mismatches
    mm_fwd_avg = float(np.mean([len(x) for x in mm_positions_fwd]))
    mm_rc_avg = float(np.mean([len(x) for x in mm_positions_rc]))
    print(f"[twt-mm] mean mismatches fwd={mm_fwd_avg:.2f}  rc={mm_rc_avg:.2f}")
    if mm_fwd_avg <= mm_rc_avg:
        mm_positions = mm_positions_fwd
        m_at_L9 = m_at_L9_fwd
        print(f"[twt-mm] using FWD alignment (better match)")
    else:
        mm_positions = mm_positions_rc
        m_at_L9 = m_at_L9_rc
        print(f"[twt-mm] using RC alignment (better match)")

    # Per-site mismatch count
    counts = Counter(len(x) for x in mm_positions)
    print(f"\n=== Per-site mismatch count in L=11 guide ===")
    for k in sorted(counts):
        print(f"  {k} mismatches: {counts[k]} sites ({counts[k]/len(mm_positions)*100:.1f}%)")

    # Position histogram (0 = 5' end of guide, 10 = 3' end)
    pos_hist = Counter()
    total_mm = 0
    for mm_list in mm_positions:
        for p in mm_list:
            pos_hist[p] += 1
            total_mm += 1
    print(f"\n=== Mismatch position histogram (over all {total_mm} mismatch events) ===")
    print(f"  pos    count  fraction  uniform_expected")
    uniform_expected = total_mm / 11
    for p in range(11):
        c = pos_hist.get(p, 0)
        print(f"  {p:>3d}  {c:>7d}   {c/total_mm*100:>6.2f}%   {uniform_expected:>7.1f}")

    # Chi-square test vs uniform
    obs = np.array([pos_hist.get(p, 0) for p in range(11)])
    exp = np.full(11, total_mm / 11)
    chi2 = float(((obs - exp) ** 2 / exp).sum())
    print(f"\n  chi2 vs uniform (df=10) = {chi2:.2f} (critical p=0.05 is 18.31)")
    print(f"  {'UNIFORM' if chi2 < 18.31 else 'NON-UNIFORM (biased)'}")

    # Spatial coherence: are 3-mismatch sites clustered?
    print(f"\n=== Clustering: for 3-mismatch sites, range(max-min) of mm positions ===")
    for target_k in (2, 3, 4):
        subs = [mm for mm in mm_positions if len(mm) == target_k]
        if not subs:
            continue
        ranges = [max(mm) - min(mm) for mm in subs]
        # For 3 uniform mm on 11 positions, expected range ~ 8. Clustered would be < 5.
        med_range = float(np.median(ranges))
        print(f"  k={target_k}: n={len(subs)}, median range={med_range:.1f} "
              f"(uniform expected ~= {8*(target_k-1)/2:.1f} for k=3)")

    # L=9 subwindow m distribution
    print(f"\n=== L=9 subwindow max m distribution (drives Mode 1 E<4 admission) ===")
    m9_counts = Counter(m_at_L9)
    for m in sorted(m9_counts):
        c = m9_counts[m]
        print(f"  m9={m}: {c} sites ({c/len(m_at_L9)*100:.1f}%)")
    frac_m9_ge_8 = float(np.mean([m >= 8 for m in m_at_L9]))
    print(f"\n  fraction with L=9 subwindow m >= 8 (E<4): {frac_m9_ge_8*100:.1f}%")
    print(f"  Mode 1 needs this on all 5 sites -> P^5 = {frac_m9_ge_8**5:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
