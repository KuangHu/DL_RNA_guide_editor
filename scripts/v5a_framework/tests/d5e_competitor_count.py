"""D5e — competitor_count comparison T-WT vs V4.2, both at L=11.

Fixes D5d's tie-blind measurement:
  P(delta <= 0) = 15.3% but planted within argmax +/-5 = 10.0% in T-WT is
  an impossibility (delta >= 0 always). The 5.3-pp gap is 100% ties: T-WT
  planted positions are all tied at max with distant nc positions.

  Third recurrence of the tie problem (peak-level PPV, exact-hit centroid
  tie-break, now argmax).

Correct metric: competitor_count = #{nc positions : m >= m_planted}
  Tie-robust (no argmax dependence)
  Direct input to E-value analytic (E = N_windows * P(Bin >= m_planted))
  Direct input to P(S=k conjunction) probability
  Cannot be gamed by fidelity axis — fidelity moves planted_m, competitor
  count moves accordingly, so the ratio is what characterizes the task.

Generator Requirement 0 (revised): fraction(competitor_count <= 10) < 5%.
T-WT baseline: 2.94%. V4.2 current: 33.5%.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from scripts.v5a_framework.match_table import load as load_mt


L_DET = 11
MT_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/durrant_positive"
V42_POS = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl"


def per_pos_m(nc, flank, L):
    fwd, rc = dot_plot(nc, flank)
    w_f = windowed_matches(fwd, L)
    w_r = windowed_matches(rc, L)
    if w_f.size == 0 or w_r.size == 0:
        return np.zeros(0, dtype=np.int32)
    n = min(w_f.shape[0], w_r.shape[0])
    return np.maximum(w_f.max(axis=1)[:n], w_r.max(axis=1)[:n])


def main(n_v42_sample: int = 2000) -> int:
    mt = load_mt(MT_POS)
    twt_ids = [t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT")]

    twt_compete = []; twt_planted_m = []
    for tnp_id in twt_ids:
        nc = mt.tnps[tnp_id].nc
        for s in mt.tnps[tnp_id].sites:
            m_arr = per_pos_m(nc, s.flank, L_DET)
            if s.gold_nc is None or s.gold_nc >= len(m_arr):
                continue
            m_p = int(m_arr[s.gold_nc])
            c = int((m_arr >= m_p).sum())
            twt_compete.append(c); twt_planted_m.append(m_p)

    v42_compete = []; v42_planted_m = []
    with open(V42_POS) as f:
        for line in f:
            d = json.loads(line)
            lab = d["labels"]
            if not lab.get("is_positive"):
                continue
            span = lab.get("guide_span_in_active_noncoding")
            if not span:
                continue
            a = lab.get("active_noncoding_index", 0) or 0
            ncs = d["inputs"].get("noncoding_regions", [])
            if a >= len(ncs):
                continue
            m_arr = per_pos_m(ncs[a], d["inputs"]["flank"], L_DET)
            planted_start = int(span[0])
            if planted_start >= len(m_arr):
                continue
            m_p = int(m_arr[planted_start])
            c = int((m_arr >= m_p).sum())
            v42_compete.append(c); v42_planted_m.append(m_p)
            if len(v42_compete) >= n_v42_sample:
                break

    t = np.array(twt_compete); v = np.array(v42_compete)
    tm = np.array(twt_planted_m); vm = np.array(v42_planted_m)
    print(f"n T-WT sites: {len(t)}, n V4.2 sites: {len(v)}")
    print()
    print(f"{'metric':<32s} {'T-WT':>12s} {'V4.2':>12s}")
    print(f"  {'competitor_count mean':<30s} {t.mean():>12.2f} {v.mean():>12.2f}")
    print(f"  {'competitor_count median':<30s} {int(np.median(t)):>12d} {int(np.median(v)):>12d}")
    print(f"  {'competitor_count p95':<30s} {int(np.percentile(t,95)):>12d} {int(np.percentile(v,95)):>12d}")
    print(f"  {'planted sole max (c=1)':<30s} {(t==1).mean():>12.4f} {(v==1).mean():>12.4f}")
    print(f"  {'competitor_count <= 5':<30s} {(t<=5).mean():>12.4f} {(v<=5).mean():>12.4f}")
    print(f"  {'competitor_count <= 10':<30s} {(t<=10).mean():>12.4f} {(v<=10).mean():>12.4f}")
    print(f"  {'competitor_count <= 50':<30s} {(t<=50).mean():>12.4f} {(v<=50).mean():>12.4f}")

    print()
    print("Stratified by planted_m (at L=11):")
    print(f"  {'planted_m':>10s} {'T-WT n':>8s} {'V4.2 n':>8s} {'T-WT c_med':>12s} {'V4.2 c_med':>12s}")
    for pm in range(5, 13):
        ti = t[tm == pm]; vi = v[vm == pm]
        if len(ti) or len(vi):
            t_med = float(np.median(ti)) if len(ti) else float("nan")
            v_med = float(np.median(vi)) if len(vi) else float("nan")
            print(f"  {pm:>10d} {len(ti):>8d} {len(vi):>8d} {t_med:>12.1f} {v_med:>12.1f}")

    print()
    print("Requirement 0 (revised, competitor-count based):")
    print("  For the generator's planted sites at the target task's detector L:")
    print("  fraction with competitor_count <= 10 must be < 5%")
    print(f"  T-WT baseline : {(t<=10).mean():.4f}  ({int((t<=10).sum())}/{len(t)})")
    print(f"  V4.2 current  : {(v<=10).mean():.4f}  ({int((v<=10).sum())}/{len(v)})")
    print(f"  Gap: V4.2 is {(v<=10).mean()/(t<=10).mean():.1f}x above the baseline.")

    print()
    print("CAVEAT: T-WT is 1 natural bridge RNA (n=1 systems). Requirement 0's target")
    print("distribution is calibrated on that one system. Extending calibration to")
    print("seekRNA's 5 additional natural systems (IS1111 + IS110) would bring n=6")
    print("before any synthetic generator scale-up decisions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
