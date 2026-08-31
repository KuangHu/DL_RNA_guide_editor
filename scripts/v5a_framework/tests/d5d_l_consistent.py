"""D5d — L-consistent Durrant T-WT vs V4.2 argmax comparison.

D5c used L=guide_length for V4.2 (varying 12-16) while D5b used L=11 for
Durrant. Cross-L comparison is not valid — same shape as the m@own-L vs
m@detector-L error that broke p^5 attribution. This script measures both
at L=11 (detector window) so the deltas are apples-to-apples.

Result: the gap D5c reported (V4.2 40.7% within +/-5 vs Durrant ~0%)
shrinks to 32.5% vs 10.0% at L=11, and stratified by planted_m the two
distributions are essentially identical.

The real V4.2/T-WT gap: V4.2 SAMPLES planted_m distributions
biased high (45% at m>=9), while T-WT concentrates at m=8 (86%).
Corrected generator Requirement 0: planted_m distribution must match
the target task's operating-point distribution, not a proximity metric.
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


def per_pos_m_max_pooled(nc, flank, L):
    fwd_dot, rc_dot = dot_plot(nc, flank)
    w_fwd = windowed_matches(fwd_dot, L)
    w_rc = windowed_matches(rc_dot, L)
    if w_fwd.size == 0 or w_rc.size == 0:
        return np.zeros(0, dtype=np.int32)
    n = min(w_fwd.shape[0], w_rc.shape[0])
    return np.maximum(w_fwd.max(axis=1)[:n], w_rc.max(axis=1)[:n])


def main(n_v42_sample: int = 2000) -> int:
    mt = load_mt(MT_POS)
    twt_ids = [t for t in mt.tnp_ids if t.startswith("durrant_bridge_RNA_T-WT_D-WT")]

    twt_data = []
    for tnp_id in twt_ids:
        nc = mt.tnps[tnp_id].nc
        for s in mt.tnps[tnp_id].sites:
            planted_start = s.gold_nc
            m_arr = per_pos_m_max_pooled(nc, s.flank, L_DET)
            if planted_start is None or planted_start >= len(m_arr):
                continue
            argmax = int(m_arr.argmax())
            twt_data.append({
                "planted_m": int(m_arr[planted_start]),
                "argmax_m": int(m_arr[argmax]),
                "delta": int(m_arr[argmax]) - int(m_arr[planted_start]),
                "argmax_dist": abs(argmax - planted_start),
            })

    v42_data = []
    with open(V42_POS) as f:
        for line in f:
            d = json.loads(line)
            lab = d["labels"]
            if not lab.get("is_positive"):
                continue
            span = lab.get("guide_span_in_active_noncoding")
            if not span:
                continue
            planted_start = int(span[0])
            a = lab.get("active_noncoding_index", 0) or 0
            ncs = d["inputs"].get("noncoding_regions", [])
            if a >= len(ncs):
                continue
            nc = ncs[a]
            flank = d["inputs"]["flank"]
            m_arr = per_pos_m_max_pooled(nc, flank, L_DET)
            if planted_start >= len(m_arr):
                continue
            argmax = int(m_arr.argmax())
            v42_data.append({
                "planted_m": int(m_arr[planted_start]),
                "argmax_m": int(m_arr[argmax]),
                "delta": int(m_arr[argmax]) - int(m_arr[planted_start]),
                "argmax_dist": abs(argmax - planted_start),
            })
            if len(v42_data) >= n_v42_sample:
                break

    def _summ(data, name):
        n = len(data)
        pm = np.array([r["planted_m"] for r in data])
        am = np.array([r["argmax_m"] for r in data])
        de = np.array([r["delta"] for r in data])
        w5 = np.array([r["argmax_dist"] <= 5 for r in data])
        print(f"{name} (n={n} at L=11):")
        print(f"  planted_m         mean={pm.mean():.2f}  med={int(np.median(pm))}")
        print(f"    distribution:    {sorted(dict(zip(*np.unique(pm, return_counts=True))).items())}")
        print(f"  argmax_m          mean={am.mean():.2f}  med={int(np.median(am))}")
        print(f"  delta             mean={de.mean():.2f}  med={int(np.median(de))}  p95={float(np.percentile(de,95)):.0f}")
        print(f"  planted <=5 from argmax : {w5.sum()}/{n} = {w5.mean():.4f}")

    print("=" * 78)
    _summ(twt_data, "Durrant T-WT (natural WT)")
    print()
    _summ(v42_data, "V4.2 (synthetic)")

    # Stratified delta comparison
    print()
    print("Delta distribution at MATCHED planted_m (the fair comparison):")
    print(f"  {'planted_m':<10s} {'T-WT n':>8s} {'V4.2 n':>8s} {'T-WT delta med':>16s} {'V4.2 delta med':>16s}")
    for pm_v in range(5, 13):
        t_here = [r["delta"] for r in twt_data if r["planted_m"] == pm_v]
        v_here = [r["delta"] for r in v42_data if r["planted_m"] == pm_v]
        if t_here or v_here:
            print(f"  {pm_v:<10d} {len(t_here):>8d} {len(v_here):>8d} "
                  f"{(float(np.median(t_here)) if t_here else float('nan')):>16.1f} "
                  f"{(float(np.median(v_here)) if v_here else float('nan')):>16.1f}")

    print()
    print("Requirement 0 (revised for generator rebuild):")
    print("  V4.2 planted_m distribution differs from T-WT. Generator must sample")
    print("  planted_m matching the target task's operating-point distribution.")
    print("  T-WT operating point at L=11: mode m=8, ~86% within [7,9], zero at m>=10.")
    print("  V4.2 current: only 30% at m=8, with ~45% at m>=9.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
