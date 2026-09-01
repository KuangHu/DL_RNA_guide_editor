"""B — validate the generator acceptance tests before implementing the generator.

Two tests defined in docs/generator_spec.md:

Test 1: competitor_count distribution matches T-WT
  - fraction(competitor_count <= 10) < 5%
  - median(competitor_count / L) at planted_m=8 in [0.19, 0.24]
  - zero mass at planted_m >= 10

Test 2: planted guide is NOT per-site argmax
  - competitor_count based (not argmax-based, which is tie-blind)
  - P(competitor_count == 1) ~ T-WT baseline 15.3%
  - median competitor_count at planted position >= 2

Both tests MUST:
  - Accept Durrant T-WT (the natural reference)
  - Reject V4.2 (the known-bad synthetic)

If either test rejects T-WT, the test itself is broken. If either
test accepts V4.2, it fails to catch a documented gap.

Same shape as test_tau0_anchor.py: the acceptance tests need
their own anchor before they can be used to accept generator output.
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

# Test 1 (v3, length-conditioned, 2026-08-31):
#   Original v2 used `fraction(competitor_count <= 10) < 5%` as an
#   ABSOLUTE threshold. Analytic competitor_count ≈ 0.21 × (nc_len - L + 1),
#   so at nc_len = 70 the expected value is only 12.6 — the absolute
#   threshold would reject the entire short-nc population by construction
#   the moment nc_len ~ U[70, 300] is added. Replaced with a normalized-
#   rate lower bound at 0.10 (half the analytic mean 0.21).
TEST1_RATE_LO_MAX   = 0.05          # fraction(count / n_positions < 0.10) < 5%
TEST1_RATE_LO_FLOOR = 0.10
TEST1_RATE_MED_LO   = 0.19          # median(competitor_count / n_pos) at m=8
TEST1_RATE_MED_HI   = 0.24
TEST1_ZERO_MASS_AT_MGE10 = 0.02     # <= 2% at planted_m >= 10

TEST2_SOLE_MAX_MAX  = 0.01          # P(competitor_count == 1) < 1%
                                     # T-WT baseline: 0/170 = 0.0%
                                     # V4.2 currently: 50/2000 = 2.5%
TEST2_MEDIAN_MIN    = 2             # median competitor_count >= 2


def per_pos_m(nc, flank, L):
    fwd, rc = dot_plot(nc, flank)
    w_f = windowed_matches(fwd, L)
    w_r = windowed_matches(rc, L)
    if w_f.size == 0 or w_r.size == 0:
        return np.zeros(0, dtype=np.int32)
    n = min(w_f.shape[0], w_r.shape[0])
    return np.maximum(w_f.max(axis=1)[:n], w_r.max(axis=1)[:n])


def measure_corpus(name, sites_iter, use_L=L_DET) -> dict:
    """sites_iter yields (nc, flank, planted_start) tuples."""
    compete = []
    planted_m_list = []
    nc_len_list = []
    for nc, flank, planted_start in sites_iter:
        m_arr = per_pos_m(nc, flank, use_L)
        if planted_start is None or planted_start >= len(m_arr):
            continue
        m_p = int(m_arr[planted_start])
        c = int((m_arr >= m_p).sum())
        compete.append(c)
        planted_m_list.append(m_p)
        nc_len_list.append(len(nc))
    return {
        "name": name,
        "n": len(compete),
        "competitor_count": np.array(compete),
        "planted_m": np.array(planted_m_list),
        "nc_len": np.array(nc_len_list),
    }


def run_test1(stats: dict, use_L: int) -> tuple[bool, dict]:
    n = stats["n"]
    if n == 0:
        return False, {"reason": "no sites"}
    c = stats["competitor_count"]
    pm = stats["planted_m"]
    nl = stats["nc_len"]

    # n_positions per site = nc_len - use_L + 1 (number of L-windows on nc).
    # Rate = competitor_count / n_positions is length-invariant (per user's
    # analytic constant ~0.21). Use rate-based lower bound instead of an
    # absolute count threshold.
    n_pos = np.maximum(nl - use_L + 1, 1)
    rate = c / n_pos
    frac_below_floor = float((rate < TEST1_RATE_LO_FLOOR).mean())
    m8_mask = (pm == 8)
    rate_at_m8 = rate[m8_mask]
    rate_med = float(np.median(rate_at_m8)) if len(rate_at_m8) else float("nan")
    zero_mass = float((pm >= 10).mean())

    checks = {
        f"fraction(count / n_pos < {TEST1_RATE_LO_FLOOR})": (frac_below_floor, TEST1_RATE_LO_MAX, "<"),
        "median(c / n_pos) at planted_m=8":                  (rate_med, (TEST1_RATE_MED_LO, TEST1_RATE_MED_HI), "in"),
        "P(planted_m >= 10)":                                (zero_mass, TEST1_ZERO_MASS_AT_MGE10, "<"),
    }
    result = {}
    passes = True
    for name, (val, target, op) in checks.items():
        if op == "<":
            ok = val <= target
            result[name] = {"value": val, "target": f"< {target}", "pass": ok}
        elif op == "in":
            lo, hi = target
            ok = (not np.isnan(val)) and (lo <= val <= hi)
            result[name] = {"value": val, "target": f"in [{lo}, {hi}]", "pass": ok}
        if not ok:
            passes = False
    return passes, result


def run_test2(stats: dict) -> tuple[bool, dict]:
    n = stats["n"]
    if n == 0:
        return False, {"reason": "no sites"}
    c = stats["competitor_count"]

    sole_max = float((c == 1).mean())
    med = float(np.median(c))

    checks = {
        "P(competitor_count == 1) [planted sole max]": (sole_max, TEST2_SOLE_MAX_MAX, "<"),
        "median(competitor_count)":                    (med, TEST2_MEDIAN_MIN, ">="),
    }
    result = {}
    passes = True
    for name, (val, target, op) in checks.items():
        if op == "<":
            ok = val <= target
            result[name] = {"value": val, "target": f"< {target}", "pass": ok}
        elif op == ">=":
            ok = val >= target
            result[name] = {"value": val, "target": f">= {target}", "pass": ok}
        if not ok:
            passes = False
    return passes, result


def _twt_sites():
    mt = load_mt(MT_POS)
    for tnp_id in mt.tnp_ids:
        if not tnp_id.startswith("durrant_bridge_RNA_T-WT_D-WT"):
            continue
        nc = mt.tnps[tnp_id].nc
        for s in mt.tnps[tnp_id].sites:
            yield (nc, s.flank, s.gold_nc)


def _v42_sites(n_max: int = 2000):
    n = 0
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
            yield (ncs[a], d["inputs"]["flank"], int(span[0]))
            n += 1
            if n >= n_max:
                break


def _report(label, checks, pw):
    print(f"  {label}: {'PASS' if pw else 'FAIL'}")
    for name, r in checks.items():
        mark = "✓" if r.get("pass") else "✗"
        val = r.get("value")
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        print(f"    {mark} {name}: {val_str}  (target: {r['target']})")


def main() -> int:
    print("=== Measuring Durrant T-WT (natural reference) ===")
    twt = measure_corpus("Durrant T-WT", _twt_sites(), use_L=L_DET)
    print(f"  n = {twt['n']}")
    print()
    print("=== Measuring V4.2 (known-bad synthetic) ===")
    v42 = measure_corpus("V4.2", _v42_sites(2000), use_L=L_DET)
    print(f"  n = {v42['n']}")
    print()

    print("=== Test 1: competitor_count distribution matches T-WT ===")
    t1_twt_pass, t1_twt_checks = run_test1(twt, L_DET)
    t1_v42_pass, t1_v42_checks = run_test1(v42, L_DET)
    _report("T-WT (must PASS)", t1_twt_checks, t1_twt_pass)
    _report("V4.2 (must FAIL)", t1_v42_checks, t1_v42_pass)

    print()
    print("=== Test 2: planted NOT per-site sole max (competitor_count-based) ===")
    t2_twt_pass, t2_twt_checks = run_test2(twt)
    t2_v42_pass, t2_v42_checks = run_test2(v42)
    _report("T-WT (must PASS)", t2_twt_checks, t2_twt_pass)
    _report("V4.2 (must FAIL)", t2_v42_checks, t2_v42_pass)

    print()
    print("=== B validation ===")
    ok = True
    if not t1_twt_pass:
        print("  ✗ Test 1 rejects T-WT (natural reference)")
        ok = False
    if t1_v42_pass:
        print("  ✗ Test 1 accepts V4.2 (should reject)")
        ok = False
    if not t2_twt_pass:
        print("  ✗ Test 2 rejects T-WT (natural reference)")
        ok = False
    if t2_v42_pass:
        print("  ✗ Test 2 accepts V4.2 (should reject)")
        ok = False

    if ok:
        print("  PASS: both tests accept T-WT and reject V4.2")
        print("  Acceptance criteria are validated. Ready to build generator against them.")
    else:
        print("  FAIL: acceptance criteria need adjustment before generator implementation.")
        print("  See failing rows above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
