"""X1' + X2a — Corrected window scan + synthetic N_nc concat test.

X1' changes:
  - Both fixed L=11 AND min-E-over-L ∈ {9,10,11,12} tested per τ. Fixed-L
    alone couples τ to length mismatch; min-E-over-L uses per-L Bin null,
    E-value comparable across L (that's the reason E-value exists).
  - Report FOUR metrics per variant (not just rate):
      coverage       fraction of Tnps with any detected peak above threshold
      PPV            fraction of detected peaks overlapping annotated TBL span (IoU ≥ 0.5)
      exact-hit      fraction of Tnps where primary peak distance to gold_nc ≤ 1
      within-5nt     fraction of Tnps where primary peak within 5 nt of gold_nc
  - Discipline: matched shuffled null per variant, safe_ratio(varying_dim="null_model").

X2a: synthetic N_nc concat.
  - Take each Tnp's ncRNA. Concatenate with 1 or 2 UNRELATED Tnps' ncRNAs (random
    from other Tnps) to simulate N_nc = 2 and N_nc = 3.
  - Recompute Channel A at the best-window from X1' (τ=?, L=min-E).
  - Report degradation of PPV and coverage as N_nc grows.
  - Gold_nc positions get remapped into the concatenated coordinate. Search space
    scales linearly with N_nc.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.stats import binom

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")
sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/scripts")

from preprocess.alignment import dot_plot, windowed_matches
from v5a_eval_asserts import Metric, MetricCondition, safe_ratio


def _fwd_win_max(nc: str, flank: str, L: int) -> np.ndarray:
    fwd, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd, L)
    if win.size == 0: return np.zeros(0, dtype=np.int32)
    return win.max(axis=1)


def _E(m: int, L: int, nc_len: int, flank_len: int, p: float = 0.25) -> float:
    Nw = max(1, (nc_len - L + 1) * (flank_len - L + 1))
    return Nw * float(1.0 - binom.cdf(m - 1, L, p))


def _site_hits_fixed(nc: str, flank: str, L: int, m: int) -> dict[int, float]:
    """Positions and their E-values under fixed L, threshold m."""
    per_pos = _fwd_win_max(nc, flank, L)
    hits = {}
    for pos in range(len(per_pos)):
        m_pos = int(per_pos[pos])
        if m_pos >= m:
            hits[pos] = _E(m_pos, L, len(nc), len(flank))
    return hits


def _site_hits_minE(nc: str, flank: str, L_range=(9, 10, 11, 12),
                      E_thresh: float = 4.0) -> dict[int, float]:
    """Positions where min-E across L range is below threshold."""
    per_pos_min_E = defaultdict(lambda: float("inf"))
    nc_len = len(nc); flank_len = len(flank)
    for L in L_range:
        per_pos = _fwd_win_max(nc, flank, L)
        for pos in range(len(per_pos)):
            m_pos = int(per_pos[pos])
            if m_pos <= 0: continue
            E = _E(m_pos, L, nc_len, flank_len)
            if E < per_pos_min_E[pos]:
                per_pos_min_E[pos] = E
    return {p: e for p, e in per_pos_min_E.items() if e < E_thresh}


def _apply_kernel(hits_lists: list[dict], nc_len_pos: int, tau: float) -> np.ndarray:
    S = np.zeros(nc_len_pos, dtype=np.float64)
    for pos in range(nc_len_pos):
        for h in hits_lists:
            if not h: continue
            positions = np.asarray(list(h.keys()))
            dists = np.abs(positions - pos)
            j = int(dists.argmin()); d = int(dists[j])
            if tau <= 0:
                w = 1.0 if d == 0 else 0.0
            else:
                w = float(np.exp(-0.5 * (d / tau) ** 2))
            S[pos] += w
    return S


def _find_peaks(S: np.ndarray, thresh: float, min_dist: int = 5) -> list[int]:
    """Local maxima above threshold, at least min_dist apart."""
    peaks = []
    L = len(S)
    for i in range(L):
        if S[i] < thresh: continue
        is_local_max = True
        for j in range(max(0, i - min_dist), min(L, i + min_dist + 1)):
            if j != i and S[j] > S[i]:
                is_local_max = False; break
        if is_local_max: peaks.append(i)
    return peaks


def _iou_overlap(p, L_win, gold_nc, gold_L, thresh=0.5):
    a0, a1 = p, p + L_win
    b0, b1 = gold_nc, gold_nc + gold_L
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return (inter / max(1e-9, union)) >= thresh


def eval_variant(tnps_5, tnp_nc, hits_maker, hits_maker_name: str, tau: float,
                    all_flanks: list, rng, thresh: float, L_win_for_iou: int = 11,
                    n_perm: int = 20):
    """Evaluate one (hits_maker, tau) variant: coverage, PPV, exact-hit, within-5nt.
    Also matched shuffled control."""
    real_peaks = []
    peaks_by_tnp = {}
    gold_by_tnp = {}
    gold_L_by_tnp = {}
    for tnp, sites in tnps_5.items():
        nc = tnp_nc[tnp]
        hits = [hits_maker(nc, s["flank"]) for s in sites[:5]]
        S = _apply_kernel(hits, len(nc) - L_win_for_iou + 1, tau)
        peaks = _find_peaks(S, thresh)
        peaks_by_tnp[tnp] = peaks
        gold_by_tnp[tnp] = sites[0]["gold_nc"]
        gold_L_by_tnp[tnp] = sites[0].get("gold_L", 11)
        real_peaks.extend(peaks)

    n_tnps = len(tnps_5)
    coverage = sum(1 for p in peaks_by_tnp.values() if p) / n_tnps
    total_peaks = sum(len(p) for p in peaks_by_tnp.values())
    ppv_num = sum(1 for tnp, peaks in peaks_by_tnp.items()
                     for p in peaks
                     if _iou_overlap(p, L_win_for_iou, gold_by_tnp[tnp], gold_L_by_tnp[tnp]))
    ppv = ppv_num / max(1, total_peaks)
    # Exact-hit + within-5nt: distance of PRIMARY peak (nearest to gold or max S) to gold_nc
    exact = 0; w5 = 0
    for tnp, peaks in peaks_by_tnp.items():
        if not peaks: continue
        best_peak = min(peaks, key=lambda p: abs(p - gold_by_tnp[tnp]))
        d = abs(best_peak - gold_by_tnp[tnp])
        if d <= 1: exact += 1
        if d <= 5: w5 += 1
    exact_rate = exact / n_tnps
    w5_rate = w5 / n_tnps

    # Shuffled null
    n_shuf_peaks_total = 0; shuf_covered = 0
    for tnp, sites in tnps_5.items():
        nc = tnp_nc[tnp]
        for _ in range(n_perm):
            idx = rng.choice(len(all_flanks), size=5, replace=False)
            fake_flanks = [all_flanks[int(i)][1] for i in idx]
            fake_hits = [hits_maker(nc, fl) for fl in fake_flanks]
            S = _apply_kernel(fake_hits, len(nc) - L_win_for_iou + 1, tau)
            peaks = _find_peaks(S, thresh)
            if peaks: shuf_covered += 1
            n_shuf_peaks_total += len(peaks)
    shuf_coverage = shuf_covered / (n_tnps * n_perm)

    a_m = Metric(f"real:{hits_maker_name}:tau={tau}", coverage,
                    MetricCondition(match_rule="strict_WC",
                                       null_model="real_flanks_target_intact",
                                       coordinate_system="absolute_nc",
                                       targeting_intact=True, tie_break="soft_kernel",
                                       denominator="tnp"))
    s_m = Metric(f"shuf:{hits_maker_name}:tau={tau}", shuf_coverage,
                    MetricCondition(match_rule="strict_WC",
                                       null_model="shuffled_intra_family_flanks",
                                       coordinate_system="absolute_nc",
                                       targeting_intact=True,
                                       tie_break="soft_kernel", denominator="tnp"))
    try:
        ratio = safe_ratio(a_m, s_m, varying_dim="null_model")
    except ValueError as e:
        print(f"    [safe_ratio violation] {e}")
        ratio = float("nan")

    return {"coverage": coverage, "PPV": ppv, "exact_hit": exact_rate,
              "within_5nt": w5_rate, "shuf_coverage": shuf_coverage,
              "ratio": ratio, "total_peaks_real": total_peaks,
              "total_peaks_shuf": n_shuf_peaks_total}


def load_tnps(cog_path, gold_path):
    gold = {json.loads(l)["site_id"]: json.loads(l) for l in open(gold_path)}
    tnp_sites = defaultdict(list); tnp_nc = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"])
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; tnp = r["transposase_id"]
            if tnp not in tnp_nc: tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc: continue
            tnp_sites[tnp].append({"flank": r["inputs"]["flank"],
                                       "gold_nc": g["guide_start_in_nc"],
                                       "gold_L":  g["target_binding_loop_length"]})
    return {t: s for t, s in tnp_sites.items() if len(s) >= 5}, tnp_nc


def x1_scan(tnps_5, tnp_nc, taus=(0, 1, 2, 3, 5, 10)):
    print(f"\n=== X1' :: window scan with fixed-L=11 vs min-E-over-L ∈ {{9..12}} ===")
    rng = np.random.default_rng(0)
    all_flanks = [(t, s["flank"]) for t, ss in tnps_5.items() for s in ss[:5]]

    variants = []
    for tau in taus:
        variants.append(("fixed_L11", lambda nc, fl: _site_hits_fixed(nc, fl, 11, 8), tau, 5.0 if tau == 0 else 4.5))
        variants.append(("min_E_9_12", lambda nc, fl: _site_hits_minE(nc, fl, (9,10,11,12), 4.0), tau, 5.0 if tau == 0 else 4.5))

    print(f"  {'variant':<24} {'coverage':>9} {'PPV':>6} {'exact':>7} {'≤5nt':>7} {'shuf_cov':>9} {'ratio':>7}")
    results = []
    for name, maker, tau, thresh in variants:
        r = eval_variant(tnps_5, tnp_nc, maker, name, tau, all_flanks, rng, thresh)
        label = f"{name}, τ={tau}"
        print(f"  {label:<24} {r['coverage']:>9.3f} {r['PPV']:>6.3f} {r['exact_hit']:>7.3f} {r['within_5nt']:>7.3f} {r['shuf_coverage']:>9.4f} {r['ratio']:>7.2f}×")
        results.append({"name": name, "tau": tau, **r})
    return results


def x2a_synthetic_N_nc(tnps_5, tnp_nc, best_hits_maker, best_hits_maker_name,
                          best_tau, thresh: float, rng):
    print(f"\n=== X2a :: synthetic N_nc concatenation test (best-window from X1') ===")
    print(f"  using best variant: {best_hits_maker_name}, τ={best_tau}")
    all_flanks = [(t, s["flank"]) for t, ss in tnps_5.items() for s in ss[:5]]
    other_ncs = {t: nc for t, nc in tnp_nc.items()}

    for N_nc in (1, 2, 3):
        real_covered = 0; real_exact = 0; total_peaks_real = 0
        peaks_that_are_correct = 0
        shuf_covered = 0; total_peaks_shuf = 0
        n_tnps = 0
        for tnp, sites in tnps_5.items():
            nc = tnp_nc[tnp]
            if N_nc > 1:
                # Concatenate with N_nc-1 unrelated ncRNAs
                other_tnp_ids = [t for t in tnp_nc if t != tnp]
                pick = rng.choice(len(other_tnp_ids), size=N_nc - 1, replace=False)
                extras = [tnp_nc[other_tnp_ids[int(i)]] for i in pick]
                combined_nc = nc + "N" * 20 + ("N" * 20).join(extras)
            else:
                combined_nc = nc
            n_tnps += 1
            gold_nc = sites[0]["gold_nc"]     # gold_nc stays in the original nc, which is at position 0..len(nc)
            gold_L  = sites[0]["gold_L"]
            hits = [best_hits_maker(combined_nc, s["flank"]) for s in sites[:5]]
            S = _apply_kernel(hits, len(combined_nc) - 11 + 1, best_tau)
            peaks = _find_peaks(S, thresh)
            total_peaks_real += len(peaks)
            if peaks:
                real_covered += 1
                for p in peaks:
                    if _iou_overlap(p, 11, gold_nc, gold_L):
                        peaks_that_are_correct += 1
                best_peak = min(peaks, key=lambda p: abs(p - gold_nc))
                if abs(best_peak - gold_nc) <= 1: real_exact += 1
            # shuffled
            for _ in range(20):
                idx = rng.choice(len(all_flanks), size=5, replace=False)
                fake_flanks = [all_flanks[int(i)][1] for i in idx]
                fake_hits = [best_hits_maker(combined_nc, fl) for fl in fake_flanks]
                S_s = _apply_kernel(fake_hits, len(combined_nc) - 11 + 1, best_tau)
                peaks_s = _find_peaks(S_s, thresh)
                if peaks_s: shuf_covered += 1
                total_peaks_shuf += len(peaks_s)
        coverage = real_covered / n_tnps
        PPV = peaks_that_are_correct / max(1, total_peaks_real)
        exact = real_exact / n_tnps
        shuf_cov = shuf_covered / (n_tnps * 20)
        print(f"  N_nc={N_nc}: coverage={coverage:.3f}   PPV={PPV:.3f}   exact-hit={exact:.3f}   shuf_cov={shuf_cov:.4f}   ratio={coverage/max(1e-9, shuf_cov):.2f}×")
    print(f"\n  Reading: PPV drop with N_nc growth = pure multiple-testing cost (no same-region constraint).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tnps_5, tnp_nc = load_tnps(args.durrant_cog, args.durrant_gold)
    print(f"  n Tnps with >=5 sites = {len(tnps_5)}")

    x1_results = x1_scan(tnps_5, tnp_nc)

    # Pick the best variant by PPV (with coverage tiebreak)
    best = max(x1_results, key=lambda r: (r["PPV"], r["coverage"]))
    print(f"\n  BEST X1 variant: {best['name']}, τ={best['tau']}   PPV={best['PPV']:.3f}, coverage={best['coverage']:.3f}")
    rng = np.random.default_rng(1)
    if best["name"] == "fixed_L11":
        maker = lambda nc, fl: _site_hits_fixed(nc, fl, 11, 8)
    else:
        maker = lambda nc, fl: _site_hits_minE(nc, fl, (9,10,11,12), 4.0)
    thresh = 5.0 if best["tau"] == 0 else 4.5
    x2a_synthetic_N_nc(tnps_5, tnp_nc, maker, best["name"], best["tau"], thresh, rng)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"X1": x1_results, "X1_best": {k: v for k, v in best.items() if not callable(v)}}, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
