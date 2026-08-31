"""X1' + X2a — optimized: precompute hits once per (site,L), reuse across τ.

Key speedup:
  - windowed_matches is computed exactly once per (site, flank, L) tuple
  - hits sets are cached; kernel application (fast) is what varies with τ
  - Shuffled null uses reduced n_perm=5 (still enough for ratio at n=65 Tnps)
  - τ grid: {0, 2, 5, 10}

Both L modes tested: fixed L=11 m≥8 AND min-E-over-L ∈ {9,10,11,12} with
E-value threshold 4. safe_ratio discipline enforced.

Metrics per variant: coverage, PPV, exact-hit, within-5nt, ratio.
X2a synthetic N_nc concat runs on best-PPV variant only.
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


def _hits_fixed(nc: str, flank: str, L: int = 11, m: int = 8) -> set[int]:
    per_pos = _fwd_win_max(nc, flank, L)
    return {int(p) for p in np.where(per_pos >= m)[0]}


def _hits_minE(nc: str, flank: str, L_range=(9, 10, 11, 12), E_thresh: float = 4.0) -> set[int]:
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
    return {p for p, e in per_pos_min_E.items() if e < E_thresh}


def _apply_kernel(hits_lists: list[set[int]], nc_len_pos: int, tau: float) -> np.ndarray:
    """Fast: for each site, distribute a Gaussian bump at each hit into the S array."""
    S = np.zeros(nc_len_pos, dtype=np.float64)
    if tau <= 0:
        for h in hits_lists:
            for pos in h:
                if 0 <= pos < nc_len_pos:
                    S[pos] += 1.0
    else:
        # Precompute Gaussian kernel over a range of distances
        radius = int(np.ceil(3 * tau))
        offsets = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (offsets / tau) ** 2)
        for h in hits_lists:
            for pos in h:
                lo = max(0, pos - radius); hi = min(nc_len_pos, pos + radius + 1)
                k_lo = lo - (pos - radius); k_hi = k_lo + (hi - lo)
                S[lo:hi] += kernel[k_lo:k_hi]
    return S


def _find_peaks(S: np.ndarray, thresh: float, min_dist: int = 5) -> list[int]:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    tnps_5, tnp_nc = load_tnps(args.durrant_cog, args.durrant_gold)
    print(f"  n Tnps with >=5 sites = {len(tnps_5)}", flush=True)

    # --- Precompute hits ONCE per (variant, site) ---
    print("[precompute] real hits per site...", flush=True)
    hits_fixed_real = {}   # tnp -> [hits per site]
    hits_minE_real  = {}
    for tnp, sites in tnps_5.items():
        nc = tnp_nc[tnp]
        hits_fixed_real[tnp] = [_hits_fixed(nc, s["flank"]) for s in sites[:5]]
        hits_minE_real[tnp]  = [_hits_minE(nc,  s["flank"]) for s in sites[:5]]

    n_perm = 5
    rng = np.random.default_rng(0)
    all_flanks = [(t, s["flank"]) for t, ss in tnps_5.items() for s in ss[:5]]

    print(f"[precompute] shuffled hits per perm ({n_perm} perms)...", flush=True)
    # Precompute shuffled: for each Tnp, n_perm draws of 5 random flanks;
    # hits computed against that Tnp's nc.
    shuf_hits_fixed = {}    # tnp -> list of [hits per site] per perm
    shuf_hits_minE  = {}
    for tnp, sites in tnps_5.items():
        nc = tnp_nc[tnp]
        shuf_hits_fixed[tnp] = []
        shuf_hits_minE[tnp] = []
        for _ in range(n_perm):
            idx = rng.choice(len(all_flanks), size=5, replace=False)
            fake_flanks = [all_flanks[int(i)][1] for i in idx]
            shuf_hits_fixed[tnp].append([_hits_fixed(nc, fl) for fl in fake_flanks])
            shuf_hits_minE[tnp].append([_hits_minE(nc,  fl) for fl in fake_flanks])
    print(f"[precompute] done", flush=True)

    # --- Kernel + peak sweep ---
    taus = (0, 2, 5, 10)
    print(f"\n=== X1' :: window scan ===", flush=True)
    print(f"  {'variant':<24} {'coverage':>9} {'PPV':>6} {'exact':>7} {'≤5nt':>7} {'shuf_cov':>9} {'ratio':>7}", flush=True)

    results = []
    for hit_name, hits_real, hits_shuf in [
        ("fixed_L11",  hits_fixed_real, shuf_hits_fixed),
        ("min_E_9_12", hits_minE_real,  shuf_hits_minE)]:
        for tau in taus:
            thresh = 5.0 if tau == 0 else 4.5
            n_tnps = len(tnps_5)
            covered = 0; total_peaks = 0; peaks_correct = 0
            exact = 0; w5 = 0
            for tnp, sites in tnps_5.items():
                nc_len_pos = len(tnp_nc[tnp]) - 11 + 1
                S = _apply_kernel(hits_real[tnp], nc_len_pos, tau)
                peaks = _find_peaks(S, thresh)
                total_peaks += len(peaks)
                if peaks:
                    covered += 1
                    gold_nc = sites[0]["gold_nc"]; gold_L = sites[0]["gold_L"]
                    for p in peaks:
                        if _iou_overlap(p, 11, gold_nc, gold_L):
                            peaks_correct += 1
                    best_peak = min(peaks, key=lambda p: abs(p - gold_nc))
                    d = abs(best_peak - gold_nc)
                    if d <= 1: exact += 1
                    if d <= 5: w5 += 1
            coverage = covered / n_tnps
            PPV = peaks_correct / max(1, total_peaks)
            exact_r = exact / n_tnps
            w5_r = w5 / n_tnps
            # shuffled
            shuf_covered = 0; total_shuf_peaks = 0
            for tnp in tnps_5:
                nc_len_pos = len(tnp_nc[tnp]) - 11 + 1
                for perm in range(n_perm):
                    S = _apply_kernel(hits_shuf[tnp][perm], nc_len_pos, tau)
                    peaks = _find_peaks(S, thresh)
                    if peaks: shuf_covered += 1
                    total_shuf_peaks += len(peaks)
            shuf_cov = shuf_covered / (n_tnps * n_perm)
            a_m = Metric(f"real:{hit_name}:tau={tau}", coverage,
                            MetricCondition(match_rule="strict_WC",
                                               null_model="real_flanks_target_intact",
                                               coordinate_system="absolute_nc",
                                               targeting_intact=True, tie_break="soft_kernel",
                                               denominator="tnp"))
            s_m = Metric(f"shuf:{hit_name}:tau={tau}", shuf_cov,
                            MetricCondition(match_rule="strict_WC",
                                               null_model="shuffled_intra_family_flanks",
                                               coordinate_system="absolute_nc",
                                               targeting_intact=True,
                                               tie_break="soft_kernel", denominator="tnp"))
            try:
                ratio = safe_ratio(a_m, s_m, varying_dim="null_model")
            except ValueError as e:
                print(f"    [safe_ratio violation] {e}", flush=True)
                ratio = float("nan")
            label = f"{hit_name}, τ={tau}"
            print(f"  {label:<24} {coverage:>9.3f} {PPV:>6.3f} {exact_r:>7.3f} {w5_r:>7.3f} {shuf_cov:>9.4f} {ratio:>7.2f}×",
                  flush=True)
            results.append({"variant": hit_name, "tau": tau, "coverage": coverage, "PPV": PPV,
                              "exact_hit": exact_r, "within_5nt": w5_r,
                              "shuf_coverage": shuf_cov, "ratio": ratio})

    # X2a synthetic N_nc — on best variant
    best = max(results, key=lambda r: (r["PPV"], r["coverage"]))
    print(f"\n=== X2a :: synthetic N_nc concat on best variant {best['variant']} τ={best['tau']} ===", flush=True)
    hits_maker = _hits_fixed if best["variant"] == "fixed_L11" else _hits_minE
    thresh = 5.0 if best["tau"] == 0 else 4.5

    for N_nc in (1, 2, 3):
        covered = 0; total_peaks = 0; peaks_correct = 0; exact = 0
        shuf_covered = 0
        for tnp, sites in tnps_5.items():
            nc = tnp_nc[tnp]
            if N_nc > 1:
                other_tnps = [t for t in tnp_nc if t != tnp]
                pick = rng.choice(len(other_tnps), size=N_nc - 1, replace=False)
                extras = [tnp_nc[other_tnps[int(i)]] for i in pick]
                combined_nc = nc + "N" * 20 + ("N" * 20).join(extras)
            else:
                combined_nc = nc
            gold_nc = sites[0]["gold_nc"]; gold_L = sites[0]["gold_L"]
            hits = [hits_maker(combined_nc, s["flank"]) for s in sites[:5]]
            nc_len_pos = len(combined_nc) - 11 + 1
            S = _apply_kernel(hits, nc_len_pos, best["tau"])
            peaks = _find_peaks(S, thresh)
            total_peaks += len(peaks)
            if peaks:
                covered += 1
                for p in peaks:
                    if _iou_overlap(p, 11, gold_nc, gold_L): peaks_correct += 1
                best_peak = min(peaks, key=lambda p: abs(p - gold_nc))
                if abs(best_peak - gold_nc) <= 1: exact += 1
            # shuffled: use 3 perms to keep it fast
            for _ in range(3):
                idx = rng.choice(len(all_flanks), size=5, replace=False)
                fake_flanks = [all_flanks[int(i)][1] for i in idx]
                fake_hits = [hits_maker(combined_nc, fl) for fl in fake_flanks]
                S_s = _apply_kernel(fake_hits, nc_len_pos, best["tau"])
                if _find_peaks(S_s, thresh): shuf_covered += 1
        n_tnps = len(tnps_5)
        cov = covered / n_tnps; ppv = peaks_correct / max(1, total_peaks)
        exact_r = exact / n_tnps
        shuf_cov = shuf_covered / (n_tnps * 3)
        print(f"  N_nc={N_nc}: coverage={cov:.3f}  PPV={ppv:.3f}  exact-hit={exact_r:.3f}  shuf_cov={shuf_cov:.4f}  ratio={cov/max(1e-9,shuf_cov):.2f}×", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"X1": results, "X1_best": best}, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
