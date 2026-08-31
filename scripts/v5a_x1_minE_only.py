"""X1' v3b — min_E variant only, vectorized E-lookup table.

Fills in the min_E_9_12 variants that timed out. Reuses X1' v3's fixed_L11
results (already reported) so we can compare.

Optimization: E-value precomputed as small lookup table (L ∈ {9,10,11,12}, m ∈ {0..16}).
Avoids the 14M binom.cdf calls that stalled X1' v3.
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


def _apply_kernel_max(hits_lists, nc_len_pos, tau):
    S = np.zeros(nc_len_pos, dtype=np.float64)
    if tau <= 0:
        for h in hits_lists:
            per_site = np.zeros(nc_len_pos)
            for pos in h:
                if 0 <= pos < nc_len_pos: per_site[pos] = 1.0
            S += per_site
        return S
    radius = int(np.ceil(3 * tau))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / tau) ** 2)
    for h in hits_lists:
        per_site = np.zeros(nc_len_pos)
        for pos in h:
            lo = max(0, pos - radius); hi = min(nc_len_pos, pos + radius + 1)
            k_lo = lo - (pos - radius); k_hi = k_lo + (hi - lo)
            np.maximum(per_site[lo:hi], kernel[k_lo:k_hi], out=per_site[lo:hi])
        S += per_site
    return S


def _find_peaks(S, thresh, min_dist=5):
    peaks = []
    L = len(S)
    for i in range(L):
        if S[i] < thresh: continue
        is_max = True
        for j in range(max(0, i - min_dist), min(L, i + min_dist + 1)):
            if j != i and S[j] > S[i]:
                is_max = False; break
        if is_max: peaks.append(i)
    return peaks


def _iou(p, L_win, gold_nc, gold_L, thresh=0.5):
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
    n_tnps = len(tnps_5)
    print(f"  n Tnps >= 5 sites = {n_tnps}", flush=True)

    tnp_ids = sorted(tnps_5.keys())
    all_flanks = []
    for t in tnp_ids:
        for s in tnps_5[t][:5]:
            all_flanks.append((t, s["flank"]))
    n_flanks = len(all_flanks)
    own_flanks = {t: [i for i, (tt, _) in enumerate(all_flanks) if tt == t] for t in tnp_ids}

    # E-value lookup table: E_TABLE[L][m] for L in {9..12}, m in {0..16}
    NC_LEN = 177; FLANK_LEN = 120     # Durrant constants
    E_TABLE = {}
    for L in (9, 10, 11, 12):
        Nw = max(1, (NC_LEN - L + 1) * (FLANK_LEN - L + 1))
        E_TABLE[L] = {m: Nw * float(1.0 - binom.cdf(m - 1, L, 0.25)) for m in range(0, 17)}
    print(f"  E table built ({len(E_TABLE)} × 17 entries)", flush=True)

    print("[precompute] (nc, flank, L) match tables ...", flush=True)
    match_table = {}
    for t_idx, t in enumerate(tnp_ids):
        nc = tnp_nc[t]
        for f_idx, (_, fl) in enumerate(all_flanks):
            for L in (9, 10, 11, 12):
                match_table[(t_idx, f_idx, L)] = _fwd_win_max(nc, fl, L)
        if (t_idx + 1) % 20 == 0: print(f"  {t_idx+1}/{n_tnps}", flush=True)

    # Precompute hits_cache using E lookup — vectorized min-E per (t, f)
    print("[precompute] min_E hits per (tnp, flank) ...", flush=True)
    hits_cache = {}
    for t_idx in range(n_tnps):
        for f_idx in range(n_flanks):
            # Vectorized: for each nc position, get min E across L in {9,10,11,12}
            best_E = np.full(200, np.inf, dtype=np.float64)
            for L in (9, 10, 11, 12):
                arr = match_table[(t_idx, f_idx, L)]
                # arr[pos] = m_at_pos_L; E = E_TABLE[L][m] (m capped at 16)
                lengths = min(len(arr), 200)
                Es = np.asarray([E_TABLE[L][min(16, int(arr[pos]))] if arr[pos] > 0 else np.inf
                                    for pos in range(lengths)], dtype=np.float64)
                best_E[:lengths] = np.minimum(best_E[:lengths], Es)
            hits_cache[(t_idx, f_idx)] = {int(p) for p in np.where(best_E < 4.0)[0]}
        if (t_idx + 1) % 20 == 0: print(f"  {t_idx+1}/{n_tnps} tnps done", flush=True)

    # Common shuffled draws
    rng = np.random.default_rng(0)
    n_perm = 200
    shuf_flank_idx = {t: [rng.choice(n_flanks, size=5, replace=False) for _ in range(n_perm)]
                          for t in tnp_ids}

    print(f"\n=== X1' v3b :: min_E_9_12 variants (n_perm={n_perm}) ===", flush=True)
    print(f"  {'variant':<24} {'coverage':>9} {'PPV':>6} {'exact':>7} {'≤5nt':>7} {'shuf_cov':>9} {'ratio':>7}", flush=True)
    results = []
    for tau in (0, 1, 2, 3, 5):
        thresh = 5.0 if tau == 0 else 4.5
        covered = 0; total_peaks = 0; peaks_correct = 0; exact = 0; w5 = 0
        for t_idx, t in enumerate(tnp_ids):
            sites = tnps_5[t]
            nc_len_pos = len(tnp_nc[t]) - 11 + 1
            hits_lists = [hits_cache[(t_idx, f_idx)] for f_idx in own_flanks[t]]
            S = _apply_kernel_max(hits_lists, nc_len_pos, tau)
            peaks = _find_peaks(S, thresh)
            total_peaks += len(peaks)
            if peaks:
                covered += 1
                gold_nc = sites[0]["gold_nc"]; gold_L = sites[0]["gold_L"]
                for p in peaks:
                    if _iou(p, 11, gold_nc, gold_L): peaks_correct += 1
                best_peak = min(peaks, key=lambda p: abs(p - gold_nc))
                d = abs(best_peak - gold_nc)
                if d <= 1: exact += 1
                if d <= 5: w5 += 1
        coverage = covered / n_tnps
        PPV = peaks_correct / max(1, total_peaks)
        exact_r = exact / n_tnps
        w5_r = w5 / n_tnps
        shuf_covered = 0
        for t_idx, t in enumerate(tnp_ids):
            nc_len_pos = len(tnp_nc[t]) - 11 + 1
            for draw in shuf_flank_idx[t]:
                hits_lists = [hits_cache[(t_idx, int(f_idx))] for f_idx in draw]
                S = _apply_kernel_max(hits_lists, nc_len_pos, tau)
                if _find_peaks(S, thresh): shuf_covered += 1
        shuf_cov = shuf_covered / (n_tnps * n_perm)
        ratio = coverage / max(1e-9, shuf_cov)
        label = f"min_E_9_12, τ={tau}"
        print(f"  {label:<24} {coverage:>9.3f} {PPV:>6.3f} {exact_r:>7.3f} {w5_r:>7.3f} {shuf_cov:>9.4f} {ratio:>7.2f}×", flush=True)
        results.append({"variant": "min_E_9_12", "tau": tau, "coverage": coverage, "PPV": PPV,
                          "exact_hit": exact_r, "within_5nt": w5_r, "shuf_coverage": shuf_cov, "ratio": ratio})

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"X1_minE_only": results}, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
