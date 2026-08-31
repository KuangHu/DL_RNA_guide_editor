"""X1' v3 — semantically correct + precomputed tables + common random numbers.

Fixes from the previous version:
  1. Kernel semantic: S_soft(i) = Σ_k max_j K(|pos_kj − i|). Per-site max is
     INSIDE the site sum. Previously additive within a site → sites with
     multiple nearby hits over-contributed.
  2. Threshold-matched variants:
     - fixed_L11_m8:   fixed L=11, m ≥ 8  (E ≈ 21.8)
     - fixed_L11_m9:   fixed L=11, m ≥ 9  (E ≈ 2.3)  ← E-matched to min_E
     - min_E_9_12:     min-E over L ∈ {9,10,11,12}, E < 4
     Comparing fixed_L11_m9 vs min_E isolates L-marginalization from strictness.
  3. n_perm = 200 shuffled draws via COMMON RANDOM NUMBERS: same 5-flank
     tuples used across all variants (per Tnp). Reduces variance in
     variant-vs-variant comparisons dramatically.
  4. τ grid {0, 1, 2, 3, 5} — populates the informative [1, 4] band per W7.
  5. Precomputed (nc, flank, L) match arrays → all variants are lookups.

Sanity gate: fixed_L11_m8 at τ=0 MUST equal the base Channel A coverage 0.338.
Any higher value would indicate the kernel bug is still present.
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


def _apply_kernel_max(hits_lists: list[set[int]], nc_len_pos: int, tau: float) -> np.ndarray:
    """CORRECT: S_soft(i) = Σ_k max_j K(|pos_kj − i|). Per-site max INSIDE sum."""
    S = np.zeros(nc_len_pos, dtype=np.float64)
    if tau <= 0:
        for h in hits_lists:
            per_site = np.zeros(nc_len_pos, dtype=np.float64)
            for pos in h:
                if 0 <= pos < nc_len_pos:
                    per_site[pos] = 1.0
            S += per_site
        return S
    radius = int(np.ceil(3 * tau))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (offsets / tau) ** 2)
    for h in hits_lists:
        per_site = np.zeros(nc_len_pos, dtype=np.float64)
        for pos in h:
            lo = max(0, pos - radius); hi = min(nc_len_pos, pos + radius + 1)
            k_lo = lo - (pos - radius); k_hi = k_lo + (hi - lo)
            # PER-SITE MAX, not add
            np.maximum(per_site[lo:hi], kernel[k_lo:k_hi], out=per_site[lo:hi])
        S += per_site      # only NOW sum across sites
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
    n_tnps = len(tnps_5)
    print(f"  n Tnps with >=5 sites = {n_tnps}", flush=True)

    # ---- Precompute (nc_id, flank_id, L) → per_pos_max_m table ----
    print("[precompute] building (nc, flank, L) match tables...", flush=True)
    tnp_ids = sorted(tnps_5.keys())
    tnp_to_idx = {t: i for i, t in enumerate(tnp_ids)}
    # Flatten all flanks: (tnp_id, flank_str, site_j_within_tnp)
    all_flanks = []
    for t in tnp_ids:
        for j, s in enumerate(tnps_5[t][:5]):
            all_flanks.append((t, s["flank"]))
    n_flanks = len(all_flanks)
    L_range = (9, 10, 11, 12)
    match_table = {}    # (tnp_idx, flank_idx, L) -> per_pos array
    for t_idx, t in enumerate(tnp_ids):
        nc = tnp_nc[t]
        for f_idx, (_, fl) in enumerate(all_flanks):
            for L in L_range:
                match_table[(t_idx, f_idx, L)] = _fwd_win_max(nc, fl, L)
        if (t_idx + 1) % 10 == 0:
            print(f"  [precompute] {t_idx + 1}/{n_tnps} tnps done", flush=True)

    # Own-flank indices per Tnp (for the "real" evaluation)
    own_flanks = {t: [i for i, (tt, _) in enumerate(all_flanks) if tt == t] for t in tnp_ids}

    # ---- Common random shuffled draws ----
    n_perm = 200
    print(f"[perms] drawing {n_perm} common shuffled tuples per Tnp...", flush=True)
    rng = np.random.default_rng(0)
    shuf_flank_idx = {}   # tnp -> list of 5-tuples of flank indices
    for t in tnp_ids:
        draws = []
        for _ in range(n_perm):
            draws.append(rng.choice(n_flanks, size=5, replace=False))
        shuf_flank_idx[t] = draws

    def _hits_from_table(t_idx: int, f_idx: int, mode: str) -> set[int]:
        nc_len = len(tnp_nc[tnp_ids[t_idx]])
        if mode == "fixed_L11_m8":
            arr = match_table[(t_idx, f_idx, 11)]
            return {int(p) for p in np.where(arr >= 8)[0]}
        elif mode == "fixed_L11_m9":
            arr = match_table[(t_idx, f_idx, 11)]
            return {int(p) for p in np.where(arr >= 9)[0]}
        elif mode == "min_E_9_12":
            per_pos_min_E = {}
            flank_len = 120    # constant on Durrant
            for L in L_range:
                arr = match_table[(t_idx, f_idx, L)]
                for pos in range(len(arr)):
                    m_pos = int(arr[pos])
                    if m_pos <= 0: continue
                    E = _E(m_pos, L, nc_len, flank_len)
                    if pos not in per_pos_min_E or E < per_pos_min_E[pos]:
                        per_pos_min_E[pos] = E
            return {p for p, e in per_pos_min_E.items() if e < 4.0}
        else:
            raise ValueError(mode)

    # ---- Sweep ----
    taus = (0, 1, 2, 3, 5)
    print(f"\n=== X1' v3 :: window scan (kernel semantics FIXED, n_perm={n_perm}) ===", flush=True)
    print(f"  {'variant':<26} {'coverage':>9} {'PPV':>6} {'exact':>7} {'≤5nt':>7} {'shuf_cov':>9} {'ratio':>7}", flush=True)

    results = []
    for mode in ("fixed_L11_m8", "fixed_L11_m9", "min_E_9_12"):
        # Precompute hits sets per (tnp, flank) for THIS mode
        hits_cache = {}
        for t_idx, t in enumerate(tnp_ids):
            for f_idx in range(n_flanks):
                hits_cache[(t_idx, f_idx)] = _hits_from_table(t_idx, f_idx, mode)
        for tau in taus:
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
                        if _iou_overlap(p, 11, gold_nc, gold_L): peaks_correct += 1
                    best_peak = min(peaks, key=lambda p: abs(p - gold_nc))
                    d = abs(best_peak - gold_nc)
                    if d <= 1: exact += 1
                    if d <= 5: w5 += 1
            coverage = covered / n_tnps
            PPV = peaks_correct / max(1, total_peaks)
            exact_r = exact / n_tnps
            w5_r = w5 / n_tnps
            # shuffled with common random numbers
            shuf_covered = 0
            for t_idx, t in enumerate(tnp_ids):
                nc_len_pos = len(tnp_nc[t]) - 11 + 1
                for draw in shuf_flank_idx[t]:
                    hits_lists = [hits_cache[(t_idx, int(f_idx))] for f_idx in draw]
                    S = _apply_kernel_max(hits_lists, nc_len_pos, tau)
                    peaks = _find_peaks(S, thresh)
                    if peaks: shuf_covered += 1
            shuf_cov = shuf_covered / (n_tnps * n_perm)
            a_m = Metric(f"real:{mode}:tau={tau}", coverage,
                            MetricCondition(match_rule="strict_WC",
                                               null_model="real_flanks_target_intact",
                                               coordinate_system="absolute_nc",
                                               targeting_intact=True, tie_break="soft_kernel_max",
                                               denominator="tnp"))
            s_m = Metric(f"shuf:{mode}:tau={tau}", shuf_cov,
                            MetricCondition(match_rule="strict_WC",
                                               null_model="shuffled_intra_family_flanks",
                                               coordinate_system="absolute_nc",
                                               targeting_intact=True,
                                               tie_break="soft_kernel_max", denominator="tnp"))
            try:
                ratio = safe_ratio(a_m, s_m, varying_dim="null_model")
            except ValueError as e:
                print(f"    [safe_ratio violation] {e}", flush=True)
                ratio = float("nan")
            label = f"{mode}, τ={tau}"
            print(f"  {label:<26} {coverage:>9.3f} {PPV:>6.3f} {exact_r:>7.3f} {w5_r:>7.3f} {shuf_cov:>9.4f} {ratio:>7.2f}×",
                  flush=True)
            results.append({"variant": mode, "tau": tau, "coverage": coverage, "PPV": PPV,
                              "exact_hit": exact_r, "within_5nt": w5_r,
                              "shuf_coverage": shuf_cov, "ratio": ratio})

    # Sanity anchor
    baseline = next(r for r in results if r["variant"] == "fixed_L11_m8" and r["tau"] == 0)
    print(f"\n  SANITY: fixed_L11_m8 τ=0 coverage = {baseline['coverage']:.4f}   (expected ~0.338)", flush=True)
    if baseline["coverage"] > 0.35:
        print(f"  WARNING: coverage {baseline['coverage']:.3f} is above baseline 0.338 — kernel semantic may still be wrong.", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"X1": results}, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
