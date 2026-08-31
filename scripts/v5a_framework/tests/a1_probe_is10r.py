"""A1 probe — IS10-R × 30 Tnps × 65 Durrant ncs, three detection modes.

Question: does Channel A produce non-trivial FP on IS10-R flanks scored
against Durrant nc substrate, and can we discriminate TSD-driven FP from
non-TSD FP?

Three excl modes, all on the same 5-flank / 1-nc scoring:
  S_all           : excl_w=0 (no exclusion)                     — full FP rate
  S_outside_TSD   : excl_w=9 from FLANK START (TSD-region)      — post-partition FP rate
  S_shift_matched : excl_w=9 from FLANK END   (control)         — same search-space
                                                                  reduction, no TSD

Comparison:
  S_all vs S_outside_TSD  : partition efficacy (some might be TSD-driven)
  S_shift_matched vs S_outside_TSD : if S_shift_matched >= S_outside_TSD,
                                      partition adds no TSD-specific removal
                                      beyond search-space reduction

Also:
  - detection matrix 30x65 -> row and column marginals
  - per-Tnp fp_hazard predicted from the 5-flank consensus vs each nc,
    checked against actual FP -> if sticky Tnps have elevated hazard,
    fp_hazard is a valid predictor
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.alignment import dot_plot, windowed_matches
from v5a_framework.e_match_table import load_e
from v5a_framework.flank_coherence import (
    _positional_consensus, _poisson_binom_tail, UNIFORM_BG, _BASE_INDEX,
)
from v5a_framework.variant import _apply_kernel_max, _find_peaks


IS10R_SITES = "/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/formatted/real_IS10-R_sites.jsonl"
E_DIAGONAL = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5a_framework_cache/e_durrant_diagonal"

N_TNP_A1 = 30
CAP_FLANKS = 10
FLANK_GROUP_SIZE = 5
SEED = 0
L_FIX = 11
M_THRESH = 8
S_THRESH = 5
TSD_WIDTH = 9

DURRANT_SHUFFLED_NULL = 0.0226


def load_is10r_paired_downstream_flanks(sites_path, n_tnp, cap_flanks, seed):
    tnp_ins = defaultdict(dict)
    with open(sites_path) as f:
        for line in f:
            d = json.loads(line)
            tnp = d.get("transposase_id")
            m = d.get("generator_metadata", {})
            side = m.get("flank_side")
            key = (m.get("insertion_start"), m.get("sample_id"))
            fl = d.get("inputs", {}).get("flank")
            if tnp is None or side not in ("upstream", "downstream") or not fl:
                continue
            tnp_ins[tnp].setdefault(key, {})[side] = fl
    tnp_dn = {}
    for tnp, m in tnp_ins.items():
        both = [p["downstream"] for p in m.values() if "upstream" in p and "downstream" in p]
        if len(both) >= 5:
            tnp_dn[tnp] = both
    sorted_ids = sorted(tnp_dn.keys())
    rng = np.random.default_rng(seed)
    picked = list(rng.choice(sorted_ids, size=min(n_tnp, len(sorted_ids)), replace=False))
    out = {}
    for tnp in picked:
        pool = tnp_dn[tnp]
        r = np.random.default_rng(hash((seed, tnp)) & 0xFFFFFFFF)
        idx = r.choice(len(pool), size=min(cap_flanks, len(pool)), replace=False)
        out[tnp] = [pool[int(i)] for i in idx]
    return out


def _m_max_three_from_win(win: np.ndarray, tsd_w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """From a (nc_pos, flank_starts) windowed-match matrix, return the three
    m_max variants: full, TSD-excluded from START, shift-matched (excluded
    from END).
    """
    if win.size == 0:
        z = np.zeros(0, dtype=np.int32)
        return z, z, z
    n_starts = win.shape[1]
    m_all = win.max(axis=1)
    m_out = win[:, tsd_w:].max(axis=1) if tsd_w < n_starts else np.zeros_like(m_all)
    m_sft = win[:, : n_starts - tsd_w].max(axis=1) if n_starts - tsd_w > 0 else np.zeros_like(m_all)
    return m_all, m_out, m_sft


def m_max_all_three_per_orient(nc: str, flank: str, L: int, tsd_w: int
                                ) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Both orientations, all three excl modes. Returns dict keyed by orient."""
    fwd, rc = dot_plot(nc, flank)
    win_fwd = windowed_matches(fwd, L)
    win_rc  = windowed_matches(rc,  L)
    return {
        "fwd": _m_max_three_from_win(win_fwd, tsd_w),
        "rc":  _m_max_three_from_win(win_rc,  tsd_w),
    }


def detect_at_thresh(m_stacked_5sites: np.ndarray, nc_len_pos: int) -> bool:
    """Given (5, nc_len_pos) int array of m per site, run Channel A at
    m>=M_THRESH, S=S_THRESH, tau=0. Return True iff any peak survives.
    """
    hits_lists = [set(int(p) for p in np.where(m_stacked_5sites[s] >= M_THRESH)[0])
                   for s in range(m_stacked_5sites.shape[0])]
    S = _apply_kernel_max(hits_lists, nc_len_pos, tau=0.0)
    peaks = _find_peaks(S, thresh=float(S_THRESH), min_dist=5)
    return len(peaks) > 0


def compute_family_bg(all_flanks: list[str]) -> np.ndarray:
    counts = np.zeros(4)
    n = 0
    for fl in all_flanks:
        for b in fl:
            i = _BASE_INDEX.get(b.upper())
            if i is not None:
                counts[i] += 1; n += 1
    return counts / max(1, n) if n > 0 else UNIFORM_BG.copy()


def per_position_freqs(flanks: list[str], window_len: int) -> np.ndarray:
    """Positional freq matrix over the first window_len nt of each flank."""
    windows = [fl[:window_len] for fl in flanks]
    _, _, freqs = _positional_consensus(windows, window_len, UNIFORM_BG)
    return freqs


def fp_hazard_joint(consensus_freqs: np.ndarray, nc: str, threshold: int,
                     S_threshold: int, L_eff: int) -> float:
    """Joint FP hazard = sum over nc windows of g(w) ** S_threshold, where
    g(w) is the Poisson-binomial tail at threshold given per-position match
    probs p_i(w) = consensus_freqs[i, nc_base(w+i)]."""
    nc_int = np.array([_BASE_INDEX.get(b.upper(), -1) for b in nc], dtype=np.int32)
    total = 0.0
    for w0 in range(len(nc) - L_eff + 1):
        base_ids = nc_int[w0:w0 + L_eff]
        p_pos = np.zeros(L_eff)
        valid = base_ids >= 0
        p_pos[valid] = consensus_freqs[np.arange(L_eff)[valid], base_ids[valid]]
        g_w = _poisson_binom_tail(p_pos, threshold)
        total += g_w ** S_threshold
    return total


def main() -> int:
    emt = load_e(E_DIAGONAL)
    print(f"[A1] Durrant nc substrates loaded: {len(emt.nc_source_tnp_ids)}", flush=True)
    is10r_flanks = load_is10r_paired_downstream_flanks(
        IS10R_SITES, N_TNP_A1, CAP_FLANKS, SEED)
    tnp_ids = list(is10r_flanks.keys())
    print(f"[A1] IS10-R Tnps loaded: {len(tnp_ids)}", flush=True)

    nc_ids = emt.nc_source_tnp_ids
    n_tnp = len(tnp_ids); n_nc = len(nc_ids)

    # Compute per-family bg for fp_hazard (from ALL 10 flanks per Tnp)
    all_is10r_flanks_flat = [fl for lst in is10r_flanks.values() for fl in lst]
    fam_bg = compute_family_bg(all_is10r_flanks_flat)
    print(f"[A1] IS10-R composition A/C/G/T = {fam_bg.round(3).tolist()}", flush=True)

    # Detection matrices for the three modes
    det_all = np.zeros((n_tnp, n_nc), dtype=bool)
    det_out = np.zeros((n_tnp, n_nc), dtype=bool)
    det_shft = np.zeros((n_tnp, n_nc), dtype=bool)

    # Per-Tnp: predicted fp_hazard aggregated across the 65 ncs
    hazard_per_tnp_per_nc = np.zeros((n_tnp, n_nc))

    t0 = time.time()
    for ti, tnp in enumerate(tnp_ids):
        flanks5 = is10r_flanks[tnp][:FLANK_GROUP_SIZE]
        # positional freqs over first 15 nt of each flank -> consensus window for fp_hazard
        cons_freqs = per_position_freqs(flanks5, window_len=15)
        # take 11-nt slice centered around max-info position (or default first 11 for simplicity)
        cons_L = cons_freqs[:11]

        for ni, nc_tnp in enumerate(nc_ids):
            nc = emt.nc_source_ncs[nc_tnp]
            nc_len_pos = len(nc) - L_FIX + 1
            # Accumulate per-orient detection under each excl mode; final
            # detection = OR over orients (Channel A convention).
            det_all_any = False; det_out_any = False; det_shft_any = False
            for orient, (m_all_arr, m_out_arr, m_shft_arr) in \
                    (lambda: [(o, m_max_all_three_per_orient(nc, flanks5[0], L_FIX, TSD_WIDTH)[o])
                                for o in ("fwd", "rc")])():
                pass  # trick to establish keys; will actually loop below
            per_orient_mats: dict[str, list[list[np.ndarray]]] = {"fwd": [[], [], []], "rc": [[], [], []]}
            for fl in flanks5:
                m_dict = m_max_all_three_per_orient(nc, fl, L_FIX, TSD_WIDTH)
                for orient in ("fwd", "rc"):
                    m_all, m_out, m_shft = m_dict[orient]
                    per_orient_mats[orient][0].append(m_all)
                    per_orient_mats[orient][1].append(m_out)
                    per_orient_mats[orient][2].append(m_shft)
            for orient in ("fwd", "rc"):
                m_all_5 = np.zeros((5, nc_len_pos), dtype=np.int32)
                m_out_5 = np.zeros((5, nc_len_pos), dtype=np.int32)
                m_shft_5 = np.zeros((5, nc_len_pos), dtype=np.int32)
                for si, arr in enumerate(per_orient_mats[orient][0]):
                    nn = min(nc_len_pos, len(arr)); m_all_5[si, :nn] = arr[:nn]
                for si, arr in enumerate(per_orient_mats[orient][1]):
                    nn = min(nc_len_pos, len(arr)); m_out_5[si, :nn] = arr[:nn]
                for si, arr in enumerate(per_orient_mats[orient][2]):
                    nn = min(nc_len_pos, len(arr)); m_shft_5[si, :nn] = arr[:nn]
                if detect_at_thresh(m_all_5, nc_len_pos):  det_all_any  = True
                if detect_at_thresh(m_out_5, nc_len_pos):  det_out_any  = True
                if detect_at_thresh(m_shft_5, nc_len_pos): det_shft_any = True
            det_all[ti, ni]  = det_all_any
            det_out[ti, ni]  = det_out_any
            det_shft[ti, ni] = det_shft_any
            hazard_per_tnp_per_nc[ti, ni] = fp_hazard_joint(
                cons_L, nc, threshold=M_THRESH, S_threshold=S_THRESH, L_eff=L_FIX)
    elapsed = time.time() - t0

    def _report(name, mat):
        n_fired = int(mat.sum())
        rate = n_fired / mat.size
        row_sums = mat.sum(axis=1)
        col_sums = mat.sum(axis=0)
        print(f"  {name:<20s}: total_fires={n_fired:5d}/{mat.size} = {rate:.4f}"
              f"    per-Tnp: mean={row_sums.mean():5.2f}, med={float(np.median(row_sums)):5.2f}, max={row_sums.max():3d}"
              f"    per-nc: mean={col_sums.mean():5.2f}, med={float(np.median(col_sums)):5.2f}, max={col_sums.max():3d}")
        return row_sums, col_sums, rate

    print()
    print("=== A1 detection matrices ===")
    r_all,  c_all,  rate_all  = _report("S_all (excl_w=0)",   det_all)
    r_out,  c_out,  rate_out  = _report("S_outside_TSD (w=9)", det_out)
    r_shft, c_shft, rate_shft = _report("S_shift_matched (end-9)", det_shft)

    print()
    print(f"  baseline durrant_shuffled null (historical) = {DURRANT_SHUFFLED_NULL:.4f}")
    print(f"  ratio S_all / baseline           = {rate_all/DURRANT_SHUFFLED_NULL:.2f}x")
    print(f"  partition removed FP: S_all - S_outside_TSD = {rate_all - rate_out:.4f}"
          f"  ({100*(rate_all - rate_out)/max(1e-9,rate_all):.1f}% of S_all)")
    print(f"  search-space effect (shift):     S_all - S_shift_matched = {rate_all - rate_shft:.4f}")
    print(f"  net TSD-specific effect:         (S_all - S_outside_TSD) - (S_all - S_shift_matched)")
    print(f"                                    = {(rate_all - rate_out) - (rate_all - rate_shft):.4f}"
          f" = {rate_shft - rate_out:.4f}")

    print()
    print("=== Sticky Tnp identification (top 5 by S_all row sum) ===")
    order = np.argsort(-r_all)
    print(f"  {'rank':>4} {'tnp':<38} {'S_all fires':>11} {'S_out fires':>11} {'S_shft fires':>12} {'mean_hazard':>12}")
    for rank_i in range(min(5, len(order))):
        ti = int(order[rank_i])
        mean_h = hazard_per_tnp_per_nc[ti].mean()
        print(f"  {rank_i+1:>4} {tnp_ids[ti][:38]:<38} {int(r_all[ti]):>11} {int(r_out[ti]):>11} {int(r_shft[ti]):>12} {mean_h:>12.6f}")

    print()
    print("=== Susceptible-nc identification (top 5 by S_all col sum) ===")
    order_nc = np.argsort(-c_all)
    print(f"  {'rank':>4} {'nc_tnp':<50} {'fires':>7}")
    for rank_i in range(min(5, len(order_nc))):
        ni = int(order_nc[rank_i])
        print(f"  {rank_i+1:>4} {nc_ids[ni][:50]:<50} {int(c_all[ni]):>7}")

    print()
    print("=== fp_hazard predictor validation ===")
    mean_h_per_tnp = hazard_per_tnp_per_nc.mean(axis=1)
    if n_tnp > 3:
        rho, p = spearmanr(mean_h_per_tnp, r_all)
        print(f"  Spearman rho (mean fp_hazard vs S_all row sum): {rho:.3f}   p={p:.4g}")
        rho2, p2 = spearmanr(mean_h_per_tnp, r_out)
        print(f"  Spearman rho (mean fp_hazard vs S_outside_TSD): {rho2:.3f}   p={p2:.4g}")

    print()
    print(f"[A1] wall time = {elapsed:.1f} s over 3 excl modes x {n_tnp} Tnps x {n_nc} ncs x 5 flanks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
