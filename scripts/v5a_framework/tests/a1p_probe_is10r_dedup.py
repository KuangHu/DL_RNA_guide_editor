"""A1' — A1 rerun with similarity-based dedup + continuous n_eff.

Under exact-string dedup (A1 finding), 45/312 IS10-R Tnps lose >=5-site
status because they contain flank duplicates. But real amplification
produces 97-99% similar flanks that PASS exact dedup and still trigger
trivial coherence (two 98%-similar flanks give near-identical m at every
nc position -> automatic support).

This script:
  1. Sweep single-linkage similarity thresholds (0.80 / 0.90 / 0.95) on
     IS10-R flank pools; report Tnp retention at each threshold.
  2. Compute a continuous n_eff per Tnp:
        n_eff = 5 / (1 + max(0, mean_pairwise_identity - 0.25))
     which gives ~5 for pairwise-distinct pools and drops toward the
     unique-flank count as duplicates and near-duplicates appear.
  3. Rerun the three-mode probe (S_all / S_outside_TSD / S_shift_matched)
     using ONE flank per cluster (threshold 0.90) with an ordering that
     picks 5 representatives from the 5 largest clusters.
  4. Do the same n_eff computation on Durrant's 65 Tnps to confirm the
     positive set is clean under the same code.
  5. Report FP rate stratified by n_eff bucket.
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
CHANCE_IDENTITY = 0.25
DURRANT_SHUFFLED_NULL = 0.0226


def identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / n


def cluster_single_linkage(flanks: list[str], threshold: float) -> list[list[int]]:
    """Single-linkage: any two flanks with identity >= threshold merge."""
    n = len(flanks)
    parent = list(range(n))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry
    for i in range(n):
        for j in range(i + 1, n):
            if identity(flanks[i], flanks[j]) >= threshold:
                union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def n_eff_continuous(flanks: list[str]) -> float:
    """n_eff = |P| / (1 + max(0, mean_pairwise - chance))
    On pairwise-distinct pools this returns ~|P|; on pools with duplicates
    (identity=1) it drops toward the unique-flank count.
    """
    n = len(flanks)
    if n <= 1:
        return float(n)
    total = 0.0
    n_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += identity(flanks[i], flanks[j])
            n_pairs += 1
    mean_id = total / max(1, n_pairs)
    excess = max(0.0, mean_id - CHANCE_IDENTITY)
    return n / (1.0 + excess)


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
        both = [p["downstream"] for p in m.values()
                if "upstream" in p and "downstream" in p]
        if len(both) >= 5:
            tnp_dn[tnp] = both
    return tnp_dn


def m_max_three_per_orient(nc: str, flank: str, L: int, tsd_w: int):
    fwd, rc = dot_plot(nc, flank)
    result = {}
    for orient, mat in (("fwd", fwd), ("rc", rc)):
        win = windowed_matches(mat, L)
        if win.size == 0:
            z = np.zeros(0, dtype=np.int32)
            result[orient] = (z, z, z)
            continue
        n_starts = win.shape[1]
        m_all = win.max(axis=1)
        m_out = win[:, tsd_w:].max(axis=1) if tsd_w < n_starts else np.zeros_like(m_all)
        m_sft = win[:, : n_starts - tsd_w].max(axis=1) if n_starts - tsd_w > 0 else np.zeros_like(m_all)
        result[orient] = (m_all, m_out, m_sft)
    return result


def detect_at_thresh(m5: np.ndarray, nc_len_pos: int) -> bool:
    hits = [set(int(p) for p in np.where(m5[s] >= M_THRESH)[0])
            for s in range(m5.shape[0])]
    S = _apply_kernel_max(hits, nc_len_pos, tau=0.0)
    return len(_find_peaks(S, thresh=float(S_THRESH), min_dist=5)) > 0


def main() -> int:
    print("[A1'] loading IS10-R and Durrant substrate", flush=True)
    tnp_dn = load_is10r_paired_downstream_flanks(IS10R_SITES, N_TNP_A1, CAP_FLANKS, SEED)
    emt = load_e(E_DIAGONAL)
    nc_ids = emt.nc_source_tnp_ids
    n_nc = len(nc_ids)

    # 1. Threshold sweep on the FULL IS10-R >=5 corpus (312 Tnps)
    print()
    print("=== Threshold sweep: Tnp retention at >=5 clusters ===")
    for thresh in (1.00, 0.95, 0.90, 0.80):
        n_qual = 0
        n_clusters_dist = []
        for tnp, flanks in tnp_dn.items():
            clus = cluster_single_linkage(flanks, thresh)
            if len(clus) >= 5:
                n_qual += 1
            n_clusters_dist.append(len(clus))
        arr = np.array(n_clusters_dist)
        print(f"  thresh={thresh:.2f}: {n_qual:>4}/{len(tnp_dn)} qualify (>=5 clusters),  "
              f"n_clusters dist: mean={arr.mean():.2f} med={float(np.median(arr)):.1f} min={arr.min()} max={arr.max()}")

    # 2. Pick threshold=0.90 as primary; requalify the 30 A1 Tnps under this rule
    THRESH = 0.90
    print()
    print(f"=== Requalify A1's 30 Tnps under threshold={THRESH} ===")
    sorted_ids = sorted(tnp_dn.keys())
    rng = np.random.default_rng(SEED)
    picked_all = list(rng.choice(sorted_ids, size=min(30, len(sorted_ids)), replace=False))

    # For each picked Tnp: cluster, keep at most one representative per cluster
    # (largest-cluster representatives first, then random), take up to 5.
    tnp_flanks_after: dict[str, list[str]] = {}
    n_eff_after: dict[str, float] = {}
    tnp_dropped: list[str] = []
    for tnp in picked_all:
        pool = tnp_dn[tnp]
        clus = cluster_single_linkage(pool, THRESH)
        # sort clusters by size desc so we prefer well-represented ones
        clus_sorted = sorted(clus, key=len, reverse=True)
        reps = [pool[c[0]] for c in clus_sorted]
        if len(reps) < 5:
            tnp_dropped.append(tnp)
            continue
        reps5 = reps[:5]
        tnp_flanks_after[tnp] = reps5
        n_eff_after[tnp] = n_eff_continuous(reps5)

    print(f"  qualifying Tnps (>=5 clusters): {len(tnp_flanks_after)}/{len(picked_all)}")
    print(f"  dropped: {tnp_dropped}")

    # 3. Run three-mode probe on the qualifying Tnps
    tnp_ids = list(tnp_flanks_after.keys())
    n_tnp = len(tnp_ids)
    if n_tnp == 0:
        print("[A1'] no Tnps qualified after dedup; aborting")
        return 1

    det_all = np.zeros((n_tnp, n_nc), dtype=bool)
    det_out = np.zeros((n_tnp, n_nc), dtype=bool)
    det_shft = np.zeros((n_tnp, n_nc), dtype=bool)

    t0 = time.time()
    for ti, tnp in enumerate(tnp_ids):
        flanks5 = tnp_flanks_after[tnp]
        for ni, nc_tnp in enumerate(nc_ids):
            nc = emt.nc_source_ncs[nc_tnp]
            nc_len_pos = len(nc) - L_FIX + 1
            per_orient_mats = {"fwd": [[], [], []], "rc": [[], [], []]}
            for fl in flanks5:
                m_dict = m_max_three_per_orient(nc, fl, L_FIX, TSD_WIDTH)
                for orient in ("fwd", "rc"):
                    for k, m in enumerate(m_dict[orient]):
                        per_orient_mats[orient][k].append(m)
            det_a = det_o = det_s = False
            for orient in ("fwd", "rc"):
                for k, dst in enumerate((det_all, det_out, det_shft)):
                    m5 = np.zeros((5, nc_len_pos), dtype=np.int32)
                    for si, arr in enumerate(per_orient_mats[orient][k]):
                        nn = min(nc_len_pos, len(arr))
                        m5[si, :nn] = arr[:nn]
                    if detect_at_thresh(m5, nc_len_pos):
                        if k == 0: det_a = True
                        if k == 1: det_o = True
                        if k == 2: det_s = True
            det_all[ti, ni]  = det_a
            det_out[ti, ni]  = det_o
            det_shft[ti, ni] = det_s
    elapsed = time.time() - t0

    def _report(name, mat):
        rate = mat.sum() / mat.size
        r = mat.sum(axis=1)
        c = mat.sum(axis=0)
        print(f"  {name:<22s} fires={int(mat.sum()):5d}/{mat.size}  rate={rate:.4f}  "
              f"per-Tnp med={float(np.median(r)):5.1f} max={r.max():3d}  "
              f"per-nc med={float(np.median(c)):5.1f} max={c.max():3d}")
        return rate, r

    print()
    print(f"=== A1' three-mode probe (IS10-R, dedup thresh={THRESH}, n_tnp={n_tnp}, wall {elapsed:.1f}s) ===")
    r_all,  row_all = _report("S_all",           det_all)
    r_out,  row_out = _report("S_outside_TSD",   det_out)
    r_shft, row_shft = _report("S_shift_matched", det_shft)
    print(f"  baseline durrant_shuffled = {DURRANT_SHUFFLED_NULL:.4f}")
    print(f"  ratio S_all / baseline    = {r_all/DURRANT_SHUFFLED_NULL:.2f}x")
    print(f"  partition effect (S_all - S_outside_TSD) = {r_all - r_out:.4f}")
    print(f"  shift-matched effect (S_all - S_shift)   = {r_all - r_shft:.4f}")
    print(f"  net TSD-specific = (S_all - S_out) - (S_all - S_shift) = {r_shft - r_out:.4f}")

    print()
    print("=== n_eff distribution (IS10-R after dedup, n_tnp={} ) ===".format(n_tnp))
    n_effs = np.array([n_eff_after[t] for t in tnp_ids])
    print(f"  n_eff: mean={n_effs.mean():.3f}  med={float(np.median(n_effs)):.3f}  "
          f"min={n_effs.min():.3f}  max={n_effs.max():.3f}")
    print(f"  n_eff by bucket (fires vs count):")
    for lo, hi in [(4.9, 5.01), (4.5, 4.9), (4.0, 4.5), (0.0, 4.0)]:
        mask = (n_effs >= lo) & (n_effs < hi)
        n_here = mask.sum()
        if n_here > 0:
            fires_all = row_all[mask].sum()
            print(f"    n_eff in [{lo:.2f}, {hi:.2f}):  {n_here} Tnps, S_all fires={int(fires_all)}, "
                  f"rate={fires_all/(n_here*n_nc):.4f}")

    print()
    print("=== Durrant baseline: n_eff on 65 positive Tnps ===")
    dur_n_effs = []
    for tnp_id in emt.flank_tnp_ids:
        durrant_flanks = [s.flank for s in emt.flank_tnps[tnp_id].sites][:5]
        if len(durrant_flanks) >= 2:
            dur_n_effs.append(n_eff_continuous(durrant_flanks))
    dur_n_effs = np.array(dur_n_effs)
    print(f"  Durrant n_eff: mean={dur_n_effs.mean():.3f}  med={float(np.median(dur_n_effs)):.3f}  "
          f"min={dur_n_effs.min():.3f}  max={dur_n_effs.max():.3f}")
    print(f"  Durrant Tnps with n_eff < 4.5: {int((dur_n_effs < 4.5).sum())} / {len(dur_n_effs)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
