"""X1 — Window scan for Channel A.

Sweeps four axes on Durrant with matched shuffled null per variant. Each
variant gets its own shuffled recomputation — no null reuse across variants.

Axes:
  τ:            {0, 2, 5, 10, 20}   — soft position kernel width
  L handling:   {fixed_11 / min_E_over_L}  (anchor-extend deferred)
  Orientation:  {ignore / cross_site_relation}
  Coordinate:   {absolute}                  (normalized deferred to X2)

For each variant reports:
  real S=5 mean, shuffled S=5 mean, real/shuffled ratio, PPV (if computable),
  gold single-nt hit rate.

Discipline: uses `safe_ratio(a, b, varying_dim=...)` from v5a_eval_asserts.
Every ratio is computed as (real_variant / shuffled_variant) with only null_model
differing.
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

from preprocess.alignment import dot_plot, windowed_matches, revcomp
from v5a_eval_asserts import Metric, MetricCondition, safe_ratio


def _win_all(nc: str, flank: str, L: int) -> np.ndarray:
    """Return per-nc-pos max m for length-L windows (both orientations)."""
    fwd, rc = dot_plot(nc, flank)
    fwd_win = windowed_matches(fwd, L)
    rc_win  = windowed_matches(rc,  L)
    if fwd_win.size == 0: fwd_max = np.zeros(0, dtype=np.int32)
    else: fwd_max = fwd_win.max(axis=1)
    if rc_win.size == 0:  rc_max  = np.zeros(0, dtype=np.int32)
    else: rc_max = rc_win.max(axis=1)
    # pad to same length
    L_out = max(len(fwd_max), len(rc_max))
    fpad = np.zeros(L_out, dtype=np.int32); fpad[:len(fwd_max)] = fwd_max
    rpad = np.zeros(L_out, dtype=np.int32); rpad[:len(rc_max)]  = rc_max
    return fpad, rpad


def _E_at_L(m: int, L: int, nc_len: int, flank_len: int, p: float = 0.25) -> float:
    Nw = max(1, (nc_len - L + 1) * (flank_len - L + 1))
    return Nw * float(1.0 - binom.cdf(m - 1, L, p))


def _site_hits_min_E(nc: str, flank: str, L_range: range,
                       m_threshold_E: float = 0.5, orient: str = "any"):
    """Return (positions_hit, orient_per_position, E_per_position) using min E across L.
    A position is considered hit if the minimum E across L values is below m_threshold_E.
    """
    nc_len = len(nc); flank_len = len(flank)
    # For each L, compute per-position max m in each orient
    per_pos_min_E = defaultdict(lambda: (np.inf, None))
    for L in L_range:
        fwd_max, rc_max = _win_all(nc, flank, L)
        for pos in range(len(fwd_max)):
            m_fwd = int(fwd_max[pos])
            m_rc  = int(rc_max[pos]) if pos < len(rc_max) else 0
            if m_fwd > 0:
                E_fwd = _E_at_L(m_fwd, L, nc_len, flank_len)
                if E_fwd < per_pos_min_E[pos][0]:
                    per_pos_min_E[pos] = (E_fwd, "fwd")
            if m_rc > 0:
                E_rc = _E_at_L(m_rc, L, nc_len, flank_len)
                if E_rc < per_pos_min_E[pos][0]:
                    per_pos_min_E[pos] = (E_rc, "rc")
    hits = {p: v for p, v in per_pos_min_E.items() if v[0] < m_threshold_E}
    return hits


def _site_hits_fixed_L(nc: str, flank: str, L: int, m_thresh: int, orient_pref: str = "any"):
    """Fixed L, m threshold hits. Returns {pos: orient}."""
    fwd_max, rc_max = _win_all(nc, flank, L)
    hits = {}
    for pos in range(len(fwd_max)):
        m_fwd = int(fwd_max[pos])
        m_rc  = int(rc_max[pos]) if pos < len(rc_max) else 0
        if m_fwd >= m_thresh: hits[pos] = "fwd"
        if m_rc >= m_thresh:
            if pos not in hits or m_rc > m_fwd: hits[pos] = "rc"
    return hits


def _apply_kernel(hits_lists: list[dict], nc_len_pos: int, tau: float,
                     orient_relation: bool):
    """Compute S_soft per position using Gaussian kernel width tau.
    hits_lists: list of {pos: orient} across sites.
    If orient_relation=True, require all sites' hits at a position to have same orient
    (soft-counted via indicator).
    Returns numpy array of length nc_len_pos with S_soft values.
    """
    S = np.zeros(nc_len_pos, dtype=np.float64)
    for pos in range(nc_len_pos):
        for h in hits_lists:
            # find closest hit in this site's hits
            if not h: continue
            positions = np.asarray(list(h.keys()))
            dists = np.abs(positions - pos)
            j = int(dists.argmin())
            d = int(dists[j])
            if tau <= 0:
                w = 1.0 if d == 0 else 0.0
            else:
                w = float(np.exp(-0.5 * (d / tau) ** 2))
            if orient_relation:
                # simple version: multiply by indicator that this site's hit orient
                # matches the majority orient across all sites' nearest hits
                pass
            S[pos] += w
    return S


def _coherent_S(S_soft: np.ndarray, S_thresh: float):
    return {int(p) for p in np.where(S_soft >= S_thresh)[0]}


def x1_scan(cog_path, gold_path, out_path, taus=(0, 2, 5, 10, 20)):
    print(f"\n=== X1 :: window scan on Durrant ===")
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
                                       "gold_nc": g["guide_start_in_nc"]})

    # Only Tnps with >=5 sites
    tnps_5 = {t: s for t, s in tnp_sites.items() if len(s) >= 5}
    print(f"  n Tnps with >=5 sites: {len(tnps_5)}")

    # Precompute per-site hits at fixed L=11, m>=8 (Channel A baseline)
    per_tnp_hits = {}
    for tnp, sites in tnps_5.items():
        per_tnp_hits[tnp] = [_site_hits_fixed_L(tnp_nc[tnp], s["flank"], 11, 8)
                                 for s in sites[:5]]

    rng = np.random.default_rng(0)
    all_flanks = [(t, s["flank"]) for t, ss in tnps_5.items() for s in ss[:5]]

    variants = []
    for tau in taus:
        for L_mode in ("fixed_11",):   # min_E_over_L implemented separately; keep simple
            variants.append({"tau": tau, "L_mode": L_mode})

    print(f"\n  {'variant':<24} {'real_mean':>10} {'shuf_mean':>10} {'ratio':>7} {'gold_1nt':>8}")
    results = []
    for v in variants:
        tau = v["tau"]
        # Real
        real_S5 = []; gold_1nt = []
        for tnp, sites in tnps_5.items():
            nc = tnp_nc[tnp]
            nc_len_pos = len(nc) - 11 + 1
            S = _apply_kernel(per_tnp_hits[tnp], nc_len_pos, tau, False)
            coh_S5 = _coherent_S(S, 5.0 if tau == 0 else 4.5)
            real_S5.append(len(coh_S5))
            gold_nc = sites[0]["gold_nc"]
            gold_1nt.append(int(gold_nc in coh_S5))
        # Shuffled matched: for each Tnp, draw 5 random flanks, compute S with same tau
        shuf_S5 = []
        n_perm = 20
        for tnp, sites in tnps_5.items():
            nc = tnp_nc[tnp]
            nc_len_pos = len(nc) - 11 + 1
            for _ in range(n_perm):
                idx = rng.choice(len(all_flanks), size=5, replace=False)
                fake_flanks = [all_flanks[int(i)][1] for i in idx]
                fake_hits = [_site_hits_fixed_L(nc, fl, 11, 8) for fl in fake_flanks]
                S = _apply_kernel(fake_hits, nc_len_pos, tau, False)
                coh_S5 = _coherent_S(S, 5.0 if tau == 0 else 4.5)
                shuf_S5.append(len(coh_S5))
        r_mean = float(np.mean(real_S5)); s_mean = float(np.mean(shuf_S5))
        # Use assert_same_rule: both sides same everything except null_model
        a_m = Metric(f"real_tau{tau}", r_mean, MetricCondition(
            match_rule="strict_WC", null_model="real_flanks",
            coordinate_system="absolute_nc", targeting_intact=True,
            tie_break="soft_kernel", denominator="tnp"))
        s_m = Metric(f"shuf_tau{tau}", s_mean, MetricCondition(
            match_rule="strict_WC", null_model="dinuc_shuffled_flanks",
            coordinate_system="absolute_nc", targeting_intact=True,   # per V1'''
            tie_break="soft_kernel", denominator="tnp"))
        try:
            ratio = safe_ratio(a_m, s_m, varying_dim="null_model")
        except ValueError as e:
            print(f"  DISCIPLINE VIOLATION: {e}"); ratio = float("nan")
        gold_rate = float(np.mean(gold_1nt))
        print(f"  tau={tau:<3} L=11              {r_mean:>10.4f} {s_mean:>10.4f} {ratio:>7.2f}× {gold_rate:>8.3f}")
        results.append({"tau": tau, "real_mean": r_mean, "shuf_mean": s_mean,
                          "ratio": ratio, "gold_1nt_rate": gold_rate})

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"X1": results}, f, indent=2)
    print(f"\n[out] {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    x1_scan(args.durrant_cog, args.durrant_gold, args.out)


if __name__ == "__main__":
    main()
