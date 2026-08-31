"""V5A Gate 0: reweight V4.2 val to Durrant (L, pool_m) marginal.

For each in-pool val bag, assign sample weight
  w_i = p_Durrant(L_i, m_i) / p_V42(L_i, m_i)
where (L_i, m_i) is c*'s tolerant-matched slot's (length, matches). Both
distributions are on the same in-pool subset (Durrant n=236, V4.2 val n = c*_in_pool).

Report:
  - unweighted vs weighted pooled MRR / R@1 / R@4 / R@8 (with proper tie-break)
    for raw_m, identity, length_pen (best from grid).
  - effective sample size ESS = (Σw)² / Σw²
  - weight distribution (min/max/median/p99).
  - paired bootstrap 95% CI on Δ MRR (raw_m − identity) after reweight.
Verdict:
  (a) reweighted raw_m MRR drops into [0.09, 0.15]  → V4.2 contains Durrant regime;
      just reweight training loss + acceptance set; generator stays as-is.
  (b) reweighted raw_m MRR still > 0.15           → V4.2 does not cover Durrant;
      generator has to be rebuilt with matched planted-c* strength.
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

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX


def _overlap(a0, a1, b0, b1):
    return max(0, min(a1, b1) - max(a0, b0))


def _find_gold(feats, mask, cands, orient, L, nc, fl, of=0.5):
    valid = np.where(mask)[0]
    if len(valid) == 0: return -1, 0.0
    matches = feats[:, 3]
    best_slot = -1; best_m = -1.0
    for i in valid:
        c = cands[int(i)]
        if c.orient != orient: continue
        mn = min(c.L, L)
        nc_ov = _overlap(c.nc_start, c.nc_start + c.L, nc, nc + L)
        f_ov = _overlap(c.flank_start, c.flank_start + c.L, fl, fl + L)
        if nc_ov < of*mn or f_ov < of*mn: continue
        if matches[i] > best_m: best_m = float(matches[i]); best_slot = int(i)
    return best_slot, best_m


def _rank_stats(qs, cs_local, k_list=(1,4,8)):
    q_cs = qs[cs_local]
    other = np.delete(qs, cs_local)
    n_gt = int((other > q_cs).sum()); n_eq = int((other == q_cs).sum())
    tie = n_eq + 1
    rank_avg = n_gt + 1 + n_eq / 2.0
    R = {k: (0.0 if n_gt >= k else min(1.0, (k - n_gt) / tie)) for k in k_list}
    E_recip = float(np.mean(1.0 / (n_gt + 1 + np.arange(tie, dtype=np.float64))))
    return rank_avg, R, E_recip


def score(slot, method, coefs=None):
    m = float(slot["matches"]); L = int(slot["L"])
    if method == "raw_m":    return m
    if method == "identity": return m / max(1, L)
    if method == "length_pen":
        a, L0 = coefs
        return m - a * max(0.0, L - L0)
    raise ValueError(method)


def build_durrant_pool_gold(cog, gold_json):
    """Return list of (L_gold, m_gold) for each in-pool Durrant record."""
    gold = {json.loads(l)["site_id"]: json.loads(l)
             for l in open(gold_json)} if False else None
    gold = {}
    with open(gold_json) as f:
        for line in f:
            g = json.loads(line); gold[g["site_id"]] = g
    pairs = []
    with open(cog) as f:
        for line in f:
            r = json.loads(line)
            g = gold.get(r["site_id"]);
            if g is None: continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs): a = 0
            nc = ncs[a]; flank = r["inputs"]["flank"]
            prof = np.zeros((len(nc), 16), dtype=np.float32); val = np.zeros((len(nc), 16), dtype=bool)
            _, feats, mask, cands = build_candidate_arrays(
                nc, flank, prof, val, L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX)
            slot, m = _find_gold(feats, mask, cands,
                                    g["target_flank_orientation"],
                                    g["target_binding_loop_length"],
                                    g["guide_start_in_nc"],
                                    g["target_flank_start"])
            if slot < 0: continue
            pairs.append((int(cands[slot].L), int(m)))
    return pairs


def _weighted_stats(vals, weights):
    weights = np.asarray(weights, dtype=np.float64)
    vals = np.asarray(vals, dtype=np.float64)
    w = weights.sum()
    if w == 0: return float("nan")
    return float((vals * weights).sum() / w)


def _bootstrap_delta(w, a_vals, b_vals, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(w)
    w = np.asarray(w); a = np.asarray(a_vals); b = np.asarray(b_vals)
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ww = w[idx].sum()
        deltas[i] = ((a[idx] - b[idx]) * w[idx]).sum() / max(1e-12, ww)
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pool", required=True)
    ap.add_argument("--durrant-cog", required=True)
    ap.add_argument("--durrant-gold", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("[durrant] computing gold (L, m) distribution ...", flush=True)
    dur_pairs = build_durrant_pool_gold(args.durrant_cog, args.durrant_gold)
    print(f"  Durrant in-pool n={len(dur_pairs)}", flush=True)

    # Empirical distribution p_Durrant(L, m) with Laplace smoothing on unseen cells.
    dur_counts = Counter(dur_pairs)
    total_dur = sum(dur_counts.values())

    # Build V4.2 val stats for each in-pool bag.
    print("[val] scoring val pool ...", flush=True)
    per_bag = []
    with open(args.val_pool) as f:
        for line in f:
            rec = json.loads(line)
            if rec["cstar_slot"] < 0: continue
            slots = rec["slots"]
            cs_local = None
            for j, s in enumerate(slots):
                if s["slot"] == rec["cstar_slot"]: cs_local = j; break
            if cs_local is None: continue
            cs = slots[cs_local]
            L_i = int(cs["L"]); m_i = int(cs["matches"])
            recips = {}
            for method, coefs in (("raw_m", None), ("identity", None),
                                    ("length_pen", (0.5, 12))):
                qs = np.asarray([score(s, method, coefs) for s in slots], dtype=np.float32)
                _, R, MRR = _rank_stats(qs, cs_local)
                recips[method] = MRR
            per_bag.append({"L": L_i, "m": m_i, "recips": recips})
    print(f"  val in-pool n={len(per_bag)}", flush=True)

    # V4.2 pool marginal p_V42(L, m).
    v42_counts = Counter((b["L"], b["m"]) for b in per_bag)
    total_v42 = sum(v42_counts.values())

    # Compute per-bag weight w = (p_Durrant / p_V42) with Laplace smoothing.
    alpha = 1.0  # smoothing
    n_cells = 17 * 17
    weights = []
    for b in per_bag:
        Lm = (b["L"], b["m"])
        p_d = (dur_counts.get(Lm, 0) + alpha) / (total_dur + alpha * n_cells)
        p_v = (v42_counts.get(Lm, 0) + alpha) / (total_v42 + alpha * n_cells)
        weights.append(p_d / max(1e-12, p_v))
    W = np.asarray(weights, dtype=np.float64)
    ESS = float(W.sum() ** 2 / max(1e-12, (W ** 2).sum()))
    print(f"[weights] median={np.median(W):.4f}  p99={np.percentile(W,99):.4f}  max={W.max():.4f}")
    print(f"          ESS={ESS:.1f}  ESS/N={ESS/len(W):.3%}", flush=True)

    # Weighted pooled MRR + R@K per method.
    print("[eval] unweighted vs Durrant-reweighted MRR:")
    print(f"  {'method':<12} {'unweighted MRR':>16}   {'reweighted MRR':>18}")
    result = {}
    for method in ("raw_m", "identity", "length_pen"):
        recips = np.asarray([b["recips"][method] for b in per_bag])
        unw = float(np.mean(recips))
        rew = _weighted_stats(recips, W)
        result[method] = {"MRR_unweighted": unw, "MRR_reweighted": rew}
        print(f"  {method:<12} {unw:>16.4f}   {rew:>18.4f}")

    # Paired-bootstrap CI on Δ(raw_m − identity) after reweight
    a = np.asarray([b["recips"]["raw_m"]    for b in per_bag])
    b = np.asarray([b["recips"]["identity"] for b in per_bag])
    lo, hi = _bootstrap_delta(W, a, b)
    result["delta_bootstrap_raw_m_vs_identity"] = {"lo_2.5": lo, "hi_97.5": hi}
    print(f"  Δ MRR (raw_m − identity) 95% CI = [{lo:+.4f}, {hi:+.4f}]  (reweighted)")

    # Verdict
    rew_raw_m = result["raw_m"]["MRR_reweighted"]
    if 0.09 <= rew_raw_m <= 0.15:
        verdict = "REWEIGHT SUFFICIENT (V4.2 already contains Durrant regime; reweight training + acceptance)"
    elif rew_raw_m > 0.15:
        verdict = "GENERATOR REBUILD REQUIRED (V4.2 does not cover Durrant regime; V5 generator must plant weaker c*)"
    else:
        verdict = f"BELOW EXPECTED FLOOR ({rew_raw_m:.3f}); investigate — Durrant may have unmodelled structure"
    print(f"\n[verdict] reweighted raw_m MRR = {rew_raw_m:.4f}")
    print(f"          → {verdict}")

    result["ESS"] = ESS
    result["ESS_over_N"] = ESS / len(W)
    result["N_val_in_pool"] = len(per_bag)
    result["N_durrant_in_pool"] = len(dur_pairs)
    result["verdict"] = verdict

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[out] {args.out}")


if __name__ == "__main__":
    main()
