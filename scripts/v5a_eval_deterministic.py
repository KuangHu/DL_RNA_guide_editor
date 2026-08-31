"""V5A-3a0b sanity + deterministic baselines.

Scores each val-pool candidate with a chosen deterministic (or train-fit-linear)
function and reports the SAME evaluation matrix used by v5a_train_3a0 — so we can
compare against the trained selector on the identical eval codepath.

Methods:
  raw_m      : q(c) = matches                       (proposer-parity baseline)
  identity   : q(c) = matches / L
  linear     : q(c) = a*m + b*L + c*(m/L), a,b,c fit on TRAIN pairs by
                minimizing softplus(margin - q(c*) + q(d))
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")


def _fit_linear(mining_aug_path: str, train_tnps: set, margin: float = 0.5,
                 steps: int = 800, lr: float = 5e-2, seed: int = 0) -> tuple[float, float, float]:
    """Fit q = a*m + b*L + c*(m/L) on stratified train pairs by torch."""
    import torch, torch.nn.functional as F
    rng = np.random.default_rng(seed)

    by_regime = defaultdict(list)
    with open(mining_aug_path) as f:
        for line in f:
            m = json.loads(line)
            if m["transposase_id"] not in train_tnps: continue
            if not m["gold_in_pool"]: continue
            r = m["cstar_rank_regime"]
            cs = m["cstar"]
            cs_f = (float(cs["matches"]), int(cs["L"]))
            d_f = [(float(d["matches"]), int(d["L"])) for d in m["decoys"]]
            by_regime[r].append((cs_f, d_f))
    for r in by_regime: print(f"  {r:<10} n_records={len(by_regime[r])}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = torch.zeros(3, requires_grad=True, device=dev)
    opt = torch.optim.Adam([W], lr=lr)

    def feats(m_, L_):
        return np.stack([m_/16.0, L_/16.0, m_/np.maximum(1, L_)], axis=-1).astype(np.float32)

    for step in range(1, steps + 1):
        all_cs = []; all_d = []
        for r in list(by_regime.keys()):
            recs = by_regime[r]
            picks = rng.choice(len(recs), size=64, replace=len(recs) < 64)
            for i in picks:
                cs_f, d_f = recs[int(i)]
                d_idx = rng.choice(len(d_f), size=6, replace=False)
                for j in d_idx:
                    all_cs.append(feats(*cs_f))
                    all_d.append(feats(*d_f[j]))
        cs = torch.from_numpy(np.stack(all_cs)).to(dev)
        dc = torch.from_numpy(np.stack(all_d)).to(dev)
        q_cs = (cs @ W)
        q_dc = (dc @ W)
        loss = F.softplus(margin - q_cs + q_dc).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 200 == 0:
            print(f"    lin step {step:>4}  loss={loss.item():.4f}  W={W.detach().cpu().numpy()}",
                  flush=True)
    a, b, c = W.detach().cpu().numpy().tolist()
    print(f"  [linear] a(m)={a:.3f}  b(L)={b:.3f}  c(m/L)={c:.3f}", flush=True)
    return a, b, c


def score(slot: dict, method: str, coefs=None) -> float:
    m = float(slot["matches"]); L = int(slot["L"])
    if method == "raw_m":    return m
    if method == "identity": return m / max(1, L)
    if method == "linear":
        a, b, c = coefs
        return a * (m/16.0) + b * (L/16.0) + c * (m/max(1, L))
    if method == "length_pen":
        # q = m - alpha * max(0, L - L0)
        alpha, L0 = coefs
        return m - alpha * max(0.0, L - L0)
    raise ValueError(method)


def _rank_stats(qs: np.ndarray, cs_local: int, k_list=(1, 4, 8)):
    """Expected rank / R@k / MRR under uniform random tie-break within tied groups.
    Returns (rank_avg, R@k dict, MRR)."""
    q_cs = qs[cs_local]
    n_gt = int((np.delete(qs, cs_local) > q_cs).sum())
    n_eq = int((np.delete(qs, cs_local) == q_cs).sum())     # ties among OTHERS
    tie_group = n_eq + 1                                     # incl. c*
    rank_avg = n_gt + 1 + n_eq / 2.0
    # Expected R@k = P(c* lands in top k under uniform placement in tie group).
    R = {}
    for k in k_list:
        if n_gt >= k: R[k] = 0.0
        else:         R[k] = min(1.0, (k - n_gt) / tie_group)
    # Expected reciprocal rank: mean over the tie group of 1/(n_gt+1+i).
    idx = np.arange(tie_group, dtype=np.float64)
    E_recip = float(np.mean(1.0 / (n_gt + 1 + idx)))
    return rank_avg, R, E_recip


def evaluate(val_pool_path: str, method: str, coefs=None):
    """Uses EXPECTED R@k + MRR under uniform random tie-break (see _rank_stats)."""
    B_buckets = ((0,0), (1,5), (6,20), (21,50), (51, 10**9))
    stats_by_regime = {r: [] for r in ("r1_4", "r5_20", "r21_50", "r51_plus")}
    stats_by_B = {b: [] for b in B_buckets}
    pooled = []
    p_beats = {b: [] for b in
        ("wrong_orientation","different_region","same_region_longer_L",
          "same_region_shorter_L","same_region_same_L_wrong_flank","near_gold")}
    n_bags = n_in = 0
    with open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            n_bags += 1
            if rec["cstar_slot"] < 0: continue
            slots = rec["slots"]
            qs = np.asarray([score(s, method, coefs) for s in slots], dtype=np.float32)
            cs_local = None
            for j, s in enumerate(slots):
                if s["slot"] == rec["cstar_slot"]: cs_local = j; break
            if cs_local is None: continue
            n_in += 1
            rank_avg, R, E_recip = _rank_stats(qs, cs_local)
            entry = {"rank_avg": rank_avg, "R1": R[1], "R4": R[4], "R8": R[8], "MRR": E_recip}
            pooled.append(entry)
            cr = rec["cstar_rank"]
            regime = ("r1_4" if cr <= 4 else "r5_20" if cr <= 20
                       else "r21_50" if cr <= 50 else "r51_plus")
            stats_by_regime[regime].append(entry)
            B = rec["full_pool_burden_ge"]
            for (lo, hi) in B_buckets:
                if lo <= B <= hi:
                    stats_by_B[(lo, hi)].append(entry); break
            # taxonomy P(c*>d): uses > strict, so ties don't count as c* win.
            q_cs = qs[cs_local]
            for j, s in enumerate(slots):
                if j == cs_local: continue
                if s["bucket"] in p_beats:
                    p_beats[s["bucket"]].append(int(q_cs > qs[j]))
    def _s(rs):
        if not rs: return {"n":0}
        arr_R1 = np.asarray([r["R1"] for r in rs])
        arr_R4 = np.asarray([r["R4"] for r in rs])
        arr_R8 = np.asarray([r["R8"] for r in rs])
        arr_MRR = np.asarray([r["MRR"] for r in rs])
        arr_rank = np.asarray([r["rank_avg"] for r in rs])
        return {"n": int(len(rs)),
                 "R@1":    float(arr_R1.mean()),
                 "R@4":    float(arr_R4.mean()),
                 "R@8":    float(arr_R8.mean()),
                 "median_rank_avg": float(np.median(arr_rank)),
                 "MRR":    float(arr_MRR.mean())}
    return {
        "n_bags":     n_bags,
        "n_c*_in":    n_in,
        "pooled":     _s(pooled),
        "by_regime":  {k: _s(v) for k, v in stats_by_regime.items()},
        "by_burden":  {f"B_{lo}_{hi if hi<10**9 else 'inf'}": _s(stats_by_B[(lo, hi)])
                         for (lo, hi) in B_buckets},
        "taxonomy_p_beats": {k: (float(np.mean(v)) if v else float("nan"), len(v))
                                for k, v in p_beats.items()},
    }


def _report(name: str, r: dict):
    print(f"\n=== {name}   n_bags={r['n_bags']}  n_c*_in_pool={r['n_c*_in']} ===")
    p = r["pooled"]
    print(f"  POOLED (expected R@k, MRR under uniform tie-break):")
    print(f"    MRR={p['MRR']:.4f}  R@1={p['R@1']:.4f}  R@4={p['R@4']:.4f}  R@8={p['R@8']:.4f}  med_rank={p['median_rank_avg']:.1f}  n={p['n']}")
    print(f"  by regime  MRR | R@1 | R@4 | R@8   (bins defined by raw_m rank — partly definitional)")
    for k in ("r1_4","r5_20","r21_50","r51_plus"):
        d = r["by_regime"][k]
        if d.get("n",0)==0: print(f"    {k:<10} n=0"); continue
        print(f"    {k:<10} {d['MRR']:.3f} | {d['R@1']:.3f} | {d['R@4']:.3f} | {d['R@8']:.3f}   n={d['n']}")
    print(f"  taxonomy P(c*>d):")
    for k, (v, n) in r["taxonomy_p_beats"].items():
        print(f"    {k:<32} {v:.3f}  n={n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-pool", required=True)
    ap.add_argument("--methods", nargs="+", default=["raw_m", "identity"],
                     choices=["raw_m", "identity", "linear", "length_pen"])
    ap.add_argument("--length-pen-alpha-grid", type=str, default="0.25,0.5,0.75,1.0")
    ap.add_argument("--length-pen-L0-grid", type=str, default="10,11,12,13,14")
    ap.add_argument("--mining-aug", help="Required when --methods includes linear")
    ap.add_argument("--splits", help="Required when --methods includes linear")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    results = {}
    for m in args.methods:
        if m == "linear":
            assert args.mining_aug and args.splits, "linear needs --mining-aug and --splits"
            train_tnps = set(json.load(open(args.splits))["train"])
            print(f"[linear] fitting on train pairs ...", flush=True)
            coefs = _fit_linear(args.mining_aug, train_tnps)
            r = evaluate(args.val_pool, "linear", coefs)
            results["linear"] = {"coefs": coefs, "eval": r}
            _report(m, r)
        elif m == "length_pen":
            best = None
            alphas = [float(x) for x in args.length_pen_alpha_grid.split(",")]
            L0s = [int(x) for x in args.length_pen_L0_grid.split(",")]
            print(f"[length_pen] grid over alpha={alphas}  L0={L0s}", flush=True)
            per_grid = {}
            for a in alphas:
                for L0 in L0s:
                    r = evaluate(args.val_pool, "length_pen", (a, L0))
                    mrr = r["pooled"]["MRR"]
                    per_grid[f"a={a}_L0={L0}"] = {"MRR": mrr, "R@1": r["pooled"]["R@1"]}
                    if best is None or mrr > best[0]:
                        best = (mrr, (a, L0), r)
            print(f"[length_pen] best (alpha, L0) = {best[1]}   pooled MRR = {best[0]:.4f}")
            results["length_pen"] = {"coefs": list(best[1]), "grid": per_grid, "eval": best[2]}
            _report(f"length_pen(alpha={best[1][0]}, L0={best[1][1]})", best[2])
        else:
            r = evaluate(args.val_pool, m)
            results[m] = {"eval": r}
            _report(m, r)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
