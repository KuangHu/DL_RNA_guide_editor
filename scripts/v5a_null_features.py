"""V5A-2: empirical per-L null tail features from the mining data.

Estimates p_null(M_L >= m) directly from the training-split decoy distribution
(no Binomial assumption). Empirical tail includes the proposer's real extreme-
value behavior at length L.

  p_null(m, L) = (1 + #{d in train_decoys : L_d = L, m_d >= m}) / (1 + N_L)
  log_tail     = log(1 / p_null)

After building the null tables, runs a paired-AUROC sanity check:
  For each c*_in_pool train record, compare log_tail(c*) to the max log_tail
  across its top-12 decoys, and compute AUROC. Do the same for raw matches m.
  Predicted: log_tail beats m substantially on r5_20 / r21_50 / r51+ where the
  raw-matches ranker fails.

Output:
  --out-null: {L: [(m, log_tail)]} table (JSON, small)
  --out-report: sanity-check AUROCs
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np


def _load_splits(path: str) -> dict:
    d = json.load(open(path))
    return {"train": set(d["train"]), "val": set(d["val"]), "test": set(d["test"])}


def _build_null(train_mining_path: str, splits: dict) -> dict:
    """Return {L: (matches_sorted_desc, log_tail_at_that_m)}."""
    train_tnps = splits["train"]
    per_L = defaultdict(list)      # L -> list of decoy matches
    with open(train_mining_path) as f:
        for line in f:
            m = json.loads(line)
            if m["transposase_id"] not in train_tnps: continue
            for d in m["decoys"]:
                per_L[int(d["L"])].append(float(d["matches"]))
    tables = {}
    for L, ms in per_L.items():
        arr = np.asarray(ms, dtype=np.float32)
        N_L = len(arr)
        # For each unique m, count decoys with m_d >= m; convert to log-tail.
        uniq = np.unique(arr)
        counts_ge = np.zeros_like(uniq, dtype=np.float64)
        arr_sorted = np.sort(arr)                 # ascending
        for i, u in enumerate(uniq):
            counts_ge[i] = N_L - np.searchsorted(arr_sorted, u, side="left")
        p_null = (1.0 + counts_ge) / (1.0 + N_L)
        log_tail = np.log(1.0 / p_null)           # >= 0
        # Store as monotone lookup indexed by (m -> log_tail).
        tables[int(L)] = {
            "N_L":       int(N_L),
            "m_values":  uniq.astype(float).tolist(),
            "log_tail":  log_tail.tolist(),
        }
    return tables


def _lookup(tables: dict, m: float, L: int) -> float:
    """Piecewise-constant tail lookup — the log_tail for the smallest m_v >= m.
    If m is above all observed m_v, return the max log_tail (capped)."""
    t = tables.get(int(L))
    if t is None: return 0.0
    mvals = t["m_values"]; lt = t["log_tail"]
    # find first mv >= m via binary search
    lo, hi = 0, len(mvals)
    while lo < hi:
        mid = (lo + hi) // 2
        if mvals[mid] < m: lo = mid + 1
        else: hi = mid
    if lo == len(mvals):
        return lt[-1]                    # tail cap
    return lt[lo]


def _sanity_check(train_mining_path: str, splits: dict, tables: dict) -> dict:
    """Paired AUROC on TRAIN c*_in_pool records: log_tail vs raw m."""
    from sklearn.metrics import roc_auc_score
    train_tnps = splits["train"]
    regime_rows = defaultdict(lambda: {"cstar_m": [], "decoy_m": [],
                                          "cstar_lt": [], "decoy_lt": []})
    with open(train_mining_path) as f:
        for line in f:
            m = json.loads(line)
            if m["transposase_id"] not in train_tnps: continue
            if not m["gold_in_pool"]: continue
            r = m["cstar_rank_regime"]
            cs = m["cstar"]
            cs_m = float(cs["matches"]); cs_L = int(cs["L"])
            cs_lt = _lookup(tables, cs_m, cs_L)
            for d in m["decoys"]:
                regime_rows[r]["cstar_m"].append(cs_m)
                regime_rows[r]["decoy_m"].append(float(d["matches"]))
                regime_rows[r]["cstar_lt"].append(cs_lt)
                regime_rows[r]["decoy_lt"].append(_lookup(tables, d["matches"], int(d["L"])))
    out = {}
    for r, d in regime_rows.items():
        cs_m = np.asarray(d["cstar_m"]); dc_m = np.asarray(d["decoy_m"])
        cs_lt = np.asarray(d["cstar_lt"]); dc_lt = np.asarray(d["decoy_lt"])
        n = len(cs_m)
        if n == 0:
            out[r] = {"n": 0}; continue
        y = np.concatenate([np.ones(n), np.zeros(n)])
        auroc_m  = float(roc_auc_score(y, np.concatenate([cs_m, dc_m])))
        auroc_lt = float(roc_auc_score(y, np.concatenate([cs_lt, dc_lt])))
        p_beats_m  = float((cs_m  > dc_m ).mean())
        p_beats_lt = float((cs_lt > dc_lt).mean())
        p_ties_m   = float((cs_m == dc_m).mean())
        p_ties_lt  = float((cs_lt == dc_lt).mean())
        out[r] = {
            "n":               n,
            "auroc_m":         auroc_m,
            "auroc_log_tail":  auroc_lt,
            "p_c*_beats_decoy_m":       p_beats_m,
            "p_c*_beats_decoy_log_tail": p_beats_lt,
            "p_ties_m":        p_ties_m,
            "p_ties_log_tail": p_ties_lt,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mining-jsonl", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out-null", required=True)
    ap.add_argument("--out-report", required=True)
    args = ap.parse_args()

    print("[splits] loading", flush=True)
    splits = _load_splits(args.splits)
    print(f"  train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}",
          flush=True)

    print("[null] building empirical null tables from TRAIN decoys", flush=True)
    tables = _build_null(args.mining_jsonl, splits)
    for L in sorted(tables):
        t = tables[L]
        print(f"  L={L:<3}  N_L={t['N_L']:>10}  unique m={len(t['m_values'])}  "
              f"m_range=[{t['m_values'][0]:.0f}, {t['m_values'][-1]:.0f}]  "
              f"log_tail_max={max(t['log_tail']):.2f}",
              flush=True)

    Path(args.out_null).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_null, "w") as f:
        json.dump({"per_L": tables}, f, indent=2)
    print(f"[out-null] {args.out_null}", flush=True)

    print("\n[sanity] AUROC(c* vs decoy) on TRAIN c*_in_pool, by regime:", flush=True)
    report = _sanity_check(args.mining_jsonl, splits, tables)
    print(f"  {'regime':<12} {'n_rows':>9} {'AUROC_m':>10} {'AUROC_lt':>10}  "
          f"{'P(m)':>8} {'P(lt)':>8}", flush=True)
    for r in ("r1_4", "r5_20", "r21_50", "r51_plus"):
        d = report.get(r, {})
        if d.get("n", 0) == 0:
            print(f"  {r:<12} n=0"); continue
        print(f"  {r:<12} {d['n']:>9} {d['auroc_m']:>10.4f} {d['auroc_log_tail']:>10.4f}  "
              f"{d['p_c*_beats_decoy_m']:>8.3f} {d['p_c*_beats_decoy_log_tail']:>8.3f}")

    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[out-report] {args.out_report}", flush=True)


if __name__ == "__main__":
    main()
