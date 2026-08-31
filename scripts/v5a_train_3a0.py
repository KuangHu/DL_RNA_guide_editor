"""V5A-3a0: minimal local selector trained by pairwise ranking loss.

Features per candidate (9 dims, normalized):
  [ m/16, L/16, m/L, log_tail/8,
    mm_count/16, mm_frac_5p, mm_frac_3p, mm_at_pos_0, mm_at_pos_last ]

Model: 9 → 32 → 32 → 1  MLP (~1.4k params).

Loss:
  L_rank = softplus( margin  −  q(c*)  +  q(d) )   for each c* and hard-decoy d.

Sampling: for each train step, stratified across c*-rank regimes {r1_4, r5_20,
r21_50, r51_plus} with equal weight. For each c*, sample K decoys biased toward:
  1. m_d > m_{c*}
  2. taxonomy diversity (different_region + wrong_orientation + same_region_longer_L)
  3. remainder uniform

Train on `augmented_mining`, evaluate periodically on the pre-built val pool.

NEVER feeds planted labels, c* coords, or decoy bucket to the model.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_DIM = 9      # 9 with log_tail (3a0), 8 without (3a0b)
USE_LOG_TAIL = True   # set by main() from CLI; _features() reads this


def _load_null_table(path: str):
    d = json.load(open(path))["per_L"]
    tables = {}
    for L, t in d.items():
        L = int(L)
        mvals = np.asarray(t["m_values"], dtype=np.float32)
        lt = np.asarray(t["log_tail"], dtype=np.float32)
        tables[L] = (mvals, lt)
    return tables


def _log_tail(tables, m: float, L: int) -> float:
    t = tables.get(int(L))
    if t is None: return 0.0
    mvals, lt = t
    idx = int(np.searchsorted(mvals, m, side="left"))
    if idx == len(mvals):
        return float(lt[-1])
    return float(lt[idx])


def _features(cand: dict, tables) -> np.ndarray:
    m = float(cand["matches"]); L = int(cand["L"])
    base = [
        m / 16.0,
        L / 16.0,
        m / max(1, L),
    ]
    if USE_LOG_TAIL:
        lt = float(cand.get("log_tail", _log_tail(tables, m, L)))
        base.append(lt / 8.0)
    base.extend([
        float(cand.get("mm_count", L - m)) / 16.0,
        float(cand.get("mm_frac_5p", 0.0)),
        float(cand.get("mm_frac_3p", 0.0)),
        float(cand.get("mm_at_pos_0", 0)),
        float(cand.get("mm_at_pos_last", 0)),
    ])
    return np.asarray(base, dtype=np.float32)


class Selector(nn.Module):
    def __init__(self, dim=FEATURE_DIM, hidden=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_training_by_regime(mining_aug_path: str, train_tnps: set, tables):
    """Group records by cstar_rank_regime → list of (c*_feats, [decoy_feats,...],
    [decoy_matches,...], cstar_matches, [decoy_buckets,...])."""
    by_regime = defaultdict(list)
    with open(mining_aug_path) as f:
        for line in f:
            m = json.loads(line)
            if m["transposase_id"] not in train_tnps: continue
            if not m["gold_in_pool"]: continue
            r = m["cstar_rank_regime"]
            cs = m["cstar"]
            cs_feats = _features(cs, tables)
            decoys = m["decoys"]
            decoy_feats = np.stack([_features(d, tables) for d in decoys], axis=0)
            decoy_matches = np.asarray([float(d["matches"]) for d in decoys], dtype=np.float32)
            decoy_buckets = [d["bucket"] for d in decoys]
            by_regime[r].append({
                "cs_feats":       cs_feats,
                "cs_matches":     float(cs["matches"]),
                "decoy_feats":    decoy_feats,
                "decoy_matches":  decoy_matches,
                "decoy_buckets":  decoy_buckets,
            })
    return by_regime


def sample_decoys(rec: dict, K: int, rng) -> np.ndarray:
    """Pick K decoy indices from the 12 with hard-negative bias.
    - >=50% biased toward m_d > cs_matches (strict overrank).
    - remainder biased toward diverse taxonomy.
    """
    n_decoys = rec["decoy_feats"].shape[0]
    stronger = np.where(rec["decoy_matches"] > rec["cs_matches"])[0]
    diverse_targets = ("different_region", "wrong_orientation", "same_region_longer_L")
    diverse = np.asarray([i for i in range(n_decoys)
                             if rec["decoy_buckets"][i] in diverse_targets])
    K_hard = max(1, K // 2)
    K_diverse = max(1, (K - K_hard) // 2)
    picks = []
    if len(stronger):
        picks.extend(rng.choice(stronger, size=min(K_hard, len(stronger)), replace=False).tolist())
    if len(diverse) and len(picks) < K:
        remaining = np.asarray([i for i in diverse if i not in picks])
        if len(remaining):
            picks.extend(rng.choice(remaining,
                                       size=min(K_diverse, len(remaining)),
                                       replace=False).tolist())
    if len(picks) < K:
        pool = [i for i in range(n_decoys) if i not in picks]
        picks.extend(rng.choice(pool, size=K - len(picks), replace=False).tolist())
    return np.asarray(picks[:K], dtype=np.int64)


def train_step(model, by_regime, K, margin, rng, device, batch_records_per_regime=32):
    model.train()
    regimes = list(by_regime.keys())
    all_cs = []; all_d = []
    for r in regimes:
        records = by_regime[r]
        picks = rng.choice(len(records), size=min(batch_records_per_regime, len(records)),
                             replace=len(records) < batch_records_per_regime)
        for i in picks:
            rec = records[int(i)]
            decoy_idx = sample_decoys(rec, K=K, rng=rng)
            all_cs.append(np.tile(rec["cs_feats"], (K, 1)))
            all_d.append(rec["decoy_feats"][decoy_idx])
    cs = torch.from_numpy(np.concatenate(all_cs, axis=0)).to(device)
    dc = torch.from_numpy(np.concatenate(all_d, axis=0)).to(device)
    q_cs = model(cs); q_dc = model(dc)
    loss = F.softplus(margin - q_cs + q_dc).mean()
    return loss


def evaluate_pool(model, val_pool_path: str, tables, device):
    """Load val pool jsonl, score every slot, rank c*, aggregate stats."""
    from collections import Counter
    n_bags = 0; n_in_pool = 0
    ranks = {r: [] for r in ("r1_4", "r5_20", "r21_50", "r51_plus")}
    B_buckets = ((0,0), (1,5), (6,20), (21,50), (51, 10**9))
    ranks_by_B = {b: [] for b in B_buckets}
    p_beats_by_taxon = {b: [] for b in
        ("wrong_orientation","different_region","same_region_longer_L",
          "same_region_shorter_L","same_region_same_L_wrong_flank","near_gold")}
    model.eval()
    with torch.no_grad(), open(val_pool_path) as f:
        for line in f:
            rec = json.loads(line)
            n_bags += 1
            if rec["cstar_slot"] < 0: continue
            slots = rec["slots"]
            feats = np.stack([_features(s, tables) for s in slots], axis=0)
            q = model(torch.from_numpy(feats).to(device)).cpu().numpy()
            # rank c*: 1 + count(q_d > q_cs) with tie-break by proposer order.
            cstar_slot_local = None
            for j, s in enumerate(slots):
                if s["slot"] == rec["cstar_slot"]:
                    cstar_slot_local = j; break
            if cstar_slot_local is None: continue
            n_in_pool += 1
            q_cs = q[cstar_slot_local]
            q_others = np.delete(q, cstar_slot_local)
            rank = 1 + int((q_others > q_cs).sum())
            # regime by CSTAR_RANK (proposer's original)
            cr = rec["cstar_rank"]
            regime = ("r1_4" if cr <= 4 else "r5_20" if cr <= 20
                       else "r21_50" if cr <= 50 else "r51_plus")
            ranks[regime].append(rank)
            # Burden bucket
            B = rec["full_pool_burden_ge"]
            for (lo, hi) in B_buckets:
                if lo <= B <= hi:
                    ranks_by_B[(lo, hi)].append(rank); break
            # Taxonomy P(q(c*)>q(d))
            for j, s in enumerate(slots):
                if j == cstar_slot_local: continue
                if s["bucket"] in p_beats_by_taxon:
                    p_beats_by_taxon[s["bucket"]].append(int(q_cs > q[j]))

    def _summ(rs):
        if not rs: return {"n": 0}
        arr = np.asarray(rs)
        return {"n": int(len(arr)),
                 "R@1": float((arr == 1).mean()),
                 "R@4": float((arr <= 4).mean()),
                 "R@8": float((arr <= 8).mean()),
                 "median": float(np.median(arr)),
                 "MRR":  float(np.mean(1.0 / arr))}

    out = {
        "n_bags":            n_bags,
        "n_c*_in_pool":      n_in_pool,
        "by_regime":         {k: _summ(v) for k, v in ranks.items()},
        "by_burden":         {f"B_{lo}_{hi if hi<10**9 else 'inf'}": _summ(ranks_by_B[(lo, hi)])
                                 for (lo, hi) in B_buckets},
        "taxonomy_p_beats":  {k: (float(np.mean(v)) if v else float("nan"), int(len(v)))
                                 for k, v in p_beats_by_taxon.items()},
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mining-aug", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--null-table", required=True)
    ap.add_argument("--val-pool", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--k-decoys", type=int, default=6)
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--batch-per-regime", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-log-tail", action="store_true",
                     help="V5A-3a0b: drop empirical log_tail feature (falsified by 3a0 eval).")
    args = ap.parse_args()

    global USE_LOG_TAIL, FEATURE_DIM
    if args.no_log_tail:
        USE_LOG_TAIL = False
        FEATURE_DIM = 8
    print(f"[features] USE_LOG_TAIL={USE_LOG_TAIL}  FEATURE_DIM={FEATURE_DIM}", flush=True)

    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}", flush=True)

    splits = json.load(open(args.splits))
    train_tnps = set(splits["train"])
    tables = _load_null_table(args.null_table)

    print(f"[load] scanning augmented mining for TRAIN records ({len(train_tnps)} tnps)...",
          flush=True)
    by_regime = load_training_by_regime(args.mining_aug, train_tnps, tables)
    for r in ("r1_4","r5_20","r21_50","r51_plus"):
        print(f"  {r:<12} n_records={len(by_regime.get(r, []))}", flush=True)

    model = Selector(dim=FEATURE_DIM).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(args.out_dir) / "train_log.jsonl"
    ckpt_path = Path(args.out_dir) / "selector_3a0.pt"

    for step in range(1, args.steps + 1):
        loss = train_step(model, by_regime, args.k_decoys, args.margin,
                            rng, device, batch_records_per_regime=args.batch_per_regime)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"  step {step:>5}  loss={loss.item():.4f}", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            report = evaluate_pool(model, args.val_pool, tables, device)
            with open(log_path, "a") as f:
                f.write(json.dumps({"step": step, "loss": loss.item(),
                                       "eval": report}) + "\n")
            r = report["by_regime"]
            b = report["by_burden"]
            print(f"\n[eval @ step {step}] n_bags={report['n_bags']}  "
                  f"n_c*_in_pool={report['n_c*_in_pool']}", flush=True)
            print(f"  by regime  R@1: r1_4={r['r1_4'].get('R@1',0):.3f}  "
                  f"r5_20={r['r5_20'].get('R@1',0):.3f}  "
                  f"r21_50={r['r21_50'].get('R@1',0):.3f}  "
                  f"r51+={r['r51_plus'].get('R@1',0):.3f}", flush=True)
            print(f"  by burden  R@1: B=0={b['B_0_0'].get('R@1',0):.3f}  "
                  f"B1-5={b['B_1_5'].get('R@1',0):.3f}  "
                  f"B6-20={b['B_6_20'].get('R@1',0):.3f}  "
                  f"B21-50={b['B_21_50'].get('R@1',0):.3f}  "
                  f"B51+={b['B_51_inf'].get('R@1',0):.3f}", flush=True)
            print("  taxonomy P(c*>d):")
            for k, (v, n) in report["taxonomy_p_beats"].items():
                print(f"    {k:<32} {v:.3f}  n={n}", flush=True)
            print()
            torch.save({"state_dict": model.state_dict(),
                          "step": step, "report": report}, ckpt_path)
    print(f"[done] checkpoint at {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
