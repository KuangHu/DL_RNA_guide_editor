"""R1 proposal-level diagnostic — proposal vs scorer decomposition.

For each POS/NEG bag, run the candidate proposer, extract per-bag
proposal-strength statistics, and compare distributions matched by parent tnp.

No neural network in this script. Purely proposal-side dot-plot analysis:

  - `best_matches`  : top-1 candidate's matches score (out of L)
  - `best_identity` : best_matches / L
  - `top4_mean`     : mean matches over top-4 candidates
  - `top20_mean`    : mean matches over top-20 candidates

Then per-tnp mean across the tnp's sites → bag-level score. Report:
  1. Unpaired AUROC (POS bags vs NEG bags)
  2. Paired Δ = S(POS_i) − S(NEG_i^shuffled_from_same_parent) matched by tnp suffix
  3. Median / MAD / P(Δ>0)

Success case (proposal contains real cognate enrichment):
  P(Δ_proposal > 0) ≫ 0.5   AND   pair-scorer P(Δ_pair > 0) ~ 0.5
  → conclusion: proposal has signal, scorer loses it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

sys.path.insert(0, "/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer")

from preprocess.candidates import build_candidate_arrays, DEFAULT_L_MIN, DEFAULT_L_MAX


def _auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(set(labels.tolist())) < 2:
        return float("nan")
    return roc_auc_score(labels, scores)


def per_bag_stats(nc: str, flank: str) -> dict:
    """Run candidate proposer with zero structure and return bag stats."""
    # Zero-structure profile / valid — proposer doesn't need real structure for matching.
    prof = np.zeros((len(nc), 16), dtype=np.float32)
    val = np.zeros((len(nc), 16), dtype=bool)
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, prof, val,
        L_min=DEFAULT_L_MIN, L_max=DEFAULT_L_MAX,
    )
    # Features: [orient_fwd, orient_rc, L, matches, mismatches, score, ...]
    matches = feats[:, 3][mask]                        # (n_cands,)
    Ls      = feats[:, 2][mask]
    if len(matches) == 0:
        return {"best_matches": 0.0, "best_identity": 0.0,
                "top4_mean": 0.0, "top20_mean": 0.0, "n_cands": 0}
    sorted_matches = np.sort(matches)[::-1]
    best = float(sorted_matches[0])
    best_L = float(Ls[np.argmax(matches)])
    top4 = float(sorted_matches[:4].mean())
    top20 = float(sorted_matches[:20].mean())
    return {
        "best_matches":  best,
        "best_identity": best / max(1.0, best_L),
        "top4_mean":     top4,
        "top20_mean":    top20,
        "n_cands":       int(len(matches)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--neg-suffix", required=True,
                    help="Suffix mapping POS tnp -> NEG tnp for paired comparison.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    by_tnp: dict[str, list[dict]] = defaultdict(list)
    is_pos_by_tnp: dict[str, bool] = {}
    n_read = 0
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            tnp = r["transposase_id"]
            active_nc = r["labels"].get("active_noncoding_index", 0)
            ncs = r["inputs"]["noncoding_regions"]
            if active_nc >= len(ncs):
                active_nc = 0
            nc = ncs[active_nc]
            flank = r["inputs"]["flank"]
            if not nc or not flank:
                continue
            stats = per_bag_stats(nc, flank)
            by_tnp[tnp].append(stats)
            is_pos_by_tnp[tnp] = bool(r["labels"]["is_positive"])
            n_read += 1
            if n_read % 100 == 0:
                print(f"  processed {n_read} records...", flush=True)

    print(f"[data] {n_read} records, {len(by_tnp)} unique tnps", flush=True)

    # Aggregate per tnp
    tnp_ids = sorted(by_tnp.keys())
    metrics = ["best_matches", "best_identity", "top4_mean", "top20_mean"]
    per_tnp_score = {m: {} for m in metrics}
    for tnp in tnp_ids:
        for m in metrics:
            vals = [s[m] for s in by_tnp[tnp]]
            per_tnp_score[m][tnp] = float(np.mean(vals))

    # Overall AUROC per metric
    print("\n=== unpaired AUROC (POS bags vs NEG bags, tnp-level) ===")
    for m in metrics:
        scores = np.asarray([per_tnp_score[m][t] for t in tnp_ids], dtype=np.float32)
        labels = np.asarray([int(is_pos_by_tnp[t]) for t in tnp_ids], dtype=np.int32)
        au = _auroc(scores, labels)
        pos_med = float(np.median(scores[labels == 1]))
        neg_med = float(np.median(scores[labels == 0]))
        print(f"  {m:<16} AUROC={au:.4f}  POS_med={pos_med:.3f}  NEG_med={neg_med:.3f}")

    # Paired Δ
    print(f"\n=== paired Δ = proposal(POS_i) − proposal(NEG_i), matched via suffix {args.neg_suffix!r} ===")
    paired_out = {}
    for m in metrics:
        deltas = []
        for tnp in tnp_ids:
            if not is_pos_by_tnp[tnp]:
                continue
            neg_tnp = tnp + args.neg_suffix
            if neg_tnp not in per_tnp_score[m]:
                continue
            d = per_tnp_score[m][tnp] - per_tnp_score[m][neg_tnp]
            deltas.append(d)
        if not deltas:
            print(f"  {m:<16} no paired matches")
            continue
        arr = np.asarray(deltas, dtype=np.float32)
        stats = {
            "n":      int(len(arr)),
            "median": float(np.median(arr)),
            "MAD":    float(np.median(np.abs(arr - np.median(arr)))),
            "std":    float(arr.std()),
            "p_gt_0": float((arr > 0).mean()),
            "q10":    float(np.quantile(arr, 0.10)),
            "q90":    float(np.quantile(arr, 0.90)),
        }
        paired_out[m] = stats
        print(f"  {m:<16} n={stats['n']:>3}  median={stats['median']:+.3f}  MAD={stats['MAD']:.3f}  P(Δ>0)={stats['p_gt_0']:.3f}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "jsonl": args.jsonl,
            "neg_suffix": args.neg_suffix,
            "n_records": n_read,
            "n_tnps": len(by_tnp),
            "unpaired_auroc": {
                m: {
                    "auroc": float(_auroc(
                        np.asarray([per_tnp_score[m][t] for t in tnp_ids], dtype=np.float32),
                        np.asarray([int(is_pos_by_tnp[t]) for t in tnp_ids], dtype=np.int32),
                    )),
                }
                for m in metrics
            },
            "paired_delta": paired_out,
        }, f, indent=2)
    print(f"\n[out] {args.out}", flush=True)


if __name__ == "__main__":
    main()
