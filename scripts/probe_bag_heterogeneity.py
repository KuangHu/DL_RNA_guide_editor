"""Measure the V3 shortcut directly: is "how many sites deviate from the
bag's consensus" a label?

For every site we locate its single best ungapped alignment between any
non-coding region and the flank (max over L = 8..16, both orientations)
and keep WHERE it lands, not just how good it is:

    orient, L, flank_start, score

A site is called **deviant** when its best alignment does not sit on the
bag's consensus: a different orientation, or a flank start more than
``--pos-tol`` bp from the bag median. Then per bag:

    frac_deviant, pos_iqr, frac_modal_orient, score_p10, score_std

``frac_deviant`` is the quantity V4 is designed to equalise. In V3:

  - positives carry off_target / unresolved noise -> frac_deviant ~ 0.3,
    and the *weaker* the positive the higher it goes;
  - level3 negatives are 50/50 clean, all on the bag anchor
    -> frac_deviant ~ 0.

so the feature separates the classes *in the wrong direction*, which is
exactly the reported AUROC(weak vs level3)=0.99 > AUROC(strong vs
level3)=0.84 inversion. Note frac_deviant is a legitimate signal against
level1/level2 (those bags really have no shared rule); it is only a
shortcut where the bag is supposed to be matched.

Usage:
    python -m scripts.probe_bag_heterogeneity --split <splits/x.jsonl> \
        --n-tnp-per-group 80 --sites 25 --out probe.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocess.alignment import dot_plot, windowed_matches

L_MIN, L_MAX = 8, 16


def best_alignment(rec: dict) -> tuple[int, int, int, float]:
    """(orient, L, flank_start, score) of this site's single best alignment."""
    flank = rec["inputs"]["flank"]
    best = (0, 0, 0, -1.0)
    for nc in rec["inputs"]["noncoding_regions"]:
        fwd, rc = dot_plot(nc, flank)
        for orient, dot in ((0, fwd), (1, rc)):
            for L in range(L_MIN, L_MAX + 1):
                win = windowed_matches(dot, L)
                if not win.size:
                    continue
                idx = int(np.argmax(win))
                score = float(win.flat[idx]) / L
                if score > best[3]:
                    j = idx % win.shape[1]
                    best = (orient, L, int(j), score)
    return best


def bag_features(sites: list[dict], pos_tol: int) -> dict:
    al = np.array([best_alignment(s) for s in sites], dtype=float)
    orient, _L, pos, score = al[:, 0], al[:, 1], al[:, 2], al[:, 3]

    modal_orient = Counter(orient.tolist()).most_common(1)[0][0]
    med_pos = float(np.median(pos[orient == modal_orient])) if (orient == modal_orient).any() \
        else float(np.median(pos))
    deviant = (orient != modal_orient) | (np.abs(pos - med_pos) > pos_tol)
    return {
        "frac_deviant": float(deviant.mean()),
        "pos_iqr": float(np.quantile(pos, 0.75) - np.quantile(pos, 0.25)),
        "frac_modal_orient": float((orient == modal_orient).mean()),
        "score_p10": float(np.quantile(score, 0.10)),
        "score_std": float(score.std()),
    }


FEATURES = ["frac_deviant", "pos_iqr", "frac_modal_orient", "score_p10", "score_std"]


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Rank AUROC with proper mid-ranks for ties."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(order, dtype=float)
    sorted_v = allv[order]
    i = 0
    while i < len(sorted_v):
        j = i
        while j + 1 < len(sorted_v) and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rp = ranks[:pos.size].sum()
    return float((rp - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=Path)
    ap.add_argument("--n-tnp-per-group", type=int, default=80)
    ap.add_argument("--sites", type=int, default=25)
    ap.add_argument("--pos-tol", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    group_of: dict[str, str] = {}
    with a.split.open() as fh:
        for line in fh:
            r = json.loads(line)
            tid = r["transposase_id"]
            if tid not in group_of:
                lab = r["labels"]
                group_of[tid] = "positive" if lab["is_positive"] else lab["violation_profile"]
    by_group = defaultdict(list)
    for tid, g in group_of.items():
        by_group[g].append(tid)
    chosen = set()
    for g, tids in by_group.items():
        tids = sorted(tids)
        rng.shuffle(tids)
        chosen.update(tids[: a.n_tnp_per_group])
    print(f"{a.split.name}: {len(group_of)} bags, probing {len(chosen)}", flush=True)

    recs = defaultdict(list)
    with a.split.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r["transposase_id"] in chosen:
                recs[r["transposase_id"]].append(r)

    rows, groups, strengths, guided_frac = [], [], [], []
    for k, tid in enumerate(sorted(recs)):
        sites = recs[tid]
        sub = sites if len(sites) <= a.sites else rng.sample(sites, a.sites)
        f = bag_features(sub, a.pos_tol)
        rows.append([f[n] for n in FEATURES])
        groups.append(group_of[tid])
        md = sites[0].get("generator_metadata", {})
        strengths.append(md.get("tnp_strength", "n/a"))
        n_guided = md.get("n_guided_in_tnp")
        guided_frac.append(n_guided / len(sites) if n_guided is not None else float("nan"))
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(recs)}", flush=True)

    X = np.array(rows)
    g = np.array(groups)
    y = g == "positive"
    negprofiles = sorted(set(g[~y]))

    print("\n=== mean bag feature by group ===")
    print(f"{'group':<42}" + "".join(f"{n:>18}" for n in FEATURES))
    for name in ["positive"] + negprofiles:
        m = X[g == name].mean(axis=0)
        print(f"{name:<42}" + "".join(f"{v:>18.4f}" for v in m))

    print("\n=== AUROC(positive vs profile) per feature ===")
    print(f"{'feature':<20}" + "".join(f"{p[:16]:>18}" for p in negprofiles))
    for i, n in enumerate(FEATURES):
        print(f"{n:<20}" + "".join(
            f"{auroc(X[y, i], X[(~y) & (g == p), i]):>18.4f}" for p in negprofiles))

    print("\n=== frac_deviant of positives by strength (the inversion driver) ===")
    for s in ["strong", "moderate", "weak"]:
        m = (np.array(strengths) == s) & y
        if m.any():
            print(f"  {s:<10} n={int(m.sum()):<4} frac_deviant={X[m, 0].mean():.4f}")
    for p in negprofiles:
        print(f"  [neg] {p:<38} frac_deviant={X[g == p, 0].mean():.4f}")

    print("\n=== AUROC on frac_deviant, positives split by strength ===")
    for p in negprofiles:
        neg = X[g == p, 0]
        line = f"  vs {p:<38}"
        for s in ["strong", "moderate", "weak"]:
            m = (np.array(strengths) == s) & y
            line += f"  {s}={auroc(X[m, 0], neg):.3f}" if m.any() else ""
        print(line)

    if a.out:
        a.out.write_text(json.dumps({
            "split": str(a.split), "features": FEATURES,
            "X": X.tolist(), "groups": groups, "strengths": strengths,
            "guided_fraction": guided_frac,
        }))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
