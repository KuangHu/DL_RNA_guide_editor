"""Item 3b — structure vs scalar discrimination on Channel A's un-detected bags.

Item 3 measured on Channel A-detected bags (5-of-5 coherence at planted p) —
the easy slice, not Channel B's target. Item 3b samples V5 L=11 bags where
exactly S ∈ {3, 4} sites hit m>=8 at the planted position: partial coherence,
Channel A does not fire, and per-site evidence has to substitute for the
cross-site aggregator.

Same features, same 5-fold CV, same 1-vs-1 gold-vs-competitor discrimination.
If Δ AUROC on this slice is significantly larger than the +3pp Item 3 found
on the easy slice, the multi-scale structural patch has a downstream reason
to exist. If it is the same +3pp, the tensor layer stays scalar-only.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.candidates_v2 import enumerate_position_arrays


V5_JSONL = "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_50k.jsonl"
L_TARGET = 11
M_THRESH = 8
MAX_BAGS = 5000
S_TARGET = {3, 4}      # partial-coherence slice

STRUCT_CHANNELS = ("dG_open_uL_pn", "H_pair_win", "cooperativity_win_pn", "E_span_win")


def _window_mean(arr: list, start: int, length: int) -> tuple[float, float]:
    if not arr:
        return float("nan"), 0.0
    seg = arr[start:start + length]
    if not seg:
        return float("nan"), 0.0
    valid = [v for v in seg if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not valid:
        return float("nan"), 0.0
    return float(np.mean(valid)), len(valid) / len(seg)


def main() -> int:
    bags = defaultdict(list)
    with open(V5_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r['labels']['arch']['L'] != L_TARGET:
                continue
            tnp = r['transposase_id']
            bags[tnp].append(r)
            if len(bags) >= MAX_BAGS * 4:
                good = [(k, v) for k, v in bags.items() if len(v) == 5]
                if len(good) >= MAX_BAGS:
                    bags = dict(good[:MAX_BAGS])
                    break
    bags = {k: v for k, v in bags.items() if len(v) == 5}
    if len(bags) > MAX_BAGS:
        bags = dict(list(bags.items())[:MAX_BAGS])
    print(f"Loaded {len(bags)} L=11 complete-5 bags")

    X_scalar_rows, X_struct_rows, y = [], [], []
    S_hist = defaultdict(int)
    n_bags_used = 0
    for tnp, sites in bags.items():
        s0 = sites[0]
        lab0 = s0['labels']; arch0 = lab0['arch']
        active = lab0['active_noncoding_index']
        planted_nc = lab0['guide_span_in_active_noncoding'][0]
        orient = 'rc' if arch0['orient'] == 'rev' else arch0['orient']

        # Determine S at planted position across the 5 sites
        m_per_site = []
        arrs_per_site = []
        for s in sites:
            nc = s['inputs']['noncoding_regions'][s['labels']['active_noncoding_index']]
            flank = s['inputs']['flank']
            arrs = enumerate_position_arrays(nc, flank)
            arr = arrs.get((orient, L_TARGET))
            if arr is None or planted_nc >= len(arr):
                m_per_site.append(-1)
            else:
                m_per_site.append(int(arr[planted_nc]))
            arrs_per_site.append(arrs)

        if any(m < 0 for m in m_per_site):
            continue
        S = sum(1 for m in m_per_site if m >= M_THRESH)
        S_hist[S] += 1

        if S not in S_TARGET:
            continue

        # For the sites that DO hit planted (>=8), extract per-site
        # gold-vs-competitor pairs. Structure means come from the SAME
        # site's nc_channels (position-invariant molecule; guide window
        # varies per site's flank interactions).
        nc_ch = lab0.get('nc_channels', [])
        active_ch = nc_ch[active] if active < len(nc_ch) else {}
        flank_len = len(s0['inputs']['flank'])

        used_from_bag = 0
        for i, s in enumerate(sites):
            gold_m = m_per_site[i]
            if gold_m < M_THRESH:
                continue    # this site didn't fire — no gold candidate

            arr = arrs_per_site[i].get((orient, L_TARGET))
            # Competitor: max m at nc_start not in planted's neighborhood
            mask = np.ones_like(arr, dtype=bool)
            skip_lo = max(0, planted_nc - L_TARGET // 2)
            skip_hi = min(len(arr), planted_nc + L_TARGET // 2 + 1)
            mask[skip_lo:skip_hi] = False
            if not mask.any():
                continue
            comp_nc = int(np.argmax(arr * mask.astype(np.int32)))
            comp_m = int(arr[comp_nc])
            if comp_m < 6:
                continue

            bag_median = int(np.median(arr))
            for cand_type, cand_nc, cand_m in (("gold", planted_nc, gold_m),
                                                  ("comp", comp_nc, comp_m)):
                scalar = [
                    cand_m,
                    cand_m / L_TARGET,
                    cand_m - bag_median,
                    cand_m / max(1, flank_len / L_TARGET),
                ]
                struct = []
                for ch in STRUCT_CHANNELS:
                    arr_ch = active_ch.get(ch, [])
                    mean, valid = _window_mean(arr_ch, cand_nc, L_TARGET)
                    if np.isnan(mean):
                        mean = 0.0
                    struct.append(mean)
                    struct.append(valid)

                X_scalar_rows.append(scalar)
                X_struct_rows.append(scalar + struct)
                y.append(1 if cand_type == "gold" else 0)
            used_from_bag += 1

        if used_from_bag > 0:
            n_bags_used += 1

    print(f"S distribution (planted-p m>=8 count across sites): {dict(sorted(S_hist.items()))}")
    print(f"Bags in target S∈{S_TARGET}: {n_bags_used}")
    X_scalar = np.asarray(X_scalar_rows, dtype=np.float64)
    X_struct = np.asarray(X_struct_rows, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    print(f"n_rows: {len(y)}  ({int(y.sum())} gold + {int(len(y)-y.sum())} comp)")

    if len(y) < 200:
        print("!! insufficient rows for reliable CV — increase MAX_BAGS")
        return 1

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    def cv_auroc(X, y):
        scores = []
        for tr, te in skf.split(X, y):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr])
            Xte = sc.transform(X[te])
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(Xtr, y[tr])
            p = clf.predict_proba(Xte)[:, 1]
            scores.append(roc_auc_score(y[te], p))
        return float(np.mean(scores)), float(np.std(scores))

    auroc_scalar, std_scalar = cv_auroc(X_scalar, y)
    auroc_struct, std_struct = cv_auroc(X_struct, y)

    print()
    print(f"=== 5-fold CV AUROC (gold vs top-m competitor, S∈{S_TARGET}) ===")
    print(f"  scalar-only          : {auroc_scalar:.4f} ± {std_scalar:.4f}")
    print(f"  scalar + structure   : {auroc_struct:.4f} ± {std_struct:.4f}")
    delta = auroc_struct - auroc_scalar
    print(f"  Δ AUROC              : {delta:+.4f}")
    print(f"  (Item 3 on S=5 easy slice: Δ = +0.030)")

    print()
    if delta >= 0.05:
        verdict = "PATCH-JUSTIFIED on the Channel B slice: multi-scale patch build authorized"
    elif delta >= 0.02:
        verdict = "SAME AS EASY SLICE: structure helps at the same +2-3pp level, patch not justified"
    else:
        verdict = "STRUCTURE LESS USEFUL: on partial coherence, structure adds < 2pp — patch definitely not justified"
    print(f"  Verdict: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
