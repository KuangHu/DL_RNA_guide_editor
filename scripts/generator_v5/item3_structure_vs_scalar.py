"""Item 3 — does the 22-channel structural patch add over 4 scalars?

Consumer-first check for the tensor layer redesign. The old candidate
layer's patch (64 nt × 22 channels) has never had its discrimination
value measured on real competitors. This script gives it a data-driven
opportunity to justify itself.

Setup:
  For V5 L=11 bags where Channel A Mode 2 fires:
    gold        = candidate at (planted orient, L=11, planted nc_start)
    competitor  = top-m candidate at (planted orient, L=11, nc_start != planted)
                  chosen from the same site's m_max array
  Features:
    scalar-only          = (matches, m/L, m_delta_from_bag_median,
                             matches_normalized_by_flank_len)
    scalar + structure   = above + guide-window means of
                             (dG_open_uL_pn, H_pair_win, cooperativity_win_pn,
                              E_span_win with NaN-mask)
  Train logistic regression on the (gold=1, competitor=0) labels with
  5-fold CV; compare AUROC.

Decision rule:
  Δ AUROC >= 5 pp   → structural patch geometry justifies the tensor layer
  Δ AUROC 2-5 pp    → structure adds; scalars-of-structure may be enough
  Δ AUROC < 2 pp    → structure adds negligibly at aggregate; the 22-channel
                       patch is not carrying its weight and should not be
                       rebuilt in the same form
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
M_THRESH = 8    # Channel A Mode 2 threshold
MAX_BAGS = 2000  # enough for reliable AUROC comparison

STRUCT_CHANNELS = ("dG_open_uL_pn", "H_pair_win", "cooperativity_win_pn", "E_span_win")


def _window_mean(arr: list, start: int, length: int) -> tuple[float, float]:
    """Return (mean, valid_frac) over arr[start:start+length].
    NaN entries are excluded from mean; valid_frac is (n_non_nan / length).
    """
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
    # Load bags; keep L=11 complete-5 only
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

    # Extract gold vs competitor pairs
    X_scalar_rows, X_struct_rows, y = [], [], []
    n_bags_used = 0
    for tnp, sites in bags.items():
        # Use site 0 (all sites share planted nc_start / L / orient)
        s = sites[0]
        lab = s['labels']; arch = lab['arch']
        active = lab['active_noncoding_index']
        nc = s['inputs']['noncoding_regions'][active]
        flank = s['inputs']['flank']
        planted_nc = lab['guide_span_in_active_noncoding'][0]
        orient = 'rc' if arch['orient'] == 'rev' else arch['orient']

        # v2 position arrays at L=11
        arrs = enumerate_position_arrays(nc, flank)
        arr = arrs.get((orient, L_TARGET))
        if arr is None or len(arr) < planted_nc + 1:
            continue

        gold_m = int(arr[planted_nc])
        if gold_m < M_THRESH:
            continue    # skip bags where Channel A wouldn't fire on gold

        # Competitor: max m at nc_start != planted_nc (with IoU-tolerance skipping)
        mask = np.ones_like(arr, dtype=bool)
        skip_lo = max(0, planted_nc - L_TARGET // 2)
        skip_hi = min(len(arr), planted_nc + L_TARGET // 2 + 1)
        mask[skip_lo:skip_hi] = False
        if not mask.any():
            continue
        comp_nc = int(np.argmax(arr * mask.astype(np.int32)))
        comp_m = int(arr[comp_nc])
        if comp_m < 6:
            continue    # skip trivially-easy bags with no real competitor

        # Bag-level median m at this L for the delta feature
        bag_median = int(np.median(arr))
        flank_len = len(flank)

        # Structure channels from nc_channels blob
        nc_ch = lab.get('nc_channels', [])
        active_ch = nc_ch[active] if active < len(nc_ch) else {}

        for cand_type, cand_nc, cand_m in (("gold", planted_nc, gold_m),
                                              ("comp", comp_nc, comp_m)):
            # Scalars
            scalar = [
                cand_m,
                cand_m / L_TARGET,
                cand_m - bag_median,
                cand_m / max(1, flank_len / L_TARGET),
            ]
            # Structure means over the L=11 guide window
            struct = []
            for ch in STRUCT_CHANNELS:
                arr_ch = active_ch.get(ch, [])
                mean, valid = _window_mean(arr_ch, cand_nc, L_TARGET)
                if np.isnan(mean):
                    mean = 0.0
                struct.append(mean)
                struct.append(valid)     # companion mask (valid fraction)

            X_scalar_rows.append(scalar)
            X_struct_rows.append(scalar + struct)
            y.append(1 if cand_type == "gold" else 0)

        n_bags_used += 1

    X_scalar = np.asarray(X_scalar_rows, dtype=np.float64)
    X_struct = np.asarray(X_struct_rows, dtype=np.float64)
    y = np.asarray(y, dtype=np.int64)
    print(f"n_bags used: {n_bags_used}, n_rows: {len(y)} ({int(y.sum())} gold + {int(len(y)-y.sum())} comp)")
    print(f"scalar features: {X_scalar.shape[1]}, +structure features: {X_struct.shape[1]}")

    # 5-fold CV with logistic regression
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
    print(f"=== 5-fold CV AUROC (gold vs top-m competitor) ===")
    print(f"  scalar-only          : {auroc_scalar:.4f} ± {std_scalar:.4f}")
    print(f"  scalar + structure   : {auroc_struct:.4f} ± {std_struct:.4f}")
    delta = auroc_struct - auroc_scalar
    print(f"  Δ AUROC              : {delta:+.4f}")

    print()
    if delta >= 0.05:
        verdict = "STRUCTURE-JUSTIFIED: patch geometry justified (Δ >= 5pp)"
    elif delta >= 0.02:
        verdict = "MARGINAL: structure adds but the 4 scalar means may suffice — don't build 22 channels blindly"
    else:
        verdict = "NEGLIGIBLE: structure adds < 2pp at aggregate — DO NOT rebuild the 22-channel patch"
    print(f"  Verdict: {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
