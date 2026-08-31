"""Linear-head-only probe: can 6 cross-site dispersion features alone
close the wrong_position and wrong_structure_role AUROC gaps?

Features (per tnp, from model-picked candidates on val_v4):
    MAD(target_position)
    STD(target_position)
    IQR(target_position)
    STD(guide_nc_start)          # analog for structural-role dispersion
    STD(alignment_length)
    orientation entropy

Excluded on purpose:
    mean cand_raw / max_cand_score  — that reintroduces already-learned local
    evidence and confounds the "can dispersion alone close the gap?" question.

For each hard-negative profile:
    Train logistic regression (POS vs profile) with 5-fold stratified CV
    Report mean test-fold AUROC.

Also run the ORACLE version (label-based dispersions) to bound above what
picks-based features could ever achieve.

No GPU. Runs on login in a few seconds.
"""
from __future__ import annotations

import collections
import json
import math

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


ROWS_PATH = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/diag_v4_candidate_selection.jsonl'
FEATURE_NAMES = ['pos_MAD', 'pos_STD', 'pos_IQR', 'ncstart_STD', 'L_STD', 'orient_H']
N_FOLDS = 5
SEED = 0


def mad(v):
    a = np.asarray(v, dtype=np.float64)
    return float(np.median(np.abs(a - np.median(a))))


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0))


def _tnp_features(sites, source: str) -> list[float]:
    """Compute the 6 dispersion features for one tnp bag.
    source='model' uses model_* fields; source='oracle' uses true_* fields.
    Returns a list of 6 floats; NaN when unavailable.
    """
    key = 'model' if source == 'model' else 'true'
    pos = np.asarray([s.get(f'{key}_target_start') for s in sites
                      if s.get(f'{key}_target_start') is not None], dtype=np.float64)
    nc = np.asarray([s.get(f'{key}_guide_start' if source == 'oracle' else 'model_nc_start')
                     for s in sites
                     if s.get(f'{key}_guide_start' if source == 'oracle' else 'model_nc_start') is not None],
                    dtype=np.float64)
    L = np.asarray([s.get(f'{key}_L') for s in sites
                    if s.get(f'{key}_L') is not None], dtype=np.float64)
    ori = [s.get(f'{key}_orient') for s in sites if s.get(f'{key}_orient')]
    if len(pos) < 3 or len(nc) < 3 or len(L) < 3 or len(ori) < 3:
        return [float('nan')] * len(FEATURE_NAMES)
    q25, q75 = np.percentile(pos, [25, 75])
    return [
        mad(pos),
        float(pos.std()),
        float(q75 - q25),
        float(nc.std()),
        float(L.std()),
        entropy(collections.Counter(ori)),
    ]


def _load_and_group():
    per_tnp = collections.defaultdict(list)
    tnp_info: dict[str, dict] = {}
    with open(ROWS_PATH) as f:
        for line in f:
            r = json.loads(line)
            if r.get('selected_slot', -1) < 0:
                continue
            tid = r['tnp_id']
            per_tnp[tid].append(r)
            if tid not in tnp_info:
                tnp_info[tid] = {
                    'is_positive': r['is_positive'],
                    'violation_profile': r['violation_profile'],
                }
    return per_tnp, tnp_info


def _cv_auroc(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """5-fold stratified CV. Returns (mean_test_auroc, std_test_auroc)."""
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < N_FOLDS:
        return (float('nan'), float('nan'))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    aurocs = []
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
        clf.fit(Xtr, y[tr])
        s = clf.decision_function(Xte)
        aurocs.append(roc_auc_score(y[te], s))
    a = np.asarray(aurocs)
    return float(a.mean()), float(a.std())


def _linear_coefs(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
    clf.fit(sc.transform(X), y)
    return clf.coef_[0]


def main():
    per_tnp, tnp_info = _load_and_group()
    print(f'loaded {len(per_tnp)} tnps with selected candidates')

    # Build per-tnp feature vectors under two sources: model-picked and oracle.
    Xm, Xo, meta = [], [], []
    for tid, sites in per_tnp.items():
        fm = _tnp_features(sites, 'model')
        fo = _tnp_features(sites, 'oracle')
        if any(math.isnan(v) for v in fm) or any(math.isnan(v) for v in fo):
            continue
        Xm.append(fm); Xo.append(fo); meta.append(tnp_info[tid])
    Xm = np.asarray(Xm, dtype=np.float64)
    Xo = np.asarray(Xo, dtype=np.float64)
    print(f'   feature matrix: {Xm.shape}')

    # Comparison groups
    is_pos = np.asarray([m['is_positive'] for m in meta])
    profiles = np.asarray([(m['violation_profile'] or '') for m in meta])
    n_pos = int(is_pos.sum())
    print(f'   positives: {n_pos}')
    unique_profs, counts = np.unique(profiles[~is_pos], return_counts=True)
    for p, c in zip(unique_profs, counts):
        if p:
            print(f'   negative "{p}": {c}')

    targets = [
        'wrong_position_consistency',
        'wrong_structure_role_consistency',
        'wrong_orientation_consistency',
        'wrong_length_consistency',
        'level1_marginal_matched',
        'level3_paired_counterfactual',
    ]

    print()
    print('=' * 92)
    print(f'6-feature linear probe (POS vs each hard-negative profile)  ({N_FOLDS}-fold CV, mean±std)')
    print('=' * 92)
    print(f'{"target profile":<40} {"MODEL-picked AUROC":>25} {"ORACLE AUROC":>20}')
    print('-' * 92)
    for prof in targets:
        m = is_pos | (profiles == prof)
        y = is_pos[m].astype(int)
        if y.sum() == 0 or (y == 0).sum() == 0:
            continue
        auc_m, sd_m = _cv_auroc(Xm[m], y)
        auc_o, sd_o = _cv_auroc(Xo[m], y)
        print(f'{prof:<40} {auc_m:>13.3f} ± {sd_m:.3f}      {auc_o:>7.3f} ± {sd_o:.3f}')

    # Multi-class probe: POS vs ALL negatives (level3 excluded, matching main-arm
    # training semantics).
    print()
    print('=' * 92)
    print('Multi-negative combined (POS vs ALL Level-1 + Level-2, level3 excluded)')
    print('=' * 92)
    keep = is_pos | (
        (profiles != 'level3_paired_counterfactual') & (~is_pos)
    )
    y_all = is_pos[keep].astype(int)
    auc_m, sd_m = _cv_auroc(Xm[keep], y_all)
    auc_o, sd_o = _cv_auroc(Xo[keep], y_all)
    print(f'{"MODEL-picked":<40} {auc_m:>13.3f} ± {sd_m:.3f}')
    print(f'{"ORACLE":<40} {auc_o:>13.3f} ± {sd_o:.3f}')

    # And per-feature ranking on the model-picked overall problem
    print()
    print('=' * 92)
    print('Feature coefficients (standardized, positive coef = higher value pushes toward POS)')
    print('=' * 92)
    coef_model = _linear_coefs(Xm[keep], y_all)
    coef_oracle = _linear_coefs(Xo[keep], y_all)
    print(f'{"feature":<20} {"MODEL coef":>12} {"ORACLE coef":>14}')
    for name, cm, co in zip(FEATURE_NAMES, coef_model, coef_oracle):
        print(f'{name:<20} {cm:>+12.3f} {co:>+14.3f}')

    print()
    print('Note: negative coefficient => higher feature value predicts NEGATIVE, as expected')
    print('for dispersion features (positives have LOW dispersion, negatives have HIGH).')


if __name__ == '__main__':
    main()
