"""Audit existing V4 negatives against V4.2 positives, per violation profile.

For each of the 6 negative profiles in negatives_v4.jsonl, and V4.2 positives:
  Phase A: extract per-record features
    layout: [flank_len, active_nc_len, n_ncs]
    sequence marginal: [gc, entropy, mono4, dinuc16] on flank + on active NC
    candidate summary: [best_match, best_identity, best_flank_start, best_L,
                        best_orient_fwd, second_best_match]
  Phase B: per-profile per-feature median/mean vs V4.2 pos → distribution overlap
  Phase C: per-profile AUROC(V4.2 pos vs this negative profile) with:
    layout_only, marginal_only, candidate_only, combined
  Phase D: negative-only profile classifier — can we tell profiles apart from
           dumb features?  If yes, model may learn 'which recipe' rather than
           'cognate interaction'.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

from v42_shortcut_probes import (
    seq_to_arr, rc_seq, feat_layout, feat_flank_marginal,
    feat_nc_marginal, feat_flank_plus_nc, _best_ungapped, _seq_stats,
)


def feat_candidate_summary(rec):
    ncs = rec['inputs']['noncoding_regions']
    active = rec['labels'].get('active_noncoding_index', 0)
    nc = ncs[active] if active < len(ncs) else ''
    flank = rec['inputs']['flank']
    if len(nc) < 8 or len(flank) < 8:
        return np.zeros(7, dtype=np.float32)
    b = _best_ungapped(nc, flank)
    L = b['L']
    matches = b['matches']
    identity = matches / max(1, L)
    return np.asarray([matches, identity, b['flank_start'], L,
                        1.0 if b['orient'] == 'fwd' else 0.0,
                        # junction distance (proximity to flank start)
                        min(b['flank_start'], 120 - b['flank_start'] - L),
                        # length of best match / len(flank)
                        L / 120.0], dtype=np.float32)


def load_positives(path, n, seed=42):
    recs = []
    with open(path) as f:
        for line in f:
            recs.append(json.loads(line))
            if len(recs) >= n * 2: break
    rng = random.Random(seed)
    rng.shuffle(recs)
    return recs[:n]


def load_negatives_by_profile(path, n_per_profile, seed=42):
    """Stream through negatives file, bucket by violation_profile."""
    buckets = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            prof = r['labels'].get('violation_profile', 'unknown')
            if len(buckets[prof]) < n_per_profile * 3:
                buckets[prof].append(r)
            # Continue reading — some profiles come later in the file
            if all(len(v) >= n_per_profile * 3
                    for v in buckets.values() if v) and len(buckets) >= 6:
                break
    rng = random.Random(seed)
    out = {}
    for prof, recs in buckets.items():
        rng.shuffle(recs)
        out[prof] = recs[:n_per_profile]
    return out


def build_features(recs):
    """Return dict of feature-group -> np.ndarray (N, D)."""
    layouts, flanks, ncs, combos, cands = [], [], [], [], []
    for r in recs:
        layouts.append(feat_layout(r))
        flanks.append(feat_flank_marginal(r))
        ncs.append(feat_nc_marginal(r))
        combos.append(feat_flank_plus_nc(r))
        cands.append(feat_candidate_summary(r))
    return {
        'layout':          np.stack(layouts, axis=0),
        'flank_marginal':  np.stack(flanks, axis=0),
        'nc_marginal':     np.stack(ncs, axis=0),
        'combined_marginal': np.stack(combos, axis=0),
        'candidate_summary': np.stack(cands, axis=0),
        'all':             np.concatenate([np.stack(combos, axis=0),
                                            np.stack(cands, axis=0)], axis=1),
    }


def train_auroc(X_pos, X_neg, seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(X_neg))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd[sd == 0] = 1
    Xt = (X_tr - mu) / sd; Xe = (X_te - mu) / sd
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr.fit(Xt, y_tr)
    au_lr = roc_auc_score(y_te, lr.predict_proba(Xe)[:, 1])
    rf = RandomForestClassifier(n_estimators=100, max_depth=8,
                                  random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    au_rf = roc_auc_score(y_te, rf.predict_proba(X_te)[:, 1])
    return au_lr, au_rf


def profile_classifier_auroc(neg_features_by_profile, seed=42):
    """Multiclass profile classifier on negatives only. If AUROC(macro) is
    close to 1, negatives have easily-distinguishable recipes."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    profiles = list(neg_features_by_profile.keys())
    # Stack X and encode y as profile index
    X_list = []; y_list = []
    for i, p in enumerate(profiles):
        f = neg_features_by_profile[p]
        # Use combined feature set
        X_p = np.concatenate([f['combined_marginal'], f['candidate_summary'],
                                f['layout']], axis=1)
        X_list.append(X_p); y_list.append(np.full(len(X_p), i))
    X = np.concatenate(X_list, axis=0); y = np.concatenate(y_list)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                random_state=seed, stratify=y)
    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                  random_state=seed, n_jobs=1)
    rf.fit(X_tr, y_tr)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_te, rf.predict(X_te))
    print(f'\n  [phase D] negative-only profile classifier: '
          f'{len(profiles)} classes, {len(X)} records')
    print(f'    RandomForest accuracy: {acc:.4f}   '
          f'(random baseline = {1/len(profiles):.3f})')
    return acc


def report_feature_medians(pos_feats, neg_by_profile, feature_names):
    print(f'\n  Feature medians (Phase B):')
    print(f'  {"feature":<32} {"POSITIVE":>10}  ', end='')
    for prof in neg_by_profile:
        print(f'{prof[:14]:>16}', end='')
    print()
    # For each feature, print median
    for idx, name in enumerate(feature_names):
        pos_med = float(np.median(pos_feats[:, idx]))
        print(f'  {name:<32} {pos_med:>10.3f}  ', end='')
        for prof, f in neg_by_profile.items():
            neg_med = float(np.median(f[:, idx]))
            print(f'{neg_med:>16.3f}', end='')
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default='/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl')
    ap.add_argument('--neg', default='/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives_v4.jsonl')
    ap.add_argument('--n-pos', type=int, default=3000)
    ap.add_argument('--n-per-profile', type=int, default=1000)
    args = ap.parse_args()

    print(f'\n{"="*105}')
    print(f'  V4 NEGATIVES AUDIT vs V4.2 POSITIVES — per violation_profile')
    print(f'{"="*105}')

    print(f'\n[load] {args.n_pos} V4.2 positives from {args.pos}', flush=True)
    pos_recs = load_positives(args.pos, args.n_pos)
    print(f'[extract] positive features', flush=True)
    pos_feats = build_features(pos_recs)

    print(f'\n[load] up to {args.n_per_profile} per profile from {args.neg}', flush=True)
    neg_by_prof = load_negatives_by_profile(args.neg, args.n_per_profile)
    print(f'[loaded] profiles: {list((k, len(v)) for k, v in neg_by_prof.items())}',
          flush=True)

    print(f'\n[extract] per-profile features', flush=True)
    neg_feats_by_prof = {p: build_features(recs) for p, recs in neg_by_prof.items()}

    # Phase B: feature medians on candidate_summary + layout
    print(f'\n{"="*105}')
    print(f'  PHASE B — Candidate summary medians  (7 features)')
    print(f'{"="*105}')
    cand_names = ['best_match', 'best_identity', 'best_flank_start', 'best_L',
                   'orient_fwd_frac', 'junction_dist', 'L_ratio']
    report_feature_medians(pos_feats['candidate_summary'],
                            {p: f['candidate_summary'] for p, f in neg_feats_by_prof.items()},
                            cand_names)

    print(f'\n{"="*105}')
    print(f'  PHASE B — Layout medians  (4 features)')
    print(f'{"="*105}')
    layout_names = ['flank_len', 'n_ncs', 'active_nc_len', 'ncs_max_len']
    report_feature_medians(pos_feats['layout'],
                            {p: f['layout'] for p, f in neg_feats_by_prof.items()},
                            layout_names)

    # Phase C: per-profile AUROC
    print(f'\n{"="*105}')
    print(f'  PHASE C — AUROC (V4.2 positives vs each negative profile)')
    print(f'{"="*105}')
    print(f'  Threshold reminders:')
    print(f'    layout / marginal / candidate_summary : each < 0.65 = healthy')
    print(f'    "all" (combined) : dumb features should not exceed ~0.75 (else model can shortcut)')
    print()
    feature_sets = ['layout', 'flank_marginal', 'nc_marginal',
                    'combined_marginal', 'candidate_summary', 'all']
    print(f'  {"profile":<36} ', end='')
    for fs in feature_sets:
        print(f'{fs[:12]:>16}', end='')
    print()
    print(f'  {"":<36} ', end='')
    for _ in feature_sets:
        print(f'{"LR / RF":>16}', end='')
    print()
    for prof, feats in neg_feats_by_prof.items():
        print(f'  {prof:<36} ', end='')
        for fs in feature_sets:
            X_pos = pos_feats[fs]; X_neg = feats[fs]
            au_lr, au_rf = train_auroc(X_pos, X_neg)
            print(f'  {au_lr:.3f}/{au_rf:.3f}', end='')
        print()

    # Phase D: profile classifier on negatives only
    profile_classifier_auroc(neg_feats_by_prof)

    print(f'\n{"="*105}')
    print(f'  DECISION KEY')
    print(f'{"="*105}')
    print(f'  A profile with AUROC > 0.75 on layout / marginal / candidate is TOO EASY.')
    print(f'  A profile with balanced feature medians and combined-AUROC ~0.65 is HEALTHY.')
    print(f'  Phase D accuracy >> 0.75 → negative recipes are distinguishable — risk that')
    print(f'    model learns "which recipe" instead of "cognate interaction". Consider')
    print(f'    reducing profile-specific fingerprints (or removing profile-diagnostic')
    print(f'    features from generator).')


if __name__ == '__main__':
    main()
