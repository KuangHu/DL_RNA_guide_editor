"""48A summary-features MLP baseline.

Reads per-site summary features (from extract_summary_features.py), aggregates
to bag level, trains a small MLP with:

  positives      : all POS records for train TNPs
  negatives      : balanced 5-profile mix (20% each) from same train TNPs

Evaluates on val + test (per-profile AUROC + paired Δscore + overall).

Design: no candidate-level nn, no MIL, no structure. This is the "what can
you do with summary alone?" ceiling.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np

FEAT_DIR = '/global/scratch/users/kh36969/DL_novel_guide_editor/features_v42'
SPLITS = '/global/scratch/users/kh36969/DL_novel_guide_editor/splits/splits_v42.json'
PROFILE_TAGS = ('shuffle', 'wrongori', 'wrongpos', 'wronglen', 'wrongstr')
PROFILE_TO_LABEL = {
    'shuffle':  'paired_shuffle_v42',
    'wrongori': 'wrong_orientation_v42',
    'wrongpos': 'wrong_position_v42',
    'wronglen': 'wrong_length_v42',
    'wrongstr': 'wrong_structure_role_v42',
}


def load_features(tag):
    p = f'{FEAT_DIR}/feats_{tag}.npz'
    d = np.load(p, allow_pickle=True)
    return {'X': d['X'], 'y': d['y'],
             'site_ids': d['site_ids'].astype(str),
             'tnp_ids': d['tnp_ids'].astype(str),
             'feature_names': d['feature_names'].astype(str)}


def aggregate_bag(X_site, feature_names):
    """X_site: (n_sites, n_feats). Return (n_bag_feats,) aggregate vector."""
    F = X_site
    # mean, max, min, std per feature; plus bag size
    aggs = np.concatenate([
        F.mean(axis=0),
        F.max(axis=0),
        F.min(axis=0),
        F.std(axis=0),
    ]).astype(np.float32)
    return np.concatenate([aggs, [len(F)]]).astype(np.float32)


def build_bag_dataset(recs_by_tag, feat_names, tnps, tag_labels):
    """recs_by_tag: {tag: dict(X, site_ids, tnp_ids)}. tag_labels: {tag: int}.
    tnps: iterable of TNP ids to include.
    Returns X (n_bags, n_bag_feats), y, tnp_ids, parent_site_ids_by_tag."""
    tnps = set(tnps)
    X_bags = []; y_bags = []; tnp_bags = []; profile_bags = []
    for tag, data in recs_by_tag.items():
        label = tag_labels[tag]
        # group by tnp
        by_tnp = defaultdict(list)
        for i in range(len(data['X'])):
            t = data['tnp_ids'][i]
            if t not in tnps: continue
            by_tnp[t].append(i)
        for t, idxs in by_tnp.items():
            X_bag = aggregate_bag(data['X'][idxs], feat_names)
            X_bags.append(X_bag); y_bags.append(label); tnp_bags.append(t)
            profile_bags.append(tag)
    return (np.stack(X_bags, axis=0),
            np.asarray(y_bags, dtype=np.int8),
            np.asarray(tnp_bags), np.asarray(profile_bags))


def _auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(set(labels)) < 2: return float('nan')
    return roc_auc_score(labels, scores)


def _auprc(scores, labels):
    from sklearn.metrics import average_precision_score
    if len(set(labels)) < 2: return float('nan')
    return average_precision_score(labels, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/global/scratch/users/kh36969/DL_novel_guide_editor/results/48a_summary_mlp.json')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--n-per-neg-train', type=int, default=None,
                    help='Cap per-profile negatives in training for speed')
    args = ap.parse_args()

    splits = json.loads(open(SPLITS).read())
    print(f'[splits] train={len(splits["train"])} val={len(splits["val"])} '
          f'test={len(splits["test"])}', flush=True)

    print(f'[load] positive features', flush=True)
    pos = load_features('pos')
    feat_names = pos['feature_names']
    neg_by_tag = {}
    for tag in PROFILE_TAGS:
        print(f'[load] {tag}', flush=True)
        neg_by_tag[tag] = load_features(tag)

    # Balanced 5-profile: each profile contributes equally; positives get all
    tag_labels = {'pos': 1, **{t: 0 for t in PROFILE_TAGS}}

    train_recs = {'pos': pos, **neg_by_tag}
    print(f'[build] train bags ...', flush=True)
    X_tr, y_tr, tnp_tr, prof_tr = build_bag_dataset(
        train_recs, feat_names, splits['train'], tag_labels)

    print(f'[build] val bags ...', flush=True)
    X_val, y_val, tnp_val, prof_val = build_bag_dataset(
        train_recs, feat_names, splits['val'], tag_labels)

    print(f'[build] test bags ...', flush=True)
    X_te, y_te, tnp_te, prof_te = build_bag_dataset(
        train_recs, feat_names, splits['test'], tag_labels)

    print(f'\n[shapes] train X={X_tr.shape}  y_pos={(y_tr==1).sum()}  y_neg={(y_tr==0).sum()}', flush=True)
    print(f'         val   X={X_val.shape}  y_pos={(y_val==1).sum()}  y_neg={(y_val==0).sum()}', flush=True)
    print(f'         test  X={X_te.shape}  y_pos={(y_te==1).sum()}  y_neg={(y_te==0).sum()}', flush=True)

    # Standardize
    mu = X_tr.mean(axis=0); sd = X_tr.std(axis=0); sd[sd == 0] = 1
    def _norm(A): return (A - mu) / sd
    Xt_n = _norm(X_tr); Xv_n = _norm(X_val); Xe_n = _norm(X_te)

    # Weight positives to balance the 5:1 ratio (5 profiles vs 1 pos)
    n_pos = (y_tr == 1).sum(); n_neg = (y_tr == 0).sum()
    sample_weight = np.where(y_tr == 1, n_neg / max(1, n_pos), 1.0)
    print(f'[weight] pos_weight={n_neg / max(1, n_pos):.3f}', flush=True)

    # Try both linear LR and small MLP
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier

    print(f'\n[train] LogisticRegression ...', flush=True)
    lr = LogisticRegression(max_iter=2000, solver='lbfgs')
    lr.fit(Xt_n, y_tr, sample_weight=sample_weight)
    p_val_lr = lr.predict_proba(Xv_n)[:, 1]
    p_te_lr = lr.predict_proba(Xe_n)[:, 1]

    print(f'[train] MLPClassifier (32, 32) ...', flush=True)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 32), activation='relu',
                         solver='lbfgs', max_iter=800, random_state=args.seed,
                         alpha=1e-2)
    mlp.fit(Xt_n, y_tr)  # MLPClassifier doesn't accept sample_weight in sklearn
    p_val_mlp = mlp.predict_proba(Xv_n)[:, 1]
    p_te_mlp = mlp.predict_proba(Xe_n)[:, 1]

    def _report(name, p_val, p_te):
        r = {'name': name}
        for split_name, p, y, prof, tnp in [
            ('val', p_val, y_val, prof_val, tnp_val),
            ('test', p_te, y_te, prof_te, tnp_te),
        ]:
            r[split_name] = {
                'overall_auroc': _auroc(p, y),
                'overall_auprc': _auprc(p, y),
                'per_profile_auroc': {},
                'paired_delta': {},
            }
            # Per profile: POS vs one profile
            pos_mask = (prof == 'pos')
            for tag in PROFILE_TAGS:
                neg_mask = (prof == tag)
                mask = pos_mask | neg_mask
                sub_scores = p[mask]; sub_labels = y[mask]
                r[split_name]['per_profile_auroc'][tag] = _auroc(sub_scores, sub_labels)

                # Paired Δ: match by parent tnp (unique tnp between pos-bag and profile-bag)
                # Both have same tnp_id since counterfactual inherits
                pos_by_tnp = {tnp[i]: p[i] for i in range(len(tnp)) if pos_mask[i]}
                neg_by_tnp = {tnp[i]: p[i] for i in range(len(tnp)) if neg_mask[i]}
                deltas = []
                for t, ps in pos_by_tnp.items():
                    if t in neg_by_tnp:
                        deltas.append(float(ps - neg_by_tnp[t]))
                deltas = np.asarray(deltas)
                if len(deltas):
                    r[split_name]['paired_delta'][tag] = {
                        'n': int(len(deltas)),
                        'median': float(np.median(deltas)),
                        'q10': float(np.quantile(deltas, 0.10)),
                        'q75': float(np.quantile(deltas, 0.75)),
                        'p_gt_0': float((deltas > 0).mean()),
                    }
        return r

    reports = [_report('LogReg', p_val_lr, p_te_lr),
                _report('MLP', p_val_mlp, p_te_mlp)]

    print(f'\n{"="*95}')
    print(f'  48A SUMMARY MLP RESULTS')
    print(f'{"="*95}')
    for r in reports:
        print(f'\n  === {r["name"]} ===')
        for split_name in ('val', 'test'):
            s = r[split_name]
            print(f'    [{split_name}] overall AUROC={s["overall_auroc"]:.4f}  '
                  f'AUPRC={s["overall_auprc"]:.4f}')
            print(f'      per-profile AUROC:')
            for tag, au in s['per_profile_auroc'].items():
                print(f'        {tag:<10} {au:.4f}')
            print(f'      paired Δ (score(POS_i) - score(NEG_i^p)):')
            for tag, d in s['paired_delta'].items():
                print(f'        {tag:<10} n={d["n"]:>4}  '
                      f'med={d["median"]:>+.3f}  '
                      f'Q10={d["q10"]:>+.3f}  '
                      f'P(Δ>0)={d["p_gt_0"]:.3f}')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump({'experiment': '48A_summary_mlp',
                    'seed': args.seed,
                    'reports': reports}, f, indent=2)
    print(f'\n[out] {args.out}', flush=True)


if __name__ == '__main__':
    main()
