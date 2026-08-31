"""Oracle-style dispersion probe.

Reads per-site selected-candidate rows from
   logs/diag_v4_candidate_selection.jsonl
and asks: **do simple per-tnp dispersion features (std/MAD of the selected
target position, orientation entropy, L variance) already separate positives
from Level-2 negatives?**

If yes with high AUROC, the signal is present in the current candidate picks —
V1's Set Transformer just isn't extracting it. That's the go-ahead to build V5
with explicit cross-site relation features.

Reports for each feature x per (positive vs each Level-2 profile):
  AUROC (with x as score, positives = class 1)
  If AUROC < 0.5, that's evidence x correlates with negatives (higher x -> more
  likely to be a wrong_position derangement). We report both raw and the
  discrimination power |AUROC - 0.5| + 0.5.

No GPU. Runs on login node in seconds.
"""
from __future__ import annotations

import collections
import json
import math

import numpy as np

ROWS_PATH = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/diag_v4_candidate_selection.jsonl'


def _auroc(scores_pos, scores_neg):
    """Trapezoidal AUROC with pos=class 1, neg=class 0. Higher score => more likely positive."""
    s = np.concatenate([np.asarray(scores_pos, dtype=np.float64),
                        np.asarray(scores_neg, dtype=np.float64)])
    y = np.concatenate([np.ones(len(scores_pos), dtype=np.int8),
                        np.zeros(len(scores_neg), dtype=np.int8)])
    order = np.argsort(-s, kind='mergesort')
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tps = np.concatenate([[0], tps])
    fps = np.concatenate([[0], fps])
    tpr = tps / max(1, tps[-1])
    fpr = fps / max(1, fps[-1])
    return float(np.trapezoid(tpr, fpr))


def mad(vals):
    a = np.asarray(vals, dtype=np.float64)
    med = np.median(a)
    return float(np.median(np.abs(a - med)))


def entropy(counter):
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return float(-sum((c / total) * math.log2(c / total) for c in counter.values() if c > 0))


def main():
    per_tnp = collections.defaultdict(list)
    tnp_info: dict[str, dict] = {}
    n_rows = 0
    with open(ROWS_PATH) as f:
        for line in f:
            r = json.loads(line)
            n_rows += 1
            tid = r['tnp_id']
            if r.get('selected_slot', -1) < 0:
                continue
            per_tnp[tid].append(r)
            if tid not in tnp_info:
                tnp_info[tid] = {
                    'is_positive': r['is_positive'],
                    'violation_profile': r['violation_profile'],
                }
    print(f'loaded {n_rows} site rows, {len(per_tnp)} tnps with selected slots')

    # Per-tnp features
    features = []
    for tid, sites in per_tnp.items():
        if len(sites) < 5:
            continue
        # model-picked axes
        mtp = np.asarray([s['model_target_start'] for s in sites], dtype=np.float64)
        mnc = np.asarray([s['model_nc_start'] for s in sites], dtype=np.float64)
        mL = np.asarray([s['model_L'] for s in sites], dtype=np.float64)
        m_orient = [s['model_orient'] for s in sites]
        # oracle axes: labelled truth
        ttp = np.asarray([s['true_target_start'] for s in sites
                          if s.get('true_target_start') is not None], dtype=np.float64)
        tnc = np.asarray([s['true_guide_start'] for s in sites
                          if s.get('true_guide_start') is not None], dtype=np.float64)
        tL = np.asarray([s['true_L'] for s in sites
                         if s.get('true_L') is not None], dtype=np.float64)
        t_orient = [s['true_orient'] for s in sites if s.get('true_orient')]

        info = tnp_info[tid]
        feat = {
            'tnp_id': tid,
            'is_positive': info['is_positive'],
            'violation_profile': info['violation_profile'],
            # model-picked dispersion
            'm_pos_std': float(mtp.std()),
            'm_pos_mad': mad(mtp),
            'm_pos_iqr': float(np.percentile(mtp, 75) - np.percentile(mtp, 25)),
            'm_ncstart_std': float(mnc.std()),
            'm_L_std': float(mL.std()),
            'm_orient_entropy': entropy(collections.Counter(m_orient)),
            # oracle (label-based) dispersion
            'o_pos_std': float(ttp.std()) if len(ttp) >= 5 else None,
            'o_pos_mad': mad(ttp) if len(ttp) >= 5 else None,
            'o_pos_iqr': float(np.percentile(ttp, 75) - np.percentile(ttp, 25)) if len(ttp) >= 5 else None,
            'o_ncstart_std': float(tnc.std()) if len(tnc) >= 5 else None,
            'o_L_std': float(tL.std()) if len(tL) >= 5 else None,
            'o_orient_entropy': entropy(collections.Counter(t_orient)),
        }
        features.append(feat)

    # Split by class
    POS = [f for f in features if f['is_positive']]
    NEG_by_prof: dict[str, list] = collections.defaultdict(list)
    for f in features:
        if not f['is_positive']:
            NEG_by_prof[f['violation_profile'] or 'unknown'].append(f)

    print(f'\ncomparison groups: POS={len(POS)}, ' +
          ', '.join(f'{p}={len(v)}' for p, v in NEG_by_prof.items()))

    print()
    print('=' * 100)
    print('AUROC per feature vs each profile comparison')
    print('(discriminating power |AUROC-0.5|+0.5 in parens; direction: HI=pos means high value -> more positive)')
    print('=' * 100)

    feature_axes = [
        # (name, description, oracle_or_model)
        ('m_pos_std',        'model-picked target-position std',           'model'),
        ('m_pos_mad',        'model-picked target-position MAD',           'model'),
        ('m_pos_iqr',        'model-picked target-position IQR',           'model'),
        ('m_ncstart_std',    'model-picked NC-start std',                  'model'),
        ('m_L_std',          'model-picked L std',                         'model'),
        ('m_orient_entropy', 'model-picked orientation entropy',           'model'),
        ('o_pos_std',        'ORACLE target-position std (labels)',        'oracle'),
        ('o_pos_mad',        'ORACLE target-position MAD (labels)',        'oracle'),
        ('o_ncstart_std',    'ORACLE guide NC-start std (labels)',         'oracle'),
        ('o_L_std',          'ORACLE guide-length std (labels)',           'oracle'),
        ('o_orient_entropy', 'ORACLE orientation entropy (labels)',        'oracle'),
    ]

    target_profiles = [
        'wrong_position_consistency',
        'wrong_structure_role_consistency',
        'wrong_orientation_consistency',
        'wrong_length_consistency',
        'level1_marginal_matched',
        'level3_paired_counterfactual',
    ]

    header = f'{"feature":<50} ' + ' '.join(f'{p[:16]:>17}' for p in target_profiles)
    print(header)
    for fname, fdesc, ftype in feature_axes:
        row = f'{fdesc:<50}'
        for prof in target_profiles:
            neg_group = NEG_by_prof.get(prof, [])
            if not neg_group:
                row += f' {"n/a":>17}'
                continue
            pos_vals = [f[fname] for f in POS if f.get(fname) is not None]
            neg_vals = [f[fname] for f in neg_group if f.get(fname) is not None]
            if not pos_vals or not neg_vals:
                row += f' {"n/a":>17}'
                continue
            auroc = _auroc(pos_vals, neg_vals)
            power = abs(auroc - 0.5) + 0.5
            direction = 'HI=pos' if auroc >= 0.5 else 'LO=pos'
            row += f' {auroc:.3f}[{power:.2f}]{direction[:2]}'
            # pad to width 17
            # (a small hack: keep formatting concise)
        print(row)

    # Print a compact summary highlighting the crucial ones
    print()
    print('=' * 100)
    print('CRUCIAL SIGNALS (oracle-based, showing whether the signal exists at all)')
    print('=' * 100)
    for fname, fdesc, ftype in feature_axes:
        if ftype != 'oracle':
            continue
        pos_vals = [f[fname] for f in POS if f.get(fname) is not None]
        for prof in target_profiles:
            neg_vals = [f[fname] for f in NEG_by_prof.get(prof, []) if f.get(fname) is not None]
            if not neg_vals:
                continue
            auroc = _auroc(pos_vals, neg_vals)
            power = abs(auroc - 0.5) + 0.5
            if power >= 0.75:  # meaningful signal
                pos_med = float(np.median(pos_vals))
                neg_med = float(np.median(neg_vals))
                direction = 'higher=neg' if auroc < 0.5 else 'higher=pos'
                print(f'{fdesc:<45} vs {prof:<35} AUROC={auroc:.3f} '
                      f'(power={power:.2f}, {direction}, pos_med={pos_med:.2f} neg_med={neg_med:.2f})')


if __name__ == '__main__':
    main()
