"""LEVEL 1 of the Gold Signal Localization Ladder.

Question: given the CURRENT candidate generator (one contiguous RNA segment
↔ one contiguous DNA segment, L in [5..16], top-K per (orient, L)) — does
its POOL contain any candidate that separates Durrant cognate from Durrant
shuffled? No neural network involved.

If oracle AUROC ≈ 0.5–0.6 → the candidate grammar itself cannot represent
the LtG+core+RtG biological object. Fixing pair_head weights won't help.
The V7 candidate representation must be rewritten (dual-arm).

If oracle AUROC ≈ 0.9+ → the signal exists in the current candidate pool.
Then the loss is at neural selection / aggregation / propagation. Fixing
the model architecture (not the candidate grammar) is enough.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

from preprocess.candidates import build_candidate_arrays


COG_JSONL = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_cognate.jsonl')
SHU_JSONL = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_shuffled.jsonl')


def _auroc(scores, labels):
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float('nan')
    order = np.argsort(-scores, kind='mergesort')
    y = labels[order]
    tps = np.cumsum(y); fps = np.cumsum(~y)
    tps = np.concatenate([[0], tps]); fps = np.concatenate([[0], fps])
    tpr = tps / max(1, tps[-1]); fpr = fps / max(1, fps[-1])
    return float(np.trapezoid(tpr, fpr))


def audit_split(jsonl_path, label):
    """For each site in the JSONL, build candidate pool and return per-site
    oracle features. `label` = 'cognate' or 'shuffled'."""
    records = [json.loads(l) for l in Path(jsonl_path).read_text().splitlines()]
    print(f'[{label}] {len(records)} sites')
    rows = []
    for i, rec in enumerate(records):
        nc = rec['inputs']['noncoding_regions'][rec['labels']['active_noncoding_index']]
        flank = rec['inputs']['flank']
        # Dummy structure profile: zero unpaired probs (only affects patches, not selection)
        u_max = 16
        prof = np.zeros((len(nc), u_max), dtype=np.float32)
        valid = np.ones((len(nc), u_max), dtype=bool)
        patches, feats, mask, cands = build_candidate_arrays(
            nc=nc, flank=flank,
            structure_profile=prof, structure_valid=valid,
            top_k_per_combo=4, L_min=5, L_max=16,
            orientations=('fwd', 'rc'),
            patch_width=64, nc_max=350,
        )
        # cands is a list of Candidate | None (K_max = 2 orient × 12 L × 4 K = 96)
        real = [c for c in cands if c is not None]

        # Per site oracle features
        best = {'max_matches': -1, 'max_ident': -1., 'max_ident_L8_14': -1.,
                'best_L': None, 'best_orient': None, 'best_L_at_junction': None,
                'best_matches_at_junction': -1, 'best_ident_at_junction': -1.,
                'best_ident_L11': -1., 'best_matches_L11': -1,
                'best_ident_L11_at_junction': -1., 'best_matches_L11_at_junction': -1}
        for c in real:
            ident = c.matches / c.L
            if c.matches > best['max_matches']:
                best['max_matches'] = c.matches
                best['best_L'] = c.L
                best['best_orient'] = c.orient
            if ident > best['max_ident']:
                best['max_ident'] = ident
            if 8 <= c.L <= 14 and ident > best['max_ident_L8_14']:
                best['max_ident_L8_14'] = ident
            # Junction-restricted: alignment lies inside flank[0..15] (downstream junction)
            if c.flank_start <= 5:
                if c.matches > best['best_matches_at_junction']:
                    best['best_matches_at_junction'] = c.matches
                    best['best_L_at_junction'] = c.L
                if ident > best['best_ident_at_junction']:
                    best['best_ident_at_junction'] = ident
            # L=11 specific (IS110 target = 11bp) — anywhere and at junction
            if c.L == 11:
                if ident > best['best_ident_L11']:
                    best['best_ident_L11'] = ident
                    best['best_matches_L11'] = c.matches
                if c.flank_start <= 5:
                    if ident > best['best_ident_L11_at_junction']:
                        best['best_ident_L11_at_junction'] = ident
                        best['best_matches_L11_at_junction'] = c.matches
        rec_row = {'label': label, 'site_id': rec['site_id'], 'bag_id': rec['transposase_id'],
                   'n_cands': len(real), **best}
        rows.append(rec_row)
        if (i + 1) % 100 == 0:
            print(f'  processed {i + 1}/{len(records)}')
    return rows


def compare(cog_rows, shu_rows):
    print(f'\n{"="*100}')
    print(f'  Level 1 oracle: does the current candidate pool contain any candidate that')
    print(f'  separates cognate from shuffled? For each feature, higher = better pairing.')
    print(f'{"="*100}\n')
    metrics = [
        ('max_matches',                 'best matches (any L, any orient, anywhere)'),
        ('max_ident',                   'best identity (any L, any orient, anywhere)'),
        ('max_ident_L8_14',             'best identity restricted to L in [8,14]'),
        ('best_ident_L11',              'best identity at L=11 (IS110 target length, anywhere)'),
        ('best_matches_L11',            'best matches at L=11 (anywhere)'),
        ('best_matches_at_junction',    'best matches within 5bp of junction (any L)'),
        ('best_ident_at_junction',      'best identity within 5bp of junction (any L)'),
        ('best_ident_L11_at_junction',  'best identity, L=11, within 5bp of junction'),
        ('best_matches_L11_at_junction','best matches, L=11, within 5bp of junction'),
    ]
    print(f'  {"feature":<40} {"cog_med":>9} {"shu_med":>9} {"cog_mean":>10} '
          f'{"shu_mean":>10}   {"AUROC":>7}   {"P(c>s)":>7}')
    print('  ' + '-' * 100)
    for key, desc in metrics:
        cog_v = np.asarray([r[key] for r in cog_rows], dtype=float)
        shu_v = np.asarray([r[key] for r in shu_rows], dtype=float)
        # some -1 sentinels for unmatched — set to nan for stats
        cog_v = np.where(cog_v < 0, np.nan, cog_v)
        shu_v = np.where(shu_v < 0, np.nan, shu_v)
        cog_ok = cog_v[~np.isnan(cog_v)]
        shu_ok = shu_v[~np.isnan(shu_v)]
        if len(cog_ok) < 5 or len(shu_ok) < 5:
            print(f'  {desc:<40}  (insufficient data: n_cog={len(cog_ok)}, n_shu={len(shu_ok)})')
            continue
        scores = np.concatenate([cog_ok, shu_ok])
        labels = np.concatenate([np.ones(len(cog_ok)), np.zeros(len(shu_ok))])
        au = _auroc(scores, labels)
        n_p = min(len(cog_ok), len(shu_ok))
        p_c_gt_s = float(np.mean(np.random.default_rng(42).permutation(cog_ok)[:n_p]
                                > np.random.default_rng(43).permutation(shu_ok)[:n_p]))
        print(f'  {desc:<40} {np.nanmedian(cog_v):>+9.3f} {np.nanmedian(shu_v):>+9.3f} '
              f'{np.nanmean(cog_v):>+10.3f} {np.nanmean(shu_v):>+10.3f}   '
              f'{au:>7.4f}   {p_c_gt_s:>7.3f}')
    print()
    # Also: what is the "best L" distribution in cognate vs shuffled?
    print(f'  Best-L distribution (L of max-matches candidate) — cognate vs shuffled:')
    from collections import Counter
    for label, rows in [('cognate', cog_rows), ('shuffled', shu_rows)]:
        c = Counter(r['best_L'] for r in rows if r['best_L'])
        top = sorted(c.items())
        s = '  '.join(f'L{L}:{n:>3}' for L, n in top)
        print(f'    {label:<10} {s}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-json', default=None)
    args = p.parse_args()
    cog_rows = audit_split(COG_JSONL, 'cognate')
    shu_rows = audit_split(SHU_JSONL, 'shuffled')
    compare(cog_rows, shu_rows)
    if args.out_json:
        with open(args.out_json, 'w') as f:
            json.dump({'cognate': cog_rows, 'shuffled': shu_rows}, f, indent=2)
        print(f'\n[out] {args.out_json}')


if __name__ == '__main__':
    main()
