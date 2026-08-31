"""Create TNP-level train/val/test split for the V4.2 dataset.

Unit = transposase_id. Every counterfactual negative inherits the split of
its parent positive TNP automatically (since site_id is a suffix of parent
site_id, and transposase_id is preserved).

Stratifies on:
  bag_size            (number of sites per bag)
  active_nc_len       (median across sites in bag)
  L_mode              (modal guide_length across sites in bag)
  mean_identity       (median identity across sites in bag)

Split ratios: 70 / 15 / 15 → 3500 / 750 / 750 for 5000 TNPs.

Fixed seed. Persisted as JSON with a snapshot of stratification stats per
split so future runs can verify no drift.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEFAULT_POS = '/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl'
DEFAULT_OUT = '/global/scratch/users/kh36969/DL_novel_guide_editor/splits/splits_v42.json'


def compute_per_bag_features(pos_path):
    """Aggregate per-tnp features across sites for stratification."""
    by_tnp = defaultdict(list)
    with open(pos_path) as f:
        for line in f:
            r = json.loads(line)
            tnp = r['transposase_id']
            ncs = r['inputs']['noncoding_regions']
            active = r['labels'].get('active_noncoding_index', 0)
            active_len = len(ncs[active]) if active < len(ncs) else 0
            L = r['labels'].get('guide_length', 0)
            mm = r['labels'].get('n_mismatches', 0)
            identity = (L - mm) / max(1, L) if L else 0
            by_tnp[tnp].append({
                'active_nc_len': active_len,
                'L': L,
                'identity': identity,
            })
    out = {}
    for tnp, rows in by_tnp.items():
        out[tnp] = {
            'bag_size': len(rows),
            'active_nc_len_med': float(np.median([r['active_nc_len'] for r in rows])),
            'L_mode': int(Counter([r['L'] for r in rows]).most_common(1)[0][0]),
            'mean_identity': float(np.median([r['identity'] for r in rows])),
        }
    return out


def _stratum_key(feats):
    """Coarse stratum label for a TNP."""
    L_bucket = 'S' if feats['L_mode'] <= 10 else 'M' if feats['L_mode'] <= 13 else 'L'
    id_bucket = 'lo' if feats['mean_identity'] < 0.70 else 'hi'
    nc_bucket = 'short' if feats['active_nc_len_med'] < 220 else 'long'
    return f'{L_bucket}_{id_bucket}_{nc_bucket}'


def stratified_split(features_by_tnp, ratios=(0.70, 0.15, 0.15), seed=0):
    """Group by stratum, then split each stratum by the ratio."""
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for tnp, feats in features_by_tnp.items():
        by_stratum[_stratum_key(feats)].append(tnp)
    train, val, test = [], [], []
    for stratum, tnps in by_stratum.items():
        rng.shuffle(tnps)
        n = len(tnps)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        train.extend(tnps[:n_train])
        val.extend(tnps[n_train:n_train + n_val])
        test.extend(tnps[n_train + n_val:])
    return train, val, test, by_stratum


def summarize_split(name, tnps, features):
    fset = [features[t] for t in tnps]
    return {
        'name': name,
        'n_tnps': len(tnps),
        'bag_size_median': float(np.median([f['bag_size'] for f in fset])),
        'active_nc_len_median': float(np.median([f['active_nc_len_med'] for f in fset])),
        'L_mode_dist': dict(Counter([f['L_mode'] for f in fset]).most_common(20)),
        'mean_identity_median': float(np.median([f['mean_identity'] for f in fset])),
        'stratum_dist': dict(Counter([_stratum_key(f) for f in fset]).most_common(20)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default=DEFAULT_POS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    print(f'[load] {args.pos}', flush=True)
    features = compute_per_bag_features(args.pos)
    print(f'  {len(features)} unique TNPs', flush=True)

    train, val, test, by_stratum = stratified_split(features, seed=args.seed)
    print(f'\n[split] train={len(train)} val={len(val)} test={len(test)}', flush=True)
    print(f'  strata:')
    for s, tnps in sorted(by_stratum.items()):
        print(f'    {s:<18}  {len(tnps):>4} tnps', flush=True)

    manifest = {
        'seed': args.seed,
        'positive_source': args.pos,
        'unit': 'transposase_id',
        'ratios': [0.70, 0.15, 0.15],
        'stratification': ['L_mode', 'mean_identity', 'active_nc_len_med'],
        'train': sorted(train),
        'val': sorted(val),
        'test': sorted(test),
        'summary': {
            'train': summarize_split('train', train, features),
            'val':   summarize_split('val', val, features),
            'test':  summarize_split('test', test, features),
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f'\n[out] {out_path}', flush=True)

    # Also print a compact drift check
    print(f'\n  stratification drift check:')
    print(f'  {"split":<8} {"n":>5} {"bag_med":>8} {"nc_med":>7} {"id_med":>7}')
    for s in ('train', 'val', 'test'):
        d = manifest['summary'][s]
        print(f'  {s:<8} {d["n_tnps"]:>5} {d["bag_size_median"]:>8.1f} '
              f'{d["active_nc_len_median"]:>7.1f} {d["mean_identity_median"]:>7.3f}')


if __name__ == '__main__':
    main()
