"""Build train/val/test JSONL for experiment 48C0.

Positives + BALANCED 5-profile counterfactual negatives (no structure):
  paired_shuffle_v42        -> tnp += '__neg_paired_shuffle_v42'
  wrong_orientation_v42     -> tnp += '__neg_wrong_orientation_v42'
  wrong_position_v42        -> tnp += '__neg_wrong_position_v42'
  wrong_length_v42          -> tnp += '__neg_wrong_length_v42'
  wrong_structure_role_v42  -> tnp += '__neg_wrong_structure_role_v42'

Each profile keeps its parent's split (via splits_v42.json), giving 5
disjoint neg-TNP sets per split. So POS:NEG bag ratio = 1:5. Total records
also 1:5 since each profile mirrors the positive set 1:1 in records.

We keep the ratio 1:5 (not downsample) so 48C0 sees the same number of
positives as 48B, and 5x as many negatives; positive class weight is
applied at loss time by the trainer already.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


NEG_ROOT = '/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives_v42_counterfactual'
PROFILES = [
    ('paired_shuffle_v42',       f'{NEG_ROOT}/negatives_v42_paired_shuffle.jsonl'),
    ('wrong_orientation_v42',    f'{NEG_ROOT}/negatives_v42_wrong_orientation.jsonl'),
    ('wrong_position_v42',       f'{NEG_ROOT}/negatives_v42_wrong_position.jsonl'),
    ('wrong_length_v42',         f'{NEG_ROOT}/negatives_v42_wrong_length.jsonl'),
    ('wrong_structure_role_v42', f'{NEG_ROOT}/negatives_v42_wrong_structure_role.jsonl'),
]


def sanitize(rec, tnp_suffix=None):
    """Strip audit-only metadata; for negatives, mangle tnp id so TnpDataset
    treats POS/NEG (and each NEG profile) as separate bags."""
    tnp = rec['transposase_id']
    parent_tnp = tnp
    if tnp_suffix:
        tnp = tnp + tnp_suffix
    out = {
        'site_id': rec['site_id'],
        'transposase_id': tnp,
        'parent_transposase_id': parent_tnp,
        'ncrna_id': rec.get('ncrna_id'),
        'inputs': {
            'flank': rec['inputs']['flank'],
            'noncoding_regions': rec['inputs']['noncoding_regions'],
        },
        'labels': {
            'is_positive': bool(rec['labels']['is_positive']),
            'active_noncoding_index': rec['labels'].get('active_noncoding_index'),
            'num_noncoding_regions': rec['labels'].get(
                'num_noncoding_regions', len(rec['inputs']['noncoding_regions'])),
            'site_class': rec['labels'].get('site_class', 'guided'),
            'ncrna_length': rec['labels'].get('ncrna_length'),
            'violation_profile': rec['labels'].get('violation_profile'),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default='/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl')
    ap.add_argument('--splits', default='/global/scratch/users/kh36969/DL_novel_guide_editor/splits/splits_v42.json')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--train-neg-frac-per-profile', type=float, default=0.20,
                    help='Fraction of train pos TNPs to keep per profile (0.20 → 700/3500). '
                         'Val/test unchanged. 1.0 disables downsampling.')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    tnp2split = {}
    for k in ('train', 'val', 'test'):
        for t in splits[k]:
            tnp2split[t] = k
    print(f'[splits] train={len(splits["train"])} val={len(splits["val"])} test={len(splits["test"])}', flush=True)

    # Per-profile downsampled train tnp subsets (RNG per profile so subsets differ)
    train_tnps = sorted(splits['train'])
    n_keep = int(round(len(train_tnps) * args.train_neg_frac_per_profile))
    train_neg_tnps_by_profile = {}
    for i, (prof, _) in enumerate(PROFILES):
        rng = random.Random(args.seed + i)
        kept = set(rng.sample(train_tnps, n_keep))
        train_neg_tnps_by_profile[prof] = kept
    print(f'[downsample] train negs: kept {n_keep}/{len(train_tnps)} pos-tnps per profile', flush=True)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    handles = {k: (out_dir / f'{k}.jsonl').open('w') for k in ('train', 'val', 'test')}
    counts = {k: {'pos': 0} for k in ('train', 'val', 'test')}
    for prof, _ in PROFILES:
        for k in ('train', 'val', 'test'):
            counts[k][prof] = 0

    def _process(path, kind, suffix=None, train_keep_tnps=None):
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                tnp = r['transposase_id']
                if tnp not in tnp2split: continue
                split = tnp2split[tnp]
                # Downsample only in TRAIN split, only for negs (train_keep_tnps set)
                if split == 'train' and train_keep_tnps is not None:
                    if tnp not in train_keep_tnps: continue
                clean = sanitize(r, tnp_suffix=suffix)
                handles[split].write(json.dumps(clean) + '\n')
                counts[split][kind] += 1

    print(f'[process] positives from {args.pos}', flush=True)
    _process(args.pos, 'pos', suffix=None)
    for prof, path in PROFILES:
        print(f'[process] {prof} from {path}', flush=True)
        _process(path, prof, suffix=f'__neg_{prof}',
                 train_keep_tnps=train_neg_tnps_by_profile[prof])

    for h in handles.values(): h.close()
    print(f'\n[out] {out_dir}')
    for k, c in counts.items():
        total = sum(c.values())
        pos_frac = c['pos'] / max(1, total)
        line = f'  {k}: pos={c["pos"]}  '
        for prof, _ in PROFILES:
            line += f' {prof.split("_v42")[0]}={c[prof]}'
        line += f'  total={total}  pos_frac={pos_frac:.2%}'
        print(line)


if __name__ == '__main__':
    main()
