"""Build train/val/test JSONL for experiment 48B.

Combines V4.2 positives + paired_shuffle_v42 negatives, split by TNP per
splits_v42.json. Each output JSONL contains records with `is_positive`
labels; the training loader (training/train_v1.py) already discriminates
by that field. Loader only reads inputs.flank + inputs.noncoding_regions
per project spec, but we still filter metadata fields here for safety.

Outputs:
  <out-dir>/train.jsonl
  <out-dir>/val.jsonl
  <out-dir>/test.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


NEG_TNP_SUFFIX = '__neg_paired_shuffle_v42'


def sanitize(rec, is_neg=False):
    """Strip audit-only metadata; keep only fields the loader may touch.
    For negative records, mangle transposase_id so the TNP-level index
    treats POS/NEG as separate bags (they otherwise collide by tnp_id
    and TnpDataset assigns one is_positive per tnp)."""
    tnp = rec['transposase_id']
    if is_neg:
        tnp = tnp + NEG_TNP_SUFFIX
    out = {
        'site_id': rec['site_id'],
        'transposase_id': tnp,
        'parent_transposase_id': rec['transposase_id'],
        'ncrna_id': rec.get('ncrna_id'),
        'inputs': {
            'flank': rec['inputs']['flank'],
            'noncoding_regions': rec['inputs']['noncoding_regions'],
        },
        'labels': {
            'is_positive': bool(rec['labels']['is_positive']),
            'active_noncoding_index': rec['labels'].get('active_noncoding_index'),
            'num_noncoding_regions': rec['labels'].get('num_noncoding_regions', len(rec['inputs']['noncoding_regions'])),
            'site_class': rec['labels'].get('site_class', 'guided'),
            # Keep ncrna_length only if needed by the loader
            'ncrna_length': rec['labels'].get('ncrna_length'),
            # For counterfactual negatives, expose the violation_profile in labels
            # for eval-time stratification (loader must ignore for model input).
            'violation_profile': rec['labels'].get('violation_profile'),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pos', default='/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives_v42.jsonl')
    ap.add_argument('--neg', default='/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives_v42_counterfactual/negatives_v42_paired_shuffle.jsonl')
    ap.add_argument('--splits', default='/global/scratch/users/kh36969/DL_novel_guide_editor/splits/splits_v42.json')
    ap.add_argument('--out-dir', required=True)
    args = ap.parse_args()

    splits = json.loads(Path(args.splits).read_text())
    tnp2split = {}
    for k in ('train', 'val', 'test'):
        for t in splits[k]:
            tnp2split[t] = k
    print(f'[splits] train={len(splits["train"])} val={len(splits["val"])} test={len(splits["test"])}')

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    handles = {k: (out_dir / f'{k}.jsonl').open('w') for k in ('train', 'val', 'test')}
    counts = {k: {'pos': 0, 'neg': 0} for k in ('train', 'val', 'test')}

    def _process(path, kind):
        is_neg = (kind == 'neg')
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                tnp = r['transposase_id']  # parent tnp; used for split lookup
                if tnp not in tnp2split: continue
                split = tnp2split[tnp]
                clean = sanitize(r, is_neg=is_neg)
                handles[split].write(json.dumps(clean) + '\n')
                counts[split][kind] += 1

    print(f'[process] positives from {args.pos}', flush=True)
    _process(args.pos, 'pos')
    print(f'[process] negatives from {args.neg}', flush=True)
    _process(args.neg, 'neg')

    for h in handles.values(): h.close()
    print(f'\n[out] {out_dir}')
    for k, c in counts.items():
        print(f'  {k}: pos={c["pos"]}  neg={c["neg"]}  total={c["pos"] + c["neg"]}')


if __name__ == '__main__':
    main()
