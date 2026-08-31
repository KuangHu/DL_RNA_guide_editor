"""Diagnostic: strength-group score distribution.

For each positive tnp in val_v3 + test_v3, extract:
  - the model's predicted P(positive)
  - tnp_strength   (strong / moderate / weak) from noisy_positives metadata
  - n_guided / n_off_target / n_unresolved counts (per-tnp constants in metadata)
  - num_noncoding_regions
  - mean best-alignment score across the 50 sites (proxy for raw alignment quality)

Report per-strength:
  - score quantiles Q10, Q25, median, Q75, Q90
  - other feature distributions to see if strength groups differ in ways
    that could explain the counterintuitive recall ordering.

If strong tnps have SYSTEMATICALLY LOWER scores than weak, and this
correlates with any generation feature (e.g. n_off_target proportion),
that's evidence of a generation artifact.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset


BASE = '/global/scratch/users/kh36969/DL_novel_guide_editor'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v1_on_v3/best.pt'


def score_split(split_name, model, device):
    cache = StructureCache(f'{BASE}/structure/{split_name}_v3_u16.index.json')
    ds = TnpGroupedDataset(f'{BASE}/splits/{split_name}_v3.jsonl', cache,
                            site_subsample_size=50, rng_seed=0)
    dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=8, shuffle=False,
                    num_workers=4,
                    collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
                    persistent_workers=True, pin_memory=True)
    scores, tnp_ids, is_pos = [], [], []
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(b['candidate_patches'], b['candidate_features'],
                             b['candidate_mask'], b['nc_region_mask'])
            scores.append(torch.sigmoid(out['logit']).float().cpu().numpy())
            tnp_ids.extend(list(b['tnp_id']))
            is_pos.append(b['is_positive'].cpu().numpy())
    return np.concatenate(scores), np.concatenate(is_pos), tnp_ids


def per_tnp_metadata(jsonl_path):
    """Return dict tnp_id -> dict of per-tnp fields (from first site's labels/metadata)."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            tnp = r['transposase_id']
            if tnp in out:
                continue
            lab = r['labels']
            meta = r.get('generator_metadata', {}) or {}
            out[tnp] = {
                'is_positive': lab.get('is_positive'),
                'tnp_strength': meta.get('tnp_strength'),
                'guided_frac_target': meta.get('guided_fraction_target'),
                'n_guided': meta.get('n_guided_in_tnp'),
                'n_off_target': meta.get('n_off_target_in_tnp'),
                'n_unresolved': meta.get('n_unresolved_in_tnp'),
                'num_noncoding_regions': lab.get('num_noncoding_regions'),
                'active_noncoding_index': lab.get('active_noncoding_index'),
                'guide_length': lab.get('guide_length'),
                'match_orientation': lab.get('match_orientation'),
            }
    return out


def quantiles(vals, qs=(0.10, 0.25, 0.50, 0.75, 0.90)):
    a = np.asarray(vals, dtype=np.float64)
    return {f'Q{int(q*100):02d}': float(np.quantile(a, q)) for q in qs}


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'ckpt: epoch {ckpt["epoch"]}, saved val AUPRC={ckpt["auprc"]:.4f}', flush=True)

    for split in ('val', 'test'):
        print()
        print('=' * 70)
        print(f'  {split}_v3')
        print('=' * 70)
        t0 = time.time()
        scores, is_pos, tnp_ids = score_split(split, model, device)
        meta = per_tnp_metadata(f'{BASE}/splits/{split}_v3.jsonl')
        print(f'  scored {len(scores)} tnps in {time.time()-t0:.1f}s')

        # Only positives with strength annotation
        by_strength = defaultdict(list)
        rows = []
        for score, positive, tnp in zip(scores, is_pos, tnp_ids):
            if not positive:
                continue
            m = meta.get(tnp, {})
            strength = m.get('tnp_strength', 'unknown')
            by_strength[strength].append(score)
            rows.append({
                'tnp_id': tnp, 'score': float(score), 'strength': strength,
                **{k: m.get(k) for k in ('guided_frac_target', 'n_guided', 'n_off_target',
                                          'n_unresolved', 'num_noncoding_regions',
                                          'guide_length', 'match_orientation')},
            })

        print()
        print('  score quantiles per strength:')
        print(f'    {"strength":<12} {"n":>4}  {"Q10":>7} {"Q25":>7} {"median":>8} {"Q75":>7} {"Q90":>7}  {"mean":>7} {"std":>7} {"recall@0.5":>10}')
        for lvl in ('strong', 'moderate', 'weak'):
            vals = by_strength.get(lvl, [])
            if not vals:
                continue
            q = quantiles(vals)
            recall = float(np.mean(np.asarray(vals) > 0.5))
            print(f'    {lvl:<12} {len(vals):>4}  {q["Q10"]:>7.4f} {q["Q25"]:>7.4f} '
                  f'{q["Q50"]:>8.4f} {q["Q75"]:>7.4f} {q["Q90"]:>7.4f}  '
                  f'{np.mean(vals):>7.4f} {np.std(vals):>7.4f} {recall:>10.4f}')

        # Compare other tnp-level features across strengths.
        print()
        print('  generation stats per strength (mean ± std):')
        by_strength_meta = defaultdict(list)
        for row in rows:
            by_strength_meta[row['strength']].append(row)
        for lvl in ('strong', 'moderate', 'weak'):
            recs = by_strength_meta.get(lvl, [])
            if not recs:
                continue
            def _mean_std(k):
                vals = [r[k] for r in recs if r.get(k) is not None]
                if not vals:
                    return 'n/a'
                arr = np.asarray(vals, dtype=np.float64)
                return f'{arr.mean():.3f} ± {arr.std():.3f}'
            print(f'    {lvl:<12}  guided_frac={_mean_std("guided_frac_target")}  '
                  f'n_guided={_mean_std("n_guided")}  '
                  f'n_off_target={_mean_std("n_off_target")}  '
                  f'n_unresolved={_mean_std("n_unresolved")}')
            print(f'                num_ncs={_mean_std("num_noncoding_regions")}  '
                  f'guide_length={_mean_std("guide_length")}')

        # Also look at rank of strong-positives among positives by score
        # (are strong tnps landing at the bottom?)
        print()
        print(f'  Fraction of positives BELOW each threshold, per strength:')
        for lvl in ('strong', 'moderate', 'weak'):
            vals = np.asarray(by_strength.get(lvl, []))
            if len(vals) == 0:
                continue
            frac_below = {th: float(np.mean(vals < th)) for th in (0.1, 0.25, 0.5, 0.75, 0.9)}
            print(f'    {lvl:<12} n={len(vals):>4}   ' +
                  '  '.join(f'<{th}:{frac_below[th]:.3f}' for th in (0.1, 0.25, 0.5, 0.75, 0.9)))


if __name__ == '__main__':
    main()
