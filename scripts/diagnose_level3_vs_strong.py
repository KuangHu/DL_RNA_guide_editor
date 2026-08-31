"""Diagnostic prediction test #1:

Compare the score distributions of:
  (a) strong-positive tnps       (guided_frac >= 0.75)
  (b) moderate-positive tnps      (0.50 <= guided_frac < 0.75)
  (c) weak-positive tnps          (0.30 <= guided_frac < 0.50)
  (d) level3_counterfactual_within_tnp negatives (hardest V3 negative)
  (e) all other negatives combined

If (a) overlaps heavily with (d), the model is confusing structurally-
homogeneous positive bags with the level3 counterfactual — evidence that
heterogeneity (off_target/unresolved sites) is being read as a positive
signal.
"""
from __future__ import annotations

import json
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


def load_metadata(jsonl_path):
    """Return dict tnp_id -> {is_positive, tnp_strength, violation_profile}."""
    out = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            tid = r['transposase_id']
            if tid in out:
                continue
            lab = r['labels']
            meta = r.get('generator_metadata', {}) or {}
            out[tid] = {
                'is_positive': lab.get('is_positive'),
                'tnp_strength': meta.get('tnp_strength'),
                'violation_profile': lab.get('violation_profile'),
            }
    return out


def score_split(split, model, device):
    cache = StructureCache(f'{BASE}/structure/{split}_v3_u16.index.json')
    ds = TnpGroupedDataset(f'{BASE}/splits/{split}_v3.jsonl', cache,
                            site_subsample_size=50, rng_seed=0)
    dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=8, shuffle=False,
                    num_workers=4,
                    collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
                    persistent_workers=True, pin_memory=True)
    scores, tnp_ids = [], []
    with torch.no_grad():
        for b in dl:
            b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(b['candidate_patches'], b['candidate_features'],
                             b['candidate_mask'], b['nc_region_mask'])
            scores.append(torch.sigmoid(out['logit']).float().cpu().numpy())
            tnp_ids.extend(list(b['tnp_id']))
    return np.concatenate(scores), tnp_ids


def summarize(name, vals):
    if len(vals) == 0:
        print(f'  {name:<40} n=0')
        return
    v = np.asarray(vals)
    qs = {p: float(np.quantile(v, p / 100)) for p in (10, 25, 50, 75, 90)}
    print(f'  {name:<40} n={len(v):>4}  '
          f'Q10={qs[10]:.4f}  Q25={qs[25]:.4f}  '
          f'median={qs[50]:.4f}  Q75={qs[75]:.4f}  Q90={qs[90]:.4f}  '
          f'mean={v.mean():.4f}  std={v.std():.4f}')


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device); model.load_state_dict(ckpt['model']); model.eval()
    print(f'ckpt: epoch {ckpt["epoch"]}', flush=True)

    for split in ('test',):
        print()
        print('=' * 70)
        print(f'  {split}_v3')
        print('=' * 70)
        t0 = time.time()
        scores, tnp_ids = score_split(split, model, device)
        meta = load_metadata(f'{BASE}/splits/{split}_v3.jsonl')
        print(f'  scored {len(scores)} tnps in {time.time()-t0:.1f}s')
        print()

        buckets = defaultdict(list)
        for score, tid in zip(scores, tnp_ids):
            m = meta.get(tid, {})
            if m.get('is_positive'):
                buckets[m.get('tnp_strength', 'unknown_pos')].append(float(score))
            else:
                vp = m.get('violation_profile', 'unknown_neg')
                buckets[vp].append(float(score))

        print('  POSITIVES (by strength):')
        for k in ('strong', 'moderate', 'weak'):
            summarize(k, buckets.get(k, []))

        print()
        print('  NEGATIVES (by violation_profile):')
        for k in sorted(buckets):
            if k in ('strong', 'moderate', 'weak', 'unknown_pos'):
                continue
            summarize(k, buckets.get(k, []))

        # Overlap analysis: fraction of strong positives with scores below various
        # negative profile medians.
        print()
        print('  --- overlap analysis ---')
        strong = np.asarray(buckets.get('strong', []))
        weak = np.asarray(buckets.get('weak', []))
        level3 = np.asarray(buckets.get('level3_counterfactual_within_tnp', []))
        if len(strong) and len(level3):
            level3_p90 = np.quantile(level3, 0.90)
            print(f'    fraction of strong positives scoring below level3 Q90 ({level3_p90:.4f}): '
                  f'{float((strong < level3_p90).mean()):.3f}')
            print(f'    fraction of strong positives scoring below level3 median ({np.median(level3):.4f}): '
                  f'{float((strong < np.median(level3)).mean()):.3f}')
            print(f'    fraction of weak positives scoring below level3 Q90 ({level3_p90:.4f}): '
                  f'{float((weak < level3_p90).mean()):.3f}')
            # AUROC restricted to strong-vs-level3 vs weak-vs-level3
            from training.metrics import _auroc
            s_all = np.concatenate([strong, level3])
            y_all = np.concatenate([np.ones(len(strong), bool), np.zeros(len(level3), bool)])
            auroc_strong_vs_l3 = _auroc(s_all, y_all)
            w_all = np.concatenate([weak, level3])
            y_all = np.concatenate([np.ones(len(weak), bool), np.zeros(len(level3), bool)])
            auroc_weak_vs_l3 = _auroc(w_all, y_all)
            print(f'    AUROC(strong vs level3): {auroc_strong_vs_l3:.4f}')
            print(f'    AUROC(weak   vs level3): {auroc_weak_vs_l3:.4f}')


if __name__ == '__main__':
    main()
