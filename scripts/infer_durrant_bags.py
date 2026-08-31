"""Score Durrant cognate + shuffled bags with V5.2 and V6, compare distributions.

For each ckpt (V5.2 production, V6 selected ep4) and each split (cognate, shuffled):
  - Load records + StructureCache
  - Group by transposase_id (each bag = 5 sites)
  - Forward the model bag-by-bag
  - Save per-bag (score, logit, base_logit, per-site diagnostics)

Then compare cognate vs shuffled score distributions per checkpoint. If V5.2 or V6
can rank cognate above shuffled, the model is picking up gold-standard LTG/RTG.
If not (score distributions overlap), it isn't — this justifies training V7.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch

CKPTS = {
    'V5.2': '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt',
    'V6':   '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v6_selected/best.pt',
}
BASE = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference')
SPLITS = {
    'cognate':  {'jsonl': BASE / 'durrant_cognate.jsonl',
                 'cache': BASE / 'struct' / 'durrant_cognate_u16.index.json'},
    'shuffled': {'jsonl': BASE / 'durrant_shuffled.jsonl',
                 'cache': BASE / 'struct' / 'durrant_shuffled_u16.index.json'},
}
OUT = BASE / 'model_scores.jsonl'


def score_one_split(ckpt_name, ckpt_path, split_name, jsonl, cache_index, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    cache = StructureCache(cache_index)
    ds = TnpGroupedDataset(str(jsonl), cache, site_subsample_size=100, rng_seed=0)
    rows = []
    for i, tnp in enumerate(ds.tnp_ids):
        item = ds[i]
        batch = collate_tnp_batch([item], to_torch=True)
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(batch['candidate_patches'], batch['candidate_features'],
                         batch['candidate_mask'], batch['nc_region_mask'])
        rows.append({
            'ckpt': ckpt_name, 'split': split_name, 'tnp_id': tnp,
            'n_sites': int(item['candidate_patches'].shape[0]),
            'logit': float(out['logit'].item()),
            'base_logit': float(out['base_logit'].item()),
            'score': float(torch.sigmoid(out['logit']).item()),
        })
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    device = torch.device('cuda')
    all_rows = []
    for ckpt_name, ckpt_path in CKPTS.items():
        for split_name, sp in SPLITS.items():
            t0 = time.time()
            rows = score_one_split(ckpt_name, ckpt_path, split_name,
                                    sp['jsonl'], sp['cache'], device)
            print(f'[{ckpt_name}][{split_name}] {len(rows)} bags in {time.time()-t0:.1f}s')
            all_rows.extend(rows)

    with OUT.open('w') as f:
        for r in all_rows:
            f.write(json.dumps(r) + '\n')
    print(f'\n[out] {OUT}')

    # Report cognate vs shuffled per ckpt
    from collections import defaultdict
    by_ck_sp = defaultdict(list)
    for r in all_rows:
        by_ck_sp[(r['ckpt'], r['split'])].append(r)

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

    print(f'\n{"="*95}')
    print(f'  Cognate vs Shuffled Durrant bags — per checkpoint')
    print(f'{"="*95}\n')
    print(f'  {"ckpt":<6} {"cognate_n":>10} {"cog_med":>8} {"cog_mean":>9} '
          f'{"shuf_n":>7} {"shuf_med":>9} {"shuf_mean":>10}   {"AUROC":>7}')
    for ck in CKPTS:
        cog = np.asarray([r['score'] for r in by_ck_sp.get((ck, 'cognate'), [])])
        shu = np.asarray([r['score'] for r in by_ck_sp.get((ck, 'shuffled'), [])])
        if not len(cog) or not len(shu): continue
        scores = np.concatenate([cog, shu])
        labels = np.concatenate([np.ones(len(cog)), np.zeros(len(shu))])
        au = _auroc(scores, labels)
        print(f'  {ck:<6} {len(cog):>10}  {np.median(cog):>7.3f} {cog.mean():>9.3f} '
              f'{len(shu):>7} {np.median(shu):>9.3f} {shu.mean():>10.3f}   {au:>7.4f}')

    print(f'\n  Also compare each bag as cognate-minus-shuffled Δ (paired by bag index):')
    print(f'  {"ckpt":<6} {"n_pairs":>8} {"Δ_med":>8} {"Δ_mean":>8} {"P(Δ>0)":>8}')
    for ck in CKPTS:
        cog = [r['score'] for r in by_ck_sp.get((ck, 'cognate'), [])]
        shu = [r['score'] for r in by_ck_sp.get((ck, 'shuffled'), [])]
        n = min(len(cog), len(shu))
        if n == 0: continue
        delta = np.asarray(cog[:n]) - np.asarray(shu[:n])
        print(f'  {ck:<6} {n:>8} {np.median(delta):>+8.3f} {delta.mean():>+8.3f} '
              f'{(delta>0).mean():>8.3f}')


if __name__ == '__main__':
    main()
