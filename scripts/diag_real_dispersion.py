"""Priority-2 diagnostic: extract V5.2 dispersion features per bag on real data.

Runs V5.2 forward on all n_native >= 25 bags (~613 bags), saves per-bag:
  - disp_phi[0..5]: [pos_MAD, pos_STD, pos_IQR, ncstart_STD, L_STD, orient_H]
  - final_score, base_score
  - family, n_native, tnp_id

Then reports per-feature median + IQR by family, so we can see which
dispersion axis is causing IS110 to look different from the controls.
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch

RUN = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt'
OUT_JSONL = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/real_dispersion_n25.jsonl'
N_MIN = 25


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] epoch {ckpt["epoch"]}, val AUPRC {ckpt["auprc"]:.4f}', flush=True)

    cache = StructureCache(f'{RUN}/real_all_u16.index.json')
    tnp_family: dict[str, str] = {}
    with open(f'{RUN}/real_all.jsonl') as f:
        for line in f:
            r = json.loads(line)
            tid = r['transposase_id']
            if tid not in tnp_family:
                tnp_family[tid] = r.get('generator_metadata', {}).get('is_family', '?')

    ds = TnpGroupedDataset(
        f'{RUN}/real_all.jsonl', cache,
        site_subsample_size=50, rng_seed=0,
    )
    # Filter to n_native >= 25
    keep_idx = [i for i, tnp in enumerate(ds.tnp_ids)
                 if len(ds._tnp_lines[tnp]) >= N_MIN]
    print(f'[dataset] {len(keep_idx)} bags with n_native >= {N_MIN}', flush=True)

    feat_names = ('pos_MAD', 'pos_STD', 'pos_IQR', 'ncstart_STD', 'L_STD', 'orient_H')
    t0 = time.time()
    rows = []
    with torch.no_grad():
        for k, i in enumerate(keep_idx):
            tnp = ds.tnp_ids[i]
            n_native = len(ds._tnp_lines[tnp])
            item = ds[i]
            batch = collate_tnp_batch([item], to_torch=True)
            batch = {k2: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k2, v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(batch['candidate_patches'], batch['candidate_features'],
                             batch['candidate_mask'], batch['nc_region_mask'])
            phi = out['disp_phi'].float().cpu().numpy()[0]   # (6,)
            logit = float(out['logit'].item())
            base_logit = float(out['base_logit'].item())
            row = {
                'tnp_id': tnp,
                'family': tnp_family.get(tnp, '?'),
                'is_positive_label': bool(ds._tnp_is_positive[tnp]),
                'n_native': n_native,
                'final_score': float(torch.sigmoid(out['logit']).item()),
                'base_score': float(1.0 / (1.0 + np.exp(-base_logit))),
                'final_logit': logit,
                'base_logit': base_logit,
            }
            for j, name in enumerate(feat_names):
                row[name] = float(phi[j])
            rows.append(row)
            if (k+1) % 100 == 0:
                el = time.time() - t0
                print(f'  [{k+1}/{len(keep_idx)}] {el:.0f}s', flush=True)

    with open(OUT_JSONL, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print(f'\n[done] {len(rows)} bags in {time.time()-t0:.0f}s')
    print(f'[out]  {OUT_JSONL}\n')

    # ============ Report per family ============
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r['family']].append(r)

    print('=' * 108)
    print(f'  PRIORITY 2 — dispersion features per family, n_native >= {N_MIN}')
    print('=' * 108)
    print(f'  Values reported: median [Q25, Q75]')
    print()
    hdr = f'  {"family":<10} {"bags":>5}'
    for name in feat_names:
        hdr += f'  {name:>17}'
    print(hdr)
    print('  ' + '-' * 106)
    for fam in ('IS110','IS30','IS903','IS10-R','ISLdl1','ISAjo2'):
        rs = by_fam.get(fam, [])
        if not rs: continue
        label = 'POS' if fam == 'IS110' else 'NEG'
        line = f'  {fam:<10}({label}) {len(rs):>4}'
        for name in feat_names:
            v = np.asarray([r[name] for r in rs])
            med = np.median(v)
            q25, q75 = np.quantile(v, [.25, .75])
            line += f'  {med:>4.1f} [{q25:>4.1f},{q75:>5.1f}]'
        print(line)

    # ============ IS110 sub-population diagnostic ============
    print()
    print('=' * 108)
    print(f'  PRIORITY 3 — IS110 stratified by V5.2 score (n_native >= {N_MIN})')
    print('=' * 108)
    is110 = sorted(by_fam.get('IS110', []), key=lambda r: r['final_score'], reverse=True)
    if len(is110) >= 15:
        n = len(is110)
        top = is110[:max(1, n//5)]           # top 20%
        bot = is110[-max(1, n//5):]          # bottom 20%
        mid = is110[len(top):-len(bot)]      # middle 60%
        print()
        for name, group in (('IS110-top-20%', top), ('IS110-mid-60%', mid),
                             ('IS110-bot-20%', bot)):
            print(f'  {name}  n={len(group)}')
            n_native_stat = np.asarray([g['n_native'] for g in group])
            score_stat = np.asarray([g['final_score'] for g in group])
            base_stat = np.asarray([g['base_score'] for g in group])
            print(f'    n_native med={np.median(n_native_stat):.0f}, '
                  f'final_score med={np.median(score_stat):.3f}, '
                  f'base_score med={np.median(base_stat):.3f}')
            for fname in feat_names:
                v = np.asarray([g[fname] for g in group])
                print(f'    {fname:<15}: med={np.median(v):.3f}  [Q25={np.quantile(v,.25):.3f}, Q75={np.quantile(v,.75):.3f}]')
            print()


if __name__ == '__main__':
    main()
