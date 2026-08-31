"""Full inference: V5.2 Stage A on all 19,676 real bags across 6 IS families.

Per bag we compute:
  - P(RNA-guided) score (sigmoid of logit)
  - raw logit
  - disp_delta / disp_beta contribution (V5.2 diagnostic)
  - n_sites in the bag (native, not subsampled)

Output: JSONL to logs/real_all_v52_scores.jsonl with per-bag records.
Also emits a summary table stratified by family + n_sites bucket.
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
OUT_JSONL = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/real_all_v52_scores.jsonl'
MAX_SITES = 50   # subsample bags larger than this to match training


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] {CKPT}: epoch {ckpt["epoch"]}, val AUPRC {ckpt["auprc"]:.4f}',
          flush=True)

    cache = StructureCache(f'{RUN}/real_all_u16.index.json')
    print(f'[cache] {cache.N} records, nc_max={cache.nc_max}, u_max={cache.u_max}',
          flush=True)

    # Build a family lookup once (parse jsonl for is_family per tnp_id).
    print('[meta] building tnp_id -> is_family map', flush=True)
    tnp_family: dict[str, str] = {}
    with open(f'{RUN}/real_all.jsonl') as f:
        for line in f:
            r = json.loads(line)
            tid = r['transposase_id']
            if tid not in tnp_family:
                tnp_family[tid] = r.get('generator_metadata', {}).get('is_family', '?')

    # Sub-sampled dataset for large bags; ds returns all sites when bag <= MAX_SITES.
    ds = TnpGroupedDataset(
        f'{RUN}/real_all.jsonl', cache,
        site_subsample_size=MAX_SITES, rng_seed=0,
    )
    print(f'[dataset] tnps={len(ds)}', flush=True)

    # Resume-skip: read tnp_ids already in OUT_JSONL and skip them.
    already = set()
    results = []
    try:
        with open(OUT_JSONL) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    already.add(r['tnp_id'])
                    results.append(r)
                except json.JSONDecodeError:
                    pass
        print(f'[resume] {len(already)} tnps already scored, skipping', flush=True)
    except FileNotFoundError:
        print(f'[resume] no prior output at {OUT_JSONL}, starting fresh', flush=True)

    t_start = time.time()
    with open(OUT_JSONL, 'a') as fh:
        for i, tnp in enumerate(ds.tnp_ids):
            if tnp in already:
                continue
            n_native = len(ds._tnp_lines[tnp])
            item = ds[i]
            n_used = int(item['candidate_patches'].shape[0])
            is_pos = ds._tnp_is_positive[tnp]

            batch = collate_tnp_batch([item], to_torch=True)
            batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(batch['candidate_patches'], batch['candidate_features'],
                             batch['candidate_mask'], batch['nc_region_mask'])
            logit = float(out['logit'].item())
            base_logit = float(out['base_logit'].item())
            score = float(torch.sigmoid(out['logit']).item())
            beta = out.get('disp_alpha_or_beta')
            beta_val = float(beta.item()) if beta is not None else float('nan')

            rec = {
                'tnp_id': tnp,
                'family': tnp_family.get(tnp, '?'),
                'is_positive': bool(is_pos),
                'n_native_sites': n_native,
                'n_used_sites': n_used,
                'score': score,
                'logit': logit,
                'base_logit': base_logit,
                'beta': beta_val,
            }
            fh.write(json.dumps(rec) + '\n')
            results.append(rec)

            n_done = len(results)
            if n_done % 500 == 0 and n_done > 0:
                el = time.time() - t_start
                new_since_start = n_done - len(already)
                rate = new_since_start / max(1, el)
                print(f'  [{n_done}/{len(ds)}] {el:.0f}s elapsed '
                      f'({rate:.1f} new bags/s)', flush=True)

    print(f'\n[done] {len(results)} bags scored total, '
          f'new this run in {time.time()-t_start:.0f}s')
    print(f'[out]  {OUT_JSONL}')

    # ---------------- Summary tables ----------------
    print()
    print('=' * 92)
    print('  TABLE — Score distribution per family (all bags)')
    print('=' * 92)
    print(f'  {"family":<10} {"label":<7} {"bags":>6} {"mean":>7} {"med":>7} '
          f'{"Q10":>7} {"Q90":>7} {"score>0.5":>10}')
    by_fam = defaultdict(list)
    for r in results:
        by_fam[r['family']].append(r['score'])
    for fam in ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2'):
        s = np.asarray(by_fam.get(fam, []))
        if not len(s):
            continue
        label = 'POS' if fam == 'IS110' else 'NEG'
        print(f'  {fam:<10} {label:<7} {len(s):>6} '
              f'{s.mean():>7.3f} {np.median(s):>7.3f} '
              f'{np.quantile(s, .1):>7.3f} {np.quantile(s, .9):>7.3f} '
              f'{(s > 0.5).mean():>10.3f}')

    print()
    print('=' * 92)
    print('  TABLE — Score distribution per family, filtered to n_native >= 5')
    print('=' * 92)
    print(f'  {"family":<10} {"label":<7} {"bags":>6} {"mean":>7} {"med":>7} '
          f'{"Q10":>7} {"Q90":>7} {"score>0.5":>10}')
    by_fam5 = defaultdict(list)
    for r in results:
        if r['n_native_sites'] >= 5:
            by_fam5[r['family']].append(r['score'])
    for fam in ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2'):
        s = np.asarray(by_fam5.get(fam, []))
        if not len(s):
            continue
        label = 'POS' if fam == 'IS110' else 'NEG'
        print(f'  {fam:<10} {label:<7} {len(s):>6} '
              f'{s.mean():>7.3f} {np.median(s):>7.3f} '
              f'{np.quantile(s, .1):>7.3f} {np.quantile(s, .9):>7.3f} '
              f'{(s > 0.5).mean():>10.3f}')

    # Ranking-based metric — AUROC treating IS110 as positive, all others as negative
    from training.metrics import _auroc, _auprc
    y = np.asarray([r['family'] == 'IS110' for r in results], dtype=bool)
    s = np.asarray([r['score'] for r in results])
    if y.any() and (~y).any():
        auroc_all = _auroc(s, y)
        auprc_all = _auprc(s, y)
        print()
        print(f'  Family-level ranking (IS110 vs all others, all bag sizes):')
        print(f'    AUROC = {auroc_all:.4f}   AUPRC = {auprc_all:.4f}   n_pos={int(y.sum())}   n_neg={int((~y).sum())}')

        # Filter to n_native >= 5
        mask5 = np.asarray([r['n_native_sites'] >= 5 for r in results])
        if mask5.any():
            auroc_5 = _auroc(s[mask5], y[mask5])
            auprc_5 = _auprc(s[mask5], y[mask5])
            print(f'  Family-level ranking (IS110 vs all others, n_native >= 5):')
            print(f'    AUROC = {auroc_5:.4f}   AUPRC = {auprc_5:.4f}   '
                  f'n_pos={int(y[mask5].sum())}   n_neg={int((~y[mask5]).sum())}')
        mask25 = np.asarray([r['n_native_sites'] >= 25 for r in results])
        if mask25.any():
            auroc_25 = _auroc(s[mask25], y[mask25])
            auprc_25 = _auprc(s[mask25], y[mask25])
            print(f'  Family-level ranking (IS110 vs all others, n_native >= 25):')
            print(f'    AUROC = {auroc_25:.4f}   AUPRC = {auprc_25:.4f}   '
                  f'n_pos={int(y[mask25].sum())}   n_neg={int((~y[mask25]).sum())}')


if __name__ == '__main__':
    main()
