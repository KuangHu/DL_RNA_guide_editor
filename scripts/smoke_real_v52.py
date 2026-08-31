"""Smoke test: run frozen V5.2 checkpoint on 10 real bags (5 IS110 + 5 IS30).

Purpose:
  Verify the classifier's inference pipeline works end-to-end on the real
  data schema — no NaN, sensible per-tnp scores, correct behavior on
  variable-size bags.

This is not a scientific evaluation. n=10 is far too small to conclude
anything about generalization; we're only checking that the plumbing works.
"""
from __future__ import annotations

import json
import sys
import time

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset


SMOKE_DIR = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/smoke'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt'


def main():
    device = torch.device('cuda')
    print(f'[env] device: {device} — cuda={torch.cuda.is_available()}', flush=True)

    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] loaded {CKPT}: epoch {ckpt["epoch"]}, val AUPRC {ckpt["auprc"]:.4f}',
          flush=True)
    print(f'[cfg]  use_dispersion={cfg.use_dispersion} mode={cfg.dispersion_mode} '
          f'disp_hidden={cfg.disp_hidden}', flush=True)

    cache = StructureCache(f'{SMOKE_DIR}/smoke_u16.index.json')
    print(f'[cache] {cache.N} records, nc_max={cache.nc_max}, u_max={cache.u_max}',
          flush=True)

    # Real bags have variable site counts (2..20 here). Per-bag inference
    # (batch_size=1) is the simplest path — collate requires uniform S per
    # batch, and site_subsample_size=50 does NOT pad up (only subsamples down).
    ds = TnpGroupedDataset(
        f'{SMOKE_DIR}/smoke.jsonl', cache,
        site_subsample_size=None, rng_seed=0,
    )
    print(f'\n[dataset] tnps={len(ds)}', flush=True)
    for i, tnp in enumerate(ds.tnp_ids):
        n_sites = len(ds._tnp_lines[tnp])
        is_pos = ds._tnp_is_positive[tnp]
        print(f'    {tnp}: {n_sites} native sites, is_positive={is_pos}',
              flush=True)

    print()
    print('=' * 76)
    print('INFERENCE (per-bag, native site counts, model was trained on 50)')
    print('=' * 76)
    _run_variable(ds, model, device)


def _run(dl, model, device):
    scores, tnp_ids = [], []
    with torch.no_grad():
        for b in dl:
            b_dev = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(b_dev['candidate_patches'], b_dev['candidate_features'],
                             b_dev['candidate_mask'], b_dev['nc_region_mask'])
            s = torch.sigmoid(out['logit']).float().cpu().numpy()
            scores.append(s)
            tnp_ids.extend(list(b['tnp_id']))
            if not np.isfinite(s).all():
                print(f'    !! NaN or Inf in logits: {s}', flush=True)
    return dict(zip(tnp_ids, np.concatenate(scores).tolist()))


def _run_variable(ds, model, device):
    print(f'  {"tnp_id":<24} {"is_pos":>7} {"n_sites":>8} {"score":>8}   {"logit":>8} {"alpha_beta":>10}')
    scores_pos, scores_neg = [], []
    for i, tnp in enumerate(ds.tnp_ids):
        n_native = len(ds._tnp_lines[tnp])
        item = ds[i]
        batch = collate_tnp_batch([item], to_torch=True)
        batch = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(batch['candidate_patches'], batch['candidate_features'],
                         batch['candidate_mask'], batch['nc_region_mask'])
        logit = float(out['logit'].item())
        s = float(torch.sigmoid(out['logit']).item())
        beta = out.get('disp_alpha_or_beta')
        beta_val = float(beta.item()) if beta is not None else float('nan')
        is_pos = ds._tnp_is_positive[tnp]
        (scores_pos if is_pos else scores_neg).append(s)
        finite = '' if np.isfinite(s) else ' !!NAN/INF!!'
        print(f'  {tnp:<24} {str(is_pos):>7} {n_native:>8} {s:>8.4f}   {logit:>+8.4f} {beta_val:>+10.4f}{finite}',
              flush=True)
    print()
    print(f'  POS scores (IS110):  mean={np.mean(scores_pos):.4f}  '
          f'range=[{min(scores_pos):.4f}, {max(scores_pos):.4f}]')
    print(f'  NEG scores (IS30):   mean={np.mean(scores_neg):.4f}  '
          f'range=[{min(scores_neg):.4f}, {max(scores_neg):.4f}]')
    print(f'  \\Delta means: pos - neg = {np.mean(scores_pos)-np.mean(scores_neg):+.4f}')


def _print_scores(ds, scores):
    print(f'  {"tnp_id":<24} {"is_pos":>7} {"native_n":>8} {"score":>8}')
    for tnp, s in scores.items():
        n = len(ds._tnp_lines[tnp])
        is_pos = ds._tnp_is_positive[tnp]
        print(f'  {tnp:<24} {str(is_pos):>7} {n:>8} {s:>8.4f}', flush=True)

    # Split by class
    pos = [s for tnp, s in scores.items() if ds._tnp_is_positive[tnp]]
    neg = [s for tnp, s in scores.items() if not ds._tnp_is_positive[tnp]]
    print(f'\n  POS scores (IS110):   mean={np.mean(pos):.4f}  '
          f'range=[{min(pos):.4f}, {max(pos):.4f}]')
    print(f'  NEG scores (IS30):    mean={np.mean(neg):.4f}  '
          f'range=[{min(neg):.4f}, {max(neg):.4f}]')


if __name__ == '__main__':
    main()
