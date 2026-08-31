"""Cross-family 6x6 flank swap matrix.

For each NC-source family i × each flank-donor family j, score bags where
each site's flank is replaced with a random flank drawn from family j
(with rejection if the sampled flank == the original).

Then fit a linear decomposition on the per-bag base logits:
    logit_{i,j,b} = mu + alpha_i (NC family) + beta_j (flank family)
                       + gamma_{i,j} (interaction) + eps
and report variance explained by each effect.

Diagonal (i == j) is the within-family swap, which Test C already established
is essentially identical to no-swap.

Output:
  logs/test_c_matrix.jsonl   — per (nc_family, flank_family, bag) row
  Console table:
    - 6x6 median base_score matrix (V4)
    - 6x6 median final_score matrix (V5.2)
    - ANOVA-style variance decomposition on logits
"""
from __future__ import annotations

import copy
import json
import random
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache, preprocess_site
from preprocess.tnp_dataset import collate_tnp_batch

import argparse

RUN = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference'
_DEFAULT_CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt'
_DEFAULT_OUT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/test_c_matrix.jsonl'
N_MIN = 25
BAGS_PER_FAMILY = 30
SEED = 0
FAMILIES = ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2')

# Filled in main() from argparse; retained as module-level for the helper fns.
CKPT = _DEFAULT_CKPT
OUT_JSONL = _DEFAULT_OUT


def load_records():
    by_tnp: dict[str, list[dict]] = defaultdict(list)
    family_flanks: dict[str, list[str]] = defaultdict(list)
    with open(f'{RUN}/real_all.jsonl') as f:
        for line in f:
            r = json.loads(line)
            tid = r['transposase_id']
            fam = r.get('generator_metadata', {}).get('is_family', '?')
            by_tnp[tid].append(r)
            family_flanks[fam].append(r['inputs']['flank'])
    return by_tnp, family_flanks


def sample_bags(by_tnp):
    rng = random.Random(SEED)
    fam_to_tnps: dict[str, list[str]] = defaultdict(list)
    for tid, recs in by_tnp.items():
        if len(recs) < N_MIN:
            continue
        fam = recs[0].get('generator_metadata', {}).get('is_family', '?')
        fam_to_tnps[fam].append(tid)
    sampled = {}
    for fam in FAMILIES:
        tnps = fam_to_tnps.get(fam, [])
        rng.shuffle(tnps)
        sampled[fam] = tnps[:BAGS_PER_FAMILY]
    return sampled


def score_bag(model, cache, records, device, flank_override_list):
    """Preprocess `records` with flank replaced per site; forward; return
    logit + base_logit + per-site nc_max & cand_top1 medians."""
    site_outs = []
    for i, rec in enumerate(records):
        rec_use = copy.copy(rec)
        rec_use['inputs'] = dict(rec['inputs'])
        rec_use['inputs']['flank'] = flank_override_list[i]
        out = preprocess_site(rec_use, structure_cache=cache)
        site_outs.append(out)
    active_ncs = [-1] * len(records)
    true_slots = [-1] * len(records)
    item = {
        'tnp_id': records[0]['transposase_id'],
        'site_ids': [o['site_id'] for o in site_outs],
        'is_positive': False,
        'candidate_patches':  np.stack([o['candidate_patches'] for o in site_outs], axis=0),
        'candidate_features': np.stack([o['candidate_features'] for o in site_outs], axis=0),
        'candidate_mask':     np.stack([o['candidate_mask'] for o in site_outs], axis=0),
        'nc_region_mask':     np.stack([o['nc_region_mask'] for o in site_outs], axis=0),
        'true_slot_idx':      np.asarray(true_slots, dtype=np.int32),
        'active_nc_index':    np.asarray(active_ncs, dtype=np.int32),
    }
    batch = collate_tnp_batch([item], to_torch=True)
    batch_dev = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                 for k, v in batch.items()}
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        out = model(batch_dev['candidate_patches'], batch_dev['candidate_features'],
                     batch_dev['candidate_mask'], batch_dev['nc_region_mask'])
    nc_attn = out['nc_attn'].float().cpu().numpy()[0]
    cand_raw = out['cand_raw'].float().cpu().numpy()[0]
    cand_mask = batch['candidate_mask'].cpu().numpy()[0]
    active_nc_pick = nc_attn.argmax(axis=-1)
    S, N, K = cand_raw.shape
    cr_at = cand_raw[np.arange(S), active_nc_pick, :]
    cm_at = cand_mask[np.arange(S), active_nc_pick, :]
    cr_at_masked = np.where(cm_at, cr_at, -np.inf)
    return {
        'logit': float(out['logit'].item()),
        'base_logit': float(out['base_logit'].item()),
        'base_score': float(1.0 / (1.0 + np.exp(-float(out['base_logit'].item())))),
        'final_score': float(torch.sigmoid(out['logit']).item()),
        'nc_max_med': float(np.median(nc_attn.max(axis=-1))),
        'cand_top1_med': float(np.median(cr_at_masked.max(axis=-1))),
        'active_nc_picks': active_nc_pick.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-path', default=_DEFAULT_CKPT,
                         help='V6 checkpoint to score with (default: V5.2 baseline).')
    parser.add_argument('--out-jsonl', default=_DEFAULT_OUT,
                         help='Where to write the per-bag scoring output.')
    parser.add_argument('--label', default='',
                         help='Optional label prefix for the printed tables.')
    args = parser.parse_args()
    global CKPT, OUT_JSONL
    CKPT = args.ckpt_path
    OUT_JSONL = args.out_jsonl

    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    print(f'[ckpt] {CKPT}', flush=True)
    if 'epoch' in ckpt:
        print(f'         epoch={ckpt["epoch"]}, val AUPRC={ckpt.get("auprc", float("nan")):.4f}',
              flush=True)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] epoch {ckpt["epoch"]}', flush=True)

    cache = StructureCache(f'{RUN}/real_all_u16.index.json')
    by_tnp, family_flanks = load_records()
    sampled = sample_bags(by_tnp)
    for fam in FAMILIES:
        print(f'  NC family {fam}: {len(sampled[fam])} bags', flush=True)

    rng = random.Random(SEED + 1)
    t0 = time.time()
    rows = []
    total_forwards = sum(len(sampled[fam]) for fam in FAMILIES) * len(FAMILIES)
    done = 0
    for nc_fam in FAMILIES:
        for tid in sampled[nc_fam]:
            recs = by_tnp[tid]
            if len(recs) > 50:
                recs = rng.sample(recs, 50)
            for flank_fam in FAMILIES:
                pool = family_flanks[flank_fam]
                perturbed = []
                for r in recs:
                    orig = r['inputs']['flank']
                    for _ in range(20):
                        donor = rng.choice(pool)
                        if donor != orig:
                            break
                    perturbed.append(donor)
                out = score_bag(model, cache, recs, device, perturbed)
                rows.append({
                    'tnp_id': tid,
                    'nc_family': nc_fam,
                    'flank_family': flank_fam,
                    'n_sites_used': len(recs),
                    **out,
                })
                done += 1
                if done % 30 == 0:
                    el = time.time() - t0
                    rate = done / max(1, el)
                    eta = (total_forwards - done) / max(1e-6, rate)
                    print(f'  [{done}/{total_forwards}] {el:.0f}s '
                          f'({rate:.2f} forwards/s, ETA {eta:.0f}s)', flush=True)

    with open(OUT_JSONL, 'w') as fh:
        for r in rows:
            # drop lists for size — keep summaries
            r_slim = {k: v for k, v in r.items() if k != 'active_nc_picks'}
            fh.write(json.dumps(r_slim) + '\n')
    print(f'\n[done] {len(rows)} rows in {time.time()-t0:.0f}s')
    print(f'[out]  {OUT_JSONL}')

    # ============ 6x6 median matrices ============
    def make_matrix(key):
        M = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=np.float64)
        for i, nc_fam in enumerate(FAMILIES):
            for j, flank_fam in enumerate(FAMILIES):
                vals = [r[key] for r in rows if r['nc_family'] == nc_fam and r['flank_family'] == flank_fam]
                M[i, j] = float(np.median(vals)) if vals else float('nan')
        return M

    for key, label in (('base_score',  'V4 BASE SCORE median (rows = NC source, columns = flank donor)'),
                         ('final_score', 'V5.2 FINAL SCORE median'),
                         ('cand_top1_med', 'candidate top-1 (median across sites in bag)'),
                         ('nc_max_med',   'NC-attn max (median across sites in bag)')):
        M = make_matrix(key)
        print()
        print('=' * 96)
        print(f'  {label}')
        print('=' * 96)
        label_row = 'NC↓ / flank→'
        hdr = f'  {label_row:<12}'
        for fam in FAMILIES:
            hdr += f' {fam:>8}'
        print(hdr)
        print('  ' + '-' * (12 + 9 * len(FAMILIES)))
        for i, nc_fam in enumerate(FAMILIES):
            line = f'  {nc_fam:<12}'
            for j in range(len(FAMILIES)):
                v = M[i, j]
                marker = '*' if i == j else ' '
                line += f' {v:>7.3f}{marker}'
            print(line)

    # ============ Variance decomposition on base logit ============
    # logit[i,j,b] = mu + alpha_i + beta_j + gamma_{ij} + eps
    # We use type-III/regression-style decomposition on per-bag logits.
    idx_nc = {f: i for i, f in enumerate(FAMILIES)}
    idx_fk = {f: i for i, f in enumerate(FAMILIES)}
    y = np.asarray([r['base_logit'] for r in rows], dtype=np.float64)
    n = len(y)
    nc = np.asarray([idx_nc[r['nc_family']] for r in rows])
    fk = np.asarray([idx_fk[r['flank_family']] for r in rows])
    mu = y.mean()
    alpha = np.zeros(len(FAMILIES))
    beta  = np.zeros(len(FAMILIES))
    for i in range(len(FAMILIES)):
        alpha[i] = y[nc == i].mean() - mu
        beta[i]  = y[fk == i].mean() - mu
    y_pred_main = mu + alpha[nc] + beta[fk]
    # Interaction cell means
    cell_mean = np.zeros((len(FAMILIES), len(FAMILIES)))
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            mask = (nc == i) & (fk == j)
            if mask.any():
                cell_mean[i, j] = y[mask].mean()
            else:
                cell_mean[i, j] = np.nan
    y_pred_cell = cell_mean[nc, fk]
    gamma_ij = y_pred_cell - y_pred_main

    ss_total = float(np.sum((y - mu) ** 2))
    ss_nc    = float(np.sum((alpha[nc]) ** 2))
    ss_flank = float(np.sum((beta[fk]) ** 2))
    ss_interact = float(np.sum((gamma_ij) ** 2))
    ss_residual = float(np.sum((y - y_pred_cell) ** 2))

    print()
    print('=' * 96)
    print('  Variance decomposition on base LOGIT (V4 base path only)')
    print('=' * 96)
    print(f'  Total SS:                 {ss_total:>10.3f}')
    print(f'  NC-family main effect:    {ss_nc:>10.3f}   ({100*ss_nc/max(1e-9,ss_total):>5.1f}%)')
    print(f'  Flank-family main effect: {ss_flank:>10.3f}   ({100*ss_flank/max(1e-9,ss_total):>5.1f}%)')
    print(f'  Interaction (NC x flank): {ss_interact:>10.3f}   ({100*ss_interact/max(1e-9,ss_total):>5.1f}%)')
    print(f'  Residual (within cell):   {ss_residual:>10.3f}   ({100*ss_residual/max(1e-9,ss_total):>5.1f}%)')

    print()
    print(f'  alpha_NC (per-family base logit shift):')
    for i, f in enumerate(FAMILIES):
        print(f'    {f:<10} {alpha[i]:+.3f}')
    print()
    print(f'  beta_flank (per-family base logit shift):')
    for i, f in enumerate(FAMILIES):
        print(f'    {f:<10} {beta[i]:+.3f}')


if __name__ == '__main__':
    main()
