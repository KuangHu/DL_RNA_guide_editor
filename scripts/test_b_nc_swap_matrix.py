"""Test B — cross-family NC-swap matrix.

For each base bag (whose flank we keep) × donor NC family:
    - Replace every site's NC regions with the corresponding donor site's
      NC regions (chosen at random from the donor family, per site).
    - Route the structure cache from (target_site, slot) to
      (donor_site, slot) — donor NCs already have their RNAplfold profile
      precomputed in the full-corpus structure cache.
    - Preprocess and score.

Rows of the matrix = base flank family (bag's flank identity).
Cols of the matrix = donor NC family.
Diagonal (i == j) = within-family NC swap (still an NC identity swap).

Combined with Test C (flank swap), this closes the causal loop:
    Test C: swap flank -> score barely changes (flank ignored)
    Test B: swap NC   -> score follows NC family (NC dominates)
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

RUN = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt'
OUT_JSONL = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/test_b_nc_matrix.jsonl'
N_MIN = 25
BAGS_PER_FAMILY = 30
SEED = 0
FAMILIES = ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2')


class RemappedStructureCache:
    """Wraps a StructureCache so lookups on (site_id, slot) route to
    (donor_site_id, donor_slot). Used to feed a swapped NC's precomputed
    structure without re-running RNAplfold."""

    def __init__(self, base: StructureCache, remap: dict):
        self.base = base
        self.remap = remap
        # StructureCache dataloader-safe attributes
        self._meta = base._meta

    def get(self, site_id: str, slot: int, nc_len: int):
        key = (site_id, slot)
        if key in self.remap:
            ds, dslot = self.remap[key]
            return self.base.get(ds, dslot, nc_len)
        return self.base.get(site_id, slot, nc_len)

    def has(self, site_id: str, slot: int) -> bool:
        key = (site_id, slot)
        if key in self.remap:
            ds, dslot = self.remap[key]
            return self.base.has(ds, dslot)
        return self.base.has(site_id, slot)


def load_records():
    by_tnp: dict[str, list[dict]] = defaultdict(list)
    family_sites: dict[str, list[dict]] = defaultdict(list)
    with open(f'{RUN}/real_all.jsonl') as f:
        for line in f:
            r = json.loads(line)
            tid = r['transposase_id']
            fam = r.get('generator_metadata', {}).get('is_family', '?')
            by_tnp[tid].append(r)
            family_sites[fam].append(r)
    return by_tnp, family_sites


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


def score_bag(model, cache_for_this_bag, records_with_swapped_ncs, device):
    site_outs = []
    for rec in records_with_swapped_ncs:
        out = preprocess_site(rec, structure_cache=cache_for_this_bag)
        site_outs.append(out)
    active_ncs = [-1] * len(records_with_swapped_ncs)
    true_slots = [-1] * len(records_with_swapped_ncs)
    item = {
        'tnp_id': records_with_swapped_ncs[0]['transposase_id'],
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
    }


def make_swapped_records_and_remap(base_records, donor_sites_pool, rng):
    """For each base site, pick a donor site (random from donor_sites_pool).
    Return (new records, remap dict for RemappedStructureCache)."""
    new_records = []
    remap = {}
    for base_r in base_records:
        donor = rng.choice(donor_sites_pool)
        mod = copy.copy(base_r)
        mod['inputs'] = dict(base_r['inputs'])
        mod['inputs']['noncoding_regions'] = list(donor['inputs']['noncoding_regions'])
        # Route structure lookup: (base site, slot) -> (donor site, slot).
        # Some donors may have fewer NCs than 3; only route the slots that exist.
        for slot in range(len(donor['inputs']['noncoding_regions'])):
            remap[(base_r['site_id'], slot)] = (donor['site_id'], slot)
        new_records.append(mod)
    return new_records, remap


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] epoch {ckpt["epoch"]}', flush=True)

    base_cache = StructureCache(f'{RUN}/real_all_u16.index.json')
    by_tnp, family_sites = load_records()
    sampled = sample_bags(by_tnp)
    for fam in FAMILIES:
        print(f'  base bags from {fam}: {len(sampled[fam])} (donor pool sizes: '
              + ', '.join(f'{f}={len(family_sites[f])}' for f in FAMILIES) + ')', flush=True)

    rng = random.Random(SEED + 2)
    t0 = time.time()
    rows = []
    total = sum(len(sampled[f]) for f in FAMILIES) * len(FAMILIES)
    done = 0
    for base_fam in FAMILIES:
        for tid in sampled[base_fam]:
            recs = by_tnp[tid]
            if len(recs) > 50:
                recs = rng.sample(recs, 50)
            for donor_nc_fam in FAMILIES:
                donor_pool = family_sites[donor_nc_fam]
                mod_recs, remap = make_swapped_records_and_remap(recs, donor_pool, rng)
                wrapped = RemappedStructureCache(base_cache, remap)
                out = score_bag(model, wrapped, mod_recs, device)
                rows.append({
                    'tnp_id': tid,
                    'base_flank_family': base_fam,   # bag's original flank stays
                    'donor_nc_family': donor_nc_fam, # NC content came from this family
                    'n_sites_used': len(recs),
                    **out,
                })
                done += 1
                if done % 30 == 0:
                    el = time.time() - t0
                    rate = done / max(1, el)
                    eta = (total - done) / max(1e-6, rate)
                    print(f'  [{done}/{total}] {el:.0f}s '
                          f'({rate:.2f} forwards/s, ETA {eta:.0f}s)', flush=True)

    with open(OUT_JSONL, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print(f'\n[done] {len(rows)} rows in {time.time()-t0:.0f}s')
    print(f'[out]  {OUT_JSONL}')

    # ============ 6x6 median matrices ============
    def make_matrix(key):
        M = np.zeros((len(FAMILIES), len(FAMILIES)), dtype=np.float64)
        for i, base_fam in enumerate(FAMILIES):
            for j, donor_fam in enumerate(FAMILIES):
                vals = [r[key] for r in rows
                         if r['base_flank_family'] == base_fam
                         and r['donor_nc_family'] == donor_fam]
                M[i, j] = float(np.median(vals)) if vals else float('nan')
        return M

    for key, label in (('base_score',  'V4 BASE SCORE median (rows = base FLANK family, cols = donor NC family)'),
                         ('final_score', 'V5.2 FINAL SCORE median'),
                         ('cand_top1_med', 'candidate top-1 (median across sites in bag)'),
                         ('nc_max_med',   'NC-attn max (median across sites in bag)')):
        M = make_matrix(key)
        print()
        print('=' * 96)
        print(f'  {label}')
        print('=' * 96)
        label_row = 'flank↓ / NC→'
        hdr = f'  {label_row:<12}'
        for fam in FAMILIES:
            hdr += f' {fam:>8}'
        print(hdr)
        print('  ' + '-' * (12 + 9 * len(FAMILIES)))
        for i, base_fam in enumerate(FAMILIES):
            line = f'  {base_fam:<12}'
            for j in range(len(FAMILIES)):
                v = M[i, j]
                marker = '*' if i == j else ' '
                line += f' {v:>7.3f}{marker}'
            print(line)

    # ============ Variance decomposition on base logit ============
    idx_flank = {f: i for i, f in enumerate(FAMILIES)}
    idx_nc    = {f: i for i, f in enumerate(FAMILIES)}
    y = np.asarray([r['base_logit'] for r in rows], dtype=np.float64)
    flank = np.asarray([idx_flank[r['base_flank_family']] for r in rows])
    nc    = np.asarray([idx_nc[r['donor_nc_family']] for r in rows])
    mu = y.mean()
    alpha_flank = np.zeros(len(FAMILIES))   # base flank main effect
    beta_nc     = np.zeros(len(FAMILIES))   # donor NC main effect
    for i in range(len(FAMILIES)):
        alpha_flank[i] = y[flank == i].mean() - mu
        beta_nc[i]     = y[nc == i].mean() - mu
    y_pred_main = mu + alpha_flank[flank] + beta_nc[nc]
    cell_mean = np.zeros((len(FAMILIES), len(FAMILIES)))
    for i in range(len(FAMILIES)):
        for j in range(len(FAMILIES)):
            mask = (flank == i) & (nc == j)
            if mask.any():
                cell_mean[i, j] = y[mask].mean()
            else:
                cell_mean[i, j] = np.nan
    y_pred_cell = cell_mean[flank, nc]
    gamma_ij = y_pred_cell - y_pred_main

    ss_total   = float(np.sum((y - mu) ** 2))
    ss_flank   = float(np.sum((alpha_flank[flank]) ** 2))
    ss_nc      = float(np.sum((beta_nc[nc]) ** 2))
    ss_interact = float(np.sum((gamma_ij) ** 2))
    ss_residual = float(np.sum((y - y_pred_cell) ** 2))

    print()
    print('=' * 96)
    print('  Variance decomposition on base LOGIT (V4 base path only)')
    print('=' * 96)
    print(f'  Total SS:                    {ss_total:>10.3f}')
    print(f'  Base flank main effect:      {ss_flank:>10.3f}   ({100*ss_flank/max(1e-9,ss_total):>5.1f}%)')
    print(f'  Donor NC main effect:        {ss_nc:>10.3f}   ({100*ss_nc/max(1e-9,ss_total):>5.1f}%)')
    print(f'  Interaction (flank x NC):    {ss_interact:>10.3f}   ({100*ss_interact/max(1e-9,ss_total):>5.1f}%)')
    print(f'  Residual (within cell):      {ss_residual:>10.3f}   ({100*ss_residual/max(1e-9,ss_total):>5.1f}%)')

    print()
    print(f'  alpha_flank (per-family base logit shift, bag flank identity):')
    for i, f in enumerate(FAMILIES):
        print(f'    {f:<10} {alpha_flank[i]:+.3f}')
    print()
    print(f'  beta_NC (per-family base logit shift, donor NC content):')
    for i, f in enumerate(FAMILIES):
        print(f'    {f:<10} {beta_nc[i]:+.3f}')


if __name__ == '__main__':
    main()
