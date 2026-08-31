"""Stagewise diagnostic on real data (n_native >= 25):
localize WHERE in the V4 base path IS110 starts scoring below controls.

For each bag, extract four layers of statistics:

  Layer 1 (Candidate scorer):
    - per-site best candidate score (max cand_raw at active NC)
    - per-site top1-top2 margin
    - per-site candidate-attention entropy
    - per-site selected L, orient
    - fraction of sites with best cand score > threshold

  Layer 2 (NC selector):
    - per-site max nc_attn weight
    - per-site nc_attn entropy

  Layer 3 (Site → Tnp: bypass Set Transformer):
    - per-site "score" via classifier applied to pre-SetTransformer site token
    - per-bag aggregation: mean, max, top10%-mean of per-site scores
    - AUROC of each aggregation vs IS110-vs-others

  Layer 4 (SetTransformer): the V4 base_logit itself, already known
    (AUROC 0.298 on n>=25 from Priority 1).

The core question this answers:
  Does IS110 rank below controls already at Layer 1 (candidate),
  or does the deficit first appear at Layer 3/4 (aggregator)?
"""
from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch
import torch.nn.functional as F

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch
from preprocess.candidates import FEATURE_NAMES

RUN = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt'
OUT_JSONL = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/real_stagewise_n25.jsonl'
N_MIN = 25

IDX_ORIENT_FWD = FEATURE_NAMES.index("orient_fwd")
IDX_L          = FEATURE_NAMES.index("L")


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


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] epoch {ckpt["epoch"]}', flush=True)

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
    keep_idx = [i for i, tnp in enumerate(ds.tnp_ids)
                 if len(ds._tnp_lines[tnp]) >= N_MIN]
    print(f'[dataset] {len(keep_idx)} bags with n_native >= {N_MIN}', flush=True)

    # V4 base classifier: apply directly to a site token to get a per-site "score"
    # (what would V4 predict if this bag had only this one site).
    classifier = model.classifier   # nn.Sequential (128 -> 64 -> 1)

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

            # (B=1, S, N=3, K=96) — active NC per site (label-known but here we
            # use MODEL-picked active NC (argmax nc_attn), same as V5.2 does).
            nc_attn = out['nc_attn'].float().cpu().numpy()[0]   # (S, N)
            cand_raw = out['cand_raw'].float().cpu().numpy()[0] # (S, N, K)
            cand_attn = out['cand_attn'].float().cpu().numpy()[0]  # (S, N, K)
            cand_mask = batch['candidate_mask'].cpu().numpy()[0]   # (S, N, K)
            cand_feats = batch['candidate_features'].cpu().numpy()[0]  # (S, N, K, F)
            site_pre_set = out['site_repr_pre_set'].float()   # (1, S, D)

            S, N, K = cand_raw.shape

            # Layer 2: NC attention stats (per site)
            nc_max = nc_attn.max(axis=-1)                       # (S,)
            nc_ent = -np.sum(np.where(nc_attn > 1e-9,
                                        nc_attn * np.log2(nc_attn + 1e-12), 0.0),
                              axis=-1)                             # (S,)  in bits

            # Active NC = argmax nc_attn (V5.2's picking policy)
            active_nc = nc_attn.argmax(axis=-1)                 # (S,)

            # Layer 1: candidate scorer stats
            cr_at = cand_raw[np.arange(S), active_nc, :]         # (S, K)
            cm_at = cand_mask[np.arange(S), active_nc, :]        # (S, K)
            cr_at_masked = np.where(cm_at, cr_at, -np.inf)
            # top-1, top-2 for margin
            sorted_idx = np.argsort(-cr_at_masked, axis=-1)      # (S, K) desc
            top1_val = cr_at_masked[np.arange(S), sorted_idx[:, 0]]
            top2_val = np.where(sorted_idx.shape[-1] > 1,
                                 cr_at_masked[np.arange(S), sorted_idx[:, 1]],
                                 top1_val - 1.0)
            top1_margin = top1_val - np.where(np.isfinite(top2_val), top2_val, top1_val - 1.0)
            # candidate-attention entropy at active NC
            ca_at = cand_attn[np.arange(S), active_nc, :]        # (S, K)
            ca_ent = -np.sum(np.where(ca_at > 1e-9,
                                        ca_at * np.log2(ca_at + 1e-12), 0.0),
                              axis=-1)                             # (S,)

            # Feature of the top-1 candidate (orient/L)
            feat_at = cand_feats[np.arange(S), active_nc, sorted_idx[:, 0], :]  # (S, F)
            sel_L = feat_at[:, IDX_L]
            sel_orient_fwd = feat_at[:, IDX_ORIENT_FWD]

            # Layer 3: apply V4 classifier to PRE-SetTransformer per-site tokens.
            # (What V4 would say if this bag had 1 site.)
            per_site_logits = classifier(site_pre_set.squeeze(0))  # (S, 1)
            per_site_score = torch.sigmoid(per_site_logits).squeeze(-1).cpu().numpy()  # (S,)

            row = {
                'tnp_id': tnp,
                'family': tnp_family.get(tnp, '?'),
                'n_native': n_native,
                'S': S,

                # --- Layer 1 (candidate scorer) ---
                'cand_top1_med':   float(np.median(top1_val[np.isfinite(top1_val)])),
                'cand_margin_med': float(np.median(top1_margin[np.isfinite(top1_margin)])),
                'cand_ent_med':    float(np.median(ca_ent)),
                'sel_L_med':       float(np.median(sel_L)),
                'frac_fwd':        float(np.mean(sel_orient_fwd > 0.5)),
                # top-1 alignment score above 0 threshold
                'frac_top1_gt0':   float(np.mean(top1_val > 0)),

                # --- Layer 2 (NC selector) ---
                'nc_max_med':      float(np.median(nc_max)),
                'nc_ent_med':      float(np.median(nc_ent)),
                'frac_ncmax_gt80': float(np.mean(nc_max > 0.80)),

                # --- Layer 3 (site → tnp aggregation, bypassing SetTransformer) ---
                'per_site_score_med':    float(np.median(per_site_score)),
                'per_site_score_max':    float(per_site_score.max()),
                'per_site_score_mean':   float(per_site_score.mean()),
                'per_site_score_top10p': float(np.mean(np.sort(per_site_score)[-max(1, S // 10):])),

                # --- Layer 4 (V4 base — for context) ---
                'v4_base_score':   float(1.0 / (1.0 + np.exp(-float(out['base_logit'].item())))),
                'v52_final_score': float(torch.sigmoid(out['logit']).item()),
            }
            rows.append(row)

            if (k+1) % 100 == 0:
                print(f'  [{k+1}/{len(keep_idx)}] {time.time()-t0:.0f}s', flush=True)

    with open(OUT_JSONL, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print(f'\n[done] {len(rows)} bags in {time.time()-t0:.0f}s')
    print(f'[out]  {OUT_JSONL}')

    # ============================ ANALYSIS ============================
    by_fam = defaultdict(list)
    for r in rows:
        by_fam[r['family']].append(r)

    def print_table(title, metrics):
        print()
        print('=' * 118)
        print(f'  {title}')
        print('=' * 118)
        hdr = f'  {"family":<12} {"bags":>5}'
        for name in metrics:
            hdr += f'  {name:>16}'
        print(hdr)
        print('  ' + '-' * 116)
        for fam in ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2'):
            rs = by_fam.get(fam, [])
            if not rs: continue
            label = 'POS' if fam == 'IS110' else 'NEG'
            line = f'  {fam:<10}({label}) {len(rs):>4}'
            for name in metrics:
                v = np.asarray([r[name] for r in rs])
                med = np.median(v)
                q25, q75 = np.quantile(v, [.25, .75])
                line += f' {med:>5.2f}[{q25:>4.2f},{q75:>4.2f}]'
            print(line)

    print_table('LAYER 1 — Candidate scorer (per-site medians per bag, aggregate over bags)',
                ('cand_top1_med', 'cand_margin_med', 'cand_ent_med',
                  'sel_L_med', 'frac_fwd', 'frac_top1_gt0'))
    print_table('LAYER 2 — NC selector (per-site medians per bag)',
                ('nc_max_med', 'nc_ent_med', 'frac_ncmax_gt80'))
    print_table('LAYER 3 — Site → Tnp (bypass SetTransformer)',
                ('per_site_score_med', 'per_site_score_max',
                  'per_site_score_mean', 'per_site_score_top10p'))
    print_table('LAYER 4 — V4 base (reference)',
                ('v4_base_score', 'v52_final_score'))

    # AUROCs — IS110-vs-others
    y = np.asarray([r['family'] == 'IS110' for r in rows], dtype=bool)
    print()
    print('=' * 118)
    print(f'  IS110-vs-others AUROC by layer / metric (n_native >= {N_MIN}, n_pos={int(y.sum())}, n_neg={int((~y).sum())})')
    print('=' * 118)
    print(f'  {"metric":<40} {"AUROC":>7}   {"note":<50}')
    print('  ' + '-' * 116)
    def _try(name, note='', invert=False):
        s = np.asarray([r[name] for r in rows])
        a = _auroc(-s if invert else s, y)
        arrow = '↓' if invert else '↑'
        print(f'  {name:<40} {a:>7.4f}   {note} ({arrow}=pos)')

    # Layer 1
    print('  LAYER 1 (higher = more like synthetic POS ⇒ should score IS110 UP if pipeline works):')
    _try('cand_top1_med',   'best alignment strength')
    _try('cand_margin_med', 'top1-top2 gap (more distinctive candidate)')
    _try('frac_top1_gt0',   'fraction of sites with a strong candidate')
    # entropy: HIGHER entropy = MORE spread = LESS like a clear synthetic positive
    _try('cand_ent_med', 'candidate-attn entropy (low=focused)', invert=True)

    # Layer 2
    print('  LAYER 2:')
    _try('nc_max_med',       'confident NC pick')
    _try('nc_ent_med',       'NC-attn entropy (low=focused)', invert=True)
    _try('frac_ncmax_gt80',  'fraction of sites with NC-attn > 0.80')

    # Layer 3 — pure site-level aggregations (no SetTransformer)
    print('  LAYER 3 (V4 classifier applied to pre-SetTransformer site tokens):')
    _try('per_site_score_med',    'median of per-site V4-classifier scores')
    _try('per_site_score_mean',   'mean of per-site scores')
    _try('per_site_score_max',    'max of per-site scores')
    _try('per_site_score_top10p', 'mean of top 10% per-site scores')

    # Layer 4
    print('  LAYER 4:')
    _try('v4_base_score',    'V4 base (with SetTransformer) — from Priority 1')
    _try('v52_final_score',  'V5.2 final (V4 + fusion) — from Priority 1')


if __name__ == '__main__':
    main()
