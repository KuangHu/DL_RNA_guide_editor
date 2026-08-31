"""Terminal-repeat shortcut probe: where does the model pick candidates
inside the NC region, per family?

For each n_native >= 25 real bag, extract the model-picked candidate at each
site's active NC region and record its NC start position (bp and normalized).

Then per family:
  - Distribution of nc_start_bp (median, quantiles)
  - Fraction of sites with picked candidate in first 30 bp of NC (5' edge)
  - Fraction in last 30 bp of NC (3' edge)
  - Fraction "in the middle" (30 bp < nc_start < nc_len-30-L)
  - Distance to nearest NC boundary (min(nc_start, nc_len - nc_end))

Hypothesis: if IS30/IS903 candidates cluster near NC start (or end) — much more
than IS110 — that suggests the model is finding element-terminal cis motifs
(inverted repeats / terminal sequences), not ncRNA guides.
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
from preprocess.candidates import FEATURE_NAMES

RUN = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt'
OUT_JSONL = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/real_ncpos_n25.jsonl'
N_MIN = 25

IDX_ORIENT_FWD = FEATURE_NAMES.index("orient_fwd")
IDX_L          = FEATURE_NAMES.index("L")
IDX_NC_START   = FEATURE_NAMES.index("nc_start_norm")


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    cache = StructureCache(f'{RUN}/real_all_u16.index.json')
    tnp_family: dict[str, str] = {}
    # Also build tnp_id -> {site_id -> nc_lengths per NC region} for exact per-site
    # NC-length lookup during inference.
    site_nc_lens: dict[tuple[str, str], list[int]] = {}
    with open(f'{RUN}/real_all.jsonl') as f:
        for line in f:
            r = json.loads(line)
            tid = r['transposase_id']
            sid = r['site_id']
            if tid not in tnp_family:
                tnp_family[tid] = r.get('generator_metadata', {}).get('is_family', '?')
            site_nc_lens[(tid, sid)] = [len(x) for x in r['inputs']['noncoding_regions']]

    ds = TnpGroupedDataset(
        f'{RUN}/real_all.jsonl', cache,
        site_subsample_size=50, rng_seed=0,
    )
    keep_idx = [i for i, tnp in enumerate(ds.tnp_ids)
                 if len(ds._tnp_lines[tnp]) >= N_MIN]
    print(f'[dataset] {len(keep_idx)} bags with n_native >= {N_MIN}', flush=True)

    # For each bag we'll dump SITE-level rows (not bag-level) so we can compute
    # distributions of "where in NC" the model picks.
    t0 = time.time()
    site_rows = []
    with torch.no_grad():
        for k, i in enumerate(keep_idx):
            tnp = ds.tnp_ids[i]
            fam = tnp_family.get(tnp, '?')
            n_native = len(ds._tnp_lines[tnp])
            item = ds[i]
            # Per-site NC lengths — look up by (tnp_id, site_id).
            per_site_nc_lens = [site_nc_lens.get((tnp, sid), []) for sid in item['site_ids']]
            batch = collate_tnp_batch([item], to_torch=True)
            batch_dev = {k2: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                          for k2, v in batch.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(batch_dev['candidate_patches'], batch_dev['candidate_features'],
                             batch_dev['candidate_mask'], batch_dev['nc_region_mask'])

            nc_attn   = out['nc_attn'].float().cpu().numpy()[0]        # (S, N)
            cand_raw  = out['cand_raw'].float().cpu().numpy()[0]       # (S, N, K)
            cand_mask = batch['candidate_mask'].cpu().numpy()[0]        # (S, N, K)
            cand_feats = batch['candidate_features'].cpu().numpy()[0]   # (S, N, K, F)

            S, N, K = cand_raw.shape
            active_nc = nc_attn.argmax(axis=-1)                          # (S,)
            cr_at = cand_raw[np.arange(S), active_nc, :]
            cm_at = cand_mask[np.arange(S), active_nc, :]
            cr_at_masked = np.where(cm_at, cr_at, -np.inf)
            slot = cr_at_masked.argmax(axis=-1)                          # (S,)

            feat_at = cand_feats[np.arange(S), active_nc, slot, :]       # (S, F)
            sel_L = feat_at[:, IDX_L].astype(np.float32)
            sel_nc_start_norm = feat_at[:, IDX_NC_START].astype(np.float32)
            sel_orient_fwd = feat_at[:, IDX_ORIENT_FWD].astype(np.float32)

            # Look up per-site NC lengths from JSONL. Pad short lists to 3.
            def _nc_len(si: int) -> int:
                lens = per_site_nc_lens[si] if si < len(per_site_nc_lens) else []
                lens = list(lens) + [0] * (3 - len(lens))
                return lens[active_nc[si]]
            nc_len_per_site = np.asarray([_nc_len(si) for si in range(S)],
                                          dtype=np.float32)
            # Model output nc_start_norm is (cand.nc_start / max(1, nc_len)).
            sel_nc_start_bp = (sel_nc_start_norm * nc_len_per_site).astype(np.int32)
            # End position of the picked candidate span
            sel_nc_end_bp = sel_nc_start_bp + sel_L.astype(np.int32)
            # Distance from picked span to the nearest NC boundary
            dist_5p = sel_nc_start_bp                                    # distance to 5' end
            dist_3p = (nc_len_per_site - sel_nc_end_bp).astype(np.int32) # distance to 3' end
            dist_nearest = np.minimum(dist_5p, dist_3p)

            # Save per-SITE rows (one per bag * S sites — for aggregation)
            for si in range(S):
                site_rows.append({
                    'tnp_id': tnp,
                    'family': fam,
                    'n_native': n_native,
                    'S_used': S,
                    'active_nc': int(active_nc[si]),
                    'nc_len_at_active': int(nc_len_per_site[si]),
                    'sel_L': int(sel_L[si]),
                    'sel_orient_fwd': bool(sel_orient_fwd[si] > 0.5),
                    'sel_nc_start_bp': int(sel_nc_start_bp[si]),
                    'sel_nc_start_norm': float(sel_nc_start_norm[si]),
                    'dist_5p': int(dist_5p[si]),
                    'dist_3p': int(dist_3p[si]),
                    'dist_nearest_boundary': int(dist_nearest[si]),
                })

            if (k+1) % 100 == 0:
                print(f'  [{k+1}/{len(keep_idx)}] {time.time()-t0:.0f}s', flush=True)

    with open(OUT_JSONL, 'w') as fh:
        for r in site_rows:
            fh.write(json.dumps(r) + '\n')
    print(f'\n[done] {len(site_rows)} site rows in {time.time()-t0:.0f}s')
    print(f'[out]  {OUT_JSONL}')

    # ============ Per-family site-level distribution ============
    by_fam = defaultdict(list)
    for r in site_rows:
        by_fam[r['family']].append(r)

    print()
    print('=' * 108)
    print(f'  Where does the model pick candidates in the NC region? (per site, n_native >= {N_MIN})')
    print('=' * 108)
    print(f'  {"family":<12} {"sites":>6}   {"nc_len":<14} {"nc_start bp":<21}   '
          f'{"nc_start norm":<15}  {"dist to nearest bd":<20}')
    print(f'  {"":<12} {"":>6}   {"med [Q10,Q90]":<14} {"med [Q10,Q90]":<21}   '
          f'{"med [Q10,Q90]":<15}  {"med [Q10,Q90]":<20}')
    print('  ' + '-' * 106)
    for fam in ('IS110','IS30','IS903','IS10-R','ISLdl1','ISAjo2'):
        rs = by_fam.get(fam, [])
        if not rs: continue
        nc_len = np.asarray([r['nc_len_at_active'] for r in rs])
        nc_start = np.asarray([r['sel_nc_start_bp'] for r in rs])
        nc_norm = np.asarray([r['sel_nc_start_norm'] for r in rs])
        dist = np.asarray([r['dist_nearest_boundary'] for r in rs])
        label = 'POS' if fam == 'IS110' else 'NEG'
        print(f'  {fam:<10}({label}) {len(rs):>4}   '
              f'{int(np.median(nc_len)):>3}[{int(np.quantile(nc_len,.1)):>3},{int(np.quantile(nc_len,.9)):>3}]  '
              f'{int(np.median(nc_start)):>4}[{int(np.quantile(nc_start,.1)):>4},{int(np.quantile(nc_start,.9)):>4}]   '
              f'{np.median(nc_norm):>4.2f}[{np.quantile(nc_norm,.1):>4.2f},{np.quantile(nc_norm,.9):>4.2f}]  '
              f'{int(np.median(dist)):>3}[{int(np.quantile(dist,.1)):>3},{int(np.quantile(dist,.9)):>3}]')

    print()
    print('=' * 108)
    print(f'  Terminal-repeat shortcut check — fraction of sites with picked candidate at NC edge')
    print('=' * 108)
    print(f'  {"family":<12} {"sites":>6}   {"5p ≤ 30 bp":>12}  {"3p ≤ 30 bp":>12}  {"any edge ≤ 30":>15}  {"middle":>10}')
    print('  ' + '-' * 106)
    for fam in ('IS110','IS30','IS903','IS10-R','ISLdl1','ISAjo2'):
        rs = by_fam.get(fam, [])
        if not rs: continue
        dist_5p = np.asarray([r['dist_5p'] for r in rs])
        dist_3p = np.asarray([r['dist_3p'] for r in rs])
        near_5 = (dist_5p <= 30).mean()
        near_3 = (dist_3p <= 30).mean()
        near_any = ((dist_5p <= 30) | (dist_3p <= 30)).mean()
        middle = ((dist_5p > 30) & (dist_3p > 30)).mean()
        label = 'POS' if fam == 'IS110' else 'NEG'
        print(f'  {fam:<10}({label}) {len(rs):>4}   {near_5:>12.3f}  {near_3:>12.3f}  '
              f'{near_any:>15.3f}  {middle:>10.3f}')

    # NC start histogram in bins of 25 bp for the first 300 bp of NC
    print()
    print('=' * 108)
    print(f'  NC-start position histogram (bp bins) per family — fraction of sites in each bin')
    print('=' * 108)
    bins = np.arange(0, 351, 25)
    hdr = f'  {"family":<12}'
    for lo in bins[:-1]:
        hdr += f'  {lo:>3}-{lo+24:<3}'
    print(hdr)
    print('  ' + '-' * (12 + 8 * (len(bins) - 1)))
    for fam in ('IS110','IS30','IS903','IS10-R','ISLdl1','ISAjo2'):
        rs = by_fam.get(fam, [])
        if not rs: continue
        nc_start = np.asarray([r['sel_nc_start_bp'] for r in rs])
        counts, _ = np.histogram(nc_start, bins=bins)
        frac = counts / max(1, counts.sum())
        label = 'POS' if fam == 'IS110' else 'NEG'
        line = f'  {fam:<10}({label})'
        for fr in frac:
            line += f'  {fr:>6.3f}'
        print(line)


if __name__ == '__main__':
    main()
