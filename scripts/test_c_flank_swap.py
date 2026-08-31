"""Test C — within-family flank swap.

Question:
  Is V4's NC MIL a REAL pairing detector (NC-attn depends on the flank
  content it can pair with), or just an NC-looks-positive detector
  (NC-attn is dominated by NC's own statistics)?

Method:
  For each n_native >= 25 bag, per site, replace the flank with a random
  flank drawn from ANOTHER site in the SAME FAMILY. NC content unchanged
  (structure cache still valid, no RNAplfold needed). Score with V5.2
  and compare to the original.

Records saved per site:
  - original: nc_attn_max at picked active NC, active_nc index,
    cand_top1 at that NC
  - perturbed: same fields
  - Δ for each

Aggregated per family (bags):
  - Δ V4 base score
  - fraction of bags where V5.2 final score drops > 0.05 (or > 0.20)
  - fraction of sites where perturbed still picks the SAME active NC
  - median Δ nc_attn max

Interpretation:
  A. If NC-attn is stable + score doesn't drop under flank swap
     → NC-MIL is content-based ("NC-looking-positive" detector).
     Model isn't doing cognate pairing; it's just NC classification.
  B. If NC-attn breaks down + score drops
     → Model is using the flank↔NC relationship (real pairing signal).
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
OUT_JSONL = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/test_c_flank_swap.jsonl'
N_MIN = 25
BAGS_PER_FAMILY = 30   # small sample per family for speed
SEED = 0


def load_records():
    """Load all real records into memory. Group by tnp_id; also index by family."""
    print('[load] scanning real_all.jsonl ...', flush=True)
    by_tnp: dict[str, list[dict]] = defaultdict(list)
    family_flanks: dict[str, list[str]] = defaultdict(list)   # family -> list of flank strings
    with open(f'{RUN}/real_all.jsonl') as f:
        for line in f:
            r = json.loads(line)
            tid = r['transposase_id']
            fam = r.get('generator_metadata', {}).get('is_family', '?')
            by_tnp[tid].append(r)
            family_flanks[fam].append(r['inputs']['flank'])
    print(f'[load] {len(by_tnp)} bags, ' +
          ', '.join(f'{fam}={len(v)}flanks' for fam, v in family_flanks.items()),
          flush=True)
    return by_tnp, family_flanks


def sample_bags(by_tnp, family_flanks):
    """Sample BAGS_PER_FAMILY tnps per family (only bags with n_native >= N_MIN)."""
    rng = random.Random(SEED)
    fam_to_tnps: dict[str, list[str]] = defaultdict(list)
    for tid, recs in by_tnp.items():
        if len(recs) < N_MIN:
            continue
        fam = recs[0].get('generator_metadata', {}).get('is_family', '?')
        fam_to_tnps[fam].append(tid)

    sampled = {}
    for fam, tnps in fam_to_tnps.items():
        rng.shuffle(tnps)
        sampled[fam] = tnps[:BAGS_PER_FAMILY]
    return sampled


def score_bag(model, cache, records, device, flank_override_list=None):
    """Preprocess `records` (per-site), collate as a single bag, forward.

    If `flank_override_list` is given (same length as records), replace each
    site's flank string with that entry before preprocessing.
    """
    site_outs = []
    for i, rec in enumerate(records):
        rec_use = rec
        if flank_override_list is not None:
            rec_use = copy.copy(rec)
            rec_use['inputs'] = dict(rec['inputs'])
            rec_use['inputs']['flank'] = flank_override_list[i]
        out = preprocess_site(rec_use, structure_cache=cache)
        site_outs.append(out)

    # Build a bag item.
    active_ncs, true_slots = [], []
    for o in site_outs:
        # active_noncoding_index from record labels (or -1 if unavailable).
        # We just need this to satisfy the batch collate; we won't use it
        # for the diagnostic.
        active_ncs.append(int(o.get('active_noncoding_index', -1)) if o.get('active_noncoding_index') is not None else -1)
        true_slots.append(-1)
    item = {
        'tnp_id': records[0]['transposase_id'],
        'site_ids': [o['site_id'] for o in site_outs],
        'is_positive': False,   # ignored at inference time
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
    # Extract per-site diagnostics.
    nc_attn = out['nc_attn'].float().cpu().numpy()[0]      # (S, N)
    cand_raw = out['cand_raw'].float().cpu().numpy()[0]     # (S, N, K)
    cand_mask = batch['candidate_mask'].cpu().numpy()[0]    # (S, N, K)
    active_nc_pick = nc_attn.argmax(axis=-1)                # (S,)
    S, N, K = cand_raw.shape
    cr_at = cand_raw[np.arange(S), active_nc_pick, :]
    cm_at = cand_mask[np.arange(S), active_nc_pick, :]
    cr_at_masked = np.where(cm_at, cr_at, -np.inf)
    cand_top1 = cr_at_masked.max(axis=-1)                    # (S,)
    nc_max = nc_attn.max(axis=-1)                            # (S,)
    return {
        'logit': float(out['logit'].item()),
        'base_logit': float(out['base_logit'].item()),
        'base_score': float(1.0 / (1.0 + np.exp(-float(out['base_logit'].item())))),
        'final_score': float(torch.sigmoid(out['logit']).item()),
        'active_nc_pick': active_nc_pick.tolist(),
        'nc_max': nc_max.tolist(),
        'cand_top1': cand_top1.tolist(),
    }


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'[ckpt] epoch {ckpt["epoch"]}', flush=True)

    cache = StructureCache(f'{RUN}/real_all_u16.index.json')
    by_tnp, family_flanks = load_records()
    sampled = sample_bags(by_tnp, family_flanks)
    for fam, tnps in sampled.items():
        print(f'  {fam}: {len(tnps)} bags sampled', flush=True)

    rng = random.Random(SEED + 1)
    t0 = time.time()
    results = []
    for fam, tnps in sampled.items():
        for tnp_idx, tid in enumerate(tnps):
            recs = by_tnp[tid]
            # If a bag is huge (>50 sites), subsample to 50 for consistency
            # with training regime.
            if len(recs) > 50:
                recs_use = rng.sample(recs, 50)
            else:
                recs_use = recs

            # Draw a random flank from another site in the SAME family for each
            # site (with rejection if same string; we don't index by tnp id).
            flank_pool = family_flanks[fam]
            perturbed_flanks = []
            for r in recs_use:
                orig = r['inputs']['flank']
                for _ in range(20):
                    donor = rng.choice(flank_pool)
                    if donor != orig:
                        break
                perturbed_flanks.append(donor)

            orig_out = score_bag(model, cache, recs_use, device)
            pert_out = score_bag(model, cache, recs_use, device,
                                    flank_override_list=perturbed_flanks)

            # Per-site diagnostics — same active-NC decision?
            same_nc = int(sum(1 for a, b in zip(orig_out['active_nc_pick'],
                                                  pert_out['active_nc_pick']) if a == b))
            n_sites = len(orig_out['active_nc_pick'])
            row = {
                'tnp_id': tid, 'family': fam, 'n_sites_used': n_sites,
                'orig_base_score':  orig_out['base_score'],
                'orig_final_score': orig_out['final_score'],
                'pert_base_score':  pert_out['base_score'],
                'pert_final_score': pert_out['final_score'],
                'd_base':  pert_out['base_score']  - orig_out['base_score'],
                'd_final': pert_out['final_score'] - orig_out['final_score'],
                'same_nc_frac':      same_nc / max(1, n_sites),
                'orig_nc_max_med':   float(np.median(orig_out['nc_max'])),
                'pert_nc_max_med':   float(np.median(pert_out['nc_max'])),
                'orig_cand_top1_med': float(np.median(orig_out['cand_top1'])),
                'pert_cand_top1_med': float(np.median(pert_out['cand_top1'])),
                'd_nc_max_med':       float(np.median(pert_out['nc_max']) -
                                              np.median(orig_out['nc_max'])),
                'd_cand_top1_med':    float(np.median(pert_out['cand_top1']) -
                                              np.median(orig_out['cand_top1'])),
            }
            results.append(row)
            if (len(results)) % 20 == 0:
                print(f'  [{len(results)}] {time.time()-t0:.0f}s '
                      f'({len(results)/max(1,time.time()-t0):.2f} bags/s)', flush=True)

    with open(OUT_JSONL, 'w') as fh:
        for r in results:
            fh.write(json.dumps(r) + '\n')
    print(f'\n[done] {len(results)} bags in {time.time()-t0:.0f}s')
    print(f'[out]  {OUT_JSONL}')

    # ============ Analysis per family ============
    by_fam = defaultdict(list)
    for r in results:
        by_fam[r['family']].append(r)

    print()
    print('=' * 108)
    print(f'  TEST C — within-family flank swap  (bags per family = {BAGS_PER_FAMILY})')
    print('=' * 108)
    print(f'  {"family":<12} {"bags":>4}  '
          f'{"orig base":>10} {"orig final":>10} '
          f'{"pert base":>10} {"pert final":>10}   '
          f'{"Δ base (med)":>13} {"Δ final (med)":>13}   '
          f'{"same_NC frac":>13}   '
          f'{"Δ nc_max":>10} {"Δ cand_top1":>12}')
    print('  ' + '-' * 106)
    for fam in ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2'):
        rs = by_fam.get(fam, [])
        if not rs: continue
        def _med(k):
            return float(np.median([r[k] for r in rs]))
        label = 'POS' if fam == 'IS110' else 'NEG'
        print(f'  {fam:<10}({label}) {len(rs):>3}  '
              f'{_med("orig_base_score"):>10.3f} {_med("orig_final_score"):>10.3f} '
              f'{_med("pert_base_score"):>10.3f} {_med("pert_final_score"):>10.3f}   '
              f'{_med("d_base"):>+13.3f} {_med("d_final"):>+13.3f}   '
              f'{_med("same_nc_frac"):>13.3f}   '
              f'{_med("d_nc_max_med"):>+10.3f} {_med("d_cand_top1_med"):>+12.3f}')

    print()
    print('  Interpretation:')
    print('    - If same_NC frac >= 0.90 AND Δ nc_max ~ 0 AND Δ base ~ 0 for a family,')
    print('      NC-MIL for that family is FLANK-INDEPENDENT (content-only detector).')
    print('    - If same_NC frac < 0.5 AND Δ nc_max << 0 AND Δ base << 0,')
    print('      NC-MIL for that family IS using flank↔NC pairing signal.')


if __name__ == '__main__':
    main()
