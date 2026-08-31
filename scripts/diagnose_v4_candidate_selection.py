"""Candidate-selection diagnostic on V4 main-arm epoch-14 checkpoint.

For each site in val_v4, compute:
  - the model's top-1 candidate at the active NC region
  - its (orient, L, flank_start, nc_start)
  - the true (labeled) (orient, L, target_position, guide_position)
  - agreement flags per axis

Aggregate per (is_positive, violation_profile):
  - fraction of sites where model.orient == label.orient
  - fraction where model.L == label.L
  - fraction where |model.flank_start - label.target_start| <= tolerance
  - fraction where |model.nc_start - label.guide_start| <= tolerance
  - mean cand_raw score at model.top-1 vs. mean cand_raw at label-true candidate

Also emit per-tnp final scores stratified by profile — including level3 —
to check where the paired counterfactual lands in the P(positive) distribution.

Question this answers:
  For wrong_position_consistency negatives —
    (A) does the model still pick candidates whose flank_start matches the
        labeled (moved) target position? If yes -> the per-site scorer works;
        the failure is at the Set Transformer aggregator (it doesn't compare
        positions across sites).
    (B) or does the model drift toward "typical" positions (i.e. ignore the
        moved target and pick alignments elsewhere)? If yes -> the per-site
        candidate scorer is the problem.

  Similarly for wrong_structure_role_consistency (though the position axis is
  less informative here; check orient/L consistency and cand_raw magnitude).
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
from preprocess.candidates import FEATURE_NAMES
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset

BASE = '/global/scratch/users/kh36969/DL_novel_guide_editor'
CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v1_on_v4_main/best.pt'
SPLIT = 'val_v4'   # val_v4 has ALL profiles including level3_paired
POS_TOL = 2         # bp tolerance for "position match"

# indices into candidate_features (F axis)
IDX_ORIENT_FWD = FEATURE_NAMES.index('orient_fwd')
IDX_ORIENT_RC = FEATURE_NAMES.index('orient_rc')
IDX_L = FEATURE_NAMES.index('L')
IDX_FLANK_START = FEATURE_NAMES.index('flank_start_norm')
IDX_NC_START = FEATURE_NAMES.index('nc_start_norm')


def _select_top_active(cand_raw, cand_mask, active_nc_idx):
    """For each (batch, site), return top-1 candidate slot in the active NC.

    cand_raw:   (B, S, N_nc, K)
    cand_mask:  (B, S, N_nc, K)   (1 for valid, 0 for pad)
    active_nc:  (B, S)             int64, or -1 for sites without active NC

    Returns (B, S) int slot index (0..K-1) — or -1 if no active NC / masked.
    """
    B, S, N, K = cand_raw.shape
    # Gather cand_raw at active_nc_idx (must be int64 for gather)
    idx = active_nc_idx.long().clamp(min=0)[..., None, None].expand(-1, -1, 1, K)
    cr = cand_raw.gather(2, idx).squeeze(2)  # (B, S, K)
    cm = cand_mask.gather(2, idx).squeeze(2)  # (B, S, K)
    cr = cr.masked_fill(~cm.bool(), float('-inf'))
    slot = cr.argmax(dim=-1)  # (B, S)
    # Sites without active NC → -1
    slot = torch.where(active_nc_idx.long() < 0, torch.full_like(slot, -1), slot)
    return slot, cr


def main():
    device = torch.device('cuda')
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f'ckpt: epoch {ckpt["epoch"]}, val AUPRC {ckpt["auprc"]:.4f}', flush=True)

    cache = StructureCache(f'{BASE}/structure/{SPLIT}_u16.index.json')
    ds = TnpGroupedDataset(f'{BASE}/splits/{SPLIT}.jsonl', cache,
                            site_subsample_size=50, rng_seed=0)
    dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=8, shuffle=False,
                    num_workers=4,
                    collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
                    persistent_workers=True, pin_memory=True)
    print(f'{SPLIT}: {len(ds)} tnps', flush=True)

    # Also need per-site labels not in the collate. Re-parse the JSONL to
    # attach (violation_profile, target_position, guide_position, guide_length,
    # match_orientation) per site.
    print(f'building per-site label lookup ...', flush=True)
    per_site_meta: dict[tuple[str, str], dict] = {}
    pos_by_tnp: dict[str, dict] = {}
    with open(f'{BASE}/splits/{SPLIT}.jsonl') as f:
        for line in f:
            r = json.loads(line)
            lab = r['labels']
            tnp = r['transposase_id']
            per_site_meta[(tnp, r['site_id'])] = {
                'is_positive': lab.get('is_positive'),
                'violation_profile': lab.get('violation_profile'),
                'target_start': lab.get('target_position_in_flank', [None, None])[0],
                'target_end': lab.get('target_position_in_flank', [None, None])[1],
                'guide_start': (lab.get('guide_span_in_active_noncoding') or [None, None])[0],
                'guide_end': (lab.get('guide_span_in_active_noncoding') or [None, None])[1],
                'guide_length': lab.get('guide_length'),
                'match_orientation': lab.get('match_orientation'),
                'active_nc_index': lab.get('active_noncoding_index'),
                'num_nc_regions': lab.get('num_noncoding_regions'),
                'nc_lengths': [len(x) for x in r['inputs']['noncoding_regions']],
            }
            if tnp not in pos_by_tnp:
                pos_by_tnp[tnp] = {
                    'is_positive': lab.get('is_positive'),
                    'violation_profile': lab.get('violation_profile'),
                    'tnp_strength': (r.get('generator_metadata') or {}).get('tnp_strength'),
                }

    print(f'running inference on {len(ds)} tnps ...', flush=True)
    t0 = time.time()
    per_site_rows = []
    per_tnp_scores = []
    with torch.no_grad():
        for b in dl:
            b_dev = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
                     for k, v in b.items()}
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model(b_dev['candidate_patches'], b_dev['candidate_features'],
                             b_dev['candidate_mask'], b_dev['nc_region_mask'])
            # tnp-level score
            score = torch.sigmoid(out['logit']).float().cpu().numpy()  # (B,)
            for i, tid in enumerate(b['tnp_id']):
                m = pos_by_tnp.get(tid, {})
                per_tnp_scores.append({
                    'tnp_id': tid,
                    'score': float(score[i]),
                    'is_positive': m.get('is_positive'),
                    'violation_profile': m.get('violation_profile'),
                    'tnp_strength': m.get('tnp_strength'),
                })

            # per-site: extract top-1 candidate at active NC
            cand_raw = out['cand_raw'].float().cpu()  # (B, S, N, K)
            cand_mask = b['candidate_mask']            # (B, S, N, K) already CPU tensor
            cand_feats = b['candidate_features']       # (B, S, N, K, F)
            active_nc = b['active_nc_index']           # (B, S)
            site_ids = b['site_ids']                   # list[list[str]]
            tnp_ids = b['tnp_id']

            slot, cr = _select_top_active(cand_raw, cand_mask, active_nc)  # (B, S), (B, S, K)
            B, S = slot.shape
            for bi in range(B):
                tid = tnp_ids[bi]
                for si in range(S):
                    sid_list = site_ids[bi]
                    if si >= len(sid_list):
                        continue
                    sid = sid_list[si]
                    key = (tid, sid)
                    lab = per_site_meta.get(key)
                    if lab is None:
                        continue
                    anc = int(active_nc[bi, si].item())
                    if anc < 0 or int(slot[bi, si].item()) < 0:
                        per_site_rows.append({
                            'tnp_id': tid, 'site_id': sid,
                            'is_positive': lab['is_positive'],
                            'violation_profile': lab['violation_profile'],
                            'active_nc_index': anc,
                            'selected_slot': -1,
                        })
                        continue
                    slot_k = int(slot[bi, si].item())
                    max_cand_score = float(cr[bi, si, slot_k].item())
                    # gather features at (bi, si, anc, slot_k, :)
                    fvec = cand_feats[bi, si, anc, slot_k]  # (F,)
                    is_fwd = bool(fvec[IDX_ORIENT_FWD] > 0.5)
                    is_rc = bool(fvec[IDX_ORIENT_RC] > 0.5)
                    model_orient = 'forward' if is_fwd else ('reverse_complement' if is_rc else '?')
                    model_L = int(round(float(fvec[IDX_L])))
                    flank_len = 120  # canonical
                    model_target_start = int(round(float(fvec[IDX_FLANK_START]) * flank_len))
                    nc_len = lab['nc_lengths'][anc]
                    model_nc_start = int(round(float(fvec[IDX_NC_START]) * max(1, nc_len)))

                    per_site_rows.append({
                        'tnp_id': tid, 'site_id': sid,
                        'is_positive': lab['is_positive'],
                        'violation_profile': lab['violation_profile'],
                        'active_nc_index': anc,
                        'selected_slot': slot_k,
                        'max_cand_score': max_cand_score,
                        'true_orient': lab['match_orientation'],
                        'true_L': lab['guide_length'],
                        'true_target_start': lab['target_start'],
                        'true_guide_start': lab['guide_start'],
                        'model_orient': model_orient,
                        'model_L': model_L,
                        'model_target_start': model_target_start,
                        'model_nc_start': model_nc_start,
                    })

    print(f'inference done in {time.time()-t0:.1f}s ({len(per_site_rows)} site rows)', flush=True)
    print()

    # Aggregate per (is_positive, violation_profile)
    print('=' * 90)
    print(f'CANDIDATE-SELECTION ACCURACY per profile (val_v4, pos_tol={POS_TOL}bp)')
    print('=' * 90)
    groups = defaultdict(list)
    for r in per_site_rows:
        if r['selected_slot'] < 0:
            continue
        key = 'POS' if r['is_positive'] else (r['violation_profile'] or 'unknown')
        groups[key].append(r)

    print(f'{"profile":<40} {"n":>7} {"orient=":>7} {"L=":>6} '
          f'{"tgt≤"+str(POS_TOL):>7} {"tgt≤5":>6} {"gs≤"+str(POS_TOL):>6} '
          f'{"score":>7}')
    for k in ('POS', 'level1_marginal_matched', 'level3_paired_counterfactual',
               'wrong_orientation_consistency', 'wrong_length_consistency',
               'wrong_position_consistency', 'wrong_structure_role_consistency'):
        rows = groups.get(k, [])
        if not rows:
            continue
        n = len(rows)
        # Accuracy per axis
        orient_hit = 0; L_hit = 0; tgt_hit_tol = 0; tgt_hit_5 = 0; gs_hit_tol = 0
        scores = []
        for r in rows:
            if r.get('true_orient') and r['true_orient'] == r['model_orient']:
                orient_hit += 1
            if r.get('true_L') and r['true_L'] == r['model_L']:
                L_hit += 1
            if r.get('true_target_start') is not None:
                d = abs(r['model_target_start'] - r['true_target_start'])
                if d <= POS_TOL: tgt_hit_tol += 1
                if d <= 5: tgt_hit_5 += 1
            if r.get('true_guide_start') is not None:
                d = abs(r['model_nc_start'] - r['true_guide_start'])
                if d <= POS_TOL: gs_hit_tol += 1
            scores.append(r['max_cand_score'])
        mean_score = float(np.mean(scores)) if scores else float('nan')
        print(f'{k:<40} {n:>7} '
              f'{orient_hit/n:>7.3f} {L_hit/n:>6.3f} '
              f'{tgt_hit_tol/n:>7.3f} {tgt_hit_5/n:>6.3f} '
              f'{gs_hit_tol/n:>6.3f} {mean_score:>7.2f}')

    # Per-tnp final score distribution
    print()
    print('=' * 90)
    print('PER-TNP SCORE DISTRIBUTION by profile (val_v4)')
    print('=' * 90)
    by_prof = defaultdict(list)
    for r in per_tnp_scores:
        key = ('POS-' + (r['tnp_strength'] or 'unknown')) if r['is_positive'] else (r['violation_profile'] or 'unknown')
        by_prof[key].append(r['score'])
    print(f'{"group":<45} {"n":>5} {"Q10":>7} {"Q25":>7} {"median":>8} {"Q75":>7} {"Q90":>7} {"mean":>7}')
    for k in ('POS-strong', 'POS-moderate', 'POS-weak',
               'level1_marginal_matched', 'wrong_orientation_consistency',
               'wrong_length_consistency', 'wrong_position_consistency',
               'wrong_structure_role_consistency', 'level3_paired_counterfactual'):
        vals = by_prof.get(k, [])
        if not vals:
            continue
        a = np.asarray(vals)
        q = np.quantile(a, [.10, .25, .5, .75, .90])
        print(f'{k:<45} {len(a):>5} {q[0]:>7.3f} {q[1]:>7.3f} '
              f'{q[2]:>8.3f} {q[3]:>7.3f} {q[4]:>7.3f} {a.mean():>7.3f}')

    # Focused wrong-position analysis: distribution of model_target_start
    # per-tnp for wrong_position_consistency negatives vs positives.
    print()
    print('=' * 90)
    print('WRONG-POSITION FAILURE MODE: model vs. true target start position')
    print('=' * 90)
    for prof in ('POS', 'wrong_position_consistency'):
        rows = groups.get(prof, [])
        if not rows:
            continue
        rows = [r for r in rows if r.get('true_target_start') is not None]
        deltas = [r['model_target_start'] - r['true_target_start'] for r in rows]
        abs_deltas = [abs(d) for d in deltas]
        a = np.asarray(deltas); b = np.asarray(abs_deltas)
        print(f'{prof} n={len(rows)}')
        print(f'  |Δ| quantiles: Q10={np.quantile(b,.1):.1f} Q25={np.quantile(b,.25):.1f} '
              f'median={np.median(b):.1f} Q75={np.quantile(b,.75):.1f} Q90={np.quantile(b,.9):.1f}')
        print(f'  frac |Δ|<=2: {(b<=2).mean():.3f}   <=5: {(b<=5).mean():.3f}   <=10: {(b<=10).mean():.3f}')
        print(f'  Δ signed mean: {a.mean():+.2f}   std: {a.std():.2f}')

    # Save raw per-site rows for further offline analysis
    out_json = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/diag_v4_candidate_selection.jsonl'
    with open(out_json, 'w') as fh:
        for r in per_site_rows:
            fh.write(json.dumps(r) + '\n')
    print(f'\nsaved per-site rows to {out_json}')


if __name__ == '__main__':
    main()
