"""Zero-shot: load a saved V1 checkpoint and evaluate on ANY val/test jsonl+cache.

Reports overall AUROC/AUPRC + per-profile AUROC (via labels.violation_profile).
No training. Meant for sanity checks like: does 48B's best.pt still discriminate
paired_shuffle when the val pool contains other profiles too?
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

from model.v1 import V1Config, V1Model
from preprocess.tnp_dataset import (
    TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset,
)
from preprocess.site import StructureCache
from torch.utils.data import DataLoader


def load_ckpt(path):
    obj = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(obj, dict) and 'model' in obj:
        state = obj['model']; meta = obj
    else:
        state = obj; meta = {}
    return state, meta


def _auroc(scores, labels):
    from sklearn.metrics import roc_auc_score
    if len(set(labels.tolist())) < 2: return float('nan')
    return roc_auc_score(labels, scores)


def _auprc(scores, labels):
    from sklearn.metrics import average_precision_score
    if len(set(labels.tolist())) < 2: return float('nan')
    return average_precision_score(labels, scores)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--cache', required=True, help='structure cache index.json')
    ap.add_argument('--tnp-batch', type=int, default=8)
    ap.add_argument('--sites', type=int, default=50)
    ap.add_argument('--num-workers', type=int, default=4)
    ap.add_argument('--out', default=None)
    ap.add_argument('--use-multi-branch', action='store_true',
                    help='Load ckpt into a V1Model with use_multi_branch=True; also report '
                         'per-profile AUROC on s_pair_aux and s_geom_aux heads.')
    ap.add_argument('--use-explicit-geom-stats', action='store_true')
    ap.add_argument('--use-and-fusion', action='store_true',
                    help='Use AND fusion (product-of-sigmoids) for logit_final. Cannot be '
                         'autodetected — pass explicitly when evaluating a 48C1f checkpoint.')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[dev] {device}', flush=True)

    # Load ckpt
    state, meta = load_ckpt(args.ckpt)
    # Auto-detect optional multi-branch sub-features from the checkpoint state dict.
    use_additive = 'alpha_pair' in state or 'alpha_geom' in state
    normalize_aux = any(k.startswith('bn_pair_aux.') or k.startswith('bn_geom_aux.')
                          for k in state.keys())
    use_orient = any(k.startswith('orient_mlp.') or k.startswith('h_orient_aux.')
                       for k in state.keys())
    v1_cfg = V1Config(
        use_multi_branch=args.use_multi_branch,
        use_explicit_geom_stats=args.use_explicit_geom_stats or args.use_multi_branch,
        use_additive_fusion=use_additive,
        normalize_aux_logits=normalize_aux,
        use_and_fusion=args.use_and_fusion,
        use_orient_branch=use_orient,
    )
    print(f'[cfg] use_dispersion={v1_cfg.use_dispersion} use_pairing={v1_cfg.use_pairing} '
          f'use_multi_branch={v1_cfg.use_multi_branch} '
          f'use_explicit_geom_stats={v1_cfg.use_explicit_geom_stats}', flush=True)
    model = V1Model(v1_cfg).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[load] missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    if missing: print(f'  missing (first 5): {missing[:5]}', flush=True)
    if unexpected: print(f'  unexpected (first 5): {unexpected[:5]}', flush=True)
    model.eval()

    # Dataset
    cache = StructureCache(args.cache)
    ds = TnpGroupedDataset(args.jsonl, cache, site_subsample_size=args.sites)
    print(f'[data] tnps={len(ds)}', flush=True)
    dl = DataLoader(
        make_torch_tnp_dataset(ds),
        batch_size=args.tnp_batch, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda items: collate_tnp_batch(items, to_torch=True),
        pin_memory=(device.type == 'cuda'),
    )

    # Also map tnp_id -> violation_profile (via 1st record) for stratified AUROC.
    tnp2profile = {}
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            t = r['transposase_id']
            if t not in tnp2profile:
                if r['labels'].get('is_positive'):
                    tnp2profile[t] = 'positive'
                else:
                    tnp2profile[t] = r['labels'].get('violation_profile', 'unknown')

    # Inference
    all_logits = []
    all_pair_aux = []
    all_geom_aux = []
    all_orient_aux = []
    all_labels = []
    all_profiles = []
    all_tnp_ids = []
    n_batches = 0
    with torch.no_grad():
        for batch in dl:
            batch_gpu = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_gpu[k] = v.to(device, non_blocking=True)
                else:
                    batch_gpu[k] = v
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=(device.type == 'cuda')):
                out = model(
                    batch_gpu['candidate_patches'],
                    batch_gpu['candidate_features'],
                    batch_gpu['candidate_mask'],
                    batch_gpu['nc_region_mask'],
                )
            logits = out['logit'].detach().float().cpu().numpy()
            labels = batch['is_positive'].numpy().astype(np.int32)
            tnp_ids = list(batch['tnp_id'])
            all_logits.extend(logits.tolist())
            all_labels.extend(labels.tolist())
            all_tnp_ids.extend(tnp_ids)
            if out.get('s_pair_aux') is not None:
                all_pair_aux.extend(out['s_pair_aux'].detach().float().cpu().numpy().tolist())
            if out.get('s_geom_aux') is not None:
                all_geom_aux.extend(out['s_geom_aux'].detach().float().cpu().numpy().tolist())
            if out.get('s_orient_aux') is not None:
                all_orient_aux.extend(out['s_orient_aux'].detach().float().cpu().numpy().tolist())
            for t in tnp_ids:
                all_profiles.append(tnp2profile.get(t, 'unknown'))
            n_batches += 1
            if n_batches % 50 == 0:
                print(f'  [batch {n_batches}] done', flush=True)

    scores = np.asarray(all_logits, dtype=np.float32)
    labels = np.asarray(all_labels, dtype=np.int32)
    profiles = np.asarray(all_profiles)

    # Overall
    overall_auroc = _auroc(scores, labels)
    overall_auprc = _auprc(scores, labels)
    print(f'\n[overall] AUROC={overall_auroc:.4f}  AUPRC={overall_auprc:.4f}  n_pos={(labels==1).sum()}  n_neg={(labels==0).sum()}')

    # Per profile
    pos_mask = (profiles == 'positive')
    pos_scores = scores[pos_mask]
    print(f'\n[per-profile: POS vs profile]')
    print(f'  {"profile":<28} {"n_neg":>6}  {"auroc":>6}  {"auprc":>6}  score_med(POS/NEG)')
    per_prof = {}
    unique_profiles = sorted(set(profiles.tolist()) - {'positive'})
    for prof in unique_profiles:
        neg_mask = (profiles == prof)
        sub_scores = np.concatenate([pos_scores, scores[neg_mask]])
        sub_labels = np.concatenate([np.ones(len(pos_scores), dtype=int),
                                       np.zeros(int(neg_mask.sum()), dtype=int)])
        au = _auroc(sub_scores, sub_labels)
        ap_ = _auprc(sub_scores, sub_labels)
        med_pos = float(np.median(pos_scores))
        med_neg = float(np.median(scores[neg_mask])) if neg_mask.sum() else float('nan')
        per_prof[prof] = {'n_neg': int(neg_mask.sum()), 'auroc': float(au), 'auprc': float(ap_),
                          'score_med_pos': med_pos, 'score_med_neg': med_neg}
        print(f'  {prof:<28} {int(neg_mask.sum()):>6}  {au:>6.4f}  {ap_:>6.4f}  {med_pos:>+.3f}/{med_neg:>+.3f}')

    # 48C1f-style paired Δ analysis. For each parent tnp (strip the __neg_ suffix),
    # find matched POS and NEG bags and compute per-branch Δ. This is a direct
    # invariance/discrimination check that AUROC alone can't tell us.
    NEG_SUFFIXES = {
        'paired_shuffle_v42':       '__neg_paired_shuffle_v42',
        'wrong_length_v42':         '__neg_wrong_length_v42',
        'wrong_orientation_v42':    '__neg_wrong_orientation_v42',
        'wrong_position_v42':       '__neg_wrong_position_v42',
        'wrong_structure_role_v42': '__neg_wrong_structure_role_v42',
    }
    paired_report = {}
    if all_pair_aux and all_geom_aux:
        pair_scores = np.asarray(all_pair_aux, dtype=np.float32)
        geom_scores = np.asarray(all_geom_aux, dtype=np.float32)
        has_orient = bool(all_orient_aux)
        orient_scores = np.asarray(all_orient_aux, dtype=np.float32) if has_orient else None
        # index by tnp_id
        idx_by_tnp = {t: i for i, t in enumerate(all_tnp_ids)}
        for prof, suffix in NEG_SUFFIXES.items():
            deltas_pair = []
            deltas_geom = []
            deltas_orient = []
            deltas_final = []
            for i, t in enumerate(all_tnp_ids):
                if profiles[i] != 'positive':
                    continue
                # Look for the paired NEG bag under the profile's suffix.
                neg_t = t + suffix
                j = idx_by_tnp.get(neg_t)
                if j is None:
                    continue
                deltas_pair.append(pair_scores[i] - pair_scores[j])
                deltas_geom.append(geom_scores[i] - geom_scores[j])
                deltas_final.append(scores[i] - scores[j])
                if has_orient:
                    deltas_orient.append(orient_scores[i] - orient_scores[j])
            if not deltas_pair:
                continue
            def _stats(vals):
                a = np.asarray(vals, dtype=np.float32)
                return {
                    'n':        int(len(a)),
                    'median':   float(np.median(a)),
                    'MAD':      float(np.median(np.abs(a - np.median(a)))),
                    'std':      float(a.std()),
                    'p_gt_0':   float((a > 0).mean()),
                    'q10':      float(np.quantile(a, 0.10)),
                    'q90':      float(np.quantile(a, 0.90)),
                }
            entry = {
                'pair':  _stats(deltas_pair),
                'geom':  _stats(deltas_geom),
                'final': _stats(deltas_final),
            }
            if has_orient:
                entry['orient'] = _stats(deltas_orient)
            paired_report[prof] = entry

        print(f'\n[paired Δ = s(POS_i) − s(NEG_i^profile), matched by parent tnp]')
        print(f'  {"profile":<26} {"head":<6} {"n":>4}  {"median":>7}  {"MAD":>5}  {"P(Δ>0)":>6}')
        for prof, s in paired_report.items():
            heads = ['pair', 'geom']
            if 'orient' in s:
                heads.append('orient')
            heads.append('final')
            for head in heads:
                d = s[head]
                print(f'  {prof:<26} {head:<6} {d["n"]:>4}  {d["median"]:>+7.3f}  {d["MAD"]:>5.3f}  {d["p_gt_0"]:>6.3f}')

    # Per-head diagnostic (48C1c multi-branch): what does each aux head see?
    aux_report = {}
    if all_pair_aux and all_geom_aux:
        pair_scores = np.asarray(all_pair_aux, dtype=np.float32)
        geom_scores = np.asarray(all_geom_aux, dtype=np.float32)
        head_pairs = [('s_pair_aux', pair_scores), ('s_geom_aux', geom_scores)]
        if all_orient_aux:
            head_pairs.append(('s_orient_aux', np.asarray(all_orient_aux, dtype=np.float32)))
        for head_name, head_scores in head_pairs:
            print(f'\n[per-profile: POS vs profile — {head_name}]')
            print(f'  {"profile":<28} {"n_neg":>6}  {"auroc":>6}  score_med(POS/NEG)')
            per_prof_head = {}
            for prof in unique_profiles:
                neg_mask = (profiles == prof)
                sub_scores = np.concatenate([head_scores[pos_mask], head_scores[neg_mask]])
                sub_labels = np.concatenate([np.ones(int(pos_mask.sum()), dtype=int),
                                               np.zeros(int(neg_mask.sum()), dtype=int)])
                au = _auroc(sub_scores, sub_labels)
                med_p = float(np.median(head_scores[pos_mask]))
                med_n = float(np.median(head_scores[neg_mask])) if neg_mask.sum() else float('nan')
                per_prof_head[prof] = {'n_neg': int(neg_mask.sum()), 'auroc': float(au),
                                        'score_med_pos': med_p, 'score_med_neg': med_n}
                print(f'  {prof:<28} {int(neg_mask.sum()):>6}  {au:>6.4f}  {med_p:>+.3f}/{med_n:>+.3f}')
            aux_report[head_name] = per_prof_head

    # Score distribution diagnostics
    pos_stats = {'mean': float(pos_scores.mean()), 'std': float(pos_scores.std()),
                  'q10': float(np.quantile(pos_scores, 0.10)),
                  'q50': float(np.quantile(pos_scores, 0.50)),
                  'q90': float(np.quantile(pos_scores, 0.90))}
    neg_scores = scores[~pos_mask]
    neg_stats = {'mean': float(neg_scores.mean()), 'std': float(neg_scores.std()),
                  'q10': float(np.quantile(neg_scores, 0.10)),
                  'q50': float(np.quantile(neg_scores, 0.50)),
                  'q90': float(np.quantile(neg_scores, 0.90))}
    print(f'\n[score stats POS] {pos_stats}')
    print(f'[score stats NEG] {neg_stats}')

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'ckpt': args.ckpt, 'jsonl': args.jsonl,
                        'overall': {'auroc': overall_auroc, 'auprc': overall_auprc,
                                    'n_pos': int((labels==1).sum()), 'n_neg': int((labels==0).sum())},
                        'per_profile': per_prof,
                        'per_head_per_profile': aux_report,
                        'paired_delta': paired_report,
                        'pos_stats': pos_stats, 'neg_stats': neg_stats}, f, indent=2)
        print(f'\n[out] {args.out}', flush=True)


if __name__ == '__main__':
    main()
