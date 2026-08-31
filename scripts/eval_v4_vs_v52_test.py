"""Held-out test comparison: V4 main vs V5.2 Stage A on test_v4.jsonl.

Produces:
  - Full metric table (AUROC, AUPRC, HARD_*, per-profile AUROC, strength recall, nc_top1)
  - Per-group score-distribution quantiles (Q10/Q25/med/Q75/Q90) for both models
    * positives split by strength (strong/moderate/weak)
    * negatives split by violation_profile (incl. level3_paired)
  - Weak-positive alternative metrics: AUROC, AUPRC, quantiles, recall@1/5

Sanity check to watch:
  - level3_paired AUROC should stay near 0.5 (indistinguishable from positives
    on observable-sequence evidence). If V5.2 pushes it to 0.8+, that's a
    synthetic-shortcut warning.
  - Positive score distributions (strong/moderate/weak) should be stable
    between V4 and V5.2; large drops on strong/moderate would mean V5.2 is
    just re-calibrating positives rather than pushing hard negatives down.

No writes outside logs/. Single-GPU eval.
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.v1 import V1Config, V1Model
from preprocess.site import StructureCache
from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch, make_torch_tnp_dataset
from training.metrics import tnp_metrics, stratified_auroc, candidate_recall, nc_selection_accuracy, _auroc, _auprc
from training.train_v1 import _violation_profile_by_tnp, _tnp_strength_by_tnp, EASY_PROFILES

BASE = '/global/scratch/users/kh36969/DL_novel_guide_editor'
V4_CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v1_on_v4_main/best.pt'
V5_CKPT = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/checkpoints/v5_2_stageA_from_v4/best.pt'
SPLIT = 'test_v4'


def _load_model(ckpt_path: str, device: torch.device) -> tuple[V1Model, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt['cfg'])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


@torch.no_grad()
def _score_all(model: V1Model, dl: DataLoader, device: torch.device) -> dict:
    scores, labels, tnp_ids = [], [], []
    cand_at_active, true_slot_all, active_all, nc_attn_all = [], [], [], []
    for b in dl:
        b = {k: (v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
             for k, v in b.items()}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(b['candidate_patches'], b['candidate_features'],
                         b['candidate_mask'], b['nc_region_mask'])
        scores.append(torch.sigmoid(out['logit']).float().cpu().numpy())
        labels.append(b['is_positive'].cpu().numpy())
        tnp_ids.extend(list(b['tnp_id']))
        cr = out['cand_raw'].float().cpu().numpy()
        na = out['nc_attn'].float().cpu().numpy()
        ac = b['active_nc_index'].cpu().numpy()
        ts = b['true_slot_idx'].cpu().numpy()
        B, S, N, K = cr.shape
        for bi in range(B):
            for si in range(S):
                if int(ac[bi, si]) < 0 or int(ts[bi, si]) < 0:
                    continue
                cand_at_active.append(cr[bi, si, int(ac[bi, si])])
                true_slot_all.append(int(ts[bi, si]))
                active_all.append(int(ac[bi, si]))
                nc_attn_all.append(na[bi, si])
    return {
        'scores':      np.concatenate(scores),
        'labels':      np.concatenate(labels),
        'tnp_ids':     tnp_ids,
        'cand_scores': np.stack(cand_at_active, 0) if cand_at_active else None,
        'true_slots':  np.asarray(true_slot_all),
        'active_ncs':  np.asarray(active_all),
        'nc_attn':     np.stack(nc_attn_all, 0) if nc_attn_all else None,
    }


def _compute_all_metrics(res: dict, groups: np.ndarray, strengths: np.ndarray) -> dict:
    scores, labels = res['scores'], res['labels']
    labels_bool = labels.astype(bool)
    m = tnp_metrics(scores, labels)
    strat = stratified_auroc(scores, labels, groups)
    cand = candidate_recall(res['cand_scores'], res['true_slots'].tolist(), ks=(1, 5, 10))
    nc = nc_selection_accuracy(res['nc_attn'], res['active_ncs'].tolist())

    hard_mask = labels_bool | np.array([g not in EASY_PROFILES for g in groups])
    s_h, y_h = scores[hard_mask], labels_bool[hard_mask]
    hard_auroc = _auroc(s_h, y_h)
    hard_auprc = _auprc(s_h, y_h)

    called = scores > 0.5
    strength_recall = {}
    for lvl in ('strong', 'moderate', 'weak'):
        m2 = labels_bool & (strengths == lvl)
        if m2.any():
            strength_recall[lvl] = (float(called[m2].mean()), int(m2.sum()))

    return {
        'n_pos':   m['n_pos'],
        'n_neg':   m['n_neg'],
        'auroc':   m['auroc'],
        'auprc':   m['auprc'],
        'hard_auroc': hard_auroc,
        'hard_auprc': hard_auprc,
        'strat':   strat,
        'cand':    cand,
        'nc_top1': nc['nc_top1'],
        'strength_recall': strength_recall,
    }


def _quantile_row(vals: list[float]) -> tuple[float, float, float, float, float]:
    a = np.asarray(vals, dtype=np.float64)
    q = np.quantile(a, [0.10, 0.25, 0.50, 0.75, 0.90])
    return float(q[0]), float(q[1]), float(q[2]), float(q[3]), float(q[4])


def _weak_positive_stats(res: dict, strengths: np.ndarray) -> dict:
    scores = res['scores']
    labels_bool = res['labels'].astype(bool)
    # weak-positive AUROC/AUPRC: weak positives vs ALL negatives (i.e. same as full,
    # but restricted to weak positives + all negatives)
    weak_mask = labels_bool & (strengths == 'weak')
    neg_mask = ~labels_bool
    keep = weak_mask | neg_mask
    if not weak_mask.any() or not neg_mask.any():
        return {}
    s_w = scores[keep]
    y_w = weak_mask[keep].astype(bool)
    return {
        'n_weak_pos': int(weak_mask.sum()),
        'n_neg':      int(neg_mask.sum()),
        'weak_auroc': _auroc(s_w, y_w),
        'weak_auprc': _auprc(s_w, y_w),
    }


def main():
    device = torch.device('cuda')

    # Data
    cache = StructureCache(f'{BASE}/structure/{SPLIT}_u16.index.json')
    ds = TnpGroupedDataset(f'{BASE}/splits/{SPLIT}.jsonl', cache,
                            site_subsample_size=50, rng_seed=0)
    dl = DataLoader(make_torch_tnp_dataset(ds), batch_size=8, shuffle=False,
                    num_workers=4,
                    collate_fn=lambda x: collate_tnp_batch(x, to_torch=True),
                    persistent_workers=True, pin_memory=True)

    print(f'test tnps={len(ds)}', flush=True)
    gmap = _violation_profile_by_tnp(f'{BASE}/splits/{SPLIT}.jsonl')
    smap = _tnp_strength_by_tnp(f'{BASE}/splits/{SPLIT}.jsonl')

    # Score with V4 main
    print(f'\n=== V4 main ({os.path.basename(V4_CKPT)}) ===', flush=True)
    m_v4, ck_v4 = _load_model(V4_CKPT, device)
    print(f'  ckpt epoch: {ck_v4["epoch"]}, val AUPRC: {ck_v4["auprc"]:.4f}', flush=True)
    t0 = time.time()
    r_v4 = _score_all(m_v4, dl, device)
    print(f'  scored {len(r_v4["scores"])} tnps in {time.time()-t0:.1f}s', flush=True)
    del m_v4

    # Score with V5.2 Stage A
    print(f'\n=== V5.2 Stage A ({os.path.basename(V5_CKPT)}) ===', flush=True)
    m_v5, ck_v5 = _load_model(V5_CKPT, device)
    print(f'  ckpt epoch: {ck_v5["epoch"]}, val AUPRC: {ck_v5["auprc"]:.4f}', flush=True)
    t0 = time.time()
    r_v5 = _score_all(m_v5, dl, device)
    print(f'  scored {len(r_v5["scores"])} tnps in {time.time()-t0:.1f}s', flush=True)
    del m_v5

    # Group labels
    groups   = np.asarray([gmap[t] for t in r_v4['tnp_ids']])
    strengths = np.asarray([smap.get(t, 'unknown') for t in r_v4['tnp_ids']])

    m4 = _compute_all_metrics(r_v4, groups, strengths)
    m5 = _compute_all_metrics(r_v5, groups, strengths)
    ws4 = _weak_positive_stats(r_v4, strengths)
    ws5 = _weak_positive_stats(r_v5, strengths)

    # ================== Table 1: overall + per-profile ==================
    print()
    print('=' * 96)
    print('  TABLE 1 — held-out test_v4 metrics: V4 main vs V5.2 Stage A')
    print('=' * 96)
    def _fmt(a, b):
        if isinstance(a, float) and isinstance(b, float):
            d = b - a
            return f'{a:>10.4f}   {b:>10.4f}   {d:+7.4f}'
        return f'{a:>10}   {b:>10}'
    print(f'  {"Metric":<40} {"V4 test":>10}   {"V5.2 test":>10}   {"Δ":>7}')
    print('  ' + '-' * 76)
    print(f'  {"AUROC":<40} ' + _fmt(m4["auroc"], m5["auroc"]))
    print(f'  {"AUPRC":<40} ' + _fmt(m4["auprc"], m5["auprc"]))
    print(f'  {"HARD_AUROC (excl level1)":<40} ' + _fmt(m4["hard_auroc"], m5["hard_auroc"]))
    print(f'  {"HARD_AUPRC (excl level1)":<40} ' + _fmt(m4["hard_auprc"], m5["hard_auprc"]))
    print(f'  {"nc_top1 (NC region selection)":<40} ' + _fmt(m4["nc_top1"], m5["nc_top1"]))
    print(f'  {"cand R@1":<40} ' + _fmt(m4["cand"]["recall@1"], m5["cand"]["recall@1"]))
    print(f'  {"cand R@5":<40} ' + _fmt(m4["cand"]["recall@5"], m5["cand"]["recall@5"]))
    print(f'  {"cand R@10":<40} ' + _fmt(m4["cand"]["recall@10"], m5["cand"]["recall@10"]))
    print()
    print(f'  {"Per-profile AUROC:":<40}')
    for k in sorted(m4['strat']):
        prof = k[6:-1]
        a, b = m4['strat'][k], m5['strat'][k]
        print(f'    {prof:<38} ' + _fmt(a, b))
    print()
    print(f'  {"Strength recall @ 0.5:":<40}')
    for lvl in ('strong', 'moderate', 'weak'):
        if lvl in m4['strength_recall'] and lvl in m5['strength_recall']:
            a, na = m4['strength_recall'][lvl]
            b, nb = m5['strength_recall'][lvl]
            print(f'    {lvl:<38} ' + _fmt(a, b) + f'   (n={na})')

    # ================== Table 2: weak-positive alternative metrics ==================
    print()
    print('=' * 96)
    print('  TABLE 2 — weak-positive alternative metrics (weak POS vs all NEG)')
    print('=' * 96)
    if ws4 and ws5:
        print(f'  {"metric":<40} {"V4 test":>10}   {"V5.2 test":>10}   {"Δ":>7}')
        print('  ' + '-' * 76)
        print(f'  {"weak-vs-all AUROC":<40} ' + _fmt(ws4["weak_auroc"], ws5["weak_auroc"]))
        print(f'  {"weak-vs-all AUPRC":<40} ' + _fmt(ws4["weak_auprc"], ws5["weak_auprc"]))
        print(f'  {"n_weak_pos":<40} {ws4["n_weak_pos"]:>10}')
        print(f'  {"n_neg":<40} {ws4["n_neg"]:>10}')

    # ================== Table 3: score-distribution quantiles ==================
    print()
    print('=' * 100)
    print('  TABLE 3 — score-distribution quantiles per group (V4 → V5.2)')
    print('=' * 100)
    def _group_scores(res: dict, positive: bool, group_key: str) -> list[float]:
        s, l = res['scores'], res['labels'].astype(bool)
        tids = res['tnp_ids']
        out = []
        for i, tid in enumerate(tids):
            if positive:
                if l[i] and smap.get(tid, '') == group_key:
                    out.append(float(s[i]))
            else:
                if (not l[i]) and gmap.get(tid, '') == group_key:
                    out.append(float(s[i]))
        return out

    print(f'  {"group":<40} {"n":>4}  {"model":<7}  {"Q10":>6} {"Q25":>6} {"med":>6} {"Q75":>6} {"Q90":>6}')
    print('  ' + '-' * 92)
    pos_groups = ('strong', 'moderate', 'weak')
    neg_groups = ('level1_marginal_matched',
                    'wrong_orientation_consistency',
                    'wrong_length_consistency',
                    'wrong_position_consistency',
                    'wrong_structure_role_consistency',
                    'level3_paired_counterfactual')
    for grp in pos_groups:
        v4v = _group_scores(r_v4, True, grp)
        v5v = _group_scores(r_v5, True, grp)
        if v4v:
            q = _quantile_row(v4v)
            print(f'  POS-{grp:<36} {len(v4v):>4}  {"V4":<7}  {q[0]:>6.3f} {q[1]:>6.3f} {q[2]:>6.3f} {q[3]:>6.3f} {q[4]:>6.3f}')
        if v5v:
            q = _quantile_row(v5v)
            print(f'  {"":<40} {len(v5v):>4}  {"V5.2":<7}  {q[0]:>6.3f} {q[1]:>6.3f} {q[2]:>6.3f} {q[3]:>6.3f} {q[4]:>6.3f}')
    print()
    for grp in neg_groups:
        v4v = _group_scores(r_v4, False, grp)
        v5v = _group_scores(r_v5, False, grp)
        if v4v:
            q = _quantile_row(v4v)
            print(f'  NEG-{grp:<36} {len(v4v):>4}  {"V4":<7}  {q[0]:>6.3f} {q[1]:>6.3f} {q[2]:>6.3f} {q[3]:>6.3f} {q[4]:>6.3f}')
        if v5v:
            q = _quantile_row(v5v)
            print(f'  {"":<40} {len(v5v):>4}  {"V5.2":<7}  {q[0]:>6.3f} {q[1]:>6.3f} {q[2]:>6.3f} {q[3]:>6.3f} {q[4]:>6.3f}')

    # ================== Table 4: signal-source diagnostic ==================
    print()
    print('=' * 96)
    print('  TABLE 4 — signal-source diagnostic: median score shifts (V4 → V5.2)')
    print('=' * 96)
    print(f'  {"group":<40}  {"V4 med":>7}  {"V5.2 med":>7}   {"Δ med":>7}')
    print('  ' + '-' * 76)
    for grp in pos_groups:
        v4v = _group_scores(r_v4, True, grp)
        v5v = _group_scores(r_v5, True, grp)
        if v4v and v5v:
            m4v = float(np.median(v4v))
            m5v = float(np.median(v5v))
            print(f'  POS-{grp:<36}  {m4v:>7.3f}  {m5v:>7.3f}   {m5v-m4v:>+7.3f}')
    for grp in neg_groups:
        v4v = _group_scores(r_v4, False, grp)
        v5v = _group_scores(r_v5, False, grp)
        if v4v and v5v:
            m4v = float(np.median(v4v))
            m5v = float(np.median(v5v))
            print(f'  NEG-{grp:<36}  {m4v:>7.3f}  {m5v:>7.3f}   {m5v-m4v:>+7.3f}')
    print()
    print('Δ interpretation: positives should stay stable; hard negatives (wrong_position,')
    print('wrong_structure_role) should have negative Δ (pushed down). level3 should have')
    print('near-zero Δ (dispersion cannot distinguish it — expected).')

    # Save raw per-tnp scores for offline analysis
    out_json = '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/logs/test_v4_scores_v4_vs_v52.jsonl'
    with open(out_json, 'w') as fh:
        for i, tid in enumerate(r_v4['tnp_ids']):
            fh.write(json.dumps({
                'tnp_id': tid,
                'is_positive': bool(r_v4['labels'][i]),
                'violation_profile': gmap.get(tid, ''),
                'tnp_strength': smap.get(tid, ''),
                'v4_score': float(r_v4['scores'][i]),
                'v5_2_score': float(r_v5['scores'][i]),
            }) + '\n')
    print(f'\nsaved per-tnp scores to {out_json}')


if __name__ == '__main__':
    main()
