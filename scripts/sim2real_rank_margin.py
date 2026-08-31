"""Apples-to-apples rank + margin comparison.

Runs V4 / V4.1 / Durrant-real-background through the SAME candidate pipeline.
For each positive record, identifies the TRUE candidate from labels,
computes:
  - rank of true candidate in per-(orient, L) proposal
  - Recall@K for K in {4, 8, 12, 20}
  - S_true              : #matches at true candidate
  - S_best_decoy         : max #matches over all NON-true candidates (same L range)
  - margin              : S_true - S_best_decoy

Reports side-by-side across the three datasets, plus per-NC-length strata if
requested.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
RC = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
LENGTHS = tuple(range(5, 17))   # match preprocess/candidates.py L in [5..16]


def seq_to_arr(s): return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)
def rc_seq(s): return ''.join(RC.get(c, 'N') for c in s[::-1].upper())


def per_L_orient_M(nc, flank, L):
    """Return (M_fwd, M_rc) match matrices, shape (n_nc, n_fk_pos)."""
    nc_a = seq_to_arr(nc); fk_a = seq_to_arr(flank); fk_rc_a = seq_to_arr(rc_seq(flank))
    if len(nc_a) < L or len(fk_a) < L: return None
    nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)
    a_oh = np.eye(5, dtype=np.int8)[nc_win]
    fw_win = np.lib.stride_tricks.sliding_window_view(fk_a, L)
    b_oh = np.eye(5, dtype=np.int8)[fw_win]
    M_fwd = np.einsum('nlc,mlc->nm', a_oh, b_oh)
    fw_rc = np.lib.stride_tricks.sliding_window_view(fk_rc_a, L)
    b_oh_rc = np.eye(5, dtype=np.int8)[fw_rc]
    M_rc = np.einsum('nlc,mlc->nm', a_oh, b_oh_rc)
    return M_fwd, M_rc


def rank_true_and_max_decoy(nc, flank, true):
    """Vectorized: return (rank, s_true, s_best_decoy_per_orient_L).

    Rank and best_decoy are BOTH computed within the true's own (orient, L)
    match matrix — consistent with the preprocess/candidates.py proposal which
    keeps top-K PER (orient, L). Both quantities are bounded above by L_true.
    """
    nc_a = seq_to_arr(nc); fk_a = seq_to_arr(flank); fk_rc_a = seq_to_arr(rc_seq(flank))
    L_true = true['L']
    if len(nc_a) < L_true or len(fk_a) < L_true: return None

    res = per_L_orient_M(nc, flank, L_true)
    if res is None: return None
    M_fwd, M_rc = res
    M_target = M_fwd if true['orient'] == 'fwd' else M_rc
    if true['orient'] == 'fwd':
        j = true['flank_start']
    else:
        j = len(fk_a) - true['flank_start'] - L_true
    i = true['nc_start']
    if not (0 <= i < M_target.shape[0] and 0 <= j < M_target.shape[1]): return None
    s_true = int(M_target[i, j])
    rank = int((M_target > s_true).sum()) + 1

    # best decoy: max matches in same (orient, L=L_true) EXCLUDING the true cell
    M_ex = M_target.copy()
    M_ex[i, j] = -1
    best_decoy = int(M_ex.max())
    return rank, s_true, best_decoy


def find_true_candidate(rec):
    """From labels, extract (orient, L, nc_start, flank_start) of the true positive.
    Returns None if labels don't identify a specific candidate (e.g. Durrant)."""
    lab = rec['labels']
    L = lab.get('guide_length')
    orient_lab = lab.get('match_orientation')
    if orient_lab == 'reverse_complement':
        orient = 'rc'
    elif orient_lab == 'forward':
        orient = 'fwd'
    else:
        return None
    tp = lab.get('target_position_in_flank')
    gs = lab.get('guide_span_in_active_noncoding')
    if not (L and tp and gs): return None
    return {'orient': orient, 'L': int(L),
            'nc_start': int(gs[0]), 'flank_start': int(tp[0])}


def durrant_true_candidate(rec):
    """For Durrant real-bg records: L=11, orient=fwd, nc_start = LTG loop position, flank_start=0."""
    # Find LTG position by max-match to spec — we baked bRNA loop at nc_start=49 for all Durrant bRNAs
    nc = rec['inputs']['noncoding_regions'][rec['labels']['active_noncoding_index']]
    # Take target 11bp = flank[0:11]
    flank = rec['inputs']['flank']
    target_11 = flank[:11]
    L = 11
    if len(nc) < L: return None
    # Slide L=11 over NC, find max-match position vs target_11 in fwd orient
    best_pos, best_m = -1, -1
    tgt_arr = np.asarray([BASE_MAP.get(c, 4) for c in target_11.upper()], dtype=np.int8)
    for i in range(len(nc) - L + 1):
        win = nc[i:i + L].upper()
        win_arr = np.asarray([BASE_MAP.get(c, 4) for c in win], dtype=np.int8)
        m = int(((win_arr == tgt_arr) & (win_arr < 4)).sum())
        if m > best_m:
            best_m = m; best_pos = i
    return {'orient': 'fwd', 'L': L, 'nc_start': best_pos, 'flank_start': 0}


def audit_dataset(records, label, is_durrant=False, max_records=500):
    """Compute per-site rank + margin stats."""
    if len(records) > max_records:
        rng = random.Random(42)
        records = rng.sample(records, max_records)
    print(f'  [{label}] processing {len(records)} records', flush=True)

    ranks_per_orient = []
    s_true, s_best_decoy = [], []
    nc_lens = []
    n_true_found = 0
    for rec in records:
        if not rec['labels'].get('is_positive'): continue
        acn = rec['labels'].get('active_noncoding_index', 0)
        ncs = rec['inputs']['noncoding_regions']
        if acn >= len(ncs): continue
        nc = ncs[acn]; flank = rec['inputs']['flank']
        nc_lens.append(len(nc))
        if is_durrant:
            true = durrant_true_candidate(rec)
        else:
            true = find_true_candidate(rec)
        if true is None: continue
        res = rank_true_and_max_decoy(nc, flank, true)
        if res is None: continue
        rank, s_true_here, s_best_dec = res
        n_true_found += 1
        ranks_per_orient.append(rank)
        s_true.append(s_true_here)
        s_best_decoy.append(s_best_dec)

    if not ranks_per_orient:
        print(f'  [{label}] no valid true candidates found')
        return None
    r_arr = np.asarray(ranks_per_orient)
    st = np.asarray(s_true)
    sd = np.asarray(s_best_decoy)
    margin = st - sd
    return {
        'label': label,
        'n': int(n_true_found),
        'nc_len_median': float(np.median(nc_lens)),
        'rank_median': float(np.median(r_arr)),
        'rank_q75': float(np.quantile(r_arr, .75)),
        'rank_q90': float(np.quantile(r_arr, .90)),
        'recall_at_4': float(np.mean(r_arr <= 4)),
        'recall_at_8': float(np.mean(r_arr <= 8)),
        'recall_at_12': float(np.mean(r_arr <= 12)),
        'recall_at_20': float(np.mean(r_arr <= 20)),
        's_true_median': float(np.median(st)),
        's_best_decoy_median': float(np.median(sd)),
        'margin_median': float(np.median(margin)),
        'margin_q25': float(np.quantile(margin, .25)),
        'margin_frac_positive': float(np.mean(margin > 0)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v4', default='/global/scratch/users/kh36969/tmp/v41_cal/assembled_baseline.jsonl')
    ap.add_argument('--v41', default='/global/scratch/users/kh36969/tmp/v41_cal/assembled_v41.jsonl')
    ap.add_argument('--durrant-realbg', default='/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_cognate_realbg.jsonl')
    ap.add_argument('--durrant-realbg-ncpad', default='/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_cognate_realbg_ncpad240.jsonl')
    ap.add_argument('--max-records', type=int, default=500)
    args = ap.parse_args()

    def _load(p):
        with open(p) as f: return [json.loads(l) for l in f]

    print(f'[load] V4        : {args.v4}')
    v4 = _load(args.v4)
    print(f'[load] V4.1      : {args.v41}')
    v41 = _load(args.v41)
    print(f'[load] Durrant native NC (177nt) + real bg: {args.durrant_realbg}')
    dur = _load(args.durrant_realbg)
    print(f'[load] Durrant NC-padded to 240nt + real bg: {args.durrant_realbg_ncpad}')
    dur_pad = _load(args.durrant_realbg_ncpad)

    print(f'\n{"="*100}')
    print(f'  APPLES-TO-APPLES rank + margin comparison')
    print(f'{"="*100}\n')

    results = []
    for label, recs, is_dur in [
        ('V4 baseline (uniform)',            v4,      False),
        ('V4.1 durrant_calibrated',           v41,     False),
        ('Durrant real-bg, NC=177nt (biology)', dur,   True),
        ('Durrant real-bg, NC=240nt (padded)',  dur_pad, True),
    ]:
        r = audit_dataset(recs, label, is_durrant=is_dur, max_records=args.max_records)
        if r: results.append(r)

    # Report table
    print(f'\n  {"dataset":<40} {"n":>4} {"NC_len":>7} {"rank_med":>9} '
          f'{"rank_Q75":>9} {"rank_Q90":>9}')
    print('  ' + '-' * 95)
    for r in results:
        print(f'  {r["label"]:<40} {r["n"]:>4} {int(r["nc_len_median"]):>7} '
              f'{r["rank_median"]:>9.1f} {r["rank_q75"]:>9.1f} {r["rank_q90"]:>9.1f}')

    print(f'\n  {"dataset":<40} {"R@4":>6} {"R@8":>6} {"R@12":>6} {"R@20":>6}')
    print('  ' + '-' * 75)
    for r in results:
        print(f'  {r["label"]:<40} {r["recall_at_4"]:>6.3f} '
              f'{r["recall_at_8"]:>6.3f} {r["recall_at_12"]:>6.3f} '
              f'{r["recall_at_20"]:>6.3f}')

    print(f'\n  {"dataset":<40} {"S_true":>7} {"S_decoy":>8} {"margin":>7} '
          f'{"marQ25":>7} {"P(m>0)":>7}')
    print('  ' + '-' * 90)
    for r in results:
        print(f'  {r["label"]:<40} {r["s_true_median"]:>7.1f} '
              f'{r["s_best_decoy_median"]:>8.1f} {r["margin_median"]:>+7.1f} '
              f'{r["margin_q25"]:>+7.1f} {r["margin_frac_positive"]:>7.3f}')


if __name__ == '__main__':
    main()
