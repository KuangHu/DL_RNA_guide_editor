"""Per-L stratified rank/margin audit.

For V4.1 synthetic: bucket positive sites by guide length L, compute per-L
identity/rank/margin distributions. Tests whether the Q90 tail (~360) is
dominated by long-L (14-16) weak positives — if yes, a length-aware
difficulty cap or rank-aware rejection sampler is the right fix; if no, the
tail comes from a different source.

Also runs the same stratification on the Durrant real-bg reference (which
is all L=11 by construction, so serves as a single-row calibration target).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

from sim2real_rank_margin import (
    seq_to_arr, rc_seq, per_L_orient_M,
    find_true_candidate, durrant_true_candidate,
    rank_true_and_max_decoy, LENGTHS,
)


def audit_by_L(records, label, is_durrant=False, max_records=500):
    if len(records) > max_records:
        rng = random.Random(42)
        records = rng.sample(records, max_records)
    print(f'  [{label}] processing {len(records)} records', flush=True)
    by_L = defaultdict(lambda: {'rank': [], 's_true': [], 's_decoy': [],
                                 'identity': [], 'n': 0})
    for rec in records:
        if not rec['labels'].get('is_positive'): continue
        acn = rec['labels'].get('active_noncoding_index', 0)
        ncs = rec['inputs']['noncoding_regions']
        if acn >= len(ncs): continue
        nc = ncs[acn]; flank = rec['inputs']['flank']
        if is_durrant:
            true = durrant_true_candidate(rec)
        else:
            true = find_true_candidate(rec)
        if true is None: continue
        L = true['L']
        res = rank_true_and_max_decoy(nc, flank, true)
        if res is None: continue
        rank, s_true, s_dec = res
        by_L[L]['rank'].append(rank)
        by_L[L]['s_true'].append(s_true)
        by_L[L]['s_decoy'].append(s_dec)
        by_L[L]['identity'].append(s_true / L)
        by_L[L]['n'] += 1
    return by_L


def print_L_table(label, by_L):
    print(f'\n  {label}')
    print(f'  {"L":>3} {"n":>4}  {"id_med":>7} {"rank_med":>9} {"rank_Q75":>9} '
          f'{"rank_Q90":>9}  {"R@4":>5} {"R@8":>5} {"R@20":>5}  '
          f'{"S_t":>4} {"S_d":>4} {"mar_med":>7} {"P(m>0)":>7}')
    print('  ' + '-' * 105)
    for L in sorted(by_L.keys()):
        d = by_L[L]
        if d['n'] < 5:
            print(f'  {L:>3} {d["n"]:>4}  (n<5, skipping)')
            continue
        r = np.asarray(d['rank']); st = np.asarray(d['s_true']); sd = np.asarray(d['s_decoy'])
        idt = np.asarray(d['identity'])
        margin = st - sd
        print(f'  {L:>3} {d["n"]:>4}  '
              f'{np.median(idt):>7.3f} {np.median(r):>9.1f} {np.quantile(r,.75):>9.1f} '
              f'{np.quantile(r,.90):>9.1f}  '
              f'{np.mean(r<=4):>5.2f} {np.mean(r<=8):>5.2f} {np.mean(r<=20):>5.2f}  '
              f'{np.median(st):>4.1f} {np.median(sd):>4.1f} '
              f'{np.median(margin):>+7.1f} {np.mean(margin>0):>7.3f}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v4', default='/global/scratch/users/kh36969/tmp/v41_cal/assembled_baseline.jsonl')
    ap.add_argument('--v41', default='/global/scratch/users/kh36969/tmp/v41_cal/assembled_v41.jsonl')
    ap.add_argument('--durrant-realbg', default='/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/durrant_cognate_realbg.jsonl')
    ap.add_argument('--max-records', type=int, default=1500)
    args = ap.parse_args()

    def _load(p):
        with open(p) as f: return [json.loads(l) for l in f]

    v4 = _load(args.v4)
    v41 = _load(args.v41)
    dur = _load(args.durrant_realbg)

    print(f'\n{"="*105}')
    print(f'  PER-L STRATIFIED RANK/MARGIN AUDIT')
    print(f'{"="*105}')

    by_L_v4 = audit_by_L(v4, 'V4 baseline (uniform)', False, args.max_records)
    print_L_table('V4 baseline (uniform)', by_L_v4)

    by_L_v41 = audit_by_L(v41, 'V4.1 durrant_calibrated', False, args.max_records)
    print_L_table('V4.1 durrant_calibrated', by_L_v41)

    by_L_dur = audit_by_L(dur, 'Durrant real-bg NC=177', True, args.max_records)
    print_L_table('Durrant real-bg NC=177', by_L_dur)

    # Also — what fraction of V4.1 pathological cases (rank > 100) come from each L?
    print(f'\n  V4.1 pathological analysis (sites with rank > 100):')
    total = 0; by_L_bad = defaultdict(int); by_L_total = defaultdict(int)
    for L, d in by_L_v41.items():
        r = np.asarray(d['rank'])
        by_L_total[L] = len(r)
        by_L_bad[L] = int((r > 100).sum())
        total += by_L_bad[L]
    print(f'  {"L":>3} {"n_bad(>100)":>12} {"n_total":>9} {"%_bad":>7}')
    print('  ' + '-' * 45)
    for L in sorted(by_L_total.keys()):
        if by_L_total[L] < 5: continue
        pct = 100 * by_L_bad[L] / max(1, by_L_total[L])
        print(f'  {L:>3} {by_L_bad[L]:>12} {by_L_total[L]:>9} {pct:>7.1f}%')
    print(f'\n  Total sites with rank>100 in V4.1: {total}')


if __name__ == '__main__':
    main()
