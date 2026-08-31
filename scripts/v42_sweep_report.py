"""Score V4.2 sweep outputs on BOTH pairing-strength AND difficulty gates.

For each candidate config, report:
  identity_median, frac_perfect, frac_ge_0.9, frac_le_0.75
  rank_median, Q75, Q90, R@4, R@8, R@20, P(rank>100)
  L distribution shift
  mean retries, hard_tail_frac
Then a joint pass/fail table.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np
from sim2real_rank_margin import find_true_candidate, rank_true_and_max_decoy


def audit(path, max_records=2000, seed=42):
    with open(path) as f:
        recs = [json.loads(l) for l in f]
    if len(recs) > max_records:
        rng = random.Random(seed)
        recs = rng.sample(recs, max_records)
    identity = []; Ls = []; ranks = []
    retries = []; hard_tail = []
    for rec in recs:
        L = rec['labels']['guide_length']
        n_mm = rec['labels']['n_mismatches']
        identity.append((L - n_mm) / L)
        Ls.append(L)
        # If rankaware meta present, use rank_final
        ra = rec['generator_metadata'].get('rankaware')
        if ra:
            ranks.append(ra['rank_final'])
            retries.append(ra['n_retries'])
            hard_tail.append(int(ra['accepted_hard_tail']))
            continue
        # Else compute rank directly
        acn = rec['labels'].get('active_noncoding_index', 0)
        nc = rec['inputs']['noncoding_regions'][acn]
        flank = rec['inputs']['flank']
        true = find_true_candidate(rec)
        if true is None: continue
        res = rank_true_and_max_decoy(nc, flank, true)
        if res is None: continue
        rank, _, _ = res
        ranks.append(rank)
    identity = np.asarray(identity); ranks = np.asarray(ranks)
    Ls = np.asarray(Ls)
    return {
        'n': len(identity),
        'identity_med': float(np.median(identity)),
        'frac_perfect': float(np.mean(identity == 1.0)),
        'frac_ge_0.9': float(np.mean(identity >= 0.9)),
        'frac_le_0.75': float(np.mean(identity <= 0.75)),
        'rank_med': float(np.median(ranks)),
        'rank_q75': float(np.quantile(ranks, .75)),
        'rank_q90': float(np.quantile(ranks, .90)),
        'r_at_4': float(np.mean(ranks <= 4)),
        'r_at_8': float(np.mean(ranks <= 8)),
        'r_at_20': float(np.mean(ranks <= 20)),
        'p_rank_gt_100': float(np.mean(ranks > 100)),
        'L_dist': dict(Counter(Ls.tolist())),
        'mean_retries': float(np.mean(retries)) if retries else 0.0,
        'hard_tail_frac': float(np.mean(hard_tail)) if hard_tail else 0.0,
    }


PAIR_GATES = {
    'identity_med': (0.65, 0.80),
    'frac_perfect': (0.0, 0.05),
    'frac_ge_0.9':  (0.0, 0.15),
    'frac_le_0.75': (0.40, 1.0),
}
DIFF_GATES = {
    'rank_med': (2, 5),
    'r_at_4':   (0.50, 0.65),
    'r_at_20':  (0.75, 0.90),
    'p_rank_gt_100': (0.05, 0.10),
    'rank_q90': (30, 80),
}


def check_gates(r, gates):
    fails = []
    for k, (lo, hi) in gates.items():
        v = r.get(k)
        if v is None or not (lo <= v <= hi):
            fails.append(k)
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sweep-dir', default='/global/scratch/users/kh36969/tmp/v41_cal')
    ap.add_argument('--max-records', type=int, default=2000)
    args = ap.parse_args()

    configs = [
        ('V4 baseline (uniform)',       'assembled_baseline.jsonl'),
        ('V4.1 durrant_calibrated',      'assembled_v41.jsonl'),
        ('V4.2 rc=100 tp=0.10',           'assembled_v42_rc100_tp0.10.jsonl'),
        ('V4.2 rc=100 tp=0.20',           'assembled_v42_rc100_tp0.20.jsonl'),
        ('V4.2 rc=100 tp=0.30',           'assembled_v42_rc100_tp0.30.jsonl'),
        ('V4.2 rc=200 tp=0.10',           'assembled_v42_rc200_tp0.10.jsonl'),
        ('V4.2 rc=200 tp=0.20',           'assembled_v42_rc200_tp0.20.jsonl'),
    ]
    results = []
    for label, fn in configs:
        path = Path(args.sweep_dir) / fn
        if not path.exists():
            print(f'[skip] {path} not found'); continue
        r = audit(path, max_records=args.max_records)
        results.append((label, r))

    # Report
    print(f'\n{"="*135}')
    print(f'  V4.2 SWEEP — pairing strength + rank difficulty')
    print(f'{"="*135}')
    print(f'\n  {"config":<32} {"n":>4}   '
          f'{"id_med":>7} {"perf%":>6} {"≥.9%":>5} {"≤.75%":>6}   '
          f'{"rk_med":>7} {"rk_Q90":>7} {"R@4":>6} {"R@20":>6} {"rk>100":>6} '
          f'{"retry":>6} {"htail":>6}')
    for label, r in results:
        print(f'  {label:<32} {r["n"]:>4}   '
              f'{r["identity_med"]:>7.3f} {100*r["frac_perfect"]:>6.1f} '
              f'{100*r["frac_ge_0.9"]:>5.1f} {100*r["frac_le_0.75"]:>6.1f}   '
              f'{r["rank_med"]:>7.1f} {r["rank_q90"]:>7.0f} '
              f'{r["r_at_4"]:>6.3f} {r["r_at_20"]:>6.3f} '
              f'{100*r["p_rank_gt_100"]:>6.1f} {r["mean_retries"]:>6.2f} '
              f'{100*r["hard_tail_frac"]:>6.1f}')

    print(f'\n  Joint pass/fail (P = pass pairing gates, D = pass difficulty gates)')
    print(f'  {"config":<32} {"P?":>4} {"P fails":<40} {"D?":>4} {"D fails":<40}')
    for label, r in results:
        p_fails = check_gates(r, PAIR_GATES)
        d_fails = check_gates(r, DIFF_GATES)
        print(f'  {label:<32} {"OK" if not p_fails else "FAIL":>4} '
              f'{",".join(p_fails):<40} {"OK" if not d_fails else "FAIL":>4} '
              f'{",".join(d_fails):<40}')


if __name__ == '__main__':
    main()
