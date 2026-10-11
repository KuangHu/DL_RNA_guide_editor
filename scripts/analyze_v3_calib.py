"""Fit a bag threshold, then evaluate it on disjoint held-out bags.

Reports empirical null pass rates, not FDR or biological recall. Old
support-first calibration files and silently incomplete nulls are rejected.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.reporting_statistic import STATISTIC_VERSION

CONDITIONS = ('real', 'null1_dinuc', 'null2_4mer', 'null3_unpaired', 'null4_rc_shuf')


def read_details(path):
    by_bag = defaultdict(dict)
    fingerprints = set()
    excluded = []
    with path.open(encoding='utf-8') as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r.get('statistic_version') != STATISTIC_VERSION:
                raise ValueError('old or incompatible calibration statistic; rerun null calibration')
            if r.get('err'):
                raise ValueError(f"calibration error: {r['bag_id']} {r['cond']}: {r['err']}")
            bid, cond = r['bag_id'], r['cond']
            if cond not in CONDITIONS or cond in by_bag[bid]:
                raise ValueError(f'duplicate/unknown calibration condition: {bid} {cond}')
            fingerprints.add(r['config_fingerprint'])
            r['max_mean_m'] = float(r['max_mean_m'])
            if not math.isfinite(r['max_mean_m']) or not 0 <= r['max_mean_m'] <= 1:
                raise ValueError('invalid calibration score')
            by_bag[bid][cond] = r
    if len(fingerprints) != 1:
        raise ValueError('calibration contains different configurations or no data')
    eligible = {}
    for bid, conditions in by_bag.items():
        if set(conditions) != set(CONDITIONS):
            raise ValueError(f'missing null condition(s): {bid}')
        if all(r.get('eligible') == 'True' and r.get('status') == 'ready' for r in conditions.values()):
            eligible[bid] = conditions
        else:
            excluded.append(bid)
    return eligible, fingerprints.pop(), excluded


def pass_rates(rows, ids, threshold):
    if not ids:
        raise ValueError('empty calibration partition')
    return {c: sum(rows[bid][c]['max_mean_m'] >= threshold for bid in ids) / len(ids) for c in CONDITIONS}


def fit_threshold(rows, ids, target):
    observed = {rows[bid][c]['max_mean_m'] for bid in ids for c in CONDITIONS if c != 'real'}
    candidates = sorted({0.5, 1.0} | {math.nextafter(v, math.inf) for v in observed if .5 <= v < 1})
    for t in candidates:
        rates = pass_rates(rows, ids, t)
        if max(rates[c] for c in CONDITIONS if c != 'real') <= target:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--details', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--target-null-pass', type=float, default=.01)
    ap.add_argument('--holdout-fraction', type=float, default=.3)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--min-partition-bags', type=int, default=20)
    args = ap.parse_args()
    if not 0 <= args.target_null_pass < 1 or not 0 < args.holdout_fraction < 1 or args.min_partition_bags < 1:
        ap.error('invalid pass rate, split fraction or minimum bag count')
    rows, fingerprint, excluded = read_details(args.details)
    ids = sorted(rows)
    random.Random(args.seed).shuffle(ids)
    nhold = int(len(ids) * args.holdout_fraction)
    holdout, train = ids[:nhold], ids[nhold:]
    if min(len(train), len(holdout)) < args.min_partition_bags:
        raise ValueError(f'insufficient eligible bags for train/holdout: {len(train)}/{len(holdout)}; excluded={len(excluded)}')
    threshold = fit_threshold(rows, train, args.target_null_pass)
    train_rates = pass_rates(rows, train, threshold) if threshold is not None else None
    held_rates = pass_rates(rows, holdout, threshold) if threshold is not None else None
    validated = held_rates is not None and max(held_rates[c] for c in CONDITIONS if c != 'real') <= args.target_null_pass
    value = dict(statistic_version=STATISTIC_VERSION, config_fingerprint=fingerprint,
                 status='empirically_validated' if validated else 'not_validated', threshold=threshold,
                 target_null_pass=args.target_null_pass, train_bag_ids=train, holdout_bag_ids=holdout,
                 excluded_bag_ids=excluded, train_pass_rates=train_rates, holdout_pass_rates=held_rates,
                 interpretation='Empirical pass rates on synthetic nulls; neither FDR nor biological recall.',
                 source_details=str(args.details), seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, indent=2), encoding='utf-8')
    print(f"{value['status']}: threshold={threshold}; train={len(train)}, holdout={len(holdout)}, excluded={len(excluded)}")
    print('Held-out empirical pass rates:', held_rates)
    print('Saved', args.out)


if __name__ == '__main__':
    main()
