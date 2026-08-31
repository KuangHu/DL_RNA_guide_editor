"""CLI entry for the V4.2 counterfactual negative generator.

Reads a YAML config, generates one JSONL per enabled profile with full
invariant checks + provenance metadata, then optionally runs the auto-audit
hook and fails on layout leak.

Usage:
    python -m scripts.generate_negatives_v42 --config configs/negative_generator_v42.yaml

Runtime: ~1-2 min per 250K sites per profile with single-thread numpy.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

# Force single-thread numpy — small einsum ops parallelize badly.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import yaml

from preprocess.negative_generator_v42 import (
    build_profiles, InvariantViolation, NEG_GEN_VERSION,
)


def load_positives_by_bag(path, max_bags=None):
    by_bag = defaultdict(list)
    n = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by_bag[r['transposase_id']].append(r)
            n += 1
    print(f'  {n} records across {len(by_bag)} bags', flush=True)
    if max_bags is not None:
        keys = list(by_bag.keys())[:max_bags]
        by_bag = {k: by_bag[k] for k in keys}
        print(f'  restricted to first {max_bags} bags', flush=True)
    return by_bag


def emit_profile(profile, pos_by_bag, out_path, rng):
    t0 = time.time()
    n_emit = 0; n_invariant_fail = 0; n_bags_ok = 0
    with out_path.open('w') as fh:
        for bag_i, (tnp, bag) in enumerate(pos_by_bag.items()):
            try:
                recs = profile.transform_bag(bag, rng)
            except InvariantViolation as e:
                n_invariant_fail += 1
                continue
            for r in recs:
                fh.write(json.dumps(r) + '\n')
            n_emit += len(recs); n_bags_ok += 1
            if (bag_i + 1) % 1000 == 0:
                print(f'    {bag_i+1}/{len(pos_by_bag)} bags — emitted {n_emit}',
                      flush=True)
    dt = time.time() - t0
    print(f'    done: {n_emit} sites, {n_bags_ok} bags ok, '
          f'{n_invariant_fail} invariant fails, {dt:.1f}s', flush=True)
    return {'n_emit': n_emit, 'n_bags_ok': n_bags_ok,
            'n_invariant_fail': n_invariant_fail, 'seconds': dt}


def run_audit(cfg, out_paths, pos_path):
    """Auto-audit: for each generated profile, verify layout AUROC < threshold.
    Emits a fail if any nuisance-matched probe leaks."""
    import numpy as np
    sys.path.insert(0, str(Path(__file__).parent))
    from v42_shortcut_probes import (
        feat_layout, feat_flank_marginal, feat_nc_marginal, feat_flank_plus_nc,
    )
    from v42_negatives_audit import feat_candidate_summary, build_features, train_auroc

    def _load_sample(path, n):
        rs = []
        with open(path) as f:
            for line in f:
                rs.append(json.loads(line))
                if len(rs) >= n * 2: break
        random.Random(42).shuffle(rs)
        return rs[:n]

    n_per = cfg['audit']['n_per_class_sample']
    threshold = cfg['audit']['layout_leak_threshold']
    layout_probes = cfg['audit']['layout_probes']

    print(f'\n[audit] loading {n_per} positives + {n_per} per profile ...',
          flush=True)
    pos = _load_sample(pos_path, n_per)
    pos_feats = build_features(pos)

    print(f'[audit] AUROC per profile (POS vs NEG, layout probes must be < '
          f'{threshold}):', flush=True)
    header_probes = ['layout', 'flank_marginal', 'nc_marginal',
                     'combined_marginal', 'candidate_summary']
    print(f'  {"profile":<32} ', end='')
    for p in header_probes: print(f'{p[:16]:>16}', end='')
    print(f'  {"verdict":>10}')

    any_fail = False
    for prof_key, path in out_paths.items():
        neg_recs = _load_sample(str(path), n_per)
        neg_feats = build_features(neg_recs)
        row = []
        for p in header_probes:
            au_lr, au_rf = train_auroc(pos_feats[p], neg_feats[p])
            row.append(max(au_lr, au_rf))
        # Check the layout probes only
        fails = [p for p, a in zip(header_probes, row)
                 if p in layout_probes and a > threshold]
        verdict = 'PASS' if not fails else 'FAIL'
        if fails: any_fail = True
        print(f'  {prof_key:<32} ', end='')
        for a in row: print(f'{a:>16.4f}', end='')
        print(f'  {verdict:>10}')
    return not any_fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True, type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    if cfg.get('version') != NEG_GEN_VERSION:
        print(f'[warn] config version {cfg.get("version")} != code '
              f'{NEG_GEN_VERSION}', flush=True)

    out_dir = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[{time.strftime("%H:%M:%S")}] loading positives from '
          f'{cfg["positive_source"]}', flush=True)
    pos_by_bag = load_positives_by_bag(cfg['positive_source'],
                                         max_bags=cfg.get('max_bags'))

    rng = random.Random(cfg.get('seed', 0))
    profiles = build_profiles(cfg)
    print(f'\n[profiles] enabled: {list(profiles.keys())}', flush=True)

    out_paths = {}; stats = {}
    for prof_key, profile in profiles.items():
        out_path = out_dir / f'negatives_v42_{prof_key}.jsonl'
        print(f'\n[{time.strftime("%H:%M:%S")}] emitting {profile.name} '
              f'-> {out_path.name}', flush=True)
        stats[prof_key] = emit_profile(profile, pos_by_bag, out_path, rng)
        out_paths[prof_key] = out_path

    # Emit a manifest summarising the run
    manifest = {
        'version': NEG_GEN_VERSION,
        'config': cfg,
        'output_paths': {k: str(v) for k, v in out_paths.items()},
        'stats_per_profile': stats,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    manifest_path = out_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'\n[manifest] {manifest_path}', flush=True)

    if cfg.get('audit', {}).get('auto_run', False):
        ok = run_audit(cfg, out_paths, cfg['positive_source'])
        if not ok and cfg['audit'].get('fail_on_layout_leak', True):
            print(f'\n[audit] FAIL — layout leak detected. See table above.',
                  flush=True)
            sys.exit(1)
        else:
            print(f'\n[audit] all layout probes pass threshold.', flush=True)


if __name__ == '__main__':
    main()
