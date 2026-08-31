"""Extract per-site summary features for the summary MLP baseline (48A).

Reads a JSONL, computes per-record:
  best_matches, best_identity, best_L, best_flank_start, best_orient_fwd,
  junction_distance, n_high_match_candidates (>=8/L), best_second_best_gap,
  active_nc_len, n_ncs

Emits a compact numpy file plus the corresponding site_id list.

Feature extraction uses an oracle sweep over L ∈ {8,10,12,14,16}, both
orientations. Runtime ~15ms/record single-thread.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')

import numpy as np

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
def _rc(s): return s.translate(_COMP)[::-1]
def _s2a(s): return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)


LENGTHS = (8, 10, 12, 14, 16)


def per_site_features(nc, flank):
    """Return dict of summary features."""
    if not nc or not flank or len(nc) < 8 or len(flank) < 8:
        return None
    nc_a = _s2a(nc)
    fk_a = _s2a(flank)
    fk_rc_a = _s2a(_rc(flank))
    best_matches, best_L, best_orient, best_fs, best_ns = -1, -1, 'fwd', 0, 0
    all_matches = []
    for L in LENGTHS:
        if len(nc_a) < L or len(fk_a) < L: continue
        nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)
        a_oh = np.eye(5, dtype=np.int8)[nc_win]
        for orient, fw in (('fwd', fk_a), ('rc', fk_rc_a)):
            fw_win = np.lib.stride_tricks.sliding_window_view(fw, L)
            b_oh = np.eye(5, dtype=np.int8)[fw_win]
            M = np.einsum('nlc,mlc->nm', a_oh, b_oh)
            m_max = int(M.max())
            all_matches.append(m_max)
            if m_max > best_matches:
                best_matches = m_max
                best_L = L
                best_orient = orient
                idx = np.unravel_index(np.argmax(M), M.shape)
                if orient == 'fwd':
                    best_fs = int(idx[1])
                else:
                    best_fs = len(fk_a) - int(idx[1]) - L
                best_ns = int(idx[0])
    if best_matches < 0: return None
    best_identity = best_matches / max(1, best_L)
    junction_dist = min(best_fs, max(0, 120 - best_fs - best_L))
    high_match = sum(1 for m in all_matches if m >= 8)
    all_matches_arr = np.asarray(all_matches, dtype=np.float32)
    return {
        'best_matches':  best_matches,
        'best_identity': best_identity,
        'best_L':        best_L,
        'best_flank_start': best_fs,
        'best_orient_fwd': 1 if best_orient == 'fwd' else 0,
        'junction_dist': junction_dist,
        'n_high_match':  high_match,
        'match_dispersion': float(all_matches_arr.std()) if len(all_matches_arr) > 1 else 0.0,
        'max_over_median': best_matches - float(np.median(all_matches_arr)),
    }


FEATURE_NAMES = [
    'best_matches', 'best_identity', 'best_L', 'best_flank_start',
    'best_orient_fwd', 'junction_dist', 'n_high_match', 'match_dispersion',
    'max_over_median', 'active_nc_len', 'n_ncs',
]
N_FEATS = len(FEATURE_NAMES)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    site_ids = []
    tnp_ids = []
    labels = []
    feats = []
    n_read = 0; n_ok = 0
    with open(args.jsonl) as f:
        for line in f:
            if args.limit and n_read >= args.limit: break
            r = json.loads(line)
            n_read += 1
            ncs = r['inputs']['noncoding_regions']
            active = r['labels'].get('active_noncoding_index', 0)
            nc = ncs[active] if active < len(ncs) else ''
            flank = r['inputs']['flank']
            fs = per_site_features(nc, flank)
            if fs is None:
                continue
            row = np.asarray([fs[k] for k in FEATURE_NAMES[:-2]], dtype=np.float32)
            active_len = len(ncs[active]) if active < len(ncs) else 0
            row = np.concatenate([row, [active_len, len(ncs)]]).astype(np.float32)
            feats.append(row)
            site_ids.append(r['site_id'])
            tnp_ids.append(r['transposase_id'])
            labels.append(1 if r['labels']['is_positive'] else 0)
            n_ok += 1
            if n_ok % 10000 == 0:
                print(f'  {n_ok} records processed', flush=True)
    X = np.stack(feats, axis=0)
    y = np.asarray(labels, dtype=np.int8)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f'{args.out_prefix}.npz',
                          X=X, y=y,
                          site_ids=np.asarray(site_ids),
                          tnp_ids=np.asarray(tnp_ids),
                          feature_names=np.asarray(FEATURE_NAMES))
    print(f'[out] {args.out_prefix}.npz  shape={X.shape}  '
          f'n_ok={n_ok}/{n_read}', flush=True)


if __name__ == '__main__':
    main()
