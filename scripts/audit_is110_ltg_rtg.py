"""Test whether real IS110 bags follow the LtG/RtG rule.

Biology (from README):
  IS110 uses a bridge RNA (bRNA) with TWO short guides.
    LTG = Left Target Guide  -> pairs with the LEFT target flank
    RTG = Right Target Guide -> pairs with the RIGHT target flank
  The paired target is ~9 bp on each side of the junction.

Flank convention (from build_real_is110.py):
    upstream flank    = last  FLANK_LEN=120 bp before insertion_start
                        -> junction sits at flank[119] (right end)
    downstream flank  = first FLANK_LEN=120 bp after  insertion_end
                        -> junction sits at flank[0]   (left end)
  So for LtG/RtG to be visible:
    - upstream sites   : best short guide should hit flank[100..119]
                          (junction distance = 120 - flank_start - L)
    - downstream sites : best short guide should hit flank[0..20]
                          (junction distance = flank_start)

Test:
  For each bag, split sites by flank_side. For each subgroup run an oracle
  ungapped alignment sweep over L in {7,8,9,10,11,12} (short = closer to the
  ~9 bp IS110 target). Report:

    Test A -- junction concentration
      distribution of "junction_distance" for the best alignment per site;
      the % of sites whose best alignment lies within 15 bp of the junction.

    Test B -- side-specific NC position clustering
      median nc_start for upstream vs downstream subgroups within a bag;
      if LtG/RtG is real, they should DIFFER (the two guides live at
      different residues of the bRNA), and each should have a low MAD.

    Test C -- shuffle control
      Same subgroup, but re-pair each site's NC with a random other-site
      flank FROM THE SAME SIDE within the bag. If the junction-concentration
      is a real pairing signal it drops under shuffle; if it is a marginal-
      distribution artifact (e.g. the first 15 bp of downstream flanks are
      systematically special because of TSD or IR-derived motifs) it persists.

Compare IS110 vs IS30/IS903/IS10-R controls.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REAL_JSONL = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference/real_all.jsonl'
FLANK_LEN = 120
JUNCTION_WINDOW = 15  # sites within this bp of junction are "junction-hit"
LENGTHS = (7, 8, 9, 10, 11, 12)  # short guides -- IS110 target is ~9 bp
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
RC = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}


def seq_to_arr(s: str) -> np.ndarray:
    return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)


def rc_seq(s: str) -> str:
    return ''.join(RC.get(c, 'N') for c in s[::-1].upper())


def best_short_alignment(nc: str, flank: str) -> dict:
    """Best ungapped alignment; tie-break: highest matches, then highest identity, then smallest L."""
    nc_a = seq_to_arr(nc)
    flank_a = seq_to_arr(flank)
    flank_rc_a = seq_to_arr(rc_seq(flank))
    best = {'matches': -1}
    for L in LENGTHS:
        if len(nc_a) < L or len(flank_a) < L:
            continue
        nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)
        fk_win = np.lib.stride_tricks.sliding_window_view(flank_a, L)
        fk_rc_win = np.lib.stride_tricks.sliding_window_view(flank_rc_a, L)
        for label, fw in (('fwd', fk_win), ('rc', fk_rc_win)):
            a_oh = np.eye(5, dtype=np.int8)[nc_win]
            b_oh = np.eye(5, dtype=np.int8)[fw]
            M = np.einsum('nlc,mlc->nm', a_oh, b_oh)
            idx = np.unravel_index(np.argmax(M), M.shape)
            m = int(M[idx])
            better = (m > best['matches'] or
                      (m == best['matches'] and (m / L) > (best['matches'] / best.get('L', 1))) or
                      (m == best['matches'] and L < best.get('L', 999)))
            if better:
                nc_start = int(idx[0])
                flank_start = int(idx[1])
                if label == 'rc':
                    flank_start = len(flank) - flank_start - L
                best = {
                    'L': L, 'orient': label, 'matches': m, 'identity': m / L,
                    'nc_start': nc_start, 'flank_start': flank_start,
                    'guide_seq': nc[nc_start:nc_start + L],
                    'target_seq': flank[flank_start:flank_start + L],
                }
    return best


def junction_distance(a: dict, flank_side: str) -> int:
    """bp from junction to the near edge of the best alignment.

    downstream: junction = flank[0], near edge = flank_start
    upstream  : junction = flank[FLANK_LEN-1], near edge = flank_start + L - 1
                distance = FLANK_LEN - (flank_start + L)
    """
    if flank_side == 'downstream':
        return int(a['flank_start'])
    else:
        return int(FLANK_LEN - a['flank_start'] - a['L'])


def load_bags():
    by_tnp = defaultdict(list)
    fams = {}
    with open(REAL_JSONL) as f:
        for line in f:
            r = json.loads(line)
            by_tnp[r['transposase_id']].append(r)
            fams[r['transposase_id']] = r['generator_metadata']['is_family']
    return by_tnp, fams


def align_records(recs) -> list[dict]:
    out = []
    for r in recs:
        acn = r['labels'].get('active_noncoding_index')
        if acn is None:
            continue
        ncs = r['inputs']['noncoding_regions']
        if acn >= len(ncs):
            continue
        nc = ncs[acn]
        flank = r['inputs']['flank']
        if len(nc) < 10 or len(flank) < 10:
            continue
        a = best_short_alignment(nc, flank)
        a['site_id'] = r['site_id']
        a['flank_side'] = r['generator_metadata']['flank_side']
        a['rev_comp'] = r['generator_metadata']['reverse_complemented']
        a['nc_len'] = len(nc)
        a['jdist'] = junction_distance(a, a['flank_side'])
        out.append(a)
    return out


def _mad(x):
    x = np.asarray(x)
    return float(np.median(np.abs(x - np.median(x))))


def summarize_side(alns_side: list[dict]) -> dict:
    if not alns_side:
        return {'n': 0}
    jdists = [a['jdist'] for a in alns_side]
    nc_starts = [a['nc_start'] for a in alns_side]
    orient_rc = sum(1 for a in alns_side if a['orient'] == 'rc') / len(alns_side)
    return {
        'n': len(alns_side),
        'jdist_median': float(np.median(jdists)),
        'jdist_mad': _mad(jdists),
        'jdist_frac_near': float(np.mean([j <= JUNCTION_WINDOW for j in jdists])),
        'nc_start_median': float(np.median(nc_starts)),
        'nc_start_mad': _mad(nc_starts),
        'nc_start_iqr': float(np.percentile(nc_starts, 75) - np.percentile(nc_starts, 25)),
        'orient_rc_frac': float(orient_rc),
        'identity_median': float(np.median([a['identity'] for a in alns_side])),
        'nc_len_median': float(np.median([a['nc_len'] for a in alns_side])),
    }


def shuffle_side_flanks(recs, flank_side: str, seed: int) -> list[dict]:
    """Within a bag, take sites with the given flank_side, and permute their
    flank strings across sites (destroys site-specific NC<->flank pairing).
    Return the shuffled recs (only that subgroup)."""
    subset = [r for r in recs
              if r['generator_metadata']['flank_side'] == flank_side]
    if len(subset) < 2:
        return subset
    flanks = [r['inputs']['flank'] for r in subset]
    rng = random.Random(seed)
    perm = list(range(len(subset)))
    for _ in range(50):
        rng.shuffle(perm)
        if all(i != perm[i] for i in range(len(subset))):
            break
    out = []
    for i, r in enumerate(subset):
        r2 = json.loads(json.dumps(r))
        r2['inputs']['flank'] = flanks[perm[i]]
        out.append(r2)
    return out


def audit_bag(recs, seed=1234) -> dict:
    alns = align_records(recs)
    by_side = defaultdict(list)
    for a in alns:
        by_side[a['flank_side']].append(a)
    up = summarize_side(by_side.get('upstream', []))
    dn = summarize_side(by_side.get('downstream', []))
    # NC-position separation between LtG (upstream) and RtG (downstream) --
    # if the rule holds, they live at DIFFERENT residues of the bRNA
    nc_sep = float('nan')
    if up.get('n', 0) >= 3 and dn.get('n', 0) >= 3:
        nc_sep = abs(up['nc_start_median'] - dn['nc_start_median'])
    # Shuffle controls
    up_shuf_recs = shuffle_side_flanks(recs, 'upstream', seed=seed)
    dn_shuf_recs = shuffle_side_flanks(recs, 'downstream', seed=seed + 1)
    up_shuf = summarize_side(align_records(up_shuf_recs))
    dn_shuf = summarize_side(align_records(dn_shuf_recs))
    return {
        'up': up, 'dn': dn,
        'up_shuf': up_shuf, 'dn_shuf': dn_shuf,
        'nc_sep_up_dn': nc_sep,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-per-family', type=int, default=10)
    p.add_argument('--n-control-per-family', type=int, default=5)
    p.add_argument('--controls', nargs='+', default=['IS30', 'IS903', 'IS10-R'])
    p.add_argument('--min-sites', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    by_tnp, fams = load_bags()
    rng = random.Random(args.seed)
    picks = {}
    for fam in ['IS110'] + args.controls:
        cands = [t for t, recs in by_tnp.items()
                 if fams[t] == fam and len(recs) >= args.min_sites]
        n_pick = args.n_per_family if fam == 'IS110' else args.n_control_per_family
        pool = list(cands); rng.shuffle(pool)
        picks[fam] = pool[:n_pick]
        print(f'  {fam:<10}: {len(cands)} bags with n>={args.min_sites}, picked {len(picks[fam])}')

    all_bags = {}
    for fam, tnps in picks.items():
        for tnp in tnps:
            print(f'  [{fam}] {tnp}  n={len(by_tnp[tnp])} ...')
            all_bags[tnp] = {'family': fam, 'n_sites': len(by_tnp[tnp]),
                             **audit_bag(by_tnp[tnp], seed=1234)}

    # Family-level summary
    lines = []
    lines.append('=' * 120)
    lines.append('  LTG / RTG audit -- do best short alignments concentrate at the junction end of the flank?')
    lines.append('=' * 120)
    lines.append('')
    lines.append('  For each bag, sites are split by flank_side:')
    lines.append('    upstream flank   -> junction sits at flank[119]  ->  test: LtG hits last ~15 bp')
    lines.append('    downstream flank -> junction sits at flank[0]    ->  test: RtG hits first ~15 bp')
    lines.append(f'  A site is a "junction hit" if best alignment is within {JUNCTION_WINDOW} bp of the junction.')
    lines.append(f'  Guide length swept over {list(LENGTHS)} bp.')
    lines.append('')

    header = (f'  {"family":<10} {"n_bags":>6}  '
              f'{"jd50_up":>7} {"jd50_dn":>7}  '
              f'{"near_up":>7} {"near_dn":>7}  '
              f'{"near_up_shuf":>12} {"near_dn_shuf":>12}  '
              f'{"nc_sep_updn":>11}  {"rc_frac":>7}')
    lines.append(header)
    lines.append('  ' + '-' * (len(header) - 2))

    fam_bags = defaultdict(list)
    for tnp, info in all_bags.items():
        fam_bags[info['family']].append(info)

    for fam in ['IS110'] + args.controls:
        rows = fam_bags.get(fam, [])
        if not rows: continue
        def _agg(f):
            v = [f(r) for r in rows]
            v = [x for x in v if x is not None and not (isinstance(x, float) and np.isnan(x))]
            return float(np.median(v)) if v else float('nan')
        lines.append(
            f'  {fam:<10} {len(rows):>6}  '
            f'{_agg(lambda r: r["up"].get("jdist_median")):>7.1f} '
            f'{_agg(lambda r: r["dn"].get("jdist_median")):>7.1f}  '
            f'{_agg(lambda r: r["up"].get("jdist_frac_near")):>7.2f} '
            f'{_agg(lambda r: r["dn"].get("jdist_frac_near")):>7.2f}  '
            f'{_agg(lambda r: r["up_shuf"].get("jdist_frac_near")):>12.2f} '
            f'{_agg(lambda r: r["dn_shuf"].get("jdist_frac_near")):>12.2f}  '
            f'{_agg(lambda r: r.get("nc_sep_up_dn")):>11.1f}  '
            f'{_agg(lambda r: 0.5 * ((r["up"].get("orient_rc_frac") or 0) + (r["dn"].get("orient_rc_frac") or 0))):>7.2f}'
        )

    lines.append('')
    lines.append('Column key:')
    lines.append('  jd50_up / jd50_dn      : median junction distance (bp) for upstream / downstream sites')
    lines.append(f'                            LtG/RtG expect: <= {JUNCTION_WINDOW}')
    lines.append(f'  near_up / near_dn      : fraction of sites within {JUNCTION_WINDOW} bp of junction')
    lines.append('                            LtG/RtG expect: high (>> 15/120 = 0.125 random baseline)')
    lines.append('  near_up/dn_shuf        : same, after within-side shuffle of NC<->flank pairing')
    lines.append('                            LtG/RtG expect: DROPS from unshuffled')
    lines.append('  nc_sep_up_dn           : |median nc_start_up - median nc_start_dn| across the bag')
    lines.append('                            LtG/RtG expect: > 20 (LtG and RtG at different bRNA residues)')
    lines.append('  rc_frac                : fraction of best alignments in RC orientation')
    lines.append('                            LtG/RtG expect: ~0.5 (either strand of dsDNA target)')
    lines.append('')
    lines.append('DECISION:')
    lines.append('  If IS110 shows HIGHER near_up + near_dn than controls, AND shuffle-shuffle drops it,')
    lines.append('  AND nc_sep_up_dn > 20 -> LtG/RtG rule IS visible in the raw data.')
    lines.append('  If none of these -> the raw data does NOT expose LtG/RtG at ungapped short alignment.')

    (out / 'ltg_rtg_summary.txt').write_text('\n'.join(lines))
    with open(out / 'ltg_rtg_raw.json', 'w') as f:
        json.dump(all_bags, f, indent=2, default=str)

    print()
    print('\n'.join(lines))
    print(f'\n[out] {out}')


if __name__ == '__main__':
    main()
