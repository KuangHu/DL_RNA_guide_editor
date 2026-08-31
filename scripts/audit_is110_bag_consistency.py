"""Per-bag biology audit: do real IS110 bags share a consistent recognition rule?

For each bag in a family, and for each site in the bag, compute the ORACLE best
ungapped short alignment between the active NC region and the flank, sweeping:

    L      ∈ {10, 12, 14, 16}
    orient ∈ {fwd, rc}
    nc_start, flank_start over all valid positions.

Best = highest #matches; ties → highest identity ratio → smallest L.
Record for each site: (L*, orient*, matches, identity, nc_start, flank_start,
guide_seq_on_nc, target_seq_on_flank).

Then for each bag, compute WITHIN-BAG consistency:
  - fraction of dominant orientation
  - MAD of flank_start (does the target hit a similar strip of the flank?)
  - MAD of nc_start   (does the guide come from a similar strip of the NC?)
  - MAD of L*
  - median identity of the top match
  - centroid target motif + mean per-site identity to centroid
      (measures "do sites share a consensus target motif?")
  - centroid guide motif  + mean per-site identity to centroid

Finally compare IS110 vs IS30 / IS903 / IS10-R controls — if IS110 real bags
truly share a recognition rule, they should be MORE consistent than controls.

Sample size (default): 10 IS110 bags + 5 each of IS30, IS903, IS10-R controls,
all with n_sites >= 30. Adjustable via --n-per-family.

Output:
  --out-dir/audit_summary.json     per-bag + per-family consistency metrics
  --out-dir/bag_<tnp_id>.txt       human-readable per-site table + consensus
  --out-dir/family_summary.txt     side-by-side family comparison
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REAL_JSONL = '/global/scratch/users/kh36969/DL_novel_guide_editor/real_data/inference/real_all.jsonl'
BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
RC = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
FAMILIES = ('IS110', 'IS30', 'IS903', 'IS10-R', 'ISLdl1', 'ISAjo2')


def seq_to_arr(s: str) -> np.ndarray:
    return np.asarray([BASE_MAP.get(c, 4) for c in s.upper()], dtype=np.int8)


def rc_seq(s: str) -> str:
    return ''.join(RC.get(c, 'N') for c in s[::-1].upper())


def best_ungapped_alignment(nc: str, flank: str,
                            lengths=(10, 12, 14, 16)) -> dict:
    """Sweep L, orient, positions; return best ungapped match.

    Match rule: L bp on NC vs L bp on flank. For fwd orient the flank is used
    as-is (guide 5'→3' pairs with target 3'→5' — but we score by direct base
    equality on nc vs flank for a "hybridization-like" ungapped rule using
    reverse-complement pairing? A guide on the RNA/DNA pairs with the target's
    complement. To keep the audit simple and interpretable, we compute BOTH:

        raw_match     : nc[i:i+L] == flank[j:j+L]        (same-sense identity)
        rc_match      : nc[i:i+L] == RC(flank[j:j+L])   (guide<->target pairing)

    and take whichever is higher. "orient=fwd" means raw_match won,
    "orient=rc" means rc_match won.
    """
    nc_a = seq_to_arr(nc)
    flank_a = seq_to_arr(flank)
    flank_rc_a = seq_to_arr(rc_seq(flank))

    best = {'matches': -1}
    for L in lengths:
        if len(nc_a) < L or len(flank_a) < L:
            continue
        # Build sliding windows: shape (n_nc_pos, L) and (n_flank_pos, L)
        n_nc = len(nc_a) - L + 1
        n_fk = len(flank_a) - L + 1
        # nc_win[i,k] = nc_a[i+k]; use stride trick via broadcasting
        nc_win = np.lib.stride_tricks.sliding_window_view(nc_a, L)  # (n_nc, L)
        fk_win = np.lib.stride_tricks.sliding_window_view(flank_a, L)  # (n_fk, L)
        fk_rc_win = np.lib.stride_tricks.sliding_window_view(flank_rc_a, L)

        # Match count matrix M[i,j] = sum(nc_win[i] == fk_win[j])
        # Do it via one-hot expansion: (n_nc, L, 5) and (n_fk, L, 5) → einsum
        def _match_matrix(a, b):
            # a: (na, L), b: (nb, L)
            a_oh = np.eye(5, dtype=np.int8)[a]  # (na, L, 5)
            b_oh = np.eye(5, dtype=np.int8)[b]  # (nb, L, 5)
            return np.einsum('nlc,mlc->nm', a_oh, b_oh)  # (na, nb)

        M_raw = _match_matrix(nc_win, fk_win)
        M_rc = _match_matrix(nc_win, fk_rc_win)

        # Find best over both orients
        for label, M, flank_ref in (('fwd', M_raw, flank),
                                     ('rc', M_rc, flank_a)):
            idx = np.unravel_index(np.argmax(M), M.shape)
            m = int(M[idx])
            if (m > best['matches'] or
                (m == best['matches'] and L < best.get('L', 999))):
                nc_start = int(idx[0])
                flank_start = int(idx[1])
                if label == 'fwd':
                    target_seq = flank[flank_start:flank_start + L]
                    guide_seq = nc[nc_start:nc_start + L]
                else:  # rc
                    # flank_start indexes into flank_rc; convert to fwd flank coord
                    # rc[j:j+L] == RC(flank[N-j-L:N-j])
                    N = len(flank)
                    fwd_flank_start = N - flank_start - L
                    target_seq = flank[fwd_flank_start:fwd_flank_start + L]
                    guide_seq = nc[nc_start:nc_start + L]
                    flank_start = fwd_flank_start  # report in fwd coords
                best = {
                    'L': L,
                    'orient': label,
                    'matches': m,
                    'identity': m / L,
                    'nc_start': nc_start,
                    'flank_start': flank_start,
                    'guide_seq': guide_seq,
                    'target_seq': target_seq,
                }
    return best


def load_bags():
    by_tnp = defaultdict(list)
    fams = {}
    with open(REAL_JSONL) as f:
        for line in f:
            r = json.loads(line)
            by_tnp[r['transposase_id']].append(r)
            fams[r['transposase_id']] = r['generator_metadata']['is_family']
    return by_tnp, fams


def compute_bag_alignments(recs) -> list[dict]:
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
        aln = best_ungapped_alignment(nc, flank)
        aln['site_id'] = r['site_id']
        aln['nc_len'] = len(nc)
        aln['flank_len'] = len(flank)
        aln['active_nc_idx'] = acn
        aln['flank_side'] = r['generator_metadata'].get('flank_side')
        aln['rev_comp'] = r['generator_metadata'].get('reverse_complemented')
        out.append(aln)
    return out


def consensus_column(seqs: list[str]) -> tuple[str, float]:
    """Column-wise consensus of equal-length seqs. Returns (consensus, mean_identity)."""
    if not seqs:
        return '', float('nan')
    L = min(len(s) for s in seqs)
    seqs = [s[:L] for s in seqs]
    cols = np.asarray([[BASE_MAP.get(c, 4) for c in s] for s in seqs])
    cons = []
    for c in range(L):
        u, counts = np.unique(cols[:, c], return_counts=True)
        cons.append('ACGTN'[u[np.argmax(counts)]])
    cons_s = ''.join(cons)
    # mean per-site identity to consensus
    ids = []
    for s in seqs:
        m = sum(1 for a, b in zip(s, cons_s) if a == b)
        ids.append(m / L)
    return cons_s, float(np.mean(ids))


def analyze_bag(alns: list[dict]) -> dict:
    if not alns:
        return {'n': 0}
    orients = [a['orient'] for a in alns]
    orient_counts = Counter(orients)
    top_orient, top_orient_n = orient_counts.most_common(1)[0]

    flank_starts = np.asarray([a['flank_start'] for a in alns])
    nc_starts = np.asarray([a['nc_start'] for a in alns])
    Ls = np.asarray([a['L'] for a in alns])
    idents = np.asarray([a['identity'] for a in alns])

    def _mad(a):
        m = np.median(a)
        return float(np.median(np.abs(a - m)))

    # Consensus target/guide (over dominant orient at modal L)
    modal_L = int(np.bincount(Ls).argmax())
    dom_sub = [a for a in alns if a['orient'] == top_orient and a['L'] == modal_L]
    tgt_cons, tgt_mean_id = consensus_column([a['target_seq'] for a in dom_sub])
    gd_cons, gd_mean_id = consensus_column([a['guide_seq'] for a in dom_sub])

    return {
        'n': len(alns),
        'orient_fwd_frac': orients.count('fwd') / len(alns),
        'orient_dominant': top_orient,
        'orient_dominant_frac': top_orient_n / len(alns),
        'modal_L': modal_L,
        'modal_L_frac': int((Ls == modal_L).sum()) / len(alns),
        'flank_start_median': float(np.median(flank_starts)),
        'flank_start_mad': _mad(flank_starts),
        'flank_start_iqr': float(np.percentile(flank_starts, 75) - np.percentile(flank_starts, 25)),
        'nc_start_median': float(np.median(nc_starts)),
        'nc_start_mad': _mad(nc_starts),
        'nc_start_iqr': float(np.percentile(nc_starts, 75) - np.percentile(nc_starts, 25)),
        'identity_median': float(np.median(idents)),
        'identity_max': float(np.max(idents)),
        'identity_min': float(np.min(idents)),
        'target_consensus': tgt_cons,
        'target_cons_mean_id': tgt_mean_id,
        'target_cons_n_sites': len(dom_sub),
        'guide_consensus': gd_cons,
        'guide_cons_mean_id': gd_mean_id,
    }


def write_bag_report(bag_id, family, alns, summary, path: Path):
    lines = []
    lines.append(f'BAG {bag_id}   family={family}   n_sites={len(alns)}')
    lines.append('=' * 100)
    lines.append(f'Dominant orient : {summary["orient_dominant"]}  '
                 f'({summary["orient_dominant_frac"]:.2f} of sites)')
    lines.append(f'Modal L          : {summary["modal_L"]}bp  '
                 f'({summary["modal_L_frac"]:.2f} of sites)')
    lines.append(f'Identity of best : median={summary["identity_median"]:.3f}  '
                 f'range=[{summary["identity_min"]:.3f}, {summary["identity_max"]:.3f}]')
    lines.append(f'Flank position   : median={summary["flank_start_median"]:.0f}  '
                 f'MAD={summary["flank_start_mad"]:.1f}  '
                 f'IQR={summary["flank_start_iqr"]:.1f}')
    lines.append(f'NC position      : median={summary["nc_start_median"]:.0f}  '
                 f'MAD={summary["nc_start_mad"]:.1f}  '
                 f'IQR={summary["nc_start_iqr"]:.1f}')
    lines.append('')
    lines.append(f'Consensus (dominant orient + modal L, n={summary["target_cons_n_sites"]}):')
    lines.append(f'  target_cons: {summary["target_consensus"]}  '
                 f'(mean site→cons identity = {summary["target_cons_mean_id"]:.3f})')
    lines.append(f'  guide_cons : {summary["guide_consensus"]}  '
                 f'(mean site→cons identity = {summary["guide_cons_mean_id"]:.3f})')
    lines.append('')
    lines.append('Per-site best ungapped alignment:')
    lines.append(f'  {"site":<40} {"L":>2} {"ori":>3} {"match":>5} {"id":>5} '
                 f'{"nc":>5} {"flnk":>5}  {"target":<18} {"guide":<18}')
    for a in sorted(alns, key=lambda x: x['site_id']):
        lines.append(f'  {a["site_id"]:<40} {a["L"]:>2} {a["orient"]:>3} '
                     f'{a["matches"]:>5} {a["identity"]:>5.3f} '
                     f'{a["nc_start"]:>5} {a["flank_start"]:>5}  '
                     f'{a["target_seq"]:<18} {a["guide_seq"]:<18}')
    path.write_text('\n'.join(lines))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', required=True)
    p.add_argument('--n-per-family', type=int, default=10)
    p.add_argument('--n-control-per-family', type=int, default=5)
    p.add_argument('--controls', nargs='+',
                    default=['IS30', 'IS903', 'IS10-R'])
    p.add_argument('--min-sites', type=int, default=30)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[load] {REAL_JSONL}')
    by_tnp, fams = load_bags()
    print(f'  {len(by_tnp)} bags total')

    rng = random.Random(args.seed)
    picks = {}
    for fam in ['IS110'] + args.controls:
        candidates = [t for t, recs in by_tnp.items()
                      if fams[t] == fam and len(recs) >= args.min_sites]
        n_pick = args.n_per_family if fam == 'IS110' else args.n_control_per_family
        pool = list(candidates)
        rng.shuffle(pool)
        picks[fam] = pool[:n_pick]
        print(f'  {fam:<10}: {len(candidates)} bags with n>={args.min_sites}, '
              f'picked {len(picks[fam])}')

    all_bag_summaries = {}
    for fam, tnps in picks.items():
        for tnp in tnps:
            print(f'  [{fam}] {tnp}  n={len(by_tnp[tnp])} ...')
            alns = compute_bag_alignments(by_tnp[tnp])
            summ = analyze_bag(alns)
            all_bag_summaries[tnp] = {
                'family': fam,
                'summary': summ,
                'n_sites_used': len(alns),
            }
            write_bag_report(tnp, fam, alns, summ,
                             out_dir / f'bag_{tnp}.txt')

    # Family-level summary
    fam_sums = defaultdict(list)
    for tnp, info in all_bag_summaries.items():
        fam_sums[info['family']].append(info['summary'])

    lines = []
    lines.append('FAMILY-LEVEL CONSISTENCY COMPARISON')
    lines.append('=' * 100)
    lines.append('')
    lines.append(f'  {"family":<10} {"n_bags":>7} {"orient_dom":>10} '
                 f'{"modal_L":>8} {"L_frac":>7} '
                 f'{"idt_med":>8} {"fk_MAD":>7} {"nc_MAD":>7} '
                 f'{"tgt_cons_id":>12} {"gd_cons_id":>11}')
    lines.append(f'  {"-"*10} {"-"*7} {"-"*10} {"-"*8} {"-"*7} '
                 f'{"-"*8} {"-"*7} {"-"*7} {"-"*12} {"-"*11}')

    for fam in ['IS110'] + args.controls:
        rows = fam_sums.get(fam, [])
        if not rows:
            continue
        def _agg(key):
            v = [r[key] for r in rows if r.get(key) is not None
                 and not (isinstance(r[key], float) and np.isnan(r[key]))]
            return float(np.median(v)) if v else float('nan')
        lines.append(
            f'  {fam:<10} {len(rows):>7}  '
            f'{_agg("orient_dominant_frac"):>9.2f} '
            f'{_agg("modal_L"):>7.0f} {_agg("modal_L_frac"):>7.2f} '
            f'{_agg("identity_median"):>8.3f} '
            f'{_agg("flank_start_mad"):>7.1f} {_agg("nc_start_mad"):>7.1f} '
            f'{_agg("target_cons_mean_id"):>12.3f} '
            f'{_agg("guide_cons_mean_id"):>11.3f}'
        )

    lines.append('')
    lines.append('Column key:')
    lines.append('  orient_dom  : fraction of sites agreeing on best-orient (higher = consistent)')
    lines.append('  modal_L     : most common best-L across sites in the bag')
    lines.append('  L_frac      : fraction of sites at modal L')
    lines.append('  idt_med     : median identity of best ungapped alignment')
    lines.append('  fk_MAD      : median absolute deviation of best-align flank_start (lower = consistent target position)')
    lines.append('  nc_MAD      : median absolute deviation of best-align nc_start (lower = consistent guide position)')
    lines.append('  tgt_cons_id : mean per-site identity to consensus target motif (higher = shared target)')
    lines.append('  gd_cons_id  : mean per-site identity to consensus guide motif (higher = shared guide)')
    lines.append('')
    lines.append('CLAIM to test:')
    lines.append('  If IS110 real bags share a consistent recognition rule, IS110 should show')
    lines.append('  HIGHER orient_dom, HIGHER L_frac, LOWER fk_MAD, LOWER nc_MAD, HIGHER cons_id')
    lines.append('  than control families (IS30, IS903, IS10-R).')

    (out_dir / 'family_summary.txt').write_text('\n'.join(lines))
    with open(out_dir / 'audit_summary.json', 'w') as f:
        json.dump({tnp: v for tnp, v in all_bag_summaries.items()}, f, indent=2)

    print()
    print('\n'.join(lines))
    print(f'\n[out] {out_dir}')


if __name__ == '__main__':
    main()
