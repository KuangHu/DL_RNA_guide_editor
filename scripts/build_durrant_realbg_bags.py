"""Rebuild Durrant fake bags with REAL bacterial 120bp flanks.

For each row in IS110_gold_v0.jsonl:
  - Take the cognate 14bp genome_target
  - Pick a random real bacterial 120bp flank from intermediate/sites_part2.jsonl
  - Splice: flank_out = cognate_14bp + real_flank[14:]  (junction at [0])
  - Use the bRNA (177nt) as the active NC
  - Emit a JSONL record matching real_all.jsonl schema

Also emits a NC-padded version where the 177nt bRNA is padded to synthetic-median
length (~240nt) with random-ACGT flanking padding — for search-space-matched
comparison with V4 synthetic NCs.

Both outputs go to /global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference/
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer')
sys.path.insert(0, '/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/scripts')

from audit_level2c_gold_inject import find_ltg_position, load_ltg_specs

GOLD_JSONL = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/curated/is110_gold_v0.jsonl')
SITES_PART2 = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/intermediate/sites_part2.jsonl')
OUT_DIR = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference')


def load_real_bacterial_flanks(n_needed, seed=42):
    """Read the first n_needed 120bp bacterial flanks from sites_part2.jsonl."""
    flanks = []
    with open(SITES_PART2) as f:
        for line in f:
            r = json.loads(line)
            fl = r['inputs']['flank']
            if len(fl) == 120:
                flanks.append(fl.upper())
            if len(flanks) >= n_needed * 3:
                break
    rng = random.Random(seed)
    rng.shuffle(flanks)
    return flanks[:n_needed]


def rna_to_dna(s):
    return s.replace('U', 'T').replace('u', 't').upper() if s else s


def random_dna(rng, n):
    return ''.join(rng.choice('ACGT') for _ in range(n))


def make_record(site_id, bag_id, brna_id, brna_dna, flank_120,
                 cognate_14bp, flank_side='downstream'):
    """Splice cognate_14bp at flank[0:14], keep flank[14:] from bacterial source."""
    assert len(flank_120) == 120
    assert len(cognate_14bp) == 14
    if flank_side == 'downstream':
        flank_out = cognate_14bp + flank_120[14:]
    else:
        flank_out = flank_120[:-14] + cognate_14bp
    return {
        'site_id': site_id,
        'transposase_id': bag_id,
        'ncrna_id': f'{bag_id}_ncrna',
        'inputs': {
            'flank': flank_out,
            'noncoding_regions': [brna_dna],
        },
        'labels': {
            'is_positive': True,
            'target_position_in_flank': [0, 14] if flank_side == 'downstream' else [106, 120],
            'target_dna': cognate_14bp,
            'match_orientation': 'forward',
            'guide_dna': None,
            'perfect_guide_dna': None,
            'guide_length': 11,
            'n_mismatches': None,
            'mismatch_positions': None,
            'active_noncoding_index': 0,
            'num_noncoding_regions': 1,
            'ncrna_span_in_active_noncoding': [0, len(brna_dna)],
            'guide_span_in_active_noncoding': None,
            'designed_structure': None,
            'guide_unpaired_in_fold': None,
            'ncrna_length': len(brna_dna),
            'site_class': 'durrant_gold',
        },
        'generator_metadata': {
            'data_source': 'Durrant2024_realbg',
            'is_family': 'IS110',
            'is_id': brna_id,
            'flank_side': flank_side,
            'reverse_complemented': False,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--min-sites-per-brna', type=int, default=5)
    ap.add_argument('--bag-size', type=int, default=5)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--nc-target-length', type=int, default=None,
                    help='If set, pad the bRNA with random ACGT padding to this length '
                         'to match synthetic NC search-space size.')
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs = load_ltg_specs()

    # Load Durrant curated rows
    curated = []
    with open(GOLD_JSONL) as f:
        for line in f:
            r = json.loads(line)
            b = r.get('ortholog_id')
            brna_seq = r.get('brna_sequence')
            tgt14 = r.get('genome_target_14bp')
            if not (b and brna_seq and tgt14): continue
            if len(tgt14) != 14: continue
            curated.append(r)

    # Group by ortholog
    from collections import defaultdict
    by_bR = defaultdict(list)
    for r in curated:
        by_bR[r['ortholog_id']].append(r)

    print(f'[gold] {len(curated)} rows across {len(by_bR)} bRNAs')

    # Load enough real bacterial flanks
    rng = random.Random(args.seed)
    n_needed = sum(len(v) for v in by_bR.values() if len(v) >= args.min_sites_per_brna)
    n_needed *= 2  # cognate + shuffled
    print(f'[flanks] loading {n_needed} real bacterial 120bp flanks')
    flanks = load_real_bacterial_flanks(n_needed, seed=args.seed)
    flank_idx = 0

    # Also build a target pool for shuffled bags
    target_pool = []
    for r in curated:
        target_pool.append({'bR': r['ortholog_id'], 'tgt14': r['genome_target_14bp']})

    tag = 'realbg'
    if args.nc_target_length:
        tag = f'realbg_ncpad{args.nc_target_length}'
    cog_path = OUT_DIR / f'durrant_cognate_{tag}.jsonl'
    shu_path = OUT_DIR / f'durrant_shuffled_{tag}.jsonl'

    n_cog = n_shu = 0
    with cog_path.open('w') as fcog, shu_path.open('w') as fshu:
        for bR, rows in by_bR.items():
            if len(rows) < args.min_sites_per_brna: continue
            # bRNA DNA
            brna_dna = rna_to_dna(rows[0]['brna_sequence'])
            if args.nc_target_length and args.nc_target_length > len(brna_dna):
                pad_total = args.nc_target_length - len(brna_dna)
                pad5 = pad_total // 2; pad3 = pad_total - pad5
                brna_dna = random_dna(rng, pad5) + brna_dna + random_dna(rng, pad3)

            rows_filtered = [r for r in rows if r.get('genome_target_14bp')]
            rng.shuffle(rows_filtered)
            for i in range(0, len(rows_filtered), args.bag_size):
                chunk = rows_filtered[i:i + args.bag_size]
                if len(chunk) < args.bag_size: break
                cog_bag = f'durrant_realbg_{bR}_cog_bag{i//args.bag_size:03d}'
                shu_bag = f'durrant_realbg_{bR}_shu_bag{i//args.bag_size:03d}'
                for j, r in enumerate(chunk):
                    if flank_idx + 1 >= len(flanks): break
                    fl_cog = flanks[flank_idx]; flank_idx += 1
                    fl_shu = flanks[flank_idx]; flank_idx += 1
                    tgt_cog = r['genome_target_14bp']
                    others = [t for t in target_pool if t['bR'] != bR]
                    tgt_shu = rng.choice(others)['tgt14']
                    rec_cog = make_record(
                        site_id=f'{cog_bag}_site_{j:04d}', bag_id=cog_bag,
                        brna_id=bR, brna_dna=brna_dna, flank_120=fl_cog,
                        cognate_14bp=tgt_cog)
                    rec_shu = make_record(
                        site_id=f'{shu_bag}_site_{j:04d}', bag_id=shu_bag,
                        brna_id=bR, brna_dna=brna_dna, flank_120=fl_shu,
                        cognate_14bp=tgt_shu)
                    fcog.write(json.dumps(rec_cog) + '\n'); n_cog += 1
                    fshu.write(json.dumps(rec_shu) + '\n'); n_shu += 1

    print(f'[out] {cog_path}   ({n_cog} sites)')
    print(f'[out] {shu_path}   ({n_shu} sites)')


if __name__ == '__main__':
    main()
