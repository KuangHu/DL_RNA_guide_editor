"""Build model-format JSONL bags from Durrant IS110_gold_v0.

For each unique bRNA in Table 2 with >= N cognate insertion sites in Table 3:

  Cognate bag:
    site_i.flank = (14 bp genome-actual target from Table 3) || 106 bp random ACGT
    site_i.noncoding_regions = [WT bRNA DNA sequence from Table 2]
    site_i.labels.active_noncoding_index = 0
    flank_side = 'downstream'  (junction at flank[0], target embedded there)

  Shuffled bag:
    Same bRNA, but each site's flank uses a DIFFERENT bRNA's cognate 14 bp
    target as the 'target' embedded at flank[0:14].

Output:
  <out-dir>/durrant_cognate.jsonl
  <out-dir>/durrant_shuffled.jsonl

Both matching preprocess/tnp_dataset.py's expected record schema.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

GOLD_JSONL = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/curated/is110_gold_v0.jsonl')
OUT_DIR = Path('/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference')
OUT_DIR.mkdir(parents=True, exist_ok=True)

FLANK_LEN = 120
TARGET_LEN = 14
PAD_LEN = FLANK_LEN - TARGET_LEN


def rna_to_dna(s):
    return s.replace('U', 'T').replace('u', 't') if s else s


def make_record(site_id, bag_id, bR_id, brna_seq_dna, target_14bp, padding, flank_side='downstream'):
    """Build a record matching real_all.jsonl schema."""
    assert len(target_14bp) == TARGET_LEN, target_14bp
    assert len(padding) == PAD_LEN, padding
    flank = target_14bp + padding if flank_side == 'downstream' else padding + target_14bp
    return {
        'site_id': site_id,
        'transposase_id': bag_id,
        'ncrna_id': f'{bag_id}_ncrna',
        'inputs': {
            'flank': flank,
            'noncoding_regions': [brna_seq_dna],
        },
        'labels': {
            'is_positive': True,
            'target_position_in_flank': None,
            'target_dna': None,
            'match_orientation': None,
            'guide_dna': None,
            'perfect_guide_dna': None,
            'guide_length': None,
            'n_mismatches': None,
            'mismatch_positions': None,
            'designed_structure': None,
            'guide_unpaired_in_fold': None,
            'ncrna_span_in_active_noncoding': None,
            'ncrna_length': None,
            'guide_span_in_active_noncoding': None,
            'active_noncoding_index': 0,
            'num_noncoding_regions': 1,
            'site_class': 'durrant_gold',
        },
        'generator_metadata': {
            'data_source': 'Durrant2024',
            'is_family': 'IS110',
            'is_id': bR_id,
            'flank_side': flank_side,
            'reverse_complemented': False,
        },
    }


def random_dna(rng, n):
    return ''.join(rng.choice('ACGT') for _ in range(n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--min-sites-per-brna', type=int, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--bag-size', type=int, default=5,
                    help='Sites per output bag; the WT bRNA has ~173 sites, split into bags of this size.')
    args = p.parse_args()

    # Load gold
    rows = [json.loads(l) for l in GOLD_JSONL.read_text().splitlines()]
    # Group by ortholog_id (bRNA name)
    by_bR = defaultdict(list)
    for r in rows:
        if not r.get('ortholog_id') or not r.get('brna_sequence'):
            continue
        if not r.get('genome_target_14bp'):
            continue
        by_bR[r['ortholog_id']].append(r)

    print(f'[gold] bRNAs with cognate 14bp genome targets: {len(by_bR)}')
    for bR, sites in sorted(by_bR.items(), key=lambda kv: -len(kv[1]))[:10]:
        print(f'  {bR:<40} n_sites={len(sites)}  brna_len={len(sites[0]["brna_sequence"])}nt')

    # Build target pool from ALL bRNAs' cognate targets for shuffling
    target_pool = []
    for bR, sites in by_bR.items():
        for s in sites:
            if s['genome_target_14bp'] and len(s['genome_target_14bp']) == TARGET_LEN:
                target_pool.append({'bR': bR, 'target': s['genome_target_14bp']})
    print(f'[pool] {len(target_pool)} 14bp targets total; {len(set(x["target"] for x in target_pool))} unique')

    rng = random.Random(args.seed)

    cognate_records = []
    shuffled_records = []

    n_bags_written = 0
    for bR, sites in by_bR.items():
        if len(sites) < args.min_sites_per_brna:
            continue
        brna_dna = rna_to_dna(sites[0]['brna_sequence']).upper()
        # Split sites into bags of bag_size
        sites_filtered = [s for s in sites if s['genome_target_14bp']
                          and len(s['genome_target_14bp']) == TARGET_LEN]
        rng.shuffle(sites_filtered)
        bag_idx = 0
        for i in range(0, len(sites_filtered), args.bag_size):
            chunk = sites_filtered[i:i + args.bag_size]
            if len(chunk) < args.bag_size: break
            # Cognate bag
            cognate_bag_id = f'durrant_{bR}_cog_bag{bag_idx:03d}'
            for j, s in enumerate(chunk):
                padding = random_dna(rng, PAD_LEN)
                cognate_records.append(make_record(
                    site_id=f'{cognate_bag_id}_site_{j:04d}',
                    bag_id=cognate_bag_id,
                    bR_id=bR,
                    brna_seq_dna=brna_dna,
                    target_14bp=s['genome_target_14bp'],
                    padding=padding,
                ))
            # Shuffled bag: same bRNA, but each site's target replaced with a random OTHER-bRNA's target
            shuf_bag_id = f'durrant_{bR}_shu_bag{bag_idx:03d}'
            for j, s in enumerate(chunk):
                # pick a target from a different bRNA
                others = [t for t in target_pool if t['bR'] != bR]
                pick = rng.choice(others)
                padding = random_dna(rng, PAD_LEN)
                shuffled_records.append(make_record(
                    site_id=f'{shuf_bag_id}_site_{j:04d}',
                    bag_id=shuf_bag_id,
                    bR_id=bR,
                    brna_seq_dna=brna_dna,
                    target_14bp=pick['target'],
                    padding=padding,
                ))
            bag_idx += 1
            n_bags_written += 1

    print(f'\n[write] cognate:  {len(cognate_records)} sites in {len(set(r["transposase_id"] for r in cognate_records))} bags')
    print(f'[write] shuffled: {len(shuffled_records)} sites in {len(set(r["transposase_id"] for r in shuffled_records))} bags')

    cog_path = OUT_DIR / 'durrant_cognate.jsonl'
    shu_path = OUT_DIR / 'durrant_shuffled.jsonl'
    with cog_path.open('w') as f:
        for r in cognate_records: f.write(json.dumps(r) + '\n')
    with shu_path.open('w') as f:
        for r in shuffled_records: f.write(json.dumps(r) + '\n')
    print(f'\n[out] {cog_path}')
    print(f'[out] {shu_path}')


if __name__ == '__main__':
    main()
