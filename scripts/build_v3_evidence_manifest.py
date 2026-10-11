"""Bind supplied allele/ORF/MSA evidence to the actual scorer input records.

The index is JSONL: bag_id, site_idx, occupied_fasta, empty_fasta, msa_fasta, msa_record,
and evidence (anchor, alignment_id, context_lo, target,
valid_insert_intervals, locus). File paths resolve relative to the index.
This packages verified source data; it does not infer insertion boundaries,
ORF homology, or target origins from motif hits.
"""
import argparse
import json
from pathlib import Path
import sys
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_v3_scorer import _bags_from_jsonl
from scripts.v5a_framework.verified_coordinates import input_digest, _prepare


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', type=Path, required=True)
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    index = [json.loads(line) for line in args.index.read_text(encoding='utf-8').splitlines() if line.strip()]
    bags = _bags_from_jsonl(args.jsonl, {row['bag_id'] for row in index})
    result, seen, unresolved = [], set(), 0
    for row in index:
        bid, idx = row['bag_id'], int(row['site_idx'])
        if (bid, idx) in seen or idx < 0:
            raise ValueError(f'duplicate or invalid site index: {bid} {idx}')
        seen.add((bid, idx))
        site = dict(bags[bid][idx])
        e = dict(row['evidence'])
        occupied = list(SeqIO.parse(args.index.parent / row['occupied_fasta'], 'fasta'))
        if len(occupied) != 1:
            raise ValueError('occupied_fasta must contain exactly one allele')
        empty = list(SeqIO.parse(args.index.parent / row['empty_fasta'], 'fasta'))
        if len(empty) != 1:
            raise ValueError('empty_fasta must contain exactly one allele')
        msa_rows = list(SeqIO.parse(args.index.parent / row['msa_fasta'], 'fasta'))
        if len({r.id for r in msa_rows}) != len(msa_rows):
            raise ValueError('MSA contains duplicate record IDs')
        aligned = next(r for r in msa_rows if r.id == row['msa_record'])
        e['occupied'] = str(occupied[0].seq).upper()
        e['empty'] = str(empty[0].seq).upper()
        e['oriented_context_msa'] = str(aligned.seq).upper()
        site['scorer_evidence'] = e
        prepared = _prepare(site)
        if any(s.status != 'unique' for s in prepared['sources']):
            unresolved += 1
        result.append(dict(bag_id=bid, site_idx=idx, input_sha256=input_digest(site), evidence=e))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(''.join(json.dumps(r)+'\n' for r in result), encoding='utf-8')
    print(f'Saved {len(result)} site records; {unresolved} have unresolved fragment sources: {args.out}')


if __name__ == '__main__':
    main()
