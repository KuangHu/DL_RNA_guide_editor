#!/usr/bin/env python3
"""Audit copy joins and actual forward-genome slices without inheriting boundaries."""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from Bio import SeqIO


def rc(seq):
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]


def rows(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def audit(manifest, resolved, fasta, outdir):
    requests = {(r['bag_id'], r['site_id']): r for r in rows(manifest)}
    records = {}
    for record in SeqIO.parse(fasta, 'fasta'):
        fields = record.id.split('|')
        key = tuple(fields[:2])
        if key in records:
            raise ValueError(f'duplicate slice key {key}')
        records[key] = (fields, str(record.seq).upper())
    counts = Counter()
    issues = []
    usable = []
    seen = set()
    for row in rows(resolved):
        counts['resolved_rows'] += 1
        key = (row['bag_id'], row['site_id'])
        if key in seen:
            raise ValueError(f'duplicate resolved key {key}')
        seen.add(key)
        req = requests.get(key)
        problems = []
        if req is None:
            problems.append('request_missing')
        else:
            for column in ('carrier_assembly', 'source_event'):
                if row[column] != req[column]:
                    problems.append(column + '_mismatch')
            if row['flank_120'] != req['carrier_empty_flank120']:
                problems.append('requested_flank_mismatch')
        if key not in records:
            counts['slice_missing'] += 1
        else:
            counts['slices_checked'] += 1
            fields, seq = records[key]
            lo, hi = int(row['slice_lo_0b']), int(row['slice_hi_0b_exclusive'])
            start = int(row['elem_start_0b']) - lo
            end = int(row['elem_end_0b_exclusive']) - lo
            if (fields[2:6] != [row['carrier_assembly'], row['contig'], str(lo), str(hi)]
                    or len(seq) != hi - lo):
                problems.append('slice_header_or_length_mismatch')
            if not (60 <= start < end <= len(seq) - 60):
                problems.append('slice_interval_or_flank_bounds')
            else:
                flank = seq[start-60:start] + seq[end:end+60]
                expected = req['carrier_empty_flank120'] if req else ''
                if min(flank, rc(flank)) != min(expected, rc(expected)):
                    problems.append('slice_does_not_reproduce_request_flank')
                insert = seq[start:end]
                source_hash = req.get('insert_md5', '') if req else ''
                forward = hashlib.md5(insert.encode()).hexdigest()[:len(source_hash)]
                reverse = hashlib.md5(rc(insert).encode()).hexdigest()[:len(source_hash)]
                if source_hash and forward == source_hash:
                    counts['forward_exact_source_md5'] += 1
                elif source_hash and reverse == source_hash:
                    counts['reverse_complement_exact_source_md5'] += 1
                else:
                    counts['actual_copy_not_exact_source_on_either_strand'] += 1
            if not problems:
                usable.append({'bag_id': key[0], 'site_id': key[1],
                               'status': 'join_and_slice_verified_not_boundary_or_nc_verified'})
        if problems:
            counts['quarantined_join_rows'] += 1
            issues.append({'bag_id': key[0], 'site_id': key[1],
                           'expected_carrier': req['carrier_assembly'] if req else '',
                           'resolved_carrier': row['carrier_assembly'],
                           'issues': ';'.join(problems)})
    counts['usable_slice_joins'] = len(usable)
    counts['metadata_joins_verified'] = counts['resolved_rows'] - counts['quarantined_join_rows']
    outdir.mkdir(parents=True, exist_ok=True)
    for name, items, fields in [('join_issues.tsv', issues,
                                ['bag_id', 'site_id', 'expected_carrier', 'resolved_carrier', 'issues']),
                               ('verified_slice_joins.tsv', usable, ['bag_id', 'site_id', 'status'])]:
        with (outdir / name).open('w', encoding='utf-8', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
            writer.writeheader()
            writer.writerows(items)
    summary = {'counts': dict(counts), 'input_sha256': {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest, resolved, fasta)},
        'limits': ['Metadata insert_md5 is inherited, not actual-copy identity evidence.',
                   'Hash counts include all checked slices, including quarantined joins.',
                   'Verified joins do not verify insertion boundaries, nc extraction or event independence.',
                   'Copysites orient is flank canonicalization, not element orientation.',
                   'Reconstructed empty flanks are hypotheses, not observed empty alleles.']}
    (outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ('manifest', 'resolved', 'fasta', 'outdir'):
        parser.add_argument('--' + argument, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.manifest, args.resolved, args.fasta, args.outdir), indent=2))
