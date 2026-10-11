#!/usr/bin/env python3
"""Export exact copy-carrier evidence requests for a selected bag cohort.

The v5 bag records contain the carrier assembly name and reconstructed empty
flank, but not the copy's contig or occupied interval. This exporter preserves
the original site key and inherited nc payload while marking unavailable
carrier coordinates/sequence as unknown. It never treats the source event's
alleles or nc sequence as copy-level evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--jsonl', required=True, type=Path)
    ap.add_argument('--audit', required=True, type=Path,
                    help='exact selected bag/site audit table')
    ap.add_argument('--bag-list', required=True, type=Path)
    ap.add_argument('--outdir', required=True, type=Path)
    a = ap.parse_args()

    bags = {x.strip() for x in a.bag_list.read_text().splitlines() if x.strip()}
    with a.audit.open(encoding='utf-8-sig', newline='') as fh:
        audit_all = {(r['bag_id'], r['site_id']): r for r in csv.DictReader(fh, delimiter='\t')
                     if r['bag_id'] in bags}
    audit = {k: r for k, r in audit_all.items()
             if re.search(r'\.copy\d+$', r['site_id'])}
    selected = set(audit)
    copies = {}
    with a.jsonl.open(encoding='utf-8') as fh:
        for line in fh:
            d = json.loads(line)
            key = (d.get('bag_id', ''), d.get('site_id', ''))
            if key in selected and re.search(r'\.copy\d+$', key[1]):
                copies.setdefault(key, d)
    missing = selected - copies.keys()
    if missing:
        raise SystemExit(f'{len(missing)} selected copy rows missing from JSONL')

    a.outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    carriers = Counter()
    for (bag, site), d in sorted(copies.items()):
        carrier = d.get('carrier', '')
        carriers[carrier] += 1
        nc = d.get('noncoding_regions') or []
        rows.append({
            'bag_id': bag, 'site_id': site,
            'source_event': d.get('source_event', ''),
            'species': d.get('species', ''), 'carrier_assembly': carrier,
            'contig': 'UNKNOWN_NOT_IN_HF_PROVENANCE',
            'occupied_start0': 'UNKNOWN_NOT_IN_HF_PROVENANCE',
            'occupied_end0': 'UNKNOWN_NOT_IN_HF_PROVENANCE',
            'inserted_len_bp': d.get('inserted_len', ''),
            'insert_md5': d.get('insert_md5', ''),
            'copy_pct_ident': d.get('copy_pct_ident', ''),
            'tsd_len_reported': d.get('tsd_len', ''),
            'carrier_empty_flank120': d.get('flank', ''),
            'flank_source': d.get('flank_source', ''),
            'nc_sequence_hash_inherited': d.get('nc_sequence_hash', ''),
            'nc_sequence_inherited_regions': '|'.join(nc),
            'nc_copy_status': 'UNKNOWN_INHERITED_FROM_SOURCE_EVENT',
            'coordinate_status': 'UNKNOWN_REQUIRES_COPY_SITE_TABLE_OR_CARRIER_FASTA',
            'audit_locus_id': audit[(bag, site)].get('locus_id_full', ''),
        })
    fields = list(rows[0]) if rows else ['bag_id', 'site_id']
    with (a.outdir / 'copy_carrier_requests.tsv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)
    with (a.outdir / 'carrier_assembly_counts.tsv').open('w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh, delimiter='\t', lineterminator='\n')
        w.writerow(['carrier_assembly', 'n_selected_copy_sites'])
        w.writerows(sorted(carriers.items(), key=lambda x: (-x[1], x[0])))
    summary = {
        'selected_bags': len(bags), 'selected_sites': len(audit_all),
        'copy_sites': len(rows), 'unique_carrier_assemblies': len(carriers),
        'copy_rows_with_coordinates': 0,
        'copy_rows_with_actual_carrier_occupied_sequence': 0,
        'copy_rows_with_actual_copy_nc': 0,
        'unknown_status': 'No copy-site coordinate table or carrier genome FASTA is present in the pinned HF provenance package. Parent/source-event sequence is not substituted.',
    }
    (a.outdir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
