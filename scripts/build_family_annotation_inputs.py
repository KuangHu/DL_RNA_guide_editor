"""Predict all insertion ORFs for an entire bag cohort, without score selection."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pyrodigal

STATE = threading.local()


def predict(item):
    iid, sequence = item
    if not hasattr(STATE, 'finder'):
        STATE.finder = pyrodigal.GeneFinder(meta=True)
    if len(sequence) < 90:
        return iid, []
    genes = STATE.finder.find_genes(sequence.encode())
    return iid, [dict(lo=g.begin-1, hi=g.end, strand=g.strand,
                     partial_begin=bool(g.partial_begin), partial_end=bool(g.partial_end),
                     protein=g.translate().rstrip('*')) for g in genes]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--provenance-root', type=Path, required=True)
    ap.add_argument('--bags-tsv', type=Path, required=True)
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    bags = {r['bag_id'] for r in csv.DictReader(args.bags_tsv.open(), delimiter='\t')}
    db_to_bag = {r['db_id']: r['cds_cluster_id'] for r in csv.DictReader(
        (args.provenance_root/'cds_v3/insert_cds.tsv').open(), delimiter='\t')
        if r['cds_cluster_id'] in bags}
    sequences, events, invalid = {}, [], []
    with (args.provenance_root/'database_v3/insertions.tsv').open() as fh:
        for r in csv.DictReader(fh, delimiter='\t'):
            if r['db_id'] not in db_to_bag:
                continue
            sequence = r['insert_seq']
            digest = hashlib.md5(sequence.encode()).hexdigest()
            if digest[:12] != r['insert_md5'] or len(sequence) != int(r['inserted_len']):
                invalid.append(r['db_id']); continue
            iid = 'I_' + digest
            if iid in sequences and sequences[iid] != sequence:
                raise ValueError('insert identifier collision')
            sequences[iid] = sequence
            events.append(dict(bag_id=db_to_bag[r['db_id']], db_id=r['db_id'], insert_id=iid,
                               species=r['species'], length=len(sequence), insert_md5=r['insert_md5']))
    args.outdir.mkdir(parents=True, exist_ok=True)
    with (args.outdir/'source_events.tsv').open('w', newline='', encoding='utf8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(events[0]), delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(events)
    with (args.outdir/'insert_sequences.fna').open('w') as fh:
        for iid, sequence in sequences.items():
            fh.write(f'>{iid}\n{sequence}\n')
    proteins, ngenes, no_orfs = {}, 0, 0
    fields = ['insert_id','gene_idx','protein_id','lo','hi','strand','partial_begin','partial_end','protein_len']
    with (args.outdir/'gene_origins.tsv').open('w', newline='', encoding='utf8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields, delimiter='\t', lineterminator='\n'); w.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for number, (iid, genes) in enumerate(pool.map(predict, sequences.items()), 1):
                no_orfs += not genes
                for idx, g in enumerate(genes):
                    sequence = g.pop('protein')
                    pid = 'P_' + hashlib.sha256(sequence.encode()).hexdigest()
                    if pid in proteins and proteins[pid] != sequence:
                        raise ValueError('protein identifier collision')
                    proteins[pid] = sequence
                    w.writerow(dict(insert_id=iid, gene_idx=idx, protein_id=pid,
                                    protein_len=len(sequence), **g))
                    ngenes += 1
                if number % 5000 == 0:
                    print(f'Predicted {number}/{len(sequences)} unique insert sequences', flush=True)
    with (args.outdir/'proteins.faa').open('w') as fh:
        for pid, protein in proteins.items():
            fh.write(f'>{pid}\n{protein}\n')
    manifest = dict(n_bags_requested=len(bags), n_bags_with_source_sequences=len({e['bag_id'] for e in events}),
                    n_source_events=len(events), n_unique_insert_sequences=len(sequences),
                    n_gene_occurrences=ngenes, n_unique_proteins=len(proteins), n_inserts_without_predicted_orfs=no_orfs,
                    invalid_source_event_ids=invalid, pyrodigal_version=pyrodigal.__version__,
                    scope='All source events for every requested bag; no signal scores or rankings read.',
                    sequence_source='database_v3/insertions.tsv insert_seq, verified length and MD5; source-event sequences do not validate actual carrier-copy sequence.')
    (args.outdir/'input_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__': main()
