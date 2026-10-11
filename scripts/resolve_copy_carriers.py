#!/usr/bin/env python3
"""Resolve the 4,128 quarantined copy-site evidence from on-cluster carrier data.

Reads evidence/_step2/copy_carrier_requests/copy_carrier_requests.tsv (the
inventory produced off-cluster), joins each `.copyNNNN` site against the
local fna_ins_discovery copysites tables (per-species + shards), and emits:

  - copy_carrier_resolved.tsv : one row per copy site with carrier
    contig/elem_start/elem_end/tsd/pct_ident/orient/empty_site_source and
    cross-check against the inherited insert_md5.
  - copy_carrier_slices.fna   : carrier contig slice `[elem_start - PAD :
    elem_end + PAD]` per site, forward strand, header
    `>bag_id|site_id|carrier|contig|slice_lo|slice_hi|orient_from_copysites`.

Nothing is substituted from the source event. If a copy row cannot be
located in the cluster copysites tables, or if the inherited insert_md5
disagrees with the copysites `insert_md5`, the site is written to
`copy_carrier_unresolved.tsv` with the reason, never silently defaulted.

The 15-species sweep uses two schemas:
  * 17-col: db_id, source_event, species, carrier, contig, elem_start,
    elem_end, tsd_len, pct_ident, q_cov, flank, flank_fwd, orient, elem_len,
    insert_md5, inserted_len, empty_site_source
  * 14-col (spneumoniae legacy variants): no flank_fwd/orient/elem_len
    (orient defaults to `+` because the legacy scanner only emitted
    forward hits; verify downstream).

Carrier FASTAs are discovered by filesystem-wide rglob of
`genomes*` dirs (same scheme as scripts/step1_corpus_audit.py), so the
species->dir hardcoding bug from Step 1 round 2 doesn't recur.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO


COPYSITES_SPECIES_FILES = {
    'abaum': ['abaum_sites.tsv'],
    'bsubtilis': ['bsubtilis_sites.tsv'],
    'cjejuni': ['cjejuni_sites.tsv'],
    'ctrachomatis': ['ctrachomatis_sites.tsv'],
    'ecoli': [],
    'efaecalis': ['efaecalis_sites.tsv'],
    'efaecium': ['efaecium_sites.tsv'],
    'hpylori': ['hpylori_sites.tsv'],
    'kpneu': [],
    'lmonocytogenes': ['lmonocytogenes_sites.tsv'],
    'mpneumoniae': ['mpneumoniae_sites.tsv'],
    'mtuberculosis': ['mtuberculosis_sites.tsv'],
    'ngonorrhoeae': ['ngonorrhoeae_sites.tsv'],
    'paeruginosa': ['paeruginosa_sites.tsv'],
    'sa_k24': [],
    'senterica': ['senterica_sites.tsv'],
    'spneumoniae': ['spneumoniae_sites.tsv', 'spneu_v3_sites.tsv',
                    'spneu_v4_sites.tsv'],
    'tpallidum': ['tpallidum_sites.tsv'],
}


RESOLVED_COLS = [
    'bag_id', 'site_id', 'source_event', 'species', 'carrier_assembly',
    'contig', 'elem_start_0b', 'elem_end_0b_exclusive', 'elem_len_from_copysites',
    'tsd_len', 'pct_ident', 'q_cov', 'orient', 'empty_site_source',
    'flank_120', 'flank_fwd_120', 'copysites_insert_md5',
    'inherited_insert_md5', 'md5_match',
    'carrier_fasta_relpath', 'contig_len_bp',
    'slice_lo_0b', 'slice_hi_0b_exclusive', 'slice_pad_bp',
    'copysites_source_file',
]


def iter_copysites_rows(root: Path, species: str):
    """Yield (filename, dict) for every row of every copysites tsv the given
    species may live in. Includes the per-species top-level file and all
    `shards/<species>.s##_sites.tsv` shards for 14/15 species. Spneumoniae
    additionally reads spneu_v3/v4_sites.tsv."""
    for name in COPYSITES_SPECIES_FILES.get(species, [f'{species}_sites.tsv']):
        p = root / name
        if p.exists() and p.stat().st_size > 0:
            with p.open(encoding='utf-8', newline='') as fh:
                for r in csv.DictReader(fh, delimiter='\t'):
                    yield name, r
    for sh in sorted((root / 'shards').glob(f'{species}.s*_sites.tsv')):
        with sh.open(encoding='utf-8', newline='') as fh:
            for r in csv.DictReader(fh, delimiter='\t'):
                yield f'shards/{sh.name}', r


def build_assembly_index(discovery_root: Path) -> dict:
    """Filesystem-wide rglob('*.fna') over every `genomes*/` dir. Non-archive
    dirs (not containing `__done_`) win when both carry the same stem."""
    cands = []
    for p in discovery_root.iterdir():
        if not p.is_dir():
            continue
        if p.name.startswith('genomes'):
            cands.append(p)
        for sub in p.glob('genomes*'):
            if sub.is_dir():
                cands.append(sub)
    idx: dict[str, Path] = {}
    for d in cands:
        archive = '__done_' in str(d)
        for fna in d.rglob('*.fna'):
            prior = idx.get(fna.stem)
            if prior is None:
                idx[fna.stem] = fna
            elif archive:
                continue
            elif '__done_' in str(prior):
                idx[fna.stem] = fna
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True,
                    help='evidence/_step2/copy_carrier_requests/copy_carrier_requests.tsv')
    ap.add_argument('--discovery-root', type=Path, required=True,
                    help='/global/scratch/users/kh36969/fna_ins_discovery')
    ap.add_argument('--outdir', type=Path, required=True)
    ap.add_argument('--pad-bp', type=int, default=200,
                    help='flank bp retained on each side of elem_start/elem_end')
    args = ap.parse_args()

    with args.manifest.open(encoding='utf-8', newline='') as fh:
        needed = [r for r in csv.DictReader(fh, delimiter='\t')]
    needed_by_species = defaultdict(list)
    needed_by_siteid: dict[str, dict] = {}
    for r in needed:
        needed_by_species[r['species']].append(r)
        needed_by_siteid[r['site_id']] = r
    print(f'manifest: {len(needed)} copy sites across {len(needed_by_species)} species')

    copysites_root = args.discovery_root / 'copysites'
    assert copysites_root.exists(), copysites_root

    found: dict[str, dict] = {}
    source_file: dict[str, str] = {}
    for species, rows in needed_by_species.items():
        wanted = {r['site_id'] for r in rows}
        hits_here = 0
        for fname, cr in iter_copysites_rows(copysites_root, species):
            if cr['db_id'] in wanted and cr['db_id'] not in found:
                found[cr['db_id']] = cr
                source_file[cr['db_id']] = fname
                hits_here += 1
                if len(found) % 500 == 0:
                    print(f'  resolved {len(found)}/{len(needed)}  (species={species} file={fname})')
        print(f'  {species:16s} needed={len(wanted):>5d}  resolved_in_cluster_copysites={len(wanted & found.keys()):>5d}')
    missing = [r for r in needed if r['site_id'] not in found]
    print(f'resolved: {len(found)}/{len(needed)}  unresolved: {len(missing)}')

    print('indexing carrier FASTAs...')
    asm_idx = build_assembly_index(args.discovery_root)
    print(f'  assembly stems in filesystem: {len(asm_idx)}')

    args.outdir.mkdir(parents=True, exist_ok=True)
    resolved_path = args.outdir / 'copy_carrier_resolved.tsv'
    fasta_path = args.outdir / 'copy_carrier_slices.fna'
    unresolved_path = args.outdir / 'copy_carrier_unresolved.tsv'

    unresolved_rows = []
    for r in missing:
        unresolved_rows.append({
            'bag_id': r['bag_id'], 'site_id': r['site_id'],
            'species': r['species'], 'carrier_assembly': r['carrier_assembly'],
            'reason': 'not_in_cluster_copysites_tables',
        })

    assembly_cache: dict[str, dict[str, str]] = {}
    def load_assembly(stem: str) -> dict[str, str] | None:
        if stem in assembly_cache:
            return assembly_cache[stem]
        p = asm_idx.get(stem)
        if p is None:
            assembly_cache[stem] = None  # type: ignore
            return None
        contigs: dict[str, str] = {}
        for rec in SeqIO.parse(p, 'fasta'):
            contigs[rec.id] = str(rec.seq).upper()
            short = rec.id.split()[0]
            if short != rec.id:
                contigs.setdefault(short, contigs[rec.id])
        assembly_cache[stem] = contigs
        return contigs

    carrier_fasta_relpath: dict[str, str] = {}
    for stem, p in asm_idx.items():
        try:
            carrier_fasta_relpath[stem] = str(p.relative_to(args.discovery_root))
        except ValueError:
            carrier_fasta_relpath[stem] = str(p)

    out_rows = []
    md5_disagreements = []
    slice_misses = []
    with fasta_path.open('w', encoding='utf-8') as ff:
        for site_id, cr in found.items():
            req = needed_by_siteid[site_id]
            elem_start = int(cr['elem_start'])
            elem_end = int(cr['elem_end'])
            carrier = cr['carrier']
            contigs = load_assembly(carrier)
            contig_len = -1
            slice_lo = slice_hi = -1
            if contigs is not None and cr['contig'] in contigs:
                seq = contigs[cr['contig']]
                contig_len = len(seq)
                slice_lo = max(0, elem_start - args.pad_bp)
                slice_hi = min(len(seq), elem_end + args.pad_bp)
                orient_from_copysites = cr.get('orient', '') or '+'
                header = (f'>{req["bag_id"]}|{site_id}|{carrier}|{cr["contig"]}|'
                          f'{slice_lo}|{slice_hi}|{orient_from_copysites}')
                ff.write(header + '\n')
                # 60-col wrap for readability
                sub = seq[slice_lo:slice_hi]
                for i in range(0, len(sub), 60):
                    ff.write(sub[i:i + 60] + '\n')
            else:
                reason = 'carrier_fasta_missing' if contigs is None else 'contig_not_in_carrier'
                slice_misses.append({
                    'bag_id': req['bag_id'], 'site_id': site_id,
                    'species': req['species'], 'carrier_assembly': carrier,
                    'contig': cr['contig'], 'reason': reason,
                })
            md5_match = cr.get('insert_md5', '') == req.get('insert_md5', '')
            if not md5_match:
                md5_disagreements.append({
                    'bag_id': req['bag_id'], 'site_id': site_id,
                    'inherited': req.get('insert_md5', ''),
                    'copysites': cr.get('insert_md5', ''),
                })
            out_rows.append({
                'bag_id': req['bag_id'],
                'site_id': site_id,
                'source_event': cr.get('source_event', ''),
                'species': req['species'],
                'carrier_assembly': carrier,
                'contig': cr['contig'],
                'elem_start_0b': elem_start,
                'elem_end_0b_exclusive': elem_end,
                'elem_len_from_copysites': cr.get('elem_len', '') or (elem_end - elem_start),
                'tsd_len': cr.get('tsd_len', ''),
                'pct_ident': cr.get('pct_ident', ''),
                'q_cov': cr.get('q_cov', ''),
                'orient': cr.get('orient', ''),
                'empty_site_source': cr.get('empty_site_source', ''),
                'flank_120': cr.get('flank', ''),
                'flank_fwd_120': cr.get('flank_fwd', ''),
                'copysites_insert_md5': cr.get('insert_md5', ''),
                'inherited_insert_md5': req.get('insert_md5', ''),
                'md5_match': 'yes' if md5_match else 'NO',
                'carrier_fasta_relpath': carrier_fasta_relpath.get(carrier, ''),
                'contig_len_bp': contig_len,
                'slice_lo_0b': slice_lo,
                'slice_hi_0b_exclusive': slice_hi,
                'slice_pad_bp': args.pad_bp,
                'copysites_source_file': source_file[site_id],
            })
    with resolved_path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=RESOLVED_COLS, delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(out_rows)

    all_unresolved = unresolved_rows + slice_misses
    with unresolved_path.open('w', encoding='utf-8', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['bag_id', 'site_id', 'species',
                                           'carrier_assembly', 'contig',
                                           'reason'], delimiter='\t',
                           lineterminator='\n', extrasaction='ignore')
        w.writeheader()
        w.writerows(all_unresolved)

    print(f'\n=== summary ===')
    print(f'resolved rows written: {len(out_rows)}')
    print(f'slice miss (carrier/contig missing locally): {len(slice_misses)}')
    print(f'manifest rows not found in cluster copysites: {len(unresolved_rows)}')
    print(f'md5 disagreements (insert_md5 drift copy vs source): {len(md5_disagreements)}')
    if md5_disagreements[:5]:
        print('  sample disagreements:')
        for d in md5_disagreements[:5]:
            print('   ', d)
    print(f'\noutputs:')
    print(f'  {resolved_path}')
    print(f'  {fasta_path}')
    print(f'  {unresolved_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
