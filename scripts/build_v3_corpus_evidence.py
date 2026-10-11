"""Build sequence-bound Step-2 evidence from local locus/allele provenance.

The target frame is the ORF-oriented midpoint of the extracted empty flank,
not a claim that an ambiguous insertion junction is biologically resolved.
Every globally optimal equal-length deletion boundary is retained. Unsupported
indel decompositions, count disagreements, ambiguous sources and paralogs stay
explicitly unresolved. MSA uses an anchor-split global star alignment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import pyrodigal
from Bio import Align, SeqIO
from Bio.Align import substitution_matrices

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.alignment import revcomp
from scripts.run_v3_scorer import _bags_from_jsonl
from scripts.v5a_framework.verified_coordinates import _prepare, input_digest


def md5(sequence):
    return hashlib.md5(sequence.encode()).hexdigest()[:12]


def all_boundaries(occupied, empty, row):
    """Full-domain minimum-Hamming deletion boundaries; no probe window."""
    ilen = int(row['inserted_len'])
    n = len(empty)
    if ilen <= 0 or len(occupied) - n != ilen:
        raise ValueError('unsupported_indel_or_replacement_decomposition')
    prefix = [0]
    for a, b in zip(occupied[:n], empty):
        prefix.append(prefix[-1] + (a != b))
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + (occupied[i + ilen] != empty[i])
    scores = [prefix[i] + suffix[i] for i in range(n + 1)]
    optimum = min(scores)
    starts = [i for i, score in enumerate(scores) if score == optimum]
    method = row.get('decomposition_method')
    if method == 'exact' and optimum != 0:
        raise ValueError('exact_decomposition_does_not_reproduce_empty')
    if method not in ('exact', 'tolerant_alignment'):
        raise ValueError('unsupported_decomposition_method')
    if method == 'tolerant_alignment':
        identity = float(row['homologous_identity'])
        if identity > 1:
            identity /= 100
        if 1 - optimum / n + 0.005 < identity:
            raise ValueError('deletion_identity_below_upstream_identity')
    if len(starts) != int(row['n_equivalent_decompositions']):
        raise ValueError(f'boundary_count_disagreement:{len(starts)}:{row["n_equivalent_decompositions"]}')
    stored = row.get('inserted_md5', '').strip()
    if len(stored) != 12:
        raise ValueError('missing_insert_md5')
    chosen = [i for i in starts if md5(occupied[i:i + ilen]) == stored]
    if not chosen:
        raise ValueError('no_optimal_boundary_matches_insert_md5')
    explicit = row.get('insert_start_in_long_bp', '')
    if explicit.lstrip('-').isdigit() and int(explicit) not in chosen:
        raise ValueError('stored_boundary_not_optimal_or_md5_mismatch')
    return [(i, i + ilen) for i in starts], optimum


def unique_source(sequence, source):
    hits = []
    for strand, query in (('+', sequence), ('-', revcomp(sequence))):
        start = source.find(query)
        while start >= 0:
            hits.append((start, start + len(query), strand))
            start = source.find(query, start + 1)
    if len(hits) != 1:
        raise ValueError(f'flank_source_not_unique:{len(hits)}')
    return hits[0]


PROTEIN = Align.PairwiseAligner()
PROTEIN.substitution_matrix = substitution_matrices.load('BLOSUM62')
PROTEIN.open_gap_score = -10
PROTEIN.extend_gap_score = -0.5
DNA = Align.PairwiseAligner()
DNA.match_score = 2
DNA.mismatch_score = -3
DNA.open_gap_score = -5
DNA.extend_gap_score = -0.5


@lru_cache(maxsize=65536)
def protein_homology(a, b):
    if not a or not b or min(len(a), len(b)) / max(len(a), len(b)) < 0.9:
        return False
    alignment = PROTEIN.align(a, b)[0]
    paired = matches = 0
    for x, y in zip(*alignment.indices):
        if x >= 0 and y >= 0:
            paired += 1
            matches += a[x] == b[y]
    return paired / max(len(a), len(b)) >= 0.9 and matches / paired >= 0.95


def genes_in_stable_insert(sequence, intervals):
    lo, hi = max(x[0] for x in intervals), min(x[1] for x in intervals)
    genes = pyrodigal.GeneFinder(meta=True).find_genes(sequence.encode())
    result = []
    for gene in genes:
        start, end = gene.begin - 1, gene.end
        protein = gene.translate().rstrip('*')
        if lo <= start < end <= hi and len(protein) >= 60 and not gene.partial_begin and not gene.partial_end:
            result.append(dict(lo=start, hi=end, strand='+' if gene.strand == 1 else '-', protein=protein))
    return result


def choose_anchor(sites):
    """No transitive family chains; every unique member matches one reference."""
    best = None
    for idx, site in sorted(sites.items()):
        for gene in site['genes']:
            members = {}
            for other_idx, other in sites.items():
                hits = [g for g in other['genes'] if protein_homology(gene['protein'], g['protein'])]
                if len(hits) == 1:
                    members[other_idx] = hits[0]
            if idx not in members or len(members) < 2:
                continue
            score = (len(members), min(len(g['protein']) for g in members.values()), -idx)
            if best is None or score > best[0]:
                best = score, idx, gene, members
    if best is None:
        raise ValueError('no_unique_conserved_orf_anchor')
    return best[1:]


def pair_strings(reference, query):
    if not reference:
        return '-' * len(query), query, dict(identity=None, coverage=0 if query else 1)
    if not query:
        return reference, '-' * len(reference), dict(identity=None, coverage=0)
    alignment = DNA.align(reference, query)[0]
    a, b = [], []
    paired = matches = 0
    for x, y in zip(*alignment.indices):
        a.append(reference[x] if x >= 0 else '-')
        b.append(query[y] if y >= 0 else '-')
        if x >= 0 and y >= 0:
            paired += 1
            matches += reference[x] == query[y]
    return ''.join(a), ''.join(b), dict(identity=matches / paired if paired else 0,
                                      coverage=paired / max(len(reference), len(query)))


def star_segment(sequences, reference_idx):
    reference = sequences[reference_idx]
    representations, metrics = {}, {}
    width = [0] * (len(reference) + 1)
    for idx, query in sequences.items():
        a, b, metrics[idx] = pair_strings(reference, query)
        inserted = [''] * (len(reference) + 1)
        bases = ['-'] * len(reference)
        position = 0
        for x, y in zip(a, b):
            if x == '-':
                inserted[position] += y
            else:
                bases[position] = y
                position += 1
        for p, value in enumerate(inserted):
            width[p] = max(width[p], len(value))
        representations[idx] = inserted, bases
    rows = {}
    for idx, (inserted, bases) in representations.items():
        row = []
        for p in range(len(reference) + 1):
            row.append(inserted[p] + '-' * (width[p] - len(inserted[p])))
            if p < len(reference):
                row.append(bases[p])
        rows[idx] = ''.join(row)
    return rows, metrics


def anchored_msa(sites, members, reference_idx):
    contexts, starts, parts = {}, {}, {}
    for idx, gene in members.items():
        site = sites[idx]
        sequence = site['occupied']
        clo = min(x[0] for x in site['intervals'])
        chi = max(x[1] for x in site['intervals'])
        if chi - clo > 10000:
            raise ValueError('context_exceeds_10000_bp_alignment_limit')
        alo, ahi = gene['lo'], gene['hi']
        if gene['strand'] == '-':
            sequence = revcomp(sequence)
            clo, chi = len(sequence) - chi, len(sequence) - clo
            alo, ahi = len(sequence) - ahi, len(sequence) - alo
        contexts[idx] = sequence[clo:chi]
        starts[idx] = clo
        parts[idx] = (sequence[clo:alo], sequence[alo:ahi], sequence[ahi:chi])
    segments, metrics = [], {}
    for part in range(3):
        rows, quality = star_segment({i: p[part] for i, p in parts.items()}, reference_idx)
        segments.append(rows)
        for idx, values in quality.items():
            metrics.setdefault(idx, []).append(values)
    accepted = {}
    for idx in members:
        # Large unmatched segments are unsafe even when the ORF is conserved.
        for part in range(3):
            length = max(len(parts[idx][part]), len(parts[reference_idx][part]))
            q = metrics[idx][part]
            if length >= 30 and (q['identity'] is None or q['identity'] < 0.70 or q['coverage'] < 0.70):
                break
        else:
            accepted[idx] = ''.join(segment[idx] for segment in segments)
            assert accepted[idx].replace('-', '') == contexts[idx]
    return accepted, starts, metrics


def read_sources(provenance_root, audit_rows):
    wanted = defaultdict(dict)
    for key, audit in audit_rows.items():
        full = audit['locus_id_full']
        if not full or '|' not in full:
            continue
        prefix, short = full.split('|', 1)
        panel, directory = prefix.split('/', 1)
        wanted[(panel, directory)][short] = None
    result = {}
    for pkey, shorts in wanted.items():
        base = provenance_root / pkey[0] / 'l2' / pkey[1]
        with (base / 'loci.tsv').open(encoding='utf-8', newline='') as fh:
            rows = {r['locus_id']: r for r in csv.DictReader(fh, delimiter='\t') if r['locus_id'] in shorts}
        alleles = defaultdict(dict)
        for record in SeqIO.parse(base / 'alleles.fna', 'fasta'):
            fields = record.id.split('|')
            if len(fields) >= 2 and fields[0] in rows:
                row = rows[fields[0]]
                if fields[1] in (row['longest_allele'], row['shortest_allele']):
                    alleles[fields[0]][fields[1]] = str(record.seq).upper()
        for short, row in rows.items():
            result[f'{pkey[0]}/{pkey[1]}|{short}'] = (row, alleles.get(short, {}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jsonl', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--provenance-root', type=Path, required=True)
    parser.add_argument('--bag-list', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    args = parser.parse_args()
    ids = {line.split()[0] for line in args.bag_list.read_text().splitlines() if line.strip()}
    bags = _bags_from_jsonl(args.jsonl, ids)
    if ids - set(bags):
        raise ValueError(f'requested_bags_missing_from_input:{sorted(ids - set(bags))}')
    args.outdir.mkdir(parents=True, exist_ok=True)
    audit_rows = {}
    with args.audit.open(encoding='utf-8', newline='') as fh:
        for row in csv.DictReader(fh, delimiter='\t'):
            if row['bag_id'] not in ids:
                continue
            key = row['bag_id'], row['site_id']
            if key in audit_rows:
                raise ValueError(f'duplicate_audit_key:{key}')
            audit_rows[key] = row
    sources = read_sources(args.provenance_root, audit_rows)
    statuses, summaries, manifest = [], [], []
    for bi, (bid, inputs) in enumerate(bags.items(), 1):
        prepared, errors = {}, {}
        for idx, site in enumerate(inputs):
            try:
                # 162_build_bags_v5.py copies nc from source_event but
                # reconstructs the target from a DIFFERENT carrier locus.
                # Parent event alleles cannot validate that carrier's nc.
                if re.search(r'\.copy\d+$', site['site_id']):
                    raise ValueError('copy_requires_actual_carrier_alleles_and_nc')
                audit = audit_rows[(bid, site['site_id'])]
                row, alleles = sources[audit['locus_id_full']]
                occupied = alleles[row['longest_allele']]
                empty = alleles[row['shortest_allele']]
                target_source = unique_source(site['inputs']['flank'].upper(), empty)
                intervals, minimum_mm = all_boundaries(occupied, empty, row)
                genes = genes_in_stable_insert(occupied, intervals)
                if not genes:
                    raise ValueError('no_complete_orf_in_stable_insert')
                prepared[idx] = dict(row=row, occupied=occupied, empty=empty,
                                     target_source=target_source, intervals=intervals,
                                     minimum_mm=minimum_mm, genes=genes)
            except (KeyError, ValueError) as exc:
                errors[idx] = str(exc)
        members, aligned, metrics = {}, {}, {}
        try:
            reference_idx, reference, members = choose_anchor(prepared)
            aligned, starts, metrics = anchored_msa(prepared, members, reference_idx)
            family = bid + ':' + hashlib.sha256(reference['protein'].encode()).hexdigest()[:16]
            aid = bid + ':' + hashlib.sha256(json.dumps(aligned, sort_keys=True).encode()).hexdigest()[:16]
            bdir = args.outdir / bid
            bdir.mkdir(exist_ok=True)
            with (bdir / 'oriented_context_msa.fna').open('w', encoding='utf-8') as fh:
                for idx, sequence in aligned.items():
                    fh.write(f'>site_{idx}\n{sequence}\n')
            (bdir / 'alignment_quality.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
            with (bdir / 'anchor_proteins.faa').open('w', encoding='utf-8') as fh:
                for idx, gene in members.items():
                    fh.write(f'>site_{idx}\n{gene["protein"]}\n')
            for idx in prepared:
                if idx not in members:
                    errors[idx] = 'no_unique_member_of_anchor_family'
                elif idx not in aligned:
                    errors[idx] = 'context_alignment_quality_failed'
                else:
                    p, gene = prepared[idx], members[idx]
                    lo, hi, strand = p['target_source']
                    # Physical interbase midpoint; independent of any motif hit.
                    midpoint = lo + 60 if strand == '+' else hi - 60
                    direction = 1 if gene['strand'] == '+' else -1
                    source_first = lo if strand == '+' else hi - 1
                    if hi - lo != 120:
                        errors[idx] = 'unexpected_flank_geometry'
                        continue
                    row = p['row']
                    e = dict(occupied=p['occupied'], empty=p['empty'],
                             anchor=dict(family=family, strand=gene['strand'], occupied_lo=gene['lo'], occupied_hi=gene['hi']),
                             alignment_id=aid, oriented_context_msa=aligned[idx], context_lo=starts[idx],
                             valid_insert_intervals=p['intervals'],
                             target=dict(frame='orf_oriented_empty_flank_midpoint_v1',
                                         strand='+' if (1 if strand == '+' else -1) * direction == 1 else '-',
                                         origin=(source_first - midpoint) * direction, origin_status='verified',
                                         empty_lo=lo, empty_hi=hi, source_strand=strand,
                                         frame_strand=gene['strand'], frame_anchor_empty_pos=midpoint),
                             locus=dict(pre_event_genome=row.get('pre_event_genome', ''),
                                        pre_event_contig=row.get('pre_event_contig', ''),
                                        reference_genome=row.get('seed_genome', ''),
                                        reference_contig=row.get('contig', ''),
                                        reference_start=int(row['start']), reference_end=int(row['end'])),
                             builder=dict(method='anchor_split_global_star_v1', pyrodigal_version=pyrodigal.__version__,
                                          boundary_method='global_min_hamming_deletion', minimum_mm=p['minimum_mm'],
                                          target_frame_is_extraction_midpoint=True, locus_id_full=audit_rows[(bid, inputs[idx]['site_id'])]['locus_id_full']))
                    candidate = dict(inputs[idx], scorer_evidence=e)
                    check = _prepare(candidate)
                    if any(s.status != 'unique' for s in check['sources']):
                        errors[idx] = 'unresolved_fragment_sources'
                    manifest.append(dict(bag_id=bid, site_idx=idx, input_sha256=input_digest(inputs[idx]), evidence=e))
        except ValueError as exc:
            for idx in prepared:
                errors.setdefault(idx, str(exc))
        n_evidence = sum(r['bag_id'] == bid for r in manifest)
        for idx, site in enumerate(inputs):
            statuses.append(dict(bag_id=bid, site_idx=idx, site_id=site['site_id'],
                                 status=errors.get(idx, 'ready'), n_boundaries=len(prepared[idx]['intervals']) if idx in prepared else 0,
                                 anchor_strand=members[idx]['strand'] if idx in members else '',
                                 anchor_lo=members[idx]['lo'] if idx in members else '',
                                 anchor_hi=members[idx]['hi'] if idx in members else '',
                                 msa_columns=len(aligned[idx]) if idx in aligned else 0))
        summaries.append(dict(bag_id=bid, n_sites=len(inputs), n_evidence=n_evidence,
                              n_ready=len(inputs) - len(errors), complete=not errors,
                              errors=errors))
        print(f'[{bi}/{len(bags)}] {bid}: evidence={n_evidence}/{len(inputs)} ready={len(inputs)-len(errors)}', flush=True)
    with (args.outdir / 'verified_evidence.jsonl').open('w', encoding='utf-8') as fh:
        for row in manifest:
            fh.write(json.dumps(row) + '\n')
    with (args.outdir / 'site_status.tsv').open('w', encoding='utf-8', newline='') as fh:
        fields = ['bag_id', 'site_idx', 'site_id', 'status', 'n_boundaries', 'anchor_strand', 'anchor_lo', 'anchor_hi', 'msa_columns']
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter='\t')
        writer.writeheader()
        writer.writerows(statuses)
    (args.outdir / 'bag_status.json').write_text(json.dumps(summaries, indent=2), encoding='utf-8')
    def file_sha256(path):
        digest = hashlib.sha256()
        with path.open('rb') as fh:
            for chunk in iter(lambda: fh.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
    run = dict(complete=True, method='anchor_split_global_star_v1',
               input_jsonl=str(args.jsonl), input_sha256=file_sha256(args.jsonl),
               audit_tsv=str(args.audit), audit_sha256=file_sha256(args.audit),
               provenance_root=str(args.provenance_root), script_sha256=file_sha256(Path(__file__)),
               pyrodigal_version=pyrodigal.__version__,
               parameters=dict(protein_identity=0.95, protein_coverage=0.9, minimum_protein_aa=60,
                               context_identity=0.70, context_coverage=0.70, minimum_checked_segment_bp=30,
                               maximum_context_bp=10000, flank_bp=120, target_frame='orf_oriented_empty_flank_midpoint_v1'),
               n_bags=len(bags), n_sites=len(statuses), n_evidence=len(manifest),
               n_complete_bags=sum(s['complete'] for s in summaries),
               site_status_counts=dict(Counter(s['status'].split(':')[0] for s in statuses)))
    (args.outdir / 'builder_manifest.json').write_text(json.dumps(run, indent=2), encoding='utf-8')
    print(f'Saved {len(manifest)} site evidence records; complete bags={sum(s["complete"] for s in summaries)}', flush=True)


if __name__ == '__main__':
    main()
