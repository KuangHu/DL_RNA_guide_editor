"""Validate supplied ORF-anchored MSA evidence and project full hit intervals.

All coordinates are zero-based, half-open. Target origin is the canonical
coordinate of the FIRST base of the raw flank (decreases when strand='-').
No extraction-boundary, fragment-origin or target-origin guesses are made.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, replace
from hashlib import sha256
import json
from pathlib import Path

from preprocess.alignment import revcomp
from scripts.nc_coord_mapping import map_fragments_to_occupied, project_interval
from scripts.v5a_framework.multi_hit_retention import BagRetention, SiteRetention, SPACER_N


def input_digest(site: dict) -> str:
    value = json.dumps(dict(inputs=site['inputs'], site_id=site.get('site_id'),
                            transposase_id=site.get('transposase_id')), sort_keys=True, separators=(',', ':'))
    return sha256(value.encode()).hexdigest()


def attach_manifest(bags: dict, path: Path | None) -> None:
    """Attach a sequence-bound sidecar keyed by bag ID and original site index."""
    if path is None:
        return
    seen = set()
    with path.open(encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row['bag_id'], int(row['site_idx'])
            if key in seen:
                raise ValueError(f'duplicate evidence key {key}, line {line_no}')
            seen.add(key)
            bid, idx = key
            if bid not in bags:
                continue
            if idx < 0 or idx >= len(bags[bid]):
                raise ValueError(f'evidence site index out of range: {key}')
            site = bags[bid][idx]
            if row['input_sha256'] != input_digest(site):
                raise ValueError(f'evidence/input sequence mismatch: {key}')
            site['scorer_evidence'] = row['evidence']


@dataclass
class CoordinateReport:
    status: str
    site_errors: dict[int, str]
    rejected_hits: dict[str, int]
    n_hits_mapped: int


def _prepare(site: dict, allow_conditional_null: bool = False) -> dict:
    e = site.get('scorer_evidence')
    if not e:
        raise ValueError('missing_evidence')
    occupied = e['occupied'].upper()
    anchor = e['anchor']
    strand = anchor['strand']
    if strand not in ('+', '-') or not anchor['family']:
        raise ValueError('invalid_anchor')
    alo, ahi = int(anchor['occupied_lo']), int(anchor['occupied_hi'])
    if not 0 <= alo < ahi <= len(occupied):
        raise ValueError('anchor_out_of_bounds')
    aligned = e['oriented_context_msa'].upper()
    context_lo = int(e['context_lo'])
    oriented = occupied if strand == '+' else revcomp(occupied)
    ungapped = aligned.replace('-', '')
    if not ungapped or context_lo < 0 or oriented[context_lo:context_lo + len(ungapped)] != ungapped:
        raise ValueError('msa_context_sequence_mismatch')
    oriented_anchor_lo = alo if strand == '+' else len(occupied) - ahi
    oriented_anchor_hi = ahi if strand == '+' else len(occupied) - alo
    if not context_lo <= oriented_anchor_lo < oriented_anchor_hi <= context_lo + len(ungapped):
        raise ValueError('msa_context_excludes_anchor')
    alignment_id = e['alignment_id']
    target = e['target']
    if not alignment_id or not target['frame'] or target['strand'] not in ('+', '-'):
        raise ValueError('invalid_coordinate_frame')
    if target.get('origin_status') != 'verified':
        raise ValueError('target_origin_unverified')
    empty = e['empty'].upper()
    tlo, thi = int(target['empty_lo']), int(target['empty_hi'])
    source_strand, frame_strand = target['source_strand'], target['frame_strand']
    if source_strand not in ('+', '-') or frame_strand not in ('+', '-') or not 0 <= tlo < thi <= len(empty):
        raise ValueError('invalid_target_source_interval')
    raw_flank = site['inputs']['flank'].upper()
    if e.get('conditional_null'):
        if not allow_conditional_null:
            raise ValueError('conditional_null_evidence_is_not_real_evidence')
        raw_flank = e['null_original_flank'].upper()
    source = empty[tlo:thi]
    if source_strand == '-':
        source = revcomp(source)
    if source != raw_flank or len(raw_flank) != len(site['inputs']['flank']):
        raise ValueError('target_empty_source_sequence_mismatch')
    frame_direction = 1 if frame_strand == '+' else -1
    source_first = tlo if source_strand == '+' else thi - 1
    origin = (source_first - int(target['frame_anchor_empty_pos'])) * frame_direction
    direction = frame_direction * (1 if source_strand == '+' else -1)
    if int(target['origin']) != origin or target['strand'] != ('+' if direction == 1 else '-'):
        raise ValueError('target_canonical_origin_mismatch')
    # A declaration of stable boundaries is backed by the actual intervals.
    intervals = [tuple(map(int, v)) for v in e['valid_insert_intervals']]
    if not intervals or any(not 0 <= lo < hi <= len(occupied) for lo, hi in intervals):
        raise ValueError('missing_or_invalid_insert_boundaries')
    fragments = site['inputs']['noncoding_regions']
    sources = map_fragments_to_occupied(fragments, occupied)
    return dict(e=e, occupied=occupied, strand=strand, sources=sources,
                columns=[j for j, b in enumerate(aligned) if b != '-'],
                context_lo=context_lo, intervals=intervals,
                frame=json.dumps([alignment_id, anchor['family'], target['frame']], separators=(',', ':')))


def project_bag(bag: BagRetention, sites: list[dict], allow_conditional_null: bool = False) -> tuple[BagRetention, CoordinateReport]:
    """Keep only unique, full, boundary-stable nc windows in a shared MSA frame.

    Input-adequacy K_valid stays frozen. A bag with incomplete evidence or
    unresolved fragment sources is explicitly unsuitable for negative calls
    or null calibration. Valid mapped hits are still available for review.
    """
    retained, site_errors, rejected = [], {}, Counter()
    prepared = {}
    for idx in bag.valid_site_idx:
        try:
            prepared[idx] = _prepare(sites[idx], allow_conditional_null)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            site_errors[idx] = str(exc)
    lengths = {}
    for idx, p in prepared.items():
        frame = p['e']['alignment_id']
        ncols = len(p['e']['oriented_context_msa'])
        if frame in lengths and lengths[frame] != ncols:
            raise ValueError('same alignment_id has inconsistent MSA lengths')
        lengths[frame] = ncols
        if any(s.status != 'unique' for s in p['sources']):
            site_errors[idx] = 'unresolved_fragment_sources'
    for sr in bag.retained:
        p = prepared.get(sr.site_idx)
        if p is None:
            rejected['missing_site_evidence'] += sum(map(len, sr.hits_by_nc_pos_L.values()))
            continue
        site = sites[sr.site_idx]
        target = p['e']['target']
        mapped = {}
        for hits in sr.hits_by_nc_pos_L.values():
            for hit in hits:
                proj = project_interval(site['inputs']['noncoding_regions'], p['sources'],
                                        hit.nc_pos, hit.nc_pos + hit.L, len(p['occupied']))
                if proj.status != 'ok':
                    rejected[proj.status] += 1
                    continue
                if not all(lo <= proj.occupied_lo and proj.occupied_hi <= hi for lo, hi in p['intervals']):
                    rejected['boundary_unstable'] += 1
                    continue
                lo, hi = proj.occupied_lo, proj.occupied_hi
                if p['strand'] == '-':
                    lo, hi = len(p['occupied']) - hi, len(p['occupied']) - lo
                offset = lo - p['context_lo']
                if offset < 0 or offset + hit.L > len(p['columns']):
                    rejected['outside_msa_context'] += 1
                    continue
                columns = tuple(p['columns'][offset:offset + hit.L])
                direction = 1 if target['strand'] == '+' else -1
                first = int(target['origin']) + direction * hit.flank_pos
                last = int(target['origin']) + direction * (hit.flank_pos + hit.L - 1)
                fp = min(first, last)
                guide_reversed = (proj.strand != p['strand'])
                target_reversed = target['strand'] == '-'
                orient_rc = (hit.orient == 'rc') ^ guide_reversed ^ target_reversed
                mh = replace(hit, nc_pos=columns[0], flank_pos=fp,
                             orient='rc' if orient_rc else 'fwd',
                             raw_nc_pos=hit.nc_pos, raw_flank_pos=hit.flank_pos,
                             raw_orient=hit.orient, coordinate_frame=p['frame'],
                             nc_columns=columns, target_strand=target['strand'])
                # Full column path and relative orientation are part of the key.
                key = (columns[0], hit.L, p['frame'], columns, mh.orient)
                mapped.setdefault(key, []).append(mh)
        for hs in mapped.values():
            hs.sort(key=lambda h: (-h.m, h.flank_pos))
        retained.append(SiteRetention(sr.site_idx, mapped))
    count = sum(len(hs) for sr in retained for hs in sr.hits_by_nc_pos_L.values())
    report = CoordinateReport('ready' if not site_errors else 'needs_evidence', site_errors, dict(rejected), count)
    return BagRetention(bag.K_valid, bag.valid_site_idx, retained, bag.dropped), report
