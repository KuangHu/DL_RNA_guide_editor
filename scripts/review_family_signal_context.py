"""Human-readable selected-anchor audit; diagnostic, never a family classifier."""
import argparse
import csv
import html
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from Bio import SeqIO
from scripts.build_v3_corpus_evidence import read_sources, all_boundaries, unique_source
from scripts.nc_coord_mapping import map_fragments_to_occupied, project_interval
from preprocess.alignment import revcomp
from scripts.v5a_framework.motif_gates import evaluate_motif, GateConfig
import re


def read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def write(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as handle:
        w = csv.DictWriter(handle, fieldnames=fields, delimiter='\t', lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def terminal_repeat(insert, inverted=False, window=200, seed=10):
    """Longest exact local end-window match >=10; not a biological TIR call."""
    width = min(window, len(insert) // 2)
    left, original_right = insert[:width], insert[-width:]
    right = revcomp(original_right) if inverted else original_right
    index = defaultdict(list)
    for j in range(max(0, len(right)-seed+1)):
        index[right[j:j+seed]].append(j)
    best = None
    for i in range(max(0, len(left)-seed+1)):
        for j in index[left[i:i+seed]]:
            if any(b not in 'ACGT' for b in left[i:i+seed]): continue
            a, b, length = i, j, seed
            while a and b and left[a-1] == right[b-1] and left[a-1] in 'ACGT':
                a -= 1; b -= 1; length += 1
            while a+length < len(left) and b+length < len(right) and left[a+length] == right[b+length] and left[a+length] in 'ACGT':
                length += 1
            rlo = len(insert)-b-length if inverted else len(insert)-width+b
            result = dict(length=length, left_lo=a, left_hi=a+length,
                          right_lo=rlo, right_hi=rlo+length,
                          sequence=left[a:a+length])
            if best is None or length > best['length']: best = result
    return best


def junction_direct(occupied, lo, hi):
    lengths = [k for k in range(1, min(120, lo, len(occupied)-hi)+1)
               if occupied[lo-k:lo] == occupied[hi:hi+k]
               and all(b in 'ACGT' for b in occupied[lo-k:lo])]
    return max(lengths, default=0)


def overlaps(a, b, c, d): return max(0, min(b, d)-max(a, c))


def rna_coverage(sites, features, annotation_root):
    """Source RNA-model intervals covered by actual input fragments; no EBS claim."""
    chosen=[s for ss in sites.values() for s in ss if features[s['transposase_id']]['family_label']=='group_II_intron']
    events={r['db_id']:r for r in read(annotation_root/'source_events.tsv')}
    def insert_id(site): return events[re.sub(r'\.(?:site|copy\d+)$','',site['site_id'])]['insert_id']
    wanted={insert_id(s) for s in chosen}
    sequences={r.id:str(r.seq) for r in SeqIO.parse(annotation_root/'insert_sequences.fna','fasta') if r.id in wanted}
    hits=defaultdict(list)
    for h in read(annotation_root/'rna_hits.tsv'):
        if h['insert_id'] in wanted: hits[h['insert_id']].append(h)
    rows=[]
    for site in chosen:
        iid=insert_id(site)
        sources=map_fragments_to_occupied(site['inputs']['noncoding_regions'],sequences[iid])
        mapped=[r for r in sources if r.status=='unique']
        for h in hits[iid]:
            lo=min(int(h['seq_from']),int(h['seq_to']))-1;hi=max(int(h['seq_from']),int(h['seq_to']))
            covered={i for r in mapped for i in range(max(lo,r.occupied_lo),min(hi,r.occupied_hi))}
            rows.append(dict(bag_id=site['transposase_id'],site_id=site['site_id'],source_insert_id=iid,
                             model=h['model'],source_model_lo=lo,source_model_hi=hi,source_model_strand=h['strand'],
                             bp_model_region=hi-lo,bp_in_current_inputs=len(covered),fraction_in_current_inputs=len(covered)/(hi-lo),
                             n_current_fragments=len(site['inputs']['noncoding_regions']),
                             current_frag_lengths=','.join(str(len(f)) for f in site['inputs']['noncoding_regions']),
                             mapping_statuses=','.join(r.status for r in sources),copy_nc_is_inherited='.copy' in site['site_id']))
    return rows


def track(label, length, spans):
    width, x0 = 850, 165
    items = [f'<text x="0" y="24">{html.escape(label)} ({length} bp)</text>',
             f'<line x1="{x0}" y1="20" x2="{x0+width}" y2="20" stroke="#475569"/>']
    for lo, hi, color, name in spans:
        if 0 <= lo < hi <= length:
            x = x0+width*lo/max(length, 1); w = max(2, width*(hi-lo)/max(length, 1))
            items.append(f'<rect x="{x:.2f}" y="12" width="{w:.2f}" height="16" fill="{color}"><title>{html.escape(name)} [{lo},{hi})</title></rect>')
    items += [f'<text x="{x0}" y="48">0</text>', f'<text x="{x0+width-45}" y="48">{length}</text>']
    return '<svg viewBox="0 0 1040 55">'+''.join(items)+'</svg>'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('scan', 'audit', 'provenance-root', 'copy-bundle', 'outdir'):
        ap.add_argument('--'+name, type=Path, required=True)
    ap.add_argument('--annotation-root',type=Path)
    args = ap.parse_args(); args.outdir.mkdir(parents=True, exist_ok=True)
    features = {r['bag_id']: r for r in read(args.scan/'family_signal_by_bag.tsv')}
    aligns = read(args.scan/'family_best_alignments.tsv')
    selected = set()
    selections = {}
    for family in dict.fromkeys(r['family_label'] for r in features.values()):
        members = sorted([r for r in features.values() if r['family_label']==family and r['input_status']=='scannable'], key=lambda r: float(r['exact_dinuc_delta']))
        for role, row in [('lowest excess', members[0]), ('highest excess', members[-1]), ('highest raw', max(members, key=lambda r: float(r['raw_score'])))]:
            selected.add(row['bag_id']); selections.setdefault(row['bag_id'], []).append(role)
    sites = defaultdict(list); all_sites = defaultdict(list)
    for line in (args.scan/'cohort_inputs.jsonl').read_text().splitlines():
        row = json.loads(line)
        all_sites[row['transposase_id']].append(row)
        if row['transposase_id'] in selected: sites[row['transposase_id']].append(row)
    audit = {(r['bag_id'], r['site_id']): r for r in read(args.audit) if r['bag_id'] in selected and '.copy' not in r['site_id']}
    sources = read_sources(args.provenance_root, audit)
    copy_rows = {(r['bag_id'], r['site_id']): r for r in read(args.copy_bundle/'copy_carrier_resolved.tsv')}
    quarantined = {(r['bag_id'],r['site_id']) for r in read(args.copy_bundle.parent/'copy_carrier_bundle_audit/join_issues.tsv')}
    copy_slices = {tuple(r.id.split('|')[:2]): str(r.seq).upper() for r in SeqIO.parse(args.copy_bundle/'copy_carrier_slices.fna', 'fasta')}
    site_rows, display = [], {}
    for alignment in aligns:
        bid, sid = alignment['bag_id'], alignment['site_id']
        if bid not in selected: continue
        site = sites[bid][int(alignment['site_idx'])]
        p, L, fp = int(alignment['nc_pos']), int(alignment['core_L']), int(alignment['flank_pos'])
        key = (bid, sid)
        result = dict(bag_id=bid, family=features[bid]['family_label'], site_id=sid,
                      site_idx=alignment['site_idx'], nc_pos=p, L=L, flank_pos=fp,
                      orientation=alignment['orientation'], match_count=alignment['match_count'],
                      nc_source=alignment['nc_source'], extraction_midpoint_distance=fp+L/2-60,
                      crosses_extraction_midpoint=fp<60<fp+L,
                      evidence_status='unknown')
        occupied = empty = None; row = None; boundaries = None; lo = hi = None
        try:
            if '.copy' in sid:
                if key in quarantined: raise ValueError('quarantined_copy_join')
                if key not in copy_slices: raise ValueError('actual_copy_slice_missing')
                occupied = copy_slices[key]; row = copy_rows[key]
                lo = int(row['elem_start_0b'])-int(row['slice_lo_0b'])
                hi = int(row['elem_end_0b_exclusive'])-int(row['slice_lo_0b'])
                result['evidence_status'] = 'actual_copy_slice_boundary_is_aligner_interval_only'
            else:
                row, alleles = sources[audit[key]['locus_id_full']]
                occupied, empty = alleles[row['longest_allele']], alleles[row['shortest_allele']]
                result['evidence_status'] = 'observed_alleles_stored_boundary_not_verified'
                try:
                    boundaries, _ = all_boundaries(occupied, empty, row)
                    result.update(boundary_count=len(boundaries), boundary_min=min(b[0] for b in boundaries), boundary_max=max(b[0] for b in boundaries))
                    result['evidence_status'] = 'observed_alleles_complete_best_boundary_set'
                except ValueError as exc: result['boundary_error'] = str(exc)
                if row.get('insert_start_in_long_bp', '').isdigit():
                    lo = int(row['insert_start_in_long_bp'])
                    result['display_boundary_source'] = 'stored_cutpoint'
                else:
                    ilen = int(row['inserted_len'])
                    candidates = [a for a,b in boundaries] if boundaries else range(len(occupied)-ilen+1)
                    matches = [a for a in candidates if hashlib.md5(occupied[a:a+ilen].encode()).hexdigest()[:12] == row['inserted_md5']]
                    if not matches: raise ValueError('no_display_cutpoint_reproduces_actual_source_hash')
                    lo = min(matches)
                    result['display_boundary_source'] = 'leftmost_source_hash_matching_cutpoint_not_biological_truth'
                    result['n_source_hash_matching_cutpoints'] = len(matches)
                hi = lo+int(row['inserted_len'])
            if not 0 <= lo < hi <= len(occupied): raise ValueError('stored_interval_invalid')
            fragments = site['inputs']['noncoding_regions']
            src = map_fragments_to_occupied(fragments, occupied)
            projection = project_interval(fragments, src, p, p+L, len(occupied))
            result.update(nc_projection_status=projection.status, nc_occupied_lo=projection.occupied_lo,
                          nc_occupied_hi=projection.occupied_hi, nc_strand=projection.strand)
            if projection.status == 'ok':
                result.update(nc_offset_from_stored_insert=projection.occupied_lo-lo,
                              nc_distance_to_nearest_stored_end=min(projection.occupied_lo-lo, hi-projection.occupied_hi))
                if boundaries:
                    inside = [a<=projection.occupied_lo and projection.occupied_hi<=b for a,b in boundaries]
                    result['nc_inside_all_best_boundaries'] = all(inside)
                    result['nc_inside_some_best_boundaries'] = any(inside)
            insert = occupied[lo:hi]
            direct, inverted = terminal_repeat(insert), terminal_repeat(insert, True)
            result.update(stored_insert_len=len(insert), end_window_direct_repeat_bp=direct['length'] if direct else 0,
                          end_window_inverted_repeat_bp=inverted['length'] if inverted else 0,
                          exterior_junction_direct_bp=junction_direct(occupied,lo,hi))
            for name, repeat in [('direct',direct), ('inverted',inverted)]:
                if repeat and projection.status=='ok':
                    a,b=projection.occupied_lo-lo, projection.occupied_hi-lo
                    result['nc_overlap_'+name+'_end_repeat_bp']=sum(overlaps(a,b,repeat[c+'_lo'],repeat[c+'_hi']) for c in ('left','right'))
            if empty:
                try:
                    tlo,thi,tstrand=unique_source(site['inputs']['flank'],empty)
                    elo,ehi=(tlo+fp,tlo+fp+L) if tstrand=='+' else (thi-fp-L,thi-fp)
                    result.update(target_empty_lo=elo,target_empty_hi=ehi, target_source_strand=tstrand)
                except ValueError as exc: result['target_source_error']=str(exc)
            display[key] = dict(occupied=occupied, empty=empty, insert=insert, lo=lo, hi=hi,
                                projection=asdict(projection), boundaries=boundaries,
                                direct=direct,inverted=inverted)
        except (ValueError, KeyError) as exc: result['evidence_error']=str(exc)
        site_rows.append(result)
    write(args.outdir/'selected_site_context.tsv',site_rows)
    bag_rows = []
    for bid in sorted(selected):
        f=features[bid]; aa=[a for a in aligns if a['bag_id']==bid]; ss=[s for s in site_rows if s['bag_id']==bid]
        bag_rows.append(dict(bag_id=bid,family=f['family_label'], selection=';'.join(selections[bid]),
                             raw_score=f['raw_score'],exact_dinuc_delta=f['exact_dinuc_delta'],
                             nc_pos=f['nc_pos'],L=f['L'],flank_pos=f['flank_pos_median'],K=f['K'],
                             n_support=len(aa), n_orientations=len({a['orientation'] for a in aa}),
                             n_exact_core=sum(bool(a['total_exact_contiguous_bp']) for a in aa),
                             n_projectable=sum(s.get('nc_projection_status')=='ok' for s in ss),
                             n_unstable_boundary=sum(s.get('nc_inside_all_best_boundaries') is False for s in ss),
                             n_nc_overlaps_direct=sum(s.get('nc_overlap_direct_end_repeat_bp',0)>0 for s in ss),
                             n_nc_overlaps_inverted=sum(s.get('nc_overlap_inverted_end_repeat_bp',0)>0 for s in ss)))
    write(args.outdir/'selected_bag_review.tsv',bag_rows)
    scored=[r for r in features.values() if r['input_status']=='scannable']
    summary=dict(n_scanned=len(scored),best_L_counts=dict(Counter(r['L'] for r in scored)),
                 n_best_support_alignments=len(aligns),n_exact_core_alignments=sum(bool(a['total_exact_contiguous_bp']) for a in aligns),
                 n_bags_mixed_raw_orientations=sum(len({a['orientation'] for a in aligns if a['bag_id']==r['bag_id']})>1 for r in scored),
                 n_selected=len(selected),n_selected_supports=len(site_rows),
                 selected=bag_rows,limits=['Selected high/low examples are a diagnostic case series, not a new performance estimate.',
                 'Mixed raw orientations are not contradictions until biological frames are resolved.',
                 'Terminal repeat matches are exact local diagnostics in chosen first/last 200 bp; absence does not exclude imperfect/longer TIR.',
                 'Stored event boundaries and aligner copy intervals are not biological boundary truth.',
                 'Exterior exact direct repeat is TSD-like evidence, not proof of target duplication.',
                 '13 ineligible bags remain unknown; current scorer is single 9-14 bp, ungapped and raw-coordinate.'])
    summary['short_motif_current_gate_results']={seq:asdict(evaluate_motif(seq,GateConfig())) for seq in ('TTAC','GTAA','TTGAT','ATCAA')}
    summary['short_motif_note']='Published IS608/ISDra2 target examples and reverse complements; gate diagnostics only. These lengths are not scanned in the current architecture.'
    if args.annotation_root:
        coverage=rna_coverage(all_sites,features,args.annotation_root)
        write(args.outdir/'groupII_rna_input_coverage.tsv',coverage)
        summary['rna_coverage_note']='Rfam source-model interval overlap, not verified EBS positions or actual-copy RNA validation.'
    summary['input_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in
                             (args.scan/'family_signal_by_bag.tsv',args.scan/'family_best_alignments.tsv',
                              args.scan/'cohort_inputs.jsonl',args.audit,args.copy_bundle/'copy_carrier_resolved.tsv',
                              args.copy_bundle/'copy_carrier_slices.fna',Path(__file__))}
    (args.outdir/'review_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    blocks=['<h1>Family signal: selected-anchor context review</h1><p>Exploratory raw coordinates. Blue = nc anchor; orange = target; purple = boundary range; green/red = direct/inverted terminal-window repeat. Tracks use independent scales. Sequence matching is identity to a strand-oriented target, not a structure or RNA pairing assay.</p>']
    for b in bag_rows:
        bid=b['bag_id']; blocks.append(f'<h2>{bid} — {html.escape(b["family"])}</h2><p>{html.escape(b["selection"])}; raw={float(b["raw_score"]):.3f}; excess={float(b["exact_dinuc_delta"]):+.3f}; L={b["L"]}; support={b["n_support"]}/{b["K"]}</p>')
        for s in [s for s in site_rows if s['bag_id']==bid]:
            key=(bid,s['site_id']); a=next(a for a in aligns if (a['bag_id'],a['site_id'])==key)
            blocks.append('<details open><summary>'+html.escape(s['site_id'])+f' — {s["match_count"]}/{s["L"]}, raw {s["orientation"]}</summary>')
            blocks.append('<pre>'+html.escape('NC      '+a['guide_context']+'\n        '+a['match_line']+'\nTARGET  '+a['target_context']+'\nCORE    '+' '*int(a['core_start'] or 0)+'^'*int(a['core_L']))+'</pre>')
            L=int(s['L']); fp=int(s['flank_pos'])
            blocks.append(track('Empty flank input',120,[(fp,fp+L,'#ef8a23','target'),(59,61,'#64748b','extraction midpoint, not verified biological junction')]))
            if key in display:
                d=display[key]; n=len(d['insert']); spans=[]; pr=d['projection']
                for name,color in [('direct','#39a96b'),('inverted','#ce4c5b')]:
                    rep=d[name]
                    if rep:
                        spans.extend((rep[t+'_lo'],rep[t+'_hi'],color,name+' end-window local match') for t in ('left','right'))
                if pr['status']=='ok': spans.append((pr['occupied_lo']-d['lo'],pr['occupied_hi']-d['lo'],'#3088d8','nc anchor'))
                blocks.append(track('Stored insert interval',n,spans))
                occsp=[(d['lo'],d['hi'],'#cbd5e1','stored interval')]
                if d['boundaries']:
                    bounds=d['boundaries']
                    for q in (0,1):
                        blo,bhi=min(x[q] for x in bounds),max(x[q] for x in bounds)
                        if blo<bhi:occsp.append((blo,bhi,'#9d78ca','all optimal boundary endpoints range'))
                if pr['status']=='ok':occsp.append((pr['occupied_lo'],pr['occupied_hi'],'#3088d8','nc anchor'))
                blocks.append(track('Actual occupied allele',len(d['occupied']),occsp))
            blocks.append('<pre class="metrics">'+html.escape(json.dumps(s,indent=2))+'</pre></details>')
    document='<!doctype html><meta charset="utf-8"><title>Family signal context review</title><style>body{font:16px system-ui;margin:32px auto;max-width:1200px;color:#17202b}h2{margin-top:48px;border-top:2px solid #ddd;padding-top:24px}summary{font-weight:600}pre{font:14px Consolas,monospace;overflow:auto}svg{width:100%;font:12px system-ui}.metrics{font-size:12px;color:#45556a}details{padding:14px;background:#f8fafc;margin:12px 0}</style>'+''.join(blocks)
    (args.outdir/'human_review.html').write_text(document,encoding='utf-8')
    # Four actual selected anchors, including two terminal-repeat overlaps.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    examples=[('CDS07967','senterica.E000031.site'), ('CDS00712','ecoli.E024252.site'),
              ('CDS00149','abaum.E003897.copy25557'), ('CDS00730','ecoli.E027450.site')]
    fig, axes=plt.subplots(4,1,figsize=(16,13),constrained_layout=True)
    for ax,key in zip(axes,examples):
        s=next(s for s in site_rows if (s['bag_id'],s['site_id'])==key)
        a=next(a for a in aligns if (a['bag_id'],a['site_id'])==key)
        d=display.get(key); f=features[key[0]]
        ax.set_xlim(-28,125);ax.set_ylim(-3.8,4.8);ax.axis('off')
        ax.text(-27,4.1,f'{key[0]} | {s["family"]} | {key[1]}',fontsize=12,fontweight='bold')
        ax.text(-27,3.4,f'Raw {float(f["raw_score"]):.3f}; dinucleotide excess {float(f["exact_dinuc_delta"]):+.3f}; shown site {a["match_count"]}/{a["core_L"]}, {a["orientation"]}',fontsize=10)
        ax.plot([0,120],[2.5,2.5],color='#64748b',lw=2)
        ax.text(-27,2.5,'Target flank 120 bp',va='center',fontsize=9)
        fp=int(a['flank_pos']);L=int(a['core_L'])
        ax.add_patch(Rectangle((fp,2.27),L,.46,color='#ef8a23'))
        ax.plot([60,60],[2.1,2.9],color='#64748b',ls='--')
        ax.text(60,2.95,'extraction midpoint',ha='center',fontsize=8,color='#64748b')
        if d:
            for y,label,seq,spans in [(1.45,'Insert interval',d['insert'],[]),(.35,'Occupied allele',d['occupied'],[])]:
                n=len(seq);ax.plot([0,120],[y,y],color='#64748b',lw=2)
                ax.text(-27,y,label+'\n'+str(n)+' bp',va='center',fontsize=9)
                if y==1.45:
                    for typ,color in [('direct','#39a96b'),('inverted','#ce4c5b')]:
                        rep=d[typ]
                        if rep:
                            for t in ('left','right'):
                                ax.add_patch(Rectangle((120*rep[t+'_lo']/n,y-.19),max(.7,120*rep['length']/n),.38,color=color))
                else:
                    ax.add_patch(Rectangle((120*d['lo']/n,y-.14),120*(d['hi']-d['lo'])/n,.28,color='#cbd5e1'))
                pr=d['projection']
                if pr['status']=='ok':
                    lo=pr['occupied_lo']-(d['lo'] if y==1.45 else 0)
                    hi=pr['occupied_hi']-(d['lo'] if y==1.45 else 0)
                    if 0<=lo<hi<=n:ax.add_patch(Rectangle((120*lo/n,y-.25),max(.9,120*(hi-lo)/n),.5,color='#3088d8'))
        text='NC      '+a['guide_context']+'\n        '+a['match_line']+'\nTARGET  '+a['target_context']+'\nCORE    '+' '*int(a['core_start'] or 0)+'^'*L
        ax.text(-27,-.65,text,fontsize=9.4,family='monospace',va='top',linespacing=1.4)
        note=('No exact >=10 bp end-window repeat detected; nc core stays inside all 3 best boundaries.' if key[0]=='CDS07967'
              else 'NC core overlaps 3 bp of a 13 bp inverted end-window match; boundary remains ambiguous.' if key[0]=='CDS00712'
              else 'NC core overlaps 7 bp of an 11 bp inverted end-window match; actual copy DNA shown, aligner boundaries only.' if key[0]=='CDS00149'
              else 'The selected 9 bp anchor is approximate; it does not test the family\'s short DNA-guide mechanism.')
        ax.text(-27,-3.35,note,fontsize=9,color='#45556a')
    fig.suptitle('Human review: actual sequences and selected anchors\nBlue: nc core | Orange: target core | Red/green: inverted/direct end-window matches | Each track has its own scale',fontsize=13)
    fig.savefig(args.outdir/'selected_alignment_architecture.png',dpi=160)
    fig.savefig(args.outdir/'selected_alignment_architecture.svg')
    plt.close(fig)
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
