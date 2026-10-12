"""Separate family annotations from experimentally validated pairing controls."""
import argparse
import csv
import json
import sys
import hashlib
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.motif_gates import evaluate_motif,GateConfig
from scripts.v5a_framework.multi_hit_retention import m_floor_retain_default
from preprocess.alignment import revcomp
from scripts.build_v3_corpus_evidence import read_sources, all_boundaries, unique_source
from scripts.nc_coord_mapping import map_fragments_to_occupied
from Bio import SeqIO


def inspect_bag_context(inputs_path, audit_path, provenance_root, annotation_root, outdir, bid='CDS00730'):
    sites=[json.loads(l) for l in inputs_path.read_text().splitlines() if json.loads(l)['transposase_id']==bid]
    audit={(r['bag_id'],r['site_id']):r for r in read(audit_path) if r['bag_id']==bid}
    sources=read_sources(provenance_root,audit)
    events={r['db_id']:r for r in read(annotation_root/'source_events.tsv') if r['bag_id']==bid}
    wanted={events[s['site_id'].removesuffix('.site')]['insert_id'] for s in sites}
    annotations={r['insert_id']:r for r in read(annotation_root/'insert_family_annotations.tsv') if r['insert_id'] in wanted}
    genes={i:[] for i in wanted}
    for r in read(annotation_root/'gene_origins.tsv'):
        if r['insert_id'] in wanted: genes[r['insert_id']].append(r)
    contexts=[]
    with (outdir/(bid+'_actual_context.fna')).open('w',encoding='utf-8') as fasta:
        for index,site in enumerate(sites):
            row,alleles=sources[audit[(bid,site['site_id'])]['locus_id_full']]
            occupied,empty=alleles[row['longest_allele']],alleles[row['shortest_allele']]
            iid=events[site['site_id'].removesuffix('.site')]['insert_id']
            ilen=int(row['inserted_len'])
            matching=[p for p in range(len(occupied)-ilen+1) if hashlib.md5(occupied[p:p+ilen].encode()).hexdigest()[:12]==row['inserted_md5']]
            if not matching: raise ValueError('source interval cannot reproduce hash: '+site['site_id'])
            start=min(matching);insert=occupied[start:start+ilen]
            anchor=next(g for g in genes[iid] if g['protein_id']==annotations[iid]['dominant_protein_id'])
            strand='+' if anchor['strand']=='1' else '-'
            canonical=insert if strand=='+' else revcomp(insert)
            error=''
            try:bounds,_=all_boundaries(occupied,empty,row)
            except ValueError as exc:bounds=[];error=str(exc)
            tlo,thi,tstrand=unique_source(site['inputs']['flank'],empty)
            adjacent=[]
            for lo,hi in bounds:
                if strand=='+': four,five=empty[max(0,lo-4):lo],empty[max(0,lo-5):lo]
                else: four,five=revcomp(empty[lo:lo+4]),revcomp(empty[lo:lo+5])
                adjacent.append(dict(empty_cutpoint=lo,canonical_adjacent_4mer=four,canonical_adjacent_5mer=five))
            mapped=map_fragments_to_occupied(site['inputs']['noncoding_regions'],occupied)
            contexts.append(dict(bag_id=bid,site_id=site['site_id'],site_idx=index,
                                 actual_source_family=annotations[iid]['label'],actual_source_dominant_role=annotations[iid]['dominant_role'],
                                 source_ORF_strand=strand,source_ORF_lo=anchor['lo'],source_ORF_hi=anchor['hi'],
                                 display_insert_start=start,display_insert_end=start+ilen,
                                 display_cutpoint_status='leftmost_actual_hash_matching_interval_not_biological_boundary_truth',
                                 n_best_boundaries=len(bounds) if bounds else '',boundary_error=error,
                                 target_input_empty_lo=tlo,target_input_empty_hi=thi,target_input_strand=tstrand,
                                 current_nc_fragment_lengths=','.join(str(len(f)) for f in site['inputs']['noncoding_regions']),
                                 current_nc_fragment_origins=json.dumps([asdict(s) for s in mapped]),
                                 canonical_left_end_150=canonical[:150],canonical_right_end_150=canonical[-150:],
                                 adjacent_motifs_json=json.dumps(adjacent),
                                 adjacent_motif_status='junction_adjacent_candidates_not_validated_guide_targets'))
            for name,seq in [('occupied',occupied),('observed_empty',empty),('source_insert',insert),('ORF_oriented_source_insert',canonical)]:
                fasta.write(f'>{bid}|{site["site_id"]}|{name}\n{seq}\n')
    write(outdir/(bid+'_actual_context.tsv'),contexts)


def read(path):
    with path.open(encoding='utf-8',newline='') as handle:
        return list(csv.DictReader(handle,delimiter='\t'))


def write(path, rows):
    with path.open('w',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n')
        writer.writeheader();writer.writerows(rows)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--annotation',type=Path,required=True)
    ap.add_argument('--scan',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--inputs',type=Path)
    ap.add_argument('--audit',type=Path)
    ap.add_argument('--provenance-root',type=Path)
    ap.add_argument('--annotation-root',type=Path)
    args=ap.parse_args();args.outdir.mkdir(parents=True,exist_ok=True)
    signal={r['bag_id']:r for r in read(args.scan)}
    rows=[]
    for r in read(args.annotation):
        if r['family_label'] not in ('IS110','group_II_intron','IS200/IS605-TnpA','TnpB-like'):continue
        s=signal.get(r['bag_id'],{})
        # The source annotation schema supplies no measured guide-target ground truth.
        rows.append(dict(bag_id=r['bag_id'],family=r['family_label'],
                         annotation_status=r['annotation_status'],mechanism_status=r['mechanism_status'],
                         subtype=r['subtype'],control_status='family_comparator_not_validated_pairing_positive',
                         experimental_guide_target_ground_truth='not_supplied_by_annotation_or_scan',
                         signal_input_status=s.get('input_status','not_in_cohort'),
                         scan_status=s.get('coordinate_status','unscored'),
                         raw_score=s.get('raw_score',''),
                         score_interpretation='exploratory_family_association_not_positive_control_sensitivity'))
    write(args.outdir/'family_positive_status.tsv',rows)
    guide,target='AAAG','TTAC'
    direct=sum(a==b for a,b in zip(guide,target))
    reverse=sum(a==b for a,b in zip(guide,revcomp(target)))
    sanity=dict(system='Helicobacter pylori IS608',scope='published_motif_level_sanity_case_not_end_to_end_bag_control',
                reference='https://pmc.ncbi.nlm.nih.gov/articles/PMC5934647/',
                independent_mechanism_reference='https://pmc.ncbi.nlm.nih.gov/articles/PMC3201880/',
                guide_5to3=guide,target_5to3=target,
                direct_identity_matches=direct,reverse_complement_identity_matches=reverse,
                current_retention_floor_if_L4_enabled=m_floor_retain_default(4),
                guide_current_gate=asdict(evaluate_motif(guide,GateConfig())),
                target_current_gate=asdict(evaluate_motif(target,GateConfig())),
                current_search_includes_length=False,
                conclusion='Reducing L alone cannot represent the published guide-target interaction; consecutive index-wise identity is the wrong contact model.',
                structural_constraint='The published system uses subterminal hairpins and ordered canonical/noncanonical contacts; these fourmers are not asserted to be a simple uninterrupted reverse-complement duplex.')
    (args.outdir/'IS608_published_motif_sanity.json').write_text(json.dumps(sanity,indent=2)+'\n',encoding='utf-8')
    summary=dict(n_family_comparators=len(rows),n_marked_not_experimentally_verified=sum(r['mechanism_status']=='not_experimentally_verified' for r in rows),
                 n_experimental_pairing_controls_established_from_these_tables=0,
                 pending='Acquire and verify published full element/RNA/target sequences, strand, endpoints and actual contact maps for IS608/ISDra2, IS621, group-II intron, and a TnpB system before reporting detector sensitivity.',
                 published_motif_sanity=sanity)
    (args.outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    if args.inputs:
        if not all((args.audit,args.provenance_root,args.annotation_root)):
            ap.error('--inputs requires --audit, --provenance-root and --annotation-root')
        inspect_bag_context(args.inputs,args.audit,args.provenance_root,args.annotation_root,args.outdir)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
