"""Conservative, score-independent bag annotation from protein/RNA evidence.

Labels mean computational sequence support, not measured mechanism or accuracy.
All ORF roles, competing families, partial genes and unclassified members remain
visible. The clean-control flag excludes any supported guided-family co-encoding.
"""
import argparse
import csv
import json
import re
from collections import Counter,defaultdict
from pathlib import Path


def read(path):
    with path.open(encoding='utf8',newline='') as fh:return list(csv.DictReader(fh,delimiter='\t'))


def write(path,rows,fields):
    with path.open('w',encoding='utf8',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,delimiter='\t',lineterminator='\n',extrasaction='ignore');w.writeheader();w.writerows(rows)


def family(model):
    return 'IS200/IS605' if model.startswith('IS200_IS605_') else model.split('_')[0]


def overlap(a,b):
    lo=max(int(a['ali_from']),int(b['ali_from']));hi=min(int(a['ali_to']),int(b['ali_to']))
    return max(0,hi-lo+1)/min(int(a['ali_to'])-int(a['ali_from'])+1,int(b['ali_to'])-int(b['ali_from'])+1)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--bags-tsv',type=Path,required=True)
    a=ap.parse_args();out=a.inputs
    byprotein=defaultdict(list)
    for r in read(out/'profile_hits.tsv'):byprotein[r['protein_id']].append(r)
    refmeta={r['Entry']:r for r in read(a.models/'uniprot_reviewed_references.tsv')}
    refs=defaultdict(list)
    for line in (out/'reviewed_reference_hits.tsv').read_text().splitlines():
        p=line.split('\t');acc=p[1].split('|')[1] if '|' in p[1] else p[1]
        refs[p[0]].append(dict(accession=acc,name=refmeta.get(acc,{}).get('Protein names',''),
                            identity=float(p[2])/100,bits=float(p[9]),evalue=float(p[8]),
                            qcov=(int(p[5])-int(p[4])+1)/int(p[10]),scov=(int(p[7])-int(p[6])+1)/int(p[11])))
    protein_annotations={}
    for pid, hits in byprotein.items():
        pfam={}
        for h in hits:
            acc=h['accession'].split('.')[0]
            # Known full-length LtrA aligns 68% of these two broad models.
            # RNA D5 is still required for the final intron element label.
            min_coverage=.6 if acc in ['PF00078','PF01348'] else .7
            if h['library']=='pfam' and float(h['hmm_coverage'])>=min_coverage:
                if acc not in pfam or float(h['domain_score'])>float(pfam[acc]['domain_score']):pfam[acc]=h
        ises=[h for h in hits if h['library']=='isescan' and float(h['hmm_coverage'])>=.7]
        ises.sort(key=lambda h:float(h['domain_score']),reverse=True)
        top=ises[0] if ises else None
        competing=[h for h in ises[1:] if family(h['model'])!=family(top['model']) and overlap(h,top)>=.5] if top else []
        margin=float(top['domain_score'])-max((float(h['domain_score']) for h in competing),default=0) if top else 0
        f=family(top['model']) if top and margin>=20 else ''
        role='unresolved';reason=[]
        qualified_refs=sorted((r for r in refs[pid] if r['qcov']>=.8 and r['scov']>=.7 and r['identity']>=.4),key=lambda r:r['bits'],reverse=True)
        closest=qualified_refs[0] if qualified_refs else None
        if {'PF01548','PF02371'}<=pfam.keys() and int(pfam['PF01548']['ali_to'])<int(pfam['PF02371']['ali_from']) and f=='IS110':
            role='IS110_transposase';reason=['ordered_DEDD_and_Transposase20_GA','competitive_IS110_profile']
        elif 'PF01797' in pfam:
            rayt=[r for r in qualified_refs if 'REP-associated' in r['name']]
            known_tnpa=[r for r in qualified_refs if re.search(r'IS200|IS605|IS608|ISDra2',r['name'])]
            if rayt and rayt[0]['bits']-max((r['bits'] for r in known_tnpa),default=0)>=20:
                role='RAYT_supported';reason=['Y1_GA','reviewed_RAYT_reference_preferred']
            elif f=='IS200/IS605' and int(top['model_length'])<220:
                role='IS200_IS605_TnpA_supported';reason=['Y1_GA','competitive_short_IS200_IS605_profile']
            else:
                role='Y1_TnpA_RAYT_ambiguous';reason=['Y1_GA_without_unambiguous_family']
        elif {'PF01385','PF12323'}<=pfam.keys() and int(pfam['PF12323']['ali_to'])<int(pfam['PF01385']['ali_from']):
            role='TnpB_like';reason=['ordered_HTH_OrfB_and_OrfB_GA']
            if closest and ('CRISPR' in closest['name'] or 'IscB' in closest['name']):
                role='RuvC_effector_ambiguous';reason.append('reviewed_reference_competitor')
        elif 'PF00078' in pfam and ('PF08388' in pfam or 'PF01348' in pfam):
            x=pfam.get('PF08388') or pfam['PF01348']
            if int(pfam['PF00078']['ali_to'])<int(x['ali_from']):
                role='groupII_IEP_supported';reason=['ordered_RT_and_maturase_GA']
        elif f=='IS1' and 'PF03400' in pfam:
            role='IS1_supported';reason=['DDE_Tnp_IS1_GA','competitive_IS1_profile']
        elif f=='IS3' and 'PF00665' in pfam:
            role='IS3_supported';reason=['rve_GA','competitive_IS3_profile']
        elif f=='IS5' and 'PF01609' in pfam:
            role='IS5_supported';reason=['DDE_Tnp1_GA','competitive_IS5_profile']
        elif 'PF00078' in pfam:
            role='RT_unclassified';reason=['RT_alone_not_groupII']
        potential_guided_domain=any(h['library']=='pfam' and h['accession'].split('.')[0] in
            {'PF01548','PF02371','PF01797','PF01385','PF12323','PF00078','PF08388','PF01348'} for h in hits)
        protein_annotations[pid]=dict(protein_id=pid,role=role,reason=';'.join(reason),
            potential_guided_domain=potential_guided_domain,
            pfam_accessions=';'.join(sorted(pfam)),best_isescan_model=top['model'] if top else '',
            best_isescan_family=family(top['model']) if top else '',isescan_margin_bits=margin,
            closest_reviewed_accession=closest['accession'] if closest else '',
            closest_reviewed_name=closest['name'] if closest else '',
            closest_reviewed_identity=closest['identity'] if closest else '',
            closest_reviewed_qcov=closest['qcov'] if closest else '',
            closest_reviewed_scov=closest['scov'] if closest else '')
    protein_fields=list(next(iter(protein_annotations.values())))
    for line in (out/'proteins.faa').read_text().splitlines():
        if not line.startswith('>'):continue
        pid=line[1:].split()[0]
        if pid not in protein_annotations:
            row={k:'' for k in protein_fields}
            row.update(protein_id=pid,role='unresolved',reason='no_reported_profile_hit',isescan_margin_bits=0,potential_guided_domain=False)
            protein_annotations[pid]=row
    write(out/'protein_role_annotations.tsv',list(protein_annotations.values()),protein_fields)
    genes=defaultdict(list)
    for g in read(out/'gene_origins.tsv'):genes[g['insert_id']].append(g)
    events=read(out/'source_events.tsv');lengths={r['insert_id']:int(r['length']) for r in events}
    rnas=defaultdict(list)
    for r in read(out/'rna_hits.tsv'):rnas[r['insert_id']].append(r)
    inserts={}
    for iid, gs in genes.items():
        # Match upstream 111_cds_cluster.py: longest nucleotide ORF span.
        # Translated proteins omit terminal '*', which changes aa tie order.
        gs.sort(key=lambda g:int(g['hi'])-int(g['lo']),reverse=True)
        all_roles=[];complete_roles=[];iep_d5=False;iep_d1=False
        for g in gs:
            role=protein_annotations.get(g['protein_id'],{}).get('role','unresolved');all_roles.append(role)
            complete=g['partial_begin']=='False' and g['partial_end']=='False'
            if complete:complete_roles.append(role)
            if role=='groupII_IEP_supported' and complete:
                minus=g['strand']=='-1';n=lengths[iid]
                gl=n-int(g['hi']) if minus else int(g['lo']);gh=n-int(g['lo']) if minus else int(g['hi'])
                for r in rnas[iid]:
                    if r['strand']!=('-' if minus else '+') or r['truncation']!='no':continue
                    lo=min(int(r['seq_from']),int(r['seq_to']))-1;hi=max(int(r['seq_from']),int(r['seq_to']))
                    rl=n-hi if minus else lo;rh=n-lo if minus else hi
                    if r['rfam_accession']=='RF00029' and rl>=gh:iep_d5=True
                    elif r['rfam_accession']!='RF00029' and rh<=gl:iep_d1=True
        dominant=protein_annotations.get(gs[0]['protein_id'],{}).get('role','unresolved')
        partial=gs[0]['partial_begin']=='True' or gs[0]['partial_end']=='True'
        label='UNKNOWN';note=[]
        if partial:note.append('dominant_ORF_partial')
        elif dominant=='IS110_transposase':label='IS110'
        elif dominant=='IS200_IS605_TnpA_supported':label='IS200/IS605-TnpA'
        elif dominant=='TnpB_like':label='TnpB-like'
        elif dominant=='groupII_IEP_supported':
            label='group_II_intron' if iep_d5 else 'group_II_IEP_candidate'
            note.append('RNA_D5_same_strand_downstream' if iep_d5 else 'RNA_D5_not_verified')
            if iep_d1:note.append('RNA_D1D4_same_strand_upstream')
        elif dominant in ['IS1_supported','IS3_supported','IS5_supported']:label=dominant.split('_')[0]
        elif dominant=='RAYT_supported':label='RAYT'
        elif dominant in ['Y1_TnpA_RAYT_ambiguous','RT_unclassified']:note.append(dominant)
        confounding_roles={'IS110_transposase','IS200_IS605_TnpA_supported','Y1_TnpA_RAYT_ambiguous','TnpB_like','groupII_IEP_supported','RuvC_effector_ambiguous'}
        # Partial / single-domain hits cannot confirm a guided family, but
        # they are sufficient to keep an example out of the clean controls.
        possible_guided=any(protein_annotations[g['protein_id']]['potential_guided_domain'] for g in gs)
        confound=bool(confounding_roles.intersection(all_roles) or rnas[iid] or possible_guided)
        clean_control=label in ['IS1','IS3','IS5'] and not confound
        inserts[iid]=dict(insert_id=iid,length=lengths[iid],dominant_protein_id=gs[0]['protein_id'],dominant_role=dominant,
                         dominant_partial=partial,label=label,all_roles=';'.join(sorted(set(all_roles))),
                         clean_family_control=clean_control,potential_guided_domain=possible_guided,iep_with_downstream_D5=iep_d5,iep_with_upstream_D1D4=iep_d1,
                         rna_models=';'.join(sorted({r['rfam_accession'] for r in rnas[iid]})),notes=';'.join(note))
    write(out/'insert_family_annotations.tsv',list(inserts.values()),list(next(iter(inserts.values()))))
    bybag=defaultdict(set)
    for e in events:bybag[e['bag_id']].add(e['insert_id'])
    bags=[]
    for b in read(a.bags_tsv):
        bid=b['bag_id'];members=[inserts[i] for i in bybag[bid]];counts=Counter(r['label'] for r in members)
        non_unknown=Counter({k:v for k,v in counts.items() if k!='UNKNOWN'})
        supported=Counter({k:v for k,v in non_unknown.items() if k!='group_II_IEP_candidate'})
        candidate,count=supported.most_common(1)[0] if supported else (non_unknown.most_common(1)[0] if non_unknown else ('UNKNOWN',0))
        fraction=count/len(members)
        competing=[k for k in non_unknown if k!=candidate and not
                   (candidate=='group_II_intron' and k=='group_II_IEP_candidate')]
        confident=candidate not in ['UNKNOWN','group_II_IEP_candidate'] and fraction>=.9 and not competing
        label=candidate if confident else 'UNKNOWN'
        clean=confident and label in ['IS1','IS3','IS5'] and all(r['clean_family_control'] for r in members)
        review='supported_consistent_sequence_annotation' if confident else ('mixed_or_conflicting_members' if competing else 'incomplete_or_insufficient_evidence')
        profile_counts=Counter(protein_annotations.get(r['dominant_protein_id'],{}).get('best_isescan_family','')
                               for r in members)
        profile_counts.pop('',None)
        bags.append(dict(bag_id=bid,family_label=label,leading_candidate=candidate,annotation_status=review,
                     n_unique_insert_sequences=len(members),n_member_labels=count,label_member_fraction=fraction,
                     member_label_counts=json.dumps(counts,sort_keys=True),
                     dominant_profile_family_counts=json.dumps(profile_counts,sort_keys=True),
                     coencoded_roles=';'.join(sorted({role for r in members for role in r['all_roles'].split(';') if role!='unresolved'})),
                     clean_family_control=clean,n_inserts_IEP_D5=sum(r['iep_with_downstream_D5'] for r in members),
                     n_inserts_IEP_D1D4=sum(r['iep_with_upstream_D1D4'] for r in members),
                     n_inserts_dominant_partial=sum(r['dominant_partial'] for r in members),
                     mechanism_status='not_experimentally_verified',subtype='IS200_vs_IS605_unresolved' if candidate=='IS200/IS605-TnpA' else '',
                     actual_copy_sequence_status='not_validated_by_source_event_annotation'))
    bags.sort(key=lambda r:(r['family_label'],r['bag_id']))
    write(out/'bag_family_annotations.tsv',bags,list(bags[0]))
    selected=[r for r in bags if r['family_label'] in ['IS110','group_II_intron','IS200/IS605-TnpA','TnpB-like'] or r['clean_family_control']]
    write(out/'supported_target_and_control_bags.tsv',selected,list(bags[0]))
    needs_review=[r for r in bags if r['family_label']=='UNKNOWN' and (
        r['leading_candidate']!='UNKNOWN' or
        {'IS110','IS200/IS605','IS1','IS3','IS5'}.intersection(json.loads(r['dominant_profile_family_counts'])) or
        any(x in r['coencoded_roles'] for x in ['groupII_IEP','RT_unclassified','Y1_TnpA_RAYT']))]
    write(out/'manual_review_queue.tsv',needs_review,list(bags[0]))
    summary=dict(n_bags=len(bags),family_labels=dict(Counter(r['family_label'] for r in bags)),
                 clean_control_families=dict(Counter(r['family_label'] for r in bags if r['clean_family_control'])),
                 manual_review_queue=len(needs_review),
                 policy='No signal scores read. Pfam curated GA and >=70% model coverage (RVT_1/Domain_X >=60%, checked on full-length LtrA); competitive ISEScan >=20 domain-bit margin; complete dominant ORF; >=90% unique-member consistency with no conflicting supported label. group II requires same-strand downstream D5 plus RT/maturase on one ORF. These are conservative annotation policies, not measured classification accuracy.',
                 limitations=['No experimental mechanism inferred for individual bags','IS200 vs IS605 subtype not forced','RAYT exclusion based on sparse reviewed references; residual uncertainty possible','TnpB-like architecture is not an activity assay; Cas12/IscB homolog discrimination may require further references','Unrecognized, frameshifted or partial proteins stay unresolved','Source-event annotation does not validate actual carrier-copy sequence'])
    (out/'annotation_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8');print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
