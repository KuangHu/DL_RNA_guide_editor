"""Continuous family signal comparison; no guided/not-guided score cutoff."""
import argparse
import csv
import itertools
import json
import statistics
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(20_000_000)
FAMILIES=['IS110','group_II_intron','IS200/IS605-TnpA','TnpB-like','IS1','IS3','IS5']
DISPLAY=['IS110','Group II','IS200/605 TnpA','TnpB-like','IS1','IS3','IS5']


def read(path):
    with path.open(encoding='utf8',newline='') as fh:return list(csv.DictReader(fh,delimiter='\t'))


def write(path,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf8',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)


def auc(a,b):
    if not len(a) or not len(b):return None
    x=np.asarray(a)[:,None];y=np.asarray(b)[None,:]
    return float(np.mean((x>y)+.5*(x==y)))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--previous-scan',type=Path,required=True)
    ap.add_argument('--raw-bags-tsv',type=Path,required=True)
    a=ap.parse_args();out=a.outdir
    bgmanifest=json.loads((out/'background_manifest.json').read_text());assert bgmanifest['complete']
    oldmanifest=json.loads((a.previous_scan/'run_manifest.json').read_text())
    newmanifest=json.loads((out/'foreground_new/run_manifest.json').read_text())
    assert newmanifest['complete'] and oldmanifest['complete']
    assert bgmanifest['config_fingerprint']==newmanifest['config_fingerprint']==oldmanifest['config_fingerprint']
    reused=set((out/'reused_bag_ids.txt').read_text().split());newids=set((out/'new_bag_ids.txt').read_text().split())
    assert not reused.intersection(newids)
    coverage=read(out/'cohort_coverage.tsv');rawsizes={r['bag_id']:r['n_sites'] for r in read(a.raw_bags_tsv)}
    rows={r['bag_id']:dict(r,n_sites_in_raw_bag=rawsizes[r['bag_id']]) for r in coverage}
    sites=defaultdict(list)
    for line in (out/'cohort_inputs.jsonl').read_text().splitlines():
        r=json.loads(line);sites[r['transposase_id']].append(r)
    features=['nc_pos','L','flank_pos_median','flank_pos_mad','n_support_raw','support_frac_raw','n_local_bg_groups',
              'mean_m_frac','mean_guide_H_base','mean_guide_H_dinuc','mean_guide_at_frac','guide_at_cluster',
              'left_median_identity','right_median_identity','median_exact_contiguous_bp','max_exact_contiguous_bp']
    for r in read(a.previous_scan/'signal_by_bag.tsv'):
        if r['bag_id'] in reused:rows[r['bag_id']].update({k:r.get(k,'') for k in features},n_phase_D_candidates=r['n_phase_D_candidates'])
    best={}
    with (out/'foreground_new/candidates_v3.tsv').open(encoding='utf8',newline='') as fh:
        for r in csv.DictReader(fh,delimiter='\t'):
            if r['passes_phase_D']!='True':continue
            key=(float(r['mean_m_frac']),int(r['n_support_raw']),int(r['n_local_bg_groups']),-float(r['flank_pos_mad']))
            if r['bag_id'] not in best or key>best[r['bag_id']][0]:best[r['bag_id']]=key,r
    summaries={r['bag_id']:r for r in map(json.loads,(out/'foreground_new/bag_summary.jsonl').read_text().splitlines())}
    for bid in newids:
        rows[bid]['n_phase_D_candidates']=summaries[bid]['n_passing']
        if bid in best:
            r=best[bid][1];rows[bid].update({k:r.get(k,'') for k in features})
            supports=json.loads(r['support_sites_json'])
            lengths=[s['local_alignment']['total_exact_contiguous_bp'] for s in supports if s.get('local_alignment') and s['local_alignment']['total_exact_contiguous_bp'] is not None]
            rows[bid].update(median_exact_contiguous_bp=statistics.median(lengths) if lengths else '',max_exact_contiguous_bp=max(lengths) if lengths else '')
    for bid,ss in sites.items():
        nc=[('N'*13).join(s['inputs']['noncoding_regions']) for s in ss]
        rows[bid].update(copy_site_fraction=sum('.copy' in s['site_id'] for s in ss)/len(ss),
                         n_distinct_nc_sequences=len(set(nc)),mean_nc_length=sum(len(s) for s in nc)/len(nc),
                         raw_score=float(rows[bid].get('mean_m_frac') or 0),coordinate_status='legacy_raw_unverified')
    backgrounds=defaultdict(list)
    for r in read(out/'background_scores.tsv'):backgrounds[(r['bag_id'],r['background'])].append(float(r['max_mean_m']))
    for bid in sites:
        for kind in bgmanifest['backgrounds']:
            values=backgrounds[(bid,kind)];assert len(values)==bgmanifest['repeats_per_background']
            score=rows[bid]['raw_score'];avg=float(np.mean(values));sd=float(np.std(values,ddof=1))
            rows[bid].update({kind+'_mean':avg,kind+'_sd':sd,kind+'_delta':score-avg,
                             kind+'_z':(score-avg)/sd if sd>0 else '',
                             kind+'_empirical_rank':(sum(x<score for x in values)+.5*sum(x==score for x in values))/len(values)})
    complete=sorted(rows.values(),key=lambda r:(FAMILIES.index(r['family_label']),r['bag_id']))
    write(out/'family_signal_by_bag.tsv',complete)
    alignments=[]
    for r in read(a.previous_scan/'signal_best_alignments.tsv'):
        if r['bag_id'] in reused:alignments.append(dict(r,family_label=rows[r['bag_id']]['family_label']))
    for bid,(_,r) in best.items():
        for s in json.loads(r['support_sites_json']):
            local=s.get('local_alignment') or {};sid=sites[bid][s['site_idx']]['site_id']
            alignments.append(dict(bag_id=bid,family_label=rows[bid]['family_label'],site_id=sid,site_idx=s['site_idx'],
                                   nc_source='inherited_source_event' if '.copy' in sid else 'observed_event_nc',coordinate_status='raw_unverified',
                                   nc_pos=s['raw_nc_pos'],flank_pos=s['raw_flank_pos'],orientation=s['raw_orient'],core_L=r['L'],match_count=s['match_count'],
                                   guide_context=local.get('guide',''),match_line=local.get('match_line',''),target_context=local.get('target_in_guide_orientation',''),
                                   core_start=local.get('core_start',''),total_exact_contiguous_bp=local.get('total_exact_contiguous_bp','')))
    write(out/'family_best_alignments.tsv',alignments)
    metrics=['raw_score','exact_dinuc_delta','unpaired_same_K_delta']
    famrows=[];groups={f:[r for r in complete if r['family_label']==f and r['input_status']=='scannable'] for f in FAMILIES}
    for f in FAMILIES:
        rr=groups[f];n=len(rr);total=sum(r['family_label']==f for r in complete)
        record=dict(family=f,n_annotated=total,n_scanned=n,coverage=n/total,n_K8=sum(int(r['K'])==8 for r in rr),
                    median_K=float(np.median([int(r['K']) for r in rr])),median_nc_length=float(np.median([r['mean_nc_length'] for r in rr])),median_copy_fraction=float(np.median([r['copy_site_fraction'] for r in rr])))
        for metric in metrics:
            values=[r[metric] for r in rr]
            record.update({metric+'_median':float(np.median(values)),metric+'_q25':float(np.quantile(values,.25)),metric+'_q75':float(np.quantile(values,.75))})
        famrows.append(record)
    write(out/'family_distributions.tsv',famrows)
    controls=[r for f in ['IS1','IS3','IS5'] for r in groups[f]]
    pairs=list(itertools.combinations(FAMILIES,2))+[(f,'pooled_strict_controls') for f in FAMILIES[:4]]
    rng=np.random.default_rng(20261011);aucrows=[]
    for first,second in pairs:
        aa=groups[first];bb=controls if second=='pooled_strict_controls' else groups[second]
        for metric in metrics:
            av=[r[metric] for r in aa];bv=[r[metric] for r in bb]
            boot=[auc(rng.choice(av,len(av),replace=True),rng.choice(bv,len(bv),replace=True)) for _ in range(1000)]
            matched=[];npairs=0;weighted=0.0
            for k in range(3,9):
                x=[r[metric] for r in aa if int(r['K'])==k];y=[r[metric] for r in bb if int(r['K'])==k]
                if x and y:
                    value=auc(x,y);weight=len(x)*len(y);matched.append(value);npairs+=weight;weighted+=value*weight
            x8=[r[metric] for r in aa if int(r['K'])==8];y8=[r[metric] for r in bb if int(r['K'])==8]
            aucrows.append(dict(family_A=first,family_B=second,metric=metric,n_A=len(aa),n_B=len(bb),
                                AUC_A_greater=auc(av,bv),bag_bootstrap_q025=float(np.quantile(boot,.025)),bag_bootstrap_q975=float(np.quantile(boot,.975)),
                                macro_AUC_within_same_K=float(np.mean(matched)) if matched else '',
                                pair_weighted_AUC_same_K=weighted/npairs if npairs else '',
                                AUC_K8_only=auc(x8,y8) if x8 and y8 else '',n_A_K8=len(x8),n_B_K8=len(y8),
                                n_K_strata=len(matched),n_same_K_pairs=npairs))
    write(out/'pairwise_auc.tsv',aucrows)
    colors=['#2c7fb8','#41ab5d','#9467bd','#e69f00','#888888','#b56758','#718080']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for ax,metric,title in [(axes[0,0],'raw_score','Raw maximum mean match'),(axes[0,1],'exact_dinuc_delta','Excess over exact dinucleotide backgrounds'),(axes[1,0],'unpaired_same_K_delta','Excess over unpaired donor bags')]:
        for i,f in enumerate(FAMILIES):
            values=np.asarray([r[metric] for r in groups[f]])
            jit=np.linspace(-.16,.16,len(values))
            ax.scatter(i+jit,values,color=colors[i],s=22,alpha=.7)
            ax.plot([i-.2,i+.2],[np.median(values)]*2,color='black',lw=2)
        if metric!='raw_score':ax.axhline(0,color='gray',lw=.7,ls='--')
        ax.set_xticks(range(7),[f'{d}\nn={len(groups[f])}' for d,f in zip(DISPLAY,FAMILIES)],rotation=20,ha='right',fontsize=8)
        ax.set_title(title);ax.set_ylabel('Match fraction' if metric=='raw_score' else 'Foreground minus background mean')
    ax=axes[1,1]
    matrix=np.full((7,7),np.nan)
    for i,f in enumerate(FAMILIES):
        for j,g in enumerate(FAMILIES):
            if i!=j:matrix[i,j]=auc([r['exact_dinuc_delta'] for r in groups[f]],[r['exact_dinuc_delta'] for r in groups[g]])
    im=ax.imshow(matrix,vmin=0,vmax=1,cmap='coolwarm')
    for i in range(7):
        for j in range(7):
            if i!=j:ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=8)
    ax.set_xticks(range(7),DISPLAY,rotation=35,ha='right',fontsize=8);ax.set_yticks(range(7),DISPLAY,fontsize=8);ax.set_title('Pairwise ranking AUC: row > column\nExact-dinucleotide excess score');fig.colorbar(im,ax=ax,shrink=.8)
    fig.suptitle('Independent family cohort: 73/86 bags scanned; 16 backgrounds per type per bag\nCurrent single-segment 9–14 bp architecture; raw coordinates; descriptive comparisons',fontsize=12)
    fig.savefig(out/'family_signal_overview.png',dpi=160);fig.savefig(out/'family_signal_overview.svg');plt.close(fig)
    summary=dict(n_annotated=len(complete),n_scanned=len(sites),n_not_scored=len(complete)-len(sites),n_reused=len(reused),n_new=len(newids),
                 n_background_runs=bgmanifest['n_runs'],family_distributions=famrows,
                 target_vs_controls=[r for r in aucrows if r['family_B']=='pooled_strict_controls'],
                 interpretation='Descriptive family association in a conservative labeled subset; no binary cutoff or activity/independence claims. K-stratified AUC and coverage are provided. Small-family CIs resample bags, not genome events.',
                 complete=True)
    (out/'comparison_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
