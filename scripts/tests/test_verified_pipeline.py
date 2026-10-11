"""Integration regressions for full-coordinate scoring and shared calibration."""
import copy
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.alignment import revcomp
from scripts.v5a_framework.pipeline import run_bag, PipelineConfig
from scripts.v5a_framework.multi_hit_retention import BagRetention, SiteRetention, Hit
from scripts.v5a_framework.verified_coordinates import project_bag, input_digest, attach_manifest
from scripts.v5a_framework.cluster_enumeration import enumerate_clusters
from scripts.v5a_framework.background_alignment import side_metrics, select_side, same_locus, gapped_identity
from scripts.v5a_framework.background_grouping import BgConfig, apply_phase_d_gate
from scripts.v5a_framework.reporting_statistic import bag_statistics, config_fingerprint, STATISTIC_VERSION
from scripts.v5a_framework.null_generators import null3_unpaired_bag
from scripts.run_v3_scorer import _row_for_cluster, _local_anchor_alignment
from scripts.analyze_v3_calib import read_details, fit_threshold, pass_rates, CONDITIONS

CORE = 'GCATGCTAGCA'


def dna(n, seed):
    rng = random.Random(seed)
    return ''.join(rng.choice('ACGT') for _ in range(n))


def make_sites(bag_id='synthetic', opposite_end=False):
    canonical = list(dna(180, 101))
    canonical[50:61] = CORE
    canonical[150:161] = CORE
    canonical = ''.join(canonical)
    sites, hits = [], []
    for i in range(3):
        reverse = i == 1 and not opposite_end
        occupied = revcomp(canonical) if reverse else canonical
        frag_lo = 135 if opposite_end and i == 1 else 35
        fragment = canonical[frag_lo:frag_lo+40]
        if reverse:
            fragment = revcomp(fragment)
        raw_np = 14 if reverse else 15  # 40 - 15 - 11 = 14
        raw_fp = 69 if reverse else 40
        flank = list(dna(120, 300+i))
        flank[raw_fp:raw_fp+11] = revcomp(CORE) if reverse else CORE
        evidence = dict(occupied=occupied, empty=''.join(flank), alignment_id='synthetic_element_msa',
                        anchor=dict(family='synthetic_anchor', strand='-' if reverse else '+',
                                    occupied_lo=70 if reverse else 80,
                                    occupied_hi=100 if reverse else 110),
                        oriented_context_msa=canonical, context_lo=0,
                        valid_insert_intervals=[[0,180]],
                        target=dict(frame='synthetic_junction', strand='-' if reverse else '+',
                                    origin=119 if reverse else 0, origin_status='verified',
                                    empty_lo=0, empty_hi=120, source_strand='+',
                                    frame_strand='-' if reverse else '+',frame_anchor_empty_pos=119 if reverse else 0),
                        locus=dict(pre_event_genome=f'g{i}', pre_event_contig='c',start=100,end=500))
        sites.append(dict(transposase_id=bag_id, inputs=dict(noncoding_regions=[fragment], flank=''.join(flank)),
                          scorer_evidence=evidence))
        hits.append(SiteRetention(i, {(raw_np,11):[Hit(raw_np,11,raw_fp,11,'fwd')]}))
    return sites, BagRetention(3,list(range(3)),hits,{})


class VerifiedPipelineTests(unittest.TestCase):
    def test_opposite_orientation_projects_full_window(self):
        sites, bag = make_sites()
        mapped, report = project_bag(bag, sites)
        self.assertEqual(report.status, 'ready')
        clusters = enumerate_clusters(mapped, min_frac=1)
        self.assertEqual(len(clusters),1)
        c = clusters[0]
        self.assertEqual((c.nc_pos,c.flank_pos_median,c.n_support),(50,40,3))
        self.assertEqual(c.nc_columns,tuple(range(50,61)))
        passing, failing = apply_phase_d_gate(clusters, {i:s['inputs']['flank'] for i,s in enumerate(sites)},
                                             BgConfig(method='gapped'),
                                             {i:s['scorer_evidence']['locus'] for i,s in enumerate(sites)})
        self.assertEqual(len(passing),1)
        row = _row_for_cluster('synthetic',1,c,passing[0][1],sites,True,None)
        evidence = json.loads(row['support_sites_json'])
        self.assertEqual(evidence[1]['guide_raw'],revcomp(CORE))
        self.assertEqual(evidence[1]['raw_nc_pos'],14)
        self.assertEqual(row['passes_publication_threshold'],'')

    def test_same_raw_offset_opposite_ends_do_not_cluster(self):
        sites, bag = make_sites(opposite_end=True)
        mapped, report = project_bag(bag,sites)
        self.assertEqual(report.status,'ready')
        self.assertEqual(enumerate_clusters(mapped,min_frac=1),[])

    def test_boundary_flip_rejects_full_window(self):
        sites,bag=make_sites()
        sites[0]['scorer_evidence']['valid_insert_intervals'].append([60,180])
        mapped,report=project_bag(bag,sites)
        self.assertEqual(report.rejected_hits['boundary_unstable'],1)
        self.assertEqual(enumerate_clusters(mapped,min_frac=1),[])

    def test_missing_evidence_is_not_a_negative_call(self):
        sites,_=make_sites()
        for site in sites:
            del site['scorer_evidence']
        result=run_bag('missing',sites,PipelineConfig(Ls=(11,)))
        self.assertEqual(result.status,'needs_evidence')
        self.assertFalse(bag_statistics(result)['eligible'])
        self.assertEqual(result.K_valid,3)

    def test_wrong_target_source_is_quarantined(self):
        sites,bag=make_sites()
        sites[0]['scorer_evidence']['empty']='A'*120
        _,report=project_bag(bag,sites)
        self.assertEqual(report.status,'needs_evidence')
        self.assertEqual(report.site_errors[0],'target_empty_source_sequence_mismatch')

    def test_context_alignment_is_anchor_fixed_and_spacer_bounded(self):
        core='GCATGCTAGCA'
        frag='TT'+core+'CC'
        site=dict(inputs=dict(noncoding_regions=['ACGT',frag],flank='AA'+frag+'GG'))
        hit=Hit(4+13+2,11,4,11,'fwd')
        display=_local_anchor_alignment(site,hit)
        self.assertEqual(display['guide'],frag)
        self.assertNotIn('N',display['guide'])
        self.assertEqual(display['total_exact_contiguous_bp'],len(frag))
        site['inputs']['flank']=revcomp(site['inputs']['flank'])
        rc_hit=Hit(hit.nc_pos,hit.L,len(site['inputs']['flank'])-hit.flank_pos-hit.L,11,'rc')
        self.assertEqual(_local_anchor_alignment(site,rc_hit),display)

    def test_null_template_cannot_be_used_as_real_evidence(self):
        sites,bag=make_sites()
        null=null3_unpaired_bag(sites,sites,seed=1)
        _,report=project_bag(bag,null)
        self.assertEqual(report.status,'needs_evidence')
        _,report=project_bag(bag,null,allow_conditional_null=True)
        self.assertEqual(report.status,'ready')

    def test_real_pipeline_retains_known_mapped_candidate(self):
        sites,_=make_sites()
        result=run_bag('synthetic',sites,PipelineConfig(Ls=(11,),min_frac=1))
        self.assertEqual(result.status,'ready')
        self.assertTrue(any(c.nc_pos==50 and c.mean_m_frac==1 and c.flank_pos_median==40 for c,b in result.passing))

    def test_insufficient_short_side_does_not_hide_long_side(self):
        a=dna(80,901)
        b=''.join(('T' if c!='T' else 'A') if i%10==0 else c for i,c in enumerate(a))
        left=side_metrics('ACGTTGCAAG','ACGTTGCAAG',20,.8)
        right=side_metrics(a,b,20,.8)
        side,metrics=select_side(left,right)
        self.assertEqual(side,'right')
        self.assertGreater(metrics['identity'],.85)

    def test_indel_and_asymmetric_coverage(self):
        a=dna(100,123)
        metrics=side_metrics(a,a[:50]+'C'+a[50:],20,.8)
        self.assertEqual(metrics['paired_bp'],100)
        self.assertEqual(metrics['indels'],1)
        self.assertTrue(metrics['eligible'])
        short=side_metrics('A'*100,'A'*20,20,.8)
        self.assertFalse(short['eligible'])
        self.assertEqual(short['coverage_longer'],.2)

    def test_provenance_merge_and_empty_guard(self):
        self.assertFalse(same_locus(dict(pre_event_genome='',pre_event_contig='',start=100,end=500),
                                    dict(pre_event_genome='',pre_event_contig='',start=100,end=500)))
        sites,bag=make_sites()
        mapped,_=project_bag(bag,sites)
        cluster=enumerate_clusters(mapped,min_frac=1)[0]
        loci={i:s['scorer_evidence']['locus'] for i,s in enumerate(sites)}
        loci[1]=copy.deepcopy(loci[0])
        passing,failing=apply_phase_d_gate([cluster],{i:s['inputs']['flank'] for i,s in enumerate(sites)},
                                          BgConfig(method='gapped'),loci)
        self.assertFalse(passing)
        self.assertEqual(failing[0][1].n_local_bg_groups,2)

    def test_unknown_pairwise_comparison_does_not_inflate_groups(self):
        sites,bag=make_sites()
        mapped,_=project_bag(bag,sites)
        c=enumerate_clusters(mapped,min_frac=1)[0]
        flanks={i:'N'*120 for i in range(3)}
        passing,failing=apply_phase_d_gate([c],flanks,BgConfig(method='gapped'))
        self.assertFalse(passing)
        self.assertEqual(failing[0][1].n_local_bg_groups,1)
        self.assertEqual(len(failing[0][1].uncertain_pairs),3)

    def test_shared_statistic_does_not_choose_support_first(self):
        clusters=[(SimpleNamespace(mean_m_frac=m,n_support=n,L=10,flank_pos_mad=0),
                   SimpleNamespace(n_local_bg_groups=3)) for n,m in [(8,.8),(4,1.)]]
        stat=bag_statistics(SimpleNamespace(passing=clusters,status='ready'))
        self.assertEqual(stat['max_mean_m'],1.)
        self.assertEqual(stat['max_n_support'],8)
        self.assertEqual(stat['best_m_at_top'],10.)

    def test_manifest_bound_to_actual_inputs(self):
        sites,_=make_sites()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'manifest.jsonl'
            row=dict(bag_id='synthetic',site_idx=0,input_sha256=input_digest(sites[0]),evidence=sites[0]['scorer_evidence'])
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
            attach_manifest({'synthetic':sites},path)
            sites[0]['inputs']['flank']='A'*120
            with self.assertRaises(ValueError):
                attach_manifest({'synthetic':sites},path)

    def test_donor_background_provenance_follows_null3(self):
        sites,_=make_sites()
        donor=copy.deepcopy(sites[:1])
        donor[0]['scorer_evidence']['locus']['pre_event_genome']='donor'
        null=null3_unpaired_bag(sites,donor,seed=1)
        self.assertTrue(all(s['scorer_evidence']['locus']['pre_event_genome']=='donor' for s in null))
        self.assertEqual(sites[0]['scorer_evidence']['locus']['pre_event_genome'],'g0')

    def test_calibration_requires_complete_conditions_and_shared_version(self):
        cfg=PipelineConfig()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'details.tsv'
            fields=['bag_id','cond','max_mean_m','statistic_version','config_fingerprint','status','eligible','err']
            rows=[dict(bag_id=f'b{i}',cond=c,max_mean_m=.95 if c=='real' else .6,
                       statistic_version=STATISTIC_VERSION,config_fingerprint=config_fingerprint(cfg),
                       status='ready',eligible=True,err='') for i in range(4) for c in CONDITIONS]
            def write(rs):
                with path.open('w',newline='') as fh:
                    w=csv.DictWriter(fh,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rs)
            write(rows)
            data,fingerprint,excluded=read_details(path)
            threshold=fit_threshold(data,['b0','b1'],.01)
            self.assertGreater(threshold,.6)
            self.assertEqual(pass_rates(data,['b2','b3'],threshold)['real'],1)
            write(rows[:-1])
            with self.assertRaises(ValueError):
                read_details(path)


if __name__=='__main__':
    unittest.main()
