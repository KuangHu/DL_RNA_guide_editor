"""Counterexample checks for family annotation; no external models required."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'classify_bag_families.py'


def table(path,rows,fields=None):
    fields=fields or list(rows[0])
    with path.open('w',newline='',encoding='utf8') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,delimiter='\t');w.writeheader();w.writerows(rows)


def hit(pid,acc,lo,hi,library='pfam',model='',length=150,score=100):
    return dict(protein_id=pid,library=library,model=model or acc,accession=acc,
                model_length=length,protein_length=600,seq_evalue=1e-30,seq_score=score,
                domain_i_evalue=1e-30,domain_score=score,hmm_from=1,hmm_to=length,
                ali_from=lo,ali_to=hi,hmm_coverage=1,protein_coverage=.8)


class AnnotationRules(unittest.TestCase):
    def run_fixture(self,hits,rnas=(),references=None,secondary_gene=False):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);models=p/'models';models.mkdir()
            pids=sorted({h['protein_id'] for h in hits})
            plen={pid:50 if secondary_gene and pid=='Q' else 600 for pid in pids}
            (p/'proteins.faa').write_text(''.join(f'>{pid}\n'+('A'*plen[pid])+'\n' for pid in pids))
            table(p/'profile_hits.tsv',hits)
            table(p/'gene_origins.tsv',[dict(insert_id='P' if secondary_gene else pid,gene_idx=0,protein_id=pid,lo=100,hi=100+plen[pid]*3,strand=1,partial_begin=secondary_gene and pid=='Q',partial_end=False,protein_len=plen[pid]) for pid in pids])
            table(p/'source_events.tsv',[dict(bag_id='BAG',db_id=pid,insert_id=pid,species='fixture',length=2200,insert_md5='') for pid in (['P'] if secondary_gene else pids)])
            table(p/'bags.tsv',[dict(bag_id='BAG')])
            table(p/'rna_hits.tsv',list(rnas),['insert_id','rfam_accession','model','model_from','model_to','seq_from','seq_to','strand','truncation','score','evalue','included'])
            table(models/'uniprot_reviewed_references.tsv',references or [],['Entry','Protein names'])
            (p/'reviewed_reference_hits.tsv').write_text('')
            if references:
                (p/'reviewed_reference_hits.tsv').write_text('P\tsp|RAYT|ref\t95\t150\t1\t150\t1\t150\t1e-80\t400\t150\t150\n')
            subprocess.run([sys.executable,str(SCRIPT),'--inputs',str(p),'--models',str(models),'--bags-tsv',str(p/'bags.tsv')],check=True,capture_output=True)
            with (p/'protein_role_annotations.tsv').open() as fh:
                roles={r['protein_id']:r['role'] for r in csv.DictReader(fh,delimiter='\t')}
            with (p/'bag_family_annotations.tsv').open() as fh:
                bag=next(csv.DictReader(fh,delimiter='\t'))
            return roles,bag

    def test_long_IS200_profile_is_TnpB_not_TnpA(self):
        roles,bag=self.run_fixture([hit('P','PF12323',1,50),hit('P','PF01385',70,190),hit('P','',1,400,'isescan','IS200_IS605_58.faa',429,250)])
        self.assertEqual(roles['P'],'TnpB_like');self.assertEqual(bag['family_label'],'TnpB-like')

    def test_RT_alone_and_RNA_hit_do_not_confirm_intron(self):
        roles,bag=self.run_fixture([hit('P','PF00078',90,240)],[dict(insert_id='P',rfam_accession='RF00029',model='D5',model_from=1,model_to=77,seq_from=2000,seq_to=2076,strand='+',truncation='no',score=50,evalue=1e-10,included='!')])
        self.assertEqual(roles['P'],'RT_unclassified');self.assertEqual(bag['family_label'],'UNKNOWN')

    def test_groupII_requires_correct_RNA_strand_and_layout(self):
        hits=[hit('P','PF00078',90,240),hit('P','PF08388',380,460)]
        for strand,start,end,label in [('+',2000,2076,'group_II_intron'),('-',2076,2000,'UNKNOWN'),('+',10,86,'UNKNOWN')]:
            with self.subTest(strand=strand,start=start):
                roles,bag=self.run_fixture(hits,[dict(insert_id='P',rfam_accession='RF00029',model='D5',model_from=1,model_to=77,seq_from=start,seq_to=end,strand=strand,truncation='no',score=50,evalue=1e-10,included='!')])
                self.assertEqual(roles['P'],'groupII_IEP_supported');self.assertEqual(bag['family_label'],label)

    def test_element_name_IS1106_does_not_override_domains(self):
        roles,bag=self.run_fixture([hit('IS1106','PF01609',90,260),hit('IS1106','',1,320,'isescan','IS5_20.faa',320,250)])
        self.assertEqual(roles['IS1106'],'IS5_supported');self.assertEqual(bag['family_label'],'IS5')

    def test_RAYT_reference_prevents_TnpA_assignment(self):
        roles,bag=self.run_fixture([hit('P','PF01797',10,130),hit('P','',1,140,'isescan','IS200_IS605_384.faa',144,250)],references=[{'Entry':'RAYT','Protein names':'REP-associated tyrosine transposase'}])
        self.assertEqual(roles['P'],'RAYT_supported');self.assertEqual(bag['family_label'],'RAYT')

    def test_conflicting_source_members_are_unknown(self):
        roles,bag=self.run_fixture([hit('A','PF01609',90,260),hit('A','',1,320,'isescan','IS5_20.faa',320,250),
                                   hit('B','PF01548',1,155),hit('B','PF02371',200,280),hit('B','',1,350,'isescan','IS110_1.faa',350,250)])
        self.assertEqual(bag['family_label'],'UNKNOWN');self.assertEqual(bag['annotation_status'],'mixed_or_conflicting_members')

    def test_partial_guided_domain_disqualifies_clean_control(self):
        partial=hit('Q','PF12323',1,20);partial['hmm_coverage']=.4
        roles,bag=self.run_fixture([hit('P','PF01609',90,260),hit('P','',1,320,'isescan','IS5_20.faa',320,250),partial],secondary_gene=True)
        self.assertEqual(bag['family_label'],'IS5');self.assertEqual(bag['clean_family_control'],'False')


if __name__=='__main__':unittest.main()
