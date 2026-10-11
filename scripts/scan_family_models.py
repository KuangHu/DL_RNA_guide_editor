"""Search family/role profiles against ALL deduplicated insertion proteins."""
import argparse
import csv
import json
import sys
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tools-dir',type=Path)
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--cpus',type=int,default=8)
    a=ap.parse_args()
    if a.tools_dir:sys.path.insert(0,str(a.tools_dir.resolve()))
    import pyhmmer
    with pyhmmer.easel.SequenceFile(str(a.inputs/'proteins.faa'),digital=True,alphabet=pyhmmer.easel.Alphabet.amino()) as fh:
        seqs=fh.read_block()
    print('Unique protein targets:',len(seqs),flush=True)
    fields=['protein_id','library','model','accession','model_length','protein_length','seq_evalue','seq_score',
            'domain_i_evalue','domain_score','hmm_from','hmm_to','ali_from','ali_to','hmm_coverage','protein_coverage']
    counts={}
    with (a.inputs/'profile_hits.tsv').open('w',encoding='utf8',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader()
        for library in ['pfam','isescan']:
            if library=='pfam':
                hmms=[]
                for file in sorted(a.models.glob('PF*.hmm')):
                    with pyhmmer.plan7.HMMFile(str(file)) as h:hmms.extend(h)
                kwargs=dict(bit_cutoffs='gathering')
            else:
                with pyhmmer.plan7.HMMFile(str(a.models/'clusters.faa.hmm')) as h:hmms=list(h)
                kwargs=dict(E=1e-10,domE=1e-5,incE=1e-10,incdomE=1e-5)
            n=0
            for idx,(hmm,hits) in enumerate(zip(hmms,pyhmmer.hmmer.hmmsearch(hmms,seqs,cpus=a.cpus,**kwargs)),1):
                for hit in hits:
                    if not hit.reported:continue
                    for dom in hit.domains:
                        if not dom.reported:continue
                        aln=dom.alignment
                        w.writerow(dict(protein_id=hit.name,library=library,model=hmm.name,accession=hmm.accession or '',
                          model_length=hmm.M,protein_length=hit.length,seq_evalue=hit.evalue,seq_score=hit.score,
                          domain_i_evalue=dom.i_evalue,domain_score=dom.score,hmm_from=aln.hmm_from,hmm_to=aln.hmm_to,
                          ali_from=aln.target_from,ali_to=aln.target_to,
                          hmm_coverage=(aln.hmm_to-aln.hmm_from+1)/hmm.M,
                          protein_coverage=(aln.target_to-aln.target_from+1)/hit.length))
                        n+=1
                if idx%25==0 or idx==len(hmms):
                    fh.flush();print(f'{library}: {idx}/{len(hmms)} models, {n} domain hits',flush=True)
            counts[library]=n
    manifest=dict(pyhmmer_version=pyhmmer.__version__,protein_targets=len(seqs),domain_hits=counts,
                  pfam_threshold='curated gathering cutoff (sequence AND domain)',
                  isescan_threshold='sequence E<=1e-10; reported domain E<=1e-5; no family decision made by this search',
                  models_manifest=json.loads((a.models/'download_manifest.json').read_text()))
    (a.inputs/'search_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
