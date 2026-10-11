"""Search all source inserts with official group-II Rfam covariance models."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs',type=Path,required=True)
    ap.add_argument('--models',type=Path,required=True)
    ap.add_argument('--cmsearch',required=True)
    ap.add_argument('--wsl-distro')
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--cpus-per-worker',type=int,default=2)
    a=ap.parse_args()
    out=a.inputs/'rna_models';out.mkdir(exist_ok=True)
    def native(path):
        p=str(Path(path).resolve())
        if a.wsl_distro:
            p=p.replace('\\','/')
            return '/mnt/'+p[0].lower()+p[2:]
        return p
    models=sorted(a.models.glob('RF*.cm'))
    def run(model):
        cmd=[native(a.cmsearch),'--cut_ga','--noali','--cpu',str(a.cpus_per_worker),
             '--tblout',native(out/(model.stem+'.tbl')),native(model),native(a.inputs/'insert_sequences.fna')]
        if a.wsl_distro:cmd=['wsl','-d',a.wsl_distro,'--']+cmd
        with (out/(model.stem+'.log')).open('w') as log:
            subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
        print('Completed covariance model',model.stem,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        list(pool.map(run,models))
    fields=['insert_id','rfam_accession','model','model_from','model_to','seq_from','seq_to','strand','truncation','score','evalue','included']
    rows=[]
    for model in models:
        for line in (out/(model.stem+'.tbl')).read_text().splitlines():
            if line.startswith('#') or not line.strip():continue
            p=line.split()
            rows.append(dict(insert_id=p[0],rfam_accession=p[3],model=p[2],model_from=p[5],model_to=p[6],
                             seq_from=p[7],seq_to=p[8],strand=p[9],truncation=p[10],score=p[14],evalue=p[15],included=p[16]))
    with (a.inputs/'rna_hits.tsv').open('w',encoding='utf8',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
    manifest=dict(models=[dict(name=p.stem,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in models],
                  n_reported_hits=len(rows),n_inserts_with_hits=len({r['insert_id'] for r in rows}),
                  threshold='Official Rfam model-specific GA cutoff',sequence_scope='ALL unique source inserts; both strands; no signal selection',
                  executable=a.cmsearch,wsl_distro=a.wsl_distro)
    (a.inputs/'rna_search_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
