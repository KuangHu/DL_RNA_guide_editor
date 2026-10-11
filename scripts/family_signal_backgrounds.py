"""Exploratory continuous-score backgrounds for an independently labeled cohort.

No binary cutoff. Dinucleotide shuffles preserve exact counts (random Euler
trails, not uniformly sampled sequences). Whole donor bags of the same K retain
real flank structure; they are a pairing control, not a guaranteed signal-free
null. Both use the same raw-coordinate pipeline as the foreground.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.v5a_framework.pipeline import PipelineConfig,run_bag
from scripts.v5a_framework.reporting_statistic import bag_statistics,config_fingerprint,effective_configuration

POOLS={}
ERROR_DIR=None


def load(path):
    bags=defaultdict(list)
    with path.open() as fh:
        for line in fh:
            r=json.loads(line);bags[r['transposase_id']].append(r)
    return dict(bags)


def exact_dinuc_shuffle(sequence,seed):
    if len(sequence)<2:return sequence
    rng=random.Random(seed);edges=defaultdict(list)
    for left,right in zip(sequence,sequence[1:]):edges[left].append(right)
    for choices in edges.values():rng.shuffle(choices)
    stack=[sequence[0]];trail=[]
    while stack:
        if edges[stack[-1]]:stack.append(edges[stack[-1]].pop())
        else:trail.append(stack.pop())
    return ''.join(reversed(trail))


def initialize(donor_path,error_dir):
    global POOLS,ERROR_DIR
    ERROR_DIR=Path(error_dir)
    POOLS=defaultdict(dict)
    for bid,sites in load(Path(donor_path)).items():
        POOLS[len(sites)][bid]=[s['inputs']['flank'] for s in sites]


def run_one(task):
    bid,sites,repeats=task;cfg=PipelineConfig(coordinate_mode='legacy');rows=[]
    donors=sorted(set(POOLS[len(sites)])-{bid})
    if not donors:raise ValueError(f'{bid}: no distinct donor with same K')
    for kind in ['exact_dinuc','unpaired_same_K']:
        for rep in range(repeats):
            seed=int.from_bytes(hashlib.sha256(f'family_signal_v1|{bid}|{kind}|{rep}'.encode()).digest()[:8],'big')
            donor=''
            if kind=='exact_dinuc':
                cache={};flanks=[]
                for i,s in enumerate(sites):
                    f=s['inputs']['flank']
                    if f not in cache:cache[f]=exact_dinuc_shuffle(f,seed+i)
                    flanks.append(cache[f])
            else:
                rng=random.Random(seed);donor=rng.choice(donors);flanks=list(POOLS[len(sites)][donor]);rng.shuffle(flanks)
            mutated=[{**s,'inputs':{**s['inputs'],'flank':f}} for s,f in zip(sites,flanks)]
            try:
                result=run_bag(bid,mutated,cfg);stat=bag_statistics(result)
            except Exception as error:
                if ERROR_DIR:
                    (ERROR_DIR/f'failure_{bid}_{kind}_{rep}.json').write_text(json.dumps(dict(bag_id=bid,kind=kind,rep=rep,seed=seed,donor=donor,error=repr(error),sites=mutated)),encoding='utf8')
                raise RuntimeError(f'{bid} {kind} replicate {rep}: {error!r}') from error
            rows.append(dict(bag_id=bid,background=kind,replicate=rep,seed=seed,donor_bag=donor,
                             K=result.K_valid,max_mean_m=stat['max_mean_m'],n_passing=stat['n_passing'],
                             max_n_bg_groups=stat['max_n_bg_groups'],status=result.status))
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cohort-jsonl',type=Path,required=True)
    ap.add_argument('--donor-jsonl',type=Path,required=True)
    ap.add_argument('--outdir',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=16)
    ap.add_argument('--workers',type=int,default=8)
    a=ap.parse_args();bags=load(a.cohort_jsonl);a.outdir.mkdir(parents=True,exist_ok=True)
    t0=time.time();cfg=PipelineConfig(coordinate_mode='legacy')
    manifest=dict(complete=False,n_bags=len(bags),repeats_per_background=a.repeats,
                  backgrounds=['exact_dinuc','unpaired_same_K'],config=effective_configuration(cfg),config_fingerprint=config_fingerprint(cfg),
                  score='maximum mean_m_frac among current Phase D passing clusters, 0 if no candidate; not a biological label',
                  donor_source=str(a.donor_jsonl),no_threshold=True,
                  limitations=['Raw coordinates; copy nc may be inherited','Exact count shuffle samples randomized Euler trails, not uniform sequences','Donor bags can contain real signals','Background perturbations can change local grouping; not a fixed-group conditional test'])
    (a.outdir/'background_manifest.json').write_text(json.dumps(manifest,indent=2))
    fields=['bag_id','background','replicate','seed','donor_bag','K','max_mean_m','n_passing','max_n_bg_groups','status']
    n=0
    with (a.outdir/'background_scores.tsv').open('w',newline='',encoding='utf8') as fh:
        w=csv.DictWriter(fh,fieldnames=fields,delimiter='\t',lineterminator='\n');w.writeheader()
        with ProcessPoolExecutor(max_workers=a.workers,initializer=initialize,initargs=(str(a.donor_jsonl.resolve()),str(a.outdir.resolve()))) as pool:
            futures=[pool.submit(run_one,(bid,sites,a.repeats)) for bid,sites in sorted(bags.items())]
            for number,future in enumerate(as_completed(futures),1):
                rows=future.result();w.writerows(rows);fh.flush();n+=len(rows)
                print(f'Backgrounds {number}/{len(bags)} bags, {time.time()-t0:.1f}s',flush=True)
    assert n==len(bags)*a.repeats*2
    manifest.update(complete=True,n_runs=n,elapsed_seconds=time.time()-t0)
    (a.outdir/'background_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
