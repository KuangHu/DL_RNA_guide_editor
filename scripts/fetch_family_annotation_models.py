"""Fetch hash-pinned protein/RNA family models without silently upgrading."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import requests


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,default=Path('config/family_annotation_models.json'))
    ap.add_argument('--outdir',type=Path,required=True)
    a=ap.parse_args();config=json.loads(a.config.read_text());a.outdir.mkdir(parents=True,exist_ok=True)
    verified=[]
    for item in config['files']:
        name=item['filename']
        if Path(name).name!=name:raise ValueError('model filename must be a basename')
        dest=a.outdir/name
        if dest.exists():
            content=dest.read_bytes()
        else:
            response=requests.get(item['url'],timeout=120);response.raise_for_status();content=response.content
            if content.startswith(b'\x1f\x8b') and item['encoding']=='optional_gzip':content=gzip.decompress(content)
        if hashlib.sha256(content).hexdigest()!=item['sha256']:
            raise ValueError(f'{name}: model version/hash changed; recover the pinned model or explicitly define a new model snapshot')
        if not dest.exists():dest.write_bytes(content)
        verified.append(dict(**item,bytes=len(content)))
        print('Verified',name,flush=True)
    (a.outdir/'download_manifest.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':main()
