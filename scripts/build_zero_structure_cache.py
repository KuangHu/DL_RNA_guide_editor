"""Build a ZERO structure cache for a JSONL.

For 48B/48C0 ablations: we want structure channels in candidate patches to
be all zeros (no structural signal), while still going through the normal
pipeline. This trivially creates a memmap of zeros in the shape the loader
expects.

No RNAplfold; runs in seconds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--jsonl', required=True)
    ap.add_argument('--out-prefix', required=True)
    ap.add_argument('--nc-max', type=int, default=350)
    ap.add_argument('--u-max', type=int, default=16)
    ap.add_argument('--W', type=int, default=120)
    ap.add_argument('--L', type=int, default=60)
    args = ap.parse_args()

    # Pass 1: enumerate site slots
    N = 0
    site2slots = {}
    with open(args.jsonl) as f:
        for line in f:
            r = json.loads(line)
            sid = r['site_id']
            ncs = r['inputs']['noncoding_regions']
            slots = {}
            for slot_i in range(len(ncs)):
                slots[str(slot_i)] = N
                N += 1
            site2slots[sid] = {'num_slots': len(ncs), 'slots': slots}
    print(f'[pass1] {len(site2slots)} sites, {N} NC rows')

    # Allocate zero mmaps
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    mmap_path = f'{args.out_prefix}.mmap'
    valid_path = f'{args.out_prefix}.valid'
    prof = np.memmap(mmap_path, dtype=np.float16, mode='w+',
                       shape=(N, args.nc_max, args.u_max))
    valid = np.memmap(valid_path, dtype=np.uint8, mode='w+',
                        shape=(N, args.nc_max, args.u_max))
    prof[:] = 0.0
    valid[:] = 1
    prof.flush(); valid.flush()
    print(f'[alloc] {mmap_path} float16 (N={N}, nc_max={args.nc_max}, u_max={args.u_max})')

    # Write index
    idx = {
        '_meta': {
            'N': N, 'nc_max': args.nc_max, 'u_max': args.u_max,
            'W': args.W, 'L': args.L,
            'mmap_path': mmap_path,
            'valid_path': valid_path,
            'split': 'zero_structure',
        },
        **site2slots,
    }
    idx_path = f'{args.out_prefix}.index.json'
    with open(idx_path, 'w') as f:
        json.dump(idx, f)
    print(f'[write] {idx_path}')


if __name__ == '__main__':
    main()
