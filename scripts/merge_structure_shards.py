"""Merge per-shard RNAplfold precompute memmaps into a single memmap + index.

Assumes each shard was produced by scripts/precompute_structure.py with a
distinct --out base. Discovers shards via --in-glob (matched against the
per-shard index.json files), then:

  1. Sanity-checks that all shards share nc_max / u_max / W / L.
  2. Allocates a combined memmap of shape (sum(N_shard), nc_max, u_max).
  3. Copies each shard's rows into the combined memmap; renumbers the
     per-site row indices with the shard offset.
  4. Writes a merged index.json (same schema as the per-shard files).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-glob", required=True,
                    help="glob for per-shard index.json files, e.g. "
                         "'/groups/.../val_u16_shard*.index.json'")
    p.add_argument("--out", required=True, type=Path,
                    help="output base path (writes <out>.mmap, <out>.valid, <out>.index.json)")
    p.add_argument("--delete-shards", action="store_true",
                    help="after successful merge, delete the per-shard files")
    args = p.parse_args()

    shard_index_paths = sorted(glob.glob(args.in_glob))
    if not shard_index_paths:
        raise SystemExit(f"no shard index files match {args.in_glob!r}")
    print(f"[merge] found {len(shard_index_paths)} shards")

    shards = []
    for ip in shard_index_paths:
        with open(ip) as f:
            idx = json.load(f)
        shards.append((Path(ip), idx))

    ref_meta = shards[0][1]["_meta"]
    for ip, idx in shards[1:]:
        m = idx["_meta"]
        for k in ("nc_max", "u_max", "W", "L"):
            if m[k] != ref_meta[k]:
                raise SystemExit(
                    f"shard param mismatch at {ip}: {k}={m[k]} vs ref {ref_meta[k]}"
                )
    nc_max = int(ref_meta["nc_max"])
    u_max = int(ref_meta["u_max"])
    total_N = sum(int(idx["_meta"]["N"]) for _, idx in shards)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_base = str(args.out)
    out_mmap = out_base + ".mmap"
    out_valid = out_base + ".valid"
    out_index = out_base + ".index.json"

    print(f"[merge] allocating combined memmap: ({total_N}, {nc_max}, {u_max}) float16")
    combined_prof = np.memmap(out_mmap, dtype=np.float16, mode="w+",
                                shape=(total_N, nc_max, u_max))
    combined_valid = np.memmap(out_valid, dtype=np.uint8, mode="w+",
                                shape=(total_N, nc_max, u_max))
    combined_prof[:] = 0
    combined_valid[:] = 0

    merged_index: dict = {
        "_meta": {
            **{k: ref_meta[k] for k in ("nc_max", "u_max", "W", "L")},
            "N": total_N,
            "dtype_prof": "float16",
            "dtype_valid": "uint8",
            "mmap_path": str(Path(out_mmap).resolve()),
            "valid_path": str(Path(out_valid).resolve()),
            "merged_from": shard_index_paths,
        }
    }

    offset = 0
    for ip, idx in shards:
        m = idx["_meta"]
        n = int(m["N"])
        shard_mmap = m["mmap_path"]
        shard_valid = m["valid_path"]
        print(f"[merge] shard {ip.name}: N={n}, rows [{offset}, {offset + n})")

        shard_prof_arr = np.memmap(shard_mmap, dtype=np.float16, mode="r",
                                     shape=(n, nc_max, u_max))
        shard_valid_arr = np.memmap(shard_valid, dtype=np.uint8, mode="r",
                                      shape=(n, nc_max, u_max))
        combined_prof[offset:offset + n] = shard_prof_arr[:]
        combined_valid[offset:offset + n] = shard_valid_arr[:]

        for site_id, entry in idx.items():
            if site_id.startswith("_"):
                continue
            if site_id in merged_index:
                raise SystemExit(
                    f"duplicate site_id {site_id} across shards: {ip}"
                )
            new_entry = {"num_slots": entry["num_slots"], "slots": {}}
            for slot_s, row in entry["slots"].items():
                new_entry["slots"][slot_s] = int(row) + offset
            merged_index[site_id] = new_entry

        offset += n
        del shard_prof_arr, shard_valid_arr

    combined_prof.flush()
    combined_valid.flush()
    with open(out_index, "w") as f:
        json.dump(merged_index, f)
    print(f"[merge] wrote {out_mmap}, {out_valid}, {out_index}")
    print(f"[merge] total rows: {total_N}, sites: {sum(1 for k in merged_index if not k.startswith('_'))}")

    if args.delete_shards:
        for ip, idx in shards:
            m = idx["_meta"]
            for path in (m["mmap_path"], m["valid_path"], str(ip)):
                try:
                    Path(path).unlink()
                    print(f"[merge] deleted {path}")
                except FileNotFoundError:
                    pass


if __name__ == "__main__":
    main()
