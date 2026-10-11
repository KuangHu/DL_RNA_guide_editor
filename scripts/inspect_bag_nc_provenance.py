"""Step 2 (continued) — per-site nc provenance inspection.

For a specified bag, dump each site's nc segments with:
  - segment index, length, direction (if available)
  - the segment-local position of a specified v3 cluster motif
  - segment identity to site 0's corresponding segment

Does NOT claim 'coord-comparable' or 'incomparable' — that is a homology
question, not a structural one.  This tool prints the raw per-site
segments so a reviewer can decide whether a candidate's reported
nc_pos lands in the SAME homologous segment across supporting sites.

Usage:
  python scripts/inspect_bag_nc_provenance.py \
    --jsonl .../adapted.jsonl \
    --bag CDS03597 \
    --nc-pos 1 --L 14 \
    [--only-sites 0,1,2,3]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


SPACER_N = 13   # nc_concat spacer length (same as multi_hit_retention)


def _per_site(sites: list[dict]) -> list[dict]:
    out = []
    for i, s in enumerate(sites):
        nc_list = s["inputs"]["noncoding_regions"]
        out.append({
            "site": i,
            "n_regions": len(nc_list),
            "segments": [{"idx": j, "len": len(seg), "seq": seg}
                          for j, seg in enumerate(nc_list)],
        })
    return out


def _concat_offsets(nc_list: list[str]) -> list[tuple[int, int]]:
    """Return [(start, end)] of each segment within the N-spacered concat."""
    out = []
    cursor = 0
    for i, seg in enumerate(nc_list):
        out.append((cursor, cursor + len(seg)))
        cursor += len(seg) + (SPACER_N if i < len(nc_list) - 1 else 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--bag", required=True)
    ap.add_argument("--nc-pos", type=int, default=None)
    ap.add_argument("--L", type=int, default=None)
    ap.add_argument("--only-sites", default=None)
    args = ap.parse_args()

    only = set(int(s) for s in args.only_sites.split(",")) if args.only_sites else None

    sites = []
    with args.jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            if r["transposase_id"] == args.bag:
                sites.append(r)
    if not sites:
        print(f"ERROR: bag {args.bag} not found in {args.jsonl}")
        return
    print(f"# bag {args.bag}: K={len(sites)}")

    # Site 0 reference
    nc0 = sites[0]["inputs"]["noncoding_regions"]
    offs0 = _concat_offsets(nc0)
    print(f"\n## site 0 segments")
    for j, (s, e) in enumerate(offs0):
        tag = ""
        if args.nc_pos is not None and s <= args.nc_pos < e:
            tag = f"  <-- nc_pos={args.nc_pos} lands here (segment-local offset {args.nc_pos - s})"
        print(f"  seg{j}: concat[{s}:{e}]  len={e-s}  seq={nc0[j][:60]!r}{tag}")

    # Each other site
    for i, s in enumerate(sites):
        if only is not None and i not in only: continue
        nc = s["inputs"]["noncoding_regions"]
        offs = _concat_offsets(nc)
        same_count = len(nc) == len(nc0)
        same_lens = [len(x) for x in nc] == [len(x) for x in nc0]
        print(f"\n## site {i}  n_regions={len(nc)}  "
              f"same_count_as_s0={same_count}  "
              f"same_lens={same_lens}")
        for j, (ss, ee) in enumerate(offs):
            tag = ""
            if args.nc_pos is not None and ss <= args.nc_pos < ee:
                tag = f"  <-- nc_pos={args.nc_pos} lands here (seg-local {args.nc_pos - ss})"
            # Compare segment to s0 segment at same index (if exists)
            if j < len(nc0):
                a = nc[j]; b = nc0[j]
                ab_len = min(len(a), len(b))
                id_hits = sum(1 for k in range(ab_len) if a[k] == b[k])
                id_pct = 100 * id_hits / ab_len if ab_len > 0 else 0
                cmp_desc = (f"vs s0.seg{j}: id={id_pct:.1f}% over {ab_len} bp"
                            f"  Δlen={len(a) - len(b):+d}")
            else:
                cmp_desc = "no s0 segment at this index"
            print(f"  seg{j}: concat[{ss}:{ee}]  len={ee-ss}  "
                  f"{cmp_desc}  seq={nc[j][:60]!r}{tag}")

        # Also: if nc_pos / L is specified, extract the actual guide window at site i
        if args.nc_pos is not None and args.L is not None:
            concat = ("N" * SPACER_N).join(nc)
            if args.nc_pos + args.L <= len(concat):
                guide_here = concat[args.nc_pos:args.nc_pos + args.L]
                # Also check what segment this falls in
                for j, (ss, ee) in enumerate(offs):
                    if ss <= args.nc_pos < ee:
                        if args.nc_pos + args.L <= ee:
                            location = f"inside seg{j} at seg-local {args.nc_pos - ss}:{args.nc_pos - ss + args.L}"
                        else:
                            location = f"STRADDLES seg{j} into spacer or next segment"
                        break
                else:
                    location = "in a spacer region"
                print(f"  → guide at nc_pos={args.nc_pos} L={args.L}: {guide_here!r}  ({location})")
            else:
                print(f"  → nc too short for nc_pos={args.nc_pos} L={args.L}")


if __name__ == "__main__":
    main()
