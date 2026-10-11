"""Step 2 of the v3 fix queue — audit nc coord comparability across sites.

For each bag in the sweep input:
  - Compare each site's noncoding_regions to site 0.
  - Classify the bag by whether raw nc_pos is comparable across sites:
      coord_comparable    = all sites have same nc region lengths AND
                             same number of regions (substitutions only)
      has_indel          = at least one site has different nc region length
      has_reorder        = at least one site has different number of regions
      incomparable       = either of the above

Also report per-bag:
  n_identical_nc, n_same_lens, nc_lens_mode

Writes:
  logs/audit/nc_coord_audit.tsv       - one row per bag with the diagnostic numbers
  logs/audit/nc_coord_summary.tsv     - aggregate counts

The intent is to decide for each sweep bag whether Phase C clustering on
raw nc_pos is well-defined, and to mark the rest unassessable (not to
back-fix the score for them).
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def classify_bag(sites: list[dict]) -> dict:
    K = len(sites)
    if K == 0:
        return {"K": 0, "class": "empty"}
    nc0 = sites[0]["inputs"]["noncoding_regions"]
    lens0 = [len(x) for x in nc0]
    n_regions_variation = len({len(s["inputs"]["noncoding_regions"]) for s in sites}) > 1
    n_identical = sum(1 for s in sites
                       if s["inputs"]["noncoding_regions"] == nc0)
    n_same_lens = sum(1 for s in sites
                       if [len(x) for x in s["inputs"]["noncoding_regions"]] == lens0)
    cls = "coord_comparable" if (n_same_lens == K and not n_regions_variation) else (
          "has_reorder"     if n_regions_variation else
          "has_indel")
    return {
        "K": K, "class": cls,
        "n_identical_nc": n_identical,
        "n_same_lens": n_same_lens,
        "has_reorder": int(n_regions_variation),
        "nc_regions_s0": len(nc0),
        "nc_total_bp_s0": sum(lens0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--cids", type=Path, required=True,
                      help="file with CIDs to audit (one per line)")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    want = set(args.cids.read_text().split())
    print(f"# auditing {len(want)} bags")

    bags = defaultdict(list)
    with args.jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            bid = r["transposase_id"]
            if bid in want:
                bags[bid].append(r)

    rows = []
    for bid in sorted(bags):
        rows.append({"bag_id": bid, **classify_bag(bags[bid])})

    out = args.outdir / "nc_coord_audit.tsv"
    cols = ["bag_id","K","class","n_identical_nc","n_same_lens",
            "has_reorder","nc_regions_s0","nc_total_bp_s0"]
    with out.open("w") as f:
        w = csv.DictWriter(f, delimiter="\t", fieldnames=cols)
        w.writeheader()
        for r in rows: w.writerow({k: r.get(k, "") for k in cols})
    print(f"# wrote {out}")

    # Summary
    cls_count = Counter(r["class"] for r in rows)
    comparable = cls_count.get("coord_comparable", 0)
    n = len(rows)
    print(f"\n## per-class counts (of {n} bags):")
    for c, k in sorted(cls_count.items(), key=lambda x: -x[1]):
        print(f"  {c:<20s}  {k:>5d}  ({100*k/n:>5.1f}%)")
    print(f"\n## coord-comparable fraction: {comparable}/{n}  ({100*comparable/n:.1f}%)")
    print(f"## INCOMPARABLE (indel or reorder): {n - comparable}/{n}  "
          f"({100*(n-comparable)/n:.1f}%)")

    summary_path = args.outdir / "nc_coord_summary.tsv"
    with summary_path.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["class","n_bags","fraction"])
        for c, k in sorted(cls_count.items(), key=lambda x: -x[1]):
            w.writerow([c, k, f"{k/n:.4f}"])
    print(f"# wrote {summary_path}")


if __name__ == "__main__":
    main()
