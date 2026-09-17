"""Bootstrap-subsample builder for Durrant v2 real-flank JSONLs.

Groups by bridge_RNA_ID (per user's 2026-09-13 directive — Programmed
mixing across bridge RNAs would test a semantically non-existent condition).

For each group, draw K bootstrap subsets, each of size n ~ Uniform{3..8}.
Each subset becomes a bootstrap "bag" with n_sites=n; sites within a
bootstrap are drawn WITHOUT replacement; bags overlap across bootstraps
(standard bootstrap-of-subsets).

Emits v7-compatible JSONL where each line is one site record (loader groups
by transposase_id). All sites in a bootstrap bag share the same
transposase_id.

Outputs (default seed 0):
  durrant_wt_bootstrap_K50.jsonl         (50 bags × ~5.5 sites/bag ≈ 275 site rows)
  durrant_programmed_bootstrap_K50.jsonl (8 groups × 50 bags × ~5.5 sites ≈ 2200 rows)

CI computed downstream via K bootstrap-quantile (2.5/50/97.5) across the K
per-group scores — the K bags are NOT independent observations so no paired
CI over K.

Usage:
    python -m scripts.build_durrant_bootstrap_bags --K 50 --seed 0 \\
        --n-min 3 --n-max 8
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DATA_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/IS110_gold/inference")
WT_IN     = DATA_ROOT / "durrant_wt_realbg_v2.jsonl"
PROG_IN   = DATA_ROOT / "durrant_programmed_realbg_v2.jsonl"

BUILD_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_grouped(path: Path) -> dict[str, list[dict]]:
    """Return {bridge_RNA_ID: [site_record, ...]}."""
    groups = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        groups[r["labels"]["durrant_bridge_rna_id"]].append(r)
    return groups


def build_bootstrap_bags(groups: dict[str, list[dict]], K: int, seed: int,
                              n_min: int, n_max: int, dataset_tag: str
                              ) -> list[dict]:
    """For each group, draw K bootstrap subsets of size ~U{n_min..n_max}.
    Return a flat list of site records (with rewritten transposase_id)."""
    rng = random.Random(seed)
    all_recs = []
    for brna, sites in sorted(groups.items()):
        pool_n = len(sites)
        if pool_n < n_min:
            print(f"  SKIP bridge RNA {brna!r} — only {pool_n} sites (< n_min={n_min})")
            continue
        for k in range(K):
            # Cap n_subsample at pool_n so we don't over-draw a small pool
            n_sub_max = min(n_max, pool_n)
            n_sub = rng.randint(n_min, n_sub_max)
            picks = rng.sample(sites, n_sub)
            bag_id = f"{dataset_tag}_{brna}_boot{k:04d}"
            for s_i, src in enumerate(picks):
                rec = json.loads(json.dumps(src))   # deep copy
                rec["transposase_id"] = bag_id
                rec["site_id"] = f"{bag_id}_site{s_i:02d}"
                rec["ncrna_id"] = f"{bag_id}_ncrna"
                # Overwrite arch.n_sites to match this bootstrap bag size
                rec["labels"]["arch"]["n_sites"] = n_sub
                # Provenance: which bridge RNA + which bootstrap
                rec["labels"]["bootstrap_group"] = brna
                rec["labels"]["bootstrap_index"] = k
                rec["labels"]["bootstrap_source_site_id"] = src["site_id"]
                # generator_metadata — record subsampling info
                rec["generator_metadata"] = dict(src.get("generator_metadata", {}))
                rec["generator_metadata"].update({
                    "data_source":               f"{dataset_tag}_bootstrap",
                    "build_date":                BUILD_DATE,
                    "bootstrap_seed":            seed,
                    "bootstrap_n_subsample":     n_sub,
                    "bootstrap_pool_size":       pool_n,
                    "bootstrap_from":            src["generator_metadata"]["data_source"],
                })
                all_recs.append(rec)
    return all_recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=50,
                     help="Bootstrap draws per bridge-RNA group (pilot: 50)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-min", type=int, default=3)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--out-dir", type=Path, default=DATA_ROOT)
    args = ap.parse_args()

    for label, in_path, tag, out_name in [
        ("WT",         WT_IN,   "durrant_wt",         f"durrant_wt_bootstrap_K{args.K}.jsonl"),
        ("Programmed", PROG_IN, "durrant_programmed", f"durrant_programmed_bootstrap_K{args.K}.jsonl"),
    ]:
        print(f"\n=== {label}: {in_path.name} ===")
        groups = load_grouped(in_path)
        print(f"  loaded {sum(len(v) for v in groups.values())} sites across "
              f"{len(groups)} bridge RNAs")
        recs = build_bootstrap_bags(groups, K=args.K, seed=args.seed,
                                            n_min=args.n_min, n_max=args.n_max,
                                            dataset_tag=tag)
        # Count bags produced
        n_bags = len({r["transposase_id"] for r in recs})
        n_rows = len(recs)
        out_path = args.out_dir / out_name
        with open(out_path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        print(f"  wrote {n_rows} site rows / {n_bags} bags → {out_path}")
        # n_sites distribution
        from collections import Counter
        ns = Counter()
        by_bag = defaultdict(int)
        for r in recs:
            by_bag[r["transposase_id"]] += 1
        for cnt in by_bag.values():
            ns[cnt] += 1
        print(f"  n_sites distribution across bags: {dict(sorted(ns.items()))}")


if __name__ == "__main__":
    main()
