"""Score v8.4 on the fna_ins_discovery corpus.

Steps:
  1. Adapt the flat fna_ins_discovery schema to V7-real
     (wrap inputs/labels, add arch, junction, spacer metadata).
  2. Filter to K ≥ 3 (V8.4 scope declaration).
  3. Build a shard.
  4. Score bags with v84_main/best.pt → bag_max scores.
  5. Report score distribution + top-N bags for follow-up.

Discovery corpus fields → V7 mapping:
  flat.flank              → inputs.flank
  flat.noncoding_regions  → inputs.noncoding_regions
  flat.bag_id             → transposase_id
  flat.orient             → labels.arch.orient
  no guide_length (unknown) → default 11 for label; m_max scan iterates
                              all Ls=(9..14) regardless
  no ground truth         → labels.guide_span_in_active_noncoding = None;
                              is_positive = None
"""
from __future__ import annotations
import argparse
import json
import shutil
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.model import ChannelBModel
from scripts.build_v7_shard import build_v7_shard


SRC = Path("/global/scratch/users/kh36969/fna_ins_discovery/bags_v4/"
              "insertions_sites.jsonl")
OUT_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                    "v84_evals/fna_ins_discovery")
CKPT = Path("checkpoints/channel_b/v84_main/best.pt")
K_MIN = 3
K_MAX_UNCAPPED = 8   # matches shard --cap-sites 8


IUPAC_TO_ACGT = {"R": "A", "Y": "C", "S": "G", "W": "A",
                     "K": "G", "M": "A", "B": "C", "D": "A",
                     "H": "A", "V": "A", "N": "A"}
NC_REGION_MAX = 250   # match training MAX_NC_LEN band; truncate longer regions


def _sanitize(seq: str) -> str:
    return "".join(IUPAC_TO_ACGT.get(c, c) if c not in "ACGT" else c
                        for c in seq.upper())


def adapt_record(r: dict, n_sites_in_bag: int) -> dict:
    """Convert flat fna_ins_discovery record to V7-real schema.

    Sanitization:
      - Uppercase + IUPAC ambiguity code → ACGT (ViennaRNA can't fold N/R/…);
      - Non-standard chars → 'A' fallback;
      - Truncate any nc region > NC_REGION_MAX to keep ViennaRNA pf() O(N^3)
        cost bounded (training data was 120-250 bp per region).
    """
    flank = _sanitize(r["flank"])
    # Cap: first 2 regions × 250bp each (matches training distribution).
    # Fold cost is O(concat_len^3); this keeps per-bag folds bounded.
    nc_regions = [_sanitize(reg)[:NC_REGION_MAX]
                      for reg in r["noncoding_regions"][:2]]
    # Ensure at least min-length nc region (loader needs nc_len ≥ MAX_L).
    if not nc_regions or all(len(x) < 20 for x in nc_regions):
        nc_regions = [_sanitize(r["noncoding_regions"][0])[:NC_REGION_MAX]
                          if r["noncoding_regions"] else "A" * 120]
    orient = r.get("orient", "fwd")
    return {
        "site_id":        r["site_id"],
        "transposase_id": r["bag_id"],
        "ncrna_id":       r["ncrna_id"],
        "inputs": {
            "flank":             flank,
            "noncoding_regions": nc_regions,
        },
        "labels": {
            "is_positive":     0,         # dummy — inference-only; loader requires castable-int
            "is_planted":      False,
            "negative_mode":   "discovery",
            "target_position_in_flank":    [60, 71],  # default, not used
            "planted_start":               None,
            "guide_span_in_active_noncoding": None,   # unknown
            "guide_length":                11,        # nominal; m_max scans all L
            "planted_m":                   11,
            "m_at_planted":                11,
            "n_mismatches":                0,
            "num_noncoding_regions":       len(nc_regions),
            "active_noncoding_index":      0,
            "ncrna_length":                sum(len(x) for x in nc_regions),
            "guide_dna":                   None,
            "junction_position_in_flank":  r.get("insertion_point_in_flank", 60),
            "arch": {
                "n_sites":                     n_sites_in_bag,
                "nc_multi_region_scoring":     "concat_with_N_spacer",
                "nc_homology_rate":            1.0,
                "gc_target":                   0.5,
                "orient":                      orient,
                "orient_p_same":               1.0,
                "is_reversed":                 (orient == "rc"),
                "target_m_at_planted":         11,
            },
        },
        "generator_metadata": {
            "data_source":                  r.get("data_source", "fna_ins_discovery"),
            "build_date":                    r.get("build_date", "unknown"),
            "generator_version_or_commit":   r.get("generator_version_or_commit", "unknown"),
            "flank_pool_source":             "fna_ins_discovery",
            "nc_multi_region_scoring":       "concat_with_N_spacer",
            "reversed_flow":                 True,
            "concat_spacer_len":             13,   # MAX_L - 1
            "max_l_at_generation":           14,
            "nc_planted_positions":          [],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-min", type=int, default=K_MIN,
                    help="V8.4 K≥3 scope; skip bags below")
    ap.add_argument("--dry-run", action="store_true",
                    help="Only adapt + write JSONL; skip shard/scoring")
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    adapted_jl = OUT_ROOT / "adapted.jsonl"
    shard_dir  = OUT_ROOT / "adapted_shard"

    # Read all records, group by bag, filter K
    print(f"# reading {SRC}")
    bags = defaultdict(list)
    for line in open(SRC):
        r = json.loads(line)
        bags[r["bag_id"]].append(r)
    all_k = [len(bs) for bs in bags.values()]
    print(f"# n_bags_total = {len(bags)}  n_records = {sum(all_k)}")
    print(f"# K distribution: min={min(all_k)}  p50={int(np.median(all_k))}  "
          f"max={max(all_k)}")

    kept_bags = {bid: bs for bid, bs in bags.items() if len(bs) >= args.k_min}
    dropped = len(bags) - len(kept_bags)
    kept_records = sum(len(bs) for bs in kept_bags.values())
    print(f"# K ≥ {args.k_min}: kept {len(kept_bags)} bags ({kept_records} records)  "
          f"dropped {dropped} bags")

    # Adapt + subsample-if-needed + write. For K > cap_sites, take 8
    # sites (deterministic hash-based pick so output is reproducible).
    import random
    rng = random.Random(0)
    n_subsampled = 0
    with open(adapted_jl, "w") as out:
        for bid, records in kept_bags.items():
            if len(records) > K_MAX_UNCAPPED:
                records = rng.sample(records, K_MAX_UNCAPPED)
                n_subsampled += 1
            k = len(records)
            for r in records:
                out.write(json.dumps(adapt_record(r, k)) + "\n")
    print(f"# subsampled {n_subsampled}/{len(kept_bags)} bags to K={K_MAX_UNCAPPED}")
    print(f"# wrote {adapted_jl}\n")

    if args.dry_run:
        return

    # Build shard (resume-friendly: existing NPZs are skipped, so a
    # timed-out prior run continues instead of restarting from zero).
    print(f"# building shard → {shard_dir} (resume-capable)")
    build_v7_shard(adapted_jl, shard_dir, min_sites=args.k_min,
                       cap_sites=K_MAX_UNCAPPED,
                       family_label="fna_ins_discovery")
    print()

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    hp = ck.get("args", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                            n_heads=hp.get("n_heads", 4),
                            n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"# loaded {CKPT}  epoch={ck.get('epoch','?')}  device={device}\n")

    # Score
    ds = ChannelBDataset(adapted_jl, shard_dir, preload_mt=True,
                            cache_dir=OUT_ROOT / "cache",
                            sort_by_nc_len=False)
    print(f"# scoring {len(ds)} bags ...", flush=True)
    results = []
    n_fail = 0
    for i in range(len(ds)):
        bid = ds._bag_index[i][0]
        try:
            b = ds[i]
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL loader] bag {i} ({bid}): {type(e).__name__}: "
                  f"{str(e)[:200]}", flush=True)
            continue
        n_sites = int(b.site_mask.sum().item())
        try:
            x = b.x.unsqueeze(0).to(device)
            sm = b.site_mask.unsqueeze(0).to(device)
            if not torch.isfinite(x).all():
                print(f"  [SKIP non-finite x] bag {i} ({bid}) "
                      f"n_sites={n_sites}", flush=True)
                n_fail += 1
                continue
            with torch.no_grad():
                pred = model(x, sm).squeeze(0)
            top_p = int(pred.argmax().item())
            top_v = float(pred.max().item())
        except Exception as e:
            n_fail += 1
            print(f"  [FAIL forward] bag {i} ({bid}) n_sites={n_sites}: "
                  f"{type(e).__name__}: {str(e)[:200]}", flush=True)
            continue
        results.append({"bag_id": bid, "n_sites": n_sites,
                              "top_p": top_p, "bag_max": top_v})
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(ds)}] scored (fails={n_fail})", flush=True)

    # Save + report
    out_tsv = OUT_ROOT / "scores.tsv"
    with open(out_tsv, "w") as out:
        out.write("bag_id\tn_sites\ttop_p\tbag_max\n")
        for r in sorted(results, key=lambda x: -x["bag_max"]):
            out.write(f"{r['bag_id']}\t{r['n_sites']}\t"
                        f"{r['top_p']}\t{r['bag_max']:.4f}\n")
    print(f"# wrote {out_tsv}\n")

    # Distribution report
    vals = [r["bag_max"] for r in results]
    vals_sorted = sorted(vals, reverse=True)
    print(f"## bag_max distribution across {len(results)} discovery bags")
    print(f"  min={min(vals):+.3f}  p05={np.percentile(vals,5):+.3f}  "
          f"p25={np.percentile(vals,25):+.3f}  p50={statistics.median(vals):+.3f}  "
          f"p75={np.percentile(vals,75):+.3f}  p95={np.percentile(vals,95):+.3f}  "
          f"max={max(vals):+.3f}  mean={statistics.mean(vals):+.3f}")

    # Split by K-bucket
    print(f"\n## bag_max p50 by n_sites bucket (K)")
    kbuckets = {"K=3-4": [], "K=5-8": [], "K=9-16": [],
                    "K=17-32": [], "K=33+": []}
    for r in results:
        k = r["n_sites"]
        if k <= 4:   kbuckets["K=3-4"].append(r["bag_max"])
        elif k <= 8: kbuckets["K=5-8"].append(r["bag_max"])
        elif k <= 16: kbuckets["K=9-16"].append(r["bag_max"])
        elif k <= 32: kbuckets["K=17-32"].append(r["bag_max"])
        else:        kbuckets["K=33+"].append(r["bag_max"])
    for name, vs in kbuckets.items():
        if vs:
            print(f"  {name:>10s}  n={len(vs):5d}  "
                  f"p50={statistics.median(vs):+.3f}  "
                  f"p95={np.percentile(vs,95):+.3f}  "
                  f"max={max(vs):+.3f}")

    # Top-20 for follow-up
    print(f"\n## top 20 bags by bag_max")
    top_sorted = sorted(results, key=lambda x: -x["bag_max"])[:20]
    for i, r in enumerate(top_sorted):
        print(f"  #{i+1:2d}  bag_id={r['bag_id']}  n_sites={r['n_sites']}  "
              f"bag_max={r['bag_max']:.3f}  top_p={r['top_p']}")

    print(f"\n# discovery scoring complete")


if __name__ == "__main__":
    main()
