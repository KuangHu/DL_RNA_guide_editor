"""Stratify partial-negative FPR by n_planted for both Channel A modes.

Reads the cached MatchTable at $CACHE_DIR + the partial JSONL, adds a
bag_n_planted key to tnp_arch computed as sum(is_planted for site in bag),
then invokes compute_channel_a with stratify_by="bag_n_planted".
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generator_v5.channel_a_v5 import (
    build_v5_positive, _resolve_spec, _parallel_run_variant, compute_channel_a,
)


import os
V5_JSONL = os.environ.get("V5_JSONL",
    "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch/positives_v5_neg5k_partial_v2.jsonl")
SHARD_DIR = os.environ.get("SHARD_DIR",
    "/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a/mt_neg5k_partial_v2")


def main() -> int:
    # 1. Load MatchTable from cache + extract tnp_arch metadata.
    mt, tnp_arch = build_v5_positive(V5_JSONL, SHARD_DIR)

    # 2. Compute per-bag n_planted from the JSONL and inject into tnp_arch.
    bag_n_planted: dict[str, int] = defaultdict(int)
    with open(V5_JSONL) as f:
        for line in f:
            r = json.loads(line)
            if r["labels"].get("is_planted", True):
                bag_n_planted[r["transposase_id"]] += 1
    for tnp in tnp_arch:
        tnp_arch[tnp]["bag_n_planted"] = bag_n_planted.get(tnp, 0)

    print(f"n_planted distribution over bags:")
    from collections import Counter
    print(f"  {dict(sorted(Counter(bag_n_planted.values()).items()))}")

    # 3. For each spec, run scan + stratify by n_planted.
    for spec_name in ("m8", "min_E"):
        spec = _resolve_spec(spec_name)
        print(f"\n=== spec={spec_name} ===")
        peaks = _parallel_run_variant(mt, spec, SHARD_DIR, workers=60)

        for restrict_L in (None, 11):
            r_kwargs = {"stratify_by": "bag_n_planted"}
            if restrict_L is not None:
                r_kwargs["restrict_to"] = {"L": restrict_L}
            res = compute_channel_a(mt, peaks, tnp_arch, **r_kwargs)
            L_tag = f"L={restrict_L}" if restrict_L else "all L"
            print(f"\n{L_tag}:")
            print(f"  {'n_planted':<12s} {'n_tnps':>7s} {'FPR':>8s} {'PPV_pk':>8s} {'PPV_Tnp':>9s} {'exact':>8s}")
            for k in sorted(res, key=lambda x: int(x) if str(x).isdigit() else -1):
                row = res[k]
                print(f"  {str(k):<12s} {row['n_tnps']:>7d} {row['coverage']:>8.4f} "
                      f"{row['ppv_peak']:>8.4f} {row['ppv_tnp']:>9.4f} {row['exact_rate']:>8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
