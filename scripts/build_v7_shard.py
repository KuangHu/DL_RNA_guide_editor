"""Build a MatchTable shard from a v7 JSONL.

v7 JSONL structure (per bag_v2.to_v42_jsonl v7-mode):
  - transposase_id groups sites into bags
  - Each site record has inputs.flank (120 nt) and inputs.noncoding_regions=[nc]
  - labels.planted_start / .guide_length populate SiteRecord.gold_nc / .gold_L
    (used by builders that care about gold coords — the shard itself only
    needs flank + nc for _compute_site_arrays).

Emits:
  shard_dir/{tnp_id}.npz      one file per bag with per-site m_max/argmax arrays
  shard_dir/_index.json       tnp_ids + orients + Ls + meta
  shard_dir/_seqs.json        {tnp_id: {nc, sites[serialized]}}

Usage:
    python -m scripts.build_v7_shard \\
        --jsonl /path/to/v7_bags.jsonl \\
        --shard-dir /path/to/shard \\
        [--max-bags N] [--min-sites 1] [--cap-sites 8]
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v5a_framework.match_table import (
    ORIENTS, DEFAULT_LS, SiteRecord, TnpRecord, _build_common,
)
from model.channel_b.constants import MAX_L, Ls

# Multi-region concat parameters — MUST match model/channel_b/data.py to
# keep shard site-position indices consistent with loader.
# See data.py's _MULTI_REGION_SPACER_LEN — this is the same constant.
_MULTI_REGION_SPACER_LEN = MAX_L - 1
assert MAX_L == max(Ls), (
    f"MAX_L != max(Ls); v7 shard multi-region spacer formula stale.")
assert max(Ls) < _MULTI_REGION_SPACER_LEN + 2


def build_v7_shard(jsonl_path: Path, shard_dir: Path,
                    max_bags: int | None = None,
                    min_sites: int = 1, cap_sites: int | None = None,
                    family_label: str = "v7") -> None:
    """Fail-loud policy (2026-09-11 rev):

    Silent drops led to a KeyError-at-loader class of bug (per-site nc
    variant → shard site missing → training crash). Every drop path in
    this builder is now either a hard raise (multi-region, cap_sites
    exceeded, arch.n_sites mismatch) or accepted + reported (per-site
    nc variants under hom<1.0 — legitimate v7 semantics).

    Raises RuntimeError on:
      - a site record with len(noncoding_regions) != 1
      - a bag whose collected site count differs from its arch.n_sites
      - cap_sites is set and any bag has more sites than cap_sites
      - min_sites is set and any bag has fewer sites than min_sites

    Accepts + reports:
      - per-site nc variants under hom<1.0 (counter printed)
    """
    tnp_nc: dict[str, str] = {}
    tnp_arch_n: dict[str, int] = {}
    tnp_sites: dict[str, list[SiteRecord]] = defaultdict(list)
    n_records = 0
    n_persite_nc_variants = 0
    # Collect violations here rather than raising on first — for large corpora
    # (160k bags, 90-min shard builds) we want to see ALL bad bags in one report
    # rather than die on the first and restart. Any non-empty list triggers a
    # summary raise at the end.
    violations: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            n_records += 1
            tnp = r["transposase_id"]
            sid = r.get("site_id", "?")
            ncs = r["inputs"].get("noncoding_regions", [])
            if len(ncs) == 0:
                violations.append(
                    f"empty_noncoding_regions: {tnp}/{sid} noncoding_regions is empty")
                continue
            if len(ncs) > 1:
                # Multi-region: MUST have arch.nc_multi_region_scoring declared
                # (same rule as data.py loader). Only supported policy:
                # "concat_with_N_spacer". Concat with MAX_L-1 N's here so the
                # shard's per-site m_max/argmax arrays key against the same
                # canonical nc the loader will materialize at read time.
                policy = r["labels"].get("arch", {}).get("nc_multi_region_scoring")
                if policy is None:
                    violations.append(
                        f"multi_region_no_policy: {tnp}/{sid} has {len(ncs)} regions "
                        f"but no arch.nc_multi_region_scoring declared (CANONICAL_BAG_SPEC §3.2)")
                    continue
                if policy != "concat_with_N_spacer":
                    violations.append(
                        f"multi_region_bad_policy: {tnp}/{sid} arch.nc_multi_region_scoring="
                        f"{policy!r}; only 'concat_with_N_spacer' is implemented")
                    continue
                nc = ("N" * _MULTI_REGION_SPACER_LEN).join(ncs)
            else:
                nc = ncs[0]
            arch_n = r["labels"].get("arch", {}).get("n_sites")
            if tnp not in tnp_nc:
                tnp_nc[tnp] = nc
                if arch_n is None:
                    violations.append(
                        f"missing_arch_n_sites: {tnp}/{sid} first-seen site record has no "
                        f"labels.arch.n_sites")
                    continue
                tnp_arch_n[tnp] = int(arch_n)
                if max_bags is not None and len(tnp_nc) > max_bags:
                    tnp_nc.pop(tnp)
                    tnp_arch_n.pop(tnp)
                    break
            elif tnp_nc[tnp] != nc:
                # Legitimate v7 semantics under homology<1.0: each site has
                # its own nc variant, but the loader uses noncoding_regions[0]
                # (first-site nc) as the bag's canonical nc for the whole
                # tensor. `match_table._write_shard` falls back to
                # `_compute_site_arrays(tnp.nc, site.flank, ...)` when
                # site_nc is None — arrays are keyed by first-site nc
                # coordinates for every site, matching what the loader
                # then reads. See docs/CANONICAL_BAG_SPEC.md §3.2.
                n_persite_nc_variants += 1
            labels = r["labels"]
            tnp_sites[tnp].append(SiteRecord(
                site_idx=len(tnp_sites[tnp]),
                flank=r["inputs"]["flank"],
                upstream_flank=None,
                target_flank_start=labels.get("target_position_in_flank", [None])[0]
                    if labels.get("target_position_in_flank") else None,
                gold_nc=labels.get("planted_start"),
                gold_L=labels.get("guide_length"),
            ))

    # Per-bag consistency: collected site count must equal arch.n_sites, and
    # must obey cap_sites / min_sites. Collect all violations then raise once.
    for tnp in list(tnp_sites.keys()):
        got = len(tnp_sites[tnp])
        want = tnp_arch_n.get(tnp)
        if want is None:
            continue   # already flagged as missing_arch_n_sites
        if got != want:
            violations.append(
                f"n_sites_mismatch: {tnp} collected {got} site records but arch.n_sites={want} "
                f"(loader iterates range({want}) → KeyError at training time)")
        if cap_sites is not None and got > cap_sites:
            violations.append(
                f"over_cap: {tnp} has {got} sites, cap_sites={cap_sites}")
        if got < min_sites:
            violations.append(
                f"under_min: {tnp} has {got} sites, min_sites={min_sites}")

    if violations:
        # Summary + first 20 detail lines. Full list preserved in the raise
        # traceback so the sbatch log carries it all — no silent truncation.
        by_kind: dict[str, int] = {}
        for v in violations:
            k = v.split(":", 1)[0]
            by_kind[k] = by_kind.get(k, 0) + 1
        summary = "  ".join(f"{k}={n}" for k, n in sorted(by_kind.items()))
        preview = "\n    ".join(violations[:20])
        more = f"\n    ... (+{len(violations) - 20} more)" if len(violations) > 20 else ""
        raise RuntimeError(
            f"[build_v7_shard] {len(violations)} violation(s) found across {len(tnp_sites)} "
            f"scanned bags — {summary}. First 20:\n    {preview}{more}"
        )

    records = [
        TnpRecord(tnp_id=t, family=family_label, nc=tnp_nc[t], sites=ss)
        for t, ss in tnp_sites.items()
    ]
    print(f"[build_v7_shard] read {n_records} site records → "
          f"{len(records)} bags kept (min_sites={min_sites}, cap_sites={cap_sites}; "
          f"per-site nc variants under hom<1.0={n_persite_nc_variants}; "
          f"NO silent drops)", flush=True)

    meta = {
        "builder": "build_v7_shard",
        "jsonl_path": str(jsonl_path),
        "family_label": family_label,
        "min_sites": min_sites,
        "cap_sites": cap_sites,
    }
    _build_common(records, Path(shard_dir), ORIENTS, DEFAULT_LS, meta,
                     progress_every=50)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=Path, required=True)
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--max-bags", type=int, default=None)
    ap.add_argument("--min-sites", type=int, default=1)
    ap.add_argument("--cap-sites", type=int, default=None,
                     help="If set, raise if any bag has more sites than this. "
                          "Default None disables the check (silent truncation is banned).")
    ap.add_argument("--family-label", type=str, default="v7")
    args = ap.parse_args()

    t0 = time.perf_counter()
    build_v7_shard(args.jsonl, args.shard_dir,
                   max_bags=args.max_bags,
                   min_sites=args.min_sites, cap_sites=args.cap_sites,
                   family_label=args.family_label)
    dt = time.perf_counter() - t0
    print(f"[build_v7_shard] done in {dt:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
