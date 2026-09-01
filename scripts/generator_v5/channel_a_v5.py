"""Channel A on V5 generator output.

Builds a MatchTable from V5 positives JSONL (each bag = one Tnp;
active_noncoding_index picks the shared nc; each site's flank + planted
target on nc supply SiteRecord fields), then runs the historical
Channel A spec (fixed L=11, m>=8, tau=0, S=5) via run_variant.

Reports:
  overall: coverage_rate, PPV_peak_level, PPV_Tnp_level, exact_rate
  per-L stratified (11, 12, 13, 14)
  per-arch axis stratified:
    is_split (False/True)
    orient (fwd/rev)
    N_nc (1/2/3)
    tsd_width (0/2/5/8/9/12)
    tsd_relation (none/before/after/both_sides)
    has_5p_stem_loop_active (False/True)
    ncr_pos_rel_orf (upstream/downstream/inline)

Baseline anchors (Durrant, 65 Tnps):
  coverage 0.338, PPV_peak 0.9565, PPV_Tnp 0.9545, exact 0.3231

Any stratum where PPV drops materially below the pool number reveals a
Channel A architecture-dependence that the generator existed to expose.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.v5a_framework.match_table import (
    DEFAULT_LS, ORIENTS, MatchTable, SiteRecord, TnpRecord,
    _build_common, _write_index, _write_shard, load as load_mt,
)
from scripts.v5a_framework.variant import (
    Peak, spec_m_threshold_L11, spec_min_E_9_12, run_variant,
)


def _resolve_spec(name: str, tau: int | None = None,
                    S: int | None = None, m: int | None = None,
                    negatives: bool = False):
    """Named specs with optional (tau, S, m) overrides."""
    if name == "m8":
        # Mode 2 defaults: fixed L=11, m=8, tau=0, S=5.
        return spec_m_threshold_L11(m=(m if m is not None else 8),
                                       tau=(tau if tau is not None else 0),
                                       S=(S if S is not None else 5))
    if name == "min_E":
        # Mode 1 defaults: min-E over L in {9..12}, E=4, tau=5, S=5.
        return spec_min_E_9_12(E=4.0,
                                 tau=(tau if tau is not None else 5),
                                 S=(S if S is not None else 5))
    if name == "m9":
        # Test the m=9 rule on L=11 (equivalent to E<4 at L=11 for min_E).
        return spec_m_threshold_L11(m=9,
                                       tau=(tau if tau is not None else 0),
                                       S=(S if S is not None else 5))
    raise ValueError(f"unknown spec: {name!r}")


# Worker-scope arg storage (set by initializer).
_W_SHARD_DIR: Path | None = None
_W_ORIENTS: tuple = ORIENTS
_W_LS: tuple = DEFAULT_LS

_W_MT: MatchTable | None = None
_W_SPEC = None


def _worker_init(shard_dir: str, orients: tuple, Ls: tuple) -> None:
    global _W_SHARD_DIR, _W_ORIENTS, _W_LS
    _W_SHARD_DIR = Path(shard_dir)
    _W_ORIENTS = orients
    _W_LS = Ls


def _worker_init_scan(shard_dir: str, spec) -> None:
    """Init for the variant-scan Pool: load MatchTable index + hold spec."""
    global _W_MT, _W_SPEC
    _W_MT = load_mt(shard_dir)
    _W_SPEC = spec


def _worker_scan_chunk(tnp_ids: list[str]) -> dict[str, list[Peak]]:
    """Run the variant scan on a subset of Tnp ids in this worker."""
    # Reuse run_variant's per-Tnp logic by temporarily restricting mt.tnp_ids.
    original = _W_MT.tnp_ids
    try:
        _W_MT.tnp_ids = list(tnp_ids)
        return run_variant(_W_MT, _W_SPEC)
    finally:
        _W_MT.tnp_ids = original


def _worker_write_shard(t: TnpRecord) -> str:
    _write_shard(_W_SHARD_DIR, t, _W_ORIENTS, _W_LS)
    return t.tnp_id


def _parallel_run_variant(mt: MatchTable, spec, shard_dir: str,
                            workers: int) -> dict[str, list[Peak]]:
    """Parallel replacement for the serial run_variant loop over Tnps."""
    tnp_ids = list(mt.tnp_ids)
    n = len(tnp_ids)
    # Chunk so each worker processes a contiguous slab; chunksize picked
    # so ~4 chunks per worker for load balancing.
    n_chunks = max(workers * 4, 1)
    chunk_size = max(1, (n + n_chunks - 1) // n_chunks)
    chunks = [tnp_ids[i:i + chunk_size] for i in range(0, n, chunk_size)]
    print(f"  [variant] parallel scan over {n} Tnps in {len(chunks)} chunks "
          f"({workers} workers)", flush=True)
    peaks_all: dict[str, list[Peak]] = {}
    t0 = time.perf_counter()
    done = 0
    with mp.Pool(workers, initializer=_worker_init_scan,
                   initargs=(shard_dir, spec)) as pool:
        for peaks_chunk in pool.imap_unordered(_worker_scan_chunk, chunks):
            peaks_all.update(peaks_chunk)
            done += len(peaks_chunk)
            dt = time.perf_counter() - t0
            print(f"  [variant] {done}/{n}  {dt:.1f}s "
                  f"({dt/max(done,1)*1000:.0f} ms/tnp)", flush=True)
    return peaks_all


def _parallel_build(records: list[TnpRecord], shard_dir: Path,
                      orients: tuple, Ls: tuple, meta: dict,
                      workers: int) -> MatchTable:
    """Parallel replacement for _build_common's per-Tnp shard write.

    Serial upstream (~51 Tnp/min single-thread) can't cover 50K Tnps
    within a 2 h wall budget. Parallelize the write-shard step across
    a Pool; _write_index still runs centrally.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    print(f"  [match_table] parallel build over {len(records)} Tnps "
          f"with {workers} workers", flush=True)
    with mp.Pool(workers, initializer=_worker_init,
                   initargs=(str(shard_dir), orients, Ls)) as pool:
        for i, _ in enumerate(pool.imap_unordered(_worker_write_shard,
                                                    records, chunksize=16), 1):
            if i % 500 == 0:
                dt = time.perf_counter() - t0
                print(f"  [match_table] {i}/{len(records)} tnps  "
                      f"{dt:.1f}s ({dt/i*1000:.0f} ms/tnp)", flush=True)
    _write_index(shard_dir, records, orients, Ls, meta)
    print(f"  [match_table] index written", flush=True)
    return load_mt(str(shard_dir))


def build_v5_positive(v5_jsonl_path: str, shard_dir: str,
                        orients: tuple = ORIENTS,
                        Ls: tuple = DEFAULT_LS,
                        min_sites: int = 5,
                        cap_sites: int = 5,
                        family_label: str = "v5_positive",
                        workers: int = 1,
                        ) -> tuple[MatchTable, dict]:
    """Build MatchTable from V5 positives JSONL.

    Returns (mt, arch_by_tnp). arch_by_tnp[tnp_id] holds the per-bag
    architecture metadata for downstream stratification.

    workers > 1 parallelizes the per-Tnp shard writes via
    multiprocessing.Pool. Required for 50K Tnps within a 2 h budget.
    """
    tnp_sites: dict[str, list[SiteRecord]] = defaultdict(list)
    tnp_nc: dict[str, str] = {}
    tnp_arch: dict[str, dict] = {}
    with open(v5_jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            # Accept both positives (is_positive=True with real gold_nc) and
            # negatives (is_positive=False with gold_nc=-1). The MatchTable
            # holds the same per-Tnp arrays either way; downstream analysis
            # decides whether coverage means PPV_denom (positives) or FP rate
            # (negatives) based on the sentinel gold_nc value.
            pass
            tnp = r["transposase_id"]
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs):
                a = 0
            nc = ncs[a]
            if tnp not in tnp_nc:
                tnp_nc[tnp] = nc
                arch_meta = dict(r["labels"].get("arch", {}))
                nc_len = int(r["labels"].get("ncrna_length", len(nc)))
                # nc_len bucket for scale-invariance analysis
                if nc_len < 120:
                    arch_meta["nc_len_bucket"] = "070-119"
                elif nc_len < 180:
                    arch_meta["nc_len_bucket"] = "120-179"
                elif nc_len < 240:
                    arch_meta["nc_len_bucket"] = "180-239"
                else:
                    arch_meta["nc_len_bucket"] = "240-300"
                arch_meta["nc_len"] = nc_len
                tnp_arch[tnp] = arch_meta
            elif tnp_nc[tnp] != nc:
                continue
            gs = r["labels"].get("guide_span_in_active_noncoding")
            if not gs:
                continue
            tnp_sites[tnp].append(SiteRecord(
                site_idx=len(tnp_sites[tnp]),
                flank=r["inputs"]["flank"],
                upstream_flank=None,
                target_flank_start=r["labels"].get("planted_start"),
                gold_nc=int(gs[0]),
                gold_L=int(r["labels"]["guide_length"]),
            ))
    records = [TnpRecord(tnp_id=t, family=family_label, nc=tnp_nc[t],
                          sites=ss[:cap_sites])
                for t, ss in tnp_sites.items() if len(ss) >= min_sites]
    print(f"[build_v5] {len(records)} Tnps with >= {min_sites} sites", flush=True)
    meta = {"builder": "build_v5_positive", "src": str(v5_jsonl_path),
            "min_sites": min_sites, "cap_sites": cap_sites,
            "workers": workers}
    if workers > 1:
        mt = _parallel_build(records, Path(shard_dir), orients, Ls, meta, workers)
    else:
        mt = _build_common(records, Path(shard_dir), orients, Ls, meta)
    return mt, tnp_arch


def _iou(p, L_win, gold_nc, gold_L, thresh=0.5) -> bool:
    a0, a1 = p, p + L_win
    b0, b1 = gold_nc, gold_nc + gold_L
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = (a1 - a0) + (b1 - b0) - inter
    return union > 0 and inter / union >= thresh


def _primary_pos(pks) -> float:
    max_S = max(pk.S_all for pk in pks)
    top = [pk.position for pk in pks if pk.S_all == max_S]
    return sum(top) / len(top)


def compute_channel_a(mt: MatchTable, peaks_by_tnp: dict, tnp_arch: dict,
                        stratify_by: str | None = None,
                        restrict_to: dict | None = None) -> dict:
    """Return {stratum -> metrics} where stratum is either 'all' or a
    value taken from tnp_arch[tnp_id][stratify_by].

    restrict_to: optional dict of {arch_key: expected_value} to filter Tnps
    (e.g. {"L": 11} to look only at the L=11 subset where the fixed-L=11
    Channel A spec actually operates without floor effects).
    """
    groups: dict[str, list[str]] = defaultdict(list)
    for tnp_id in mt.tnp_ids:
        if restrict_to:
            arch = tnp_arch.get(tnp_id, {})
            if any(arch.get(k) != v for k, v in restrict_to.items()):
                continue
        if stratify_by is None:
            groups["all"].append(tnp_id)
        else:
            v = tnp_arch.get(tnp_id, {}).get(stratify_by, "unknown")
            groups[str(v)].append(tnp_id)
    out: dict[str, dict] = {}
    for stratum, tnps in groups.items():
        n_tnps = len(tnps)
        covered = 0; total_peaks = 0; peaks_correct = 0
        tnps_with_correct = 0; exact = 0
        for tnp_id in tnps:
            pks = peaks_by_tnp.get(tnp_id, [])
            if not pks:
                continue
            covered += 1
            tnp = mt.tnps[tnp_id]
            gold_nc = tnp.sites[0].gold_nc
            gold_L = tnp.sites[0].gold_L
            any_ok = False
            for pk in pks:
                total_peaks += 1
                if _iou(pk.position, pk.L_at_peak, gold_nc, gold_L):
                    peaks_correct += 1
                    any_ok = True
            if any_ok:
                tnps_with_correct += 1
            pp = _primary_pos(pks)
            if abs(pp - gold_nc) <= 1:
                exact += 1
        out[stratum] = {
            "n_tnps":            n_tnps,
            "covered":           covered,
            "total_peaks":       total_peaks,
            "peaks_correct":     peaks_correct,
            "tnps_with_correct": tnps_with_correct,
            "exact":             exact,
            "coverage":          covered / max(1, n_tnps),
            "ppv_peak":          peaks_correct / max(1, total_peaks),
            "ppv_tnp":           tnps_with_correct / max(1, covered),
            "exact_rate":        exact / max(1, n_tnps),
        }
    return out


def _print_table(title: str, results: dict, sort_key=None) -> None:
    print(f"\n=== {title} ===")
    print(f"  {'stratum':<20s} {'n_tnps':>7s} {'covered':>8s} "
          f"{'coverage':>9s} {'ppv_peak':>9s} {'ppv_tnp':>8s} {'exact_rate':>11s}")
    strata = sorted(results.items(), key=(sort_key or (lambda kv: kv[0])))
    for s, m in strata:
        print(f"  {s:<20s} {m['n_tnps']:>7d} {m['covered']:>8d} "
              f"{m['coverage']:>9.4f} {m['ppv_peak']:>9.4f} "
              f"{m['ppv_tnp']:>8.4f} {m['exact_rate']:>11.4f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5-jsonl", required=True)
    ap.add_argument("--shard-dir", required=True)
    ap.add_argument("--report-out", default=None)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--spec", default="m8",
                     choices=["m8", "min_E", "m9"],
                     help="Channel A spec base")
    ap.add_argument("--tau", type=int, default=None, help="Override spec tau")
    ap.add_argument("--S", type=int, default=None, help="Override spec S")
    ap.add_argument("--m", type=int, default=None, help="Override spec m (m8/m9 only)")
    ap.add_argument("--grid", action="store_true",
                     help="Sweep (tau, S) grid on the cached MatchTable")
    ap.add_argument("--treat-as-negatives", action="store_true",
                     help="Interpret coverage as FP rate; skip PPV meaningfulness")
    args = ap.parse_args()

    shard_dir_path = Path(args.shard_dir)
    if (shard_dir_path / "_index.json").exists():
        print(f"[chA-v5] loading cached MatchTable from {args.shard_dir}", flush=True)
        mt = load_mt(str(shard_dir_path))
        # Re-parse JSONL only to reconstruct arch metadata (light — arch dicts
        # are small, but we do skip the whole line if we can).
        print(f"[chA-v5] re-reading arch metadata from JSONL", flush=True)
        tnp_arch: dict = {}
        with open(args.v5_jsonl) as f:
            for line in f:
                r = json.loads(line)
                tnp = r["transposase_id"]
                if tnp in tnp_arch:
                    continue
                arch_meta = dict(r["labels"].get("arch", {}))
                nc_len = int(r["labels"].get("ncrna_length", 0))
                if nc_len < 120:
                    arch_meta["nc_len_bucket"] = "070-119"
                elif nc_len < 180:
                    arch_meta["nc_len_bucket"] = "120-179"
                elif nc_len < 240:
                    arch_meta["nc_len_bucket"] = "180-239"
                else:
                    arch_meta["nc_len_bucket"] = "240-300"
                arch_meta["nc_len"] = nc_len
                tnp_arch[tnp] = arch_meta
    else:
        print(f"[chA-v5] building MatchTable from {args.v5_jsonl} "
              f"({args.workers} workers)", flush=True)
        mt, tnp_arch = build_v5_positive(args.v5_jsonl, args.shard_dir,
                                           workers=args.workers)
    if args.grid:
        # (tau, S) sweep on the cached MatchTable. Report only overall +
        # L=11 per config; skip the per-arch stratifications for size.
        grid_tau = [0, 1, 2, 3, 5]
        grid_S = [3, 4, 5]
        grid_out: dict = {}
        for tau in grid_tau:
            for S in grid_S:
                cfg = f"{args.spec}_tau{tau}_S{S}"
                print(f"\n[chA-v5][grid] {cfg}", flush=True)
                spec = _resolve_spec(args.spec, tau=tau, S=S,
                                       negatives=args.treat_as_negatives)
                if args.workers > 1:
                    peaks = _parallel_run_variant(mt, spec, args.shard_dir,
                                                     args.workers)
                else:
                    peaks = run_variant(mt, spec)
                res_all = compute_channel_a(mt, peaks, tnp_arch, stratify_by=None)
                res_L11 = compute_channel_a(mt, peaks, tnp_arch,
                                              stratify_by=None,
                                              restrict_to={"L": 11})
                res_L = compute_channel_a(mt, peaks, tnp_arch, stratify_by="L")
                grid_out[cfg] = {"tau": tau, "S": S, "overall": res_all,
                                   "overall_L11": res_L11, "by_L": res_L}
                pk = res_all.get("all", {})
                pkL = res_L11.get("all", {})
                print(f"  overall cov={pk.get('coverage',0):.4f} "
                      f"L=11 cov={pkL.get('coverage',0):.4f}", flush=True)
        if args.report_out:
            Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
            with open(args.report_out, "w") as f:
                json.dump({"grid": grid_out, "spec_base": args.spec}, f, indent=2)
            print(f"[chA-v5] grid report written to {args.report_out}")
        return 0

    tau_str = str(args.tau) if args.tau is not None else "default"
    S_str = str(args.S) if args.S is not None else "default"
    m_str = str(args.m) if args.m is not None else "default"
    print(f"[chA-v5] running Channel A spec={args.spec} tau={tau_str} S={S_str} m={m_str}",
          flush=True)
    spec = _resolve_spec(args.spec, tau=args.tau, S=args.S, m=args.m,
                          negatives=args.treat_as_negatives)
    if args.workers > 1:
        peaks_by_tnp = _parallel_run_variant(mt, spec, args.shard_dir,
                                                args.workers)
    else:
        peaks_by_tnp = run_variant(mt, spec)

    # Overall
    overall = compute_channel_a(mt, peaks_by_tnp, tnp_arch, stratify_by=None)
    _print_table("overall vs Durrant anchor (0.338 / 0.9565 / 0.9545 / 0.3231)",
                    overall)

    # Per-arch stratifications on the FULL corpus
    all_reports = {"overall": overall}
    for axis in ("L", "is_split", "orient", "n_nc", "tsd_width",
                   "tsd_relation", "has_5p_stem_loop_active", "ncr_pos_rel_orf",
                   "nc_len_bucket"):
        r = compute_channel_a(mt, peaks_by_tnp, tnp_arch, stratify_by=axis)
        all_reports[axis] = r
        _print_table(f"stratified by {axis} (all L)", r)

    # Per-arch stratifications RESTRICTED TO L=11 (per user directive
    # 2026-08-31: fixed-L=11 Channel A spec floors at L=13, L=14, so
    # arch-axis analysis is only meaningful within the L=11 subset).
    print("\n\n" + "=" * 60)
    print("=== Per-arch stratifications restricted to L=11 subset ===")
    print("=" * 60)
    overall_L11 = compute_channel_a(mt, peaks_by_tnp, tnp_arch,
                                      stratify_by=None, restrict_to={"L": 11})
    all_reports["overall_L11"] = overall_L11
    _print_table("overall L=11 only", overall_L11)
    for axis in ("is_split", "orient", "n_nc", "tsd_width",
                   "tsd_relation", "has_5p_stem_loop_active", "ncr_pos_rel_orf",
                   "nc_len_bucket"):
        r = compute_channel_a(mt, peaks_by_tnp, tnp_arch, stratify_by=axis,
                                restrict_to={"L": 11})
        all_reports[f"{axis}_L11"] = r
        _print_table(f"stratified by {axis} (L=11 only)", r)

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump(all_reports, f, indent=2)
        print(f"\n[chA-v5] report written to {args.report_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
