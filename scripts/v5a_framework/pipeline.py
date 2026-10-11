"""Unified Phase A→B→C→D runner for model-free scorer v3.

Keeps the configuration surface in one place so the main run and the
null-calibration driver see identical settings.

Returns per-bag:
  BagResult
    bag_id
    K_valid
    n_hits_in           — hits before Phase B
    n_hits_kept         — hits after Phase B
    passing             — list[(Cluster, BgReport)] that passed Phase D
    failing             — list[(Cluster, BgReport)] that failed Phase D
    phase_b_log         — RejectionLog (gate reasons)
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v5a_framework.multi_hit_retention import (
    retain_bag, m_floor_retain_default, BagRetention,
    compute_K_valid,
)
from scripts.v5a_framework.motif_gates import (
    apply_phase_b, GateConfig, RejectionLog,
)
from scripts.v5a_framework.cluster_enumeration import (
    enumerate_clusters, Cluster,
)
from scripts.v5a_framework.background_grouping import (
    apply_phase_d_gate, BgConfig, BgReport,
)
from scripts.v5a_framework.verified_coordinates import project_bag, CoordinateReport


DEFAULT_LS = (9, 10, 11, 12, 13, 14)


@dataclass(frozen=True)
class PipelineConfig:
    coordinate_mode: str = "verified"
    Ls: tuple[int, ...] = DEFAULT_LS
    # Phase A
    min_nc_len: int | None = None
    min_flank_len: int | None = None
    # Phase B
    gate: GateConfig = GateConfig()
    # Phase C
    max_offset: int = 5
    min_frac: float = 0.5
    dedup_jaccard_min: float = 0.80
    # Phase D
    bg: BgConfig = BgConfig()


@dataclass
class BagResult:
    bag_id: str
    K_valid: int
    n_hits_in: int
    n_hits_kept: int
    passing: list                 # list[tuple[Cluster, BgReport]]
    failing: list                 # list[tuple[Cluster, BgReport]]
    phase_b_log: RejectionLog
    bag_retention: BagRetention   # for diagnostics
    coordinate_report: CoordinateReport | None = None
    status: str = "ready"


def run_bag(bag_id: str, sites: list[dict],
             cfg: PipelineConfig = PipelineConfig(), *, conditional_null: bool = False) -> BagResult:
    if cfg.coordinate_mode == "verified":
        K, valid, dropped = compute_K_valid(sites, cfg.min_nc_len or max(cfg.Ls),
                                            cfg.min_flank_len or max(cfg.Ls))
        if not any(sites[i].get("scorer_evidence") for i in valid):
            status = "needs_evidence" if K else "no_valid_inputs"
            empty = BagRetention(K, valid, [], dropped)
            report = CoordinateReport(status, {i: "missing_evidence" for i in valid}, {}, 0)
            return BagResult(bag_id, K, 0, 0, [], [], RejectionLog(), empty, report, status)
    bag = retain_bag(sites, cfg.Ls,
                      m_floor_fn=m_floor_retain_default,
                      min_nc_len=cfg.min_nc_len,
                      min_flank_len=cfg.min_flank_len)
    bag_f, log = apply_phase_b(bag, sites, cfg.gate)
    coord_report = None
    if cfg.coordinate_mode == "verified":
        bag_f, coord_report = project_bag(bag_f, sites, allow_conditional_null=conditional_null)
    elif cfg.coordinate_mode != "legacy":
        raise ValueError(f"unknown coordinate mode: {cfg.coordinate_mode}")
    clusters = enumerate_clusters(bag_f,
                                    max_offset=cfg.max_offset,
                                    min_frac=cfg.min_frac,
                                    dedup_jaccard_min=cfg.dedup_jaccard_min)
    flanks_per_site = {i: sites[i]["inputs"]["flank"] for i in bag.valid_site_idx}
    bg_cfg = replace(cfg.bg, method="gapped") if cfg.coordinate_mode == "verified" else cfg.bg
    loci = {i: sites[i].get("scorer_evidence", {}).get("locus", {}) for i in bag.valid_site_idx}
    passing, failing = apply_phase_d_gate(clusters, flanks_per_site, bg_cfg, loci)
    return BagResult(
        bag_id=bag_id, K_valid=bag.K_valid,
        n_hits_in=log.n_hits_in, n_hits_kept=log.n_hits_kept,
        passing=passing, failing=failing,
        phase_b_log=log, bag_retention=bag_f,
        coordinate_report=coord_report,
        status=(coord_report.status if coord_report else "legacy_unverified") if bag.K_valid else "no_valid_inputs",
    )
