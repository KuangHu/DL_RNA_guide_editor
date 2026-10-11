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
from dataclasses import dataclass, field
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v5a_framework.multi_hit_retention import (
    retain_bag, m_floor_retain_default, BagRetention,
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


DEFAULT_LS = (9, 10, 11, 12, 13, 14)


@dataclass(frozen=True)
class PipelineConfig:
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


def run_bag(bag_id: str, sites: list[dict],
             cfg: PipelineConfig = PipelineConfig()) -> BagResult:
    bag = retain_bag(sites, cfg.Ls,
                      m_floor_fn=m_floor_retain_default,
                      min_nc_len=cfg.min_nc_len,
                      min_flank_len=cfg.min_flank_len)
    bag_f, log = apply_phase_b(bag, sites, cfg.gate)
    clusters = enumerate_clusters(bag_f,
                                    max_offset=cfg.max_offset,
                                    min_frac=cfg.min_frac,
                                    dedup_jaccard_min=cfg.dedup_jaccard_min)
    flanks_per_site = {i: sites[i]["inputs"]["flank"] for i in bag.valid_site_idx}
    passing, failing = apply_phase_d_gate(clusters, flanks_per_site, cfg.bg)
    return BagResult(
        bag_id=bag_id, K_valid=bag.K_valid,
        n_hits_in=log.n_hits_in, n_hits_kept=log.n_hits_kept,
        passing=passing, failing=failing,
        phase_b_log=log, bag_retention=bag_f,
    )
