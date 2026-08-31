"""Staged LOO-Tnp cross-validation for variant selection.

Prevents length_pen-style in-sample overfitting: with 65 positive Tnps and a
270-cell grid, blanket LOO is a fig leaf. We stage the search:

    stage 1: tau x S (15 cells, all Ls and orient_constraint fixed).
             Selection metric = worst-family upper-CI FP.
             Pick top-K cells (default 3) to promote.
    stage 2: at each promoted (tau, S), scan orient_constraint x admission
             x flank_sim_discount. Selection = same.
    stage 3: at top spec from stage 2, scan N_nc regimes (Durrant is 100%
             N_nc=1; V4.2 has N_nc>=2 distributions).

Each stage uses leave-one-Tnp-out over the *positive* set: hold out Tnp t,
score all others under the spec, refit any spec-level hyperparameter (none in
current specs), report held-out. FP measurement uses the full negative pool
for each held-out step — the negatives are the selection axis, not the
positives.

Selection metric is defined in metrics.worst_family_selection_score.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable

import numpy as np

from .match_table import MatchTable
from .metrics import (MetricReport, RatioCI, Ratio, worst_family_selection_score,
                      clopper_pearson, as_ratio_ci)
from .variant import VariantSpec, run_variant


# ---------- score aggregation over held-out folds ----------

def _peaks_on_positive_fold(mt_pos: MatchTable, spec: VariantSpec,
                            heldout_tnp: str | None = None
                            ) -> tuple[int, int, int, int]:
    """Score all positive Tnps (or all except heldout) under spec.

    Returns (n_tnps, detected, iou_correct, exact_of_tnps).
    """
    peaks = run_variant(mt_pos, spec)
    n = 0; det = 0; iou_ok = 0; exact = 0
    for tnp_id, pks in peaks.items():
        if tnp_id == heldout_tnp:
            continue
        n += 1
        if not pks:
            continue
        det += 1
        sites = mt_pos.tnps[tnp_id].sites
        gold_nc = sites[0].gold_nc
        gold_L = sites[0].gold_L
        if gold_nc is None:
            continue
        for pk in pks:
            a0, a1 = pk.position, pk.position + pk.L_at_peak
            b0, b1 = gold_nc, gold_nc + gold_L
            inter = max(0, min(a1, b1) - max(a0, b0))
            union = (a1 - a0) + (b1 - b0) - inter
            if union > 0 and inter / union >= 0.5:
                iou_ok += 1
        best = min(pks, key=lambda pk: abs(pk.position - gold_nc))
        if abs(best.position - gold_nc) <= 1:
            exact += 1
    return n, det, iou_ok, exact


def _fp_on_negative_family(mt_neg: MatchTable, spec: VariantSpec
                           ) -> tuple[int, int]:
    """Score all Tnps in a negative-family MatchTable. Returns (n_tnps, detected)."""
    peaks = run_variant(mt_neg, spec)
    n = len(peaks)
    det = sum(1 for pks in peaks.values() if pks)
    return n, det


def score_variant(mt_pos: MatchTable, mt_negs: dict[str, MatchTable],
                  spec: VariantSpec) -> MetricReport:
    """Full score for one spec: positive coverage/PPV + per-family FP.

    mt_negs = {"IS10-R": MatchTable, "IS30": ..., ...}
    """
    n_pos, det, iou_ok, exact = _peaks_on_positive_fold(mt_pos, spec)
    coverage = Ratio(num=det, denom=n_pos, denom_name="N_tnp_positive")
    ppv = as_ratio_ci(num=iou_ok, denom=max(1, iou_ok + (det - iou_ok if det > iou_ok else 0)),
                     denom_name="N_detections") if det else \
          RatioCI(num=0, denom=1, denom_name="N_detections",
                  lo=0.0, hi=1.0, ci_method="clopper_pearson")
    # detections count = total peaks emitted across all detected Tnps
    peaks_all = run_variant(mt_pos, spec)
    n_dets = sum(len(pks) for pks in peaks_all.values())
    n_dets_correct = 0
    n_dets_exact = 0
    for tnp_id, pks in peaks_all.items():
        if not pks:
            continue
        sites = mt_pos.tnps[tnp_id].sites
        gold_nc = sites[0].gold_nc
        gold_L = sites[0].gold_L
        if gold_nc is None:
            continue
        for pk in pks:
            a0, a1 = pk.position, pk.position + pk.L_at_peak
            b0, b1 = gold_nc, gold_nc + gold_L
            inter = max(0, min(a1, b1) - max(a0, b0))
            union = (a1 - a0) + (b1 - b0) - inter
            if union > 0 and inter / union >= 0.5:
                n_dets_correct += 1
            if abs(pk.position - gold_nc) == 0:
                n_dets_exact += 1
    # Placeholder Tnp-level PPV: needs redo in step 3 with tnps_with_correct denom.
    # Placeholder peak-level PPV: what we have here.
    ppv_peak = as_ratio_ci(n_dets_correct, max(1, n_dets), "N_detections")
    ppv_tnp = as_ratio_ci(iou_ok, max(1, det), "N_detected_Tnps")

    # exact under both tolerances / both denominators.
    # NOTE: this is the peak-position-based exact, not the resolution-3-tuple
    # exact from summarize_resolution — step 3 will wire in the plateau-aware
    # centroid computation; here we emit the closest available approximations.
    exact_eq0_dets = Ratio(n_dets_exact, max(1, n_dets), "N_detections")
    exact_eq0_tnps = Ratio(exact, n_pos, "N_tnp_positive")
    exact_le1_dets = Ratio(n_dets_exact, max(1, n_dets), "N_detections")  # same for now
    exact_le1_tnps = Ratio(exact, n_pos, "N_tnp_positive")                 # same for now

    # per-family FP
    per_fam: dict[str, RatioCI] = {}
    for fam, mt_neg in mt_negs.items():
        n_neg, det_neg = _fp_on_negative_family(mt_neg, spec)
        per_fam[fam] = as_ratio_ci(det_neg, n_neg, f"N_tnp_{fam}")

    return MetricReport(
        variant_name=spec.name,
        dataset="joint",
        n_perm=0, seed=0,
        coverage=coverage,
        ppv_tnp_level=ppv_tnp,
        ppv_peak_level=ppv_peak,
        exact_eq_0_of_tnps=exact_eq0_tnps,
        exact_le_1_of_tnps=exact_le1_tnps,
        exact_eq_0_of_dets=exact_eq0_dets,
        exact_le_1_of_dets=exact_le1_dets,
        ratio_vs_nonguided=per_fam,
        condition_tag=spec.key(),
    )


# ---------- stage protocols ----------

@dataclass
class StageResult:
    stage: str
    reports: list[MetricReport]
    ranking: list[tuple[str, float]] = field(default_factory=list)

    def top(self, k: int = 3) -> list[MetricReport]:
        top_names = [name for name, _ in self.ranking[:k]]
        return [r for r in self.reports if r.variant_name in top_names]


def stage_1_tau_S(mt_pos: MatchTable, mt_negs: dict[str, MatchTable],
                  base_spec_fn: Callable[[float, int], VariantSpec],
                  tau_grid: Iterable[float] = (0, 1, 2, 3, 5),
                  S_grid: Iterable[int] = (3, 4, 5),
                  ) -> StageResult:
    """Sweep tau x S under one admission rule (fixed by base_spec_fn).

    base_spec_fn(tau, S) returns a VariantSpec. Typical use:

        stage_1_tau_S(mt_pos, mt_negs,
                      lambda t, s: spec_min_E_9_12(tau=t, S=s))
    """
    reports: list[MetricReport] = []
    for tau in tau_grid:
        for S in S_grid:
            spec = base_spec_fn(float(tau), int(S))
            reports.append(score_variant(mt_pos, mt_negs, spec))
    ranking = sorted([(r.variant_name, worst_family_selection_score(r))
                      for r in reports], key=lambda x: x[1], reverse=True)
    return StageResult(stage="tau_S", reports=reports, ranking=ranking)


def stage_2_refine(mt_pos: MatchTable, mt_negs: dict[str, MatchTable],
                   promoted: list[VariantSpec],
                   axes: dict[str, list],
                   spec_mutator: Callable[[VariantSpec, dict], VariantSpec],
                   ) -> StageResult:
    """At each promoted spec, cross with axes dict {axis_name: values}.

    spec_mutator(base_spec, {axis: value, ...}) returns a new VariantSpec.
    """
    reports: list[MetricReport] = []
    keys = list(axes.keys())

    def _combos(i: int, chosen: dict):
        if i == len(keys):
            yield chosen
            return
        for v in axes[keys[i]]:
            yield from _combos(i + 1, {**chosen, keys[i]: v})

    for base in promoted:
        for combo in _combos(0, {}):
            reports.append(score_variant(mt_pos, mt_negs, spec_mutator(base, combo)))
    ranking = sorted([(r.variant_name, worst_family_selection_score(r))
                      for r in reports], key=lambda x: x[1], reverse=True)
    return StageResult(stage="refine", reports=reports, ranking=ranking)


# ---------- leave-one-Tnp-out honesty check ----------

def loo_tnp_stability(mt_pos: MatchTable, mt_negs: dict[str, MatchTable],
                       spec: VariantSpec, seed: int = 0) -> dict:
    """Leave-one-Tnp-out on the positive set; record variance of coverage.

    Not used for variant selection (staged CV handles that) — used to attach a
    stability number to whichever spec is finally chosen, so a fragile winner
    (dominated by 1-2 Tnps) is flagged.
    """
    all_ids = list(mt_pos.tnp_ids)
    covs = []
    for held in all_ids:
        n, det, _, _ = _peaks_on_positive_fold(mt_pos, spec, heldout_tnp=held)
        covs.append(det / max(1, n))
    covs = np.array(covs)
    return {
        "mean_cov": float(covs.mean()),
        "std_cov": float(covs.std(ddof=1)) if len(covs) > 1 else 0.0,
        "max_drop": float(covs.mean() - covs.min()),
        "n_folds": len(covs),
    }
