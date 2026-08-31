"""MetricReport contract — every reported number carries its denominator + CI.

Enforcement design:
    - `Ratio` requires a named `denom_name` string ("N_tnp" / "N_detections" / ...).
      No naked float ratios anywhere in the framework.
    - `RatioCI` extends Ratio with a lower/upper 95% bound. For proportions
      (detections/Tnps), CI is Clopper-Pearson. For real/null ratios, CI is
      Tnp-clustered bootstrap.
    - `MetricReport` is frozen; missing a required field is a TypeError, not a
      convention we can forget.

Categorical FP fields are stored per-family so worst-family selection can look
up the family axis without recomputing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from scipy.stats import beta


# ---------- primitive types ----------

@dataclass(frozen=True)
class Ratio:
    """num / denom with named denominator. Never bare floats."""
    num: int
    denom: int
    denom_name: str

    @property
    def value(self) -> float:
        return self.num / max(1, self.denom)

    def __repr__(self) -> str:
        return f"{self.value:.4f} ({self.num}/{self.denom} [{self.denom_name}])"


@dataclass(frozen=True)
class RatioCI:
    """Ratio with lower/upper 95% CI + CI method label."""
    num: int
    denom: int
    denom_name: str
    lo: float
    hi: float
    ci_method: str        # "clopper_pearson" | "tnp_clustered_bootstrap"

    @property
    def value(self) -> float:
        return self.num / max(1, self.denom)

    def __repr__(self) -> str:
        return (f"{self.value:.4f} ({self.num}/{self.denom} [{self.denom_name}], "
                f"95% {self.ci_method} [{self.lo:.4f}, {self.hi:.4f}])")


# ---------- CI helpers ----------

def clopper_pearson(num: int, denom: int, alpha: float = 0.05) -> tuple[float, float]:
    """95% Clopper-Pearson CI for a binomial proportion."""
    if denom == 0:
        return (0.0, 1.0)
    lo = 0.0 if num == 0 else float(beta.ppf(alpha / 2, num, denom - num + 1))
    hi = 1.0 if num == denom else float(beta.ppf(1 - alpha / 2, num + 1, denom - num))
    return (lo, hi)


def as_ratio_ci(num: int, denom: int, denom_name: str, alpha: float = 0.05) -> RatioCI:
    lo, hi = clopper_pearson(num, denom, alpha)
    return RatioCI(num=num, denom=denom, denom_name=denom_name,
                   lo=lo, hi=hi, ci_method="clopper_pearson")


def tnp_clustered_ratio_bootstrap(
    real_hits_by_tnp: Mapping[str, int],
    real_denom_by_tnp: Mapping[str, int],
    null_hits_by_tnp: Mapping[str, int],
    null_denom_by_tnp: Mapping[str, int],
    n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Bootstrap CI for a real/null coverage ratio, Tnp-clustered.

    Real and null must be indexed by the same set of Tnp ids; resampling
    draws Tnps with replacement and recomputes the aggregate ratio.

    Returns (point, lo, hi).
    """
    tnp_ids = list(real_hits_by_tnp.keys())
    rh = np.array([real_hits_by_tnp[t] for t in tnp_ids])
    rd = np.array([real_denom_by_tnp[t] for t in tnp_ids])
    nh = np.array([null_hits_by_tnp[t] for t in tnp_ids])
    nd = np.array([null_denom_by_tnp[t] for t in tnp_ids])

    def _ratio(idx):
        num = rh[idx].sum() / max(1, rd[idx].sum())
        den = nh[idx].sum() / max(1e-9, nd[idx].sum())
        return num / max(1e-9, den)

    point = _ratio(np.arange(len(tnp_ids)))
    rng = np.random.default_rng(seed)
    boots = np.zeros(n_boot)
    N = len(tnp_ids)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = _ratio(idx)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (point, lo, hi)


# ---------- nested bootstrap (outer Tnp / inner random-5) ----------

def nested_bootstrap_ratio_ci(
    per_tnp_scoring: "callable",
    tnp_ids: list[str],
    n_inner_draws: int = 20,
    n_outer_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Two-level bootstrap that propagates BOTH sources of variance:
      - outer: resample Tnps (with replacement), captures Tnp-clustered variance
      - inner: for each drawn Tnp, pick one of `n_inner_draws` random-5 indices
               (captures within-Tnp sampling variance from the 5-flank subsample)

    `per_tnp_scoring(tnp_id, draw_idx) -> tuple[num, denom]` — returns a
    (numerator, denominator) pair for that Tnp under the given inner draw.
    The aggregate ratio is sum(num) / max(1, sum(denom)).

    Returns (point_estimate, ci_lo, ci_hi).

    Correct construction per the 'not a second variance source' correction:
    the k=20 inner draws do NOT get compounded with an outer Tnp-clustered
    bootstrap; instead, one inner draw is picked per drawn Tnp per outer
    iteration, so both variances propagate exactly once.
    """
    rng = np.random.default_rng(seed)
    N = len(tnp_ids)

    # Point estimate: use draw 0 for each Tnp
    nums0 = np.zeros(N); dens0 = np.zeros(N)
    for i, t in enumerate(tnp_ids):
        n, d = per_tnp_scoring(t, 0)
        nums0[i] = n; dens0[i] = d
    point = float(nums0.sum() / max(1, dens0.sum()))

    # Outer + inner bootstrap
    boots = np.empty(n_outer_boot)
    for b in range(n_outer_boot):
        idx = rng.integers(0, N, size=N)
        inner = rng.integers(0, n_inner_draws, size=N)
        num_sum = 0.0; den_sum = 0.0
        for k in range(N):
            n, d = per_tnp_scoring(tnp_ids[idx[k]], int(inner[k]))
            num_sum += n; den_sum += d
        boots[b] = num_sum / max(1e-9, den_sum)
    lo = float(np.quantile(boots, alpha / 2))
    hi = float(np.quantile(boots, 1 - alpha / 2))
    return (point, lo, hi)


# ---------- 3-tuple resolution report + both exact tolerances ----------

@dataclass(frozen=True)
class ResolutionReport:
    """Per-detection resolution summary that replaces the single 'exact-hit'
    percentage. Reports what plateau structure the detector produces.

    plateau_width       nt count of positions sharing the primary S_all value
    contains_gold       whether the plateau contains gold_nc (bool)
    centroid_dist       nt distance from plateau centroid to gold_nc
    """
    tnp_id: str
    plateau_width: int
    contains_gold: bool
    centroid_dist: float
    primary_position: float   # centroid (may be non-integer for even plateaus)
    gold_nc: int


def summarize_resolution(reports: list[ResolutionReport]) -> dict:
    """Aggregate a list of ResolutionReports into a distributional summary.

    Returns:
      n_detections, plateau_width_median, plateau_width_mean,
      contains_gold_frac, centroid_dist_median, centroid_dist_mean,
      exact_eq_0 (fraction with centroid_dist == 0),
      exact_le_1 (fraction with centroid_dist <= 1).
    """
    if not reports:
        return {"n_detections": 0}
    widths = np.array([r.plateau_width for r in reports])
    contains = np.array([r.contains_gold for r in reports], dtype=bool)
    dists = np.array([r.centroid_dist for r in reports])
    return {
        "n_detections": len(reports),
        "plateau_width_median": float(np.median(widths)),
        "plateau_width_mean": float(widths.mean()),
        "plateau_width_p95": float(np.percentile(widths, 95)),
        "contains_gold_frac": float(contains.mean()),
        "centroid_dist_median": float(np.median(dists)),
        "centroid_dist_mean": float(dists.mean()),
        "exact_eq_0": float((dists == 0).mean()),
        "exact_le_1": float((dists <= 1).mean()),
    }


# ---------- primary report ----------

@dataclass(frozen=True)
class MetricReport:
    """One variant's complete result on one evaluation set.

    Every derived comparison (worst-family, ratio-vs-null) must be computed
    THROUGH this object so denominators stay explicit.
    """
    variant_name: str
    dataset: str                    # "durrant_positive" | "durrant_shuffled" | "IS10-R" | ...
    n_perm: int
    seed: int

    coverage: Ratio                 # detected Tnps / total Tnps in dataset
    ppv_tnp_level: RatioCI          # Tnps with correct peak / detected Tnps  (canonical for P3)
    ppv_peak_level: RatioCI         # correct peaks / total peaks (diagnostic; tau-comparable only within tau)

    # exact-hit reported under BOTH tolerances + BOTH denominators (four numbers).
    # Under gold-blind centroid, dist == 0 is unreachable on even-width plateaus,
    # so exact_eq_0 is strictly stricter than exact_le_1 and both must ship.
    exact_eq_0_of_tnps: Ratio       # denom N_tnp    -- Tnps with centroid_dist == 0
    exact_le_1_of_tnps: Ratio       # denom N_tnp    -- Tnps with centroid_dist <= 1
    exact_eq_0_of_dets: Ratio       # denom N_dets   -- detections with centroid_dist == 0
    exact_le_1_of_dets: Ratio       # denom N_dets   -- detections with centroid_dist <= 1

    # Detector resolution — the 3-tuple that replaces the single "exact-hit" percentage.
    resolution: dict = field(default_factory=dict)   # from summarize_resolution(...)

    # Null-relative comparisons
    ratio_vs_shuffled: RatioCI | None = None            # random-flank null
    ratio_vs_target_destroyed: RatioCI | None = None    # durrant_shuffled null
    ratio_vs_nonguided: Mapping[str, RatioCI] = field(default_factory=dict)
    # ^ per non-guided family. Missing families = not measured (positive-only run).

    # FP classification (populated only for negative datasets)
    fp_by_class: Mapping[str, Ratio] = field(default_factory=dict)
    # keys: "motif" | "palindrome" | "composition" | "candidate"

    condition_tag: str = ""         # human-readable summary of MetricCondition

    def worst_family_fp_upper_ci(self) -> tuple[str, float] | None:
        """Return (family, upper CI of FP rate) for the worst family.
        Returns None if no negative-family measurements are attached."""
        worst = None
        for fam, rci in self.ratio_vs_nonguided.items():
            fp_rate = rci.num / max(1, rci.denom)
            fp_lo, fp_hi = clopper_pearson(rci.num, rci.denom)
            if worst is None or fp_hi > worst[1]:
                worst = (fam, fp_hi)
        return worst

    def summary_row(self) -> dict:
        row = {
            "variant": self.variant_name,
            "dataset": self.dataset,
            "coverage": self.coverage.value,
            "coverage_denom": f"{self.coverage.num}/{self.coverage.denom}",
            "ppv_tnp": self.ppv_tnp_level.value,
            "ppv_tnp_ci": f"[{self.ppv_tnp_level.lo:.3f}, {self.ppv_tnp_level.hi:.3f}]",
            "ppv_peak": self.ppv_peak_level.value,
            "ppv_peak_ci": f"[{self.ppv_peak_level.lo:.3f}, {self.ppv_peak_level.hi:.3f}]",
            "exact_eq0_tnp": self.exact_eq_0_of_tnps.value,
            "exact_le1_tnp": self.exact_le_1_of_tnps.value,
            "exact_eq0_dets": self.exact_eq_0_of_dets.value,
            "exact_le1_dets": self.exact_le_1_of_dets.value,
        }
        # 3-tuple resolution (from summarize_resolution)
        if self.resolution:
            row["plateau_width_median"] = self.resolution.get("plateau_width_median")
            row["contains_gold_frac"]   = self.resolution.get("contains_gold_frac")
            row["centroid_dist_median"] = self.resolution.get("centroid_dist_median")
        if self.ratio_vs_shuffled:
            row["ratio_shuf"] = self.ratio_vs_shuffled.value
            row["ratio_shuf_ci"] = f"[{self.ratio_vs_shuffled.lo:.2f}, {self.ratio_vs_shuffled.hi:.2f}]"
        for fam, r in self.ratio_vs_nonguided.items():
            row[f"fp_{fam}"] = r.num / max(1, r.denom)
            row[f"fp_{fam}_hi"] = r.hi
        worst = self.worst_family_fp_upper_ci()
        if worst:
            row["worst_family"] = worst[0]
            row["worst_fp_hi"] = worst[1]
        return row


# ---------- selection metric ----------

def worst_family_selection_score(report: MetricReport) -> float:
    """Selection metric for staged CV: negative of worst-family upper-CI FP.

    Higher = better. Uses upper CI (not point estimate) to prevent small-N
    families (ISAjo2 N=80) from noise-dominating the ordering.

    If no negative-family data attached, returns -inf (variant not comparable).
    """
    worst = report.worst_family_fp_upper_ci()
    if worst is None:
        return -math.inf
    return -worst[1]


def paired_decision_gate(
    m_fp_by_tnp: Mapping[str, bool],
    e_fp_by_tnp: Mapping[str, bool],
    n_boot: int = 5000,
    seed: int = 0,
    safety_upper: float = 0.02,
    both_high_cutoff: float = 0.10,
) -> dict:
    """Paired gate on ISLdl1 (or any negative family): is admission=min_E locked?

    Rebuilds the earlier two-independent-CIs construction as a same-Tnp paired
    comparison. m_fp_by_tnp and e_fp_by_tnp must be indexed by the SAME set of
    Tnp ids.

    Lock conditions:
      - direction 'admission_min_E' (min_E strictly better and safe):
          (a) paired Delta = mean(m_fp - e_fp) has 95% CI excluding 0
          (b) e_fp upper CP CI < safety_upper
      - direction 'composition_correction' (admission does not help):
          (a) paired Delta CI contains 0
          (b) both e_fp and m_fp upper CP CI > both_high_cutoff

    Returns a dict with:
      locked, direction, reason,
      delta_point, delta_ci_lo, delta_ci_hi,
      m_fp_lo, m_fp_hi, e_fp_lo, e_fp_hi,
      discordant_m_only, discordant_e_only  (paired 2x2 counts)
    """
    tnp_ids = sorted(set(m_fp_by_tnp) & set(e_fp_by_tnp))
    if len(tnp_ids) < 20:
        return {"locked": False,
                "reason": f"paired gate needs >=20 shared Tnps; got {len(tnp_ids)}"}
    m = np.array([int(m_fp_by_tnp[t]) for t in tnp_ids])
    e = np.array([int(e_fp_by_tnp[t]) for t in tnp_ids])
    delta = m - e
    N = len(tnp_ids)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boots[b] = delta[idx].mean()
    delta_lo = float(np.quantile(boots, 0.025))
    delta_hi = float(np.quantile(boots, 0.975))
    delta_point = float(delta.mean())

    m_lo, m_hi = clopper_pearson(int(m.sum()), N)
    e_lo, e_hi = clopper_pearson(int(e.sum()), N)
    m_only = int(((m == 1) & (e == 0)).sum())
    e_only = int(((m == 0) & (e == 1)).sum())

    result = {
        "delta_point": delta_point,
        "delta_ci_lo": delta_lo, "delta_ci_hi": delta_hi,
        "m_fp_lo": m_lo, "m_fp_hi": m_hi,
        "e_fp_lo": e_lo, "e_fp_hi": e_hi,
        "discordant_m_only": m_only, "discordant_e_only": e_only,
        "n_paired": N,
    }
    delta_excludes_zero_pos = delta_lo > 0
    delta_ci_contains_zero = (delta_lo <= 0 <= delta_hi)
    e_safe = e_hi < safety_upper
    both_high = (m_hi > both_high_cutoff) and (e_hi > both_high_cutoff)

    if delta_excludes_zero_pos and e_safe:
        result.update({
            "locked": True, "direction": "admission_min_E",
            "reason": (f"paired Delta CI=[{delta_lo:.3f}, {delta_hi:.3f}] excludes 0 "
                       f"AND e_fp upper CI {e_hi:.4f} < {safety_upper}"),
        })
    elif delta_ci_contains_zero and both_high:
        result.update({
            "locked": True, "direction": "composition_correction",
            "reason": (f"paired Delta CI=[{delta_lo:.3f}, {delta_hi:.3f}] contains 0 "
                       f"AND m_hi={m_hi:.3f}, e_hi={e_hi:.3f} both > {both_high_cutoff} "
                       "-> admission rule is NOT the answer; move to per-position p_hat"),
        })
    else:
        result.update({
            "locked": False,
            "reason": (f"gate not met: Delta CI=[{delta_lo:.3f}, {delta_hi:.3f}], "
                       f"m_hi={m_hi:.3f}, e_hi={e_hi:.3f} -> continue tau x S scan"),
        })
    return result


def decision_gate_from_reports_paired(
    m_report: MetricReport, e_report: MetricReport,
    m_fp_by_tnp: Mapping[str, bool], e_fp_by_tnp: Mapping[str, bool],
    dataset_expected: str = "ISLdl1",
    **gate_kwargs,
) -> dict:
    """Convenience wrapper for the paired gate on ISLdl1 reports.

    Per-Tnp FP indicators must be attached separately — the aggregate
    MetricReport carries only the summed count, which is not sufficient for
    the paired construction.
    """
    if m_report.dataset != dataset_expected or e_report.dataset != dataset_expected:
        return {"locked": False,
                "reason": f"gate requires reports.dataset == {dataset_expected!r}"}
    return paired_decision_gate(m_fp_by_tnp, e_fp_by_tnp, **gate_kwargs)
