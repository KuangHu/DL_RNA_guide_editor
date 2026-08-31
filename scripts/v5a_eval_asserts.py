"""Assertion helpers for methodology defense — target Category-A one-sided-arithmetic errors.

Any statistic reported for comparison must carry a condition tuple:

  MetricCondition(
      match_rule:        str    # "strict_WC" / "wobble" / "SW_gapped" / ...
      null_model:        str    # "unshuffled" / "dinuc_shuffled_flanks" / "Bin(N,q)_indep" / ...
      coordinate_system: str    # "absolute_nc" / "normalized_nc" / "aligned" / ...
      targeting_intact:  bool   # True if guide-target pairing preserved, False if destroyed
      tie_break:         str    # "average_rank" / "optimistic" / "pessimistic" / ...
      denominator:       str    # "in_pool" / "full_panel" / "conditional_on_hit" / ...
  )

Rule: when comparing two statistics, at most ONE field of the tuple may differ.
That field IS the axis being tested. All other fields must be identical.

This targets the 7/10 Category-A recurrences in the diagnostic phase (log_tail
p-null mismatch, D1 V-shape vs hinge, wobble asymmetric null, tie-break
inconsistency, m-rank stratification-as-target, W4+ gapped one-sided, V1''
destroyed-targeting-instead-of-family-bias).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class MetricCondition:
    """Every condition under which a numeric statistic was computed.
    Any two statistics compared as a ratio or difference must have identical
    tuples EXCEPT for the one dimension being deliberately varied.
    """
    match_rule:        str    # e.g. "strict_WC" | "wobble" | "SW_gapped" | "IoU_correct"
    null_model:        str    # e.g. "unshuffled_intra_family" | "dinuc_shuffled_flanks" | "Bin_indep"
    coordinate_system: str    # e.g. "absolute_nc" | "normalized_nc" | "sequence_aligned"
    targeting_intact:  bool
    tie_break:         str    # e.g. "average_rank" | "optimistic" | "pessimistic"
    denominator:       str    # e.g. "in_pool" | "full_panel" | "conditional_on_hit"
    n_sites_per_tnp:   int    # 5 for Durrant; must match on both sides of every ratio.
                              # Prevents 5-of-5 vs 5-of-16 architectural inflation.

    def diff(self, other: "MetricCondition") -> list[str]:
        return [f.name for f in fields(self)
                    if getattr(self, f.name) != getattr(other, f.name)]


@dataclass(frozen=True)
class Metric:
    """A single reported statistic + the condition tuple under which it was computed."""
    label:     str
    value:     float
    condition: MetricCondition


def assert_same_rule(a: Metric, b: Metric,
                        varying_dim: str | None = None) -> None:
    """Raise ValueError iff `a` and `b` differ in any dimension other than
    `varying_dim` (or, if varying_dim=None, they must be identical).

    Use this before computing ratios, deltas, or "degradation factors."
    """
    d = a.condition.diff(b.condition)
    if varying_dim is None:
        if d:
            raise ValueError(
                f"assert_same_rule: comparing '{a.label}' ({a.value}) vs '{b.label}' ({b.value}) "
                f"but they differ in {d}. Ratio/delta will confound multiple axes.")
    else:
        if varying_dim not in d:
            raise ValueError(
                f"assert_same_rule: expected varying_dim='{varying_dim}' but that "
                f"dimension is IDENTICAL between '{a.label}' and '{b.label}'. "
                f"Are you sure this is the comparison you meant?")
        extra = [x for x in d if x != varying_dim]
        if extra:
            raise ValueError(
                f"assert_same_rule: comparing '{a.label}' vs '{b.label}' along "
                f"'{varying_dim}' but they ALSO differ in {extra}. This is a "
                f"one-sided-arithmetic error (Category A). Reconcile the extra "
                f"axes before reporting the ratio.")


def safe_ratio(a: Metric, b: Metric, varying_dim: str | None = None) -> float:
    """Compute a.value / b.value only after passing assert_same_rule."""
    assert_same_rule(a, b, varying_dim)
    return a.value / max(1e-12, b.value)


# --- Retroactive examples from the diagnostic phase ---

def _example_v1pp_would_have_caught():
    """V1'' example: computing 'cross-family degradation' as (a/c) / (b/d),
    but b vs a differs in targeting_intact (dinuc-shuffled flanks lost
    biological pairing) AS WELL AS null_model — two axes moved at once."""
    a = Metric("V1''(a) REAL", 0.354, MetricCondition(
        match_rule="strict_WC", null_model="unshuffled_intra_family",
        coordinate_system="absolute_nc", targeting_intact=True,
        tie_break="average_rank", denominator="in_pool", n_sites_per_tnp=5))
    b = Metric("V1''(b) REAL_shuffled_own_flanks", 0.0154, MetricCondition(
        match_rule="strict_WC", null_model="unshuffled_intra_family",
        coordinate_system="absolute_nc", targeting_intact=False,   # ← changed
        tie_break="average_rank", denominator="in_pool"))
    # This would raise: a/c compared to b/d with varying_dim="null_model" but
    # targeting_intact also differs. Category-A error caught.
    try:
        safe_ratio(a, b, varying_dim="null_model")
    except ValueError as e:
        return str(e)


if __name__ == "__main__":
    print(_example_v1pp_would_have_caught())
