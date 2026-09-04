"""Regression test: assert_non_degenerate_stratification catches the
orient='rev' silent-empty bug from 2026-09-01.

If someone reintroduces the same class of bug — a stratum label that no
downstream filter matches, so the stratum is silently empty and all its
rate metrics are 0.0 — this test verifies the assert fires.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.v5a_framework.metrics import (
    assert_non_degenerate_stratification,
    DegenerateStratumError,
)


def test_catches_all_zero_stratum():
    """Reproduces the orient='rev' bug: one stratum has all metrics == 0.0
    while n is large — the assert must raise DegenerateStratumError."""
    stratified = {
        "fwd": {"n_tnps": 124695, "strict": 0.071, "L_full": 0.419,
                "subL": 0.601, "any_L": 0.863},
        "rev": {"n_tnps": 125305, "strict": 0.000, "L_full": 0.000,
                "subL": 0.000, "any_L": 0.000},   # the bug signature
    }
    with pytest.raises(DegenerateStratumError, match="rev"):
        assert_non_degenerate_stratification(
            stratified, ("strict", "L_full", "subL", "any_L"),
            axis_name="orient", min_n=50,
        )


def test_catches_all_one_stratum():
    """Reciprocal: a stratum with all metrics == 1.0 is also suspicious
    (a tautology / swapped comparison)."""
    stratified = {
        "normal": {"n_tnps": 5000, "strict": 0.07, "L_full": 0.42},
        "tautology": {"n_tnps": 5000, "strict": 1.0, "L_full": 1.0},
    }
    with pytest.raises(DegenerateStratumError, match="tautology"):
        assert_non_degenerate_stratification(
            stratified, ("strict", "L_full"),
            axis_name="mode", min_n=50,
        )


def test_passes_healthy_stratification():
    """A non-degenerate stratification (the corrected fwd/rev numbers from
    the post-fix run) must pass silently."""
    stratified = {
        "fwd": {"n_tnps": 124695, "strict": 0.071, "L_full": 0.419,
                "subL": 0.601, "any_L": 0.863},
        "rev": {"n_tnps": 125305, "strict": 0.070, "L_full": 0.425,
                "subL": 0.605, "any_L": 0.865},
    }
    assert_non_degenerate_stratification(
        stratified, ("strict", "L_full", "subL", "any_L"),
        axis_name="orient", min_n=50,
    )


def test_ignores_small_strata():
    """A stratum with n < min_n is allowed to have all zeros — small-n noise
    is not diagnostic and would produce false positives."""
    stratified = {
        "big": {"n_tnps": 5000, "strict": 0.07},
        "tiny": {"n_tnps": 3, "strict": 0.0},   # legitimate small-n zero
    }
    assert_non_degenerate_stratification(
        stratified, ("strict",),
        axis_name="edge_case", min_n=50,
    )


if __name__ == "__main__":
    test_catches_all_zero_stratum()
    test_catches_all_one_stratum()
    test_passes_healthy_stratification()
    test_ignores_small_strata()
    print("all tests passed")
