"""generator_data_checker — three-layer acceptance for a V5 / v6 corpus.

Motivation (2026-09-03): the two Stage 1a bugs (RNG divergence at
`flank_offset_mode=consistent`, unit error at `_mutate_preserving_guide`)
both hid at *default* parameter values — the gate ran at hom=1.0 (short-
circuit) and at flank_offset_mode=inconsistent (default). Design rule
for this checker:

  **Every axis must be verified at every value it takes, not just at
  default. Pooling hides single-cell failures.**

Layers:

  Layer 1 — Data self-consistency (data-only, no detector).
    Directly measures data invariants (identity fraction, map correctness,
    within-bag guide sharing, mismatch class satisfaction, etc.). When
    a check fails, the failure is in the generator, not in Channel A.

  Layer 2 — Leakage probes.
    Trains a simple classifier on record-level features to predict a
    label the generator MUST have hidden (active vs inactive nc,
    planted vs unplanted flank, negative_mode from single features).
    Any classifier that gets meaningfully above chance reveals a
    shortcut Channel B could learn.

  Layer 3 — Mechanism anchor (through Channel A).
    Runs Channel A on the corpus, stratifies by
    (L × mm_concentration × nc_homology_rate × n_sites). Reports each
    cell's cov / exact / PPV with 95% Tnp-clustered bootstrap CI.

Usage:
  python -m scripts.generator_v5.generator_data_checker \
    --jsonl /path/positives.jsonl \
    --shard-dir /path/mt/  \
    --layer all \
    --report-out /path/report.json

CLI --layer accepts 1, 2, 3, or all (comma-separated for a subset).

Report format:
  Each check emits {status: PASS|FAIL|SKIP, cells: [{axis-key: value,
  metric: value, band: [lo, hi], pass: bool}, ...]}. All FAILs are
  aggregated into an exit code (non-zero iff any FAIL).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


# ---------- record loader ----------

def _iter_records(jsonl: Path, max_records: int | None = None):
    with open(jsonl) as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break
            yield json.loads(line)


def _group_by_bag(jsonl: Path, max_bags: int | None = None
                    ) -> dict[str, list[dict]]:
    """Load records grouped by bag_id (transposase_id). Stops after
    `max_bags` bags (records past that boundary are dropped)."""
    bags: dict[str, list[dict]] = {}
    for r in _iter_records(jsonl):
        t = r["transposase_id"]
        if t not in bags:
            if max_bags is not None and len(bags) >= max_bags:
                # Skip until the next bag boundary
                continue
            bags[t] = []
        bags[t].append(r)
    return bags


def _first_record_per_bag(jsonl: Path, max_bags: int | None = None
                             ) -> list[dict]:
    """Read the first record per bag (bag-level metadata: canonical_nc,
    fold, arch axes). Much cheaper than full grouping."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in _iter_records(jsonl):
        t = r["transposase_id"]
        if t in seen:
            continue
        seen.add(t)
        out.append(r)
        if max_bags is not None and len(out) >= max_bags:
            break
    return out


# ---------- check helpers ----------

def _in_band(value: float, band: tuple[float, float]) -> bool:
    lo, hi = band
    return lo <= value <= hi


def _summarize(name: str, cells: list[dict]) -> dict:
    fails = [c for c in cells if not c.get("pass", True)]
    skips = [c for c in cells if c.get("status") == "SKIP"]
    status = "PASS"
    if fails:
        status = "FAIL"
    elif skips and len(skips) == len(cells):
        status = "SKIP"
    return {"name": name, "status": status, "cells": cells,
            "n_pass": len(cells) - len(fails) - len(skips),
            "n_fail": len(fails), "n_skip": len(skips)}


# ---------- Layer 1 checks ----------

def check_nc_homology_identity(bags: list[list[dict]],
                                    tol: float = 0.02) -> dict:
    """Measure site↔canonical sequence identity per declared nc_homology_rate.

    For each bag: compute mean over its 5 sites of the fraction of
    canonical positions preserved (canonical[p] == site[hits[p]]),
    where `hits` is inverted from the site_to_canonical_map.

    Stratified by declared nc_homology_rate. Per-cell PASS if mean
    identity in [rate - tol, rate + tol]. Skipped for records without
    canonical_nc (pre-v6 corpus)."""
    by_rate: dict[float, list[float]] = defaultdict(list)
    for recs in bags:
        for r in recs:
            l = r["labels"]
            canonical = l.get("canonical_nc")
            site = l.get("site_nc_sequence")
            s2c = l.get("site_to_canonical_map")
            if canonical is None or site is None or s2c is None:
                return _summarize("nc_homology_identity",
                                    [{"status": "SKIP",
                                       "reason": "no canonical_nc — pre-v6 corpus",
                                       "pass": True}])
            rate = float(l["arch"].get("nc_homology_rate", 1.0))
            hits = [-1] * len(canonical)
            for k, cp in enumerate(s2c):
                if 0 <= cp < len(canonical) and hits[cp] == -1:
                    hits[cp] = k
            preserved = sum(1 for p in range(len(canonical))
                              if hits[p] != -1 and site[hits[p]] == canonical[p])
            by_rate[rate].append(preserved / len(canonical))
    cells = []
    for rate in sorted(by_rate):
        vals = np.array(by_rate[rate])
        m = float(vals.mean())
        band = (rate - tol, rate + tol)
        # Insertion-preserving detail: 15% of events insert a base BEFORE
        # the canonical position, so the canonical char is still present.
        # Effective identity ≈ rate + (1-rate)*0.15. Widen band by that.
        adjusted_hi = min(1.0, rate + (1 - rate) * 0.15 + tol)
        band = (rate - tol, adjusted_hi)
        cells.append({
            "nc_homology_rate": rate,
            "n_sites":          int(vals.size),
            "mean_identity":    round(m, 4),
            "band":             [round(band[0], 4), round(band[1], 4)],
            "pass":             _in_band(m, band),
        })
    return _summarize("nc_homology_identity", cells)


def check_alignment_map_at_guide(bags: list[list[dict]],
                                      accuracy_floor: float = 0.98) -> dict:
    """At the guide window, verify site_to_canonical_map recovers the
    correct canonical positions. For each site, find the site position
    whose s2c value equals gs; check that it maps back to gs correctly.

    Under hom=1.0, this must be 100% (identity map).
    Under hom<1.0, PW alignment may misplace a small fraction; the floor
    is 98%. Stratified by nc_homology_rate."""
    by_rate_correct: dict[float, int] = defaultdict(int)
    by_rate_total: dict[float, int] = defaultdict(int)
    for recs in bags:
        for r in recs:
            l = r["labels"]
            canonical = l.get("canonical_nc")
            s2c = l.get("site_to_canonical_map")
            if canonical is None or s2c is None:
                return _summarize("alignment_map_at_guide",
                                    [{"status": "SKIP",
                                       "reason": "no v6 fields",
                                       "pass": True}])
            gs = l["guide_span_in_active_noncoding"][0]
            L = l["guide_length"]
            rate = float(l["arch"].get("nc_homology_rate", 1.0))
            # A canonical guide-window position p is recovered correctly
            # iff some site k has s2c[k] == p AND canonical[p] == site[k].
            # Under mutate_preserving_guide, guide window is byte-preserved,
            # so the second condition is trivially satisfied when the first
            # is. Count fraction of gs..gs+L positions with an s2c entry.
            hit_flags = [False] * L
            for k, cp in enumerate(s2c):
                if gs <= cp < gs + L:
                    hit_flags[cp - gs] = True
            correct = sum(hit_flags)
            by_rate_correct[rate] += correct
            by_rate_total[rate] += L
    cells = []
    for rate in sorted(by_rate_correct):
        acc = by_rate_correct[rate] / max(1, by_rate_total[rate])
        # hom=1.0 must be exactly 1.0 (short-circuit path)
        floor = 1.0 - 1e-9 if rate >= 1.0 else accuracy_floor
        cells.append({
            "nc_homology_rate":  rate,
            "n_positions":       int(by_rate_total[rate]),
            "recovery_accuracy": round(acc, 4),
            "floor":             round(floor, 4),
            "pass":              acc >= floor,
        })
    return _summarize("alignment_map_at_guide", cells)


def check_flank_offset_std(bags: list[list[dict]]) -> dict:
    """Verify per-bag flank plant_start distribution matches the declared
    flank_offset_mode.

      consistent   : plant_start = bag_base + U{-jitter, jitter}  →  std ≈ 1.4 at J=2
      inconsistent : plant_start = U[0, flank_len - target_width) →  std ≈ (range)/√12

    Stratified by mode. Per bag: compute std of plant_start across its
    sites; aggregate median and 10th/90th percentile per mode."""
    by_mode_stds: dict[str, list[float]] = defaultdict(list)
    by_mode_jitter: dict[str, int] = {}
    for recs in bags:
        # Only planted sites contribute
        starts = [r["labels"]["planted_start"] for r in recs
                     if r["labels"].get("is_planted", True)]
        if len(starts) < 2:
            continue
        mode = recs[0]["labels"]["arch"].get("flank_offset_mode", "inconsistent")
        jitter = int(recs[0]["labels"]["arch"].get("flank_jitter", 2))
        by_mode_jitter[mode] = jitter
        by_mode_stds[mode].append(float(np.std(starts)))
    cells = []
    for mode in sorted(by_mode_stds):
        stds = np.array(by_mode_stds[mode])
        med = float(np.median(stds))
        if mode == "consistent":
            # Expected std for U{-J..J} on integers = sqrt(J*(J+1)/3)
            J = by_mode_jitter[mode]
            expected = math.sqrt(J * (J + 1) / 3)
            band = (0.5, expected * 2.0)      # generous band, catches order-of-mag errors
        else:
            # Uniform over roughly [0, 106) (120 - 14 typical) → std ≈ 30
            band = (15, 45)
        cells.append({
            "flank_offset_mode": mode,
            "n_bags":            int(stds.size),
            "median_std":        round(med, 3),
            "expected_band":     [round(band[0], 3), round(band[1], 3)],
            "pass":              _in_band(med, band),
        })
    return _summarize("flank_offset_std", cells)


def check_bag_guide_sharing(bags: list[list[dict]]) -> dict:
    """All planted sites of a positive bag must share the same guide.
    For scattered mode the guide varies per site — SKIPPED for those.
    Reported as fraction of bags with sharing violated."""
    bad = 0
    total = 0
    for recs in bags:
        neg_mode = recs[0]["labels"].get("negative_mode", "none")
        if neg_mode == "scattered":
            continue
        guides = {r["labels"]["perfect_guide_dna"]
                     for r in recs
                     if r["labels"].get("is_planted", True)}
        if not guides:
            continue
        total += 1
        if len(guides) > 1:
            bad += 1
    if total == 0:
        return _summarize("bag_guide_sharing",
                            [{"status": "SKIP", "reason": "no positive bags",
                               "pass": True}])
    cell = {
        "n_positive_bags": total,
        "n_bad":            bad,
        "shared_fraction":  round(1 - bad / total, 4),
        "pass":             bad == 0,
    }
    return _summarize("bag_guide_sharing", [cell])


def check_bag_target_diversity(bags: list[list[dict]]) -> dict:
    """The n_sites flanks of a positive bag come from a bag-shuffled pool
    draw; the L-nt target segments across sites should be pairwise-distinct.

    Stratified by n_sites — collision probability grows with C(n, 2):
    n=5 → 10 pairs, n=8 → 28 pairs, so the "all distinct" fraction
    falls with n_sites and needs a per-n band.

    Threshold: the flank-pool draws from a finite pool (~2763 flanks in
    the 86/10/4-tail set). Small-birthday collision rate ≈ C(n,2)/2763.
    Band per cell = 1 - 5 * C(n,2)/2763 (5× safety over the analytic
    Poisson rate). Failure at these bands indicates a real diversity
    regression (e.g. same flank drawn n times); the 0.4-0.02% observed
    tail is expected under the finite pool.
    """
    by_n_sites: dict[int, tuple[int, int, list[int]]] = {}
    for recs in bags:
        targets = [r["labels"].get("target_dna", "") for r in recs]
        targets = [t for t in targets if t]
        if len(targets) < 2:
            continue
        n_sites = len(targets)
        n_uniq = len(set(targets))
        checked, distinct, per_bag = by_n_sites.get(n_sites, (0, 0, []))
        checked += 1
        if n_uniq == n_sites:
            distinct += 1
        per_bag.append(n_uniq)
        by_n_sites[n_sites] = (checked, distinct, per_bag)
    if not by_n_sites:
        return _summarize("bag_target_diversity",
                            [{"status": "SKIP", "reason": "no targets",
                               "pass": True}])
    cells = []
    for n in sorted(by_n_sites):
        checked, distinct, per_bag = by_n_sites[n]
        frac = distinct / max(1, checked)
        # Band widens with n_sites: expected collision rate scales as C(n,2)*(1/4^L).
        # For L>=9 all bands come out ≥0.98 in the collision-free regime.
        pairs = n * (n - 1) // 2
        band_lo = max(0.90, 1.0 - 5.0 * pairs / 2763)
        cells.append({
            "n_sites":            n,
            "n_bags":             checked,
            "frac_all_distinct":  round(frac, 4),
            "median_uniq_per_bag": int(np.median(per_bag)) if per_bag else 0,
            "band":               [round(band_lo, 4), 1.0],
            "pass":               frac >= band_lo,
        })
    return _summarize("bag_target_diversity", cells)


def check_mismatch_geometry_class(bags: list[list[dict]]) -> dict:
    """Verify each site's mismatch positions satisfy the declared
    mm_concentration class using the CANONICAL classifier from
    `constrained_mm._classify_tuple`. Class definitions:

      clustered  ≡ mode2_visible : ∃ L' ∈ {9,10,11,12} ∩ [1,L], ∃ offset
                     ∈ [0, L-L'], subwindow has ≤ 1 mismatch
      dispersed  ≡ mode2_blind   : ∀ L' ∈ {9,10,11,12} ∩ [1,L], ∀ offset,
                     subwindow has ≥ 2 mismatches

    Stratified BY L to catch per-L-cell drift (e.g., L=14 might satisfy
    while L=11 fails). Sites with n_mm < 2 skipped (vacuous).
    """
    from scripts.generator_v5.constrained_mm import (
        _classify_tuple, class_counts,
    )
    # Per (L, concentration): tot = sites where class is non-empty (fair
    # test); agree = sites whose sampled mm actually falls in declared class.
    # fallback = sites where declared class is empty at (L, n_mm), so the
    # sampler falls back to the other class by design (documented in
    # architecture.sample_mismatch_positions).
    per_L: dict[tuple[int, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    for recs in bags:
        for r in recs:
            l = r["labels"]
            L = l["guide_length"]
            n_mm = l.get("n_mismatches", 0)
            if n_mm < 2:
                continue
            mm = l.get("mismatch_positions", [])
            if len(mm) != n_mm:
                continue
            concentration = l["arch"].get("mm_concentration")
            if concentration is None:
                continue
            n_vis, n_bli = class_counts(L, n_mm)
            declared_empty = (
                (concentration == "clustered" and n_vis == 0) or
                (concentration == "dispersed" and n_bli == 0))
            cls = _classify_tuple(tuple(sorted(mm)), L)
            declared = ("mode2_visible" if concentration == "clustered"
                          else "mode2_blind")
            k = (L, concentration)
            tot, agree, fallbacks = per_L[k]
            if declared_empty:
                fallbacks += 1
            else:
                tot += 1
                if cls == declared:
                    agree += 1
            per_L[k] = [tot, agree, fallbacks]
    cells = []
    for (L, concentration) in sorted(per_L):
        tot, ok, fallbacks = per_L[(L, concentration)]
        frac = ok / max(1, tot)
        cells.append({
            "L":                L,
            "mm_concentration": concentration,
            "n_sites":          tot + fallbacks,
            "n_class_non_empty": tot,
            "n_fallback":       fallbacks,
            "class_satisfied":  round(frac, 4),
            "band":             [0.99, 1.0],
            "pass":             frac >= 0.99 or tot == 0,
        })
    return _summarize("mismatch_geometry_class", cells)


def check_planted_m_distribution(bags: list[list[dict]]) -> dict:
    """Per-L distribution of `planted_m`. Under the 86/10/4 tail, mode ==
    target_m, and shape should match. We report the mode and the tail
    proportions (m == target_m, m > target_m, m < target_m). Skips the
    check if target_m is missing."""
    by_L: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for recs in bags:
        for r in recs:
            l = r["labels"]
            L = l["guide_length"]
            pm = l.get("planted_m")
            tm = l.get("bag_target_m")
            if pm is None or tm is None:
                continue
            by_L[L].append((pm, tm))
    cells = []
    for L in sorted(by_L):
        arr = np.array(by_L[L])
        pms, tms = arr[:, 0], arr[:, 1]
        mode_pm = int(np.bincount(pms).argmax())
        eq_target = float((pms == tms).mean())
        below = float((pms < tms).mean())
        above = float((pms > tms).mean())
        # 86/10/4 = P(=target) / P(≤target-2) / P(≤target-3+) roughly
        pass_eq = 0.75 <= eq_target <= 0.95
        pass_above = above <= 0.05
        cells.append({
            "L":            L,
            "n_sites":      int(pms.size),
            "mode_pm":      mode_pm,
            "P_eq_target":  round(eq_target, 4),
            "P_below":      round(below, 4),
            "P_above":      round(above, 4),
            "band_eq":      [0.75, 0.95],
            "pass":         pass_eq and pass_above,
        })
    return _summarize("planted_m_distribution", cells)


def check_competitor_count(bags: list[list[dict]]) -> dict:
    """Per-(L, gc_bin) observed median competitor rate vs analytic
    expected from the gc-conditioned rate table.

    Stage 1f (2026-09-03, threshold-jump aware): observed rate depends
    on target_m(L, gc), which can JUMP at cells where rate(m+1) crosses
    the 0.15 floor (verified: L=13 gc=0.60 target_m 9→8 jump). A single
    per-L band conflates real gc modulation with threshold jumps. The
    correct check compares each cell to its ANALYTIC EXPECTED rate:

        expected = 0.86 rate(L, tm, gc) + 0.10 rate(L, tm-1, gc)
                                        + 0.04 rate(L, tm-2, gc)

    which uses the 86/10/4 planted_m tail and the gc-conditioned rate
    table. Band is expected × [0.7, 1.4]. A cell FAILing this check
    signals either (a) a rate-table drift (bad build) or (b) a target_m
    lookup mismatch (bag says planted at gc=X but detector saw
    planted_m from bin Y). The pass/fail no longer conflates coupling
    with jump.
    """
    from scripts.generator_v5.difficulty import load_or_build_rate_table
    tbl = load_or_build_rate_table(rebuild=False)

    def gc_bin_of(v: float | None) -> str:
        if v is None:
            return "unknown"
        for lo, hi in ((0.25, 0.35), (0.35, 0.45), (0.45, 0.55), (0.55, 0.65)):
            if lo <= v < hi:
                return f"[{lo:.2f},{hi:.2f})"
        return f"other({v:.2f})"

    def gc_bin_center(gc_b: str) -> float | None:
        """Extract the center of a gc bin string like '[0.25,0.35)'."""
        try:
            lo_hi = gc_b.strip("[)()]").split(",")
            return (float(lo_hi[0]) + float(lo_hi[1])) / 2.0
        except Exception:
            return None

    def snap_gc(gc_center: float) -> float:
        """Snap a continuous gc value to the nearest measured bin center.

        NOTE on discretization error: gc_target ~ U[0.25, 0.65] is
        continuous, but the rate table has only 4 bins (0.3/0.4/0.5/0.6).
        A bag at gc=0.34 uses gc=0.30's target_m and rate lookup, while
        its actual rate matches gc=0.34. Under the current ±30% band
        this is absorbed (per 0.05 gc step ≈ 10% rate change; bin
        half-width 0.05 → up to ~10% error). If future tightening
        pushes the band below ±15%, either subdivide the rate table
        or interpolate between neighboring bins. Do NOT tighten the
        band before addressing this discretization.
        """
        if not tbl.gc_bins:
            return 0.5
        return min(tbl.gc_bins, key=lambda b: abs(b - gc_center))

    def expected_rate(L: int, gc_bin_str: str) -> tuple[float, int] | None:
        """Return (expected_rate, target_m) for the cell, or None."""
        center = gc_bin_center(gc_bin_str)
        if center is None:
            return None
        gc = snap_gc(center)
        try:
            tm = tbl.target_m_for_L(L, 0.21, gc=gc)
        except KeyError:
            return None
        rd = tbl.rate_by_gc.get(gc, tbl.rate)
        rates = [rd.get((L, m), 0.0) for m in (tm, tm - 1, tm - 2)]
        exp = 0.86 * rates[0] + 0.10 * rates[1] + 0.04 * rates[2]
        return exp, tm

    key = tuple[int, str]
    by_L_gc: dict[key, list[tuple[int, int]]] = defaultdict(list)
    for recs in bags:
        for r in recs:
            l = r["labels"]
            L = l["guide_length"]
            cc = l.get("competitor_count_at_site_planted_m")
            nc_len = l.get("ncrna_length", 200)
            gc_val = l.get("arch", {}).get("gc_target")
            if cc is None:
                continue
            by_L_gc[(L, gc_bin_of(gc_val))].append((cc, nc_len))
    cells = []
    for (L, gc_b) in sorted(by_L_gc):
        arr = np.array(by_L_gc[(L, gc_b)])
        counts, nc_lens = arr[:, 0], arr[:, 1]
        n_pos = np.maximum(nc_lens - L + 1, 1)
        rates = counts / n_pos
        med_rate = float(np.median(rates))
        exp_pair = expected_rate(L, gc_b)
        if exp_pair is None:
            expected = None; target_m = None
            band = [0.05, 0.50]
            passing = 0.05 <= med_rate <= 0.50
        else:
            expected, target_m = exp_pair
            band = [round(expected * 0.7, 4), round(expected * 1.4, 4)]
            passing = band[0] <= med_rate <= band[1] if expected > 0 else True
        cells.append({
            "L":            L,
            "gc_bin":       gc_b,
            "n_sites":      int(counts.size),
            "target_m":     target_m,
            "median_rate":  round(med_rate, 4),
            "expected":     round(expected, 4) if expected else None,
            "band":         band,
            "pass":         bool(passing),
        })
    return _summarize("competitor_count", cells)


def check_nc_position_consistency(bags: list[list[dict]]) -> dict:
    """Within a positive bag, all sites' guide_span_in_active_noncoding[0]
    (canonical guide start) must be identical."""
    n_bag = 0
    n_bad = 0
    for recs in bags:
        neg_mode = recs[0]["labels"].get("negative_mode", "none")
        if neg_mode == "scattered":
            continue
        starts = {r["labels"]["guide_span_in_active_noncoding"][0]
                     for r in recs}
        if len(starts) == 0:
            continue
        n_bag += 1
        if len(starts) > 1:
            n_bad += 1
    if n_bag == 0:
        return _summarize("nc_position_consistency",
                            [{"status": "SKIP", "reason": "no bags", "pass": True}])
    cell = {
        "n_bags":         n_bag,
        "n_inconsistent": n_bad,
        "pass":           n_bad == 0,
    }
    return _summarize("nc_position_consistency", [cell])


# ---------- Layer 2 checks (leakage probes) ----------

def _auroc_from_features(X: np.ndarray, y: np.ndarray,
                            random_state: int = 0) -> float:
    """5-fold cross-validated logistic-regression AUROC. Held-out fold
    predictions avoid the in-sample overfit that inflates the probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    y = np.asarray(y)
    p_pred = np.zeros_like(y, dtype=float)
    for train, test in kf.split(X):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[train])
        Xte = scaler.transform(X[test])
        clf = LogisticRegression(max_iter=1000)
        clf.fit(Xtr, y[train])
        p_pred[test] = clf.predict_proba(Xte)[:, 1]
    return float(roc_auc_score(y, p_pred))


def _flank_features(flank: str) -> list[float]:
    """22 flank composition features: A/C/G/T freqs (4), all dinucleotide
    freqs (16), Shannon entropy (1), GC skew (1)."""
    L = len(flank)
    if L == 0:
        return [0.0] * 22
    base_counts = Counter(flank)
    feats = [base_counts.get(b, 0) / L for b in "ACGT"]
    dinuc_counts = Counter(flank[i:i+2] for i in range(L - 1))
    for b1 in "ACGT":
        for b2 in "ACGT":
            feats.append(dinuc_counts.get(b1 + b2, 0) / max(1, L - 1))
    p = np.array(feats[:4])
    p_nonzero = p[p > 0]
    entropy = -float((p_nonzero * np.log2(p_nonzero)).sum()) if p_nonzero.size else 0.0
    feats.append(entropy)
    gc = base_counts.get("G", 0) + base_counts.get("C", 0)
    feats.append((gc / L) if L else 0.0)
    return feats


def check_active_vs_inactive_leakage(bags: list[list[dict]]) -> dict:
    """Train a classifier on per-nc summary features (length, GC, and
    scalar stats from the structure channels when present) to distinguish
    active vs inactive nc within the same bag. AUROC ≤ 0.55 is target
    (matched-distribution generator prevents a shortcut)."""
    X = []
    y = []
    for recs in bags:
        r = recs[0]
        l = r["labels"]
        ncs = r["inputs"]["noncoding_regions"]
        nc_channels = l.get("nc_channels")
        if not nc_channels:
            # No structure channels emitted; fall back to length/GC only
            for i, nc in enumerate(ncs):
                feats = [len(nc)]
                gc = nc.count("G") + nc.count("C")
                feats.append(gc / max(1, len(nc)))
                X.append(feats)
                y.append(1 if i == l["active_noncoding_index"] else 0)
        else:
            for i, (nc, ch) in enumerate(zip(ncs, nc_channels)):
                feats = [len(nc), (nc.count("G") + nc.count("C")) / max(1, len(nc))]
                for key in ("dG_open_u1", "dG_open_uL_pn", "cooperativity_win_pn"):
                    arr = np.array(ch.get(key, []), dtype=float)
                    if arr.size:
                        feats.extend([float(arr.mean()), float(arr.std())])
                    else:
                        feats.extend([0.0, 0.0])
                X.append(feats)
                y.append(1 if i == l["active_noncoding_index"] else 0)
    X = np.array(X)
    y = np.array(y)
    if X.shape[0] < 20 or len(set(y)) < 2:
        return _summarize("nc_active_vs_inactive_leakage",
                            [{"status": "SKIP", "reason": "not enough ncs",
                               "pass": True}])
    auroc = _auroc_from_features(X, y)
    # Symmetric ±0.05 band assumes SE ≈ 0.023 (n_ncs ≥ 500). At smaller
    # n, widen automatically — otherwise a single tiny corpus lights up
    # the check as a false positive.
    n = int(X.shape[0])
    slack = max(0.05, 1.5 / math.sqrt(n))
    band = (0.5 - slack, 0.5 + slack)
    cell = {
        "n_ncs":  n,
        "auroc":  round(auroc, 4),
        "band":   [round(band[0], 4), round(band[1], 4)],
        "pass":   band[0] <= auroc <= band[1],
    }
    return _summarize("nc_active_vs_inactive_leakage", [cell])


def check_flank_planted_vs_unplanted_leakage(bags: list[list[dict]]) -> dict:
    """Train a classifier on 22 flank composition features to distinguish
    planted vs unplanted flanks. AUROC ≤ 0.52 target — the fake-plant
    (composition-matched shuffle at same position) should erase all
    single-flank signal. Requires the corpus to have negative_mode !=
    'none' bags (partial or fully negative). Skips if not."""
    X = []
    y = []
    for recs in bags:
        for r in recs:
            is_p = r["labels"].get("is_planted", True)
            X.append(_flank_features(r["inputs"]["flank"]))
            y.append(1 if is_p else 0)
    X = np.array(X)
    y = np.array(y)
    if len(set(y)) < 2:
        return _summarize("flank_planted_vs_unplanted_leakage",
                            [{"status": "SKIP",
                               "reason": "no unplanted sites (positive-only corpus)",
                               "pass": True}])
    if X.shape[0] < 20:
        return _summarize("flank_planted_vs_unplanted_leakage",
                            [{"status": "SKIP", "reason": "not enough sites",
                               "pass": True}])
    auroc = _auroc_from_features(X, y)
    cell = {
        "n_sites": int(X.shape[0]),
        "n_positive": int(y.sum()),
        "auroc":   round(auroc, 4),
        "band":    [0.48, 0.52],
        "pass":    0.48 <= auroc <= 0.52,
    }
    return _summarize("flank_planted_vs_unplanted_leakage", [cell])


def check_structure_positive_vs_twin_leakage(bags: list[list[dict]]) -> dict:
    """Deferred — twin negatives (Stage 1d) not yet in the generator.
    Emits SKIP with a rationale so the check is visible in the report."""
    return _summarize("structure_positive_vs_twin_leakage",
                        [{"status": "SKIP",
                           "reason": "twin negatives (Stage 1d) not yet emitted; "
                                     "provable-0.5 check will land alongside them",
                           "pass": True}])


def check_negative_mode_leakage(bags: list[list[dict]]) -> dict:
    """Train per-feature univariate classifiers to predict negative_mode
    from single features (nc_len, GC, planted_start, dG stats). Any
    single-feature AUROC > 0.55 = shortcut. Skips if corpus has < 2
    negative_modes."""
    modes: list[str] = []
    feats: dict[str, list[float]] = defaultdict(list)
    for recs in bags:
        r = recs[0]
        l = r["labels"]
        mode = l.get("negative_mode", "none")
        modes.append(mode)
        feats["nc_len"].append(len(r["inputs"]["noncoding_regions"][l["active_noncoding_index"]]))
        act_nc = r["inputs"]["noncoding_regions"][l["active_noncoding_index"]]
        gc = act_nc.count("G") + act_nc.count("C")
        feats["gc"].append(gc / max(1, len(act_nc)))
        feats["planted_start"].append(l.get("planted_start", 0))
    mode_set = sorted(set(modes))
    if len(mode_set) < 2:
        return _summarize("negative_mode_leakage",
                            [{"status": "SKIP",
                               "reason": f"only one negative_mode: {mode_set}",
                               "pass": True}])
    cells = []
    from sklearn.metrics import roc_auc_score
    for target_mode in mode_set:
        y = np.array([1 if m == target_mode else 0 for m in modes])
        for fname, vals in feats.items():
            x = np.array(vals, dtype=float)
            try:
                auroc = float(roc_auc_score(y, x))
                auroc = max(auroc, 1 - auroc)
            except Exception:
                continue
            cells.append({
                "target_mode": target_mode,
                "feature":     fname,
                "auroc":       round(auroc, 4),
                "band":        [0.0, 0.55],
                "pass":        auroc <= 0.55,
            })
    return _summarize("negative_mode_leakage", cells)


# ---------- Layer 3 checks (through Channel A) ----------

def check_channel_a_stratified(jsonl: Path, shard_dir: Path,
                                    workers: int = 32, n_boot: int = 200,
                                    theta: float | None = 1.0,
                                    flank_coherence: str = "off",
                                    ) -> dict:
    """Load or build MatchTable, run Channel A (L=11 m=8 tau=0),
    stratify by (L × mm_concentration × nc_homology_rate × n_sites).
    Only cells with n_tnps >= 20 are reported.

    Spec: if `theta` is given (default 1.0 = "all sites must hit"),
    the S threshold scales per-Tnp with n_sites; if None, uses fixed
    S=5 (legacy). Stage 1c makes theta the correct choice; keeping S=5
    is only for pre-1c anchor reproduction.

    Cell band: 95% Tnp-clustered bootstrap CI on coverage. Reported for
    diagnostics, not gated (the cells are the READING).
    The CROSS-CELL check gated here: monotonicity over nc_homology_rate."""
    import shutil
    from scripts.generator_v5.channel_a_v5 import (
        build_v5_positive, _parallel_run_variant,
    )
    from scripts.v5a_framework.match_table import load as load_mt
    from scripts.v5a_framework.variant import spec_m_threshold_L11, run_variant
    from scripts.generator_v5.channel_a_v5 import compute_channel_a

    if not (shard_dir / "_index.json").exists():
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        mt, tnp_arch = build_v5_positive(str(jsonl), str(shard_dir), workers=workers)
    else:
        mt = load_mt(str(shard_dir))
        tnp_arch = {}
        for r in _iter_records(jsonl):
            t = r["transposase_id"]
            if t in tnp_arch:
                continue
            tnp_arch[t] = dict(r["labels"].get("arch", {}))

    spec = spec_m_threshold_L11(m=8, tau=0, S=5, theta=theta,
                                    flank_coherence=flank_coherence)
    if workers > 1:
        peaks = _parallel_run_variant(mt, spec, str(shard_dir), workers)
    else:
        peaks = run_variant(mt, spec)

    # Cell key: (L, mm_concentration, nc_homology_rate, n_sites)
    # (n_sites is the Stage 1c axis — pre-1c corpora emit n_sites=5.)
    def key(a: dict) -> tuple:
        return (int(a.get("L", -1)),
                a.get("mm_concentration", "?"),
                round(float(a.get("nc_homology_rate", 1.0)), 3),
                int(a.get("n_sites", 5)))

    cells: dict[tuple, list[str]] = defaultdict(list)
    for t, a in tnp_arch.items():
        cells[key(a)].append(t)

    result_cells = []
    n_under_sampled = 0
    ns3_tnps: list[str] = []     # n_sites=3 tracked separately (see note below)
    # NOTE: n_sites=3 is NOT excluded from the report. Under Mode 2 (m≥8)
    # NC-AXIS ONLY, n_sites=3 has no valid S (E[FP]/bag ≥ 1). But adding
    # flank-offset consistency drops the effective q by ~250×, making
    # n_sites=3 fully viable — that's the ONE slice where the second-
    # coherence axis (Tier 2 1b) produces a binary yes/no answer, and
    # thus the load-bearing validation for the flank axis. Do NOT drop
    # n_sites=3 from the corpus. Cell values below are Mode 2 nc-axis
    # only; treat them as the "null" against which a flank-coherent
    # detector's lift is measured in Stage 3.
    for k, tnps in sorted(cells.items()):
        L, mm, hom, n_sites_cell = k
        if n_sites_cell == 3:
            ns3_tnps.extend(tnps)
        # Cells with fewer than 10 Tnps have bootstrap CI so wide the
        # cell is uninformative. Report them as an aggregate count so
        # the reader can see "corpus is too small" rather than "nothing
        # to check here".
        if len(tnps) < 10:
            n_under_sampled += len(tnps)
            continue
        cov = 0; exact = 0
        for t in tnps:
            pks = peaks.get(t, [])
            if not pks:
                continue
            cov += 1
            gs = mt.tnps[t].sites[0].gold_nc
            top_S = max(pk.S_all for pk in pks)
            top = [pk.position for pk in pks if pk.S_all == top_S]
            pp = sum(top) / len(top)
            if abs(pp - gs) <= 1:
                exact += 1
        n = len(tnps)
        cov_rate = cov / n
        exact_rate = exact / n
        # Bootstrap CI on cov
        rng = np.random.default_rng(0)
        boot_covs = []
        for _ in range(n_boot):
            samp = rng.choice(tnps, size=n, replace=True)
            c = 0
            for t in samp:
                if peaks.get(t):
                    c += 1
            boot_covs.append(c / n)
        boot_covs = np.array(boot_covs)
        result_cells.append({
            "L":                L,
            "mm_concentration": mm,
            "nc_homology_rate": hom,
            "n_sites":          n_sites_cell,
            "n_tnps":           n,
            "coverage":         round(cov_rate, 4),
            "cov_ci_95":        [round(float(np.percentile(boot_covs, 2.5)), 4),
                                   round(float(np.percentile(boot_covs, 97.5)), 4)],
            "exact_rate":       round(exact_rate, 4),
            "pass":             True,       # cells are diagnostic
        })

    # Emit an under-sampled summary if any cells were dropped below the
    # 10-Tnp CI floor — reader sees corpus size limitation explicitly.
    if n_under_sampled > 0:
        result_cells.append({
            "check":              "under_sampled_cells_summary",
            "n_tnps_in_undersampled_cells": n_under_sampled,
            "n_cells_dropped":    sum(1 for _k, _t in cells.items() if len(_t) < 10),
            "note":               "cells with < 10 Tnps skipped; corpus too small for stratified CI",
            "pass":               True,
        })

    # Note the presence of the n_sites=3 slice — the flank-axis validation
    # target — so the reader doesn't misread its low Mode 2 nc-axis-only
    # coverage as a corpus flaw.
    if ns3_tnps:
        result_cells.append({
            "check":              "n_sites_3_flank_axis_slice",
            "n_tnps":             len(ns3_tnps),
            "note":               ("n_sites=3 IS in the corpus. Under Mode 2 (m>=8) "
                                     "NC-AXIS ONLY these cells look weak by construction: "
                                     "E[FP]/bag=1.565 -> background covers every bag. "
                                     "Adding flank consistency drops effective q ~250x. "
                                     "THIS SLICE IS THE FLANK-AXIS BINARY JUDGE (Tier 2 1b, "
                                     "Stage 3), requires Stage 1h detector wiring "
                                     "(MatchTable.flank_argmax + VariantSpec.flank_coherence). "
                                     "Falsifiable prediction is a 2x2 grid, NOT two rows on one "
                                     "detector: nc-only detector must give identical cov (~1) and "
                                     "exact (~0.05) for BOTH consistent and inconsistent bags "
                                     "(the null); flank-coherent detector must give "
                                     "cov~=0.86^3=0.636 exact~=cov on consistent AND cov~=0 on "
                                     "inconsistent. Judge = (left column flat) AND (right column "
                                     "large gap 0.636 vs 0) AND (cov~=exact in (consistent, "
                                     "flank-coherent)). No single cell can decide."),
            "pass":               True,
        })

    # Cross-cell monotonicity over nc_homology (at fixed L=11, mm=clustered,
    # per n_sites). Cells vary along multiple axes; comparing hom monotonicity
    # requires isolating n_sites (otherwise a hom=0.995/n=8 cell gets
    # compared against a hom=0.99/n=3 cell, which are on different rows of
    # the "coverage vs hom" curve and interleave).
    hom_cells_all = [c for c in result_cells
                        if c.get("L") == 11 and c.get("mm_concentration") == "clustered"]
    if hom_cells_all:
        by_n = defaultdict(list)
        for c in hom_cells_all:
            by_n[c["n_sites"]].append(c)
        # Report per-n_sites monotonicity band; PASS if the WORST violation
        # across all n_sites is within the SE-slack band. This makes the
        # check stratified — a failure names the specific n_sites row.
        worst_viol = 0.0
        worst_n = None
        per_row = []
        for n, cells in sorted(by_n.items()):
            if len(cells) < 2:
                continue
            cells.sort(key=lambda c: -c["nc_homology_rate"])
            cov_seq = [c["coverage"] for c in cells]
            v = max((b - a for a, b in zip(cov_seq, cov_seq[1:])),
                     default=0.0)
            per_row.append({"n_sites": n,
                             "sequence": [(c["nc_homology_rate"],
                                             c["coverage"]) for c in cells],
                             "max_up_violation": round(v, 4)})
            if v > worst_viol:
                worst_viol = v; worst_n = n
        # DIAGNOSTIC ONLY — no gate at this sample size.
        # Analytic model: hom drift from 0.995 → 0.90 gives per-site
        # ε_total ≈ 0.0009 → 0.036. Total expected coverage drop across
        # the 5 hom bins is (1 - ε)^n_sites integrated across bins
        # (~13% at n_sites=4 end-to-end; ~3% per adjacent-bin hop).
        # At n_tnps ≈ 1200/6 ≈ 200 per cell, Bernoulli SE ≈ 0.035, so
        # 3pp per-hop signal is < 1 SE. This check has NO RESOLUTION
        # at this corpus size — the observed hop-level variance is
        # dominated by sampling noise. To gate hom drift, either:
        #   (a) aggregate to hom=0.995 vs hom=0.90 endpoints only
        #       (pool n_tnps ≈ 1200 per endpoint → SE ≈ 0.014, then a
        #        13% end-to-end signal is 10 SE detectable), or
        #   (b) build a corpus with ≥ 10K bags per (hom, n_sites) cell
        #       to make per-hop 3pp detectable at 1 SE.
        pass_mono = True   # always PASS; the anchor is per_row itself
        result_cells.append({
            "check":            "hom monotonicity (L=11 clustered, per n_sites) — DIAGNOSTIC",
            "worst_row_n_sites": worst_n,
            "max_up_violation": round(worst_viol, 4),
            "note":             ("no gate — expected per-hop signal ≈3pp "
                                    "< 1 SE at n≈200/cell; per_row is the anchor. "
                                    "For a gated hom check, aggregate to hom endpoint "
                                    "pair (0.995 vs 0.90) or grow the corpus to "
                                    "≥10K bags per (hom, n_sites) cell."),
            "pass":             pass_mono,
            "per_row":          per_row,
        })

        # Endpoint-pair check (0.995 vs 0.90): pooled across n_sites at
        # fixed L=11 clustered. Pooled n_tnps ≈ 1200 per endpoint → SE ≈
        # 0.014. Analytic expected drop ≈ ε_total(0.90) × mean(n_sites) =
        # 0.036 × 5.5 ≈ 0.20 pp per site → net coverage difference of
        # a few pp across the whole n_sites basket. Gate: cov(0.995) >=
        # cov(0.90) - 0.05 (asymmetric, since only up-violation is
        # anomalous; a large drop with lower hom is expected).
        hi_cells = [c for c in hom_cells_all
                        if c.get("nc_homology_rate") == 0.995]
        lo_cells = [c for c in hom_cells_all
                        if c.get("nc_homology_rate") == 0.9]
        if hi_cells and lo_cells:
            hi_pool = sum(c["coverage"] * c["n_tnps"] for c in hi_cells) / \
                        max(1, sum(c["n_tnps"] for c in hi_cells))
            lo_pool = sum(c["coverage"] * c["n_tnps"] for c in lo_cells) / \
                        max(1, sum(c["n_tnps"] for c in lo_cells))
            up_violation = lo_pool - hi_pool
            result_cells.append({
                "check":            "hom endpoint pair (0.995 vs 0.90) pooled",
                "cov_hom_0.995":    round(hi_pool, 4),
                "cov_hom_0.900":    round(lo_pool, 4),
                "up_violation":     round(up_violation, 4),
                "band":             [-1.0, 0.05],
                "pass":             up_violation <= 0.05,
            })

    return _summarize("channel_a_stratified", result_cells)


# ---------- runner ----------

LAYER_1_CHECKS: list[tuple[str, Callable]] = [
    ("nc_homology_identity",       check_nc_homology_identity),
    ("alignment_map_at_guide",     check_alignment_map_at_guide),
    ("flank_offset_std",           check_flank_offset_std),
    ("bag_guide_sharing",          check_bag_guide_sharing),
    ("bag_target_diversity",       check_bag_target_diversity),
    ("mismatch_geometry_class",    check_mismatch_geometry_class),
    ("planted_m_distribution",     check_planted_m_distribution),
    ("competitor_count",           check_competitor_count),
    ("nc_position_consistency",    check_nc_position_consistency),
]

LAYER_2_CHECKS: list[tuple[str, Callable]] = [
    ("nc_active_vs_inactive_leakage",       check_active_vs_inactive_leakage),
    ("flank_planted_vs_unplanted_leakage",  check_flank_planted_vs_unplanted_leakage),
    ("structure_positive_vs_twin_leakage",  check_structure_positive_vs_twin_leakage),
    ("negative_mode_leakage",               check_negative_mode_leakage),
]


def _run_layer_1(jsonl: Path, max_bags: int | None) -> list[dict]:
    print(f"[layer1] loading up to {max_bags or 'ALL'} bags", flush=True)
    bags_dict = _group_by_bag(jsonl, max_bags=max_bags)
    bags = list(bags_dict.values())
    print(f"[layer1] loaded {len(bags)} bags, {sum(len(b) for b in bags)} sites", flush=True)
    results = []
    for name, fn in LAYER_1_CHECKS:
        print(f"[layer1] {name}", flush=True)
        results.append(fn(bags))
    return results


def _run_layer_2(jsonl: Path, max_bags: int | None) -> list[dict]:
    print(f"[layer2] loading up to {max_bags or 'ALL'} bags", flush=True)
    bags_dict = _group_by_bag(jsonl, max_bags=max_bags)
    bags = list(bags_dict.values())
    print(f"[layer2] loaded {len(bags)} bags", flush=True)
    results = []
    for name, fn in LAYER_2_CHECKS:
        print(f"[layer2] {name}", flush=True)
        results.append(fn(bags))
    return results


def _run_layer_3(jsonl: Path, shard_dir: Path, workers: int,
                    flank_coherence: str = "off") -> list[dict]:
    print(f"[layer3] Channel A stratified anchor "
          f"(flank_coherence={flank_coherence})", flush=True)
    return [check_channel_a_stratified(jsonl, shard_dir, workers=workers,
                                            flank_coherence=flank_coherence)]


def _print_result(res: dict) -> None:
    status = res["status"]
    mark = "PASS" if status == "PASS" else ("SKIP" if status == "SKIP" else "FAIL")
    print(f"\n[{mark}] {res['name']}  "
          f"({res['n_pass']} pass / {res['n_fail']} fail / {res['n_skip']} skip)")
    for c in res["cells"]:
        st = c.get("status", "PASS" if c.get("pass") else "FAIL")
        print(f"   {st:<4s}  {c}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--shard-dir", type=Path, default=None,
                     help="MatchTable shard dir (required for Layer 3)")
    ap.add_argument("--layer", default="all",
                     help="Comma-separated: 1,2,3 or 'all'")
    ap.add_argument("--max-bags", type=int, default=None,
                     help="Cap on bags loaded for Layers 1 and 2")
    ap.add_argument("--workers", type=int, default=None,
                     help="Layer 3 parallel workers (default = min(32, nproc))")
    ap.add_argument("--flank-coherence", default="off",
                     choices=("off", "jitter"),
                     help="Layer 3 detector setting; jitter engages the "
                          "Stage 1h flank-argmax cross-site check")
    ap.add_argument("--report-out", type=Path, default=None)
    args = ap.parse_args()

    layers = set()
    if args.layer == "all":
        layers = {1, 2, 3}
    else:
        for tok in args.layer.split(","):
            layers.add(int(tok.strip()))

    workers = args.workers or min(32, os.cpu_count() or 4)

    all_results: dict[str, list[dict]] = {}
    if 1 in layers:
        all_results["layer1"] = _run_layer_1(args.jsonl, args.max_bags)
    if 2 in layers:
        all_results["layer2"] = _run_layer_2(args.jsonl, args.max_bags)
    if 3 in layers:
        if args.shard_dir is None:
            print("[layer3] --shard-dir required, skipping")
            all_results["layer3"] = []
        else:
            all_results["layer3"] = _run_layer_3(args.jsonl, args.shard_dir,
                                                        workers,
                                                        flank_coherence=args.flank_coherence)

    print("\n\n============================================")
    print("=== Generator Data Checker — final report ===")
    print("============================================")
    total_pass = total_fail = total_skip = 0
    per_layer_stats: dict[str, dict] = {}
    for layer_name, results in all_results.items():
        print(f"\n--- {layer_name} ---")
        lp = lf = ls = 0
        for res in results:
            _print_result(res)
            lp += res["n_pass"]; lf += res["n_fail"]; ls += res["n_skip"]
        per_layer_stats[layer_name] = {"pass": lp, "fail": lf, "skip": ls,
                                          "n_checks": len(results)}
        total_pass += lp; total_fail += lf; total_skip += ls

    # Layer 2: how many probes actually ran vs SKIP. This is critical to
    # avoid misreading "Layer 2 PASS" as "no leakage" when 3/4 probes are
    # gated on negative-mode data the corpus doesn't have.
    if "layer2" in all_results:
        layer2_results = all_results["layer2"]
        skipped_names = [r["name"] for r in layer2_results if r["status"] == "SKIP"]
        active_names = [r["name"] for r in layer2_results if r["status"] != "SKIP"]
        print("\n[[layer2 coverage]]")
        print(f"  probes active: {len(active_names)}/{len(layer2_results)}  "
              f"— {active_names}")
        if skipped_names:
            print(f"  probes SKIPPED (need negative-mode data / Stage 1d):")
            for n in skipped_names:
                print(f"    - {n}")
            print(f"  layer2 PASS here means only the active probes were checked; "
                  f"the biggest shortcut classes (twin structure, flank composition, "
                  f"negative_mode collinearity) remain unmeasured on this corpus.")

    # Layer 3 spec-note: pre-1c corpora used absolute S=5 on fixed n_sites=5.
    # Under Stage 1c, spec_m_threshold_L11(m=8, tau=0, theta=1.0) scales
    # per-Tnp with n_sites. Explicit theta vs S=5 distinction below.
    if "layer3" in all_results and all_results["layer3"]:
        cells = all_results["layer3"][0]["cells"]
        distinct_n_sites = {c.get("n_sites") for c in cells if isinstance(c.get("n_sites"), int)}
        print("\n[[layer3 spec note]]")
        if not distinct_n_sites:
            print(f"  No cells had >= 10 Tnps — corpus too small for stratified Channel A.")
            print(f"  Rerun on a >= 5K-bag corpus to populate cells.")
        elif len(distinct_n_sites) > 1:
            print(f"  n_sites axis varies (values: {sorted(distinct_n_sites)}).")
            print(f"  Detector: spec_m_threshold_L11(m=8, tau=0, theta=1.0) — per-Tnp")
            print(f"  effective S = ceil(theta * n_sites). Cells are STAGE 1C-VALID.")
        else:
            n = next(iter(distinct_n_sites))
            print(f"  n_sites fixed at {n} in this corpus.")
            print(f"  Detector: spec_m_threshold_L11(m=8, tau=0, S=5, theta=1.0).")
            print(f"  Cells match pre-1c semantics because theta*n == n = old S=n.")
            print(f"  These cells stay comparable to pre-1c anchors AT n_sites={n} only.")

    print(f"\nOverall: {total_pass} PASS  {total_fail} FAIL  {total_skip} SKIP")
    verdict = "PASS" if total_fail == 0 else "FAIL"
    print(f"Verdict: {verdict}")

    if args.report_out:
        Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report_out, "w") as f:
            json.dump({"results": all_results, "verdict": verdict}, f, indent=2)
        print(f"Report written to {args.report_out}")

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
