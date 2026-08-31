"""Layer 0-5 survival + FP profile — where does the signal get lost.

Layers:
    L0: raw sequence           (nc + flanks in memory)
    L1: match table            (int8 m_max arrays)
    L2: admission applied      (site_hit sets after m/E filter)
    L3: per-site kernel        (Gaussian tau smoothing per site)
    L4: cross-site aggregation (sum over sites, then peak detection)
    L5: peaks emitted          (thresh + local max)

Survival test (P1 on V4.2, planted-c* known):
    for each Tnp with a known gold_nc, at each layer, is the gold still
    represented above the noise floor at its true position?

FP profile (P2 on negative families):
    per layer, what fraction of negative Tnps have SOMETHING that would
    survive to the next layer? Isolates which layer contributes the specificity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .match_table import MatchTable, Orient
from .metrics import Ratio, RatioCI, as_ratio_ci
from .variant import (VariantSpec, _apply_kernel_max, _admitted_positions,
                       _E_table, _fixed_L_hits, _min_over_L_hits, _find_peaks)


Layer = Literal["L1_match", "L2_admission", "L3_kernel", "L4_aggregate", "L5_peaks"]
LAYERS: tuple[Layer, ...] = ("L1_match", "L2_admission", "L3_kernel",
                              "L4_aggregate", "L5_peaks")

# Convention: L1 and L2 use binary presence (gold_present).
# L3, L4, L5 use rank + margin — the kernel does not delete hits, so binary
# "gold_present" at L3 is trivially True; useful signal is rank and margin.
# L4 is derived from the same aggregated S as L3, but with the S_threshold cut
# applied as a *summary* pass/fail (kept for interpretability). L5 is the
# emitted-peak view; distance-to-gold + rank among peaks.


@dataclass(frozen=True)
class SurvivalRow:
    tnp_id: str
    layer: Layer
    gold_present: bool
    gold_rank_at_layer: int | None
    top_score_at_gold: float
    top_score_elsewhere: float
    stratum: dict


def _gold_survives_L1(mt: MatchTable, tnp_id: str, spec: VariantSpec,
                       gold_nc: int, gold_L: int) -> SurvivalRow:
    """L1: does the max m at gold position exceed the noise floor?"""
    tnp = mt.tnps[tnp_id]
    scores_at_gold = []
    scores_elsewhere: list[float] = []
    for s in tnp.sites:
        for orient in mt.orients:
            for L in spec.L_value:
                arr = mt.m_max(tnp_id, s.site_idx, orient, L, excl_w=0)
                if gold_nc < len(arr):
                    scores_at_gold.append(int(arr[gold_nc]))
                mask = np.ones(len(arr), dtype=bool)
                lo = max(0, gold_nc - 3); hi = min(len(arr), gold_nc + 4)
                mask[lo:hi] = False
                if mask.any():
                    scores_elsewhere.append(float(arr[mask].max()))
    top_g = float(max(scores_at_gold)) if scores_at_gold else 0.0
    top_e = float(max(scores_elsewhere)) if scores_elsewhere else 0.0
    return SurvivalRow(
        tnp_id=tnp_id, layer="L1_match",
        gold_present=top_g > 0,
        gold_rank_at_layer=None,
        top_score_at_gold=top_g, top_score_elsewhere=top_e,
        stratum={"m_at_gold": top_g, "m_elsewhere": top_e},
    )


def _gold_survives_L2(mt: MatchTable, tnp_id: str, spec: VariantSpec,
                       gold_nc: int) -> SurvivalRow:
    """L2: after admission, is gold_nc in at least one site's admitted set?"""
    tnp = mt.tnps[tnp_id]
    n_sites_hit = 0
    n_sites_hit_elsewhere = 0
    for s in tnp.sites:
        flank_len = len(s.flank)
        got_gold = False
        got_other = 0
        for orient in mt.orients:
            if spec.L_mode == "fixed":
                hits = _fixed_L_hits(mt, tnp_id, s.site_idx, orient, spec, flank_len)
            else:
                hits = _min_over_L_hits(mt, tnp_id, s.site_idx, orient, spec, flank_len)
            if any(abs(h - gold_nc) <= 1 for h in hits):
                got_gold = True
            got_other += sum(1 for h in hits if abs(h - gold_nc) > 5)
        if got_gold:
            n_sites_hit += 1
        n_sites_hit_elsewhere += got_other
    return SurvivalRow(
        tnp_id=tnp_id, layer="L2_admission",
        gold_present=n_sites_hit > 0,
        gold_rank_at_layer=None,
        top_score_at_gold=float(n_sites_hit),
        top_score_elsewhere=float(n_sites_hit_elsewhere),
        stratum={"sites_hitting_gold": n_sites_hit,
                 "sites_hitting_other_nc_pos": n_sites_hit_elsewhere},
    )


def _aggregated_S(mt: MatchTable, tnp_id: str, spec: VariantSpec
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Return (S_pooled, per_orient_S_stacked) — used by L3 and L4."""
    tnp = mt.tnps[tnp_id]
    ref_L = spec.L_value[0]
    nc_len_pos = len(tnp.nc) - ref_L + 1
    per_orient: list[np.ndarray] = []
    for orient in mt.orients:
        hits_lists = []
        for s in tnp.sites:
            flank_len = len(s.flank)
            if spec.L_mode == "fixed":
                h = _fixed_L_hits(mt, tnp_id, s.site_idx, orient, spec, flank_len)
            else:
                h = _min_over_L_hits(mt, tnp_id, s.site_idx, orient, spec, flank_len)
            hits_lists.append(h)
        per_orient.append(_apply_kernel_max(hits_lists, nc_len_pos, spec.tau))
    S_stack = np.stack(per_orient)
    S_pooled = S_stack.max(axis=0)
    return S_pooled, S_stack


def _gold_survives_L3(mt: MatchTable, tnp_id: str, spec: VariantSpec,
                       gold_nc: int) -> SurvivalRow:
    """L3 (kernel + per-site aggregation): rank of gold_nc + margin.

    Binary 'gold_present' is trivially True here (kernel does not zero hits).
    The informative signal is where gold ranks in the S distribution and how
    large the margin is to the top competitor.
    """
    S_pooled, S_stack = _aggregated_S(mt, tnp_id, spec)
    if len(S_pooled) == 0 or gold_nc >= len(S_pooled):
        return SurvivalRow(tnp_id=tnp_id, layer="L3_kernel",
                            gold_present=True, gold_rank_at_layer=None,
                            top_score_at_gold=0.0, top_score_elsewhere=0.0,
                            stratum={"note": "gold_nc out of range"})
    if spec.orient_constraint:
        S_at_gold = float(max(so[gold_nc] for so in S_stack))
    else:
        S_at_gold = float(S_pooled[gold_nc])
    lo = max(0, gold_nc - 3); hi = min(len(S_pooled), gold_nc + 4)
    S_masked = S_pooled.copy()
    S_masked[lo:hi] = 0.0
    S_elsewhere = float(S_masked.max()) if len(S_masked) else 0.0
    rank = int((S_pooled > S_at_gold).sum())
    return SurvivalRow(
        tnp_id=tnp_id, layer="L3_kernel",
        gold_present=True,      # trivially true; use rank/margin instead
        gold_rank_at_layer=rank,
        top_score_at_gold=S_at_gold,
        top_score_elsewhere=S_elsewhere,
        stratum={"S_at_gold": S_at_gold,
                 "S_max_elsewhere": S_elsewhere,
                 "margin": S_at_gold - S_elsewhere,
                 "rank": rank},
    )


def _gold_survives_L4(mt: MatchTable, tnp_id: str, spec: VariantSpec,
                       gold_nc: int) -> SurvivalRow:
    """L4 (threshold): does S(gold_nc) reach S_threshold - 0.5?"""
    S_pooled, S_stack = _aggregated_S(mt, tnp_id, spec)
    if len(S_pooled) == 0 or gold_nc >= len(S_pooled):
        return SurvivalRow(tnp_id=tnp_id, layer="L4_aggregate",
                            gold_present=False, gold_rank_at_layer=None,
                            top_score_at_gold=0.0, top_score_elsewhere=0.0,
                            stratum={"note": "gold_nc out of range"})
    if spec.orient_constraint:
        S_at_gold = float(max(so[gold_nc] for so in S_stack))
    else:
        S_at_gold = float(S_pooled[gold_nc])
    lo = max(0, gold_nc - 3); hi = min(len(S_pooled), gold_nc + 4)
    S_masked = S_pooled.copy()
    S_masked[lo:hi] = 0.0
    thresh = float(spec.S_threshold) - 0.5
    return SurvivalRow(
        tnp_id=tnp_id, layer="L4_aggregate",
        gold_present=S_at_gold >= thresh,
        gold_rank_at_layer=int((S_pooled > S_at_gold).sum()),
        top_score_at_gold=S_at_gold,
        top_score_elsewhere=float(S_masked.max()),
        stratum={"S_at_gold": S_at_gold,
                 "S_max_elsewhere": float(S_masked.max()),
                 "threshold": thresh,
                 "margin_to_threshold": S_at_gold - thresh},
    )


def _gold_survives_L5(mt: MatchTable, tnp_id: str, spec: VariantSpec,
                       gold_nc: int) -> SurvivalRow:
    """L5: after peak-finding, is a peak within 1 nt of gold_nc?"""
    from .variant import run_variant
    peaks = run_variant(mt, spec).get(tnp_id, [])
    if not peaks:
        return SurvivalRow(tnp_id=tnp_id, layer="L5_peaks",
                            gold_present=False, gold_rank_at_layer=None,
                            top_score_at_gold=0.0, top_score_elsewhere=0.0,
                            stratum={"n_peaks": 0})
    best = min(peaks, key=lambda pk: abs(pk.position - gold_nc))
    return SurvivalRow(
        tnp_id=tnp_id, layer="L5_peaks",
        gold_present=abs(best.position - gold_nc) <= 1,
        gold_rank_at_layer=None,
        top_score_at_gold=float(best.score),
        top_score_elsewhere=float(max((pk.score for pk in peaks
                                        if abs(pk.position - gold_nc) > 5),
                                       default=0.0)),
        stratum={"n_peaks": len(peaks),
                 "best_dist": abs(best.position - gold_nc)},
    )


def survival_table(mt_pos: MatchTable, spec: VariantSpec
                    ) -> list[SurvivalRow]:
    """Full survival test on a positive dataset. Returns one row per (tnp, layer)."""
    rows: list[SurvivalRow] = []
    for tnp_id in mt_pos.tnp_ids:
        tnp = mt_pos.tnps[tnp_id]
        if not tnp.sites or tnp.sites[0].gold_nc is None:
            continue
        gold_nc = tnp.sites[0].gold_nc
        gold_L = tnp.sites[0].gold_L or spec.L_value[0]
        rows.append(_gold_survives_L1(mt_pos, tnp_id, spec, gold_nc, gold_L))
        rows.append(_gold_survives_L2(mt_pos, tnp_id, spec, gold_nc))
        rows.append(_gold_survives_L4(mt_pos, tnp_id, spec, gold_nc))
        rows.append(_gold_survives_L5(mt_pos, tnp_id, spec, gold_nc))
    return rows


def survival_summary(rows: list[SurvivalRow], stratify_by: str | None = None
                     ) -> dict:
    """Aggregate survival by layer (optionally within strata).

    stratify_by: name of a `stratum` key to group by (e.g. 'm_at_gold').
    """
    by_layer: dict[str, dict] = {}
    for layer in ("L1_match", "L2_admission", "L4_aggregate", "L5_peaks"):
        r = [row for row in rows if row.layer == layer]
        n = len(r)
        surv = sum(1 for row in r if row.gold_present)
        d = {"n": n, "gold_survives": surv, "rate": surv / max(1, n)}
        if stratify_by:
            strata: dict = {}
            for row in r:
                key = row.stratum.get(stratify_by, "unknown")
                strata.setdefault(key, {"n": 0, "surv": 0})
                strata[key]["n"] += 1
                strata[key]["surv"] += int(row.gold_present)
            d["strata"] = {k: {**v, "rate": v["surv"] / max(1, v["n"])}
                            for k, v in strata.items()}
        by_layer[layer] = d
    return by_layer


# ---------- FP profile ----------

@dataclass(frozen=True)
class FPLayerRow:
    tnp_id: str
    family: str
    layer: Layer
    passed: bool
    stratum: dict


def fp_profile(mt_neg: MatchTable, spec: VariantSpec) -> list[FPLayerRow]:
    """Per-Tnp per-layer pass indicator on a negative family.

    L2 pass: at least one site admits SOME position.
    L4 pass: some nc position has cross-site aggregate >= S_threshold - 0.5.
    L5 pass: at least one peak is emitted.
    """
    rows: list[FPLayerRow] = []
    ref_L = spec.L_value[0]
    for tnp_id in mt_neg.tnp_ids:
        tnp = mt_neg.tnps[tnp_id]
        family = tnp.family
        nc_len_pos = len(tnp.nc) - ref_L + 1

        any_admit = False
        max_S = 0.0
        for orient in mt_neg.orients:
            hits_lists = []
            for s in tnp.sites:
                flank_len = len(s.flank)
                if spec.L_mode == "fixed":
                    h = _fixed_L_hits(mt_neg, tnp_id, s.site_idx, orient, spec, flank_len)
                else:
                    h = _min_over_L_hits(mt_neg, tnp_id, s.site_idx, orient, spec, flank_len)
                if h:
                    any_admit = True
                hits_lists.append(h)
            S = _apply_kernel_max(hits_lists, nc_len_pos, spec.tau)
            if len(S):
                max_S = max(max_S, float(S.max()))

        rows.append(FPLayerRow(tnp_id, family, "L2_admission",
                                any_admit, {}))
        rows.append(FPLayerRow(tnp_id, family, "L4_aggregate",
                                max_S >= float(spec.S_threshold) - 0.5,
                                {"max_S": max_S}))

        from .variant import run_variant
        peaks = run_variant(mt_neg, spec).get(tnp_id, [])
        rows.append(FPLayerRow(tnp_id, family, "L5_peaks",
                                bool(peaks),
                                {"n_peaks": len(peaks)}))
    return rows


def fp_profile_summary(rows: list[FPLayerRow]) -> dict:
    """Aggregate FP profile by (family, layer). Returns CP CIs."""
    by_fam_layer: dict[tuple[str, str], dict] = {}
    fams = sorted({r.family for r in rows})
    layers = ("L2_admission", "L4_aggregate", "L5_peaks")
    for fam in fams:
        for layer in layers:
            r = [row for row in rows if row.family == fam and row.layer == layer]
            n = len(r)
            passed = sum(1 for row in r if row.passed)
            ci = as_ratio_ci(passed, n, f"N_tnp_{fam}")
            by_fam_layer[(fam, layer)] = {
                "n": n, "passed": passed,
                "rate": ci.value, "ci_lo": ci.lo, "ci_hi": ci.hi,
            }
    return {f"{fam}|{layer}": v for (fam, layer), v in by_fam_layer.items()}
