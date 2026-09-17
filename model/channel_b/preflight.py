"""Preflight channel-stats check. Call before model.forward() in every inference script.

Prints per-channel-group (min, max, mean, frac_zero, frac_nan) for the batch tensor,
compared to reference statistics if provided. Aborts with a descriptive error if
`frac_zero` for any group differs from the reference by more than `zero_thresh`
(default 0.2) — this catches silent-zero-fill bugs (like the Durrant shard's missing
flank_argmax_by_excl issue in 2026-09-08) BEFORE they contaminate downstream analysis.

Two prior incidents in this project:
1. `zeroing` floor-effect (2026-08-29-ish): zeroing structure at inference on a
   non-zero-trained model produced OOD collapse.
2. Durrant `flank_argmax = None` silent zero-fill (2026-09-08): ChannelBDataset's
   `if a_arr is not None` check silently skipped, main B was fed all-zero ch 9-12
   for 4+ rounds of Durrant experiments. Findings retracted.

Use as:
    from model.channel_b.preflight import channel_stats_preflight
    # x is the (B, N_sites, P, N_channels) tensor before model.forward
    channel_stats_preflight(x, reference_frac_zero=REF_STATS)
"""
from __future__ import annotations
import numpy as np
import torch

# 15-channel layout (from constants.py + data.py:319-334):
#   ch 0-3  : m_max per L (per-site, per-position)
#   ch 4-7  : structure (bag-level, broadcast across sites)
#   ch 8    : structure_valid mask
#   ch 9-12 : flank_dev per L (per-site)
#   ch 13-14: orient one-hot (bag-level, broadcast)
CHANNEL_GROUPS: dict[str, tuple[int, int]] = {
    "m_max (ch 0-3)":         (0, 4),
    "structure (ch 4-7)":     (4, 8),
    "struct_valid (ch 8)":    (8, 9),
    "flank_dev (ch 9-12)":    (9, 13),
    "orient one-hot (ch 13-14)": (13, 15),
}

# Reference stats MEASURED on v6r2 val (ACTIVE SITES ONLY, 2026-09-09, job 25746357,
# 15 bags across all 5 corpora). frac_zero uses site_mask masking; quantiles computed
# on finite values within active sites. See scripts/verify_failfast_and_ref.py.
#
# CAVEAT on 'flank_dev (ch 9-12)' frac_zero:
#   flank_dev = argmax - bag_median where median is over ACTIVE sites. For odd
#   n_sites_real (3, 5, 7), one site IS exactly the median → flank_dev = 0 by
#   construction on that site. So frac_zero on flank_dev has a per-n_sites structure:
#     n=3: ~0.33 (1 of 3 sites is median)
#     n=4: ~0.00 (median is mean of two, rarely exact zero)
#     n=5: ~0.20 (1 of 5)
#     n=6: ~0.00
#     n=7: ~0.14
#     n=8: ~0.00
#   Aggregate mean of ~0.19 is dominated by odd-n bags. This is NOT a defect — it's
#   the median-subtraction convention. Quantile check (p05, p95) is more reliable
#   for flank_dev; prefer quantile_thresh over zero_thresh when auditing this channel.
#
# The 2026-09-08 Durrant bug (ch 9-12 all-zero, frac_zero=1.0) is caught either way:
#   Δfrac_zero vs ref ≈ +0.81; quantile drift p05: 0 vs ref -4.24 (rel diff huge).
REF_TRAINING: dict[str, dict[str, float]] = {
    # RE-MEASURED 2026-09-10 under the new loader (arch.orient GOLD leak
    # removed; ch 0-3 = elementwise max over orients; ch 9-12 = argmax of
    # winning orient; ch 13-14 = zero). n=250 v6r2 val bags (5 corpora × 50).
    # Old (pre-2026-09-10) values kept in `_LEGACY_REF_TRAINING_PRE_ORIENT_FIX`
    # below for audit.
    "m_max (ch 0-3)": {
        "frac_zero": 0.0230,  # ± 0.0178  n=250  — was 0.0060 under old loader
        "p05":  0.4056,       # ± 0.1348           was 0.4163
        "p50":  0.5790,       # ± 0.0181           was 0.5227 (+0.056, double-orient max)
        "p95":  0.6858,       # ± 0.0349           was 0.6675
    },
    "structure (ch 4-7)": {
        "frac_zero": 0.0187,  # ± 0.0179  — was 0.0000, tiny shift; structure unaffected by loader change
        "p05":  0.1535,       # ± 0.1496
        "p50":  1.0771,       # ± 0.2486
        "p95":  3.7633,       # ± 1.5100  (40% σ — WIDE)
        # Per-channel quantile_thresh override — structure floats have intrinsically
        # wide σ (per-bag ncRNA structure varies). Global default 0.30 → false alarms;
        # override to 1.0 = ~2.5σ. Read by preflight via
        # reference[group].get("quantile_thresh", global default).
        "quantile_thresh": 1.0,
    },
    "struct_valid (ch 8)": {
        "frac_zero": 0.0187,  # ± 0.0179  n=250
        "p05":  0.9520,       # ± 0.2138
        "p50":  1.0000,
        "p95":  1.0000,
        # NOTE 2026-09-10: the OLD ref value 0.0000 for frac_zero was
        # MISLEADING — small-sample artifact. Follow-up investigation (5
        # corpora × 50 val bags) found 44-78% of bags have AT LEAST ONE
        # struct_valid=0 position; per-position rate is 0.77-2.30% depending
        # on corpus (ctrl10k/scat10k highest). This is not a defect (real
        # ViennaRNA fold-invalid positions in v6r2). Any preflight run
        # under the old ref that reported "no drift" on struct_valid was
        # not actually validating anything: 0.0 reference is unreachable by
        # real data, so the check was inert. New reference reflects reality.
    },
    "flank_dev (ch 9-12)": {
        "frac_zero": 0.1007,  # ± 0.1078  — SKIPPED in preflight (see caveat above)
        "p05": -4.4093,       # ± 0.4996  — was -4.0391 (winning-orient argmax is more spread)
        "p50":  0.0000,       # ± 0.0000  — median subtraction pins this
        "p95":  5.6195,       # ± 0.5296  — was 5.9091
    },
    "orient one-hot (ch 13-14)": {
        "frac_zero": 1.0000,  # ± 0.0000  — ch 13-14 constant zero after 2026-09-10 rev
        "p05":  0.0000,
        "p50":  0.0000,       # was 0.5 (one-hot); now zero
        "p95":  0.0000,       # was 1.0
    },
}


# Legacy REF (pre-2026-09-10 arch.orient leak fix). Kept for audit — any
# analysis run against v6r2 shards under the OLD loader path should use
# these; new loader uses REF_TRAINING above.
_LEGACY_REF_TRAINING_PRE_ORIENT_FIX: dict[str, dict[str, float]] = {
    "m_max (ch 0-3)":        {"frac_zero": 0.0060, "p05":  0.4163, "p50":  0.5227, "p95":  0.6675},
    "structure (ch 4-7)":    {"frac_zero": 0.0000, "p05":  0.2429, "p50":  1.3423, "p95":  4.7477, "quantile_thresh": 1.0},
    "struct_valid (ch 8)":   {"frac_zero": 0.0000, "p05":  1.0000, "p50":  1.0000, "p95":  1.0000},
    "flank_dev (ch 9-12)":   {"frac_zero": 0.1058, "p05": -4.0391, "p50":  0.0000, "p95":  5.9091},
    "orient one-hot (ch 13-14)": {"frac_zero": 0.5000, "p05":  0.0000, "p50":  0.5000, "p95":  1.0000},
}

# Legacy padding-inclusive reference (kept for audit; DO NOT USE for new code).
_LEGACY_REF_FRAC_ZERO_TRAINING_PADDING_INCLUSIVE: dict[str, float] = {
    "m_max (ch 0-3)":         0.5033,
    "structure (ch 4-7)":     0.5000,
    "struct_valid (ch 8)":    0.5000,
    "flank_dev (ch 9-12)":    0.5873,
    "orient one-hot (ch 13-14)": 0.7500,
}
# Backward-compat alias: any script importing REF_FRAC_ZERO_TRAINING still works,
# but its values are now the un-masked legacy ones. Explicit deprecation.
REF_FRAC_ZERO_TRAINING = _LEGACY_REF_FRAC_ZERO_TRAINING_PADDING_INCLUSIVE


def channel_stats_preflight(
    x: "torch.Tensor | np.ndarray",
    site_mask: "torch.Tensor | np.ndarray | None" = None,
    reference: dict | None = None,
    zero_thresh: float = 0.15,
    quantile_thresh: float = 0.30,
    abort_on_mismatch: bool = True,
    label: str = "",
) -> dict[str, dict[str, float]]:
    """Print per-channel-group stats on ACTIVE sites only; abort on drift from reference.

    Padding sites (site_mask=False) contribute all zeros and dominate raw frac_zero.
    Masking to site_mask=True decouples the stat from n_sites variation across bags.

    Args:
        x: (B, N_sites, P, N_channels) tensor or ndarray
        site_mask: (B, N_sites) bool tensor; True = real site, False = padding.
            If None, all sites treated as real (backward-compat).
        reference: optional {group_name: {'frac_zero': v, 'p05': v, 'p50': v, 'p95': v}}.
            Any provided value gets checked; missing ones skipped.
        zero_thresh: max allowed |observed − ref_frac_zero| before abort (default 0.15
            after masking — tighter than the old padding-inclusive 0.30).
        quantile_thresh: max allowed relative-difference in each of p05/p50/p95:
            |obs - ref| / (|ref| + 1e-6) — catches scale/offset drift not visible
            in frac_zero alone.
        abort_on_mismatch: raise RuntimeError if any group violates threshold.
        label: printed in the header.

    Returns:
        dict per group: {min, max, mean, frac_zero, frac_nan, p05, p50, p95}
    """
    if isinstance(x, torch.Tensor):
        xn = x.detach().cpu().float().numpy()
    else:
        xn = np.asarray(x, dtype=np.float32)
    assert xn.ndim == 4, f"expected 4D (B, N_sites, P, N_channels), got {xn.shape}"
    n_ch = xn.shape[-1]
    assert n_ch == 15, f"expected 15 channels, got {n_ch}"

    B, N_sites, P, _ = xn.shape
    if site_mask is None:
        mask_np = np.ones((B, N_sites), dtype=bool)
    else:
        mask_np = (site_mask.detach().cpu().numpy() if isinstance(site_mask, torch.Tensor)
                    else np.asarray(site_mask, dtype=bool))
    assert mask_np.shape == (B, N_sites), f"site_mask shape {mask_np.shape} != ({B}, {N_sites})"

    n_active = int(mask_np.sum())
    n_padded = int((~mask_np).sum())
    print(f"[preflight]{' ' + label if label else ''} shape={tuple(xn.shape)}  "
          f"active_sites={n_active}  padded_sites={n_padded}")
    print(f"  {'channel group':<28s}  {'frac_zero':>10s}  {'p05':>8s}  {'p50':>8s}  {'p95':>8s}  "
          f"{'refZ':>6s}  {'Δref':>6s}  {'flags':<20s}")

    results: dict[str, dict[str, float]] = {}
    violations: list[str] = []
    for name, (a, b) in CHANNEL_GROUPS.items():
        block = xn[..., a:b]                            # (B, N_sites, P, C_group)
        # Mask along the site axis
        mask_full = mask_np[:, :, None, None]           # broadcast to (B, N_sites, 1, 1)
        block_active = block[np.broadcast_to(mask_full, block.shape)]
        if block_active.size == 0:
            print(f"  {name:<28s}  no active data")
            continue
        frac_zero = float((block_active == 0).mean())
        frac_nan  = float(np.isnan(block_active).mean())
        finite = block_active[np.isfinite(block_active)]
        if finite.size:
            p05 = float(np.percentile(finite, 5))
            p50 = float(np.percentile(finite, 50))
            p95 = float(np.percentile(finite, 95))
        else:
            p05 = p50 = p95 = float("nan")
        stats = {"frac_zero": frac_zero, "frac_nan": frac_nan,
                  "p05": p05, "p50": p50, "p95": p95}
        results[name] = stats

        # Compare to reference (if provided)
        ref = (reference or {}).get(name, {})
        flags = []
        ref_z_str = "—"; dz_str = "—"
        # flank_dev's frac_zero has a per-n_sites structure (median-subtraction pins
        # 1/n sites to exactly 0 for odd n); the aggregate mean is misleading and
        # a fixed zero_thresh false-alarms at both n=3 (33%) and n=8 (0%). Skip the
        # frac_zero check for flank_dev entirely — rely on quantile drift, which is
        # stable and catches the Durrant-bug class of degeneracy (p05 = 0 vs ref
        # -4.24 is a huge signal).
        skip_fzero = (name == "flank_dev (ch 9-12)")
        if "frac_zero" in ref and not skip_fzero:
            rz = ref["frac_zero"]
            dz = frac_zero - rz
            ref_z_str = f"{rz:.3f}"; dz_str = f"{dz:+.3f}"
            if abs(dz) > zero_thresh:
                flags.append(f"fZ Δ={dz:+.2f}")
        elif skip_fzero and "frac_zero" in ref:
            ref_z_str = f"{ref['frac_zero']:.3f}*"; dz_str = "skip*"
        # Per-channel quantile_thresh override (some channels have intrinsically
        # wider σ and need a looser threshold to avoid false alarms).
        q_thresh = ref.get("quantile_thresh", quantile_thresh)
        for pname in ("p05", "p50", "p95"):
            if pname in ref:
                rq = ref[pname]
                # Absolute tolerance for near-zero references (avoid dividing by ~0)
                abs_tol = 0.5
                diff = stats[pname] - rq
                if abs(rq) < abs_tol:
                    if abs(diff) > abs_tol * q_thresh:
                        flags.append(f"{pname} Δ={diff:+.2g}")
                else:
                    rel = abs(diff) / (abs(rq) + 1e-6)
                    if rel > q_thresh:
                        flags.append(f"{pname} rel={rel:.2f}")
        flags_str = ",".join(flags) if flags else ""
        print(f"  {name:<28s}  {frac_zero:>10.4f}  {p05:>8.3g}  {p50:>8.3g}  {p95:>8.3g}  "
              f"{ref_z_str:>6s}  {dz_str:>6s}  {flags_str:<20s}")
        if flags:
            violations.append(f"{name}: {'; '.join(flags)}")

    if violations and abort_on_mismatch:
        msg = ("[preflight] ABORT — input distribution drifted from training reference:\n  "
               + "\n  ".join(violations)
               + "\n[preflight] Possible causes: silent zero-fill, loader bug, wrong "
                 "preprocessing path, scale/offset drift. Verify inputs before running.")
        raise RuntimeError(msg)
    if violations:
        print(f"[preflight] WARNING — {len(violations)} channel group(s) drifted; "
              f"proceeding because abort_on_mismatch=False")
    return results
