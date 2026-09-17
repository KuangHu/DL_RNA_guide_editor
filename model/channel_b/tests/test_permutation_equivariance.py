"""C3 pre-registered gate: shuffling site order per bag must give
AUROC = 0.500 exactly. That gate is at eval time on trained weights.
This unit test is the construction check on UNTRAINED weights — if the
architecture itself is permutation-equivariant on the site axis, then
the bag-level logit is INVARIANT to site permutations (any permutation
produces the same output).

Measurement (2026-09-04 first run with hidden=64/n_heads=4/n_blocks=2):
  max_over_permutations |logit_perm − logit_orig| = 8.94e-08

The architecture IS permutation-equivariant MATHEMATICALLY on the site
axis (all site-axis operations — MHA over sites, masked mean pool —
would be strict-equivariant with exact arithmetic). The 8.94e-08
residual comes from float32 accumulation order: `softmax(QK^T) @ V`
matmul reduces along the site axis in whatever order cuBLAS/MKL/CPU
picked, and the masked-mean site pool sums in the site's *presented*
order. Neither can be made bit-exact without switching to fp64 or a
deterministic-reduction attention kernel, both at compute cost.

The pragmatic bound is < 1e-5:
  - safely above the observed 1e-7 (float32 rounding envelope)
  - safely below any downstream measurable effect (C3 SE ≈ 1e-3
    at n=5000, so a 1e-5 logit shift is 8 orders under gate resolution)
  - a regression to > 1e-5 indicates a real order-dependent layer (e.g.
    accidentally using BatchNorm somewhere, or a layer indexing sites
    directly by position) and MUST be investigated
"""
from __future__ import annotations
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from model.channel_b.model import ChannelBModel
from model.channel_b.constants import MAX_N_SITES, N_CHANNELS


def _make_random_bag(n_sites_real: int, nc_len_eff: int,
                          seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    x = torch.zeros((MAX_N_SITES, nc_len_eff, N_CHANNELS), dtype=torch.float32)
    x[:n_sites_real] = torch.randn(
        (n_sites_real, nc_len_eff, N_CHANNELS), generator=rng
    )
    site_mask = torch.zeros((MAX_N_SITES,), dtype=torch.bool)
    site_mask[:n_sites_real] = True
    return x, site_mask


def test_site_permutation_bit_exact():
    """Site permutation invariance on untrained weights.
    Output is (B, P) per-position score; every position must be invariant
    (within float32 envelope) to site permutation. 20 random permutations
    on a single bag with 8 real sites."""
    torch.manual_seed(0)
    model = ChannelBModel(hidden=64, n_heads=4, n_blocks=2).eval()
    x, site_mask = _make_random_bag(n_sites_real=MAX_N_SITES, nc_len_eff=40, seed=1)
    x, site_mask = x.unsqueeze(0), site_mask.unsqueeze(0)   # (1, S, P, C)
    with torch.no_grad():
        base_out = model(x, site_mask)   # (1, P)

    max_abs_diff = 0.0
    for k in range(20):
        perm = torch.randperm(MAX_N_SITES, generator=torch.Generator().manual_seed(k + 100))
        x_perm = x[:, perm, :, :]
        site_mask_perm = site_mask[:, perm]
        with torch.no_grad():
            perm_out = model(x_perm, site_mask_perm)   # (1, P)
        d = (perm_out - base_out).abs().max().item()
        max_abs_diff = max(max_abs_diff, d)
    # < 1e-5 (float32 rounding envelope; see docstring for rationale)
    assert max_abs_diff < 1e-5, (
        f"Permutation invariance failed: max |Δ per-position score| = {max_abs_diff:.6e} "
        f"across 20 permutations, above 1e-5 float32 bound. Some site-axis "
        f"operation has a REAL order-dependent reduction (not just float rounding). "
        f"Investigate before training."
    )


def test_site_permutation_bit_exact_with_padding():
    """Same test but n_sites_real=3 (the minimum from arch.py) with 5
    padded sites. Padding sites must be ignored; every position of the
    (1, P) output vector must be invariant to the 3! permutations of the
    real sites."""
    torch.manual_seed(0)
    model = ChannelBModel(hidden=64, n_heads=4, n_blocks=2).eval()
    x, site_mask = _make_random_bag(n_sites_real=3, nc_len_eff=40, seed=2)
    x, site_mask = x.unsqueeze(0), site_mask.unsqueeze(0)
    with torch.no_grad():
        base_out = model(x, site_mask)   # (1, P)

    max_abs_diff = 0.0
    real_idxs = [0, 1, 2]
    pad_idxs = [3, 4, 5, 6, 7]
    from itertools import permutations
    for k, perm_real in enumerate(list(permutations(real_idxs))[:6]):  # 3! = 6
        perm = list(perm_real) + pad_idxs   # keep padding at same positions
        perm_t = torch.tensor(perm)
        x_perm = x[:, perm_t, :, :]
        site_mask_perm = site_mask[:, perm_t]
        with torch.no_grad():
            perm_out = model(x_perm, site_mask_perm)   # (1, P)
        d = (perm_out - base_out).abs().max().item()
        max_abs_diff = max(max_abs_diff, d)
    assert max_abs_diff < 1e-5, (
        f"Permutation invariance failed with 3 real + 5 pad sites: "
        f"max |Δ per-position score| = {max_abs_diff:.6e} (above 1e-5 float32 bound)"
    )


if __name__ == "__main__":
    import subprocess
    r = subprocess.run(["pytest", __file__, "-v"],
                          cwd=str(Path(__file__).resolve().parents[3]))
    sys.exit(r.returncode)
