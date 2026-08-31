"""V1 model shape + gradient tests.

Runs entirely inside the opfi conda env (which has torch 2.6.0+cu124).
Uses synthetic input first (fast), then a real 4-tnp × 8-site val batch.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model.v1 import (
    ALIGN_FEATURE_INDICES,
    POS_FEATURE_INDICES,
    CandidateEncoder,
    GatedAttentionMIL,
    PMA,
    SetTransformerBlock,
    V1Config,
    V1Model,
    v1_loss,
)


def encoder_shape_smoke():
    cfg = V1Config()
    enc = CandidateEncoder(cfg)
    B, S, N, K = 2, 3, 3, cfg.num_candidates
    patches = torch.randn(B, S, N, K, cfg.patch_width, cfg.patch_channels)
    feats = torch.randn(B, S, N, K, cfg.num_features)
    zs, za, zp = enc(patches, feats)
    assert zs.shape == (B, S, N, K, cfg.struct_out)
    assert za.shape == (B, S, N, K, cfg.align_out)
    assert zp.shape == (B, S, N, K, cfg.pos_out)


def mil_shape_smoke():
    torch.manual_seed(0)
    mil = GatedAttentionMIL(dim=128, hidden=64)
    x = torch.randn(4, 10, 128)
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[0, 5:] = False   # partial mask
    mask[3, :] = False    # fully empty row
    z, attn, raw = mil(x, mask)
    assert z.shape == (4, 128)
    assert attn.shape == (4, 10) and raw.shape == (4, 10)
    # attention sums to 1 across unmasked entries where any_valid; zero on masked
    valid_sums = attn[:3].sum(dim=-1)
    assert torch.allclose(valid_sums, torch.ones(3), atol=1e-5), valid_sums
    assert (attn[0, 5:] == 0).all()
    # Fully-empty row: pooled vec should be zero, no NaN
    assert not torch.isnan(z).any()
    assert (z[3] == 0).all()


def set_transformer_shape_smoke():
    torch.manual_seed(0)
    dim = 128
    sab = SetTransformerBlock(dim, heads=4)
    pma = PMA(dim, heads=4, n_seeds=1)
    x = torch.randn(2, 12, dim)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1, 8:] = False
    y = sab(x, mask)
    assert y.shape == x.shape
    p = pma(y, mask)
    assert p.shape == (2, 1, dim)


def v1_forward_synthetic():
    torch.manual_seed(0)
    cfg = V1Config()
    model = V1Model(cfg)
    B, S, N, K = 2, 4, 3, cfg.num_candidates
    patches = torch.randn(B, S, N, K, cfg.patch_width, cfg.patch_channels)
    feats = torch.randn(B, S, N, K, cfg.num_features)
    cmask = torch.ones(B, S, N, K, dtype=torch.bool)
    ncmask = torch.ones(B, S, N, dtype=torch.bool)
    ncmask[:, :, 2] = False
    out = model(patches, feats, cmask, ncmask)
    assert out["logit"].shape == (B,)
    assert out["site_repr"].shape == (B, S, cfg.site_dim)
    assert out["cand_attn"].shape == (B, S, N, K)
    assert out["cand_raw"].shape == (B, S, N, K)
    assert out["nc_attn"].shape == (B, S, N)
    # NC attn on the padded slot should be zero.
    assert torch.allclose(out["nc_attn"][:, :, 2], torch.zeros(B, S), atol=1e-6)


def v1_backward_synthetic():
    torch.manual_seed(0)
    cfg = V1Config()
    model = V1Model(cfg)
    B, S = 2, 4
    N, K = cfg.num_nc, cfg.num_candidates
    patches = torch.randn(B, S, N, K, cfg.patch_width, cfg.patch_channels, requires_grad=False)
    feats = torch.randn(B, S, N, K, cfg.num_features)
    cmask = torch.ones(B, S, N, K, dtype=torch.bool)
    ncmask = torch.ones(B, S, N, dtype=torch.bool)
    is_pos = torch.tensor([True, False])
    true_slot = torch.zeros(B, S, dtype=torch.int32)  # e.g. index 0
    # For negatives, set to -1 to mask aux loss.
    true_slot[1] = -1
    active = torch.zeros(B, S, dtype=torch.long)      # NC slot 0 as active

    out = model(patches, feats, cmask, ncmask)
    losses = v1_loss(out, is_pos, true_slot, active_nc_index=active, aux_lambda=0.1)
    losses["total"].backward()

    # All parameters should have non-None non-zero grads (or at least some do).
    n_params = 0
    n_nonzero = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            raise AssertionError(f"param {name} has no grad")
        n_params += 1
        if p.grad.abs().sum() > 0:
            n_nonzero += 1
    frac = n_nonzero / n_params
    print(f"  {n_nonzero}/{n_params} params ({frac*100:.1f}%) have non-zero grad")
    assert frac > 0.9, "expected almost all params to receive gradient"


def v1_real_val_batch():
    """Run V1 on a real 4-tnp × 8-site val batch."""
    try:
        from preprocess.site import StructureCache
        from preprocess.tnp_dataset import TnpGroupedDataset, collate_tnp_batch
    except Exception as e:
        print(f"  [skip] {e}")
        return

    BASE = "/global/scratch/users/kh36969/DL_novel_guide_editor"
    if not os.path.exists(f"{BASE}/structure/val_u16.index.json"):
        print("  [skip] no val structure cache")
        return

    cache = StructureCache(f"{BASE}/structure/val_u16.index.json")
    ds = TnpGroupedDataset(f"{BASE}/splits/val.jsonl", cache, site_subsample_size=8)

    pos_idx = [i for i, t in enumerate(ds.tnp_ids) if ds.is_positive(t)][:2]
    neg_idx = [i for i, t in enumerate(ds.tnp_ids) if not ds.is_positive(t)][:2]
    items = [ds[i] for i in pos_idx + neg_idx]
    batch = collate_tnp_batch(items, to_torch=True)

    torch.manual_seed(0)
    cfg = V1Config()
    model = V1Model(cfg)
    out = model(
        batch["candidate_patches"],
        batch["candidate_features"],
        batch["candidate_mask"],
        batch["nc_region_mask"],
    )
    print(f"  logit shape: {tuple(out['logit'].shape)}")
    print(f"  cand_attn shape: {tuple(out['cand_attn'].shape)}")
    print(f"  site_repr shape: {tuple(out['site_repr'].shape)}")
    print(f"  logit values: {out['logit'].detach().cpu().numpy()}")

    # Loss + backward (active_nc_index piped through from labels)
    losses = v1_loss(
        out, batch["is_positive"], batch["true_slot_idx"],
        active_nc_index=batch["active_nc_index"], aux_lambda=0.1,
    )
    losses["total"].backward()
    print(f"  bce={float(losses['bce']):.4f}  aux={float(losses['aux']):.4f}  "
          f"total={float(losses['total']):.4f}")
    n_params = sum(1 for _ in model.parameters())
    n_nonzero = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    print(f"  {n_nonzero}/{n_params} params received gradient")


def param_count():
    cfg = V1Config()
    model = V1Model(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  V1 params: {total:,} total, {trainable:,} trainable")


def main():
    print("param count ...")
    param_count()

    print("CandidateEncoder shape ...", end=" ")
    encoder_shape_smoke()
    print("ok")

    print("GatedAttentionMIL shape + masking ...", end=" ")
    mil_shape_smoke()
    print("ok")

    print("SetTransformer + PMA shape ...", end=" ")
    set_transformer_shape_smoke()
    print("ok")

    print("V1 synthetic forward ...", end=" ")
    v1_forward_synthetic()
    print("ok")

    print("V1 synthetic backward (grad flow) ...")
    v1_backward_synthetic()

    print("V1 real val batch (2 pos + 2 neg, 8 sites each) ...")
    v1_real_val_batch()


if __name__ == "__main__":
    main()
