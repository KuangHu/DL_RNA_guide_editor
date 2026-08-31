"""V1 training loop.

Usage:
    python -m training.train_v1 \
        --train-jsonl /groups/.../splits/train.jsonl \
        --train-cache /groups/.../structure/train_u16.index.json \
        --val-jsonl   /groups/.../splits/val.jsonl \
        --val-cache   /groups/.../structure/val_u16.index.json \
        --out-dir     ./checkpoints/v1_run1 \
        --epochs 20 --tnp-batch 8 --sites-train 16 --sites-val 50

Defaults are locked to the user's spec:
  AdamW, lr 3e-4, wd 0.01, cosine decay, ~5% warmup, grad clip 1.0,
  BCE (no class weighting), aux candidate loss lambda=0.1,
  best checkpoint by val Tnp AUPRC.

On CUDA the model runs under BF16 autocast for the forward + loss; the
optimizer state and parameters stay in FP32.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

from model.v1 import (
    V1Config, V1Model, v1_loss, v1_multi_branch_loss, v1_and_fusion_loss,
    v1_paired_geom_loss, v1_paired_orient_loss,
)

# 48C1f: property-validity tables per violation_profile.
#   y_pair = 1 iff RNA<->DNA pairing is intact (only shuffle/length break it)
#   y_geom = 1 iff bag geometry is intact (only orient/pos break it; struct we
#           treat as geom-intact until we add a structure branch)
PROFILE_Y_PAIR = {
    "positive":                 1,
    "paired_shuffle_v42":       0,
    "wrong_length_v42":         0,
    "wrong_orientation_v42":    1,
    "wrong_position_v42":       1,
    "wrong_structure_role_v42": 1,
}
PROFILE_Y_GEOM = {
    "positive":                 1,
    "paired_shuffle_v42":       1,
    "wrong_length_v42":         1,
    "wrong_orientation_v42":    0,
    "wrong_position_v42":       0,
    "wrong_structure_role_v42": 1,
}
# 48C2a: orientation-validity labels. Only wrong_orientation breaks it.
PROFILE_Y_ORIENT = {
    "positive":                 1,
    "paired_shuffle_v42":       1,
    "wrong_length_v42":         1,
    "wrong_orientation_v42":    0,
    "wrong_position_v42":       1,
    "wrong_structure_role_v42": 1,
}
from preprocess.site import StructureCache
from preprocess.tnp_dataset import (
    TnpGroupedDataset,
    collate_tnp_batch,
    make_torch_tnp_dataset,
)
from training.metrics import (
    candidate_recall,
    nc_selection_accuracy,
    stratified_auroc,
    tnp_metrics,
)


# ---------------------------------------------------------------- #
# Config
# ---------------------------------------------------------------- #

@dataclass
class TrainConfig:
    train_jsonl: str
    train_cache: str
    val_jsonl: str
    val_cache: str
    out_dir: str = "./checkpoints/v1_run"

    epochs: int = 20
    tnp_batch: int = 8
    sites_train: int = 16     # subsample this many per tnp during training
    sites_val: int = 50       # use all 50 at eval (~1x per epoch)
    num_workers: int = 4
    persistent_workers: bool = True

    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_frac: float = 0.05
    grad_clip: float = 1.0
    aux_lambda: float = 0.1
    pos_weight: float | None = None   # BCE pos_weight for imbalanced pos:neg (e.g. 5.0 for 1:5)

    # Stratified TNP batch sampler (dict is group_name -> num_tnps_per_batch).
    # None → use standard shuffle=True DataLoader with batch_size=tnp_batch.
    stratified_per_group: dict[str, int] | None = None
    stratified_steps_per_epoch: int | None = None

    # 48C1b: multi-branch pair+geom architecture with per-branch auxiliary losses.
    use_multi_branch: bool = False
    lambda_pair_aux: float = 0.5
    lambda_geom_aux: float = 0.5
    # 48C1c: LR scaling for warm-started pair backbone (protect pretrained
    # pair pipeline while geom branch catches up). 1.0 = no scaling.
    pair_backbone_lr_scale: float = 1.0

    # 48C1g: paired counterfactual batching + geom-only training.
    #   `paired_profiles`: ordered list of (profile_name, tnp_id_suffix). The
    #   first entry is treated as the parent (POS). E.g.
    #     [('positive', ''), ('paired_shuffle_v42','__neg_paired_shuffle_v42'),
    #      ('wrong_length_v42','__neg_wrong_length_v42'),
    #      ('wrong_position_v42','__neg_wrong_position_v42')]
    #   Yields batches of (K_parents * P) bags in the fixed order.
    use_paired_batch: bool = False
    paired_profiles: tuple[tuple[str, str], ...] = ()
    k_parents_per_batch: int = 2
    freeze_pair_branch: bool = False
    # Which paired profile index MUST be outranked by POS on the geom head
    # (typically wrong_position). Which indices geom must be INVARIANT to
    # (typically shuffle, length).
    geom_ranking_neg_idx: int = 3
    geom_invariance_idx: tuple[int, ...] = (1, 2)
    geom_margin: float = 0.1
    lambda_geom_inv: float = 1.0
    lambda_geom_prop: float = 0.0  # 48C1h-A: property BCE anchor for geom

    # 48C2a: freeze geom branch in addition to pair; train orient only.
    freeze_geom_branch: bool = False
    # 48C2a: paired orientation loss params (mirror geom equivalents).
    train_orient_only: bool = False
    orient_ranking_neg_idx: int = 4      # index of wrong_orientation in paired_profiles
    orient_invariance_idx: tuple[int, ...] = (1, 2, 3)  # shuf, len, wp
    orient_margin: float = 0.1
    lambda_orient_inv: float = 1.0
    lambda_orient_prop: float = 0.3
    # Profile group names that supervise each aux head (positive always supervises both).
    pair_supervised_profiles: tuple[str, ...] = (
        "positive", "paired_shuffle_v42", "wrong_length_v42",
    )
    geom_supervised_profiles: tuple[str, ...] = (
        "positive", "wrong_orientation_v42", "wrong_position_v42",
    )

    seed: int = 0
    bf16: bool = True                 # BF16 autocast on cuda
    device: str = "cuda"              # falls back to cpu automatically
    log_every: int = 20
    max_val_tnps: int | None = None   # cap val set for quick smoke
    max_train_steps: int | None = None
    v1_cfg: V1Config = field(default_factory=V1Config)

    # V5.1: warm-start from a V4 checkpoint + freeze backbone + guardrails.
    init_from: str | None = None            # path to .pt to seed model.state_dict()
    freeze_backbone: bool = False           # True: only disp_head + disp_alpha train
    nc_top1_gate: float = 0.0               # ★ best requires nc_top1 >= this
    wrong_orient_gate: float = 0.0          # ★ best requires wrong_orient AUROC >= this

    # V6: pairing branch training.
    use_pairing: bool = False               # enable pair loss + swap augmentation
    pair_lambda: float = 0.5                # final λ_pair after ramp
    pair_lambda_warmup_epochs: float = 1.0  # linear warmup 0.25*λ → λ over this many epochs
    pair_margin: float = 1.0                # margin m
    freeze_pair_beta: bool = True           # Stage A: β=0 fixed. Stage B: β trainable.
    freeze_v52: bool = False                # Stage A: freeze all V5.2 modules (only pair_head trains).
    freeze_v6_stage_b: bool = False         # Stage B: freeze only encoder + cand_mil + V5.2 disp branch.
    pair_beta_init: float = 0.0             # Stage B.1: warm-start pair_beta to force pathway usage.
    prop_lambda: float = 0.0                # Stage B.1: bag-level cognate-vs-swap logit propagation loss weight.
    prop_margin: float = 1.0                # Stage B.1: margin for L_prop.


# ---------------------------------------------------------------- #
# LR schedule
# ---------------------------------------------------------------- #

def cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float) -> float:
    """Linear warmup for `warmup` steps, then cosine decay to 0 over
    the remaining steps."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if total <= warmup:
        return 0.0
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------- #

def _to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _tnp_violation_profile(rec_labels: dict) -> str:
    if rec_labels.get("is_positive"):
        return "positive"
    return rec_labels.get("violation_profile") or "unknown"


def _violation_profile_by_tnp(jsonl_path: str) -> dict[str, str]:
    """Map tnp_id -> group name. Positives -> 'positive'; negatives get
    their (shared) violation_profile of the FIRST site (all sites of one
    negative tnp share the same profile by construction)."""
    out: dict[str, str] = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            tnp = r["transposase_id"]
            if tnp in out:
                continue
            out[tnp] = _tnp_violation_profile(r["labels"])
    return out


def _tnp_strength_by_tnp(jsonl_path: str) -> dict[str, str]:
    """Map tnp_id -> tnp_strength (strong/moderate/weak) for POSITIVE tnps,
    or 'negative' otherwise. Set by weaken_positives.py in V3 noisy positives.
    Missing → 'unknown'."""
    out: dict[str, str] = {}
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            tnp = r["transposase_id"]
            if tnp in out:
                continue
            if r["labels"].get("is_positive"):
                out[tnp] = r.get("generator_metadata", {}).get("tnp_strength", "unknown")
            else:
                out[tnp] = "negative"
    return out


# Profiles considered "easy" — excluded from AUROC_hard_only.
EASY_PROFILES = ("level1_marginal_matched",)


# ---------------------------------------------------------------- #
# Evaluation
# ---------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    model, val_dl, device, cfg: TrainConfig,
    tnp_group_map: dict[str, str],
    tnp_strength_map: dict[str, str] | None = None,
) -> dict:
    model.eval()
    all_scores: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_tnp_ids: list[list[str]] = []
    # For auxiliary metrics we track per-site info at each step (positives only).
    cand_scores_at_active_nc = []      # (N_pos_sites, K)
    true_slot_all: list[int] = []
    true_active_nc_all: list[int] = []
    nc_attn_all = []                    # (N_pos_sites, N_nc)

    # V6: track cognate-vs-swap bag logits (Δ_final) when swap augmentation is on.
    delta_final_all: list[float] = []
    delta_final_pos: list[bool] = []
    logit_cog_all: list[float] = []
    logit_swp_all: list[float] = []

    for batch in val_dl:
        batch = _to_device(batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                             enabled=(cfg.bf16 and device.type == "cuda")):
            out = model(
                batch["candidate_patches"],
                batch["candidate_features"],
                batch["candidate_mask"],
                batch["nc_region_mask"],
            )
            # V6 Stage B.1: also forward the swap version to compute Δ_final.
            out_swap = None
            if "candidate_patches_swap" in batch and cfg.use_pairing:
                out_swap = model(
                    batch["candidate_patches_swap"],
                    batch["candidate_features_swap"],
                    batch["candidate_mask_swap"],
                    batch["nc_region_mask_swap"],
                )
        all_scores.append(torch.sigmoid(out["logit"]).float().cpu().numpy())
        all_labels.append(batch["is_positive"].cpu().numpy())
        all_tnp_ids.append(list(batch["tnp_id"]))
        if out_swap is not None:
            lc = out["logit"].float().cpu().numpy()
            ls = out_swap["logit"].float().cpu().numpy()
            for bi in range(len(lc)):
                delta_final_all.append(float(lc[bi] - ls[bi]))
                delta_final_pos.append(bool(batch["is_positive"][bi].item()))
                logit_cog_all.append(float(lc[bi]))
                logit_swp_all.append(float(ls[bi]))

        # Gather aux info on positive sites within this batch.
        cand_raw = out["cand_raw"].float().cpu().numpy()        # (B, S, N, K)
        nc_attn = out["nc_attn"].float().cpu().numpy()          # (B, S, N)
        active = batch["active_nc_index"].cpu().numpy()         # (B, S) int
        true_slot = batch["true_slot_idx"].cpu().numpy()        # (B, S) int
        B, S, N, K = cand_raw.shape
        for bi in range(B):
            for si in range(S):
                anc = int(active[bi, si])
                ts = int(true_slot[bi, si])
                if anc < 0 or ts < 0:
                    continue
                cand_scores_at_active_nc.append(cand_raw[bi, si, anc])
                true_slot_all.append(ts)
                true_active_nc_all.append(anc)
                nc_attn_all.append(nc_attn[bi, si])

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    tnp_ids_flat = [t for chunk in all_tnp_ids for t in chunk]
    groups = np.asarray([tnp_group_map.get(t, "unknown") for t in tnp_ids_flat])

    m_tnp = tnp_metrics(scores, labels)
    m_strat = stratified_auroc(scores, labels, groups)
    if cand_scores_at_active_nc:
        cand_scores_arr = np.stack(cand_scores_at_active_nc, axis=0)
        m_cand = candidate_recall(cand_scores_arr, true_slot_all, ks=(1, 5, 10))
        nc_attn_arr = np.stack(nc_attn_all, axis=0)
        m_nc = nc_selection_accuracy(nc_attn_arr, true_active_nc_all)
    else:
        m_cand = {f"recall@{k}": float("nan") for k in (1, 5, 10)} | {"n": 0}
        m_nc = {"nc_top1": float("nan"), "n": 0}

    # AUROC / AUPRC excluding EASY_PROFILES (V3 hard-only view).
    labels_bool = labels.astype(bool)
    hard_mask = labels_bool | np.array([g not in EASY_PROFILES for g in groups])
    hard_metrics = {"auroc_hard_only": float("nan"), "auprc_hard_only": float("nan"), "n_hard_neg": 0}
    if hard_mask.any():
        from training.metrics import _auroc, _auprc
        s_hard = scores[hard_mask]
        y_hard = labels_bool[hard_mask]
        if y_hard.any() and (~y_hard).any():
            hard_metrics = {
                "auroc_hard_only": _auroc(s_hard, y_hard),
                "auprc_hard_only": _auprc(s_hard, y_hard),
                "n_hard_neg": int((~y_hard).sum()),
            }

    # Positive recall by tnp_strength (weak/moderate/strong).
    strength_metrics = {}
    if tnp_strength_map is not None:
        strengths = np.asarray([tnp_strength_map.get(t, "unknown") for t in tnp_ids_flat])
        pos_mask = labels_bool
        # Threshold 0.5 (tnp is called positive if score > 0.5).
        called_pos = scores > 0.5
        for level in ("strong", "moderate", "weak"):
            m = pos_mask & (strengths == level)
            if m.any():
                s_lvl = scores[m]
                strength_metrics[f"recall_{level}"] = float(called_pos[m].mean())
                strength_metrics[f"n_{level}"] = int(m.sum())
                # Score-distribution quantiles (V5.2 diagnostic — catches cases
                # where the model is not literally missing weak positives at
                # threshold 0.5 but has systematically down-scored them).
                q = np.quantile(s_lvl, [0.10, 0.50, 0.90])
                strength_metrics[f"score_q10_{level}"] = float(q[0])
                strength_metrics[f"score_med_{level}"] = float(q[1])
                strength_metrics[f"score_q90_{level}"] = float(q[2])
            else:
                strength_metrics[f"recall_{level}"] = float("nan")
                strength_metrics[f"n_{level}"] = 0
                for tag in ("score_q10_", "score_med_", "score_q90_"):
                    strength_metrics[f"{tag}{level}"] = float("nan")

    # V6: Δ_final metrics (only if swap fwd was done).
    pair_final_metrics = {}
    if delta_final_all:
        arr = np.asarray(delta_final_all)
        pos_arr = np.asarray(delta_final_pos, dtype=bool)
        arr_pos = arr[pos_arr]
        if len(arr_pos) > 0:
            q = np.quantile(arr_pos, [0.10, 0.25, 0.50, 0.75, 0.90])
            pair_final_metrics["delta_final_pos_median"] = float(q[2])
            pair_final_metrics["delta_final_pos_q10"] = float(q[0])
            pair_final_metrics["delta_final_pos_q25"] = float(q[1])
            pair_final_metrics["delta_final_pos_q75"] = float(q[3])
            pair_final_metrics["delta_final_pos_q90"] = float(q[4])
            pair_final_metrics["delta_final_pos_mean"] = float(arr_pos.mean())
            pair_final_metrics["delta_final_pos_frac_gt_0"] = float((arr_pos > 0).mean())
            pair_final_metrics["delta_final_pos_frac_gt_0p25"] = float((arr_pos > 0.25).mean())
            pair_final_metrics["delta_final_pos_frac_gt_1"] = float((arr_pos > 1.0).mean())
            # Bag-level cognate-vs-swap AUROC on positive bags
            lc_pos = np.asarray(logit_cog_all)[pos_arr]
            ls_pos = np.asarray(logit_swp_all)[pos_arr]
            pooled = np.concatenate([lc_pos, ls_pos])
            plabels = np.concatenate([np.ones(len(lc_pos), dtype=bool),
                                        np.zeros(len(ls_pos), dtype=bool)])
            from training.metrics import _auroc as _auroc_local
            pair_final_metrics["pair_final_auroc_pos"] = float(_auroc_local(pooled, plabels))

    return {
        "n_tnp_pos": m_tnp["n_pos"],
        "n_tnp_neg": m_tnp["n_neg"],
        "auroc": m_tnp["auroc"],
        "auprc": m_tnp["auprc"],
        **hard_metrics,
        **m_strat,
        **m_cand,
        **m_nc,
        **strength_metrics,
        **pair_final_metrics,
    }


# ---------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------- #

def train(cfg: TrainConfig) -> dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.json", "w") as f:
        json.dump({**{k: v for k, v in asdict(cfg).items() if k != "v1_cfg"},
                    "v1_cfg": asdict(cfg.v1_cfg)}, f, indent=2)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"[train] device: {device}   bf16: {cfg.bf16 and device.type == 'cuda'}")

    # Data
    print(f"[data] loading caches ...", flush=True)
    train_cache = StructureCache(cfg.train_cache)
    val_cache = StructureCache(cfg.val_cache)
    train_ds = TnpGroupedDataset(
        cfg.train_jsonl, train_cache,
        site_subsample_size=cfg.sites_train, rng_seed=cfg.seed,
        generate_swap=cfg.use_pairing,   # V6: augment with per-site flank swap
    )
    val_ds = TnpGroupedDataset(
        cfg.val_jsonl, val_cache,
        site_subsample_size=cfg.sites_val, rng_seed=cfg.seed,
        generate_swap=cfg.use_pairing,
    )
    if cfg.max_val_tnps is not None:
        # Restrict tnp_ids for quick smoke.
        val_ds.tnp_ids = val_ds.tnp_ids[: cfg.max_val_tnps]
    print(f"[data] train tnps={len(train_ds)}   val tnps={len(val_ds)}", flush=True)

    train_group = _violation_profile_by_tnp(cfg.train_jsonl)
    val_group = _violation_profile_by_tnp(cfg.val_jsonl)
    val_strength = _tnp_strength_by_tnp(cfg.val_jsonl)

    if cfg.use_paired_batch:
        from preprocess.tnp_dataset import PairedCounterfactualBatchSampler
        profile_suffixes = dict(cfg.paired_profiles)
        batch_sampler = PairedCounterfactualBatchSampler(
            train_ds, profile_suffixes,
            k_parents_per_batch=cfg.k_parents_per_batch,
            steps_per_epoch=cfg.stratified_steps_per_epoch,
            seed=cfg.seed,
        )
        eff_batch = cfg.k_parents_per_batch * len(profile_suffixes)
        print(f"[sampler] PAIRED profiles={list(profile_suffixes)}  "
              f"K={cfg.k_parents_per_batch}  batch={eff_batch}  "
              f"steps/epoch={len(batch_sampler)}  "
              f"n_parents_matched={len(batch_sampler.parents)}", flush=True)
        train_dl = DataLoader(
            make_torch_tnp_dataset(train_ds),
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            collate_fn=lambda items: collate_tnp_batch(items, to_torch=True),
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            pin_memory=(device.type == "cuda"),
        )
    elif cfg.stratified_per_group:
        from preprocess.tnp_dataset import StratifiedTnpBatchSampler
        batch_sampler = StratifiedTnpBatchSampler(
            train_ds, tnp_to_group=train_group,
            per_group=cfg.stratified_per_group,
            steps_per_epoch=cfg.stratified_steps_per_epoch,
            seed=cfg.seed,
        )
        eff_batch = sum(cfg.stratified_per_group.values())
        print(f"[sampler] STRATIFIED per_group={cfg.stratified_per_group}  "
              f"batch={eff_batch}  steps/epoch={len(batch_sampler)}", flush=True)
        train_dl = DataLoader(
            make_torch_tnp_dataset(train_ds),
            batch_sampler=batch_sampler,
            num_workers=cfg.num_workers,
            collate_fn=lambda items: collate_tnp_batch(items, to_torch=True),
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            pin_memory=(device.type == "cuda"),
        )
    else:
        train_dl = DataLoader(
            make_torch_tnp_dataset(train_ds),
            batch_size=cfg.tnp_batch, shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=lambda items: collate_tnp_batch(items, to_torch=True),
            persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
            pin_memory=(device.type == "cuda"),
        )
    val_dl = DataLoader(
        make_torch_tnp_dataset(val_ds),
        batch_size=cfg.tnp_batch, shuffle=False,
        num_workers=cfg.num_workers,
        collate_fn=lambda items: collate_tnp_batch(items, to_torch=True),
        persistent_workers=cfg.persistent_workers and cfg.num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )

    # Model + optim
    model = V1Model(cfg.v1_cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] V1 params: {n_params:,}", flush=True)

    # V5.1: warm-start from a V4 checkpoint if requested.
    if cfg.init_from:
        print(f"[init] loading state_dict from {cfg.init_from}", flush=True)
        ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[init] missing keys ({len(missing)}): {missing}", flush=True)
        print(f"[init] unexpected keys ({len(unexpected)}): {unexpected}", flush=True)

    # V5.1/V5.2: freeze everything except the dispersion branch.
    #   scalar (V5.1):  disp_head, disp_alpha
    #   hidden_residual (V5.2): disp_encoder, fusion_mlp, disp_beta
    if cfg.freeze_backbone:
        allowed = ("disp_head", "disp_alpha", "disp_encoder",
                    "fusion_mlp", "disp_beta")
        n_frozen = n_train = 0
        train_names = []
        for name, p in model.named_parameters():
            if any(k in name for k in allowed):
                p.requires_grad = True
                n_train += p.numel()
                train_names.append(name)
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        print(f"[freeze] frozen={n_frozen:,} trainable={n_train:,}", flush=True)
        print(f"[freeze] trainable params: {train_names}", flush=True)

    # V6 Stage A: freeze all V5.2 modules, train only pair_head + pair_fuse.
    # Optionally keep pair_beta frozen at 0 (auxiliary head only).
    if cfg.freeze_v52:
        pair_allowed = ("pair_head", "pair_fuse")
        if not cfg.freeze_pair_beta:
            pair_allowed = pair_allowed + ("pair_beta",)
        n_frozen = n_train = 0
        train_names = []
        for name, p in model.named_parameters():
            if any(k in name for k in pair_allowed):
                p.requires_grad = True
                n_train += p.numel()
                train_names.append(name)
            else:
                p.requires_grad = False
                n_frozen += p.numel()
        # Ensure pair_beta is exactly 0 and stays frozen in Stage A
        if cfg.freeze_pair_beta and getattr(model, "pair_beta", None) is not None:
            with torch.no_grad():
                model.pair_beta.zero_()
            model.pair_beta.requires_grad = False
        print(f"[freeze-v52] frozen={n_frozen:,} trainable={n_train:,} "
              f"(pair_beta frozen at 0: {cfg.freeze_pair_beta})", flush=True)
        print(f"[freeze-v52] trainable: {train_names}", flush=True)

    # V6 Stage B.1: warm-start pair_beta AFTER state_dict load so the pathway
    # depends on q_nc from step 1 (avoids the "β stays at 0" bypass).
    if cfg.pair_beta_init != 0.0 and getattr(model, "pair_beta", None) is not None:
        with torch.no_grad():
            model.pair_beta.fill_(cfg.pair_beta_init)
        print(f"[init] pair_beta warm-started to {cfg.pair_beta_init:+.4f}", flush=True)

    # V6 Stage B: freeze ONLY candidate encoder + cand_mil + V5.2 dispersion
    # branch. Train nc_mil + set_blocks + pma + classifier + pair_head +
    # pair_fuse + pair_beta. pair_beta is unfrozen so q_nc can start
    # modulating the main pathway.
    if cfg.freeze_v6_stage_b:
        # Modules to FREEZE (proven-working; don't perturb)
        frozen_prefixes = ("encoder.", "cand_mil.",
                             "disp_encoder.", "fusion_mlp.", "disp_beta")
        n_frozen = n_train = 0
        train_names = []
        for name, p in model.named_parameters():
            is_frozen = any(name.startswith(pref) for pref in frozen_prefixes)
            if is_frozen:
                p.requires_grad = False
                n_frozen += p.numel()
            else:
                p.requires_grad = True
                n_train += p.numel()
                train_names.append(name)
        print(f"[freeze-v6-stageB] frozen={n_frozen:,} trainable={n_train:,}", flush=True)
        print(f"[freeze-v6-stageB] trainable (sample): {train_names[:10]}"
              f"{' ...' if len(train_names) > 10 else ''}", flush=True)

    # 48C1g / 48C2a: freeze branches selectively.
    frozen_prefixes = []
    if cfg.freeze_pair_branch:
        frozen_prefixes.extend([
            "encoder.", "cand_mil.", "nc_mil.", "set_blocks.", "pma.",
            "classifier.", "h_pair_aux.",
        ])
    if cfg.freeze_geom_branch:
        frozen_prefixes.extend([
            "geom_input_mlp.", "geom_set_blocks.", "geom_pma.",
            "h_geom_aux.", "h_fusion.", "alpha_pair", "alpha_geom",
            "bn_pair_aux.", "bn_geom_aux.",
        ])
    if frozen_prefixes:
        n_frozen = n_train = 0
        train_names = []
        for name, p in model.named_parameters():
            is_frozen = any(name.startswith(pref) or f".{pref}" in name for pref in frozen_prefixes)
            if is_frozen:
                p.requires_grad = False
                n_frozen += p.numel()
            else:
                p.requires_grad = True
                n_train += p.numel()
                train_names.append(name)
        print(f"[freeze-branch] frozen={n_frozen:,} trainable={n_train:,}", flush=True)
        print(f"[freeze-branch] trainable modules (sample): {train_names[:8]}"
              f"{' ...' if len(train_names) > 8 else ''}", flush=True)

    # V6 Stage B: two LR groups — pair branch at higher LR (typically cfg.lr),
    # backbone at 0.1× (protects V5.2 finetuning). Others: single group.
    if cfg.freeze_v6_stage_b:
        pair_keys = ("pair_head", "pair_fuse", "pair_beta")
        pair_params, backbone_params = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            (pair_params if any(k in n for k in pair_keys) else backbone_params).append(p)
        optim = torch.optim.AdamW([
            {"params": pair_params,     "lr": cfg.lr,       "weight_decay": cfg.weight_decay},
            {"params": backbone_params, "lr": cfg.lr * 0.1, "weight_decay": cfg.weight_decay},
        ])
        # Store initial LRs so we can scale via warmup+cosine
        for g in optim.param_groups:
            g["initial_lr"] = g["lr"]
        print(f"[optim] V6 Stage B two-group LR: pair@{cfg.lr:.1e}, backbone@{cfg.lr*0.1:.1e}", flush=True)
    elif cfg.use_multi_branch and cfg.pair_backbone_lr_scale != 1.0:
        # 48C1c: protect warm-started pair pipeline with a lower LR while the
        # geometry branch (and heads) catch up at full LR.
        pair_backbone_keys = ("encoder.", "cand_mil.", "nc_mil.", "set_blocks.", "pma.", "classifier.")
        pair_bb, new_group = [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(n.startswith(k) or f".{k}" in n for k in pair_backbone_keys):
                pair_bb.append(p)
            else:
                new_group.append(p)
        lr_bb = cfg.lr * cfg.pair_backbone_lr_scale
        optim = torch.optim.AdamW([
            {"params": pair_bb,   "lr": lr_bb,    "weight_decay": cfg.weight_decay},
            {"params": new_group, "lr": cfg.lr,   "weight_decay": cfg.weight_decay},
        ])
        for g in optim.param_groups:
            g["initial_lr"] = g["lr"]
        print(f"[optim] 48C1c two-group LR: pair-backbone@{lr_bb:.1e} ({len(pair_bb)} params), "
              f"new-modules@{cfg.lr:.1e} ({len(new_group)} params)", flush=True)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        optim = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
        for g in optim.param_groups:
            g["initial_lr"] = g["lr"]

    steps_per_epoch = len(train_dl)  # respects stratified sampler if in use
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.max_train_steps is not None:
        total_steps = min(total_steps, cfg.max_train_steps)
    warmup_steps = max(1, int(cfg.warmup_frac * total_steps))
    print(f"[schedule] total_steps={total_steps} warmup={warmup_steps}", flush=True)

    best_auprc = -1.0
    best_epoch = -1
    history = []
    global_step = 0
    t_start = time.time()

    # V6: import pair_loss on demand (avoids surprising V5.x users who don't touch pairing)
    if cfg.use_pairing:
        from model.v1 import pair_loss as _pair_loss
    steps_per_epoch_true = len(train_dl)  # respects stratified sampler if in use

    for epoch in range(cfg.epochs):
        model.train()
        t_ep = time.time()
        running = {"total": 0.0, "bce": 0.0, "aux": 0.0, "pair": 0.0,
                    "delta_pair": 0.0, "prop": 0.0, "delta_final": 0.0,
                    "n": 0, "n_pair_total": 0}
        for it, batch in enumerate(train_dl):
            batch = _to_device(batch, device)
            lr_now = cosine_with_warmup(global_step, total_steps, warmup_steps, cfg.lr)
            # Scale each group's LR relative to its initial LR (V6 Stage B uses 2 groups).
            for g in optim.param_groups:
                scale = g["initial_lr"] / max(1e-12, cfg.lr)
                g["lr"] = lr_now * scale

            # V6: λ_pair ramp — linear from 0.25*λ to λ over first
            # pair_lambda_warmup_epochs.
            if cfg.use_pairing and cfg.pair_lambda_warmup_epochs > 0:
                progress = (global_step / max(1, steps_per_epoch_true)) / cfg.pair_lambda_warmup_epochs
                progress = min(1.0, max(0.0, progress))
                lam_pair_now = cfg.pair_lambda * (0.25 + 0.75 * progress)
            else:
                lam_pair_now = cfg.pair_lambda if cfg.use_pairing else 0.0

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=(cfg.bf16 and device.type == "cuda")):
                out = model(
                    batch["candidate_patches"],
                    batch["candidate_features"],
                    batch["candidate_mask"],
                    batch["nc_region_mask"],
                )
                pw_tensor = None
                if cfg.pos_weight is not None:
                    pw_tensor = torch.tensor(cfg.pos_weight, device=device, dtype=torch.float32)
                if cfg.use_paired_batch and cfg.use_multi_branch and cfg.train_orient_only:
                    # 48C2a: paired orientation-only loss.
                    y_orient_tensor = None
                    if cfg.lambda_orient_prop > 0.0:
                        tnp_ids = batch["tnp_id"]
                        profs = [train_group.get(t, "unknown") for t in tnp_ids]
                        y_orient_tensor = torch.tensor(
                            [PROFILE_Y_ORIENT.get(p, 1) for p in profs],
                            dtype=torch.float32, device=device,
                        )
                    losses = v1_paired_orient_loss(
                        out["s_orient_aux"],
                        n_profiles=len(cfg.paired_profiles),
                        profile_ranking_pos=cfg.orient_ranking_neg_idx,
                        profile_invariance_idx=tuple(cfg.orient_invariance_idx),
                        margin=cfg.orient_margin,
                        lambda_inv=cfg.lambda_orient_inv,
                        y_orient=y_orient_tensor,
                        lambda_prop=cfg.lambda_orient_prop,
                    )
                elif cfg.use_paired_batch and cfg.use_multi_branch:
                    # 48C1g/h: paired counterfactual geometry loss.
                    y_geom_tensor = None
                    if cfg.lambda_geom_prop > 0.0:
                        tnp_ids = batch["tnp_id"]
                        profs = [train_group.get(t, "unknown") for t in tnp_ids]
                        y_geom_tensor = torch.tensor(
                            [PROFILE_Y_GEOM.get(p, 1) for p in profs],
                            dtype=torch.float32, device=device,
                        )
                    losses = v1_paired_geom_loss(
                        out["s_geom_aux"],
                        n_profiles=len(cfg.paired_profiles),
                        profile_ranking_pos=cfg.geom_ranking_neg_idx,
                        profile_invariance_idx=tuple(cfg.geom_invariance_idx),
                        margin=cfg.geom_margin,
                        lambda_inv=cfg.lambda_geom_inv,
                        y_geom=y_geom_tensor,
                        lambda_prop=cfg.lambda_geom_prop,
                    )
                elif cfg.use_multi_branch and cfg.v1_cfg.use_and_fusion:
                    # 48C1f: property-supervised AND fusion. Each aux head is
                    # trained on ALL samples with property-specific validity
                    # labels (no profile masking).
                    tnp_ids = batch["tnp_id"]
                    profs = [train_group.get(t, "unknown") for t in tnp_ids]
                    y_pair = torch.tensor(
                        [PROFILE_Y_PAIR.get(p, 1) for p in profs],
                        dtype=torch.float32, device=device,
                    )
                    y_geom = torch.tensor(
                        [PROFILE_Y_GEOM.get(p, 1) for p in profs],
                        dtype=torch.float32, device=device,
                    )
                    losses = v1_and_fusion_loss(
                        out, y_pair, y_geom,
                        lambda_pair=cfg.lambda_pair_aux,
                        lambda_geom=cfg.lambda_geom_aux,
                    )
                elif cfg.use_multi_branch:
                    # 48C1b: profile-masked auxiliary losses.
                    tnp_ids = batch["tnp_id"]
                    profs = [train_group.get(t, "unknown") for t in tnp_ids]
                    pair_sup = torch.tensor(
                        [p in cfg.pair_supervised_profiles for p in profs],
                        dtype=torch.bool, device=device,
                    )
                    geom_sup = torch.tensor(
                        [p in cfg.geom_supervised_profiles for p in profs],
                        dtype=torch.bool, device=device,
                    )
                    losses = v1_multi_branch_loss(
                        out, batch["is_positive"], pair_sup, geom_sup,
                        lambda_pair=cfg.lambda_pair_aux,
                        lambda_geom=cfg.lambda_geom_aux,
                        pos_weight=pw_tensor,
                    )
                else:
                    losses = v1_loss(
                        out, batch["is_positive"], batch["true_slot_idx"],
                        active_nc_index=batch["active_nc_index"],
                        aux_lambda=cfg.aux_lambda,
                        pos_weight=pw_tensor,
                    )
                # V6: compute q_nc on the swap version, then pair loss.
                pair_stats = None
                if cfg.use_pairing and "candidate_patches_swap" in batch:
                    out_swap = model(
                        batch["candidate_patches_swap"],
                        batch["candidate_features_swap"],
                        batch["candidate_mask_swap"],
                        batch["nc_region_mask_swap"],
                    )
                    pair_stats = _pair_loss(
                        q_nc_cognate=out["q_nc"],
                        q_nc_swap=out_swap["q_nc"],
                        active_nc_index=batch["active_nc_index"],
                        pair_mask=batch["pair_mask"],
                        margin=cfg.pair_margin,
                    )
                    losses["total"] = losses["total"] + lam_pair_now * pair_stats["pair"]
                    losses["pair"] = pair_stats["pair"]
                    losses["delta_pair"] = pair_stats["delta_pair"]

                    # V6 Stage B.1: bag-level cognate-vs-swap PROPAGATION loss.
                    # Enforces logit(cognate) > logit(swap) + m on positive bags
                    # (guided sites had their flanks swapped; non-guided sites are unchanged).
                    # Negative bags: swap == cognate by dataset construction, no contribution.
                    if cfg.prop_lambda > 0.0:
                        pos_mask = batch["is_positive"].float()
                        logit_cog = out["logit"]
                        logit_swp = out_swap["logit"]
                        prop_gap = (cfg.prop_margin - logit_cog + logit_swp).clamp(min=0.0)
                        denom = pos_mask.sum().clamp(min=1)
                        l_prop = (prop_gap * pos_mask).sum() / denom
                        delta_final = ((logit_cog - logit_swp) * pos_mask).sum() / denom
                        losses["total"] = losses["total"] + cfg.prop_lambda * l_prop
                        losses["prop"] = l_prop
                        losses["delta_final"] = delta_final

            optim.zero_grad(set_to_none=True)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

            b = batch["is_positive"].size(0)
            running["total"] += float(losses["total"]) * b
            running["bce"] += float(losses["bce"]) * b
            running["aux"] += float(losses["aux"]) * b
            if "pair_aux" in losses:
                running["pair_aux"] = running.get("pair_aux", 0.0) + float(losses["pair_aux"]) * b
            if "geom_aux" in losses:
                running["geom_aux"] = running.get("geom_aux", 0.0) + float(losses["geom_aux"]) * b
            if pair_stats is not None:
                running["pair"] += float(pair_stats["pair"]) * b
                running["delta_pair"] += float(pair_stats["delta_pair"]) * b
                running["n_pair_total"] += int(pair_stats["n_pair"])
            if cfg.use_pairing and cfg.prop_lambda > 0 and "prop" in losses:
                running["prop"] += float(losses["prop"]) * b
                running["delta_final"] += float(losses["delta_final"]) * b
            running["n"] += b
            global_step += 1

            if global_step % cfg.log_every == 0:
                elapsed = time.time() - t_start
                pair_str = ""
                if cfg.use_pairing:
                    pair_str = (f" pair={running['pair']/max(1,running['n']):.4f}"
                                f" Δpair={running['delta_pair']/max(1,running['n']):+.4f}"
                                f" n_pair={running['n_pair_total']}"
                                f" λ={lam_pair_now:.3f}")
                    if cfg.prop_lambda > 0:
                        beta_val = float(model.pair_beta.item()) if getattr(model, 'pair_beta', None) is not None else float('nan')
                        pair_str += (f" prop={running['prop']/max(1,running['n']):.4f}"
                                      f" Δfinal={running['delta_final']/max(1,running['n']):+.4f}"
                                      f" β={beta_val:+.3f}")
                print(
                    f"[train] step {global_step}/{total_steps} "
                    f"lr={lr_now:.2e} "
                    f"total={running['total']/max(1,running['n']):.4f} "
                    f"bce={running['bce']/max(1,running['n']):.4f} "
                    f"aux={running['aux']/max(1,running['n']):.4f}{pair_str} "
                    f"({elapsed:.1f}s elapsed)",
                    flush=True,
                )
            if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
                break

        # Eval on val at end of epoch.
        t_val = time.time()
        val_stats = evaluate(model, val_dl, device, cfg, val_group,
                               tnp_strength_map=val_strength)
        val_stats["epoch_time_s"] = time.time() - t_ep
        val_stats["val_time_s"] = time.time() - t_val
        val_stats["epoch"] = epoch
        val_stats["global_step"] = global_step

        # Save best — guarded by primary-path health metrics (V5.1).
        auprc = val_stats["auprc"]
        val_stats["is_best"] = False
        nc_top1 = val_stats.get("nc_top1", float("nan"))
        wrong_orient = val_stats.get(
            "auroc[wrong_orientation_consistency]", float("nan"))
        # If a health metric is NaN (e.g. no wrong_orient in this profile mix),
        # ignore that gate. Only enforce gates that produced a real number.
        gates_ok = True
        if cfg.nc_top1_gate > 0.0 and not math.isnan(nc_top1):
            gates_ok = gates_ok and (nc_top1 >= cfg.nc_top1_gate)
        if cfg.wrong_orient_gate > 0.0 and not math.isnan(wrong_orient):
            gates_ok = gates_ok and (wrong_orient >= cfg.wrong_orient_gate)
        val_stats["gates_ok"] = gates_ok
        if gates_ok and not math.isnan(auprc) and auprc > best_auprc:
            best_auprc = auprc
            best_epoch = epoch
            val_stats["is_best"] = True
            torch.save(
                {
                    "model": model.state_dict(),
                    "cfg": asdict(cfg.v1_cfg),
                    "train_cfg": {k: v for k, v in asdict(cfg).items() if k != "v1_cfg"},
                    "epoch": epoch, "global_step": global_step, "auprc": auprc,
                },
                out_dir / "best.pt",
            )
        # V6: also save a per-epoch checkpoint so constrained selection can be
        # done post-hoc (val Δ_final + guardrails).
        torch.save(
            {
                "model": model.state_dict(),
                "cfg": asdict(cfg.v1_cfg),
                "train_cfg": {k: v for k, v in asdict(cfg).items() if k != "v1_cfg"},
                "epoch": epoch, "global_step": global_step, "auprc": auprc,
                "val_stats": val_stats,
            },
            out_dir / f"epoch_{epoch:02d}.pt",
        )
        history.append(val_stats)
        with open(out_dir / "history.jsonl", "a") as f:
            f.write(json.dumps(val_stats) + "\n")

        strat_keys = sorted(k for k in val_stats if k.startswith("auroc["))
        strat_str = "  ".join(
            f"{k[6:-1]}={val_stats[k]:.3f}" for k in strat_keys if not math.isnan(val_stats[k])
        )
        recall_str = "  ".join(
            f"{lvl}={val_stats.get(f'recall_{lvl}', float('nan')):.3f}"
            for lvl in ("strong", "moderate", "weak")
            if not math.isnan(val_stats.get(f'recall_{lvl}', float('nan')))
        )
        # Per-strength score quantiles (V5.2 diagnostic).
        qstr_parts = []
        for lvl in ("strong", "moderate", "weak"):
            med = val_stats.get(f"score_med_{lvl}", float("nan"))
            q10 = val_stats.get(f"score_q10_{lvl}", float("nan"))
            q90 = val_stats.get(f"score_q90_{lvl}", float("nan"))
            if not math.isnan(med):
                qstr_parts.append(f"{lvl}[q10={q10:.3f} med={med:.3f} q90={q90:.3f}]")
        quantile_str = "  ".join(qstr_parts)
        hard_str = ""
        if not math.isnan(val_stats.get("auroc_hard_only", float("nan"))):
            hard_str = (f"  HARD_AUROC={val_stats['auroc_hard_only']:.4f} "
                        f"HARD_AUPRC={val_stats['auprc_hard_only']:.4f}")
        # V6 Stage B.1: Δ_final metrics on val (positive bags).
        pair_val_str = ""
        if "delta_final_pos_median" in val_stats:
            beta_val = float(model.pair_beta.item()) if getattr(model, 'pair_beta', None) is not None else float('nan')
            pair_val_str = (f"  Δfinal_pos[med={val_stats['delta_final_pos_median']:+.3f} "
                            f"Q10={val_stats['delta_final_pos_q10']:+.3f} "
                            f"Q90={val_stats['delta_final_pos_q90']:+.3f} "
                            f"P(>0)={val_stats['delta_final_pos_frac_gt_0']:.3f} "
                            f"P(>1)={val_stats['delta_final_pos_frac_gt_1']:.3f}] "
                            f"AUROCpair={val_stats.get('pair_final_auroc_pos', float('nan')):.4f} "
                            f"β={beta_val:+.3f}")
        print(
            f"[val] ep{epoch} AUROC={val_stats['auroc']:.4f} "
            f"AUPRC={val_stats['auprc']:.4f}{hard_str} "
            f"R@1={val_stats['recall@1']:.3f} R@5={val_stats['recall@5']:.3f} "
            f"nc_top1={val_stats['nc_top1']:.3f}  "
            f"pos-recall[{recall_str}]  "
            f"{strat_str}  "
            f"score-q[{quantile_str}]{pair_val_str} "
            f"[{val_stats['val_time_s']:.1f}s]"
            f"{' ★ best' if val_stats['is_best'] else ''}",
            flush=True,
        )

        if cfg.max_train_steps is not None and global_step >= cfg.max_train_steps:
            break

    print(f"[done] best AUPRC={best_auprc:.4f} at epoch {best_epoch}. "
          f"Total time: {time.time()-t_start:.1f}s", flush=True)
    return {"best_auprc": best_auprc, "best_epoch": best_epoch, "history": history}


# ---------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------- #

def _load_stratified_spec(arg: str) -> dict[str, int]:
    """Accept either a JSON string or a path to a JSON file. Robust to shell
    escaping issues around embedded braces/commas."""
    from pathlib import Path
    p = Path(arg)
    if p.exists():
        raw = p.read_text().strip()
    else:
        raw = arg
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"--stratified-per-group must parse to dict, got {type(obj).__name__}: {obj!r}")
    return {str(k): int(v) for k, v in obj.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--train-cache", required=True)
    p.add_argument("--val-jsonl", required=True)
    p.add_argument("--val-cache", required=True)
    p.add_argument("--out-dir", default="./checkpoints/v1_run")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--tnp-batch", type=int, default=8)
    p.add_argument("--sites-train", type=int, default=16)
    p.add_argument("--sites-val", type=int, default=50)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--aux-lambda", type=float, default=0.1)
    p.add_argument("--pos-weight", type=float, default=None,
                    help="BCE pos_weight for imbalanced pos:neg (e.g. 5.0 for 1:5 ratio)")
    p.add_argument("--stratified-per-group", type=str, default=None,
                    help="JSON dict mapping violation_profile group name to per-batch TNP count. "
                         "Example: '{\"positive\":2,\"paired_shuffle_v42\":2,\"wrong_orientation_v42\":2,"
                         "\"wrong_position_v42\":2,\"wrong_length_v42\":2,\"wrong_structure_role_v42\":2}'. "
                         "When set, overrides --tnp-batch/shuffle DataLoader.")
    p.add_argument("--stratified-steps-per-epoch", type=int, default=None,
                    help="Explicit steps/epoch for stratified sampler. Defaults to min(pool/k).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--max-val-tnps", type=int, default=None)
    p.add_argument("--max-train-steps", type=int, default=None)
    # V5 cross-site dispersion branch
    p.add_argument("--use-dispersion", action="store_true",
                   help="V5: enable per-tnp dispersion features branch (detached).")
    p.add_argument("--disp-hidden", type=int, default=32,
                   help="V5: hidden dim of the dispersion MLP (default 32).")
    p.add_argument("--dispersion-mode", choices=("scalar", "hidden_residual"),
                   default="scalar",
                   help='V5 fusion mode: "scalar" (V5.1: logit+=α*δ) or '
                        '"hidden_residual" (V5.2: h+=β*Δh via fusion_mlp).')
    # V5.1: warm-start + freeze-backbone + guardrails
    p.add_argument("--init-from", type=str, default=None,
                   help="V5.1: path to a .pt checkpoint to warm-start from.")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="V5.1: freeze everything except disp_head + disp_alpha.")
    p.add_argument("--nc-top1-gate", type=float, default=0.0,
                   help="V5.1: minimum nc_top1 required for a val epoch to be eligible as ★ best.")
    p.add_argument("--wrong-orient-gate", type=float, default=0.0,
                   help="V5.1: minimum wrong_orientation AUROC required for a val epoch to be eligible as ★ best.")
    # V6: cognate-pairing branch
    p.add_argument("--use-pairing", action="store_true",
                   help="V6: enable pair_head branch + swap-flank augmentation + L_pair.")
    p.add_argument("--pair-hidden", type=int, default=32)
    p.add_argument("--pair-lambda", type=float, default=0.5)
    p.add_argument("--pair-margin", type=float, default=1.0)
    p.add_argument("--pair-lambda-warmup-epochs", type=float, default=1.0)
    p.add_argument("--freeze-pair-beta", action="store_true",
                   help="V6 Stage A: keep pair_beta = 0 (auxiliary head only).")
    p.add_argument("--no-freeze-pair-beta", dest="freeze_pair_beta", action="store_false")
    p.set_defaults(freeze_pair_beta=True)
    p.add_argument("--freeze-v52", action="store_true",
                   help="V6 Stage A: freeze all V5.2 modules, only pair_head+pair_fuse train.")
    p.add_argument("--freeze-v6-stage-b", action="store_true",
                   help="V6 Stage B: freeze encoder+cand_mil+disp branch; train nc_mil+set+cls+pair.")
    # V6 Stage B.1
    p.add_argument("--pair-beta-init", type=float, default=0.0,
                   help="V6 Stage B.1: warm-start pair_beta to this value (default 0).")
    p.add_argument("--prop-lambda", type=float, default=0.0,
                   help="V6 Stage B.1: weight on cognate-vs-swap final-logit propagation loss.")
    p.add_argument("--prop-margin", type=float, default=1.0,
                   help="V6 Stage B.1: margin for propagation loss (in logit units).")
    # 48C1a: geometry bypass diagnostic
    p.add_argument("--use-geom-bypass", action="store_true",
                   help="48C1a: enable bag-level orient+position summary added to the logit.")
    p.add_argument("--geom-hidden", type=int, default=32)
    # 48C1b: two-branch disentangled architecture
    p.add_argument("--use-multi-branch", action="store_true",
                   help="48C1b: enable pair+geom two-branch model with profile-masked aux losses.")
    p.add_argument("--geom-dim", type=int, default=32)
    p.add_argument("--geom-mlp-hidden", type=int, default=64)
    p.add_argument("--geom-set-depth", type=int, default=1)
    p.add_argument("--geom-set-heads", type=int, default=2)
    p.add_argument("--lambda-pair-aux", type=float, default=0.5)
    p.add_argument("--lambda-geom-aux", type=float, default=0.5)
    p.add_argument("--use-explicit-geom-stats", action="store_true",
                   help="48C1c: enable hand-computed bag statistics concatenated to E_set.")
    p.add_argument("--no-explicit-geom-stats", dest="use_explicit_geom_stats", action="store_false")
    p.set_defaults(use_explicit_geom_stats=True)
    p.add_argument("--use-additive-fusion", action="store_true",
                   help="48C1d: logit = alpha·s_pair_aux + beta·s_geom_aux + h_fusion([E_pair;E_geom]) "
                        "with alpha=beta=1 init and h_fusion zero-init. Prevents fusion from suppressing "
                        "geom evidence at init.")
    p.add_argument("--normalize-aux-logits", action="store_true",
                   help="48C1e: BatchNorm1d each aux logit before combining. Removes s_pair vs "
                        "s_geom magnitude imbalance that masks weaker per-axis signals.")
    p.add_argument("--use-and-fusion", action="store_true",
                   help="48C1f: property-supervised AND fusion — p_final = σ(s_pair)·σ(s_geom). "
                        "Requires property-specific supervision via PROFILE_Y_PAIR/PROFILE_Y_GEOM.")
    # 48C1g: paired counterfactual training
    p.add_argument("--use-paired-batch", action="store_true",
                   help="48C1g: paired counterfactual batching + geom-only training.")
    p.add_argument("--paired-profiles-json", type=str, default=None,
                   help="Path to JSON list of [profile_name, tnp_suffix]. First entry is the parent.")
    p.add_argument("--k-parents-per-batch", type=int, default=2)
    p.add_argument("--freeze-pair-branch", action="store_true",
                   help="48C1g: freeze encoder+MIL+set+pma+classifier+h_pair_aux; train only geom.")
    p.add_argument("--geom-ranking-neg-idx", type=int, default=3,
                   help="Profile index that POS must outrank on geom head (typically wrong_position=3).")
    p.add_argument("--geom-invariance-indices", type=str, default="1,2",
                   help="Comma-separated indices geom should be invariant to (e.g. shuffle=1,length=2).")
    p.add_argument("--geom-margin", type=float, default=0.1,
                   help="Ranking hinge margin for s_geom(POS) - s_geom(wrong_pos).")
    p.add_argument("--lambda-geom-inv", type=float, default=1.0,
                   help="Weight on invariance MSE (POS ≈ shuffle ≈ length in geom).")
    p.add_argument("--lambda-geom-prop", type=float, default=0.0,
                   help="48C1h-A: weight on absolute-level property BCE for geom head.")
    # 48C2a: orientation branch training
    p.add_argument("--use-orient-branch", action="store_true",
                   help="48C2a: instantiate the orientation branch (h_orient_aux + tiny MLP).")
    p.add_argument("--train-orient-only", action="store_true",
                   help="48C2a: use paired-orient loss (freeze pair+geom).")
    p.add_argument("--freeze-geom-branch", action="store_true",
                   help="48C2a: freeze geom_input_mlp/set/pma/h_geom_aux + fusion.")
    p.add_argument("--orient-ranking-neg-idx", type=int, default=4,
                   help="Index of wrong_orientation profile in paired_profiles.")
    p.add_argument("--orient-invariance-indices", type=str, default="1,2,3",
                   help="Comma-separated indices orient should be invariant to.")
    p.add_argument("--orient-margin", type=float, default=0.1)
    p.add_argument("--lambda-orient-inv", type=float, default=1.0)
    p.add_argument("--lambda-orient-prop", type=float, default=0.3)
    p.add_argument("--pair-backbone-lr-scale", type=float, default=1.0,
                   help="48C1c: LR multiplier for warm-started pair backbone (encoder/mil/set/pma/classifier). "
                        "0.1 protects the pretrained pair pipeline while the geometry branch catches up.")
    args = p.parse_args()

    v1_cfg = V1Config(
        use_dispersion=args.use_dispersion,
        disp_hidden=args.disp_hidden,
        dispersion_mode=args.dispersion_mode,
        use_pairing=args.use_pairing,
        pair_hidden=args.pair_hidden,
        use_geom_bypass=args.use_geom_bypass,
        geom_hidden=args.geom_hidden,
        use_multi_branch=args.use_multi_branch,
        geom_dim=args.geom_dim,
        geom_mlp_hidden=args.geom_mlp_hidden,
        geom_set_depth=args.geom_set_depth,
        geom_set_heads=args.geom_set_heads,
        use_explicit_geom_stats=args.use_explicit_geom_stats,
        use_additive_fusion=args.use_additive_fusion,
        normalize_aux_logits=args.normalize_aux_logits,
        use_and_fusion=args.use_and_fusion,
        use_orient_branch=args.use_orient_branch,
    )

    # 48C1g: load paired profiles JSON if requested.
    paired_profiles_val = ()
    if args.use_paired_batch:
        from pathlib import Path as _P
        import json as _json
        if not args.paired_profiles_json:
            raise ValueError("--use-paired-batch requires --paired-profiles-json <path.json>")
        _raw = _json.loads(_P(args.paired_profiles_json).read_text())
        paired_profiles_val = tuple((str(name), str(suf)) for name, suf in _raw)
    geom_inv_idx_val = tuple(int(x) for x in args.geom_invariance_indices.split(","))

    cfg = TrainConfig(
        train_jsonl=args.train_jsonl,
        train_cache=args.train_cache,
        val_jsonl=args.val_jsonl,
        val_cache=args.val_cache,
        out_dir=args.out_dir,
        epochs=args.epochs,
        tnp_batch=args.tnp_batch,
        sites_train=args.sites_train,
        sites_val=args.sites_val,
        num_workers=args.num_workers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_frac=args.warmup_frac,
        grad_clip=args.grad_clip,
        aux_lambda=args.aux_lambda,
        pos_weight=args.pos_weight,
        stratified_per_group=(_load_stratified_spec(args.stratified_per_group)
                                if args.stratified_per_group else None),
        stratified_steps_per_epoch=args.stratified_steps_per_epoch,
        use_multi_branch=args.use_multi_branch,
        lambda_pair_aux=args.lambda_pair_aux,
        lambda_geom_aux=args.lambda_geom_aux,
        pair_backbone_lr_scale=args.pair_backbone_lr_scale,
        use_paired_batch=args.use_paired_batch,
        paired_profiles=paired_profiles_val,
        k_parents_per_batch=args.k_parents_per_batch,
        freeze_pair_branch=args.freeze_pair_branch,
        geom_ranking_neg_idx=args.geom_ranking_neg_idx,
        geom_invariance_idx=geom_inv_idx_val,
        geom_margin=args.geom_margin,
        lambda_geom_inv=args.lambda_geom_inv,
        lambda_geom_prop=args.lambda_geom_prop,
        freeze_geom_branch=args.freeze_geom_branch,
        train_orient_only=args.train_orient_only,
        orient_ranking_neg_idx=args.orient_ranking_neg_idx,
        orient_invariance_idx=tuple(int(x) for x in args.orient_invariance_indices.split(",")),
        orient_margin=args.orient_margin,
        lambda_orient_inv=args.lambda_orient_inv,
        lambda_orient_prop=args.lambda_orient_prop,
        seed=args.seed,
        bf16=not args.no_bf16,
        log_every=args.log_every,
        max_val_tnps=args.max_val_tnps,
        max_train_steps=args.max_train_steps,
        v1_cfg=v1_cfg,
        init_from=args.init_from,
        freeze_backbone=args.freeze_backbone,
        nc_top1_gate=args.nc_top1_gate,
        wrong_orient_gate=args.wrong_orient_gate,
        use_pairing=args.use_pairing,
        pair_lambda=args.pair_lambda,
        pair_lambda_warmup_epochs=args.pair_lambda_warmup_epochs,
        pair_margin=args.pair_margin,
        freeze_pair_beta=args.freeze_pair_beta,
        freeze_v52=args.freeze_v52,
        freeze_v6_stage_b=args.freeze_v6_stage_b,
        pair_beta_init=args.pair_beta_init,
        prop_lambda=args.prop_lambda,
        prop_margin=args.prop_margin,
    )
    train(cfg)


if __name__ == "__main__":
    main()
