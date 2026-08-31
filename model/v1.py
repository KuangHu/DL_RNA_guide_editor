"""V1 classifier: candidate CNN -> candidate MIL -> NC MIL -> Tnp Set Transformer.

Input tensors (from preprocess/tnp_dataset.collate_tnp_batch, to_torch=True):

  candidate_patches   float32  (B, S, N_nc, K, W, C_patch)
  candidate_features  float32  (B, S, N_nc, K, F)
  candidate_mask      bool     (B, S, N_nc, K)   True = real candidate
  nc_region_mask      bool     (B, S, N_nc)      True = populated NC slot
  true_slot_idx       int32    (B, S)            index of GT candidate in K,
                                                  -1 if unknown / negative
  is_positive         bool     (B,)              tnp label

Shape defaults for this dataset:
  N_nc=3, K=96, W=64, C_patch=22, F=13

Layer stack (see docstring on each class):

  CandidateEncoder:  per-candidate  -> 128-D token
                      [ z_structure(64) | z_alignment(48) | z_position(16) ]

  GatedAttentionMIL: K candidates      -> 1 NC   token (128-D)
  GatedAttentionMIL: N_nc NC tokens    -> 1 site token (128-D)
  SetTransformerBlock x 2:  S site tokens  <-> S site tokens  (self-attention)
  PMA (learned queries):    S site tokens  -> 1 tnp token (128-D)
  Classifier MLP 128 -> 64 -> 1 -> BCEWithLogits

Loss (see v1_loss):
  L_tnp        = BCE(tnp_logit, is_positive)
  L_candidate  = CE(pre-softmax candidate attention at active NC slot,
                     true_slot_idx)   (positives only, masked when idx < 0)
  L_total      = L_tnp + lambda * L_candidate
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- #
#  Feature split for the 13-D per-candidate scalar feature vector.
# ---------------------------------------------------------------- #

# Matches preprocess/candidates.py::FEATURE_NAMES  (indexes 0..12):
#   0: orient_fwd       1: orient_rc
#   2: L                3: matches
#   4: mismatches       5: score
#   6: flank_start_norm 7: flank_end_norm
#   8: boundary_dist_up 9: boundary_dist_dn
#  10: target_side_up  11: nc_start_norm
#  12: nc_len_norm
ALIGN_FEATURE_INDICES: tuple[int, ...] = (0, 1, 2, 3, 4, 5)    # 6-D
POS_FEATURE_INDICES:   tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)  # 7-D

# Feature slot indices used by the V5 dispersion branch.
IDX_ORIENT_FWD = 0
IDX_L          = 2
IDX_FLANK_START = 6   # flank_start_norm in [0, 1]
IDX_NC_START   = 11   # nc_start_norm  in [0, 1]

# Bp-scale multipliers for dispersion features. `NC_LEN_SCALE` is a rough
# constant matching the ~mean NC-region length; the small MLP downstream
# adapts to whatever scale we choose, so exact value is not critical.
FLANK_LEN_BP = 120.0
NC_LEN_SCALE = 250.0


# ---------------------------------------------------------------- #
#  Config
# ---------------------------------------------------------------- #

@dataclass
class V1Config:
    # candidate input
    patch_width: int = 64
    patch_channels: int = 22
    num_features: int = 13
    num_candidates: int = 96
    num_nc: int = 3

    # candidate encoder
    conv_channels: tuple[int, ...] = (48, 64, 96)
    conv_kernel: int = 5
    struct_out: int = 64
    align_hidden: int = 32
    align_out: int = 48
    pos_hidden: int = 16
    pos_out: int = 16
    dropout: float = 0.1

    # MILs
    mil_hidden: int = 128
    site_dim: int = 128           # = struct_out + align_out + pos_out (64+48+16)

    # tnp-level set transformer
    set_heads: int = 4
    set_depth: int = 2
    set_ff_mult: int = 2
    pma_num_seeds: int = 1

    # classifier
    cls_hidden: int = 64

    # V5 cross-site dispersion branch (detached from candidate scorer).
    #   use_dispersion=True enables a 6-feature branch that reads argmax-picked
    #   candidate coordinates per site, computes per-tnp dispersion stats
    #   (MAD/STD/IQR of target position; STD of NC-start; STD of L; orientation
    #   entropy). Gradient does NOT flow back into candidate scorer.
    #
    # dispersion_mode:
    #   "scalar"          V5.1: logit = base_logit_V4 + alpha * disp_head(phi)
    #                     Additive at the logit level. Cannot learn interactions.
    #   "hidden_residual" V5.2: h_V4 ∈ R^64 -> Δh via fusion_mlp([h_V4; d]) ->
    #                     h' = h_V4 + β * Δh -> logit = classifier[3](h').
    #                     Dispersion can gate/modulate V4's hidden evidence.
    use_dispersion: bool = False
    disp_hidden: int = 32
    dispersion_mode: str = "scalar"

    # V6 cognate-pairing branch.
    #   use_pairing=True adds a small pair_head over the NC token (post cand_mil).
    #   q_nc = pair_head(nc_tok)          — scalar per NC region (for contrastive loss)
    #   nc_tok' = nc_tok + pair_beta * pair_fuse(q_nc)  — fusion back into pathway
    #   pair_beta init 0 → V6 output == V5.2 output at t=0 (bitwise, verified).
    #   Stage A of V6 freezes pair_beta at 0 (auxiliary head only).
    #   Stage B unfreezes pair_beta so the pathway starts using q_nc.
    use_pairing: bool = False
    pair_hidden: int = 32

    # 48C1a: bag-level geometry bypass diagnostic.
    #   Computes a small vector of NC-length-invariant orient+position summaries
    #   from candidate_features and injects an additive logit correction:
    #     logit_final = logit + geom_head(s_geom)
    #   Purpose: test whether orientation/position information can be used by
    #   the classifier if it is fed AROUND the MIL pathway. See PROJECT_JOURNAL
    #   for the diagnostic ladder framing.
    use_geom_bypass: bool = False
    geom_hidden: int = 32
    num_geom_feats: int = 6

    # 48C1b/c: two-branch disentangled evidence architecture.
    #   Pair branch = existing V1 pipeline through PMA (E_pair, 128-D).
    #   Geom branch = per-site (orient_top1, flank_start_top1, L_top1) → MLP →
    #     tiny SetTransformer over sites → PMA → E_set (geom_dim).
    #   48C1c: E_geom = [E_set ; S_explicit] where S_explicit is a set of
    #   HAND-COMPUTED bag-level statistics on orientation + position — so the
    #   SetTransformer doesn't have to re-derive medians/entropy from scratch.
    #   Three heads:
    #     h_pair_aux(E_pair) → s_pair             (aux)
    #     h_geom_aux(E_geom) → s_geom             (aux)
    #     h_fusion([E_pair; E_geom]) → logit_final (main)
    #   Loss uses profile-masked auxiliary BCEs — see v1_multi_branch_loss.
    use_multi_branch: bool = False
    geom_dim: int = 32
    geom_mlp_hidden: int = 64
    geom_set_depth: int = 1
    geom_set_heads: int = 2
    use_explicit_geom_stats: bool = True  # 48C1c: enable hybrid geometry branch
    num_explicit_geom_stats: int = 10
    # 48C1d: additive fusion —  logit_final = α·s_pair_aux + β·s_geom_aux + r
    #   where α,β are learnable scalars init at 1.0 and r is h_fusion([E_pair;E_geom])
    #   with its final Linear zero-initialized. Purpose: force both aux logits to
    #   contribute unconditionally at init so h_fusion cannot suppress geom evidence.
    use_additive_fusion: bool = False
    # 48C1e: normalize each aux logit via BatchNorm1d BEFORE the α/β combination.
    #   Fixes magnitude mismatch — s_pair spans ~1 unit while s_geom spans ~0.02
    #   on axes where only geom carries signal (e.g., position). With normalized
    #   inputs, α=β=1 gives equal-magnitude contributions and direction (not scale)
    #   accumulates.
    normalize_aux_logits: bool = False
    # 48C1f: AND-fusion of property-specific aux heads.
    #   p_final = sigmoid(s_pair_aux) * sigmoid(s_geom_aux)
    #   Both branches interpreted as independent probabilities of PAIR_VALID and
    #   GEOM_VALID respectively. AND rule (both must fire) gives final positive.
    #   No α/β, no h_fusion residual. Requires property-specific supervision:
    #   each aux head is trained on ALL profiles with their appropriate validity
    #   labels (e.g. wrong_position: y_pair=1, y_geom=0). See v1_and_fusion_loss.
    use_and_fusion: bool = False
    # 48C2a: dedicated orientation-validity branch.
    #   Input: 5 bag-level orientation stats (p_fwd_mean, p_fwd_top1, H_orient,
    #     C_orient=|2·p_fwd_mean-1|, top1_orient_consistency)
    #   → small MLP → h_orient_aux → s_orient_aux (log-odds of orient-validity)
    #   Deliberately NO SetTransformer — sufficient statistic is closed-form.
    #   Deliberately NO NC-length / n_NC / whole-NC coord features (anti-shortcut).
    use_orient_branch: bool = False
    orient_mlp_hidden: int = 32
    orient_dim: int = 16
    num_orient_stats: int = 5

    def __post_init__(self):
        assert self.struct_out + self.align_out + self.pos_out == self.site_dim, (
            f"struct+align+pos ({self.struct_out}+{self.align_out}+{self.pos_out}) "
            f"must equal site_dim ({self.site_dim})"
        )


# ---------------------------------------------------------------- #
#  Building blocks
# ---------------------------------------------------------------- #

class _AttentionPool1d(nn.Module):
    """Pool a variable-length sequence to a single vector using a single
    learnable query. Given x of shape (N, L, D), returns (N, D)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) * 0.02)
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, L, D)
        # score per position: (x + query) -> tanh -> linear -> softmax across L
        h = torch.tanh(x + self.query)      # (N, L, D)
        s = self.score(h).squeeze(-1)       # (N, L)
        w = torch.softmax(s, dim=-1)        # (N, L)
        return (w.unsqueeze(-1) * x).sum(dim=1)  # (N, D)


class CandidateEncoder(nn.Module):
    """Per-candidate token: 128-D = [struct(64) | align(48) | pos(16)].

    Structure branch:
      patch (B_cand, C_patch=22, W=64)
        -> Conv1d(22 -> 48, k=5) -> GELU -> Dropout
        -> Conv1d(48 -> 64, k=5) -> GELU -> Dropout
        -> Conv1d(64 -> 96, k=5) -> GELU
        -> attention pool over W -> Linear(96 -> struct_out=64)

    Alignment branch:
      align_feats (B_cand, 6) -> MLP -> align_out=48

    Position branch:
      pos_feats (B_cand, 7)   -> MLP -> pos_out=16
    """

    def __init__(self, cfg: V1Config):
        super().__init__()
        self.cfg = cfg
        c1, c2, c3 = cfg.conv_channels
        k = cfg.conv_kernel
        pad = k // 2
        self.conv1 = nn.Conv1d(cfg.patch_channels, c1, kernel_size=k, padding=pad)
        self.conv2 = nn.Conv1d(c1, c2, kernel_size=k, padding=pad)
        self.conv3 = nn.Conv1d(c2, c3, kernel_size=k, padding=pad)
        self.drop = nn.Dropout(cfg.dropout)
        self.pool = _AttentionPool1d(c3)
        self.struct_proj = nn.Linear(c3, cfg.struct_out)

        self.align_mlp = nn.Sequential(
            nn.Linear(len(ALIGN_FEATURE_INDICES), cfg.align_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.align_hidden, cfg.align_out),
            nn.GELU(),
        )
        self.pos_mlp = nn.Sequential(
            nn.Linear(len(POS_FEATURE_INDICES), cfg.pos_hidden),
            nn.GELU(),
            nn.Linear(cfg.pos_hidden, cfg.pos_out),
            nn.GELU(),
        )

    def forward(
        self, patches: torch.Tensor, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        patches:  (..., W, C_patch)     leading dims flattened by caller
        features: (..., F=13)

        Returns (z_struct, z_align, z_pos), each (..., <dim>).
        """
        # Flatten leading dims for the CNN.
        lead = patches.shape[:-2]
        W = patches.shape[-2]
        C = patches.shape[-1]
        n_lead = 1
        for d in lead:
            n_lead *= d
        x = patches.reshape(n_lead, W, C).transpose(1, 2)   # (N, C, W)
        x = self.drop(F.gelu(self.conv1(x)))
        x = self.drop(F.gelu(self.conv2(x)))
        x = F.gelu(self.conv3(x))                            # (N, c3, W)
        x = x.transpose(1, 2)                                # (N, W, c3)
        z_struct = self.pool(x)                              # (N, c3)
        z_struct = self.struct_proj(z_struct)                # (N, struct_out)
        z_struct = z_struct.reshape(*lead, self.cfg.struct_out)

        align_in = features[..., list(ALIGN_FEATURE_INDICES)]
        z_align = self.align_mlp(align_in)                   # (..., align_out)

        pos_in = features[..., list(POS_FEATURE_INDICES)]
        z_pos = self.pos_mlp(pos_in)                         # (..., pos_out)

        return z_struct, z_align, z_pos


class GatedAttentionMIL(nn.Module):
    """Ilse 2018 gated attention MIL.

    a_i = softmax( V^T ( tanh(W h_i) * sigmoid(U h_i) ) )   over i in set
    z   = sum_i a_i * h_i

    Supports a boolean mask over the set dimension.

    forward(x, mask):
      x:    (N, S, D)   set of S tokens per batch
      mask: (N, S)      True = keep, False = padded / invalid
    returns:
      z:    (N, D)
      attn: (N, S)   attention weights (masked entries are 0; sums to 1
                       across unmasked)
      raw:  (N, S)   pre-softmax scores (masked entries = -inf) — used by
                       the auxiliary candidate-localization loss.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.attn_V = nn.Linear(dim, hidden)
        self.attn_U = nn.Linear(dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (N, S, D)   mask: (N, S) bool
        h = torch.tanh(self.attn_V(x)) * torch.sigmoid(self.attn_U(x))
        raw = self.attn_w(h).squeeze(-1)               # (N, S)
        # Fully-masked rows: keep raw=0 so softmax gives a uniform distribution
        # over 0 elements? Instead, we return a zero pooled vector and a
        # zeroed attention when the row is fully masked, to avoid NaN.
        neg_inf = torch.finfo(raw.dtype).min
        masked_raw = raw.masked_fill(~mask, neg_inf)
        # Detect fully-empty rows.
        any_valid = mask.any(dim=-1, keepdim=True)     # (N, 1)
        # For empty rows, softmax(-inf all) is NaN; we replace with uniform then
        # zero out the pooled output.
        stable_raw = torch.where(any_valid, masked_raw, torch.zeros_like(masked_raw))
        attn = torch.softmax(stable_raw, dim=-1)       # (N, S)
        attn = attn * mask.float()                     # zero-out masked entries
        # renormalize (safe since any_valid guards divide-by-zero later)
        norm = attn.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        attn = attn / norm
        z = (attn.unsqueeze(-1) * x).sum(dim=1)        # (N, D)
        # Zero the output for fully-empty rows.
        z = z * any_valid.float()
        return z, attn, masked_raw


class _MultiHeadAttention(nn.Module):
    """Standard scaled-dot-product multi-head self- or cross-attention with
    an optional key padding mask."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0
        self.heads = heads
        self.dh = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        key_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # q, k, v: (N, Lq/Lk, D). key_mask: (N, Lk) bool, True = valid.
        N = q.size(0)
        Lq = q.size(1)
        Lk = k.size(1)
        Q = self.q(q).view(N, Lq, self.heads, self.dh).transpose(1, 2)  # (N, H, Lq, dh)
        K = self.k(k).view(N, Lk, self.heads, self.dh).transpose(1, 2)
        V = self.v(v).view(N, Lk, self.heads, self.dh).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dh ** 0.5)  # (N, H, Lq, Lk)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)                                       # (N, H, Lq, dh)
        out = out.transpose(1, 2).contiguous().view(N, Lq, self.heads * self.dh)
        return self.o(out)


class SetTransformerBlock(nn.Module):
    """SAB (Lee et al. 2019): a permutation-equivariant residual block.
    x <- LayerNorm(x + MHA(x, x, x))
    x <- LayerNorm(x + FF(x))
    """

    def __init__(self, dim: int, heads: int, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        self.attn = _MultiHeadAttention(dim, heads, dropout)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (N, S, D). mask: (N, S) bool.
        x = self.ln1(x + self.drop(self.attn(x, x, x, key_mask=mask)))
        x = self.ln2(x + self.drop(self.ff(x)))
        return x


class PMA(nn.Module):
    """Pooling by Multihead Attention (Lee et al. 2019).
    Learnable seed queries S: (n_seeds, D). Cross-attend to the input set."""

    def __init__(self, dim: int, heads: int, n_seeds: int = 1, dropout: float = 0.1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(n_seeds, dim) * 0.02)
        self.attn = _MultiHeadAttention(dim, heads, dropout)
        self.ln = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: (N, S, D). mask: (N, S) bool. Returns (N, n_seeds, D).
        N = x.size(0)
        q = self.seeds.unsqueeze(0).expand(N, -1, -1)   # (N, n_seeds, D)
        out = self.attn(q, x, x, key_mask=mask)
        return self.ln(out + self.ff(out))


# ---------------------------------------------------------------- #
#  Full V1 model
# ---------------------------------------------------------------- #

# ---------------------------------------------------------------- #
#  V5 helper: per-tnp cross-site dispersion of picked candidates.
# ---------------------------------------------------------------- #

@torch.amp.autocast(device_type="cuda", enabled=False)
def _compute_dispersion_features(
    cand_raw: torch.Tensor,          # (B, S, N, K)
    cand_features: torch.Tensor,     # (B, S, N, K, F)
    cand_mask: torch.Tensor,         # (B, S, N, K) bool
    nc_attn: torch.Tensor,           # (B, S, N)   softmax weights
) -> torch.Tensor:
    """Compute per-tnp 6-D dispersion features from model-picked candidates.

    All inputs are treated as detached: this function makes no assumption
    about gradient graph attachment; the caller controls that.

    Returns (B, 6) float:
        [pos_MAD, pos_STD, pos_IQR, ncstart_STD, L_STD, orient_entropy]

    pos is target position in bp (flank_start_norm * 120), nc_start uses a
    fixed 250-bp scale (nc lengths are ~140-341bp in practice; the MLP
    downstream adapts to scale). Orientation entropy is binary entropy of
    fraction of sites picked as 'forward'.

    Picking policy:
        * Active NC per site = argmax(nc_attn)
        * Top-1 candidate slot per site = argmax(cand_raw at that NC), with
          invalid candidate slots masked to -inf via cand_mask.
    """
    B, S, N, K, F = cand_features.shape
    device = cand_features.device

    # Cast all inputs to float32 explicitly (autocast-disabled context above
    # keeps ops in fp32; this line just guarantees no leftover bf16 tensor).
    cand_raw = cand_raw.float()
    cand_features = cand_features.float()
    nc_attn = nc_attn.float()

    # 1) Model-picked active NC per site.
    active_nc = nc_attn.argmax(dim=-1)                # (B, S)  int64

    # 2) cand_raw at active NC -> (B, S, K); mask invalid slots to -inf.
    nc_exp = active_nc.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, K)
    cr_at = cand_raw.gather(2, nc_exp).squeeze(2)     # (B, S, K)
    cm_at = cand_mask.gather(2, nc_exp).squeeze(2)    # (B, S, K)
    cr_at = cr_at.masked_fill(~cm_at.bool(), float("-inf"))

    # 3) Top-1 candidate slot per site.
    slot = cr_at.argmax(dim=-1)                       # (B, S)  int64

    # 4) Gather feature vector at (active_nc, slot) -> (B, S, F)
    feat_nc = cand_features.gather(
        2, active_nc.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, K, F)
    ).squeeze(2)                                       # (B, S, K, F)
    feat = feat_nc.gather(
        2, slot.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, F)
    ).squeeze(2)                                       # (B, S, F)

    pos_bp   = feat[..., IDX_FLANK_START] * FLANK_LEN_BP   # (B, S)
    ncstart  = feat[..., IDX_NC_START]   * NC_LEN_SCALE    # (B, S)
    L_val    = feat[..., IDX_L]                             # (B, S)
    ori_fwd  = feat[..., IDX_ORIENT_FWD]                    # (B, S)

    # 5) Per-tnp dispersion stats.
    # Cast to float32 to keep quantile / median accurate under bf16 autocast.
    pos_bp32 = pos_bp.float()
    pos_std = pos_bp32.std(dim=-1, unbiased=False)          # (B,)
    # Use torch.quantile for MAD & IQR (exact interpolated quantiles).
    qs = torch.tensor([0.25, 0.5, 0.75], device=device, dtype=torch.float32)
    pos_q = torch.quantile(pos_bp32, qs, dim=-1)            # (3, B)
    pos_iqr = pos_q[2] - pos_q[0]                            # (B,)
    pos_med = pos_q[1].unsqueeze(-1)                         # (B, 1)
    pos_mad = torch.quantile((pos_bp32 - pos_med).abs(),
                              torch.tensor(0.5, device=device, dtype=torch.float32),
                              dim=-1)                        # (B,)

    nc_std = ncstart.float().std(dim=-1, unbiased=False)    # (B,)
    L_std  = L_val.float().std(dim=-1, unbiased=False)      # (B,)

    # Orientation entropy (binary: forward vs reverse-complement). We use
    # a safe formulation: H = 0 when p_fwd is exactly 0 or 1, otherwise the
    # standard formula. This avoids log2(0) even if the fp32 subtraction
    # (1 - p) underflows to 0 for p_fwd extremely close to 1.
    p_fwd = ori_fwd.mean(dim=-1)                                          # (B,) fp32
    p_clamp = p_fwd.clamp(min=1e-7, max=1.0 - 1e-7)
    orient_H = -(p_clamp * torch.log2(p_clamp)
                  + (1.0 - p_clamp) * torch.log2(1.0 - p_clamp))
    orient_H = torch.where(
        (p_fwd <= 1e-7) | (p_fwd >= 1.0 - 1e-7),
        torch.zeros_like(orient_H),
        orient_H,
    )

    phi = torch.stack([pos_mad, pos_std, pos_iqr, nc_std, L_std, orient_H], dim=-1)
    return phi   # (B, 6)


def _compute_geom_summary(
    cand_features: torch.Tensor,      # (B, S, N, K, F)
    cand_mask: torch.Tensor,          # (B, S, N, K) bool
    nc_region_mask: torch.Tensor,     # (B, S, N)   bool
    site_mask: torch.Tensor,          # (B, S)     bool
) -> torch.Tensor:
    """Return (B, 6) bag-level orientation + junction-position summary.

    Features (all NC-length-invariant; junction/flank-referenced):
        [0] p_fwd_mean          — fraction of valid candidates with orient=fwd
        [1] p_fwd_top1          — fraction of sites whose top-1 candidate is fwd
        [2] H_orient            — Shannon entropy of the p_fwd_mean mixture
        [3] median flank_start  — median of top-1 flank_start_norm across sites
        [4] IQR flank_start     — Q75-Q25 of top-1 flank_start_norm across sites
        [5] junction_side_frac  — fraction of top-1 within 0.17 of flank center

    Feature indices used from cand_features:
        [0] orient_fwd, [3] matches, [6] flank_start_norm
    """
    B, S, N, K, F = cand_features.shape
    device = cand_features.device
    cand_features = cand_features.float()

    valid = (
        cand_mask
        & nc_region_mask.unsqueeze(-1)
        & site_mask.unsqueeze(-1).unsqueeze(-1)
    )                                                    # (B, S, N, K)
    vf = valid.float()

    orient_fwd = cand_features[..., 0]                    # (B, S, N, K)
    matches    = cand_features[..., 3]
    flank_st   = cand_features[..., 6]

    # (1) p_fwd_mean across all valid candidates.
    denom_all = vf.sum(dim=(1, 2, 3)).clamp(min=1.0)
    p_fwd_mean = (orient_fwd * vf).sum(dim=(1, 2, 3)) / denom_all         # (B,)

    # (2) top-1 candidate per site (max matches over (N*K)); mask invalid to -inf.
    matches_flat = matches.masked_fill(~valid, float("-inf")).reshape(B, S, N * K)
    top1 = matches_flat.argmax(dim=-1)                                   # (B, S)
    orient_flat = orient_fwd.reshape(B, S, N * K)
    flank_flat  = flank_st.reshape(B, S, N * K)
    orient_top1 = orient_flat.gather(-1, top1.unsqueeze(-1)).squeeze(-1) # (B, S)
    flank_top1  = flank_flat.gather(-1, top1.unsqueeze(-1)).squeeze(-1)  # (B, S)

    # A site is "usable" for top-1 if it has at least one valid candidate.
    site_has_valid = valid.any(dim=(2, 3)) & site_mask                    # (B, S)
    sv = site_has_valid.float()
    denom_sites = sv.sum(dim=1).clamp(min=1.0)                            # (B,)

    p_fwd_top1 = (orient_top1 * sv).sum(dim=1) / denom_sites              # (B,)

    # (3) Shannon entropy of p_fwd_mean (binary).
    eps = 1e-6
    H_orient = -(
        p_fwd_mean * (p_fwd_mean + eps).log()
        + (1.0 - p_fwd_mean) * (1.0 - p_fwd_mean + eps).log()
    )                                                                     # (B,)

    # (4-5) median + IQR of flank_top1 across sites. Use nanquantile with
    # invalid-site fill = NaN.
    flank_nan = torch.where(
        site_has_valid, flank_top1, torch.full_like(flank_top1, float("nan"))
    )
    med_flank = torch.nanquantile(flank_nan, 0.5, dim=1)                  # (B,)
    q75 = torch.nanquantile(flank_nan, 0.75, dim=1)
    q25 = torch.nanquantile(flank_nan, 0.25, dim=1)
    iqr_flank = q75 - q25                                                 # (B,)

    # NaN safety: if a bag had 0 valid sites, quantiles are NaN → fill 0.5 / 0.
    med_flank = torch.where(torch.isnan(med_flank), torch.full_like(med_flank, 0.5), med_flank)
    iqr_flank = torch.where(torch.isnan(iqr_flank), torch.zeros_like(iqr_flank), iqr_flank)

    # (6) junction_side_frac.
    dist_center = (flank_top1 - 0.5).abs()
    within = (dist_center < 0.17).float()
    junction_side_frac = (within * sv).sum(dim=1) / denom_sites            # (B,)

    s_geom = torch.stack(
        [p_fwd_mean, p_fwd_top1, H_orient, med_flank, iqr_flank, junction_side_frac],
        dim=-1,
    )                                                                     # (B, 6)
    return s_geom


def _compute_geom_stats_explicit(
    cand_features: torch.Tensor,      # (B, S, N, K, F)
    cand_mask: torch.Tensor,          # (B, S, N, K) bool
    nc_region_mask: torch.Tensor,     # (B, S, N)   bool
    site_mask: torch.Tensor,          # (B, S)     bool
) -> torch.Tensor:
    """Return (B, 10) bag-level explicit statistics for the 48C1c hybrid geometry
    branch. All interaction-geometry, junction-referenced; no NC-layout deps.

    Feature layout:
        [0] p_fwd_mean                        (all valid candidates)
        [1] p_fwd_top1                        (top-1 per site by matches)
        [2] H_orient (binary entropy of p_fwd_mean)
        [3] orient_concentration = |2·p_fwd_mean - 1|
        [4] top1_orient_consistency           (fraction of sites' top-1 matching the bag plurality orient)
        [5] median flank_start_norm           (over top-1 per site)
        [6] IQR flank_start_norm              (over top-1 per site)
        [7] MAD flank_start_norm              (median-absolute-deviation)
        [8] fraction_upstream                 (top-1 flank_start_norm < 0.5)
        [9] junction_side_frac                (|top-1 flank_start_norm - 0.5| < 0.17)
    """
    B, S, N, K, F = cand_features.shape
    cand_features = cand_features.float()

    valid = (
        cand_mask
        & nc_region_mask.unsqueeze(-1)
        & site_mask.unsqueeze(-1).unsqueeze(-1)
    )                                                                      # (B, S, N, K)
    vf = valid.float()
    orient_fwd = cand_features[..., 0]
    matches    = cand_features[..., 3]
    flank_st   = cand_features[..., 6]

    # p_fwd_mean, H, concentration
    denom_all = vf.sum(dim=(1, 2, 3)).clamp(min=1.0)
    p_fwd_mean = (orient_fwd * vf).sum(dim=(1, 2, 3)) / denom_all           # (B,)
    eps = 1e-6
    H_orient = -(
        p_fwd_mean * (p_fwd_mean + eps).log()
        + (1.0 - p_fwd_mean) * (1.0 - p_fwd_mean + eps).log()
    )
    concentration = (2.0 * p_fwd_mean - 1.0).abs()                          # (B,)

    # top-1 per site
    m_flat = matches.masked_fill(~valid, float("-inf")).reshape(B, S, N * K)
    top1 = m_flat.argmax(dim=-1)                                            # (B, S)
    orient_top1 = orient_fwd.reshape(B, S, N * K).gather(-1, top1.unsqueeze(-1)).squeeze(-1)
    flank_top1  = flank_st.reshape(B, S, N * K).gather(-1, top1.unsqueeze(-1)).squeeze(-1)

    site_has_valid = valid.any(dim=(2, 3)) & site_mask                       # (B, S)
    sv = site_has_valid.float()
    n_sites = sv.sum(dim=1).clamp(min=1.0)                                   # (B,)

    p_fwd_top1 = (orient_top1 * sv).sum(dim=1) / n_sites                     # (B,)

    # top1 orient consistency: fraction of sites whose top-1 orient == bag plurality.
    plurality = (p_fwd_top1 >= 0.5).float().unsqueeze(1)                     # (B, 1) — 1 if fwd is majority
    matches_plur = (orient_top1 == plurality).float() * sv                    # (B, S)
    top1_orient_consistency = matches_plur.sum(dim=1) / n_sites              # (B,)

    # Position stats on flank_top1 (nan-filled invalid sites).
    flank_nan = torch.where(
        site_has_valid, flank_top1, torch.full_like(flank_top1, float("nan"))
    )
    med_flank = torch.nanquantile(flank_nan, 0.5, dim=1)
    q75 = torch.nanquantile(flank_nan, 0.75, dim=1)
    q25 = torch.nanquantile(flank_nan, 0.25, dim=1)
    iqr_flank = q75 - q25
    # MAD via median of |x - median|
    abs_dev = (flank_nan - med_flank.unsqueeze(1)).abs()
    mad_flank = torch.nanquantile(abs_dev, 0.5, dim=1)

    fraction_upstream = ((flank_top1 < 0.5).float() * sv).sum(dim=1) / n_sites
    dist_center = (flank_top1 - 0.5).abs()
    junction_side_frac = ((dist_center < 0.17).float() * sv).sum(dim=1) / n_sites

    # NaN safety
    def _nz(x, fill):
        return torch.where(torch.isnan(x), torch.full_like(x, fill), x)
    med_flank = _nz(med_flank, 0.5)
    iqr_flank = _nz(iqr_flank, 0.0)
    mad_flank = _nz(mad_flank, 0.0)

    s = torch.stack([
        p_fwd_mean, p_fwd_top1, H_orient, concentration, top1_orient_consistency,
        med_flank, iqr_flank, mad_flank, fraction_upstream, junction_side_frac,
    ], dim=-1)                                                                # (B, 10)
    return s


def _compute_orient_stats(
    cand_features: torch.Tensor,      # (B, S, N, K, F)
    cand_mask: torch.Tensor,          # (B, S, N, K) bool
    nc_region_mask: torch.Tensor,     # (B, S, N)   bool
    site_mask: torch.Tensor,          # (B, S)     bool
) -> torch.Tensor:
    """48C2a: orientation-only bag-level sufficient statistics. (B, 5).

    Deliberately excludes:
      - flank_start / boundary distance (position leakage)
      - L (length leakage)
      - any NC-layout normalization

    Only orient_fwd (feature 0) and matches (feature 3, used as top-1 selector
    and confidence weight) are read.

    Feature layout:
        [0] p_fwd_mean                        (all valid candidates)
        [1] p_fwd_top1                        (top-1 per site by matches)
        [2] H_orient (binary entropy of p_fwd_mean)
        [3] C_orient = |2·p_fwd_mean - 1|     (concentration)
        [4] top1_orient_consistency           (fraction of sites' top-1 matching plurality)
    """
    B, S, N, K, F = cand_features.shape
    cand_features = cand_features.float()

    valid = (
        cand_mask
        & nc_region_mask.unsqueeze(-1)
        & site_mask.unsqueeze(-1).unsqueeze(-1)
    )                                                                       # (B, S, N, K)
    vf = valid.float()
    orient_fwd = cand_features[..., 0]
    matches    = cand_features[..., 3]

    denom_all = vf.sum(dim=(1, 2, 3)).clamp(min=1.0)
    p_fwd_mean = (orient_fwd * vf).sum(dim=(1, 2, 3)) / denom_all           # (B,)
    eps = 1e-6
    H_orient = -(
        p_fwd_mean * (p_fwd_mean + eps).log()
        + (1.0 - p_fwd_mean) * (1.0 - p_fwd_mean + eps).log()
    )
    concentration = (2.0 * p_fwd_mean - 1.0).abs()                          # (B,)

    # top-1 per site by matches
    m_flat = matches.masked_fill(~valid, float("-inf")).reshape(B, S, N * K)
    top1 = m_flat.argmax(dim=-1)                                            # (B, S)
    orient_top1 = orient_fwd.reshape(B, S, N * K).gather(-1, top1.unsqueeze(-1)).squeeze(-1)

    site_has_valid = valid.any(dim=(2, 3)) & site_mask                       # (B, S)
    sv = site_has_valid.float()
    n_sites = sv.sum(dim=1).clamp(min=1.0)

    p_fwd_top1 = (orient_top1 * sv).sum(dim=1) / n_sites                     # (B,)

    plurality = (p_fwd_top1 >= 0.5).float().unsqueeze(1)
    matches_plur = (orient_top1 == plurality).float() * sv
    top1_orient_consistency = matches_plur.sum(dim=1) / n_sites              # (B,)

    return torch.stack(
        [p_fwd_mean, p_fwd_top1, H_orient, concentration, top1_orient_consistency],
        dim=-1,
    )                                                                         # (B, 5)


class V1Model(nn.Module):
    def __init__(self, cfg: V1Config = V1Config()):
        super().__init__()
        self.cfg = cfg
        self.encoder = CandidateEncoder(cfg)
        self.cand_mil = GatedAttentionMIL(cfg.site_dim, cfg.mil_hidden)
        self.nc_mil = GatedAttentionMIL(cfg.site_dim, cfg.mil_hidden)
        self.set_blocks = nn.ModuleList([
            SetTransformerBlock(cfg.site_dim, cfg.set_heads,
                                 ff_mult=cfg.set_ff_mult, dropout=cfg.dropout)
            for _ in range(cfg.set_depth)
        ])
        self.pma = PMA(cfg.site_dim, cfg.set_heads, n_seeds=cfg.pma_num_seeds,
                        dropout=cfg.dropout)

        # Classifier is ALWAYS the V4-shape one (128 -> cls_hidden -> 1). Loading
        # a V4 checkpoint into this V1Model with use_dispersion=True must keep
        # classifier weights bitwise identical.
        self.classifier = nn.Sequential(
            nn.Linear(cfg.site_dim * cfg.pma_num_seeds, cfg.cls_hidden),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.cls_hidden, 1),
        )

        # V5 dispersion branch — detached φ, mode-dependent fusion.
        # At init (α=0 / β=0) the model is IDENTICAL to V4 regardless of mode,
        # so a warmup-from-V4-init cannot destroy the primary path.
        use_disp = getattr(cfg, "use_dispersion", False)
        mode = getattr(cfg, "dispersion_mode", "scalar")
        self.disp_mode = mode if use_disp else None

        # V5.1 scalar residual: disp_head + disp_alpha
        if use_disp and mode == "scalar":
            self.disp_head = nn.Sequential(
                nn.Linear(6, cfg.disp_hidden),
                nn.GELU(),
                nn.LayerNorm(cfg.disp_hidden),
                nn.Linear(cfg.disp_hidden, 1),
            )
            self.disp_alpha = nn.Parameter(torch.zeros(1))
        else:
            self.disp_head = None
            self.disp_alpha = None

        # V5.2 hidden-residual fusion: disp_encoder + fusion_mlp + disp_beta
        if use_disp and mode == "hidden_residual":
            self.disp_encoder = nn.Sequential(
                nn.Linear(6, cfg.disp_hidden),
                nn.GELU(),
                nn.LayerNorm(cfg.disp_hidden),
            )
            # h_V4 has size cfg.cls_hidden (=64); fuse [h_V4; d] -> Δh of size 64.
            self.fusion_mlp = nn.Sequential(
                nn.Linear(cfg.cls_hidden + cfg.disp_hidden, cfg.cls_hidden),
                nn.GELU(),
                nn.LayerNorm(cfg.cls_hidden),
            )
            self.disp_beta = nn.Parameter(torch.zeros(1))
        else:
            self.disp_encoder = None
            self.fusion_mlp = None
            self.disp_beta = None

        # V6 cognate-pairing branch (pair_head + pair_fuse + pair_beta).
        use_pair = getattr(cfg, "use_pairing", False)
        if use_pair:
            self.pair_head = nn.Sequential(
                nn.Linear(cfg.site_dim, cfg.pair_hidden),
                nn.GELU(),
                nn.LayerNorm(cfg.pair_hidden),
                nn.Linear(cfg.pair_hidden, 1),
            )
            # pair_fuse: (B, S, N, 1) -> (B, S, N, site_dim) correction on nc_tok
            self.pair_fuse = nn.Linear(1, cfg.site_dim, bias=False)
            # β init 0 → V6 forward ≡ V5.2 forward bitwise
            self.pair_beta = nn.Parameter(torch.zeros(1))
        else:
            self.pair_head = None
            self.pair_fuse = None
            self.pair_beta = None

        # 48C1a: geometry bypass head (see V1Config.use_geom_bypass).
        if cfg.use_geom_bypass:
            self.geom_head = nn.Sequential(
                nn.Linear(cfg.num_geom_feats, cfg.geom_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.geom_hidden, 1),
            )
        else:
            self.geom_head = None

        # 48C1b/c: two-branch disentangled architecture (see V1Config.use_multi_branch).
        if cfg.use_multi_branch:
            D_pair = cfg.site_dim * cfg.pma_num_seeds
            # 48C1c: geom evidence = [E_set (from SetTransformer over per-site tokens);
            #                        S_explicit (hand-computed bag statistics)].
            D_geom_set = cfg.geom_dim
            D_geom_stats = cfg.num_explicit_geom_stats if cfg.use_explicit_geom_stats else 0
            D_geom = D_geom_set + D_geom_stats
            # per-site input: (orient_top1, flank_start_top1, L_top1)
            self.geom_input_mlp = nn.Sequential(
                nn.Linear(3, cfg.geom_mlp_hidden),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.geom_mlp_hidden, cfg.geom_dim),
            )
            self.geom_set_blocks = nn.ModuleList([
                SetTransformerBlock(cfg.geom_dim, cfg.geom_set_heads,
                                     ff_mult=cfg.set_ff_mult, dropout=cfg.dropout)
                for _ in range(cfg.geom_set_depth)
            ])
            self.geom_pma = PMA(cfg.geom_dim, cfg.geom_set_heads,
                                 n_seeds=1, dropout=cfg.dropout)
            self.h_pair_aux = nn.Sequential(
                nn.Linear(D_pair, cfg.cls_hidden), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.cls_hidden, 1),
            )
            self.h_geom_aux = nn.Sequential(
                nn.Linear(D_geom, cfg.cls_hidden), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.cls_hidden, 1),
            )
            self.h_fusion = nn.Sequential(
                nn.Linear(D_pair + D_geom, cfg.cls_hidden), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.cls_hidden, 1),
            )
            # 48C1d: additive fusion parameters. When use_additive_fusion is True,
            # both aux logits contribute directly to logit_final at initialization
            # (α=β=1) and the fusion residual r is zero-initialized so it cannot
            # suppress s_pair_aux/s_geom_aux early. Both scalars are learnable.
            if cfg.use_additive_fusion:
                self.alpha_pair = nn.Parameter(torch.ones(1))
                self.alpha_geom = nn.Parameter(torch.ones(1))
                # Zero-init the final Linear of h_fusion (both weight and bias).
                _last_lin = None
                for m in self.h_fusion:
                    if isinstance(m, nn.Linear):
                        _last_lin = m
                if _last_lin is not None:
                    nn.init.zeros_(_last_lin.weight)
                    if _last_lin.bias is not None:
                        nn.init.zeros_(_last_lin.bias)
            else:
                self.alpha_pair = None
                self.alpha_geom = None
            # 48C1e: aux-logit normalization (independent of additive fusion flag).
            if cfg.normalize_aux_logits:
                self.bn_pair_aux = nn.BatchNorm1d(1, affine=False)
                self.bn_geom_aux = nn.BatchNorm1d(1, affine=False)
            else:
                self.bn_pair_aux = None
                self.bn_geom_aux = None
            # 48C2a: dedicated orientation branch (added post-hoc; independent
            # of pair and geom branches). Kept small and closed-form.
            if cfg.use_orient_branch:
                self.orient_mlp = nn.Sequential(
                    nn.Linear(cfg.num_orient_stats, cfg.orient_mlp_hidden),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.orient_mlp_hidden, cfg.orient_dim),
                )
                self.h_orient_aux = nn.Sequential(
                    nn.Linear(cfg.orient_dim, cfg.cls_hidden), nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.cls_hidden, 1),
                )
            else:
                self.orient_mlp = None
                self.h_orient_aux = None
        else:
            self.geom_input_mlp = None
            self.geom_set_blocks = None
            self.geom_pma = None
            self.h_pair_aux = None
            self.h_geom_aux = None
            self.h_fusion = None
            self.alpha_pair = None
            self.alpha_geom = None
            self.bn_pair_aux = None
            self.bn_geom_aux = None
            self.orient_mlp = None
            self.h_orient_aux = None

    def forward(
        self,
        candidate_patches: torch.Tensor,   # (B, S, N_nc, K, W, C)
        candidate_features: torch.Tensor,  # (B, S, N_nc, K, F)
        candidate_mask: torch.Tensor,      # (B, S, N_nc, K) bool
        nc_region_mask: torch.Tensor,      # (B, S, N_nc)   bool
        site_mask: Optional[torch.Tensor] = None,  # (B, S) bool; default all True
    ) -> dict:
        """Returns dict:
          logit:             (B,)
          site_repr:         (B, S, D)          per-site token (post set transformer)
          cand_attn:         (B, S, N_nc, K)    candidate attention weights
          cand_raw:          (B, S, N_nc, K)    pre-softmax candidate scores (for aux loss)
          nc_attn:           (B, S, N_nc)       NC attention weights
        """
        B, S, N_nc, K, W, C = candidate_patches.shape
        cfg = self.cfg
        assert C == cfg.patch_channels and W == cfg.patch_width
        assert N_nc == cfg.num_nc
        assert K == cfg.num_candidates

        if site_mask is None:
            site_mask = torch.ones((B, S), dtype=torch.bool, device=candidate_patches.device)

        # --- 1) per-candidate encoder ---------------------------------
        # patches: (B, S, N, K, W, C) -> flatten to (B*S*N*K, W, C) for CNN
        z_struct, z_align, z_pos = self.encoder(candidate_patches, candidate_features)
        # z_*: shape (B, S, N, K, <dim>)
        cand_tok = torch.cat([z_struct, z_align, z_pos], dim=-1)  # (B,S,N,K,D=128)

        # --- 2) candidate MIL: K candidates -> 1 NC token --------------
        # Flatten batch of (B*S*N, K, D). Mask: (B*S*N, K).
        cand_flat = cand_tok.reshape(B * S * N_nc, K, cfg.site_dim)
        cand_m_flat = candidate_mask.reshape(B * S * N_nc, K)
        nc_tok_flat, cand_attn_flat, cand_raw_flat = self.cand_mil(cand_flat, cand_m_flat)
        # (B*S*N, D)
        nc_tok = nc_tok_flat.reshape(B, S, N_nc, cfg.site_dim)
        cand_attn = cand_attn_flat.reshape(B, S, N_nc, K)
        cand_raw = cand_raw_flat.reshape(B, S, N_nc, K)

        # V6 cognate-pairing branch — applied to nc_tok BEFORE NC MIL.
        #   q_nc: (B, S, N_nc) — scalar pairing evidence per NC, for contrastive loss
        #   nc_tok gets a β-scaled correction from pair_fuse(q_nc)
        q_nc = None
        if self.pair_head is not None:
            q_nc = self.pair_head(nc_tok).squeeze(-1)     # (B, S, N_nc)
            # cast β to nc_tok dtype for autocast compatibility
            delta_nc = self.pair_fuse(q_nc.unsqueeze(-1).to(nc_tok.dtype))  # (B, S, N_nc, D)
            nc_tok = nc_tok + self.pair_beta.to(nc_tok.dtype) * delta_nc

        # --- 3) NC MIL: N_nc NC tokens -> 1 site token ----------------
        nc_flat = nc_tok.reshape(B * S, N_nc, cfg.site_dim)
        nc_m_flat = nc_region_mask.reshape(B * S, N_nc)
        site_tok_flat, nc_attn_flat, _ = self.nc_mil(nc_flat, nc_m_flat)
        site_tok = site_tok_flat.reshape(B, S, cfg.site_dim)
        nc_attn = nc_attn_flat.reshape(B, S, N_nc)

        # Save pre-SetTransformer site tokens for stagewise diagnostics.
        site_tok_pre_set = site_tok

        # --- 4) tnp-level Set Transformer -----------------------------
        for blk in self.set_blocks:
            site_tok = blk(site_tok, site_mask)

        pooled = self.pma(site_tok, site_mask)               # (B, pma_num_seeds, D)
        pooled_flat = pooled.reshape(B, cfg.pma_num_seeds * cfg.site_dim)

        # Compute dispersion φ once (mode-independent, always detached).
        disp_phi = None
        if self.disp_mode is not None:
            with torch.no_grad():
                disp_phi = _compute_dispersion_features(
                    cand_raw.detach(),
                    candidate_features.detach(),
                    candidate_mask,
                    nc_attn.detach(),
                )

        # V4 base logit — always computed exactly as V4 did.
        base_logit = self.classifier(pooled_flat).squeeze(-1)   # (B,)

        disp_delta = None
        alpha_or_beta = None
        if self.disp_mode == "scalar":
            # V5.1: logit = base_logit + α * δ(φ)
            disp_delta = self.disp_head(disp_phi.to(pooled_flat.dtype)).squeeze(-1)
            alpha_or_beta = self.disp_alpha
            logit = base_logit + self.disp_alpha * disp_delta
        elif self.disp_mode == "hidden_residual":
            # V5.2: hidden-level residual fusion.
            #   h_V4 = classifier[0..2](pooled_flat)    # Linear -> GELU -> Dropout
            #   d    = disp_encoder(φ)                  # (B, disp_hidden)
            #   Δh   = fusion_mlp([h_V4; d])            # (B, cls_hidden)
            #   h'   = h_V4 + β * Δh                    # (B, cls_hidden)
            #   logit = classifier[3](h')               # (B, 1) using the V4 output layer
            h_v4 = self.classifier[0](pooled_flat)      # Linear(128, 64)
            h_v4 = self.classifier[1](h_v4)             # GELU
            h_v4 = self.classifier[2](h_v4)             # Dropout
            d = self.disp_encoder(disp_phi.to(h_v4.dtype))
            fused_in = torch.cat([h_v4, d], dim=-1)
            delta_h = self.fusion_mlp(fused_in)
            h_prime = h_v4 + self.disp_beta * delta_h
            logit = self.classifier[3](h_prime).squeeze(-1)
            alpha_or_beta = self.disp_beta
        else:
            logit = base_logit

        # 48C1a: geometry bypass — additive logit correction from bag summary.
        s_geom = None
        geom_delta = None
        if self.geom_head is not None:
            with torch.no_grad():
                s_geom = _compute_geom_summary(
                    candidate_features, candidate_mask, nc_region_mask, site_mask
                )
            geom_delta = self.geom_head(s_geom.to(logit.dtype)).squeeze(-1)   # (B,)
            logit = logit + geom_delta

        # 48C1b: two-branch disentangled evidence — replaces the shared classifier
        # with a per-branch pair of auxiliary heads plus a fusion head.
        E_pair = None
        E_geom = None
        s_pair_aux = None
        s_geom_aux = None
        s_orient_aux = None
        if self.h_fusion is not None:
            E_pair = pooled_flat                                        # (B, D_pair)
            # Per-site geometry input: (orient_top1, flank_start_top1, L_top1).
            with torch.no_grad():
                valid = (
                    candidate_mask
                    & nc_region_mask.unsqueeze(-1)
                    & site_mask.unsqueeze(-1).unsqueeze(-1)
                )                                                        # (B, S, N, K)
                matches = candidate_features[..., 3].float()
                matches_flat = matches.masked_fill(~valid, float("-inf")).reshape(B, S, N_nc * K)
                top1 = matches_flat.argmax(dim=-1)                       # (B, S)
                orient_top1 = candidate_features[..., 0].reshape(B, S, N_nc * K).gather(
                    -1, top1.unsqueeze(-1)).squeeze(-1)
                L_top1 = candidate_features[..., 2].reshape(B, S, N_nc * K).gather(
                    -1, top1.unsqueeze(-1)).squeeze(-1)
                flank_top1 = candidate_features[..., 6].reshape(B, S, N_nc * K).gather(
                    -1, top1.unsqueeze(-1)).squeeze(-1)
                g_per_site = torch.stack([orient_top1, flank_top1, L_top1], dim=-1)  # (B, S, 3)
                site_has_valid = valid.any(dim=(2, 3)) & site_mask
                g_per_site = g_per_site * site_has_valid.unsqueeze(-1).float()
            z_geom = self.geom_input_mlp(g_per_site.to(pooled_flat.dtype))    # (B, S, D_geom)
            for blk in self.geom_set_blocks:
                z_geom = blk(z_geom, site_mask)
            E_set = self.geom_pma(z_geom, site_mask).reshape(B, cfg.geom_dim)  # (B, D_set)

            # 48C1c: hybrid geom = [E_set ; S_explicit].
            if cfg.use_explicit_geom_stats:
                with torch.no_grad():
                    S_explicit = _compute_geom_stats_explicit(
                        candidate_features, candidate_mask, nc_region_mask, site_mask
                    )
                E_geom = torch.cat([E_set, S_explicit.to(E_set.dtype)], dim=-1)
            else:
                E_geom = E_set

            s_pair_aux = self.h_pair_aux(E_pair).squeeze(-1)             # (B,)
            s_geom_aux = self.h_geom_aux(E_geom).squeeze(-1)             # (B,)

            # 48C2a: orientation branch (independent of pair/geom; does not
            # participate in fusion — used only for the aux loss and post-hoc
            # calibrated fusion in 48C2b).
            s_orient_aux = None
            if self.h_orient_aux is not None:
                with torch.no_grad():
                    S_orient = _compute_orient_stats(
                        candidate_features, candidate_mask,
                        nc_region_mask, site_mask
                    )
                E_orient = self.orient_mlp(S_orient.to(E_pair.dtype))
                s_orient_aux = self.h_orient_aux(E_orient).squeeze(-1)
            r_fusion = self.h_fusion(torch.cat([E_pair, E_geom], dim=-1)).squeeze(-1)  # (B,)
            if cfg.use_and_fusion:
                # 48C1f: AND-fusion via product of sigmoids.
                #   p_final = σ(s_pair) · σ(s_geom).
                #   For AUROC ranking we use log(p_final) = logσ(s_pair) + logσ(s_geom)
                #   which is monotonic in p_final. Loss computation uses probabilities
                #   directly via v1_and_fusion_loss.
                log_p_final = F.logsigmoid(s_pair_aux) + F.logsigmoid(s_geom_aux)
                logit = log_p_final                                       # ranking-equivalent
            elif cfg.use_additive_fusion:
                # 48C1d: additive fusion — both aux logits contribute directly.
                # 48C1e: optionally BatchNorm each aux logit first (unit-variance,
                # so α=β=1 gives balanced contributions).
                if self.bn_pair_aux is not None:
                    s_pair_for_fuse = self.bn_pair_aux(s_pair_aux.unsqueeze(-1)).squeeze(-1)
                    s_geom_for_fuse = self.bn_geom_aux(s_geom_aux.unsqueeze(-1)).squeeze(-1)
                else:
                    s_pair_for_fuse = s_pair_aux
                    s_geom_for_fuse = s_geom_aux
                # At init α=β=1 and r_fusion≈0 (zero-init), so logit≈s_pair+s_geom.
                a = self.alpha_pair.to(r_fusion.dtype)
                b = self.alpha_geom.to(r_fusion.dtype)
                logit = a * s_pair_for_fuse + b * s_geom_for_fuse + r_fusion
            else:
                logit = r_fusion
            base_logit = logit                                            # override for downstream logging

        return {
            "logit": logit,
            "base_logit": base_logit,        # V4 path only (for debugging)
            "s_geom": s_geom,                # (B, 6) or None  (48C1a bypass features)
            "geom_delta": geom_delta,        # (B,) or None
            "E_pair": E_pair,                # (B, D_pair) or None
            "E_geom": E_geom,                # (B, D_geom) or None
            "s_pair_aux": s_pair_aux,        # (B,) or None (48C1b pair aux logit)
            "s_geom_aux": s_geom_aux,        # (B,) or None (48C1b geom aux logit)
            "s_orient_aux": s_orient_aux,    # (B,) or None (48C2a orient aux logit)
            "disp_delta": disp_delta,        # (B,) or None  (scalar mode)
            "disp_alpha_or_beta": alpha_or_beta,  # (1,) or None
            "site_repr": site_tok,           # post-SetTransformer
            "site_repr_pre_set": site_tok_pre_set,  # pre-SetTransformer (for stagewise diagnostic)
            "cand_attn": cand_attn,
            "cand_raw": cand_raw,
            "nc_attn": nc_attn,
            "disp_phi": disp_phi,            # (B, 6) or None
            "q_nc": q_nc,                    # (B, S, N_nc) — V6 pairing evidence per NC, or None
            "pair_beta": self.pair_beta,     # (1,) or None — for logging
        }


# ---------------------------------------------------------------- #
#  Loss
# ---------------------------------------------------------------- #

def v1_loss(
    out: dict,
    is_positive: torch.Tensor,        # (B,) bool
    true_slot_idx: torch.Tensor,      # (B, S) int; -1 where GT unknown
    active_nc_index: Optional[torch.Tensor] = None,  # (B, S) int; -1 if none
    aux_lambda: float = 0.1,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    """Compose main BCE loss + auxiliary candidate-localization loss.

    aux loss: CE between the candidate softmax at the labeled active-NC
    slot vs true_slot_idx. Only sites with true_slot_idx >= 0 contribute.
    Auxiliary NC selection isn't included here (we let the MIL figure it
    out); we could add it later.

    If active_nc_index is None, defaults to slot 0 (the standard active
    slot for this dataset when a guide exists). This works because when
    is_positive is False (or true_slot_idx=-1) the aux loss is masked out.
    """
    B, S = true_slot_idx.shape
    device = out["logit"].device
    y = is_positive.float().to(device)
    bce = F.binary_cross_entropy_with_logits(
        out["logit"], y, pos_weight=pos_weight
    )

    # Candidate aux loss
    cand_raw = out["cand_raw"]        # (B, S, N_nc, K)
    if active_nc_index is None:
        # Default to slot 0. Positive sites always have their guide at
        # `labels.active_noncoding_index`; use `nc_attn` picking would be
        # circular for supervision so we pass the true active slot here.
        active_nc_index = torch.zeros(B, S, dtype=torch.long, device=device)
    else:
        active_nc_index = active_nc_index.to(device).long()

    # Pick the K-way logit vector at the active NC slot for each site.
    N_nc = cand_raw.size(2)
    K = cand_raw.size(3)
    idx = active_nc_index.clamp(min=0).unsqueeze(-1).unsqueeze(-1).expand(B, S, 1, K)
    cand_logits_active = cand_raw.gather(2, idx).squeeze(2)  # (B, S, K)

    # Mask: only sites with a valid true_slot_idx count.
    aux_mask = (true_slot_idx >= 0).to(device)
    if aux_mask.any():
        flat_logits = cand_logits_active[aux_mask]            # (M, K)
        flat_target = true_slot_idx[aux_mask].to(device).long()
        aux_ce = F.cross_entropy(flat_logits, flat_target)
    else:
        aux_ce = torch.zeros((), device=device)

    total = bce + aux_lambda * aux_ce
    return {"total": total, "bce": bce, "aux": aux_ce}


def v1_multi_branch_loss(
    out: dict,
    is_positive: torch.Tensor,        # (B,) bool
    pair_supervised: torch.Tensor,    # (B,) bool — sample supervises the pair aux head
    geom_supervised: torch.Tensor,    # (B,) bool — sample supervises the geom aux head
    lambda_pair: float = 0.5,
    lambda_geom: float = 0.5,
    pos_weight: Optional[torch.Tensor] = None,
) -> dict:
    """Loss for 48C1b two-branch model.

    L = BCE(logit_final, y) + λ_p · masked_mean(BCE(s_pair_aux, y), pair_supervised)
                              + λ_g · masked_mean(BCE(s_geom_aux, y), geom_supervised)
    """
    y = is_positive.float()
    device = y.device
    logit = out["logit"]
    L_final = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)

    def _masked(logits, mask):
        if not mask.any():
            return torch.zeros((), device=device)
        w = mask.float()
        # elementwise BCE, then reduce with mask
        per = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight, reduction="none")
        return (per * w).sum() / w.sum().clamp(min=1.0)

    s_pair = out.get("s_pair_aux")
    s_geom = out.get("s_geom_aux")
    if s_pair is None or s_geom is None:
        raise ValueError("v1_multi_branch_loss requires model outputs 's_pair_aux' and 's_geom_aux'; "
                          "did you set V1Config.use_multi_branch=True?")
    L_pair = _masked(s_pair, pair_supervised)
    L_geom = _masked(s_geom, geom_supervised)

    total = L_final + lambda_pair * L_pair + lambda_geom * L_geom
    return {
        "total": total,
        "bce": L_final,
        "pair_aux": L_pair,
        "geom_aux": L_geom,
        "aux": torch.zeros((), device=device),  # for logging compatibility with v1_loss
    }


def v1_and_fusion_loss(
    out: dict,
    y_pair: torch.Tensor,        # (B,) float in [0,1] — pair-validity target per sample
    y_geom: torch.Tensor,        # (B,) float in [0,1] — geom-validity target per sample
    lambda_pair: float = 1.0,
    lambda_geom: float = 1.0,
    pos_weight_pair: Optional[torch.Tensor] = None,
    pos_weight_geom: Optional[torch.Tensor] = None,
) -> dict:
    """Property-supervised AND-fusion loss (48C1f).

    Semantics:
        y_pair = 1 iff RNA<->DNA pairing is intact for this sample
        y_geom = 1 iff bag geometry is intact for this sample
        y_final = y_pair AND y_geom (both must be intact for a POS)

    Loss:
        L = BCE(p_final, y_final)  +  λ_p·BCE(s_pair_aux, y_pair)  +  λ_g·BCE(s_geom_aux, y_geom)
    where p_final = σ(s_pair_aux) · σ(s_geom_aux).

    Each aux head is trained on ALL samples with property-specific labels — no
    profile masking, no unlabeled samples.
    """
    device = y_pair.device
    y_pair = y_pair.float()
    y_geom = y_geom.float()
    y_final = (y_pair * y_geom).clamp(0.0, 1.0)  # AND for binary labels

    s_pair = out["s_pair_aux"]
    s_geom = out["s_geom_aux"]

    # Aux losses (each trained on every sample with property-specific target).
    L_pair = F.binary_cross_entropy_with_logits(s_pair, y_pair, pos_weight=pos_weight_pair)
    L_geom = F.binary_cross_entropy_with_logits(s_geom, y_geom, pos_weight=pos_weight_geom)

    # Final loss via product-of-sigmoids AND. Numerically stable:
    #   log(p_final)    = logσ(s_pair) + logσ(s_geom)
    #   log(1-p_final)  = logsumexp([log(1-σ_p), log σ_p + log(1-σ_g)])
    log_sig_pair = F.logsigmoid(s_pair)
    log_sig_geom = F.logsigmoid(s_geom)
    log_1m_sig_pair = F.logsigmoid(-s_pair)
    log_1m_sig_geom = F.logsigmoid(-s_geom)
    log_p_final = log_sig_pair + log_sig_geom
    log_1m_p_final = torch.logaddexp(log_1m_sig_pair, log_sig_pair + log_1m_sig_geom)
    L_final = -(y_final * log_p_final + (1.0 - y_final) * log_1m_p_final).mean()

    total = L_final + lambda_pair * L_pair + lambda_geom * L_geom
    return {
        "total": total,
        "bce": L_final,
        "pair_aux": L_pair,
        "geom_aux": L_geom,
        "aux": torch.zeros((), device=device),
    }


def v1_paired_geom_loss(
    s_geom: torch.Tensor,        # (B,) — s_geom_aux for the whole batch
    n_profiles: int,             # number of profiles per parent (typically 4: POS, shuffle, length, wp)
    profile_ranking_pos: int,    # index of the profile that MUST be OUTRANKED by POS (e.g. wrong_position = 3)
    profile_invariance_idx: tuple[int, ...],  # profile indices that geom should be INVARIANT to (e.g. shuffle=1, length=2)
    margin: float = 0.1,
    lambda_inv: float = 1.0,
    y_geom: Optional[torch.Tensor] = None,   # (B,) float — per-sample geom-validity labels (48C1h-A)
    lambda_prop: float = 0.0,                # 48C1h-A: weight on absolute-level property BCE
) -> dict:
    """Paired geometry loss using counterfactual twins in the same batch.

    Batch layout (from PairedCounterfactualBatchSampler):
        [parent_0_POS, parent_0_shuf, parent_0_len, parent_0_wp,
         parent_1_POS, parent_1_shuf, parent_1_len, parent_1_wp, ...]

    Losses:
        L_rank = softplus(margin - s_g(POS) + s_g(wp))           (POS must outrank wp)
        L_inv  = mean_over_prof {(s_g(POS) - s_g(prof))^2}       (POS ≈ shuf, len)
        L_prop = BCE(s_g, y_geom)                                 (absolute-level anchor; 48C1h-A)
        L_g = L_rank + λ_inv · L_inv + λ_prop · L_prop
    """
    B = s_geom.size(0)
    assert B % n_profiles == 0, f"batch {B} not divisible by n_profiles {n_profiles}"
    K = B // n_profiles
    s_kg = s_geom.reshape(K, n_profiles)          # (K, P)
    s_pos = s_kg[:, 0]
    s_rank_neg = s_kg[:, profile_ranking_pos]      # (K,)

    L_rank = F.softplus(margin - s_pos + s_rank_neg).mean()

    if profile_invariance_idx:
        inv_terms = [(s_pos - s_kg[:, i]).pow(2).mean() for i in profile_invariance_idx]
        L_inv = torch.stack(inv_terms).mean()
    else:
        L_inv = torch.zeros((), device=s_geom.device)

    if lambda_prop > 0.0 and y_geom is not None:
        L_prop = F.binary_cross_entropy_with_logits(s_geom, y_geom.float())
    else:
        L_prop = torch.zeros((), device=s_geom.device)

    total = L_rank + lambda_inv * L_inv + lambda_prop * L_prop
    return {
        "total": total,
        "rank": L_rank,
        "inv": L_inv,
        "prop": L_prop,
        # keys expected by the trainer's running-mean log:
        "bce": total,             # placeholder — this IS the training objective
        "aux": torch.zeros((), device=s_geom.device),
    }


def v1_paired_orient_loss(
    s_orient: torch.Tensor,           # (B,) — s_orient_aux for the whole batch
    n_profiles: int,                  # typically 5: POS, shuf, len, wp, wo
    profile_ranking_pos: int,         # index that must be OUTRANKED by POS (typically wo=4)
    profile_invariance_idx: tuple[int, ...],  # indices that orient should be INVARIANT to (shuf=1, len=2, wp=3)
    margin: float = 0.1,
    lambda_inv: float = 1.0,
    y_orient: Optional[torch.Tensor] = None,
    lambda_prop: float = 0.0,
) -> dict:
    """Paired orientation loss. Same shape as v1_paired_geom_loss but for the
    orientation branch.

    Batch layout (5 profiles from PairedCounterfactualBatchSampler):
        [POS_i, shuf_i, len_i, wp_i, wo_i,  POS_j, shuf_j, ...]
    """
    B = s_orient.size(0)
    assert B % n_profiles == 0, f"batch {B} not divisible by n_profiles {n_profiles}"
    K = B // n_profiles
    s_kp = s_orient.reshape(K, n_profiles)         # (K, P)
    s_pos = s_kp[:, 0]
    s_rank_neg = s_kp[:, profile_ranking_pos]       # (K,)

    L_rank = F.softplus(margin - s_pos + s_rank_neg).mean()

    if profile_invariance_idx:
        inv_terms = [(s_pos - s_kp[:, i]).pow(2).mean() for i in profile_invariance_idx]
        L_inv = torch.stack(inv_terms).mean()
    else:
        L_inv = torch.zeros((), device=s_orient.device)

    if lambda_prop > 0.0 and y_orient is not None:
        L_prop = F.binary_cross_entropy_with_logits(s_orient, y_orient.float())
    else:
        L_prop = torch.zeros((), device=s_orient.device)

    total = L_rank + lambda_inv * L_inv + lambda_prop * L_prop
    return {
        "total": total,
        "rank": L_rank,
        "inv": L_inv,
        "prop": L_prop,
        "bce": total,
        "aux": torch.zeros((), device=s_orient.device),
    }


def pair_loss(
    q_nc_cognate: torch.Tensor,     # (B, S, N_nc) — model's pairing evidence, cognate flank
    q_nc_swap: torch.Tensor,        # (B, S, N_nc) — model's pairing evidence, swap flank
    active_nc_index: torch.Tensor,  # (B, S) int; picks which NC slot to score
    pair_mask: torch.Tensor,        # (B, S) bool; True where L_pair fires (guided sites)
    margin: float = 1.0,
) -> dict:
    """V6 cognate-pairing contrastive loss.

    Only fires on sites where pair_mask is True (guided sites in positive
    bags). For those sites:
        L_pair = max(0, margin - q(NC_i, F_i) + q(NC_i, F_j))

    where q(NC_i, F_*) is q_nc_* at the site's active_nc_index. This is a
    margin loss that pushes cognate pairing evidence above swap pairing
    evidence by at least `margin` logit units.

    Returns:
        {
          "pair":       margin-loss (mean over valid sites; 0 if none),
          "delta_pair": (mean q_cognate - mean q_swap over valid sites)
                        for logging (should be positive after training).
          "n_pair":     number of sites contributing,
        }
    """
    device = q_nc_cognate.device
    B, S, N_nc = q_nc_cognate.shape
    # Gather q at the active NC slot per site.
    idx = active_nc_index.long().clamp(min=0).unsqueeze(-1)     # (B, S, 1)
    q_cog = q_nc_cognate.gather(-1, idx).squeeze(-1)             # (B, S)
    q_swp = q_nc_swap.gather(-1, idx).squeeze(-1)                # (B, S)
    # Additional gate: active_nc_index must be >= 0.
    valid = pair_mask.to(device) & (active_nc_index.to(device) >= 0)
    n = valid.sum().clamp(min=1)
    margin_gap = (margin - q_cog + q_swp).clamp(min=0.0)
    l_pair = (margin_gap * valid.float()).sum() / n
    delta = ((q_cog - q_swp) * valid.float()).sum() / n
    return {"pair": l_pair, "delta_pair": delta, "n_pair": int(valid.sum().item())}
