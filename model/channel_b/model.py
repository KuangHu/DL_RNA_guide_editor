"""Channel B model — cross-site-first attention over (site, position) grid.

Architecture (satisfies spec §1 principle-1 and principle-2):

  Input:                     (B, S, P, C=15)     already pre-normalized in data.py
  Cell-wise projection:      Linear(C → H)       → (B, S, P, H)
  ┌─ N blocks:
  │   Site attention (per position):
  │     For each p, MHA(Q,K,V) over sites 0..S-1, masked by site_mask
  │     → (B, S, P, H); permutation-equivariant on the S axis
  │   Position attention (per site):
  │     For each s, MHA(Q,K,V) over positions 0..P-1
  │     → (B, S, P, H); NOT masked because position axis is real (no padding
  │       within a bucketed batch)
  └─
  Pool over sites (masked mean): → (B, P, H)
  Per-position head:         Linear(H → H) → GELU → Linear(H → 1) → (B, P)

Output is per-position predicted `n_planted_at_position` (spec §3
ordinal regression target ∈ {0, ..., n_sites}); NO position pool. At
inference, a bag-level score is a downstream threshold on this
per-position vector — not baked into training.

Principle-2 enforcement:
  The site-axis attention treats S as a set. Given site permutation π,
  the output at π(s) equals what the unpermuted output was at s (for
  every position p). MHA is bit-exact permutation-equivariant when
  implemented on top of dot-product attention because both softmax and
  the value aggregation act uniformly on the sequence dimension.

Deploy contract:
  The model consumes only the 15-channel tensor from data.py (which is
  built from deploy-computable inputs; see whitelist in constants.py).
  No label information reaches this file. Model weights + fixed
  channel scales are the only train-time artefacts.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .constants import N_CHANNELS, MAX_N_SITES


class SiteAttentionBlock(nn.Module):
    """Multi-head attention over the SITE axis at each position (independent
    per position). Site padding respected via key_padding_mask."""

    def __init__(self, hidden: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, site_mask: torch.Tensor) -> torch.Tensor:
        # x: (B, S, P, H) — reshape so attention over S happens per-position
        B, S, P, H = x.shape
        # (B, S, P, H) -> (B, P, S, H) so the S axis is the attention seq dim
        y = x.permute(0, 2, 1, 3).contiguous()                # (B, P, S, H)
        y = y.view(B * P, S, H)                                # (B*P, S, H)
        # key_padding_mask: True where the key is PADDING (nn.MHA convention)
        # site_mask is True where REAL, so invert.
        pad_mask = ~site_mask                                  # (B, S)
        pad_mask_rep = pad_mask.unsqueeze(1).expand(B, P, S).reshape(B * P, S)
        # A tokenwise all-padding row can produce NaN under standard MHA; but
        # our loader guarantees ≥ 3 real sites per bag, and positions are all
        # real within a bucketed batch, so no fully-padded row exists.
        h, _ = self.attn(y, y, y, key_padding_mask=pad_mask_rep,
                             need_weights=False)
        y = self.ln1(y + h)
        y = self.ln2(y + self.ff(y))
        # back to (B, S, P, H)
        y = y.view(B, P, S, H).permute(0, 2, 1, 3).contiguous()
        return y


class PositionAttentionBlock(nn.Module):
    """Multi-head attention over the POSITION axis at each site (independent
    per site). No padding on the position axis inside a bucketed batch."""

    def __init__(self, hidden: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            hidden, n_heads, dropout=dropout, batch_first=True,
        )
        self.ln1 = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden),
            nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, P, H = x.shape
        y = x.view(B * S, P, H)
        h, _ = self.attn(y, y, y, need_weights=False)
        y = self.ln1(y + h)
        y = self.ln2(y + self.ff(y))
        return y.view(B, S, P, H)


class ChannelBModel(nn.Module):
    """Cross-site-first attention model. Bag-level output is P(positive)."""

    def __init__(
        self,
        hidden: int = 128,
        n_heads: int = 4,
        n_blocks: int = 3,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden = hidden
        self.input_proj = nn.Linear(N_CHANNELS, hidden)
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "site":     SiteAttentionBlock(hidden, n_heads, dropout),
                "position": PositionAttentionBlock(hidden, n_heads, dropout),
            }))
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        x: torch.Tensor,          # (B, S=MAX_N_SITES, P, C=N_CHANNELS)
        site_mask: torch.Tensor,  # (B, S) bool
    ) -> torch.Tensor:
        # 1) per-cell embedding
        h = self.input_proj(x)                             # (B, S, P, H)

        # 2) alternating site / position attention blocks
        for blk in self.blocks:
            h = blk["site"](h, site_mask)
            h = blk["position"](h)

        # 3) masked mean pool over sites first (per position)
        m = site_mask.float().unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
        h_site_pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        # h_site_pooled: (B, P, H)

        # 4) per-position head. NO position pool — output is a per-
        # position predicted n_planted_at_position (spec §3 ordinal
        # regression target). Bag-level scoring is a deployment-time
        # threshold on this vector.
        per_pos_score = self.head(h_site_pooled).squeeze(-1)   # (B, P)
        return per_pos_score
