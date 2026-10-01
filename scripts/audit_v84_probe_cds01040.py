"""Interpret v8.4's prediction for CDS01040 (or any --cds).

Reports per candidate bag:
  - top_p (nc position of peak) + bag_max
  - Full per-position score profile (top-10 peaks)
  - Per-site m_max value at top_p for each of L=9..14
  - Per-site flank_argmax at top_p (which flank position matches nc[top_p:top_p+L])
  - Structure channels (ch6-9) at top_p
  - flank_bg_identity (ch19) — bag-level
  - Actual nc sequence around top_p (the candidate "guide")
  - Actual flank sequence at each site's argmax (the matching flank window)
  - Zero-ablation: score drop per channel group
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.model import ChannelBModel
from model.channel_b.constants import CHANNELS, Ls, MAX_L


ADAPTED_JL = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                       "v84_evals/fna_ins_discovery/adapted.jsonl")
SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                  "v84_evals/fna_ins_discovery/adapted_shard")
CKPT = Path("checkpoints/channel_b/v84_main/best.pt")

CH_IDX = {c: i for i, c in enumerate(CHANNELS)}
CH_MMAX = [CH_IDX[f"m_max_L{L}"] for L in Ls]
CH_FDEV = [CH_IDX[f"flank_dev_L{L}"] for L in Ls]
CH_STRUCT = [CH_IDX[c] for c in ("dG_open_uL_pn", "H_pair_win",
                                       "cooperativity_win_pn", "E_span_win",
                                       "structure_valid")]
CH_FBG = [CH_IDX["flank_bg_identity"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cds", required=True)
    ap.add_argument("--top-peaks", type=int, default=5)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    hp = ck.get("args", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                            n_heads=hp.get("n_heads", 4),
                            n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    model.load_state_dict(ck["model"])

    # Find this bag in the dataset
    ds = ChannelBDataset(ADAPTED_JL, SHARD, preload_mt=True,
                            cache_dir=Path("/global/scratch/users/kh36969/"
                                            "DL_novel_guide_editor/v84_evals/"
                                            "fna_ins_discovery/cache"),
                            sort_by_nc_len=False)
    idx = None
    for i in range(len(ds)):
        if ds._bag_index[i][0] == args.cds:
            idx = i
            break
    if idx is None:
        print(f"ERROR: {args.cds} not in adapted dataset")
        return
    print(f"# probe target: {args.cds}  ds_idx={idx}\n")

    b = ds[idx]
    x = b.x                # (K, nc_len, C)
    sm = b.site_mask       # (K,)
    K_real = int(sm.sum().item())
    print(f"# tensor shape: x={tuple(x.shape)}  site_mask.sum={K_real}")

    # Full forward
    x_g = x.unsqueeze(0).to(device)
    sm_g = sm.unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_g, sm_g).squeeze(0).cpu().numpy()   # (nc_len,)
    print(f"# per-position pred shape={pred.shape}")

    top_p = int(pred.argmax())
    bag_max = float(pred.max())
    print(f"# TOP: pos={top_p}  score={bag_max:.4f}")

    # Top-N peaks (non-overlapping — window 12 bp)
    peaks = []
    scored = pred.copy()
    for _ in range(args.top_peaks):
        p = int(np.argmax(scored))
        peaks.append((p, float(pred[p])))
        # Suppress ±11 bp around this peak
        lo = max(0, p - 11); hi = min(len(pred), p + 12)
        scored[lo:hi] = -1e9
    print(f"\n## top-{args.top_peaks} peaks in per-position pred")
    for r, (p, v) in enumerate(peaks):
        print(f"  #{r+1}  pos={p:4d}  score={v:+.4f}")

    # Read raw records for this bag
    raw_recs = []
    with open(ADAPTED_JL) as f:
        for line in f:
            r = json.loads(line)
            if r["transposase_id"] == args.cds:
                raw_recs.append(r)
    K_raw = len(raw_recs)
    print(f"\n# raw records for {args.cds}: {K_raw}  "
          f"(loader used {K_real})")

    # Print nc sequence around top_p (candidate guide window)
    nc_regions = raw_recs[0]["inputs"]["noncoding_regions"]
    active = int(raw_recs[0]["labels"]["active_noncoding_index"])
    active_nc = nc_regions[active]
    spacer = "N" * (MAX_L - 1)
    concat = spacer.join(nc_regions)
    print(f"\n## nc concat (used for scoring): len={len(concat)}")
    for L in Ls:
        if top_p + L <= len(concat):
            print(f"  guide-candidate @ top_p (L={L:2d}):  "
                  f"{concat[top_p:top_p + L]!r}")
    print(f"  ±20 bp context: ...{concat[max(0,top_p-20):top_p]}"
          f"[{concat[top_p:top_p+11]}]{concat[top_p+11:min(len(concat),top_p+31)]}...")

    # Per-site m_max + flank_argmax at top_p, for each L
    print(f"\n## per-site channel readouts at top_p={top_p}")
    print(f"  L: " + " ".join(f"L{L:<2d}" for L in Ls))
    for s in range(K_real):
        m_vals = [float(x[s, top_p, CH_MMAX[i]].item()) for i in range(len(Ls))]
        fd_vals = [float(x[s, top_p, CH_FDEV[i]].item()) for i in range(len(Ls))]
        struct_valid = float(x[s, top_p, CH_IDX["structure_valid"]].item())
        # Also: total sum of m_max across L → indicates match strength
        print(f"  site {s} m_max (norm): " +
              " ".join(f"{v:.2f}" for v in m_vals) +
              f"    flank_dev: " + " ".join(f"{v:+.2f}" for v in fd_vals) +
              f"  struct_valid={struct_valid:.0f}")

    # Structure channels at top_p (bag-level, all sites see same value)
    print(f"\n  structure channels @ top_p:")
    for ch in ("dG_open_uL_pn", "H_pair_win", "cooperativity_win_pn",
                 "E_span_win", "structure_valid"):
        # take site 0 (structure is broadcast across sites)
        v = float(x[0, top_p, CH_IDX[ch]].item())
        print(f"    {ch:<25s}: {v:+.3f}")
    print(f"  flank_bg_identity: {float(x[0, top_p, CH_IDX['flank_bg_identity']].item()):+.3f}")

    # Per-site flank & flank_argmax
    print(f"\n## per-site flank sequences (120bp)")
    for i, r in enumerate(raw_recs[:K_real]):
        print(f"  site {i:2d}: {r['inputs']['flank']}")

    # Zero-ablation on this bag: how much does each channel group contribute?
    print(f"\n## zero-ablation @ top_p (per-channel-group contribution to bag_max)")
    baseline_max = bag_max
    for name, chs in [("m_max (L9-14)", CH_MMAX),
                        ("flank_dev (L9-14)", CH_FDEV),
                        ("structure (ch6-10)", CH_STRUCT),
                        ("flank_bg_identity (ch19)", CH_FBG),
                        ("orient_fwd+rc (ch17-18)", [CH_IDX["orient_fwd"], CH_IDX["orient_rc"]])]:
        x_abl = x.clone()
        x_abl[..., chs] = 0.0
        with torch.no_grad():
            pred_abl = model(x_abl.unsqueeze(0).to(device), sm_g).squeeze(0).cpu().numpy()
        new_max = float(pred_abl.max())
        d = new_max - baseline_max
        print(f"  zero {name:<28s}  new_bag_max={new_max:+.4f}  Δ={d:+.4f}")

    # Localized ablation: keep only top_p ±30 bp active, zero rest → verify top_p region drives
    print(f"\n## localization: zero-out score OUTSIDE top_p ± window, see if signal survives")
    for window in (5, 15, 30):
        x_abl = x.clone()
        lo, hi = max(0, top_p - window), min(x.shape[1], top_p + 11 + window)
        # Zero all positions except [lo, hi)
        mask_out = torch.ones(x.shape[1], dtype=torch.bool)
        mask_out[lo:hi] = False
        x_abl[:, mask_out, :] = 0.0
        with torch.no_grad():
            pred_abl = model(x_abl.unsqueeze(0).to(device), sm_g).squeeze(0).cpu().numpy()
        new_max = float(pred_abl.max())
        print(f"  keep only nc[{lo:4d}, {hi:4d})  (window ±{window:2d} bp)  → "
              f"new_bag_max={new_max:+.4f}  Δ={new_max - baseline_max:+.4f}")

    print(f"\n# probe complete")


if __name__ == "__main__":
    main()
