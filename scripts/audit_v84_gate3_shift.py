"""V8.4 Gate 3 — C8 shift invariance standalone.

Takes 20 positive val bags, inserts 30 zero-position vectors at the
start of the nc axis (tensor-level shift), scores original vs shifted,
reports Δtop_p and |Δbag_max|.

Pass = Δtop_p == 30 AND |Δbag_max| < 0.1.
"""
from __future__ import annotations
import hashlib
import statistics
import sys
from pathlib import Path

import torch
from torch.utils.data import Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset
from model.channel_b.model import ChannelBModel


V84_BATCH = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                     "v84_generation/50k")
V84_SHARDS = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                      "v84_shards")
CACHE_ROOT = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                       "channel_b_cache_v84")
CKPT = Path("checkpoints/channel_b/v84_main/best.pt")
SALT = "channelb_main_seed0"


def hash_split(bag_id: str) -> float:
    h = hashlib.md5(f"{SALT}::{bag_id}".encode()).hexdigest()
    return int(h[:8], 16) / 2**32


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"# v84 Gate 3 standalone — C8 shift invariance  device={device}")

    ck = torch.load(CKPT, map_location=device, weights_only=False)
    hp = ck.get("args", {})
    model = ChannelBModel(hidden=hp.get("hidden", 128),
                            n_heads=hp.get("n_heads", 4),
                            n_blocks=hp.get("n_blocks", 3)).to(device).eval()
    model.load_state_dict(ck["model"])
    print(f"# ckpt: {CKPT}  epoch={ck.get('epoch','?')}")

    # Load positive corpus + val subset
    ds = ChannelBDataset(V84_BATCH / "v84_none.jsonl",
                            V84_SHARDS / "v84_none_shard",
                            preload_mt=True,
                            cache_dir=CACHE_ROOT / "v84_none",
                            sort_by_nc_len=False)
    val_idx = [i for i in range(len(ds))
                 if hash_split(ds._bag_index[i][0]) >= 0.9]
    val = Subset(ds, val_idx)
    print(f"# n_val_bags(positive) = {len(val)}")

    picked = [val[i] for i in range(min(20, len(val)))]
    print(f"# picked {len(picked)} positive bags\n")

    shift = 30
    diffs_bag_max = []
    diffs_top_p = []
    ok = 0
    for i, b in enumerate(picked):
        x = b.x.unsqueeze(0).to(device)
        sm = b.site_mask.unsqueeze(0).to(device)
        with torch.no_grad():
            y0 = model(x, sm).squeeze(0)
        y0_max = float(y0.max().item())
        top_p0 = int(y0.argmax().item())

        C = x.shape[-1]
        n_sites = x.shape[1]
        pad = torch.zeros(1, n_sites, shift, C, device=device, dtype=x.dtype)
        x_shift = torch.cat([pad, x], dim=2)
        with torch.no_grad():
            y1 = model(x_shift, sm).squeeze(0)
        y1_max = float(y1.max().item())
        top_p1 = int(y1.argmax().item())

        d_max = abs(y0_max - y1_max)
        d_top = top_p1 - top_p0
        diffs_bag_max.append(d_max)
        diffs_top_p.append(d_top)
        passed = (d_top == shift) and (d_max < 0.1)
        if passed:
            ok += 1
        print(f"  bag {i:2d}: top_p {top_p0}→{top_p1} (Δ {d_top:+d})  "
                f"bag_max {y0_max:.4f}→{y1_max:.4f} (|Δ| {d_max:.4f})  "
                f"{'PASS' if passed else 'FAIL'}")

    print(f"\n## summary")
    print(f"  Δtop_p:      expected {shift}  observed_range=[{min(diffs_top_p)}, {max(diffs_top_p)}]  "
            f"n_equals_30={sum(1 for d in diffs_top_p if d == shift)}/{len(picked)}")
    print(f"  |Δbag_max|:  max={max(diffs_bag_max):.4f}  "
            f"mean={statistics.mean(diffs_bag_max):.4f}  "
            f"p50={statistics.median(diffs_bag_max):.4f}")
    print(f"  PASS bags: {ok}/{len(picked)}  "
            f"(pass = Δtop_p == 30 AND |Δbag_max| < 0.1)")


if __name__ == "__main__":
    main()
