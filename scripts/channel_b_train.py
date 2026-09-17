"""Channel B training runner — per-position ordinal regression on the v6r2
5-corpus mix, weighted MSE per spec §3.

Runs 3 separate configurations (user directive 2026-09-04):
  --split-mode main               : Tnp-hash split, random 80/10/10
  --split-mode n_sites_heldout    : train on n_sites ∈ {3,4,5,6}, val on {7,8}
  --split-mode gc_heldout         : train on gc ∈ [0.25,0.45], val on [0.55,0.65]

Corpus mix (all 5 v6r2 corpora, per spec §Data recipe + user 2026-09-04
correction on partial-being-necessary):
  pos50k       50000 bags, negative_mode=none, is_positive=True
  ctrl10k      10000 bags, negative_mode=none, is_positive=True (extra pos control)
  twin50k      50000 bags, negative_mode=twin, is_positive=False
  partial40k   40000 bags, negative_mode=partial, is_positive=False (n_planted ∈ 1..n-1)
  scat10k      10000 bags, negative_mode=scattered, is_positive=False

Loss: nc_len-weighted MSE on the (batch, nc_len_eff) per-position ordinal
target. Positives (n_planted_at_position > 0) weighted by nc_len to balance
the ~1:189 within-bag imbalance.

Checkpoint: val loss (cheap). C1-C6 gates run OFFLINE after the run
converges, using the checkpointed weights.

TWO TRAINING-TIME CAVEATS RECORDED FOR EVAL (user directive 2026-09-04):

  1. C2 dose-response slope is not directly comparable to 1.41. Spec's
     1.41 = ln(p/q) = ln(0.86/0.21) is per-site log-odds. Current head
     is MSE regression on integer n_planted_at_position, not K-1
     cumulative-link logits. When writing the C2 evaluator, pick a
     bag-level scalar collapse (e.g. E[n_planted] = sum over positions,
     or sigmoid-then-logit of some pooled probability) and USE THE SAME
     COLLAPSE in val_auroc_proxy above so checkpoint selection is
     self-consistent with the C2 measurement. Failure mode to avoid:
     "AUROC-best checkpoint has worst C2 slope" attribution ambiguity.
     If MSE regression turns out to give a slope that can't be made
     interpretable, revisit spec §3's option (b) cumulative-link head.

  2. pos_weight = nc_len (~190) is a single-threshold approximation.
     Position-level P(n≥1) ≈ 0.29, P(n≥8) ≈ 0.019. A single weight
     rebalances the mean; high-tier (k≥6) predictions may still collapse
     toward zero because their imbalance is 100× the mean. If first-
     run diagnostics show high-tier prediction collapse, the fix is
     per-threshold weights ∝ 1/P(n≥k), not more data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.channel_b.data import ChannelBDataset, bucket_collate_fn
from model.channel_b.model import ChannelBModel
from model.channel_b.constants import MAX_L, CHANNELS


BATCH = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/large_batch")
SHARD = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v5_gen/channel_a")

CORPORA = [
    ("pos50k",     "positives_v5_v6r2_pos50k.jsonl",     "mt_v6r2_pos50k"),
    ("twin50k",    "positives_v5_v6r2_twin50k.jsonl",    "mt_v6r2_twin50k"),
    ("partial40k", "positives_v5_v6r2_partial40k.jsonl", "mt_v6r2_partial40k"),
    ("ctrl10k",    "positives_v5_v6r2_ctrl10k.jsonl",    "mt_v6r2_ctrl10k"),
    ("scat10k",    "positives_v5_v6r2_scat10k.jsonl",    "mt_v6r2_scat10k"),
]


# ---------- splitting ---------------------------------------------------

def _hash_split(bag_id: str, salt: str) -> float:
    h = hashlib.md5(f"{salt}::{bag_id}".encode()).hexdigest()
    return int(h[:8], 16) / 2**32   # [0, 1)


def _bag_meta(ds: ChannelBDataset, idx: int) -> dict:
    """Cheap metadata lookup without folding — reads first site record only."""
    bag_id = ds._bag_index[idx][0]
    sites = ds._sites_by_bag[bag_id]
    labels = sites[0]["labels"]
    arch = labels.get("arch", {})
    return {
        "bag_id":    bag_id,
        "n_sites":   int(arch.get("n_sites", len(sites))),
        "gc":        float(arch.get("gc_target", 0.5)),
        "hom":       float(arch.get("nc_homology_rate", 1.0)),
    }


def build_split(ds: ChannelBDataset, split_mode: str, seed: int = 0
                    ) -> tuple[Subset, Subset]:
    """Return (train_ds, val_ds) as PyTorch Subsets. Bag-blocked split."""
    salt = f"channelb_{split_mode}_seed{seed}"
    train_idx, val_idx = [], []
    for i in range(len(ds)):
        m = _bag_meta(ds, i)
        if split_mode == "main":
            r = _hash_split(m["bag_id"], salt)
            (train_idx if r < 0.9 else val_idx).append(i)
        elif split_mode == "n_sites_heldout":
            (train_idx if m["n_sites"] <= 6 else val_idx).append(i)
        elif split_mode == "gc_heldout":
            if m["gc"] < 0.45:      train_idx.append(i)
            elif m["gc"] >= 0.55:   val_idx.append(i)
            # gc ∈ [0.45, 0.55) intentionally dropped — clean boundary
        else:
            raise ValueError(f"unknown split_mode {split_mode!r}")
    return Subset(ds, train_idx), Subset(ds, val_idx)


# ---------- loss --------------------------------------------------------

def weighted_mse_loss(pred: torch.Tensor, target: torch.Tensor,
                          pos_mask: torch.Tensor) -> torch.Tensor:
    """Per spec §3: MSE on (B, P) with positive-position weighting by
    nc_len (~190) to balance the ~1:189 within-bag imbalance.
    Real nc_len per bag = pos_mask.sum(dim=1)."""
    # (B, P) diff^2
    diff2 = (pred - target) ** 2
    # per-bag nc_len
    nc_len_bag = pos_mask.float().sum(dim=1, keepdim=True).clamp_min(1.0)  # (B, 1)
    # weight positive positions by nc_len, negative positions by 1
    is_pos = (target > 0).float()   # (B, P) — 1 where n_planted > 0
    weight = torch.where(is_pos.bool(), nc_len_bag.expand_as(target),
                              torch.ones_like(target))
    # mask + normalize by per-bag total-weight, then average across bags
    weighted = diff2 * weight * pos_mask.float()
    per_bag_total_weight = (weight * pos_mask.float()).sum(dim=1).clamp_min(1.0)  # (B,)
    per_bag_loss = weighted.sum(dim=1) / per_bag_total_weight
    return per_bag_loss.mean()


# ---------- eval metric -------------------------------------------------

def evaluate(model: ChannelBModel, loader: DataLoader, device: torch.device,
                 zero_ch: list[int] | None = None,
                 ) -> dict:
    """Returns val loss + per-bag AUROC on (score >= 5 threshold vs
    is_positive) — a cheap proxy for C1 without running LOO folds."""
    model.eval()
    from sklearn.metrics import roc_auc_score
    total_loss = 0.0
    n_batches = 0
    all_bag_scores = []
    all_bag_labels = []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        site_mask = batch["site_mask"].to(device)
        pos_mask = batch["pos_mask"].to(device)
        if zero_ch:
            x = x.clone(); x[..., zero_ch] = 0.0
        with torch.no_grad():
            pred = model(x, site_mask)   # (B, P)
            loss = weighted_mse_loss(pred, y, pos_mask)
        total_loss += float(loss.item())
        n_batches += 1
        # Bag-level score: max over positions (proxy for "any position looks
        # planted"). Bag label = is_positive from labels.
        bag_max_score = (pred * pos_mask.float() +
                             (-1e9) * (~pos_mask).float()).max(dim=1).values
        all_bag_scores.append(bag_max_score.cpu().numpy())
        all_bag_labels.append(batch["labels"].cpu().numpy())
    val_loss = total_loss / max(1, n_batches)
    scores = np.concatenate(all_bag_scores)
    labels = np.concatenate(all_bag_labels)
    if len(set(labels.tolist())) >= 2:
        auroc = float(roc_auc_score(labels, scores))
    else:
        auroc = float("nan")
    model.train()
    return {"val_loss": val_loss, "val_auroc_proxy": auroc, "n_val_bags": int(len(scores))}


# ---------- length-bucket sampler --------------------------------------

class LengthBucketSampler(torch.utils.data.Sampler):
    """Sort indices by nc_len then draw contiguous chunks of batch_size.
    Within an epoch the chunk order is shuffled but items in a chunk stay
    length-sorted so padding waste is small."""
    def __init__(self, subset: Subset, batch_size: int, seed: int = 0,
                    shuffle: bool = True):
        self.subset = subset
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        # For each subset item, look up nc_len via parent dataset
        parent: ChannelBDataset = subset.dataset
        pairs = [(local_i, parent._bag_index[global_i][2])
                    for local_i, global_i in enumerate(subset.indices)]
        pairs.sort(key=lambda p: p[1])
        self._sorted_local = [p[0] for p in pairs]
        self._epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self._epoch)
            self._epoch += 1
            # Shuffle chunks of batch_size
            n = len(self._sorted_local)
            n_chunks = (n + self.batch_size - 1) // self.batch_size
            chunk_order = torch.randperm(n_chunks, generator=g).tolist()
            out = []
            for c in chunk_order:
                out.extend(self._sorted_local[c * self.batch_size :
                                                    (c + 1) * self.batch_size])
            return iter(out)
        return iter(self._sorted_local)

    def __len__(self):
        return len(self._sorted_local)


# ---------- training main ----------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-mode", choices=("main", "n_sites_heldout", "gc_heldout"),
                     required=True)
    ap.add_argument("--run-name", type=str, required=True,
                     help="Output dir tag under checkpoints/channel_b/<run-name>/")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-blocks", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-bags-per-corpus", type=int, default=None,
                     help="Cap per corpus for a quick smoke run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cache-root", type=Path,
                     default=Path("/global/scratch/users/kh36969/DL_novel_guide_editor/"
                                    "channel_b_cache"),
                     help="Root dir for per-bag tensor cache. First epoch = fold + cache; "
                          "subsequent epochs = fp32 load (~10× speedup). "
                          "Cache is content-addressed by corpus tag + bag_id.")
    ap.add_argument("--eval-every", type=int, default=1)   # epochs
    ap.add_argument("--planted-m-train-mset", type=str, default="",
                     help="Comma-separated planted_m values to KEEP in TRAIN (all-site condition). "
                          "Empty = no filter. Only affects pos50k bags with arch.target_m_at_planted. "
                          "Example: '8,9,10' keeps bags where every site's planted_m ∈ {8,9,10}.")
    ap.add_argument("--planted-m-val-mset", type=str, default="",
                     help="Same as --planted-m-train-mset but applied to VAL (best.pt selection). "
                          "For heldout-m gate: set this to the same as train-mset to prevent "
                          "peeking at the held-out planted_m slice during model selection.")
    ap.add_argument("--random-subsample-n", type=int, default=0,
                     help="If >0, after loading and any planted-m filtering, randomly subsample "
                          "this many TRAIN bags total (per-corpus proportional). Used for "
                          "sample-size-matched controls in held-out-m gate.")
    ap.add_argument("--subsample-exclude-val-planted-m", type=str, default="",
                     help="Bags whose sites are ALL in this planted_m mset are EXCLUDED from the "
                          "random-subsample source pool. Used to prevent train-eval overlap with a "
                          "held-out planted-m validation slice. Example: '6,7'.")
    ap.add_argument("--subsample-seed", type=int, default=0,
                     help="Seed for --random-subsample-n. Fixed for reproducibility.")
    ap.add_argument("--save-every-epoch", action="store_true",
                     help="Save epoch_N.pt in addition to best.pt (for held-out-m gate audit).")
    ap.add_argument("--maxm-train-min", type=int, default=0,
                     help="If >0: keep POSITIVE train bags with max(m_at_planted across sites) >= N. "
                          "Negatives (no per-site m_at_planted) are kept unchanged. "
                          "Used for the 2026-09-13 max-aggregation held-out-m variant. "
                          "Interacts with --split-mode main (applied after main-split).")
    ap.add_argument("--maxm-val-eq", type=int, default=0,
                     help="If >0: keep POSITIVE val bags with max(m_at_planted across sites) == N. "
                          "Negatives are kept unchanged. Companion to --maxm-train-min for the "
                          "max-aggregation held-out-m variant.")
    ap.add_argument("--corpora-config", type=Path, default=None,
                     help="Optional JSON file overriding the hardcoded CORPORA list. "
                          "Format: {\"batch_root\": <path>, \"shard_root\": <path>, "
                          "\"corpora\": [[name, jsonl_relpath, shard_relpath], ...]}. "
                          "When set, replaces CORPORA + BATCH + SHARD for this run only "
                          "(v6r2 paths remain the default). Used for v7 training.")
    ap.add_argument("--zero-channels", type=str, default="",
                     help="Comma-separated channel indices to zero at input "
                          "(both train + val). Use to test 'model with only "
                          "subset X visible'. E.g. --zero-channels 4,5,6,7,8,"
                          "9,10,11,12,13,14 for m_max-only.")
    args = ap.parse_args()
    zero_ch = [int(x) for x in args.zero_channels.split(",") if x.strip()]
    if zero_ch:
        print(f"[chb-train] ZEROING input channels: {zero_ch} at every batch",
              flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[chb-train] device={device}  split={args.split_mode}  run={args.run_name}",
          flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1) load all 5 corpora, concat.
    # `--corpora-config` overrides the default v6r2 CORPORA + roots for v7
    # training. When absent, the module-level v6r2 CORPORA + BATCH + SHARD
    # are used unchanged (byte-compatible with the v6r2 training path).
    if args.corpora_config is not None:
        cfg = json.loads(args.corpora_config.read_text())
        batch_root = Path(cfg["batch_root"])
        shard_root = Path(cfg["shard_root"])
        corpora = [tuple(x) for x in cfg["corpora"]]
        print(f"[chb-train] --corpora-config: batch_root={batch_root}", flush=True)
        print(f"[chb-train] --corpora-config: shard_root={shard_root}", flush=True)
        print(f"[chb-train] --corpora-config: {len(corpora)} corpora", flush=True)
    else:
        batch_root = BATCH
        shard_root = SHARD
        corpora = CORPORA

    # Cache inventory (fail-loud tripwire for the cache-collision class
    # of bug — 2026-09-12 v7 incident). Print cache root + count pre-
    # existing .pt files per corpus subdir. Any non-zero count means
    # something is either being resumed intentionally or (much more
    # commonly) a stale cache from a different run is about to be
    # silently hit — even under the source-hash-in-path defense, print
    # so the operator sees it.
    print(f"[chb-train] cache root: {args.cache_root}", flush=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    for name, _jsonl, _shard in corpora:
        cdir = args.cache_root / name
        n_pre = 0
        if cdir.exists():
            n_pre = sum(1 for _ in cdir.rglob("*.pt"))
        marker = "PRE-EXISTING (check!)" if n_pre > 0 else "empty (fresh)"
        print(f"  cache/{name}: {n_pre} .pt files — {marker}", flush=True)

    print("[chb-train] loading corpora ...", flush=True)
    corpus_dsets = []
    for name, jsonl, shard in corpora:
        cache_dir = args.cache_root / name
        t0 = time.perf_counter()
        ds = ChannelBDataset(
            batch_root / jsonl, shard_root / shard,
            max_bags=args.max_bags_per_corpus,
            preload_mt=True,
            cache_dir=cache_dir,
        )
        print(f"    {name}: {len(ds)} bags  "
              f"(index+MT preload {time.perf_counter()-t0:.1f}s; "
              f"cache={cache_dir})", flush=True)
        corpus_dsets.append(ds)
    full = ConcatDataset(corpus_dsets)
    print(f"    total: {len(full)} bags", flush=True)

    # 2) split by the requested mode
    # We split each corpus with the same salt, then concat train/val
    print(f"[chb-train] splitting mode={args.split_mode} ...", flush=True)
    train_subs, val_subs = [], []
    for ds in corpus_dsets:
        tr, va = build_split(ds, args.split_mode, seed=args.seed)
        train_subs.append(tr); val_subs.append(va)
    train_full = ConcatDataset(train_subs)
    val_full = ConcatDataset(val_subs)
    print(f"    train: {len(train_full)}  val: {len(val_full)}", flush=True)

    # 2b) held-out-m gate helpers: filter TRAIN by planted-m mset, and/or
    # random-subsample TRAIN to a fixed size (with optional exclusion pool).
    train_m_mset = {int(x) for x in args.planted_m_train_mset.split(",") if x.strip()}
    exclude_val_mset = {int(x) for x in args.subsample_exclude_val_planted_m.split(",") if x.strip()}

    def _bag_all_site_planted_m(ds, local_i):
        """Return sorted tuple of planted_m across sites in a bag (dedup), or None if missing."""
        bag_id, _off, _nc_len, _label = ds._bag_index[local_i]
        sites = ds._sites_by_bag.get(bag_id, [])
        ms = []
        for s in sites:
            arch = s["labels"].get("arch", {})
            pm = arch.get("target_m_at_planted")
            if pm is None: pm = s["labels"].get("m_at_planted")
            if pm is not None:
                ms.append(int(pm))
        return tuple(sorted(set(ms))) if ms else None

    def _filter_indices_by_all_site_mset(ds, indices, mset):
        """Keep bags where EVERY site's planted_m ∈ mset.
        Bags without planted_m (negatives) are KEPT — filter only applies to positives."""
        kept = []
        for i in indices:
            ms = _bag_all_site_planted_m(ds, i)
            if ms is None:
                kept.append(i)   # negative bag — always keep
                continue
            if all(m in mset for m in ms):
                kept.append(i)
        return kept

    def _bag_max_m_at_planted(ds, local_i):
        """Return max(m_at_planted across sites) or None if no per-site m
        (typically a negative bag). Uses arch.target_m_at_planted first,
        then labels.m_at_planted — same field priority as
        _bag_all_site_planted_m to keep behavior consistent."""
        bag_id, _off, _nc_len, _label = ds._bag_index[local_i]
        sites = ds._sites_by_bag.get(bag_id, [])
        ms = []
        for s in sites:
            arch = s["labels"].get("arch", {})
            pm = arch.get("target_m_at_planted")
            if pm is None: pm = s["labels"].get("m_at_planted")
            if pm is not None:
                ms.append(int(pm))
        return max(ms) if ms else None

    def _filter_indices_by_max_m(ds, indices, cond):
        """cond(max_m: int) → bool. Bags with no per-site m (negatives) are
        KEPT unchanged. Bags with per-site m are kept iff cond(max_m) is True."""
        kept = []
        for i in indices:
            mx = _bag_max_m_at_planted(ds, i)
            if mx is None:
                kept.append(i)   # negative — always keep
                continue
            if cond(mx):
                kept.append(i)
        return kept

    if args.maxm_val_eq > 0:
        target = args.maxm_val_eq
        print(f"[chb-train] applying VAL max(m_at_planted) == {target} "
              f"filter (positives only; negs kept)", flush=True)
        from torch.utils.data import Subset
        new_val_subs = []
        for ds, va in zip(corpus_dsets, val_subs):
            idx = _filter_indices_by_max_m(ds, list(va.indices),
                                                     lambda mx, T=target: mx == T)
            new_val_subs.append(Subset(ds, idx))
        val_subs = new_val_subs
        val_full = ConcatDataset(val_subs)
        print(f"[chb-train] REVISED val after max-m=={target} filter: "
              f"{len(val_full)} bags", flush=True)

    if args.maxm_train_min > 0:
        thr = args.maxm_train_min
        print(f"[chb-train] applying TRAIN max(m_at_planted) >= {thr} "
              f"filter (positives only; negs kept)", flush=True)
        from torch.utils.data import Subset
        new_train_subs = []
        for ds, tr in zip(corpus_dsets, train_subs):
            idx = _filter_indices_by_max_m(ds, list(tr.indices),
                                                     lambda mx, T=thr: mx >= T)
            new_train_subs.append(Subset(ds, idx))
        train_subs = new_train_subs
        train_full = ConcatDataset(train_subs)
        print(f"[chb-train] REVISED train after max-m>={thr} filter: "
              f"{len(train_full)} bags", flush=True)

    val_m_mset = {int(x) for x in args.planted_m_val_mset.split(",") if x.strip()}
    if val_m_mset:
        print(f"[chb-train] applying VAL planted-m filter: keep only bags with all-site m ∈ {val_m_mset}", flush=True)
        from torch.utils.data import Subset
        new_val_subs = []
        for ds, va in zip(corpus_dsets, val_subs):
            idx = _filter_indices_by_all_site_mset(ds, list(va.indices), val_m_mset)
            new_val_subs.append(Subset(ds, idx))
        val_subs = new_val_subs
        val_full = ConcatDataset(val_subs)
        print(f"[chb-train] REVISED val after m-filter: {len(val_full)} bags", flush=True)

    if train_m_mset or exclude_val_mset or args.random_subsample_n > 0:
        print(f"[chb-train] applying held-out-m filters: train_mset={train_m_mset}  "
              f"exclude_val_mset={exclude_val_mset}  subsample_n={args.random_subsample_n}", flush=True)
        new_train_subs = []
        for ds, tr in zip(corpus_dsets, train_subs):
            idx = list(tr.indices)
            n_before = len(idx)
            # (a) planted-m TRAIN filter
            if train_m_mset:
                idx = _filter_indices_by_all_site_mset(ds, idx, train_m_mset)
            # (b) exclude bags in val-planted-m mset from the pool BEFORE subsampling
            if exclude_val_mset:
                idx = [i for i in idx
                       if not (_bag_all_site_planted_m(ds, i) is not None and
                               all(m in exclude_val_mset for m in _bag_all_site_planted_m(ds, i)))]
            print(f"    corpus filter: {n_before} → {len(idx)} bags after m-filter+exclusion", flush=True)
            new_train_subs.append((ds, idx))
        # (c) random-subsample to N total bags, per-corpus proportional
        if args.random_subsample_n > 0:
            rng = np.random.default_rng(args.subsample_seed)
            totals = sum(len(idx) for _, idx in new_train_subs)
            if totals < args.random_subsample_n:
                print(f"[chb-train] WARNING: pool has {totals} bags < requested {args.random_subsample_n}; "
                      f"using ALL {totals} instead of random subsample.", flush=True)
            else:
                subsampled = []
                for ds, idx in new_train_subs:
                    share = int(round(args.random_subsample_n * len(idx) / totals))
                    share = min(share, len(idx))
                    picked = rng.choice(idx, size=share, replace=False)
                    subsampled.append((ds, sorted(int(i) for i in picked)))
                new_train_subs = subsampled
                print(f"    random-subsample: total pool {totals} → {sum(len(idx) for _, idx in new_train_subs)} bags "
                      f"(seed={args.subsample_seed})", flush=True)
        # Rebuild train_subs as Subset objects
        from torch.utils.data import Subset
        train_subs = [Subset(ds, idx) for ds, idx in new_train_subs]
        train_full = ConcatDataset(train_subs)
        print(f"[chb-train] REVISED train after held-out-m filters: {len(train_full)} bags", flush=True)

    # 3) loaders (no bucket sampler on ConcatDataset — use SequentialSampler
    #    over the concatenated indices; length bucketing across corpora would
    #    need a custom multi-corpus sampler which is out of scope for smoke)
    train_loader = DataLoader(
        train_full, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=bucket_collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_full, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=bucket_collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    # 3.5) first-bag preflight — dump one bag's tensor summary and cross-
    # check against its JSONL header BEFORE the first backward pass. The
    # 30-s check would have caught the 2026-09-12 cache-collision incident
    # (v7 training silently consumed v6r2 tensors under v7 labels for 160k
    # bags). See [[feedback-first-bag-tensor-dump]] in memory.
    _preflight_ds = corpus_dsets[0]
    _sample_item = _preflight_ds[0]
    _sample_bag_id = _preflight_ds._bag_index[0][0]
    _sample_header = _preflight_ds._sites_by_bag[_sample_bag_id][0]
    _sample_arch = _sample_header["labels"].get("arch", {})
    _jsonl_n_sites = int(_sample_arch.get("n_sites",
                                                 len(_preflight_ds._sites_by_bag[_sample_bag_id])))
    _jsonl_nc_len = int(_sample_header["labels"].get("ncrna_length",
                                                            len(_sample_header["inputs"]["noncoding_regions"][0])))
    _tensor_n_sites = int(_sample_item.site_mask.sum().item())
    _tensor_nc_len_eff = int(_sample_item.x.shape[1])
    # 2026-09-13: multi-region-aware formula. Loader concatenates N regions
    # with (N-1) N-spacers of length (MAX_L-1) each — see data.py's
    # concat_with_N_spacer policy. Effective concat length = sum(region lens)
    # + (N-1)*(MAX_L-1). Effective post-window length = concat - MAX_L + 1.
    _n_regions = int(_sample_header["labels"].get("num_noncoding_regions",
                                                          len(_sample_header["inputs"]["noncoding_regions"])))
    _spacer_bases = max(0, _n_regions - 1) * (MAX_L - 1)
    _expected_nc_len_eff = _jsonl_nc_len + _spacer_bases - MAX_L + 1
    print(f"[chb-train] preflight (first bag of {corpora[0][0]}):", flush=True)
    print(f"  bag_id={_sample_bag_id}  tensor.shape={tuple(_sample_item.x.shape)}  "
          f"site_mask.sum={_tensor_n_sites}  label={_sample_item.label}", flush=True)
    print(f"  JSONL says:  arch.n_sites={_jsonl_n_sites}  ncrna_length={_jsonl_nc_len}  "
          f"→ expected nc_len_eff={_expected_nc_len_eff}", flush=True)
    if _tensor_n_sites != _jsonl_n_sites:
        raise RuntimeError(
            f"[chb-train] preflight FAIL: tensor site_mask.sum={_tensor_n_sites} "
            f"vs JSONL arch.n_sites={_jsonl_n_sites} — dataset returned a bag whose "
            f"tensor doesn't match its own JSONL. Almost always a stale cache "
            f"(different data source under a colliding cache path). "
            f"See feedback_cache_content_key + feedback_first_bag_tensor_dump."
        )
    if _tensor_nc_len_eff != _expected_nc_len_eff:
        raise RuntimeError(
            f"[chb-train] preflight FAIL: tensor nc_len_eff={_tensor_nc_len_eff} "
            f"vs JSONL expected {_expected_nc_len_eff} (from ncrna_length "
            f"{_jsonl_nc_len} − MAX_L {MAX_L} + 1). Same class as above."
        )
    _p50 = {c: float(_sample_item.x[..., i].median().item())
              for i, c in enumerate(CHANNELS)}
    print(f"  per-channel p50: " + "  ".join(f"{c}={v:+.3f}" for c, v in _p50.items()),
          flush=True)
    print(f"  preflight OK", flush=True)
    del _preflight_ds, _sample_item, _sample_header, _sample_arch, _p50

    # 4) model + optim
    model = ChannelBModel(hidden=args.hidden, n_heads=args.n_heads,
                            n_blocks=args.n_blocks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.wd)
    total_steps = args.epochs * max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)

    ckpt_dir = Path(f"checkpoints/channel_b/{args.run_name}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "log.jsonl"

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[chb-train] model {n_params:,} params  "
          f"opt=AdamW(lr={args.lr}, wd={args.wd})  "
          f"steps_total={total_steps}", flush=True)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        n_batches = 0
        for step, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            site_mask = batch["site_mask"].to(device, non_blocking=True)
            pos_mask = batch["pos_mask"].to(device, non_blocking=True)
            if zero_ch:
                x = x.clone(); x[..., zero_ch] = 0.0
            pred = model(x, site_mask)
            loss = weighted_mse_loss(pred, y, pos_mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += float(loss.item()); n_batches += 1
            if step % 200 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} "
                      f"train_loss={running/max(1,n_batches):.5f} "
                      f"lr={sched.get_last_lr()[0]:.2e}", flush=True)

        train_loss = running / max(1, n_batches)
        elapsed = time.perf_counter() - t0

        # eval
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            val = evaluate(model, val_loader, device, zero_ch=zero_ch)
        else:
            val = {"val_loss": float("nan"), "val_auroc_proxy": float("nan"),
                   "n_val_bags": 0}

        log = {"epoch": epoch, "train_loss": train_loss,
               "elapsed_s": round(elapsed, 1), **val,
               "lr": sched.get_last_lr()[0]}
        with open(log_path, "a") as f:
            f.write(json.dumps(log) + "\n")
        print(f"[epoch {epoch}] train_loss={train_loss:.5f} "
              f"val_loss={val['val_loss']:.5f} val_auroc_proxy={val['val_auroc_proxy']:.4f} "
              f"elapsed={elapsed:.0f}s", flush=True)

        if val["val_loss"] < best_val_loss:
            best_val_loss = val["val_loss"]
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val["val_loss"],
                "val_auroc_proxy": val["val_auroc_proxy"],
                "args": vars(args),
            }, ckpt_dir / "best.pt")
            print(f"  saved best.pt @ val_loss={val['val_loss']:.5f}", flush=True)

        if args.save_every_epoch:
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_loss": val["val_loss"],
                "val_auroc_proxy": val["val_auroc_proxy"],
                "args": vars(args),
            }, ckpt_dir / f"epoch_{epoch}.pt")

    # final ckpt too
    torch.save({
        "model": model.state_dict(),
        "epoch": args.epochs - 1,
        "args": vars(args),
    }, ckpt_dir / "final.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
