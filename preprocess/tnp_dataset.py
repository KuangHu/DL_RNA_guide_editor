"""Tnp-grouped dataset.

While `DLNovelGuideEditorDataset` yields one site per __getitem__,
`TnpGroupedDataset` yields all sites belonging to one transposase per
__getitem__. This matches the V1 model's tnp-level Set Transformer: we
need to compare sites of the SAME tnp against each other.

Each tnp in this dataset has exactly 50 sites (confirmed on the source
data). Optional `site_subsample_size` picks a random subset per epoch
for training (adds regularization + reduces per-batch memory).

Returned dict:
    'tnp_id'           : str
    'site_ids'         : list[str] len S
    'is_positive'      : bool
    'candidate_patches':  float32 (S, 3, K, W, C)
    'candidate_features': float32 (S, 3, K, F)
    'candidate_mask':     bool    (S, 3, K)
    'nc_region_mask':     bool    (S, 3)
    'true_slot_idx':      int32   (S,)   -- index of the ground-truth
                                            candidate in the 96-slot layout,
                                            or -1 if unknown (negatives or
                                            positives whose true candidate
                                            didn't survive top-K filtering)

For the collate: sites-per-tnp is fixed (50 or subsample_size), so a
simple torch.stack across tnps produces (B, S, ...) tensors.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np

from .candidates import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_ORIENTATIONS,
    PATCH_WIDTH_DEFAULT,
    TOP_K_PER_COMBO_DEFAULT,
)
from .site import (
    DEFAULT_NC_MAX,
    DEFAULT_NUM_NC_SLOTS,
    StructureCache,
    preprocess_site,
)


def _slot_for_ground_truth(
    orient: str,
    L: int,
    L_min: int,
    L_max: int,
    orientations: Sequence[str],
    top_k_per_combo: int,
) -> int:
    """Return the FIRST slot in the (orient, L) block for the ground-truth
    candidate. The actual index within the block depends on the top-K
    sorting, so we return the block start; the caller must scan the block
    to find the exact slot whose (nc_start, flank_start) matches.
    """
    try:
        orient_i = list(orientations).index(orient)
    except ValueError:
        return -1
    n_L = L_max - L_min + 1
    if not (L_min <= L <= L_max):
        return -1
    return orient_i * n_L * top_k_per_combo + (L - L_min) * top_k_per_combo


def _find_ground_truth_slot(
    rec: dict,
    candidate_features: np.ndarray,   # (3, K, F)
    candidate_mask: np.ndarray,       # (3, K)
    flank_len: int,
    L_min: int,
    L_max: int,
    orientations: Sequence[str],
    top_k_per_combo: int,
) -> int:
    """Locate the ground-truth candidate index (0..K-1) in the 3D layout
    at the active NC slot. Returns the flat slot index or -1 if not found
    (negative, or true candidate didn't survive top-K).
    """
    lbl = rec["labels"]
    if not lbl.get("is_positive"):
        return -1
    slot_nc = lbl.get("active_noncoding_index", -1)
    if slot_nc < 0:
        return -1
    # Real (unlabelled) records have is_positive=True but no site-level
    # supervision — return -1 (no ground-truth slot). Downstream auxiliary
    # losses skip -1 entries, so this only affects training-time aux loss
    # (never used at inference).
    for k in ("match_orientation", "guide_length",
              "guide_span_in_active_noncoding", "target_position_in_flank"):
        if lbl.get(k) is None:
            return -1
    orient = "fwd" if lbl["match_orientation"] == "forward" else "rc"
    L = lbl["guide_length"]
    gs = lbl["guide_span_in_active_noncoding"][0]
    ts = lbl["target_position_in_flank"][0]

    block_start = _slot_for_ground_truth(
        orient, L, L_min, L_max, orientations, top_k_per_combo
    )
    if block_start < 0:
        return -1
    block_end = block_start + top_k_per_combo

    # Feature indices: FEATURE_NAMES order is fixed by preprocess.candidates.
    from .candidates import FEATURE_NAMES

    F_L = FEATURE_NAMES.index("L")
    F_flank_start = FEATURE_NAMES.index("flank_start_norm")
    F_nc_start = FEATURE_NAMES.index("nc_start_norm")

    feats_block = candidate_features[slot_nc, block_start:block_end]
    mask_block = candidate_mask[slot_nc, block_start:block_end]

    # nc_len is exact from the record — no need to invert nc_len_norm.
    nc_seq = rec["inputs"]["noncoding_regions"][slot_nc]
    nc_len = len(nc_seq)

    for i in range(top_k_per_combo):
        if not mask_block[i]:
            continue
        if int(feats_block[i, F_L]) != L:
            continue
        flank_start_reconstructed = int(round(float(feats_block[i, F_flank_start]) * flank_len))
        if flank_start_reconstructed != ts:
            continue
        nc_start_reconstructed = int(round(float(feats_block[i, F_nc_start]) * nc_len))
        if nc_start_reconstructed != gs:
            continue
        return block_start + i
    return -1


class TnpGroupedDataset:
    """Random-access dataset over tnps in one split.

    __init__ builds:
      - offset table over the jsonl (like DLNovelGuideEditorDataset)
      - map: tnp_id -> [site line-index, ...]  (in file order)
      - tnp_ids: list[str]                     (in stable sort order)

    __getitem__(idx) reads the S sites for tnp_ids[idx], runs preprocess_site
    on each, and stacks along a leading site axis. Optionally subsamples
    `site_subsample_size` sites uniformly at random (per-call reshuffle).
    """

    def __init__(
        self,
        split_path: str | Path,
        structure_cache: StructureCache,
        *,
        site_subsample_size: Optional[int] = None,
        rng_seed: int = 0,
        generate_swap: bool = False,   # V6: also produce swap-flank version per bag
        # forwarded to preprocess_site:
        nc_max: int = DEFAULT_NC_MAX,
        num_nc_slots: int = DEFAULT_NUM_NC_SLOTS,
        top_k_per_combo: int = TOP_K_PER_COMBO_DEFAULT,
        L_min: int = DEFAULT_L_MIN,
        L_max: int = DEFAULT_L_MAX,
        orientations: Sequence[str] = DEFAULT_ORIENTATIONS,
        patch_width: int = PATCH_WIDTH_DEFAULT,
    ):
        self.split_path = Path(split_path)
        self.structure_cache = structure_cache
        self.site_subsample_size = site_subsample_size
        self.generate_swap = generate_swap
        self._rng = np.random.default_rng(rng_seed)

        self.preprocess_kwargs = dict(
            nc_max=nc_max,
            num_nc_slots=num_nc_slots,
            top_k_per_combo=top_k_per_combo,
            L_min=L_min,
            L_max=L_max,
            orientations=tuple(orientations),
            patch_width=patch_width,
        )
        self._flank_len = 120  # dataset constant

        self._offsets, self._tnp_lines, self._tnp_is_positive = self._build_index()
        # Sort tnp_ids so positive and negative tnps are interleaved deterministically
        # (positives are "tnp_XXXXX", negatives "tnp_neg_XXXXX" — alphabetical would
        # place all positives first). Interleaving via a shuffled hash keeps eval
        # deterministic while giving stratified prefixes for --max-val-tnps.
        rng = np.random.default_rng(rng_seed)
        ids = sorted(self._tnp_lines.keys())
        rng.shuffle(ids)
        self.tnp_ids: list[str] = ids

    def _build_index(self):
        offsets: list[int] = []
        tnp_lines: dict[str, list[int]] = defaultdict(list)
        tnp_is_positive: dict[str, bool] = {}
        with open(self.split_path, "rb") as f:
            line_i = 0
            while True:
                off = f.tell()
                raw = f.readline()
                if not raw:
                    break
                offsets.append(off)
                rec = json.loads(raw)
                tnp = rec["transposase_id"]
                tnp_lines[tnp].append(line_i)
                if tnp not in tnp_is_positive:
                    tnp_is_positive[tnp] = bool(rec["labels"].get("is_positive", False))
                line_i += 1
        return np.asarray(offsets, dtype=np.int64), tnp_lines, tnp_is_positive

    def __len__(self) -> int:
        return len(self.tnp_ids)

    def is_positive(self, tnp_id: str) -> bool:
        return self._tnp_is_positive[tnp_id]

    def _read_record(self, line_i: int) -> dict:
        off = int(self._offsets[line_i])
        with open(self.split_path, "rb") as f:
            f.seek(off)
            return json.loads(f.readline())

    def __getitem__(self, idx: int) -> dict:
        tnp = self.tnp_ids[idx]
        site_lines = self._tnp_lines[tnp]
        # Site subsampling.
        if self.site_subsample_size is not None and self.site_subsample_size < len(site_lines):
            picked = self._rng.choice(
                len(site_lines), size=self.site_subsample_size, replace=False
            )
            site_lines = [site_lines[i] for i in sorted(picked)]

        # First pass: load records and preprocess ORIGINAL sites.
        records = [self._read_record(i) for i in site_lines]
        sites_pat = []
        sites_feat = []
        sites_mask = []
        sites_ncmask = []
        site_ids = []
        true_slots = []
        active_ncs = []
        site_class_guided = []
        for rec in records:
            out = preprocess_site(
                rec, structure_cache=self.structure_cache, **self.preprocess_kwargs
            )
            sites_pat.append(out["candidate_patches"])
            sites_feat.append(out["candidate_features"])
            sites_mask.append(out["candidate_mask"])
            sites_ncmask.append(out["nc_region_mask"])
            site_ids.append(out["site_id"])
            true_slots.append(
                _find_ground_truth_slot(
                    rec,
                    out["candidate_features"],
                    out["candidate_mask"],
                    flank_len=self._flank_len,
                    L_min=self.preprocess_kwargs["L_min"],
                    L_max=self.preprocess_kwargs["L_max"],
                    orientations=self.preprocess_kwargs["orientations"],
                    top_k_per_combo=self.preprocess_kwargs["top_k_per_combo"],
                )
            )
            active_ncs.append(int(rec["labels"].get("active_noncoding_index", -1)))
            # V6: guided-only pairing mask requires is_positive AND site_class == "guided".
            is_pos = bool(rec["labels"].get("is_positive"))
            sc = rec["labels"].get("site_class")
            site_class_guided.append(is_pos and (sc == "guided"))

        item = {
            "tnp_id": tnp,
            "site_ids": site_ids,
            "is_positive": self._tnp_is_positive[tnp],
            "candidate_patches":  np.stack(sites_pat, axis=0),
            "candidate_features": np.stack(sites_feat, axis=0),
            "candidate_mask":     np.stack(sites_mask, axis=0),
            "nc_region_mask":     np.stack(sites_ncmask, axis=0),
            "true_slot_idx":      np.asarray(true_slots, dtype=np.int32),
            "active_nc_index":    np.asarray(active_ncs, dtype=np.int32),
            "pair_mask":          np.asarray(site_class_guided, dtype=bool),   # V6 mask
        }

        # V6: if requested, produce a swap-flank version.
        # For each guided site (pair_mask=True), replace its flank with another
        # guided site's flank drawn from the SAME bag. Structure cache lookups are
        # unaffected (structure is over NC, not flank). Non-guided sites' swap =
        # their original (never used because pair_mask is False for them).
        if self.generate_swap:
            guided_idx = [i for i, g in enumerate(site_class_guided) if g]
            swap_pat, swap_feat, swap_mask, swap_ncmask = [], [], [], []
            final_pair_mask = list(site_class_guided)
            for i, rec in enumerate(records):
                if not site_class_guided[i] or len(guided_idx) < 2:
                    # Non-guided site — swap = original (never used because pair_mask is False).
                    swap_pat.append(sites_pat[i])
                    swap_feat.append(sites_feat[i])
                    swap_mask.append(sites_mask[i])
                    swap_ncmask.append(sites_ncmask[i])
                    if len(guided_idx) < 2:
                        final_pair_mask[i] = False   # bag has < 2 guided sites -> no valid swap
                    continue
                # Pick a random OTHER guided site to donate its flank.
                candidates = [j for j in guided_idx if j != i]
                donor_j = int(self._rng.choice(len(candidates)))
                donor_idx = candidates[donor_j]
                donor_flank = records[donor_idx]["inputs"]["flank"]
                swap_rec = {
                    **rec,
                    "inputs": {**rec["inputs"], "flank": donor_flank},
                }
                out_s = preprocess_site(
                    swap_rec, structure_cache=self.structure_cache, **self.preprocess_kwargs
                )
                swap_pat.append(out_s["candidate_patches"])
                swap_feat.append(out_s["candidate_features"])
                swap_mask.append(out_s["candidate_mask"])
                swap_ncmask.append(out_s["nc_region_mask"])
            item["candidate_patches_swap"]  = np.stack(swap_pat, axis=0)
            item["candidate_features_swap"] = np.stack(swap_feat, axis=0)
            item["candidate_mask_swap"]     = np.stack(swap_mask, axis=0)
            item["nc_region_mask_swap"]     = np.stack(swap_ncmask, axis=0)
            item["pair_mask"]               = np.asarray(final_pair_mask, dtype=bool)

        return item


def collate_tnp_batch(items: list[dict], *, to_torch: bool = True) -> dict:
    """Stack B tnps into batched tensors.

    Requires all items have the same number of sites (call the dataset with
    a fixed site_subsample_size or all-50 mode).
    """
    B = len(items)
    if B == 0:
        raise ValueError("empty items")
    S_ref = items[0]["candidate_patches"].shape[0]
    for it in items:
        if it["candidate_patches"].shape[0] != S_ref:
            raise ValueError(
                f"inconsistent sites-per-tnp: {S_ref} vs {it['candidate_patches'].shape[0]}. "
                "Use a fixed site_subsample_size."
            )
    out = {
        "candidate_patches":  np.stack([it["candidate_patches"] for it in items], axis=0),
        "candidate_features": np.stack([it["candidate_features"] for it in items], axis=0),
        "candidate_mask":     np.stack([it["candidate_mask"] for it in items], axis=0),
        "nc_region_mask":     np.stack([it["nc_region_mask"] for it in items], axis=0),
        "true_slot_idx":      np.stack([it["true_slot_idx"] for it in items], axis=0),
        "active_nc_index":    np.stack([it["active_nc_index"] for it in items], axis=0),
        "is_positive":        np.asarray([it["is_positive"] for it in items], dtype=bool),
        "tnp_id":             [it["tnp_id"] for it in items],
        "site_ids":           [it["site_ids"] for it in items],
    }
    # V6: pair_mask is always present (dataset supplies it); swap fields only when
    # generate_swap=True on the dataset.
    if "pair_mask" in items[0]:
        out["pair_mask"] = np.stack([it["pair_mask"] for it in items], axis=0)
    has_swap = "candidate_patches_swap" in items[0]
    if has_swap:
        for k in ("candidate_patches_swap", "candidate_features_swap",
                    "candidate_mask_swap", "nc_region_mask_swap"):
            out[k] = np.stack([it[k] for it in items], axis=0)
    if to_torch:
        import torch
        for k in ("candidate_patches", "candidate_features"):
            out[k] = torch.from_numpy(out[k])
        for k in ("candidate_mask", "nc_region_mask", "is_positive"):
            out[k] = torch.from_numpy(out[k])
        out["true_slot_idx"] = torch.from_numpy(out["true_slot_idx"])
        out["active_nc_index"] = torch.from_numpy(out["active_nc_index"])
        if "pair_mask" in out:
            out["pair_mask"] = torch.from_numpy(out["pair_mask"])
        if has_swap:
            for k in ("candidate_patches_swap", "candidate_features_swap"):
                out[k] = torch.from_numpy(out[k])
            for k in ("candidate_mask_swap", "nc_region_mask_swap"):
                out[k] = torch.from_numpy(out[k])
    return out


def make_torch_tnp_dataset(ds: TnpGroupedDataset):
    """Small torch adapter."""
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    class _Wrapped(_TorchDataset):
        def __init__(self, base): self.base = base
        def __len__(self): return len(self.base)
        def __getitem__(self, idx): return self.base[idx]

    return _Wrapped(ds)


class StratifiedTnpBatchSampler:
    """Yield fixed-composition batches over TNP indices, stratified by profile.

    Every yielded batch is a list of exactly `sum(per_group.values())` indices
    into `ds.tnp_ids`, with `per_group[g]` slots drawn (without replacement,
    reshuffled per epoch) from the pool of TNPs belonging to group `g`.

    For 48C0: per_group={'positive':2, 'paired_shuffle_v42':2, ...}, batch=12.

    steps_per_epoch defaults to `floor(min_pool_size / max_k)`, so every TNP
    in the smallest-per-batch group is visited exactly once per epoch. Pass an
    explicit value to override.
    """

    def __init__(self, ds: TnpGroupedDataset,
                 tnp_to_group: dict[str, str],
                 per_group: dict[str, int],
                 *,
                 steps_per_epoch: int | None = None,
                 seed: int = 0):
        import random
        self.rng = random.Random(seed)
        self.per_group = dict(per_group)
        self.groups: dict[str, list[int]] = {g: [] for g in per_group}
        for i, tid in enumerate(ds.tnp_ids):
            g = tnp_to_group.get(tid)
            if g in self.groups:
                self.groups[g].append(i)
        missing = [g for g, k in per_group.items()
                    if len(self.groups[g]) < k]
        if missing:
            sizes = {g: len(self.groups[g]) for g in per_group}
            raise ValueError(
                f"StratifiedTnpBatchSampler: groups have fewer TNPs than requested per batch. "
                f"per_group={per_group}, sizes={sizes}, deficit_groups={missing}"
            )
        if steps_per_epoch is None:
            steps_per_epoch = min(
                len(self.groups[g]) // k for g, k in per_group.items()
            )
        self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        cursors = {}
        for g, pool in self.groups.items():
            perm = pool[:]
            self.rng.shuffle(perm)
            cursors[g] = (perm, 0)
        for _ in range(self.steps_per_epoch):
            batch: list[int] = []
            for g, k in self.per_group.items():
                perm, pos = cursors[g]
                if pos + k > len(perm):
                    self.rng.shuffle(perm)
                    pos = 0
                batch.extend(perm[pos:pos + k])
                cursors[g] = (perm, pos + k)
            self.rng.shuffle(batch)
            yield batch


class PairedCounterfactualBatchSampler:
    """Yield batches whose bags are paired counterfactual TWINS by parent tnp.

    For each of K parents picked per batch, the sampler emits the parent's
    variants in a fixed order defined by `profile_suffixes`. This lets the
    trainer index paired-loss terms by simple reshape:

        batch_bags = sampler yields [idx_0, idx_1, ..., idx_{K·P-1}]
        # For parent p at position p, profile q at position q:
        #   idx = p*P + q
        # so batch[k, q]  = idx of k-th parent's q-th profile

    `profile_suffixes` is an ordered dict {profile_name: tnp_id_suffix}. The
    first profile is treated as the parent (its suffix is '' by convention).
    Only parents that have ALL requested profiles are kept.
    """

    def __init__(self, ds: TnpGroupedDataset,
                 profile_suffixes: dict[str, str],
                 *,
                 k_parents_per_batch: int = 2,
                 steps_per_epoch: int | None = None,
                 seed: int = 0):
        import random
        self.rng = random.Random(seed)
        self.k = int(k_parents_per_batch)
        # Preserve insertion order in profile_suffixes
        self.profile_suffixes = dict(profile_suffixes)
        self.n_profiles = len(self.profile_suffixes)

        # Build tnp_id -> dataset index map for fast lookup.
        idx_by_tid = {t: i for i, t in enumerate(ds.tnp_ids)}
        # First profile is treated as the parent.
        parent_key = next(iter(self.profile_suffixes))
        # For every tnp that matches the parent suffix (parent_suffix == '' by convention),
        # check that every other profile's derived tnp exists in the dataset.
        parent_suffix = self.profile_suffixes[parent_key]
        parents = []
        parent_to_indices: dict[str, list[int]] = {}
        for tid, i in idx_by_tid.items():
            if parent_suffix and not tid.endswith(parent_suffix):
                continue
            if parent_suffix:
                parent_id = tid[:-len(parent_suffix)]
            else:
                parent_id = tid
            row = []
            ok = True
            for prof, suf in self.profile_suffixes.items():
                child_tid = parent_id + suf if suf else parent_id
                j = idx_by_tid.get(child_tid)
                if j is None:
                    ok = False
                    break
                row.append(j)
            if ok:
                parents.append(parent_id)
                parent_to_indices[parent_id] = row
        self.parents = parents
        self.parent_to_indices = parent_to_indices
        if steps_per_epoch is None:
            steps_per_epoch = max(1, len(self.parents) // self.k)
        self.steps_per_epoch = int(steps_per_epoch)

    def __len__(self) -> int:
        return self.steps_per_epoch

    def __iter__(self):
        parents = list(self.parents)
        self.rng.shuffle(parents)
        cursor = 0
        for _ in range(self.steps_per_epoch):
            if cursor + self.k > len(parents):
                self.rng.shuffle(parents)
                cursor = 0
            batch: list[int] = []
            for p in parents[cursor:cursor + self.k]:
                batch.extend(self.parent_to_indices[p])
            cursor += self.k
            yield batch
