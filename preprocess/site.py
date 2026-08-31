"""Record-level preprocessing.

`preprocess_site` takes one dataset record and returns candidate-based
tensors:

  candidate_patches   (num_nc_slots, K_max, patch_width, PATCH_CHANNELS)
  candidate_features  (num_nc_slots, K_max, NUM_FEATURES)
  candidate_mask      (num_nc_slots, K_max)

Each of the K_max slots per NC is one "alignment-aware structure patch":
a fixed-width guide-centered window of the NC that carries both RNAplfold
per-nt accessibility and alignment-specific overlays (guide mask, match /
mismatch state, paired flank position, guide-internal offset). See
`preprocess/candidates.py` for the full channel layout.

The precomputed RNAplfold structure cache is REQUIRED — build it once
with `scripts/precompute_structure.py`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .candidates import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_ORIENTATIONS,
    FEATURE_NAMES,
    NUM_FEATURES,
    PATCH_CHANNEL_NAMES,
    PATCH_CHANNELS,
    PATCH_WIDTH_DEFAULT,
    TOP_K_PER_COMBO_DEFAULT,
    build_candidate_arrays,
    k_max,
)


# Dataset caps (measured on the DL_novel_guide_editor dataset):
#   num_noncoding_regions ∈ {1, 2, 3}
#   NC region length      126 .. 341 bp
#   flank length          120 bp (fixed)
DEFAULT_NUM_NC_SLOTS = 3
DEFAULT_NC_MAX = 350


class StructureCache:
    """Lazy reader for the precomputed RNAplfold memmap.

    Layout produced by `scripts/precompute_structure.py`:

      <base>.mmap        float16 (N, nc_max, u_max)  unpaired probabilities
      <base>.valid       uint8   (N, nc_max, u_max)  NA mask (1 = real value)
      <base>.index.json  { site_id: {"slots": {slot: row, ...}, ...},
                            _meta: { N, nc_max, u_max, W, L, ... } }

    Usage:

        cache = StructureCache('/path/to/base.index.json')
        prof, valid = cache.get('tnp_00001_site_0001', slot=0, nc_len=275)

    Safe to construct in the parent process and share across dataloader
    workers (numpy memmaps are file-backed and copy-on-read).
    """

    def __init__(self, index_path: str | Path):
        index_path = Path(index_path)
        with open(index_path) as f:
            self._index = json.load(f)
        self._meta = self._index["_meta"]
        mmap_path = Path(self._meta["mmap_path"])
        valid_path = Path(self._meta["valid_path"])
        if not mmap_path.is_absolute():
            mmap_path = index_path.parent / mmap_path
        if not valid_path.is_absolute():
            valid_path = index_path.parent / valid_path
        self.mmap_path = mmap_path
        self.valid_path = valid_path
        self.N = int(self._meta["N"])
        self.nc_max = int(self._meta["nc_max"])
        self.u_max = int(self._meta["u_max"])
        self.W = int(self._meta["W"])
        self.L = int(self._meta["L"])
        self._prof: np.memmap | None = None
        self._valid: np.memmap | None = None

    def _open(self):
        if self._prof is None:
            self._prof = np.memmap(
                self.mmap_path, dtype=np.float16, mode="r",
                shape=(self.N, self.nc_max, self.u_max),
            )
            self._valid = np.memmap(
                self.valid_path, dtype=np.uint8, mode="r",
                shape=(self.N, self.nc_max, self.u_max),
            )

    def has(self, site_id: str, slot: int) -> bool:
        entry = self._index.get(site_id)
        if entry is None:
            return False
        return str(slot) in entry.get("slots", {})

    def get(self, site_id: str, slot: int, nc_len: int) -> tuple[np.ndarray, np.ndarray]:
        entry = self._index.get(site_id)
        if entry is None:
            raise KeyError(f"site_id {site_id!r} not in structure cache")
        slots = entry["slots"]
        row = slots.get(str(slot))
        if row is None:
            raise KeyError(
                f"site {site_id!r} slot {slot} not in structure cache "
                f"(available slots: {sorted(slots)})"
            )
        self._open()
        prof = np.asarray(self._prof[row, :nc_len, :], dtype=np.float32)   # type: ignore[index]
        valid = np.asarray(self._valid[row, :nc_len, :] != 0)              # type: ignore[index]
        return prof, valid


def preprocess_site(
    rec: dict,
    *,
    nc_max: int = DEFAULT_NC_MAX,
    num_nc_slots: int = DEFAULT_NUM_NC_SLOTS,
    top_k_per_combo: int = TOP_K_PER_COMBO_DEFAULT,
    L_min: int = DEFAULT_L_MIN,
    L_max: int = DEFAULT_L_MAX,
    orientations: Sequence[str] = DEFAULT_ORIENTATIONS,
    patch_width: int = PATCH_WIDTH_DEFAULT,
    structure_cache: Optional[StructureCache] = None,
) -> dict:
    """Preprocess one dataset record into candidate-based padded tensors.

    Args:
        rec: JSONL record (from positives/negatives/splits jsonl).
        structure_cache: REQUIRED — precomputed RNAplfold memmap reader.

    Returns dict:
        'candidate_patches':  float32 (num_nc_slots, K_max, patch_width, PATCH_CHANNELS)
        'candidate_features': float32 (num_nc_slots, K_max, NUM_FEATURES)
        'candidate_mask':     bool    (num_nc_slots, K_max)
        'nc_region_mask':     bool    (num_nc_slots,)          populated NC slots
        'nc_lengths':         tuple[int, ...]                  raw NC lengths
        'flank_len':          int                              120 in this dataset
        'patch_channel_names': list[str]  len PATCH_CHANNELS
        'feature_names':      list[str]  len NUM_FEATURES
        'K_max':              int
        'K_layout':           tuple(orientations, L_range, top_k_per_combo)
        'is_positive':        bool or None
        'violation_profile':  str or None
        'site_id':            str or None
    """
    if structure_cache is None:
        raise ValueError(
            "preprocess_site requires a structure_cache. Build one with "
            "scripts/precompute_structure.py and pass "
            "StructureCache('/path/base.index.json')."
        )

    flank: str = rec["inputs"]["flank"]
    ncs: list[str] = list(rec["inputs"]["noncoding_regions"])
    flank_len = len(flank)

    if len(ncs) > num_nc_slots:
        raise ValueError(
            f"record has {len(ncs)} NC regions > num_nc_slots={num_nc_slots}"
        )

    K_max = k_max(top_k_per_combo, L_min, L_max, orientations)

    candidate_patches = np.zeros(
        (num_nc_slots, K_max, patch_width, PATCH_CHANNELS), dtype=np.float32
    )
    candidate_features = np.zeros(
        (num_nc_slots, K_max, NUM_FEATURES), dtype=np.float32
    )
    candidate_mask = np.zeros((num_nc_slots, K_max), dtype=bool)
    nc_region_mask = np.zeros((num_nc_slots,), dtype=bool)

    site_id = rec.get("site_id")
    for slot, nc in enumerate(ncs):
        nc_len = len(nc)
        if nc_len > nc_max:
            raise ValueError(
                f"NC region length {nc_len} > nc_max={nc_max}; increase nc_max"
            )
        profile, valid = structure_cache.get(site_id, slot, nc_len)
        patches, feats, mask, _cands = build_candidate_arrays(
            nc, flank, profile, valid,
            top_k_per_combo=top_k_per_combo,
            L_min=L_min, L_max=L_max,
            orientations=orientations,
            patch_width=patch_width,
            nc_max=nc_max,
        )
        candidate_patches[slot] = patches
        candidate_features[slot] = feats
        candidate_mask[slot] = mask
        nc_region_mask[slot] = True

    labels = rec.get("labels", {}) or {}
    return {
        "candidate_patches": candidate_patches,
        "candidate_features": candidate_features,
        "candidate_mask": candidate_mask,
        "nc_region_mask": nc_region_mask,
        "nc_lengths": tuple(len(nc) for nc in ncs),
        "flank_len": flank_len,
        "patch_channel_names": list(PATCH_CHANNEL_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "K_max": K_max,
        "K_layout": {
            "orientations": tuple(orientations),
            "L_range": (L_min, L_max),
            "top_k_per_combo": top_k_per_combo,
        },
        "is_positive": labels.get("is_positive"),
        "violation_profile": labels.get("violation_profile"),
        "site_id": site_id,
    }
