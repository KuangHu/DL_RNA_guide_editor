"""Tests for preprocess/dataset.py.

Checks:
  1. Line-offset index length matches record count.
  2. __getitem__(i) round-trips to preprocess_site on the same record.
  3. collate_batch stacks the four tensor keys with correct shape/dtype.
  4. Torch conversion (skipped if torch isn't importable).

Requires the smoke structure mmap at /tmp/nc_unp_smoke.index.json.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocess.dataset import (
    DLNovelGuideEditorDataset,
    collate_batch,
)
from preprocess.site import StructureCache, preprocess_site
from preprocess.candidates import PATCH_CHANNELS, PATCH_WIDTH_DEFAULT, NUM_FEATURES, k_max

VAL_SPLIT = "/global/scratch/users/kh36969/DL_novel_guide_editor/splits/val.jsonl"
STRUCTURE_INDEX_SMOKE = os.environ.get(
    "STRUCTURE_INDEX", "/tmp/nc_unp_smoke.index.json"
)


def offset_index_smoke():
    cache = StructureCache(STRUCTURE_INDEX_SMOKE)
    ds = DLNovelGuideEditorDataset(VAL_SPLIT, cache)
    n_lines = sum(1 for _ in open(VAL_SPLIT))
    assert len(ds) == n_lines, f"index has {len(ds)}, file has {n_lines}"


def getitem_round_trip():
    cache = StructureCache(STRUCTURE_INDEX_SMOKE)
    ds = DLNovelGuideEditorDataset(VAL_SPLIT, cache)
    # Read first 5 records via dataset and via a direct file walk; compare.
    with open(VAL_SPLIT) as f:
        for i in range(5):
            rec = json.loads(f.readline())
            direct = preprocess_site(rec, structure_cache=cache)
            via_ds = ds[i]
            for k in ("candidate_patches", "candidate_features", "candidate_mask",
                      "nc_region_mask"):
                assert np.array_equal(direct[k], via_ds[k]), f"key {k} differs at idx {i}"
            assert direct["is_positive"] == via_ds["is_positive"]
            assert direct["site_id"] == via_ds["site_id"]


def collate_shapes():
    cache = StructureCache(STRUCTURE_INDEX_SMOKE)
    ds = DLNovelGuideEditorDataset(VAL_SPLIT, cache)
    items = [ds[i] for i in range(8)]
    batch = collate_batch(items)
    K = k_max()
    assert batch["candidate_patches"].shape == (8, 3, K, PATCH_WIDTH_DEFAULT, PATCH_CHANNELS), (
        batch["candidate_patches"].shape
    )
    assert batch["candidate_features"].shape == (8, 3, K, NUM_FEATURES)
    assert batch["candidate_mask"].shape == (8, 3, K)
    assert batch["nc_region_mask"].shape == (8, 3)
    assert batch["is_positive"].shape == (8,) and batch["is_positive"].dtype == bool
    assert len(batch["site_id"]) == 8
    assert batch["patch_channel_names"][17] == "guide_mask"


def collate_torch_conversion_optional():
    """Skip if torch not importable in current interpreter (system python3
    doesn't have torch; opfi env does)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        print("  [skip] torch not importable in this python; use conda env opfi to test")
        return
    cache = StructureCache(STRUCTURE_INDEX_SMOKE)
    ds = DLNovelGuideEditorDataset(VAL_SPLIT, cache)
    items = [ds[i] for i in range(4)]
    batch = collate_batch(items, to_torch=True)
    import torch
    assert isinstance(batch["candidate_patches"], torch.Tensor)
    assert batch["candidate_patches"].dtype == torch.float32
    assert batch["candidate_mask"].dtype == torch.bool
    assert batch["is_positive"].dtype == torch.bool


def main():
    if not os.path.exists(STRUCTURE_INDEX_SMOKE):
        print(f"[skip all] no structure cache at {STRUCTURE_INDEX_SMOKE}")
        return

    print("offset index length ...", end=" ")
    offset_index_smoke()
    print("ok")

    print("__getitem__ round trip vs preprocess_site ...", end=" ")
    getitem_round_trip()
    print("ok")

    print("collate_batch shapes ...", end=" ")
    collate_shapes()
    print("ok")

    print("collate_batch torch conversion ...")
    collate_torch_conversion_optional()


if __name__ == "__main__":
    main()
