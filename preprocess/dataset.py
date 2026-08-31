"""Dataset wrapper over the DL_novel_guide_editor splits.

Loads records lazily from `splits/{train,val,test}.jsonl` via a per-line
byte-offset index (built once at construction, ~1 s per split), runs
`preprocess_site` on `__getitem__`, and returns a dict of numpy arrays.

The class is deliberately framework-agnostic (pure numpy): it works
without torch. `collate_batch(..., to_torch=True)` lazy-imports torch and
converts arrays to tensors for use in a `torch.utils.data.DataLoader`
via a plain `Dataset` adapter.

Typical usage:

    from preprocess.dataset import DLNovelGuideEditorDataset, collate_batch
    from preprocess.site import StructureCache

    cache = StructureCache('/.../structure/val_u16.index.json')
    ds = DLNovelGuideEditorDataset('/.../splits/val.jsonl', cache)
    item = ds[0]                     # dict of numpy arrays

    batch = collate_batch([ds[i] for i in range(8)], to_torch=True)
    #  batch['candidate_patches']  torch.float32 (8, 3, 96, 64, 22)
    #  batch['candidate_features'] torch.float32 (8, 3, 96, 13)
    #  batch['candidate_mask']     torch.bool    (8, 3, 96)
    #  batch['nc_region_mask']     torch.bool    (8, 3)
    #  batch['is_positive']        torch.bool    (8,)
"""

from __future__ import annotations

import json
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


# Keys in a preprocess_site output that get stacked into a batch tensor.
_STACK_KEYS = (
    "candidate_patches",
    "candidate_features",
    "candidate_mask",
    "nc_region_mask",
)


class DLNovelGuideEditorDataset:
    """Random-access dataset over one split jsonl.

    Attributes:
        split_path:      Path to the jsonl.
        structure_cache: StructureCache instance (required).
        preprocess_kwargs: extra kwargs forwarded to preprocess_site
                            (nc_max, num_nc_slots, L_min, L_max,
                             orientations, patch_width, top_k_per_combo).

    Notes:
        - The line-offset index is built once at __init__ (linear scan of
          the file). For a 600k-line file this takes ~1 s.
        - `__getitem__` opens the file, seeks, reads one line, closes. This
          costs one open+seek per call; if a worker uses many items in a
          row, keep the file handle open by passing `persistent_open=True`.
        - Records missing from the structure cache raise KeyError from
          preprocess_site; catch upstream if you want to skip them.
    """

    def __init__(
        self,
        split_path: str | Path,
        structure_cache: StructureCache,
        *,
        nc_max: int = DEFAULT_NC_MAX,
        num_nc_slots: int = DEFAULT_NUM_NC_SLOTS,
        top_k_per_combo: int = TOP_K_PER_COMBO_DEFAULT,
        L_min: int = DEFAULT_L_MIN,
        L_max: int = DEFAULT_L_MAX,
        orientations: Sequence[str] = DEFAULT_ORIENTATIONS,
        patch_width: int = PATCH_WIDTH_DEFAULT,
        persistent_open: bool = False,
    ):
        self.split_path = Path(split_path)
        self.structure_cache = structure_cache
        self.preprocess_kwargs = dict(
            nc_max=nc_max,
            num_nc_slots=num_nc_slots,
            top_k_per_combo=top_k_per_combo,
            L_min=L_min,
            L_max=L_max,
            orientations=tuple(orientations),
            patch_width=patch_width,
        )
        self._offsets = self._build_offsets()
        self._persistent_open = persistent_open
        self._file = None  # type: ignore[assignment]

    def _build_offsets(self) -> np.ndarray:
        offsets: list[int] = []
        with open(self.split_path, "rb") as f:
            while True:
                off = f.tell()
                line = f.readline()
                if not line:
                    break
                offsets.append(off)
        return np.asarray(offsets, dtype=np.int64)

    def __len__(self) -> int:
        return int(self._offsets.shape[0])

    def _read_line(self, idx: int) -> bytes:
        off = int(self._offsets[idx])
        if self._persistent_open:
            if self._file is None:
                self._file = open(self.split_path, "rb")
            self._file.seek(off)
            return self._file.readline()
        with open(self.split_path, "rb") as f:
            f.seek(off)
            return f.readline()

    def __getitem__(self, idx: int) -> dict:
        rec = json.loads(self._read_line(idx))
        return preprocess_site(
            rec, structure_cache=self.structure_cache, **self.preprocess_kwargs
        )

    def iter_indices(self, indices: Iterable[int]):
        for i in indices:
            yield self[i]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def collate_batch(items: list[dict], *, to_torch: bool = False) -> dict:
    """Stack a list of preprocess_site outputs into a batched dict.

    Numpy arrays under _STACK_KEYS are stacked along a new leading batch
    axis. Scalar `is_positive` becomes a bool array of shape (B,).
    List-valued keys (site_id, violation_profile, nc_lengths) are kept
    as Python lists.

    If `to_torch=True`, arrays are converted to torch tensors after
    stacking (dtype preserved; bool stays bool).
    """
    if not items:
        raise ValueError("collate_batch: empty items")
    out: dict = {}
    for k in _STACK_KEYS:
        out[k] = np.stack([it[k] for it in items], axis=0)
    out["is_positive"] = np.asarray(
        [bool(it.get("is_positive")) if it.get("is_positive") is not None else False
          for it in items],
        dtype=bool,
    )
    # Book-keeping (Python lists — model doesn't consume these but eval does).
    out["site_id"] = [it.get("site_id") for it in items]
    out["violation_profile"] = [it.get("violation_profile") for it in items]
    out["nc_lengths"] = [it.get("nc_lengths") for it in items]
    # Channel names come from the first item; identical across items in one dataset.
    out["patch_channel_names"] = items[0]["patch_channel_names"]
    out["feature_names"] = items[0]["feature_names"]

    if to_torch:
        import torch
        for k in _STACK_KEYS:
            out[k] = torch.from_numpy(out[k])
        out["is_positive"] = torch.from_numpy(out["is_positive"])
    return out


def make_torch_dataset(
    ds: DLNovelGuideEditorDataset,
    to_torch: bool = True,
):
    """Return a torch.utils.data.Dataset wrapping the numpy dataset.

    Torch is imported lazily; call this only inside a torch-enabled env.
    Each `__getitem__` returns a dict; use `collate_batch` (with the same
    `to_torch` flag) as the DataLoader's collate_fn.
    """
    import torch
    from torch.utils.data import Dataset as _TorchDataset

    class _Wrapped(_TorchDataset):
        def __init__(self, base):
            self.base = base

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            if to_torch:
                # Convert stackable arrays only; leave metadata as-is.
                for k in _STACK_KEYS:
                    item[k] = torch.from_numpy(item[k])
            return item

    return _Wrapped(ds)
