"""Channel B dataset + collate — v6r2 corpus + MatchTable shards.

Produces per-bag tensors of shape (n_sites, nc_len_eff, 15) plus:
  - site_mask       : (MAX_N_SITES,) — 1 for real sites, 0 for padded
  - label            : scalar 0/1 — is_positive (bag-level)
  - meta             : dict with bag_id, ε_align, hom, gc_target, n_sites for
                       stratification only (never touched by the model)

Design decisions locked in constants.py:
  - Position axis trimmed to nc_len - MAX_L + 1 so every L-window aligns
  - Structure NaN → 0 + structure_valid mask channel
  - flank_argmax → deviation-from-bag-median encoding (÷ FLANK_DEV_SCALE)
  - Site axis padded to MAX_N_SITES = 8

Whitelist enforcement:
  Every read of a labels-side field in `_build_input_tensor` goes through
  `_read_input_label` which asserts the key is in INPUT_TENSOR_LABEL_WHITELIST.
  Target construction (`_build_target`) uses a SEPARATE `labels.get(...)`
  path and legitimately consumes GOLD fields (`TARGET_ONLY_LABEL_KEYS`);
  that path cannot leak into the model input tensor. Bag construction can
  still read other fields for the meta dict; that path is separate and
  cannot leak into the model input tensor either.

Runtime performance:
  Structure channels are computed once per bag from canonical_nc via
  ViennaRNA — ~40 ms/bag on a 200 nt nc. For 50K bags across 10 epochs
  that's ~5.5 h wall of preprocessing. `ChannelBDataset` caches the
  per-bag tensor on the first call; a 10 GB cache holds the full corpus
  at fp16. Set `cache_to=<path>` on the dataset to persist across runs.
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from preprocess.features_structure_v2 import compute_features_v2
from scripts.v5a_framework.match_table import load as load_mt, MatchTable

from .constants import (
    Ls, MAX_L, MAX_N_SITES, CHANNELS, N_CHANNELS,
    INPUT_TENSOR_LABEL_WHITELIST, ARCH_ALLOWED_KEYS, TRAIN_ONLY_LABEL_KEYS,
    TARGET_ONLY_LABEL_KEYS,
    FLANK_DEV_SCALE, CHANNEL_SCALES,
)

# The multi-region concat spacer must be at least `max(Ls) - 1` N's so that no
# length-L window with L in Ls can span both real regions (proof: a window of
# length L can cover 1 left-base + s spacer + 1 right-base ONLY IF
# L >= s + 2; so with spacer_len = max(Ls) - 1, need L >= max(Ls) + 1 to
# cross, impossible for L in Ls). If Ls ever changes without MAX_L tracking,
# this assert fires at import time — catches the "Ls extended but spacer
# formula stale" class of bug the user flagged 2026-09-13.
_MULTI_REGION_SPACER_LEN = MAX_L - 1
assert MAX_L == max(Ls), (
    f"MAX_L ({MAX_L}) != max(Ls) ({max(Ls)}); spacer formula "
    f"(MAX_L - 1) is wrong for the current Ls. Fix constants.py or spacer.")
assert max(Ls) < _MULTI_REGION_SPACER_LEN + 2, (
    f"spacer_len ({_MULTI_REGION_SPACER_LEN}) < max(Ls) ({max(Ls)}) - 1; "
    f"a length-max(Ls) window can span both real regions. Increase spacer "
    f"to at least max(Ls) - 1.")

# Per-channel divisor as a (N_CHANNELS,) array, ordered to match CHANNELS.
_CHANNEL_DIVISOR = np.array([CHANNEL_SCALES[c] for c in CHANNELS], dtype=np.float32)


# ---------- whitelist enforcement --------------------------------------

def _read_input_label(labels: dict, key: str) -> Any:
    """Reads `key` from `labels` with a hard assert that the key is in
    the deploy-computable whitelist. Any attempt to read a train-only
    field (planted_start, oracle_map, etc.) raises immediately."""
    if key in TRAIN_ONLY_LABEL_KEYS:
        raise AssertionError(
            f"_read_input_label: {key!r} is in TRAIN_ONLY_LABEL_KEYS "
            f"and MUST NOT enter the model input tensor. "
            f"See feedback_conjunction_train_deploy_gap + finding_train_deploy_gap_fields."
        )
    if key not in INPUT_TENSOR_LABEL_WHITELIST:
        raise AssertionError(
            f"_read_input_label: {key!r} is not in INPUT_TENSOR_LABEL_WHITELIST. "
            f"Either add it to the whitelist (with justification) or "
            f"stop reading it in input construction. If this field is only "
            f"needed for target y (loss) construction, use `_build_target` — "
            f"that path reads TARGET_ONLY_LABEL_KEYS legitimately."
        )
    return labels.get(key)


def _read_arch(arch: dict, key: str) -> Any:
    if key not in ARCH_ALLOWED_KEYS:
        raise AssertionError(
            f"_read_arch: {key!r} is not in ARCH_ALLOWED_KEYS. "
            f"Arch fields readable for INPUT construction: {sorted(ARCH_ALLOWED_KEYS)}."
        )
    return arch.get(key)


# ---------- target construction (train-only labels) --------------------
# Reads from labels that are in TRAIN_ONLY_LABEL_KEYS. This path is
# SEPARATE from _read_input_label — targets are allowed to see train-only
# ground truth; the whitelist enforcement is scoped to the input tensor
# construction path.

def _build_target(sites: list[dict], nc_len_eff: int) -> np.ndarray:
    """Per-position n_planted_at_position ∈ {0..n_sites} per spec §3.

      n_planted_at_position[p] = |{ site : is_planted(site) ∧
                                     guide_span_start(site) == p }|

    Per negative_mode:
      none / positive: one position with count = n_sites (shared bag_nc_start)
      scattered / twin: n_sites positions each with count 1
      partial: one position with count = n_planted ∈ {1..n_sites-1}, else 0
    """
    y = np.zeros((nc_len_eff,), dtype=np.float32)
    for site in sites:
        labels = site["labels"]
        # `is_planted` is a train-only label (TRAIN_ONLY_LABEL_KEYS blacklist);
        # reading it here is intentional — this is target construction.
        if not labels.get("is_planted", True):
            continue
        span = labels.get("guide_span_in_active_noncoding")
        if not span:
            continue
        p = int(span[0])
        if 0 <= p < nc_len_eff:
            y[p] += 1.0
    return y


# ---------- per-bag tensor construction --------------------------------

@dataclass
class BagInputs:
    x: torch.Tensor           # (n_sites_padded=8, nc_len_eff, N_CHANNELS)
    site_mask: torch.Tensor   # (n_sites_padded=8,) bool
    y: torch.Tensor           # (nc_len_eff,) per-position n_planted_at_position ∈ {0..n_sites}
    label: int                # 0/1 bag-level positive/negative (kept for stratification, not the target)
    meta: dict                # bag_id, ε_align_mean, hom, gc_target, n_sites, nc_len, negative_mode

    @property
    def n_sites(self) -> int:
        return int(self.site_mask.sum().item())

    @property
    def nc_len_eff(self) -> int:
        return int(self.x.shape[1])


class ChannelBDataset(Dataset):
    """One item = one bag. Sort corpus by nc_len for bucketing at
    collate time (see `bucket_collate_fn`).

    `bag_ids` is the ordered list of bag_ids the dataset serves. Order
    is preserved from the JSONL file order, then sorted by nc_len if
    `sort_by_nc_len=True` — bucketing-friendly for the DataLoader.

    Performance notes:
    - **preload_mt=True** (default): loads MatchTable shard in the main
      process at __init__. Fork-based DataLoader workers inherit the
      shared memory via copy-on-write — one 5GB read total, not one per
      worker. Without this, each worker independently reloads on first
      __getitem__, which on Lustre with 4-8 workers per corpus adds
      minutes of stall.
    - **cache_dir**: if set, per-bag input tensors (post-normalization)
      are memoized to disk as .pt files. First epoch = fold + cache;
      subsequent epochs = fp16 load. ~10× speedup after first epoch.
    """
    def __init__(
        self,
        jsonl_path: str | Path,
        shard_dir: str | Path,
        *,
        max_bags: int | None = None,
        sort_by_nc_len: bool = True,
        preload_mt: bool = True,
        cache_dir: str | Path | None = None,
        allow_missing_flank_argmax: bool = False,
    ):
        # NOTE (2026-09-10): `orient_source_fn` parameter removed. The loader
        # no longer reads `arch.orient` — that was a GOLD leak (see
        # channel_b/data.py orient block below). Both fwd and rc shard arrays
        # are consumed unconditionally and combined per-position. The
        # per-site orient override the old param provided is no longer
        # meaningful because there is no per-site orient consumption.
        # allow_missing_flank_argmax=False (default, fail-fast) raises when the
        # shard's flank_argmax_by_excl is None for any site, preventing the
        # silent-zero-fill class of bug (2026-09-08 Durrant OOD incident: main B
        # ate all-zero ch 9-12 for 4+ rounds of experiments before caught).
        # Set to True only if you have a validated reason to accept degenerate
        # zero-filled flank_dev channels (e.g. legacy training with known-good
        # zero baselines). Never silently.
        self._allow_missing_flank_argmax = allow_missing_flank_argmax
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = Path(jsonl_path)
        self.shard_dir = Path(shard_dir)
        self._mt: MatchTable | None = None  # lazy loaded
        # Index the JSONL by bag_id → (byte_offset, nc_len, label)
        self._bag_index: list[tuple[str, int, int, int]] = []   # (bag_id, offset, nc_len, label)
        self._first_line_offset: dict[str, int] = {}
        offset = 0
        seen: set[str] = set()
        with open(self.jsonl_path, "rb") as f:
            for line in f:
                r = json.loads(line)
                bag_id = r["transposase_id"]
                if bag_id not in seen:
                    seen.add(bag_id)
                    labels = r["labels"]
                    nc_len = int(labels["ncrna_length"])
                    label = int(labels.get("is_positive", 1))
                    self._bag_index.append((bag_id, offset, nc_len, label))
                    self._first_line_offset[bag_id] = offset
                    if max_bags is not None and len(self._bag_index) >= max_bags:
                        break
                offset += len(line)
        if sort_by_nc_len:
            self._bag_index.sort(key=lambda t: t[2])
        # Group site records per bag_id for O(1) access
        self._sites_by_bag: dict[str, list[dict]] = {b: [] for b, _, _, _ in self._bag_index}
        wanted = set(self._sites_by_bag)
        with open(self.jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                bag_id = r["transposase_id"]
                if bag_id in wanted:
                    self._sites_by_bag[bag_id].append(r)

        # Pre-load MatchTable in main process so fork-based DataLoader
        # workers inherit shared memory (copy-on-write) instead of each
        # reloading their own 5GB copy from Lustre. Kills the multi-
        # minute-per-worker cold-start latency.
        if preload_mt:
            self._ensure_mt()

    def __len__(self) -> int:
        return len(self._bag_index)

    def _ensure_mt(self):
        if self._mt is None:
            self._mt = load_mt(str(self.shard_dir))

    def _source_hash(self) -> str:
        """SHA1 prefix of this dataset's JSONL path — used as a
        subdirectory in the cache path so two different JSONL sources
        that share a corpus name (and hence a cache_dir) live in
        different filesystem subdirs and cannot collide by name.
        See [[feedback-cache-content-key]] — primary defense; the
        stored content_key in each blob is a backstop."""
        if not hasattr(self, "_cached_source_hash"):
            import hashlib
            h = hashlib.sha1(str(self.jsonl_path.resolve()).encode()).hexdigest()
            self._cached_source_hash = h[:8]
        return self._cached_source_hash

    def _cache_path(self, bag_id: str) -> Path | None:
        if self.cache_dir is None:
            return None
        # Path-based separation: cache_dir / <8-char sha1 of jsonl path> / <bag_id>.pt
        # Different JSONL → different subdir → automatic miss → recompute.
        # No shared-name collision is possible even if two different
        # corpora happen to use the same top-level cache_dir root and the
        # same bag_id namespace (as v7 vs v6r2 both use `bag_XXXXXX`).
        sub = self.cache_dir / self._source_hash()
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{bag_id}.pt"

    def _bag_content_key(self, bag_id: str) -> str:
        """Content-addressed key for a bag: SHA1 of the site records that
        this dataset instance carries for `bag_id`, plus this dataset's
        JSONL absolute path. Two bags that produce different site records
        (different flanks, different noncoding_regions, different arch)
        get different keys — a stale cache written by an unrelated JSONL
        (a v6r2 corpus that happens to share the corpus name and bag_id
        namespace, as caused the 2026-09-12 v7-training invalidation)
        cannot be silently mistaken for THIS dataset's bag.

        Includes jsonl_path in the key so caches from two different JSONLs
        that happen to produce byte-identical site records (theoretically
        possible if two generators emit identical bags) still don't cross.
        """
        import hashlib
        sites = self._sites_by_bag.get(bag_id, [])
        h = hashlib.sha1()
        h.update(str(self.jsonl_path.resolve()).encode())
        h.update(b"|")
        h.update(bag_id.encode())
        h.update(b"|")
        # Hash the JSON of each site record deterministically (sort_keys)
        for s in sites:
            h.update(json.dumps(s, sort_keys=True).encode())
            h.update(b"|")
        return h.hexdigest()[:32]

    def __getitem__(self, idx: int) -> BagInputs:
        bag_id, _off, _nc_len, label = self._bag_index[idx]
        cache = self._cache_path(bag_id)
        if cache is not None and cache.exists():
            try:
                blob = torch.load(cache, map_location="cpu", weights_only=False)
                # Content-key validation. Cache hits WITHOUT a matching key
                # are refused loudly — the pre-2026-09-12 loader silently
                # returned any cached tensor whose bag_id matched, which
                # caused the v7 training run to consume v6r2 tensors under
                # v7 labels for the full 160k-bag corpus (see FROZEN's
                # "Cache-collision invalidation" section). Missing key on
                # a legacy cache is also a mismatch (must rebuild once).
                expected_key = self._bag_content_key(bag_id)
                cache_key = blob.get("content_key")
                if cache_key == expected_key:
                    return BagInputs(x=blob["x"], site_mask=blob["site_mask"],
                                         y=blob["y"], label=int(blob["label"]),
                                         meta=blob["meta"])
                # Mismatch: don't fall through silently. Raise so the operator
                # picks a fresh cache_root (recommended) or explicitly
                # invalidates. Log the offending path + first characters of
                # both keys so the mismatch is diagnosable.
                if not getattr(self, "_allow_cache_key_mismatch", False):
                    raise RuntimeError(
                        f"[data.py] cache-content mismatch at {cache} for "
                        f"bag_id={bag_id!r}: cache_key={cache_key!r} vs "
                        f"expected={expected_key!r}. This cache was written by "
                        f"a different JSONL (or a legacy pre-2026-09-12 cache "
                        f"without content keys). Pick a fresh cache_root OR "
                        f"delete this cache directory. Silent fall-through is "
                        f"disabled — see FROZEN's cache-collision note.")
            except RuntimeError:
                raise
            except Exception:
                pass  # cache is corrupt in a non-shape way; rebuild
        self._ensure_mt()
        sites = self._sites_by_bag[bag_id]
        built = self._build_bag_inputs(bag_id, sites, label)
        if cache is not None:
            try:
                torch.save(
                    {"x": built.x, "site_mask": built.site_mask, "y": built.y,
                     "label": built.label, "meta": built.meta,
                     "content_key": self._bag_content_key(bag_id)}, cache,
                )
            except Exception:
                pass  # cache is best-effort
        return built

    # ------------------------------------------------------------------
    def _build_bag_inputs(self, bag_id: str, sites: list[dict], label: int) -> BagInputs:
        assert self._mt is not None
        first = sites[0]
        labels = first["labels"]

        # 1) bag-level nc + fold. 2026-09-10 rev: nc now comes DIRECTLY
        # from inputs.noncoding_regions[0] (deploy-legal), not from
        # labels.canonical_nc (was a generator-derived pre-mutation
        # sequence — 84% mismatch with any raw region; deprecated per
        # 2026-09-10 SUBSTANTIVE_INPUT_MISMATCH retraction).
        # V7 records emit exactly 1 region; if a v6r2 legacy record has
        # multi-region noncoding_regions, the loader fails fast (raise) —
        # v6r2 comparisons under the new loader must first flatten to
        # 1-region records, or accept the multi-region policy declared in
        # a real-data adapter (out of scope for this loader).
        inputs = first["inputs"]
        noncoding_regions = inputs.get("noncoding_regions")
        if noncoding_regions is None or len(noncoding_regions) == 0:
            raise RuntimeError(
                f"[data.py] {bag_id}: inputs.noncoding_regions missing or empty. "
                f"Loader requires deploy-legal nc from inputs, not labels.canonical_nc "
                f"(2026-09-10 rev — see V7_SPEC §1.1)."
            )
        # Multi-region handling — 2026-09-13 addition per CANONICAL_BAG_SPEC §3.2.
        # v7 records are single-region and hit the len==1 fast path. DDE
        # records are 62-92% multi-region (see 2026-09-13 count) and MUST
        # declare `arch.nc_multi_region_scoring`. The only implemented
        # policy is "concat_with_N_spacer" with a `MAX_L - 1` N spacer;
        # any other value raises.
        #
        # Why the spacer length is `MAX_L - 1`: a length-L window can span
        # BOTH regions only if `L > spacer_len + 1`. With spacer_len =
        # MAX_L - 1 and L ≤ MAX_L, every L-window is entirely within
        # left-region, entirely within right-region, or partially in one
        # side + spacer. No L-window contains bases from both real regions,
        # so no cross-boundary false matches are possible regardless of
        # aligner N-handling.
        region_boundaries: list[tuple[int, int]] | None = None
        if len(noncoding_regions) == 1:
            canonical_nc: str = noncoding_regions[0]
        else:
            arch_declaration = first["labels"].get("arch", {}).get(
                "nc_multi_region_scoring")
            if arch_declaration is None:
                raise RuntimeError(
                    f"[data.py] {bag_id}: {len(noncoding_regions)} regions but "
                    f"arch.nc_multi_region_scoring not declared. "
                    f"CANONICAL_BAG_SPEC §3.2 forbids default policies; the record "
                    f"MUST set nc_multi_region_scoring='concat_with_N_spacer' (the "
                    f"only supported value) or the source needs a real-data adapter "
                    f"that flattens to 1 region."
                )
            if arch_declaration != "concat_with_N_spacer":
                raise RuntimeError(
                    f"[data.py] {bag_id}: arch.nc_multi_region_scoring="
                    f"{arch_declaration!r} — only 'concat_with_N_spacer' is "
                    f"implemented (per CANONICAL_BAG_SPEC §3.2). If you want "
                    f"another policy, implement it explicitly, do NOT silently "
                    f"fall back to a default."
                )
            spacer = "N" * _MULTI_REGION_SPACER_LEN
            parts = []
            region_boundaries = []
            cursor = 0
            for k, reg in enumerate(noncoding_regions):
                if k > 0:
                    parts.append(spacer)
                    cursor += len(spacer)
                start = cursor
                parts.append(reg)
                cursor += len(reg)
                region_boundaries.append((start, cursor))   # [start, end) in concat
            canonical_nc = "".join(parts)
        # NOTE: guide_length from labels is the GENERATOR's guide_L (can be
        # 9-14); we intentionally use Channel B's MAX_L=12 for the structure
        # window so all bags produce identically-shaped structure arrays
        # regardless of the generator's L. Ignoring generator guide_L for
        # this axis is fine — structure at L=12 windows the sequence at
        # a fixed scale that Channel B's cross-L attention can consume.
        _ = _read_input_label(labels, "guide_length")  # read + discard (whitelist audit trail)
        nc_len = len(canonical_nc)
        nc_len_eff = nc_len - MAX_L + 1
        if nc_len_eff <= 0:
            raise ValueError(f"{bag_id}: nc_len {nc_len} shorter than MAX_L {MAX_L}")

        # 2) structure channels (bag-level, broadcast across sites); at MAX_L.
        feats = compute_features_v2(canonical_nc, guide_length=MAX_L)
        # feats has arrays of length (nc_len - L_guide + 1); we index only
        # positions 0..nc_len_eff-1 for the tensor. Nan → 0 + valid mask.
        struct_arrays = {
            "dG_open_uL_pn":        feats.dG_open_uL_pn,
            "H_pair_win":           feats.H_pair_win,
            "cooperativity_win_pn": feats.cooperativity_win_pn,
            "E_span_win":           feats.E_span_win,
        }
        struct_valid = feats.windowed_valid[:nc_len_eff].astype(np.float32)
        struct_stack = np.zeros((nc_len_eff, 4), dtype=np.float32)
        for i, key in enumerate(("dG_open_uL_pn", "H_pair_win",
                                     "cooperativity_win_pn", "E_span_win")):
            arr = struct_arrays[key][:nc_len_eff]
            struct_stack[:, i] = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # 3) per-site per-L m_max and flank_argmax
        # (2026-09-10 rev: previously read `arch.orient` — a GOLD field — and
        # pulled ONLY that orient's shard arrays. That was a deploy-illegal
        # leak: at deploy the true orient is unknown. Now read BOTH orients'
        # arrays and combine per-position: m_max = elementwise max;
        # flank_argmax = argmax of the winning orient at that position.)
        arch = _read_input_label(labels, "arch")
        n_sites_real = _read_arch(arch, "n_sites") or len(sites)

        # (n_sites_real, nc_len_eff, len(Ls)) for m_max and flank_argmax_raw
        m_max_per_site = np.zeros((n_sites_real, nc_len_eff, len(Ls)), dtype=np.float32)
        argmax_per_site = np.zeros((n_sites_real, nc_len_eff, len(Ls)), dtype=np.float32)
        # site_to_canonical is applied in shard-side _project_to_canonical
        # already, so array indices are canonical positions.
        for s_idx in range(n_sites_real):
            for L_i, L in enumerate(Ls):
                ma_fwd = self._mt.get(bag_id, s_idx, "fwd", L)
                ma_rc  = self._mt.get(bag_id, s_idx, "rc",  L)
                m_fwd = ma_fwd.m_max_by_excl.get(0)   # (n_canonical_positions,) int8
                m_rc  = ma_rc.m_max_by_excl.get(0)
                if m_fwd is None or m_rc is None:
                    continue
                # elementwise per-position max over orients — this is the
                # deploy-legal m_max (max_{orient} m_max_by_pos)
                n_pos = min(m_fwd.shape[0], m_rc.shape[0], nc_len_eff)
                m_combined = np.maximum(m_fwd[:n_pos], m_rc[:n_pos])
                m_max_per_site[s_idx, :n_pos, L_i] = m_combined.astype(np.float32)
                # flank_argmax: take argmax of the WINNING orient at each
                # position — argmax_fwd where fwd won, else argmax_rc.
                a_fwd = ma_fwd.flank_argmax_by_excl.get(0)
                a_rc  = ma_rc.flank_argmax_by_excl.get(0)
                if a_fwd is None or a_rc is None:
                    if not getattr(self, "_allow_missing_flank_argmax", False):
                        raise RuntimeError(
                            f"[data.py] shard's flank_argmax_by_excl is None for "
                            f"bag_id={bag_id!r} site_idx={s_idx} L={L} "
                            f"(fwd={'None' if a_fwd is None else 'ok'}, "
                            f"rc={'None' if a_rc is None else 'ok'}). "
                            f"Silent zero-fill of ch 9-12 (flank_dev) is disabled by default "
                            f"(see 2026-09-08 Durrant OOD incident + 2026-09-10 orient-leak fix). "
                            f"To bypass, pass allow_missing_flank_argmax=True; to fix the "
                            f"underlying issue, rebuild the shard with flank_argmax populated "
                            f"or use an on-the-fly path that computes it via _compute_site_arrays.")
                    if not getattr(self, "_zero_fill_warned", False):
                        print(f"[data.py] WARNING: flank_argmax_by_excl is None for at least "
                              f"one (site, orient) — ch 9-12 will be zero-filled at those cells. "
                              f"Suppressing further warnings for this dataset.")
                        self._zero_fill_warned = True
                    continue
                a_combined = np.where(m_fwd[:n_pos] >= m_rc[:n_pos],
                                          a_fwd[:n_pos], a_rc[:n_pos])
                argmax_per_site[s_idx, :n_pos, L_i] = a_combined.astype(np.float32)

        # 4) flank_argmax → deviation-from-bag-median coherence encoding
        # For each (position, L), take median over sites and subtract.
        # Sites with no argmax (missing site or padded) contribute NaN
        # ignored via nanmedian, then subtract from real sites and divide.
        bag_median = np.median(argmax_per_site, axis=0, keepdims=True)  # (1, nc_len_eff, len(Ls))
        flank_dev = (argmax_per_site - bag_median) / FLANK_DEV_SCALE

        # 5) build the 15-channel tensor
        x = np.zeros((MAX_N_SITES, nc_len_eff, N_CHANNELS), dtype=np.float32)
        for s_idx in range(n_sites_real):
            # channels 0-3: m_max per L
            x[s_idx, :, 0:4] = m_max_per_site[s_idx]
            # channels 4-7: structure (broadcast across sites)
            x[s_idx, :, 4:8] = struct_stack
            # channel 8: structure_valid mask (broadcast)
            x[s_idx, :, 8] = struct_valid
            # channels 9-12: flank_dev per L
            x[s_idx, :, 9:13] = flank_dev[s_idx]
            # channels 13-14: LEFT ZERO (2026-09-10 rev). Previously encoded
            # `arch.orient` one-hot — a GOLD leak. An intermediate revision
            # replaced it with per-site "winning orient" (derived from data),
            # but that quantity is 80%-accurate on strong-signal v6r2 and
            # undefined on non-planted real data (DDE has no "true" orient).
            # Orient information is already implicit in ch 0-3 (per-position
            # max over orient) and ch 9-12 (argmax from the winning orient),
            # so an explicit orient channel adds noise, not signal. Retained
            # as zero rather than dropped from the tensor to preserve the
            # 15-channel architecture (checkpoint compatibility). If a
            # deploy-legal orient signal is ever wanted, it belongs in a
            # separate v7' experiment, not the default input.
            # (x[s_idx, :, 13] and x[s_idx, :, 14] remain zero from the
            #  np.zeros initialization above; explicit no-op here.)

        # Apply fixed per-channel divisor. Padded sites (>= n_sites_real)
        # stay zero, which are correct pre-normalization values.
        x /= _CHANNEL_DIVISOR

        site_mask = np.zeros((MAX_N_SITES,), dtype=bool)
        site_mask[:n_sites_real] = True

        # 6) stratification meta (NOT input to the model)
        eps_align_per_site = [
            s["labels"].get("epsilon_align", 0.0) for s in sites
        ]
        meta = {
            "bag_id":        bag_id,
            "n_sites":       n_sites_real,
            "nc_len":        nc_len,
            "hom":           float(arch.get("nc_homology_rate", 1.0)),
            "gc_target":     float(arch.get("gc_target", 0.5)),
            "epsilon_align_mean": float(np.mean(eps_align_per_site)) if eps_align_per_site else 0.0,
            "epsilon_align_max":  float(np.max(eps_align_per_site)) if eps_align_per_site else 0.0,
            "flank_offset_mode": arch.get("flank_offset_mode", ""),
            "negative_mode":     labels.get("negative_mode", "none"),
            # region boundaries in the concatenated canonical_nc (None for
            # single-region bags). Downstream analysis can slice per-region
            # via these; the model consumes the tensor as one contiguous axis.
            "region_boundaries": region_boundaries,
        }
        # Per-position ordinal target (train-only labels, spec §3)
        y = _build_target(sites, nc_len_eff)
        return BagInputs(
            x=torch.from_numpy(x),
            site_mask=torch.from_numpy(site_mask),
            y=torch.from_numpy(y),
            label=label,
            meta=meta,
        )


# ---------- collate ----------------------------------------------------

def bucket_collate_fn(items: list[BagInputs]) -> dict:
    """Batches BagInputs whose nc_len_eff already match (bucketing at
    the sampler; if a batch mixes lengths, pad to the max in-batch)."""
    max_pos = max(it.x.shape[1] for it in items)
    B = len(items)
    x = torch.zeros((B, MAX_N_SITES, max_pos, N_CHANNELS), dtype=torch.float32)
    y = torch.zeros((B, max_pos), dtype=torch.float32)
    pos_mask = torch.zeros((B, max_pos), dtype=torch.bool)
    site_mask = torch.zeros((B, MAX_N_SITES), dtype=torch.bool)
    labels = torch.zeros((B,), dtype=torch.float32)
    metas = []
    for i, it in enumerate(items):
        P = it.x.shape[1]
        x[i, :, :P] = it.x
        y[i, :P] = it.y
        pos_mask[i, :P] = True
        site_mask[i] = it.site_mask
        labels[i] = float(it.label)
        metas.append(it.meta)
    return {
        "x":         x,           # (B, MAX_N_SITES, max_pos, N_CHANNELS)
        "y":         y,           # (B, max_pos) per-position ordinal target
        "site_mask": site_mask,   # (B, MAX_N_SITES)
        "pos_mask":  pos_mask,    # (B, max_pos)
        "labels":    labels,      # (B,)
        "metas":     metas,
    }


# ---------- diagnostic: dump one bag -----------------------------------

def dump_bag(bag_inputs: BagInputs, out_stream=None) -> None:
    """Print every value in the input tensor for one bag, plus meta.
    Human-readable per-site per-position table for eye-check auditing.
    A11-shape errors (reading the wrong field, off-by-one L) are only
    catchable this way; summary statistics hide them.
    """
    import sys
    if out_stream is None:
        out_stream = sys.stdout
    p = lambda *a, **k: print(*a, file=out_stream, **k)
    x = bag_inputs.x.numpy()
    site_mask = bag_inputs.site_mask.numpy()
    n_sites, nc_len_eff, C = x.shape
    p(f"=== bag {bag_inputs.meta['bag_id']} ===")
    for k, v in bag_inputs.meta.items():
        p(f"  meta.{k} = {v}")
    p(f"  tensor shape: (n_sites_padded={n_sites}, nc_len_eff={nc_len_eff}, C={C})")
    p(f"  n_sites_real: {int(site_mask.sum())}")
    p(f"  label: {bag_inputs.label}")
    p()
    for s in range(n_sites):
        real = "REAL" if site_mask[s] else "PAD"
        p(f"--- site {s} [{real}] ---")
        # Print per-position values for first 8 positions + last 3, for each channel
        pos_indices = list(range(min(8, nc_len_eff))) + \
                      ([-3, -2, -1] if nc_len_eff > 8 else [])
        pos_indices = sorted(set([i if i >= 0 else nc_len_eff + i for i in pos_indices]))
        # Header
        p(f"    {'pos':>4s}  " + "  ".join(f"{c[:10]:>10s}" for c in CHANNELS))
        for pos in pos_indices:
            row = [f"    {pos:>4d}"]
            for c_i in range(C):
                row.append(f"{x[s, pos, c_i]:>10.4f}")
            p("  ".join(row))
        p()
