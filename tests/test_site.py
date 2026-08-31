"""Site-level preprocess tests (new candidate-based output).

Verifies preprocess_site() on real records:
 - Output shape / dtypes.
 - Requires a structure cache; error path when missing.
 - Ground truth positive appears in a candidate slot at the correct NC.
 - Padded NC slots (num_noncoding_regions < num_nc_slots) are all-zero and masked.
 - Throughput sanity.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocess.candidates import (
    FEATURE_NAMES,
    NUM_FEATURES,
    PATCH_CHANNELS,
    PATCH_WIDTH_DEFAULT,
    TOP_K_PER_COMBO_DEFAULT,
    k_max,
)
from preprocess.site import (
    DEFAULT_NC_MAX,
    DEFAULT_NUM_NC_SLOTS,
    StructureCache,
    preprocess_site,
)

VAL_SPLIT = "/global/scratch/users/kh36969/DL_novel_guide_editor/splits/val.jsonl"
STRUCTURE_INDEX_SMOKE = os.environ.get(
    "STRUCTURE_INDEX", "/tmp/nc_unp_smoke.index.json"
)


def load_head(path: str, n: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            yield json.loads(line)


def check_shapes(out: dict) -> None:
    K = out["K_max"]
    assert K == k_max()
    assert out["candidate_patches"].shape == (
        DEFAULT_NUM_NC_SLOTS, K, PATCH_WIDTH_DEFAULT, PATCH_CHANNELS
    ), out["candidate_patches"].shape
    assert out["candidate_patches"].dtype == np.float32
    assert out["candidate_features"].shape == (DEFAULT_NUM_NC_SLOTS, K, NUM_FEATURES)
    assert out["candidate_features"].dtype == np.float32
    assert out["candidate_mask"].shape == (DEFAULT_NUM_NC_SLOTS, K)
    assert out["candidate_mask"].dtype == bool
    assert out["nc_region_mask"].shape == (DEFAULT_NUM_NC_SLOTS,)
    assert out["nc_region_mask"].dtype == bool
    assert len(out["patch_channel_names"]) == PATCH_CHANNELS
    assert len(out["feature_names"]) == NUM_FEATURES


def check_unpopulated_slots(out: dict) -> None:
    """NC slots beyond nc_lengths must be zero and mask=False."""
    nls = out["nc_lengths"]
    for slot in range(DEFAULT_NUM_NC_SLOTS):
        if slot >= len(nls):
            assert not out["nc_region_mask"][slot]
            assert not out["candidate_mask"][slot].any(), (
                f"unpopulated slot {slot} candidate_mask has True entries"
            )
            assert (out["candidate_patches"][slot] == 0.0).all(), (
                f"unpopulated slot {slot} patches non-zero"
            )
            assert (out["candidate_features"][slot] == 0.0).all(), (
                f"unpopulated slot {slot} features non-zero"
            )
        else:
            assert out["nc_region_mask"][slot]


def check_ground_truth_candidate(rec: dict, out: dict) -> None:
    """The labeled (orient, L, gs, ts) must appear as a candidate in the
    active NC slot with matches == L - n_mismatches."""
    lbl = rec["labels"]
    if not lbl.get("is_positive"):
        return
    slot = lbl["active_noncoding_index"]
    if slot < 0:
        return
    L = lbl["guide_length"]
    gs = lbl["guide_span_in_active_noncoding"][0]
    ts = lbl["target_position_in_flank"][0]
    orient = "fwd" if lbl["match_orientation"] == "forward" else "rc"
    expected_matches = L - lbl["n_mismatches"]

    # Locate the (orient, L) block.
    orient_offset = 0 if orient == "fwd" else 12 * TOP_K_PER_COMBO_DEFAULT
    block_start = orient_offset + (L - 5) * TOP_K_PER_COMBO_DEFAULT
    block_end = block_start + TOP_K_PER_COMBO_DEFAULT

    # Search within this block for a candidate whose scalar features match.
    feats = out["candidate_features"][slot, block_start:block_end]
    mask = out["candidate_mask"][slot, block_start:block_end]
    found = False
    for i in range(TOP_K_PER_COMBO_DEFAULT):
        if not mask[i]:
            continue
        # scalar checks: L, matches, orient
        L_ok = int(feats[i, FEATURE_NAMES.index("L")]) == L
        matches_ok = int(feats[i, FEATURE_NAMES.index("matches")]) == expected_matches
        orient_ok = (
            feats[i, FEATURE_NAMES.index("orient_fwd")]
            == (1.0 if orient == "fwd" else 0.0)
        )
        # flank_start: derive by inverting flank_start_norm (which is / 120).
        flank_start_norm = feats[i, FEATURE_NAMES.index("flank_start_norm")]
        flank_start = int(round(flank_start_norm * out["flank_len"]))
        # nc_start: from nc_start_norm × nc_len.
        nc_len = out["nc_lengths"][slot]
        nc_start_norm = feats[i, FEATURE_NAMES.index("nc_start_norm")]
        nc_start = int(round(nc_start_norm * nc_len))
        if L_ok and matches_ok and orient_ok and flank_start == ts and nc_start == gs:
            found = True
            break
    assert found, (
        f"[{rec['site_id']}] ground truth (orient={orient} L={L} gs={gs} ts={ts} "
        f"matches={expected_matches}) not found in candidate block "
        f"[{block_start}, {block_end})"
    )


def main():
    if not os.path.exists(STRUCTURE_INDEX_SMOKE):
        print(f"[skip all] no structure cache at {STRUCTURE_INDEX_SMOKE}")
        return

    cache = StructureCache(STRUCTURE_INDEX_SMOKE)

    print("shape / dtype / mask invariants on 50 records ...", end=" ")
    n = 0
    for rec in load_head(VAL_SPLIT, 50):
        out = preprocess_site(rec, structure_cache=cache)
        check_shapes(out)
        check_unpopulated_slots(out)
        n += 1
    print(f"ok ({n})")

    print("ground-truth candidate present on positive records ...", end=" ")
    n_pos_checked = 0
    for rec in load_head(VAL_SPLIT, 100):
        if not rec["labels"].get("is_positive"):
            continue
        out = preprocess_site(rec, structure_cache=cache)
        check_ground_truth_candidate(rec, out)
        n_pos_checked += 1
        if n_pos_checked >= 20:
            break
    assert n_pos_checked > 0, "no positives in first 100 val records"
    print(f"ok ({n_pos_checked})")

    print("error path (no structure_cache) ...", end=" ")
    with open(VAL_SPLIT) as f:
        rec = json.loads(f.readline())
    try:
        preprocess_site(rec)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "structure_cache" in str(e), f"wrong error: {e}"
    print("ok")

    print("throughput: 50 records ...", end=" ")
    t0 = time.time()
    for rec in load_head(VAL_SPLIT, 50):
        preprocess_site(rec, structure_cache=cache)
    dt = time.time() - t0
    print(f"ok ({dt*1000:.0f} ms, {dt*20:.1f} ms/record)")
    assert dt < 30.0, f"preprocess too slow: {dt:.2f} s for 50 records"


if __name__ == "__main__":
    main()
