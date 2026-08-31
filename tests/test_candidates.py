"""Tests for preprocess/candidates.py — the alignment-aware structure-patch
generator.

Checks:
  1. Shape / layout / dtype invariants.
  2. Ground truth: on a positive record, the labeled (orient, L, gs, ts) is
     the top-1 (or a top-few) candidate in the correct (orient, L) combo, its
     patch structure equals the cache slice, guide mask lights up exactly on
     the guide span, match_state matches the labeled n_mismatches, and the
     paired_flank_pos + align_position channels are correct.
  3. Feature vector: orient one-hot, L, matches, mismatches, score,
     flank position / boundary fields all correct.
  4. Boundary: for a synthetic guide near the NC boundary, patch pads with 0
     and struct_valid drops to 0 in the padded region.
  5. Padding to K_max: (orient, L) combos with fewer real candidates leave
     mask=False slots; their patches/features are zero.
  6. RC coordinate handling: for an rc positive, paired_flank_pos decreases
     along guide offset (guide 0 pairs with a HIGHER flank pos than guide L-1).

Requires the smoke structure mmap at /tmp/nc_unp_smoke.index.json (build
with `python -m scripts.precompute_structure --split val.jsonl --limit 200
--out /tmp/nc_unp_smoke`).
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocess.candidates import (
    FEATURE_NAMES,
    NUM_FEATURES,
    PATCH_CHANNEL_NAMES,
    PATCH_CHANNELS,
    PATCH_WIDTH_DEFAULT,
    TOP_K_PER_COMBO_DEFAULT,
    build_candidate_arrays,
    k_max,
)
from preprocess.site import StructureCache

VAL_SPLIT = "/global/scratch/users/kh36969/DL_novel_guide_editor/splits/val.jsonl"
NEGATIVES = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives.jsonl"
STRUCTURE_INDEX_SMOKE = os.environ.get(
    "STRUCTURE_INDEX", "/tmp/nc_unp_smoke.index.json"
)


def _load_smoke_val_recs(n: int) -> list[dict]:
    recs = []
    with open(VAL_SPLIT) as f:
        for line in f:
            recs.append(json.loads(line))
            if len(recs) >= n:
                break
    return recs


def shape_layout_smoke():
    """Trivial random NC + flank; check shape/dtype/layout invariants."""
    rng = np.random.default_rng(0)
    nc = "".join(rng.choice(list("ACGT"), size=200))
    flank = "".join(rng.choice(list("ACGT"), size=120))
    # Fake structure: uniform 0.5 profile, all valid.
    prof = np.full((200, 16), 0.5, dtype=np.float32)
    valid = np.ones((200, 16), dtype=bool)
    K = k_max()
    assert K == 2 * 12 * TOP_K_PER_COMBO_DEFAULT == 96

    patches, feats, mask, cands = build_candidate_arrays(nc, flank, prof, valid)
    assert patches.shape == (K, PATCH_WIDTH_DEFAULT, PATCH_CHANNELS), patches.shape
    assert feats.shape == (K, NUM_FEATURES), feats.shape
    assert mask.shape == (K,) and mask.dtype == bool
    assert len(cands) == K
    # Random NC big enough that every (orient, L) fills its top-K.
    assert mask.all(), "expected all K slots filled on a 200bp NC"

    # Layout: first 12*4 slots are fwd, next 12*4 are rc.
    for i in range(48):
        assert cands[i].orient == "fwd", cands[i]
    for i in range(48, 96):
        assert cands[i].orient == "rc", cands[i]
    # Within one (orient, L) block, top-K sorted descending by matches.
    for start in range(0, K, TOP_K_PER_COMBO_DEFAULT):
        block = [cands[start + i] for i in range(TOP_K_PER_COMBO_DEFAULT)]
        scores = [c.matches for c in block]
        assert scores == sorted(scores, reverse=True), (
            f"block starting at {start} not sorted desc: {scores}"
        )
        # All same L within the block.
        Ls = {c.L for c in block}
        assert len(Ls) == 1, f"block at {start} has mixed L: {Ls}"


def ground_truth_positive_fwd():
    """The labeled (orient=fwd, L, gs, ts) of a positive must appear
    among the top candidates of its (orient, L) block, with matches
    = L - n_mismatches, and its patch must faithfully embed the cache
    slice + correct alignment channels."""
    cache = StructureCache(STRUCTURE_INDEX_SMOKE)
    found = None
    for rec in _load_smoke_val_recs(50):
        lbl = rec["labels"]
        if lbl.get("is_positive") and lbl["match_orientation"] == "forward":
            found = rec
            break
    assert found is not None, "no fwd positive in first 50 val records"
    rec = found
    lbl = rec["labels"]
    slot = lbl["active_noncoding_index"]
    nc = rec["inputs"]["noncoding_regions"][slot]
    flank = rec["inputs"]["flank"]
    L = lbl["guide_length"]
    gs = lbl["guide_span_in_active_noncoding"][0]
    ts = lbl["target_position_in_flank"][0]
    n_mm = lbl["n_mismatches"]

    profile, valid = cache.get(rec["site_id"], slot, len(nc))
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, profile, valid,
    )

    # Find the ground-truth candidate in its (fwd, L) block.
    orient_L_start = 0 + (L - 5) * TOP_K_PER_COMBO_DEFAULT  # fwd block first
    block_end = orient_L_start + TOP_K_PER_COMBO_DEFAULT
    match_idx = None
    for i in range(orient_L_start, block_end):
        c = cands[i]
        if c is not None and c.orient == "fwd" and c.L == L and c.nc_start == gs and c.flank_start == ts:
            match_idx = i
            break
    assert match_idx is not None, (
        f"[{rec['site_id']}] labeled fwd L={L} gs={gs} ts={ts} not in "
        f"top-{TOP_K_PER_COMBO_DEFAULT} of its block; candidates were: "
        f"{[cands[i] for i in range(orient_L_start, block_end)]}"
    )
    c = cands[match_idx]
    assert c.matches == L - n_mm

    # Patch structure channels equal the cache slice at overlapping positions.
    W = PATCH_WIDTH_DEFAULT
    guide_center = c.nc_start + L // 2
    patch_start = guide_center - W // 2
    # First check within-NC region.
    for p in range(W):
        nc_pos = patch_start + p
        if 0 <= nc_pos < len(nc):
            assert np.allclose(patches[match_idx, p, :16], profile[nc_pos, :16], atol=1e-6), (
                f"unp mismatch at patch pos {p} (nc pos {nc_pos})"
            )
            assert patches[match_idx, p, 16] == float(valid[nc_pos, 0])
        else:
            # Padded region: struct_valid must be 0.
            assert patches[match_idx, p, 16] == 0.0, f"padded pos {p} has struct_valid != 0"

    # Guide mask lights up EXACTLY on the guide span.
    gm = patches[match_idx, :, 17]
    guide_slots = [p for p in range(W) if 0 <= patch_start + p < len(nc)
                    and gs <= patch_start + p < gs + L]
    for p in range(W):
        expected = 1.0 if p in guide_slots else 0.0
        assert gm[p] == expected, f"guide_mask[{p}] = {gm[p]}, expected {expected}"

    # match_state channels: match_match + match_mismatch inside guide, both 0 outside.
    mm_ch = patches[match_idx, :, 18]
    mmiss_ch = patches[match_idx, :, 19]
    total_matches = int(mm_ch.sum())
    total_mm = int(mmiss_ch.sum())
    assert total_matches == L - n_mm, f"expected {L - n_mm} matches, got {total_matches}"
    assert total_mm == n_mm, f"expected {n_mm} mismatches, got {total_mm}"
    # Outside guide -> both must be 0.
    outside = [p for p in range(W) if p not in guide_slots]
    assert (mm_ch[outside] == 0).all() and (mmiss_ch[outside] == 0).all()

    # Paired flank pos: fwd -> increases along guide offset.
    pfp = patches[match_idx, :, 20]
    prev = -1.0
    for p in guide_slots:
        cur = pfp[p]
        assert cur > prev, f"fwd paired flank should increase, got {prev}->{cur}"
        prev = cur

    # Align position in guide: linear 0..1 across guide.
    ap = patches[match_idx, :, 21]
    assert ap[guide_slots[0]] == 0.0
    assert abs(ap[guide_slots[-1]] - 1.0) < 1e-6

    # Scalar features.
    fv = feats[match_idx]
    assert fv[FEATURE_NAMES.index("orient_fwd")] == 1.0
    assert fv[FEATURE_NAMES.index("orient_rc")] == 0.0
    assert fv[FEATURE_NAMES.index("L")] == float(L)
    assert fv[FEATURE_NAMES.index("matches")] == float(L - n_mm)
    assert fv[FEATURE_NAMES.index("mismatches")] == float(n_mm)
    assert abs(fv[FEATURE_NAMES.index("score")] - (L - n_mm) / L) < 1e-6


def ground_truth_positive_rc():
    """Same as fwd but for a reverse-complement positive: paired_flank_pos
    must DECREASE along guide offset, matches count still equals L-n_mm."""
    cache = StructureCache(STRUCTURE_INDEX_SMOKE)
    found = None
    for rec in _load_smoke_val_recs(50):
        lbl = rec["labels"]
        if lbl.get("is_positive") and lbl["match_orientation"] == "reverse_complement":
            found = rec
            break
    assert found is not None, "no rc positive in first 50 val records"
    rec = found
    lbl = rec["labels"]
    slot = lbl["active_noncoding_index"]
    nc = rec["inputs"]["noncoding_regions"][slot]
    flank = rec["inputs"]["flank"]
    L = lbl["guide_length"]
    gs = lbl["guide_span_in_active_noncoding"][0]
    ts = lbl["target_position_in_flank"][0]
    n_mm = lbl["n_mismatches"]

    profile, valid = cache.get(rec["site_id"], slot, len(nc))
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, profile, valid,
    )

    # RC block starts at index 48 (48 fwd slots first).
    block_start = 48 + (L - 5) * TOP_K_PER_COMBO_DEFAULT
    block_end = block_start + TOP_K_PER_COMBO_DEFAULT
    match_idx = None
    for i in range(block_start, block_end):
        c = cands[i]
        if c is not None and c.orient == "rc" and c.L == L and c.nc_start == gs and c.flank_start == ts:
            match_idx = i
            break
    assert match_idx is not None, (
        f"[{rec['site_id']}] labeled rc L={L} gs={gs} ts={ts} not in "
        f"top-{TOP_K_PER_COMBO_DEFAULT} of its block; candidates were: "
        f"{[cands[i] for i in range(block_start, block_end)]}"
    )
    c = cands[match_idx]
    assert c.matches == L - n_mm

    # Paired flank pos DECREASES along guide offset for RC.
    W = PATCH_WIDTH_DEFAULT
    patch_start = (c.nc_start + L // 2) - W // 2
    guide_slots = [p for p in range(W) if gs <= patch_start + p < gs + L
                    and 0 <= patch_start + p < len(nc)]
    pfp = patches[match_idx, :, 20]
    prev = 2.0  # anything > 1
    for p in guide_slots:
        cur = pfp[p]
        assert cur < prev, f"rc paired flank should decrease, got {prev}->{cur}"
        prev = cur

    # Match counts still consistent.
    mm_ch = patches[match_idx, :, 18]
    mmiss_ch = patches[match_idx, :, 19]
    assert int(mm_ch.sum()) == L - n_mm
    assert int(mmiss_ch.sum()) == n_mm

    # Feature vector: rc one-hot.
    fv = feats[match_idx]
    assert fv[FEATURE_NAMES.index("orient_fwd")] == 0.0
    assert fv[FEATURE_NAMES.index("orient_rc")] == 1.0


def boundary_padding():
    """Guide placed near the NC boundary must produce a patch where struct_valid
    is 0 in the padded region and 1 in the real region."""
    nc = "A" * 20 + "CCCCCCCC" + "A" * 20  # guide is at pos 20..28, near left edge? no, symmetric
    # Shift so guide is at the start: pos 0..8
    nc = "CCCCCCCC" + "A" * 40  # nc_len = 48; guide at pos 0..8
    flank = "GGGGGGGG" + "T" * 112  # flank length 120; target matches "CCCCCCCC" at pos 0
    # ... but this only works if fwd matches. C vs G no. Let's flip:
    flank = "CCCCCCCC" + "T" * 112  # target = CCCCCCCC at flank[0:8]
    # Structure: uniform 0.5 for all 48 positions.
    profile = np.full((48, 16), 0.5, dtype=np.float32)
    valid = np.ones((48, 16), dtype=bool)
    patches, feats, mask, cands = build_candidate_arrays(
        nc, flank, profile, valid, patch_width=64,
    )
    # Find the L=8 fwd top-1 candidate (should be nc_start=0, flank_start=0, matches=8).
    L = 8
    orient_L_start = 0 + (L - 5) * TOP_K_PER_COMBO_DEFAULT
    c = cands[orient_L_start]
    assert c is not None
    assert c.orient == "fwd" and c.L == L and c.matches == L, c
    assert c.nc_start == 0 and c.flank_start == 0

    # guide_center = 4; patch_start = 4 - 32 = -28
    # So patch pos [0, 27] map to NC pos [-28, -1] (padded); [28, 47] map to NC pos [0, 19] (valid).
    struct_valid = patches[orient_L_start, :, 16]
    assert (struct_valid[:28] == 0).all(), "padded prefix should have struct_valid=0"
    # And within the real NC, struct_valid should be 1 (uniform valid array).
    real_positions = 64 - 28  # positions [28, 63]
    assert (struct_valid[28:] == 1).all(), "real positions should have struct_valid=1"


def padded_mask_when_low_variety():
    """If nc_len < L for some L, that (orient, L) block has fewer or zero
    candidates. Check that the mask correctly flags padded slots."""
    # nc_len = 6: only L∈{5, 6} produce valid windows. L∈{7..16} → 0 candidates.
    nc = "ACGTAC"
    flank = "ACGTAC" + "T" * 114  # flank_len = 120; matches nc at pos 0
    profile = np.full((6, 16), 0.5, dtype=np.float32)
    valid = np.ones((6, 16), dtype=bool)
    patches, feats, mask, cands = build_candidate_arrays(nc, flank, profile, valid)

    # For L=5, fwd block: nc_len - L + 1 = 2 possible nc_starts × (flank_len - L + 1) = 116
    #   -> more than K=4, so 4 real candidates.
    # For L=6, 1 nc_start × 115 -> 4 real candidates.
    # For L>=7, empty window -> 0 real candidates.
    n_orient = 2
    for orient_i in range(n_orient):
        for L in range(5, 17):
            block_start = orient_i * 12 * 4 + (L - 5) * 4
            block_slice = mask[block_start:block_start + 4]
            if L <= 6:
                assert block_slice.any(), f"L={L} should have some candidates"
            else:
                assert not block_slice.any(), f"L={L} block should be empty (nc_len=6)"
                # Also patches must be zero.
                assert (patches[block_start:block_start + 4] == 0).all()


def main():
    print("shape/layout smoke ...", end=" ")
    shape_layout_smoke()
    print("ok")

    if os.path.exists(STRUCTURE_INDEX_SMOKE):
        print("ground truth positive (fwd) ...", end=" ")
        ground_truth_positive_fwd()
        print("ok")
        print("ground truth positive (rc) ...", end=" ")
        ground_truth_positive_rc()
        print("ok")
    else:
        print(f"[skip] ground-truth tests — no structure cache at {STRUCTURE_INDEX_SMOKE}")

    print("boundary padding ...", end=" ")
    boundary_padding()
    print("ok")

    print("padded mask when nc_len too small ...", end=" ")
    padded_mask_when_low_variety()
    print("ok")


if __name__ == "__main__":
    main()
