"""Verify pairwise_alignment_array on real dataset records.

For every positive we probe: the (guide_span, target_span, orientation) from
`labels` must correspond to a match-count equal to `guide_length - n_mismatches`
in the appropriate windowed matrix. This is the ground-truth alignment; if the
preprocess is correct it must be exactly reproducible.

For negatives with `no_alignment*` violation profiles we additionally check
that the ground-truth cell in the ORIGINAL positive coordinate no longer scores
`guide_length` — i.e. alignment truly was destroyed. (Skipped when a profile
sets guide_dna / active_noncoding_index to null.)
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocess.alignment import (
    alignment_feature_stack,
    direction_fusion,
    dot_plot,
    encode_dna,
    one_hot_dna,
    pairwise_alignment_array,
    perfect_seed_density,
    rc_flank_pos_to_flank_pos,
    revcomp,
    windowed_matches,
)

POSITIVES = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives.jsonl"
NEGATIVES = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives.jsonl"


def check_positive(rec: dict) -> None:
    lbl = rec["labels"]
    flank = rec["inputs"]["flank"]
    ncs = rec["inputs"]["noncoding_regions"]
    nc = ncs[lbl["active_noncoding_index"]]
    L = lbl["guide_length"]
    gs, ge = lbl["guide_span_in_active_noncoding"]
    ts, te = lbl["target_position_in_flank"]
    assert ge - gs == L, f"guide span length mismatch: {ge - gs} vs {L}"
    assert te - ts == L, f"target span length mismatch: {te - ts} vs {L}"

    arr = pairwise_alignment_array(nc, flank, L_min=L, L_max=L)
    orient = lbl["match_orientation"]
    expected = L - lbl["n_mismatches"]

    if orient == "forward":
        score = int(arr["fwd_L"][L][gs, ts])
    else:
        j_rc = rc_flank_pos_to_flank_pos(ts, L, len(flank))
        # sanity: mapping is its own inverse
        assert rc_flank_pos_to_flank_pos(j_rc, L, len(flank)) == ts
        score = int(arr["rc_L"][L][gs, j_rc])

    assert score == expected, (
        f"[{rec['site_id']}] {orient} L={L} n_mm={lbl['n_mismatches']} "
        f"expected {expected} matches at (gs={gs}, ts={ts}), got {score}"
    )

    # Group A feature stack: same ground-truth cell should read
    # (L - n_mismatches) / L in flank coordinates.
    stack = alignment_feature_stack(nc, flank, L_min=L, L_max=L, include_groups=("A",))
    m = stack["map"]
    names = stack["channel_names"]
    stack_ch = f"fwd_L{L}" if orient == "forward" else f"rc_L{L}"
    stack_score = float(m[gs, ts, names.index(stack_ch)])
    expected_norm = expected / L
    assert abs(stack_score - expected_norm) < 1e-6, (
        f"[{rec['site_id']}] stack {stack_ch} at (gs={gs}, ts={ts}) got "
        f"{stack_score} expected {expected_norm}"
    )

    guide_dna = lbl["guide_dna"]
    target_dna = lbl["target_dna"]
    assert nc[gs:ge] == guide_dna, f"NC guide slice != labels.guide_dna"
    assert flank[ts:te] == target_dna, f"flank target slice != labels.target_dna"
    if orient == "reverse_complement":
        # guide_dna should be revcomp(target_dna) up to n_mismatches
        rc_target = revcomp(target_dna)
        mm = sum(1 for a, b in zip(guide_dna, rc_target) if a != b)
        assert mm == lbl["n_mismatches"], f"RC mismatch count mismatch: {mm} vs {lbl['n_mismatches']}"
    else:
        mm = sum(1 for a, b in zip(guide_dna, target_dna) if a != b)
        assert mm == lbl["n_mismatches"], f"fwd mismatch count mismatch: {mm} vs {lbl['n_mismatches']}"


def check_negative_alignment_broken(rec: dict) -> None:
    """For no_alignment* negatives, guide_dna is random. It should score
    strictly fewer than `guide_length` matches at the labeled cell (else it
    would coincidentally be a perfect match, which is possible but rare).
    We just check the guide_dna vs perfect_guide_dna disagree — the score is
    then trivially < L."""
    lbl = rec["labels"]
    prof = lbl.get("violation_profile", "")
    if not prof.startswith("no_alignment"):
        return
    if lbl.get("guide_dna") is None:
        return  # no_active_noncoding
    if lbl.get("perfect_guide_dna") is None:
        return
    assert lbl["guide_dna"] != lbl["perfect_guide_dna"], (
        f"[{rec['site_id']}] no_alignment profile but guide == perfect guide"
    )


def check_alignment_only(rec: dict) -> None:
    """alignment_only: guide perfectly pairs with target. Score at the labeled
    cell should equal guide_length (assuming spans/orientation are meaningful).
    Also verify the feature-stack cell reads 1.0.
    """
    lbl = rec["labels"]
    if lbl.get("violation_profile") != "alignment_only":
        return
    if lbl.get("guide_dna") is None or lbl.get("active_noncoding_index", -1) < 0:
        return
    flank = rec["inputs"]["flank"]
    nc = rec["inputs"]["noncoding_regions"][lbl["active_noncoding_index"]]
    L = lbl["guide_length"]
    gs = lbl["guide_span_in_active_noncoding"][0]
    ts = lbl["target_position_in_flank"][0]
    arr = pairwise_alignment_array(nc, flank, L_min=L, L_max=L)
    if lbl["match_orientation"] == "forward":
        score = int(arr["fwd_L"][L][gs, ts])
    else:
        j_rc = rc_flank_pos_to_flank_pos(ts, L, len(flank))
        score = int(arr["rc_L"][L][gs, j_rc])
    # alignment_only means guide DOES pair; but the schema says target may be
    # at a random position - the labeled positions still describe the guide
    # and target that pair with 0 mismatches.
    assert score == L, (
        f"[{rec['site_id']}] alignment_only expected {L} matches, got {score}"
    )

    stack = alignment_feature_stack(nc, flank, L_min=L, L_max=L, include_groups=("A",))
    names = stack["channel_names"]
    ch = f"fwd_L{L}" if lbl["match_orientation"] == "forward" else f"rc_L{L}"
    stack_score = float(stack["map"][gs, ts, names.index(ch)])
    assert abs(stack_score - 1.0) < 1e-6, (
        f"[{rec['site_id']}] alignment_only stack expected 1.0 got {stack_score}"
    )


def stack_shape_smoke() -> None:
    """Group A stack: 20 channels, correct shape, correct channel order."""
    nc = "ACGT" * 60  # 240 bp
    flank = "A" * 120
    stack = alignment_feature_stack(nc, flank, L_min=8, L_max=16, include_groups=("A",))
    assert stack["map"].shape == (240, 120, 20), f"shape {stack['map'].shape}"
    assert stack["map"].dtype == np.float32
    expected_names = ["fwd_dot", "rc_dot_flank"]
    for L in range(8, 17):
        expected_names.append(f"fwd_L{L}")
        expected_names.append(f"rc_L{L}")
    assert stack["channel_names"] == expected_names, "channel order mismatch"


def stack_full_group_smoke() -> None:
    """Groups A+B+C: 20 + 14 + 1 = 35 channels."""
    nc = "ACGT" * 60
    flank = "A" * 120
    stack = alignment_feature_stack(nc, flank, L_min=8, L_max=16, include_groups=("A", "B", "C"))
    assert stack["map"].shape == (240, 120, 35), f"shape {stack['map'].shape}"
    names = stack["channel_names"]
    # Group A first
    assert names[0] == "fwd_dot" and names[1] == "rc_dot_flank"
    # Group B block
    b_start = 20
    for k, base in enumerate("ACGT"):
        assert names[b_start + k] == f"nc_{base}"
        assert names[b_start + 4 + k] == f"flank_{base}"
        assert names[b_start + 8 + k] == f"rcflank_{base}"
    assert names[b_start + 12] == "dir_fusion_0"
    assert names[b_start + 13] == "dir_fusion_1"
    # Group C
    assert names[-1] == "rel_pos"

    # NC one-hot channel for 'A' should be 1 exactly where nc[i]=='A'
    m = stack["map"]
    nc_A = m[..., names.index("nc_A")]
    for i, c in enumerate(nc):
        expected = 1.0 if c == "A" else 0.0
        assert nc_A[i, 0] == expected, f"nc_A at ({i}, 0) got {nc_A[i, 0]} expected {expected}"

    # Flank is all A, so flank_A should be 1 everywhere
    flank_A = m[..., names.index("flank_A")]
    assert np.all(flank_A == 1.0), "flank_A should be 1 everywhere for all-A flank"

    # rel_pos at (0, 0) = 0; at (nc_len-1, 0) = (nc_len-1)/(nc_len+flank_len)
    rp = m[..., names.index("rel_pos")]
    assert rp[0, 0] == 0.0
    assert abs(rp[239, 0] - 239.0 / 360.0) < 1e-6
    assert abs(rp[0, 119] - (-119.0 / 360.0)) < 1e-6


def one_hot_smoke() -> None:
    oh = one_hot_dna("ACGTN")
    assert oh.shape == (5, 4)
    assert np.array_equal(oh[0], [1, 0, 0, 0])
    assert np.array_equal(oh[1], [0, 1, 0, 0])
    assert np.array_equal(oh[2], [0, 0, 1, 0])
    assert np.array_equal(oh[3], [0, 0, 0, 1])
    assert np.array_equal(oh[4], [0, 0, 0, 0]), "N should be all zeros"


def direction_fusion_ambiguity_smoke() -> None:
    """The CRISPR-MFH argument: after plain OR-fusion, TC == CT, but direction
    channels must distinguish them.
    """
    # NC = 'T' at pos 0, flank = 'C' at pos 0  -> pair (T, C)
    # NC = 'C' at pos 0, flank = 'T' at pos 0  -> pair (C, T)
    tc = direction_fusion(encode_dna("T"), encode_dna("C"))  # (1, 1, 2)
    ct = direction_fusion(encode_dna("C"), encode_dna("T"))
    assert tc.shape == (1, 1, 2)
    assert not np.array_equal(tc, ct), "direction_fusion must disambiguate TC vs CT"
    # And they must be exact opposites (antisymmetric under nc<->flank swap)
    assert np.array_equal(tc, -ct), "direction_fusion should be antisymmetric"

    # A match cell (A vs A) should be all zero.
    aa = direction_fusion(encode_dna("A"), encode_dna("A"))
    assert np.all(aa == 0.0), "match cells should produce zero directional signature"

    # N should produce zero signature regardless of partner.
    an = direction_fusion(encode_dna("A"), encode_dna("N"))
    na = direction_fusion(encode_dna("N"), encode_dna("A"))
    assert np.all(an == 0.0) and np.all(na == 0.0), "N cells should be zero"


def seed_density_smoke() -> None:
    """One perfect L=5 seed at (i, j) should raise density in a
    (2r+1)^2 neighborhood by exactly 1; elsewhere unchanged."""
    nc = "AAAAA" + "T" * 100
    flank = "AAAAA" + "G" * 100
    stack = alignment_feature_stack(
        nc, flank, L_min=8, L_max=16, include_groups=("D",), seed_lengths=(5,), seed_radius=3,
    )
    m = stack["map"]
    names = stack["channel_names"]
    ch = names.index("fwd_seed_dens_L5_r3")
    dens = m[..., ch]
    # Neighborhood: (2*3+1)^2 = 49. One seed -> value 1/49 at every cell whose
    # box contains anchor (0, 0). The anchor is at (0, 0) => cells in
    # i∈[0, 3], j∈[0, 3] have density 1/49.
    assert abs(dens[0, 0] - 1.0 / 49.0) < 1e-6
    assert abs(dens[3, 3] - 1.0 / 49.0) < 1e-6
    assert dens[4, 4] == 0.0, "beyond radius should be 0"


def short_seed_group_D_shape_smoke() -> None:
    """Group D shape and channel naming: 2 * len(seed_lengths) channels."""
    stack = alignment_feature_stack(
        "ACGT" * 20, "ACGT" * 30, include_groups=("D",),
        seed_lengths=(5, 6), seed_radius=8,
    )
    assert stack["map"].shape[-1] == 4
    assert stack["channel_names"] == [
        "fwd_seed_dens_L5_r8",
        "rc_seed_dens_L5_r8",
        "fwd_seed_dens_L6_r8",
        "rc_seed_dens_L6_r8",
    ]


def short_L_smoke() -> None:
    """Lowering L_min exposes fwd_L5, rc_L5, ... in Group A."""
    stack = alignment_feature_stack(
        "ACGT" * 20, "ACGT" * 30, L_min=5, L_max=8, include_groups=("A",),
    )
    for L in range(5, 9):
        assert f"fwd_L{L}" in stack["channel_names"]
        assert f"rc_L{L}" in stack["channel_names"]


def seed_density_matches_manual_count() -> None:
    """On a random pair, the fwd_seed_dens channel equals a brute-force
    neighborhood count of perfect L_seed anchors."""
    rng = np.random.default_rng(1)
    nc = "".join(rng.choice(list("ACGT"), size=100))
    flank = "".join(rng.choice(list("ACGT"), size=80))
    L_seed = 5
    r = 4
    stack = alignment_feature_stack(
        nc, flank, L_min=8, L_max=16, include_groups=("D",),
        seed_lengths=(L_seed,), seed_radius=r,
    )
    m = stack["map"]
    ch = stack["channel_names"].index(f"fwd_seed_dens_L{L_seed}_r{r}")
    fwd_dens = m[..., ch]

    # Independent brute-force: enumerate perfect seed anchors.
    fwd_dot, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd_dot, L_seed)
    anchors = np.argwhere(win == L_seed)  # (K, 2) in (nc, flank) coords
    H, W = len(nc), len(flank)
    manual = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            i0 = max(0, i - r); i1 = min(H, i + r + 1)
            j0 = max(0, j - r); j1 = min(W, j + r + 1)
            count = 0
            for ai, aj in anchors:
                if i0 <= ai < i1 and j0 <= aj < j1:
                    count += 1
            manual[i, j] = count
    neighborhood = (2 * r + 1) ** 2
    assert np.allclose(fwd_dens, manual.astype(np.float32) / neighborhood), (
        "fwd_seed_dens disagrees with brute-force neighborhood count"
    )


def stack_matches_array_smoke() -> None:
    """Group A windowed channels equal the standalone pairwise_alignment_array
    values at every anchor position (up to normalization and rc→flank flip)."""
    rng = np.random.default_rng(0)
    nc = "".join(rng.choice(list("ACGT"), size=200))
    flank = "".join(rng.choice(list("ACGT"), size=120))
    arr = pairwise_alignment_array(nc, flank, L_min=8, L_max=16)
    stack = alignment_feature_stack(nc, flank, L_min=8, L_max=16, include_groups=("A",))
    m = stack["map"]
    names = stack["channel_names"]
    flank_len = 120
    for L in range(8, 17):
        f_ch = m[..., names.index(f"fwd_L{L}")]
        r_ch = m[..., names.index(f"rc_L{L}")]
        f_win_h, f_win_w = 200 - L + 1, flank_len - L + 1
        # fwd: direct match after normalization
        assert np.allclose(
            f_ch[:f_win_h, :f_win_w], arr["fwd_L"][L].astype(np.float32) / L, atol=1e-6
        ), f"fwd_L{L} mismatch"
        # rc: stack is flank-coord; array is rc-coord. flank_coord[j] = rc_coord[flank_L-1-j]
        expected_rc = arr["rc_L"][L].astype(np.float32) / L
        expected_rc_flank = expected_rc[:, ::-1]
        assert np.allclose(
            r_ch[:f_win_h, :f_win_w], expected_rc_flank, atol=1e-6
        ), f"rc_L{L} mismatch"


def dot_plot_smoke() -> None:
    """Tiny sanity check: hand-computed dot plot."""
    nc = "ACGTAC"
    flank = "GTACGT"
    fwd, rc = dot_plot(nc, flank)
    # forward: nc[i]==flank[j]
    expected_fwd = np.array(
        [[c == d for d in flank] for c in nc], dtype=bool
    )
    assert np.array_equal(fwd, expected_fwd), "forward dot plot mismatch"
    rc_flank = revcomp(flank)  # ACGTAC
    expected_rc = np.array([[c == d for d in rc_flank] for c in nc], dtype=bool)
    assert np.array_equal(rc, expected_rc), "rc dot plot mismatch"
    # revcomp of "GTACGT" is "ACGTAC" so nc == rc_flank, diagonal all True
    assert np.all(np.diag(rc)), "rc self-alignment should be perfect"


def load_head(path: str, n: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            yield json.loads(line)


def main():
    print("dot plot smoke ...", end=" ")
    dot_plot_smoke()
    print("ok")

    print("stack shape smoke ...", end=" ")
    stack_shape_smoke()
    print("ok")

    print("one-hot smoke ...", end=" ")
    one_hot_smoke()
    print("ok")

    print("direction fusion ambiguity smoke ...", end=" ")
    direction_fusion_ambiguity_smoke()
    print("ok")

    print("full-group (A+B+C) stack smoke ...", end=" ")
    stack_full_group_smoke()
    print("ok")

    print("short-seed density smoke ...", end=" ")
    seed_density_smoke()
    print("ok")

    print("Group D shape smoke ...", end=" ")
    short_seed_group_D_shape_smoke()
    print("ok")

    print("short-L (L_min=5) smoke ...", end=" ")
    short_L_smoke()
    print("ok")

    print("seed density vs brute-force count ...", end=" ")
    seed_density_matches_manual_count()
    print("ok")

    print("stack vs standalone-array smoke ...", end=" ")
    stack_matches_array_smoke()
    print("ok")

    n_pos = 500
    print(f"positives: checking first {n_pos} records ...", end=" ")
    n_fwd = n_rc = 0
    for rec in load_head(POSITIVES, n_pos):
        check_positive(rec)
        if rec["labels"]["match_orientation"] == "forward":
            n_fwd += 1
        else:
            n_rc += 1
    print(f"ok (fwd={n_fwd}, rc={n_rc})")

    n_neg = 500
    print(f"negatives: checking first {n_neg} records ...", end=" ")
    prof_counts = {}
    for rec in load_head(NEGATIVES, n_neg):
        prof = rec["labels"].get("violation_profile", "?")
        prof_counts[prof] = prof_counts.get(prof, 0) + 1
        check_negative_alignment_broken(rec)
        check_alignment_only(rec)
    print(f"ok")
    print("  profile distribution in first 500 negatives:")
    for p, c in sorted(prof_counts.items()):
        print(f"    {p}: {c}")


if __name__ == "__main__":
    main()
