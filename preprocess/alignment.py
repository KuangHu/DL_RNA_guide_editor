"""Pairwise short-alignment features between a non-coding region and the flank.

Given one non-coding region `nc` (variable length, ~140-270 bp) and the 120 bp
`flank`, we enumerate every possible short ungapped alignment in two
orientations:

  forward:            nc[i:i+L]  ==  flank[j:j+L]
  reverse_complement: nc[i:i+L]  ==  revcomp(flank[j:j+L])

For a range of guide lengths L (dataset uses 8..16).

Two representations are produced:

1. Base-level dot plot (per orientation):
       shape (len(nc), len(flank))
       fwd_dot[i, j] = 1 iff nc[i] == flank[j]
       rc_dot [i, j] = 1 iff nc[i] == revcomp(flank)[j]
   Any length-L ungapped alignment is a length-L diagonal streak in this matrix.

2. Windowed match counts (per orientation, per L):
       shape (len(nc) - L + 1, len(flank) - L + 1)
       match[i, j] = number of matching bases in the L-window starting at
                     (nc position i, flank position j) under the given orientation.
   This is the diagonal sum of the dot plot with a length-L box filter, and
   equals `L - n_mismatches` for the true guide/target alignment.

Coordinate note for RC:
   The RC dot plot / match matrix indexes into `revcomp(flank)`. To convert a
   position j' in rc-space back to the flank position of the L-window on the
   original flank, use `rc_flank_pos_to_flank_pos(j', L, len(flank))`.
"""

from __future__ import annotations

import numpy as np

_BASE_TO_INT = {"A": 0, "C": 1, "G": 2, "T": 3}
_N_CODE = 4
_COMPLEMENT = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")


def encode_dna(seq: str) -> np.ndarray:
    """DNA string -> int8 array (A=0, C=1, G=2, T=3, N/other=4).

    Case-insensitive. Any non-ACGT letter is coded as N (4) and will never
    match anything in the dot plot.
    """
    arr = np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)
    out = np.full(arr.shape, _N_CODE, dtype=np.int8)
    for b, i in _BASE_TO_INT.items():
        out[arr == ord(b)] = i
    return out


def revcomp(seq: str) -> str:
    return seq.translate(_COMPLEMENT)[::-1]


def one_hot_dna(seq: str) -> np.ndarray:
    """DNA string -> one-hot (len, 4) float32. A=0,C=1,G=2,T=3. N/other -> all zero."""
    codes = encode_dna(seq)
    out = np.zeros((len(codes), 4), dtype=np.float32)
    valid = codes < _N_CODE
    idx = np.arange(len(codes))[valid]
    out[idx, codes[valid]] = 1.0
    return out


def direction_fusion(nc_codes: np.ndarray, flank_codes: np.ndarray) -> np.ndarray:
    """CRISPR-MFH directional fusion. Two-channel signed indicator per (i, j) cell.

    Given int-coded NC (len N) and flank (len F) sequences, returns
    shape (N, F, 2) float32:

      channel 0: +1 if nc[i] < flank[j], -1 if nc[i] > flank[j], 0 if equal or N
      channel 1: +1 if nc[i] > flank[j], -1 if nc[i] < flank[j], 0 if equal or N

    This disambiguates the OR-fusion of one-hots (which cannot distinguish
    the pair TC from CT). Following Zhang 2025 (CRISPR-MFH, PMC12026807):
    adding a directional pair of channels preserves the ordering of the
    mismatch pair. N bases (code 4) produce all-zero signature.
    """
    nc = nc_codes.astype(np.int16)
    fl = flank_codes.astype(np.int16)
    diff = nc[:, None] - fl[None, :]                # (N, F)
    valid = (nc[:, None] < _N_CODE) & (fl[None, :] < _N_CODE)
    lt = ((diff < 0) & valid).astype(np.float32)
    gt = ((diff > 0) & valid).astype(np.float32)
    ch0 = lt - gt
    ch1 = gt - lt
    return np.stack([ch0, ch1], axis=-1)


def dot_plot(nc: str, flank: str) -> tuple[np.ndarray, np.ndarray]:
    """Base-level match matrices for one NC region vs the flank.

    Returns:
        fwd_dot: bool array (len(nc), len(flank)); nc[i] == flank[j]
        rc_dot:  bool array (len(nc), len(flank)); nc[i] == revcomp(flank)[j]
    N bases never match (they are coded 4 and compared against 0..3 codes).
    """
    nc_arr = encode_dna(nc)
    flank_arr = encode_dna(flank)
    rc_arr = encode_dna(revcomp(flank))

    valid_nc = nc_arr != _N_CODE
    fwd = (nc_arr[:, None] == flank_arr[None, :]) & valid_nc[:, None] & (flank_arr[None, :] != _N_CODE)
    rc = (nc_arr[:, None] == rc_arr[None, :]) & valid_nc[:, None] & (rc_arr[None, :] != _N_CODE)
    return fwd, rc


def windowed_matches(dot: np.ndarray, L: int) -> np.ndarray:
    """Length-L diagonal sum of a base-level dot plot.

    dot: 2D array (H, W). L: window length.

    Returns match_count[i, j] = sum_{k=0..L-1} dot[i+k, j+k], with shape
    (H - L + 1, W - L + 1). Returns an empty (0, 0) array if either side is
    too short.
    """
    H, W = dot.shape
    if H < L or W < L or L < 1:
        return np.zeros((max(0, H - L + 1), max(0, W - L + 1)), dtype=np.int32)
    d = dot.astype(np.int32, copy=False)
    out_h = H - L + 1
    out_w = W - L + 1
    out = np.zeros((out_h, out_w), dtype=np.int32)
    for k in range(L):
        out += d[k : k + out_h, k : k + out_w]
    return out


def _box_sum_2d(x: np.ndarray, radius: int) -> np.ndarray:
    """Neighborhood sum with Chebyshev radius, via 2D summed-area table.

    x: (H, W) int-like. Cells outside the array boundary are treated as 0.
    Returns int32 (H, W): out[i, j] = sum of x over the (2r+1)^2 box
    centered at (i, j), clipped at boundaries.
    """
    H, W = x.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.int32)
    ii[1:, 1:] = x.astype(np.int32, copy=False).cumsum(axis=0).cumsum(axis=1)
    r = radius
    i_idx = np.arange(H)
    j_idx = np.arange(W)
    i0 = np.clip(i_idx - r, 0, H)
    i1 = np.clip(i_idx + r + 1, 0, H)
    j0 = np.clip(j_idx - r, 0, W)
    j1 = np.clip(j_idx + r + 1, 0, W)
    return (
        ii[i1[:, None], j1[None, :]]
        - ii[i0[:, None], j1[None, :]]
        - ii[i1[:, None], j0[None, :]]
        + ii[i0[:, None], j0[None, :]]
    )


def perfect_seed_density(
    dot: np.ndarray,
    L_seed: int,
    radius: int,
) -> np.ndarray:
    """Local density of perfect L_seed matches, one number per (i, j) cell.

    A perfect L_seed match at anchor (i, j) means dot[i:i+L_seed, j:j+L_seed]
    has L_seed 1's along the main diagonal. We compute the L_seed-window
    match count via `windowed_matches`, threshold at == L_seed to get a
    binary "perfect-seed anchor" map, then sum inside a (2*radius+1) square
    neighborhood centered at each cell.

    Interpretation:
      - dense seeds (many perfect L_seed matches in one region) -> high value
      - isolated 5/5 hits scattered across the map -> ~baseline
      - captures BLAST / IntaRNA seed-clustering intuition

    Coordinate note: output is in the SAME coordinate frame as `dot`.
    For the reverse-complement dot plot in rc-flank coordinates, flip the
    perfect-seed indicator to flank coordinates BEFORE box-summing (that
    logic lives inline in `alignment_feature_stack`, not here).

    Args:
        dot: base-level dot plot (H, W), bool or int.
        L_seed: seed length (e.g. 5).
        radius: neighborhood radius (Chebyshev). Total window is (2r+1)^2.

    Returns:
        int32 (H, W) density map.
    """
    H, W = dot.shape
    perf_win = (windowed_matches(dot, L_seed) == L_seed).astype(np.int32)
    perf_full = np.zeros((H, W), dtype=np.int32)
    ph, pw = perf_win.shape
    perf_full[:ph, :pw] = perf_win
    return _box_sum_2d(perf_full, radius)


def pairwise_alignment_array(
    nc: str,
    flank: str,
    L_min: int = 5,
    L_max: int = 16,
) -> dict:
    """All possible short ungapped alignments between one NC region and the flank.

    Returns dict:
        'fwd_dot': (len(nc), len(flank)) bool  - base-level forward matches
        'rc_dot' : (len(nc), len(flank)) bool  - base-level RC matches (vs revcomp(flank))
        'fwd_L'  : {L: (len(nc)-L+1, len(flank)-L+1) int32 match counts}
        'rc_L'   : {L: same shape} in rc-flank coordinates
        'L_range': (L_min, L_max)
        'flank_len': int
        'nc_len': int
    """
    fwd_dot, rc_dot = dot_plot(nc, flank)
    fwd_L = {L: windowed_matches(fwd_dot, L) for L in range(L_min, L_max + 1)}
    rc_L = {L: windowed_matches(rc_dot, L) for L in range(L_min, L_max + 1)}
    return {
        "fwd_dot": fwd_dot,
        "rc_dot": rc_dot,
        "fwd_L": fwd_L,
        "rc_L": rc_L,
        "L_range": (L_min, L_max),
        "flank_len": len(flank),
        "nc_len": len(nc),
    }


def rc_flank_pos_to_flank_pos(j_rc: int, L: int, flank_len: int) -> int:
    """Map a start position in revcomp(flank) space back to the corresponding
    L-window start position in the original flank.

    An L-window starting at j_rc in revcomp(flank) covers rc_flank[j_rc:j_rc+L],
    which is revcomp(flank[flank_len - j_rc - L : flank_len - j_rc]). So the
    original flank window starts at `flank_len - j_rc - L`.
    """
    return flank_len - j_rc - L


def alignment_feature_stack(
    nc: str,
    flank: str,
    L_min: int = 5,
    L_max: int = 16,
    include_groups: tuple[str, ...] = ("A",),
    seed_lengths: tuple[int, ...] = (5, 6),
    seed_radius: int = 8,
    nc_structure: np.ndarray | None = None,
    nc_structure_valid: np.ndarray | None = None,
) -> dict:
    """Stacked 2D feature tensor per (nc_pos, flank_pos) cell.

    All channels are laid out on a common (nc_len, flank_len) grid where
    axis-0 is the NC position and axis-1 is the position in the ORIGINAL
    flank (not revcomp). RC channels are converted to flank coordinates by
    flipping the rc-flank axis, so forward and RC channels overlay on the
    same 2D map.

    Groups selected via include_groups:

      'A' - Alignment scores (Cas-OFFinder / dot plot). Default.
         fwd_dot        - base-level fwd match indicator (nc[i] == flank[j])
         rc_dot_flank   - base-level RC pairability   (nc[i] == complement(flank[j]))
         fwd_L{L}       - fwd L-window match count / L,  anchor at (nc_start, flank_start)
         rc_L{L}        - RC  L-window match count / L,  anchor at (nc_start, flank_start)
                          in flank coordinates
         (2 + 2 * (L_max-L_min+1) channels; default range 5..16 -> 26 channels
          — includes short "seed" lengths 5, 6, 7 alongside the guide-length
          range 8..16 so a seed-and-extend head can see clustered short hits)

      'B' - Sequence identity + directional fusion (CRISPR-MFH-inspired):
         nc_A, nc_C, nc_G, nc_T       - NC one-hot broadcast over flank axis
         flank_A..T                   - flank one-hot broadcast over NC axis
         rcflank_A..T                 - revcomp(flank) one-hot broadcast (for RC-oriented heads)
         dir_fusion_0, dir_fusion_1   - CRISPR-MFH directional pair signature
                                        (disambiguates TC vs CT after any OR fusion)
         (14 channels)

      'C' - Positional prior:
         rel_pos                      - (nc_pos - flank_pos) / (nc_len + flank_len);
                                        diagonal streaks (fwd alignments) lie on constant-value stripes
         (1 channel)

      'D' - Short-seed density (BLAST / IntaRNA seed intuition):
         fwd_seed_dens_L{Ls}_r{r}    - count of perfect fwd L_seed matches whose anchor
                                        lies within Chebyshev distance `seed_radius` of (i, j)
         rc_seed_dens_L{Ls}_r{r}     - same for RC (in flank coordinates)
         One pair per L_seed in `seed_lengths` (default (5, 6)). Radius default = 8.
         Isolated short seeds are noise; a cluster of them in one region -> strong signal
         even when the guide has 2-3 mismatches (so no long window ever scores near 1.0).
         (2 * len(seed_lengths) channels; default -> 4 channels)

      'E' - Per-nt NC structure (RNAplfold), broadcast over flank axis:
         nc_unp_u{k}                  - P([i-k+1 .. i] all unpaired), for k=1..u_max
         nc_unp_valid                 - 1 where RNAplfold reported a real value, 0 where NA
                                        (padded prefixes; u > i cannot be evaluated)
         Requires nc_structure kwarg of shape (nc_len, u_max) — pass the
         `profile` returned by preprocess.structure.nc_unpaired_profile /
         batch_unpaired_profile. Optionally pass nc_structure_valid
         (bool (nc_len, u_max)) to override the default all-True mask.
         (u_max + 1 channels)

      Additionally, L_min can be lowered (e.g. to 5) to expose short-window
      normalized match channels (fwd_L5, rc_L5, ...) directly in Group A. These
      are noisy on their own but the network can learn the noise floor per L.

    Anchor convention: the value at (i, j, L-channel) is the score for an
    L-window whose NC substring starts at nc[i] and whose flank substring
    starts at flank[j]. For cells where the window would run past the end
    of NC or flank (i > nc_len - L or j > flank_len - L), the value is 0.

    Returns dict:
      'map'          : float32 (nc_len, flank_len, C)
      'channel_names': list[str] length C, one name per channel in map order
      'L_range'      : (L_min, L_max)
      'nc_len'       : int
      'flank_len'    : int
    """
    nc_len = len(nc)
    flank_len = len(flank)
    fwd_dot_bool, rc_dot_rc_bool = dot_plot(nc, flank)

    # Precompute int codes / one-hots — needed by Groups B and C.
    nc_codes = encode_dna(nc)
    flank_codes = encode_dna(flank)
    rc_flank = revcomp(flank)

    channels: list[np.ndarray] = []
    names: list[str] = []

    if "A" in include_groups:
        # Base-level: fwd match; rc pairability in flank coordinates.
        # rc_dot_flank[i, j] = 1 iff nc[i] pairs with flank[j] via WC.
        # This is the rc-space base dot flipped along the flank axis
        # (offset j' = flank_len - 1 - j).
        rc_dot_flank_bool = rc_dot_rc_bool[:, ::-1]

        channels.append(fwd_dot_bool.astype(np.float32))
        names.append("fwd_dot")
        channels.append(rc_dot_flank_bool.astype(np.float32))
        names.append("rc_dot_flank")

        for L in range(L_min, L_max + 1):
            fwd_L_int = windowed_matches(fwd_dot_bool, L)
            rc_L_int_rc = windowed_matches(rc_dot_rc_bool, L)
            # Convert rc window scores to flank coords.
            # rc_L_flank[i, j] = rc_L_rc[i, flank_L - 1 - j] where
            # flank_L = flank_len - L + 1. This is a flip along axis 1.
            rc_L_int_flank = rc_L_int_rc[:, ::-1]

            f_ch = np.zeros((nc_len, flank_len), dtype=np.float32)
            r_ch = np.zeros((nc_len, flank_len), dtype=np.float32)
            f_h, f_w = fwd_L_int.shape
            r_h, r_w = rc_L_int_flank.shape
            f_ch[:f_h, :f_w] = fwd_L_int.astype(np.float32) / float(L)
            r_ch[:r_h, :r_w] = rc_L_int_flank.astype(np.float32) / float(L)
            channels.append(f_ch)
            names.append(f"fwd_L{L}")
            channels.append(r_ch)
            names.append(f"rc_L{L}")

    if "B" in include_groups:
        # NC one-hot broadcast over flank axis: (nc_len, 4) -> (nc_len, flank_len, 4)
        nc_oh = one_hot_dna(nc)  # (nc_len, 4)
        nc_bcast = np.broadcast_to(
            nc_oh[:, None, :], (nc_len, flank_len, 4)
        )
        flank_oh = one_hot_dna(flank)  # (flank_len, 4)
        flank_bcast = np.broadcast_to(
            flank_oh[None, :, :], (nc_len, flank_len, 4)
        )
        rc_flank_oh = one_hot_dna(rc_flank)  # (flank_len, 4)  in rc-flank order
        # Convert rc-flank one-hot to flank coordinates by reversing along the
        # flank axis (base-level flip: j <- flank_len - 1 - j'). Note this is
        # the rc-pairing one-hot: rcflank_bcast[i, j, k] = 1 iff k == complement(flank[j])
        # (up to code ordering; here we're indexing rc_flank at position
        # flank_len-1-j).
        rc_flank_bcast = np.broadcast_to(
            rc_flank_oh[::-1][None, :, :], (nc_len, flank_len, 4)
        )
        for label, arr4 in (
            ("nc", nc_bcast),
            ("flank", flank_bcast),
            ("rcflank", rc_flank_bcast),
        ):
            for k, base in enumerate("ACGT"):
                channels.append(np.ascontiguousarray(arr4[..., k], dtype=np.float32))
                names.append(f"{label}_{base}")

        # Directional fusion (2 channels)
        dir_ch = direction_fusion(nc_codes, flank_codes)  # (nc_len, flank_len, 2)
        channels.append(np.ascontiguousarray(dir_ch[..., 0]))
        names.append("dir_fusion_0")
        channels.append(np.ascontiguousarray(dir_ch[..., 1]))
        names.append("dir_fusion_1")

    if "C" in include_groups:
        # Relative position: (nc_pos - flank_pos) / (nc_len + flank_len).
        # Constant along the main diagonal — a fwd alignment sits on a
        # constant-value stripe, so a simple linear head can learn to
        # attend to a specific diagonal band.
        i_idx = np.arange(nc_len, dtype=np.float32)
        j_idx = np.arange(flank_len, dtype=np.float32)
        rel = (i_idx[:, None] - j_idx[None, :]) / float(nc_len + flank_len)
        channels.append(rel.astype(np.float32))
        names.append("rel_pos")

    if "D" in include_groups:
        # Short-seed clustering (BLAST/IntaRNA style).
        # For fwd, seed anchors live in flank coords already; for rc, the
        # anchor map is in rc-flank coords -> flip axis-1 of the L_seed-window
        # indicator BEFORE the box-sum, so neighborhoods pool in flank coords.
        neighborhood = float((2 * seed_radius + 1) ** 2)
        for L_seed in seed_lengths:
            if L_seed < 1:
                raise ValueError(f"seed_lengths must be positive; got {L_seed}")

            # Forward: perfect-seed indicator in flank coords.
            fwd_perf_win = (windowed_matches(fwd_dot_bool, L_seed) == L_seed).astype(np.int32)
            fwd_perf_full = np.zeros((nc_len, flank_len), dtype=np.int32)
            fh, fw = fwd_perf_win.shape
            fwd_perf_full[:fh, :fw] = fwd_perf_win
            fwd_dens = _box_sum_2d(fwd_perf_full, seed_radius)

            # RC: perfect-seed indicator in rc-flank coords; flip axis-1 (of
            # the window-shape array, size flank_len - L_seed + 1) to place
            # each anchor at its original flank position (seed at rc pos j'
            # corresponds to flank pos flank_len - L_seed - j').
            rc_perf_win_rc = (windowed_matches(rc_dot_rc_bool, L_seed) == L_seed).astype(np.int32)
            rc_perf_win_flank = rc_perf_win_rc[:, ::-1]
            rc_perf_full = np.zeros((nc_len, flank_len), dtype=np.int32)
            rh, rw = rc_perf_win_flank.shape
            rc_perf_full[:rh, :rw] = rc_perf_win_flank
            rc_dens = _box_sum_2d(rc_perf_full, seed_radius)

            channels.append(fwd_dens.astype(np.float32) / neighborhood)
            names.append(f"fwd_seed_dens_L{L_seed}_r{seed_radius}")
            channels.append(rc_dens.astype(np.float32) / neighborhood)
            names.append(f"rc_seed_dens_L{L_seed}_r{seed_radius}")

    if "E" in include_groups:
        # Per-nt NC unpaired-stretch probability (RNAplfold), broadcast
        # over the flank axis so a 2D CNN sees it in the same (i, j) frame.
        if nc_structure is None:
            raise ValueError(
                "include_groups requested 'E' but nc_structure is None. "
                "Compute it via preprocess.structure.nc_unpaired_profile "
                "(or precompute for the whole dataset) and pass it in."
            )
        nc_structure = np.asarray(nc_structure)
        if nc_structure.ndim != 2 or nc_structure.shape[0] != nc_len:
            raise ValueError(
                f"nc_structure shape {nc_structure.shape} incompatible with "
                f"nc_len={nc_len}; expected (nc_len, u_max)"
            )
        u_max = nc_structure.shape[1]
        if nc_structure_valid is None:
            valid_arr = np.ones_like(nc_structure, dtype=bool)
        else:
            valid_arr = np.asarray(nc_structure_valid, dtype=bool)
            if valid_arr.shape != nc_structure.shape:
                raise ValueError(
                    f"nc_structure_valid shape {valid_arr.shape} must match "
                    f"nc_structure {nc_structure.shape}"
                )
        # Broadcast each per-nt column over the flank axis.
        for k in range(u_max):
            col = nc_structure[:, k].astype(np.float32)
            # Broadcast to (nc_len, flank_len) -> contiguous copy so the
            # channel is a normal float32 array (not a view).
            channels.append(
                np.broadcast_to(col[:, None], (nc_len, flank_len)).copy()
            )
            names.append(f"nc_unp_u{k + 1}")
        # Single valid-mask channel: True where AT LEAST the l=1 column is
        # valid (i.e., the position exists in the sequence and wasn't NA
        # anywhere). Padded / out-of-range NC positions will read 0 here.
        base_valid = valid_arr.any(axis=1).astype(np.float32)
        channels.append(
            np.broadcast_to(base_valid[:, None], (nc_len, flank_len)).copy()
        )
        names.append("nc_unp_valid")

    if not channels:
        raise ValueError(
            f"alignment_feature_stack: no channels selected "
            f"(include_groups={include_groups!r})"
        )

    stack = np.stack(channels, axis=-1)  # (nc_len, flank_len, C)
    return {
        "map": stack,
        "channel_names": names,
        "L_range": (L_min, L_max),
        "nc_len": nc_len,
        "flank_len": flank_len,
    }
