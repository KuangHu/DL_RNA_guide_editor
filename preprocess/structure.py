"""ncRNA structure preprocess via ViennaRNA's RNAplfold.

Given a non-coding-region DNA/RNA sequence, `RNAplfold -u X -W W -L L`
computes, for each nucleotide position `i`, the probability that
consecutive stretches ending at `i` are entirely single-stranded:

    column k  ->  P( nts [i-k, ..., i] are all unpaired )   for k = 0..X-1

This is the "seed accessibility" signal IntaRNA uses. For each NC we get
a shape (len, u_max) array plus a boolean valid mask (False where
RNAplfold reported `NA` because the stretch would extend past position 1).

Two entry points:

  nc_unpaired_profile(seq)
      One sequence per call. Convenient for tests / small workloads.
      One RNAplfold subprocess start-up per call (~50-100 ms).

  batch_unpaired_profile(seqs)
      Many sequences in a single RNAplfold invocation via multi-FASTA on
      stdin. Amortizes start-up cost to ~5-10 ms per sequence at batch
      size >= 100. Used by the precompute script.

Both work in a private tempdir; RNAplfold writes `<id>_lunp` and
`<id>_dp.ps` per sequence; we parse `_lunp` and discard the rest.

DNA input is auto-converted to RNA (T -> U). Non-ACGU letters map to N
which RNAplfold treats as unknown; the resulting stretches typically
still fold correctly.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import numpy as np

RNAPLFOLD_BIN = "RNAplfold"

DEFAULT_U_MAX = 16
DEFAULT_W = 120
DEFAULT_L = 60


def _dna_to_rna(seq: str) -> str:
    return seq.upper().replace("T", "U")


def _parse_lunp(path: Path, expected_length: int, u_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Parse an RNAplfold `_lunp` text file.

    Returns:
        profile: float32 (expected_length, u_max). NA values -> 0.0.
        valid:   bool    (expected_length, u_max). False where NA.
    """
    profile = np.zeros((expected_length, u_max), dtype=np.float32)
    valid = np.zeros((expected_length, u_max), dtype=bool)
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            # parts[0] = 1-indexed position; parts[1..u_max] = probabilities
            pos = int(parts[0]) - 1
            if not (0 <= pos < expected_length):
                raise ValueError(
                    f"position {pos + 1} in {path} outside expected length {expected_length}"
                )
            # Number of probability columns present may be < u_max at the
            # tail if the file was written with a smaller -u; enforce match.
            cols = parts[1:]
            if len(cols) < u_max:
                raise ValueError(
                    f"{path}: got {len(cols)} probability columns, expected {u_max}"
                )
            for k in range(u_max):
                v = cols[k]
                if v == "NA":
                    continue
                profile[pos, k] = float(v)
                valid[pos, k] = True
    return profile, valid


def _run_rnaplfold(
    fasta: str,
    cwd: str | Path,
    u_max: int,
    W: int,
    L: int,
    timeout: float,
) -> subprocess.CompletedProcess:
    """Run RNAplfold with stdin=fasta in cwd. Returns the completed process."""
    return subprocess.run(
        [
            RNAPLFOLD_BIN,
            "-u", str(u_max),
            "-W", str(W),
            "-L", str(L),
        ],
        input=fasta,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(cwd),
    )


def nc_unpaired_profile(
    seq: str,
    u_max: int = DEFAULT_U_MAX,
    W: int = DEFAULT_W,
    L: int = DEFAULT_L,
    timeout: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold ONE NC region with RNAplfold and return per-nt unpaired-stretch
    probabilities.

    Args:
        seq:   DNA or RNA sequence (case-insensitive). T is converted to U.
        u_max: max stretch length to consider (returns u_max columns).
        W:     sliding-window size (base pairs only within a window of this size).
        L:     max base-pair span (<= W).
        timeout: seconds allotted to the RNAplfold subprocess.

    Returns:
        profile: float32 (len(seq), u_max)  — NA cells are 0.0
        valid:   bool    (len(seq), u_max)  — False where NA

    profile[i, k] = P(nts [i-k, ..., i] are all unpaired).
    In particular profile[i, 0] = P(nt i unpaired).
    """
    if not seq:
        raise ValueError("empty sequence")
    if L > W:
        raise ValueError(f"L ({L}) must be <= W ({W})")
    seq_rna = _dna_to_rna(seq)
    seq_id = "nc"
    fasta = f">{seq_id}\n{seq_rna}\n"
    with tempfile.TemporaryDirectory(prefix="rnaplfold_") as tmp:
        proc = _run_rnaplfold(fasta, tmp, u_max, W, L, timeout)
        lunp = Path(tmp) / f"{seq_id}_lunp"
        if not lunp.exists():
            raise RuntimeError(
                f"RNAplfold did not produce {lunp.name}: "
                f"stdout={proc.stdout!r} stderr={proc.stderr!r} rc={proc.returncode}"
            )
        return _parse_lunp(lunp, len(seq), u_max)


def batch_unpaired_profile(
    seqs: list[str],
    u_max: int = DEFAULT_U_MAX,
    W: int = DEFAULT_W,
    L: int = DEFAULT_L,
    timeout: float = 600.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Fold many NC regions in ONE RNAplfold invocation via multi-FASTA on stdin.

    Args:
        seqs: list of DNA/RNA strings.
        u_max, W, L: RNAplfold parameters (see nc_unpaired_profile).
        timeout: seconds allotted to the whole batch.

    Returns:
        list of (profile, valid) tuples, one per input sequence, in the same order.
    """
    if not seqs:
        return []
    if L > W:
        raise ValueError(f"L ({L}) must be <= W ({W})")
    # Use fixed-width numeric IDs so the output file basename is
    # predictable and deterministic per input index.
    id_fmt = "nc_{:07d}"
    fasta_chunks = []
    for idx, seq in enumerate(seqs):
        if not seq:
            raise ValueError(f"empty sequence at index {idx}")
        fasta_chunks.append(f">{id_fmt.format(idx)}\n{_dna_to_rna(seq)}")
    fasta = "\n".join(fasta_chunks) + "\n"

    with tempfile.TemporaryDirectory(prefix="rnaplfold_batch_") as tmp:
        proc = _run_rnaplfold(fasta, tmp, u_max, W, L, timeout)
        out: list[tuple[np.ndarray, np.ndarray]] = []
        for idx, seq in enumerate(seqs):
            lunp = Path(tmp) / f"{id_fmt.format(idx)}_lunp"
            if not lunp.exists():
                raise RuntimeError(
                    f"RNAplfold batch: missing {lunp.name} for index {idx} "
                    f"(rc={proc.returncode}, stderr={proc.stderr[:400]!r})"
                )
            out.append(_parse_lunp(lunp, len(seq), u_max))
        return out
