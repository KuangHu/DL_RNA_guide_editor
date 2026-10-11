"""Phase G of model-free scorer v3 — null-flank constructors.

Four null constructions for calibrating the Phase D output against sequence
backgrounds that preserve some, but not all, of the signal structure of
real flanks.  Each takes a bag's sites and returns a new sites list with
one of the four flank-replacement rules.

Nulls:
  null1_dinuc_shuffle  : flank replaced by a dinucleotide-shuffled version
                          (preserves dinuc composition, destroys higher-period
                          repeats and real motif placement)
  null2_kmer_shuffle   : same but for 4-mer composition
  null3_unpaired_bag   : flanks replaced by flanks from a different donor
                          bag (preserves per-flank composition exactly but
                          breaks the nc<->flank pairing). CAVEAT: the donor
                          bag may itself carry real signal; interpret as a
                          plausibility bound, not a strict null.
  null4_rc_then_shuffle: flank reverse-complemented then dinuc-shuffled;
                          isolates RC-strand leakage from the retention
                          machinery.

Shuffle implementation:
  Random-walk approximation on the k-mer transition graph: for each position
  i >= k-1, draw the next base from the empirical conditional distribution
  P(seq[i] | seq[i-k+1 .. i-1]).  This preserves (k-1)-mer conditional
  probabilities in expectation but NOT exact k-mer counts; good enough as
  a null that scales to tens of thousands of flanks.  For publication-grade
  exact preservation, swap in an Altschul-Erickson Eulerian walk.
"""
from __future__ import annotations
import random
from pathlib import Path
import sys
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.alignment import revcomp


def _kmer_shuffle_once(seq: str, k: int, rng: random.Random) -> str:
    """Random-walk shuffle preserving (k-1)-mer conditional composition."""
    if k < 2 or len(seq) <= k:
        # Nothing to preserve; fall back to a simple permutation.
        chars = list(seq); rng.shuffle(chars); return "".join(chars)
    # P(next | (k-1)-mer): collect lists of next bases.
    conds: dict[str, list[str]] = defaultdict(list)
    for i in range(len(seq) - k + 1):
        prefix = seq[i : i + k - 1]
        conds[prefix].append(seq[i + k - 1])
    out = [seq[0]]
    for i in range(1, k - 1):
        out.append(seq[i])
    for _ in range(len(seq) - (k - 1)):
        prefix = "".join(out[-(k - 1):])
        choices = conds.get(prefix)
        if not choices:
            # dead end: pick a random base from anywhere
            pool = [b for bs in conds.values() for b in bs]
            if not pool:
                out.append(rng.choice("ACGT"))
            else:
                out.append(rng.choice(pool))
        else:
            out.append(rng.choice(choices))
    return "".join(out)


def dinuc_shuffle(seq: str, seed: int) -> str:
    return _kmer_shuffle_once(seq, k=2, rng=random.Random(seed))


def kmer_shuffle(seq: str, k: int, seed: int) -> str:
    return _kmer_shuffle_once(seq, k=k, rng=random.Random(seed))


def _replace_flanks(sites: list[dict], new_flanks: list[str]) -> list[dict]:
    """Return a copy of sites with flanks replaced one-to-one."""
    assert len(new_flanks) == len(sites), (
        f"flank count mismatch: {len(new_flanks)} vs {len(sites)}")
    out = []
    for s, f in zip(sites, new_flanks):
        new_s = {**s, "inputs": {**s["inputs"], "flank": f}}
        out.append(new_s)
    return out


def null1_dinuc_shuffle(sites: list[dict], seed: int) -> list[dict]:
    new_flanks = [dinuc_shuffle(s["inputs"]["flank"], seed=seed + 100 * i)
                   for i, s in enumerate(sites)]
    return _replace_flanks(sites, new_flanks)


def null2_kmer_shuffle(sites: list[dict], k: int, seed: int) -> list[dict]:
    new_flanks = [kmer_shuffle(s["inputs"]["flank"], k=k, seed=seed + 100 * i)
                   for i, s in enumerate(sites)]
    return _replace_flanks(sites, new_flanks)


def null3_unpaired_bag(sites: list[dict],
                        donor_sites: list[dict],
                        seed: int) -> list[dict]:
    """Replace this bag's K flanks with K flanks sampled from donor_sites.
    If donor_sites has fewer than K flanks, cycle through them.
    """
    rng = random.Random(seed)
    donor_flanks = [s["inputs"]["flank"] for s in donor_sites]
    assert donor_flanks, "donor bag has no flanks"
    rng.shuffle(donor_flanks)
    new_flanks = [donor_flanks[i % len(donor_flanks)] for i in range(len(sites))]
    return _replace_flanks(sites, new_flanks)


def null4_rc_then_shuffle(sites: list[dict], seed: int) -> list[dict]:
    new_flanks = []
    for i, s in enumerate(sites):
        rc = revcomp(s["inputs"]["flank"])
        new_flanks.append(dinuc_shuffle(rc, seed=seed + 100 * i))
    return _replace_flanks(sites, new_flanks)
