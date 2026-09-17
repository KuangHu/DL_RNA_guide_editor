"""Real bacterial flank sampler for v7 refactor.

Loads the 50-genome pool fetched by `scripts.fetch_v7_genome_pool` and
draws 120bp (60+60) windows uniformly across the pooled bacterial genome
mass. Applies an AT-content filter (reject if AT > threshold, default 0.70)
and an N-mask filter (reject if any N in the window).

Junction convention: position 60 in the returned 120bp string (between
chars [0:60] and [60:120]) — matches the Durrant v2 real-flank convention
so the model sees a consistent "junction at 60" semantics across synthetic
v7 bags and real-Durrant evaluation.

Usage in v7 generator (planned):
    pool = RealFlankPool.load_default()
    flank120 = pool.sample_flank_120(rng, at_max=0.70)
"""
from __future__ import annotations
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from Bio import SeqIO


DEFAULT_POOL_DIR = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v7_refactor/genome_pool")


@dataclass
class RealFlankPool:
    genomes: list[tuple[str, str]]  # (label, upper-case ACGTN sequence)
    lengths: list[int] = field(default_factory=list)
    total_len: int = 0

    def __post_init__(self):
        self.lengths = [len(g[1]) for g in self.genomes]
        self.total_len = sum(self.lengths)

    @classmethod
    def load_default(cls, pool_dir: Path = DEFAULT_POOL_DIR) -> "RealFlankPool":
        genomes = []
        for fna in sorted(pool_dir.glob("*.fna")):
            rec = next(SeqIO.parse(fna, "fasta"))
            genomes.append((fna.stem, str(rec.seq).upper()))
        return cls(genomes=genomes)

    def sample_flank_120(self, rng: random.Random, at_max: float = 0.70,
                              max_reject: int = 1000) -> str:
        """Draw one 120bp window uniformly across the pool.
        Filters: AT-fraction ≤ at_max, no N in window.
        Junction position (for planting) = 60 in the returned string.
        Raises RuntimeError if rejection exceeds max_reject (probably a
        pool-composition issue)."""
        for _ in range(max_reject):
            # Weighted by genome length
            r = rng.uniform(0, self.total_len)
            cum = 0
            for i, L in enumerate(self.lengths):
                cum += L
                if r <= cum:
                    idx = i
                    break
            genome_seq = self.genomes[idx][1]
            pos = rng.randint(60, len(genome_seq) - 60)   # need 60 upstream + 60 downstream
            flank = genome_seq[pos - 60 : pos + 60]
            if len(flank) != 120: continue
            if "N" in flank: continue
            at = (flank.count("A") + flank.count("T")) / 120
            if at > at_max: continue
            return flank
        raise RuntimeError(
            f"[RealFlankPool] rejected {max_reject} draws without a valid window; "
            f"pool composition may be wrong. at_max={at_max}, pool size={self.total_len}"
        )
