"""Tests for preprocess/structure.py (RNAplfold-based per-nt unpaired profile).

Checks:
  1. Hairpin smoke: GGGGAAAACCCC -> loop unpaired > 0.9, stem < 0.2.
  2. Batch consistency: single-call and batched profiles agree exactly.
  3. Positive record: mean unpaired-prob inside the labeled guide span is
     > 0.5 (guides are placed inside a designed loop by the generator).
  4. alignment_only negative: the active NC's ncRNA span is random DNA,
     so the "guide is in a loop" pattern should NOT hold for it. We
     verify that on a small batch the alignment_only records have a
     significantly LOWER mean guide-span unpaired probability than
     matched positives.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from preprocess.structure import (
    batch_unpaired_profile,
    nc_unpaired_profile,
)

POSITIVES = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/positives.jsonl"
NEGATIVES = "/global/scratch/users/kh36969/DL_novel_guide_editor/data/negatives.jsonl"


def hairpin_smoke() -> None:
    p, v = nc_unpaired_profile("GGGGAAAACCCC", u_max=6, W=20, L=20)
    assert p.shape == (12, 6) and p.dtype == np.float32
    assert v.shape == (12, 6) and v.dtype == bool
    loop_mean = p[4:8, 0].mean()
    stem_mean = p[np.r_[0:4, 8:12], 0].mean()
    assert loop_mean > 0.9, f"hairpin loop mean unpaired {loop_mean:.3f} not > 0.9"
    assert stem_mean < 0.2, f"hairpin stem mean unpaired {stem_mean:.3f} not < 0.2"
    # Valid mask: for l=1, all positions have a value; for l>1, first (l-1) are NA.
    assert v[:, 0].all(), "l=1 col should have no NAs"
    for l in range(1, 6):
        assert not v[l - 1, l], f"pos {l-1} l={l+1} should be NA"


def batch_consistency() -> None:
    seqs = [
        "GGGGAAAACCCC",
        "GGGGAAAACCCC",         # duplicate: batch should return identical arrays
        "ACGUACGUACGUACGU",
        "GGGGGGCCCCCCUUUUAAAAGGGGGGCCCCCC",  # nested hairpin
    ]
    batch = batch_unpaired_profile(seqs, u_max=8, W=30, L=30)
    for i, seq in enumerate(seqs):
        single_p, single_v = nc_unpaired_profile(seq, u_max=8, W=30, L=30)
        batch_p, batch_v = batch[i]
        assert single_p.shape == batch_p.shape
        # Batch and single should be bit-identical (deterministic RNAplfold).
        max_diff = float(np.abs(single_p - batch_p).max())
        assert max_diff < 1e-6, (
            f"seq[{i}]: single-vs-batch diff = {max_diff}"
        )
        assert np.array_equal(single_v, batch_v), f"seq[{i}]: valid masks differ"
    # Duplicate check: two identical seqs in the batch produce identical arrays.
    assert np.array_equal(batch[0][0], batch[1][0])


def load_head(path: str, n: int):
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            yield json.loads(line)


def guide_span_unpaired_signal() -> None:
    """On a modest batch of positives, mean unpaired probability of the
    guide span in the active NC should be substantially higher than the
    NC-wide baseline (guides sit in loops)."""
    n = 50
    pos_recs = list(load_head(POSITIVES, n))
    seqs = [r["inputs"]["noncoding_regions"][r["labels"]["active_noncoding_index"]] for r in pos_recs]
    profiles = batch_unpaired_profile(seqs, u_max=16, W=120, L=60)

    span_means = []
    baseline_means = []
    for rec, (p, _v) in zip(pos_recs, profiles):
        gs, ge = rec["labels"]["guide_span_in_active_noncoding"]
        # Column 0 = P(nt is unpaired).
        span_means.append(float(p[gs:ge, 0].mean()))
        baseline_means.append(float(p[:, 0].mean()))

    guide_mean = float(np.mean(span_means))
    baseline_mean = float(np.mean(baseline_means))
    print(f"  positives: guide-span unpaired = {guide_mean:.3f}, "
          f"NC baseline = {baseline_mean:.3f}, n={n}")
    assert guide_mean > 0.5, (
        f"guide-span mean unpaired {guide_mean:.3f} not > 0.5 "
        f"(guides should be in loops)"
    )
    assert guide_mean > baseline_mean, (
        f"guide-span mean {guide_mean:.3f} should exceed NC baseline "
        f"{baseline_mean:.3f}"
    )


def alignment_only_signal_differs() -> None:
    """`alignment_only` negatives replace the ncRNA with random DNA. Their
    guide-span mean unpaired probability should NOT show the loop-like
    signal; on a matched batch it should be significantly lower than the
    positive baseline."""
    n_pos = 40
    n_neg = 40
    pos_recs = list(load_head(POSITIVES, n_pos))
    neg_recs = []
    with open(NEGATIVES) as f:
        for line in f:
            r = json.loads(line)
            if r["labels"].get("violation_profile") == "alignment_only":
                neg_recs.append(r)
                if len(neg_recs) >= n_neg:
                    break

    all_seqs = (
        [r["inputs"]["noncoding_regions"][r["labels"]["active_noncoding_index"]] for r in pos_recs]
        + [r["inputs"]["noncoding_regions"][r["labels"]["active_noncoding_index"]] for r in neg_recs]
    )
    profiles = batch_unpaired_profile(all_seqs, u_max=16, W=120, L=60)

    pos_span_means = []
    neg_span_means = []
    for i, rec in enumerate(pos_recs):
        gs, ge = rec["labels"]["guide_span_in_active_noncoding"]
        p, _ = profiles[i]
        pos_span_means.append(float(p[gs:ge, 0].mean()))
    for j, rec in enumerate(neg_recs):
        gs, ge = rec["labels"]["guide_span_in_active_noncoding"]
        p, _ = profiles[n_pos + j]
        neg_span_means.append(float(p[gs:ge, 0].mean()))

    pos_mean = float(np.mean(pos_span_means))
    neg_mean = float(np.mean(neg_span_means))
    print(f"  positives guide-span unpaired = {pos_mean:.3f} (n={len(pos_span_means)})")
    print(f"  alignment_only guide-span unpaired = {neg_mean:.3f} (n={len(neg_span_means)})")
    assert pos_mean > neg_mean + 0.1, (
        f"expected positives {pos_mean:.3f} substantially higher than "
        f"alignment_only {neg_mean:.3f}"
    )


def main():
    print("hairpin smoke ...", end=" ")
    hairpin_smoke()
    print("ok")

    print("batch consistency ...", end=" ")
    batch_consistency()
    print("ok")

    print("guide-span unpaired signal on positives ...")
    guide_span_unpaired_signal()

    print("alignment_only vs positive signal ...")
    alignment_only_signal_differs()

    print("all structure tests passed.")


if __name__ == "__main__":
    main()
