"""Single entry point that runs every V5 frozen anchor and reports PASS/FAIL.

Purpose: after any change to bag_v2, difficulty, features_structure_v2,
match_table, variant, metrics, or channel_a_v5, this script confirms
every locked anchor still holds. If any anchor drifts, the freeze is
broken and the change must be reviewed.

Anchors covered (see FROZEN.md for full context):

  A1. Cross-implementation Durrant anchor via channel_a_v5
      (proves compute_channel_a is equivalent to the framework path)

  A2. Framework Durrant anchor via test_tau0_anchor
      (canonical Durrant reference: 22/23/22/21/21)

  A3. features_structure_v2 T-WT unit tests
      (BPP module invariants + gold-window percentile 83.2%)

  A4. difficulty.py rate table + target_m_for_L(11)==8
      (empirical rate 0.22 at (L=11, m=8))

  A5. architecture.py per-axis uniformity + composition invariants

  A6. Acceptance b: T-WT accepts, V4.2 rejects
      (Tests 1a/1b/1c/2a/2b)

Any FAIL requires investigation before the change is merged.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ANCHORS = [
    ("A1 cross-impl Durrant anchor",
     [sys.executable, "-m", "scripts.generator_v5.tests.test_durrant_anchor_via_channel_a_v5"]),
    ("A2 framework Durrant anchor",
     [sys.executable, "-m", "scripts.v5a_framework.tests.test_tau0_anchor"]),
    ("A3 features_structure_v2 T-WT",
     [sys.executable, "-m", "preprocess.tests.test_features_structure_v2"]),
    ("A4 difficulty rate table + T-WT anchor",
     [sys.executable, "-m", "scripts.generator_v5.tests.test_difficulty"]),
    ("A5 architecture axes",
     [sys.executable, "-m", "scripts.generator_v5.tests.test_architecture"]),
    ("A6 acceptance b (T-WT PASS, V4.2 FAIL)",
     [sys.executable, "-m", "scripts.v5a_framework.tests.b_acceptance_validation"]),
    ("A7 candidates_v2 → Channel A → Durrant anchor (T-WT, L=11 only)",
     [sys.executable, "-m", "scripts.generator_v5.tests.test_candidates_v2_channel_a_parity"]),
    ("A8 candidates_v2 → Channel A → V5 50K anchor (all L, mm_geometry, is_split)",
     [sys.executable, "-m", "scripts.generator_v5.tests.test_candidates_v2_channel_a_v5_anchor"]),
]


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    print(f"[anchors] running {len(ANCHORS)} anchors from {repo_root}")
    print()
    results = []
    for name, cmd in ANCHORS:
        print(f"=== {name} ===")
        r = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
        ok = r.returncode == 0
        results.append((name, ok, r.stdout, r.stderr))
        # Print condensed tail so overall log stays readable
        tail = "\n".join(r.stdout.strip().splitlines()[-6:])
        print(tail)
        if not ok and r.stderr:
            print("--- stderr ---")
            print(r.stderr[-500:])
        print(f"[{name}] {'PASS' if ok else 'FAIL'}")
        print()

    print("=" * 60)
    print("=== SUMMARY ===")
    for name, ok, _, _ in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    all_ok = all(ok for _, ok, _, _ in results)
    print()
    print(f"[anchors] {'ALL PASS' if all_ok else 'SOME FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
