"""V5A framework — shared precompute, variant interface, metric contract, CV, layers.

Foundation for Channel A window selection (P0 for P1-P3). Structure:

    match_table.py   MatchTable[(tnp, site, orient, L)] -> np.ndarray[int8]
                     Per-Tnp sharded on-disk cache; built once per dataset.
    variant.py       VariantSpec (params only) + run_variant(mt, spec) pure fn.
    metrics.py       MetricReport with denominators + CP CI + Tnp-clustered bootstrap.
    cv.py            Staged LOO-Tnp: stage1 (tau x S) -> stage2 (orient/admission)
                     -> stage3 (N_nc). Selection = worst-family upper-CI FP.
    layers.py        Layer 0-5 survival + FP profile helpers.
    fp_classify.py   Motif-explained vs candidate FP classifier, per-family rules.
"""
from __future__ import annotations
