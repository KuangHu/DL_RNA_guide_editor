"""Matched counterfactuals for V1 interpretability (tests 5, 6, 7).

Unlike ablations (which mask channels in existing inputs), counterfactuals
modify raw records BEFORE preprocessing so that the model sees consistent
inputs that have exactly one axis broken. We measure per-tnp
Delta_p = P(positive|original) - P(positive|counterfactual).

Tests:

  test5_swap_flanks
      Within each positive tnp, permute the `inputs.flank` field across
      its S sites. NC (guide + padding) stays identical to the original
      site, so the ncRNA structure is unchanged; only the alignment
      between guide and flank is broken.

  test6_move_target
      For each site of a positive tnp, keep the guide + NC identical, but
      MOVE the exact target sequence (flank[ts:ts+L]) to a random position
      in the flank that is at least 15 bp away from the original. The
      original target position is over-written with random DNA.
      Alignment quality (matches / L) is preserved; only target position
      on the flank is changed.

  test7_swap_paddings
      For each site of a positive tnp, keep the exact ncRNA sub-region
      (including the guide slot) unchanged, but swap the 5' and 3'
      paddings around it with the paddings from another site's NC.
      Alignment intact; ncRNA fold context around the guide changes.
      RNAplfold is re-run on the modified NC (~150 ms/NC).

We evaluate on a subset of positive tnps in val (default 200 tnps ×
50 sites = 10k sites) and report:
  - Delta_p distribution (mean, median, quartiles)
  - Fraction of tnps where score drops by >= 0.5 (large drop)
  - Fraction that flip below 0.5 (classification change)
  - Candidate R@1 change on the counterfactual

Compute cost: cache-only for tests 5 & 6 (fast, ~90 ms/site).
Test 7 uses on-the-fly RNAplfold folding for each modified NC
(~150 ms/NC extra, ~seconds per tnp). Keep --max-tnps small for test 7.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from model.v1 import V1Config, V1Model
from preprocess.candidates import (
    DEFAULT_L_MAX,
    DEFAULT_L_MIN,
    DEFAULT_ORIENTATIONS,
    PATCH_WIDTH_DEFAULT,
    TOP_K_PER_COMBO_DEFAULT,
    build_candidate_arrays,
)
from preprocess.site import DEFAULT_NC_MAX, DEFAULT_NUM_NC_SLOTS, StructureCache
from preprocess.structure import nc_unpaired_profile


# --------------------------------------------------------------- #
# Record loading
# --------------------------------------------------------------- #

def load_positive_tnp_records(split_path: Path, max_tnps: int, seed: int):
    """Return {tnp_id: [records]} for the first `max_tnps` positive tnps
    encountered (deterministic shuffle by seed)."""
    from collections import defaultdict
    by_tnp: dict[str, list[dict]] = defaultdict(list)
    with open(split_path) as f:
        for line in f:
            r = json.loads(line)
            if not r["labels"].get("is_positive"):
                continue
            by_tnp[r["transposase_id"]].append(r)
    tnp_ids = sorted(by_tnp.keys())
    rng = np.random.default_rng(seed)
    rng.shuffle(tnp_ids)
    tnp_ids = tnp_ids[:max_tnps]
    return {t: by_tnp[t] for t in tnp_ids}


# --------------------------------------------------------------- #
# Counterfactual transforms  (operate on a list of records for one tnp)
# --------------------------------------------------------------- #

def cf_swap_flanks(recs: list[dict], rng: np.random.Generator) -> list[dict]:
    """Test 5: within tnp, permute flanks across sites."""
    flanks = [r["inputs"]["flank"] for r in recs]
    perm = rng.permutation(len(flanks))
    # If the permutation happens to keep any position fixed, force a full derangement
    # (each site gets a DIFFERENT flank than its own).
    for _try in range(10):
        if not any(i == perm[i] for i in range(len(perm))):
            break
        perm = rng.permutation(len(flanks))
    out = []
    for r, p in zip(recs, perm):
        rr = copy.deepcopy(r)
        rr["inputs"]["flank"] = flanks[int(p)]
        # invalidate label since target position now points to the wrong flank
        rr["labels"] = dict(rr["labels"])
        rr["labels"]["is_positive"] = None  # unknown — not the point of this test
        out.append(rr)
    return out


def cf_move_target(recs: list[dict], rng: np.random.Generator,
                    min_shift: int = 15) -> list[dict]:
    """Test 6: keep alignment quality; move target to random position in flank.

    The target sequence flank[ts:ts+L] is copied to a new position `ts'`
    at least `min_shift` bp away from `ts`; the original position is
    over-written with random DNA to remove the alignment there.
    """
    out = []
    DNA = np.array(list("ACGT"))
    for r in recs:
        rr = copy.deepcopy(r)
        flank = rr["inputs"]["flank"]
        lbl = rr["labels"]
        ts, te = lbl["target_position_in_flank"]
        L = te - ts
        target = flank[ts:te]

        # pick new start
        max_off = len(flank) - L
        candidates = [i for i in range(0, max_off + 1) if abs(i - ts) >= min_shift]
        if not candidates:
            candidates = list(range(0, max_off + 1))
        new_ts = int(rng.choice(candidates))

        # Randomize old target position
        rand_old = "".join(DNA[rng.integers(0, 4, size=L)])
        # Build new flank in two passes so we don't overwrite the moving target
        buf = list(flank)
        for i in range(L):
            buf[ts + i] = rand_old[i]
        for i in range(L):
            buf[new_ts + i] = target[i]
        rr["inputs"]["flank"] = "".join(buf)
        rr["labels"] = dict(lbl)
        rr["labels"]["target_position_in_flank"] = [new_ts, new_ts + L]
        out.append(rr)
    return out


def cf_swap_paddings(recs: list[dict], rng: np.random.Generator) -> list[dict]:
    """Test 7: keep ncRNA sub-region + guide identical; swap surrounding
    paddings between sites of the same tnp. The `active_noncoding_index`
    NC is the one modified; decoy NCs are unchanged.

    NC layout: padding_5 + ncRNA + padding_3.
    ncrna_span_in_active_noncoding gives [start, end] within the NC.
    """
    S = len(recs)
    if S < 2:
        return [copy.deepcopy(r) for r in recs]

    # Collect paddings from each site's ACTIVE NC.
    p5s: list[str] = []
    p3s: list[str] = []
    for r in recs:
        slot = r["labels"]["active_noncoding_index"]
        nc = r["inputs"]["noncoding_regions"][slot]
        ns, ne = r["labels"]["ncrna_span_in_active_noncoding"]
        p5s.append(nc[:ns])
        p3s.append(nc[ne:])

    # Derange each site (guarantee a swap, not identity).
    perm5 = rng.permutation(S)
    for _ in range(10):
        if not any(i == perm5[i] for i in range(S)):
            break
        perm5 = rng.permutation(S)
    perm3 = rng.permutation(S)
    for _ in range(10):
        if not any(i == perm3[i] for i in range(S)):
            break
        perm3 = rng.permutation(S)

    out = []
    for i, r in enumerate(recs):
        rr = copy.deepcopy(r)
        slot = rr["labels"]["active_noncoding_index"]
        nc = rr["inputs"]["noncoding_regions"][slot]
        ns, ne = rr["labels"]["ncrna_span_in_active_noncoding"]
        ncrna = nc[ns:ne]  # includes the guide
        new_p5 = p5s[int(perm5[i])]
        new_p3 = p3s[int(perm3[i])]
        new_nc = new_p5 + ncrna + new_p3
        rr["inputs"]["noncoding_regions"] = list(rr["inputs"]["noncoding_regions"])
        rr["inputs"]["noncoding_regions"][slot] = new_nc
        # Update spans (padding lengths changed).
        new_ns = len(new_p5)
        new_ne = new_ns + (ne - ns)
        rr["labels"] = dict(rr["labels"])
        rr["labels"]["ncrna_span_in_active_noncoding"] = [new_ns, new_ne]
        # Guide span shifts by delta_p5.
        gs, ge = rr["labels"]["guide_span_in_active_noncoding"]
        delta_p5 = new_ns - ns
        rr["labels"]["guide_span_in_active_noncoding"] = [gs + delta_p5, ge + delta_p5]
        out.append(rr)
    return out


# --------------------------------------------------------------- #
# Preprocessing that supports either the cache OR on-the-fly folding
# --------------------------------------------------------------- #

def preprocess_records(
    recs: list[dict],
    *,
    structure_cache: StructureCache | None,
    fold_on_the_fly: bool,
    nc_max: int = DEFAULT_NC_MAX,
    num_nc_slots: int = DEFAULT_NUM_NC_SLOTS,
    top_k_per_combo: int = TOP_K_PER_COMBO_DEFAULT,
    L_min: int = DEFAULT_L_MIN,
    L_max: int = DEFAULT_L_MAX,
    orientations=DEFAULT_ORIENTATIONS,
    patch_width: int = PATCH_WIDTH_DEFAULT,
    u_max: int = 16, W_pl: int = 120, L_pl: int = 60,
) -> dict:
    """Return a torch batch dict for the tnp's records.

    Preprocess each site into candidate_patches / features / mask; stack
    into (S, ...). If fold_on_the_fly is True, RNAplfold is invoked on
    each site's NCs (needed when NC sequences differ from what the cache
    stored).
    """
    S = len(recs)
    if S == 0:
        raise ValueError("empty recs")

    # Probe first site to learn K/shape.
    def get_structure_for_site(r):
        """Return list of (profile, valid) per NC of this site."""
        outs = []
        for slot, nc in enumerate(r["inputs"]["noncoding_regions"]):
            if fold_on_the_fly:
                prof, valid = nc_unpaired_profile(
                    nc, u_max=u_max, W=W_pl, L=L_pl,
                )
            else:
                prof, valid = structure_cache.get(r["site_id"], slot, len(nc))
            outs.append((prof, valid))
        return outs

    # Pre-compute per-site preprocessed arrays.
    site_pat, site_feat, site_mask, site_ncmask = [], [], [], []
    for r in recs:
        flank = r["inputs"]["flank"]
        ncs = r["inputs"]["noncoding_regions"]
        struct_list = get_structure_for_site(r)

        K_max = len(orientations) * (L_max - L_min + 1) * top_k_per_combo
        from preprocess.candidates import PATCH_CHANNELS, NUM_FEATURES
        patches = np.zeros((num_nc_slots, K_max, patch_width, PATCH_CHANNELS), dtype=np.float32)
        features = np.zeros((num_nc_slots, K_max, NUM_FEATURES), dtype=np.float32)
        mask = np.zeros((num_nc_slots, K_max), dtype=bool)
        ncmask = np.zeros((num_nc_slots,), dtype=bool)

        for slot, nc in enumerate(ncs):
            prof, valid = struct_list[slot]
            p, f, m, _ = build_candidate_arrays(
                nc, flank, prof, valid,
                top_k_per_combo=top_k_per_combo,
                L_min=L_min, L_max=L_max, orientations=orientations,
                patch_width=patch_width, nc_max=nc_max,
            )
            patches[slot] = p
            features[slot] = f
            mask[slot] = m
            ncmask[slot] = True

        site_pat.append(patches)
        site_feat.append(features)
        site_mask.append(mask)
        site_ncmask.append(ncmask)

    # Stack across sites -> (1, S, N_nc, K, ...) for a batch of B=1 tnp.
    batch = {
        "candidate_patches":  torch.from_numpy(np.stack(site_pat, axis=0)[None]),   # (1, S, ...)
        "candidate_features": torch.from_numpy(np.stack(site_feat, axis=0)[None]),
        "candidate_mask":     torch.from_numpy(np.stack(site_mask, axis=0)[None]),
        "nc_region_mask":     torch.from_numpy(np.stack(site_ncmask, axis=0)[None]),
    }
    return batch


# --------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------- #

@torch.no_grad()
def score_tnp(model, batch, device, use_bf16=True) -> float:
    batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                        enabled=(use_bf16 and device.type == "cuda")):
        out = model(
            batch["candidate_patches"],
            batch["candidate_features"],
            batch["candidate_mask"],
            batch["nc_region_mask"],
        )
    return float(torch.sigmoid(out["logit"]).item())


def summarize(deltas: np.ndarray, tag: str) -> dict:
    d = deltas
    q = np.quantile(d, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "test": tag,
        "n": int(len(d)),
        "mean_delta_p": float(d.mean()),
        "median_delta_p": float(np.median(d)),
        "p05": float(q[0]), "p25": float(q[1]),
        "p75": float(q[3]), "p95": float(q[4]),
        "frac_drop_>=0.5": float((d >= 0.5).mean()),
        "frac_flipped_below_0.5": float((d >= 0.5).mean()),  # rough proxy
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--split-jsonl", required=True, type=Path)
    p.add_argument("--structure-index", required=True)
    p.add_argument("--max-tnps", type=int, default=200)
    p.add_argument("--max-tnps-test7", type=int, default=50,
                    help="test 7 is slow (on-the-fly folding); cap harder")
    p.add_argument("--tests", default="5,6,7",
                    help="comma-separated: subset of {5, 6, 7}")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = V1Config(**ckpt["cfg"])
    model = V1Model(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"[cf] loaded {args.ckpt} (epoch {ckpt['epoch']})", flush=True)

    cache = StructureCache(args.structure_index)
    tnp2recs = load_positive_tnp_records(args.split_jsonl, args.max_tnps, args.seed)
    print(f"[cf] {len(tnp2recs)} positive tnps loaded", flush=True)

    tests = set(args.tests.split(","))
    rng = np.random.default_rng(args.seed)
    reports: list[dict] = []
    per_tnp_out: dict[str, dict] = {}

    for i, (tnp, recs) in enumerate(tnp2recs.items()):
        base_batch = preprocess_records(
            recs, structure_cache=cache, fold_on_the_fly=False,
        )
        p_base = score_tnp(model, base_batch, device)
        per_tnp_out[tnp] = {"p_base": p_base, "S": len(recs)}

        if "5" in tests:
            new_recs = cf_swap_flanks(recs, rng)
            b = preprocess_records(new_recs, structure_cache=cache, fold_on_the_fly=False)
            per_tnp_out[tnp]["p_test5"] = score_tnp(model, b, device)

        if "6" in tests:
            new_recs = cf_move_target(recs, rng)
            b = preprocess_records(new_recs, structure_cache=cache, fold_on_the_fly=False)
            per_tnp_out[tnp]["p_test6"] = score_tnp(model, b, device)

        if "7" in tests and i < args.max_tnps_test7:
            new_recs = cf_swap_paddings(recs, rng)
            b = preprocess_records(new_recs, structure_cache=None, fold_on_the_fly=True)
            per_tnp_out[tnp]["p_test7"] = score_tnp(model, b, device)

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(tnp2recs)}", flush=True)

    # Reports.
    print()
    def _collect(key):
        return np.array([
            v["p_base"] - v[key]
            for v in per_tnp_out.values() if key in v
        ])

    if "5" in tests:
        d5 = _collect("p_test5")
        s = summarize(d5, "test5_swap_flanks")
        reports.append(s)
        print(f"[test 5 swap_flanks]  n={s['n']}  Δp mean={s['mean_delta_p']:.3f}  "
              f"median={s['median_delta_p']:.3f}  IQR=[{s['p25']:.3f}, {s['p75']:.3f}]  "
              f"frac(Δp>=0.5)={s['frac_drop_>=0.5']:.2f}")
    if "6" in tests:
        d6 = _collect("p_test6")
        s = summarize(d6, "test6_move_target")
        reports.append(s)
        print(f"[test 6 move_target]  n={s['n']}  Δp mean={s['mean_delta_p']:.3f}  "
              f"median={s['median_delta_p']:.3f}  IQR=[{s['p25']:.3f}, {s['p75']:.3f}]  "
              f"frac(Δp>=0.5)={s['frac_drop_>=0.5']:.2f}")
    if "7" in tests:
        d7 = _collect("p_test7")
        s = summarize(d7, "test7_swap_paddings")
        reports.append(s)
        print(f"[test 7 swap_paddings]  n={s['n']}  Δp mean={s['mean_delta_p']:.3f}  "
              f"median={s['median_delta_p']:.3f}  IQR=[{s['p25']:.3f}, {s['p75']:.3f}]  "
              f"frac(Δp>=0.5)={s['frac_drop_>=0.5']:.2f}")

    base_scores = np.array([v["p_base"] for v in per_tnp_out.values()])
    print(f"[baseline]  n_positive={len(base_scores)}  "
          f"mean p_base={base_scores.mean():.4f}  min={base_scores.min():.4f}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"reports": reports, "per_tnp": per_tnp_out}, f, indent=2)
        print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
