"""MatchTable — precomputed per-position m_max + argmax lookup.

Key: (tnp_id, site_idx, orient, L)
Value: MatchArrays(m_max: int8[N_pos], argmax: int16[N_pos])
       m_max[p]  = max over flank offset f of matches between nc[p:p+L] and flank[f:f+L]
       argmax[p] = the f that achieved the max (enables S_outside_TSD partitioning
                   without any cache rebuild)

Built ONCE per dataset. Two builders:
  build_positive(cog_path, gold_path, ...)   Durrant-style; target_flank_start known
  build_negative(sites_path, family, ...)    real_{fam}_sites.jsonl — paired up+downstream
                                             per physical insertion; downstream flank is
                                             stored for scoring (matches Durrant target-at-
                                             flank-start convention), upstream is retained
                                             for within-site TSD detection.

Sites per Tnp are capped at max_sites_per_tnp (default 50) at build time to keep
shard sizes bounded. Downstream sampling (exactly-5 vs random-5 k=20) is done by
the caller of run_variant via a site_selector callable, not baked into the table.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from preprocess.alignment import dot_plot, windowed_matches


Orient = Literal["fwd", "rc"]
ORIENTS: tuple[Orient, ...] = ("fwd", "rc")
DEFAULT_LS: tuple[int, ...] = (9, 10, 11, 12)

# Exclusion widths for TSD-partitioning m_max_by_excl.
# w=0  : no exclusion (Durrant + any record with no detected TSD)
# w=2  : IS30 palindrome-hotspot minimum TSD
# w=8  : ISLdl1 8 bp AT-rich TSD
# w=9  : IS10-R / IS903 canonical 9 bp TSD
# w=12 : upper safety margin for novel-family TSDs
EXCL_WIDTHS: tuple[int, ...] = (0, 2, 8, 9, 12)


# Family TSD width table. resolve_excl_w() dispatches variant.py's excl_w
# lookup by flank-source family — Durrant (IS110-family) resolves to 0 (no
# characteristic TSD), the five DDE families resolve to their canonical
# TSD length. The runtime assertion in build_e_positive_diagonal /
# build_e_negative refuses any flank source not registered here.
FAMILY_TSD_WIDTH: dict[str, int] = {
    "durrant_positive": 0,
    "IS10-R": 9,
    "IS30":   2,
    "IS903":  9,
    "ISAjo2": 0,
    "ISLdl1": 8,
}


def resolve_excl_w(flank_source_family: str) -> int:
    """Return the excl_w for a given flank source family.

    Raises ValueError if the family isn't registered — this catches the
    "silently defaulted to 0" class of bug (the one that would make an
    IS10-R E-table look like it had no TSD confound).
    """
    if flank_source_family not in FAMILY_TSD_WIDTH:
        raise ValueError(
            f"unknown flank_source_family: {flank_source_family!r}. "
            f"Register it in FAMILY_TSD_WIDTH (module match_table.py).")
    return FAMILY_TSD_WIDTH[flank_source_family]


@dataclass(frozen=True)
class SiteRecord:
    """One insertion site.

    For Durrant positives: flank is the 120 nt window containing the target at
      `target_flank_start`. upstream_flank is None.
    For negatives: flank is the DOWNSTREAM 120 nt flank (junction at position 0);
      upstream_flank is the paired UPSTREAM 120 nt (junction at the last position);
      target_flank_start is None (no known guide target).
    """
    site_idx: int
    flank: str
    upstream_flank: str | None
    target_flank_start: int | None
    gold_nc: int | None
    gold_L: int | None


@dataclass
class TnpRecord:
    tnp_id: str
    family: str
    nc: str
    sites: list[SiteRecord]


@dataclass
class MatchArrays:
    """Per-position m_max, indexed by junction-exclusion width.

    m_max_by_excl[w][p] = max over flank offsets f in [w, W-L+1) of m_max
                          for the L-window at nc position p.
    w=0 means "no exclusion" — identical to the original m_max.
    """
    m_max_by_excl: dict[int, np.ndarray]   # {w: int8[n_nc_positions]}

    @property
    def m_max(self) -> np.ndarray:
        """Backward-compat alias: m_max_by_excl[0]."""
        return self.m_max_by_excl[0]


@dataclass
class MatchTable:
    tnp_ids: list[str]
    tnps: dict[str, TnpRecord]
    orients: tuple[Orient, ...]
    Ls: tuple[int, ...]
    shard_dir: Path
    meta: dict[str, object] = field(default_factory=dict)
    _cache: dict[str, dict] = field(default_factory=dict)

    def get(self, tnp_id: str, site_idx: int, orient: Orient, L: int) -> MatchArrays:
        if tnp_id not in self._cache:
            self._load_shard(tnp_id)
        return self._cache[tnp_id][(site_idx, orient, L)]

    def m_max(self, tnp_id: str, site_idx: int, orient: Orient, L: int,
              excl_w: int = 0) -> np.ndarray:
        """Per-position m_max at a given junction-exclusion width.

        excl_w=0 (default) is the standard m_max (no exclusion).
        excl_w>0 excludes flank offsets f in [0, excl_w).
        """
        return self.get(tnp_id, site_idx, orient, L).m_max_by_excl[excl_w]

    def _load_shard(self, tnp_id: str) -> None:
        path = self.shard_dir / f"{_safe_name(tnp_id)}.npz"
        z = np.load(path)
        by_key: dict[tuple[int, Orient, int], dict[int, np.ndarray]] = {}
        for k in z.files:
            # keys are like "3|fwd|11|excl9"
            parts = k.split("|")
            if len(parts) != 4 or not parts[3].startswith("excl"):
                continue
            s_idx = int(parts[0]); orient = parts[1]; L = int(parts[2])
            w = int(parts[3][4:])
            by_key.setdefault((s_idx, orient, L), {})[w] = z[k]
        d: dict[tuple[int, Orient, int], MatchArrays] = {}
        for key, w_dict in by_key.items():
            d[key] = MatchArrays(m_max_by_excl=w_dict)
        self._cache[tnp_id] = d

    def evict(self, tnp_id: str) -> None:
        self._cache.pop(tnp_id, None)

    def summary(self) -> dict:
        by_fam: dict[str, int] = defaultdict(int)
        sites_dist: list[int] = []
        for t in self.tnps.values():
            by_fam[t.family] += 1
            sites_dist.append(len(t.sites))
        return {
            "n_tnps": len(self.tnp_ids),
            "by_family": dict(by_fam),
            "orients": list(self.orients),
            "Ls": list(self.Ls),
            "sites_per_tnp": {
                "min": min(sites_dist) if sites_dist else 0,
                "median": int(np.median(sites_dist)) if sites_dist else 0,
                "max": max(sites_dist) if sites_dist else 0,
            },
            "shard_dir": str(self.shard_dir),
            "meta": self.meta,
        }


def _safe_name(tnp_id: str) -> str:
    return tnp_id.replace("/", "_").replace("|", "_")


def _dataset_hash(records: list[TnpRecord],
                  orients: tuple[Orient, ...], Ls: tuple[int, ...]) -> str:
    h = hashlib.sha256()
    for t in records:
        h.update(t.tnp_id.encode())
        h.update(t.nc.encode())
        for s in t.sites:
            h.update(f"{s.site_idx}|{s.flank}".encode())
    h.update(json.dumps({"orients": list(orients), "Ls": list(Ls)}, sort_keys=True).encode())
    return h.hexdigest()[:16]


def _windowed_max_by_excl(win: np.ndarray, excl_widths: tuple[int, ...]
                          ) -> dict[int, np.ndarray]:
    """For each exclusion width w, return win[p, w:].max(axis=1) as int8.

    Handles empty windows (returns zero-length arrays for every w).
    """
    out: dict[int, np.ndarray] = {}
    if win.size == 0:
        for w in excl_widths:
            out[w] = np.zeros(0, dtype=np.int8)
        return out
    n_pos, n_offsets = win.shape
    for w in excl_widths:
        if w >= n_offsets:
            out[w] = np.zeros(n_pos, dtype=np.int8)
        else:
            out[w] = win[:, w:].max(axis=1).astype(np.int8)
    return out


def _fwd_win_max_by_excl(nc: str, flank: str, L: int,
                          excl_widths: tuple[int, ...]) -> dict[int, np.ndarray]:
    fwd, _ = dot_plot(nc, flank)
    win = windowed_matches(fwd, L)
    return _windowed_max_by_excl(win, excl_widths)


def _rc_win_max_by_excl(nc: str, flank: str, L: int,
                         excl_widths: tuple[int, ...]) -> dict[int, np.ndarray]:
    _, rc = dot_plot(nc, flank)
    win = windowed_matches(rc, L)
    return _windowed_max_by_excl(win, excl_widths)


def _compute_site_arrays(
    nc: str, flank: str, orients: tuple[Orient, ...], Ls: tuple[int, ...],
    excl_widths: tuple[int, ...] = EXCL_WIDTHS,
) -> dict[tuple[Orient, int], MatchArrays]:
    out: dict[tuple[Orient, int], MatchArrays] = {}
    for orient in orients:
        for L in Ls:
            if orient == "fwd":
                m_dict = _fwd_win_max_by_excl(nc, flank, L, excl_widths)
            else:
                m_dict = _rc_win_max_by_excl(nc, flank, L, excl_widths)
            out[(orient, L)] = MatchArrays(m_max_by_excl=m_dict)
    return out


def _write_shard(shard_dir: Path, tnp: TnpRecord,
                 orients: tuple[Orient, ...], Ls: tuple[int, ...],
                 excl_widths: tuple[int, ...] = EXCL_WIDTHS) -> None:
    arrays: dict[str, np.ndarray] = {}
    for s in tnp.sites:
        site_arrs = _compute_site_arrays(tnp.nc, s.flank, orients, Ls, excl_widths)
        for (orient, L), ma in site_arrs.items():
            for w, arr in ma.m_max_by_excl.items():
                arrays[f"{s.site_idx}|{orient}|{L}|excl{w}"] = arr
    np.savez(shard_dir / f"{_safe_name(tnp.tnp_id)}.npz", **arrays)


def _serialize_site(s: SiteRecord) -> dict:
    return {"site_idx": s.site_idx, "flank": s.flank,
            "upstream_flank": s.upstream_flank,
            "target_flank_start": s.target_flank_start,
            "gold_nc": s.gold_nc, "gold_L": s.gold_L}


def _deserialize_site(d: dict) -> SiteRecord:
    return SiteRecord(
        site_idx=d["site_idx"], flank=d["flank"],
        upstream_flank=d.get("upstream_flank"),
        target_flank_start=d.get("target_flank_start"),
        gold_nc=d.get("gold_nc"), gold_L=d.get("gold_L"),
    )


def _write_index(shard_dir: Path, records: list[TnpRecord],
                 orients: tuple[Orient, ...], Ls: tuple[int, ...],
                 meta: dict) -> None:
    idx = {
        "dataset_hash": _dataset_hash(records, orients, Ls),
        "tnp_ids": [t.tnp_id for t in records],
        "orients": list(orients),
        "Ls": list(Ls),
        "tnps": {t.tnp_id: {"family": t.family, "n_sites": len(t.sites),
                             "nc_len": len(t.nc)} for t in records},
        "meta": meta,
    }
    (shard_dir / "_index.json").write_text(json.dumps(idx, indent=2))
    seqs = {t.tnp_id: {"nc": t.nc, "sites": [_serialize_site(s) for s in t.sites]}
            for t in records}
    (shard_dir / "_seqs.json").write_text(json.dumps(seqs))


def _load_from_index(shard_dir: Path) -> MatchTable:
    idx = json.loads((shard_dir / "_index.json").read_text())
    seqs = json.loads((shard_dir / "_seqs.json").read_text())
    tnps: dict[str, TnpRecord] = {}
    for tnp_id in idx["tnp_ids"]:
        seq_t = seqs[tnp_id]
        sites = [_deserialize_site(s) for s in seq_t["sites"]]
        tnps[tnp_id] = TnpRecord(tnp_id=tnp_id, family=idx["tnps"][tnp_id]["family"],
                                  nc=seq_t["nc"], sites=sites)
    return MatchTable(
        tnp_ids=list(idx["tnp_ids"]), tnps=tnps,
        orients=tuple(idx["orients"]), Ls=tuple(idx["Ls"]),
        shard_dir=shard_dir,
        meta={**idx.get("meta", {}), "dataset_hash": idx["dataset_hash"]},
    )


def load(shard_dir: str | Path) -> MatchTable:
    return _load_from_index(Path(shard_dir))


def _build_common(records: list[TnpRecord], shard_dir: Path,
                  orients: tuple[Orient, ...], Ls: tuple[int, ...],
                  meta: dict, progress_every: int = 20) -> MatchTable:
    shard_dir.mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(records):
        _write_shard(shard_dir, t, orients, Ls)
        if (i + 1) % progress_every == 0:
            print(f"  [match_table] {i+1}/{len(records)} tnps written", flush=True)
    _write_index(shard_dir, records, orients, Ls, meta)
    print(f"  [match_table] index written to {shard_dir}", flush=True)
    return _load_from_index(shard_dir)


# ---------- positive builder (Durrant) ----------

def build_positive(cog_path: str | Path, gold_path: str | Path,
                   shard_dir: str | Path,
                   orients: tuple[Orient, ...] = ORIENTS,
                   Ls: tuple[int, ...] = DEFAULT_LS,
                   min_sites: int = 5, cap_sites: int = 5,
                   family_label: str = "durrant_positive") -> MatchTable:
    """Build MatchTable from Durrant cog_vs_shuf JSONL + durrant_gold_v1 JSONL.

    Uses labels.active_noncoding_index to pick the nc region. Keeps only Tnps
    where all included sites share the same nc string. Filters to min_sites
    and caps to cap_sites (default 5 to match architecture).
    """
    gold_by_site = {}
    with open(gold_path) as f:
        for line in f:
            r = json.loads(line)
            gold_by_site[r["site_id"]] = r
    tnp_sites: dict[str, list[SiteRecord]] = defaultdict(list)
    tnp_nc: dict[str, str] = {}
    with open(cog_path) as f:
        for line in f:
            r = json.loads(line)
            if not r.get("labels", {}).get("is_positive"):
                continue
            g = gold_by_site.get(r["site_id"])
            if g is None:
                continue
            a = r["labels"].get("active_noncoding_index", 0) or 0
            ncs = r["inputs"]["noncoding_regions"]
            if a >= len(ncs):
                a = 0
            nc = ncs[a]
            tnp = r["transposase_id"]
            if tnp not in tnp_nc:
                tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc:
                continue
            tnp_sites[tnp].append(SiteRecord(
                site_idx=len(tnp_sites[tnp]),
                flank=r["inputs"]["flank"],
                upstream_flank=None,
                target_flank_start=g.get("target_flank_start"),
                gold_nc=g.get("guide_start_in_nc"),
                gold_L=g.get("target_binding_loop_length"),
            ))
    records = [TnpRecord(tnp_id=t, family=family_label, nc=tnp_nc[t], sites=ss[:cap_sites])
               for t, ss in tnp_sites.items() if len(ss) >= min_sites]
    print(f"  [build_positive] {len(records)} Tnps with >= {min_sites} sites "
          f"(capped to {cap_sites})", flush=True)
    meta = {"builder": "build_positive", "cog_path": str(cog_path),
            "gold_path": str(gold_path), "min_sites": min_sites, "cap_sites": cap_sites}
    return _build_common(records, Path(shard_dir), orients, Ls, meta)


# ---------- negative builder (5 families) ----------

def build_negative(sites_path: str | Path, family: str,
                   shard_dir: str | Path,
                   orients: tuple[Orient, ...] = ORIENTS,
                   Ls: tuple[int, ...] = DEFAULT_LS,
                   min_insertions: int = 5,
                   max_insertions_per_tnp: int = 50) -> MatchTable:
    """Build MatchTable from real_{family}_sites.jsonl (not bagdedup — we want
    per-insertion paired records).

    Each physical insertion has TWO records (flank_side=upstream + downstream);
    they are paired via (transposase_id, insertion_start, sample_id). We store
    ONE SiteRecord per physical insertion:
        flank          = downstream flank (matches Durrant target-at-position-0)
        upstream_flank = paired upstream flank (for within-site TSD detection)

    Keeps Tnps with >= min_insertions physical insertions. Caps at
    max_insertions_per_tnp to bound shard size (some Tnps have >2000 copies).
    Downsampling to exactly-5 vs random-5 is done at scoring time via a
    site_selector, not at build time.
    """
    from collections import defaultdict
    pairs: dict[tuple, dict[str, dict]] = defaultdict(dict)
    tnp_nc: dict[str, str] = {}
    with open(sites_path) as f:
        for line in f:
            r = json.loads(line)
            inputs = r.get("inputs", {})
            ncs = inputs.get("noncoding_regions", [])
            if not ncs:
                continue
            a = r.get("labels", {}).get("active_noncoding_index", 0) or 0
            if a >= len(ncs):
                a = 0
            nc = ncs[a]
            tnp = r.get("transposase_id")
            m = r.get("generator_metadata", {})
            side = m.get("flank_side")
            key = (tnp, m.get("insertion_start"), m.get("sample_id"))
            if tnp is None or side not in ("upstream", "downstream"):
                continue
            if tnp not in tnp_nc:
                tnp_nc[tnp] = nc
            elif tnp_nc[tnp] != nc:
                continue
            pairs[key][side] = r

    tnp_sites: dict[str, list[SiteRecord]] = defaultdict(list)
    for (tnp, _, _), pair in pairs.items():
        if "upstream" not in pair or "downstream" not in pair:
            continue
        dn = pair["downstream"]["inputs"]["flank"]
        up = pair["upstream"]["inputs"]["flank"]
        if not dn or not up:
            continue
        tnp_sites[tnp].append(SiteRecord(
            site_idx=len(tnp_sites[tnp]),
            flank=dn, upstream_flank=up,
            target_flank_start=None, gold_nc=None, gold_L=None,
        ))

    records: list[TnpRecord] = []
    for tnp, ss in tnp_sites.items():
        if len(ss) < min_insertions:
            continue
        if len(ss) > max_insertions_per_tnp:
            ss = ss[:max_insertions_per_tnp]   # deterministic head cap; caller
                                                # selects downstream for random-5.
        records.append(TnpRecord(tnp_id=tnp, family=family,
                                  nc=tnp_nc[tnp], sites=ss))
    print(f"  [build_negative:{family}] {len(records)} Tnps with >= {min_insertions} "
          f"physical insertions (capped to {max_insertions_per_tnp})", flush=True)
    meta = {"builder": "build_negative", "sites_path": str(sites_path),
            "family": family, "min_insertions": min_insertions,
            "max_insertions_per_tnp": max_insertions_per_tnp}
    return _build_common(records, Path(shard_dir), orients, Ls, meta)


# ---------- site selectors (sampling modes) ----------

def select_exactly_5(mt: MatchTable) -> dict[str, list[int]]:
    """Mode A: keep only Tnps with exactly 5 sites; use those 5 indices."""
    out: dict[str, list[int]] = {}
    for tnp_id in mt.tnp_ids:
        n = len(mt.tnps[tnp_id].sites)
        if n == 5:
            out[tnp_id] = [0, 1, 2, 3, 4]
    return out


def select_random_5(mt: MatchTable, seed: int, resample_idx: int
                     ) -> dict[str, list[int]]:
    """Mode B: for every Tnp with >= 5 sites, randomly pick 5 site indices.

    `resample_idx` picks a distinct random draw. `seed` is the base seed;
    per-Tnp draws are seeded by (seed, resample_idx, tnp_id_hash) so nested
    bootstrap can jointly resample Tnps and pick their inner draws.
    """
    out: dict[str, list[int]] = {}
    for tnp_id in mt.tnp_ids:
        n = len(mt.tnps[tnp_id].sites)
        if n < 5:
            continue
        h = hash((seed, resample_idx, tnp_id)) & 0xFFFFFFFF
        rng = np.random.default_rng(h)
        out[tnp_id] = sorted(rng.choice(n, size=5, replace=False).tolist())
    return out


def stack_positions(mt: MatchTable, tnp_id: str, orient: Orient, L: int,
                    site_indices: list[int] | None = None,
                    fixed_len: int | None = None) -> np.ndarray:
    """Return (n_sites, n_positions) stacked m_max array. site_indices=None
    uses all sites of the Tnp."""
    tnp = mt.tnps[tnp_id]
    if site_indices is None:
        site_indices = list(range(len(tnp.sites)))
    arrs = [mt.m_max(tnp_id, si, orient, L) for si in site_indices]
    if not arrs:
        return np.zeros((0, 0), dtype=np.int8)
    n = fixed_len if fixed_len is not None else min(a.shape[0] for a in arrs)
    out = np.zeros((len(arrs), n), dtype=np.int8)
    for i, a in enumerate(arrs):
        out[i, : min(n, a.shape[0])] = a[:n]
    return out
