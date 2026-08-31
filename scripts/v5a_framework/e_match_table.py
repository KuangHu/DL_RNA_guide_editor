"""EMatchTable — Option E precomputed table.

Semantic (Option E): negative flanks are scored against Durrant nc substrate
so nc composition, nc length, search space, and coordinate system are held
constant. Only the flank source varies across the 7-way negative axis
(Durrant self + Durrant-shuffled + 5 non-guided families).

Key: (flank_tnp_id, site_idx, nc_source_tnp_id, orient, L, excl_w).
The `flank_source_family` field of the table is baked into the flank_tnps
(each TnpRecord.family) and also stamped on the table itself. The
family field enables `resolve_excl_w()` runtime dispatch, so scoring can
cross-check that the excl_w it is asking for matches the excl_w baked into
the shard.

Storage: per-flank-Tnp shards; each shard is one .npz where each entry
name is `{site_idx}|{nc_source_tnp_id}|{orient}|{L}|excl{w}` and each
value is an int8 array of per-position m_max at that exclusion width.

Diagonal build (this file): each Durrant Tnp T's 5 flanks scored against
T's OWN nc. This is the historical anchor case and is the minimum-cost
verification that the E-table's build path is byte-equivalent to the
existing MatchTable.

Full build (later): each flank Tnp × each Durrant nc (65 × 65 for Durrant;
sampled per family for negatives with cap=10 flanks per Tnp).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .match_table import (
    MatchTable, TnpRecord, SiteRecord, Orient, ORIENTS, DEFAULT_LS, EXCL_WIDTHS,
    FAMILY_TSD_WIDTH, resolve_excl_w,
    _compute_site_arrays, _safe_name, _serialize_site, _deserialize_site,
)


@dataclass
class EMatchTable:
    """Option E precomputed match table.

    flank_source_family : e.g. "durrant_positive", "IS10-R" — dispatched via
                          resolve_excl_w() to the family's TSD width. This is
                          the *primary* runtime-safety key: scoring must ask
                          for the excl_w that resolve_excl_w() returns, not
                          a different one.
    flank_tnp_ids       : ordered list of flank source Tnp IDs
    flank_tnps          : per-Tnp records; sites carry the flank sequences
    nc_source_tnp_ids   : ordered list of Durrant Tnp IDs providing nc substrate
    nc_source_ncs       : nc_source_tnp_id -> nc sequence (needed for the
                          E-value table lookup which is a function of nc_len)
    """
    flank_source_family: str
    flank_tnp_ids: list[str]
    flank_tnps: dict[str, TnpRecord]
    nc_source_tnp_ids: list[str]
    nc_source_ncs: dict[str, str]
    orients: tuple[Orient, ...]
    Ls: tuple[int, ...]
    shard_dir: Path
    meta: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict)

    def m_max(self, flank_tnp_id: str, site_idx: int,
              nc_source_tnp_id: str, orient: Orient, L: int,
              excl_w: int = 0) -> np.ndarray:
        if flank_tnp_id not in self._cache:
            self._load_shard(flank_tnp_id)
        return self._cache[flank_tnp_id][
            (site_idx, nc_source_tnp_id, orient, L, excl_w)
        ]

    def resolved_excl_w(self) -> int:
        """Return the excl_w the flank family dispatches to.

        Callers wanting the "natural" excl_w for this table's flank family
        should ask through this method, not hardcode.
        """
        return resolve_excl_w(self.flank_source_family)

    def _load_shard(self, flank_tnp_id: str) -> None:
        path = self.shard_dir / f"{_safe_name(flank_tnp_id)}.npz"
        z = np.load(path)
        d: dict[tuple, np.ndarray] = {}
        for key in z.files:
            parts = key.split("|")
            if len(parts) != 5 or not parts[4].startswith("excl"):
                continue
            s_idx = int(parts[0])
            nc_tnp = parts[1]
            orient = parts[2]
            L = int(parts[3])
            w = int(parts[4][4:])
            d[(s_idx, nc_tnp, orient, L, w)] = z[key]
        self._cache[flank_tnp_id] = d

    def evict(self, flank_tnp_id: str) -> None:
        self._cache.pop(flank_tnp_id, None)

    def summary(self) -> dict:
        return {
            "flank_source_family": self.flank_source_family,
            "resolved_excl_w": self.resolved_excl_w(),
            "n_flank_tnps": len(self.flank_tnp_ids),
            "n_nc_source_tnps": len(self.nc_source_tnp_ids),
            "orients": list(self.orients),
            "Ls": list(self.Ls),
            "shard_dir": str(self.shard_dir),
            "meta": self.meta,
        }


# ---------- diagonal builder ----------

def build_e_positive_diagonal(mt_pos: MatchTable,
                                shard_dir: str | Path,
                                orients: tuple[Orient, ...] = ORIENTS,
                                Ls: tuple[int, ...] = DEFAULT_LS,
                                excl_widths: tuple[int, ...] = EXCL_WIDTHS
                                ) -> EMatchTable:
    """Build the Durrant-self DIAGONAL of the E-table.

    For each Durrant Tnp T:
      T's 5 flanks × T's own nc × orient × L × excl_w -> m_max arrays.

    This is the minimum-cost verification path for the E-table's semantic
    equivalence to the historical MatchTable. The full E-table (T × T' for
    all Durrant pairs, plus 5 negative families × Durrant nc grid) is built
    by build_e_positive_full and build_e_negative separately.

    Runtime assertion: expects every Tnp in mt_pos to have family label
    'durrant_positive' (registered in FAMILY_TSD_WIDTH -> 0). Refuses to
    build if the family is unregistered or resolves to a nonzero excl_w
    for a Durrant table.
    """
    family = "durrant_positive"
    resolved = resolve_excl_w(family)
    if resolved != 0:
        raise ValueError(
            f"family {family!r} must resolve to excl_w=0 for the Durrant path; "
            f"got {resolved}. Check FAMILY_TSD_WIDTH.")
    for tnp_id in mt_pos.tnp_ids:
        actual = mt_pos.tnps[tnp_id].family
        if actual != family:
            raise ValueError(
                f"build_e_positive_diagonal expects all Tnps in mt_pos to have "
                f"family={family!r}; {tnp_id!r} has family={actual!r}.")

    shard_dir = Path(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    n_total = len(mt_pos.tnp_ids)
    for i, tnp_id in enumerate(mt_pos.tnp_ids):
        tnp = mt_pos.tnps[tnp_id]
        nc = tnp.nc     # diagonal: T's own nc
        arrays: dict[str, np.ndarray] = {}
        for site in tnp.sites:
            site_arrs = _compute_site_arrays(nc, site.flank, orients, Ls, excl_widths)
            for (orient, L), ma in site_arrs.items():
                for w, arr in ma.m_max_by_excl.items():
                    key = f"{site.site_idx}|{tnp_id}|{orient}|{L}|excl{w}"
                    arrays[key] = arr
        np.savez(shard_dir / f"{_safe_name(tnp_id)}.npz", **arrays)
        if (i + 1) % 20 == 0:
            print(f"  [build_e_diagonal] {i+1}/{n_total} tnps", flush=True)

    idx = {
        "flank_source_family": family,
        "resolved_excl_w": resolved,
        "flank_tnp_ids": list(mt_pos.tnp_ids),
        "nc_source_tnp_ids": list(mt_pos.tnp_ids),
        "orients": list(orients),
        "Ls": list(Ls),
        "excl_widths": list(excl_widths),
        "shape": "diagonal",
        "meta": {"builder": "build_e_positive_diagonal"},
    }
    (shard_dir / "_e_index.json").write_text(json.dumps(idx, indent=2))
    seqs = {tnp_id: {"nc": mt_pos.tnps[tnp_id].nc,
                     "sites": [_serialize_site(s) for s in mt_pos.tnps[tnp_id].sites]}
            for tnp_id in mt_pos.tnp_ids}
    (shard_dir / "_e_seqs.json").write_text(json.dumps(seqs))
    print(f"  [build_e_diagonal] index written to {shard_dir}", flush=True)
    return load_e(shard_dir)


def load_e(shard_dir: str | Path) -> EMatchTable:
    """Load an EMatchTable from an E-table shard directory."""
    p = Path(shard_dir)
    idx = json.loads((p / "_e_index.json").read_text())
    seqs = json.loads((p / "_e_seqs.json").read_text())
    family = idx["flank_source_family"]
    flank_tnps: dict[str, TnpRecord] = {}
    nc_ncs: dict[str, str] = {}
    for tnp_id in idx["flank_tnp_ids"]:
        seq_t = seqs[tnp_id]
        sites = [_deserialize_site(s) for s in seq_t["sites"]]
        flank_tnps[tnp_id] = TnpRecord(tnp_id=tnp_id, family=family,
                                        nc=seq_t["nc"], sites=sites)
        nc_ncs[tnp_id] = seq_t["nc"]
    # Cross-check runtime dispatch matches what the table was built at
    idx_resolved = idx.get("resolved_excl_w")
    if idx_resolved is not None and idx_resolved != resolve_excl_w(family):
        raise ValueError(
            f"E-table shard at {p} recorded resolved_excl_w={idx_resolved} for "
            f"family {family!r}, but current FAMILY_TSD_WIDTH resolves to "
            f"{resolve_excl_w(family)}. Either the table is stale or the "
            f"family registry drifted; rebuild.")
    return EMatchTable(
        flank_source_family=family,
        flank_tnp_ids=list(idx["flank_tnp_ids"]),
        flank_tnps=flank_tnps,
        nc_source_tnp_ids=list(idx["nc_source_tnp_ids"]),
        nc_source_ncs=nc_ncs,
        orients=tuple(idx["orients"]),
        Ls=tuple(idx["Ls"]),
        shard_dir=p,
        meta=idx.get("meta", {}),
    )


# ---------- MatchTable interface shim (diagonal case) ----------

class DiagonalShim:
    """Adapts an EMatchTable to the MatchTable interface used by variant.py
    by defaulting the nc source to == flank tnp (diagonal case).

    Useful for running the standard anchor spec on the E-table's Durrant-self
    diagonal without re-plumbing variant.py.
    """
    def __init__(self, emt: EMatchTable) -> None:
        self._emt = emt

    @property
    def tnp_ids(self) -> list[str]:
        return self._emt.flank_tnp_ids

    @property
    def tnps(self) -> dict[str, TnpRecord]:
        return self._emt.flank_tnps

    @property
    def orients(self) -> tuple[Orient, ...]:
        return self._emt.orients

    @property
    def Ls(self) -> tuple[int, ...]:
        return self._emt.Ls

    def m_max(self, tnp_id: str, site_idx: int, orient: Orient, L: int,
              excl_w: int = 0) -> np.ndarray:
        # diagonal: nc source == flank tnp
        return self._emt.m_max(tnp_id, site_idx, tnp_id, orient, L, excl_w)
