"""Phase D of model-free scorer v3 — local background grouping + gate.

Grouping rule:
  - Per supporting site, mask the cluster's target span [flank_pos, flank_pos+L]
    plus margin (default 3 bp).
  - Compute LEFT and RIGHT background per site; measure available bp per side;
    flag sites whose side has < min_bg_bp as edge_insufficient_<side>.
  - Pairwise identity computed BOTH on each side independently (reported) AND
    on concatenated bg (used for grouping); common length is trimmed adjacent
    to each site's mask.
  - Edge between two sites if concatenated identity > clonal_bg_threshold.
  - Connected components = local background groups.
  - chain_linked flag set True if any non-adjacent pair within a component
    falls at/below threshold — honest signal that transitivity bridged them.

Hard gate:  n_local_bg_groups >= min_bg_groups (default 3).
Clusters failing it are stored with rejection_reason set, not removed.

CAVEAT (part of module contract, cite in output):
  n_local_bg_groups is a LOCAL 120 bp observation, NOT a genome-level
  independence claim.  Chain-linked components may bridge dissimilar pairs
  through a transitive partner.  For genome-level distinctness, cross-reference
  carriers_derived / pre_event_genome in top4_candidates/*/provenance.tsv.
  Distinct assembly accessions do not themselves prove sequence context
  differs; the local bg check is the only sequence-level signal here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.v5a_framework.cluster_enumeration import Cluster, SupportingSite


@dataclass(frozen=True)
class BgConfig:
    margin: int = 3
    clonal_bg_threshold: float = 0.85
    min_bg_bp: int = 20
    min_bg_groups: int = 3
    require_both_sides: bool = False   # strict mode: fail if any supporting
                                       # site has edge_insufficient_both OR
                                       # if both sides insufficient for the
                                       # pair; default = permissive


@dataclass
class BgGroup:
    members: list[int]
    chain_linked: bool
    min_pair_identity: float
    max_pair_identity: float


@dataclass
class BgReport:
    cluster_key: tuple[int, int, int]
    n_local_bg_groups: int
    groups: list[BgGroup]
    pairwise_identities: dict[tuple[int, int], float]       # concat-bg, used for grouping
    masked_flank_per_site: dict[int, str]                   # left + right concatenated (full)
    any_chain_linked: bool
    # per-site edge availability
    n_bp_left_available: dict[int, int]
    n_bp_right_available: dict[int, int]
    edge_insufficient_left: set[int]
    edge_insufficient_right: set[int]
    edge_insufficient_both: set[int]
    # per-side pairwise identities (None when either site is edge-insufficient on that side)
    pairwise_left_identity: dict[tuple[int, int], float]    # omit keys when either side missing
    pairwise_right_identity: dict[tuple[int, int], float]
    median_left_identity: float | None
    median_right_identity: float | None
    # gate
    passes_bg_gate: bool
    rejection_reason: str | None


def _mask_boundaries(flank_len: int, flank_pos: int, L: int, margin: int,
                      ) -> tuple[int, int]:
    lo = max(0, flank_pos - margin)
    hi = min(flank_len, flank_pos + L + margin)
    return lo, hi


def _pair_identity(a: str, b: str) -> float:
    n = min(len(a), len(b))
    if n == 0: return 0.0
    m = sum(1 for x, y in zip(a[:n], b[:n]) if x == y)
    return m / n


def _side_pair(flank_A: str, flank_B: str,
                lo_A: int, hi_A: int, lo_B: int, hi_B: int, side: str,
                ) -> tuple[str, str]:
    """Return the (A_chunk, B_chunk) pair of equal length adjacent to each
    site's mask on the requested side.  For 'left': take the max-common
    tail length of the pre-mask region of each site (adjacent to target
    on left).  For 'right': take the max-common head length of the
    post-mask region of each site (adjacent to target on right)."""
    if side == "left":
        left_A = flank_A[:lo_A]
        left_B = flank_B[:lo_B]
        n = min(len(left_A), len(left_B))
        return left_A[len(left_A) - n:], left_B[len(left_B) - n:]
    else:  # right
        right_A = flank_A[hi_A:]
        right_B = flank_B[hi_B:]
        n = min(len(right_A), len(right_B))
        return right_A[:n], right_B[:n]


def compute_bg_groups(cluster: Cluster,
                       flanks_per_site: dict[int, str],
                       margin: int = 3,
                       clonal_bg_threshold: float = 0.85,
                       min_bg_bp: int = 20,
                       min_bg_groups: int = 3,
                       require_both_sides: bool = False,
                       ) -> BgReport:
    cfg = BgConfig(margin=margin, clonal_bg_threshold=clonal_bg_threshold,
                    min_bg_bp=min_bg_bp, min_bg_groups=min_bg_groups,
                    require_both_sides=require_both_sides)

    # Per-site mask boundaries + per-side bp available
    mask_bounds: dict[int, tuple[int, int]] = {}
    n_left: dict[int, int] = {}
    n_right: dict[int, int] = {}
    masked_concat: dict[int, str] = {}
    edge_L: set[int] = set()
    edge_R: set[int] = set()
    for ss in cluster.supporting_sites:
        f = flanks_per_site[ss.site_idx]
        lo, hi = _mask_boundaries(len(f), ss.chosen_hit.flank_pos, cluster.L, cfg.margin)
        mask_bounds[ss.site_idx] = (lo, hi)
        n_left[ss.site_idx]  = lo
        n_right[ss.site_idx] = len(f) - hi
        masked_concat[ss.site_idx] = f[:lo] + f[hi:]
        if n_left[ss.site_idx]  < cfg.min_bg_bp: edge_L.add(ss.site_idx)
        if n_right[ss.site_idx] < cfg.min_bg_bp: edge_R.add(ss.site_idx)
    edge_both = edge_L & edge_R

    # Which sites participate in grouping?
    # Permissive: keep every site with ANY available bp.
    # Strict (require_both_sides): drop sites in edge_both.
    all_sites = sorted(masked_concat.keys())
    if cfg.require_both_sides:
        bg_sites = [s for s in all_sites if s not in edge_both]
    else:
        bg_sites = [s for s in all_sites if masked_concat[s]]

    # Pairwise identities
    pw_concat: dict[tuple[int, int], float] = {}
    pw_left:   dict[tuple[int, int], float] = {}
    pw_right:  dict[tuple[int, int], float] = {}
    for i, si in enumerate(bg_sites):
        for sj in bg_sites[i + 1:]:
            fA, fB = flanks_per_site[si], flanks_per_site[sj]
            loA, hiA = mask_bounds[si]
            loB, hiB = mask_bounds[sj]
            # left pair
            a_l, b_l = _side_pair(fA, fB, loA, hiA, loB, hiB, "left")
            if len(a_l) >= cfg.min_bg_bp:
                pw_left[(si, sj)] = _pair_identity(a_l, b_l)
            # right pair
            a_r, b_r = _side_pair(fA, fB, loA, hiA, loB, hiB, "right")
            if len(a_r) >= cfg.min_bg_bp:
                pw_right[(si, sj)] = _pair_identity(a_r, b_r)
            # concat for grouping: concatenate the common-length left and right
            # chunks (one or both may be empty); require at least min_bg_bp
            # total.  If 0 bp usable, drop this edge from the graph.
            cat_A = a_l + a_r
            cat_B = b_l + b_r
            if len(cat_A) >= cfg.min_bg_bp:
                pw_concat[(si, sj)] = _pair_identity(cat_A, cat_B)

    med_left  = median(pw_left.values())  if pw_left  else None
    med_right = median(pw_right.values()) if pw_right else None

    # Build adjacency on concat threshold (only for pairs with a concat identity)
    adj: dict[int, set[int]] = {s: set() for s in bg_sites}
    for (si, sj), ident in pw_concat.items():
        if ident > cfg.clonal_bg_threshold:
            adj[si].add(sj); adj[sj].add(si)

    # Connected components
    visited: set[int] = set()
    components: list[list[int]] = []
    for s in bg_sites:
        if s in visited: continue
        stack = [s]; comp = []
        while stack:
            x = stack.pop()
            if x in visited: continue
            visited.add(x); comp.append(x)
            stack.extend(adj[x] - visited)
        components.append(sorted(comp))

    # Chain-link / per-group stats
    groups: list[BgGroup] = []
    any_chain = False
    for comp in components:
        if len(comp) == 1:
            groups.append(BgGroup(members=comp, chain_linked=False,
                                   min_pair_identity=1.0,
                                   max_pair_identity=1.0))
            continue
        pair_ids = []
        for i, a in enumerate(comp):
            for b in comp[i + 1:]:
                key = (min(a, b), max(a, b))
                if key in pw_concat:
                    pair_ids.append(pw_concat[key])
        if not pair_ids:
            # Component built only via other partners (shouldn't happen since
            # a single-size>1 component implies at least one edge), but guard.
            groups.append(BgGroup(members=comp, chain_linked=True,
                                   min_pair_identity=0.0, max_pair_identity=0.0))
            any_chain = True
            continue
        min_p = min(pair_ids); max_p = max(pair_ids)
        chain = min_p <= cfg.clonal_bg_threshold
        if chain: any_chain = True
        groups.append(BgGroup(members=comp, chain_linked=chain,
                               min_pair_identity=min_p,
                               max_pair_identity=max_p))

    # Gate
    n_groups = len(components)
    passes = n_groups >= cfg.min_bg_groups
    reason = None
    if not passes:
        reason = (f"insufficient_bg_groups({n_groups}<{cfg.min_bg_groups})")
    # Strict mode: require every supporting site to have >= min_bg_bp on
    # BOTH sides.  This gives a cleaner "both backgrounds differ" claim at
    # the cost of more rejections for edge-adjacent targets.  Permissive
    # (default) mode still runs the grouping with whatever bg is available
    # per-side and flags insufficient sides for user review.
    if cfg.require_both_sides:
        insufficient = edge_L | edge_R
        if insufficient:
            strict_reason = f"edge_insufficient_sites({sorted(insufficient)})"
            reason = strict_reason if reason is None else f"{reason};{strict_reason}"
            passes = False

    return BgReport(
        cluster_key=cluster.key(),
        n_local_bg_groups=n_groups,
        groups=groups,
        pairwise_identities=pw_concat,
        masked_flank_per_site=masked_concat,
        any_chain_linked=any_chain,
        n_bp_left_available=n_left,
        n_bp_right_available=n_right,
        edge_insufficient_left=edge_L,
        edge_insufficient_right=edge_R,
        edge_insufficient_both=edge_both,
        pairwise_left_identity=pw_left,
        pairwise_right_identity=pw_right,
        median_left_identity=med_left,
        median_right_identity=med_right,
        passes_bg_gate=passes,
        rejection_reason=reason,
    )


def apply_phase_d_gate(clusters: list[Cluster],
                        flanks_per_site: dict[int, str],
                        cfg: BgConfig = BgConfig(),
                        ) -> tuple[list[tuple[Cluster, BgReport]],
                                    list[tuple[Cluster, BgReport]]]:
    """Return (passing, failing) lists of (cluster, bg_report)."""
    passing: list[tuple[Cluster, BgReport]] = []
    failing: list[tuple[Cluster, BgReport]] = []
    for c in clusters:
        br = compute_bg_groups(c, flanks_per_site,
                                margin=cfg.margin,
                                clonal_bg_threshold=cfg.clonal_bg_threshold,
                                min_bg_bp=cfg.min_bg_bp,
                                min_bg_groups=cfg.min_bg_groups,
                                require_both_sides=cfg.require_both_sides)
        (passing if br.passes_bg_gate else failing).append((c, br))
    return passing, failing
