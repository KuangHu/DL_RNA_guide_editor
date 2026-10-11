"""Locus-aware Phase D, comparing each oriented background side separately."""
from statistics import median
from preprocess.alignment import revcomp
from scripts.v5a_framework.background_alignment import side_metrics, select_side, same_locus
from scripts.v5a_framework.background_grouping import BgGroup, BgReport, _mask_boundaries


def compute_verified_bg(cluster, flanks, loci, cfg):
    sides, nleft, nright, masked = {}, {}, {}, {}
    for ss in cluster.supporting_sites:
        h, idx = ss.chosen_hit, ss.site_idx
        f = flanks[idx].upper()
        lo, hi = _mask_boundaries(len(f), h.source_flank_pos(), cluster.L, cfg.margin)
        left, right = f[:lo], f[hi:]
        if h.target_strand == '-':
            left, right = revcomp(right), revcomp(left)
        sides[idx] = left, right
        nleft[idx], nright[idx] = len(left), len(right)
        masked[idx] = left + right
    indices = sorted(sides)
    parent = {i: i for i in indices}
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        parent[find(a)] = find(b)
    pw, pwleft, pwright, metrics, sources, uncertain = {}, {}, {}, {}, {}, []
    for n, i in enumerate(indices):
        for j in indices[n+1:]:
            key = (i, j)
            left = side_metrics(sides[i][0], sides[j][0], cfg.min_bg_bp, cfg.min_coverage_ratio)
            right = side_metrics(sides[i][1], sides[j][1], cfg.min_bg_bp, cfg.min_coverage_ratio)
            selected_side, selected = select_side(left, right)
            metrics[key] = dict(left=left, right=right, selected_side=selected_side)
            if left['eligible']:
                pwleft[key] = left['identity']
            if right['eligible']:
                pwright[key] = right['identity']
            if selected is not None:
                pw[key] = selected['identity']
            if same_locus(loci.get(i, {}), loci.get(j, {}), cfg.ortho_overlap_bp):
                union(i, j)
                sources[key] = 'same_reference_locus'
            elif selected is None:
                # Conservatively collapse unresolved pairs. Missing comparison
                # evidence must never inflate the distinct-background count.
                union(i, j)
                sources[key] = 'insufficient_comparison'
                uncertain.append(key)
            elif selected['identity'] > cfg.clonal_bg_threshold:
                union(i, j)
                sources[key] = selected_side + '_homology'
    components = {}
    for i in indices:
        components.setdefault(find(i), []).append(i)
    groups = []
    for members in components.values():
        ids = [pw[(i, j)] for n, i in enumerate(members) for j in members[n+1:] if (i, j) in pw]
        chain = any((i, j) not in sources for n, i in enumerate(members) for j in members[n+1:])
        groups.append(BgGroup(members, chain, min(ids) if ids else 1.0, max(ids) if ids else 1.0))
    edge_l = {i for i in indices if nleft[i] < cfg.min_bg_bp}
    edge_r = {i for i in indices if nright[i] < cfg.min_bg_bp}
    reasons = []
    if len(groups) < cfg.min_bg_groups:
        reasons.append(f'insufficient_bg_groups({len(groups)}<{cfg.min_bg_groups})')
    if cfg.require_both_sides and (edge_l or edge_r):
        reasons.append(f'edge_insufficient_sites({sorted(edge_l | edge_r)})')
    if cfg.require_both_sides and any(not v['left']['eligible'] or not v['right']['eligible'] for v in metrics.values()):
        reasons.append('insufficient_pairwise_side_coverage')
    return BgReport(cluster.key(), len(groups), groups, pw, masked,
                    any(g.chain_linked for g in groups), nleft, nright,
                    edge_l, edge_r, edge_l & edge_r, pwleft, pwright,
                    median(pwleft.values()) if pwleft else None,
                    median(pwright.values()) if pwright else None,
                    not reasons, ';'.join(reasons) if reasons else None,
                    metrics, sources, uncertain)
