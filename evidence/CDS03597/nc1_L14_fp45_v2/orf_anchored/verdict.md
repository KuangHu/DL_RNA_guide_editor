# ORF-anchored clustering verdict for CDS03597 cluster at (nc_pos=1, L=14)

## Setup
Anchor ORF family: {'4': 3, '6': 3, '7': 2, '0': 1}
Context window: ±400 bp around anchor ORF start

## Per-site anchor
| site | orf_idx | strand | insert_start_in_occupied | anchor_pos_in_oriented |
|---|---|---|---|---|
| 0 | 1 | + | 220 | 483 |
| 4 | 3 | - | 97 | 589 |
| 6 | 3 | - | 249 | 510 |
| 7 | 2 | - | 139 | 589 |

## MSA
Rows: 4, columns: 1913

## Full motif interval [p, p+L) projected into MSA columns
For strand='-' sites the reverse interval in oriented frame is [N-p-L, N-p). The reviewer's expected intervals are site 0 [89,103), site 4 [1875,1889), site 6 [1810,1824), site 7 [1833,1847).

| site | strand | core occ [lo, hi) | core ori [lo, hi) | MSA col [lo, hi) |
|---|---|---|---|---|
| 0 | + | [221, 235) | [221, 235) | [89, 103) |
| 4 | - | [98, 112) | [2112, 2126) | [1875, 1889) |
| 6 | - | [250, 264) | [1968, 1982) | [1810, 1824) |
| 7 | - | [140, 154) | [2071, 2085) | [1833, 1847) |

**MSA-column spread of motif START across sites:**
- cols: [89, 1810, 1833, 1875]
- median: 1833
- max deviation: 1744

**Verdict (CDS03597):** cores scatter across MSA columns (spread=1744). In ORF-anchored canonical orientation, site 0's core sits near the 5' end of the element while sites 4/6/7's cores sit near the 3' end. The raw-nc_pos cluster is pulling matches from OPPOSITE ends of the oriented element — the extraction convention 'nc_frag_0 starts at insert 5' edge' picks up the 5'-TDR copy for sites with the element on the + strand of occupied and the 3'-TDR copy for sites with the element on the - strand. The cluster therefore reflects terminal direct repeat (TDR) homology, not a shared ncRNA guide at a conserved scaffold column.

Note the per-site anchor strands match the pattern: site 0 strand +, sites 4/6/7 strand -. The element is in opposite polarity between the two groups.
