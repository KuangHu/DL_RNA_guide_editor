# ORF-anchored clustering verdict for CDS03597 cluster at (nc_pos=1, L=14)

## Setup
Anchor ORF family: {'4': 3, '7': 2, '6': 3, '0': 1}
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

## Core-a position (guide inside insert) projected into MSA columns
| site | core_a_pos (occupied) | MSA column |
|---|---|---|
| 0 | 221 | 89 |
| 4 | 98 | 1888 |
| 6 | 250 | 1823 |
| 7 | 140 | 1846 |

**MSA-column spread of core_a across sites:**
- cols: [89, 1823, 1846, 1888]
- median: 1846
- max deviation: 1757

**Verdict (CDS03597):** cores scatter across MSA columns (spread=1757). In ORF-anchored canonical orientation, site 0's core sits near the 5' end of the element while sites 4/6/7's cores sit near the 3' end. The raw-nc_pos cluster is pulling matches from OPPOSITE ends of the oriented element — the extraction convention 'nc_frag_0 starts at insert 5' edge' picks up the 5'-TDR copy for sites with the element on the + strand of occupied and the 3'-TDR copy for sites with the element on the - strand. The cluster therefore reflects terminal direct repeat (TDR) homology, not a shared ncRNA guide at a conserved scaffold column.

Note the per-site anchor strands match the pattern: site 0 strand +, sites 4/6/7 strand -. The element is in opposite polarity between the two groups.
