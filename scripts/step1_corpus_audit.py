"""Step 1 audit: per-site data-source availability for the v3 scorer corpus.

Explicit (bag_id, site_id) join. Row-order equality reported as a
consistency check only. Six independent axes per site; nothing is collapsed
into a single "resolved" column.

Scope reported twice:
  (a) 1173-bag sweep scope — config/v3_sweep_cids.txt
  (b) 4380-bag upstream   — all distinct transposase_id in adapted.jsonl

Axes:
  joined                — (bag_id, db_id) in insert_cds.tsv AND
                           db_id resolved in insertions.tsv AND
                           short locus present in panel loci.tsv.
  raw_files_exist       — panel loci.tsv + alleles.fna present AND seed-genome
                           FASTA present. Fail buckets distinguish all three.
  sequence_recovered    — flank matches **empty** allele exactly, 1 copy any
                           strand; AND each nc fragment matches **occupied**
                           exactly, 1 copy each (ambiguity preserved).
  boundary_set_complete — strict ok ONLY when insert_start_in_long_bp is
                           numeric AND junction_L_abs, junction_R_abs,
                           n_equivalent_decompositions are present.
                           Alternate tier `partial:single_cutpoint_only` when
                           only the fallback cutpoint is derivable; the full
                           boundary SET is still missing.
  orf_msa_ready         — pending; Step 2 deliverable.
  target_coord_ready    — pending; Step 2 deliverable.

v2 corrections (2026-10-11, after reviewer feedback):
  - db_id parser strips BOTH `.site` AND `.copyNNNN` suffix
    (the raw audit's 19131 "no_event_in_insert_cds" was 100% a parser bug)
  - allele key uses full split (`locus|allele|len=…|n=…` → allele at index 1)
  - flank-recovery check is in `empty`, nc-recovery in `occupied`
  - boundary_set_complete=ok only when insert_start_in_long_bp is numeric;
    partial:single_cutpoint_only is reported but is NOT ok.
"""
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path("/global/scratch/users/kh36969/fna_ins_discovery")
ADAPTED = Path("/global/scratch/users/kh36969/DL_novel_guide_editor/v84_evals/fna_ins_discovery_v5_md_seed2/adapted.jsonl")
SWEEP = Path("/global/home/users/kh36969/tools/DL_RNA_guide_edotor_classifer/config/v3_sweep_cids.txt")
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Build a filesystem-wide {assembly_id -> path} index at startup. The
# previous hardcoded species→dir map was wrong for most species (genomes
# live in `genomes_<species>/` not `genomes/<species>/`, and some species
# tags don't match the dir name — `sa_k24` → `genomes_saureus`). Scanning
# removes the mapping entirely: existence is a direct lookup by assembly.
def _build_assembly_index(root: Path) -> dict:
    """Walk every `genomes*` dir (top-level and nested one deep) and index
    every .fna recursively. Non-`__done_` copies win on duplicates.
    """
    idx = {}
    cands = []
    for p in root.iterdir():
        if not p.is_dir(): continue
        if p.name.startswith("genomes"): cands.append(p)
        for sub in p.glob("genomes*"):
            if sub.is_dir(): cands.append(sub)
    for d in cands:
        archive = "__done_" in str(d)
        for fna in d.rglob("*.fna"):
            if archive and fna.stem in idx:
                continue
            idx[fna.stem] = fna
    return idx
ASSEMBLY_INDEX = {}  # populated at main() time

SITE_SUFFIX_RE = re.compile(r'\.(?:site|copy\d+)$')
COMP = str.maketrans("ACGT", "TGCA")
def rc(s): return s.translate(COMP)[::-1]


def log(*a):
    print(*a, flush=True)


log("[0/7] building filesystem-wide assembly index …")
ASSEMBLY_INDEX = _build_assembly_index(ROOT)
log(f"  indexed {len(ASSEMBLY_INDEX)} assembly FASTAs across genomes* dirs")

log("[1/7] loading sweep scope + adapted.jsonl …")
sweep_bags = set()
with SWEEP.open() as f:
    for line in f:
        line = line.strip()
        if line: sweep_bags.add(line)
log(f"  sweep scope: {len(sweep_bags)} bag IDs")

sites = []
with ADAPTED.open() as f:
    for row_idx, line in enumerate(f):
        r = json.loads(line)
        bid = r["transposase_id"]
        sid = r["site_id"]
        db_id = SITE_SUFFIX_RE.sub("", sid)
        flank = r["inputs"]["flank"].upper()
        ncs = [x.upper() for x in r["inputs"]["noncoding_regions"]]
        sites.append(dict(
            jsonl_row=row_idx, bag_id=bid, site_id=sid, db_id=db_id,
            flank=flank, nc_frags=ncs,
        ))
log(f"  adapted.jsonl: {len(sites)} sites across {len(set(s['bag_id'] for s in sites))} bags")


log("[2/7] loading cds_v3/insert_cds.tsv (bag → events) …")
cds_f = ROOT / "cds_v3" / "insert_cds.tsv"
bag_db_events = {}
bag_event_count = Counter()
all_bag_ids = {s["bag_id"] for s in sites}
with cds_f.open() as f:
    hdr = f.readline().rstrip("\n").split("\t")
    c_cid = hdr.index("cds_cluster_id"); c_db = hdr.index("db_id")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[c_cid] in all_bag_ids:
            bag_db_events[(p[c_cid], p[c_db])] = dict(zip(hdr, p))
            bag_event_count[p[c_cid]] += 1
log(f"  insert_cds.tsv: {len(bag_db_events)} (bag, db_id) pairs")
log(f"  bags present: {len(bag_event_count)}")


log("[3/7] loading database_v3/insertions.tsv (db_id → locus) …")
ins_f = ROOT / "database_v3" / "insertions.tsv"
db_to_locus = {}
need_db_ids = {s["db_id"] for s in sites}
with ins_f.open() as f:
    hdr = f.readline().rstrip("\n").split("\t")
    c_db = hdr.index("db_id")
    for line in f:
        p = line.rstrip("\n").split("\t")
        if p[c_db] in need_db_ids:
            db_to_locus[p[c_db]] = dict(zip(hdr, p))
log(f"  insertions.tsv: {len(db_to_locus)} / {len(need_db_ids)} db_ids resolved")


log("[4/7] loading each referenced l2/loci.tsv …")
panel_cache = {}
need_panels = set()
for d in db_to_locus.values():
    lid = d.get("locus_id", "")
    if "|" not in lid: continue
    pref, _ = lid.split("|", 1); pr, gd = pref.split("/", 1)
    need_panels.add((pr, gd))
log(f"  unique panel/genome_dir pairs: {len(need_panels)}")

missing_loci_file = 0
loaded = 0
for pr, gd in need_panels:
    pfile = ROOT / pr / "l2" / gd / "loci.tsv"
    if not pfile.exists():
        missing_loci_file += 1
        panel_cache[(pr, gd)] = None
        continue
    rows = {}
    try:
        with pfile.open() as f:
            hdr = f.readline().rstrip("\n").split("\t")
            c_lid = hdr.index("locus_id")
            for line in f:
                p = line.rstrip("\n").split("\t")
                rows[p[c_lid]] = dict(zip(hdr, p))
    except Exception as e:
        rows = None
    panel_cache[(pr, gd)] = rows
    if rows is not None: loaded += 1
log(f"  loaded loci.tsv for {loaded} panels; missing: {missing_loci_file}")

alleles_exists = {pkey: (ROOT / pkey[0] / "l2" / pkey[1] / "alleles.fna").exists()
                   for pkey in need_panels}


log("[5/7] computing joined / raw_files_exist / boundary_set_complete …")

def get_locus_row(db_id):
    d = db_to_locus.get(db_id)
    if not d: return None, None, None
    lid = d.get("locus_id", "")
    if "|" not in lid: return None, None, None
    pref, short = lid.split("|", 1); pr, gd = pref.split("/", 1)
    cache = panel_cache.get((pr, gd))
    if cache is None: return (pr, gd), None, lid
    return (pr, gd), cache.get(short), lid


for s in sites:
    # joined
    d = db_to_locus.get(s["db_id"])
    pkey, locus_row, locus_full = None, None, ""
    joined = "fail:no_event_in_insert_cds"
    if (s["bag_id"], s["db_id"]) in bag_db_events:
        if d:
            pkey, locus_row, locus_full = get_locus_row(s["db_id"])
            if locus_row: joined = "ok"
            elif pkey and panel_cache.get(pkey) is None:
                joined = "fail:panel_loci_missing"
            else:
                joined = "fail:locus_short_id_not_in_panel"
        else:
            joined = "fail:db_id_not_in_insertions"
    s["locus_row"] = locus_row
    s["panel_key"] = pkey
    s["locus_id_full"] = locus_full
    s["joined"] = joined

    # raw_files_exist — DISTINCT buckets per file type
    raw = "fail:no_locus"
    if locus_row and pkey:
        pr, gd = pkey
        base = ROOT / pr / "l2" / gd
        have_loci = (base / "loci.tsv").exists()
        have_alleles = alleles_exists.get(pkey, False)
        seed = locus_row.get("seed_genome", "")
        have_seed = bool(seed and seed in ASSEMBLY_INDEX)
        if have_loci and have_alleles and have_seed: raw = "ok"
        elif not have_loci: raw = "fail:loci_tsv_missing"
        elif not have_alleles: raw = "fail:alleles_fna_missing"
        else: raw = "fail:seed_genome_missing"
    s["raw_files_exist"] = raw

    # boundary_set_complete — STRICT: insert_start_in_long_bp present
    #   partial:single_cutpoint_only: has junction_L_abs + start → fallback
    #   fail: no locus or no boundary fields at all
    bset = "fail:no_locus"
    if locus_row:
        istart = locus_row.get("insert_start_in_long_bp", "")
        jL = locus_row.get("junction_L_abs", "")
        jR = locus_row.get("junction_R_abs", "")
        ndec = locus_row.get("n_equivalent_decompositions", "")
        start = locus_row.get("start", "")
        if istart.lstrip("-").isdigit() and jL and jR and ndec:
            bset = "ok"
        elif jL and start and ndec and jL.lstrip("-").isdigit() and start.lstrip("-").isdigit():
            bset = "partial:single_cutpoint_only"
        else:
            missing = []
            if not istart: missing.append("insert_start_in_long_bp")
            if not jL: missing.append("junction_L_abs")
            if not jR: missing.append("junction_R_abs")
            if not ndec: missing.append("n_equivalent_decompositions")
            bset = "fail:missing(" + ",".join(missing) + ")"
    s["boundary_set_complete"] = bset

    s["orf_msa_ready"] = "pending"
    s["target_coord_ready"] = "pending"
log("  axes 1,2,4 computed")


log("[6/7] sequence_recovered (flank in empty + nc in occupied) …")
sites_by_panel = defaultdict(list)
for s in sites:
    if s["raw_files_exist"] == "ok" and s["locus_row"]:
        sites_by_panel[s["panel_key"]].append(s)
log(f"  panels to scan: {len(sites_by_panel)}")

def read_fasta_select(fna_path, want):
    """want: {short_id -> set of allele_ids}. Returns
    {short_id: {allele_id: seq}}."""
    out = defaultdict(dict)
    cur_id = None; cur_seq = []
    def flush():
        if cur_id is None: return
        p = cur_id.split("|")
        if len(p) >= 2 and p[0] in want and p[1] in want[p[0]]:
            out[p[0]][p[1]] = "".join(cur_seq).upper()
    with fna_path.open() as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith(">"):
                flush(); cur_id = line[1:]; cur_seq = []
            else:
                cur_seq.append(line)
    flush()
    return out

for pidx, (pkey, slist) in enumerate(sites_by_panel.items(), 1):
    pr, gd = pkey
    fna = ROOT / pr / "l2" / gd / "alleles.fna"
    want = defaultdict(set)
    pair_by_short = {}
    for s in slist:
        if "|" not in s["locus_id_full"]: continue
        short = s["locus_id_full"].split("|", 1)[1]
        lr = s["locus_row"]
        long_id = lr.get("longest_allele", "")
        short_id = lr.get("shortest_allele", "")
        if long_id and short_id:
            want[short].update([long_id, short_id])
            pair_by_short[short] = (long_id, short_id)
    try:
        alleles = read_fasta_select(fna, want)
    except Exception as e:
        for s in slist: s["sequence_recovered"] = f"fail:alleles_read_error({str(e)[:60]})"
        continue
    for s in slist:
        if "|" not in s["locus_id_full"]:
            s["sequence_recovered"] = "fail:no_short_id"; continue
        short = s["locus_id_full"].split("|", 1)[1]
        pair = pair_by_short.get(short)
        if not pair:
            s["sequence_recovered"] = "fail:no_allele_ids"; continue
        long_id, short_id = pair
        a_map = alleles.get(short, {})
        occ = a_map.get(long_id, "")
        emp = a_map.get(short_id, "")
        if not occ: s["sequence_recovered"] = f"fail:longest({long_id})_absent"; continue
        if not emp: s["sequence_recovered"] = f"fail:shortest({short_id})_absent"; continue
        flank = s["flank"]
        f_total = emp.count(flank) + emp.count(rc(flank))
        nc_hits = [occ.count(nf) + occ.count(rc(nf)) for nf in s["nc_frags"]]
        if f_total == 1 and all(h == 1 for h in nc_hits):
            s["sequence_recovered"] = "ok"
        elif f_total == 0:
            s["sequence_recovered"] = "fail:flank_not_in_empty"
        elif any(h == 0 for h in nc_hits):
            n_zero = sum(1 for h in nc_hits if h == 0)
            s["sequence_recovered"] = f"fail:nc_frag_not_in_occupied(n_zero={n_zero})"
        else:
            s["sequence_recovered"] = f"ambiguous:flank={f_total},nc={','.join(map(str,nc_hits))}"
    if pidx % 100 == 0 or pidx == len(sites_by_panel):
        log(f"  …panel {pidx}/{len(sites_by_panel)}")

for s in sites:
    if "sequence_recovered" not in s:
        s["sequence_recovered"] = "fail:raw_files_not_ok"


log("[7/7] emitting per-site TSV + per-bag + summaries …")

cols = ["jsonl_row", "bag_id", "site_id", "db_id", "in_sweep_1173",
        "locus_id_full", "joined", "raw_files_exist", "sequence_recovered",
        "boundary_set_complete", "orf_msa_ready", "target_coord_ready"]
per_site = OUT_DIR / "per_site_audit.final.tsv"
with per_site.open("w") as fh:
    w = csv.writer(fh, delimiter="\t")
    w.writerow(cols)
    for s in sites:
        in_sweep = "1" if s["bag_id"] in sweep_bags else "0"
        w.writerow([s["jsonl_row"], s["bag_id"], s["site_id"], s["db_id"],
                    in_sweep, s["locus_id_full"], s["joined"],
                    s["raw_files_exist"], s["sequence_recovered"],
                    s["boundary_set_complete"], s["orf_msa_ready"],
                    s["target_coord_ready"]])
log(f"  wrote {per_site}")

# Per-bag summary (strict + partial-boundary variants, counted separately)
by_bag = defaultdict(list)
for s in sites: by_bag[s["bag_id"]].append(s)
with (OUT_DIR / "per_bag_summary.tsv").open("w") as f:
    f.write("bag_id\tin_sweep_1173\tn_sites_total\tn_joined_ok\tn_raw_files_ok\t"
            "n_sequence_recovered_ok\tn_boundary_strict_ok\tn_boundary_partial\t"
            "n_raw_source_ready_strict\tn_raw_source_ready_with_partial_bndry\n")
    bag_rows = []
    for b, bl in by_bag.items():
        in_sweep = "1" if b in sweep_bags else "0"
        def c(ax, prefix="ok"): return sum(1 for r in bl if r[ax].startswith(prefix))
        n_bnd_ok = c("boundary_set_complete")
        n_bnd_partial = sum(1 for r in bl if r["boundary_set_complete"] == "partial:single_cutpoint_only")
        n_strict = sum(1 for r in bl if r["joined"] == "ok"
                        and r["raw_files_exist"] == "ok"
                        and r["sequence_recovered"] == "ok"
                        and r["boundary_set_complete"] == "ok")
        n_with_partial = sum(1 for r in bl if r["joined"] == "ok"
                              and r["raw_files_exist"] == "ok"
                              and r["sequence_recovered"] == "ok"
                              and r["boundary_set_complete"] in ("ok", "partial:single_cutpoint_only"))
        bag_rows.append((b, in_sweep, len(bl), c("joined"), c("raw_files_exist"),
                         c("sequence_recovered"), n_bnd_ok, n_bnd_partial,
                         n_strict, n_with_partial))
    bag_rows.sort(key=lambda t: (t[1] != "1", -t[8], -t[9], t[0]))
    for row in bag_rows:
        f.write("\t".join(str(x) for x in row) + "\n")
log(f"  wrote per_bag_summary.tsv ({len(bag_rows)} bags)")

# Pilot lists (strict-only and with-partial)
sweep_strict = sorted([t for t in bag_rows if t[1] == "1" and t[8] >= 4],
                       key=lambda t: (-t[8], t[0]))
sweep_partial = sorted([t for t in bag_rows if t[1] == "1" and t[9] >= 4],
                        key=lambda t: (-t[9], t[0]))
all_strict = sorted([t for t in bag_rows if t[8] >= 4],
                     key=lambda t: (-t[8], t[0]))
all_partial = sorted([t for t in bag_rows if t[9] >= 4],
                      key=lambda t: (-t[9], t[0]))
with (OUT_DIR / "pilot_candidate_bags_sweep_strict.txt").open("w") as f:
    for t in sweep_strict: f.write(f"{t[0]}\t{t[8]}\n")
with (OUT_DIR / "pilot_candidate_bags_sweep_with_partial.txt").open("w") as f:
    for t in sweep_partial: f.write(f"{t[0]}\t{t[9]}\n")
with (OUT_DIR / "pilot_candidate_bags_upstream_strict.txt").open("w") as f:
    for t in all_strict: f.write(f"{t[0]}\t{t[8]}\n")
with (OUT_DIR / "pilot_candidate_bags_upstream_with_partial.txt").open("w") as f:
    for t in all_partial: f.write(f"{t[0]}\t{t[9]}\n")


def summarize(label, rr):
    log(f"\n### COVERAGE — {label} ({len(rr)} sites, "
        f"{len(set(r['bag_id'] for r in rr))} bags) ###")
    for ax in ("joined", "raw_files_exist", "sequence_recovered",
               "boundary_set_complete"):
        c = Counter(r[ax] for r in rr)
        log(f"  {ax}:")
        for k in sorted(c, key=lambda kk: (not kk.startswith("ok"), -c[kk])):
            log(f"    {c[k]:7d}  {k}")

def bag_hist(label, rr):
    by_b = defaultdict(list)
    for r in rr: by_b[r["bag_id"]].append(r)
    strict_hist = Counter(); partial_hist = Counter()
    for b, bl in by_b.items():
        ns = sum(1 for r in bl if r["joined"]=="ok" and r["raw_files_exist"]=="ok"
                 and r["sequence_recovered"]=="ok" and r["boundary_set_complete"]=="ok")
        np = sum(1 for r in bl if r["joined"]=="ok" and r["raw_files_exist"]=="ok"
                 and r["sequence_recovered"]=="ok"
                 and r["boundary_set_complete"] in ("ok","partial:single_cutpoint_only"))
        strict_hist[ns] += 1; partial_hist[np] += 1
    log(f"\n### RAW-SOURCE-READY HISTOGRAM — {label} ###")
    log(f"  (strict = boundary_set_complete=ok; +partial includes partial_single_cutpoint)")
    log(f"  threshold | strict | +partial")
    for thr in (8,7,6,5,4,3,2,1):
        s = sum(v for k,v in strict_hist.items() if k >= thr)
        p = sum(v for k,v in partial_hist.items() if k >= thr)
        log(f"      >={thr}   |  {s:4d}  |   {p:4d}")

summarize("1173-bag sweep scope", [s for s in sites if s["bag_id"] in sweep_bags])
summarize("4380-bag upstream (all)", sites)
bag_hist("1173-bag sweep scope", [s for s in sites if s["bag_id"] in sweep_bags])
bag_hist("4380-bag upstream (all)", sites)

log(f"\nDONE — per-site TSV: {per_site}")
