# export_hits.py — extract corpus hit counts from znou_exchange.db
#
# token_sweep.py's ghost test (SPEC 4.4) reads MasterHitCounts out of the live
# database. The database is not published: it holds operator activity. The hit
# counts are not operator activity — they are corpus coverage per neuron per
# quadrant, and SPEC 4.4 and 8.3 both rest on them.
#
# This pulls out that one table and nothing else, so 4.4 becomes reproducible
# from the published bundle.
#
# Opens the database READ-ONLY. Nothing is modified.
#
# USAGE
#     python export_hits.py --db starmap\znou_exchange.db
#     python export_hits.py --db starmap\znou_exchange.db --out results\master_hit_counts.tsv
#     python export_hits.py --db starmap\znou_exchange.db --inspect
#
# --inspect lists the tables and columns without exporting, in case the schema
# differs from what is assumed here.

import argparse
import hashlib
import os
import sqlite3
import sys

DEFAULT_OUT = "master_hit_counts.tsv"
M = 3072  # |J|


def connect_ro(path):
    if not os.path.isfile(path):
        sys.exit(f"no such file: {path}")
    uri = "file:" + os.path.abspath(path).replace("\\", "/") + "?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inspect(db):
    cur = db.execute("select name from sqlite_master where type='table' order by name")
    tables = [r[0] for r in cur]
    if not tables:
        print("no tables found")
        return
    for t in tables:
        try:
            cols = [(r[1], r[2]) for r in db.execute(f'pragma table_info("{t}")')]
            n = db.execute(f'select count(*) from "{t}"').fetchone()[0]
        except sqlite3.Error as e:
            print(f"{t}: unreadable ({e})")
            continue
        print(f"\n{t}  ({n} rows)")
        for name, typ in cols:
            print(f"    {name} {typ}")


def export(db, out_path):
    try:
        rows = list(db.execute(
            "select quadrant_key, neuron_id, corpus_hits "
            "from MasterHitCounts order by quadrant_key, neuron_id"))
    except sqlite3.Error as e:
        sys.exit(f"could not read MasterHitCounts: {e}\n"
                 f"run with --inspect to see the actual schema")

    if not rows:
        sys.exit("MasterHitCounts is empty")

    by_quad = {}
    out_of_range = 0
    for quad, j, hits in rows:
        if not isinstance(j, int) or j < 0 or j >= M:
            out_of_range += 1
            continue
        by_quad.setdefault(quad, {})[j] = int(hits or 0)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        fh.write("# corpus hit counts per neuron per quadrant\n")
        fh.write("# source: MasterHitCounts, znou_exchange.db (read-only export)\n")
        fh.write("# |J| = 3072, neuron ids 0-based\n")
        fh.write("quadrant\tneuron\tcorpus_hits\n")
        for quad in sorted(by_quad):
            counts = by_quad[quad]
            for j in range(M):
                fh.write(f"{quad}\t{j}\t{counts.get(j, 0)}\n")

    h = hashlib.sha256()
    with open(out_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)

    print(f"wrote {out_path}")
    print(f"  sha256 {h.hexdigest()}")
    print()
    print(f"{'quadrant':22} {'reached':>8} {'unreached':>10} {'total hits':>12}")
    for quad in sorted(by_quad):
        counts = by_quad[quad]
        reached = sum(1 for j in range(M) if counts.get(j, 0) > 0)
        print(f"{quad:22} {reached:8d} {M - reached:10d} "
              f"{sum(counts.values()):12d}")
    if out_of_range:
        print(f"\n  {out_of_range} row(s) had a neuron id outside [0, {M}) and were skipped")
    print()
    print("Cross-check against SPEC 8.3: imp_r 1452, exp_r 1339, imp_i 2225,")
    print("exp_i 2151, union 2336, never reached 736. If these disagree, the")
    print("database is not the one 8.3 was computed from — say so rather than")
    print("adjusting the document.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to znou_exchange.db")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--inspect", action="store_true",
                    help="list tables and columns, export nothing")
    args = ap.parse_args()

    db = connect_ro(args.db)
    try:
        if args.inspect:
            inspect(db)
        else:
            export(db, args.out)
    finally:
        db.close()


if __name__ == "__main__":
    main()
