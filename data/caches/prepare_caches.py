#!/usr/bin/env python3
"""
prepare_caches.py — inventory, verify and stage the corpus caches for publication.

Drop this into  znou-publish\\data\\caches\\  and run it.

    python prepare_caches.py                  # inventory + verify, writes nothing but reports
    python prepare_caches.py --extract         # also extract each archive under _extracted/
    python prepare_caches.py --extract --strip-text wiki103_partial870k

It NEVER modifies or deletes the .zip files. Everything it produces goes into
_extracted/ and _reports/ next to them.

What it does
------------
1. Finds every *.zip, hashes it, flags byte-identical duplicates.
2. Extracts (with --extract) and renames folder + inner files together, so
   cache_analyzer.py keeps working: it derives run_name from the directory name
   and expects {run_name}.jsonl and {run_name}_{quadrant}.bin inside.
3. Verifies each archive against its own report.md:
     - the four .bin are 3072 x uint32 = 12288 bytes
     - nonzero count per .bin == the coverage number in report.md
     - sum of hits per .bin == jsonl row count
     - jsonl rows vs report.md "Sentences Processed" -> the silent-filter gap
       (data_pipeline.py drops <3-word fragments and swallowed exceptions,
        but increments total_sentences anyway)
4. Scans checkpoint.json for any long string values, i.e. leaked corpus text.
5. Computes the terminal-character distribution and its Shannon entropy per
   corpus. This is the cheap stdlib proxy for SPEC 8.2's "final char" table and
   the composition axis Q6 wants. Use --terminal-tokens for real GPT-2 token
   ids (needs transformers; CPU only, no model forward pass).
6. Writes a 50-row sample.jsonl per corpus so a reader can see the format.
7. Optionally strips the "s" field from named corpora (--strip-text), keeping
   destinations and terminal token. For wiki103, which is CC BY-SA.
8. Writes _reports/summary.tsv and _reports/CACHES.md.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

DIMENSION = 3072
QUADRANTS = ["exp_r", "exp_i", "imp_r", "imp_i"]

# Gutenberg download naming -> readable slug. The run date is kept: it is the
# only thing distinguishing this run from a future re-chunked one.
TITLES = {
    "alice_full":                        "alice-in-wonderland",
    "bookdead_full":                     "book-of-the-dead",
    "buddinggrove_full":                 "within-a-budding-grove",
    "crimandpunishment_full":            "crime-and-punishment",
    "dollshouse_full":                   "a-dolls-house",
    "DuCoteProust_full":                 "du-cote-de-chez-swann-FR",
    "kinginYellow_full":                 "the-king-in-yellow",
    "Leviathan_full":                    "leviathan",
    "Machiavelli_full":                  "the-prince",
    "Metamorphosis_full":                "the-metamorphosis",
    "moby_dick_full":                    "moby-dick",
    "Poe_full":                          "poe-collected",
    "shakespeare_full":                  "shakespeare-complete",
    "swannsway_full":                    "swanns-way-EN",
    "TheYellowWallpaper_full":           "the-yellow-wallpaper",
    "Tractatus Logico-Philosophicus_full": "tractatus",
    "Ulysses_full":                      "ulysses",
    "war_and_peace_full":                "war-and-peace",
    "Zarathustra_full":                  "thus-spoke-zarathustra",
    "wiki103_partial870k":               "wikitext-103-partial870k",
}

# Files that should not be sitting in a caches directory.
MISPLACED = {
    "moby_dick.txt": "source corpus - belongs at data/moby_dick.txt (SPEC 8 "
                     "says both states ship; currently only one does)",
    "the_sea.json": "the UNTRIMMED 7394-entry raw corpus - NOT a duplicate of "
                    "data/the_sea.json, which is the 7353-row computed cache. "
                    "The two names mean opposite things here. Provenance only; "
                    "do not delete",
    "the_sea_sailed.json": "prototype save file, contains 13 hand-typed "
                           "arrivals incl. neuron 2256 ('as', 'asdf')",
}


def sha256_file(path: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def slug_for(run_name: str) -> str:
    """Map a run directory name like 'alice_full_2025-11-05' to a readable slug."""
    m = re.match(r"^(.*)_(\d{4}-\d{2}-\d{2}.*)$", run_name)
    if not m:
        return run_name
    stem, date = m.group(1), m.group(2)
    return f"{TITLES.get(stem, stem)}_{date}"


def read_u32(path: Path):
    """Read a .bin as a list of uint32. Stdlib only, avoids a numpy dependency."""
    import array
    data = path.read_bytes()
    a = array.array("I")
    a.frombytes(data)
    if sys.byteorder != "little":
        a.byteswap()
    return a


def parse_report(path: Path) -> dict:
    """Pull the numbers out of a report.md written by data_pipeline.py."""
    out = {"coverage": {}, "sentences": None, "lines": None, "source": None}
    if not path.is_file():
        return out
    text = path.read_text(encoding="utf-8", errors="replace")

    m = re.search(r"Sentences Processed:\*\*\s*`([\d,]+)`", text)
    if m:
        out["sentences"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Lines Processed:\*\*\s*`([\d,]+)\s*/", text)
    if m:
        out["lines"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Source Dataset:\*\*\s*`([^`]+)`", text)
    if m:
        out["source"] = m.group(1)

    label_to_key = {"exp r": "exp_r", "exp i": "exp_i",
                    "imp r": "imp_r", "imp i": "imp_i"}
    for label, num in re.findall(r"\|\s*(Exp R|Exp I|Imp R|Imp I)\s*\|\s*`([\d,]+)`", text):
        out["coverage"][label_to_key[label.lower()]] = int(num.replace(",", ""))
    return out


def shannon_entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    e = -sum((c / total) * math.log2(c / total) for c in counter.values())
    return 0.0 if e == 0 else e


def scan_jsonl(path: Path, want_tokens: bool, tokenizer=None) -> dict:
    """One pass over the jsonl. Returns row counts, distinct destinations,
    terminal-character and (optionally) terminal-token distributions."""
    rows = 0
    malformed = 0
    no_text = 0
    distinct = {q: set() for q in QUADRANTS}
    hits = {q: Counter() for q in QUADRANTS}
    final_char = Counter()
    final_token = Counter()
    lengths = []

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            rows += 1
            for q in QUADRANTS:
                v = d.get(q)
                if isinstance(v, int):
                    distinct[q].add(v)
                    hits[q][v] += 1
            s = d.get("s")
            if isinstance(s, str) and s:
                lengths.append(len(s))
                final_char[s[-1]] += 1
                if want_tokens and tokenizer is not None:
                    ids = tokenizer(s)["input_ids"]
                    if ids:
                        final_token[ids[-1]] += 1
            else:
                no_text += 1

    lengths.sort()
    return {
        "rows": rows,
        "malformed": malformed,
        "no_text": no_text,
        "distinct": {q: len(distinct[q]) for q in QUADRANTS},
        "hits": hits,
        "final_char": final_char,
        "final_token": final_token,
        "char_entropy": shannon_entropy(final_char),
        "token_entropy": shannon_entropy(final_token) if final_token else None,
        "len_median": lengths[len(lengths) // 2] if lengths else 0,
        "len_mean": (sum(lengths) / len(lengths)) if lengths else 0,
    }


def inspect_checkpoint(path: Path) -> list:
    """Flag any long string values - i.e. leaked corpus text in a checkpoint."""
    findings = []
    if not path.is_file():
        return findings
    try:
        d = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except Exception as e:
        return [f"unreadable: {e}"]

    def walk(node, trail="")\
            :
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{trail}.{k}" if trail else str(k))
        elif isinstance(node, list):
            for i, v in enumerate(node[:50]):
                walk(v, f"{trail}[{i}]")
        elif isinstance(node, str) and len(node) > 40:
            findings.append(f"{trail}: {len(node)} chars — {node[:60]!r}")

    walk(d)
    return findings


def strip_text_field(src: Path, dst: Path, tokenizer=None) -> int:
    """Write a copy of the jsonl with 's' removed. Keeps destinations, and adds
    the final token id when a tokenizer is supplied, so terminal-token entropy
    stays computable without redistributing text."""
    n = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fi, \
         open(dst, "w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            s = d.pop("s", None)
            if tokenizer is not None and isinstance(s, str) and s:
                ids = tokenizer(s)["input_ids"]
                if ids:
                    d["t"] = ids[-1]
            fo.write(json.dumps(d) + "\n")
            n += 1
    return n


def write_sample(src: Path, dst: Path, n: int = 50):
    with open(src, "r", encoding="utf-8", errors="replace") as fi, \
         open(dst, "w", encoding="utf-8") as fo:
        for i, line in enumerate(fi):
            if i >= n:
                break
            fo.write(line)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="directory holding the .zip archives (default: .)")
    ap.add_argument("--extract", action="store_true",
                    help="extract archives into _extracted/ with renamed files")
    ap.add_argument("--strip-text", nargs="*", default=[], metavar="NAME",
                    help="corpora whose jsonl 's' field should be removed "
                         "(substring match, e.g. wiki103)")
    ap.add_argument("--terminal-tokens", action="store_true",
                    help="compute GPT-2 terminal token ids (needs transformers)")
    ap.add_argument("--sample-rows", type=int, default=50)
    args = ap.parse_args()

    root = args.root.resolve()
    work = root / "_extracted"
    reports = root / "_reports"
    reports.mkdir(exist_ok=True)

    tokenizer = None
    if args.terminal_tokens:
        try:
            from transformers import GPT2TokenizerFast
            # Pinned to match SPEC section 7. Without this the token ids are
            # not reproducible.
            tokenizer = GPT2TokenizerFast.from_pretrained(
                "gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
        except Exception as e:
            print(f"! could not load tokenizer ({e}); continuing without token ids")

    zips = sorted(p for p in root.glob("*.zip"))
    if not zips:
        print(f"No .zip archives found in {root}")
        return

    print(f"{len(zips)} archives in {root}\n")

    # --- duplicate detection -------------------------------------------------
    by_hash = {}
    for z in zips:
        h = sha256_file(z)
        by_hash.setdefault(h, []).append(z.name)
    for h, names in by_hash.items():
        if len(names) > 1:
            print(f"! DUPLICATE ({h[:12]}…): {', '.join(names)}")
            print("  keep one, delete the rest\n")

    # --- misplaced loose files ----------------------------------------------
    for name, why in MISPLACED.items():
        if (root / name).exists():
            print(f"! {name} is here — {why}\n")

    rows_out = []

    for z in zips:
        with zipfile.ZipFile(z) as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
        if not members:
            print(f"── {z.name}\n   ! empty archive\n")
            continue

        # Strip a shared top-level folder if the archive has one.
        tops = {Path(m).parts[0] for m in members if len(Path(m).parts) > 1}
        prefix = tops.pop() if len(tops) == 1 and all(
            len(Path(m).parts) > 1 for m in members) else None

        def rel_of(member: str) -> Path:
            p = Path(member)
            return Path(*p.parts[1:]) if prefix and p.parts[0] == prefix else p

        # The run name is whatever the .jsonl inside is called — not the zip
        # filename, which can disagree after copying or renaming.
        jsonl_members = [m for m in members if m.endswith(".jsonl")]
        if jsonl_members:
            run_name = Path(jsonl_members[0]).stem
        else:
            run_name = prefix or z.stem
        slug = slug_for(run_name)
        print(f"── {run_name}   [{z.name}]\n   → {slug}")

        dest = work / slug
        if args.extract:
            if dest.exists():
                shutil.rmtree(dest)
            dest.mkdir(parents=True)
            with zipfile.ZipFile(z) as zf:
                for member in members:
                    rel = rel_of(member)
                    # rename inner files so cache_analyzer.py still resolves them
                    new_name = rel.name.replace(run_name, slug)
                    target = dest / rel.parent / new_name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as fi, open(target, "wb") as fo:
                        shutil.copyfileobj(fi, fo)
        elif not dest.is_dir():
            print("   (not extracted — rerun with --extract for full checks)\n")
            continue

        report = parse_report(dest / "report.md")
        jsonl = dest / f"{slug}.jsonl"
        if not jsonl.is_file():
            print("   ! no jsonl found\n")
            continue

        # --- verify the bins against their own report ------------------------
        bin_ok = True
        bin_nonzero, bin_sum = {}, {}
        for q in QUADRANTS:
            p = dest / f"{slug}_{q}.bin"
            if not p.is_file():
                print(f"   ! missing {p.name}")
                bin_ok = False
                continue
            size = p.stat().st_size
            if size != DIMENSION * 4:
                print(f"   ! {p.name} is {size} bytes, expected {DIMENSION * 4}")
                bin_ok = False
                continue
            arr = read_u32(p)
            bin_nonzero[q] = sum(1 for v in arr if v)
            bin_sum[q] = sum(arr)
            claimed = report["coverage"].get(q)
            if claimed is not None and claimed != bin_nonzero[q]:
                print(f"   ! {q}: report.md says {claimed}, bin has {bin_nonzero[q]}")
                bin_ok = False

        stats = scan_jsonl(jsonl, args.terminal_tokens, tokenizer)

        for q in QUADRANTS:
            if q in bin_sum and bin_sum[q] != stats["rows"]:
                print(f"   ! {q}: bin hits {bin_sum[q]:,} != jsonl rows {stats['rows']:,}")
            if q in bin_nonzero and bin_nonzero[q] != stats["distinct"][q]:
                print(f"   ! {q}: bin nonzero {bin_nonzero[q]} != "
                      f"jsonl distinct {stats['distinct'][q]}")

        dropped = (report["sentences"] - stats["rows"]) if report["sentences"] else None
        if dropped:
            pct = 100 * dropped / report["sentences"]
            print(f"   filtered out: {dropped:,} of {report['sentences']:,} "
                  f"units ({pct:.1f}%) — <3 words or swallowed exception")

        leaks = inspect_checkpoint(dest / "checkpoint.json")
        for line in leaks:
            print(f"   ! checkpoint.json holds text — {line}")

        print(f"   rows {stats['rows']:,} · mean len {stats['len_mean']:.0f} "
              f"· final-char entropy {stats['char_entropy']:.3f} bits"
              + (f" · token entropy {stats['token_entropy']:.3f} bits"
                 if stats["token_entropy"] else "")
              + (" · bins OK" if bin_ok else " · BINS FAILED"))

        write_sample(jsonl, dest / "sample.jsonl", args.sample_rows)

        if any(k.lower() in run_name.lower() or k.lower() in z.name.lower()
               for k in args.strip_text):
            out = dest / f"{slug}.stripped.jsonl"
            n = strip_text_field(jsonl, out, tokenizer)
            jsonl.unlink()
            print(f"   text stripped → {out.name} ({n:,} rows, no source text)")

        big = [p for p in dest.rglob("*") if p.is_file() and p.stat().st_size > 95 * 1024**2]
        for p in big:
            print(f"   ! {p.name} is {p.stat().st_size / 1024**2:.0f}MB — "
                  f"GitHub blocks files over 100MB")

        rows_out.append({
            "slug": slug,
            "original_run": run_name,
            "source": report["source"] or "",
            "lines": report["lines"] or "",
            "units_reported": report["sentences"] or "",
            "jsonl_rows": stats["rows"],
            "filtered_out": dropped if dropped is not None else "",
            "mean_len": round(stats["len_mean"], 1),
            "median_len": stats["len_median"],
            "final_char_entropy_bits": round(stats["char_entropy"], 4),
            "final_token_entropy_bits": (round(stats["token_entropy"], 4)
                                         if stats["token_entropy"] else ""),
            **{f"coverage_{q}": stats["distinct"][q] for q in QUADRANTS},
            "zip_sha256": sha256_file(z)[:16],
        })
        print()

    # --- outputs -------------------------------------------------------------
    if rows_out:
        tsv = reports / "summary.tsv"
        with open(tsv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(rows_out)
        print(f"wrote {tsv}")

        md = reports / "CACHES.md"
        with open(md, "w", encoding="utf-8") as f:
            f.write("# Corpus caches\n\n")
            f.write(
                "Produced by `data_pipeline.py` on 2025-11-05. **The unit is not a\n"
                "sentence.** The pipeline applied NLTK punkt to each physical line of\n"
                "hard-wrapped source text, so units are line-bounded fragments; most\n"
                "end mid-clause on an ordinary word. Coverage figures here are\n"
                "therefore **not comparable with SPEC 8.2, 8.2.1 or 8.3**, whose corpus\n"
                "was prepared with paragraphs unwrapped first.\n\n"
                "These runs also predate SPEC 7's configuration discipline: no\n"
                "`revision` pin, no TF32 flags, no `use_deterministic_algorithms`.\n"
                "The function itself is unchanged — layer 5, `mlp.hook_post`, same four\n"
                "quadrants — but the stack was torch 2.4.0+cu121 on an RTX A4500 under\n"
                "Ubuntu 24.04, not the reference stack.\n\n"
                "`units_reported` counts every fragment the pipeline saw;\n"
                "`jsonl_rows` counts those that survived its under-3-words filter.\n\n"
            )
            f.write("| corpus | units | rows | coverage imp_r | imp_i | "
                    "final-char entropy |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            for r in rows_out:
                f.write(f"| {r['slug']} | {r['units_reported']} | {r['jsonl_rows']} | "
                        f"{r['coverage_imp_r']} | {r['coverage_imp_i']} | "
                        f"{r['final_char_entropy_bits']} |\n")
        print(f"wrote {md}")


if __name__ == "__main__":
    main()
