#!/usr/bin/env python3
"""
Rank neurons as candidate 'game rooms' by phrase-family concentration.

For each sufficiently-sampled neuron, report the word n-grams most *enriched*
among its hits relative to the corpus background.

Enrichment, not raw frequency, is the point: raw frequency returns 'of the'
for every neuron. Enrichment recovers 'up and down' / 'here and there' /
'now and then' for N541 (matching the probe v9 certificates) and gives the
default-basin neurons (1888, 1594) a flat ~1x, correctly marking them as
background.

This is a SCREEN, not a result. It finds phrases *correlated* with a
destination, not the span that *causes* it. Confirm candidates with
peak-anchored minimisation before believing anything.

Reads either cache format:
  .json   list of {"sentence": ..., "neuron_id": ...}
  .jsonl  one object per line, {"s": ..., "exp_r":.., "exp_i":.., "imp_r":.., "imp_i":..}

Usage:
  # single corpus
  python screen_neurons.py ../data/the_sea_implicit_resonance.json
  python screen_neurons.py ../data/caches/_extracted/ulysses_2025-11-05/ulysses_2025-11-05.jsonl

  # a whole directory of extracted runs
  python screen_neurons.py ../data/caches/_extracted --quadrant imp_r

  # cross-corpus: keep only neurons that concentrate in >= 3 corpora
  python screen_neurons.py ../data/caches/_extracted --cross --min-corpora 3
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

WORD = re.compile(r"[a-z']+")
QUADRANTS = ("imp_r", "imp_i", "exp_r", "exp_i")


def tokens(text):
    return WORD.findall(text.lower())


def ngrams(text, sizes=(2, 3)):
    """Document-level n-gram set: presence, not count."""
    t = tokens(text)
    out = set()
    for k in sizes:
        for i in range(len(t) - k + 1):
            out.add(" ".join(t[i : i + k]))
    return out


# ---------------------------------------------------------------- loading


def load_any(path, quadrant="imp_r"):
    """Return [(sentence, neuron_id), ...] from either cache format."""
    rows = []
    if path.endswith(".jsonl"):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                sentence = obj.get("s") or obj.get("sentence")
                neuron = obj.get(quadrant)
                if neuron is None:
                    neuron = obj.get("neuron_id")
                if sentence and neuron is not None:
                    rows.append((sentence, int(neuron)))
    else:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            return []  # metadata, not a sentence cache
        for obj in data:
            if not isinstance(obj, dict):
                continue
            sentence = obj.get("sentence") or obj.get("s")
            neuron = obj.get("neuron_id", obj.get(quadrant))
            if sentence and neuron is not None:
                rows.append((sentence, int(neuron)))
    return rows


def discover(target):
    """A file, a single run folder, or a directory of runs -> [(label, path)].

    Raises FileNotFoundError with a useful message rather than silently
    returning nothing.
    """
    target = os.path.expanduser(target)

    if not os.path.exists(target):
        raise FileNotFoundError(
            f"path does not exist: {os.path.abspath(target)}\n"
            f"  (cwd is {os.getcwd()} — if you are in tools/, "
            f"you probably want ..{os.sep}data{os.sep}...)"
        )

    if os.path.isfile(target):
        return [(os.path.splitext(os.path.basename(target))[0], target)]

    # not sentence caches: 50-line previews and pipeline metadata
    IGNORE = {"sample.jsonl", "checkpoint.json", "datasets.json", "report.json"}

    def usable(path):
        return os.path.basename(path) not in IGNORE

    # a directory of run folders:  _extracted/<run>/<run>.jsonl
    nested = [p for p in sorted(glob.glob(os.path.join(target, "*", "*.jsonl")))
              if usable(p)]
    if nested:
        return [(os.path.basename(os.path.dirname(p)), p) for p in nested]

    # a single run folder, or a flat directory of caches
    flat = [p for p in sorted(glob.glob(os.path.join(target, "*.json")))
            + sorted(glob.glob(os.path.join(target, "*.jsonl")))
            if usable(p)]
    if flat:
        return [(os.path.splitext(os.path.basename(p))[0], p) for p in flat]

    raise FileNotFoundError(
        f"no .json or .jsonl caches found under {os.path.abspath(target)}\n"
        f"  expected either a cache file, a run folder containing "
        f"<name>.jsonl, or a directory of such folders"
    )


# ---------------------------------------------------------------- screen


def screen(rows, min_hits=20, min_support=3, top_phrases=5, sizes=(2, 3),
           min_background=None):
    by_neuron = collections.defaultdict(list)
    for sentence, neuron in rows:
        by_neuron[neuron].append(sentence)

    total = len(rows)
    background = collections.Counter()
    for sentence, _ in rows:
        background.update(ngrams(sentence, sizes))

    # A phrase occurring only a handful of times in the whole corpus produces
    # enormous enrichment for trivial reasons: if it appears 3 times and all 3
    # land on one neuron, enrichment is T/|H|, which grows without bound in
    # corpus size. At 7k sentences this is invisible; at 866k (WikiText-103) it
    # dominates the ranking with proper nouns. Require a minimum background
    # count, scaled to the corpus.
    if min_background is None:
        min_background = max(3 * min_support, total // 10000)

    results = []
    for neuron, sentences in by_neuron.items():
        n = len(sentences)
        if n < min_hits:
            continue

        local = collections.Counter()
        for sentence in sentences:
            local.update(ngrams(sentence, sizes))

        scored = []
        for gram, count in local.items():
            if count < min_support or background[gram] < min_background:
                continue
            enrichment = (count / n) / (background[gram] / total)
            # weight by support so a 200x phrase seen 3 times doesn't
            # outrank a 60x phrase seen 12 times
            scored.append(
                (enrichment * count, enrichment, count, background[gram], gram)
            )
        scored.sort(reverse=True)
        phrases = [(g, c, b, e) for _, e, c, b, g in scored[:top_phrases]]

        peak = phrases[0][3] if phrases else 0.0
        covered = set()
        for gram, _, _, _ in phrases:
            for i, sentence in enumerate(sentences):
                if gram in " ".join(tokens(sentence)):
                    covered.add(i)
        coverage = len(covered) / n if n else 0.0

        results.append(
            {
                "neuron": neuron,
                "hits": n,
                "peak_enrichment": peak,
                "coverage": coverage,
                "phrases": phrases,
                # rank on both: high enrichment AND broad coverage of the sample
                "score": peak * coverage,
                "min_background": min_background,
            }
        )

    results.sort(key=lambda r: -r["score"])
    # Enrichment is not comparable across corpora of different sizes, so record
    # each neuron's rank percentile within its own corpus. Cross-corpus ranking
    # uses this, never the raw score.
    m = len(results)
    for i, r in enumerate(results):
        r["percentile"] = 1.0 - (i / m) if m else 0.0
    return results


def print_table(results, limit, header=""):
    if header:
        print(f"\n{header}")
    print(f"{'neuron':>6} {'hits':>6} {'peak':>7} {'cov':>5}   top enriched phrases")
    for r in results[:limit]:
        shown = "  ".join(f"{g!r}({c},{e:.0f}x)" for g, c, _b, e in r["phrases"][:4])
        print(
            f"{r['neuron']:6d} {r['hits']:6d} {r['peak_enrichment']:6.0f}x "
            f"{r['coverage']:5.2f}   {shown}"
        )


# ---------------------------------------------------------------- main



def _ensure(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def write_single_tsv(out_dir, args, per_corpus):
    """One file per corpus, named for it, so successive runs don't collide."""
    out_dir = _ensure(out_dir)
    written = []
    for label, results in per_corpus.items():
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", label)
        path = os.path.join(out_dir, f"screen_{args.quadrant}_{safe}.tsv")
        with open(path, "w", encoding="utf-8", newline="") as handle:
            handle.write(
                "corpus\tneuron\thits\tpeak_enrichment\tcoverage\tscore\tpercentile"
                "\tphrase\thits_with_phrase\tcorpus_count\tenrichment\n"
            )
            for r in results:
                for gram, count, bgcount, enrichment in r["phrases"]:
                    handle.write(
                        f"{label}\t{r['neuron']}\t{r['hits']}\t"
                        f"{r['peak_enrichment']:.2f}\t{r['coverage']:.4f}\t"
                        f"{r['score']:.2f}\t{r['percentile']:.4f}\t"
                        f"{gram}\t{count}\t{bgcount}\t{enrichment:.2f}\n"
                    )
        written.append(path)
    print()
    for path in written:
        print(f"wrote {os.path.abspath(path)}")


def write_cross_tsv(out_dir, args, per_corpus, survivors, med_pct):
    out_dir = _ensure(out_dir)

    summary = os.path.join(out_dir, f"screen_{args.quadrant}_cross_summary.tsv")
    with open(summary, "w", encoding="utf-8", newline="") as handle:
        handle.write("neuron\tn_corpora\tmedian_percentile\tcorpora\ttop_phrases\n")
        for neuron, v in survivors:
            labels = ",".join(sorted(lab.split("_")[0] for lab, _ in v))
            grams = collections.Counter()
            for _lab, r in v:
                for gram, _c, _b, _e in r["phrases"][:3]:
                    grams[gram] += 1
            top = ",".join(g for g, _ in grams.most_common(5))
            handle.write(f"{neuron}\t{len(v)}\t{med_pct(v):.4f}\t{labels}\t{top}\n")

    detail = os.path.join(out_dir, f"screen_{args.quadrant}_cross_detail.tsv")
    with open(detail, "w", encoding="utf-8", newline="") as handle:
        handle.write(
            "neuron\tcorpus\thits\tcoverage\tpercentile\tphrase"
            "\thits_with_phrase\tcorpus_count\tenrichment\n"
        )
        for neuron, v in survivors:
            for label, r in sorted(v, key=lambda t: -t[1]["percentile"]):
                for gram, count, bgcount, enrichment in r["phrases"]:
                    handle.write(
                        f"{neuron}\t{label}\t{r['hits']}\t{r['coverage']:.4f}\t"
                        f"{r['percentile']:.4f}\t{gram}\t{count}\t{bgcount}\t"
                        f"{enrichment:.2f}\n"
                    )

    print(f"\nwrote {os.path.abspath(summary)}")
    print(f"wrote {os.path.abspath(detail)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="cache file, or directory of _extracted runs")
    ap.add_argument("--quadrant", default="imp_r", choices=QUADRANTS,
                    help="which destination column to read from .jsonl (default imp_r)")
    ap.add_argument("--min-hits", type=int, default=20)
    ap.add_argument("--min-support", type=int, default=3)
    ap.add_argument("--top", type=int, default=30, help="rows to print")
    ap.add_argument("--cross", action="store_true",
                    help="screen each corpus separately and report neurons surviving several")
    ap.add_argument("--min-corpora", type=int, default=3,
                    help="with --cross, how many corpora a neuron must concentrate in")
    ap.add_argument("--out-dir", default="results",
                    help="directory for TSV output (default ./results)")
    ap.add_argument("--min-background", type=int, default=None,
                    help="minimum corpus occurrences for a phrase to count "
                         "(default: max(3*min_support, sentences/10000))")
    args = ap.parse_args()

    try:
        corpora = discover(args.target)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out_dir = args.out_dir
    per_corpus = {}
    for label, path in corpora:
        rows = load_any(path, args.quadrant)
        if len(rows) < args.min_hits * 2:
            print(f"  skipping {label}: only {len(rows)} sentences")
            continue
        results = screen(rows, args.min_hits, args.min_support,
                         min_background=args.min_background)
        per_corpus[label] = results
        floor = results[0]["min_background"] if results else "-"
        print(f"  {label}: {len(rows)} sentences, "
              f"{len(results)} neurons with >= {args.min_hits} hits "
              f"(phrase background floor {floor})")

    if not args.cross:
        label, results = next(iter(per_corpus.items()))
        print_table(results, args.top, f"[{args.quadrant}] {label}")
    else:
        appearances = collections.defaultdict(list)
        for label, results in per_corpus.items():
            for r in results:
                appearances[r["neuron"]].append((label, r))

        survivors = [(n, v) for n, v in appearances.items()
                     if len(v) >= args.min_corpora]
        # Rank on median within-corpus percentile. Raw enrichment is not
        # comparable across corpora of different sizes -- the same phrase on
        # the same neuron scores 67x in a 7k corpus and 508x in a 129k one.
        def med_pct(v):
            s = sorted(r["percentile"] for _, r in v)
            return s[len(s) // 2]
        survivors.sort(key=lambda t: -med_pct(t[1]))

        print(f"\n{len(survivors)} neurons concentrate in >= {args.min_corpora} "
              f"of {len(per_corpus)} corpora [{args.quadrant}]\n")
        print(f"{'neuron':>6} {'corpora':>7} {'medpct':>7}   phrases per corpus")
        for neuron, v in survivors[: args.top]:
            best = "; ".join(
                f"{lab.split('_')[0]}: " + ", ".join(g for g, _c, _b, _e in r["phrases"][:2])
                for lab, r in sorted(v, key=lambda t: -t[1]["percentile"])[:3]
            )
            print(f"{neuron:6d} {len(v):7d} {med_pct(v):7.2f}   {best}")

        write_cross_tsv(out_dir, args, per_corpus, survivors, med_pct)

    if not args.cross:
        write_single_tsv(out_dir, args, per_corpus)

    return 0


if __name__ == "__main__":
    sys.exit(main())
