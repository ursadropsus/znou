"""
probe_v5.py — which perturbation axis actually finds new destinations?

Four strategies, equal budget, same seed certificate. The question is
distinct destinations per string evaluated, which is the only quantity a
coverage generator should be optimised for.

  A CARRIER    vary the prefix, hold the gate      'X Mackinaw'
  B GATE       hold the prefix, vary the word      'at X'
  C LOCAL      random single-token substitution anywhere in the seed
  D RANDOM     random token sequences, length-matched (the ascent's regime)

D is the control: whatever a directed strategy is worth, it has to beat
sampling. A and B are the two halves of §2's taxonomy. C is the Q5
neighbourhood, included because it is the axis the spec cares about
independently of coverage.

Canonicality is checked throughout: every candidate is decoded to a string
and re-encoded, and rows where that round trip changes the token sequence
are flagged rather than dropped. dom(D) is strings (§4.5.2).

Writes results/probe_v5_yield_*.tsv
"""

import random

import znou_probe as zp

SEED_STRING = "at Mackinaw"
BUDGET = 60
RNG_SEED = 0

WATCH = [20, 1430, 1888, 2256]

CARRIERS = """at the on near was from of to in by for with into upon beyond
island fort lake north old great we he they past under aboard left toward
beside sailed whale ship Michigan Boston Iceland north south east west
after before through across around behind between during without within
along among against about above below since until while where when
""".split()

WORDS = """Mackinaw Nantucket Iceland Greenland Labrador Patagonia Zanzibar
harbour anchor rigging fathom leeward starboard binnacle scrimshaw
whale porpoise kraken albatross petrel cormorant
tavern chowder ledger pulpit sermon parson
compass quadrant sextant chronometer lantern
robin cabin napkin coffin muffin goblin basin margin
Franklin Dublin Berlin Austin Merlin Latin
thunder marrow gospel iron canvas tallow
""".split()


def canonical(s):
    """True if s survives encode -> decode -> encode unchanged."""
    ids = zp.tok(s)["input_ids"]
    return zp.tok(zp.tok.decode(ids))["input_ids"] == ids


def safe_token_pool(rng, n=4000):
    """Sample token ids that round-trip cleanly on their own."""
    pool = []
    while len(pool) < n:
        i = rng.randrange(50257)
        d = zp.tok.decode([i])
        if not d or not d.strip():
            continue
        if zp.tok(d)["input_ids"] == [i]:
            pool.append(i)
    return pool


def evaluate(rec, arm, strings):
    """Measure each string once, dedup, record, return the rows."""
    seen, rows = set(), []
    for s in strings:
        if s in seen or not s.strip():
            continue
        seen.add(s)
        r = zp.measure(s, watch=WATCH, arm=arm, canonical=int(canonical(s)))
        rec.add(r)
        rows.append(r)
    return rows


def summarise(arm, rows, known):
    dests = [r["dest"] for r in rows]
    distinct = sorted(set(dests))
    novel = sorted(set(dests) - known)
    nulls = sum(d in (2256, 1888) for d in dests)
    deltas = sorted(r["delta"] for r in rows)
    med = deltas[len(deltas) // 2] if deltas else float("nan")
    noncanon = sum(1 - r["canonical"] for r in rows)
    print(f"  {arm:<9}{len(rows):>6}{len(distinct):>10}{len(novel):>8}"
          f"{nulls:>8}{med:>10.4f}{noncanon:>11}")
    return {"arm": arm, "n": len(rows), "distinct": len(distinct),
            "novel": len(novel), "null_2256_1888": nulls,
            "median_delta": med, "non_canonical": noncanon,
            "dests": " ".join(str(d) for d in distinct)}


def main():
    rng = random.Random(RNG_SEED)
    rec = zp.Recorder("probe_v5_yield")

    seed_row = zp.measure(SEED_STRING, watch=WATCH, arm="seed", canonical=1)
    rec.add(seed_row)
    seed_ids = zp.tok(SEED_STRING)["input_ids"]
    print(f"seed {SEED_STRING!r} -> {seed_row['dest']} "
          f"delta {seed_row['delta']:.4f}  {len(seed_ids)} tokens\n")

    # A. carrier: substitute position 0
    a = [f"{c} Mackinaw" for c in CARRIERS[:BUDGET]]

    # B. gate: substitute the content word
    b = [f"at {w}" for w in WORDS[:BUDGET]]

    # C. local: one random token substituted at a random position
    pool = safe_token_pool(rng)
    c = []
    while len(c) < BUDGET:
        ids = list(seed_ids)
        ids[rng.randrange(len(ids))] = rng.choice(pool)
        c.append(zp.tok.decode(ids))

    # D. random: length-matched random token sequences
    d = [zp.tok.decode([rng.choice(pool) for _ in seed_ids])
         for _ in range(BUDGET)]

    print(f"  {'arm':<9}{'n':>6}{'distinct':>10}{'novel':>8}"
          f"{'null':>8}{'med Δ':>10}{'noncanon':>11}")

    known = set()
    summaries = []
    for arm, strings in (("A_carrier", a), ("B_gate", b),
                         ("C_local", c), ("D_random", d)):
        rows = evaluate(rec, arm, strings)
        s = summarise(arm, rows, known)
        summaries.append(s)
        known |= {r["dest"] for r in rows}

    print(f"\n  union across all arms: {len(known)} distinct destinations")
    print("  'novel' is relative to arms already run, so order matters;")
    print("  'distinct' is the comparable column.\n")

    srec = zp.Recorder("probe_v5_summary")
    for s in summaries:
        srec.add(s)
    rec.write()
    srec.write()


if __name__ == "__main__":
    main()
