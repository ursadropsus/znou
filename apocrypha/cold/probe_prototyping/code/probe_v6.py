"""
probe_v6.py — retention and yield around several known routes.

V6 keeps all experiments in one versioned script, with explicit branches:

  A_CARRIER       vary material before the gate
  B_GATE          hold the carrier; substitute the gate word
  C_LOCAL         one realised-token substitution
  D_RANDOM_TOKEN  random realised strings, token-length matched
  E_CONTINUATION  hold the candidate gate; vary only what follows

The primary outputs are deliberately plural:

  retention  how often the seed destination survives
  yield      how many distinct destinations the branch visits

The candidate pipeline records intended and realised token sequences. A
string can be round-trip stable without still being the token intervention
the generator intended; v5 conflated those properties.

Default run: 3 seed families × (3 deterministic branches + 2 stochastic
branches × 3 RNG seeds) × 60 unique realised strings = 1,620 candidates,
plus the three seeds. Adjust --budget and --rng-seeds for smoke/full runs.

Requires znou_probe.py beside this file. Writes timestamped TSVs through
znou_probe.Recorder:

  probe_v6_yield_*.tsv       one row per measured string
  probe_v6_summary_*.tsv     one row per seed/branch/RNG block
  probe_v6_curve_*.tsv       cumulative discovery at shared budgets
  probe_v6_overlap_*.tsv     symmetric destination overlap

Examples
--------
  python probe_v6.py --budget 10 --rng-seeds 0
  python probe_v6.py --budget 100 --rng-seeds 0,1,2,3,4
  python probe_v6.py --reference reached_imp_r.txt
"""

import argparse
import json
import math
import random
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import znou_probe as zp


# ---------------------------------------------------------------------------
# Configuration

WATCH = [20, 38, 281, 464, 1430, 1790, 1888, 2094, 2256]


@dataclass(frozen=True)
class SeedSpec:
    name: str
    string: str
    expected_dest: int | None
    carrier_tail: str
    gate_carrier: str
    continuation_stem: str


SEEDS = [
    SeedSpec(
        name="mackinaw_20",
        string="at Mackinaw",
        expected_dest=20,
        carrier_tail=" Mackinaw",
        gate_carrier="at",
        continuation_stem="at Mackin",
    ),
    SeedSpec(
        name="robin_281",
        string="at robin",
        expected_dest=281,
        carrier_tail=" robin",
        gate_carrier="at",
        continuation_stem="at rob",
    ),
    SeedSpec(
        name="cold_38",
        string="it was cold",
        expected_dest=38,
        carrier_tail=" cold",
        gate_carrier="it was",
        continuation_stem="it was cold",
    ),
]

# Add live discoveries here. expected_dest may be None: v6 will use the
# measured seed destination. Supply the positional decomposition explicitly.
EXTRA_SEEDS: list[SeedSpec] = []


# Single-token-like carriers and explicit multiword cold-family carriers are
# combined below. The strings are candidates, not linguistic categories.
_SINGLE_CARRIERS = """at the on near was from of to in by for with into upon
beyond island fort lake north old great we he she they I you past under aboard
left toward beside sailed whale ship Michigan Boston Iceland northward
southward eastward westward after before through across around behind between
during without within along among against about above below since until while
where when here there now then once still almost never always already perhaps
maybe quietly suddenly merely partly barely deeply extremely strangely
apparently formerly eventually generally locally remotely today yesterday
tomorrow someone nobody everybody winter midnight morning evening offshore
upstairs outside inside elsewhere home away ahead astern
""".split()

_PHRASE_CARRIERS = [
    "it was", "it is", "it felt", "it seemed", "this was", "that was",
    "everything was", "nothing was", "the air was", "the water was",
    "the night was", "the room was", "the world was", "the sea was",
    "the wind was", "the ground was", "the ship was", "the morning was",
    "the evening was", "the weather was", "I was", "he was", "she was",
    "we were", "they were", "you were", "there was", "outside it was",
    "inside it was", "yesterday was", "today was", "winter was",
]

CARRIER_POOL = list(dict.fromkeys(_PHRASE_CARRIERS + _SINGLE_CARRIERS))

GATE_WORDS = """Mackinaw Nantucket Iceland Greenland Labrador Patagonia
Zanzibar harbour anchor rigging fathom leeward starboard binnacle scrimshaw
whale porpoise kraken albatross petrel cormorant tavern chowder ledger pulpit
sermon parson compass quadrant sextant chronometer lantern robin cabin napkin
coffin muffin goblin basin margin Franklin Dublin Berlin Austin Merlin Latin
thunder marrow gospel iron canvas tallow cold colder coldest cool cooler coolest
frigid freezing frozen icy warm warmer warmest hot hotter hottest mild bitter
bleak crisp chill chilly wintry arctic polar glacial numb dark darker darkest
light bright silent quiet loud empty full near nearer far distant lost hidden
open closed broken whole strange ordinary ancient modern northern southern
eastern western ocean forest mountain desert island station signal passage
door threshold chamber vessel engine orbit system neuron token string
""".split()

SUFFIXES = [
    "", ".", ",", "!", "?", ";", ":", "...", "er", "est", "s", "ed",
    "ing", "ly", "ness", "ish", "ward", "wards", " water", " air",
    " weather", " night", " morning", " again", " now", " then", " today",
    " yesterday", " outside", " inside", " here", " there", " nearby",
    " below", " above", " ahead", " behind", " at sea", " in winter",
    " for hours", " all day", " all night", " and dark", " and still",
    " and silent", " and empty", " and clear", " and bright", " and warm",
    " but calm", " but changing", " before dawn", " after sunset",
    " near shore", " offshore", " aboard", " within", " without",
    " beneath us", " around us", " beyond them", " as ever", " as stone",
    " as ice", " as iron", " as glass", " to touch", " to me", " to him",
    " to her", " to them", " enough", " indeed", " perhaps", " somehow",
    " suddenly", " gradually", " slightly", " deeply", " extremely",
    " almost", " barely", " merely", " unusually", " impossibly",
    " when we arrived", " when it began", " when the wind rose",
    " where we stood", " where the road ended", " under the stars",
    " across the lake", " beside the fire", " beyond the ridge",
    " until morning", " since yesterday", " throughout", " forever",
    " no longer", " once more", " at last", " in the distance",
    " on the surface", " underfoot", " around midnight", " by morning",
]


# ---------------------------------------------------------------------------
# Token and candidate utilities

def ids_for(s):
    return list(zp.tok(s)["input_ids"])


def decode(ids):
    return zp.tok.decode(list(ids))


def stable_ids(s):
    """Return realised IDs and whether another round trip preserves them."""
    realised = ids_for(s)
    stable = ids_for(decode(realised)) == realised
    return realised, stable


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1,
                           prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else None


def complete_safe_token_pool():
    """All nonblank token IDs that round-trip cleanly in isolation."""
    pool = []
    for token_id in range(50257):
        s = decode([token_id])
        if s and s.strip() and ids_for(s) == [token_id]:
            pool.append(token_id)
    return pool


def realised_candidate(source, intended_ids, *, require_len=None,
                       require_hamming=None):
    """Decode intended IDs and describe the actual string-domain candidate."""
    s = decode(intended_ids)
    realised, stable = stable_ids(s)
    source_ids = ids_for(source)
    agrees = stable
    if require_len is not None:
        agrees = agrees and len(realised) == require_len
    if require_hamming is not None:
        agrees = agrees and hamming(source_ids, realised) == require_hamming
    return {
        "string": s,
        "source_string": source,
        "source_token_ids": json.dumps(source_ids),
        "intended_token_ids": json.dumps(list(intended_ids)),
        "realised_token_ids": json.dumps(realised),
        "intended_n_tok": len(intended_ids),
        "realised_n_tok": len(realised),
        "intended_edit_distance": levenshtein(source_ids, intended_ids),
        "realised_edit_distance": levenshtein(source_ids, realised),
        "roundtrip_stable": int(stable),
        "intervention_agrees": int(agrees),
        "edit_position": "NA",
    }


def string_candidate(source, s):
    realised, stable = stable_ids(s)
    source_ids = ids_for(source)
    return {
        "string": s,
        "source_string": source,
        "source_token_ids": json.dumps(source_ids),
        "intended_token_ids": json.dumps(realised),
        "realised_token_ids": json.dumps(realised),
        "intended_n_tok": len(realised),
        "realised_n_tok": len(realised),
        "intended_edit_distance": levenshtein(source_ids, realised),
        "realised_edit_distance": levenshtein(source_ids, realised),
        "roundtrip_stable": int(stable),
        "intervention_agrees": int(stable),
        "edit_position": "NA",
    }


def take_unique(candidates, budget, *, label):
    out, seen = [], set()
    for candidate in candidates:
        s = candidate["string"]
        if not s.strip() or s in seen:
            continue
        seen.add(s)
        out.append(candidate)
        if len(out) == budget:
            return out
    raise RuntimeError(
        f"{label}: generated {len(out)} unique strings; budget is {budget}"
    )


def generate_exact_local(seed, budget, rng, pool):
    source_ids = ids_for(seed.string)
    out, seen = [], set()
    attempts = 0
    max_attempts = budget * 500
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        intended = list(source_ids)
        pos = rng.randrange(len(intended))
        replacement = rng.choice(pool)
        if replacement == intended[pos]:
            continue
        intended[pos] = replacement
        c = realised_candidate(
            seed.string, intended,
            require_len=len(source_ids), require_hamming=1,
        )
        if not c["intervention_agrees"] or c["string"] in seen:
            continue
        seen.add(c["string"])
        c["edit_position"] = pos
        out.append(c)
    if len(out) < budget:
        raise RuntimeError(
            f"{seed.name}/C_LOCAL: only {len(out)} valid candidates after "
            f"{attempts} attempts"
        )
    return out


def generate_exact_random(seed, budget, rng, pool):
    n_tok = len(ids_for(seed.string))
    out, seen = [], set()
    attempts = 0
    max_attempts = budget * 1000
    while len(out) < budget and attempts < max_attempts:
        attempts += 1
        intended = [rng.choice(pool) for _ in range(n_tok)]
        c = realised_candidate(seed.string, intended, require_len=n_tok)
        if not c["intervention_agrees"] or c["string"] in seen:
            continue
        seen.add(c["string"])
        out.append(c)
    if len(out) < budget:
        raise RuntimeError(
            f"{seed.name}/D_RANDOM_TOKEN: only {len(out)} valid candidates "
            f"after {attempts} attempts"
        )
    return out


def deterministic_candidates(seed, branch, budget):
    if branch == "A_CARRIER":
        strings = (carrier + seed.carrier_tail for carrier in CARRIER_POOL)
    elif branch == "B_GATE":
        strings = (seed.gate_carrier + " " + word for word in GATE_WORDS)
    elif branch == "E_CONTINUATION":
        strings = (seed.continuation_stem + suffix for suffix in SUFFIXES)
    else:
        raise ValueError(branch)
    candidates = (string_candidate(seed.string, s) for s in strings)
    return take_unique(candidates, budget, label=f"{seed.name}/{branch}")


# ---------------------------------------------------------------------------
# Measurement and summaries

def evaluate_block(rec, seed, target, branch, rng_seed, candidates):
    rows = []
    watch = list(dict.fromkeys(WATCH + [target]))
    for evaluation_index, candidate in enumerate(candidates, 1):
        fields = {k: v for k, v in candidate.items() if k != "string"}
        row = zp.measure(
            candidate["string"], watch=watch,
            seed_name=seed.name,
            seed_string=seed.string,
            target_dest=target,
            branch=branch,
            rng_seed=rng_seed,
            evaluation_index=evaluation_index,
            **fields,
        )
        rec.add(row)
        rows.append(row)
    return rows


def entropy(counter):
    n = sum(counter.values())
    return -sum((k / n) * math.log2(k / n) for k in counter.values())


def quantile(values, q):
    """Linear interpolation, matching the common type-7 sample quantile."""
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    p = (len(xs) - 1) * q
    lo, hi = math.floor(p), math.ceil(p)
    return xs[lo] + (p - lo) * (xs[hi] - xs[lo])


def summarise(seed, target, branch, rng_seed, rows, reference,
              reference_id="NA"):
    dests = [int(r["dest"]) for r in rows]
    counts = Counter(dests)
    deltas = [float(r["delta"]) for r in rows]
    dominant_dest, dominant_n = counts.most_common(1)[0]
    distinct = set(dests)
    novel = distinct - reference if reference is not None else set()
    return {
        "seed_name": seed.name,
        "seed_string": seed.string,
        "target_dest": target,
        "branch": branch,
        "rng_seed": rng_seed,
        "n_requested": len(rows),
        "n_evaluated": len(rows),
        "n_unique_strings": len({r["string"] for r in rows}),
        "target_retained": sum(d == target for d in dests),
        "retention_rate": sum(d == target for d in dests) / len(dests),
        "distinct_destinations": len(distinct),
        "distinct_per_evaluation": len(distinct) / len(dests),
        "destination_entropy_bits": entropy(counts),
        "dominant_dest": dominant_dest,
        "dominant_n": dominant_n,
        "dominant_share": dominant_n / len(dests),
        "median_delta": statistics.median(deltas),
        "delta_q25": quantile(deltas, 0.25),
        "delta_q75": quantile(deltas, 0.75),
        "destination_counts": json.dumps(dict(sorted(counts.items()))),
        "destinations": " ".join(str(d) for d in sorted(distinct)),
        "novel_against_reference": len(novel) if reference is not None else "NA",
        "reference_id": reference_id if reference is not None else "NA",
        "novel_destinations": (
            " ".join(str(d) for d in sorted(novel))
            if reference is not None else "NA"
        ),
    }


def add_curve_rows(rec, seed, target, branch, rng_seed, rows):
    seen = set()
    retained = 0
    for i, row in enumerate(rows, 1):
        dest = int(row["dest"])
        seen.add(dest)
        retained += dest == target
        rec.add({
            "seed_name": seed.name,
            "target_dest": target,
            "branch": branch,
            "rng_seed": rng_seed,
            "n_evaluated": i,
            "distinct_destinations": len(seen),
            "target_retained": retained,
            "retention_rate": retained / i,
        })


def add_overlap_rows(rec, seed, blocks):
    labels = sorted(blocks)
    for i, left in enumerate(labels):
        dl = {int(r["dest"]) for r in blocks[left]}
        for right in labels[i + 1:]:
            dr = {int(r["dest"]) for r in blocks[right]}
            inter = dl & dr
            union = dl | dr
            rec.add({
                "seed_name": seed.name,
                "left_block": left,
                "right_block": right,
                "left_distinct": len(dl),
                "right_distinct": len(dr),
                "intersection_n": len(inter),
                "union_n": len(union),
                "jaccard": len(inter) / len(union),
                "intersection": " ".join(str(d) for d in sorted(inter)),
            })


def load_reference(path):
    if path is None:
        return None
    values = set()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        values.add(int(line.split()[0]))
    return values


def parse_rng_seeds(text):
    seeds = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not seeds:
        raise argparse.ArgumentTypeError("need at least one RNG seed")
    return seeds


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget", type=int, default=60)
    parser.add_argument("--rng-seeds", type=parse_rng_seeds, default=[0, 1, 2])
    parser.add_argument(
        "--reference",
        help="frozen known-destination file, one integer per line",
    )
    parser.add_argument(
        "--seed",
        action="append",
        dest="seed_names",
        help="run only this seed name; repeat for several",
    )
    args = parser.parse_args()
    if args.budget < 1:
        parser.error("--budget must be positive")

    seeds = SEEDS + EXTRA_SEEDS
    if args.seed_names:
        wanted = set(args.seed_names)
        seeds = [s for s in seeds if s.name in wanted]
        missing = wanted - {s.name for s in seeds}
        if missing:
            parser.error(f"unknown seed name(s): {', '.join(sorted(missing))}")

    reference = load_reference(args.reference)
    print("building complete safe-token pool...")
    safe_pool = complete_safe_token_pool()
    print(f"safe tokens: {len(safe_pool)}/50257")

    yield_rec = zp.Recorder("probe_v6_yield")
    summary_rec = zp.Recorder("probe_v6_summary")
    curve_rec = zp.Recorder("probe_v6_curve")
    overlap_rec = zp.Recorder("probe_v6_overlap")

    for seed in seeds:
        seed_ids = ids_for(seed.string)
        seed_row = zp.measure(
            seed.string, watch=WATCH, seed_name=seed.name,
            seed_string=seed.string,
            target_dest=seed.expected_dest if seed.expected_dest is not None else "NA",
            branch="SEED", rng_seed="NA", evaluation_index=0,
            source_string=seed.string,
            source_token_ids=json.dumps(seed_ids),
            intended_token_ids=json.dumps(seed_ids),
            realised_token_ids=json.dumps(seed_ids),
            intended_n_tok=len(seed_ids), realised_n_tok=len(seed_ids),
            intended_edit_distance=0, realised_edit_distance=0,
            roundtrip_stable=1, intervention_agrees=1,
            edit_position="NA",
        )
        yield_rec.add(seed_row)
        target = int(seed_row["dest"])
        if seed.expected_dest is not None and target != seed.expected_dest:
            raise RuntimeError(
                f"{seed.name}: expected {seed.expected_dest}, measured {target}"
            )
        print(f"\n{seed.name}: {seed.string!r} -> {target} "
              f"delta {float(seed_row['delta']):.6f}")

        blocks = {}

        # Deterministic branches run once. RNG seed is NA by construction.
        for branch in ("A_CARRIER", "B_GATE", "E_CONTINUATION"):
            candidates = deterministic_candidates(seed, branch, args.budget)
            rows = evaluate_block(
                yield_rec, seed, target, branch, "NA", candidates,
            )
            block_name = branch
            blocks[block_name] = rows
            summary = summarise(
                seed, target, branch, "NA", rows, reference,
                args.reference or "NA",
            )
            summary_rec.add(summary)
            add_curve_rows(curve_rec, seed, target, branch, "NA", rows)
            print(f"  {branch:<16} retain {summary['retention_rate']:.1%}  "
                  f"distinct {summary['distinct_destinations']:>3}/"
                  f"{summary['n_evaluated']}")

        # Stochastic branches are independent blocks for every declared seed.
        for rng_seed in args.rng_seeds:
            # Independent streams: changing C's budget or rejection rate must
            # not silently change D's control candidates.
            local_rng = random.Random(rng_seed * 2 + 1)
            random_rng = random.Random(rng_seed * 2 + 2)
            local = generate_exact_local(
                seed, args.budget, local_rng, safe_pool,
            )
            random_rows = generate_exact_random(
                seed, args.budget, random_rng, safe_pool,
            )
            for branch, candidates in (
                ("C_LOCAL", local), ("D_RANDOM_TOKEN", random_rows)
            ):
                rows = evaluate_block(
                    yield_rec, seed, target, branch, rng_seed, candidates,
                )
                block_name = f"{branch}:rng={rng_seed}"
                blocks[block_name] = rows
                summary = summarise(
                    seed, target, branch, rng_seed, rows, reference,
                    args.reference or "NA",
                )
                summary_rec.add(summary)
                add_curve_rows(
                    curve_rec, seed, target, branch, rng_seed, rows,
                )
                print(f"  {block_name:<16} retain "
                      f"{summary['retention_rate']:.1%}  distinct "
                      f"{summary['distinct_destinations']:>3}/"
                      f"{summary['n_evaluated']}")

        add_overlap_rows(overlap_rec, seed, blocks)

    yield_rec.write()
    summary_rec.write()
    curve_rec.write()
    overlap_rec.write()


if __name__ == "__main__":
    main()
