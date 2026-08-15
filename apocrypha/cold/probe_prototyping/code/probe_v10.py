"""
probe_v10.py — controlled discrimination panel for layer-5 neuron 541.

V9 found 80 Melville routes whose shortest certificates consistently ended in
paired constructions.  V10 asks what part of that pattern matters.  It keeps
feature response separate from map victory:

  raw_completion_541   raw neuron-541 activation on the final token of member 2
  peak_541             maximum neuron-541 activation anywhere in the string
  signed_541_margin    peak_541 minus the strongest non-541 peak
  dest / retained      argmax destination and the thresholded route outcome

The centrepiece is a multi-seed 2x2 member factorial.  Each established pair
is crossed with tokenizer-matched arbitrary first and second members, then run
in four declared position/carrier regimes.  Endpoint-sufficiency controls,
connector substitutions, held-out conventional pairs, arbitrary pairs,
frame-breaking variants, and controlled continuations are separate branches.

Raw per-position traces are written for every case.  Running peaks are retained
for operational interpretation but are not treated as independent evidence of
completion sensitivity: under Resonance they are monotone by construction.

Requires the CUDA-corrected znou_probe.py beside this file.

Examples
--------
  python probe_v10.py --smoke
  python probe_v10.py
  python probe_v10.py --batch-size 32 --outdir results
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import torch

import znou_probe as zp


TARGET = 541
CONNECTORS = ("and", "or", "to", "by", "after")

# Six v9-supported endpoint families.  The canonical connector is part of the
# seed definition, not inferred from v10 results.
SEEDS = (
    ("UP_DOWN", "up", "and", "down"),
    ("NOW_THEN", "now", "and", "then"),
    ("HERE_THERE", "here", "and", "there"),
    ("RIGHT_LEFT", "right", "and", "left"),
    ("TRUE_FALSE", "true", "or", "false"),
    ("EAST_WEST", "east", "and", "west"),
)

# Selection from this pool depends only on GPT-2 realised-token length, never
# activation.  Ordering is fixed and recorded in PREFLIGHT.
ARBITRARY_POOL = (
    "gravel", "lantern", "velvet", "engine", "marble", "copper",
    "window", "garden", "rabbit", "ocean", "candle", "button",
    "forest", "pocket", "hammer", "violin", "mustard", "planet",
    "basket", "chimney", "teacup", "ribbon", "anchor", "pencil",
)

HELDOUT_CONVENTIONAL = (
    ("SALT_PEPPER", "salt", "and", "pepper"),
    ("KNIFE_FORK", "knife", "and", "fork"),
    ("BACK_FORTH", "back", "and", "forth"),
    ("BLACK_WHITE", "black", "and", "white"),
    ("PROS_CONS", "pros", "and", "cons"),
    ("SOONER_LATER", "sooner", "or", "later"),
    ("GIVE_TAKE", "give", "and", "take"),
    ("LIFE_DEATH", "life", "and", "death"),
)

ARBITRARY_PAIRS = (
    ("CATS_ELEPHANTS", "cats", "and", "elephants"),
    ("BREAD_GRAVEL", "bread", "and", "gravel"),
    ("LANTERNS_ARITHMETIC", "lanterns", "and", "arithmetic"),
    ("VELVET_ENGINES", "velvet", "or", "engines"),
    ("CLOUDS_TEASPOONS", "clouds", "and", "teaspoons"),
    ("NORTH_VELVET", "north", "or", "velvet"),
    ("MARBLE_RABBITS", "marble", "and", "rabbits"),
    ("BUTTON_OCEAN", "button", "and", "ocean"),
)

# BARE tests ordinary sentence-initial tokenisation.  V9_SPACE reproduces the
# common leading-space certificate regime without moving the first token away
# from position 1.  The two fixed carriers move the construction later.
CARRIERS = (
    ("BARE", "", ""),
    ("V9_SPACE", "", " "),
    ("THE_PHRASE", "the phrase ", ""),
    ("THEY_SAID", "they said ", ""),
)

CONTINUATIONS = (
    ("NONE", ""),
    ("AGAIN", " again"),
    ("PERIOD", "."),
    ("ROAD", " the road"),
    ("CLAUSE", ", without stopping"),
    ("XYZ", " xyz"),
)


@dataclass
class Case:
    case_id: str
    branch: str
    seed_id: str
    variant: str
    carrier_id: str
    carrier: str
    leading_prefix: str
    first: str
    connector: str
    second: str
    construction: str
    continuation_id: str = "NONE"
    continuation: str = ""
    baseline_variant: str = ""
    notes: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def text(self):
        return self.carrier + self.leading_prefix + self.construction + self.continuation

    @property
    def construction_start(self):
        return len(self.carrier) + len(self.leading_prefix)

    @property
    def second_start(self):
        if not self.second:
            return self.construction_start
        local = self.construction.rfind(self.second)
        if local < 0:
            raise ValueError(f"{self.case_id}: second member absent from construction")
        return self.construction_start + local

    @property
    def second_end(self):
        return self.second_start + len(self.second)


def slug(*parts):
    return "__".join(
        str(p).upper().replace(" ", "_").replace("-", "_") for p in parts
    )


def construction(first, connector, second):
    return " ".join(x for x in (first, connector, second) if x)


def token_ids(text):
    return list(zp.tok(text)["input_ids"])


def member_token_count(word):
    """Token count for a member in its normal space-prefixed regime."""
    return len(token_ids(" " + word))


def choose_arbitrary(word, forbidden, offset=0):
    wanted = member_token_count(word)
    candidates = [
        x for x in ARBITRARY_POOL
        if x not in forbidden and member_token_count(x) == wanted
    ]
    if not candidates:
        raise RuntimeError(f"no arbitrary member matches token length of {word!r}")
    return candidates[offset % len(candidates)]


def make_case(branch, seed_id, variant, carrier, first, connector, second,
              *, continuation=("NONE", ""), baseline_variant="CC", notes="",
              construction_text=None, extra=None):
    carrier_id, carrier_text, leading_prefix = carrier
    text = construction_text
    if text is None:
        text = construction(first, connector, second)
    continuation_id, continuation_text = continuation
    return Case(
        case_id=slug(branch, seed_id, variant, carrier_id, continuation_id),
        branch=branch,
        seed_id=seed_id,
        variant=variant,
        carrier_id=carrier_id,
        carrier=carrier_text,
        leading_prefix=leading_prefix,
        first=first,
        connector=connector,
        second=second,
        construction=text,
        continuation_id=continuation_id,
        continuation=continuation_text,
        baseline_variant=baseline_variant,
        notes=notes,
        extra=extra or {},
    )


def build_cases(smoke=False):
    cases = []
    preflight = []
    seeds = SEEDS[:2] if smoke else SEEDS
    carriers = CARRIERS[:2] if smoke else CARRIERS

    replacements = {}
    used = set()
    for seed_index, (seed_id, first, connector, second) in enumerate(seeds):
        arbitrary_first = choose_arbitrary(first, used | {first, second}, seed_index)
        used.add(arbitrary_first)
        arbitrary_second = choose_arbitrary(
            second, used | {first, second, arbitrary_first}, seed_index + 5
        )
        used.add(arbitrary_second)
        replacements[seed_id] = (arbitrary_first, arbitrary_second)
        preflight.append({
            "seed_id": seed_id,
            "canonical_first": first,
            "canonical_connector": connector,
            "canonical_second": second,
            "arbitrary_first": arbitrary_first,
            "arbitrary_second": arbitrary_second,
            "canonical_first_space_tokens": member_token_count(first),
            "arbitrary_first_space_tokens": member_token_count(arbitrary_first),
            "canonical_second_space_tokens": member_token_count(second),
            "arbitrary_second_space_tokens": member_token_count(arbitrary_second),
            "selection_rule": "first predeclared pool member matching space-token count",
        })

        factorial = (
            ("CC", first, second, "canonical first × canonical second"),
            ("CA", first, arbitrary_second, "canonical first × arbitrary second"),
            ("AC", arbitrary_first, second, "arbitrary first × canonical second"),
            ("AA", arbitrary_first, arbitrary_second, "arbitrary first × arbitrary second"),
        )
        for carrier in carriers:
            for variant, left, right, note in factorial:
                cases.append(make_case(
                    "CORE_FACTORIAL", seed_id, variant, carrier,
                    left, connector, right, notes=note,
                ))

            # Endpoint and frame controls share the canonical endpoint.  Their
            # position-match status is measured after tokenisation, not assumed.
            controls = (
                ("FULL_FRAME", first, connector, second,
                 construction(first, connector, second)),
                ("SECOND_ONLY", "", "", second, second),
                ("CONNECTOR_SECOND", "", connector, second,
                 construction("", connector, second)),
                ("FIRST_SECOND", first, "", second,
                 construction(first, "", second)),
                ("ARBFIRST_SECOND", arbitrary_first, "", second,
                 construction(arbitrary_first, "", second)),
                ("FIRST_REPEAT", first, "", first,
                 construction(first, "", first)),
                ("REVERSED", second, connector, first,
                 construction(second, connector, first)),
            )
            for variant, left, conn, right, text in controls:
                cases.append(make_case(
                    "ENDPOINT_CONTROL", seed_id, variant, carrier,
                    left, conn, right, baseline_variant="FULL_FRAME",
                    construction_text=text,
                ))

    if not smoke:
        # Connector substitutions keep members fixed.  Canonical connector rows
        # deliberately repeat CORE_FACTORIAL/CC under a different declared test.
        for seed_id, first, canonical_connector, second in seeds:
            for carrier in carriers:
                for connector in CONNECTORS:
                    cases.append(make_case(
                        "CONNECTOR_MATRIX", seed_id, connector.upper(), carrier,
                        first, connector, second,
                        baseline_variant=canonical_connector.upper(),
                        notes=("canonical connector" if connector == canonical_connector
                               else "connector substitution"),
                    ))

        # Pair rows by predeclared index.  Comparisons are labelled strict only
        # when realised token counts and completion positions actually match.
        for pair_index, (conventional, arbitrary) in enumerate(
            zip(HELDOUT_CONVENTIONAL, ARBITRARY_PAIRS), 1
        ):
            panel_id = f"PAIR_{pair_index:02d}"
            for variant, item in (
                ("CONVENTIONAL", conventional),
                ("ARBITRARY", arbitrary),
            ):
                lexical_id, first, connector, second = item
                for carrier in carriers:
                    cases.append(make_case(
                        "PAIR_PANEL", panel_id, variant, carrier,
                        first, connector, second,
                        baseline_variant="CONVENTIONAL",
                        notes=("held out from v9 certificate sample"
                               if variant == "CONVENTIONAL"
                               else "predeclared non-conventional pair"),
                        extra={"lexical_pair_id": lexical_id},
                    ))

        # Shared continuation panel on the three most recurrent exact v9 seeds.
        continuation_seeds = SEEDS[:3]
        for seed_id, first, connector, second in continuation_seeds:
            for carrier in carriers:
                for continuation in CONTINUATIONS:
                    cases.append(make_case(
                        "CONTINUATION", seed_id, continuation[0], carrier,
                        first, connector, second,
                        continuation=continuation, baseline_variant="NONE",
                    ))

        # Carrier controls quantify what the fixed carrier itself contributes.
        for carrier_id, carrier_text, leading_prefix in CARRIERS[2:]:
            words = carrier_text.strip().split()
            second = words[-1]
            prefix = carrier_text.strip()[:-len(second)]
            cases.append(Case(
                case_id=slug("CARRIER_CONTROL", carrier_id),
                branch="CARRIER_CONTROL",
                seed_id=carrier_id,
                variant="CARRIER_ONLY",
                carrier_id=carrier_id,
                carrier=prefix,
                leading_prefix="",
                first="",
                connector="",
                second=second,
                construction=second,
                notes="fixed carrier measured without construction",
            ))

    ids = [c.case_id for c in cases]
    if len(ids) != len(set(ids)):
        duplicates = [x for x, n in Counter(ids).items() if n > 1]
        raise RuntimeError("duplicate case IDs: " + ", ".join(duplicates))
    return cases, preflight


# ---------------------------------------------------------------------------
# Completion-aware batched scoring

def locate_case_tokens(case):
    encoded = zp.tok(
        case.text, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = list(encoded["input_ids"])
    offsets = [tuple(x) for x in encoded["offset_mapping"]]
    if not ids:
        raise ValueError(f"{case.case_id}: empty token sequence")
    second_positions = [
        i for i, (start, end) in enumerate(offsets)
        if end > case.second_start and start < case.second_end
    ]
    if not second_positions:
        raise ValueError(
            f"{case.case_id}: no token overlaps second member "
            f"{case.second!r} at {case.second_start}:{case.second_end}"
        )
    first_second_body = min(second_positions)
    completion_body = max(second_positions)
    return {
        "ids": ids,
        "offsets": offsets,
        "first_second_body": first_second_body,
        "completion_body": completion_body,
        "pre_second_body_tokens": first_second_body,
        "second_realised_tokens": len(second_positions),
        "completion_pos": completion_body + 1,  # account for BOS
        "pre_second_pos": first_second_body,    # position before first member-2 token
    }


class CompletionScorer:
    def __init__(self, batch_size=64):
        self.batch_size = batch_size
        self.device = next(zp.mdl.parameters()).device
        self.buf = {}
        self.handle = zp.mdl.transformer.h[zp.ELL].mlp.act.register_forward_hook(
            lambda module, inputs, output: self.buf.__setitem__(
                "A", output.detach()
            )
        )

    def close(self):
        self.handle.remove()

    def score(self, cases):
        results = []
        for offset in range(0, len(cases), self.batch_size):
            batch = cases[offset:offset + self.batch_size]
            located = [locate_case_tokens(c) for c in batch]
            sequences = [[zp.BOS] + x["ids"] for x in located]
            if any(len(x) > zp.N_CTX for x in sequences):
                raise ValueError("v10 case outside dom(D)")
            lengths = [len(x) for x in sequences]
            width = max(lengths)
            input_ids = torch.full(
                (len(batch), width), zp.BOS,
                dtype=torch.long, device=self.device,
            )
            attention_mask = torch.zeros(
                (len(batch), width), dtype=torch.long, device=self.device,
            )
            for i, ids in enumerate(sequences):
                input_ids[i, :len(ids)] = torch.tensor(
                    ids, dtype=torch.long, device=self.device
                )
                attention_mask[i, :len(ids)] = 1
            with torch.inference_mode():
                zp.mdl.transformer(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            activations = self.buf["A"]

            for i, (case, loc, length) in enumerate(zip(batch, located, lengths)):
                A = activations[i, :length, :]
                peaks = A.max(dim=0).values
                top2 = torch.topk(peaks, 2)
                dest = int(top2.indices[0])
                peak_541 = float(peaks[TARGET])
                if dest == TARGET:
                    best_other = int(top2.indices[1])
                    best_other_peak = float(top2.values[1])
                else:
                    best_other = dest
                    best_other_peak = float(top2.values[0])

                completion_pos = loc["completion_pos"]
                raw = A[completion_pos, :]
                raw_541 = float(raw[TARGET])
                raw_copy = raw.clone()
                raw_copy[TARGET] = -torch.inf
                raw_other_peak, raw_other_id = raw_copy.max(dim=0)
                target_peakpos = int(A[:, TARGET].argmax())
                ids_with_bos = sequences[i]
                result = {
                    "case_id": case.case_id,
                    "branch": case.branch,
                    "seed_id": case.seed_id,
                    "variant": case.variant,
                    "carrier_id": case.carrier_id,
                    "carrier": case.carrier,
                    "leading_prefix": case.leading_prefix,
                    "first": case.first,
                    "connector": case.connector,
                    "second": case.second,
                    "construction": case.construction,
                    "continuation_id": case.continuation_id,
                    "continuation": case.continuation,
                    "string": case.text,
                    "n_tok": len(loc["ids"]),
                    "tokens": json.dumps(
                        [zp.tok.decode([x]) for x in loc["ids"]],
                        ensure_ascii=False,
                    ),
                    "token_ids": json.dumps(loc["ids"]),
                    "pre_second_body_tokens": loc["pre_second_body_tokens"],
                    "second_realised_tokens": loc["second_realised_tokens"],
                    "completion_pos": completion_pos,
                    "completion_token": zp.tok.decode(
                        [ids_with_bos[completion_pos]]
                    ),
                    "raw_pre_second_541": round(
                        float(A[loc["pre_second_pos"], TARGET]), 6
                    ),
                    "raw_completion_541": round(raw_541, 6),
                    "raw_completion_increment": round(
                        raw_541 - float(A[loc["pre_second_pos"], TARGET]), 6
                    ),
                    "raw_completion_best_other": int(raw_other_id),
                    "raw_completion_best_other_value": round(
                        float(raw_other_peak), 6
                    ),
                    "raw_completion_margin": round(
                        raw_541 - float(raw_other_peak), 6
                    ),
                    "peak_541": round(peak_541, 6),
                    "peakpos_541": target_peakpos,
                    "peak_token_541": (
                        "<BOS>" if target_peakpos == 0 else
                        zp.tok.decode([ids_with_bos[target_peakpos]])
                    ),
                    "best_other": best_other,
                    "best_other_peak": round(best_other_peak, 6),
                    "signed_541_margin": round(peak_541 - best_other_peak, 6),
                    "dest": dest,
                    "retained_541": int(dest == TARGET),
                    "winner_peak": round(float(top2.values[0]), 6),
                    "reported_delta": round(
                        float(top2.values[0] - top2.values[1]), 6
                    ),
                    "target_peaks_at_completion": int(
                        target_peakpos == completion_pos
                    ),
                    "baseline_variant": case.baseline_variant,
                    "notes": case.notes,
                    **case.extra,
                }
                results.append(result)
        return results


def trace_case(case):
    loc = locate_case_tokens(case)
    ids, A = zp.activations(case.text, bos=True)
    rows = []
    for pos in range(A.shape[0]):
        running = A[:pos + 1, :].max(dim=0).values
        leaders = torch.topk(running, 2)
        raw_top = torch.topk(A[pos, :], 5)
        rows.append({
            "case_id": case.case_id,
            "branch": case.branch,
            "seed_id": case.seed_id,
            "variant": case.variant,
            "carrier_id": case.carrier_id,
            "string": case.text,
            "pos": pos,
            "token": "<BOS>" if pos == 0 else zp.tok.decode([ids[pos]]),
            "token_id": ids[pos],
            "is_second_member_token": int(
                loc["first_second_body"] + 1 <= pos <= loc["completion_pos"]
            ),
            "is_completion_token": int(pos == loc["completion_pos"]),
            "raw_541": round(float(A[pos, TARGET]), 6),
            "running_541": round(float(running[TARGET]), 6),
            "running_leader": int(leaders.indices[0]),
            "running_margin": round(
                float(leaders.values[0] - leaders.values[1]), 6
            ),
            "raw_top5": json.dumps([
                [int(j), round(float(x), 6)]
                for j, x in zip(raw_top.indices, raw_top.values)
            ]),
        })
    return rows


# ---------------------------------------------------------------------------
# Matched comparisons and summaries

def comparison_key(row):
    continuation_group = (
        "" if row["branch"] == "CONTINUATION" else row["continuation_id"]
    )
    return row["branch"], row["seed_id"], row["carrier_id"], continuation_group


def build_comparisons(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[comparison_key(row)].append(row)
    out = []
    for key, members in groups.items():
        by_variant = {r["variant"]: r for r in members}
        for row in members:
            baseline_name = row.get("baseline_variant", "")
            baseline = by_variant.get(baseline_name)
            if baseline is None or baseline["case_id"] == row["case_id"]:
                continue
            same_profile = int(
                int(row["pre_second_body_tokens"])
                    == int(baseline["pre_second_body_tokens"])
                and int(row["second_realised_tokens"])
                    == int(baseline["second_realised_tokens"])
                and int(row["completion_pos"])
                    == int(baseline["completion_pos"])
            )
            out.append({
                "comparison_id": slug(baseline["case_id"], "TO", row["variant"]),
                "comparison_type": "WITHIN_CARRIER",
                "branch": row["branch"],
                "seed_id": row["seed_id"],
                "carrier_id": row["carrier_id"],
                "baseline_variant": baseline_name,
                "variant": row["variant"],
                "baseline_case_id": baseline["case_id"],
                "case_id": row["case_id"],
                "baseline_string": baseline["string"],
                "string": row["string"],
                "strict_token_position_match": same_profile,
                "baseline_pre_second_body_tokens": baseline["pre_second_body_tokens"],
                "pre_second_body_tokens": row["pre_second_body_tokens"],
                "baseline_second_realised_tokens": baseline["second_realised_tokens"],
                "second_realised_tokens": row["second_realised_tokens"],
                "raw_completion_541_change": round(
                    float(row["raw_completion_541"])
                    - float(baseline["raw_completion_541"]), 6
                ),
                "raw_completion_increment_change": round(
                    float(row["raw_completion_increment"])
                    - float(baseline["raw_completion_increment"]), 6
                ),
                "peak_541_change": round(
                    float(row["peak_541"]) - float(baseline["peak_541"]), 6
                ),
                "signed_541_margin_change": round(
                    float(row["signed_541_margin"])
                    - float(baseline["signed_541_margin"]), 6
                ),
                "baseline_dest": baseline["dest"],
                "dest": row["dest"],
                "baseline_retained_541": baseline["retained_541"],
                "retained_541": row["retained_541"],
                "retention_changed": int(
                    row["retained_541"] != baseline["retained_541"]
                ),
            })

    # Position/carrier effects are not nuisance rows off to the side: every
    # linguistic case is paired back to its BARE form.  They are intentionally
    # not labelled strict position matches because moving the construction is
    # the intervention.  Completion-token identity and member token count are
    # retained so tokenisation changes remain visible.
    carrier_groups = defaultdict(list)
    for row in rows:
        if row["branch"] == "CARRIER_CONTROL":
            continue
        carrier_groups[(
            row["branch"], row["seed_id"], row["variant"],
            row["continuation_id"],
        )].append(row)
    for key, members in carrier_groups.items():
        by_carrier = {r["carrier_id"]: r for r in members}
        baseline = by_carrier.get("BARE")
        if baseline is None:
            continue
        for row in members:
            if row["carrier_id"] == "BARE":
                continue
            out.append({
                "comparison_id": slug(baseline["case_id"], "TO", row["carrier_id"]),
                "comparison_type": "CARRIER_VS_BARE",
                "branch": row["branch"],
                "seed_id": row["seed_id"],
                "carrier_id": row["carrier_id"],
                "baseline_variant": "BARE",
                "variant": row["carrier_id"],
                "baseline_case_id": baseline["case_id"],
                "case_id": row["case_id"],
                "baseline_string": baseline["string"],
                "string": row["string"],
                "strict_token_position_match": 0,
                "completion_token_match": int(
                    row["completion_token"] == baseline["completion_token"]
                ),
                "second_token_count_match": int(
                    int(row["second_realised_tokens"])
                    == int(baseline["second_realised_tokens"])
                ),
                "baseline_pre_second_body_tokens": baseline["pre_second_body_tokens"],
                "pre_second_body_tokens": row["pre_second_body_tokens"],
                "baseline_second_realised_tokens": baseline["second_realised_tokens"],
                "second_realised_tokens": row["second_realised_tokens"],
                "raw_completion_541_change": round(
                    float(row["raw_completion_541"])
                    - float(baseline["raw_completion_541"]), 6
                ),
                "raw_completion_increment_change": round(
                    float(row["raw_completion_increment"])
                    - float(baseline["raw_completion_increment"]), 6
                ),
                "peak_541_change": round(
                    float(row["peak_541"]) - float(baseline["peak_541"]), 6
                ),
                "signed_541_margin_change": round(
                    float(row["signed_541_margin"])
                    - float(baseline["signed_541_margin"]), 6
                ),
                "baseline_dest": baseline["dest"],
                "dest": row["dest"],
                "baseline_retained_541": baseline["retained_541"],
                "retained_541": row["retained_541"],
                "retention_changed": int(
                    row["retained_541"] != baseline["retained_541"]
                ),
            })
    return out


def mean(values):
    return round(statistics.mean(values), 6) if values else ""


def median(values):
    return round(statistics.median(values), 6) if values else ""


def build_summary(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["branch"], row["variant"], row["carrier_id"])].append(row)
    out = []
    for (branch, variant, carrier_id), members in sorted(groups.items()):
        raw = [float(r["raw_completion_541"]) for r in members]
        increments = [float(r["raw_completion_increment"]) for r in members]
        peaks = [float(r["peak_541"]) for r in members]
        margins = [float(r["signed_541_margin"]) for r in members]
        retained = [int(r["retained_541"]) for r in members]
        out.append({
            "branch": branch,
            "variant": variant,
            "carrier_id": carrier_id,
            "n": len(members),
            "retained_541_n": sum(retained),
            "retained_541_rate": round(sum(retained) / len(retained), 6),
            "raw_completion_541_mean": mean(raw),
            "raw_completion_541_median": median(raw),
            "raw_completion_increment_mean": mean(increments),
            "raw_completion_increment_median": median(increments),
            "peak_541_mean": mean(peaks),
            "peak_541_median": median(peaks),
            "signed_541_margin_mean": mean(margins),
            "signed_541_margin_median": median(margins),
            "completion_is_target_peak_n": sum(
                int(r["target_peaks_at_completion"]) for r in members
            ),
            "member_case_ids": json.dumps([r["case_id"] for r in members]),
        })
    return out


def preflight_rows(cases, selection_rows):
    out = list(selection_rows)
    for case in cases:
        loc = locate_case_tokens(case)
        out.append({
            "case_id": case.case_id,
            "branch": case.branch,
            "seed_id": case.seed_id,
            "variant": case.variant,
            "carrier_id": case.carrier_id,
            "string": case.text,
            "n_tok": len(loc["ids"]),
            "pre_second_body_tokens": loc["pre_second_body_tokens"],
            "second_realised_tokens": loc["second_realised_tokens"],
            "completion_pos": loc["completion_pos"],
            "tokens": json.dumps(
                [zp.tok.decode([x]) for x in loc["ids"]], ensure_ascii=False
            ),
            "token_ids": json.dumps(loc["ids"]),
        })
    return out


def validate_design(cases, rows, comparisons):
    failures = []
    if len(cases) != len(rows):
        failures.append(f"scored {len(rows)} rows for {len(cases)} cases")
    if len({r["case_id"] for r in rows}) != len(rows):
        failures.append("duplicate scored case IDs")
    core = [r for r in rows if r["branch"] == "CORE_FACTORIAL"]
    expected = {"CC", "CA", "AC", "AA"}
    for key, members in _group(core, comparison_key).items():
        got = {r["variant"] for r in members}
        if got != expected:
            failures.append(f"core factorial {key}: variants {sorted(got)}")
    if not any(c["strict_token_position_match"] for c in comparisons):
        failures.append("no strict token/position-matched comparisons")
    if any(int(r["n_tok"]) + 1 > zp.N_CTX for r in rows):
        failures.append("one or more rows outside dom(D)")
    return failures


def _group(rows, key_fn):
    out = defaultdict(list)
    for row in rows:
        out[key_fn(row)].append(row)
    return out


# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--outdir", default="results")
    parser.add_argument(
        "--smoke", action="store_true",
        help="two seeds, bare/leading-space regimes, core and endpoint branches",
    )
    parser.add_argument(
        "--no-trace", action="store_true",
        help="omit per-position trace output after scoring",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    device = next(zp.mdl.parameters()).device
    if device.type != "cuda":
        raise SystemExit(f"znou_probe model is on {device}; v10 requires CUDA")

    cases, selection_rows = build_cases(smoke=args.smoke)
    preflight = preflight_rows(cases, selection_rows)
    print(f"model device: {device}")
    print(f"cases: {len(cases)}")
    print("branches:")
    for branch, n in sorted(Counter(c.branch for c in cases).items()):
        print(f"  {branch:<24} {n}")

    scorer = CompletionScorer(args.batch_size)
    try:
        rows = scorer.score(cases)
    finally:
        scorer.close()

    comparisons = build_comparisons(rows)
    summaries = build_summary(rows)
    traces = []
    if not args.no_trace:
        for index, case in enumerate(cases, 1):
            if index == 1 or index % 50 == 0 or index == len(cases):
                print(f"trace [{index}/{len(cases)}]")
            traces.extend(trace_case(case))

    failures = validate_design(cases, rows, comparisons)
    strict = sum(int(r["strict_token_position_match"]) for r in comparisons)
    retained = sum(int(r["retained_541"]) for r in rows)
    completion_peaks = sum(int(r["target_peaks_at_completion"]) for r in rows)
    print("\nvalidation")
    print(f"  strict matched comparisons  {strict}/{len(comparisons)}")
    print(f"  retained 541                {retained}/{len(rows)}")
    print(f"  541 peak at completion      {completion_peaks}/{len(rows)}")

    outdir = Path(args.outdir)
    outputs = []
    for name, data in (
        ("probe_v10_preflight", preflight),
        ("probe_v10_variants", rows),
        ("probe_v10_comparisons", comparisons),
        ("probe_v10_summary", summaries),
        ("probe_v10_trace", traces),
    ):
        recorder = zp.Recorder(name, outdir=outdir)
        recorder.extend(data)
        outputs.append(recorder.write(suffix="smoke" if args.smoke else ""))

    print("\noutputs")
    for path in outputs:
        print(f"  {path}")
    if failures:
        raise SystemExit("v10 validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
