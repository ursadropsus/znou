"""
probe_v11.py — recurrence and frame-compatibility panel for layer-5 neuron 541.

V10 established that the response of neuron 541 to six distinct-member pairs
depends jointly on both lexical members and a compatible connector.  It did
not properly test the reduplicative and recurrent forms recovered by v9, such
as ``Wave after wave``, ``one by one``, and ``again and again``.  V11 covers
that design gap while keeping feature response separate from map victory:

  raw_completion_541   raw neuron-541 activation on the final token of member 2
  peak_541             maximum neuron-541 activation anywhere in the string
  signed_541_margin    peak_541 minus the strongest non-541 peak
  dest / retained      argmax destination and the thresholded route outcome

The centrepiece crosses lexical status with frame shape.  Each declared panel
contains a known repeated member, a semantically related distinct member, two
predeclared arbitrary members, and connector-removal controls.  A separate
connector sweep holds exact repetition fixed while varying and/or/to/by/after.
A small pronominal-recurrence branch covers forms that are recurrent without
being literal X-connector-X strings.

Raw per-position traces are written for every case.  Running peaks are retained
for operational interpretation but are not treated as independent evidence of
completion sensitivity: under Resonance they are monotone by construction.

Requires the CUDA-corrected znou_probe.py beside this file.

Examples
--------
  python probe_v11.py --smoke
  python probe_v11.py
  python probe_v11.py --batch-size 32 --outdir results
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

# Each tuple is:
#   panel_id, native connector, known member, related distinct member,
#   arbitrary member A, arbitrary member B, provenance.
#
# Words were declared before inference.  The arbitrary labels mean "selected
# as unrelated controls", not semantically proven nulls.  PREFLIGHT records
# realised token counts rather than assuming matching tokenisation.
FRAME_PANELS = (
    ("AFTER_WAVE", "after", "wave", "tide", "gravel", "window", "V9_ATTESTED"),
    ("AFTER_DAY", "after", "day", "night", "marble", "forest", "HELDOUT_FAMILIAR"),
    ("AFTER_MILE", "after", "mile", "road", "candle", "basket", "HELDOUT_FAMILIAR"),
    ("BY_ONE", "by", "one", "two", "velvet", "ocean", "V9_ATTESTED"),
    ("BY_ZONE", "by", "zone", "region", "pocket", "pencil", "V9_ATTESTED"),
    ("BY_STEP", "by", "step", "stride", "lantern", "engine", "HELDOUT_FAMILIAR"),
    ("AND_ROUND", "and", "round", "straight", "rabbit", "button", "V9_ATTESTED"),
    ("AND_AGAIN", "and", "again", "then", "hammer", "violin", "V9_ATTESTED"),
    ("AND_TIME", "and", "time", "space", "mustard", "planet", "HELDOUT_FAMILIAR"),
    ("TO_HAND", "to", "hand", "foot", "teacup", "ribbon", "V9_ATTESTED"),
    ("TO_FACE", "to", "face", "back", "anchor", "chimney", "HELDOUT_FAMILIAR"),
    ("TO_SHOULDER", "to", "shoulder", "knee", "copper", "garden", "HELDOUT_FAMILIAR"),
)

# These forms express recurrence or succession but do not all instantiate
# literal lexical identity on the two sides of the connector.
PRONOMINAL_FORMS = (
    ("ONE_AFTER_ANOTHER", "one", "after", "another", "one after another"),
    ("ONE_AFTER_THE_OTHER", "one", "after", "other", "one after the other"),
    ("ONE_AFTER_ONE", "one", "after", "one", "one after one"),
    ("ONE_AND_ANOTHER", "one", "and", "another", "one and another"),
    ("ONE_ANOTHER", "one", "", "another", "one another"),
    ("GRAVEL_AFTER_ANOTHER", "gravel", "after", "another", "gravel after another"),
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
    panels = FRAME_PANELS[:2] if smoke else FRAME_PANELS
    carriers = CARRIERS[:2] if smoke else CARRIERS

    for (panel_id, native_connector, known, related, arbitrary_a,
         arbitrary_b, provenance) in panels:
        preflight.append({
            "panel_id": panel_id,
            "native_connector": native_connector,
            "known_member": known,
            "related_member": related,
            "arbitrary_a": arbitrary_a,
            "arbitrary_b": arbitrary_b,
            "provenance": provenance,
            "known_space_tokens": member_token_count(known),
            "related_space_tokens": member_token_count(related),
            "arbitrary_a_space_tokens": member_token_count(arbitrary_a),
            "arbitrary_b_space_tokens": member_token_count(arbitrary_b),
            "selection_rule": "predeclared lexical panel; no activation-based selection",
        })

        # The primary frame crossing.  KNOWN_REPEAT is the within-carrier
        # baseline.  ARBITRARY_REPEAT is the decisive structural-generalisation
        # cell: the repeated frame is preserved while familiar lexical content
        # is removed.  Connector-removal variants test whether repetition alone
        # is sufficient in the exact lexical regimes.
        frame_variants = (
            ("KNOWN_REPEAT", known, native_connector, known,
             "KNOWN_REPEAT", "known member repeated in native connector frame"),
            ("KNOWN_RELATED", known, native_connector, related,
             "RELATED_DISTINCT", "known first plus related distinct second"),
            ("KNOWN_ARBITRARY", known, native_connector, arbitrary_b,
             "ARBITRARY_DISTINCT", "known first plus arbitrary distinct second"),
            ("ARBITRARY_KNOWN", arbitrary_a, native_connector, known,
             "ARBITRARY_DISTINCT", "arbitrary first plus known second"),
            ("ARBITRARY_REPEAT", arbitrary_a, native_connector, arbitrary_a,
             "NOVEL_REPEAT", "predeclared arbitrary member repeated"),
            ("ARBITRARY_DISTINCT", arbitrary_a, native_connector, arbitrary_b,
             "NOVEL_DISTINCT", "two predeclared arbitrary members"),
            ("KNOWN_NO_CONNECTOR", known, "", known,
             "NO_CONNECTOR", "known repetition with connector removed"),
            ("ARBITRARY_NO_CONNECTOR", arbitrary_a, "", arbitrary_a,
             "NO_CONNECTOR", "arbitrary repetition with connector removed"),
        )
        for carrier in carriers:
            for variant, first, connector, second, frame_class, note in frame_variants:
                cases.append(make_case(
                    "FRAME_CROSS", panel_id, variant, carrier,
                    first, connector, second,
                    baseline_variant="KNOWN_REPEAT", notes=note,
                    extra={
                        "native_connector": native_connector,
                        "frame_class": frame_class,
                        "source_provenance": provenance,
                        "known_member": known,
                        "related_member": related,
                        "arbitrary_a": arbitrary_a,
                        "arbitrary_b": arbitrary_b,
                    },
                ))

        # Hold exact lexical repetition fixed and change only the connector.
        # This is separate from FRAME_CROSS so a connector's repeated-frame
        # behaviour is never confused with its distinct-member behaviour.
        for carrier in carriers:
            for connector in CONNECTORS:
                cases.append(make_case(
                    "REPEAT_CONNECTOR", panel_id, connector.upper(), carrier,
                    known, connector, known,
                    baseline_variant=native_connector.upper(),
                    notes=("native connector" if connector == native_connector
                           else "connector substitution under exact repetition"),
                    extra={
                        "native_connector": native_connector,
                        "frame_class": "KNOWN_REPEAT",
                        "source_provenance": provenance,
                        "known_member": known,
                        "related_member": related,
                        "arbitrary_a": arbitrary_a,
                        "arbitrary_b": arbitrary_b,
                    },
                ))

    # A small non-literal recurrence branch.  In smoke mode it uses two forms
    # and two carriers; the full run uses the declared panel and all carriers.
    pronouns = PRONOMINAL_FORMS[:2] if smoke else PRONOMINAL_FORMS
    for form_id, first, connector, second, text in pronouns:
        for carrier in carriers:
            cases.append(make_case(
                "PRONOMINAL_RECURRENCE", "PRONOMINAL", form_id, carrier,
                first, connector, second, construction_text=text,
                baseline_variant="ONE_AFTER_ANOTHER",
                notes="declared pronominal or correlative recurrence control",
                extra={
                    "native_connector": "after",
                    "frame_class": "PRONOMINAL_RECURRENCE",
                    "source_provenance": (
                        "V9_ATTESTED" if form_id in {
                            "ONE_AFTER_ANOTHER", "ONE_AFTER_THE_OTHER"
                        } else "CONTROL"
                    ),
                    "known_member": "one",
                    "related_member": "another",
                    "arbitrary_a": "gravel",
                    "arbitrary_b": "",
                },
            ))

    # Carrier-only controls quantify the two fixed carrier contributions.
    if not smoke:
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
                extra={
                    "native_connector": "",
                    "frame_class": "CARRIER_ONLY",
                    "source_provenance": "CONTROL",
                    "known_member": "",
                    "related_member": "",
                    "arbitrary_a": "",
                    "arbitrary_b": "",
                },
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
                raise ValueError("v11 case outside dom(D)")
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
    return row["branch"], row["seed_id"], row["carrier_id"]


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
                "native_connector": row.get("native_connector", ""),
                "frame_class": row.get("frame_class", ""),
                "source_provenance": row.get("source_provenance", ""),
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
                "native_connector": row.get("native_connector", ""),
                "frame_class": row.get("frame_class", ""),
                "source_provenance": row.get("source_provenance", ""),
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


def build_cells(rows):
    """Focused v11 summaries retaining connector and provenance structure."""
    groups = defaultdict(list)
    for row in rows:
        if row["branch"] == "CARRIER_CONTROL":
            continue
        groups[(
            row["branch"], row.get("native_connector", ""),
            row["variant"], row.get("frame_class", ""),
            row.get("source_provenance", ""),
        )].append(row)
    out = []
    for key, members in sorted(groups.items()):
        branch, native_connector, variant, frame_class, provenance = key
        raw = [float(r["raw_completion_541"]) for r in members]
        peaks = [float(r["peak_541"]) for r in members]
        margins = [float(r["signed_541_margin"]) for r in members]
        retained = [int(r["retained_541"]) for r in members]
        out.append({
            "branch": branch,
            "native_connector": native_connector,
            "variant": variant,
            "frame_class": frame_class,
            "source_provenance": provenance,
            "n": len(members),
            "retained_541_n": sum(retained),
            "retained_541_rate": round(sum(retained) / len(retained), 6),
            "raw_completion_541_mean": mean(raw),
            "raw_completion_541_median": median(raw),
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
    frame = [r for r in rows if r["branch"] == "FRAME_CROSS"]
    expected = {
        "KNOWN_REPEAT", "KNOWN_RELATED", "KNOWN_ARBITRARY",
        "ARBITRARY_KNOWN", "ARBITRARY_REPEAT", "ARBITRARY_DISTINCT",
        "KNOWN_NO_CONNECTOR", "ARBITRARY_NO_CONNECTOR",
    }
    for key, members in _group(frame, comparison_key).items():
        got = {r["variant"] for r in members}
        if got != expected:
            failures.append(f"frame cross {key}: variants {sorted(got)}")
    repeat = [r for r in rows if r["branch"] == "REPEAT_CONNECTOR"]
    connector_expected = {x.upper() for x in CONNECTORS}
    for key, members in _group(repeat, comparison_key).items():
        got = {r["variant"] for r in members}
        if got != connector_expected:
            failures.append(f"repeat connector {key}: variants {sorted(got)}")
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
        help="two panels, two carriers, frame/connector/pronominal branches",
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
        raise SystemExit(f"znou_probe model is on {device}; v11 requires CUDA")

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
    cells = build_cells(rows)
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
        ("probe_v11_preflight", preflight),
        ("probe_v11_variants", rows),
        ("probe_v11_comparisons", comparisons),
        ("probe_v11_summary", summaries),
        ("probe_v11_cells", cells),
        ("probe_v11_trace", traces),
    ):
        recorder = zp.Recorder(name, outdir=outdir)
        recorder.extend(data)
        outputs.append(recorder.write(suffix="smoke" if args.smoke else ""))

    print("\noutputs")
    for path in outputs:
        print(f"  {path}")
    if failures:
        raise SystemExit("v11 validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
