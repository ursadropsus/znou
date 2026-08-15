"""
probe_v9.py — boundary profiles and within-neuron route families.

V9 is designed for a large Atlas export containing many full-sentence routes
to one or more layer-5 neurons.  It verifies every route, then characterises
the source region around the labelled neuron's full-sentence peak without
performing an exhaustive quadratic search on every sentence.

For every verified route V9 computes:

  PREFIX          source prefix ending at the target peak
  TOKEN_MIN       shortest winning peak-anchored left deletion
  WORD_ANCHORED   shortest winning source-word-bounded span around that peak
  MARGIN_MAX      widest-margin winner in the anchored left-deletion ladder
  PARETO          winners not beaten on both realised length and margin
  EXTENSION       TOKEN_MIN extended rightward through the remaining source

An evenly spaced, length-stratified audit subset additionally receives a true
exhaustive contiguous-source-span search (CONTIG_AUDIT).  This tests rather
than assumes that peak anchoring recovers the same minimum.  The audit streams
candidates in batches and checkpoints after every completed span width.

"Word bounded" is deliberately not called globally word-minimal.  It means
that both source character boundaries lie outside a word-like run, where
internal apostrophes and hyphens are treated as part of the written word.

Route checkpoints are written after each stage.  Re-running the same command
resumes completed work if the input hash and analysis options agree.

Requires the CUDA-corrected znou_probe.py beside this file.

Example
-------
  python probe_v9.py --input "J5-541 (80 Hits).txt"
  python probe_v9.py --input routes.txt --limit 3 --audit-size 1
  python probe_v9.py --input routes.txt --audit-size 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

import znou_probe as zp


SCRIPT_VERSION = 9
HEADER_RE = re.compile(r"^J5-(\d+)\s+\((\d+)\s+Hits?\)\s*$")
WORD_RE = re.compile(r"[^\W_]+(?:[-'’][^\W_]+)*", re.UNICODE)


@dataclass(frozen=True)
class Route:
    neuron: int
    string: str
    source: str
    source_index: int
    declared_hits: int | None = None

    @property
    def route_id(self):
        return f"J5-{self.neuron}-{self.source_index:03d}"


# ---------------------------------------------------------------------------
# Input and deterministic audit selection

def unwrap_export_quote(line):
    """Remove one paired ASCII wrapper quote and preserve inner punctuation."""
    if len(line) >= 2 and line.startswith('"') and line.endswith('"'):
        return line[1:-1]
    return line


def parse_atlas_export(path):
    routes = []
    current_neuron = None
    declared_hits = None
    in_inputs = False
    per_neuron_index = 0

    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        line = raw.strip()
        match = HEADER_RE.match(line)
        if match:
            current_neuron = int(match.group(1))
            declared_hits = int(match.group(2))
            in_inputs = False
            per_neuron_index = 0
            continue
        if line == "All Inputs:":
            if current_neuron is None:
                raise ValueError(f"line {line_number}: All Inputs before header")
            in_inputs = True
            continue
        if not in_inputs or not line:
            continue

        per_neuron_index += 1
        s = unwrap_export_quote(line)
        if not s:
            raise ValueError(f"line {line_number}: empty Atlas input")
        routes.append(Route(
            neuron=current_neuron,
            string=s,
            source=f"atlas:{Path(path).name}",
            source_index=per_neuron_index,
            declared_hits=declared_hits,
        ))

    if not routes:
        raise ValueError(f"no Atlas routes parsed from {path}")

    counts = Counter(r.neuron for r in routes)
    declarations = {r.neuron: r.declared_hits for r in routes}
    mismatches = [
        f"J5-{j}: parsed {counts[j]}, declared {declarations[j]}"
        for j in sorted(counts)
        if declarations[j] is not None and counts[j] != declarations[j]
    ]
    if mismatches:
        raise ValueError("Atlas hit-count mismatch: " + "; ".join(mismatches))
    if len({(r.neuron, r.string) for r in routes}) != len(routes):
        raise ValueError("duplicate neuron/string rows in Atlas export")
    return routes


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def select_audit_ids(routes, token_counts, audit_size):
    """Pre-reduction selection: focus row plus even length quantiles."""
    if audit_size <= 0:
        return set()
    k = min(audit_size, len(routes))
    ordered = sorted(routes, key=lambda r: (token_counts[r.route_id], r.route_id))
    # The first Atlas row is the export's declared Focus prompt.  Select the
    # remaining slots across the full length range after removing that row;
    # this prevents the focus reservation from displacing the longest route.
    chosen = [routes[0].route_id] if routes else []
    if k == 1:
        return set(chosen)

    remaining = [r for r in ordered if r.route_id not in chosen]
    slots = k - len(chosen)
    for i in range(slots):
        pos = round(i * (len(remaining) - 1) / max(1, slots - 1))
        chosen.append(remaining[pos].route_id)
    for route in ordered:
        if len(chosen) >= k:
            break
        if route.route_id not in chosen:
            chosen.append(route.route_id)
    return set(chosen[:k])


# ---------------------------------------------------------------------------
# Batched activation scorer

class BatchScorer:
    """Peak readout without allocating vocabulary logits."""

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

    def score(self, strings, target):
        results = []
        for offset in range(0, len(strings), self.batch_size):
            batch = strings[offset:offset + self.batch_size]
            bodies = [list(zp.tok(s)["input_ids"]) for s in batch]
            if any(not ids for ids in bodies):
                raise ValueError("empty token sequence outside dom(D)")
            sequences = [[zp.BOS] + ids for ids in bodies]
            if any(len(ids) > zp.N_CTX for ids in sequences):
                raise ValueError("candidate outside dom(D)")
            lengths = [len(ids) for ids in sequences]
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

            for i, length in enumerate(lengths):
                A = activations[i, :length, :]
                peaks = A.max(dim=0).values
                top2 = torch.topk(peaks, 2)
                target_peak = float(peaks[target])
                dest = int(top2.indices[0])
                if dest == target:
                    other_id = int(top2.indices[1])
                    other_peak = float(top2.values[1])
                else:
                    other_id = dest
                    other_peak = float(top2.values[0])
                results.append({
                    "dest": dest,
                    "delta": round(float(top2.values[0] - top2.values[1]), 6),
                    "runner_up": int(top2.indices[1]),
                    "winner_peak": round(float(top2.values[0]), 6),
                    "target_peak": round(target_peak, 6),
                    "target_peakpos": int(A[:, target].argmax()),
                    "best_other": other_id,
                    "best_other_peak": round(other_peak, 6),
                    "signed_target_margin": round(target_peak - other_peak, 6),
                    "n_tok": length - 1,
                })
        return results


# ---------------------------------------------------------------------------
# Source spans, boundaries, and certificate selection

def realised_span(source_ids, start, end):
    intended = source_ids[start:end]
    string = zp.tok.decode(intended)
    realised = list(zp.tok(string)["input_ids"])
    if not string or not realised:
        return None
    return {
        "string": string,
        "source_start": start,
        "source_end": end,
        "source_span_n_tok": end - start,
        "realised_token_ids": realised,
        "realised_n_tok": len(realised),
        "roundtrip_stable": int(
            list(zp.tok(zp.tok.decode(realised))["input_ids"]) == realised
        ),
    }


def attach_scores(candidates, scorer, target):
    if not candidates:
        return candidates
    scores = scorer.score([c["string"] for c in candidates], target)
    for candidate, score in zip(candidates, scores):
        candidate.update(score)
    return candidates


def token_at(string, peak_position):
    if int(peak_position) == 0:
        return "<BOS>"
    ids = [zp.BOS] + list(zp.tok(string)["input_ids"])
    return zp.tok.decode([ids[int(peak_position)]])


def candidate_row(route, method, candidate, full_target_peak, **extra):
    if candidate is None:
        return {
            "route_id": route.route_id,
            "target_neuron": route.neuron,
            "source_index": route.source_index,
            "method": method,
            "found": 0,
            **extra,
        }
    return {
        "route_id": route.route_id,
        "target_neuron": route.neuron,
        "source_index": route.source_index,
        "method": method,
        "found": 1,
        "string": candidate["string"],
        "source_start": candidate["source_start"],
        "source_end": candidate["source_end"],
        "source_span_n_tok": candidate["source_span_n_tok"],
        "realised_n_tok": candidate["realised_n_tok"],
        "dest": candidate["dest"],
        "target_wins": int(candidate["dest"] == route.neuron),
        "target_peak": candidate["target_peak"],
        "target_peakpos": candidate["target_peakpos"],
        "target_peak_token": token_at(
            candidate["string"], candidate["target_peakpos"]
        ),
        "target_peak_drift": round(
            float(candidate["target_peak"]) - full_target_peak, 6
        ),
        "best_other": candidate["best_other"],
        "best_other_peak": candidate["best_other_peak"],
        "signed_target_margin": candidate["signed_target_margin"],
        "reported_delta": candidate["delta"],
        "runner_up": candidate["runner_up"],
        "roundtrip_stable": candidate["roundtrip_stable"],
        "realised_token_ids": json.dumps(candidate["realised_token_ids"]),
        **extra,
    }


def choose_token_min(winners, full_target_peak):
    if not winners:
        return None
    return min(winners, key=lambda c: (
        c["realised_n_tok"],
        len(c["string"]),
        abs(float(c["target_peak"]) - full_target_peak),
        c["source_start"],
    ))


def choose_margin_max(winners):
    if not winners:
        return None
    return min(winners, key=lambda c: (
        -float(c["signed_target_margin"]),
        c["realised_n_tok"],
        len(c["string"]),
        c["source_start"],
    ))


def pareto_winners(winners):
    """Unique strings not dominated on realised token count and margin."""
    unique = {}
    for c in winners:
        key = c["string"]
        old = unique.get(key)
        if old is None or float(c["signed_target_margin"]) > float(
            old["signed_target_margin"]
        ):
            unique[key] = c
    candidates = list(unique.values())
    frontier = []
    for c in candidates:
        dominated = any(
            d is not c
            and d["realised_n_tok"] <= c["realised_n_tok"]
            and float(d["signed_target_margin"])
                >= float(c["signed_target_margin"])
            and (
                d["realised_n_tok"] < c["realised_n_tok"]
                or float(d["signed_target_margin"])
                    > float(c["signed_target_margin"])
            )
            for d in candidates
        )
        if not dominated:
            frontier.append(c)
    return sorted(frontier, key=lambda c: (
        c["realised_n_tok"], -float(c["signed_target_margin"]), c["string"]
    ))


def word_boundaries(source):
    """Character positions not strictly inside a word-like regex span."""
    inside = set()
    for match in WORD_RE.finditer(source):
        inside.update(range(match.start() + 1, match.end()))
    return set(range(len(source) + 1)) - inside


def source_token_offsets(source):
    encoded = zp.tok(
        source, add_special_tokens=False, return_offsets_mapping=True
    )
    return list(encoded["input_ids"]), [tuple(x) for x in encoded["offset_mapping"]]


def anchored_candidates(route, source_ids, peak_end, scorer):
    candidates = [
        realised_span(source_ids, start, peak_end)
        for start in range(0, peak_end)
    ]
    candidates = [c for c in candidates if c is not None]
    attach_scores(candidates, scorer, route.neuron)
    return candidates


def word_anchored_candidates(route, source_ids, offsets, peak_end, scorer):
    boundaries = word_boundaries(route.string)
    token_starts = [start for start, _ in offsets]
    token_ends = [end for _, end in offsets]

    # First source-token endpoint at or after the peak that closes the written
    # word-like run containing the peak.  Hyphens/apostrophes internal to a
    # regex word are therefore not accepted as end boundaries.
    right_end = None
    for end in range(peak_end, len(source_ids) + 1):
        char_end = token_ends[end - 1]
        if char_end in boundaries:
            right_end = end
            break
    if right_end is None:
        return [], None

    candidates = []
    for start in range(0, peak_end):
        char_start = token_starts[start]
        if char_start not in boundaries:
            continue
        candidate = realised_span(source_ids, start, right_end)
        if candidate is not None:
            candidates.append(candidate)
    attach_scores(candidates, scorer, route.neuron)
    return candidates, right_end


def extension_candidates(route, source_ids, fixed_start, peak_end, scorer):
    candidates = [
        realised_span(source_ids, fixed_start, end)
        for end in range(peak_end, len(source_ids) + 1)
    ]
    candidates = [c for c in candidates if c is not None]
    attach_scores(candidates, scorer, route.neuron)
    return candidates


# ---------------------------------------------------------------------------
# Checkpointing and exhaustive audit

def atomic_json(path, value):
    """Durably replace a JSON checkpoint, tolerating Windows scan locks.

    A fixed ``.tmp`` name can be opened briefly by Defender, an indexer, or an
    editor immediately after it is written.  Windows then rejects os.replace
    with WinError 5.  Use a unique temporary name for every write and retry the
    atomic rename; never truncate the last good checkpoint in place.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=path.name + ".",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())

        last_error = None
        for attempt in range(10):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as error:
                last_error = error
                # 0.05, 0.10, 0.20 ... capped at one second.  The unique temp
                # remains intact throughout, and the previous checkpoint is
                # never deleted before a successful atomic replacement.
                time.sleep(min(0.05 * (2 ** attempt), 1.0))
        raise last_error
    finally:
        # Normally os.replace has moved the temp path.  On persistent failure,
        # remove only this call's private temporary file where Windows allows.
        try:
            temporary.unlink(missing_ok=True)
        except PermissionError:
            pass


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def audit_checkpoint_path(checkpoint_dir, route):
    return Path(checkpoint_dir) / f"audit_{route.route_id}.json"


def exhaustive_audit(route, source_ids, scorer, full_target_peak,
                     checkpoint_dir, max_span=0):
    """Stream all declared contiguous spans; resume at completed width."""
    n = len(source_ids)
    limit = min(n, max_span) if max_span else n
    path = audit_checkpoint_path(checkpoint_dir, route)
    state = read_json(path, {
        "route_id": route.route_id,
        "next_width": 1,
        "evaluated": 0,
        "hits": 0,
        "best": None,
        "complete": False,
    })
    if state.get("complete"):
        return state

    best = state.get("best")
    for width in range(int(state["next_width"]), limit + 1):
        for offset in range(0, n - width + 1, scorer.batch_size * 8):
            candidates = []
            stop = min(n - width + 1, offset + scorer.batch_size * 8)
            for start in range(offset, stop):
                c = realised_span(source_ids, start, start + width)
                if c is not None:
                    candidates.append(c)
            attach_scores(candidates, scorer, route.neuron)
            state["evaluated"] += len(candidates)
            winners = [c for c in candidates if c["dest"] == route.neuron]
            state["hits"] += len(winners)
            for candidate in winners:
                if best is None:
                    best = candidate
                else:
                    best = choose_token_min([best, candidate], full_target_peak)
        state["best"] = best
        state["next_width"] = width + 1
        atomic_json(path, state)

    state["complete"] = True
    state["max_span"] = max_span
    state["source_n_tok"] = n
    atomic_json(path, state)
    return state


# ---------------------------------------------------------------------------
# One-route analysis

def analyse_route(route, scorer, audit_selected, checkpoint_dir, max_audit_span):
    route_path = Path(checkpoint_dir) / f"route_{route.route_id}.json"
    cached = read_json(route_path)
    if cached is not None and cached.get("complete"):
        return cached

    source_ids, offsets = source_token_offsets(route.string)
    full_candidate = realised_span(source_ids, 0, len(source_ids))
    # Verification is against the exact supplied Atlas string, not merely the
    # decoded token slice used by subsequent source-span searches.
    full_candidate["string"] = route.string
    full_candidate["realised_token_ids"] = list(
        zp.tok(route.string)["input_ids"]
    )
    full_candidate["realised_n_tok"] = len(full_candidate["realised_token_ids"])
    attach_scores([full_candidate], scorer, route.neuron)
    full = full_candidate
    verified = int(full["dest"] == route.neuron)
    base = {
        "route_id": route.route_id,
        "target_neuron": route.neuron,
        "source": route.source,
        "source_index": route.source_index,
        "declared_hits": route.declared_hits,
        "full_string": route.string,
        "full_n_tok": len(source_ids),
        "full_dest": full["dest"],
        "full_delta": full["delta"],
        "full_target_peak": full["target_peak"],
        "full_target_peakpos": full["target_peakpos"],
        "full_target_peak_token": token_at(
            route.string, full["target_peakpos"]
        ),
        "full_best_other": full["best_other"],
        "full_best_other_peak": full["best_other_peak"],
        "verified": verified,
        "audit_selected": int(audit_selected),
    }
    result = {
        "summary": base,
        "certificates": [],
        "ladder": [],
        "extensions": [],
        "audit": None,
        "complete": False,
    }

    if not verified:
        result["summary"]["status"] = "LABEL_MISMATCH"
        result["complete"] = True
        atomic_json(route_path, result)
        return result

    peak_end = int(full["target_peakpos"])
    if peak_end <= 0:
        result["summary"]["status"] = "TARGET_BOS_PEAK"
        result["complete"] = True
        atomic_json(route_path, result)
        return result

    full_target_peak = float(full["target_peak"])
    ladder = anchored_candidates(route, source_ids, peak_end, scorer)
    winners = [c for c in ladder if c["dest"] == route.neuron]
    token_min = choose_token_min(winners, full_target_peak)
    margin_max = choose_margin_max(winners)
    pareto = pareto_winners(winners)

    prefix = next((c for c in ladder if c["source_start"] == 0), None)
    word_candidates, word_right_end = word_anchored_candidates(
        route, source_ids, offsets, peak_end, scorer
    )
    word_winners = [c for c in word_candidates if c["dest"] == route.neuron]
    word_anchored = choose_token_min(word_winners, full_target_peak)

    result["ladder"] = [
        candidate_row(
            route, "ANCHORED_LADDER", c, full_target_peak,
            target_wins=int(c["dest"] == route.neuron),
            is_token_min=int(c is token_min),
            is_margin_max=int(c is margin_max),
            is_pareto=int(c in pareto),
        )
        for c in ladder
    ]
    result["certificates"] = [
        candidate_row(route, "PREFIX", prefix, full_target_peak),
        candidate_row(route, "TOKEN_MIN", token_min, full_target_peak),
        candidate_row(
            route, "WORD_ANCHORED", word_anchored, full_target_peak,
            word_right_source_end=(
                word_right_end if word_right_end is not None else ""
            ),
        ),
        candidate_row(route, "MARGIN_MAX", margin_max, full_target_peak),
    ] + [
        candidate_row(
            route, "PARETO", c, full_target_peak,
            pareto_index=i, pareto_size=len(pareto),
        )
        for i, c in enumerate(pareto, 1)
    ]

    if token_min is not None:
        extensions = extension_candidates(
            route, source_ids, token_min["source_start"], peak_end, scorer
        )
        base_margin = float(extensions[0]["signed_target_margin"])
        base_runner = int(extensions[0]["best_other"])
        result["extensions"] = [
            candidate_row(
                route, "RIGHT_EXTENSION", c, full_target_peak,
                extension_tokens=c["source_end"] - peak_end,
                margin_change_from_certificate=round(
                    float(c["signed_target_margin"]) - base_margin, 6
                ),
                competitor_switched=int(int(c["best_other"]) != base_runner),
            )
            for c in extensions
        ]
    else:
        extensions = []

    if audit_selected:
        audit_state = exhaustive_audit(
            route, source_ids, scorer, full_target_peak,
            checkpoint_dir, max_span=max_audit_span,
        )
        audit_best = audit_state.get("best")
        result["audit"] = candidate_row(
            route, "CONTIG_AUDIT", audit_best, full_target_peak,
            candidates_evaluated=audit_state["evaluated"],
            successful_candidates=audit_state["hits"],
            max_source_span=max_audit_span,
            agrees_token_min=int(
                audit_best is not None and token_min is not None
                and audit_best["string"] == token_min["string"]
            ),
        )

    first_loss = next(
        (c for c in extensions[1:] if c["dest"] != route.neuron), None
    )
    first_narrower = next(
        (
            c for c in extensions[1:]
            if float(c["signed_target_margin"])
                < float(extensions[0]["signed_target_margin"]) - 0.0000005
        ),
        None,
    ) if extensions else None
    first_competitor_switch = next(
        (
            c for c in extensions[1:]
            if c["best_other"] != extensions[0]["best_other"]
        ),
        None,
    ) if extensions else None

    def pick(candidate, key, default=""):
        return candidate.get(key, default) if candidate is not None else default

    result["summary"].update({
        "peak_source_end": peak_end,
        "anchored_candidates": len(ladder),
        "anchored_winners": len(winners),
        "token_min_string": pick(token_min, "string"),
        "token_min_n_tok": pick(token_min, "realised_n_tok"),
        "token_min_margin": pick(token_min, "signed_target_margin"),
        "token_min_target_peak": pick(token_min, "target_peak"),
        "token_min_peak_drift": (
            round(float(token_min["target_peak"]) - full_target_peak, 6)
            if token_min is not None else ""
        ),
        "word_anchored_string": pick(word_anchored, "string"),
        "word_anchored_n_tok": pick(word_anchored, "realised_n_tok"),
        "word_anchored_margin": pick(word_anchored, "signed_target_margin"),
        "margin_max_string": pick(margin_max, "string"),
        "margin_max_n_tok": pick(margin_max, "realised_n_tok"),
        "margin_max_margin": pick(margin_max, "signed_target_margin"),
        "pareto_size": len(pareto),
        "extension_rows": len(extensions),
        "first_narrower_extension_tokens": (
            pick(first_narrower, "source_end", 0) - peak_end
            if first_narrower is not None else ""
        ),
        "first_narrower_extension_token": (
            zp.tok.decode([source_ids[first_narrower["source_end"] - 1]])
            if first_narrower is not None else ""
        ),
        "first_competitor_switch_extension_tokens": (
            pick(first_competitor_switch, "source_end", 0) - peak_end
            if first_competitor_switch is not None else ""
        ),
        "first_loss_extension_tokens": (
            pick(first_loss, "source_end", 0) - peak_end
            if first_loss is not None else ""
        ),
        "first_loss_extension_token": (
            zp.tok.decode([source_ids[first_loss["source_end"] - 1]])
            if first_loss is not None else ""
        ),
        "audit_agrees_token_min": (
            result["audit"].get("agrees_token_min", "")
            if result["audit"] is not None else ""
        ),
        "status": "OK" if token_min is not None else "NO_ANCHORED_CERTIFICATE",
    })
    result["complete"] = True
    atomic_json(route_path, result)
    return result


# ---------------------------------------------------------------------------
# Cross-route exact families and token-overlap measurements

def multiset_jaccard(a, b):
    ca, cb = Counter(a), Counter(b)
    intersection = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return intersection / union if union else 1.0


def bigram_jaccard(a, b):
    aa = set(zip(a, a[1:]))
    bb = set(zip(b, b[1:]))
    if not aa and not bb:
        return 1.0 if a == b else 0.0
    return len(aa & bb) / len(aa | bb)


def longest_common_contiguous(a, b):
    previous = [0] * (len(b) + 1)
    best = 0
    for x in a:
        current = [0]
        for j, y in enumerate(b, 1):
            value = previous[j - 1] + 1 if x == y else 0
            current.append(value)
            best = max(best, value)
        previous = current
    return best


def build_family_outputs(summaries):
    usable = [s for s in summaries if s.get("token_min_string")]
    groups = {}
    for row in usable:
        key = (row["target_neuron"], row["token_min_string"])
        groups.setdefault(key, []).append(row)
    ordered = sorted(
        groups.items(), key=lambda item: (-len(item[1]), item[0])
    )
    family_rows = []
    family_by_route = {}
    for index, ((target_neuron, string), members) in enumerate(ordered, 1):
        family_id = f"EXACT-{index:03d}"
        member_ids = [m["route_id"] for m in members]
        token_ids = list(zp.tok(string)["input_ids"])
        family_rows.append({
            "family_id": family_id,
            "target_neuron": target_neuron,
            "certificate": string,
            "count": len(members),
            "realised_n_tok": len(token_ids),
            "member_route_ids": json.dumps(member_ids),
            "member_source_indices": json.dumps(
                [m["source_index"] for m in members]
            ),
            "recurrent": int(len(members) > 1),
        })
        for route_id in member_ids:
            family_by_route[route_id] = family_id

    overlap_rows = []
    for i, left in enumerate(usable):
        a = list(zp.tok(left["token_min_string"])["input_ids"])
        for right in usable[i + 1:]:
            b = list(zp.tok(right["token_min_string"])["input_ids"])
            lcc = longest_common_contiguous(a, b)
            overlap_rows.append({
                "target_neuron_left": left["target_neuron"],
                "target_neuron_right": right["target_neuron"],
                "route_id_left": left["route_id"],
                "route_id_right": right["route_id"],
                "certificate_left": left["token_min_string"],
                "certificate_right": right["token_min_string"],
                "exact_match": int(left["token_min_string"] == right["token_min_string"]),
                "unigram_multiset_jaccard": round(multiset_jaccard(a, b), 6),
                "bigram_jaccard": round(bigram_jaccard(a, b), 6),
                "longest_common_contiguous_tokens": lcc,
                "left_fraction_in_lcc": round(lcc / len(a), 6),
                "right_fraction_in_lcc": round(lcc / len(b), 6),
            })
    return family_rows, overlap_rows, family_by_route


# ---------------------------------------------------------------------------
# CLI and final recording

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Atlas text export")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--audit-size", type=int, default=12,
        help="length-stratified exhaustive audit routes; 0 disables",
    )
    parser.add_argument(
        "--max-audit-span", type=int, default=0,
        help="maximum source-token width in audit; 0 searches all widths",
    )
    parser.add_argument(
        "--only-neuron", type=int, action="append",
        help="restrict to a neuron; repeat for several",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--outdir", default="results", help="final TSV output directory",
    )
    parser.add_argument(
        "--checkpoint-dir", default="results/probe_v9_checkpoint",
        help="resumable per-route checkpoint directory",
    )
    return parser.parse_args()


def validate_args(args):
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.audit_size < 0:
        raise SystemExit("--audit-size cannot be negative")
    if args.max_audit_span < 0:
        raise SystemExit("--max-audit-span cannot be negative")
    if args.limit < 0:
        raise SystemExit("--limit cannot be negative")


def main():
    args = parse_args()
    validate_args(args)
    device = next(zp.mdl.parameters()).device
    if device.type != "cuda":
        raise SystemExit(f"znou_probe model is on {device}; v9 requires CUDA")

    routes = parse_atlas_export(args.input)
    if args.only_neuron:
        wanted = set(args.only_neuron)
        routes = [r for r in routes if r.neuron in wanted]
    if args.limit:
        routes = routes[:args.limit]
    if not routes:
        raise SystemExit("no routes remain after filtering")

    token_counts = {
        r.route_id: len(zp.tok(r.string)["input_ids"]) for r in routes
    }
    if any(n + 1 > zp.N_CTX for n in token_counts.values()):
        outside = [rid for rid, n in token_counts.items() if n + 1 > zp.N_CTX]
        raise SystemExit(f"routes outside dom(D): {', '.join(outside)}")
    audit_ids = select_audit_ids(routes, token_counts, args.audit_size)

    checkpoint_dir = Path(args.checkpoint_dir)
    manifest_path = checkpoint_dir / "manifest.json"
    manifest = {
        "script_version": SCRIPT_VERSION,
        "input_path": str(Path(args.input).resolve()),
        "input_sha256": file_sha256(args.input),
        "route_ids": [r.route_id for r in routes],
        "batch_size": args.batch_size,
        "audit_size": args.audit_size,
        "max_audit_span": args.max_audit_span,
        "audit_ids": sorted(audit_ids),
    }
    old_manifest = read_json(manifest_path)
    if old_manifest is not None and old_manifest != manifest:
        raise SystemExit(
            f"checkpoint manifest mismatch at {manifest_path}; use a new "
            "--checkpoint-dir for changed input or options"
        )
    atomic_json(manifest_path, manifest)

    print(f"model device: {device}")
    print(f"routes: {len(routes)}")
    print(f"audit routes: {len(audit_ids)}")
    print(f"checkpoint: {checkpoint_dir}")

    analyses = []
    scorer = BatchScorer(args.batch_size)
    try:
        for index, route in enumerate(routes, 1):
            audit_tag = " · exhaustive audit" if route.route_id in audit_ids else ""
            print(
                f"[{index}/{len(routes)}] {route.route_id} "
                f"{token_counts[route.route_id]} tokens{audit_tag}"
            )
            analysis = analyse_route(
                route, scorer, route.route_id in audit_ids,
                checkpoint_dir, args.max_audit_span,
            )
            analyses.append(analysis)
            summary = analysis["summary"]
            if not summary["verified"]:
                print(f"  LABEL MISMATCH: measured {summary['full_dest']}")
            else:
                print(
                    f"  TOKEN_MIN {summary.get('token_min_n_tok', '-')} tok · "
                    f"WORD_ANCHORED {summary.get('word_anchored_n_tok', '-')} tok · "
                    f"PARETO {summary.get('pareto_size', '-')}"
                )
    finally:
        scorer.close()

    summaries = [a["summary"] for a in analyses]
    certificates = [r for a in analyses for r in a["certificates"]]
    ladders = [r for a in analyses for r in a["ladder"]]
    extensions = [r for a in analyses for r in a["extensions"]]
    audits = [a["audit"] for a in analyses if a["audit"] is not None]
    families, overlaps, family_by_route = build_family_outputs(summaries)
    for row in summaries:
        row["exact_family_id"] = family_by_route.get(row["route_id"], "")

    outdir = Path(args.outdir)
    outputs = []
    for name, rows in (
        ("probe_v9_summary", summaries),
        ("probe_v9_certificates", certificates),
        ("probe_v9_ladder", ladders),
        ("probe_v9_extensions", extensions),
        ("probe_v9_audit", audits),
        ("probe_v9_exact_families", families),
        ("probe_v9_overlap", overlaps),
    ):
        recorder = zp.Recorder(name, outdir=outdir)
        recorder.extend(rows)
        outputs.append(recorder.write())

    verified = sum(int(r["verified"]) for r in summaries)
    audit_agree = sum(int(r.get("agrees_token_min", 0)) for r in audits)
    recurrent_routes = sum(
        family["count"] for family in families if family["recurrent"]
    )
    print("\nvalidation")
    print(f"  verified destinations  {verified}/{len(routes)}")
    print(f"  audit agreement        {audit_agree}/{len(audits)}")
    print(f"  exact-recurrent routes {recurrent_routes}/{len(routes)}")
    print("\noutputs")
    for path in outputs:
        print(f"  {path}")

    if verified != len(routes):
        raise SystemExit("one or more Atlas labels did not reproduce")
    if any(int(r.get("roundtrip_stable", 1)) != 1 for r in certificates):
        raise SystemExit("one or more selected certificates failed round trip")


if __name__ == "__main__":
    main()
