"""
probe_v7.py — verify and whittle full-sentence Atlas routes.

V7 is the minimisation stage that v6 did not contain. It accepts the text
format exported from the Atlas, verifies every full sentence against its
labelled neuron, and produces three certificates:

  PREFIX     free right-truncation at the winner's peak (§1.1 ratchet)
  ANCHORED   shortest successful source-token span ending at that peak
  CONTIG     shortest successful contiguous source-token span anywhere

ANCHORED and CONTIG require new forward passes. CONTIG is globally minimal
only among decoded contiguous spans of the supplied source token sequence. It
is not a globally minimal string and does not search arbitrary internal
deletions or strings absent from the source.

The built-in neuron-906 fixture checks the full Melville sentence and the
manual certificate:

  the drabbest drab, to a harpooneer in a broad shad-bellied

An automatic certificate passes if it reaches 906 and is no longer in realised
tokens than the manual certificate. The automatic result need not be identical:
a shorter valid span is a stronger result, though peak drift is reported so a
collision can be distinguished from close preservation of the original route.

Requires the CUDA-corrected znou_probe.py beside this file.

Outputs
-------
  results/probe_v7_minimise_*.tsv      one row per supplied route
  results/probe_v7_certificates_*.tsv one row per certificate/method

Examples
--------
  python probe_v7.py --input "candidates for probe v7.txt" --limit 2
  python probe_v7.py --input "candidates for probe v7.txt"
  python probe_v7.py --input routes.txt --only-neuron 508 --batch-size 64
"""

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import torch

import znou_probe as zp


# ---------------------------------------------------------------------------
# Acceptance fixture supplied with the v7 design

FIXTURE_906_FULL = (
    "Rising from a little cabin-boy in short clothes of the drabbest drab, "
    "to a harpooneer in a broad shad-bellied waistcoat; from that becoming "
    "boat-header, chief-mate, and captain, and finally a ship owner; Bildad, "
    "as I hinted before, had concluded his adventurous career by wholly "
    "retiring from active life at the goodly age of sixty, and dedicating "
    "his remaining days to the quiet receiving of his well-earned income."
)

FIXTURE_906_MANUAL = (
    "the drabbest drab, to a harpooneer in a broad shad-bellied"
)


@dataclass(frozen=True)
class Route:
    neuron: int
    string: str
    source: str
    source_index: int
    declared_hits: int | None = None
    manual_certificate: str | None = None


# ---------------------------------------------------------------------------
# Atlas parser

HEADER_RE = re.compile(r"^J5-(\d+)\s+\((\d+)\s+Hits?\)\s*$")


def unwrap_export_quote(line):
    """Remove only a paired ASCII wrapper quote; preserve all inner text."""
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
            raise ValueError(f"line {line_number}: empty input")
        routes.append(Route(
            neuron=current_neuron,
            string=s,
            source=f"atlas:{Path(path).name}",
            source_index=per_neuron_index,
            declared_hits=declared_hits,
        ))

    if not routes:
        raise ValueError(f"no Atlas routes parsed from {path}")

    counts = {}
    declarations = {}
    for route in routes:
        counts[route.neuron] = counts.get(route.neuron, 0) + 1
        declarations[route.neuron] = route.declared_hits
    mismatches = {
        neuron: (counts[neuron], declarations[neuron])
        for neuron in counts
        if declarations[neuron] is not None
        and counts[neuron] != declarations[neuron]
    }
    if mismatches:
        detail = ", ".join(
            f"J5-{j}: parsed {got}, declared {want}"
            for j, (got, want) in sorted(mismatches.items())
        )
        raise ValueError(f"Atlas hit-count mismatch: {detail}")
    return routes


def fixture_route():
    return Route(
        neuron=906,
        string=FIXTURE_906_FULL,
        source="acceptance_fixture_906",
        source_index=1,
        declared_hits=1,
        manual_certificate=FIXTURE_906_MANUAL,
    )


# ---------------------------------------------------------------------------
# Batched string scorer

class BatchScorer:
    """Destination and target-peak readout without forming LM logits."""

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
            batch = strings[offset: offset + self.batch_size]
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
                # Call the transformer, not GPT2LMHeadModel: the probe needs
                # MLP activations and should not allocate vocabulary logits.
                zp.mdl.transformer(
                    input_ids=input_ids, attention_mask=attention_mask
                )
            activations = self.buf["A"]

            for i, length in enumerate(lengths):
                A = activations[i, :length, :]
                peaks = A.max(dim=0).values
                top2 = torch.topk(peaks, 2)
                destination = int(top2.indices[0])
                results.append({
                    "dest": destination,
                    "delta": round(float(top2.values[0] - top2.values[1]), 6),
                    "runner_up": int(top2.indices[1]),
                    "winner_peak": round(float(top2.values[0]), 6),
                    "target_peak": round(float(peaks[target]), 6),
                    "target_peakpos": int(A[:, target].argmax()),
                    "n_tok": length - 1,
                })
        return results


# ---------------------------------------------------------------------------
# Certificate search

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
        "intended_token_ids": intended,
        "realised_token_ids": realised,
        "realised_n_tok": len(realised),
        "roundtrip_stable": int(
            list(zp.tok(zp.tok.decode(realised))["input_ids"]) == realised
        ),
    }


def unique_spans(source_ids, *, end=None, max_span=0):
    """Decoded contiguous source-token spans, deduplicated by string."""
    n = len(source_ids)
    seen = set()
    out = []
    if end is not None:
        starts = range(0, end)
        pairs = ((start, end) for start in starts)
    else:
        limit = min(n, max_span) if max_span else n
        pairs = (
            (start, start + width)
            for width in range(1, limit + 1)
            for start in range(0, n - width + 1)
        )
    for start, stop in pairs:
        candidate = realised_span(source_ids, start, stop)
        if candidate is None or candidate["string"] in seen:
            continue
        seen.add(candidate["string"])
        out.append(candidate)
    return out


def attach_scores(candidates, scorer, target):
    scores = scorer.score([c["string"] for c in candidates], target)
    for candidate, score in zip(candidates, scores):
        candidate.update(score)
    return candidates


def choose_certificate(hits, full_target_peak):
    """Shortest realised certificate; deterministic tie-breaks."""
    if not hits:
        return None
    return min(hits, key=lambda c: (
        c["realised_n_tok"],
        len(c["string"]),
        abs(c["target_peak"] - full_target_peak),
        c["source_start"],
    ))


def search_anchored(source_ids, peak_body_end, scorer, target,
                    full_target_peak):
    candidates = unique_spans(source_ids, end=peak_body_end)
    attach_scores(candidates, scorer, target)
    hits = [c for c in candidates if c["dest"] == target]
    return choose_certificate(hits, full_target_peak), len(candidates), len(hits)


def search_contiguous(source_ids, scorer, target, full_target_peak,
                      max_span=0):
    candidates = unique_spans(source_ids, max_span=max_span)
    attach_scores(candidates, scorer, target)
    hits = [c for c in candidates if c["dest"] == target]
    return choose_certificate(hits, full_target_peak), len(candidates), len(hits)


def certificate_row(route, method, candidate, full_target_peak):
    if candidate is None:
        return {
            "target_neuron": route.neuron,
            "source": route.source,
            "source_index": route.source_index,
            "method": method,
            "found": 0,
        }
    return {
        "target_neuron": route.neuron,
        "source": route.source,
        "source_index": route.source_index,
        "method": method,
        "found": 1,
        "string": candidate["string"],
        "source_start": candidate["source_start"],
        "source_end": candidate["source_end"],
        "source_span_n_tok": candidate["source_span_n_tok"],
        "realised_n_tok": candidate["realised_n_tok"],
        "dest": candidate["dest"],
        "delta": candidate["delta"],
        "runner_up": candidate["runner_up"],
        "target_peak": candidate["target_peak"],
        "target_peakpos": candidate["target_peakpos"],
        "target_peak_drift": round(
            candidate["target_peak"] - full_target_peak, 6
        ),
        "roundtrip_stable": candidate["roundtrip_stable"],
        "realised_token_ids": json.dumps(candidate["realised_token_ids"]),
    }


def prefix_candidate(full_row, source_ids, scorer, target):
    prefix = full_row.get("prefix", "")
    if not prefix:
        return None
    realised = list(zp.tok(prefix)["input_ids"])
    score = scorer.score([prefix], target)[0]
    end = int(full_row["t_star"])  # BOS position t maps to body[:t]
    candidate = {
        "string": prefix,
        "source_start": 0,
        "source_end": end,
        "source_span_n_tok": end,
        "intended_token_ids": source_ids[:end],
        "realised_token_ids": realised,
        "realised_n_tok": len(realised),
        "roundtrip_stable": int(
            list(zp.tok(zp.tok.decode(realised))["input_ids"]) == realised
        ),
    }
    candidate.update(score)
    return candidate


# ---------------------------------------------------------------------------

def minimise_route(route, scorer, max_span=0):
    source_ids = list(zp.tok(route.string)["input_ids"])
    full = zp.measure(route.string, watch=[route.neuron])
    verified = int(full["dest"] == route.neuron)
    result = {
        "target_neuron": route.neuron,
        "source": route.source,
        "source_index": route.source_index,
        "declared_hits": route.declared_hits,
        "full_string": route.string,
        "full_n_tok": len(source_ids),
        "full_dest": full["dest"],
        "full_delta": full["delta"],
        "full_runner_up": full["runner_up"],
        "full_t_star": full["t_star"],
        "full_target_peak": full[f"peak_{route.neuron}"],
        "verified": verified,
    }
    certificates = []

    if not verified:
        result.update(
            prefix_string="", anchored_string="", contiguous_string="",
            status="LABEL_MISMATCH",
        )
        return result, certificates

    full_target_peak = float(full[f"peak_{route.neuron}"])
    prefix = prefix_candidate(full, source_ids, scorer, route.neuron)
    certificates.append(certificate_row(
        route, "PREFIX", prefix, full_target_peak
    ))

    # t_star includes BOS. body[:t_star] ends at the winning body position.
    peak_body_end = int(full["t_star"])
    if peak_body_end <= 0:
        anchored = None
        anchored_evaluated = anchored_hits = 0
    else:
        anchored, anchored_evaluated, anchored_hits = search_anchored(
            source_ids, peak_body_end, scorer, route.neuron, full_target_peak
        )
    certificates.append(certificate_row(
        route, "ANCHORED", anchored, full_target_peak
    ))

    contiguous, contiguous_evaluated, contiguous_hits = search_contiguous(
        source_ids, scorer, route.neuron, full_target_peak, max_span=max_span
    )
    certificates.append(certificate_row(
        route, "CONTIG", contiguous, full_target_peak
    ))

    def value(candidate, key, default=""):
        return candidate[key] if candidate is not None else default

    result.update({
        "prefix_string": value(prefix, "string"),
        "prefix_n_tok": value(prefix, "realised_n_tok"),
        "prefix_delta": value(prefix, "delta"),
        "anchored_string": value(anchored, "string"),
        "anchored_n_tok": value(anchored, "realised_n_tok"),
        "anchored_delta": value(anchored, "delta"),
        "anchored_target_peak": value(anchored, "target_peak"),
        "anchored_peak_drift": (
            round(value(anchored, "target_peak", full_target_peak)
                  - full_target_peak, 6)
            if anchored is not None else ""
        ),
        "anchored_candidates_evaluated": anchored_evaluated,
        "anchored_hits": anchored_hits,
        "contiguous_string": value(contiguous, "string"),
        "contiguous_n_tok": value(contiguous, "realised_n_tok"),
        "contiguous_delta": value(contiguous, "delta"),
        "contiguous_target_peak": value(contiguous, "target_peak"),
        "contiguous_peak_drift": (
            round(value(contiguous, "target_peak", full_target_peak)
                  - full_target_peak, 6)
            if contiguous is not None else ""
        ),
        "contiguous_source_start": value(contiguous, "source_start"),
        "contiguous_source_end": value(contiguous, "source_end"),
        "contiguous_candidates_evaluated": contiguous_evaluated,
        "contiguous_hits": contiguous_hits,
        "contiguous_reduction": (
            round(value(contiguous, "realised_n_tok") / len(source_ids), 6)
            if contiguous is not None else ""
        ),
        "status": "OK" if contiguous is not None else "NO_CONTIG_CERTIFICATE",
    })

    if route.manual_certificate is not None:
        manual_ids = list(zp.tok(route.manual_certificate)["input_ids"])
        manual_score = scorer.score([route.manual_certificate], route.neuron)[0]
        manual_candidate = {
            "string": route.manual_certificate,
            "source_start": "NA",
            "source_end": "NA",
            "source_span_n_tok": "NA",
            "realised_token_ids": manual_ids,
            "realised_n_tok": len(manual_ids),
            "roundtrip_stable": int(
                list(zp.tok(zp.tok.decode(manual_ids))["input_ids"])
                == manual_ids
            ),
            **manual_score,
        }
        certificates.append(certificate_row(
            route, "MANUAL", manual_candidate, full_target_peak
        ))
        fixture_pass = int(
            manual_score["dest"] == route.neuron
            and anchored is not None
            and anchored["dest"] == route.neuron
            and anchored["realised_n_tok"] <= len(manual_ids)
            and contiguous is not None
            and contiguous["dest"] == route.neuron
            and contiguous["realised_n_tok"] <= len(manual_ids)
        )
        result.update({
            "manual_string": route.manual_certificate,
            "manual_n_tok": len(manual_ids),
            "manual_dest": manual_score["dest"],
            "manual_delta": manual_score["delta"],
            "fixture_pass": fixture_pass,
        })

    return result, certificates


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Atlas text export")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--max-source-span", type=int, default=0,
        help="maximum source-token span for CONTIG; 0 searches all spans",
    )
    parser.add_argument(
        "--only-neuron", type=int, action="append",
        help="restrict Atlas routes to this neuron; repeat for several",
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="limit Atlas routes after filtering; fixture is still included",
    )
    parser.add_argument(
        "--no-fixture-906", action="store_true",
        help="omit the built-in neuron-906 acceptance fixture",
    )
    parser.add_argument(
        "--no-strict-fixture", action="store_true",
        help="write a failed 906 fixture instead of aborting",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_source_span < 0:
        raise SystemExit("--max-source-span cannot be negative")

    device = next(zp.mdl.parameters()).device
    if device.type != "cuda":
        raise SystemExit(
            f"znou_probe model is on {device}; v7 requires CUDA"
        )

    routes = parse_atlas_export(args.input)
    if args.only_neuron:
        wanted = set(args.only_neuron)
        routes = [r for r in routes if r.neuron in wanted]
    if args.limit:
        routes = routes[:args.limit]
    atlas_n = len(routes)
    if not args.no_fixture_906:
        routes.append(fixture_route())

    print(f"model device: {device}")
    print(f"routes: {atlas_n} Atlas + {len(routes) - atlas_n} fixture")
    print("CONTIG search:",
          "all source spans" if args.max_source_span == 0 else
          f"source spans up to {args.max_source_span} tokens")

    minimise_rec = zp.Recorder("probe_v7_minimise")
    certificate_rec = zp.Recorder("probe_v7_certificates")
    scorer = BatchScorer(args.batch_size)
    fixture_failed = False
    try:
        for index, route in enumerate(routes, 1):
            n_tok = len(zp.tok(route.string)["input_ids"])
            n_spans = n_tok * (n_tok + 1) // 2
            if args.max_source_span:
                m = min(n_tok, args.max_source_span)
                n_spans = m * (n_tok + 1) - m * (m + 1) // 2
            print(
                f"[{index}/{len(routes)}] J5-{route.neuron} "
                f"{n_tok} tokens, up to {n_spans} contiguous spans"
            )
            result, certificates = minimise_route(
                route, scorer, max_span=args.max_source_span
            )
            minimise_rec.add(result)
            certificate_rec.extend(certificates)
            if result["verified"]:
                print(
                    f"  prefix {result.get('prefix_n_tok', '-')} tok · "
                    f"anchored {result.get('anchored_n_tok', '-')} tok · "
                    f"contig {result.get('contiguous_n_tok', '-')} tok"
                )
            else:
                print(
                    f"  LABEL MISMATCH: measured {result['full_dest']}"
                )
            if route.manual_certificate is not None:
                fixture_failed = result.get("fixture_pass") != 1
                print(
                    f"  906 fixture: {'PASS' if not fixture_failed else 'FAIL'}"
                )
    finally:
        scorer.close()

    minimise_path = minimise_rec.write()
    certificate_path = certificate_rec.write()
    print(f"minimise output: {minimise_path}")
    print(f"certificate output: {certificate_path}")

    if fixture_failed and not args.no_strict_fixture:
        raise SystemExit("neuron-906 acceptance fixture failed")


if __name__ == "__main__":
    main()
