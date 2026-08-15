"""
probe_v15.py — Melville route extension for GPT-2 Small L5-N38.

V13-v14 found that N38 is strongly context-dependent: it responds broadly in
predicative-property contexts and causally tilts matched next-token ratios
toward `cold`.  Three Melville Atlas routes now provide a held-out observational
test, two without explicit temperature language.

V15 preserves the supplied Unicode strings exactly and records:

  * whole-string Atlas destination and N38 peak
  * N38 activation and local winner at every token position
  * every decoded contiguous source-token span and its score
  * PREFIX, ANCHORED, and CONTIG destination-preserving certificates
  * shortest local-winner and 90%-of-full-peak diagnostic spans
  * separately labelled apostrophe and dash normalization controls

Certificate definitions
-----------------------
PREFIX
    The source prefix ending at the full route's N38 peak token.  This is a
    free right-truncation check rather than a global minimum.

ANCHORED
    The shortest successful contiguous source-token span ending at the full
    route's N38 peak token.

CONTIG
    The shortest successful contiguous source-token span anywhere in the
    supplied source token sequence.

A successful Atlas certificate has destination N38 under the project's
Resonance rule.  Minimality is confined to decoded contiguous spans of the
supplied source tokens; it does not cover arbitrary rewrites or internal
deletions.

Diagnostic spans are not certificates.  LOCAL_WINNER requires N38 to win only
at its own peak token.  PEAK_90 requires a target peak at least 90% of the full
route peak.  Either can retain strong N38 activity while another neuron wins
the whole-string Atlas competition.

Requires the CUDA-corrected znou_probe.py beside this file.

Examples
--------
  python probe_v15.py --smoke
  python probe_v15.py
  python probe_v15.py --batch-size 64 --outdir results
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch

import znou_probe as zp


TARGET = 38
PEAK_FRACTION = 0.90
# The primary route is scored alone, while typography variants are scored in a
# padded batch.  CUDA kernels can differ in the last few floating-point digits
# across those batch shapes.  Discrete identity checks remain exact; only the
# repeated scalar peak comparison receives this tolerance.
REPEAT_PEAK_TOL = 5e-5


@dataclass(frozen=True)
class Route:
    route_id: str
    text: str
    notes: str


ROUTES = (
    Route(
        "MELVILLE_ICELAND",
        "It was cold as Iceland—no fire at all—the landlord said he couldn’t afford it.",
        "explicit cold route; exact Unicode em dashes and curly apostrophe",
    ),
    Route(
        "MELVILLE_CREAKING",
        "We all heard a faint creaking, as of ropes and yards hitherto muffled by the storm.",
        "held-out route without explicit temperature lexeme",
    ),
    Route(
        "MELVILLE_GREY_DAWN",
        "The grey dawn came on, and the slumbering crew arose from the boat’s bottom, and ere noon the dead whale was brought to the ship.",
        "held-out route without explicit temperature lexeme; curly apostrophe preserved",
    ),
)


def decode_token(token_id):
    return zp.tok.decode([int(token_id)])


def json_tokens(ids):
    return json.dumps([decode_token(x) for x in ids], ensure_ascii=False)


def body_ids(text):
    ids = list(zp.tok(text, add_special_tokens=False)["input_ids"])
    if not ids:
        raise ValueError("empty token sequence outside dom(D)")
    if len(ids) + 1 > zp.N_CTX:
        raise ValueError(f"string has {len(ids)+1} tokens outside dom(D)")
    return ids


class BatchScorer:
    """Destination and target trace scorer without allocating LM logits."""

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

    def score(self, strings, include_trace=False):
        out = []
        for offset in range(0, len(strings), self.batch_size):
            batch = strings[offset:offset + self.batch_size]
            bodies = [body_ids(text) for text in batch]
            sequences = [[zp.BOS] + ids for ids in bodies]
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
                target_pos = int(A[:, TARGET].argmax())
                local = A[target_pos]
                local_top2 = torch.topk(local, 2)
                row = {
                    "destination": int(top2.indices[0]),
                    "runner_up": int(top2.indices[1]),
                    "margin": round(float(top2.values[0]-top2.values[1]), 6),
                    "winner_peak": round(float(top2.values[0]), 6),
                    "target_peak": round(float(peaks[TARGET]), 6),
                    "target_peak_pos": target_pos,
                    "target_local_winner": int(local_top2.indices[0] == TARGET),
                    "target_local_rank": int((local > local[TARGET]).sum()) + 1,
                    "target_local_margin": round(float(
                        local[TARGET] - torch.cat(
                            [local[:TARGET], local[TARGET+1:]]
                        ).max()
                    ), 6),
                    "n_tok": length - 1,
                }
                if include_trace:
                    row["A"] = A.detach().cpu()
                    row["body_ids"] = bodies[i]
                out.append(row)
        return out


def realised_span(source_ids, start, end):
    intended = source_ids[start:end]
    text = zp.tok.decode(intended)
    if not text:
        return None
    realised = body_ids(text)
    return {
        "text": text,
        "source_start": start,
        "source_end": end,
        "source_width": end-start,
        "intended_ids": intended,
        "realised_ids": realised,
        "realised_n_tok": len(realised),
        "roundtrip_stable": int(
            list(zp.tok(zp.tok.decode(realised), add_special_tokens=False)[
                "input_ids"
            ]) == realised
        ),
    }


def all_contiguous_spans(source_ids, max_source_span=0):
    n = len(source_ids)
    limit = min(n, max_source_span) if max_source_span else n
    seen = set()
    spans = []
    for width in range(1, limit+1):
        for start in range(0, n-width+1):
            item = realised_span(source_ids, start, start+width)
            if item is None or item["text"] in seen:
                continue
            seen.add(item["text"])
            spans.append(item)
    return spans


def attach_scores(spans, scorer):
    scores = scorer.score([x["text"] for x in spans])
    for span, score in zip(spans, scores):
        span.update(score)
    return spans


def choose_shortest(spans, predicate):
    hits = [x for x in spans if predicate(x)]
    if not hits:
        return None
    return min(hits, key=lambda x: (
        x["realised_n_tok"], x["source_width"], x["source_start"], x["text"]
    ))


def certificate_row(route, method, candidate, full_score):
    base = {
        "route_id": route.route_id,
        "method": method,
        "criterion": {
            "PREFIX": "right truncation ending at full-route target peak",
            "ANCHORED": "shortest destination-38 span ending at full-route target peak",
            "CONTIG": "shortest destination-38 contiguous source-token span",
            "LOCAL_WINNER": "shortest span where N38 wins at its own peak token",
            "PEAK_90": "shortest span reaching at least 90% of full-route N38 peak",
        }[method],
        "full_text": route.text,
        "full_destination": full_score["destination"],
        "full_target_peak": full_score["target_peak"],
        "certificate_found": int(candidate is not None),
    }
    if candidate is None:
        return {**base, "certificate": "", "source_start": "",
                "source_end": "", "source_width": "", "realised_n_tok": "",
                "intended_token_ids": "", "realised_token_ids": "",
                "roundtrip_stable": "", "destination": "", "runner_up": "",
                "margin": "", "target_peak": "", "target_peak_pos": "",
                "target_peak_drift": "", "target_local_winner": "",
                "target_local_rank": "", "target_local_margin": ""}
    return {
        **base,
        "certificate": candidate["text"],
        "source_start": candidate["source_start"],
        "source_end": candidate["source_end"],
        "source_width": candidate["source_width"],
        "realised_n_tok": candidate["realised_n_tok"],
        "intended_token_ids": json.dumps(candidate["intended_ids"]),
        "realised_token_ids": json.dumps(candidate["realised_ids"]),
        "roundtrip_stable": candidate["roundtrip_stable"],
        "destination": candidate["destination"],
        "runner_up": candidate["runner_up"],
        "margin": candidate["margin"],
        "target_peak": candidate["target_peak"],
        "target_peak_pos": candidate["target_peak_pos"],
        "target_peak_drift": round(
            candidate["target_peak"]-full_score["target_peak"], 6
        ),
        "target_local_winner": candidate["target_local_winner"],
        "target_local_rank": candidate["target_local_rank"],
        "target_local_margin": candidate["target_local_margin"],
    }


def route_analysis(route, scorer, max_source_span=0):
    full = scorer.score([route.text], include_trace=True)[0]
    ids = full.pop("body_ids")
    A = full.pop("A")
    trace = []
    sequence = [zp.BOS] + ids
    for pos, token_id in enumerate(sequence):
        local = A[pos]
        top2 = torch.topk(local, 2)
        trace.append({
            "route_id": route.route_id,
            "text": route.text,
            "position": pos,
            "token_id": token_id,
            "token": "<BOS>" if pos == 0 else decode_token(token_id),
            "target_activation": round(float(local[TARGET]), 6),
            "position_winner": int(top2.indices[0]),
            "position_winner_activation": round(float(top2.values[0]), 6),
            "target_local_rank": int((local > local[TARGET]).sum()) + 1,
            "target_local_margin": round(float(
                local[TARGET] - torch.cat(
                    [local[:TARGET], local[TARGET+1:]]
                ).max()
            ), 6),
            "is_target_peak": int(pos == full["target_peak_pos"]),
        })

    spans = attach_scores(
        all_contiguous_spans(ids, max_source_span=max_source_span), scorer
    )
    peak_body_end = full["target_peak_pos"]  # BOS position equals body end index
    prefix = realised_span(ids, 0, peak_body_end)
    if prefix is not None:
        prefix.update(scorer.score([prefix["text"]])[0])
    anchored = choose_shortest(
        spans,
        lambda x: x["source_end"] == peak_body_end and
                  x["destination"] == TARGET,
    )
    contig = choose_shortest(spans, lambda x: x["destination"] == TARGET)
    local_winner = choose_shortest(
        spans, lambda x: bool(x["target_local_winner"])
    )
    peak90 = choose_shortest(
        spans,
        lambda x: x["target_peak"] >= PEAK_FRACTION*full["target_peak"],
    )
    certificates = [
        certificate_row(route, "PREFIX", prefix, full),
        certificate_row(route, "ANCHORED", anchored, full),
        certificate_row(route, "CONTIG", contig, full),
        certificate_row(route, "LOCAL_WINNER", local_winner, full),
        certificate_row(route, "PEAK_90", peak90, full),
    ]
    span_rows = []
    for span in spans:
        span_rows.append({
            "route_id": route.route_id,
            "full_text": route.text,
            "source_start": span["source_start"],
            "source_end": span["source_end"],
            "source_width": span["source_width"],
            "span": span["text"],
            "realised_n_tok": span["realised_n_tok"],
            "intended_token_ids": json.dumps(span["intended_ids"]),
            "realised_token_ids": json.dumps(span["realised_ids"]),
            "roundtrip_stable": span["roundtrip_stable"],
            "destination": span["destination"],
            "target_retained": int(span["destination"] == TARGET),
            "runner_up": span["runner_up"],
            "margin": span["margin"],
            "winner_peak": span["winner_peak"],
            "target_peak": span["target_peak"],
            "target_peak_fraction_of_full": round(
                span["target_peak"]/max(full["target_peak"], 1e-8), 6
            ),
            "target_peak_pos": span["target_peak_pos"],
            "target_local_winner": span["target_local_winner"],
            "target_local_rank": span["target_local_rank"],
            "target_local_margin": span["target_local_margin"],
        })

    route_row = {
        "route_id": route.route_id,
        "text": route.text,
        "notes": route.notes,
        "n_tok": len(ids),
        "tokens": json_tokens(ids),
        "token_ids": json.dumps(ids),
        **full,
        "target_retained": int(full["destination"] == TARGET),
        "n_unique_contiguous_spans": len(spans),
        "n_destination_38_spans": sum(x["destination"] == TARGET for x in spans),
        "n_local_winner_spans": sum(bool(x["target_local_winner"]) for x in spans),
        "n_peak90_spans": sum(
            x["target_peak"] >= PEAK_FRACTION*full["target_peak"] for x in spans
        ),
    }
    return route_row, trace, certificates, span_rows


def variant_specs(route):
    specs = [("ORIGINAL", route.text, "primary exact supplied string")]
    ascii_apostrophe = route.text.replace("’", "'").replace("‘", "'")
    if ascii_apostrophe != route.text:
        specs.append(("ASCII_APOSTROPHE", ascii_apostrophe,
                      "curly apostrophe replaced by ASCII apostrophe"))
    ascii_dash = route.text.replace("—", " -- ")
    if ascii_dash != route.text:
        specs.append(("ASCII_DASH", ascii_dash,
                      "em dash replaced by space-double-hyphen-space"))
    both = ascii_apostrophe.replace("—", " -- ")
    if both not in {text for _, text, _ in specs}:
        specs.append(("ASCII_BOTH", both,
                      "apostrophe and em dash normalized"))
    return specs


def variant_rows(routes, scorer):
    out = []
    for route in routes:
        specs = variant_specs(route)
        scores = scorer.score([text for _, text, _ in specs])
        original = scores[0]
        for (variant, text, note), score in zip(specs, scores):
            out.append({
                "route_id": route.route_id,
                "variant": variant,
                "text": text,
                "notes": note,
                **score,
                "target_retained": int(score["destination"] == TARGET),
                "target_peak_change_from_original": round(
                    score["target_peak"]-original["target_peak"], 6
                ),
                "destination_changed": int(
                    score["destination"] != original["destination"]
                ),
            })
    return out


def validate(routes, route_rows, traces, certificates, spans, variants,
             max_source_span):
    failures = []
    if len({x.route_id for x in routes}) != len(routes):
        failures.append("duplicate route IDs")
    if len(route_rows) != len(routes):
        failures.append("route-row count mismatch")
    for route, row in zip(routes, route_rows):
        if row["destination"] != TARGET:
            failures.append(
                f"{route.route_id}: supplied Atlas route destinations at "
                f"{row['destination']}, not {TARGET}"
            )
        expected_trace = row["n_tok"] + 1
        got_trace = sum(x["route_id"] == route.route_id for x in traces)
        if got_trace != expected_trace:
            failures.append(
                f"{route.route_id}: trace {got_trace} != {expected_trace}"
            )
        route_certs = [x for x in certificates
                       if x["route_id"] == route.route_id]
        if len(route_certs) != 5:
            failures.append(f"{route.route_id}: certificate row count")
        for method in ("PREFIX", "ANCHORED", "CONTIG"):
            cert = next(x for x in route_certs if x["method"] == method)
            if cert["certificate_found"] and cert["destination"] != TARGET:
                failures.append(
                    f"{route.route_id}/{method}: non-target certificate"
                )
        if not max_source_span:
            n = row["n_tok"]
            if row["n_unique_contiguous_spans"] > n*(n+1)//2:
                failures.append(f"{route.route_id}: impossible span count")
    originals = [x for x in variants if x["variant"] == "ORIGINAL"]
    if len(originals) != len(routes):
        failures.append("variant originals missing")
    for row in route_rows:
        original = next(x for x in originals if x["route_id"] == row["route_id"])
        for field in ("destination", "target_peak_pos"):
            if original[field] != row[field]:
                failures.append(
                    f"{row['route_id']}: route/variant mismatch in {field}"
                )
        peak_difference = abs(
            float(original["target_peak"])-float(row["target_peak"])
        )
        if peak_difference > REPEAT_PEAK_TOL:
            failures.append(
                f"{row['route_id']}: route/variant target-peak difference "
                f"{peak_difference:.9g} exceeds tolerance {REPEAT_PEAK_TOL}"
            )
    return failures


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-source-span", type=int, default=0,
                   help="maximum contiguous source width; 0 searches all")
    p.add_argument("--smoke", action="store_true",
                   help="run the first route through the full machinery")
    return p.parse_args()


def main():
    args = parse_args()
    device = next(zp.mdl.parameters()).device
    if device.type != "cuda":
        raise SystemExit(f"znou_probe model is on {device}; v15 requires CUDA")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.max_source_span < 0:
        raise SystemExit("--max-source-span cannot be negative")

    routes = list(ROUTES[:1] if args.smoke else ROUTES)
    print(f"model device: {device}")
    print(f"routes: {len(routes)}")
    for route in routes:
        n = len(body_ids(route.text))
        m = min(n, args.max_source_span) if args.max_source_span else n
        possible = m*(n+1)-m*(m+1)//2
        print(f"  {route.route_id:<24} {n} tokens · up to {possible} spans")

    scorer = BatchScorer(batch_size=args.batch_size)
    route_rows, traces, certificates, spans = [], [], [], []
    try:
        for i, route in enumerate(routes, 1):
            print(f"analyse [{i}/{len(routes)}] {route.route_id}")
            rr, tr, cr, sr = route_analysis(
                route, scorer, max_source_span=args.max_source_span
            )
            route_rows.append(rr)
            traces.extend(tr)
            certificates.extend(cr)
            spans.extend(sr)
        variants = variant_rows(routes, scorer)
    finally:
        scorer.close()

    failures = validate(
        routes, route_rows, traces, certificates, spans, variants,
        args.max_source_span,
    )
    print("\nvalidation")
    print(f"  route rows              {len(route_rows)}")
    print(f"  trace rows              {len(traces)}")
    print(f"  certificate rows        {len(certificates)}")
    print(f"  span rows               {len(spans)}")
    print(f"  variant rows            {len(variants)}")
    print(f"  validation failures     {len(failures)}")

    outdir = Path(args.outdir)
    suffix = "smoke" if args.smoke else ""
    outputs = []
    for name, data in (
        ("probe_v15_routes", route_rows),
        ("probe_v15_trace", traces),
        ("probe_v15_certificates", certificates),
        ("probe_v15_spans", spans),
        ("probe_v15_variants", variants),
    ):
        recorder = zp.Recorder(name, outdir=outdir)
        recorder.extend(data)
        outputs.append(recorder.write(suffix=suffix))
    print("\noutputs")
    for path in outputs:
        print(f"  {path}")
    if failures:
        raise SystemExit("v15 validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
