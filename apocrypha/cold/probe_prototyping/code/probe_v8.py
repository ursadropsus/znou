"""
probe_v8.py — neuron-906 target-versus-rival micro-probe.

V7 found that the token-minimal certificate for a Melville route to neuron 906
had a lower target peak but a wider winning margin than the manual word-bounded
certificate. V8 measures the arithmetic and position-level mechanism directly.

Variants
--------
  FULL     complete Melville sentence
  PREFIX   free right-truncation at the full sentence's target peak
  MANUAL   manually whittled word-bounded certificate
  AUTO     v7 token-minimal, peak-anchored certificate

Measurements
------------
  probe_v8_variants_*.tsv
      final destination, target and runner peaks, peak positions/tokens,
      top-eight final peak ranking

  probe_v8_comparisons_*.tsv
      exact target change, competitor-envelope change, margin change, runner
      identity switch, and arithmetic reconciliation for declared pairs

  probe_v8_trace_*.tsv
      one row per token position for all variants; raw and running activations
      for neuron 906 and every runner observed in the four variants

  probe_v8_ladder_*.tsv
      every left trim of the full sentence's peak-ending prefix, from the full
      prefix to the shortest possible suffix; target margin is signed even
      after 906 stops winning

Requires the CUDA-corrected znou_probe.py beside this file.
"""

import json

import torch

import znou_probe as zp


TARGET = 906

FULL = (
    "Rising from a little cabin-boy in short clothes of the drabbest drab, "
    "to a harpooneer in a broad shad-bellied waistcoat; from that becoming "
    "boat-header, chief-mate, and captain, and finally a ship owner; Bildad, "
    "as I hinted before, had concluded his adventurous career by wholly "
    "retiring from active life at the goodly age of sixty, and dedicating "
    "his remaining days to the quiet receiving of his well-earned income."
)

MANUAL = "the drabbest drab, to a harpooneer in a broad shad-bellied"
AUTO = "er in a broad shad"


def token_at(ids_with_bos, position):
    if position == 0:
        return "<BOS>"
    return zp.tok.decode([ids_with_bos[position]])


def top_ranking(s, k=8):
    ids, A = zp.activations(s, bos=True)
    peaks = A.max(dim=0).values
    values, indices = torch.topk(peaks, k)
    return [
        {
            "rank": rank,
            "neuron": int(neuron),
            "peak": round(float(value), 6),
            "peakpos": int(A[:, int(neuron)].argmax()),
            "token": token_at(ids, int(A[:, int(neuron)].argmax())),
        }
        for rank, (neuron, value) in enumerate(zip(indices, values), 1)
    ]


def measure_variants():
    # First pass discovers the prefix and runner identities. The second pass
    # watches the union, allowing each runner to be evaluated in every string.
    full_first = zp.measure(FULL)
    if int(full_first["dest"]) != TARGET:
        raise RuntimeError(
            f"FULL expected {TARGET}, measured {full_first['dest']}"
        )
    prefix = full_first["prefix"]
    variants = {
        "FULL": FULL,
        "PREFIX": prefix,
        "MANUAL": MANUAL,
        "AUTO": AUTO,
    }
    first = {name: zp.measure(s) for name, s in variants.items()}
    runners = sorted({int(row["runner_up"]) for row in first.values()})
    watch = [TARGET] + [j for j in runners if j != TARGET]
    measured = {
        name: zp.measure(s, watch=watch, variant=name)
        for name, s in variants.items()
    }
    return variants, measured, watch


def variant_summary(name, s, row, watch):
    ids, _ = zp.activations(s, bos=True)
    runner = int(row["runner_up"])
    target_pos = int(row[f"peakpos_{TARGET}"])
    runner_pos = int(row[f"peakpos_{runner}"])
    result = {
        "variant": name,
        "string": s,
        "n_tok": row["n_tok"],
        "dest": row["dest"],
        "delta": row["delta"],
        "target_neuron": TARGET,
        "target_peak": row[f"peak_{TARGET}"],
        "target_peakpos": target_pos,
        "target_peak_token": token_at(ids, target_pos),
        "runner_up": runner,
        "runner_peak": row[f"peak_{runner}"],
        "runner_peakpos": runner_pos,
        "runner_peak_token": token_at(ids, runner_pos),
        "top8": json.dumps(top_ranking(s), ensure_ascii=False),
    }
    for neuron in watch:
        result[f"peak_{neuron}"] = row[f"peak_{neuron}"]
        result[f"peakpos_{neuron}"] = row[f"peakpos_{neuron}"]
    return result


def competitor_envelope(row):
    """Best non-target peak and its neuron, whether or not target wins."""
    dest = int(row["dest"])
    if dest == TARGET:
        return (
            int(row["runner_up"]),
            round(float(row["peak_val"]) - float(row["delta"]), 6),
        )
    return dest, float(row["peak_val"])


def comparison_row(label, baseline_name, reduced_name, measured):
    baseline = measured[baseline_name]
    reduced = measured[reduced_name]
    baseline_other_j, baseline_other = competitor_envelope(baseline)
    reduced_other_j, reduced_other = competitor_envelope(reduced)
    baseline_target = float(baseline[f"peak_{TARGET}"])
    reduced_target = float(reduced[f"peak_{TARGET}"])
    baseline_delta = baseline_target - baseline_other
    reduced_delta = reduced_target - reduced_other
    target_change = reduced_target - baseline_target
    other_change = reduced_other - baseline_other
    delta_change = reduced_delta - baseline_delta
    reconciled = target_change - other_change

    baseline_runner = int(baseline["runner_up"])
    reduced_runner = int(reduced["runner_up"])
    return {
        "comparison": label,
        "baseline": baseline_name,
        "reduced": reduced_name,
        "baseline_string": baseline["string"],
        "reduced_string": reduced["string"],
        "baseline_target_peak": baseline_target,
        "reduced_target_peak": reduced_target,
        "target_peak_change": round(target_change, 6),
        "baseline_best_other": baseline_other_j,
        "baseline_best_other_peak": baseline_other,
        "reduced_best_other": reduced_other_j,
        "reduced_best_other_peak": reduced_other,
        "best_other_peak_change": round(other_change, 6),
        "baseline_margin_target": round(baseline_delta, 6),
        "reduced_margin_target": round(reduced_delta, 6),
        "margin_change": round(delta_change, 6),
        "target_change_minus_other_change": round(reconciled, 6),
        "arithmetic_error": round(delta_change - reconciled, 9),
        "baseline_runner_up": baseline_runner,
        "reduced_runner_up": reduced_runner,
        "runner_switched": int(baseline_runner != reduced_runner),
        "baseline_runner_peak_in_baseline": baseline[f"peak_{baseline_runner}"],
        "baseline_runner_peak_in_reduced": reduced[f"peak_{baseline_runner}"],
        "reduced_runner_peak_in_baseline": baseline[f"peak_{reduced_runner}"],
        "reduced_runner_peak_in_reduced": reduced[f"peak_{reduced_runner}"],
    }


def trace_variants(variants, watch):
    rows = []
    for name, s in variants.items():
        rows.extend(zp.trace_rows(
            s, watch=watch, k=5, variant=name,
            target_neuron=TARGET,
        ))
    return rows


def anchored_ladder(full_row, watch):
    body = list(zp.tok(FULL)["input_ids"])
    end = int(full_row["t_star"])
    if end <= 0:
        raise RuntimeError("906 full-sentence peak is the BOS footprint")
    full_target_peak = float(full_row[f"peak_{TARGET}"])
    rows = []
    for start in range(0, end):
        s = zp.tok.decode(body[start:end])
        row = zp.measure(
            s, watch=watch, source_start=start, source_end=end,
            removed_left_tokens=start,
        )
        other_j, other_peak = competitor_envelope(row)
        target_peak = float(row[f"peak_{TARGET}"])
        signed_margin = target_peak - other_peak
        ids, _ = zp.activations(s, bos=True)
        target_pos = int(row[f"peakpos_{TARGET}"])
        rows.append({
            "source_start": start,
            "source_end": end,
            "removed_left_tokens": start,
            "string": s,
            "n_tok": row["n_tok"],
            "dest": row["dest"],
            "target_wins": int(int(row["dest"]) == TARGET),
            "target_peak": target_peak,
            "target_peakpos": target_pos,
            "target_peak_token": token_at(ids, target_pos),
            "target_peak_drift": round(target_peak - full_target_peak, 6),
            "best_other": other_j,
            "best_other_peak": other_peak,
            "signed_target_margin": round(signed_margin, 6),
            "reported_delta": row["delta"],
            "runner_up": row["runner_up"],
            "matches_v7_auto": int(s == AUTO),
            **{
                f"peak_{j}": row[f"peak_{j}"]
                for j in watch
            },
            **{
                f"peakpos_{j}": row[f"peakpos_{j}"]
                for j in watch
            },
        })
    return rows


def validate(variants, measured, ladder):
    failures = []
    for name in ("FULL", "PREFIX", "MANUAL", "AUTO"):
        if int(measured[name]["dest"]) != TARGET:
            failures.append(
                f"{name}: expected {TARGET}, measured {measured[name]['dest']}"
            )
    matches = [row for row in ladder if row["matches_v7_auto"]]
    if len(matches) != 1:
        failures.append(
            f"anchored ladder contains {len(matches)} exact AUTO matches"
        )
    winners = [row for row in ladder if row["target_wins"]]
    selected = min(
        winners,
        key=lambda row: (
            int(row["n_tok"]), len(row["string"]),
            abs(float(row["target_peak_drift"])), int(row["source_start"]),
        ),
    )
    if selected["string"] != AUTO:
        failures.append(
            f"shortest anchored winner is {selected['string']!r}, not AUTO"
        )
    return failures


def main():
    device = next(zp.mdl.parameters()).device
    if device.type != "cuda":
        raise SystemExit(f"znou_probe model is on {device}; v8 requires CUDA")

    variants, measured, watch = measure_variants()
    variant_rows = [
        variant_summary(name, s, measured[name], watch)
        for name, s in variants.items()
    ]
    comparison_rows = [
        comparison_row("FULL_TO_PREFIX", "FULL", "PREFIX", measured),
        comparison_row("FULL_TO_MANUAL", "FULL", "MANUAL", measured),
        comparison_row("PREFIX_TO_AUTO", "PREFIX", "AUTO", measured),
        comparison_row("MANUAL_TO_AUTO", "MANUAL", "AUTO", measured),
    ]
    trace_rows = trace_variants(variants, watch)
    ladder_rows = anchored_ladder(measured["FULL"], watch)
    failures = validate(variants, measured, ladder_rows)

    variant_rec = zp.Recorder("probe_v8_variants")
    comparison_rec = zp.Recorder("probe_v8_comparisons")
    trace_rec = zp.Recorder("probe_v8_trace")
    ladder_rec = zp.Recorder("probe_v8_ladder")
    variant_rec.extend(variant_rows)
    comparison_rec.extend(comparison_rows)
    trace_rec.extend(trace_rows)
    ladder_rec.extend(ladder_rows)

    variant_path = variant_rec.write()
    comparison_path = comparison_rec.write()
    trace_path = trace_rec.write()
    ladder_path = ladder_rec.write()

    print(f"model device: {device}")
    print("watched neurons:", " ".join(str(j) for j in watch))
    for row in variant_rows:
        print(
            f"{row['variant']:<7} dest {row['dest']}  "
            f"target {float(row['target_peak']):.6f}  "
            f"runner {row['runner_up']}:{float(row['runner_peak']):.6f}  "
            f"delta {float(row['delta']):.6f}"
        )
    manual_auto = next(
        row for row in comparison_rows
        if row["comparison"] == "MANUAL_TO_AUTO"
    )
    print("\nMANUAL -> AUTO")
    print(f"  target change      {manual_auto['target_peak_change']:+.6f}")
    print(f"  best-other change  {manual_auto['best_other_peak_change']:+.6f}")
    print(f"  margin change      {manual_auto['margin_change']:+.6f}")
    print(f"  runner switched    {manual_auto['runner_switched']}")
    print("\noutputs:")
    for path in (variant_path, comparison_path, trace_path, ladder_path):
        print(f"  {path}")

    if failures:
        raise SystemExit("v8 validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
