"""
probe_v14.py — absolute-value steering repair for GPT-2 Small L5-N38.

V13 found a consistent cold-over-control steering advantage, but its
“natural-range” values were used as increments: ADD 2.87 on top of a clean
activation of 0.68–2.02 produces a final value of 3.55–4.89, beyond the
observed natural maximum of 3.43.  V14 directly repairs that calibration issue.

It uses the same twelve paired pre-adjective cases and compares:

  * CLEAN, deterministic repeat, and exact sham
  * absolute SET values at 0, LOW, MED, HIGH, and MAX
  * the v13 ADD ladder at LOW, MED, HIGH, and MAX as a bridge
  * declared-position and previous-position SET-HIGH controls

LOW/MED/HIGH/MAX are recomputed from the clean declared-position activations
of the same eighteen completed-predicate calibration cases used by v13.  The
primary result is the paired cold-minus-control change within identical
prefixes.  No outcome is defined using N38's own activation or Atlas
destination.

Requires the CUDA-corrected znou_probe.py beside this file.

Examples
--------
  python probe_v14.py --smoke
  python probe_v14.py
  python probe_v14.py --outdir results
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch

import znou_probe as zp


TARGET = 38
COMPANIONS = (1103, 2094)
COLD_CONSTELLATION = (1508, 234)
DECLARED_EXCLUSIONS = {38, 1103, 2094, 1508, 234, 1888, 2073, 2566}
TOP_N = 10
SHAM_TOL = 1e-7
EPS = 1e-8


@dataclass(frozen=True)
class Case:
    case_id: str
    branch: str
    prefix: str
    outcome: str
    family: str
    notes: str = ""

    @property
    def text(self):
        return self.prefix + self.outcome


# Completed predicates.  The intervention is at the final token in prefix and
# the outcome is scored by teacher forcing.  These are declared before seeing
# N38 activations; low/non-active members remain informative controls.
NATURAL_CASES = (
    Case("N_IT_WAS_COLD", "NATURAL_ABLATION", "It was cold", " and windy",
         "COLD_EXACT", "recovered N38 Atlas route"),
    Case("N_IT_IS_COLD", "NATURAL_ABLATION", "It is cold", " outside",
         "COPULA_TENSE"),
    Case("N_IT_FELT_COLD", "NATURAL_ABLATION", "It felt cold", " to the touch",
         "SENSATION"),
    Case("N_AIR_WAS_COLD", "NATURAL_ABLATION", "The air was cold", " and still",
         "PHYSICAL"),
    Case("N_ROOM_WAS_COLD", "NATURAL_ABLATION", "The room was cold", " and dark",
         "PHYSICAL"),
    Case("N_SKIN_FELT_COLD", "NATURAL_ABLATION", "Her skin felt cold", " and clammy",
         "BODILY"),
    Case("N_VOICE_WAS_COLD", "NATURAL_ABLATION", "His voice was cold", " and flat",
         "METAPHORICAL"),
    Case("N_DECISION_WAS_COLD", "NATURAL_ABLATION", "It was a cold decision", ".",
         "METAPHORICAL"),
    Case("N_CAUGHT_A_COLD", "NATURAL_ABLATION", "He caught a cold", " last winter",
         "NOUN_ILLNESS"),
    Case("N_COLD_WAR", "NATURAL_ABLATION", "The Cold War", " ended",
         "PROPER_NAME"),
    Case("N_COLD_START", "NATURAL_ABLATION", "The engine made a cold start", ".",
         "ATTRIBUTIVE"),
    Case("N_IT_WAS_FRIGID", "NATURAL_ABLATION", "It was frigid", " outside",
         "SYNONYM"),
    Case("N_IT_WAS_FREEZING", "NATURAL_ABLATION", "It was freezing", " outside",
         "SYNONYM"),
    Case("N_IT_WAS_ICY", "NATURAL_ABLATION", "It was icy", " underfoot",
         "SYNONYM"),
    Case("N_IT_WAS_CHILLY", "NATURAL_ABLATION", "It was chilly", " that morning",
         "SYNONYM"),
    Case("N_IT_WAS_COOL", "NATURAL_ABLATION", "It was cool", " in the shade",
         "TEMPERATURE_NEIGHBOUR"),
    Case("N_IT_WAS_WARM", "NATURAL_ABLATION", "It was warm", " in the sun",
         "ANTONYM"),
    Case("N_IT_WAS_HOT", "NATURAL_ABLATION", "It was hot", " in the room",
         "ANTONYM"),
)


# Prefix ends immediately before the adjective.  The intervention position is
# the last realised prefix token, which predicts the first outcome token.
STEERING_CASES = (
    Case("P_IT_WAS_COLD", "PRE_ADJECTIVE_STEERING", "It was", " cold",
         "COLD_TARGET"),
    Case("P_IT_WAS_WARM", "PRE_ADJECTIVE_STEERING", "It was", " warm",
         "ANTONYM_TARGET"),
    Case("P_AIR_IS_COLD", "PRE_ADJECTIVE_STEERING", "The air is", " cold",
         "COLD_TARGET"),
    Case("P_AIR_IS_HOT", "PRE_ADJECTIVE_STEERING", "The air is", " hot",
         "ANTONYM_TARGET"),
    Case("P_ROOM_WAS_COLD", "PRE_ADJECTIVE_STEERING", "The room was", " cold",
         "COLD_TARGET"),
    Case("P_ROOM_WAS_DARK", "PRE_ADJECTIVE_STEERING", "The room was", " dark",
         "ORTHOGONAL_TARGET"),
    Case("P_SKIN_FELT_COLD", "PRE_ADJECTIVE_STEERING", "Her skin felt", " cold",
         "COLD_TARGET"),
    Case("P_VOICE_WAS_COLD", "PRE_ADJECTIVE_STEERING", "His voice was", " cold",
         "METAPHORICAL_TARGET"),
    Case("P_SEASON_IS_COLD", "PRE_ADJECTIVE_STEERING", "The season is", " cold",
         "COLD_TARGET"),
    Case("P_SEASON_IS_WARM", "PRE_ADJECTIVE_STEERING", "The season is", " warm",
         "ANTONYM_TARGET"),
    Case("P_WORLD_IS_COLD", "PRE_ADJECTIVE_STEERING", "The world is", " cold",
         "BROAD_TARGET"),
    Case("P_WORLD_IS_STRANGE", "PRE_ADJECTIVE_STEERING", "The world is", " strange",
         "ORTHOGONAL_TARGET"),
)


# Clean traces and certificate searches use the completed prefix, without the
# continuation.  This keeps `It was cold` directly comparable to the old route.
PHENOTYPE_STRINGS = tuple(dict.fromkeys(c.prefix for c in NATURAL_CASES)) + (
    "cold", " cold", "a cold", "cold shoulder", "cold and windy",
    "warm", "hot", "temperature", "The first line of the novel is",
)


GENERATION_PROMPTS = (
    "The desert sun is beating down, the air is",
    "The summer sun is fiery, the season is",
    "The fire in the hearth is roaring, the air is",
    "The fever was raging, and her skin felt",
    "It was cold",
    "It was frigid",
    "He reached out for something and it felt",
    "The first line of the novel is",
)


@dataclass(frozen=True)
class Located:
    body_ids: tuple[int, ...]
    offsets: tuple[tuple[int, int], ...]
    intervention_pos: int
    previous_pos: int
    outcome_positions: tuple[int, ...]
    outcome_ids: tuple[int, ...]


@dataclass(frozen=True)
class Treatment:
    name: str
    neurons: tuple[int, ...]
    position_mode: str
    value_mode: str
    value: float | None = None
    role: str = ""


def decode_token(token_id):
    return zp.tok.decode([int(token_id)])


def json_tokens(ids):
    return json.dumps([decode_token(x) for x in ids], ensure_ascii=False)


def locate(case):
    encoded = zp.tok(case.text, add_special_tokens=False,
                     return_offsets_mapping=True)
    ids = tuple(int(x) for x in encoded["input_ids"])
    offsets = tuple(tuple(int(y) for y in x)
                    for x in encoded["offset_mapping"])
    boundary = len(case.prefix)
    crossing = [i for i, (a, b) in enumerate(offsets) if a < boundary < b]
    if crossing:
        raise ValueError(f"{case.case_id}: token crosses prefix boundary")
    prefix_body = [i for i, (_, b) in enumerate(offsets) if b <= boundary]
    outcome_body = [i for i, (a, b) in enumerate(offsets)
                    if a >= boundary and b > boundary]
    if not prefix_body or not outcome_body:
        raise ValueError(f"{case.case_id}: empty prefix or outcome")
    intervention_pos = max(prefix_body) + 1  # explicit BOS offset
    previous_pos = intervention_pos - 1
    outcome_positions = tuple(i + 1 for i in outcome_body)
    if outcome_positions[0] - 1 != intervention_pos:
        raise ValueError(f"{case.case_id}: predictor/intervention misalignment")
    return Located(
        ids, offsets, intervention_pos, previous_pos,
        outcome_positions, tuple(ids[i] for i in outcome_body),
    )


class ForwardRunner:
    """Deterministic teacher-forced pass with an optional multi-neuron hook."""

    def __init__(self):
        self.device = next(zp.mdl.parameters()).device

    def run(self, case, loc, neurons=(), position=None, mode="CLEAN", value=None):
        input_ids = torch.tensor([[zp.BOS] + list(loc.body_ids)],
                                 dtype=torch.long, device=self.device)
        captured = {}

        def hook(module, inputs, output):
            captured["original"] = output.detach().clone()
            changed = output.clone()
            if neurons:
                if not 0 <= position < output.shape[1]:
                    raise IndexError(f"{case.case_id}: bad position {position}")
                for neuron in neurons:
                    old = float(output[0, position, neuron])
                    if mode == "SET":
                        new = float(value)
                    elif mode == "ADD":
                        new = old + float(value)
                    elif mode == "SCALE":
                        new = old * float(value)
                    elif mode == "SHAM":
                        new = old
                    else:
                        raise ValueError(f"unknown hook mode {mode}")
                    changed[0, position, neuron] = new
            captured["modified"] = changed.detach().clone()
            return changed if neurons else None

        handle = zp.mdl.transformer.h[zp.ELL].mlp.act.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = zp.mdl(input_ids=input_ids, output_hidden_states=True,
                             use_cache=False, return_dict=True)
        finally:
            handle.remove()
        return {
            "logits": out.logits[0].detach().clone(),
            "hidden_l5": out.hidden_states[zp.ELL + 1][0].detach().clone(),
            "hidden_final": out.hidden_states[-1][0].detach().clone(),
            "act_original": captured["original"][0],
            "act_modified": captured["modified"][0],
        }


def outcome_metrics(run, loc):
    total = 0.0
    rows = []
    for pos, token_id in zip(loc.outcome_positions, loc.outcome_ids):
        pred = pos - 1
        logits = run["logits"][pred]
        log_probs = torch.log_softmax(logits, dim=-1)
        lp = float(log_probs[token_id])
        value = logits[token_id]
        rank = int((logits > value).sum()) + 1
        total += lp
        rows.append({"outcome_pos": pos, "predictor_pos": pred,
                     "token_id": int(token_id), "token": decode_token(token_id),
                     "logprob": lp, "prob": math.exp(lp), "rank": rank})
    first_pred = loc.outcome_positions[0] - 1
    logits = run["logits"][first_pred]
    probs = torch.softmax(logits, dim=-1)
    greedy = int(logits.argmax())
    return {
        "outcome_logprob": total,
        "first_prob": rows[0]["prob"],
        "first_rank": rows[0]["rank"],
        "greedy_id": greedy,
        "greedy_token": decode_token(greedy),
        "entropy": float(-(probs * torch.log(probs.clamp_min(EPS))).sum()),
        "tokens": rows,
    }


def vector_change(clean, changed):
    delta = changed - clean
    l2 = float(torch.linalg.vector_norm(delta))
    clean_norm = float(torch.linalg.vector_norm(clean))
    cosine = float(torch.nn.functional.cosine_similarity(
        clean.unsqueeze(0), changed.unsqueeze(0), dim=-1)[0])
    return l2, l2 / max(clean_norm, EPS), cosine


def top_token_changes(clean_logits, changed_logits, n=TOP_N):
    delta = changed_logits - clean_logits
    up = torch.topk(delta, n)
    down = torch.topk(-delta, n)
    return (
        [{"token_id": int(i), "token": decode_token(i),
          "delta_logit": round(float(v), 6)}
         for i, v in zip(up.indices, up.values)],
        [{"token_id": int(i), "token": decode_token(i),
          "delta_logit": round(-float(v), 6)}
         for i, v in zip(down.indices, down.values)],
    )


def cproj_directions():
    weight = zp.mdl.transformer.h[zp.ELL].mlp.c_proj.weight.detach()
    # HF GPT-2 uses Conv1D rather than nn.Linear here, so `out_features` is
    # unavailable.  The MLP width is the larger c_proj dimension (3072 for
    # GPT-2 Small); the other dimension is d_model (768).
    width = max(weight.shape)
    if weight.shape[0] == width:
        return weight
    if weight.shape[1] == width:
        return weight.T
    raise RuntimeError(f"cannot orient c_proj weight {tuple(weight.shape)}")


def matched_controls(clean_runs, cases, located, n=3):
    directions = cproj_directions()
    norms = torch.linalg.vector_norm(directions.float(), dim=1).cpu()
    acts = torch.stack([
        clean_runs[c.case_id]["act_original"][located[c.case_id].intervention_pos]
        .abs().cpu() for c in cases
    ]).mean(dim=0)
    target_norm, target_act = float(norms[TARGET]), float(acts[TARGET])
    candidates = []
    for neuron in range(len(norms)):
        if neuron in DECLARED_EXCLUSIONS:
            continue
        distance = abs(math.log((float(norms[neuron]) + EPS) /
                                (target_norm + EPS))) + abs(math.log(
            (float(acts[neuron]) + EPS) / (target_act + EPS)))
        candidates.append((distance, neuron))
    candidates.sort()
    selected = [n_id for _, n_id in candidates[:n]]
    rows = []
    for distance, neuron in candidates[:n]:
        rows.append({
            "selection_type": "MATCHED_CONTROL", "neuron": neuron,
            "distance": round(distance, 6),
            "output_direction_norm": round(float(norms[neuron]), 6),
            "mean_abs_natural_activation": round(float(acts[neuron]), 6),
            "target_output_direction_norm": round(target_norm, 6),
            "target_mean_abs_natural_activation": round(target_act, 6),
            "selection_rule": "nearest log-ratio distance on c_proj norm and mean absolute declared-position activation",
        })
    for neuron, label in ((*[(n, "DECLARED_COMPANION") for n in COMPANIONS],
                           *[(n, "COLD_CONSTELLATION") for n in COLD_CONSTELLATION])):
        rows.append({"selection_type": label, "neuron": neuron,
                     "output_direction_norm": round(float(norms[neuron]), 6),
                     "mean_abs_natural_activation": round(float(acts[neuron]), 6),
                     "target_output_direction_norm": round(target_norm, 6),
                     "target_mean_abs_natural_activation": round(target_act, 6),
                     "selection_rule": "declared from recovered pre-v13 experiments"})
    return selected, rows


def natural_doses(clean_runs, cases, located):
    positive = sorted(max(0.0, float(clean_runs[c.case_id]["act_original"][
        located[c.case_id].intervention_pos, TARGET])) for c in cases)
    positive = [x for x in positive if x > 0]
    if not positive:
        raise RuntimeError("no positive natural N38 activation for dose calibration")
    # Inclusive nearest-rank quantiles, deterministic and dependency-free.
    def q(frac):
        return positive[round((len(positive) - 1) * frac)]
    return {"LOW": q(.25), "MED": q(.50), "HIGH": q(.75), "MAX": positive[-1]}


def natural_treatments(controls):
    out = [
        Treatment("CLEAN", (), "DECLARED", "CLEAN", role="baseline"),
        Treatment("CLEAN_REPEAT", (), "DECLARED", "RERUN", role="deterministic duplicate"),
        Treatment("SHAM_38", (TARGET,), "DECLARED", "SHAM", role="hook identity"),
        Treatment("SCALE_075_38", (TARGET,), "DECLARED", "SCALE", .75, "dose reduction"),
        Treatment("SCALE_050_38", (TARGET,), "DECLARED", "SCALE", .50, "dose reduction"),
        Treatment("SCALE_025_38", (TARGET,), "DECLARED", "SCALE", .25, "dose reduction"),
        Treatment("ZERO_38", (TARGET,), "DECLARED", "SET", 0.0, "target ablation"),
        Treatment("PREVIOUS_ZERO_38", (TARGET,), "PREVIOUS", "SET", 0.0, "position control"),
        Treatment("ZERO_1103", (1103,), "DECLARED", "SET", 0.0, "declared companion ablation"),
        Treatment("ZERO_2094", (2094,), "DECLARED", "SET", 0.0, "declared companion ablation"),
        Treatment("ZERO_TEAM_38_1103_2094", (38, 1103, 2094), "DECLARED", "SET", 0.0,
                  "recovered specialist-team ablation"),
        Treatment("ZERO_1508", (1508,), "DECLARED", "SET", 0.0, "old cold-constellation control"),
        Treatment("ZERO_234", (234,), "DECLARED", "SET", 0.0, "old cold-constellation control"),
    ]
    for i, neuron in enumerate(controls, 1):
        out.append(Treatment(f"ZERO_MATCHED_{i}_{neuron}", (neuron,),
                             "DECLARED", "SET", 0.0,
                             "matched-neuron ablation control"))
    return out


def steering_treatments(doses):
    return [
        Treatment("CLEAN", (), "DECLARED", "CLEAN", role="baseline"),
        Treatment("CLEAN_REPEAT", (), "DECLARED", "RERUN", role="deterministic duplicate"),
        Treatment("SHAM_38", (TARGET,), "DECLARED", "SHAM", role="hook identity"),
        Treatment("SET_0_38", (TARGET,), "DECLARED", "SET", 0.0,
                  "absolute zero anchor"),
        Treatment("SET_LOW_38", (TARGET,), "DECLARED", "SET", doses["LOW"],
                  "absolute natural-distribution quartile"),
        Treatment("SET_MED_38", (TARGET,), "DECLARED", "SET", doses["MED"],
                  "absolute natural-distribution median"),
        Treatment("SET_HIGH_38", (TARGET,), "DECLARED", "SET", doses["HIGH"],
                  "absolute natural-distribution quartile"),
        Treatment("SET_MAX_38", (TARGET,), "DECLARED", "SET", doses["MAX"],
                  "absolute observed natural maximum"),
        Treatment("PREVIOUS_SET_HIGH_38", (TARGET,), "PREVIOUS", "SET", doses["HIGH"],
                  "absolute-value position control"),
        Treatment("ADD_POS_LOW_38", (TARGET,), "DECLARED", "ADD", doses["LOW"],
                  "v13 additive bridge; final value may exceed natural maximum"),
        Treatment("ADD_POS_MED_38", (TARGET,), "DECLARED", "ADD", doses["MED"],
                  "v13 additive bridge; final value may exceed natural maximum"),
        Treatment("ADD_POS_HIGH_38", (TARGET,), "DECLARED", "ADD", doses["HIGH"],
                  "v13 additive bridge; final value exceeds natural maximum"),
        Treatment("ADD_POS_MAX_38", (TARGET,), "DECLARED", "ADD", doses["MAX"],
                  "v13 additive bridge; final value exceeds natural maximum"),
    ]


def intervention_row(case, loc, treatment, run, clean):
    clean_m = outcome_metrics(clean, loc)
    m = outcome_metrics(run, loc)
    pred = loc.outcome_positions[0] - 1
    clean_logits, logits = clean["logits"][pred], run["logits"][pred]
    clean_lp = torch.log_softmax(clean_logits, dim=-1)
    changed_lp = torch.log_softmax(logits, dim=-1)
    kl = float((clean_lp.exp() * (clean_lp - changed_lp)).sum())
    promoted, suppressed = top_token_changes(clean_logits, logits)
    pos = loc.intervention_pos if treatment.position_mode == "DECLARED" else loc.previous_pos
    l5_l2, l5_rel, l5_cos = vector_change(clean["hidden_l5"][pos], run["hidden_l5"][pos])
    final_l2, final_rel, final_cos = vector_change(clean["hidden_final"][pos], run["hidden_final"][pos])
    originals = [float(clean["act_original"][pos, n]) for n in treatment.neurons]
    sets = [float(run["act_modified"][pos, n]) for n in treatment.neurons]
    return {
        "case_id": case.case_id, "branch": case.branch, "family": case.family,
        "treatment": treatment.name, "declared_role": treatment.role,
        "text": case.text, "prefix": case.prefix, "outcome": case.outcome,
        "n_tok": len(loc.body_ids), "tokens": json_tokens(loc.body_ids),
        "token_ids": json.dumps(loc.body_ids),
        "outcome_tokens": json_tokens(loc.outcome_ids),
        "outcome_token_ids": json.dumps(loc.outcome_ids),
        "declared_intervention_pos": loc.intervention_pos,
        "actual_intervention_pos": pos,
        "position_mode": treatment.position_mode,
        "intervened_neurons": json.dumps(treatment.neurons),
        "value_mode": treatment.value_mode, "declared_value": treatment.value if treatment.value is not None else "",
        "original_activations": json.dumps(originals), "set_activations": json.dumps(sets),
        "activation_changes": json.dumps([b-a for a, b in zip(originals, sets)]),
        "clean_outcome_logprob": round(clean_m["outcome_logprob"], 6),
        "outcome_logprob": round(m["outcome_logprob"], 6),
        "outcome_logprob_change": round(m["outcome_logprob"]-clean_m["outcome_logprob"], 12),
        "clean_first_outcome_prob": round(clean_m["first_prob"], 9),
        "first_outcome_prob": round(m["first_prob"], 9),
        "first_outcome_prob_change": round(m["first_prob"]-clean_m["first_prob"], 9),
        "clean_first_outcome_rank": clean_m["first_rank"], "first_outcome_rank": m["first_rank"],
        "first_outcome_rank_change": m["first_rank"]-clean_m["first_rank"],
        "clean_greedy_token": clean_m["greedy_token"], "greedy_token": m["greedy_token"],
        "greedy_changed": int(m["greedy_id"] != clean_m["greedy_id"]),
        "kl_clean_to_intervened": round(kl, 12),
        "clean_first_entropy": round(clean_m["entropy"], 6), "first_entropy": round(m["entropy"], 6),
        "first_entropy_change": round(m["entropy"]-clean_m["entropy"], 6),
        "first_logit_l2": round(float(torch.linalg.vector_norm(logits-clean_logits)), 12),
        "max_abs_logit_change": round(float((logits-clean_logits).abs().max()), 12),
        "promoted_tokens": json.dumps(promoted, ensure_ascii=False),
        "suppressed_tokens": json.dumps(suppressed, ensure_ascii=False),
        "hidden_l5_delta_l2": round(l5_l2, 6), "hidden_l5_relative_delta": round(l5_rel, 9),
        "hidden_l5_cosine": round(l5_cos, 9), "hidden_final_delta_l2": round(final_l2, 6),
        "hidden_final_relative_delta": round(final_rel, 9), "hidden_final_cosine": round(final_cos, 9),
        "outcome_token_metrics": json.dumps(m["tokens"], ensure_ascii=False), "notes": case.notes,
    }


def summary_rows(results):
    groups = defaultdict(list)
    for row in results:
        groups[(row["branch"], row["treatment"], row["declared_role"])].append(row)
    out = []
    for (branch, treatment, role), members in sorted(groups.items()):
        vals = lambda k: [float(x[k]) for x in members]
        out.append({
            "branch": branch, "treatment": treatment, "declared_role": role,
            "n": len(members),
            "outcome_logprob_change_mean": round(statistics.mean(vals("outcome_logprob_change")), 6),
            "outcome_logprob_change_median": round(statistics.median(vals("outcome_logprob_change")), 6),
            "first_outcome_prob_change_mean": round(statistics.mean(vals("first_outcome_prob_change")), 9),
            "kl_mean": round(statistics.mean(vals("kl_clean_to_intervened")), 9),
            "kl_median": round(statistics.median(vals("kl_clean_to_intervened")), 9),
            "first_logit_l2_mean": round(statistics.mean(vals("first_logit_l2")), 6),
            "hidden_final_delta_l2_mean": round(statistics.mean(vals("hidden_final_delta_l2")), 6),
            "greedy_changed_n": sum(int(x["greedy_changed"]) for x in members),
            "positive_outcome_logprob_change_n": sum(float(x["outcome_logprob_change"]) > 0 for x in members),
            "member_case_ids": json.dumps([x["case_id"] for x in members]),
        })
    return out


def token_effect_rows(results):
    out = []
    for row in results:
        if row["treatment"] in {"CLEAN", "CLEAN_REPEAT", "SHAM_38"}:
            continue
        for direction, field in (("PROMOTED", "promoted_tokens"),
                                 ("SUPPRESSED", "suppressed_tokens")):
            for rank, item in enumerate(json.loads(row[field]), 1):
                out.append({"case_id": row["case_id"], "branch": row["branch"],
                            "family": row["family"], "treatment": row["treatment"],
                            "direction": direction, "rank": rank, **item})
    return out


def factorial_rows(results):
    """Descriptive team-minus-sum contrasts; not assumed to be linear tests."""
    by_case = defaultdict(dict)
    for row in results:
        by_case[row["case_id"]][row["treatment"]] = row
    out = []
    for case_id, rows in sorted(by_case.items()):
        if "ADD_POS_HIGH_TEAM" in rows:
            members = ("ADD_POS_HIGH_38", "ADD_POS_HIGH_1103", "ADD_POS_HIGH_2094")
            team = rows["ADD_POS_HIGH_TEAM"]
            if all(x in rows for x in members):
                single_sum = sum(float(rows[x]["outcome_logprob_change"]) for x in members)
                team_effect = float(team["outcome_logprob_change"])
                out.append({"case_id": case_id, "branch": team["branch"],
                            "contrast": "TEAM_MINUS_SUM_OF_SINGLE_ADDITIONS",
                            "team_treatment": "ADD_POS_HIGH_TEAM",
                            "single_treatments": json.dumps(members),
                            "team_outcome_logprob_change": round(team_effect, 12),
                            "single_effect_sum": round(single_sum, 12),
                            "interaction_residual": round(team_effect-single_sum, 12),
                            "caveat": "descriptive non-additivity; downstream network and log-probability scale are nonlinear"})
        if "ZERO_TEAM_38_1103_2094" in rows:
            members = ("ZERO_38", "ZERO_1103", "ZERO_2094")
            team = rows["ZERO_TEAM_38_1103_2094"]
            if all(x in rows for x in members):
                single_sum = sum(float(rows[x]["outcome_logprob_change"]) for x in members)
                team_effect = float(team["outcome_logprob_change"])
                out.append({"case_id": case_id, "branch": team["branch"],
                            "contrast": "TEAM_MINUS_SUM_OF_SINGLE_ABLATIONS",
                            "team_treatment": "ZERO_TEAM_38_1103_2094",
                            "single_treatments": json.dumps(members),
                            "team_outcome_logprob_change": round(team_effect, 12),
                            "single_effect_sum": round(single_sum, 12),
                            "interaction_residual": round(team_effect-single_sum, 12),
                            "caveat": "descriptive non-additivity; downstream network and log-probability scale are nonlinear"})
    return out


PAIRED_OUTCOMES = (
    ("IT_WAS", "P_IT_WAS_COLD", "P_IT_WAS_WARM"),
    ("AIR_IS", "P_AIR_IS_COLD", "P_AIR_IS_HOT"),
    ("ROOM_WAS", "P_ROOM_WAS_COLD", "P_ROOM_WAS_DARK"),
    ("SEASON_IS", "P_SEASON_IS_COLD", "P_SEASON_IS_WARM"),
    ("WORLD_IS", "P_WORLD_IS_COLD", "P_WORLD_IS_STRANGE"),
)


def paired_contrast_rows(results, natural_max):
    """Cold-minus-control contrasts under an identical prefix/intervention."""
    by_key = {(r["case_id"], r["treatment"]): r for r in results}
    treatments = sorted({r["treatment"] for r in results
                         if r["treatment"] not in {"CLEAN", "CLEAN_REPEAT", "SHAM_38"}})
    out = []
    for treatment in treatments:
        pair_values = []
        for pair_id, cold_id, control_id in PAIRED_OUTCOMES:
            cold = by_key.get((cold_id, treatment))
            control = by_key.get((control_id, treatment))
            if cold is None or control is None:
                continue
            cold_change = float(cold["outcome_logprob_change"])
            control_change = float(control["outcome_logprob_change"])
            cold_set = json.loads(cold["set_activations"])[0]
            control_set = json.loads(control["set_activations"])[0]
            if abs(cold_set-control_set) > 1e-6:
                raise RuntimeError(f"{pair_id}/{treatment}: paired final activations differ")
            advantage = cold_change-control_change
            pair_values.append(advantage)
            out.append({
                "row_type": "PAIR", "treatment": treatment, "pair_id": pair_id,
                "prefix": cold["prefix"], "cold_case_id": cold_id,
                "control_case_id": control_id, "cold_outcome": cold["outcome"],
                "control_outcome": control["outcome"],
                "final_38_activation": round(cold_set, 6),
                "within_observed_natural_max": int(0.0 <= cold_set <= natural_max+1e-6),
                "cold_logprob_change": round(cold_change, 12),
                "control_logprob_change": round(control_change, 12),
                "cold_minus_control": round(advantage, 12),
                "mean_cold_minus_control": "", "n_pairs": "",
            })
        if pair_values:
            out.append({
                "row_type": "AGGREGATE", "treatment": treatment,
                "pair_id": "ALL_DECLARED_PAIRS", "prefix": "",
                "cold_case_id": "", "control_case_id": "", "cold_outcome": "",
                "control_outcome": "", "final_38_activation": "",
                "within_observed_natural_max": "",
                "cold_logprob_change": "", "control_logprob_change": "",
                "cold_minus_control": "",
                "mean_cold_minus_control": round(statistics.mean(pair_values), 12),
                "n_pairs": len(pair_values),
            })
    return out


class PhenotypeScorer:
    """Scores Atlas destination and N38 trace without forming LM logits."""
    def __init__(self):
        self.device = next(zp.mdl.parameters()).device
        self.buf = {}
        self.handle = zp.mdl.transformer.h[zp.ELL].mlp.act.register_forward_hook(
            lambda module, inputs, output: self.buf.__setitem__("A", output.detach()))

    def close(self):
        self.handle.remove()

    def score(self, text):
        body = list(zp.tok(text, add_special_tokens=False)["input_ids"])
        ids = [zp.BOS] + body
        if not body or len(ids) > zp.N_CTX:
            raise ValueError("phenotype string outside dom(D)")
        tensor = torch.tensor([ids], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            zp.mdl.transformer(input_ids=tensor)
        A = self.buf["A"][0, :len(ids)]
        peaks = A.max(dim=0).values
        top2 = torch.topk(peaks, 2)
        return body, A, {
            "destination": int(top2.indices[0]), "runner_up": int(top2.indices[1]),
            "margin": float(top2.values[0]-top2.values[1]),
            "winner_peak": float(top2.values[0]), "target_peak": float(peaks[TARGET]),
            "target_peak_pos": int(A[:, TARGET].argmax()),
        }


def phenotype_and_certificates(strings):
    scorer = PhenotypeScorer()
    summaries, traces, certs = [], [], []
    try:
        for index, text in enumerate(strings, 1):
            body, A, score = scorer.score(text)
            summaries.append({"case_index": index, "text": text, "n_tok": len(body),
                              "tokens": json_tokens(body), "token_ids": json.dumps(body),
                              **{k: round(v, 6) if isinstance(v, float) else v
                                 for k, v in score.items()},
                              "target_retained": int(score["destination"] == TARGET)})
            ids = [zp.BOS] + body
            for pos in range(len(ids)):
                traces.append({"case_index": index, "text": text, "position": pos,
                               "token_id": ids[pos], "token": "<BOS>" if pos == 0 else decode_token(ids[pos]),
                               "target_activation": round(float(A[pos, TARGET]), 6),
                               "position_winner": int(A[pos].argmax()),
                               "position_winner_activation": round(float(A[pos].max()), 6)})
            # Exhaustive contiguous source-token spans.  Certificates require
            # N38 to be the Atlas destination, matching the v7 criterion.
            candidates = []
            seen = set()
            for width in range(1, len(body)+1):
                for start in range(0, len(body)-width+1):
                    span = zp.tok.decode(body[start:start+width])
                    if not span or span in seen:
                        continue
                    seen.add(span)
                    realised, _, s = scorer.score(span)
                    if s["destination"] == TARGET:
                        candidates.append((len(realised), start, width, span, realised, s))
                if candidates:
                    break
            if candidates:
                candidates.sort(key=lambda x: (x[0], x[1], x[3]))
                n, start, width, span, realised, s = candidates[0]
                certs.append({"case_index": index, "source_text": text,
                              "source_destination": score["destination"],
                              "certificate_found": 1, "certificate": span,
                              "source_start": start, "source_width": width,
                              "realised_n_tok": n, "realised_token_ids": json.dumps(realised),
                              "certificate_target_peak": round(s["target_peak"], 6),
                              "target_peak_drift": round(s["target_peak"]-score["target_peak"], 6)})
            else:
                certs.append({"case_index": index, "source_text": text,
                              "source_destination": score["destination"],
                              "certificate_found": 0, "certificate": "",
                              "source_start": "", "source_width": "", "realised_n_tok": "",
                              "realised_token_ids": "", "certificate_target_peak": "",
                              "target_peak_drift": ""})
    finally:
        scorer.close()
    return summaries, traces, certs


def direct_logit_rows():
    directions = cproj_directions()
    # Direct effect at the residual stream is N38's c_proj direction.  GPT-2's
    # final layer norm makes exact vocabulary effects state-dependent, so this
    # is explicitly a state-free screening heuristic, not a causal outcome.
    vec = directions[TARGET].float()
    embed = zp.mdl.lm_head.weight.detach().float()
    scores = embed @ vec
    up, down = torch.topk(scores, TOP_N), torch.topk(-scores, TOP_N)
    rows = []
    for direction, result, sign in (("PROMOTED", up, 1), ("SUPPRESSED", down, -1)):
        for rank, (idx, value) in enumerate(zip(result.indices, result.values), 1):
            rows.append({"neuron": TARGET, "direction": direction, "rank": rank,
                         "token_id": int(idx), "token": decode_token(idx),
                         "unnormalised_logit_direction": round(sign*float(value), 6),
                         "caveat": "state-free c_proj-to-unembedding screen; final layer norm omitted"})
    return rows


def greedy_generate(prompt, neurons, add_value, mode, max_new_tokens):
    device = next(zp.mdl.parameters()).device
    body = list(zp.tok(prompt, add_special_tokens=False)["input_ids"])
    ids = torch.tensor([[zp.BOS] + body], dtype=torch.long, device=device)
    generated = []
    step_rows = []
    for step in range(max_new_tokens):
        captured = {}
        active = bool(neurons) and (mode == "SUSTAINED" or step == 0)
        def hook(module, inputs, output):
            changed = output.clone()
            captured["original"] = [float(output[0, -1, n]) for n in neurons]
            if active:
                for neuron in neurons:
                    changed[0, -1, neuron] += float(add_value)
            captured["set"] = [float(changed[0, -1, n]) for n in neurons]
            return changed if active else None
        handle = zp.mdl.transformer.h[zp.ELL].mlp.act.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = zp.mdl(input_ids=ids, use_cache=False, return_dict=True)
        finally:
            handle.remove()
        logits = out.logits[0, -1]
        next_id = int(logits.argmax())
        generated.append(next_id)
        step_rows.append({"step": step+1, "active": int(active),
                          "generated_token_id": next_id, "generated_token": decode_token(next_id),
                          "intervention_originals": json.dumps(captured.get("original", [])),
                          "intervention_sets": json.dumps(captured.get("set", []))})
        ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
        if ids.shape[1] >= zp.N_CTX:
            break
    return generated, step_rows


def generation_rows(prompts, high_dose, max_new_tokens):
    treatments = (
        ("CLEAN", (), 0.0, "ONE_STEP", "baseline"),
        ("N38_ONE_STEP_NATURAL", (38,), high_dose, "ONE_STEP", "natural-range dose"),
        ("N38_SUSTAINED_NATURAL", (38,), high_dose, "SUSTAINED", "autoregressive feedback test"),
        ("TEAM_ONE_STEP_N38_CALIBRATED", (38,1103,2094), high_dose, "ONE_STEP", "old team at shared N38-calibrated dose"),
        ("TEAM_SUSTAINED_N38_CALIBRATED", (38,1103,2094), high_dose, "SUSTAINED", "old team feedback test at shared N38-calibrated dose"),
        ("N38_SUSTAINED_LEGACY_PLUS10", (38,), 10.0, "SUSTAINED", "labelled legacy replication"),
        ("TEAM_SUSTAINED_LEGACY_PLUS10", (38,1103,2094), 10.0, "SUSTAINED", "labelled legacy replication"),
        ("N38_SUSTAINED_NEG_NATURAL", (38,), -high_dose, "SUSTAINED", "signed direction control"),
    )
    rows, steps = [], []
    for prompt_id, prompt in enumerate(prompts, 1):
        clean_generated = None
        for name, neurons, value, mode, role in treatments:
            generated, trace = greedy_generate(prompt, neurons, value, mode, max_new_tokens)
            if name == "CLEAN":
                clean_generated = list(generated)
            divergence = ""
            exact_clean_match = ""
            if clean_generated is not None:
                exact_clean_match = int(generated == clean_generated)
                for i, (clean_id, changed_id) in enumerate(zip(clean_generated, generated), 1):
                    if clean_id != changed_id:
                        divergence = i
                        break
                if divergence == "" and len(generated) != len(clean_generated):
                    divergence = min(len(generated), len(clean_generated)) + 1
            generated_text = zp.tok.decode(generated)
            cold_lexeme_count = sum(generated_text.lower().count(stem)
                                    for stem in ("cold", "frigid", "freez", "icy", "chill"))
            rows.append({"prompt_id": prompt_id, "prompt": prompt, "treatment": name,
                         "declared_role": role, "neurons": json.dumps(neurons),
                         "add_value": value, "intervention_mode": mode,
                         "max_new_tokens": max_new_tokens,
                         "generated_token_ids": json.dumps(generated),
                         "generated_text": generated_text,
                         "full_text": prompt + generated_text,
                         "exact_clean_match": exact_clean_match,
                         "first_divergence_step": divergence,
                         "cold_lexeme_count": cold_lexeme_count})
            for item in trace:
                steps.append({"prompt_id": prompt_id, "prompt": prompt,
                              "treatment": name, "neurons": json.dumps(neurons),
                              "add_value": value, "intervention_mode": mode, **item})
    return rows, steps


def validate(cases, results, controls):
    failures = []
    if len({c.case_id for c in cases}) != len(cases):
        failures.append("duplicate case IDs")
    if len(set(controls)) != 3 or any(n in DECLARED_EXCLUSIONS for n in controls):
        failures.append(f"invalid matched controls {controls}")
    for treatment in ("CLEAN_REPEAT", "SHAM_38"):
        rows = [r for r in results if r["treatment"] == treatment]
        if len(rows) != len(cases):
            failures.append(f"{treatment}: {len(rows)} rows != {len(cases)} cases")
        for row in rows:
            if (abs(float(row["max_abs_logit_change"])) > SHAM_TOL or
                    abs(float(row["outcome_logprob_change"])) > SHAM_TOL):
                failures.append(f"{row['case_id']}: {treatment} mismatch")
    return failures


def validate_v14(cases, results, doses, paired):
    failures = []
    if len({c.case_id for c in cases}) != len(cases):
        failures.append("duplicate case IDs")
    for treatment in ("CLEAN_REPEAT", "SHAM_38"):
        rows = [r for r in results if r["treatment"] == treatment]
        if len(rows) != len(cases):
            failures.append(f"{treatment}: {len(rows)} rows != {len(cases)} cases")
        for row in rows:
            if (abs(float(row["max_abs_logit_change"])) > SHAM_TOL or
                    abs(float(row["outcome_logprob_change"])) > SHAM_TOL):
                failures.append(f"{row['case_id']}: {treatment} mismatch")
    expected_set = {
        "SET_0_38": 0.0, "SET_LOW_38": doses["LOW"],
        "SET_MED_38": doses["MED"], "SET_HIGH_38": doses["HIGH"],
        "SET_MAX_38": doses["MAX"],
    }
    for treatment, expected in expected_set.items():
        rows = [r for r in results if r["treatment"] == treatment]
        if len(rows) != len(cases):
            failures.append(f"{treatment}: missing rows")
        for row in rows:
            actual = json.loads(row["set_activations"])[0]
            if abs(actual-expected) > 1e-5:
                failures.append(f"{row['case_id']}: {treatment} set {actual} != {expected}")
    expected_aggregates = len([
        r for r in paired if r["row_type"] == "AGGREGATE"
    ])
    nonidentity = len({r["treatment"] for r in results
                       if r["treatment"] not in {"CLEAN", "CLEAN_REPEAT", "SHAM_38"}})
    if expected_aggregates != nonidentity:
        failures.append(
            f"paired aggregates {expected_aggregates} != treatments {nonidentity}"
        )
    return failures


def preflight_rows(cases, located, clean_runs, selection, doses):
    rows = list(selection)
    for label, value in doses.items():
        rows.append({"selection_type": "NATURAL_DOSE", "neuron": TARGET,
                     "dose_label": label, "dose_value": round(value, 6),
                     "selection_rule": "nearest-rank quantile of positive clean N38 declared-position activations"})
    for case in cases:
        loc, run = located[case.case_id], clean_runs[case.case_id]
        m = outcome_metrics(run, loc)
        rows.append({"selection_type": "CASE", "case_id": case.case_id,
                     "branch": case.branch, "family": case.family,
                     "prefix": case.prefix, "outcome": case.outcome, "text": case.text,
                     "n_tok": len(loc.body_ids), "tokens": json_tokens(loc.body_ids),
                     "token_ids": json.dumps(loc.body_ids),
                     "intervention_pos": loc.intervention_pos,
                     "intervention_token": decode_token(([zp.BOS]+list(loc.body_ids))[loc.intervention_pos]),
                     "previous_pos": loc.previous_pos,
                     "outcome_positions": json.dumps(loc.outcome_positions),
                     "outcome_tokens": json_tokens(loc.outcome_ids),
                     "clean_38_activation": round(float(run["act_original"][loc.intervention_pos,TARGET]),6),
                     "clean_outcome_logprob": round(m["outcome_logprob"],6),
                     "clean_first_outcome_prob": round(m["first_prob"],9),
                     "clean_first_outcome_rank": m["first_rank"], "notes": case.notes})
    return rows


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--outdir", default="results")
    p.add_argument("--smoke", action="store_true",
                   help="two complete prefix pairs; full dose/validation machinery")
    return p.parse_args()


def main():
    args = parse_args()
    device = next(zp.mdl.parameters()).device
    if device.type != "cuda":
        raise SystemExit(f"znou_probe model is on {device}; v14 requires CUDA")

    # Always calibrate on the full preregistered natural panel so smoke and
    # full runs use identical absolute values.
    calibration = list(NATURAL_CASES)
    if args.smoke:
        smoke_ids = {"P_IT_WAS_COLD", "P_IT_WAS_WARM",
                     "P_AIR_IS_COLD", "P_AIR_IS_HOT"}
        steering = [c for c in STEERING_CASES if c.case_id in smoke_ids]
    else:
        steering = list(STEERING_CASES)
    cache_cases = calibration + steering
    located = {c.case_id: locate(c) for c in cache_cases}
    print(f"model device: {device}")
    print(f"calibration cases: {len(calibration)}")
    print(f"reported steering cases: {len(steering)}")

    runner = ForwardRunner()
    clean_runs = {}
    for i, case in enumerate(cache_cases, 1):
        clean_runs[case.case_id] = runner.run(case, located[case.case_id])
        if i == 1 or i == len(cache_cases):
            print(f"clean [{i}/{len(cache_cases)}]")

    doses = natural_doses(clean_runs, calibration, located)
    print("N38 natural doses: " + ", ".join(f"{k}={v:.6f}" for k,v in doses.items()))

    treatments = steering_treatments(doses)
    results = []
    total = len(steering) * len(treatments)
    count = 0
    for case in steering:
        loc, clean = located[case.case_id], clean_runs[case.case_id]
        for treatment in treatments:
            count += 1
            pos = loc.intervention_pos if treatment.position_mode == "DECLARED" else loc.previous_pos
            if treatment.value_mode == "CLEAN":
                run = clean
            elif treatment.value_mode == "RERUN":
                run = runner.run(case, loc)
            else:
                run = runner.run(case, loc, treatment.neurons, pos,
                                 treatment.value_mode, treatment.value)
            results.append(intervention_row(case, loc, treatment, run, clean))
            if count == 1 or count % 50 == 0 or count == total:
                print(f"intervention [{count}/{total}]")

    summaries = summary_rows(results)
    effects = token_effect_rows(results)
    paired = paired_contrast_rows(results, doses["MAX"])
    preflight = preflight_rows(cache_cases, located, clean_runs, [], doses)
    failures = validate_v14(steering, results, doses, paired)
    print("\nvalidation")
    print(f"  intervention rows       {len(results)}")
    print(f"  paired-contrast rows    {len(paired)}")
    print(f"  validation failures     {len(failures)}")

    outdir = Path(args.outdir)
    suffix = "smoke" if args.smoke else ""
    outputs = []
    datasets = (
        ("probe_v14_preflight", preflight),
        ("probe_v14_interventions", results),
        ("probe_v14_token_effects", effects),
        ("probe_v14_paired", paired),
        ("probe_v14_summary", summaries),
    )
    for name, data in datasets:
        if not data:
            continue
        recorder = zp.Recorder(name, outdir=outdir)
        recorder.extend(data)
        outputs.append(recorder.write(suffix=suffix))
    print("\noutputs")
    for path in outputs:
        print(f"  {path}")
    if failures:
        raise SystemExit("v14 validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
