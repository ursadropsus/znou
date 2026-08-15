"""
probe_v12.py — causal ablation and steering panel for layer-5 neuron 541.

V9-v11 established a reproducible observational phenotype around completion of
connected parallel-member constructions.  V12 asks whether neuron 541 makes a
causal downstream contribution.  It never uses the directly manipulated
neuron's Resonance destination as an outcome.  Primary outcomes are downstream:

  * teacher-forced outcome-token log probability
  * next-token KL divergence, entropy and token rank
  * promoted/suppressed output logits
  * residual-stream change after layer 5 and at the final hidden state

Branches
--------
NATURAL_COMPLETION
    Ablate or scale natural 541 activation at construction completion.

COMPLETION_INJECTION
    Inject realistic values or a matched donor into weak/destructive controls.

PRECOMPLETION_STEERING
    Artificially inject 541 at the connector position and measure probability
    of a declared completing member.  This is sufficiency/steering evidence,
    not a claim about 541's naturally observed firing time.

The intervention acts on blocks.5.mlp.act (post-GELU), one neuron at one token
position.  A sham hook writes the exact clean scalar back and must reproduce
the unhooked logits within the declared tolerance.

Requires the CUDA-corrected znou_probe.py beside this file.

Examples
--------
  python probe_v12.py --smoke
  python probe_v12.py
  python probe_v12.py --outdir results
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


TARGET = 541
RIVAL = 2659
TOP_N = 10
SHAM_TOL = 1e-7
EPS = 1e-8


@dataclass(frozen=True)
class Case:
    case_id: str
    branch: str
    prefix: str
    outcome: str
    donor_id: str = ""
    family: str = ""
    notes: str = ""

    @property
    def text(self):
        return self.prefix + self.outcome


# The outcome is present for teacher forcing.  The intervention position is the
# last realised token wholly inside prefix; outcome positions are scored from
# the logits immediately preceding each realised outcome token.
NATURAL_CASES = (
    Case("N_ONE_AFTER_ANOTHER", "NATURAL_COMPLETION",
         "one after another", " arrived", family="PRONOMINAL",
         notes="strong v11 slot frame"),
    Case("N_ONE_BY_ONE", "NATURAL_COMPLETION",
         "one by one", " they came", family="RECURRENT",
         notes="strong v9/v11 recurrent frame"),
    Case("N_HAND_TO_HAND", "NATURAL_COMPLETION",
         "hand to hand", " combat", family="RECURRENT",
         notes="strong v9/v11 recurrent frame"),
    Case("N_UP_AND_DOWN", "NATURAL_COMPLETION",
         "up and down", " the road", family="PAIRED",
         notes="strong v9/v10 distinct-member frame"),
    Case("N_AGAIN_AND_AGAIN", "NATURAL_COMPLETION",
         "again and again", " he tried", family="RECURRENT",
         notes="strong v9/v11 recurrent frame"),
    Case("N_WAVE_AFTER_WAVE", "NATURAL_COMPLETION",
         "wave after wave", " struck", family="RECURRENT",
         notes="v9-attested after frame"),
    Case("N_SHOULDER_TO_SHOULDER", "NATURAL_COMPLETION",
         "shoulder to shoulder", " they stood", family="RECURRENT",
         notes="strong v9/v11 recurrent frame"),
    Case("N_ROUND_BARE", "NATURAL_COMPLETION",
         "round and round", " the ring", family="CONTEXT_FRAGILE",
         notes="v11 bare form; v9 predicts context dependence"),
    Case("N_ROUND_CONTEXT", "NATURAL_COMPLETION",
         " and thus round and round", " the ship", family="CONTEXT_ASSISTED",
         notes="exact v9 token-minimum context-assisted form"),
    Case("N_GO_ROUND_CONTEXT", "NATURAL_COMPLETION",
         ", go round and round", ".", family="CONTEXT_ASSISTED",
         notes="exact v9 token-minimum context-assisted form"),
)


INJECTION_CASES = (
    Case("I_GRAVEL_AFTER_GRAVEL", "COMPLETION_INJECTION",
         "gravel after gravel", " again", "N_WAVE_AFTER_WAVE",
         "NOVEL_REPEAT", "novel connected repetition"),
    Case("I_GRAVEL_AFTER_WINDOW", "COMPLETION_INJECTION",
         "gravel after window", " again", "N_WAVE_AFTER_WAVE",
         "ARBITRARY_DISTINCT", "novel distinct connected control"),
    Case("I_GRAVEL_GRAVEL", "COMPLETION_INJECTION",
         "gravel gravel", " again", "N_WAVE_AFTER_WAVE",
         "NO_CONNECTOR", "novel connectorless repetition"),
    Case("I_HAMMER_AND_HAMMER", "COMPLETION_INJECTION",
         "hammer and hammer", " again", "N_AGAIN_AND_AGAIN",
         "NOVEL_REPEAT", "v11 moderate novel repetition"),
    Case("I_HAMMER_AND_VIOLIN", "COMPLETION_INJECTION",
         "hammer and violin", " again", "N_AGAIN_AND_AGAIN",
         "ARBITRARY_DISTINCT", "v11 arbitrary distinct control"),
    Case("I_HAMMER_HAMMER", "COMPLETION_INJECTION",
         "hammer hammer", " again", "N_AGAIN_AND_AGAIN",
         "NO_CONNECTOR", "connectorless repetition"),
    Case("I_WAVE_AFTER_TIDE", "COMPLETION_INJECTION",
         "wave after tide", " again", "N_WAVE_AFTER_WAVE",
         "RELATED_DISTINCT", "v11 related distinct control"),
    Case("I_GRAVEL_AFTER_ANOTHER", "COMPLETION_INJECTION",
         "gravel after another", " arrived", "N_ONE_AFTER_ANOTHER",
         "PRONOMINAL_LEXICAL_CONTROL", "v11 first-slot destruction"),
    Case("I_ONE_ANOTHER", "COMPLETION_INJECTION",
         "one another", " arrived", "N_ONE_AFTER_ANOTHER",
         "PRONOMINAL_CONNECTOR_CONTROL", "v11 connector removal"),
)


PRECOMPLETION_CASES = (
    Case("P_ONE_AFTER_ANOTHER", "PRECOMPLETION_STEERING",
         "one after", " another", "N_ONE_AFTER_ANOTHER", "PRONOMINAL"),
    Case("P_ONE_AFTER_ONE", "PRECOMPLETION_STEERING",
         "one after", " one", "N_ONE_AFTER_ANOTHER", "PRONOMINAL"),
    Case("P_ONE_AFTER_THE_OTHER", "PRECOMPLETION_STEERING",
         "one after", " the other", "N_ONE_AFTER_ANOTHER", "PRONOMINAL"),
    Case("P_WAVE_AFTER_WAVE", "PRECOMPLETION_STEERING",
         "wave after", " wave", "N_WAVE_AFTER_WAVE", "RECURRENT"),
    Case("P_ONE_BY_ONE", "PRECOMPLETION_STEERING",
         "one by", " one", "N_ONE_BY_ONE", "RECURRENT"),
    Case("P_HAND_TO_HAND", "PRECOMPLETION_STEERING",
         "hand to", " hand", "N_HAND_TO_HAND", "RECURRENT"),
    Case("P_UP_AND_DOWN", "PRECOMPLETION_STEERING",
         "up and", " down", "N_UP_AND_DOWN", "PAIRED"),
    Case("P_NOW_OR_THEN", "PRECOMPLETION_STEERING",
         "now or", " then", "N_UP_AND_DOWN", "PAIRED",
         notes="related paired-family donor declared in advance"),
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
    treatment: str
    neuron: int | None
    position_mode: str
    value_mode: str
    value: float | None = None
    declared_role: str = ""


def locate(case):
    encoded = zp.tok(
        case.text, add_special_tokens=False, return_offsets_mapping=True
    )
    ids = tuple(int(x) for x in encoded["input_ids"])
    offsets = tuple(tuple(int(y) for y in x)
                    for x in encoded["offset_mapping"])
    if not ids:
        raise ValueError(f"{case.case_id}: empty text")
    boundary = len(case.prefix)
    prefix_body = [i for i, (_, end) in enumerate(offsets) if end <= boundary]
    crossing = [i for i, (start, end) in enumerate(offsets)
                if start < boundary < end]
    if crossing:
        raise ValueError(
            f"{case.case_id}: token crosses prefix/outcome boundary: {crossing}"
        )
    if not prefix_body:
        raise ValueError(f"{case.case_id}: prefix has no realised token")
    outcome_body = [i for i, (start, end) in enumerate(offsets)
                    if end > boundary and start >= boundary]
    if not outcome_body:
        raise ValueError(f"{case.case_id}: outcome has no realised token")
    intervention_pos = max(prefix_body) + 1  # BOS offset
    previous_pos = intervention_pos - 1
    if previous_pos < 0:
        raise ValueError(f"{case.case_id}: no previous-position control")
    outcome_positions = tuple(i + 1 for i in outcome_body)
    if outcome_positions[0] - 1 != intervention_pos:
        raise ValueError(
            f"{case.case_id}: intervention is not the primary predictor position"
        )
    return Located(
        body_ids=ids,
        offsets=offsets,
        intervention_pos=intervention_pos,
        previous_pos=previous_pos,
        outcome_positions=outcome_positions,
        outcome_ids=tuple(ids[i] for i in outcome_body),
    )


def decode_token(token_id):
    return zp.tok.decode([int(token_id)])


def json_tokens(ids):
    return json.dumps([decode_token(x) for x in ids], ensure_ascii=False)


class ForwardRunner:
    """Runs one deterministic forward pass with an optional scalar hook."""

    def __init__(self):
        self.device = next(zp.mdl.parameters()).device

    def run(self, case, loc, neuron=None, position=None, set_value=None):
        ids = [zp.BOS] + list(loc.body_ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        captured = {}

        def hook(module, inputs, output):
            captured["original"] = output.detach().clone()
            if neuron is None:
                captured["modified"] = output.detach().clone()
                return None
            if not (0 <= position < output.shape[1]):
                raise IndexError(
                    f"{case.case_id}: intervention position {position} outside "
                    f"0:{output.shape[1]}"
                )
            changed = output.clone()
            captured["original_scalar"] = float(output[0, position, neuron])
            changed[0, position, neuron] = float(set_value)
            captured["modified_scalar"] = float(changed[0, position, neuron])
            captured["modified"] = changed.detach().clone()
            return changed

        handle = zp.mdl.transformer.h[zp.ELL].mlp.act.register_forward_hook(hook)
        try:
            with torch.inference_mode():
                out = zp.mdl(
                    input_ids=input_ids,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            handle.remove()
        if "original" not in captured:
            raise RuntimeError(f"{case.case_id}: layer-5 hook did not fire")
        return {
            "logits": out.logits[0].detach().clone(),
            "hidden_l5": out.hidden_states[zp.ELL + 1][0].detach().clone(),
            "hidden_final": out.hidden_states[-1][0].detach().clone(),
            "act_original": captured["original"][0],
            "act_modified": captured["modified"][0],
            "original_scalar": captured.get("original_scalar"),
            "modified_scalar": captured.get("modified_scalar"),
        }


def outcome_metrics(run, loc):
    total = 0.0
    token_rows = []
    for pos, token_id in zip(loc.outcome_positions, loc.outcome_ids):
        pred_pos = pos - 1
        log_probs = torch.log_softmax(run["logits"][pred_pos], dim=-1)
        lp = float(log_probs[token_id])
        total += lp
        value = run["logits"][pred_pos, token_id]
        rank = int((run["logits"][pred_pos] > value).sum()) + 1
        token_rows.append({
            "outcome_pos": pos,
            "predictor_pos": pred_pos,
            "token_id": int(token_id),
            "token": decode_token(token_id),
            "logprob": lp,
            "prob": math.exp(lp),
            "rank": rank,
        })
    first_pred = loc.outcome_positions[0] - 1
    first_logits = run["logits"][first_pred]
    first_probs = torch.softmax(first_logits, dim=-1)
    entropy = float(-(first_probs * torch.log(first_probs.clamp_min(EPS))).sum())
    greedy = int(first_logits.argmax())
    return {
        "outcome_logprob": total,
        "outcome_mean_logprob": total / len(token_rows),
        "first_outcome_logprob": token_rows[0]["logprob"],
        "first_outcome_prob": token_rows[0]["prob"],
        "first_outcome_rank": token_rows[0]["rank"],
        "first_greedy_id": greedy,
        "first_greedy_token": decode_token(greedy),
        "first_entropy": entropy,
        "token_rows": token_rows,
    }


def vector_change(clean, changed):
    delta = changed - clean
    l2 = float(torch.linalg.vector_norm(delta))
    clean_norm = float(torch.linalg.vector_norm(clean))
    relative = l2 / max(clean_norm, EPS)
    cosine = float(torch.nn.functional.cosine_similarity(
        clean.unsqueeze(0), changed.unsqueeze(0), dim=-1
    )[0])
    return l2, relative, cosine


def top_token_changes(clean_logits, changed_logits, n=TOP_N):
    delta = changed_logits - clean_logits
    up = torch.topk(delta, n)
    down = torch.topk(-delta, n)
    promoted = [
        {
            "token_id": int(i), "token": decode_token(i),
            "delta_logit": round(float(v), 6),
        }
        for i, v in zip(up.indices, up.values)
    ]
    suppressed = [
        {
            "token_id": int(i), "token": decode_token(i),
            "delta_logit": round(-float(v), 6),
        }
        for i, v in zip(down.indices, down.values)
    ]
    return promoted, suppressed


def matched_control_neurons(clean_runs, cases, located, n=3):
    """Match target on c_proj output norm and natural completion activation."""
    weight = zp.mdl.transformer.h[zp.ELL].mlp.c_proj.weight.detach()
    width = clean_runs[cases[0].case_id]["act_original"].shape[-1]
    if weight.shape[0] == width:
        directions = weight
    elif weight.shape[1] == width:
        directions = weight.T
    else:
        raise RuntimeError(
            f"cannot orient c_proj weight {tuple(weight.shape)} for width {width}"
        )
    if directions.shape[0] != width:
        raise RuntimeError(
            f"cannot orient c_proj weight {tuple(weight.shape)} for width {width}"
        )
    norms = torch.linalg.vector_norm(directions.float(), dim=1).cpu()
    activations = torch.stack([
        clean_runs[c.case_id]["act_original"][located[c.case_id].intervention_pos]
        .abs().cpu()
        for c in cases
    ]).mean(dim=0)
    target_norm = float(norms[TARGET])
    target_act = float(activations[TARGET])
    rows = []
    excluded = {TARGET, RIVAL, 1888, 2073, 2566}
    for neuron in range(width):
        if neuron in excluded:
            continue
        norm = float(norms[neuron])
        act = float(activations[neuron])
        distance = abs(math.log((norm + EPS) / (target_norm + EPS))) + \
            abs(math.log((act + EPS) / (target_act + EPS)))
        rows.append((distance, neuron, norm, act))
    rows.sort()
    selected = rows[:n]
    selection_rows = [{
        "selection_type": "MATCHED_CONTROL",
        "neuron": neuron,
        "distance": round(distance, 6),
        "output_direction_norm": round(norm, 6),
        "mean_abs_natural_activation": round(act, 6),
        "target_output_direction_norm": round(target_norm, 6),
        "target_mean_abs_natural_activation": round(target_act, 6),
        "selection_rule": "nearest log-ratio distance on norm and mean abs activation",
    } for distance, neuron, norm, act in selected]
    selection_rows.append({
        "selection_type": "DECLARED_RIVAL",
        "neuron": RIVAL,
        "distance": "",
        "output_direction_norm": round(float(norms[RIVAL]), 6),
        "mean_abs_natural_activation": round(float(activations[RIVAL]), 6),
        "target_output_direction_norm": round(target_norm, 6),
        "target_mean_abs_natural_activation": round(target_act, 6),
        "selection_rule": "v11 recurring winner declared before v12 inference",
    })
    return [x[1] for x in selected], selection_rows


def natural_treatments(control_neurons):
    out = [
        Treatment("CLEAN", None, "DECLARED", "CLEAN", declared_role="baseline"),
        Treatment("CLEAN_REPEAT", None, "DECLARED", "RERUN",
                  declared_role="deterministic duplicate"),
        Treatment("SHAM", TARGET, "DECLARED", "SHAM", declared_role="hook identity"),
        Treatment("ZERO_541", TARGET, "DECLARED", "SET", 0.0,
                  declared_role="natural ablation"),
        Treatment("MEAN_REPLACE_541", TARGET, "DECLARED", "CONTROL_MEAN",
                  declared_role="matched control-state replacement"),
        Treatment("SCALE_025_541", TARGET, "DECLARED", "SCALE", 0.25,
                  declared_role="dose reduction"),
        Treatment("SCALE_050_541", TARGET, "DECLARED", "SCALE", 0.50,
                  declared_role="dose reduction"),
        Treatment("SCALE_075_541", TARGET, "DECLARED", "SCALE", 0.75,
                  declared_role="dose reduction"),
        Treatment("PREVIOUS_ZERO_541", TARGET, "PREVIOUS", "SET", 0.0,
                  declared_role="position control"),
        Treatment(f"ZERO_{RIVAL}", RIVAL, "DECLARED", "SET", 0.0,
                  declared_role="declared rival-neuron control"),
    ]
    for i, neuron in enumerate(control_neurons, 1):
        out.append(Treatment(
            f"ZERO_MATCHED_{i}_{neuron}", neuron, "DECLARED", "SET", 0.0,
            declared_role="matched-neuron ablation control",
        ))
    return out


def steering_treatments():
    return [
        Treatment("CLEAN", None, "DECLARED", "CLEAN", declared_role="baseline"),
        Treatment("CLEAN_REPEAT", None, "DECLARED", "RERUN",
                  declared_role="deterministic duplicate"),
        Treatment("SHAM", TARGET, "DECLARED", "SHAM", declared_role="hook identity"),
        Treatment("SET_1_541", TARGET, "DECLARED", "SET", 1.0,
                  declared_role="realistic fixed dose"),
        Treatment("SET_2_541", TARGET, "DECLARED", "SET", 2.0,
                  declared_role="realistic fixed dose"),
        Treatment("SET_3_541", TARGET, "DECLARED", "SET", 3.0,
                  declared_role="realistic fixed dose"),
        Treatment("SET_4_541", TARGET, "DECLARED", "SET", 4.0,
                  declared_role="realistic fixed dose"),
        Treatment("DONOR_541", TARGET, "DECLARED", "DONOR",
                  declared_role="matched donor patch"),
        Treatment("SHUFFLED_DONOR_541", TARGET, "DECLARED", "SHUFFLED_DONOR",
                  declared_role="donor identity control"),
        Treatment("PREVIOUS_DONOR_541", TARGET, "PREVIOUS", "DONOR",
                  declared_role="position control"),
    ]


def resolve_value(treatment, case, clean, clean_runs, located, control_mean,
                  donor_order):
    if treatment.value_mode in {"CLEAN", "RERUN"}:
        return None, None, ""
    position = (located[case.case_id].intervention_pos
                if treatment.position_mode == "DECLARED"
                else located[case.case_id].previous_pos)
    original = float(clean["act_original"][position, treatment.neuron])
    donor_used = ""
    if treatment.value_mode == "SHAM":
        value = original
    elif treatment.value_mode == "SET":
        value = float(treatment.value)
    elif treatment.value_mode == "SCALE":
        value = original * float(treatment.value)
    elif treatment.value_mode == "CONTROL_MEAN":
        value = float(control_mean)
    elif treatment.value_mode in {"DONOR", "SHUFFLED_DONOR"}:
        donor_id = case.donor_id
        if treatment.value_mode == "SHUFFLED_DONOR":
            donor_id = donor_order[(donor_order.index(case.donor_id) + 1)
                                   % len(donor_order)]
        donor_used = donor_id
        donor_loc = located[donor_id]
        value = float(clean_runs[donor_id]["act_original"][
            donor_loc.intervention_pos, TARGET
        ])
    else:
        raise ValueError(f"unknown value mode {treatment.value_mode}")
    return position, value, donor_used


def result_row(case, loc, treatment, run, clean, clean_metrics, position,
               donor_used):
    metrics = outcome_metrics(run, loc)
    first_pred = loc.outcome_positions[0] - 1
    clean_logits = clean["logits"][first_pred]
    changed_logits = run["logits"][first_pred]
    clean_logp = torch.log_softmax(clean_logits, dim=-1)
    changed_logp = torch.log_softmax(changed_logits, dim=-1)
    clean_prob = clean_logp.exp()
    kl = float((clean_prob * (clean_logp - changed_logp)).sum())
    promoted, suppressed = top_token_changes(clean_logits, changed_logits)
    intervention_pos = (loc.intervention_pos if position is None else position)
    l5_l2, l5_rel, l5_cos = vector_change(
        clean["hidden_l5"][intervention_pos], run["hidden_l5"][intervention_pos]
    )
    final_l2, final_rel, final_cos = vector_change(
        clean["hidden_final"][intervention_pos],
        run["hidden_final"][intervention_pos],
    )
    original_scalar = (
        float(clean["act_original"][intervention_pos, treatment.neuron])
        if treatment.neuron is not None else ""
    )
    actual_set = (
        float(run["act_modified"][intervention_pos, treatment.neuron])
        if treatment.neuron is not None else ""
    )
    return {
        "case_id": case.case_id,
        "branch": case.branch,
        "family": case.family,
        "treatment": treatment.treatment,
        "declared_role": treatment.declared_role,
        "text": case.text,
        "prefix": case.prefix,
        "outcome": case.outcome,
        "n_tok": len(loc.body_ids),
        "tokens": json_tokens(loc.body_ids),
        "token_ids": json.dumps(loc.body_ids),
        "outcome_tokens": json_tokens(loc.outcome_ids),
        "outcome_token_ids": json.dumps(loc.outcome_ids),
        "declared_intervention_pos": loc.intervention_pos,
        "actual_intervention_pos": "" if position is None else position,
        "position_mode": treatment.position_mode,
        "intervened_neuron": "" if treatment.neuron is None else treatment.neuron,
        "value_mode": treatment.value_mode,
        "original_activation": original_scalar,
        "set_activation": actual_set,
        "activation_change": (
            round(float(actual_set) - float(original_scalar), 6)
            if treatment.neuron is not None else ""
        ),
        "declared_donor_id": case.donor_id,
        "donor_used": donor_used,
        "clean_outcome_logprob": round(clean_metrics["outcome_logprob"], 6),
        "outcome_logprob": round(metrics["outcome_logprob"], 6),
        "outcome_logprob_change": round(
            metrics["outcome_logprob"] - clean_metrics["outcome_logprob"], 12
        ),
        "clean_first_outcome_prob": round(clean_metrics["first_outcome_prob"], 9),
        "first_outcome_prob": round(metrics["first_outcome_prob"], 9),
        "first_outcome_prob_change": round(
            metrics["first_outcome_prob"] - clean_metrics["first_outcome_prob"], 9
        ),
        "clean_first_outcome_rank": clean_metrics["first_outcome_rank"],
        "first_outcome_rank": metrics["first_outcome_rank"],
        "first_outcome_rank_change": (
            metrics["first_outcome_rank"] - clean_metrics["first_outcome_rank"]
        ),
        "clean_greedy_token": clean_metrics["first_greedy_token"],
        "greedy_token": metrics["first_greedy_token"],
        "greedy_changed": int(
            metrics["first_greedy_id"] != clean_metrics["first_greedy_id"]
        ),
        "kl_clean_to_intervened": round(kl, 12),
        "clean_first_entropy": round(clean_metrics["first_entropy"], 6),
        "first_entropy": round(metrics["first_entropy"], 6),
        "first_entropy_change": round(
            metrics["first_entropy"] - clean_metrics["first_entropy"], 6
        ),
        "first_logit_l2": round(float(torch.linalg.vector_norm(
            changed_logits - clean_logits
        )), 12),
        "max_abs_logit_change": round(float(
            (changed_logits - clean_logits).abs().max()
        ), 12),
        "promoted_tokens": json.dumps(promoted, ensure_ascii=False),
        "suppressed_tokens": json.dumps(suppressed, ensure_ascii=False),
        "hidden_l5_delta_l2": round(l5_l2, 6),
        "hidden_l5_relative_delta": round(l5_rel, 9),
        "hidden_l5_cosine": round(l5_cos, 9),
        "hidden_final_delta_l2": round(final_l2, 6),
        "hidden_final_relative_delta": round(final_rel, 9),
        "hidden_final_cosine": round(final_cos, 9),
        "outcome_token_metrics": json.dumps(metrics["token_rows"], ensure_ascii=False),
        "notes": case.notes,
    }


def token_effect_rows(result):
    if result["treatment"] in {"CLEAN", "CLEAN_REPEAT", "SHAM"}:
        return []
    out = []
    for direction, field in (
        ("PROMOTED", "promoted_tokens"),
        ("SUPPRESSED", "suppressed_tokens"),
    ):
        for rank, item in enumerate(json.loads(result[field]), 1):
            out.append({
                "case_id": result["case_id"],
                "branch": result["branch"],
                "family": result["family"],
                "treatment": result["treatment"],
                "direction": direction,
                "rank": rank,
                **item,
            })
    return out


def summary_rows(results):
    groups = defaultdict(list)
    for row in results:
        groups[(row["branch"], row["treatment"], row["declared_role"])].append(row)
    out = []
    for (branch, treatment, role), members in sorted(groups.items()):
        def vals(field):
            return [float(x[field]) for x in members]
        out.append({
            "branch": branch,
            "treatment": treatment,
            "declared_role": role,
            "n": len(members),
            "outcome_logprob_change_mean": round(statistics.mean(
                vals("outcome_logprob_change")), 6),
            "outcome_logprob_change_median": round(statistics.median(
                vals("outcome_logprob_change")), 6),
            "first_outcome_prob_change_mean": round(statistics.mean(
                vals("first_outcome_prob_change")), 9),
            "kl_mean": round(statistics.mean(vals("kl_clean_to_intervened")), 9),
            "kl_median": round(statistics.median(vals("kl_clean_to_intervened")), 9),
            "first_logit_l2_mean": round(statistics.mean(
                vals("first_logit_l2")), 6),
            "hidden_final_delta_l2_mean": round(statistics.mean(
                vals("hidden_final_delta_l2")), 6),
            "greedy_changed_n": sum(int(x["greedy_changed"]) for x in members),
            "positive_outcome_logprob_change_n": sum(
                float(x["outcome_logprob_change"]) > 0 for x in members
            ),
            "member_case_ids": json.dumps([x["case_id"] for x in members]),
        })
    return out


def preflight_rows(cases, located, clean_runs, control_selection, control_mean):
    out = list(control_selection)
    out.append({
        "selection_type": "CONTROL_MEAN",
        "neuron": TARGET,
        "control_mean_activation": round(control_mean, 6),
        "selection_rule": "mean clean 541 activation across completion-injection controls",
    })
    for case in cases:
        loc = located[case.case_id]
        clean = clean_runs[case.case_id]
        metrics = outcome_metrics(clean, loc)
        out.append({
            "selection_type": "CASE",
            "case_id": case.case_id,
            "branch": case.branch,
            "family": case.family,
            "prefix": case.prefix,
            "outcome": case.outcome,
            "text": case.text,
            "donor_id": case.donor_id,
            "n_tok": len(loc.body_ids),
            "tokens": json_tokens(loc.body_ids),
            "token_ids": json.dumps(loc.body_ids),
            "intervention_pos": loc.intervention_pos,
            "intervention_token": (
                "<BOS>" if loc.intervention_pos == 0
                else decode_token(([zp.BOS] + list(loc.body_ids))[loc.intervention_pos])
            ),
            "previous_pos": loc.previous_pos,
            "outcome_positions": json.dumps(loc.outcome_positions),
            "outcome_tokens": json_tokens(loc.outcome_ids),
            "outcome_token_ids": json.dumps(loc.outcome_ids),
            "clean_541_activation": round(float(
                clean["act_original"][loc.intervention_pos, TARGET]
            ), 6),
            "clean_outcome_logprob": round(metrics["outcome_logprob"], 6),
            "clean_first_outcome_prob": round(metrics["first_outcome_prob"], 9),
            "clean_first_outcome_rank": metrics["first_outcome_rank"],
            "notes": case.notes,
        })
    return out


def validate(cases, located, results, controls):
    failures = []
    if len({c.case_id for c in cases}) != len(cases):
        failures.append("duplicate case IDs")
    if len(set(controls)) != 3 or TARGET in controls or RIVAL in controls:
        failures.append(f"invalid matched controls: {controls}")
    for case in cases:
        loc = located[case.case_id]
        if len(loc.body_ids) + 1 > zp.N_CTX:
            failures.append(f"{case.case_id}: outside dom(D)")
    sham = [r for r in results if r["treatment"] == "SHAM"]
    if len(sham) != len(cases):
        failures.append(f"sham rows {len(sham)} != cases {len(cases)}")
    for row in sham:
        if (abs(float(row["max_abs_logit_change"])) > SHAM_TOL or
                abs(float(row["outcome_logprob_change"])) > SHAM_TOL):
            failures.append(
                f"{row['case_id']}: sham mismatch "
                f"{row['max_abs_logit_change']}, {row['outcome_logprob_change']}"
            )
    repeats = [r for r in results if r["treatment"] == "CLEAN_REPEAT"]
    if len(repeats) != len(cases):
        failures.append(f"clean-repeat rows {len(repeats)} != cases {len(cases)}")
    for row in repeats:
        if (abs(float(row["max_abs_logit_change"])) > SHAM_TOL or
                abs(float(row["outcome_logprob_change"])) > SHAM_TOL):
            failures.append(
                f"{row['case_id']}: deterministic repeat mismatch "
                f"{row['max_abs_logit_change']}, {row['outcome_logprob_change']}"
            )
    clean = [r for r in results if r["treatment"] == "CLEAN"]
    if len(clean) != len(cases):
        failures.append(f"clean rows {len(clean)} != cases {len(cases)}")
    donors = {c.case_id for c in NATURAL_CASES}
    for case in cases:
        if case.donor_id and case.donor_id not in donors:
            failures.append(f"{case.case_id}: unknown donor {case.donor_id}")
    return failures


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results")
    parser.add_argument(
        "--smoke", action="store_true",
        help="two cases per branch; full treatment/validation machinery",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = next(zp.mdl.parameters()).device
    if device.type != "cuda":
        raise SystemExit(f"znou_probe model is on {device}; v12 requires CUDA")

    natural = list(NATURAL_CASES[:2] if args.smoke else NATURAL_CASES)
    injection = list(INJECTION_CASES[:2] if args.smoke else INJECTION_CASES)
    precompletion = list(PRECOMPLETION_CASES[:2] if args.smoke else PRECOMPLETION_CASES)
    # Donor cases must always be available in the clean cache even if they are
    # not part of a smoke branch's reported natural cases.
    donor_ids = {c.donor_id for c in injection + precompletion if c.donor_id}
    donor_cases = [c for c in NATURAL_CASES if c.case_id in donor_ids]
    cases = natural + injection + precompletion
    cache_cases = list({c.case_id: c for c in cases + donor_cases}.values())
    located = {c.case_id: locate(c) for c in cache_cases}

    print(f"model device: {device}")
    print(f"reported cases: {len(cases)}")
    print(f"clean-cache cases: {len(cache_cases)}")
    for branch, n in sorted(Counter(c.branch for c in cases).items()):
        print(f"  {branch:<28} {n}")

    runner = ForwardRunner()
    clean_runs = {}
    for i, case in enumerate(cache_cases, 1):
        clean_runs[case.case_id] = runner.run(case, located[case.case_id])
        if i == 1 or i == len(cache_cases):
            print(f"clean [{i}/{len(cache_cases)}]")

    controls, control_selection = matched_control_neurons(
        clean_runs, natural, located
    )
    control_mean = statistics.mean(
        float(clean_runs[c.case_id]["act_original"][
            located[c.case_id].intervention_pos, TARGET
        ]) for c in injection
    )
    donor_order = sorted({c.donor_id for c in injection + precompletion})
    print(f"matched control neurons: {controls}")
    print(f"completion-control mean 541: {control_mean:.6f}")

    results = []
    effects = []
    treatment_map = {
        "NATURAL_COMPLETION": natural_treatments(controls),
        "COMPLETION_INJECTION": steering_treatments(),
        "PRECOMPLETION_STEERING": steering_treatments(),
    }
    total = sum(len(treatment_map[c.branch]) for c in cases)
    count = 0
    for case in cases:
        loc = located[case.case_id]
        clean = clean_runs[case.case_id]
        clean_metrics = outcome_metrics(clean, loc)
        for treatment in treatment_map[case.branch]:
            count += 1
            position, value, donor_used = resolve_value(
                treatment, case, clean, clean_runs, located, control_mean,
                donor_order,
            )
            if treatment.value_mode == "CLEAN":
                run = clean
            elif treatment.value_mode == "RERUN":
                run = runner.run(case, loc)
            else:
                run = runner.run(
                    case, loc, neuron=treatment.neuron,
                    position=position, set_value=value,
                )
            row = result_row(
                case, loc, treatment, run, clean, clean_metrics,
                position, donor_used,
            )
            results.append(row)
            effects.extend(token_effect_rows(row))
            if count == 1 or count % 50 == 0 or count == total:
                print(f"intervention [{count}/{total}]")

    summaries = summary_rows(results)
    preflight = preflight_rows(
        cases, located, clean_runs, control_selection, control_mean
    )
    failures = validate(cases, located, results, controls)
    sham_max = max(float(r["max_abs_logit_change"])
                   for r in results if r["treatment"] == "SHAM")
    print("\nvalidation")
    print(f"  intervention rows           {len(results)}")
    print(f"  token-effect rows           {len(effects)}")
    print(f"  sham max logit difference   {sham_max:.9g}")
    print(f"  validation failures         {len(failures)}")

    outdir = Path(args.outdir)
    suffix = "smoke" if args.smoke else ""
    outputs = []
    for name, data in (
        ("probe_v12_preflight", preflight),
        ("probe_v12_interventions", results),
        ("probe_v12_token_effects", effects),
        ("probe_v12_summary", summaries),
    ):
        recorder = zp.Recorder(name, outdir=outdir)
        recorder.extend(data)
        outputs.append(recorder.write(suffix=suffix))
    print("\noutputs")
    for path in outputs:
        print(f"  {path}")
    if failures:
        raise SystemExit("v12 validation failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
