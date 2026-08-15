# STRING ITERATION PROBES v13–v14 — RESULTS

## L5-N38: predicative-property response with a causal cold tilt

Status: consolidated preliminary result note. This report records v13 and its
focused v14 calibration repair as one inference sequence. Tags follow the
project convention: `MEASURED`, `DERIVED`, `INTERPRETATION`, and `PROPOSAL`.

Earlier recovered experiments associated GPT-2 Small layer-5 neuron 38 with
`cold`-related generation. Those experiments repeatedly added `+10` to neuron
teams during greedy generation. They produced striking cold outputs but did not
isolate neuron 38, did not use naturally calibrated values, and did not
separate an immediate logit effect from autoregressive feedback.

V13 reconstructed that observation under a stricter design. It first mapped
N38 across lexical, syntactic, temperature, polysemy, and control strings. It
then isolated N38 from the old specialist team, compared one-step and sustained
generation, and introduced signed teacher-forced steering. V13 found that N38
is almost inactive on isolated `cold` and weak on attributive or nominal uses,
while responding strongly at predicative property completions. Its causal
effect broadly supported tested adjective completions and consistently favoured
`cold` over matched alternatives.

V13 also exposed its own calibration problem. Values sampled from the natural
activation distribution were used as **increments**. Adding 2.87 to an existing
activation of 0.68–2.02 produced final values of 3.55–4.89, exceeding the
observed natural maximum of 3.43. V14 repaired this by setting N38 to absolute
values from zero through the observed maximum. The cold-relative effect
survived in all five matched prefix pairs and increased monotonically across
the entire absolute ladder.

The consolidated working phenotype is:

> **L5-N38 responds strongly in predicative-property contexts and contributes
> a graded, context-conditioned cold preference within that broader response.
> Increasing N38 shifts matched next-token probability ratios toward `cold`;
> removing it shifts those ratios away from `cold`.**

This supports a real cold-aligned causal component. It does not support a
context-free `cold` detector, an autonomous cold generator, or a claim that
the full internal state created by v14 occurs naturally before adjective
selection.

---

## 1. Run provenance — MEASURED

Both probes used the same pinned model state:

```text
python          3.11.15
torch           2.9.0+cu128
transformers    4.57.1
device          cuda:0
gpu             NVIDIA GeForce RTX 5060 Laptop GPU
revision        607a30d783dfa663caf39e06633721c8d4cfcd7e
layer           5
tf32_matmul     False
tf32_cudnn      False
theta_sha256    113687a222f8cf98039222c27b39aaf716493e5e8c1db94ea4e6544e0814088c
```

### 1.1 V13 outputs

```text
probe_v13_preflight_20260815-105024.tsv
probe_v13_phenotype_20260815-105025.tsv
probe_v13_trace_20260815-105025.tsv
probe_v13_certificates_20260815-105025.tsv
probe_v13_logit_direction_20260815-105025.tsv
probe_v13_interventions_20260815-105025.tsv
probe_v13_token_effects_20260815-105025.tsv
probe_v13_factorial_20260815-105025.tsv
probe_v13_summary_20260815-105025.tsv
probe_v13_generation_20260815-105025.tsv
probe_v13_generation_steps_20260815-105025.tsv
```

| Output | Rows | Unit |
|---|---:|---|
| Preflight | 41 | neuron selections, dose declarations, and 30 cases |
| Phenotype | 27 | one completed string per row |
| Trace | 113 | one token position per row |
| Certificates | 27 | one minimisation result per phenotype string |
| Direct logit direction | 20 | top ten promoted and suppressed heuristic tokens |
| Interventions | 456 | one case × treatment evaluation |
| Token effects | 7,322 | promoted and suppressed token records |
| Factorial | 30 | team-minus-single contrasts |
| Summary | 30 | branch × treatment aggregates |
| Generation | 64 | eight prompts × eight treatments |
| Generation steps | 960 | one generated token per trajectory step |

### 1.2 V14 outputs

```text
probe_v14_preflight_20260815-111604.tsv
probe_v14_interventions_20260815-111605.tsv
probe_v14_token_effects_20260815-111605.tsv
probe_v14_paired_20260815-111605.tsv
probe_v14_summary_20260815-111605.tsv
```

| Output | Rows | Unit |
|---|---:|---|
| Preflight | 34 | four dose declarations plus 30 calibration/steering cases |
| Interventions | 156 | twelve cases × thirteen treatments |
| Token effects | 2,400 | promoted and suppressed token records |
| Paired | 60 | five pairs plus aggregate for ten non-identity treatments |
| Summary | 13 | one row per treatment |

The v14 smoke run contained 52 intervention rows. Every field in every shared
smoke/full row was identical. `MEASURED`

All TSVs use literal quotation marks rather than CSV quoting. Generic readers
should use tab separation with `quoting=csv.QUOTE_NONE` or an equivalent
literal-TSV mode.

---

## 2. Background: what the recovered experiments established

The old intervention suite used neuron groups including:

```text
Specialists Only       [38, 1103, 2094]
Minimal Coordinators   [1508, 234]
Full Roster            eleven combined neurons
```

At each greedy generation step, the hook added `+10` to the selected neurons
at the current final token. This produced outputs such as:

```text
The summer sun is fiery, the season is cold ...
It was frigid, and I was cold. I was cold ...
The desert sun is beating down, the air is cold ...
```

These were genuine causal trajectory changes: the model generated different
tokens because internal activations were changed. Their attribution remained
ambiguous for three reasons.

First, N38 was never tested alone in the specialist condition. Second, `+10`
was not compared with its natural activation distribution. Third, the hook
acted again after each generated token. Once an altered token entered the
context, the next intervention acted on an already altered sequence. Repeated
`cold` or strong thematic drift could therefore arise from a modest first-step
bias, team interaction, greedy threshold crossing, autoregressive feedback, or
some combination.

V13 treats those recovered files as hypothesis-generating provenance. It does
not use their generated prose as confirmatory evidence.

---

## 3. Terms

**Target activation** is the scalar post-GELU activation of layer-5 neuron 38
at one token position.

**Position winner** is the neuron with the largest activation at a particular
token. This differs from the Atlas **destination**, which compares every
neuron's maximum activation anywhere in the complete string. N38 can win at
the adjective while another neuron retains the whole-string destination due to
an earlier token.

**Predicative property** means a property expressed as a predicate, commonly
after a copula or perception verb:

```text
It was cold
The room was dark
Her skin felt cold
```

The phrase describes the test panel. It does not assert that GPT-2 represents
a formal linguistic variable with this name.

**Attributive use** places an adjective inside a noun phrase, as in `a cold
start`. **Nominal use** treats `cold` as a noun, as in `caught a cold`.

**Ablation** sets the chosen activation to zero at one position. It does not
delete the neuron globally.

**ADD steering** adds a declared increment to the clean activation. A dose of
2.87 applied to a clean value of 1.10 produces 3.97.

**SET steering** replaces the clean activation with a declared absolute value.
A set value of 2.87 produces 2.87 regardless of the clean value.

**Cold-relative advantage** is:

```text
change in log P(cold | prefix)
  − change in log P(matched alternative | same prefix)
```

The two outcomes share the identical prefix, model state, and intervention.
Their common softmax normalization cancels, so this is also the intervention's
change in their log-probability ratio. Positive values shift the ratio toward
`cold`; negative values shift it away.

**Within the observed scalar range** means that the final N38 value does not
exceed the observed calibration maximum of 3.433706. It does not mean the
complete 3,072-dimensional activation state was naturally observed at that
token position.

---

## 4. V13 design

### 4.1 Phenotype and minimisation

The panel separated lexical identity, predicative framing, synonymy,
temperature opposition, polysemy, and attributive use:

```text
It was cold               cold
It is cold                 cold shoulder
The air was cold           He caught a cold
His voice was cold         The engine made a cold start
It was freezing            It was warm
It was chilly              It was hot
```

Every token received an N38 trace. The Atlas destination and local token winner
were both recorded. Exhaustive contiguous source-token minimisation searched
for the shortest span that retained N38 as the Atlas destination.

### 4.2 Natural ablation

Eighteen completed predicates were given declared continuations. N38 was
scaled to 75%, 50%, and 25%, then zeroed. Controls included clean repeat, sham,
previous-position ablation, isolated old team members, the three-neuron team,
two older cold-constellation neurons, and three automatically matched neurons.

The matched neurons—2761, 727, and 2394—were selected before outcome analysis
by proximity to N38 in output-direction norm and mean absolute activation.

This branch measures what N38 does to prediction *after* the completed
predicate. It does not directly measure whether N38 helps choose the adjective.

### 4.3 Pre-adjective steering

Twelve outcomes were scored after prefixes such as:

```text
It was       → cold / warm
The air is   → cold / hot
The room was → cold / dark
The season is → cold / warm
The world is → cold / strange
```

The five displayed pairs are the primary matched contrasts. `Her skin felt →
cold` and `His voice was → cold` broadened the cold panel without paired
alternatives.

N38 received positive and negative additive doses. Neurons 1103, 2094, 1508,
and 234 were tested individually, while `[38, 1103, 2094]` reproduced the old
specialist grouping at a shared N38-calibrated increment.

### 4.4 Generation feedback

Eight recovered prompts were evaluated under clean generation, N38 alone, and
the three-neuron team. One-step interventions acted only on the first generated
token. Sustained interventions acted at every greedy step. Natural-calibrated
increments and legacy `+10` were kept separate.

---

## 5. V13 integrity and calibration — MEASURED

Clean repeats and sham hooks were bit-identical to clean runs on every recorded
metric. The measurement chain therefore passed the same exact identity checks
used in v12.

The declared N38 dose values were:

| Label | Value |
|---|---:|
| Low | 2.076799 |
| Median | 2.650153 |
| High | 2.870399 |
| Maximum | 3.433706 |

They were nearest-rank summaries of positive clean N38 activations at the
declared completion position in the eighteen natural calibration cases.

V13 used these values as increments in its steering branch. Consequently:

| V13 treatment | Pairs within observed maximum | Final values across five pairs |
|---|---:|---|
| Add low | 4/5 | 2.76–4.10 |
| Add median | 1/5 | 3.33–4.67 |
| Add high | 0/5 | 3.55–4.89 |
| Add maximum | 0/5 | 4.11–5.46 |

The label “natural-range dose” correctly described the size of the increment,
but not the resulting activation. This was a design/terminology error rather
than a computational failure. V14 was written specifically to repair it.

---

## 6. V13 observational phenotype

### 6.1 N38 is not a context-free `cold` detector — MEASURED

| String | N38 peak | Atlas destination |
|---|---:|---:|
| `It was cold` | **3.434** | **38** |
| `The air was cold` | 3.143 | 1888 |
| `It is cold` | 2.977 | 1888 |
| `It felt cold` | 2.890 | 1888 |
| `It was freezing` | 2.870 | 1888 |
| `It was chilly` | 2.835 | 1888 |
| `It was warm` | 2.831 | 1888 |
| `His voice was cold` | 2.650 | 1888 |
| `He caught a cold` | 1.009 | 1888 |
| `The Cold War` | 0.643 | 1888 |
| `The engine made a cold start` | 0.289 | 1888 |
| `cold` | 0.116 whole-string peak; −0.150 on `cold` token | 1888 |
| ` cold` | 0.116 whole-string peak | 1888 |
| `temperature` | 0.116 whole-string peak | 1888 |

Isolated `cold` does not activate N38 positively on the lexical token. Nominal,
proper-name, and attributive uses are much weaker than predicative uses. Warm
and other temperature predicates can be strong. The evidence therefore points
to a contextual conjunction rather than direct word detection.

### 6.2 Local position winner and global destination are different

For `It was cold`, the trace was:

| Position | Token | N38 activation | Position winner |
|---:|---|---:|---:|
| 0 | `<BOS>` | 0.116 | 2256 |
| 1 | `It` | 0.778 | 1888 |
| 2 | ` was` | 2.024 | 1888 |
| 3 | ` cold` | **3.434** | **38** |

For `It was warm`, the first three values are identical and ` warm` raises N38
to 2.831, again making 38 the local adjective-position winner. The whole-string
destination remains 1888 because neuron 1888 had already reached 3.132 on
`It`.

`It was cold` is the sole N38 destination because ` cold` pushes the broader
predicative response high enough to exceed that earlier competitor. Atlas
destination made the route appear more exclusively cold-related than the token
trace supports. `INTERPRETATION`

### 6.3 Minimisation

Across all 27 phenotype strings, the only destination-preserving certificate
was:

```text
It was cold
```

It could not be shortened under the exhaustive contiguous source-token search.
This is evidence for context dependence under the Atlas destination criterion.
It is not proof that all three tokens form a unique mechanistic feature; the
criterion also depends on competition with every other layer-5 neuron.

---

## 7. V13 causal decomposition

### 7.1 Pre-adjective N38 steering broadly supports properties

At the v13 high additive increment, N38 increased all twelve declared outcomes:

| Aggregate | Value |
|---|---:|
| Mean outcome log-probability change | +0.542 |
| Median change | +0.526 |
| Positive cases | 12/12 |
| Mean KL divergence | 0.0253 |
| Greedy-token changes | 2/12 |

Negative high steering lowered every declared outcome, with mean change
−0.621. N38 therefore has a signed causal effect on this selected property-
completion panel. It is not exclusively cold-promoting.

### 7.2 N38 contains the consistent cold-relative component

Within identical prefixes, the high additive intervention favoured `cold` over
every matched alternative:

| Pair | Cold-relative advantage |
|---|---:|
| `It was`: cold vs warm | +0.180 |
| `The air is`: cold vs hot | +0.243 |
| `The room was`: cold vs dark | +0.209 |
| `The season is`: cold vs warm | +0.309 |
| `The world is`: cold vs strange | +0.179 |
| **Mean** | **+0.224** |

The advantage increased monotonically with additive dose:

| Additive increment | Mean cold-relative advantage |
|---:|---:|
| 2.077 | +0.162 |
| 2.650 | +0.207 |
| 2.870 | +0.224 |
| 3.434 | +0.266 |

Negative steering reversed the relationship, producing a mean advantage of
−0.189. `MEASURED`

### 7.3 Old specialist-team decomposition

| Intervention at shared high increment | Mean effect on all outcomes | Mean cold-relative advantage |
|---|---:|---:|
| N38 | +0.542 | **+0.224** |
| N1103 | +0.114 | −0.008 |
| N2094 | +0.133 | −0.005 |
| N1508 | +0.013 | −0.005 |
| N234 | +0.037 | +0.065 |
| N38+N1103+N2094 | +0.736 | +0.212 |

Neurons 1103 and 2094 broadly promoted the selected outcomes without supplying
a consistent cold-over-control advantage. N38 supplied most of the team's
directional cold tilt. Neuron 234 showed a smaller possible cold-aligned effect
that was less consistent across individual pairs.

The three-neuron team effect was mildly subadditive on the log-probability
scale: mean team-minus-sum residual −0.052. The ablation residual was much
smaller at −0.006. Because the downstream network and log-probability scale are
nonlinear, these are descriptive non-additivity measures rather than clean
biological-style interaction coefficients.

### 7.4 Natural ablation after the predicate

Zeroing N38 at the completed predicate produced:

| Metric | Value |
|---|---:|
| Mean declared-continuation change | −0.096 |
| Median change | −0.037 |
| Mean KL divergence | 0.0123 |
| Positive declared-continuation cases | 7/18 |
| Greedy-token changes | 1/18 |

The signs were heterogeneous. Examples included:

```text
It is cold → outside          −0.369
It was freezing → outside     −0.511
It was cool → in the shade    −0.676

It felt cold → to the touch   +0.319
His voice was cold → and flat +0.212
Her skin felt cold → and clammy +0.184
```

N38's normalized mean effect was −0.0141 outcome-log-probability units per unit
of layer-5 residual perturbation norm. The matched controls were −0.00588,
−0.00111, and −0.00131. N38 therefore remained more aligned with this selected
continuation panel after perturbation-size normalization, while the
continuation-specific signs resisted a single semantic summary.

This branch concerns prediction after the adjective. The pre-adjective paired
branch is the cleaner evidence about choosing `cold` relative to alternatives.

---

## 8. V13 generation and the recovered effect

### 8.1 One-step N38 steering stayed below the greedy threshold

At the high N38-calibrated increment, a one-step N38 intervention changed the
greedy trajectory in 0/8 prompts. The teacher-forced probability shifts were
real but insufficient to change the argmax.

The three-neuron team changed 1/8 one-step trajectories:

```text
The summer sun is fiery, the season is
clean:  warm ...
team:   cold ...
```

This is consistent with N38's cold-relative tilt combining with broader support
from its companions to cross a local greedy boundary.

### 8.2 Sustained steering changed trajectories without generally making them cold

| Treatment | Exact clean matches | Trajectories containing a newly generated cold-family lexeme |
|---|---:|---:|
| N38 one-step calibrated | 8/8 | 0/8 |
| N38 sustained calibrated | 1/8 | 0/8 |
| Team one-step calibrated | 7/8 | 1/8 |
| Team sustained calibrated | 1/8 | 1/8 |
| N38 sustained legacy +10 | 0/8 | 1/8 |
| Team sustained legacy +10 | 0/8 | 2/8 |

Natural-calibrated sustained N38 intervention changed seven trajectories,
usually after several clean-matching tokens, without producing a new cold
lexeme. Legacy `+10` changed every trajectory but remained only intermittently
cold-specific.

The recovered cold prose is therefore best understood as a threshold-and-
feedback phenomenon: N38 contributes a cold-relative bias; companions alter
the broader completion distribution; greedy threshold crossings insert new
tokens into the context; repeated interventions then act on the diverged
sequence. `INTERPRETATION`

---

## 9. Why v14 was required

V13 established signed direction, team attribution, and matched cold-relative
effects, but most positive steering values were numerically above the observed
natural maximum after addition. That left two explanations compatible with the
result:

1. N38 has a cold-relative output direction that matters only under unusually
   large activation.
2. The same direction produces a graded effect within values the neuron reaches
   naturally elsewhere in the panel.

V14 held the twelve steering cases and all five primary pairs fixed. It
recomputed the same four calibration values from the same eighteen natural
cases, then used absolute `SET` operations:

```text
SET 0.000000
SET 2.076799
SET 2.650153
SET 2.870399
SET 3.433706
```

The v13 additive ladder was repeated in the same run as a bridge. A previous-
position absolute-high control tested temporal concentration.

---

## 10. V14 integrity — MEASURED

Clean repeat and sham remained exact identities. Every SET row reached its
declared scalar value. Every absolute-value pair was flagged within the
declared zero-to-natural-maximum envelope.

The smoke run used two complete prefix pairs while retaining the full eighteen-
case calibration panel. All 52 shared smoke/full intervention rows matched
field for field. This independently confirms deterministic dose selection and
execution across separate invocations.

---

## 11. V14 absolute-value result

### 11.1 Aggregate paired curve — MEASURED

| Final N38 activation | Mean cold-relative advantage | Probability-ratio multiplier |
|---:|---:|---:|
| 0.000 | −0.087 | 0.917× |
| 2.077 | +0.071 | 1.073× |
| 2.650 | +0.117 | 1.124× |
| 2.870 | +0.134 | 1.144× |
| 3.434 | **+0.179** | **1.196×** |

The multiplier is `exp(cold-relative advantage)`. At the observed maximum, the
intervention increased the cold-to-control probability ratio by approximately
20% relative to each pair's clean ratio. Moving from zero to the maximum
changed the ratio by `exp(0.179 − (−0.087)) ≈ 1.305`, or approximately 31%.
`DERIVED`

### 11.2 All five pair curves are monotone

| Pair | Set 0 | Set low | Set median | Set high | Set maximum |
|---|---:|---:|---:|---:|---:|
| `It was`: cold vs warm | −0.135 | +0.004 | +0.042 | +0.056 | +0.092 |
| `The air is`: cold vs hot | −0.090 | +0.083 | +0.132 | +0.151 | +0.199 |
| `The room was`: cold vs dark | −0.048 | +0.087 | +0.129 | +0.146 | +0.190 |
| `The season is`: cold vs warm | −0.128 | +0.098 | +0.161 | +0.185 | +0.245 |
| `The world is`: cold vs strange | −0.035 | +0.081 | +0.118 | +0.133 | +0.171 |

Every prefix pair became progressively more cold-favouring at every successive
absolute value. The aggregate curve does not conceal a discordant context.

At zero, all twelve declared outcomes became less likely, confirming the broad
property-support effect. Cold declined more than its paired alternative in all
five pairs. At every positive set value, all twelve declared outcomes became
more likely, while cold gained more than its alternative in every pair.

### 11.3 Absolute and additive ladders

| Treatment | Mean outcome change | Mean cold-relative advantage | Pairs within observed maximum |
|---|---:|---:|---:|
| Absolute low | +0.193 | +0.071 | 5/5 |
| Absolute median | +0.303 | +0.117 | 5/5 |
| Absolute high | +0.344 | +0.134 | 5/5 |
| Absolute maximum | +0.445 | +0.179 | 5/5 |
| Additive low | +0.405 | +0.162 | 4/5 |
| Additive median | +0.505 | +0.207 | 1/5 |
| Additive high | +0.542 | +0.224 | 0/5 |
| Additive maximum | +0.632 | +0.266 | 0/5 |

V13 identified the correct direction and overstated its realistic magnitude.
V14 shows that the directional effect persists throughout the observed scalar
range.

### 11.4 Position control

| High absolute intervention | Mean activation change | Mean outcome change | Mean cold-relative advantage | Mean KL |
|---|---:|---:|---:|---:|
| Declared pre-adjective position | +1.783 | +0.344 | +0.134 | 0.0104 |
| Previous token | +2.540 | +0.243 | +0.072 | 0.0061 |

The previous-position intervention was larger in scalar and layer-5 residual
perturbation magnitude, yet produced roughly half the cold-relative advantage.
This supports positional concentration. The control was not null: information
introduced one token earlier can propagate forward, and N38's output direction
remains available at nearby positions.

---

## 12. Combined interpretation

The results support several overlapping components rather than a single-word
label.

### 12.1 Predicative frame sensitivity

N38 rises during copular or property-predication contexts and can become the
local winner on cold, warm, freezing, and other adjective tokens. Isolated,
nominal, and attributive `cold` forms are much weaker. This component explains
why `It was cold` is special while `cold` alone is not.

### 12.2 Broad property-completion support

Positive N38 steering increases probabilities for all tested adjectives;
zeroing or negatively steering it decreases them. This is a panel-conditioned
description, not a claim that every grammatical adjective would behave alike.

### 12.3 Cold-relative direction

Within identical prefix states, increasing N38 consistently favours `cold`
over warm, hot, dark, or strange controls. The effect is signed, graded,
monotone, replicated across five pairs, and present under absolute values no
higher than the observed natural maximum.

### 12.4 Companion and feedback effects

Old team members 1103 and 2094 provide broader support without reproducing
N38's cold-relative tilt. Their combined activity can help N38 cross a greedy
threshold. Sustained intervention then accumulates autoregressive consequences
that are broader and less reliably cold-specific than the immediate paired
logit effect.

The shorthand **predicative-property neuron with a causal cold tilt** is useful
if its scope remains explicit. N38 is one coordinate in a distributed model,
and the tested cold preference is conditional on the selected prefixes and
outcomes.

---

## 13. What is established

Under this GPT-2 Small checkpoint, layer, tokenizer, and declared panel:

- N38 is strongly context-dependent and is not activated by isolated `cold`;
- predicative uses produce much stronger N38 activity than nominal,
  proper-name, or attributive uses in the tested strings;
- `It was cold` is an irreducible three-token Atlas certificate within its
  source-token span;
- N38 has a reproducible causal effect on downstream next-token probabilities;
- this effect broadly supports the tested property completions;
- increasing N38 consistently shifts five matched outcome ratios toward
  `cold`, while setting it to zero shifts all five away from `cold`;
- the absolute cold-relative curve is monotone in every pair from zero through
  the observed natural maximum;
- old specialist-team cold direction is chiefly attributable to N38 under the
  declared decomposition, while companions contribute broader support;
- one-step N38 effects commonly remain below the greedy threshold;
- sustained generation effects reflect both intervention and autoregressive
  feedback.

---

## 14. What is not established

The probes do not establish that:

- N38 is a context-free cold, temperature, or adjective detector;
- N38 alone is necessary or sufficient for naturally generated cold language;
- the five curated pairs are an independent sample from a linguistic
  population;
- the result generalises to other models, checkpoints, layers, tokenizers, or
  languages;
- scalar values observed at predicate completion occur naturally at the
  earlier pre-adjective intervention position;
- a scalar within range makes the entire activation state in distribution;
- N38's role can be understood independently of the remaining 3,071 MLP
  coordinates and later-layer processing;
- the state-free direct-logit screen provides a semantic vocabulary for N38;
- all temperature synonyms or metaphorical cold uses belong to one mechanism.

The last calibration distinction is important. V14 resolves **amplitude
out-of-range**: the absolute values do not exceed N38's observed maximum. It
does not resolve **positional or joint-state out-of-distribution**: a
completion-level scalar was installed before adjective selection inside a
state where the rest of the MLP vector remained target-specific.

---

## 15. Candidate next steps — PROPOSAL

### 15.1 Natural mediation at the pre-adjective position

Measure naturally occurring N38 variation across a larger set of prefixes that
already differ in their probability of `cold`. Test whether natural activation
predicts the cold-versus-control logit ratio after controlling for prefix
identity or within matched templates. This would connect the intervention
curve to naturally varying states.

### 15.2 Structurally matched activation patching

Patch full or partial layer-5 activation vectors between matched pre-adjective
positions rather than inserting a completion-derived scalar alone. Compare:

```text
full donor vector
full donor vector with N38 restored to target clean value
target vector with donor N38 alone
size-matched preregistered control subsets
```

The full-donor versus donor-without-N38 comparison would estimate N38's
incremental role within a coherent transplanted state.

### 15.3 Broader lexical grid

Expand the paired panel with preregistered categories:

```text
cold-family       cold, chilly, freezing, frigid, icy
temperature       warm, hot, cool
sensory property  wet, rough, heavy, bright
evaluative        strange, pleasant, dangerous
metaphorical      cold voice, cold reception, cold decision
```

Tokenisation, baseline probability, morphology, and prefix compatibility
should be recorded. The aim would be to discover whether N38's cold tilt
generalises across cold-family tokens or is unusually concentrated on the
specific ` cold` token.

### 15.4 Greedy-threshold prediction

Use teacher-forced logit changes to predict in advance which prompts should
cross the greedy boundary under N38 alone or the old team. Then run generation
as a held-out test. This would connect the precise probability measurements to
the visually striking but discontinuous generation phenotype.

---

## 16. Bottom line

The recovered experiments contained a genuine effect but bundled together
neuron identity, team activity, extreme dose, greedy thresholding, and repeated
feedback. V13 separated those components and found a broader predicative-
property response with a consistent N38-specific cold tilt. V14 corrected
V13's additive calibration and showed that the cold-relative effect survives
throughout absolute activation values from zero to the observed natural
maximum.

N38 is therefore more informative than the phrase “cold neuron” suggests. Its
activation depends strongly on syntactic and lexical context, its downstream
effect supports a broader property-completion panel, and its distinctive cold
component appears as a graded relative preference. The strongest current
description is **a predicative-property-sensitive causal participant with a
reproducible cold-aligned direction**.

