# STRING ITERATION PROBE v12 — RESULTS

Status: preliminary causal-intervention result note. Tags follow `SPEC.md`:
`MEASURED`, `DERIVED`, `INTERPRETATION`, and `PROPOSAL`.

Probe v12 moved the neuron-541 investigation from observational phenotype
mapping to direct intervention. Earlier probes found that GPT-2 Small layer-5
neuron 541 responds strongly and recurrently at the completion of connected
parallel-member constructions such as `up and down`, `one by one`, `wave after
wave`, and `one after another`. V12 asked three narrower causal questions:

1. Does natural 541 activity at the completed construction affect what the
   model predicts next?
2. Is setting 541 to a positive value sufficient to transfer this downstream
   behaviour into weak or malformed routes?
3. Can activating 541 before the second member steer the model toward the
   expected completion?

The first answer is a qualified yes. At the construction boundary, progressively
removing 541 produced a monotone dose-response in the model's next-token
distribution and usually reduced the probability of the declared continuation.
The effect was position-specific and remained larger than three matched-neuron
controls after normalising for the size of the removed residual contribution.

The second answer is not under the tested intervention. Injecting 541 at the
completion of weak controls changed the distribution but did not reliably
increase their declared continuations or transplant the route family.

The third answer is also no under the tested intervention—and the direction is
striking. Forcing 541 positive at the connector lowered the expected second
member in every case at every dose, with increasing suppression at larger
doses. Because clean 541 activation at that position was near zero or slightly
negative, these are deliberately ectopic states. They show that prematurely
turning on 541 is not a construction generator; they do not by themselves
establish that 541 is downstream-inert within its natural co-active circuit.

The combined result supports a temporally qualified description: **neuron 541
is a position-sensitive participant in processing a resolved connected
parallel-member construction, with a modest, heterogeneous causal influence on
subsequent prediction.** V12 does not establish that 541 alone generates the
construction, represents the entire construction family, or is necessary for
the model to recognise it.

---

## 1. Run provenance — MEASURED

```text
script          probe_v12.py
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

Outputs:

```text
results/probe_v12_preflight_20260815-062746.tsv
results/probe_v12_interventions_20260815-062746.tsv
results/probe_v12_summary_20260815-062746.tsv
results/probe_v12_token_effects_20260815-062746.tsv
```

| Output | Rows | Unit |
|---|---:|---|
| Preflight | 32 | five neuron/control declarations plus 27 linguistic cases |
| Interventions | 300 | one case × treatment evaluation |
| Summary | 33 | one branch × treatment aggregate |
| Token effects | 4,380 | promoted and suppressed token records for non-identity treatments |

All four files record the same pinned revision, parameter hash, layer, and
actual CUDA device. `MEASURED`

The TSVs use the recorder's literal-quotation convention. Generic readers
should use tab separation with `quoting=csv.QUOTE_NONE` or an equivalent
literal-TSV mode.

---

## 2. What was intervened on

### 2.1 Intervention site

Neuron 541 means coordinate 541 in the post-GELU activation vector of GPT-2
Small's layer-5 MLP. The intervention altered this coordinate at one declared
token position while leaving the prompt tokens unchanged. Outcomes were read
from downstream hidden states and output logits, never from 541's own activation
or from the Atlas destination rule. This avoids defining success using the
quantity directly manipulated. `MEASURED`

The natural-completion branch intervened at the final realised token of the
second member:

```text
one by [one] → they came
up and [down] → the road
wave after [wave] → struck
```

The precompletion branch intervened one structural phase earlier, at the
connector:

```text
one [by] → one
up [and] → down
wave [after] → wave
```

Square brackets mark the intervention position, not a textual edit.

### 2.2 Three branches

**Natural completion** used ten strong or informative routes. It reduced or
removed the naturally positive 541 activation and measured the declared
continuation after the completed construction.

**Completion injection** used nine weak, novel, malformed, or lexically
disrupted routes. It set 541 to fixed positive values or donor values at their
final member and asked whether their declared continuation became more likely.

**Precompletion steering** used eight incomplete constructions. It set 541
positive at the connector and measured the probability of the anticipated
second member.

These branches address different causal questions. Failure of early steering
does not negate a downstream effect at natural completion, and downstream
necessity-like evidence does not imply that one neuron can generate the
construction in advance.

### 2.3 Controls

The run included:

- `CLEAN_REPEAT`, a deterministic duplicate of the unmodified forward pass;
- `SHAM`, which installed the intervention hook but wrote back the original
  value;
- `PREVIOUS_ZERO_541` or `PREVIOUS_DONOR_541`, which moved the intervention to
  the preceding token;
- neuron 2659, declared in advance from v11 as a recurring rival;
- neurons 1738, 1691, and 2292, selected before outcome analysis by proximity
  to 541 in output-direction norm and mean absolute natural activation;
- fixed 541 values of 1, 2, 3, and 4;
- matched and shuffled assignments of natural donor activation values.

The matched controls reduce, but do not eliminate, the possibility that any
active layer-5 neuron with a substantial output vector would produce a similar
effect when ablated.

---

## 3. Terms used in this report

**Ablation** means setting the selected activation to zero. It removes that
neuron's contribution at one token position; it does not delete the neuron from
the whole model or prove that all information associated with it has vanished.

**Steering** means deliberately assigning an activation value in order to
change a later prediction. A steering failure means the tested assignment did
not produce the declared result. It is not automatically proof that the neuron
has no causal role under naturally coordinated activity.

**Necessity-like evidence** means that removing a naturally present component
reliably harms a declared outcome. Strict necessity would require the outcome
or capacity to fail whenever that component is absent. V12 produces graded
probability changes, not categorical loss of the construction.

**Sufficiency** would mean that adding the component is enough to produce the
declared behaviour in an otherwise non-producing context. V12 tests only
single-neuron sufficiency at particular positions and values.

**Outcome log-probability change** is the intervened log-probability of the
declared continuation minus its clean value. A change of −0.30 nats corresponds
to multiplying the continuation's probability by approximately
`exp(−0.30) = 0.74`, if the outcome consists of the same declared token sequence.

**KL divergence** measures how much the full next-token probability
distribution changes, rather than only the declared outcome. It is zero only
when the two distributions are identical under the calculation used here.

**Greedy token** is the single highest-probability next token. A probability
change can be real while leaving the greedy token unchanged.

**Donor patch** assigns 541 the activation observed on a declared natural donor
route. A **shuffled donor** permutes those scalar donor values across targets.
Both still intervene on neuron 541; this control varies donor–target matching,
not neuron identity.

**Out of distribution**, in this report, means an internal activation state
not observed in the clean cases at that position. It does not assert that the
input text itself lies outside GPT-2's training distribution.

---

## 4. Integrity checks — MEASURED

`CLEAN_REPEAT` and `SHAM` were bit-exact identities on the recorded outcome
log-probabilities, KL divergences, logit norms, and hidden-state changes in all
three branches. The hook alone therefore introduced no measurable change, and
the pinned run was exactly repeatable under these checks.

At natural completion, zeroing 541 at the preceding token was also effectively
inert:

| Treatment | Mean outcome log-probability change | Median KL |
|---|---:|---:|
| `SHAM` | 0.000000 | 0.000000 |
| `CLEAN_REPEAT` | 0.000000 | 0.000000 |
| `PREVIOUS_ZERO_541` | +0.000834 | 0.000001 |

This position control is especially informative because it uses the same
neuron and operation while moving the intervention by only one token. Its near
null result supports temporal specificity. It does not show that all other
positions would be inert. `INTERPRETATION`

---

## 5. Natural-completion ablation

### 5.1 Aggregate dose-response — MEASURED

| Remaining 541 activation | Mean outcome log-probability change | Median change | Mean KL | Median KL |
|---:|---:|---:|---:|---:|
| 75% | −0.058 | −0.045 | 0.00236 | 0.00122 |
| 50% | −0.131 | −0.100 | 0.00963 | 0.00486 |
| 25% | −0.213 | −0.166 | 0.02171 | 0.01120 |
| 0% | **−0.302** | **−0.246** | **0.03838** | **0.02052** |

Removing progressively more activation produced progressively larger average
and median effects. The curve is convex-looking on the log-probability and KL
scales: partial removal was disproportionately cheap relative to full
ablation. Because downstream logits, softmax probabilities, and
log-probabilities are nonlinear, v12 does not identify where that curvature is
generated. It should not yet be described as a special nonlinear mechanism
inside neuron 541. `DERIVED`, `INTERPRETATION`

Full ablation changed the greedy next token in only 1/10 cases. This is
compatible with a real but usually non-decisive redistribution of probability:
most effects altered confidence without changing the argmax. `MEASURED`

### 5.2 Case heterogeneity — MEASURED

| Completed route and declared continuation | Full-ablation log-probability change |
|---|---:|
| `up and down` → ` the road` | **−0.889** |
| `one by one` → ` they came` | **−0.840** |
| `one after another` → ` arrived` | −0.488 |
| `again and again` → ` he tried` | −0.436 |
| `hand to hand` → ` combat` | −0.298 |
| `round and round` → ` the ship` | −0.194 |
| `, go round and round` → `.` | −0.178 |
| `wave after wave` → ` struck` | −0.072 |
| `shoulder to shoulder` → ` they stood` | **+0.131** |
| bare `round and round` → ` the ring` | **+0.243** |

Eight of ten declared continuations became less likely, while two became more
likely. The mean is influenced strongly by `up and down` and `one by one`, but
the median remains negative at −0.246. The sign reversals are part of the
phenotype rather than errors to average away: 541's fixed output direction is
interpreted through different downstream states, and its effect on any chosen
token can therefore vary by context. `INTERPRETATION`

The outcome strings were selected continuations, not direct tests of whether
the model still detected or could itself generate the preceding construction.
The branch therefore establishes causal influence on post-construction
prediction, not strict necessity for reciprocal or recurrent language.

### 5.3 Matched-neuron comparison — MEASURED and DERIVED

The raw effect of 541 is larger than all three matched-neuron controls. Since
the neurons did not contribute identical residual-vector magnitudes in these
specific cases, the comparison was also normalised by mean layer-5 residual
perturbation norm:

| Ablation | Mean outcome change | Mean layer-5 residual ΔL2 | Outcome change per residual ΔL2 |
|---|---:|---:|---:|
| `ZERO_541` | **−0.3022** | 14.2663 | **−0.02118** |
| `ZERO_MATCHED_1_1738` | −0.0685 | 6.5798 | −0.01041 |
| `ZERO_MATCHED_2_1691` | −0.0368 | 4.7211 | −0.00779 |
| `ZERO_MATCHED_3_2292` | −0.0197 | 4.7011 | −0.00420 |
| `ZERO_2659` | +0.1042 | 7.9387 | +0.01312 |

After this normalization, 541 remains approximately two to five times as
negatively aligned with the declared outcomes as the matched controls. Neuron
2659 has substantial potency in the opposite direction. This suggests that
541 is not uniquely powerful among layer-5 neurons; its output direction is
more consistently aligned with this panel's selected continuations.
`INTERPRETATION`

Neuron 541 was the most damaging of the four ablations comprising it and the
three matched controls in 6/10 cases. An exact binomial calculation under an
exchangeable 0.25 null gives approximately `p = 0.02`, but that is exploratory,
not confirmatory: the ten cases share construction families, were deliberately
selected, and the four neurons are not random exchangeable draws. The
descriptive 6/10 result is more trustworthy than the nominal p-value.

### 5.4 Mechanical and downstream distances

The layer-5 hidden-state change scales linearly with dose:

```text
75% remaining     3.567
50% remaining     7.133
25% remaining    10.700
 0% remaining    14.266
```

This is mechanically expected: changing a single activation by `Δa` changes
the MLP projection according to `|Δa|` times that neuron's output-direction
norm. It is a hook-correctness and dose-delivery check, not independent evidence
that later layers used the signal. The nonzero final-layer changes, output KL,
and log-probability changes are the downstream measurements.

Rank changes should be summarized by medians rather than means. For
`ZERO_541`, the mean first-outcome rank change was +37.8 while the median was
only +1.5. For matched neuron 1738 the mean was +125.5 while the median was
0.0. Rare extreme rank movements make the means misleading.

---

## 6. Completion-position injection

The completion-injection branch asked whether positive 541 activation could
make weak or malformed routes behave like strong natural routes. The aggregate
declared-continuation effects were small and inconsistent:

| Treatment | Mean outcome change | Median outcome change | Mean KL | Positive cases |
|---|---:|---:|---:|---:|
| Set to 1 | +0.023 | +0.007 | 0.00254 | 5/9 |
| Set to 2 | +0.013 | −0.029 | 0.00663 | 3/9 |
| Set to 3 | +0.007 | −0.024 | 0.01546 | 3/9 |
| Set to 4 | +0.004 | −0.066 | 0.02811 | 2/9 |
| Matched donor | +0.019 | −0.021 | 0.02164 | 2/9 |
| Shuffled donor | +0.013 | −0.049 | 0.02303 | 2/9 |

Larger doses increasingly changed the full output distribution, but did not
produce an increasing benefit for the declared continuations. This separates
**causal potency**—the intervention changes the network—from **behavioural
sufficiency**—the change produces the hypothesised behaviour.

One route, connectorless `one another` → ` arrived`, increased by +0.494 at
activation 4. The other eight cases ranged from +0.074 to −0.136. This isolated
response is worth retaining as a candidate interaction, but it does not support
general construction transfer by itself.

Matched and shuffled donors were similar. This means route-specific matching
of the donor scalar supplied no detectable advantage under this panel. It does
not test neuron identity: every donor value was still written into coordinate
541. `MEASURED`, `INTERPRETATION`

---

## 7. Precompletion steering

### 7.1 Uniform negative direction — MEASURED

Forcing 541 positive at the connector reduced the declared second member in
all eight cases at every fixed dose:

| Set activation | Mean completion log-probability change | Median change | Mean KL | Positive cases |
|---:|---:|---:|---:|---:|
| 1 | −0.198 | −0.136 | 0.03284 | 0/8 |
| 2 | −0.413 | −0.314 | 0.11633 | 0/8 |
| 3 | −0.654 | −0.443 | 0.25203 | 0/8 |
| 4 | **−0.916** | **−0.558** | **0.43315** | **0/8** |

At activation 4:

| Prefix → declared completion | Log-probability change | Clean probability → intervened probability |
|---|---:|---:|
| `one by` → ` one` | **−2.372** | 0.479 → 0.045 |
| `wave after` → ` wave` | **−1.949** | 0.777 → 0.111 |
| `hand to` → ` hand` | −1.058 | 0.527 → 0.183 |
| `one after` → ` one` | −0.590 | 0.018 → 0.010 |
| `now or` → ` then` | −0.526 | 0.00135 → 0.00080 |
| `one after` → ` another` | −0.375 | 0.572 → 0.393 |
| `one after` → ` the other` | −0.263 | 0.232 → 0.208 |
| `up and` → ` down` | −0.195 | 0.218 → 0.179 |

The greedy token changed in two of eight cases at the largest dose. Most
declared completions remained top-ranked or near the top despite losing
probability, so the effect is better described as graded suppression than
categorical erasure.

### 7.2 Why this is not a clean natural-role test

Clean 541 activation at the connector had a median near −0.06. The fixed-dose
interventions changed it by median amounts of approximately +1.06, +2.06,
+3.06, and +4.06. They therefore imposed a completion-like positive activation
at a position where the tested clean inputs did not naturally produce one.

Three interpretations remain live:

1. **Premature phase signal.** Positive 541 activity may indicate that a
   connected relation has resolved; forcing it early conflicts with a pattern
   that is still incomplete.
2. **Fixed output-direction effect.** At connector states, 541's output vector
   may happen to suppress the expected second-member tokens without the model
   treating it as a semantic completion flag.
3. **Ectopic-state disruption.** A large isolated activation may create an
   internal state not produced by natural inputs, with no clean functional
   interpretation.

The monotone result decisively rejects the simple prediction that increasing
541 alone at the connector will promote the anticipated completion. It does
not yet distinguish these three explanations.

Matched donor and shuffled donor effects were almost identical in their means
(`−0.861` and `−0.862`). As in the completion-injection branch, this shows no
benefit from route-specific scalar donor matching. It is consistent with
generic dose effects, but cannot by itself show that another neuron would have
the same effect.

The previous-position donor control was less disruptive than the declared
connector-position patch (median outcome changes −0.266 versus −0.582), though
it was not null. A large donor activation can therefore affect predictions
from nearby positions as well, while the declared position remains the more
consequential one.

---

## 8. What v12 establishes

### 8.1 Supported claims

Under this declared panel and model checkpoint:

- the hook and deterministic measurement chain passed exact identity checks;
- natural 541 activation at the completed construction causally affects the
  model's subsequent prediction distribution;
- the aggregate ablation effect is graded with dose and temporally specific;
- the effect is mostly supportive for the selected continuations but varies by
  context and reverses sign in two cases;
- 541 remains more outcome-aligned than the three matched-neuron controls after
  normalising by removed residual-vector magnitude;
- single-neuron activation at weak completion routes does not reliably
  transplant the hypothesised family behaviour;
- forcing 541 positive at the connector consistently suppresses, rather than
  promotes, the anticipated second member.

### 8.2 Claims not established

V12 does not establish that:

- neuron 541 is strictly necessary for reciprocal or recurrent constructions;
- neuron 541 alone generates or represents the whole construction;
- the observed effects generalise beyond the selected GPT-2 Small checkpoint,
  layer, carriers, routes, or continuations;
- donor/shuffled equivalence makes neuron identity irrelevant;
- the early-steering suppression has a uniquely semantic interpretation;
- the apparent dose-curve curvature originates specifically in a 541-centred
  circuit;
- ten related, deliberately selected cases support population-level
  significance claims.

---

## 9. Revised working phenotype — INTERPRETATION

V9–v11 described 541 observationally as concentrating around the completion of
connected parallel-member constructions, with graded contributions from
recurrence, lexical familiarity, member relation, and connector compatibility.
V12 adds a causal layer without replacing that phenotype:

> **Neuron 541 is a completion-position-sensitive member of the computation
> associated with connected parallel-member constructions. Its natural
> activation has a modest, dose-dependent and context-sensitive influence on
> what GPT-2 predicts after the construction has resolved. Positive activation
> of 541 alone is not sufficient to install the construction in weak routes or
> to elicit its missing second member when forced prematurely.**

The compact shorthand **completion-associated causal participant** is safer
than either **construction neuron** or **pure reporter**. “Participant” records
the ablation result. “Completion-associated” records the timing. The remaining
qualification matters because v12 altered one coordinate in a 3,072-unit MLP
state, while natural computation may depend on a distributed co-active set.

The word **reporter** remains a plausible hypothesis: 541 may chiefly become
active after other circuitry has resolved the construction. It is not yet a
settled conclusion, because removing the report changes downstream prediction,
and v12 did not test 541's incremental contribution inside a naturally patched
ensemble.

---

## 10. Proposed next experiment — PROPOSAL

The next discriminating step is activation-vector patching with structural
position matching. It should separate two questions that a single
completion-to-connector patch would conflate.

### 10.1 Pending-completion state transfer

For generation, copy the full layer-5 post-GELU vector from a natural donor at
its connector position into a structurally matched target connector position:

```text
donor:  wave [after] → wave
target: gravel [after] → ?
```

This transfers a naturally occurring “second member still pending” state. A
vector taken from `wave after wave` at the completed `wave` would instead move
a resolved state into an unresolved position and repeat v12's phase mismatch
at much larger dimensionality.

### 10.2 Resolved-state ensemble test

For the reporter-versus-ensemble question, patch between matched completion
positions and compare paired interventions:

```text
A. complete donor activation vector
B. the same donor vector with coordinate 541 restored to the target clean value
C. target vector with donor coordinate 541 alone
D. preregistered donor subsets containing and excluding 541
```

The key contrast is A versus B: identical transplanted states differing only
in 541. If the full vector transfers an effect and removing 541 reliably
reduces it, 541 contributes within a co-active state even though it cannot
steer alone. If the full vector works while A and B are equivalent, the ensemble
matters but 541 may be primarily correlated with it. If structurally matched
full-vector patches also fail, the reporter interpretation becomes stronger,
subject to the usual patching and alignment limitations.

Controls should include clean repeats, sham full-vector writes, same-case
self-patches, shuffled donors, donor/target tokenisation checks, position-
shifted patches, and subset-size-matched random coordinates. Outcomes and
subsets should be declared before inference.

---

## 11. Bottom line

V12 extends the earlier correlational route-family evidence: changing neuron
541 at the construction boundary causally and dose-dependently changes later
prediction.
The influence is modest, heterogeneous, and usually does not change the greedy
token. It remains stronger than matched-neuron controls after perturbation-size
normalisation.

The same neuron does not behave as a standalone construction switch. Injecting
it into weak completed routes produces no general transfer, and forcing it on
before completion uniformly pushes the expected second member down. The most
defensible current description accommodates both the readout-like timing and
the downstream causal effect: neuron 541 is a temporally situated participant
whose function is likely inseparable from the surrounding activation state.
