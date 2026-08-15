# STRING ITERATION PROBE v11 — RESULTS

Status: preliminary result note. Tags follow `SPEC.md`: `MEASURED`, `DERIVED`,
`INTERPRETATION`, `PROPOSAL`.

Probe v11 examined the recurrent and reduplicative routes left under-tested by
v10 for GPT-2 Small layer-5 neuron 541. V9 had recovered constructions such as
`Wave after wave`, `one by one`, `zone by zone`, `again and again`, and `hand
to hand`. V10 established joint lexical and connector dependence for a
different family of distinct-member pairs, but its connector matrix placed
`after`, `by`, and `to` between members such as `up` and `down`. It therefore
did not test those connectors in the recurrent frames where v9 had found them.

V11 confirms that this was a consequential design gap. Familiar connected
repetitions produced a mean neuron-541 peak of 3.554 and retained 541 in 29/48
carrier cases. Novel connected repetitions such as `gravel after gravel`,
`pocket by pocket`, and `hammer and hammer` retained it in 0/48, but produced a
mean peak of 1.414—1.172 units above matched arbitrary distinct-member forms.
The repeated form was higher in 43/48 paired cases. Removing the connector
collapsed both familiar and novel repetition almost to zero.

The result separates two contributions. Connected repetition is sufficient to
raise 541 reproducibly, including with unfamiliar lexical material. Familiar
lexeme–connector combinations amplify that structural response into the
strong, often route-winning regime. Semantic or otherwise related distinct
members occupy a similar intermediate band to novel repetition.

The connector sweep also reverses the apparent v10 ranking. Under exact
repetition, `by`, `after`, and `to` retained 541 far more often than `and` or
`or`. This does not make them universally stronger connectors: many swept
forms—`day by day`, `one to one`, `round by round`, and `time after time`—are
familiar constructions themselves. It demonstrates connector–frame–lexeme
interaction rather than a context-free connector hierarchy.

The current working phenotype can be broadened cautiously to **completion of
connected parallel-member constructions**. Exact recurrence, semantic
relation, lexical familiarity, and connector compatibility provide graded and
partly overlapping evidence. This remains a behavioural description under the
declared panel, not a final linguistic or causal interpretation of neuron 541.

---

## 1. Run provenance — MEASURED

```text
script          probe_v11.py
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
results/probe_v11_preflight_20260815-051446.tsv
results/probe_v11_variants_20260815-051446.tsv
results/probe_v11_comparisons_20260815-051446.tsv
results/probe_v11_summary_20260815-051446.tsv
results/probe_v11_cells_20260815-051446.tsv
results/probe_v11_trace_20260815-051446.tsv
```

Row counts reconcile across all six outputs:

| Output | Rows | Unit |
|---|---:|---|
| Preflight | 662 | 12 lexical declarations plus 650 realised cases |
| Variants | 650 | one per evaluated string case |
| Comparisons | 1,034 | declared within-carrier and carrier comparisons |
| Summary | 78 | branch/variant/carrier summaries |
| Cells | 110 | connector/frame/provenance summaries |
| Trace | 3,252 | one per realised token position |

All files record the same pinned model revision, layer, parameter hash, and
actual CUDA device. `MEASURED`

The TSVs use the probe-recorder convention: tab separation with quotation
marks treated literally. Generic CSV readers should use
`quoting=csv.QUOTE_NONE` or an equivalent literal-TSV mode.

---

## 2. Question and design

### 2.1 The missing question from v10

V10 varied connectors while holding distinct lexical members fixed:

```text
up and down
up or down
up to down
up by down
up after down
```

That design established that `and` and `or` work well for the selected paired
members. It could not determine how `after`, `by`, or `to` behave in recurrent
or reduplicative frames. V11 therefore asks:

```text
Does exact repetition raise neuron 541 with unfamiliar words?
Does a connector remain necessary when the members repeat?
How do connectors behave while exact repetition is held fixed?
Do related but distinct members occupy the same response regime?
Do pronominal recurrence forms behave like literal repetition?
```

### 2.2 Frame-cross panel

Twelve declared lexical panels covered native `after`, `by`, `and`, and `to`
frames. They included v9-attested members and familiar held-out members:

```text
wave after wave       day after day        mile after mile
one by one            zone by zone         step by step
round and round       again and again      time and time
hand to hand          face to face         shoulder to shoulder
```

Each panel declared a known member, a related distinct member, and two
arbitrary controls before inference. For example:

```text
KNOWN_REPEAT          wave after wave
KNOWN_RELATED         wave after tide
KNOWN_ARBITRARY       wave after window
ARBITRARY_KNOWN       gravel after wave
ARBITRARY_REPEAT      gravel after gravel
ARBITRARY_DISTINCT    gravel after window
KNOWN_NO_CONNECTOR    wave wave
ARBITRARY_NO_CONNECTOR gravel gravel
```

The labels `known`, `related`, and `arbitrary` describe the declared design
roles. They are not independently validated semantic categories. Token counts,
positions, and realised strings were recorded in preflight. Comparisons were
marked strict only when the realised-token and completion-position criteria
matched; lexical frequency, morphology, and baseline neuron effects were not
fully matched.

### 2.3 Repeated-member connector sweep

For each of the twelve known members, v11 held exact repetition fixed and
substituted five connectors:

```text
wave and wave
wave or wave
wave to wave
wave by wave
wave after wave
```

The declared native connector served as the within-carrier baseline. This
branch tests connector compatibility inside the repeated-member frame. Because
some substitutions form familiar expressions in their own right, it does not
isolate connector identity from constructional familiarity.

### 2.4 Pronominal recurrence

A separate branch tested forms that express recurrence or succession without
always repeating the same written member:

```text
one after another
one after the other
one after one
one and another
one another
gravel after another
```

### 2.5 Carriers

Every linguistic case was evaluated under four declared regimes:

```text
BARE          construction begins at the first token position
V9_SPACE      leading-space form matching common v9 certificates
THE_PHRASE    embedded after “the phrase”
THEY_SAID     embedded after “they said”
```

A **carrier** is surrounding text that moves a construction into a changed
context and token position. A carrier can change 541’s activation and can
introduce stronger competing-neuron peaks. Carrier rows are repeated measures
of related strings, not independent samples.

---

## 3. Terms

### 3.1 Repetition, recurrence, and parallel members

**Exact repetition** means that the same written lexical member occurs on both
sides of a connector, as in `wave after wave`.

**Recurrence** is broader. `One after the other` denotes succession while its
surface members are not identical.

**Parallel members** is the provisional umbrella term for members placed in
corresponding slots of a connected relation. Their relationship may involve
identity, opposition, alternation, category membership, succession, or another
learned association. The term describes the panel’s common shape; it does not
claim that GPT-2 explicitly represents a grammatical category named
“parallelism.”

### 3.2 Raw activation, target peak, and completion

**Raw activation** is neuron 541’s value at one token position. Its **target
peak** is the largest raw activation it reaches anywhere in the whole string.

**Construction completion** is the final realised token of the declared second
member. “541 peaks at completion” means its largest activation in that string
occurs at this position. It is a positional observation, not by itself proof
that the neuron computes an abstract completion operation.

### 3.3 Destination, retention, and signed margin

Under the project’s Resonance readout, every layer-5 neuron is represented by
its largest activation anywhere in the string. The neuron with the largest
peak is the string’s **destination**. Neuron 541 is **retained** when it remains
that destination.

Its signed margin is:

```text
peak(541) - highest peak reached by any other layer-5 neuron
```

A positive margin means 541 wins. A negative margin can coexist with a
substantial 541 response. Retention is therefore a competitive thresholded
readout, while target peak measures the target’s own response.

### 3.4 Familiarity

**Familiar** refers to an attested or ordinary-looking construction selected by
the probe author, such as `day after day`. It is not a measured training-corpus
frequency. GPT-2’s actual exposure to each phrase remains unmeasured.

---

## 4. Validation and overview — MEASURED

```text
evaluated string cases              650
FRAME_CROSS cases                   384
REPEAT_CONNECTOR cases              240
PRONOMINAL_RECURRENCE cases          24
CARRIER_CONTROL cases                 2
cases retaining neuron 541          114
cases with 541 peak at completion   461
within-carrier comparisons          548
strict within-carrier comparisons   408
carrier-versus-bare comparisons     486
```

The aggregate 114/650 retention rate is not a feature-frequency estimate. The
650 cases intentionally mix positive constructions, destructive controls,
connector substitutions, and carrier variants. Interpretation depends on
declared matched contrasts.

All 114 cases retaining neuron 541 placed its target peak at the construction
completion position:

```text
FRAME_CROSS retained              33/33 at completion
REPEAT_CONNECTOR retained         68/68 at completion
PRONOMINAL_RECURRENCE retained    13/13 at completion
```

These cases are strongly correlated because they share lexical panels,
connectors, and carriers. They provide 114 concordant controlled forms, not 114
independent samples. `MEASURED`, `INTERPRETATION`

---

## 5. Primary result: novel connected repetition raises 541

### 5.1 Frame-cross aggregate — MEASURED

| Frame variant | Retained 541 | Cases | Mean 541 peak | Mean signed margin |
|---|---:|---:|---:|---:|
| Known repetition | 29 | 48 | 3.554 | 0.161 |
| Known related members | 4 | 48 | 1.564 | -1.558 |
| Arbitrary repetition | 0 | 48 | 1.414 | -1.748 |
| Arbitrary distinct members | 0 | 48 | 0.242 | -2.803 |
| Known + arbitrary | 0 | 48 | 0.167 | -2.936 |
| Arbitrary + known | 0 | 48 | 0.060 | -2.983 |
| Known repetition without connector | 0 | 48 | 0.102 | -2.894 |
| Arbitrary repetition without connector | 0 | 48 | approximately 0 | -3.049 |

### 5.2 Repetition effect among arbitrary members — DERIVED

Within the same panel and carrier, `ARBITRARY_REPEAT` exceeded
`ARBITRARY_DISTINCT` by an average of 1.172 target-peak units and 1.055 signed
margin units. Its peak was higher in 43/48 paired comparisons.

The mean repetition advantage appeared under every native connector:

| Native frame | Mean peak advantage | Cases with positive advantage |
|---|---:|---:|
| `after` | 1.229 | 11/12 |
| `by` | 1.175 | 11/12 |
| `and` | 1.500 | 12/12 |
| `to` | 0.787 | 9/12 |

Representative novel repetitions included:

```text
gravel after gravel
marble after marble
pocket by pocket
hammer and hammer
copper to copper
```

No arbitrary repetition retained 541 as destination. This does not make the
structural effect absent. Some cases approached the competitor envelope:

```text
they said pocket by pocket       peak 2.780   margin -0.024
they said marble after marble    peak 2.474   margin -0.110
they said candle after candle    peak 2.212   margin -0.314
```

`Hammer and hammer` reached a maximum measured peak of 3.122, but neuron 2659
won that string. Raw target activation and route victory therefore give
different, complementary descriptions of generalisation.

`INTERPRETATION` — exact connected repetition is sufficient to raise neuron
541 systematically beyond arbitrary distinct coordination. It is generally
insufficient, under these lexical controls and carriers, to make 541 the
destination.

### 5.3 Familiarity or lexical compatibility adds a second boost — DERIVED

Known repetition exceeded arbitrary repetition by an average of 2.140 target
peak units and was higher in 47/48 matched panel/carrier comparisons. It
exceeded known-related distinct forms by 1.990 units and was higher in 46/48.

Known repetitions retained 541 in 29/48 cases. Performance varied by phrase:

```text
one by one              4/4 retained   mean peak 4.683
again and again         4/4 retained   mean peak 3.524
hand to hand            4/4 retained   mean peak 4.288
shoulder to shoulder    4/4 retained   mean peak 4.150
mile after mile         4/4 retained   mean peak 3.839
round and round         0/4 retained   mean peak 3.256
time and time           0/4 retained   mean peak 2.630
```

The last two examples again show why a destination-only account is incomplete:
familiar frames can evoke substantial target activation while losing to a
stronger competitor.

`INTERPRETATION` — the data support at least two graded contributions: a
general repeated-frame contribution and a larger contribution associated with
particular familiar or compatible lexeme–connector combinations. The present
design does not identify whether the latter arises from phrase frequency,
lexical semantics, constructional conventionality, or correlated token-level
properties.

---

## 6. The connector is part of the recurrent route

Removing the connector reduced known repetition from a mean peak of 3.554 to
0.102 and retention from 29/48 to 0/48. The known connected form exceeded its
connectorless counterpart in all 48 paired cases, by 3.452 target-peak units
on average. Arbitrary connected repetition exceeded arbitrary connectorless
repetition by 1.414 units and was higher in 44/48.

`INTERPRETATION` — mere adjacency of two identical written members is not
sufficient under this panel. The connector participates in the route. This is
a bounded conclusion about the tested short forms; other punctuation,
morphosyntactic frames, or longer contexts remain untested.

---

## 7. Connector behaviour depends on the repeated frame

### 7.1 Aggregate repeated-member sweep — MEASURED

| Connector | Retained 541 | Cases | Mean 541 peak | Mean signed margin |
|---|---:|---:|---:|---:|
| `by` | 24 | 48 | 3.260 | 0.001 |
| `after` | 20 | 48 | 2.954 | -0.211 |
| `to` | 18 | 48 | 2.757 | -0.556 |
| `and` | 4 | 48 | 2.823 | -0.909 |
| `or` | 2 | 48 | 2.382 | -1.217 |

Neuron 541 peaked at construction completion in all 240 connector-sweep cases,
including the 172 cases in which another neuron ultimately won.

This ranking differs sharply from v10’s distinct-member connector matrix,
where `or` retained 22/24 and `and` retained 18/24, while `by`, `to`, and
`after` were much weaker. V11 confirms that v10 measured connector performance
inside one lexical frame rather than general connector strength.

### 7.2 Native recurrent frames recover — MEASURED

| Declared native repeated frame | Retained 541 | Cases | Mean 541 peak |
|---|---:|---:|---:|
| `X after X` | 8 | 12 | 3.578 |
| `X by X` | 8 | 12 | 3.901 |
| `X and X` | 4 | 12 | 3.137 |
| `X to X` | 9 | 12 | 3.601 |

The clearest repair to the v10 interpretation is `after`. V10’s
distinct-member `X after Y` cases retained 541 in 0/24, with mean target peak
1.081. V11’s native repeated `X after X` cases retained it in 8/12, with mean
peak 3.578. These are not a fully matched cross-run comparison because the
lexical panels differ, but they establish that `after` can be highly effective
in its recurrent frame.

### 7.3 “Native connector” is not a fixed lexical rule

Several connector substitutions created strong familiar forms:

```text
day by day
mile by mile
one to one
round by round
time after time
step after step
```

The declared native connector was not always the highest-activation connector
for a member. `Mile by mile` exceeded `mile after mile` on mean target peak;
`round by round` exceeded `round and round`; and `time after time` exceeded
`time and time`.

`INTERPRETATION` — the result is better described as compatibility among the
connector, lexical member, and recurrent frame. A member can participate in
several effective constructions. Because many substitutions are themselves
conventional expressions, the sweep does not distinguish connector semantics
from phrase familiarity.

---

## 8. Related distinct members occupy an intermediate band

Known-related distinct forms produced a mean peak of 1.564 and retained 541 in
4/48 cases. This is far below known exact repetition at 3.554, but close to
novel arbitrary repetition at 1.414 and well above arbitrary distinct forms at
0.242.

The four route wins came from:

```text
they said day after night
one by two                         [three of four carriers]
```

The declared “related” pairs are heterogeneous: `one/two`, `day/night`,
`wave/tide`, `hand/foot`, and `time/space` express different relations. Their
aggregate proximity to novel repetition suggests that semantic relation and
structural identity may supply partly substitutable evidence. It does not show
that the model assigns them equal causal weight. A matched multi-sample
relation panel remains necessary.

---

## 9. Pronominal recurrence forms a strong related family

### 9.1 Results — MEASURED

| Form | Retained 541 | Cases | Mean 541 peak | Mean signed margin |
|---|---:|---:|---:|---:|
| `one after another` | 4 | 4 | 4.063 | 0.845 |
| `one after one` | 3 | 4 | 3.900 | 0.864 |
| `one after the other` | 4 | 4 | 3.727 | 0.733 |
| `one and another` | 2 | 4 | 2.514 | -0.491 |
| `one another` | 0 | 4 | 0.010 | -2.966 |
| `gravel after another` | 0 | 4 | 0.048 | -2.932 |

Replacing `one` with `gravel` nearly eliminates the response, so `another`
alone does not explain the route. Removing `after` from `one after another`
also collapses it. Substituting `and` preserves a moderate response but reduces
both target peak and retention.

`INTERPRETATION` — neuron 541’s recurrent family extends beyond literal
written-word identity. Pronominal or correlative succession can evoke a strong
response, but it depends on compatible lexical and connector material. This
again favors a constructional account over a detector for duplicated token
identity.

---

## 10. Carrier and competitor effects

Known repetition remained strong across all four carriers:

| Carrier | Retained 541 | Cases | Mean 541 peak | Mean signed margin |
|---|---:|---:|---:|---:|
| Bare | 9 | 12 | 3.373 | 0.235 |
| V9 leading space | 5 | 12 | 3.498 | 0.056 |
| `the phrase` | 7 | 12 | 3.576 | 0.112 |
| `they said` | 8 | 12 | 3.770 | 0.242 |

Novel repetition retained 541 in none of the four regimes, but its mean peak
rose from 0.952 bare to 1.758 after `they said`. This is not merely carrier
suppression or rescue: the carrier changes the target response and the set of
rival opportunities simultaneously.

Several novel repetitions lost consistently to recurring competitors,
including neurons 1888, 2073, 2566, and 2659. A route can therefore move closer
to 541 without arriving there. The signed margin preserves this distinction.

---

## 11. What v11 rules against, supports, and leaves open

### 11.1 Strongly disfavoured within the tested regime — INTERPRETATION

A **context-free weakness of `after`, `by`, or `to`** is inconsistent with
their strong repeated-frame performance.

A **pure duplicated-token detector** is inadequate because connectors are
necessary and `one after another` is strong without literal member identity.

A **pure memorised-phrase inventory** is incomplete because novel arbitrary
repetition raises 541 systematically above arbitrary distinct forms across all
four connector families.

A **repetition-alone sufficiency account** is also incomplete. Novel connected
repetition never made 541 the destination, and familiar repetitions were about
2.14 peak units stronger on average.

These are local exclusions under the declared strings and readout, not proofs
over all possible inputs.

### 11.2 Supported working account — INTERPRETATION

The combined v10 and v11 evidence supports a graded sensitivity to the
completion of **connected parallel-member constructions**. Several overlapping
signals can raise neuron 541:

```text
exact identity or recurrence across the member slots
semantic or lexical relation between distinct members
familiarity of the complete construction
compatibility between connector, members, and frame
the arrival of the completing second member
surrounding carrier and token position
competition from other layer-5 neurons under Resonance
```

The response need not reduce to one linguistically pure property. A single MLP
neuron may pool correlated cues found together in the training distribution.

### 11.3 Still open

V11 does not measure phrase frequency in GPT-2’s training data, so
“familiarity” remains an observer classification. It does not cleanly separate
semantic relation from morphology, part of speech, token frequency, or
construction frequency. It also does not establish specificity across ordinary
corpus text, because the panel was designed around known-positive families and
constructed controls.

The apparent intermediate equivalence of related distinct members and novel
repetition needs direct factorial testing. The competitor field also deserves
attention: novel repetition may represent meaningful sub-threshold
participation by 541 even when another neuron, especially 2659, consistently
wins.

---

## 12. Recommended next steps — PROPOSAL

The highest-value next experiment is a matched semantic-relation factorial.
For several lexical bases, it should cross exact repetition, same-category
members, opposites, conventional associates, and frequency/token-matched
unrelated controls under one connector and carrier. Multiple samples per cell
are essential; one `cats and elephants` or `one by two` example cannot identify
a general semantic effect.

A complementary competitor probe could examine the recurrent rivals exposed
by v11. Comparing neuron 541 with 2659, 1888, and 2073 at the completion token
may reveal whether novel repetitions form a distributed sub-threshold pattern
whose argmax destination changes with carrier. This would extend the project
from single-destination phenotyping toward companion or local-population
structure without changing the declared Resonance address.

Finally, a natural-corpus validation should search held-out text for high 541
activation and classify the resulting contexts without preselecting paired
constructions. That would estimate false positives, reveal unanticipated
families, and test whether the current account predicts naturally occurring
hits beyond Melville and researcher-authored controls.

---

## 13. Bottom line

`MEASURED` — known connected repetitions retained neuron 541 in 29/48 cases
with mean peak 3.554. Novel connected repetitions retained it in 0/48 but
produced a mean peak of 1.414, exceeding arbitrary distinct-member controls by
1.172 units on average and in 43/48 paired cases. Connector removal collapsed
both familiar and novel repetition. Under exact repetition, `by`, `after`, and
`to` became highly effective, repairing the frame mismatch in v10. Strong
pronominal forms showed that recurrence need not involve identical written
members. All 114 retained v11 cases peaked at construction completion.

`INTERPRETATION` — v11 identifies a real structural repetition contribution
and a larger lexical/constructional compatibility contribution. Neuron 541 is
best described provisionally as responding at the completion of connected
parallel-member constructions, with graded evidence from recurrence, semantic
relation, familiarity, and connector–frame compatibility.

`CAUTION` — the working phenotype is broader and better constrained than after
v10, but it is not a final neuron name. The next discriminating step is to
separate repetition, semantic relation, and construction frequency with
multiple matched lexical samples, while also examining the competitor
population that prevents many moderate 541 responses from becoming the
destination.
