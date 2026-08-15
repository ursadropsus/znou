# PROBE DAY OVERVIEW — v6 to v15

On 15 August 2026, the probe sequence developed from a general way of testing
Atlas routes into a small but increasingly rigorous interpretability workflow.
Across versions 6–15, the work moved through three linked questions: how stable
an Atlas route is under changes to its wording, how to reduce a long route to
the fragment actually responsible for its destination, and what causal role a
strongly associated neuron plays in the model's predictions.

The day began with v6 testing route robustness. Rather than treating a string's
Atlas destination as a fixed semantic label, it examined how that destination
survived truncation, continuation, substitution, and different surrounding
carriers. The results showed that routes have distinct behavioural profiles:
some survive changes around their apparent core, some depend strongly on their
carrier, and some remain locally porous to nearby wording changes. V6 also
caught an output-formatting issue and established a more careful vocabulary for
describing route robustness without turning preliminary patterns into neuron
identities.

V7 turned minimisation into a systematic operation. Given a full sentence and
its target neuron, it searched contiguous token spans for shorter strings that
preserved the Atlas destination. This reproduced the manually whittled N906
example while also showing that token-level certificates can be shorter than
the word-level phrases a human naturally selects. It further revealed that
several routes to the same neuron can collapse onto recurring short forms, an
early sign of route-family concentration.

V8 then examined why an automatic certificate could be shorter and win by a
wider margin despite having a lower target-neuron peak. The answer lay in the
competition: an Atlas destination depends on the target peak relative to every
other neuron's maximum. Removing context can lower the target slightly while
lowering its strongest rivals much more. This made an important distinction
explicit: the strongest raw activation, the shortest winning route, and the
widest winning margin are different optimisation objectives.

V9 applied exhaustive minimisation to 80 Melville routes for neuron 541. The
result was the strongest observational concentration encountered during the
day. Eighty routes reduced to 47 exact certificates, with a median length of
three tokens. Recurrent forms included `up and down`, `now and then`, `here and
there`, `one by one`, `right and left`, and related constructions such as
`again and again`, `east and west`, and `one after the other`. The textual
surface varied, but the certificates repeatedly completed paired, parallel, or
recurrent members linked by a connector. A Windows checkpoint replacement
error interrupted the first execution at route 52; the checkpoint handling was
repaired and the resumed audit completed without discarding prior work.

V10 and v11 tested whether this N541 pattern was merely memorised Melville
language, generic coordination, lexical familiarity, repetition, or a more
structured combination. Factorial controls showed that both members and their
connector matter. Familiar or conventional pairs often produced stronger
responses, while novel connected repetition also raised N541. Connector
effects depended on the frame, related distinct members occupied an
intermediate region, and pronominal recurrence supplied especially useful
evidence for abstract slot structure. The resulting phenotype was broader than
a list of idioms and narrower than arbitrary `X and Y`: N541 is strongly
associated with the completion of connected parallel-member and recurrent
constructions, modulated by lexical compatibility and context.

V12 asked the causal question. Removing naturally active N541 at completed
constructions had a modest, position-specific, dose-responsive effect on what
the model predicted next. The effect was heterogeneous and sometimes reversed
sign, but remained larger than several matched-neuron controls after accounting
for perturbation size. Conversely, injecting N541 before a construction's
completion did not make the expected second member more likely; it consistently
reduced it. Donor and shuffled-donor interventions behaved nearly identically,
showing that the injection result was dominated by generic disruption rather
than information uniquely carried by N541. The causal conclusion was therefore
qualified: N541 participates in processing an already resolved construction
and modestly affects subsequent prediction, while the experiment did not show
that it independently generates or represents the construction.

V13 shifted attention to neuron 38 and reconstructed older, less controlled
experiments that had associated it with cold-related generation. The new probe
separated phenotype mapping, single-neuron intervention, older neuron-team
effects, one-step prediction, and sustained generation. N38 was weak on the
isolated word `cold` and much stronger when a property was expressed
predicatively, as in `It was cold`. Steering broadly supported adjective or
property completions, while consistently shifting matched probability ratios
toward `cold`. Sustained intervention could alter generated trajectories, but
did not generally turn them into cold prose by itself.

V13 also uncovered a calibration error in its own steering design: values drawn
from the natural activation range had been added to existing activations,
creating final values above the observed natural maximum. V14 repaired this by
setting N38 to absolute values from zero through the measured maximum. Across
five matched cold-versus-alternative pairs, the cold-relative advantage changed
monotonically with N38's set value. This strengthened the causal claim while
leaving an important caveat: a scalar value can be naturally sized even when
the complete surrounding activation vector is not a naturally observed state.

V15 returned to observational Atlas work with three full Melville routes to
N38. Exhaustive span analysis reduced them to:

```text
It was cold
heard a faint creaking
The grey dawn came
```

The result overturned the most tempting reading of the long sentences. The
second route peaked on the completion of `creaking`, not on `muffled by the
storm`; the third peaked on `came`, not on `was brought to the ship`. The
broader observational family now appears to involve compact predications that
establish a perceptible property, state, or environmental occurrence. The
causal evidence remains narrower: within the tested property frames, N38 has a
reproducible cold-relative tilt. V15 also demonstrated that typography later in
a sentence can change the winning Atlas destination without changing N38's own
earlier peak, because the route is a competition among all neurons' maxima.

## Where the work ended

By the end of the sequence, the project had produced more than a collection of
interesting neuron anecdotes. It had a reusable progression from full Atlas
route, to token trace, to exhaustive certificate, to controlled linguistic
factorial, and finally to causal intervention with sham, positional, matched-
neuron, dose, and calibration controls.

Two preliminary neuron phenotypes emerged. N541 shows unusually concentrated
observational selectivity for completed connected parallel or recurrent
constructions, together with a modest and temporally specific downstream causal
role. N38 shows a broader observational response to compact state- or
event-establishing predications, together with a more specific causal tilt
toward cold-related completions. Neither has been reduced to a single settled
semantic label, and the negative or qualified causal results are part of the
finding rather than failures of the probes.

The larger methodological lesson is that an Atlas route is only a starting
address. Long strings invite plausible stories about the wrong words;
minimisation locates the responsible region, competitor analysis explains why
it wins, matched controls test the apparent family, and intervention determines
which parts of that family have downstream causal force. The day's work made
that chain substantially more concrete, reproducible, and cautious.
