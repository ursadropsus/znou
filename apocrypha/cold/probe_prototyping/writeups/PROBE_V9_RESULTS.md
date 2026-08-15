# STRING ITERATION PROBE v9 — RESULTS

Status: preliminary result note. Tags follow `SPEC.md`: `MEASURED`, `DERIVED`,
`INTERPRETATION`, `PROPOSAL`.

Probe v9 examined 80 full-sentence *Moby-Dick* routes to GPT-2 Small layer-5
neuron 541. It tested whether a large within-neuron sample would collapse onto
recurrent short routes, whether those routes remained attached to the original
target peak, and how minimum length, word legibility, winning margin, and
continuation behaviour differed.

The result is the strongest route-family concentration observed in this probe
series so far. All 80 sentences reproduced neuron 541. Their shortest
certificates overwhelmingly reduced to compact paired constructions such as
`up and down`, `now and then`, `here and there`, `one by one`, `right and
left`, `true or false`, and `hand to hand`. In every route, neuron 541 reached
its certificate maximum on the final token, as the second member completed the
pair.

This is unusually coherent positive evidence. It remains a
Melville-conditioned Atlas sample containing only hits to one neuron. Until
matched non-hits and held-out corpora are tested, the result is properly called
route-family concentration rather than a complete feature interpretation.

---

## 1. Run provenance — MEASURED

```text
script          probe_v9.py
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

Input:

```text
J5-541 (80 Hits).txt
```

Outputs:

```text
results/probe_v9_summary_20260815-041244.tsv
results/probe_v9_certificates_20260815-041244.tsv
results/probe_v9_ladder_20260815-041244.tsv
results/probe_v9_extensions_20260815-041244.tsv
results/probe_v9_audit_20260815-041244.tsv
results/probe_v9_exact_families_20260815-041244.tsv
results/probe_v9_overlap_20260815-041244.tsv
```

Row counts reconcile across the seven outputs:

| Output | Rows | Unit |
|---|---:|---|
| Summary | 80 | one per source route |
| Certificates | 566 | selected certificate classes and Pareto members |
| Anchored ladder | 2,721 | one per evaluated left boundary |
| Right extensions | 2,398 | one per restored source endpoint |
| Exhaustive audit | 12 | one per audited route |
| Exact families | 47 | one per exact `TOKEN_MIN` string |
| Pairwise overlap | 3,160 | every pair among 80 routes |

All files record the same model revision, layer, hash, and actual CUDA device.
All 566 selected certificate rows passed realised-token round-trip checking.
`MEASURED`

The run was interrupted once by a transient Windows checkpoint-file lock. The
checkpoint writer was revised to use unique temporary files and retry atomic
replacement. The run resumed from the last valid checkpoint; completed route
measurements were not recomputed or manually substituted.

The TSVs use the probe-recorder convention: tab separation with quotation marks
treated literally. Generic CSV readers should use `quoting=csv.QUOTE_NONE` or
an equivalent literal-TSV mode.

---

## 2. Design

### 2.1 Sample

The Atlas export contained 80 unique *Moby-Dick* sentences labelled J5-541.
Full-sentence length ranged from 11 to 289 realised GPT-2 tokens, with a median
of 46.5 tokens.

This is a large within-neuron positive sample. It is not a random sample of
Melville, natural language, or the route space. The Atlas selected every row
because it had already reached neuron 541.

### 2.2 Boundary profiles

For each verified route, v9 evaluated every left deletion ending at the
full-sentence position where neuron 541 reached its maximum. From this anchored
ladder it selected:

```text
PREFIX          source prefix ending at the original 541 peak
TOKEN_MIN       shortest winning peak-anchored certificate
WORD_ANCHORED   shortest winning word-bounded span around that peak
MARGIN_MAX      widest-margin winner in the anchored ladder
PARETO          winners not beaten on both token length and margin
```

Starting from `TOKEN_MIN`, the probe then restored the remaining source tokens
one endpoint at a time. This right-extension ladder measured whether the
original source continuation narrowed the margin, changed the strongest
competitor, or caused neuron 541 to lose after its earlier left context had
been removed.

### 2.3 Exhaustive audit

V9 did not assume that the shortest successful span would remain attached to
the original target peak. Twelve routes were selected before reduction using
the Atlas focus row and evenly spaced source-length strata. Every contiguous
source-token span of those routes was evaluated.

The audit tested whether the global contiguous minimum agreed with
`TOKEN_MIN`. It did not turn the remaining 68 anchored results into exhaustive
searches.

---

## 3. Terms

### 3.1 Paired or binomial construction

In this report, a **paired construction** is a sequence in which two terms are
linked by a connector or repeated frame, for example:

```text
right and left
true or false
top to bottom
one by one
night after night
```

**Binomial** is useful linguistic shorthand for many coordinated two-member
expressions. The observed family is somewhat broader than conventional fixed
binomials because it also contains repeated and correlative frames.

“Reciprocal binomial” is not used as the umbrella label. Some routes are
reciprocal or alternating, while `You and I`, `seldom or never`, `one by one`,
and `each and every` instantiate different relations. Reciprocity is one
possible subfamily rather than the measured common denominator.

### 3.2 Construction completion

The arrival of the second member of the observed pair. In all 80 shortest
certificates, neuron 541 peaks on the certificate's final token. “Completion”
describes this position-level regularity. It does not yet establish that the
neuron computes an abstract syntactic completion operation.

### 3.3 Exact family and normalized textual family

An **exact family** contains byte-identical realised certificate strings.
Leading space and capitalization are preserved because they can change GPT-2
tokenisation and activation.

A **normalized textual family** lowercases and strips surrounding whitespace
only for secondary descriptive counting. It is not used as a model-input
equivalence class.

### 3.4 Target peak and signed target margin

The **target peak** is neuron 541's maximum activation across the evaluated
string. The **signed target margin** is:

```text
peak(541) - highest non-541 peak
```

A positive margin means neuron 541 wins. Target activation and relative
selectivity are distinct: shortening can lower the target peak while lowering
the competitor envelope even further.

### 3.5 Word-anchored certificate

A certificate whose source boundaries do not fall inside a written word-like
run. Internal hyphens and apostrophes are treated as belonging to that run.

`WORD_ANCHORED` is not a globally word-minimal string. It is the shortest
successful word-bounded span around the original target peak under v9's
declared search.

### 3.6 Pareto certificate

A tested certificate for which no other candidate is both shorter and
wider-margin, with at least one strict improvement. A route can therefore have
several Pareto certificates representing genuine length/clearance trade-offs.

### 3.7 Route-family concentration

Repeated recovery of identical or structurally related local routes from
independent full sentences reaching the same neuron. It is evidence that the
positive corpus hits share route material. Without matched negatives, it does
not determine whether the material is sufficient, specific, corpus-independent,
or a complete account of the neuron.

---

## 4. Validation — MEASURED

```text
parsed source routes             80
unique source routes             80
full sentences reaching 541      80/80
successful anchored minima       80/80
round-trip-stable certificate rows 566/566
exhaustive audit agreement       12/12
```

The twelve audited routes comprised 60,749 contiguous spans, of which 25,240
reached neuron 541. In every case, exhaustive `CONTIG_AUDIT` and anchored
`TOKEN_MIN` selected the same realised string.

V7 previously found anchored/global agreement in 42/42 routes. Across the two
runs, the observed agreement is now:

```text
v7                         42/42
v9 audit                   12/12
combined observation       54/54
```

`INTERPRETATION` — peak anchoring is increasingly well supported as an
efficient screening method for these naturally discovered Resonance routes.
The 54/54 observation does not prove universal agreement and should not be
silently converted into a theorem or an unchecked implementation assumption.

---

## 5. Primary result: paired-construction concentration

### 5.1 Every shortest route retains the constructional shape — MEASURED,
INTERPRETATION

The 80 measured `TOKEN_MIN` strings can all be consistently annotated as a
recognisable paired, coordinated, alternating, reciprocal, correlative, or
repeated construction. Representative examples include:

```text
Right and left
You and I
here and there
one by one
zone by zone
Wave after wave
right to left
seldom or never
large and small
top to bottom
direct and indirect
hand to hand
whenever and wherever
night after night
first to last
better or for worse
Human or animal
shoulder to shoulder
male and female
each and every
again and again
true or false
in and out
side to side
```

The family spans several surface connectors:

```text
X and Y
X or Y
X to Y
X by X
X after X
```

This makes a single-token explanation such as “the `and` neuron” inadequate
to the positive sample. `and` is common, but many successful routes use `or`,
`to`, `by`, or `after`, and neuron 541 peaks after the connector rather than
on it.

### 5.2 Peak location — MEASURED

Neuron 541 reached its maximum on the final realised token of `TOKEN_MIN` in:

```text
80/80 routes
```

The full sentence and its `TOKEN_MIN` certificate identified the same target
peak token in:

```text
80/80 routes
```

Examples:

| Certificate | 541 peak token |
|---|---|
| `right and left` | ` left` |
| `here and there` | ` there` |
| `one by one` | second ` one` |
| `top to bottom` | ` bottom` |
| `direct and indirect` | ` indirect` |
| `true or false` | ` false` |

`INTERPRETATION` — the strongest shared positional description is that neuron
541 becomes maximally active when the second term completes a paired
construction. This is more specific than phrase co-occurrence and broader than
any one semantic relation. Whether it reflects syntactic coordination,
lexicalised binomials, expectancy, token association, or overlapping
subfeatures remains open.

---

## 6. Exact and normalized recurrence

### 6.1 Exact realised strings — MEASURED

The 80 routes reduced to 47 exact `TOKEN_MIN` strings. Eight exact families
recurred across 41 routes:

| Exact certificate | Count |
|---|---:|
| ` up and down` | 11 |
| ` now and then` | 10 |
| ` here and there` | 8 |
| ` one by one` | 3 |
| ` right and left` | 3 |
| ` old and new` | 2 |
| ` one after the other` | 2 |
| ` this or that` | 2 |

These are exact model inputs, including leading spaces. Identical strings
necessarily reproduce identical deterministic measurements on the pinned
stack; the substantive recurrence is that independent full sentences reduce
to the same short input.

### 6.2 Descriptive normalization — DERIVED

Lowercasing and stripping surrounding whitespace yields 41 unique textual
forms. Ten normalized families recur across 49 routes:

| Normalized certificate | Count |
|---|---:|
| `now and then` | 12 |
| `up and down` | 11 |
| `here and there` | 8 |
| `right and left` | 4 |
| `one after the other` | 3 |
| `one by one` | 3 |
| `by and by` | 2 |
| `east and west` | 2 |
| `old and new` | 2 |
| `this or that` | 2 |

Normalization is useful for human description but should not replace the
exact-string table. For example, sentence-initial capitalization and the
presence of a leading-space token can measurably change target peaks and
margins.

### 6.3 Textual singletons remain structurally related — MEASURED,
INTERPRETATION

Most non-recurring exact strings retain the same broad form:

```text
again and again
east and west
each and every
top to bottom
hand to hand
zone by zone
Wave after wave
direct and indirect
seldom or never
better or for worse
```

Thus exact recurrence captures only part of the concentration. The remaining
routes are often novel lexical instances of a shared constructional shape.
Calling them one feature family is a plausible interpretation to be tested,
not a clustering fact already established by English readability.

---

## 7. Compression and human legibility

| Representation | Minimum tokens | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Full sentence | 11 | 46.5 | 62.99 | 289 |
| `TOKEN_MIN` | 3 | 3 | 3.61 | 13 |
| `WORD_ANCHORED` | 3 | 3 | 3.66 | 13 |
| `MARGIN_MAX` | 3 | 5 | 10.24 | 154 |

`TOKEN_MIN` length distribution:

```text
3 tokens     63 routes
4 tokens      7 routes
5 tokens      4 routes
6 tokens      1 route
8 tokens      3 routes
9 tokens      1 route
13 tokens     1 route
```

Seventy-four of 80 routes reduced to five tokens or fewer. `TOKEN_MIN` and
`WORD_ANCHORED` selected the same string in 79/80 routes.

The exception was route 76:

```text
TOKEN_MIN
y Dick swam swiftly round and round

WORD_ANCHORED
 horizontal attitude, Moby Dick swam swiftly round and round
```

The model-native result begins inside *Moby*. Unlike several v7 route families,
however, word-internal boundaries are not a dominant feature of the 541
sample. The recovered family is overwhelmingly human-legible.

---

## 8. Peak drift and winning margin

Shortening affected the target peak in both directions:

```text
negative target-peak drift       46/80
approximately zero                2/80
positive target-peak drift       32/80
```

The shortest certificate's target margin compared with the full sentence was:

```text
wider                             59/80
approximately equal               2/80
narrower                          19/80
```

Margin distributions:

| Representation | Minimum | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Full sentence | 0.012303 | 0.329038 | 0.395468 | 1.258146 |
| `TOKEN_MIN` | 0.000351 | 0.569719 | 0.663501 | 1.827452 |
| `WORD_ANCHORED` | 0.025641 | 0.569723 | 0.663917 | 1.827454 |
| `MARGIN_MAX` | 0.047954 | 1.006871 | 0.970106 | 2.199461 |

`DERIVED` — the v8 phenomenon is common but not universal in this sample.
Shortening usually widens the target's clearance, yet 19 routes become more
competitive. Lower target activation and wider margin can coexist because the
competitor envelope may fall further than neuron 541.

The smallest `TOKEN_MIN` margin, 0.000351, belongs to:

```text
y Dick swam swiftly round and round
```

This is a destination-preserving certificate by the declared tie and margin
rules, but its clearance is extremely thin. Minimum length should not be
treated as synonymous with robust or practically preferable.

---

## 9. Minimum length versus maximum margin

`TOKEN_MIN` and `MARGIN_MAX` selected the same string in only 23/80 routes.
The median margin rose from 0.569719 to 1.006871 when the probe was allowed to
retain more left context.

Examples:

```text
right to left
margin 0.039615

 from right to left
margin 2.130554
```

```text
now and then
margin 0.569724

, divine intuitions now and then
margin 1.973842
```

The longest margin-maximising route retained 154 tokens before `right and
left`. That candidate is valid under the mathematical objective and awkward
as a human certificate. It shows why maximum margin should remain a separate
product rather than replacing minimum length.

The median route had two Pareto certificates; the observed range was 1 to 28.
A route can contain several non-dominated solutions trading compression
against clearance. There is often no unique operationally best certificate.

---

## 10. Attribution within sentences

Long sentences sometimes contained several visually plausible paired motifs.
Peak attribution and minimisation distinguished the route that actually set
the destination.

For example, six source sentences visibly contained `round and round`. One of
them reduced instead to:

```text
shoulder to shoulder
```

The later `round and round` did not overtake the earlier neuron-541 peak. Other
`round and round` routes required markedly different preceding material:

```text
hair braided and coiled round and round
and thus round and round
, go round and round
water there, wheeling round and round
y Dick swam swiftly round and round
```

`INTERPRETATION` — phrase counting alone would over-attribute the later visible
motif and understate route diversity. The `round and round` subfamily also
appears more carrier-dependent than the compact `up and down`, `now and then`,
and `here and there` families. This motivates a controlled carrier sweep.

---

## 11. Source-continuation behaviour

Starting from each `TOKEN_MIN` certificate and restoring the source's remaining
tokens one endpoint at a time:

```text
never lost destination 541             73/80
never fell below certificate margin    42/80
encountered a new strongest competitor 38/80
eventually lost destination 541         7/80
```

This is 91.25% destination retention under the exact observed source
continuations. It is not a random or generative continuation sweep.

The seven losing extension ladders began from shortened strings, so their final
rows were suffixes of the original source rather than the verified full
sentence. Their losses do not contradict 80/80 full-sentence verification.
They show that deleting early context and then retaining extensive later text
can create a different competitive route.

`INTERPRETATION` — neuron 541's compact routes are usually robust to their
attested remaining continuation, while margins and runner identities remain
more dynamic. Carrier dependence and continuation dependence should continue
to be measured separately.

---

## 12. Preliminary route phenotype — INTERPRETATION

A compact description supported by v9 is:

> Neuron 541 shows a strongly concentrated, predominantly human-legible route
> family in which its maximum tends to occur as the second member completes a
> paired construction. Exact routes recur across independent Melville
> sentences, while the relation between members varies across coordination,
> alternation, opposition, reciprocity, correlation, and iteration.

This is tighter than neuron 508's `in`-bearing concentration and better
replicated than neuron 906's single `shad` route. It is therefore the strongest
feature-like signal produced by this probe sequence so far.

“Feature-like” is deliberately provisional. Positive examples alone cannot
separate at least four readings:

```text
H1  GENERAL COORDINATION
    541 responds broadly when a second constituent completes X connector Y.

H2  LEXICALISED BINOMIALS
    541 responds preferentially to familiar conventional pairs.

H3  SEVERAL SUBFAMILIES
    multiple narrower features or routes converge on the same neuron.

H4  MELVILLE-CONDITIONED CONCENTRATION
    the recurrence reflects corpus vocabulary, style, and Atlas selection more
    than a corpus-independent neuron property.
```

These hypotheses are compatible in mixtures. The next probe should estimate
their relative support rather than force a single categorical answer.

---

## 13. What v9 establishes

- `MEASURED` All 80 supplied sentences reproduce destination 541 on the pinned
  CUDA stack.
- `MEASURED` All 80 routes admit a peak-anchored contiguous certificate.
- `MEASURED` The shortest certificates have a median length of three tokens.
- `MEASURED` The 80 routes reduce to 47 exact certificate strings.
- `MEASURED` Eight exact recurrent families cover 41 routes.
- `MEASURED` Descriptive case/space normalization yields ten recurrent families
  covering 49 routes.
- `INTERPRETATION` Every measured shortest certificate can be consistently
  annotated as retaining a paired construction.
- `MEASURED` Neuron 541 peaks on the final token of every shortest certificate.
- `MEASURED` Full sentences and shortest certificates agree on the 541 peak
  token in 80/80 routes.
- `MEASURED` Exhaustive and anchored minima agree in the 12-route audit.
- `MEASURED` Token-minimal and word-anchored certificates agree in 79/80 routes.
- `MEASURED` Seventy-three shortest routes retain 541 through their exact
  observed remaining source continuation.
- `DERIVED` Minimum length, human legibility, target activation, margin, and
  source-continuation retention are distinct certificate properties.

---

## 14. What v9 does not establish

- The sample contains no matched non-541 controls.
- The sample contains no held-out author or corpus.
- Commonness among hits is not the same as enrichment over ordinary Melville
  text.
- Pair completion has not been shown sufficient to reach 541.
- The positive data do not distinguish familiar fixed binomials from arbitrary
  syntactic coordination.
- The positive data do not determine whether 541 contains one broad feature or
  several convergent subfamilies.
- Exact source-continuation retention does not establish robustness to arbitrary
  continuations.
- A target peak on the second member does not prove that the neuron computes an
  abstract linguistic relation.
- V9 observes activation and argmax behaviour; it does not perform neuron
  ablation, activation patching, or downstream causal analysis.
- Anchored/global agreement in 12/12 v9 audits and 54/54 cumulatively remains an
  empirical observation, not a universal guarantee.

---

## 15. Next experimental step — PROPOSAL

V9 has enough positive Melville evidence. The next gain comes from a compact
discriminative panel containing positives, matched negatives, and carrier
interventions.

### 15.1 Held-out attested binomials

Test familiar paired expressions not observed in the supplied Melville hit
sample, for example:

```text
salt and pepper
knife and fork
back and forth
pros and cons
black and white
sooner or later
```

These ask whether the family transfers to conventional pairings beyond the
Atlas sample. Candidate strings should be checked against the source corpus
before being labelled absent from Melville; absence should be measured rather
than assumed from memory.

### 15.2 Matched non-idiomatic coordination

Construct or sample less conventional pairs while holding the surface frame
similar:

```text
cats and elephants
bread and gravel
lanterns and arithmetic
velvet or engines
```

Matched token length, leading-space regime, capitalization, connector, and
position should be recorded. If arbitrary well-formed `X and Y` examples
reach 541, that supports a broader coordination/completion account. If familiar
binomials substantially outperform them, lexicalised construction knowledge
becomes more plausible.

Non-idiomatic does not mean nonsensical by definition; plausibility should be
varied or matched rather than allowed to confound lexical familiarity.

### 15.3 Melville matched non-hit controls

Retrieve Melville sentences or local spans containing paired constructions
that do not reach neuron 541. Match them where possible on connector, token
length, target position, capitalization, and source context.

This supplies the missing denominator:

```text
observed now:     frequent among selected 541 hits
needed next:      enriched among 541 hits relative to comparable Melville text
```

If the same construction is common among non-hits, additional carrier,
position, lexical, or competition conditions are required.

### 15.4 Carrier sweep

Use `up and down` as the primary seed because it is short, exact across eleven
independent reductions, has a substantial margin near 1.1565, and is
human-legible. Hold the three-token string fixed while varying preceding
material separately from following continuation.

Include:

```text
position-1 / BOS form
short function-word carriers
content-word carriers
punctuation carriers
attested Melville carriers
random safe-token carriers
```

Record retention, target-peak drift, competitor envelope, runner identity, and
position of the 541 peak. V6 showed that routes with similarly compact strings
can differ sharply in carrier robustness; the eleven exact recurrences do not
themselves establish robustness because their reduced certificate input is
identical after left context is removed.

### 15.5 Frame-breaking controls

For each seed family, test interventions that isolate the apparent structure:

```text
delete the connector          up down
retain only the first term    up
retain only the second term   down
replace the second term       up and lantern
replace the first term        gravel and down
reverse conventional order    down and up
change the connector          up or down / up to down
repeat without connector      up up
```

These should be described operationally as substitutions and deletions, not as
pure semantic manipulations. Tokenisation and position can change with the
surface edit and must be logged.

### 15.6 Cross-corpus positives

Search at least one held-out literary corpus and one broader corpus for natural
routes to neuron 541. Apply the same minimisation without selecting sentences
for visible paired phrases.

Three outcomes would all be informative:

```text
same paired families recur       supports corpus transfer
new paired families emerge       supports a broader construction family
unrelated minima dominate        supports polysemy or Melville concentration
```

The probe should retain failures and unrelated 541 routes rather than reporting
only examples that agree with the present interpretation.

---

## 16. Suggested decision table for the next run

| Held-out result | Evidence favours | Remaining caution |
|---|---|---|
| Attested binomials retain; arbitrary pairs mostly fail | lexicalised/conventional construction sensitivity | familiarity may covary with frequency and tokenisation |
| Both attested and arbitrary pairs retain | broader coordination or pair-completion sensitivity | negative frames and position controls still required |
| Melville hits retain but held-out pairs fail | corpus/route concentration or carrier dependence | test exact phrases in new carriers before concluding corpus specificity |
| Several connectors retain but with distinct rival profiles | convergent subfamilies within 541 | one argmax neuron can mix several activation routes |
| Natural cross-corpus 541 hits reduce to the same families | corpus-independent route-family evidence | still not a complete causal feature interpretation |

The most efficient next version is therefore a controlled neuron-541 panel,
not another large positive-only Melville collection. V9 has already supplied a
clear candidate phenomenon. The next task is to determine its scope,
specificity, carrier dependence, and transfer.
