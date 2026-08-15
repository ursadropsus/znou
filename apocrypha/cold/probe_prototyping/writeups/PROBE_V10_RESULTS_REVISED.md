# STRING ITERATION PROBE v10 — RESULTS

Status: revised preliminary result note. This revision incorporates a second
independent reading of the v10 outputs while preserving the first report as a
separate paper-trail artifact. Tags follow `SPEC.md`: `MEASURED`, `DERIVED`,
`INTERPRETATION`, `PROPOSAL`.

Probe v10 followed up the route-family concentration found for GPT-2 Small
layer-5 neuron 541. V9 had reduced 80 Melville routes to short paired
constructions such as `up and down`, `now and then`, `here and there`, and
`one by one`. V10 asked what those positive examples did not settle: whether
541 was responding to isolated endpoint words, literal memorised strings,
arbitrary coordination, familiar lexical pairings, connector choice, or some
combination of these factors.

The principal result is strongly discriminating. Across six recurrent seed
pairs and four carriers, the canonical pair retained neuron 541 in 21/24
cases. Replacing either lexical member with a token-matched arbitrary word
reduced this to 0/24, as did replacing both. The second member alone, the two
members without a connector, an arbitrary first member followed by the
canonical second, and repetition of the first member also retained 541 in
0/24 cases each. This makes an isolated-second-word account inadequate for
the tested routes.

The response is nevertheless more flexible than literal phrase memorisation.
Reversed canonical pairs retained 541 in 16/24 cases, and replacing the
connector with `or` retained it in 22/24. Familiar held-out pairs produced
substantially higher raw 541 peaks than the nominally arbitrary panel, although
one same-category control, `cats and elephants`, also won in two carriers.

The most defensible current description of the part tested by v10 is therefore
**paired-relation construction sensitivity**: 541 is sensitive to combinations
of two lexical members and a compatible relational connector, with graded
tolerance for order, carrier, and lexical familiarity. V9 also recovered a
second conspicuous family—reduplicative or recurrent frames such as `one by
one`, `Wave after wave`, and `again and again`—which v10's connector matrix did
not properly cross. The broader phenotype may therefore contain several
related constructional subfamilies.

This is a working phenotype, not a complete feature interpretation. The
experiment does not yet separate opposition, alternation, conventional
binomiality, semantic relatedness, recurrence, and corpus familiarity cleanly
enough to name one of them as the neuron’s feature.

---

## 1. Run provenance — MEASURED

```text
script          probe_v10.py
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
results/probe_v10_preflight_20260815-044753.tsv
results/probe_v10_variants_20260815-044754.tsv
results/probe_v10_comparisons_20260815-044754.tsv
results/probe_v10_trace_20260815-044754.tsv
results/probe_v10_summary_20260815-044754.tsv
```

Row counts reconcile across the outputs:

| Output | Rows | Unit |
|---|---:|---|
| Preflight | 528 | realised-token and boundary checks |
| Variants | 522 | one per evaluated string case |
| Comparisons | 794 | declared within-carrier and carrier comparisons |
| Trace | 2,596 | one per realised token position |
| Summary | 98 | grouped descriptive summaries |

All files record the same pinned model revision, layer, parameter hash, and
actual CUDA device. `MEASURED`

The TSVs use the probe-recorder convention: tab separation with quotation
marks treated literally. Generic CSV readers should use
`quoting=csv.QUOTE_NONE` or an equivalent literal-TSV mode.

---

## 2. Question and design

### 2.1 Core seeds

V10 began with six recurrent v9 pairs:

```text
up and down
now and then
here and there
right and left
true or false
east and west
```

Each seed was evaluated under four declared carriers:

```text
BARE          construction begins at the first token position
V9_SPACE      leading-space form matching the common v9 certificate form
THE_PHRASE    embedded after “the phrase”
THEY_SAID     embedded after “they said”
```

A **carrier** is surrounding text used to place the construction in a changed
context and token position. Carrier tests ask whether the local route survives
that change. They do not make the embedded phrase semantically natural in
every case, and they can introduce new competing-neuron peaks.

### 2.2 Core lexical factorial

For each seed, v10 selected one arbitrary replacement for each canonical
member. Canonical and arbitrary replacements were matched at one realised
leading-space GPT-2 token:

| Canonical pair | Arbitrary replacement pair |
|---|---|
| `up` / `down` | `gravel` / `window` |
| `now` / `then` | `velvet` / `ocean` |
| `here` / `there` | `marble` / `forest` |
| `right` / `left` | `garden` / `violin` |
| `true` / `false` | `candle` / `basket` |
| `east` / `west` | `pocket` / `pencil` |

The factorial cells were:

```text
CC    canonical first + connector + canonical second
CA    canonical first + connector + arbitrary second
AC    arbitrary first + connector + canonical second
AA    arbitrary first + connector + arbitrary second
```

This crossing is the central test of the endpoint account. If the canonical
second member alone carried the response, `AC` strings such as
`gravel and down` should remain effective. If the first member alone carried
it, `CA` should remain effective. If arbitrary coordination were sufficient,
`AA` should often work.

### 2.3 Other branches

The endpoint branch separately tested the canonical second member alone, the
connector plus second member, both members without the connector, an arbitrary
first plus canonical second, repetition of the first, the full frame, and the
reversed canonical frame.

The connector matrix held the lexical members fixed while substituting
`and`, `or`, `to`, `by`, and `after`. A held-out pair panel compared eight
familiar expressions absent from the 47 exact v9 certificate families with
eight nominally arbitrary coordinations. Finally, the continuation branch
appended six suffixes to three established constructions under two carriers.

These branches are controlled string interventions, not samples drawn
independently from a population. Rows sharing a seed, member, or carrier are
correlated. The report therefore uses descriptive contrasts rather than
naive significance tests over rows.

---

## 3. Terms

### 3.1 Raw activation, peak, and completion

**Raw activation** is neuron 541’s value at one token position. Its **peak**
is the largest raw activation it reaches anywhere in the completed string.

**Construction completion** is the final realised token of the second lexical
member. The **raw completion increment** is the raw 541 activation at that
position minus its raw activation immediately before the second member. It is
a temporal contrast inside one forward pass, not a causal estimate of what
the second member contributed. Earlier tokens have already changed the model
state at completion.

The statement **541 peaks at completion** means its largest activation in that
string occurs at the construction-completion position. It does not by itself
show that the neuron computes an abstract completion operation.

### 3.2 Retention, destination, and signed winning margin

The probe applies the project’s Resonance readout: each layer-5 neuron is
represented by its largest activation anywhere in the string. The neuron with
the largest such peak is the string’s **destination**.

Neuron 541 is **retained** when it remains that destination. Its signed margin
is:

```text
peak(541) - highest peak reached by any other layer-5 neuron
```

A positive margin means 541 wins; a negative margin means it responds but some
competitor responds more strongly. Raw activation and retention answer
different questions. A carrier can increase 541’s peak while narrowing its
margin because it raises a competitor even more.

### 3.3 Strict matched comparison

A comparison is marked **strict** only when the relevant realised-token counts
and completion positions match. V10 recorded 404 within-carrier comparisons,
of which 271 met the declared strict criteria. Carrier-versus-bare comparisons
were intentionally not called strict because the carrier changes position and
adds tokens.

### 3.4 Working phenotype

A **phenotype** here is a reproducible behavioural profile under a declared
test panel. “Paired-relation construction sensitivity” summarises the current
profile without claiming that it is a complete causal, linguistic, or
corpus-independent description of neuron 541.

---

## 4. Validation and overview — MEASURED

```text
evaluated string cases              522
cases retaining neuron 541          189
cases with 541 peak at completion   364
retained cases peaking at completion 189/189
within-carrier comparisons          404
strict within-carrier comparisons   271
carrier-versus-bare comparisons     390
```

The aggregate 189/522 retention rate is not itself a feature estimate. The
522 cases deliberately mix positive forms, destructive controls, connector
substitutions, continuations, and carrier-only controls. Branch-specific
contrasts carry the interpretation.

Preflight confirmed that all six canonical and arbitrary lexical members used
in the factorial occupied one realised token in their declared leading-space
form. It also recorded each realised string, token sequence, second-member
boundary, and completion position before inference. `MEASURED`

Every one of the 189 v10 cases that retained neuron 541 placed its 541 peak at
construction completion. V9 had observed the same position in all 80 source
routes. This gives 269 concordant strings across the two probes. They are not
269 independent samples: v10 deliberately reuses seeds, members, carriers,
and transformations, and its starting constructions were informed by v9.
The result is nevertheless a striking positional regularity across varied
controlled forms. `MEASURED`, `INTERPRETATION`

---

## 5. Primary result: both lexical members matter

### 5.1 Core factorial — MEASURED

| Factorial cell | Cases retaining 541 | Cases |
|---|---:|---:|
| `CC` — both canonical | 21 | 24 |
| `CA` — arbitrary second | 0 | 24 |
| `AC` — arbitrary first | 0 | 24 |
| `AA` — both arbitrary | 0 | 24 |

The separation appears in 541's own raw peak, not only in whether another
neuron wins:

| Factorial cell | Mean 541 peak | Mean signed margin |
|---|---:|---:|
| `CC` | 4.007 | 0.631 |
| `CA` | 0.139 | -2.930 |
| `AC` | 0.025 | -3.025 |
| `AA` | 0.808 | -2.254 |

All six canonical pairs retained 541 in the bare and v9-leading-space forms.
Five of six retained it after `the phrase`; four of six retained it after
`they said`.

Among strict token-position-matched comparisons against `CC`, replacing the
second member lowered the 541 peak by 3.87 on average across 24 comparisons.
Replacing the first lowered it by 4.06 across 19 strict comparisons, and
replacing both lowered it by 3.29 across 19. None of the substituted cases
retained 541.

`INTERPRETATION` — the tested response depends jointly on lexical material
from both sides of the connector. In particular, the failure of every `AC`
case strongly disfavors the proposal that v9 merely rediscovered detectors
for terminal words such as `down`, `then`, `there`, `left`, or `west`.

The gap is large rather than a threshold-only change: both mixed cells reduce
the mean target peak from about 4.0 to close to zero. At the same time, `AA`
has a higher mean than either mixed cell, partly because particular arbitrary
combinations such as `candle or basket` activate 541 more than others. The
factorial therefore supports conjunction of compatible lexical material; it
does not make “conventionality” a single monotonic dose variable.

This conclusion is bounded. Each canonical member had only one arbitrary
replacement, and all replacements were selected from a small pool. Token-count
matching controls one important confound but does not match word frequency,
semantics, morphology, or each replacement’s pre-existing activation profile.

### 5.2 Endpoint controls — MEASURED

| Variant | Retained 541 | Cases |
|---|---:|---:|
| Full canonical frame | 21 | 24 |
| Reversed canonical frame | 16 | 24 |
| Second member alone | 0 | 24 |
| Connector + second | 0 | 24 |
| First + second, no connector | 0 | 24 |
| Arbitrary first + canonical second | 0 | 24 |
| Repeated first member | 0 | 24 |

The endpoint branch converges with the factorial: neither an isolated second
member nor merely placing both members in one short string reproduced the
canonical result. For these six distinct-member seeds, the connector
participates in the route rather than being incidental.

The reversal result adds a different constraint. Reversal retained 541 in all
four carriers for `down and up` and `then and now`; in three for `left and
right` and `west and east`; in two for `there and here`; and in none for
`false or true`.

`INTERPRETATION` — 541 is not adequately described as a fixed ordered-string
detector. Order tolerance is graded and pair-dependent. The complete failure
of `false or true` also prevents reversal invariance from becoming the next
overgeneralisation.

---

## 6. Connector compatibility is structured

### 6.1 Connector matrix — MEASURED

Each connector was crossed with the six core lexical pairs and four carriers:

| Connector | Retained 541 | Cases | Mean 541 peak | Mean signed margin |
|---|---:|---:|---:|---:|
| `or` | 22 | 24 | 4.135 | 0.571 |
| `and` | 18 | 24 | 3.910 | 0.520 |
| `to` | 6 | 24 | 2.449 | -0.581 |
| `by` | 3 | 24 | 1.956 | -1.070 |
| `after` | 0 | 24 | 1.081 | -1.842 |

Neuron 541 peaked at the second member’s completion in 117/120 connector cases,
including many strings that did not retain it as destination.

`OR` was not merely tolerated for the original `true or false` seed. It
generalised broadly across `up or down`, `now or then`, `here or there`,
`right or left`, and `east or west`. `TO` worked selectively, most consistently
for the east/west pair. `AFTER` never retained 541 in this matrix.

`INTERPRETATION` — literal `and` detection is inadequate, while connectors are
not interchangeable. `And` and `or` can both express a relation between paired
alternatives or poles in these seeds. The weaker, pair-specific behavior of
`to`, `by`, and `after` is compatible with relational or constructional
subfamilies rather than one connector-blind template.

The fact that `or` slightly exceeds `and` in this panel should not be ranked as
a universal preference. The matrix contains only six lexical pairs, one of
which is canonically an `or` expression, and repeated rows share their lexical
members.

### 6.2 The matrix tested one frame, not each connector's native frame

V10 held distinct members in the form `X connector Y`. That is an appropriate
test of connector substitution for `up and down`-like routes. It does not
measure every construction in which the same connector occurred in v9.

Several v9 families instead repeat a member or express recurrence:

```text
Wave after wave
one after the other
one by one
zone by zone
hand to hand
round and round
again and again
by and by
```

Consequently, `up after down` failing does not show that `after` is generally
weak for neuron 541. It shows that `after` was weak between the selected
distinct members. The v9 evidence suggests at least two candidate frames:

```text
X connector Y    paired, contrasted, or parallel members
X connector X    repeated, distributed, or recurrent member
```

`One after the other` shows that even this two-frame shorthand is incomplete;
pronominal and correlative recurrence may form a related third pattern.
`INTERPRETATION`

V10 therefore characterised the distinct-member family much more closely than
the reduplicative family. The low `after`, `by`, and `to` aggregate values must
not be promoted into context-free rankings of those connectors.

---

## 7. Familiar held-out pairs exceed nominally arbitrary pairs

### 7.1 Panel aggregate — MEASURED

The held-out panel contained eight familiar paired expressions absent from
v9’s 47 exact certificate families and eight comparison coordinations, each
under four carriers.

| Pair class | Retained 541 | Cases | Mean 541 peak | Median 541 peak | Mean margin |
|---|---:|---:|---:|---:|---:|
| Familiar/conventional | 8 | 32 | 2.937 | 2.984 | -0.441 |
| Nominally arbitrary | 2 | 32 | 0.639 | 0.224 | -2.325 |

The familiar expressions that retained 541 at least once were:

```text
back and forth
black and white
sooner or later
life and death
```

Several other familiar pairs reached substantial raw 541 activation without
winning the route. The destination criterion is therefore more conservative
than asking whether 541 participates.

`Give and take` is the clearest example: its mean 541 peak was 3.592 and its
maximum was 4.121, yet it retained 541 in 0/4 carriers because another neuron
won each complete string. A retention-only summary would make this strong
target response look like complete failure. Signed margin is therefore
load-bearing for cases in the competitive middle band.

The two arbitrary-panel wins were both forms of `cats and elephants`: bare and
after `they said`.

### 7.2 What the exception changes — INTERPRETATION

The panel supports an association between familiar pairings and stronger raw
541 response, but it is not a clean conventionality experiment. `Cats and
elephants` is grammatically ordinary and its members share the semantic
category *animals*. Calling it arbitrary describes the absence of a familiar
fixed binomial, not the absence of a relation.

That exception is informative. It suggests that 541 may respond to some novel
same-category coordination as well as lexicalised expressions. It also exposes
the next needed distinction: conventionality, semantic relatedness, opposition,
and token-level activation history were bundled together in the present panel.
The result is descriptive evidence for held-out generalisation, not yet a
measurement of conventionality by itself.

Semantic parallelism is one plausible explanation, because `cats` and
`elephants` are plural animate nouns from the same broad category. One pair
cannot establish that account: animacy, plurality, noun class, semantic
category, relative size, lexical frequency, and each token's existing network
effects all vary together. The result motivates matched semantic controls
rather than settling their interpretation.

---

## 8. Carrier and continuation behaviour

### 8.1 Established frames are raw-activation robust — MEASURED

For canonical `CC` cases, all six pairs retained 541 when bare and in the
v9-leading-space form. Embedding after `the phrase` produced five wins, while
embedding after `they said` produced four.

Mean raw 541 peak was 3.840 bare, 3.951 with the v9 leading space, 4.170 after
`the phrase`, and 4.068 after `they said`. The carrier therefore did not simply
suppress the target. Some losses occurred because the carrier created a still
stronger rival peak, narrowing the signed margin.

`INTERPRETATION` — the established local route is reasonably robust to these
three context changes at the level of target activation. Destination is more
fragile because Resonance compares 541 with every rival opportunity introduced
by the entire string.

### 8.2 Continuations preserve the recorded target peak — MEASURED

All 72 continuation cases retained 541. Its peak remained at construction
completion in all 72. Appending a period, `again`, a road continuation, a
clausal continuation, or `xyz` left the already-recorded 541 peak unchanged
apart from tiny floating-point-scale variation. Some continuations slightly
narrowed the winning margin by offering competitors new positions; `xyz`
produced the largest mean narrowing in this small panel.

This result is partly guaranteed by causal sequence processing and the
max-over-positions readout. Later tokens cannot retroactively change an
activation already recorded at an earlier position, although they can create a
larger rival peak and thereby change the destination. The 72/72 result is thus
evidence of continuation robustness under the tested competitor envelope, not
evidence that suffixes have no effect on the network.

---

## 9. What v10 rules against, supports, and leaves open

### 9.1 Strongly disfavoured within the tested regime — INTERPRETATION

An **isolated endpoint detector** is inconsistent with the 0/24 results for the
canonical second member alone and for arbitrary-first/canonical-second forms.

A **literal `X and Y` syntactic-coordination detector** is inconsistent with
the failure of arbitrary coordinations in the core factorial and the broad
success of `or`.

A **fixed memorised ordered-string detector** is incomplete because 16/24
reversed constructions and 22/24 `or` constructions retained 541.

These are strong local exclusions, not universal proofs. They apply to the six
core pairs, declared replacements, carriers, and GPT-2 revision measured here.

### 9.2 Supported working account — INTERPRETATION

The results jointly support a graded response to **lexically and relationally
compatible paired constructions**. Both members matter; an overt connector
matters; some connector substitutions preserve the response; and order can be
changed for many pairs. Familiar held-out pairs tend to produce substantially
more 541 activation than the comparison panel.

This account can include several overlapping sources of selectivity:

```text
lexical association between the two members
semantic relatedness or opposition
constructional familiarity
connector–pair compatibility
direction or order preference
token position and surrounding carrier
competition from other layer-5 neurons under Resonance
```

The data do not require these influences to collapse into one linguistically
pure feature. A neuron can pool several correlated regularities learned from
the training distribution.

### 9.3 Still open

V10 does not yet determine whether 541 responds primarily to oppositional
poles, conventional binomials, same-category pairing, alternative sets, or a
mixture. Nor does it estimate specificity across ordinary text, because the
experiment was built from known positive seeds and designed controls rather
than a balanced corpus of hits and non-hits.

It also does not determine how the distinct-member family relates to v9's
reduplicative and recurrent families. They may be separate learned frames, or
surface forms of a broader sensitivity to structured parallel slots. That
question is now a design gap rather than a minor connector detail.

The Melville origin of the seeds remains relevant, but held-out expressions
show that the response is not confined to the exact strings recovered from
Melville. Corpus-independent generalisation would require comparable tests in
other natural corpora rather than researcher-authored phrases alone.

---

## 10. Recommended next experiments — PROPOSAL

### 10.1 First priority: cross frame, connector, and lexical familiarity

The shortest high-value follow-up is a reduplication and recurrence panel. For
several connectors, it should compare an attested v9 frame, a familiar held-out
frame, a novel repeated member, a distinct related pair, an arbitrary distinct
pair, and connector removal. For example:

```text
Wave after wave       attested v9 reduplication
day after day         familiar held-out reduplication
gravel after gravel   novel repeated member
wave after tide       distinct related members
gravel after window   arbitrary distinct members
wave wave             connector removed
```

Parallel panels for `by`, `to`, and `and` would test whether each connector
prefers a frame rather than possessing a context-free strength. The crucial
novel cell is `gravel after gravel`: a strong response would show that a
compatible reduplicative structure can generalise beyond a memorised lexical
pair. If familiar repetitions respond while novel ones do not, lexical or
constructional familiarity remains important.

### 10.2 Second priority: matched semantic-relation panel

The next broader experiment should use matched relation classes rather than a
larger unsystematic phrase list. It should include multiple lexical samples per
class:

```text
conventional opposites       black and white
novel or rare opposites      shallow and deep
same-category, non-opposite  cats and elephants
associated, cross-category   knife and fork
frequency-matched unrelated  selected from a declared pool
```

Each lexical pair should be tested with `and`, `or`, reversal, and at least one
fixed carrier, with more than one arbitrary replacement per canonical member.
Word frequency, realised token count, capitalization, and leading-space form
should be recorded rather than assumed matched. A complementary corpus pass
could then estimate how often high 541 activation occurs outside known paired
constructions.

This would turn the present working phenotype into competing quantitative
accounts: conventionality, semantic relation, connector compatibility, and
lexical identity could each make different predictions on held-out cells.

---

## 11. Bottom line

`MEASURED` — canonical paired constructions retained neuron 541 in 21/24 core
carrier cases. Replacing either member, both members, or presenting the
canonical second member without its canonical first reduced retention to zero
in the corresponding 24-case panels. Reversal retained 16/24; `or` retained
22/24. Familiar held-out pairs produced markedly higher raw 541 peaks than the
nominally arbitrary panel, with `cats and elephants` providing an informative
same-category exception. All 189 retained v10 cases peaked at construction
completion, extending the positional regularity observed in v9 without
treating the related v10 cases as independent samples.

`INTERPRETATION` — v10 converts v9’s route-family concentration into a more
discriminating preliminary phenotype for distinct-member constructions.
Neuron 541 is sensitive to their joint lexical and relational structure. V9's
reduplicative and recurrent routes plausibly form another related subfamily
that v10 did not adequately cross. Across both, the response is richer than an
isolated word or fixed phrase account, while remaining graded, pair-dependent,
connector-and-frame-dependent, and entangled with the Resonance competitor
field.

`CAUTION` — “paired-relation construction sensitivity” is the best current
working label for the v10-tested family, not a final neuron name. The immediate
next step is to test reduplicative frames directly; the following step is to
separate semantic relation from conventionality and lexical familiarity using
multiple matched held-out samples.
