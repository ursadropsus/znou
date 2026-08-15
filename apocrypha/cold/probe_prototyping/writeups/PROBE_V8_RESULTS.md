# STRING ITERATION PROBE v8 — RESULTS

Status: preliminary result note. Tags follow `SPEC.md`: `MEASURED`, `DERIVED`,
`INTERPRETATION`, `PROPOSAL`.

Probe v8 followed up one striking result from v7. For a Melville route to
GPT-2 Small layer-5 neuron 906, the automatic token-minimal certificate was
shorter and won by a wider margin than the manually whittled certificate,
despite producing a lower activation peak in neuron 906 itself.

V8 resolves most of that apparent puzzle at the level of measured peaks and
token positions. The automatic certificate ends immediately after `shad`.
The manual certificate continues into `-bellied`; its first added token, the
hyphen, produces an almost tying peak in neuron 1830. The complete sentence
later supplies another rival opportunity at the `ad` token in `Bildad`, where
neuron 1790 rises close to neuron 906.

The result is therefore about both boundaries of a route. Earlier context can
change the strength of the target and its rivals. Later continuation cannot
remove a peak already recorded under Resonance, but it can add new positions
at which another neuron peaks more strongly. A shorter certificate may win
more clearly because it preserves enough context to reach its destination
while withholding opportunities from competitors.

This is a mechanism account for one route under the declared readout. It does
not yet identify what neuron 906 represents, or establish that the same
boundary mechanism is common across other routes.

---

## 1. Run provenance — MEASURED

```text
script          probe_v8.py
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
results/probe_v8_variants_20260815-030237.tsv
results/probe_v8_comparisons_20260815-030238.tsv
results/probe_v8_trace_20260815-030238.tsv
results/probe_v8_ladder_20260815-030238.tsv
```

The files contain four variant summaries, four declared pairwise comparisons,
161 token-position trace rows, and 30 anchored left-deletion rows. All record
the same pinned model hash and actual CUDA device. The comparison arithmetic
reconciles with zero recorded error. `MEASURED`

The TSVs use the existing probe-recorder convention: tab separation with no
syntactic meaning assigned to quotation marks. Generic CSV readers should use
`quoting=csv.QUOTE_NONE` or an equivalent literal-TSV mode.

---

## 2. Question and design

V7 compared four forms of the same discovered route:

```text
FULL      complete 100-token Melville sentence
PREFIX    full sentence cut immediately after neuron 906's peak
MANUAL    manually whittled, word-bounded certificate
AUTO      v7's shortest peak-anchored contiguous-token certificate
```

All four reach neuron 906. V8 records the target peak, the strongest competing
peak, the position at which each occurs, and the evolution of selected neuron
peaks as each token is processed. It also deletes the PREFIX's leftmost source
token one step at a time while holding its right boundary at the target peak.

The pairwise comparisons were chosen to separate several effects:

```text
FULL   -> PREFIX    remove continuation after the target peak
PREFIX -> AUTO      remove left context while keeping the same right boundary
MANUAL -> AUTO      compare the two v7 certificate products
FULL   -> MANUAL    locate the manual result relative to the source
```

These are descriptive interventions on strings. Only `FULL -> PREFIX` and the
anchored left-deletion ladder vary one boundary in a clean nested sequence.
`MANUAL -> AUTO` changes both boundaries and changes the runner's identity, so
its aggregate difference must not be assigned to one cause without consulting
the token trace.

---

## 3. Terms

### 3.1 Raw activation and running peak

The **raw activation** is a neuron's value at one token position. The
**running peak** is the largest raw activation that neuron has reached up to
and including that position.

Under Resonance, the final score for each neuron is its largest activation
anywhere in the whole string. The final score is therefore the running peak
at the last position.

### 3.2 Target, competitor envelope, and winning margin

Neuron 906 is the declared **target**. At any completed string, the
**competitor envelope** is the highest peak reached by any non-target neuron.
The identity of the neuron forming that envelope can change between strings.

The signed target margin is:

```text
target margin = peak(906) - highest non-906 peak
```

A positive margin means 906 is the destination. A negative margin means
another neuron wins. In a winning string this is the familiar Δ, or winning
clearance.

### 3.3 Runner and runner switch

The **runner** or **runner-up** is the neuron forming the competitor envelope
for a particular winning string. A **runner switch** means that two strings
have different strongest competitors.

This matters when interpreting aggregate changes. Saying that “the runner
fell” can be misleading if the original runner collapsed but a different
neuron became the new runner. The envelope can fall even while some individual
competitors rise.

### 3.4 Target-peak drift

Target-peak drift is the shortened string's neuron-906 peak minus the full
sentence's neuron-906 peak. Negative drift means 906 activates less strongly;
it does not necessarily mean 906 loses. If competing peaks fall further, the
target can win by a wider margin.

### 3.5 Left and right boundaries

The **left boundary** is where a contiguous certificate begins. The **right
boundary** is where it ends. V8 holds the right boundary at the `ad` token of
`shad` for the anchored ladder and moves only the left boundary.

The manual and automatic certificates have different right boundaries:

```text
MANUAL    ... in a broad shad-bellied
AUTO      ... in a broad shad
```

That difference is central to the result.

### 3.6 Carrier fragment

A **carrier** is material before the local region treated as the candidate
gate or target-bearing event. Here, `er in a broad` can provisionally be
described as carrier material for the target-bearing `shad` completion.

The word “carrier” is operational. It says that preceding material helps form
a successful measured route. It does not imply that every token in the carrier
is necessary, that the carrier has one semantic meaning, or that neuron 906 is
a detector for the following word.

`er` is the model-token suffix of *harpooneer*. It is a valid model-native
boundary inherited from v7's search, not a human word boundary.

### 3.7 Minimal versus optimal

`AUTO` is minimal under v7's declared objective: it is the shortest successful
decoded contiguous span attached to the original peak region. **Minimal** here
means shortest under that search. It does not mean highest activation, widest
margin, greatest robustness, or most human-legible route.

---

## 4. Primary measurements

All four variants reached neuron 906:

| Variant | Realised string | Tokens | 906 peak | Strongest rival | Margin |
|---|---|---:|---:|---:|---:|
| FULL | complete sentence | 100 | 3.503901 | 1790: 3.450959 | +0.052941 |
| PREFIX | `Rising ... in a broad shad` | 30 | 3.503901 | 1888: 3.367364 | +0.136537 |
| MANUAL | `the drabbest drab, to a harpooneer in a broad shad-bellied` | 21 | 3.207114 | 1830: 3.191885 | +0.015230 |
| AUTO | `er in a broad shad` | 6 | 3.095655 | 1888: 2.933532 | +0.162123 |

In every variant, neuron 906 reaches its maximum on the token `ad` that
completes `shad`. The activation magnitude depends on the preceding context,
but the target's peak-token location is stable across these four strings.
`MEASURED`

The automatic certificate has the lowest target peak of the four and the
widest final margin of the four. This is possible because destination is a
relative result: the winning neuron is determined by the target peak compared
with the competitor envelope, not by the target peak alone. `DERIVED`

---

## 5. The right-boundary mechanism

### 5.1 The manual certificate before and after the hyphen — MEASURED

Within the MANUAL token trace, neuron 906 reaches 3.207114 on `ad`. At that
moment it leads the full neuron set by 0.324461.

The next token is `-`:

| MANUAL position | Token | Raw 906 | Raw 1830 | Running 906 | Running 1830 | 906 margin |
|---:|---|---:|---:|---:|---:|---:|
| 18 | `ad` | 3.207114 | -0.152710 | 3.207114 | 0.852909 | +0.324461 |
| 19 | `-` | -0.139748 | 3.191885 | 3.207114 | 3.191885 | +0.015230 |

The hyphen does not lower neuron 906's already recorded running peak. It gives
neuron 1830 a new position at which to rise to 3.191885, almost tying 906.
The following `bell` and `ied` tokens do not surpass either recorded peak, so
the manual certificate finishes with the same narrow 0.015230 margin.

This directly explains why ending before `-bellied` can be more selective than
retaining the complete compound word. The result is stronger than a general
interpretation about “less context”: the trace locates the rival, the token,
the activation, and the exact point at which the margin collapses.

### 5.2 Continuation after the peak is not neutral under Resonance

Right-truncation from FULL to PREFIX leaves neuron 906's peak exactly unchanged
at 3.503901. It widens the margin from 0.052941 to 0.136537.

The removed continuation contains `Bildad`. On its `ad` token, neuron 1790
reaches 3.450959 and becomes the full sentence's runner-up. In PREFIX, where
that position does not exist, neuron 1790 peaks at only 2.795080 and neuron
1888 forms the competitor envelope instead.

```text
FULL -> PREFIX

906 peak change                 0.000000
competitor-envelope change     -0.083596
margin change                  +0.083596
runner                         1790 -> 1888
```

The envelope falls by only 0.083596 because neuron 1888 was already high at
3.367364. The removed runner 1790 itself falls much further, by 0.655879, but
another neuron takes its place. This is why runner identity should accompany
every margin comparison.

`DERIVED` — under a maximum-over-position readout, exact right-truncation after
the target peak is one-sided with respect to recorded positions. It retains
all earlier target and rival activations while removing later opportunities.
It can preserve or improve the target's margin; it cannot introduce a new
later rival position. This statement concerns exact token prefixes under the
declared readout, not arbitrary text editing and re-tokenisation.

---

## 6. The left-context effect

PREFIX and AUTO share the same right boundary at `shad`. Their comparison
therefore isolates the net effect of deleting the PREFIX's earlier left
context along this realised-string pair:

```text
PREFIX -> AUTO

906 peak                 3.503901 -> 3.095655    change -0.408246
1888 peak                3.367364 -> 2.933532    change -0.433832
target margin            0.136537 -> 0.162123    change +0.025586
runner                    1888 -> 1888
```

Neuron 906 loses substantial activation when the earlier sentence context is
removed. Neuron 1888 loses slightly more, so the target margin widens modestly.
`MEASURED`

This is a genuine context effect, but its contribution to selectivity is much
smaller than the raw drop in either neuron might suggest. The target and its
strongest rival move together to a considerable degree. The relative change
between them is only 0.025586. `DERIVED`

It would therefore be too broad to explain the manual/automatic contrast as
“left trimming clears competition.” Left trimming lowers the target and the
relevant competitor envelope here. The more dramatic narrowing of MANUAL is
located at its different right boundary, where the hyphen recruits neuron
1830.

---

## 7. Why the aggregate MANUAL-to-AUTO comparison needs care

The direct summary is:

```text
MANUAL -> AUTO

906 peak change                -0.111459
competitor-envelope change    -0.258352
margin change                 +0.146893
runner                         1830 -> 1888
```

Those numbers are correct, but the phrase “the competitor fell by 0.258352”
compresses a runner switch. Neuron 1830 falls from 3.191885 in MANUAL to
0.209164 in AUTO because AUTO contains no following hyphen. Meanwhile neuron
1888 rises from 2.707389 in MANUAL to 2.933532 in AUTO and becomes the new
runner.

The competitor envelope falls because the very strong 1830 event disappears,
not because every rival is suppressed. `MEASURED`

MANUAL and AUTO also change both the left and right boundaries at once. Their
0.146893 margin difference is consequently a useful outcome comparison, not a
clean estimate of one deletion effect. The token trace and the nested
PREFIX-to-AUTO comparison are what permit the more specific mechanism account.

---

## 8. The anchored deletion ladder

V8 tested every left trim of the 30-token peak-ending PREFIX. The results are
strongly non-monotonic. Removing more left context does not smoothly weaken
neuron 906, smoothly suppress its rivals, or smoothly improve the margin.

Selected ladder points are:

| Realised string | Tokens | 906 peak | Strongest rival | Signed margin | 906 wins? |
|---|---:|---:|---:|---:|:---:|
| `Rising ... in a broad shad` | 30 | 3.503901 | 1888: 3.367364 | +0.136537 | yes |
| `est drab, to a harpooneer in a broad shad` | 15 | 3.253084 | 1888: 2.748882 | +0.504202 | yes |
| ` to a harpooneer in a broad shad` | 11 | 3.130312 | 1790: 3.277112 | -0.146800 | no |
| ` harpooneer in a broad shad` | 9 | 3.114519 | 1888: 3.706467 | -0.591948 | no |
| `pooneer in a broad shad` | 8 | 3.126508 | 1888: 2.812624 | +0.313884 | yes |
| `er in a broad shad` | 6 | 3.095655 | 1888: 2.933532 | +0.162123 | yes |
| ` in a broad shad` | 5 | 3.077611 | 1888: 3.585052 | -0.507441 | no |
| `ad` | 1 | 0.443514 | 1790: 2.722723 | -2.279209 | no |

Several consequences follow.

First, `er in a broad shad` is the shortest winning member of this declared
anchored ladder. Deleting its first token makes neuron 1888 surge from 2.933532
to 3.585052 while neuron 906 changes only from 3.095655 to 3.077611. The `er`
token is therefore necessary for this particular six-token certificate under
the one-step left-deletion test. `MEASURED`

Second, `AUTO` is not the ladder's widest-margin route. The 15-token
`est drab, to a harpooneer in a broad shad` wins by 0.504202, more than three
times AUTO's margin. Other longer forms also win more clearly. Minimal length
and maximum margin are distinct objectives. `DERIVED`

Third, success is not monotone under left deletion. A losing 9-token route is
followed by winning 8-, 7-, and 6-token routes, then by losing 5-token and
shorter routes. The changing beginning of the realised string reshapes the
competition, often by changing which neuron is strongest. A certificate
cannot safely be described as a bag of individually supportive words.

Fourth, `ad` alone does not reproduce the 906 event. Its recorded 906 maximum
is the BOS footprint, 0.443514, while neuron 1790 reaches 2.722723. The target
peak occurring on `ad` in successful routes therefore does not make `ad` an
independent trigger. The completion token is a site at which preceding context
is integrated.

---

## 9. What is measured and what remains interpretive

V8 measures the following:

- neuron 906 peaks on the `ad` completing `shad` in all four variants;
- its peak magnitude changes with left context;
- the hyphen following `shad` produces a 3.191885 peak in neuron 1830 within
  MANUAL and narrows 906's margin to 0.015230;
- the later `ad` in `Bildad` produces a 3.450959 peak in neuron 1790 within
  FULL and narrows the full-sentence margin;
- PREFIX-to-AUTO left trimming lowers both 906 and runner 1888, with 1888
  falling 0.025586 further relative to the target;
- `er in a broad shad` is the shortest winner in the anchored deletion ladder,
  while several longer ladder members have wider margins;
- the ladder alternates between winning and losing spans rather than following
  a monotonic compression curve.

The following remain interpretations or open questions:

- whether neuron 906 is responding to a lexical fragment, compound-word
  structure, orthography, a contextual feature, or a superposition of features;
- what functional descriptions, if any, fit neurons 1790, 1830, and 1888;
- whether `er` supplies positive support to 906, suppresses 1888, changes a
  broader contextual state, or participates in several of these effects;
- whether the same right-boundary rivalry occurs in independent routes to 906;
- whether short, narrow-margin certificates are stable under other carriers,
  punctuation, continuations, tokenisations, or numerical perturbations.

The phrase “neuron 1830 is a hyphen neuron,” for example, would go beyond the
evidence. V8 shows one strong 1830 activation located on one hyphen in this
context. A feature claim would require varied positive examples and matched
controls.

---

## 10. Revised account of the v7 result — INTERPRETATION

A compact account supported by v8 is:

> Neuron 906 reaches its route-defining peak on the token `ad` completing
> `shad`. Its shortest discovered anchored certificate is
> `er in a broad shad`. The certificate's selectivity depends on both
> boundaries. Its left context is sufficient for 906 to outrank neuron 1888,
> while its right boundary excludes a following hyphen that would strongly
> activate neuron 1830. In the full sentence, later continuation supplies an
> additional near-rival when neuron 1790 peaks on `ad` in `Bildad`.

This description deliberately says “route-defining peak” rather than assigning
a semantic meaning to neuron 906. It also describes `er` as part of a measured
route without claiming that the suffix has a stable role across contexts.

The broader methodological lesson is that a certificate records a competitive
event, not merely a compact target stimulus. Its boundaries determine both the
target activation made available and the rival activations excluded or
admitted.

---

## 11. What v8 establishes

- `MEASURED` All four declared variants reproduce destination 906 on the
  pinned CUDA stack.
- `MEASURED` The 906 peak occurs on `ad` completing `shad` in every variant.
- `MEASURED` The manual route's narrow margin is created at the following
  hyphen, where neuron 1830 rises to 3.191885.
- `MEASURED` The full sentence's later `Bildad` supplies a 3.450959 rival peak
  in neuron 1790.
- `MEASURED` Removing left context from PREFIX to AUTO lowers both the target
  and runner 1888, widening the margin by 0.025586.
- `MEASURED` The automatic route is the shortest winner in the anchored ladder
  and loses after one further left-token deletion.
- `DERIVED` The automatic certificate's wider margin is principally explained
  by competitor opportunity and boundary placement, despite its lower target
  peak.
- `DERIVED` Minimum length, target strength, winning margin, human legibility,
  and likely robustness are different certificate objectives.

---

## 12. What v8 does not establish

- One route does not establish a general property of neuron 906 or of
  certificate minimisation.
- A peak token is not necessarily a self-sufficient trigger; `ad` alone is a
  direct counterexample here.
- A strong activation on a hyphen does not identify neuron 1830 as a general
  hyphen feature.
- The automatic certificate's positive margin of 0.162123 does not guarantee
  robustness under edits, new carriers, continuations, implementations, or
  floating-point regimes.
- The anchored ladder is not an arbitrary-string search and does not establish
  a global minimum over all strings.
- Destination preservation does not establish activation-mechanism identity.
- The probe observes activations and competitive outcomes; it does not perform
  causal neuron ablation or activation patching.

---

## 13. Consequences for the next probe — PROPOSAL

### 13.1 Record several certificate objectives

Future output should avoid treating one certificate as uniquely canonical.
For each route it can preserve:

```text
TOKEN_MIN       shortest successful model-native certificate
WORD_MIN        shortest successful certificate under declared word boundaries
MARGIN_MAX      widest-margin candidate within a declared search family
PEAK_KEEP       shortest candidate within a declared target-drift tolerance
```

These answer different questions. `TOKEN_MIN` is useful for model-native
minimality; `WORD_MIN` for human inspection; `MARGIN_MAX` for clearance; and
`PEAK_KEEP` for retaining something closer to the original activation event.

### 13.2 Add right-extension ladders

V8 exposes two continuation hazards: `-` recruits neuron 1830 and later
`Bildad` recruits neuron 1790. A generalised probe should begin at each
peak-ending certificate and append source tokens one at a time, recording:

```text
target running peak
competitor envelope
runner identity
first token that narrows the margin
first token that changes the destination, if any
```

This would turn “continuation robustness” into a position-resolved measurement
rather than only a final retention rate.

### 13.3 Test the `er` boundary with controlled variants

The one-step ladder shows that removing `er` makes neuron 1888 win, but it does
not identify why. Useful controlled comparisons include alternative word-final
fragments, complete-word boundaries, spaces, punctuation, and matched tokens
with similar frequency or shape. The outcome should track target and named
rivals separately rather than only final destination.

### 13.4 Replicate before naming a 906 feature

Independent routes to neuron 906 should be collected before proposing a
semantic or orthographic description. If several routes peak at `ad`, `shad`,
compound boundaries, or comparable fragments, those recurrences can motivate
matched positive and negative controls. If they do not, this result may be a
route-specific collision rather than a stable feature family.

### 13.5 Treat margin as a profile, not a single quality score

The ladder's maximum observed margin belongs to a longer certificate, while
the shortest certificate remains locally fragile. Future selection can retain
a Pareto set: candidates for which no other tested string is both shorter and
wider-margin. This preserves genuine trade-offs instead of choosing one scalar
objective prematurely.

That last change would give the probe a more faithful output than a single
“best certificate.” For the present route, v8 already shows that the geometry
of shortening contains several useful solutions rather than one natural end
point.
