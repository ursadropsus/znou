# STRING ITERATION PROBE v15 — RESULTS

## L5-N38 across three Melville Atlas routes

Status: preliminary observational result note. Tags follow the project
convention: `MEASURED`, `DERIVED`, `INTERPRETATION`, and `PROPOSAL`.

Probe v15 examined three full Melville strings that had been independently
recorded as routes to GPT-2 Small layer-5 neuron 38:

```text
It was cold as Iceland—no fire at all—the landlord said he couldn’t afford it.

We all heard a faint creaking, as of ropes and yards hitherto muffled by the storm.

The grey dawn came on, and the slumbering crew arose from the boat’s bottom,
and ere noon the dead whale was brought to the ship.
```

V13–v14 had already associated N38 with predicative-property contexts and a
causal cold-relative tilt. The new strings raised a broader question. Did the
two non-temperature routes reach N38 through language later in their sentences,
such as `muffled by the storm` and `was brought to the ship`, or through a
shared feature elsewhere?

V15 traced N38 at every token and exhaustively evaluated every unique decoded
contiguous source-token span. The decisive peaks occurred much earlier than
the visually suggestive later phrases:

| Route | Full-route N38 peak | Shortest destination-38 certificate | Peak retained |
|---|---:|---|---:|
| Iceland | 3.433706 on ` cold` | `It was cold` | 100.0% |
| Creaking | 3.257905 on `aking` | ` heard a faint creaking` | 98.6% |
| Grey dawn | 3.623844 on ` came` | `The grey dawn came` | 100.0% |

Everything following these certificates can be removed without changing the
essential route result. The later participial language is therefore not the
source of the observed N38 peaks. `MEASURED`

The three certificates broaden the observational phenotype beyond explicit
temperature. They are compact clause-like units which establish a state,
perceived occurrence, or environmental event. In combination with v13–v14,
the current cautious description is:

> **L5-N38 responds strongly in compact predications that establish a
> perceptible property, state, or environmental occurrence. Within that
> broader response, N38 has a demonstrated context-conditioned causal tilt
> toward `cold` over matched alternatives.**

This remains a provisional family resemblance, not a claim that N38 encodes a
single formal semantic variable. Three positive Melville routes cannot by
themselves distinguish event introduction, perceptual salience, clause shape,
lexical combinations, corpus concentration, or a mixture of these factors.

---

## 1. Run provenance — MEASURED

```text
script          probe_v15.py
generated_utc   2026-08-15T02:06:36+00:00
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
probe_v15_routes_20260815-120636.tsv
probe_v15_trace_20260815-120636.tsv
probe_v15_certificates_20260815-120636.tsv
probe_v15_spans_20260815-120636.tsv
probe_v15_variants_20260815-120636.tsv
```

| Output | Rows | Unit |
|---|---:|---|
| Routes | 3 | one supplied full string |
| Trace | 79 | one BOS/token position |
| Certificates | 15 | five certificate criteria per route |
| Spans | 1,049 | one unique realised contiguous source-token span |
| Variants | 7 | original or typography-normalised full string |

The first Iceland-only execution reported a validation failure because the
same activation was compared across single-string and padded-batch execution
with an unnecessarily exact floating-point equality test. The observed values
agreed at the scale relevant to the experiment. V15 was repaired to permit a
`5e-5` target-peak tolerance while continuing to require exact destination and
peak-position agreement. The completed three-route run passed validation.
`MEASURED`

All TSVs use literal quotation marks rather than CSV quoting. Generic readers
should use tab separation with `quoting=csv.QUOTE_NONE` or an equivalent
literal-TSV mode.

---

## 2. Terms

**N38 target activation** is the scalar post-GELU activation of layer-5 neuron
38 at one token position.

**Position winner** is the neuron with the largest activation at one position.
V15's trace indices include the beginning-of-string token as position zero, so
the displayed peak positions are one greater than zero-based source-token
indices.

**Atlas destination** compares each layer-5 neuron's largest activation
anywhere in the complete string. The destination is the neuron with the largest
of those 3,072 per-neuron maxima. This is a whole-route argmax. It is possible
for N38 to win locally at a token while another neuron wins the complete route.

**Atlas certificate** is a shorter string which preserves the specified Atlas
destination. A destination-38 certificate therefore remains a complete route
to N38 under the same whole-string rule. It need not preserve every activation
value or every competing neuron's behaviour.

**Contiguous source-token span** is a consecutive interval of the original
token sequence. V15 decodes each interval and retokenizes it before scoring.
The intended and realised token IDs are recorded so that boundary-sensitive
tokenization is visible rather than assumed away.

**Anchored certificate** is the shortest destination-38 span ending at the
source token where the full route reaches its N38 peak.

**Local winner** means N38 is the strongest neuron at its own highest-activation
position in the tested span. This is weaker than retaining destination 38.
V15 treats the shortest local-winner span as a diagnostic near-certificate,
not as an Atlas certificate.

**Peak retention** is the span's N38 peak divided by the corresponding
full-route N38 peak. The `PEAK_90` diagnostic asks for the shortest span that
retains at least 90% of the full value; it does not by itself require
destination 38.

---

## 3. Design

For each supplied route, v15 performed five related analyses.

First, it scored the full string and recorded the Atlas destination, runner-up,
destination margin, N38 peak, peak position, and local rank. Second, it traced
N38 across BOS and every source token. Third, it scored every unique decoded
contiguous source-token span. Fourth, it selected five certificate or
diagnostic records:

```text
PREFIX       shortest right-truncated prefix ending at the full N38 peak
ANCHORED     shortest destination-38 span ending at that source token
CONTIG       shortest destination-38 span anywhere in the route
LOCAL_WINNER shortest span in which N38 wins locally at its own peak
PEAK_90      shortest span retaining at least 90% of the full N38 peak
```

Finally, typography controls replaced curly apostrophes and/or em dashes with
ASCII forms. These controls were run on complete strings; they test route-level
robustness, not only the certificate fragment.

The exhaustive search covered 251 unique realised spans for Iceland, 210 for
Creaking, and 588 for Grey dawn. `MEASURED`

---

## 4. Full-route results — MEASURED

All three supplied strings retained N38 as their Atlas destination and made
N38 the local winner at its own peak.

| Route | Tokens | Destination runner-up | Destination margin | N38 peak | Peak token | Local margin |
|---|---:|---:|---:|---:|---|---:|
| Iceland | 22 | 8 | 0.136566 | 3.433706 | ` cold` | 1.253900 |
| Creaking | 20 | 1790 | 0.270219 | 3.257905 | `aking` | 1.058957 |
| Grey dawn | 34 | 2874 | 0.181501 | 3.623844 | ` came` | 1.850253 |

The destination margin and local margin answer different questions. The local
margin compares N38 with other neurons at the N38 peak token. The destination
margin compares N38's route maximum with every competing neuron's maximum at
whatever token each competitor prefers.

### 4.1 Iceland trace

```text
It       0.778225   local rank 17
 was     2.023738   local rank 2
 cold    3.433706   local rank 1   <- full peak
 as      1.615323   local rank 5
 Iceland -0.077709  local rank 1413
```

N38 rises compositionally across `It` → `It was` → `It was cold`. Isolated
` cold` does not reproduce the response: as a one-token span its N38 peak is
only 0.116321 and its Atlas destination is 1888. This agrees with v13's finding
that the response is not a context-free lexical detector for the word `cold`.

### 4.2 Creaking trace

```text
 a       1.347250   local rank 6
 faint   2.054071   local rank 1
 cre     1.003540   local rank 13
aking    3.257905   local rank 1   <- full peak
,        1.968992   local rank 1
 as      0.856619   local rank 22
 of      0.220616   local rank 236
 ropes  -0.092534   local rank 1473
```

The maximum arrives when the split token sequence ` cre` + `aking` completes
the word `creaking`. N38 does not peak on `muffled`, `storm`, or a passive
construction later in the sentence. The activation then decays as the sentence
continues.

### 4.3 Grey-dawn trace

```text
The     -0.003507   local rank 334
 grey    0.058169   local rank 270
 dawn    2.142298   local rank 1
 came    3.623844   local rank 1   <- full peak
 on      3.453443   local rank 1
,        2.524120   local rank 1
 and     1.896050   local rank 1
 the     1.579434   local rank 3
```

The decisive rise occurs across `dawn came`, near the beginning of the route.
The later `was brought` clause occurs after the route maximum and does not
explain why this string reaches N38.

---

## 5. Exhaustive minimisation — MEASURED and DERIVED

### 5.1 Iceland

All five criteria select the same three-token string:

```text
It was cold
```

It retains N38 as destination with a peak of 3.433705–3.433706, effectively
100% of the full route, and produces a destination margin of 0.301422. All 20
destination-38 spans begin at source token zero and extend from `It was cold`
to progressively longer right extensions. No span beginning at `was`, `cold`,
or later retains destination 38. `MEASURED`

This confirms that `It was cold` is an irreducible three-source-token Atlas
certificate within the supplied route. Here *irreducible* means only that no
shorter contiguous source-token span satisfies the declared destination rule.
It does not mean the phrase is a minimal semantic representation of cold, or
that arbitrary token editing could not produce another shorter route.
`DERIVED`

### 5.2 Creaking

The shortest destination-38 result is:

```text
 heard a faint creaking
```

It is a five-source-token span, reaches 3.212888 or 98.6% of the full-route
N38 peak, and gives N38 a destination margin of 0.282958. The complete prefix
`We all heard a faint creaking` exactly retains the full peak, while adding
`all` and dropping only `We` raises it slightly to 3.261597.

Every one of the 42 destination-38 spans begins at `We`, `all`, or `heard` and
extends through the completed word `creaking`. No later phrase independently
routes to N38. `MEASURED`

The shortest local-winner diagnostics are smaller:

```text
 heard a faint       N38 peak 2.188485   destination 1790
 faint creaking      N38 peak 2.318588   destination 1888
```

These fragments show two partial ways of building the response, around faint
perception and the sound/event noun. N38 wins at its preferred token in each,
yet another neuron achieves the greater whole-span maximum. They are therefore
near-certificates rather than destination-38 certificates. `DERIVED`

### 5.3 Grey dawn

The shortest destination-38 result is:

```text
The grey dawn came
```

It is a four-source-token certificate and reproduces the full peak to recorded
precision. All 31 destination-38 spans begin at source token zero and contain
this entire phrase; they differ only in how far they extend to the right.
`MEASURED`

The shortest local-winner span is:

```text
 dawn came
```

This reaches 2.869829, or 79.2% of the full peak, while retaining Atlas
destination 1888. `The grey dawn` also makes N38 the local winner with a lower
peak of 2.142298, but does not win the whole route. The determiner and adjective
provide relatively little activation by themselves; the combined subject and
event predicate raise N38 sufficiently to become the Atlas destination.
`DERIVED`

---

## 6. What the span geometry rules out

V15 was motivated partly by semantic impressions of the complete sentences.
The exhaustive results show why reading a long Atlas route by eye is hazardous.
A route contains every word after its decisive peak, and those later words may
be evocative without contributing to the destination.

For Creaking, every destination-38 span must contain the early `heard ...
creaking` material, while none begins near `muffled by the storm`. For Grey
dawn, every destination-38 span must begin with `The` and contain `grey dawn
came`; none begins near `dead whale was brought`. `MEASURED`

Accordingly, v15 does not support a passive-participle account based on
`muffled` and `brought`, nor a general claim about suppression, dormancy, or
being acted upon. Those interpretations arose from salient language outside
the causal location of the observed peak. `DERIVED`

The result also clarifies the meaning of an Atlas hit. A full route tells us
that some part of the string produced the largest layer-5 activation under the
Atlas rule. It does not tell us which visible phrase did so. Token traces and
minimisation are required before interpreting the route semantically.

---

## 7. Typography controls — MEASURED

Replacing the Iceland route's curly apostrophe with an ASCII apostrophe reduced
its token count from 22 to 20 but left the N38 peak and peak position unchanged.
It retained destination 38.

Replacing the em dashes with spaced double hyphens also left N38's peak exactly
unchanged at 3.433704 on `cold`, but changed the full-route destination from 38
to 8. Replacing both the apostrophe and dashes together retained destination
38. The Grey-dawn apostrophe replacement also left the N38 result unchanged.

The isolated dash result should not be described as N38 certificate fragility.
The N38 peak itself is stable because it occurs before either dash. Instead,
later typography changes the maxima reached by competing neurons elsewhere in
the full string; in one variant, neuron 8 narrowly overtakes N38 by 0.077770.
This is whole-route argmax sensitivity. It illustrates how an unchanged target
feature can lose the Atlas destination when an unrelated competitor changes.
`DERIVED`

The fact that the combined ASCII replacement returns destination 38 also warns
against treating typography effects as independent additive factors. Tokenized
context downstream can alter several competing route maxima jointly.

---

## 8. Phenotype update — INTERPRETATION

The three exact destination-preserving certificates are:

```text
It was cold
heard a faint creaking
The grey dawn came
```

Their most defensible shared description is presently broader than
temperature. Each forms a compact predication or predicate-bearing fragment
which establishes something perceptible about a scene: a state, an apprehended
sound, or an environmental transition. The strongest activation arrives when
the relevant property or event becomes specified: `cold`, the completion of
`creaking`, and `came`.

That description fits the data, but several narrower explanations remain live:

* N38 may respond to a family of event/state-establishing clause shapes.
* It may favour perceptually or atmospherically salient predicates.
* The three lexical combinations may be separately learned routes with only a
  loose semantic resemblance.
* Melville-conditioned sampling may concentrate a broader and more diverse
  underlying response into this apparent family.

V15 contains three positive examples and no matched negative panel, so it
cannot separate these possibilities. It also applies no causal interventions.
The causal cold-relative result still comes from v13–v14; v15 should not be
used to transfer that causal claim to `creaking`, `dawn`, perception, or event
introduction.

The useful synthesis is therefore asymmetric. The observational family has
broadened, while the established causal tilt remains specifically cold-related
within the contexts tested so far.

---

## 9. Conclusions

1. All three exact supplied Melville strings are reproducible destination-38
   routes under the pinned v15 environment. `MEASURED`
2. Their decisive N38 peaks occur on `cold`, completion of `creaking`, and
   `came`, not on the later phrases that initially drew attention. `MEASURED`
3. Exhaustive contiguous minimisation yields destination-preserving
   certificates of three, five, and four source tokens. `MEASURED`
4. Local-winner near-certificates reveal graded partial structure, but local
   N38 dominance is insufficient to establish Atlas destination 38. `DERIVED`
5. The results broaden the observational phenotype from predicative properties
   toward compact state/event-establishing language, while leaving the exact
   abstraction unresolved. `INTERPRETATION`
6. V13–v14's causal cold tilt survives as a distinct and stronger claim; v15
   supplies no causal evidence for the broader phenotype. `INTERPRETATION`
7. Full-route typography can change the Atlas destination without changing
   N38's peak, because the destination is a competition among maxima reached
   anywhere in the string. `DERIVED`

---

## 10. Focused next test — PROPOSAL

The most informative follow-up would be a small matched factorial panel built
around the three certificate shapes rather than another collection of long
positive routes. For example, preserve and perturb subject frame, predicate
class, perceptual framing, and event/state content independently:

```text
It was cold              It was ordinary
It was dark              It was considered
We heard a faint creak   We mentioned a faint creak
We heard a loud crash    We carried a wooden box
The grey dawn came       The grey package came
The warm evening came    The committee report came
```

These examples are design sketches, not a preregistered panel. Tokenization,
baseline activation, lexical frequency, and full-route competitors would need
to be balanced before inference. The goal would be to learn which alterations
move N38 while holding the surrounding clause structure as constant as
practical.

A second useful step would intervene on naturally active N38 at the `creaking`
and `came` peaks and measure declared downstream outcomes, using the v12–v14
integrity controls. That would test whether the wider observational family has
any shared causal consequence. Until then, the broadened phenotype should
remain explicitly observational.
