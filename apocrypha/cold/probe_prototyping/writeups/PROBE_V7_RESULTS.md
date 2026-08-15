# STRING ITERATION PROBE v7 — RESULTS

Status: preliminary result note. Tags follow `SPEC.md`: `MEASURED`, `DERIVED`,
`INTERPRETATION`, `PROPOSAL`.

Probe v7 tested whether full-sentence Atlas discoveries could be reduced to
short contiguous certificates for the same destination. It verified 41
Melville routes across five neurons, added one acceptance fixture for neuron
906, and exhaustively evaluated every contiguous token span of each source
sentence.

The run succeeded technically and produced two substantive findings. First,
the shortest same-neuron certificate was attached to the original winning-peak
region in all 42 cases. Second, independent sentences reaching the same neuron
often collapsed onto visibly related short route families.

The run also separated token-minimality from human-legible minimality. The
automatic search can begin or end inside a written word because GPT-2 tokens
do not respect word boundaries. These model-native certificates are valid
under `D`, but they are a different product from the word-bounded “atomic
strings” previously found by manual whittling.

---

## 1. Run provenance — MEASURED

```
script          probe_v7.py
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

```
candidates for probe v7.txt
```

Outputs:

```
results/probe_v7_minimise_20260815-023610.tsv
results/probe_v7_certificates_20260815-023611.tsv
```

The minimisation table contains 42 rows: 41 Atlas routes and the neuron-906
acceptance fixture. The certificate table contains 127 rows: three automatic
certificate methods for all 42 routes, plus the manual 906 certificate.

All rows report the same pinned model hash and actual CUDA device. `MEASURED`

---

## 2. Terms

### 2.1 Source token sequence

The GPT-2 tokenisation of the complete supplied sentence. A contiguous source
span is a run of adjacent tokens from this sequence. It does not reorder tokens
or omit a token from the middle of the selected run.

### 2.2 Realised string and realised tokens

To test a source span in the domain of `D`, v7 decodes the selected tokens into
a string and tokenises that string again. The resulting sequence is the
**realised token sequence**. All certificates are evaluated as realised
strings, not assumed to preserve the token IDs sliced from the source.

### 2.3 Prefix certificate

The source sentence cut immediately after the position `t*` where its winning
neuron reaches its peak. Under Resonance, this right-truncation is free: the
winning peak is retained and later rivals are removed. The decoded prefix must
still be re-evaluated because string reconstruction can alter tokenisation.

### 2.4 Peak-anchored certificate

A successful contiguous source span whose right boundary is the original
winning-peak position. V7 evaluates every possible left boundary ending there
and selects the shortest realised same-neuron string.

“Anchored” describes its location in the source sentence. It does not mean the
activation value remains numerically identical after the left context is
removed.

### 2.5 Contiguous certificate

A decoded contiguous source span that reaches the source sentence's labelled
neuron. V7 exhaustively evaluates spans throughout the sentence and selects
the successful candidate with the fewest realised tokens, then the fewest
characters, with deterministic tie-breaking.

This is a global minimum only over decoded contiguous spans of the supplied
source token sequence. It is not a globally minimal string. It does not search
arbitrary internal deletions, reordered text, substitutions, or strings absent
from the source.

### 2.6 Token-minimal and word-bounded

A **token-minimal certificate** is shortest under the contiguous-token search
above. Its boundary may fall inside an ordinary written word.

A **word-bounded certificate** would require boundaries that preserve complete
source words or another declared human-readable unit. V7 did not perform that
search. It is proposed as a separate output rather than a replacement for the
token-minimal result.

### 2.7 Target peak and target-peak drift

The **target peak** is the maximum activation of the labelled neuron across the
evaluated string's positions.

```
target-peak drift = certificate target peak − full-sentence target peak
```

Zero drift means the recorded maximum is numerically preserved to the stored
precision. Negative drift means the target activates less strongly in the
certificate; it can still win if rival peaks fall further. Positive drift means
the shortened context strengthens the target peak.

Target-peak drift is distinct from Δ. Δ is the target's winning clearance over
the strongest rival in that evaluated string. A certificate can have a lower
target peak and a wider Δ.

### 2.8 Destination preservation and activation preservation

**Destination preservation** means the certificate has the same argmax neuron
as the source sentence. **Activation preservation** would require a stronger
similarity criterion, such as the same target peak, peak position, rival
structure, or activation fingerprint. V7 establishes destination preservation
and reports peak drift; it does not assert full activation preservation.

### 2.9 Route-family concentration

Repeated recovery of identical or closely related certificates from distinct
full sentences reaching the same neuron. It is evidence that the corpus hits
share local route material. It is not, by itself, evidence that the neuron has
a complete semantic interpretation matching that material.

---

## 3. Input sample

The Atlas export contained 41 routes from *Moby-Dick*:

| Neuron | Full-sentence routes |
|---:|---:|
| 508 | 17 |
| 666 | 4 |
| 1079 | 1 |
| 1055 | 2 |
| 870 | 17 |

Neuron 906 was supplied separately as an acceptance fixture with one full
sentence and one manually whittled certificate.

This is a Melville-conditioned sample selected from the player's Atlas. It is
well suited to testing within-neuron route recurrence. It is not a random or
representative sample of ℛ, natural language, or all corpus discoveries.

---

## 4. Validation — MEASURED

All 42 full sentences reached their declared neuron on the v7 GPU stack:

```
verified labels             42/42
successful minimisations    42/42
automatic certificates     126/126
round-trip-stable rows      127/127
906 fixture                 PASS
```

No Atlas route was silently normalised into a different destination, and no
certificate method failed to find a same-neuron span.

The TSVs use the existing probe recorder convention: tab separation with no
syntactic meaning assigned to quotation marks. Generic CSV readers should use
`quoting=csv.QUOTE_NONE` or an equivalent literal-TSV mode.

---

## 5. Compression result

### 5.1 Atlas routes — MEASURED, DERIVED

Across the 41 Atlas routes:

| Representation | Minimum tokens | Median | Mean | Maximum |
|---|---:|---:|---:|---:|
| Full sentence | 10 | 37 | 47.78 | 209 |
| Free prefix | 2 | 11 | 22.73 | 161 |
| Peak-anchored certificate | 2 | 2 | 2.90 | 12 |
| Global contiguous certificate | 2 | 2 | 2.90 | 12 |

The median contiguous certificate retained 7.14% of its source sentence's
tokens. The mean retained fraction was 10.21%, with a measured range from
1.52% to 57.14%.

The 209-token source that dominated runtime reduced to:

```text
, if I may → 870
```

four realised tokens, or 1.91% of the source length.

### 5.2 Anchored and global searches agree — MEASURED

The peak-anchored and global contiguous searches returned the same certificate
for every route:

```
Atlas routes        41/41
906 fixture          1/1
total               42/42
```

Within this sample, the shortest same-neuron span was always attached to the
region containing the full sentence's original target peak. The exhaustive
search did not discover a shorter same-neuron collision elsewhere in any
sentence.

`INTERPRETATION` — this validates peak-anchored left-whittling as an efficient
candidate strategy for these routes. It does not prove that all routes or
neurons have this property. The exhaustive search remains the control that
made the 42/42 observation visible.

### 5.3 Peak drift — MEASURED

Across the 41 Atlas anchored certificates:

```
minimum drift       −1.269739
median drift         0.000000
maximum drift       +0.442884
median |drift|       0.145343
```

Some certificates preserve the recorded target maximum essentially exactly;
others reach the same destination after substantial activation change. A
same-neuron certificate should therefore not automatically be described as
the same activation event.

---

## 6. Neuron-906 acceptance fixture

The full sentence contained 100 tokens and reached neuron 906:

```text
Rising from a little cabin-boy in short clothes of the drabbest drab, to a
harpooneer in a broad shad-bellied waistcoat; from that becoming boat-header,
chief-mate, and captain, and finally a ship owner; Bildad, as I hinted before,
had concluded his adventurous career by wholly retiring from active life at
the goodly age of sixty, and dedicating his remaining days to the quiet
receiving of his well-earned income.
```

The reductions were:

| Method | Certificate | Tokens | Δ | Target peak | Drift |
|---|---|---:|---:|---:|---:|
| Full sentence | full sentence above | 100 | 0.052941 | 3.503901 | — |
| Free prefix | `Rising … in a broad shad` | 30 | 0.136537 | 3.503901 | 0.000000 |
| Manual | `the drabbest drab, to a harpooneer in a broad shad-bellied` | 21 | 0.015230 | 3.207114 | −0.296787 |
| Peak-anchored | `er in a broad shad` | 6 | 0.162126 | 3.095656 | −0.408245 |
| Global contiguous | `er in a broad shad` | 6 | 0.162123 | 3.095655 | −0.408246 |

The automatic certificate is shorter and has a wider winning margin than the
manual certificate, despite having a lower target peak. Rival activity fell
far enough for neuron 906 to win more clearly.

The result passed the formal v7 acceptance test. It also exposed that the test
captured token-minimality rather than the user's intended human-legible
minimality. `er` is the end of `harpooneer`; `shad` is the beginning of
`shad-bellied`. The certificate follows GPT-2 token boundaries through written
word boundaries.

`INTERPRETATION` — the manual and automatic strings answer different useful
questions:

```
manual certificate      short word-bounded route found by a human
automatic certificate   shortest model-native contiguous token route
```

Neither should displace the other in the record.

---

## 7. Within-neuron route-family concentration

### 7.1 Neuron 508 — MEASURED

Seventeen full sentences reduced to 14 unique contiguous certificates. Three
certificates recurred twice:

```
Inn       ×2
 in due   ×2
In no     ×2
```

Most remaining cores also contain a form of `in`:

```text
 In as
 are in the
First: In the
But in all
in the
in all
in this
, in the
 all in the
In due
```

The two-token `Inn` result is tokenised as `["In", "n"]`; it belongs to this
orthographic/token family despite being a different written word.

One route required a substantially longer 12-token certificate:

```text
 but I was born there.” “In the
```

`INTERPRETATION` — neuron 508's 17 Melville hits show strong concentration
around `in`-bearing local sequences. This establishes recurrent route material
within the sample. It does not establish that the neuron's function is the
English preposition *in*.

### 7.2 Neuron 870 — MEASURED

Seventeen full sentences reduced to ten unique certificates. The strongest
exact recurrences were:

```text
 at all       ×6
 after all    ×2
, as well     ×2
```

Other recovered cores included:

```text
At all
, if you will
 if possible
, if I may
 if practicable
```

Two longer outliers were:

```text
; they are Quakers with a vengeance
 were simultaneously quaffed down with a hiss
```

`INTERPRETATION` — neuron 870 shows stronger exact route recurrence than 508,
with repeated `at all` and related discourse or conditional sequences. The
source sentences were selected from an Atlas already rich in `at all`; corpus
and selection effects are part of the result, not controlled away.

### 7.3 Neuron 666 — MEASURED

Four sentences reduced to three unique cores:

```text
 seer       ×2
, reader
arpenter
```

`arpenter` is extracted from `Carpenter` and begins inside the written word.
It is a valid token-bounded certificate and another direct example of why a
word-bounded output is needed for human-facing use.

Four routes are insufficient to determine whether these strings form one
stable route family or several unrelated ways into neuron 666.

### 7.4 Neurons 1079 and 1055 — MEASURED

Neuron 1079 had one route:

```text
 accounts Tarsh
```

Neuron 1055 had two:

```text
 Van R
De bal
```

These are measurements of individual routes. They provide no within-neuron
replication.

---

## 8. What v7 establishes

- `MEASURED` Every supplied full sentence reproduced its Atlas destination on
  the pinned GPU stack.
- `MEASURED` Every route admitted a much shorter contiguous certificate.
- `MEASURED` Peak-anchored and global contiguous minima agreed in 42/42 cases.
- `MEASURED` Target-peak drift varied substantially despite destination
  preservation.
- `MEASURED` Neurons 508 and 870 show repeated short route material across
  distinct Melville sentences.
- `MEASURED` The automatic 906 certificate is shorter than the manual
  word-bounded certificate but crosses written word boundaries.
- `DERIVED` Token-minimal and human-legible certificates are distinct products
  and should be recorded separately.

---

## 9. What v7 does not establish

- A contiguous certificate is not a globally minimal string in Σ*.
- Same destination does not entail identical activation mechanism.
- Peak anchoring does not imply zero target-peak drift.
- Repeated strings in a Melville-conditioned Atlas do not establish a
  map-wide or corpus-independent neuron interpretation.
- The sample contains only five Atlas neurons, two of them with substantial
  within-neuron replication.
- V7 does not yet measure word-bounded minima.
- V7 does not yet characterise the recovered routes under carrier,
  continuation, or local-edit sweeps.

---

## 10. Method and tooling consequences — PROPOSAL

### 10.1 Preserve several certificate classes

Future output should retain at least:

```text
PREFIX       free right-truncated certificate
TOKEN_MIN    globally shortest contiguous token-derived certificate
WORD_MIN     shortest same-neuron certificate with declared word boundaries
```

The existing `ANCHORED` result remains useful as a cheaper search and a test of
whether `TOKEN_MIN` stays attached to the original peak region.

### 10.2 Classify peak preservation

Add an explicit peak-drift classification rather than treating every
same-neuron certificate alike. Thresholds should be declared only after
examining the observed distribution; provisional labels such as exact, small,
and large drift should not acquire numerical definitions post hoc.

### 10.3 Checkpoint and resume

Exhaustive contiguous search is quadratic in source length:

```
n source tokens → n(n+1)/2 contiguous spans
```

The longest v7 sentence contained 209 tokens and required 21,945 span
evaluations. The run completed safely, but results were held until every route
finished. Future versions should checkpoint after each source sentence and
resume without repeating completed work.

### 10.4 Use anchored search as a screened path

Because anchored and global results agreed in 42/42 cases, a larger survey
could run anchored minimisation on every route and reserve exhaustive global
search for a declared validation sample. That would reduce computation without
silently converting the present observation into an assumption.

### 10.5 Characterise a balanced set of recovered routes

The next phenotype comparison should select multiple certificates per neuron,
including repeated and outlying route families. For example, neuron 870 should
include both an `at all` route and a longer outlier; neuron 508 should include
recurrent `in` routes and its 12-token exception. This prevents the most common
certificate from standing in for the whole neuron.
