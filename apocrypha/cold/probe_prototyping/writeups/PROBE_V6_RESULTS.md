# STRING ITERATION PROBE v6 — RESULTS

Status: preliminary result note. Tags follow `SPEC.md`: `MEASURED`, `DERIVED`,
`INTERPRETATION`, `PROPOSAL`.

Probe v6 tested whether three known imp_r routes at GPT-2 Small layer 5 have
different local neighbourhoods under carrier changes, gate substitutions,
one-token substitutions, random strings, and continuations. The result is a
preliminary set of route phenotypes: the three routes differ sharply in
carrier robustness and local-edit robustness while all remain comparatively
robust to continuation.

These are properties of measured routes under `D`, not semantic neuron types
or claims about the neurons' complete functional roles.

---

## 1. Run provenance — MEASURED

Run completed 2026-08-15 local time / 2026-08-14 UTC using:

```
script          probe_v6.py
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

Command:

```bash
python probe_v6.py --budget 60 --rng-seeds 0,1,2
```

Result files:

```
results/probe_v6_yield_20260815-013458.tsv
results/probe_v6_summary_20260815-013459.tsv
results/probe_v6_curve_20260815-013459.tsv
results/probe_v6_overlap_20260815-013459.tsv
```

The yield table contains 1,623 rows: three seed rows and 1,620 evaluated
candidates. The curve table contains one cumulative row per candidate. The
summary contains 27 seed/branch blocks, and the overlap table contains 108
pairwise block comparisons. Counts reconcile across all four outputs.
`MEASURED`

This is the first probe run in this series whose recorded CUDA device matches
the model's actual execution device. The earlier v5 support module reported
CUDA availability without moving the model or inputs from CPU. V6 should
therefore be treated as a GPU-stack run rather than an exact device replay of
v5.

---

## 2. Design

### 2.1 Terms

The probe uses positional terms rather than treating each variant as a generic
language edit:

**Candidate gate** — the token span provisionally treated as load-bearing for
a route. A gate is found through intervention: removing or replacing it loses
the destination. It is operational shorthand, not a claim that the neuron
detects the gate's word or meaning.

**Carrier** — material before the candidate gate. A carrier sweep changes that
preceding material while holding the candidate gate text fixed.

**Continuation** — material placed after the candidate gate or gate stem. A
continuation sweep preserves the earlier route candidate and varies what
follows it.

**Local edit** — in v6, exactly one realised-token substitution anywhere in
the seed. Candidates that changed into multi-token edits during decoding and
re-encoding were rejected and resampled.

**Retention** — the proportion of tested variants that still reach the seed's
destination. “Carrier-robust,” for example, means high measured retention in
the declared carrier sweep, not invariance to every possible carrier.

**Exit diversity** — how widely variants that leave the seed destination
disperse across other destinations. Low retention and high exit diversity are
different properties: a route can lose almost every edit while those edits
all collapse onto the same competing destination.

### 2.2 Worked example

For the cold seed, the provisional decomposition is:

```
seed               it was cold  → 38
carrier             it was
candidate gate       cold
continuation         est
```

The branches ask different questions:

```
carrier change       inside it was cold  → 38     retained
continuation         it was coldest      → 38     retained
local substitution   it was MU           → 1888   lost
```

In the local example, the token ` cold` was replaced by the single token ` MU`;
the realised string remained a one-token edit after round-trip checking.

`it was colder` is a continuation of the string `it was cold`, whereas
replacing ` cold` with another token is a substitution. Under ρ=R those
operations have different causal structure: an extension preserves all prior
positions and may retain their peak; a substitution changes the activations
from the edited position onward.

### 2.3 Seed routes

```
at Mackinaw   → 20
at robin      → 281
it was cold   → 38
```

The seeds were selected because prior probing suggested contrasting route
structure. Mackinaw appeared carrier-robust and dependent on the ` Mack` +
`in` sequence; robin appeared to use a short ` rob` route; cold supplied the
known `cold` / `colder` / `coldest` continuation family.

### 2.4 Branches

Each seed was evaluated under five branches:

```
A_CARRIER       vary material before the candidate gate
B_GATE          hold the carrier template; substitute the following word
C_LOCAL         perform one realised-token substitution
D_RANDOM_TOKEN  draw a random, realised-length-matched safe-token string
E_CONTINUATION  preserve the candidate gate stem; vary what follows
```

Deterministic branches A, B, and E evaluated 60 unique realised strings per
seed. Stochastic branches C and D evaluated 60 strings under each of three RNG
seeds, for 180 evaluations per seed and branch.

V6 separates round-trip stability from intervention agreement. Every evaluated
candidate was round-trip stable and every intended intervention agreed with
the realised token sequence: 0/1,620 failures on either check. `MEASURED`

---

## 3. Primary result: three preliminary route phenotypes

### 3.1 Retention by positional operation — MEASURED

| Seed route | Carrier retention | Continuation retention | Local-edit retention |
|---|---:|---:|---:|
| `at Mackinaw → 20` | 49/60 (81.7%) | 51/60 (85.0%) | 64/180 (35.6%) |
| `at robin → 281` | 3/60 (5.0%) | 56/60 (93.3%) | 57/180 (31.7%) |
| `it was cold → 38` | 2/60 (3.3%) | 54/60 (90.0%) | 0/180 (0.0%) |

The operations are not interchangeable. Mackinaw survives changes before and
after its candidate gate. Robin is highly sensitive to carrier changes while
remaining robust to continuation. Cold is carrier-fragile and completely lost
under the sampled one-token substitutions, yet survives most extensions.

`INTERPRETATION` — compact operational descriptions of these measured routes
are:

```
Mackinaw   high carrier retention · high continuation retention
           moderate local-edit retention · high exit diversity

Robin      low carrier retention · high continuation retention
           moderate local-edit retention · high exit diversity

Cold       low carrier retention · high continuation retention
           zero local-edit retention · low exit diversity, mostly 1888
```

“Phenotype” here means a measured behavioural profile under the declared probe.
It does not imply a discrete or map-wide taxonomy. Three routes cannot show
whether these properties are continuous, clustered, or correlated with route
length, position, margin, frequency, or any semantic description.

### 3.2 Carrier profiles — MEASURED

Mackinaw retained neuron 20 under 49/60 carriers. The successful set included
multiword carriers such as `it was`, `the water was`, `outside it was`, and
`winter was`, as well as short carriers including `at`, `the`, `near`, `from`,
and `north`.

Robin retained neuron 281 under only:

```
at robin
from robin
by robin
```

Cold retained neuron 38 under only:

```
it was cold
inside it was cold
```

This extends the earlier two-door observation into a three-route measurement:
carrier robustness differs by more than an order of magnitude between the
Mackinaw route and the robin/cold routes.

### 3.3 Continuation profiles — MEASURED

All three routes survived at least 85% of the tested continuations. This is
consistent with the ρ=R running-max ratchet: later material cannot erase an
earlier peak, although it can introduce a stronger rival.

Mackinaw departures:

```
at Mackin.               → 1821
at Mackiner              → 1790
at Mackinest             → 2594
at Mackins               → 1430
at Mackined              → 1790
at Mackining             → 1790
at Mackinness            → 1790
at Mackin all night      → 1581
at Mackin and dark       → 2331
```

Robin departures:

```
at robs                  → 2256
at robed                 → 2256
at robness               → 1790
at rob all night         → 1581
```

Cold departures:

```
it was colder            → 1888
it was coldly            → 1594
it was coldward          → 1790
it was cold outside      → 1865
it was cold and dark     → 2659
it was cold and warm     → 2659
```

The cold exceptions expose at least two mechanisms. `it was colder` lands on a
peak set in the earlier `it was` prefix. Most other departures introduce a
later winner whose recorded runner-up is neuron 38. Thus “continuation failure”
does not denote one mechanism: it includes inherited-prefix winners and new
downstream winners. `INTERPRETATION`

The established family remains visible:

```
it was cold      → 38
it was colder    → 1888
it was coldest   → 38
```

Across 180 valid one-token substitutions of `it was cold`, none retained 38.
The survival of `it was coldest` is instead an extension result protected by
the ratchet. This makes the earlier cold anecdote a concrete example of why
substitution and continuation should be measured separately.

---

## 4. Destination yield

### 4.1 Local substitution versus random strings — MEASURED, DERIVED

Aggregating the three RNG blocks per seed:

| Seed | Local distinct | Local entropy | Random distinct | Random entropy |
|---|---:|---:|---:|---:|
| Mackinaw | 14 | 2.642 bits | 11 | 1.013 bits |
| Robin | 16 | 2.402 bits | 10 | 0.915 bits |
| Cold | 10 | 0.946 bits | 10 | 0.915 bits |

Local substitution produced a broader, less concentrated destination set than
random strings around Mackinaw and robin. It offered essentially no diversity
advantage around cold. Directed local perturbation is therefore seed-dependent
in this run, rather than generically coverage-productive. `INTERPRETATION`

The three local RNG blocks also vary materially. For Mackinaw, individual
blocks produced 6, 9, and 14 distinct destinations; for robin, 6, 11, and 8;
for cold, 4, 8, and 4. Multiple RNG blocks were necessary to expose that
variance.

### 4.2 Random short-string collapse — MEASURED

Across the random blocks, neuron 1888 won:

```
Mackinaw-length random strings    150/180 (83.3%)
Robin-length random strings       153/180 (85.0%)
Cold-length random strings        153/180 (85.0%)
```

Neuron 1790 was the consistent distant second. The dominance of 1888 recurred
across all RNG blocks, but its scope is narrow: uniformly sampled,
round-trip-safe, short token strings under imp_r. It is not an estimate of the
background distribution over natural strings.

### 4.3 Discovery curves — MEASURED

The deterministic Mackinaw gate sweep reached 13 destinations by evaluation
30 and added none over the remaining 30 evaluations. The same curve appears
for robin because both branches evaluate the identical `at X` candidate list.
The result describes one shared template sweep viewed against two target
labels, not two independent neighbourhood samples.

Local discovery remained more seed- and RNG-dependent. Mackinaw's RNG-2 local
block continued from 12 destinations at evaluation 30 to 14 at 60, whereas its
RNG-0 block saturated at six by evaluation 30. Endpoint yield alone therefore
conceals different saturation behaviour.

---

## 5. Paired and repeated controls

The Mackinaw and robin B_GATE branches contain the same 60 `at X` strings.
Their destination distribution is consequently identical: 13 destinations,
with neuron 1888 winning 21/60 and 2256 winning 18/60. The target-retention
counts differ only because the same destination set is read relative to target
20 or target 281.

Robin and cold have the same realised seed length. Under each RNG seed their
D_RANDOM_TOKEN branches therefore contain identical random strings. This is a
paired control and makes their random summaries directly comparable; it is not
independent replication of the random generator.

Future summaries should mark shared candidate-set IDs so paired blocks cannot
be accidentally counted as independent evidence. `PROPOSAL`

---

## 6. Output-format finding — MEASURED

One random string begins with a literal double quote. `znou_probe.Recorder`
writes tab-separated cells directly after escaping tabs and line breaks, but
does not use CSV/TSV quotation rules. A generic `csv.DictReader` with its
default quote handling therefore merged this row with the following row and
appeared to reduce one random block from 60 to 58 entries.

The file itself is intact. Parsing with tab separation and quotation disabled
recovers all 1,623 rows exactly:

```python
csv.DictReader(file, delimiter="\t", quoting=csv.QUOTE_NONE)
```

`PROPOSAL` — update `Recorder` to use an explicit tab-delimited writer, or
document that quotation has no syntactic meaning in this format. The current
files must not be analysed with a default CSV parser.

---

## 7. What v6 establishes

- `MEASURED` The probe can recover sharply different carrier, continuation,
  and local-edit retention profiles from known routes.
- `MEASURED` Mackinaw is substantially more carrier-robust than robin or cold
  under the tested candidate set.
- `MEASURED` All three routes are highly continuation-robust, with later
  winners explaining the measured failures.
- `MEASURED` Cold is lost under all 180 sampled one-token substitutions while
  surviving 54/60 continuations.
- `MEASURED` Local substitution broadens destination yield over random strings
  for Mackinaw and robin, but not for cold.
- `MEASURED` Short random safe-token strings collapse predominantly onto 1888
  under this stack and readout.
- `DERIVED` The measured differences support route-level behavioural profiles
  as a useful output of the probe.

---

## 8. What v6 does not establish

- Three routes do not establish a taxonomy or population distribution.
- The labels do not describe the complete behaviour of neurons 20, 281, or 38.
- The probe does not identify semantic detectors.
- The carrier and suffix lists are designed candidate sets, not random samples
  from a declared language distribution.
- Repeated evaluation of a shared candidate list is paired evidence, not
  independent replication.
- Distinct destinations are an exploratory yield statistic, not proof that a
  branch improves discovery against the full published atlas.
- GPU v6 and CPU v5 are not exact device-matched replications.

---

## 9. Questions opened

1. Where do additional routes fall in the carrier × continuation × local-edit
   profile space?
2. Are the apparent phenotypes clusters, or convenient names for points on
   continuous axes?
3. Does carrier robustness correlate with margin, gate length, winning
   position, corpus frequency, or route multiplicity?
4. Can continuation failures be classified automatically into inherited-prefix
   winners and new downstream winners from `t_star` and runner-up records?
5. Does runner-up-directed mutation outperform undirected local substitution
   after controlling candidate budget and realised edit distance?
6. Does 1888 retain its random-string dominance across lengths, quadrants,
   layers, token-pool definitions, and independent stacks?

The next useful expansion is more seed routes rather than more perturbation
branches. The instrument now distinguishes the selected cases; the open
question is how those distinctions populate the wider map. `INTERPRETATION`
