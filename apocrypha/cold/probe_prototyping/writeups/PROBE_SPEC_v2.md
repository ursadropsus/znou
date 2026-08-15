# STRING ITERATION PROBE — PRELIMINARY DESIGN, v2

Working note, not a SPEC section. Tags follow `SPEC.md`: MEASURED, DERIVED,
INTERPRETATION, PROPOSAL. Measurements here come from `tools/probe*.py` on
the §7 stack. Results through probe v5 were produced on 2026-08-14 and are
backed by timestamped TSVs in `results/`. Nothing here has been reproduced on
a second stack.

All findings are imp_r (β_bos=1, ρ=R) at ℓ=5. None transfer automatically to
ρ=I; see §8.

This note covers two related uses of one evolving probe script:

1. characterising the neighbourhood of a known route; and
2. comparing perturbation strategies as generators of destinations.

They may remain branches of the same versioned script. Their endpoints must
remain distinct: target retention describes a route neighbourhood, while
distinct-destination yield describes a generator.

---

## 0. Purpose

The probe supports a mine → minimise → characterise → expand loop: take a
destination reached incidentally by a corpus, find a short string that reaches
it, determine what parts of that string are load-bearing, map nearby routes,
and test whether those routes expose new destinations.

It is not a competitor to `coordinate_ascent.py`. Coordinate ascent targets
specified neurons and reaches 2805/3072 under its reported budget. The probe
instead produces short, English-legible certificates and measurements of
their local structure: the λ_perp centre of the map rather than its rim
(§5.1). If coverage alone is the objective, use ascent. If route provenance,
robustness, or local topology matters, use the probe. `PROPOSAL`

---

## 1. Established mechanics

### 1.1 The running max decomposes over positions — DERIVED

The transformer is causal. For `s = c · w`, activations at positions belonging
to `c` are identical whether or not `w` follows. Therefore, for every neuron
`j`:

```
peak_j(c · w) = max(peak_j(c), peak_j(w | c))
```

The full destination is the argmax across `j` of these joined-string peaks.
This decomposition is exact for ρ=R. It does not imply that the conditioned
tail has a fixed response, or that only the carrier's winner and the target's
usual runner-up can compete.

### 1.2 A reduced comparison holds for the measured Mackinaw family — MEASURED

For the six tested `carrier + " Mackinaw"` strings, the observed margin of
neuron 20 was reproduced by:

```
Δ_20 = peak_20 − max(peak_1430, peak_inc(carrier alone))
```

```
carrier    predicted     observed
at          0.322105     0.322105
the         0.420925     0.420925
The         0.066184     0.066184
north       0.394161     0.394161
lake        0.183836     0.183836
ship        0.130951     0.130950   (1888 wins; sign inverts)
```

`MEASURED`, `results/probe_v4_carriers_*.tsv`.

The running-max decomposition is general. This three-neuron reduction is not:
it is an empirical regularity of the tested family. Another neuron activated
in the conditioned tail could become runner-up or winner without violating
§1.1.

This remains arithmetic on measured peaks, not a cheap predictor in the §8.1
sense. It cannot infer a destination without evaluating `A`.

### 1.3 Segment responses are context-modulated — MEASURED

`peak_20` for the fixed text tail ` Mackinaw` ranged from **2.8477** (`in`) to
**3.5176** (`north`) across 44 tested carriers. The spread of 0.67 exceeds the
corpus median Δ of 0.1676 (§8.1). The response is therefore not a context-free
constant of the tail.

An observed target-peak band can be used as a heuristic screen for more
carriers from the same sampling regime. It is not a sound universal bound.
A new carrier may drive the target outside the observed band, and a carrier
whose own peak lies below the target band may still lose to a rival activated
inside the conditioned tail. A sound exclusion or admission rule would need
bounds on the target and every relevant rival under the new context.

### 1.4 Position 1 is a distinct measured regime — MEASURED, DERIVED

For the tested route, ` Mack` activates neuron 1888 at **3.4378** in position 1
and **2.2953** in position 2, a difference of 1.14. Adding any carrier moves the
tail's first token out of that regime. This is consistent with E1 and §4.3,
and is measured here for a multi-token continuation.

Because position 1 sees only BOS and its own token, a carrier's position-1
activations are bit-identical when evaluated alone or at the start of a longer
joined string. Those activations are exactly cacheable across tails.

### 1.5 Prefix truncation is free under ρ=R — DERIVED, MEASURED

Let `t*` be a position where the winning neuron's running maximum is attained.
Cutting the string after `t*` preserves that peak and removes only later rival
peaks. At the token-sequence level, the destination is preserved and Δ can
only stay equal or increase.

```
'Upon my word were I at Mackinaw, ...wigwam.'   Δ 0.2975
'Upon my word were I at Mackinaw'               Δ 0.3502
'Upon my word were I at Mackin'                 Δ 0.3502
```

At the string level, the decoded prefix must be re-encoded and evaluated.
Token-boundary changes can make the realised prefix differ from the sliced
token sequence. `prefix_agrees` records that test rather than assuming the
derivation survives decoding.

This operation finds a minimal successful **prefix**, not a globally minimal
string. Deleting material from the left or middle changes the context of later
tokens and remains a search problem.

`INTERPRETATION` — truncation-found and optimiser-found short routes may be
different populations. In this example, removing the natural continuation
widened the margin, whereas §4.5.1 found thin margins among short routes under
a heavy search budget. Route provenance should therefore be retained if Δ is
ever used in route quality or pricing.

### 1.6 Equal margins are diagnostic clues, not position proofs — MEASURED

Several extensions share Δ to six recorded decimals:

```
0.322105   'at Mackinaw'  · 'at Mackin'   · 'at Mackinac'
1.049642   'at robin'     · "at robin's"
0.564337   'at rob'       · 'at robins'   · 'at robing'
```

For these measured families, direct peak and peak-position records show that
the extensions do not outbid the existing top peaks. The ratchet in §1.1
explains the collision.

In general, equal rounded Δ does not identify a unique causal position set.
It may arise from unchanged top-two peaks, coincident rounding, or changes that
preserve their difference. Collision grouping is a cheap candidate generator;
`t_star`, runner-up position, and watched peaks provide the attribution test.

---

## 2. Route structure observed so far

### 2.1 Two doors show different gating behaviour — MEASURED

```
j=20    ' Mack' + 'in'      bigram-dependent in the tested family. Neither
                            tested half reproduces the route independently.

j=281   ' rob'               single-token route in the tested family.
                            'at rob' reaches 281; continuation modulates its
                            peak: ∅ 2.925 · 'i' 3.230 · 'in' 3.411 ·
                            'by' 3.944.
```

These are operational gates: spans whose controlled removal or substitution
loses the destination. They are not claims that the neuron detects a semantic
concept. String deletion may also retokenize the remainder, so gate-location
runs must record both the intended token intervention and the realised token
sequence.

Two doors do not establish a taxonomy. Carrier robustness may be continuous,
multimodal, or correlated with route length, position, margin, frequency, or
other variables.

### 2.2 Tokenization is a demonstrated source of route failure — MEASURED

```
'at Robin'  → 2256          ' Robin' is one token and the tested route is absent
'robin'     → 2256          tokenises as ['ro', 'bin']
' robin'    → 1888          contains ' rob' in position 1
'at robin'  → 281           Δ 1.0496
```

Capitalisation, leading space, and position change both tokenisation and
context, and destroy this route in the tested cases. These examples establish
that tokenization must be controlled before a semantic interpretation is
credible. They do not establish that semantics contributes nothing once
tokenization and position are held fixed.

---

## 3. Positional perturbation taxonomy — PROPOSAL

Generic NLP labels such as lexical, tense, register, and syntactic remain
useful annotations. They should not be the primary experimental cut for `D`.
The first cut is positional:

```
CARRIER      material before the candidate gate. It competes through retained
             earlier peaks and changes the context and absolute position of
             everything after it.

GATE         the minimal operational span whose controlled intervention loses
             the target route. It is found experimentally, not assumed from
             words or morphemes.

CONTINUATION material after the gate. Under ρ=R it cannot erase earlier peaks,
             but it can create a stronger rival or modulate later contextual
             responses.
```

A controlled sweep varies one role at a time. Every row should additionally
record whether the realised token sequence still instantiates the intended
intervention.

---

## 4. Probe stages — PROPOSAL

```
S1  MINE          corpus → (sentence, j, Δ). Exists: data_pipeline.py.

S2  TRUNCATE      read t*, slice at the token level, decode, re-encode, and
                  re-evaluate. Log intended and realised token IDs.

S3  LOCATE GATE   ablate or substitute candidate spans in the short prefix.
                  Distinguish token-space intervention from string deletion.

S4  CHARACTERISE  hold the realised gate fixed; sweep carriers and
                  continuations separately. Measure target retention, margin,
                  conditioned target peak, rival set, and sensitivity.

S5  EXPAND        generate routes from the characterised seed:
                    (a) same-target routes → robustness and route multiplicity
                    (b) near misses/new winners → local adjacency and yield

S6  COMPARE       compare expansion arms at equal realised-string budgets and
                  across multiple seeds. Report retention and yield separately.
```

S5(b) is potentially coverage-productive. Earlier probes produced targets and
controls spanning several destinations, but whether directed near-miss
expansion beats appropriate random perturbation remains under test.

---

## 5. Probe v5: perturbation-arm comparison

### 5.1 Question and construction — PROPOSAL, MEASURED

Starting from `at Mackinaw → 20`, v5 compared four branches in one script:

```
A CARRIER   substitute the carrier; hold ' Mackinaw' fixed
B GATE      hold 'at'; substitute the following word
C LOCAL     replace one intended token in the seed sequence
D RANDOM    sample a nominally length-matched sequence from a safe-token pool
```

The arms do not optimise the same behavioural endpoint. A is principally a
target-retention test. B and D emphasise destination diversity. C can express
both. The comparison is therefore a retention–yield profile, not a single
winner-takes-all ranking.

### 5.2 Results — MEASURED

`results/probe_v5_yield_20260815-005601.tsv` and
`results/probe_v5_summary_20260815-005601.tsv`:

```
arm          n    distinct    target 20 retained    dominant destination
A carrier   59        4             53 (89.8%)       20:   53/59
B gate      52       13              1 ( 1.9%)       1888: 21/52
C local     60       11             26 (43.3%)       20:   26/60
D random    60        3              0 ( 0.0%)       1888: 56/60
```

To compare unequal sample sizes descriptively, expected distinct destinations
under rarefaction to 52 rows are approximately:

```
A 3.76    B 13.00    C 10.30    D 2.87
```

These are deterministic summaries of this run, not confidence intervals.
They show three different local behaviours around the seed: high carrier
robustness, broad branching under gate substitution, and intermediate
retention/yield under local mutation. The random control was dominated by
neuron 1888.

### 5.3 Limits discovered by v5 — MEASURED, DERIVED

The nominal budgets were unequal. `CARRIERS` contained a duplicate and yielded
59 unique strings; `WORDS` contained 52 entries. Future comparisons should
construct equal numbers of evaluated, unique realised strings per arm.

`novel` in v5 means absent from arms evaluated earlier in script order. It is
order-dependent and is not a coverage result. Actual novelty must be defined
against a frozen reference set; arm overlap should be reported symmetrically.

The canonicality check asks whether the final string is stable under
encode → decode → encode. It does not ask whether the final IDs equal the IDs
the generator intended. In C, only 55/60 realised strings were at token edit
distance 1 from the seed; four were distance 2 and one was distance 4. In D,
three nominally four-token candidates re-encoded to five or six tokens.

The random pool is a sampled multiset of 4,000 individually round-trip-safe
tokens. Duplicate token IDs receive additional sampling weight, and safety of
individual tokens does not guarantee boundary stability when tokens are
concatenated. This is a usable exploratory control, not yet a uniform baseline
over safe realised strings.

The v5 summary uses the upper middle observation for even-sized samples. If
the field is called `median_delta`, future versions should average the two
middle observations or explicitly rename the statistic. Conventional medians
for B and D are approximately 0.115870 and 0.480723, rather than 0.116225 and
0.491029.

---

## 6. Output schema — PROPOSAL

Timestamped TSV under `results/`, with a commented header carrying stack and
script metadata. One row per evaluated realised string:

```
string · bos · rho
n_tok · tokens · token_ids
dest · delta · t_star · runner_up · peak_val
peak_<j> · peakpos_<j>              for each watched neuron
prefix · prefix_dest · prefix_delta · prefix_agrees
branch · block · <probe-specific columns>
```

Generators that operate on token IDs should additionally log:

```
source_string · source_token_ids
intended_token_ids · realised_token_ids
intended_edit_distance · realised_edit_distance
intended_n_tok · realised_n_tok
roundtrip_stable · intervention_agrees
```

`roundtrip_stable` and `intervention_agrees` answer different questions and
must not be collapsed into a single `canonical` field.

Summary output should include:

```
n_requested · n_evaluated · n_unique_strings
target_retained · retention_rate
distinct_destinations · distinct_per_evaluation
destination_counts · pairwise_arm_overlap
novel_against_reference · reference_id
median_delta · delta_quantiles
```

`theta_sha256_local` is currently computed by `znou_probe` and has not been
reconciled with `tools/pin_stack.py`; retain the separate field name until it
has been.

---

## 7. Requirements for probe v6 — PROPOSAL

V6 may remain a single file with clearly labelled branches. Its minimum
methodological changes are:

1. Generate an equal number of unique realised strings per comparison arm.
2. Preserve intended and realised token IDs and test intervention agreement.
3. Replace order-dependent novelty with novelty against a frozen reference,
   plus symmetric overlap between arms.
4. Report target retention alongside distinct yield; do not reduce both to one
   score unless a weighting is declared in advance.
5. Correct the even-sample median.
6. Deduplicate the safe-token pool, or construct the complete safe-token set,
   and enforce realised length when length matching is claimed.
7. Repeat stochastic arms across several RNG seeds.
8. Add at least one unrelated seed certificate before generalising from the
   Mackinaw neighbourhood.
9. Record cumulative discovery curves, allowing arms to be compared at every
   shared budget rather than only at the endpoint.

Useful optional branches:

- continuation-only perturbations with the gate and carrier fixed;
- runner-up-directed substitutions versus undirected local substitutions;
- a natural-language random control matched on realised token length;
- stratification by realised edit distance rather than dropping retokenised
  candidates;
- replay mode that evaluates a saved candidate list on a second stack.

---

## 8. Scope limit: ρ=R only

Every result above uses the running-max ratchet. Under ρ=I only the final
position is read. Prefix truncation is not protected, earlier carrier peaks do
not compete directly with the terminal readout, and §8.2's terminal-token
effect dominates. An inference-quadrant probe would hold a final token fixed
and vary its preceding context. That is a different experimental branch, not
a boolean switch whose results can share the present interpretation.

---

## 9. What this does not establish

- Nothing here bears on ℛ*. “Gate,” “route,” and “response” refer to winners
  under `D`, not to a neuron's complete functional role.
- Two characterised doors and one v5 seed do not establish a map-wide
  taxonomy.
- V5's destination counts describe four particular candidate lists, not
  expected performance over a population of seeds or prompts.
- The 1888 collapse under random strings is measured for one RNG seed and one
  sampled token pool. Its scale is striking but its generality is unmeasured.
- Linguistic labels attached to substitutions do not isolate semantic effects
  unless tokenisation, position, and realised intervention are controlled.
- All results remain one stack, one day, and one operator until replayed.

---

## 10. Feeds

```
Q5      S4 and S6 measure short-string perturbation structure, where a single
        token is a large fraction of the route.

Q12     S2 records token-prefix/string-prefix divergences from natural text.

λ_edit  S5 and S6 build and compare local adjacency under declared edit
        metrics.

λ_route S2 supplies short successful prefixes per door, with provenance.

§6.2    Search-found and truncation-found short routes remain distinct route
        populations when interpreting Δ.
```
