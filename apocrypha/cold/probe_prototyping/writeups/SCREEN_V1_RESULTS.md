# PHRASE-ENRICHMENT SCREEN v1 — RESULTS

Status: preliminary result note. Tags follow `SPEC.md`: `MEASURED`, `DERIVED`,
`INTERPRETATION`, `PROPOSAL`.

A cheap ranking pass over already-cached destinations. For each sufficiently
sampled neuron, it reports the word n-grams that occur disproportionately among
that neuron's hit sentences relative to the corpus background.

Three results:

1. The screen recovers probe v9's manually derived N541 certificates without a
   model, from surface statistics alone.
2. **Hypothesis H4 from v9 §12 — that N541's concentration is Melville-conditioned
   — is falsified.** N541's binomial family reproduces in 13 of 19 independent
   corpora spanning Shakespeare, Hobbes, Dostoevsky, Joyce, Proust, Nietzsche,
   Poe, Ibsen, Tolstoy and the Egyptian Book of the Dead.
3. N541 is not unusual. 44 neurons show comparable cross-corpus phrase
   concentration in 10 or more of the 19 corpora; 102 in 6 or more.

The screen is correlational. It identifies phrases *associated with* a
destination, not the span that *causes* it. Every candidate below requires
peak-anchored minimisation (v7/v9 method) before any claim is made about it.

---

## 1. Run provenance — MEASURED

```text
script          tools/screen_neurons.py
python          3.11 (stdlib only: json, re, collections, csv, glob)
model           NOT LOADED
gpu             NOT USED
```

No forward passes were performed. The screen reads the destination caches
produced by the November 2025 pipeline runs and computes text statistics over
them. It inherits, and cannot correct, any defect in those caches.

Inputs:

```text
data/the_sea_implicit_resonance.json          7,353 sentences  (validation)
data/caches/_extracted/*/*.jsonl             407,475 sentences  (19 corpora)
```

The 407,475 figure matches SPEC §8.3's corpus total, so this is the same
nineteen-corpus set discussed in §8.4. Quadrant `imp_r` throughout unless
stated. `sample.jsonl` files (50-line previews) are skipped.

Corpus sizes:

| Corpus | Sentences |
|---|---:|
| shakespeare-complete | 128,885 |
| war-and-peace | 59,735 |
| ulysses | 32,368 |
| crime-and-punishment | 23,407 |
| within-a-budding-grove | 22,456 |
| moby-dick | 21,713 |
| leviathan | 20,450 |
| swanns-way-EN | 18,516 |
| du-cote-de-chez-swann-FR | 16,966 |
| book-of-the-dead | 14,585 |
| thus-spoke-zarathustra | 12,223 |
| poe-collected | 10,879 |
| the-king-in-yellow | 8,081 |
| the-prince | 5,014 |
| a-dolls-house | 3,560 |
| alice-in-wonderland | 2,969 |
| the-metamorphosis | 2,447 |
| tractatus | 2,277 |
| the-yellow-wallpaper | 944 |

---

## 2. Method

### 2.1 The statistic

For neuron `j` with hit set `H_j` in a corpus of `T` sentences, and word n-gram
`g` (n ∈ {2,3}) appearing in `c` of those hits and `B_g` of the corpus:

```text
enrichment(g, j) = (c / |H_j|) / (B_g / T)
rank score       = enrichment × c
```

Document frequency is used, not term frequency: a gram counts once per
sentence regardless of repetition.

Ranking on enrichment × support, rather than enrichment alone, prevents a
200× phrase appearing three times from outranking a 60× phrase appearing
twelve times.

### 2.2 Why enrichment and not frequency — MEASURED

Raw document frequency is uninformative. Ranking N541's 80 Melville hits by
raw frequency returns `of the` — the same answer it returns for almost every
neuron, including the default basins. Enrichment against corpus background
returns the v9 certificates. The distinction is the whole method.

### 2.3 Neuron-level summary

```text
peak       highest enrichment among the neuron's top phrases
coverage   fraction of the neuron's hits containing at least one top-5 phrase
score      peak × coverage
```

Coverage is a necessary second filter. N429 in Melville shows peak 131×
(`might as well`) at coverage 0.18 — a strongly enriched phrase accounting for
under a fifth of the sample, meaning most hits arrive by some other route.
High enrichment at low coverage is a subfamily, not a phenotype.

### 2.4 Thresholds

`min_hits = 20` per neuron per corpus, `min_support = 3` per phrase. Both
arbitrary and both worth sweeping.

---

## 3. Validation against probe v9 — MEASURED

Probe v9 established, by exhaustive peak-anchored minimisation of 80 Melville
sentences on a pinned CUDA stack, that N541's shortest certificates concentrate
on paired constructions, with the eight most common exact forms being ` up and
down` (11), ` now and then` (10), ` here and there` (8), ` one by one` (3),
` right and left` (3), ` old and new` (2), ` one after the other` (2), ` this or
that` (2).

The screen, run on the same 80 sentences with no model, returns as its four
highest-scoring phrases:

```text
'up and down'     (11 hits, 67x enrichment)
'here and there'  ( 8 hits, 57x)
'and down'        (11 hits, 51x)
'now and then'    (12 hits, 50x)
```

Three of the four are v9's top three exact certificates, in near-identical
support order. The fourth is a substring of the first.

`INTERPRETATION` — the screen recovers a manually derived, GPU-confirmed result
on the one neuron where ground truth exists. This is a single positive
validation, not a demonstration that the screen is generally sound. It does not
establish that an enriched phrase is the certificate for any *other* neuron.

### 3.1 Negative control — MEASURED

The default-basin neurons return flat enrichment, correctly identifying
themselves as background:

```text
N1888   1,305 hits   top phrases 'of the', 'in the', 'it is'   all ~1x
N1594     422 hits   top phrases 'of the', 'in the'            1–2x
```

`DERIVED` — the screen separates concentrated destinations from basins without
being told which is which. A neuron whose most enriched phrase sits at ~1×
has no phrase family to find.

---

## 4. H4 is falsified — MEASURED

V9 §12 listed four readings of the N541 concentration and could not separate
them from a Melville-only positive sample. The fourth was:

> **H4 MELVILLE-CONDITIONED CONCENTRATION** — the recurrence reflects corpus
> vocabulary, style, and Atlas selection more than a corpus-independent neuron
> property.

Screening all nineteen corpora independently, N541 reaches the ≥20-hit
threshold and returns binomial phrases in **13 of 19**:

| Corpus | Top enriched phrases |
|---|---|
| a-dolls-house | `up and down`, `and down`, `walking up and` |
| book-of-the-dead | `in and out`, `and out`, `day after day` |
| crime-and-punishment | `up and down`, `and down`, `down the room` |
| leviathan | `went in and`, `up and down`, `to mouth` |
| moby-dick | `now and then`, `up and down`, `here and there` |
| poe-collected | `one by one`, `one by`, `by one` |
| shakespeare-complete | `up and down`, `and down`, `up and` |
| swanns-way-EN | `here and there`, `now and then`, `now and` |
| the-king-in-yellow | `over and over`, `up and down`, `one by one` |
| thus-spoke-zarathustra | `again and again`, `old and new` |
| ulysses | `up and down`, `and down`, `in and out` |
| war-and-peace | `up and down`, `and down`, `and down the` |
| within-a-budding-grove | `now and`, `now and then`, `here and there` |

`MEASURED` — the family reproduces across four centuries, five source
languages in translation, and genres from Jacobean drama to funerary ritual to
analytic philosophy's neighbours. Forms absent from the Melville sample appear
independently: `over and over`, `day after day`, `in and out`.

`INTERPRETATION` — H4 is not merely weakened but falsified as stated. The
recurrence cannot reflect Melville's vocabulary when it reproduces in the
Book of the Dead and *A Doll's House*.

This does **not** settle v9's H1/H2/H3 (general coordination vs. lexicalised
binomials vs. several convergent subfamilies). Probe v11's factorial evidence
remains the relevant instrument there, and its finding stands: novel connected
repetition raises N541 reproducibly but never won a destination in 48 tested
cases, while familiar repetition retained 541 in 29/48. The cross-corpus result
adds breadth to the positive sample; it does not distinguish frequency from
structure.

---

## 5. N541 is not unusual — MEASURED

Neurons reaching ≥20 hits in at least one corpus: **408**.

| Concentrating in ≥ N corpora | Neurons |
|---:|---:|
| 2 | 257 |
| 3 | 202 |
| 4 | 158 |
| 6 | 102 |
| 8 | 67 |
| 10 | 44 |
| 13 | 20 |

`DERIVED` — cross-corpus phrase concentration at N541's level is common, not
rare. Roughly a hundred neurons are plausible candidates and roughly forty are
strong ones, before any minimisation has been attempted.

`INTERPRETATION` — N541 was probed because it had a large Atlas sample and
looked interesting, not because it was selected against alternatives. Its
apparent exceptionality in the v6–v15 sequence is a sampling artifact of that
sequence, not a property of the neuron.

---

## 6. Candidate list — MEASURED (phrases) / INTERPRETATION (labels)

Neurons in ≥6 corpora, ranked by median score. Labels in the right column are
the author's reading of the phrase sets and carry no evidential weight.

| Neuron | Corpora | Recurring phrases | Provisional reading |
|---:|---:|---|---|
| 1109 | 7 | `for instance`, `for example`, `e g`, `such as` | exemplification |
| 2793 | 6 | `such a`, `to such a`, `quite a` | degree |
| 2219 | 6 | `so as to`, `so as` | purposive |
| 2241 | 6 | `going to`, `was going to`, `not going` | prospective aspect |
| 1681 | 6 | `ceased to`, `began to`, `had ceased to` | phase / aspect |
| 2476 | 6 | `different from`, `similar to`, `compared to` | comparison |
| 43 | 6 | `glad to`, `am glad to`, `thank god` | positive affect |
| 550 | 6 | `not at all`, `not half so` | emphatic negation |
| 1934 | 6 | `is it not`, `was it not`, `it not` | tag interrogative |
| 2742 | 10 | `that is to`, `is to say`, `at any rate` | reformulation |
| 870 | 9 | `as well as`, `as well` | additive |
| 2808 | 8 | `if i had`, `if i were`, `if i` | counterfactual conditional |
| 2864 | 7 | `fast as`, `white as`, `white as a` | simile frame |
| 1763 | 7 | `continued to`, `continue to` | continuative aspect |
| 664 | 6 | `as though`, `as if`, `as though he` | hypothetical comparison |
| 2661 | 11 | `a most`, `is most`, `was most` | superlative |

### 6.1 Two worth taking first — INTERPRETATION

**N1109** is the cleanest. Seven corpora, and the phrase set varies lexically
while holding function: `for instance` (Dostoevsky, Proust, Joyce), `for
example` (Hobbes, Poe), `e g` (Wittgenstein), `such as` (Joyce). A neuron
tracking surface strings would not pick up `e g`. This is the strongest
candidate for a function-rather-than-form family in the list.

**N1681** is the tidiest single phenotype: `ceased to` / `began to` / `cease
to` / `had ceased to` across six corpora. Inchoative and cessative aspect
share a slot, which is linguistically coherent — both are phase verbs marking
a boundary of an event — and would be an unusually crisp thing to find.

### 6.2 One that needs care — INTERPRETATION

**N1120** (4 corpora) initially looked like the standout: `do feign`
(Shakespeare), `they pretend to` (Hobbes), `pretending to` (Tolstoy) — the same
concept in three different lexemes across four centuries. But the fourth corpus
returns `i supposed` / `i had supposed` (Proust), and Tolstoy also gives `was
supposed to`. English *suppose* covers both "believe" and "be expected to",
which are not the pretence sense. So the family may be genuine
assertion-without-actuality, or the screen may be collapsing two unrelated
senses because they share a surface form.

Recorded here because it is either the most interesting item in the list or a
clean example of how the screen misleads, and minimisation would settle which.

---

## 7. What this does not establish

- The screen finds **correlation, not causation**. An enriched phrase may be
  the certificate, may co-occur with it, or may be an artifact of what else
  the corpus contains. Confirmed for N541 only.
- No peak position was computed. Certificates sit at the neuron's peak token;
  the screen has no access to token-level activations and searches the whole
  sentence.
- Phrase families were not minimised, so no claim is made about certificate
  length, margin, or carrier dependence for any neuron other than N541.
- No matched negatives. High enrichment among hits is not enrichment over
  matched non-hits.
- The `min_hits = 20` threshold is arbitrary and interacts with corpus size:
  Shakespeare contributes 325 screened neurons, *The Yellow Wallpaper* four.
  Large corpora dominate the cross-corpus counts.
- The nineteen corpora carry the preparation defect disclosed in SPEC §8.4:
  fragmentation varies terminal diversity as a side effect of preparation, and
  two corpora lose over a quarter of their units to the under-3-words filter.
  The screen inherits this and cannot correct it. Re-running §8.4 with
  paragraphs unwrapped, as Q6 proposes, would also improve this result.
- Coverage as defined uses substring matching over normalised tokens, which
  over-counts short grams appearing inside longer ones.
- All figures are `imp_r`. The other three quadrants are unscreened.
- Nothing here re-evaluated the model. If the caches are stale relative to
  SPEC §7's pin, every number inherits that.

---

## 8. Next steps — PROPOSAL

1. **Run against full WikiText-103.** Held out of the public repo for
   licensing, present in the local mirror. At 400k–860k units it is both the
   largest available sample and the only non-literary one; anything surviving
   both it and the nineteen is unlikely to be a genre artifact.
2. **Screen the other three quadrants.** Cheap, and SPEC §8.2 already
   establishes that the lens changes the map. Whether it changes *which
   neurons are concentrated* is unasked.
3. **Minimise the top five.** N1109 and N1681 first. This is the expensive
   step and the only one that converts a candidate into a result.
4. **Sweep `min_hits`.** The 20-hit floor is doing unexamined work.
5. **Add matched negatives.** For a candidate phrase family, sample sentences
   containing the phrase that did *not* reach the neuron. v9 §14 lists this as
   its own principal gap and it applies identically here.

### 8.1 Relation to Q4 and Q5 — INTERPRETATION

Neither is answered here. But if a subset of destinations has legible,
corpus-independent phrase families, then Q4 (does a practised operator policy
beat uniform sampling) becomes answerable on that subset rather than on the
whole space, which is a much cheaper experiment than the general form. Q5
remains untouched: this measures nothing about what happens to a destination
under small edits.

---

## 9. Reproduction

```bash
# validation against v9
python tools/screen_neurons.py data/the_sea_implicit_resonance.json

# single corpus
python tools/screen_neurons.py \
    data/caches/_extracted/ulysses_2025-11-05/ulysses_2025-11-05.jsonl

# cross-corpus, all nineteen
python tools/screen_neurons.py data/caches/_extracted \
    --cross --min-corpora 4 --tsv results/screen_v1.tsv

# other quadrants
python tools/screen_neurons.py data/caches/_extracted --quadrant exp_r --cross
```

Stdlib only. Approximately four minutes for all nineteen corpora on CPU.
Handles both cache formats (`{sentence, neuron_id}` JSON and `{s, exp_r, exp_i,
imp_r, imp_i}` JSONL) and skips `sample.jsonl` previews.
