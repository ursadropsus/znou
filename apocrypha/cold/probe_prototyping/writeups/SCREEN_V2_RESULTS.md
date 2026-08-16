# PHRASE-ENRICHMENT SCREEN v2 — RESULTS

Status: preliminary result note. Supersedes v1, which used a statistic that
breaks at large corpus size (§2.3). Tags follow `SPEC.md`: `MEASURED`,
`DERIVED`, `INTERPRETATION`, `PROPOSAL`.

A cheap ranking pass over already-cached destinations. For each sufficiently
sampled neuron, it reports the word n-grams occurring disproportionately among
that neuron's hit sentences relative to the corpus background. No model is
loaded; no forward pass is run.

Four results:

1. The screen recovers probe v9's manually derived N541 certificates from
   surface statistics alone, without a model.
2. **H4 from v9 §12 — that N541's concentration is Melville-conditioned — is
   falsified.** The binomial family reproduces in 14 of 20 corpora, including
   WikiText-103.
3. N541 is not unusual. 207 neurons concentrate in 4 or more of 20 corpora;
   58 in 10 or more.
4. `PRELIMINARY` — spot checks against Neuroscope, an external tool using a
   different readout on a different corpus, agree for the three neurons
   checked so far.

The screen is correlational. It identifies phrases *associated with* a
destination, not the span that *causes* it. Candidates require peak-anchored
minimisation (v7/v9 method) before any claim is made about them.

---

## 1. Run provenance — MEASURED

```text
script          tools/screen_neurons.py
python          3.11 (stdlib only)
model           NOT LOADED
gpu             NOT USED
```

Inputs:

```text
data/the_sea_implicit_resonance.json            7,353 sentences  (validation)
data/caches/_extracted/*/*.jsonl            1,273,584 sentences  (20 corpora)
```

The nineteen literary corpora total 407,475, matching SPEC §8.3. WikiText-103
adds 866,109 and is the only non-literary source. Quadrant `imp_r` throughout.

| Corpus | Sentences | Neurons ≥20 hits |
|---|---:|---:|
| wiki103 | 866,109 | 838 |
| shakespeare-complete | 128,885 | 325 |
| war-and-peace | 59,735 | 249 |
| ulysses | 32,368 | 139 |
| crime-and-punishment | 23,407 | 128 |
| within-a-budding-grove | 22,456 | 151 |
| moby-dick | 21,713 | 125 |
| leviathan | 20,450 | 106 |
| swanns-way-EN | 18,516 | 140 |
| du-cote-de-chez-swann-FR | 16,966 | 26 |
| book-of-the-dead | 14,585 | 65 |
| thus-spoke-zarathustra | 12,223 | 64 |
| poe-collected | 10,879 | 62 |
| the-king-in-yellow | 8,081 | 34 |
| the-prince | 5,014 | 31 |
| a-dolls-house | 3,560 | 23 |
| alice-in-wonderland | 2,969 | 11 |
| the-metamorphosis | 2,447 | 14 |
| tractatus | 2,277 | 10 |
| the-yellow-wallpaper | 944 | 4 |

---

## 2. Method

### 2.1 The statistic

For neuron `j` with hit set `H_j` in a corpus of `T` sentences, and word n-gram
`g` (n ∈ {2,3}) appearing in `c` of those hits and `B_g` of the corpus:

```text
enrichment(g, j) = (c / |H_j|) / (B_g / T)
rank score       = enrichment × c
```

Document frequency, not term frequency: a gram counts once per sentence.

### 2.2 Why enrichment and not frequency — MEASURED

Raw document frequency is uninformative. Ranking N541's 80 Melville hits by
raw frequency returns `of the`, the same answer it returns for nearly every
neuron including the default basins. Enrichment against corpus background
returns the v9 certificates.

### 2.3 The v1 defect and its correction — MEASURED

**v1 had no lower bound on `B_g`, and this breaks at scale.** If a phrase
occurs 3 times in the corpus and all 3 land on one neuron, enrichment is
`T / |H_j|`, which grows without bound in corpus size. The statistic rewards
*uniqueness*, not concentration.

At 7,353 sentences this was invisible. At 866,109 it dominated completely.
The v1-statistic top of WikiText-103:

```text
'didier drogba'   ( 3 hits, 25474x)
'levon helm'      (10 hits, 16656x)
'prof tica'       ( 8 hits, 27387x)
'titania and'     ( 3 hits, 33312x)
```

Proper nouns, not features — a birthday-paradox artifact given ~840 screened
neurons and a large tail of phrases occurring 2–4 times.

v2 requires `B_g ≥ max(3 × min_support, T / 10000)`: 9 for the seven-thousand
sentence cache, 86 for WikiText-103. After the floor, the same corpus returns:

```text
1828  'with the exception of'  (158 hits, coverage 0.98)
 744  'from the greek', 'is derived from'
 864  'his or her'
 438  'fewer than', 'no more than'
2024  'there are no'
2018  'prone to', 'risk of'
```

The floor leaves N541's certificates untouched at every corpus size tested.

`INTERPRETATION` — residual proper nouns remain (`saturday night live`,
`alice in chains`, `dunder mifflin` at 90–113 occurrences each). These clear
the floor legitimately and are not the same artifact. Whether they are real
or mark something structural those phrases share is unresolved. A
capitalisation filter would be the next mitigation and has not been applied.

### 2.4 Cross-corpus ranking uses percentile — DERIVED

Enrichment is **not comparable across corpora of different sizes**: identical
behaviour on N541 scores 67× in the 7k cache and 508× in Shakespeare, purely
because a larger background makes any phrase rarer. v1 ranked cross-corpus
survivors on median raw score, which silently ranked large corpora higher.

v2 records each neuron's rank percentile *within its own corpus* and ranks
survivors on the median of those. Raw enrichment is retained in the detail
TSV for inspection but is not used for ordering.

### 2.5 Thresholds

`min_hits = 20` per neuron per corpus, `min_support = 3` per phrase,
background floor as above. All three arbitrary; none swept.

---

## 3. Validation against probe v9 — MEASURED

Probe v9 established by exhaustive peak-anchored minimisation on a pinned CUDA
stack that N541's shortest certificates concentrate on paired constructions,
the most common exact forms being ` up and down` (11), ` now and then` (10),
` here and there` (8).

The screen, on the same 80 sentences, with no model, returns as its highest
scoring phrases:

```text
'up and down'     (11 hits, 67x)
'here and there'  ( 8 hits, 57x)
'and down'        (11 hits, 51x)
'now and then'    (12 hits, 50x)
```

Three of four are v9's top three exact certificates in near-identical support
order; the fourth is a substring of the first.

`INTERPRETATION` — one positive validation on the one neuron where ground
truth exists. Not a demonstration that the screen is generally sound.

### 3.1 Negative control — MEASURED

Default-basin neurons return flat enrichment, identifying themselves as
background:

```text
N1888   1,305 hits   'of the', 'in the', 'it is'   all ~1x
N1594     422 hits   'of the', 'in the'            1–2x
```

`DERIVED` — the screen separates concentrated destinations from basins without
being told which is which.

---

## 4. H4 is falsified — MEASURED

V9 §12 could not separate four readings of the N541 concentration from a
Melville-only positive sample. The fourth was:

> **H4 MELVILLE-CONDITIONED CONCENTRATION** — the recurrence reflects corpus
> vocabulary, style, and Atlas selection more than a corpus-independent neuron
> property.

N541 reaches the ≥20-hit threshold and returns binomial phrases in **14 of 20**
corpora: a-dolls-house, book-of-the-dead, crime-and-punishment, leviathan,
moby-dick, poe-collected, shakespeare-complete, swanns-way-EN,
the-king-in-yellow, thus-spoke-zarathustra, ulysses, war-and-peace, wiki103,
within-a-budding-grove.

Forms absent from the Melville sample appear independently: `over and over`,
`day after day`, `in and out`.

`INTERPRETATION` — H4 is falsified as stated. The recurrence cannot reflect
Melville's vocabulary when it reproduces in the Book of the Dead, *A Doll's
House*, and Wikipedia.

This does **not** settle v9's H1/H2/H3 (general coordination vs. lexicalised
binomials vs. convergent subfamilies). Probe v11's factorial finding stands:
novel connected repetition raises N541 reproducibly but won no destination in
48 tested cases, while familiar repetition retained 541 in 29/48. The
cross-corpus result adds breadth to the positive sample; it does not
distinguish frequency from structure.

---

## 5. External validation against Neuroscope — PRELIMINARY

Neuroscope (Nanda 2022, `neuroscope.io/gpt2-small/5/{N}.html`) publishes the
top-20 max-activating dataset examples for every GPT-2 Small neuron, computed
over OpenWebText. It is the same model and layer indexing (12 layers, 3,072
neurons) loaded through the same TransformerLens path.

**This is an independent axis of comparison in three respects**: different
corpus (OpenWebText vs. literary + WikiText), different readout (max
activation vs. argmax across all 3,072), different investigators.

Spot checks so far:

| Neuron | Screen phrases | Neuroscope top peaks |
|---|---|---|
| 541 | binomials | `top to bottom` ×3, `above and below`, `then and now` |
| 1109 | for instance / for example | pages dominated by `example`, `e.g.` |
| 2808 | if i had / if it were / had i | `not for`, `if it were not for` |

For N541 the five highest-activation snippets peak on ` bottom`, ` bottom`,
` below`, ` now`, ` bottom` — in every case **the second member of a paired
construction**, reproducing v9's finding that N541 peaked on the certificate's
final token in 80/80 routes.

`MEASURED` (for the five N541 snippets directly inspected).
`INTERPRETATION` — `top to bottom` appears in v9's certificate list; `above and
below` and `then and now` do not appear in any of the twenty corpora screened
here. Connectors again vary (`to`, `and`), consistent with v9 §5.1 rejecting a
single-token "the `and` neuron" account.

`OPEN` — this rests on three neurons and five snippets read by eye. A full
layer-5 pull (3,072 pages, `tools/fetch_neuroscope.py`) is in progress; the
agreement rate across all 207 candidates is not yet measured and could be low.
Nothing in this section should be treated as more than an encouraging
indication until that join is done.

---

## 6. N541 is not unusual — MEASURED

| Concentrating in ≥ N of 20 corpora | Neurons |
|---:|---:|
| 4 | 207 |
| 5 | 163 |
| 6 | 135 |
| 7 | 109 |
| 8 | 89 |
| 9 | 71 |
| 10 | 58 |

`INTERPRETATION` — N541 was probed because it had a large Atlas sample, not
because it was selected against alternatives. Its apparent exceptionality in
the v6–v15 sequence is a sampling artifact of that sequence.

### 6.1 N541's actual profile — MEASURED

```text
14 corpora (the widest reach of any neuron), median percentile 0.55
```

Under the v1 raw-score ranking this was obscured. N541 is **ubiquitous but not
dominant**: it appears nearly everywhere and ranks mid-table in each. That is
a different property from the top-ranked candidates below, and the distinction
only became visible after the §2.4 correction.

---

## 7. Candidate list — MEASURED (phrases) / INTERPRETATION (labels)

Neurons in ≥5 corpora by median percentile. Right-hand labels are the author's
reading and carry no evidential weight.

| Neuron | Corpora | pct | Recurring phrases | Provisional reading |
|---:|---:|---:|---|---|
| 1109 | 8 | 1.00 | `for example`, `for instance`, `as for example` | exemplification |
| 1151 | 5 | 0.97 | `half an hour`, `half a` | partitive duration |
| 295 | 5 | 0.95 | `i had been`, `it had been` | anterior aspect |
| 2793 | 7 | 0.92 | `such a`, `quite a`, `something of` | degree |
| 2373 | 5 | 0.89 | `so far`, `so far as`, `thus far` | extent |
| 62 | 6 | 0.88 | `nor is`, `nor is it`, `up to` | negative coordination |
| 2301 | 5 | 0.87 | `not only`, `not merely` | scalar negation |
| 2219 | 7 | 0.85 | `so as to`, `enough to` | purposive |
| 664 | 6 | 0.85 | `as though`, `as if` | hypothetical comparison |
| 43 | 7 | 0.85 | `glad to`, `glad to see` | positive affect |
| 2808 | 9 | 0.84 | `if i had`, `if it were`, `had i` | counterfactual conditional |
| 2590 | 5 | 0.84 | `for a moment`, `for a minute` | brief duration |
| 2241 | 7 | 0.82 | `going to`, `was going to` | prospective aspect |
| 2742 | 11 | 0.81 | `that is to say`, `is to say` | reformulation |
| 434 | 6 | 0.81 | `to day`, `to morrow`, `to night` | deictic day reference |
| 2432 | 6 | 0.80 | `could not`, `managed to`, `t help` | ability / inability |
| 550 | 7 | 0.79 | `not at all`, `not half so` | emphatic negation |

### 7.1 The two strongest — INTERPRETATION

**N1109 (exemplification).** 8 corpora, **median percentile 1.0000** — it
ranks first in every corpus it appears in. The realisations vary lexically
while holding function: `for instance` (Dostoevsky, Proust, Joyce), `for
example` (Hobbes, Poe, Wikipedia), `e g` (Wittgenstein), `such as` (Joyce). A
surface-string account does not unify `e g` with `for instance`.

**N2808 (counterfactual conditional).** 9 corpora, the widest of the
high-percentile group, with greater *syntactic* variation than N1109:
Shakespeare's inverted `had i` and `would it were`, Tolstoy's `if only` and
`had he not`, Hobbes's `if it were`, Wikipedia's `if i`. Inversion and
`if`-marking are different constructions realising the same category.
Neuroscope's independent readout adds `if it were not for`.

### 7.2 One requiring care — INTERPRETATION

**N1055** returns `m de charlus`, `de charlus`, `de son`, `en tous cas` — Proust
character names and French function words, clearing the background floor
legitimately because they are frequent in that corpus. This is the residual
proper-noun problem of §2.3 and marks the limit of a frequency-only filter.

---

## 8. What this does not establish

- The screen finds **correlation, not causation**. Confirmed against
  minimisation for N541 only.
- No peak position is computed. Certificates sit at the neuron's peak token;
  the screen has no token-level access and searches whole sentences.
- No phrase family other than N541's has been minimised. No claim is made
  about certificate length, margin, or carrier dependence for any other
  neuron.
- No matched negatives. Enrichment among hits is not enrichment over matched
  non-hits.
- `min_hits = 20` interacts with corpus size: WikiText-103 contributes 838
  screened neurons, *The Yellow Wallpaper* four. Large corpora dominate the
  cross-corpus counts, and the ≥N-corpora threshold does not correct for this.
- The background floor is a blunt instrument (§2.3, §7.2).
- The nineteen literary corpora carry the preparation defect disclosed in
  SPEC §8.4: fragmentation varies terminal diversity as a side effect of
  preparation, and two corpora lose over a quarter of their units to the
  under-3-words filter. The screen inherits and cannot correct this.
- All figures are `imp_r`. The other three quadrants are unscreened.
- Nothing re-evaluated the model. If the November 2025 caches are stale
  relative to SPEC §7's pin, every number inherits that.
- §5 rests on three neurons inspected by eye.

---

## 9. Next steps — PROPOSAL

1. **Complete the Neuroscope pull and join it.** In progress. Produces an
   agreement rate across all 207 candidates instead of three spot checks, and
   is the single strongest available external check.
2. **Minimise N1109 and N2808.** The expensive step, and the only one that
   converts a candidate into a result. Both now have attested routes across
   8–9 corpora rather than one author.
3. **Screen the other three quadrants.** SPEC §8.2 establishes that the lens
   changes the map; whether it changes *which neurons concentrate* is unasked.
4. **Add a capitalisation filter** and re-run, to address §7.2.
5. **Sweep `min_hits` and the background floor.** Both are doing unexamined
   work.
6. **Matched negatives.** For a candidate family, sample sentences containing
   the phrase that did *not* reach the neuron. v9 §14 lists this as its own
   principal gap.

### 9.1 Relation to Q4 and Q5 — INTERPRETATION

Neither is answered. But if a subset of destinations has legible,
corpus-independent phrase families, Q4 (does a practised operator policy beat
uniform sampling into the tail) becomes answerable on that subset rather than
the whole space — a far cheaper experiment than the general form. Q5 remains
untouched: nothing here measures what happens to a destination under small
edits.

---

## 10. Reproduction

```bash
cd tools

# validation against v9
python screen_neurons.py ../data/the_sea_implicit_resonance.json

# single corpus
python screen_neurons.py ../data/caches/_extracted/wiki103_full_2025-11-05

# cross-corpus, all twenty
python screen_neurons.py ../data/caches/_extracted --cross --min-corpora 4

# external comparison (3,072 pages, ~2h, resumable)
python fetch_neuroscope.py --probe 541
python fetch_neuroscope.py
```

Stdlib only. The screen takes roughly five minutes for all twenty corpora on
CPU. TSVs are written to `tools/results/`; single-corpus runs are named per
corpus, the cross run produces a summary and a detail file.
