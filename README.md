# znou / J-Space

(Work in progress. Restoring and documenting months-old work; the repository is
ahead of the prototype in places.)

A deterministic function from strings to neurons, and a game built on it.

`D(s)` takes a string, runs it through frozen GPT-2 small, reads the MLP
activations at layer 5, and returns the index of the neuron that fired
loudest. That index is the whole output — 3072 possible destinations, one
integer, no sampling, no randomness, no logits ever formed.

Nothing about which string lands where was designed. `it was cold` arrives at
neuron 38. `it was colder` arrives at 1888. `it was coldest` arrives back at
38. The weights are public, the function is fifteen lines, and no one can tell
you in advance where a sentence will go.

**`SPEC.md` is the document.** It defines the function, reports what has been
measured, and is explicit about what has not been. This README is how to run
and check it.

---

## What is here

```
SPEC.md          the specification. Start at §0; §4 and §8 hold the results
MANIFEST.md      every measured claim mapped to the file that produced it

tools/           the scripts that produced the numbers
data/            the corpus and the four quadrant caches
results/         their outputs

starmap/         the prototype server and the model runner
discover/        the atlas frontend
exchange/        the discovery ledger

apocrypha/       intervention pilots and axis maps. Not cited as results;
                 read the note under Status before drawing anything from them
frontier/        one addendum on J-space as a player-instantiated substrate
assets/          video, images, fonts
```

---

## Before you reproduce anything

Reproducibility was not a design goal during construction. These scripts were
written to answer questions, not to be re-run by strangers. The document was
assembled afterwards, from artifacts that already existed.

That ordering left three things a reader should know up front. None of them
invalidates a published number. All of them will cost you time if you meet them
by surprise.

**The pin is normative in §7 and unenforced in the code.** §7 fixes the model
revision, the weight hash, torch 2.9.0+cu128, transformers 4.57.1, fp32 and
TF32 disabled — and says plainly that loading without `revision=` makes the pin
decorative. Eight load sites in this repository do exactly that. They request
`gpt2` and take whatever HuggingFace currently serves. Today that is the same
snapshot every published number was measured against; nothing in the code
ensures it stays that way. `data/caches/prepare_caches.py` is the one loader
that passes a revision. The verification path below closes this manually, and
it takes about a minute. MANIFEST lists the eight sites.

**Two filenames mean opposite things.** `data/the_sea_raw.json` is the corpus
as evaluated. `data/the_sea.json` is the *computed cache* — it has destinations
in it. And `data/caches/the_sea.json` is the untrimmed source corpus, i.e. the
raw one, under the name that elsewhere means computed. Tell them apart by entry
count, never by name:

| file | entries | what it is |
|---|---|---|
| `data/the_sea_raw.json` | 7354 | the corpus as evaluated — input to `precompute_cache.py` |
| `data/the_sea.json` | 7353 rows | the computed cache — each row carries a destination |
| `data/caches/the_sea.json` | 7394 | the untrimmed corpus, provenance only |
| `discover/data/the_sea_raw.json` | 7394 | an orphan duplicate of the untrimmed corpus, under the trimmed file's name |

The 40 entries the untrimmed file holds beyond the evaluated range are Project
Gutenberg licence boilerplate, sitting past the last line of the novel.
`precompute_cache.py` does not slice them off — the shipped corpus is already
trimmed at that boundary. Substituting the untrimmed file would produce 7393
rows and the published cache would stop reproducing. SPEC §8 documents the
range; MANIFEST documents why it holds.

Watch the last row. `discover/data/precompute_cache.py` reads
`'./data/the_sea_raw.json'` as a *relative* path, and two files answer to that
name — so running it from the repository root reads the correct corpus and
running it from `discover/` reads the untrimmed one and writes 7393 rows. Run
it from the root, or delete the orphan.

**Two shipped files are UTF-16LE.** `starmap/requirements.txt` and
`results/margins.tsv`. The first sits directly on the install path below, and
`pip install -r` may reject it — if it does, re-save as UTF-8, which changes no
content. Read the second with `encoding='utf-16'`.

---

## Checking the results

Everything in `SPEC.md` tagged MEASURED has a file behind it, listed in
`MANIFEST.md` with a hash.

**Do this one first — it is the pin check.**

```bash
python tools/pin_stack.py
```

It prints your stack, your TF32 settings, and at the bottom
`sha256(theta)` — a fingerprint of the parameters actually loaded, independent
of file format or download path. Compare it by eye against the value in §7:

```
113687a222f8cf98039222c27b39aaf716493e5e8c1db94ea4e6544e0814088c
```

Agreement means you hold the same weights the spec was measured on and
everything below is meaningful. Disagreement means you do not, and nothing
below will match. This is a manual check performed by a reader who knows to
perform it, not a guarantee enforced by the code — which is the whole content
of the pin caveat above. `python tools/hasher.py` prints the current upstream
revision and per-file hashes if you want to see where a mismatch came from.

Two more that need no GPU:

```bash
# does your copy of the corpus cache match the one the spec pins?
sha256sum data/the_sea_implicit_resonance.json
# expect e2e0c5166a5a0518...

# does a database seeded from scratch reproduce the spec's coverage figures?
cd starmap && python seed_database.py
python ../tools/export_hits.py --db znou_exchange.db --out ../results/check.tsv
# expect reached: imp_r 1452 · exp_r 1339 · imp_i 2225 · exp_i 2151
# and sha256 2724a6dbea76f2a574821df456dc76dd62d208d1b8a2161aa9e6b010737ff5f0
```

The stronger one needs a GPU. `tools/replay_cache.py` re-evaluates all 7,353
sentences of the corpus against the current stack and compares to a cache
generated months earlier on different hardware. The spec reports 7353/7353
agreement, down to a winning margin of 1e-6. It should agree for you too — and
if it doesn't, that is a more interesting result than if it does.

---

## Running the prototype

Requires an NVIDIA GPU and Python 3.11.

```bash
# 1. environment
conda create -n znou python=3.11
conda activate znou

# 2. torch first — the +cu128 build is not on PyPI
pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r starmap/requirements.txt

# 3. build the database from the published corpus data
cd starmap
python seed_database.py

# 4. run
python local_dev.py
```

Then open **http://127.0.0.1:5000/discover/**

If step 2 fails on the requirements file, it is the UTF-16 encoding noted
above, not a bad dependency list. Re-save it as UTF-8 and retry.

First start is slow: it downloads and loads GPT-2. `local_dev.py` serves the
frontend and the API from one origin, standing in for the nginx setup the
hosted version used. It binds to `127.0.0.1` only.

No database ships with this repository. `seed_database.py` builds one from
`rarity_index.json` and the four `master_hits_{quadrant}.bin` files, which do
ship — so the coverage data you get is byte-identical to the author's, without
anyone having to send you a database.

---

## Two things to know before reporting a bug

**Every number here is layer 5.** ℓ is a parameter, not a discovery. Change it
and every figure changes while no definition does. See §0.1.

**TF32 is the one setting known to change `D` silently.** §7 requires it
disabled and `tools/replay_cache.py` sets the flags explicitly; most other
entry points inherit whatever your environment defaults to. `pin_stack.py`
prints the three relevant values. If your destinations disagree with the spec,
check those before anything else.

---

## What is not here

**The WikiText-103 full coverage run (§8.2.1).** Recovered partial run data
only — 172 coverage points to 860,000 units, from an interrupted run's
checkpoint. The curve survived because the run did not finish.

**The 407,475-sentence corpus behind §8.3.** Its coverage export ships as
`results/master_hit_counts.tsv` and reproduces exactly; the corpus file itself
has not been located.

**The live instance's database.** It holds operator activity from the hosted
prototype. The one table the spec's claims depend on is exported instead, as
`results/master_hit_counts.tsv`.

**Per-unit destinations for WikiText-103.** WikiText-103 is CC BY-SA and does
not sit inside this repository's Apache-2.0 licence. The sample, the checkpoint
hit arrays and the coverage log do ship, and those are what §8.2.1 rests on.

---

## Status

This is a proposal with measurements attached, not a finished result. The
foundational question — whether winning an argmax at layer 5 corresponds to
anything the network computes with — is listed as Q10 and **has not been
answered**. Until it is, read every occurrence of *system*, *destination*,
*locus* and *space* as "the neuron that won", which is what §1 says they mean.

A pilot *was* run, and its output is published under `apocrypha/` so it isn't
hidden. It does not answer Q10 and is cited nowhere as a result. Those reports
amplify neuron teams at +10.0 and print `CONCLUSION: Causal alteration
detected` for every team — including the ones that collapse into repetition.
The line detects any change in output, not a targeted effect, so it cannot
separate "this neuron carries the concept" from "amplifying anything by +10.0
breaks generation." Answering Q10 needs one neuron at a time, matched controls
sampled to the target's activation magnitude, `mlp.act` rather than `mlp.c_fc`,
and blind scoring. That has not been done.

§9 lists thirteen open questions in the order worth attacking them. Several
are cheap. Corrections and results are welcome, especially ones that break
something.

`SPEC.md` also maintains a list of claims withdrawn from earlier drafts, and
`MANIFEST.md` records where the tree and the document have disagreed and how
each disagreement was settled. Both lists are part of the documentation, not
appendices to it.

---

## Provenance and non-affiliation

The name *J-space* is borrowed from Gurnee, Sofroniew, Lindsey et al.,
*Verbalizable Representations Form a Global Workspace in Language Models*
(Transformer Circuits, 2026). **The name and nothing else** — the two objects
are nearly inverses, and the spec explains how. The question that produced this
work came from Gurnee & Tegmark, *Language Models Represent Space and Time*
(arXiv:2310.02207).

This is an independent proposal. It is not affiliated with, endorsed by, or a
product of Anthropic, MIT, CCP Games / Fenris Creations, or anyone else. The
authors cited above are cited as sources of an idea and a name; they have no
involvement in this work and no responsibility for it.

The corpus is derived from *Moby-Dick*, public domain, with Project Gutenberg's
apparatus removed from the evaluated range. Model weights are the original 2019
GPT-2 small release and are not redistributed here.

---

## Licence

Code, specification text and results in this repository are licensed under
Apache-2.0 (see `LICENSE`), except as noted below.

Third-party material, not covered by that licence and not licensed by this
repository:

- GPT-2 small weights — released by OpenAI, not redistributed here; the
  scripts download them from Hugging Face.
- *Moby-Dick* — public domain. Project Gutenberg's apparatus has been removed
  from the derived corpus's evaluated range; forty licence sentences remain in
  the untrimmed file at `data/caches/the_sea.json`, unevaluated.
- WikiText-103 — CC BY-SA, not redistributed here beyond a sample and derived
  count arrays.
- Names and trademarks of EVE Online, EVE Frontier, CCP Games / Fenris
  Creations, Anthropic and Midjourney are the property of their respective
  owners. They are referred to descriptively; no affiliation or endorsement is
  claimed, and no trademark rights are granted by this licence.
- Any third-party images included are the property of their owners and are
  reproduced for reference and commentary only.
