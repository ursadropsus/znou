# znou / J-Space

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
```

---

## Checking the results

Everything in `SPEC.md` tagged MEASURED has a file behind it, listed in
`MANIFEST.md` with a hash. Two checks need no GPU and take a minute:

```bash
# does your copy of the corpus cache match the one the spec pins?
python tools/hasher.py data/the_sea_implicit_resonance.json
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

**The stack is pinned and the pin is load-bearing.** §7 fixes the model
revision, the weight hash, torch 2.9.0+cu128, transformers 4.57.1, fp32, and
TF32 disabled. TF32 left on is the one configuration known to change `D`
silently. If your destinations disagree with the spec, check §7 first.

---

## What is not here

**The WikiText-103 full coverage run (§8.2.1).** Recovered partial run data only. 

**The live instance's database.** It holds operator activity from the hosted
prototype. The one table the spec's claims depend on is exported instead, as
`results/master_hit_counts.tsv`.

---

## Status

This is a proposal with measurements attached, not a finished result. The
foundational question — whether winning an argmax at layer 5 corresponds to
anything the network computes with — is listed as Q10 and **has not been
run**. Until it has, read every occurrence of *system*, *destination*,
*locus* and *space* as "the neuron that won", which is what §1 says they mean.

§9 lists thirteen open questions in the order worth attacking them. Several
are cheap. Corrections and results are welcome, especially ones that break
something.

`SPEC.md` also maintains a list of claims withdrawn from earlier drafts. That
list is part of the document, not an appendix to it.

---

## Provenance and non-affiliation

The name *J-space* is borrowed from Gurnee, Sofroniew, Lindsey et al.,
*Verbalizable Representations Form a Global Workspace in Language Models*
(Transformer Circuits, 2026). **The name and nothing else** — the two objects
are nearly inverses, and the spec explains how. The question that produced this
work came from Gurnee & Tegmark, *Language Models Represent Space and Time*
(arXiv:2310.02207).

This is an independent proposal. It is not affiliated with, endorsed by, or a
product of Anthropic, CCP Games / Fenris Creations, or anyone else.

The corpus is derived from *Moby-Dick*, public domain, with Project Gutenberg's
apparatus removed. Model weights are the original 2019 GPT-2 small release and
are not redistributed here.

---

## Licence

Code, specification text and results in this repository are licensed under
Apache-2.0 (see `LICENSE`), except as noted below.

Third-party material, not covered by that licence and not licensed by this
repository:

- GPT-2 small weights — released by OpenAI, not redistributed here; the
  scripts download them from Hugging Face.
- *Moby-Dick* — public domain. Project Gutenberg's apparatus has been removed
  from the derived corpus.
- Names and trademarks of EVE Online, EVE Frontier, CCP Games / Fenris
  Creations, Anthropic and Midjourney are the property of their respective
  owners. They are referred to descriptively; no affiliation or endorsement is
  claimed, and no trademark rights are granted by this licence.
- Any third-party images included are the property of their owners and are
  reproduced for reference and commentary only.
