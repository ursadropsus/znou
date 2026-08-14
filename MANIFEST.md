# MANIFEST — claims to artifacts

Maps every MEASURED claim in SPEC to the script that produced it and the file
that holds the result. Built from `inventory.md`, root `znou/code/v1/znou`,
507 files, 2026-08-12. Revised 2026-08-13: §8.2.1 recovered, §8.4 added, caches
published. Revised 2026-08-14: every row below re-verified against a fresh
clone of the published repository; missing artifacts uploaded and renamed to
canonical names; zip references retired; reproducibility posture stated.
Further revised 2026-08-14 (evening): `apocrypha/` audited after an earlier
pass checked the wrong paths, the duplicate raw corpus deleted, encodings
corrected, and `prepare_caches.py`'s misleading MISPLACED note rewritten.

Status key: `OK` mapped and consistent · `?` mapping uncertain, named below ·
`GAP` no file found · `NEW` file exists, not referenced by the spec

All paths are relative to the repository root **as published**, and as of
2026-08-14 they have been checked to exist at those paths. Sha256 prefixes are
the first 12 hex characters and were recomputed from the published files on
2026-08-14; where a prefix here disagrees with one printed in SPEC, this file
is the later measurement.

---

## Reproducibility posture — read this first

Reproducibility was not a design goal during construction. These scripts were
written to answer questions, not to be re-run by strangers. The document was
assembled afterwards, from artifacts that already existed. That ordering is
visible in the tree and it is better stated than discovered.

Three consequences, all of them live:

**The pin is normative in SPEC §7 and unenforced in the code.** Eight load
sites call `from_pretrained` without a `revision` argument, so they fetch
whatever HuggingFace currently serves at `gpt2` rather than the specific
snapshot every published number was measured against. See *The pin* below for
the sites and for the verification path that works today without any code
change.

**Filename conventions were not stable across the project's life.** The same
two names, `the_sea.json` and `the_sea_raw.json`, mean opposite things in the
published tree and in the local archive at `eve/data/01_archives`. This has
already caused one error. See *Navigating the sea files* below.

**One shipped artifact carries a Windows-era encoding.** `results/margins.tsv`
is UTF-16LE with CRLF terminators, which naive readers mis-parse. It stays that
way deliberately — its sha256 is published, and re-encoding would break the
pin. `starmap/requirements.txt` had the same defect and was corrected on
2026-08-14, since nothing depends on its hash and it sits on the install path.
See *Encoding and hygiene* below.

None of these invalidate a published number. All of them cost a stranger time,
and one of them — the pin — could in principle cost correctness if the upstream
weights ever moved.

**Filename convention.** Suffixes like `foo (1).py`, `foo (2).py` are browser
download artifacts, not variants: the plain name is the *first* download and
higher `(n)` is *later*. Confirmed against the v15 sequence — `SPEC_THREE_v15.md`
1,757 lines, `(1)` 1,959, `(2)` and `(3)` 2,034 with identical hashes, i.e. the
last one downloaded twice. As of 2026-08-14 every published file carries one
canonical name; the `(n)` forms are gone from the tree.

---

## The pin

| claim | script | evidence | status |
|---|---|---|---|
| pinned artifacts, `sha256(theta)`, file hashes | `tools/pin_stack.py` `a53c526aa7a0`, 88 lines | §7 tables | OK — see the circularity note below |
| hashing helper | `tools/hasher.py` `375c68ab3132`, 13 lines | — | OK — pins via `hf_hub_download(..., revision=sha)`, the one loader that reads its revision from a variable |
| reference `D` | inline in §7 | — | OK — **revision pinned in the snippet** |
| `transformer_lens` 2.15.0, `nltk` 3.9.2 + punkt | — | pinned in §7 | OK. TransformerLens weight processing means the prototype's `state_dict` does not hash to `sha256(theta)`; §7 says so |
| third environment | `data/caches/*/report.md` | torch 2.4.0+cu121, transformers 4.57.1, RTX A4500, Ubuntu 24.04 | NEW — the §8.4 runs. Not produced under §7's discipline; disclosed in §8.4. A replay of `the_sea` on this stack would widen §8's invariance claim from two environments to three |
| **the pin is not enforced at load** | eight sites, listed below | — | **DISCLOSED, open.** Documented here and in §7 rather than silently fixed |

The eight unpinned load sites, verified 2026-08-14:

```
starmap/experiment_runner.py:25      tools/pin_stack.py:77
tools/coordinate_ascent.py:48-49     tools/unreachable_certificate.py:101
tools/replay_cache.py:36-37          apocrypha/cold/coldchat.py:33-34
tools/token_sweep.py:102-103
```

`data/caches/prepare_caches.py:288-289` is the only loader in the tree that
passes a `revision`, at `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

**The circularity.** `pin_stack.py` computes `sha256(theta)` — the value
everything else is checked against — from an unpinned load. On its own it would
certify whatever it happened to download. It is a ruler that measures itself.

**Why this is nonetheless verifiable today.** SPEC §7 publishes the fingerprint
as a written constant. A reader runs `pin_stack.py`, reads the hash it prints,
and compares it by eye against the number in §7. Agreement means they hold the
same θ; disagreement is visible immediately. That path requires no change to
any of the eight sites and is the recommended check — but it is a *manual*
check, performed by a reader who knows to perform it, not a guarantee enforced
by the code. Anyone re-running a measurement should do it first.

**If the eight sites are ever fixed**, the shape is: lift `pin_stack.py`'s
hashing loop into a shared `verify_theta()` and have the other seven call it.
`experiment_runner.py` is not a one-liner —
`HookedTransformer.from_pretrained` takes no revision, so the HF model must be
loaded pinned and passed as `hf_model=` with an explicit tokenizer, and the
resulting `state_dict` will not hash to §7's value because of TransformerLens
weight processing. Assert a derived fingerprint and document it, or the first
run looks like a failed pin. §7 already says this.

**This must land before any re-run of §8.4**, or nineteen fresh caches are
generated against an unpinned model and the work is done twice.

---

## Navigating the sea files

Four filenames, two of which swap meaning between the repository and the local
archive. This table is the authoritative reading for the **published tree**.

| file | what it is | entries | sha256 |
|---|---|---|---|
| `data/the_sea_raw.json` | **the corpus as evaluated.** Input to `precompute_cache.py` | 7354 — 7353 strings and one malformed `[]` | `12625decc7b6` |
| `data/the_sea.json` | **the computed cache.** Output of `precompute_cache.py`; each row carries a destination | 7353 rows | `a5cdf1bdfe75` |
| `data/caches/the_sea.json` | **the untrimmed source corpus**, kept for provenance. Not an input to anything published | 7394 — 7393 strings and one malformed `[]` | `27219032df8d` |
| `data/caches/the_sea_sailed.json` | a prototype Atlas save state. **Not a measurement** | — | `d1af4bbd8a50` |

A fifth file, `discover/data/the_sea_raw.json`, held a byte-identical copy of
the untrimmed corpus under the name that everywhere else means the trimmed one.
**Deleted 2026-08-14.** The trap it created is recorded below, because clones
taken before that date still contain it.

Read that top to bottom: `data/caches/the_sea.json` is *raw*, and
`data/the_sea.json` is *computed*, which is the exact opposite of what the
names suggest. In `eve/data/01_archives` the two names again mean the opposite
things. Check the entry count before moving any of these files. The entry count
is the reliable discriminator: 7394 is untrimmed, 7354 is the evaluated corpus,
7353 rows of dictionaries is the computed cache.

**The relationship between them, verified 2026-08-14.** The evaluated corpus is
a strict prefix of the untrimmed one — its 7353 strings are byte-identical to
the first 7353 of the untrimmed file, in order. The 40 entries that follow are
Project Gutenberg licence and donation boilerplate, beginning immediately after
the closing sentence of the novel. The trim removed a contiguous tail and
nothing else.

**Why the trimmed file is the shipped input, and must stay that way.** The
published `data/the_sea.json` was computed from the trimmed corpus and its 7353
sentences match one-for-one. `discover/data/precompute_cache.py`
`e7e74a47b2a6` iterates every string entry and skips only non-strings and empty
strings; **it does not implement the `[0:7353]` slice that SPEC §8 describes.**
The boundary holds because the shipped file ends there, not because the code
enforces it. Substituting the untrimmed file without changing the code would
produce 7393 rows — forty of them Gutenberg licence text carrying real
destinations — and the published cache would no longer reproduce. The
`[0:7353]` range in §8 should be read as a description of the shipped corpus,
not as a runtime guard.

**The trap, and it is live rather than hypothetical.**
`discover/data/precompute_cache.py:8` sets
`RAW_INPUT_FILE = './data/the_sea_raw.json'` — a *relative* path. Two files in
this repository answer to that name:

```
run from repo root   →  data/the_sea_raw.json           7354 entries  correct
run from discover/   →  discover/data/the_sea_raw.json  7394 entries  wrong
```

The second produced 7393 rows including forty Gutenberg licence sentences with
destinations, and the published cache stopped reproducing.

**Resolved 2026-08-14** by deleting the duplicate. Only one file now answers to
that name, so a run from the wrong directory fails loudly with
`FileNotFoundError` instead of silently writing a different corpus. Verified:
`precompute_cache.py:8` is the sole reference to `the_sea_raw.json` anywhere in
the tree, and the atlas never loads a raw corpus at all — `main.js` fetches the
four quadrant caches from a root-absolute `/data/`, and those already carry
destinations.

---

## Corpus — §8, B2a

| claim | script | evidence | status |
|---|---|---|---|
| source text | — | `data/moby_dick.txt` `c6a5a7d69345`, 1,246,660 bytes | OK — added 2026-08-13 so §8's provenance diff is checkable rather than asserted. Byte count re-verified 2026-08-14, matches §8's pin exactly |
| the corpus as evaluated | — | `data/the_sea_raw.json` `12625decc7b6`, 7354 entries | OK — **corrected 2026-08-14.** An earlier revision of this manifest claimed the untrimmed 7394-entry file had been swapped in. It had not, and it should not be: see *Navigating the sea files*. SPEC's inventory lines are being amended to match the tree rather than the tree amended to match SPEC |
| the corpus, untrimmed | — | `data/caches/the_sea.json` `27219032df8d`, 7394 entries | OK — provenance only. Not an input to any published result |
| the corpus with destinations | `discover/data/precompute_cache.py` `e7e74a47b2a6` | `data/the_sea.json` `a5cdf1bdfe75`, 7353 rows | OK — this is the *computed cache*, not the corpus. Verified 2026-08-14 to reproduce from `the_sea_raw.json` one-for-one |
| corpus derivation (apparatus stripped, openers added) | — | diff of `moby_dick.txt` against `data/caches/the_sea.json` | OK — B2a closed. 5 openers + 7348 Melville = 7353 evaluated, then 40 licence sentences and one malformed `[]`. *Call me Ishmael.* absent, BOM defect, disclosed in §8 |
| prototype Atlas / save state | `discover/js/main.js` | `data/caches/the_sea_sailed.json` `d1af4bbd8a50`, 2.8MB | NEW — a completed autoscan save in the pre-quadrant atlas format. **Not a measurement:** 546 discovered systems against §8's 545, the extra being neuron 2256 reached by hand-typed `as` / `asdf`, and 13 of its 7366 hits are manual. Ships because the cache system needs it; must not be cited. Usable as Q10 stimulus material — it records three natural Melville sentences reaching neuron 38 |
| cache, imp_r — hash pinned in §8 as `e2e0c5166a5a0518` | — | `data/the_sea_implicit_resonance.json` `e2e0c5166a5a` | OK — **re-verified against §8, 2026-08-14** |
| caches, other three quadrants | — | `data/the_sea_{explicit_inference,explicit_resonance,implicit_inference}.json` | OK |

## §4.1.1 — certificate vacuous at ℓ=5

| claim | script | evidence | status |
|---|---|---|---|
| pre-activation range, Θ, `\|{j : L_j > x⋆}\| = 0` | `tools/unreachable_certificate.py` `00278e7180e5`, 138 lines | printed output only | `GAP` — no result file. The claim rests on a console figure nobody can re-read without re-running. Redirecting one run to a committed file would close it |

## §4.3 — one-token census

| claim | script | evidence | status |
|---|---|---|---|
| all 50,257 tokens, four quadrants | `tools/token_sweep.py` `e13d0c70d14a`, 322 lines | `results/sweep_tokens.tsv` `a96c0a6502d3`, 50,259 lines | OK — **resolved 2026-08-14.** The 322-line file is the one published, under its canonical name. The 317-line earlier iteration is not in the tree |

## §4.4 — activation suppression

| claim | script | evidence | status |
|---|---|---|---|
| per-neuron max over single tokens, reached vs unreached | `tools/token_sweep.py` `e13d0c70d14a` | `results/sweep_neurons.tsv` `1cfb11b56559`, 3,074 lines | OK — attribution resolved. `token_sweep.py` writes both tsvs from one run and its `ghost_test()` **is** §4.4 |
| — | `tools/locus_check.py` `decd9d62c637`, 306 lines | — | NEW — published for completeness. Superseded by `token_sweep.py`'s `ghost_test()`; not the source of any published figure |
| Cliff's δ (B4, open) | `tools/cliffs_delta.py` | would derive from `sweep_neurons.tsv` | `GAP` — open TODO |

## §4.5 / §4.5.1 / §4.5.2 — coordinate ascent

All outputs consolidated under `results/` on 2026-08-14. Earlier revisions of
this manifest used `tools/` prefixes carried over from the working tree.

| claim | script | evidence | status |
|---|---|---|---|
| sweep over all of J | `tools/coordinate_ascent.py` `8b815cf238d1`, 286 lines | `results/ascent_all.tsv` `a9e01d4957ce`, 3,075 lines | OK — the capitalisation discrepancy is retired; the published file is lowercase, matching SPEC |
| calibration, 100 systems | same | `results/control.tsv` `d736d51bb79b`, 103 lines | OK — this is the second-iteration run, formerly `control2.tsv`, published under the canonical name. Its footer reads `# targets 100  hit_tok 96  hit_str 88 (306s)`, which is §4.5's second budget row and the 88% denominator |
| the hard 172 | same | `results/hard172.txt` `d3a06d9d8e7b`, 172 lines | OK |
| residue detail | same | `tools/residue.tsv` `2fdfcc782c0c`, 175 lines | OK — formerly `residue2.tsv` |
| round-trip-safe token pool (49,905) | same | `results/safe_tokens.pt` `17b7e8ef6a4f` | OK |
| systems reached, imp_r | same | `results/reached_imp_r.txt` `1029d8a1fa80`, 545 lines | OK |
| systems reached, hard budget | same | `tools/reached_imp_r_hard.txt` `09e80de0318d`, 187 lines | OK |
| Melville routes, n=545 | — | `results/melville_routes_imp_r.txt` `f13530e399e0`, 545 lines | OK — the Q5 sample frame |
| median Δ by budget class | — | `results/margins.tsv` `92134b394ab3`, 546 lines | OK — **line count corrected 2026-08-14.** Earlier revisions said 1,093. The file is UTF-16LE with CRLF terminators; a counter treating `\r` and `\n` as separate breaks doubles the count. 546 = one header plus the 545 rows, consistent with n=545 |

## §8 — cross-environment replay

| claim | script | evidence | status |
|---|---|---|---|
| 7353/7353 agreement | `tools/replay_cache.py` `6ccd7f4fa7ae`, 134 lines | `results/replay_results.tsv` `cf7fa2ddac60`, 7,354 lines | OK |
| tightest arrival Δ=1e-6 | same | same | OK |
| the 18 below Δ<0.001 | same | `results/fixtures18.txt` `04155f73da8b`, 18 lines | OK |
| the 211 below Δ<0.01 | same | derived from `replay_results.tsv` | `?` no standalone file; recoverable by filtering |
| §8 fixture table (18 hand-written strings) | `--fixtures` flag | — | `GAP` — `typed_lines.txt` and `say_lines.txt` are not in the published tree. The 18 strings are printed verbatim in §8, so the table is readable; the input file that generated them is not published |
| TF32 flags set explicitly | `tools/replay_cache.py` | — | OK — the one place in the tree where numerical-precision state is pinned rather than inherited. Worth imitating elsewhere |

## §8.1 — margin distributions

| claim | script | evidence | status |
|---|---|---|---|
| n=545 first arrivals | — | `results/margins.tsv` `92134b394ab3` | OK — read with `encoding='utf-16'` |
| n=7353 all sentences | `tools/replay_cache.py` | `results/replay_results.tsv` | OK |

## §8.2 / §8.2.1 / §8.3 — coverage

| claim | script | evidence | status |
|---|---|---|---|
| imp_r 545 vs imp_i 57, terminal-char table | — | the four `data/the_sea_*.json` caches | `?` script not identified |
| WikiText-103, `wiki103test_511`, 40k, four quadrants | lost | lost | **CLOSED, superseded.** The original run took its drive with it and is not in any backup. §8.2.1 no longer rests on it: its figures are retired and replaced from the run below. The lost data stays lost; nothing now depends on it |
| WikiText-103 coverage curve, 172 points to 860,000 units | `tools/data_pipeline.py` `f941b75bf1f3` | `data/caches/_extracted/wiki103_partial870k_2025-11-05/checkpoint.json`, 42KB | OK — **the recovery.** The run was interrupted before writing `report.md`, but `coverage_log` is appended every 5,000 units and flushed every 20,000, so the curve survived *because* the run did not finish. Resolves Q6 to composition: at 400,000 units imp_r/imp_i = 1.79 against §8.3's 0.65 at 407,475, and the curves have not crossed by 860,000 |
| corpus coverage, 407,475 sentences | `tools/export_hits.py` `84095d985007`, 129 lines | `results/master_hit_counts.tsv` `2724a6dbea76`, 12,292 lines; `starmap/master_hits_{exp_i,exp_r,imp_i,imp_r}.bin`, 12,288 bytes each = 3072 × uint32 | OK — confirmed. Reproduces §8.3 exactly (1452 / 1339 / 2225 / 2151) and a freshly seeded database exports byte-identically |
| the §8.3 corpus itself | — | — | `GAP` — the 407,475-sentence corpus has not been located. Its terminal-character entropy is the cheapest open measurement in the project and cannot be taken until the file is found |
| nineteen book corpora, §8.4 | `tools/data_pipeline.py` `f941b75bf1f3` | `data/caches/_extracted/<corpus>_2025-11-05/` | OK — see §8.4 below |
| cache analysis | `tools/cache_analyzer.py` `960308a6098d`, 230 lines | — | NEW — published 2026-08-14 alongside `data_pipeline.py`. From the same 2025-11 run; not the source of a published figure |

## §8.4 — corpus composition, nineteen books

Published under `data/caches/_extracted/`, one directory per corpus, all dated
2025-11-05. Twenty directories: the nineteen books plus `wiki103_full`.

The extracted tree is published rather than archives, so the files are readable
and diffable in place. Earlier revisions of this manifest and of SPEC §7.2
described zipped archives; that described the local holding at
`eve/data/01_archives`, not the repository, and those references are retired.
The stale path `discover/data/caches/` is why the publish step missed these
until 2026-08-13.

```
Shakespeare · Ulysses · War and Peace · Swann's Way (EN) ·
Du Côté de chez Swann (FR) · Budding Grove · Crime and Punishment ·
Moby-Dick (full) · Leviathan · Machiavelli · Metamorphosis · Poe ·
The Yellow Wallpaper · Tractatus · Zarathustra · Alice ·
Book of the Dead · A Doll's House · The King in Yellow
```

Earlier drafts of this manifest said nineteen and then named eighteen. The
missing one was **Du Côté de chez Swann** — French, the same novel as
`swanns-way`, and now the most useful single item in the set: it holds author,
content, scale and preparation constant and varies only language.

`alice-in-wonderland_2025-11-05` failed to upload in the first publish and was
restored on 2026-08-13. Verified 2026-08-14: twenty directories under
`_extracted/`, twenty data rows in `_reports/summary.tsv`.

| claim | script | evidence | status |
|---|---|---|---|
| per-corpus coverage, four quadrants | `tools/data_pipeline.py` | `<corpus>_{quadrant}.bin`, 3072 × uint32, plus `report.md` | OK — `prepare_caches.py` recomputes coverage from each `.bin` and checks it against that corpus's own report |
| per-unit destinations | same | `<corpus>.jsonl` | OK for the books. **Withheld for wiki103** — WikiText-103 is CC BY-SA and does not sit inside this repository's Apache-2.0 licence. `sample.jsonl`, the checkpoint hit arrays and the coverage log do ship, and those are what §8.2.1 rests on |
| wiki103 hit arrays | same | `checkpoint_hits_{quadrant}.bin` | OK — note these are the *checkpoint* arrays, not `<corpus>_{quadrant}.bin`. The run was interrupted, so finalisation never wrote the latter. The last checkpoint is at 860,000 units |
| staging and verification | `data/caches/prepare_caches.py` `16fe635ac69c` | `_reports/summary.tsv`, `_reports/CACHES.md` | OK — the only loader in the tree that pins a `revision` |
| units are line fragments, not sentences | `tools/data_pipeline.py:262-267` | punkt applied per physical line | **DEFECT, DISCLOSED in §8.4.** Line numbers corrected 2026-08-14 against the published file. The loop at 262 iterates physical lines and calls `nltk.sent_tokenize` on each at 267, so hard-wrapped sources yield line fragments. Not comparable with §8.2, §8.2.1 or §8.3. The fix is to unwrap paragraphs before splitting and checkpoint on paragraph index rather than line index — `state.line_index` at 263-265 is *why* the bug exists, since resume must survive |
| under-3-words filter | `tools/data_pipeline.py:159` | report count minus jsonl rows | Verified at line 159. Discards 38.3% of *A Doll's House* and 26.7% of *Shakespeare*. Both excluded from any regression |
| French collapse | — | `du-cote-de-chez-swann-FR` vs `swanns-way-EN` | OK — 3.7–6.0× fewer destinations at 0.92× the units. Reported in §8.4 as a property of θ and its training distribution |

## §5 — map and lenses

| claim | script | evidence | status |
|---|---|---|---|
| λ_raw, λ_grav, λ_norm, λ_orr | `discover/js/main.js` `3b7818d5bab2`, 1,293 lines | — | OK |
| four-quadrant atlas, save migration | same, `discover/js/stateManager.js` | — | OK |
| `b_j ~ Unif`, unseeded, per client | `main.js:302-308`, `distributionSize = 70` | verified against `the_sea_sailed.json`: x spans −34.998 to 34.986 | OK — no projection of θ, so §5's "no metric on J" stands. That the draw is *unseeded* is disclosed in §5 as an open design question: every operator holds a different sky, screenshots are not comparable, and a data purge destroys any learned constellation. Deriving `b_j` from a hash of `j` would keep exactly zero information from θ, so §5 is unaffected either way |
| rarity index | `starmap/seed_database.py` | `starmap/rarity_index.json` `e51f517ade1c` | NEW — not referenced in spec |
| similarity matrices over J | `apocrypha/cold/const_mapper_*.py`, `constellation_mapper2.py` | `AXIS_OF_TEMPERATURE_inf/`, `AXIS_OF_TEMPERATURE_res/`, `AXIS_OF_TSALAL/` — each holding `*_similarity_matrix.txt`, `*_heatmap.png`, per-pole `*_fingerprint.png`, `*_report.txt` | **OPEN, and larger than §5 accounts for.** A similarity matrix over neurons is a metric-shaped object and §5 claims no metric on J is asserted by any lens. §5's open note names `AXIS_OF_TEMPERATURE` only; there are **three** axes published, two of them temperature under different quadrants and one, TSALAL, that §5 does not mention at all. Say what all three measure, or amend §5 |

## §6 — reward, exchange

| claim | script | evidence | status |
|---|---|---|---|
| exchange / claim / leaderboard | `starmap/api_server.py` `b3b49354befe`, `starmap/market_ticker.py` | `exchange/` | OK — **ambiguity resolved.** `game/starmap/` was a stale duplicate and is not published. `local_dev.py` computes SITE_ROOT one level above itself and expects `index.html`, `discover/`, `exchange/` there, which is only true of `starmap/` |
| live instance data | — | — | OK — **resolved.** No database is published. `*.db` is gitignored and no `znou_exchange*.db` is tracked. `seed_database.py` rebuilds the database from the four `master_hits_*.bin` files plus `rarity_index.json`, so none needs to ship |

## Running the thing

| purpose | file | status |
|---|---|---|
| local launcher | `starmap/local_dev.py` `e317ffde5bda` | OK |
| server | `starmap/api_server.py` `b3b49354befe` | OK |
| experiment runner (loads GPT-2) | `starmap/experiment_runner.py` `27139924f1e1`, 461 lines | OK — line 25 is unpinned; see *The pin* |
| config | `starmap/config.py` | OK |
| deps | `starmap/requirements.txt` `75c825647e17` | OK — **the seven public-IP matches were false positives**, version strings like `torch==2.9.0+cu128` matching an IPv4 pattern. No non-local address appears anywhere in the published tree. Re-encoded to plain ASCII on 2026-08-14 and verified through pip's parser |
| db init | `starmap/initialize_database.py`, `starmap/verify_db.py` | OK |
| db seed from published bins | `starmap/seed_database.py` | OK — the reason no database needs shipping |

Startup, verified against the published tree:

```
cd znou-publish/starmap
python seed_database.py      # once
python local_dev.py          # then http://127.0.0.1:5000/discover/
```

Torch must come from the PyTorch index, not PyPI:
`pip install torch==2.9.0 --index-url https://download.pytorch.org/whl/cu128`

---

## Encoding and hygiene

Small, cosmetic, and each one costs a stranger ten confused minutes.

| item | detail | status |
|---|---|---|
| `results/margins.tsv` | UTF-16LE, CRLF, no BOM | Readers must pass `encoding='utf-16'`. A UTF-8 reader sees interleaved null bytes. This is the source of the retired 1,093-line figure |
| `starmap/requirements.txt` | now ASCII, CRLF, no BOM, `75c825647e17` | **RESOLVED 2026-08-14.** Previously UTF-16LE, then briefly UTF-8-with-BOM — both break `pip install -r`, the BOM by folding three bytes into the first package name so pip searches for `\ufeffaccelerate`. Verified through pip's own parser: the current file parses |
| CRLF terminators | `ascent_all.tsv`, `control.tsv`, `replay_results.tsv`, `sweep_tokens.tsv`, `sweep_neurons.tsv`, `residue.tsv` | Harmless to Python's `csv` and to pandas; noted so nobody diffs against an LF copy and reports a mismatch |
| `data/caches/prepare_caches.py` MISPLACED note | described `data/caches/the_sea.json` as a "duplicate of data/the_sea.json one level up" | **RESOLVED 2026-08-14.** It was not a duplicate — it is the untrimmed 7394-entry raw corpus, and the file it was called redundant against is the 7353-row computed cache. Anyone acting on that string would have deleted the only surviving copy of the untrimmed corpus. The note now states which file is which and says do not delete. This was the last place in the tree still asserting the old reading |
| `results/` vs `tools/` | `control.tsv` and `margins.tsv` existed byte-identically in both | Resolved 2026-08-14 — the `tools/` copies were deleted. This manifest cites `results/` |
| `discover/data/datasets.json` | lists `alice_full_2025-11-05` and `moby_dick_full_2025-11-05` under `data/caches/` | **DEAD CONFIG, not a bug** — corrected 2026-08-14 after an earlier revision here called it functionally broken. Nothing in the tree reads this file. The atlas loads a single precomputed cache via `CACHE_BASE_URL` in `main.js`; multi-corpus browsing in-game was never built, and the twenty corpus runs were coverage and rarity experiments, not game content. The paths also predate the rename to readable slugs and the move into `_extracted/`. **Kept deliberately**, as a record of a direction that was prototyped and not built. Nothing is removed from this tree unless it is a duplicate or carries personal data |
| `discover/data/the_sea_raw.json` | orphan duplicate of the untrimmed corpus | Resolved 2026-08-14 — deleted. See *Navigating the sea files* |
| `starmap/__pycache__/` | four `.pyc` files were tracked | Resolved 2026-08-14 — untracked. They embedded the author's local Windows path |

---

## NEW — present in the tree, absent from the spec

These are not gaps in the manifest; they are work the document does not
mention, listed so it can be either cited or explicitly set aside.

**Correction, 2026-08-14.** An earlier revision of this section stated that the
intervention work was not published and that "nothing tsalal-shaped is
published or cited." **That was wrong.** It was checked against the `starmap/`
paths an older inventory named; the files had since moved to `apocrypha/`. The
accurate position is below.

### The intervention work is published, under `apocrypha/`

| what | where | status |
|---|---|---|
| eight intervention reports | `apocrypha/cold/intervention_report_surgical_2025-11-12_20-40-43.txt`, six more under `cold/barrage/` dated 2025-11-14, and `cold/AXIS_OF_TSALAL/intervention_report_tsalal_2025-11-13_06-59-28.txt` | **PUBLISHED, uncited.** Pilot data, not a result |
| the harness | `apocrypha/cold/coldchat.py`, `polarity_override_team.py`, `barrage/polarity_override_team_L.py` | PUBLISHED, uncited |
| the axis mappers | `apocrypha/cold/const_mapper_inference.py`, `const_mapper_resonance.py`, `const_mapper_resonance_ts.py`, `constellation_mapper2.py`, `verify_landmarks.py` | PUBLISHED, uncited. Produce the similarity matrices — see §5 above |
| an EVE forum thread as PDF | `apocrypha/not-our-creation.pdf` | **PUBLISHED — licence question, see below** |

**What the reports actually contain, and why they are not a Q10 answer.** Each
runs a fixed master prompt through a baseline pass and then through several
amplified neuron teams at strength +10.0, printing baseline and intervened
continuations. The surgical set tests four teams — *Minimalist Coordinators*
(2 neurons), *Specialists Only* (3), *Brute Force / Top 8* (8), *Full Roster*
(11). This is the pilot behind the Q10 note: the 3-neuron team stays fluent and
shifts toward cold and stillness, while the 2- and 8-neuron teams collapse into
repetition at the same amplification.

The reports print `CONCLUSION: Causal alteration detected` for **every** team,
including the ones that merely degenerate into repetition. That line detects
*any* change in output, not a targeted semantic effect, so it cannot distinguish
"this neuron carries the concept" from "amplifying anything by +10.0 breaks
generation." **That is precisely why Q10 is still open**, and why its four
required changes are one neuron at a time, k matched controls per target
sampled to the target's activation magnitude, hooking `mlp.act` rather than
`mlp.c_fc`, and blind scoring against a fixed lexicon. `coldchat.py` is most of
the harness needed to do it properly.

§4's statement that ℛ* has no results in it stands — none of this is cited as a
result anywhere in SPEC. But the artifacts are in the repository, so the
document should say they exist and say why they are not evidence, rather than
being silent and letting a reader find them unaccompanied.

**Licence question — `apocrypha/not-our-creation.pdf`.** Five pages,
Print-to-PDF of an EVE Online forum thread on Jove lore, captured 2026-08-12.
Forum posts belong to their authors under CCP's terms and are not the
repository's to redistribute under Apache-2.0. Either drop it and cite the
thread by URL, or keep it and add it explicitly to README's third-party list.
It is currently in neither.

Verified 2026-08-14 as **not published**: `game/`, `old/`, `oldsite/`,
`assets/sfx/`, `starmap/test_intervention.py`, `starmap/constellations/`,
`starmap/test_cold*.py`, `starmap/test_lensing*.py`, and all three
`znou_exchange*.db`.

Published and not referenced by SPEC:

```
tools/locus_check.py         superseded by token_sweep.py's ghost_test()
tools/cache_analyzer.py      from the 2025-11 corpus run
starmap/rarity_index.json    seed input
starmap/live_mapper.py · mapper_L5_quadrant.py · utils_game.py
apocrypha/                   3.3MB — see the table above
frontier/                    one addendum, "J-Space as a Player-Instantiated
                             Computational Substrate", nothing references it
assets/                      15MB video, images, fonts, screens
discover/assets/             3.5MB calibration and codex video
```

---

## Open questions for the author

RESOLVED:

1. ~~Which `token_sweep`?~~ The 322-line file, `e13d0c70d14a`, now published
   under the canonical name. It writes both `sweep_tokens.tsv` and
   `sweep_neurons.tsv` from one run, and its `ghost_test()` **is** §4.4.
2. ~~Which `control`?~~ The second-iteration run, `d736d51bb79b`, published as
   `results/control.tsv`. Footer `# targets 100  hit_tok 96  hit_str 88 (306s)`.
3. ~~Which `api_server.py`?~~ `starmap/`. `game/starmap/` was stale and is not
   published.
4. ~~Coverage counts?~~ Exported from `MasterHitCounts` to
   `results/master_hit_counts.tsv` `2724a6dbea76`. Reproduces §8.3 exactly
   (1452 / 1339 / 2225 / 2151) and §4.4's `n_unreached` (1620, 847), with
   407,475 hits in every quadrant. Live and backup databases produce a
   byte-identical export, so the coverage table never drifted. The four
   `master_hits_{quadrant}.bin` files (3072 × uint32) ARE published;
   `seed_database.py` rebuilds the whole database from them plus
   `rarity_index.json`. No database is shipped.
5. ~~§8.2.1?~~ **Recovered 2026-08-13**, not re-run. The interrupted
   `wiki103_partial870k` run's `checkpoint.json` holds 172 coverage points to
   860,000 units. §8.2.1 is now tagged MEASURED and the "cannot be reproduced
   from the published bundle" sentence is withdrawn.
6. ~~Source edition and openers?~~ Five openers, verbatim in §8; source is a
   Gutenberg plain-text edition with front matter already removed, `*** START`
   absent and `*** END` surviving at byte 1,191,770. Read off the diff.
9. ~~Where do the map's positions come from?~~ `Math.random()`, uniform in a
   cube of side 70, no projection of θ. §5 stands. That it is *unseeded* is a
   design question, now disclosed in §5.
10. ~~What is `the_sea_sailed.json`?~~ A prototype save state. See the corpus
   table above.
13. ~~Which `the_sea` file is which?~~ Resolved by entry count, tabulated in
   *Navigating the sea files*. The trimmed 7354-entry file is the shipped
   input and stays that way; SPEC's inventory lines are amended to match.
20. ~~The duplicate raw corpus?~~ Deleted 2026-08-14. `precompute_cache.py`'s
   output no longer depends on the working directory.
21. ~~Is `starmap/requirements.txt` installable?~~ Yes, since 2026-08-14. It
   was UTF-16LE, then UTF-8-with-BOM; both break `pip install -r`. Now plain
   ASCII and verified through pip's parser.
14. ~~Are the §4.5 outputs missing?~~ No. `residue.tsv`,
   `reached_imp_r_hard.txt`, `margins.tsv`, `locus_check.py`,
   `data_pipeline.py` and `cache_analyzer.py` were uploaded 2026-08-14 under
   canonical names. Every artifact this manifest names now exists at the path
   it names, with the exception of the two `GAP` rows below.

STILL OPEN:

7. What do the **three** published similarity matrices measure —
   `AXIS_OF_TEMPERATURE_inf`, `AXIS_OF_TEMPERATURE_res`, `AXIS_OF_TSALAL` —
   and does §5 need amending? §5's note names one axis; the tree holds three.
8. `token_sweep.py` contains a `c_fc + F.gelu` fallback that would compute
   **exact GELU, not gelu_new**, if `mlp.act` were ever absent. It never fires
   on GPT-2. Delete it or make it raise — this is the same confusion that put
   x⋆ = −0.7517 in the document until v6. Still present at
   `tools/token_sweep.py:113`, dead behind the `mlp.act` branch at line 110.
11. **§7 is normative in the document and unenforced in the code.** Eight load
   sites, listed in *The pin*. Documented rather than fixed, and the manual
   verification path is stated there. Must be closed before any re-run of §8.4.
12. What is the terminal-character entropy of §8.3's own 407,475-sentence
   corpus? WikiText-103's is 0.910 bits and §8.4's nineteen run 3.198–4.020.
   One pass over the file, no forward pass. It decides whether entropy alone
   predicts the coverage ratio or merely orders corpora. **Requires locating
   that corpus, which has not been done.**
15. §4.1.1 rests on printed console output with no committed result file. One
   redirected run would close it.
16. The 18 hand-written §8 fixture strings are printed in the document but the
   input file that produced them is not published.
17. The intervention reports under `apocrypha/` are published and uncited.
   Decide: cite them in §4 as pilot data with the "causal alteration detected"
   caveat stated, or move them out of the published tree. Silence is the one
   option that reads badly, since a reader who finds them unaccompanied cannot
   tell whether they were overlooked or withheld.
18. `apocrypha/not-our-creation.pdf` is third-party forum content with no
   licence note. Drop it or list it in README's third-party section.
19. ~~`discover/data/datasets.json`?~~ Dead config, kept deliberately. Nothing
   reads it; the atlas loads one precomputed cache. It records a multi-corpus
   browsing direction that was prototyped and not built, and its paths predate
   the cache rename. Listed here so nobody reads it as a live index.
