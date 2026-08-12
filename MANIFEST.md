# MANIFEST — claims to artifacts

Maps every MEASURED claim in SPEC_THREE to the script that produced it and the
file that holds the result. Built from `inventory.md`, root
`znou/code/v1/znou`, 507 files, 2026-08-12. Revised 2026-08-13: §8.2.1
recovered, §8.4 added, caches published.

Status key: `OK` mapped and consistent · `?` mapping uncertain, named below ·
`GAP` no file found · `NEW` file exists, not referenced by the spec

Paths are relative to the repository root as published, which may differ from
the working tree. Sha256 prefixes are the first 12 hex characters.

**Filename convention.** Suffixes like `foo (1).py`, `foo (2).py` are browser
download artifacts, not variants: the plain name is the *first* download and
higher `(n)` is *later*. Confirmed against the v15 sequence — `SPEC_THREE_v15.md`
1,757 lines, `(1)` 1,959, `(2)` and `(3)` 2,034 with identical hashes, i.e. the
last one downloaded twice. Published files must carry one canonical name each,
and the highest `(n)` is normally the one to keep.

---

## Reference stack — §7

| claim | script | evidence | status |
|---|---|---|---|
| pinned artifacts, `sha256(theta)`, file hashes | `tools/pin_stack.py` `a53c526aa7a0` | §7 tables | OK |
| hashing helper | `tools/hasher.py` `375c68ab3132` | — | OK |
| reference `D` | inline in §7 | — | OK — **revision now pinned in the snippet** |
| `transformer_lens` 2.15.0, `nltk` 3.9.2 + punkt | — | pinned in §7 | OK — added 2026-08-13. TransformerLens weight processing means the prototype's `state_dict` does not hash to `sha256(theta)`; §7 says so |
| third environment | `data/caches/*/report.md` | torch 2.4.0+cu121, transformers 4.57.1, RTX A4500, Ubuntu 24.04 | NEW — the §8.4 runs. Not produced under §7's discipline; disclosed in §8.4. A replay of `the_sea` on this stack would widen §8's invariance claim from two environments to three |

## Corpus — §8, B2a

| claim | script | evidence | status |
|---|---|---|---|
| source text | — | `data/moby_dick.txt`, 1,246,660 bytes | OK — added 2026-08-13 so §8's provenance diff is checkable rather than asserted |
| the corpus, untrimmed | — | `data/the_sea_raw.json`, 7394 entries | OK — **rehashed after 2026-08-13.** The published file previously held 7353; §8 describes 7394 and the untrimmed file is now the shipped one |
| the corpus with destinations | `discover/data/precompute_cache.py` | `data/the_sea.json`, 7353 rows | OK — this is the *computed cache*, not the corpus. §7.2 and §8 now say so; the two filenames read the opposite way round in `eve/data/01_archives` and that has caused one error already |
| corpus derivation (apparatus stripped, openers added) | — | diff of `moby_dick.txt` against `the_sea_raw.json` | OK — B2a closed. 5 openers + 7348 Melville = 7353 evaluated, then 40 licence sentences and one malformed `[]`. *Call me Ishmael.* absent, BOM defect, disclosed in §8 |
| prototype Atlas / save state | `discover/js/main.js` | `data/the_sea_sailed.json`, 2.8MB | NEW — a completed autoscan save in the pre-quadrant atlas format. **Not a measurement:** 546 discovered systems against §8's 545, the extra being neuron 2256 reached by hand-typed `as` / `asdf`, and 13 of its 7366 hits are manual. Ships because the cache system needs it; must not be cited |
| cache, imp_r — hash pinned in §8 as `e2e0c5166a5a0518` | — | `data/the_sea_implicit_resonance.json` `e2e0c5166a5a` | OK — **verified against §8** |
| caches, other three quadrants | — | `data/the_sea_{explicit_inference,explicit_resonance,implicit_inference}.json` | OK |

## §4.1.1 — certificate vacuous at ℓ=5

| claim | script | evidence | status |
|---|---|---|---|
| pre-activation range, Θ, `|{j : L_j > x⋆}| = 0` | `tools/unreachable_certificate.py` `00278e7180e5` | printed output only? | ? no result file found |

## §4.3 — one-token census

| claim | script | evidence | status |
|---|---|---|---|
| all 50,257 tokens, four quadrants | `tools/token_sweep.py` `571236e34af0` (317 lines) | `tools/sweep_tokens.tsv` `a96c0a6502d3`, 50,259 lines | ? |
| — | **or** `tools/token_sweep (1).py` `e13d0c70d14a` (322 lines) | — | download-order artifact: `(1)` is the later iteration. Confirm which matches the tsv's columns, keep one, drop the other |

## §4.4 — activation suppression

| claim | script | evidence | status |
|---|---|---|---|
| per-neuron max over single tokens, reached vs unreached | `tools/token_sweep.py`? `tools/locus_check.py` `decd9d62c637`? | `tools/sweep_neurons.tsv` `1cfb11b56559`, 3,074 lines | ? script attribution unconfirmed |
| Cliff's δ (B4, open) | not written | would derive from the same tsv | GAP — open TODO |

## §4.5 / §4.5.1 / §4.5.2 — coordinate ascent

| claim | script | evidence | status |
|---|---|---|---|
| sweep over all of J | `tools/Coordinate_ascent.py` `8b815cf238d1` | `tools/ascent_all.tsv` `a9e01d4957ce`, 3,075 lines | OK — note **spec says `coordinate_ascent.py`, file is `Coordinate_ascent.py`** |
| calibration, 100 systems | same | `tools/control.tsv` `7f0ec6ca6482`, `tools/control2.tsv` `d736d51bb79b`, 103 lines each | ? `control2` is presumably the second run; confirm which §4.5 reports |
| the hard 172 | same | `tools/hard172.txt` `d3a06d9d8e7b`, 172 lines | OK |
| residue detail | same | `tools/residue2.tsv` `2fdfcc782c0c`, 175 lines | ? |
| round-trip-safe token pool (49,905) | same | `tools/safe_tokens.pt` `17b7e8ef6a4f` | OK |
| systems reached, imp_r | same | `tools/reached_imp_r.txt` `1029d8a1fa80`, 545 lines | OK |
| systems reached, hard budget | same | `tools/reached_imp_r_hard.txt` `09e80de0318d`, 187 lines | OK |
| Melville routes, n=545 | — | `tools/melville_routes_imp_r.txt` `f13530e399e0`, 545 lines | OK |
| median Δ by budget class | — | `tools/margins.tsv` `92134b394ab3`, 1,093 lines | ? |

## §8 — cross-environment replay

| claim | script | evidence | status |
|---|---|---|---|
| 7353/7353 agreement | `tools/replay_cache.py` `6ccd7f4fa7ae` | `tools/replay_results.tsv` `cf7fa2ddac60`, 7,354 lines | OK |
| tightest arrival Δ=1e-6 | same | same | OK |
| the 18 below Δ<0.001 | same | `tools/fixtures18.txt` `04155f73da8b`, 18 lines | OK |
| the 211 below Δ<0.01 | same | derived from `replay_results.tsv` | ? no standalone file |
| §8 fixture table (18 hand-written strings) | `--fixtures` flag | `tools/typed_lines.txt` `d09fdcf353bb` (21)? `tools/say_lines.txt` (38)? | ? |

## §8.1 — margin distributions

| claim | script | evidence | status |
|---|---|---|---|
| n=545 first arrivals | — | `tools/margins.tsv` | ? |
| n=7353 all sentences | `tools/replay_cache.py` | `tools/replay_results.tsv` | OK |

## §8.2 / §8.2.1 / §8.3 — coverage

| claim | script | evidence | status |
|---|---|---|---|
| imp_r 545 vs imp_i 57, terminal-char table | — | the four `data/the_sea_*.json` caches | ? script not identified |
| WikiText-103, `wiki103test_511`, 40k, four quadrants | lost | lost | **CLOSED, superseded.** The original run took its drive with it and is not in any backup. §8.2.1 no longer rests on it: its figures are retired and replaced from the run below. The lost data stays lost; nothing now depends on it. |
| WikiText-103 coverage curve, 172 points to 860,000 units | `data_pipeline.py` | `data/caches/wiki103_partial870k_2025-11-05/checkpoint.json`, 42KB | OK — **the recovery.** The run was interrupted before writing `report.md`, but `coverage_log` is appended every 5,000 units and flushed every 20,000, so the curve survived *because* the run did not finish. Resolves Q6 to composition: at 400,000 units imp_r/imp_i = 1.79 against §8.3's 0.65 at 407,475, and the curves have not crossed by 860,000. |
| corpus coverage, 407,475 sentences | `tools/export_hits.py` | `results/master_hit_counts.tsv` `2724a6dbea76`; `starmap/master_hits_{quadrant}.bin`, 12,288 bytes each = 3072 × uint32 | OK — confirmed. Reproduces §8.3 exactly (1452 / 1339 / 2225 / 2151) and a freshly seeded database exports byte-identically |
| nineteen book corpora, §8.4 | `data_pipeline.py` | `data/caches/*_2025-11-05.zip` | OK — see §8.4 below |

## §8.4 — corpus composition, nineteen books

Published under `data/caches/_extracted/`, one directory per corpus, all
2025-11-05. Held locally at `eve/data/01_archives` as `.zip`, **not** at
`discover/data/caches/` — the stale path is why the publish step missed them
until 2026-08-13. The extracted form is published rather than the archives so
that the files are readable and diffable in place.

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

| claim | script | evidence | status |
|---|---|---|---|
| per-corpus coverage, four quadrants | `data_pipeline.py` | `<corpus>_{quadrant}.bin`, 3072 × uint32, plus `report.md` | OK — `prepare_caches.py` recomputes coverage from each `.bin` and checks it against that corpus's own report |
| per-unit destinations | same | `<corpus>.jsonl` | OK for the books. **Withheld for wiki103** — WikiText-103 is CC BY-SA and does not sit inside this repository's Apache-2.0 licence. `sample.jsonl`, the checkpoint hit arrays and the coverage log do ship, and those are what §8.2.1 rests on |
| wiki103 hit arrays | same | `checkpoint_hits_{quadrant}.bin` | OK — note these are the *checkpoint* arrays, not `<corpus>_{quadrant}.bin`. The run was interrupted, so finalisation never wrote the latter. The last checkpoint is at 860,000 units |
| staging and verification | `data/caches/prepare_caches.py` | `_reports/summary.tsv`, `_reports/CACHES.md` | OK |
| units are line fragments, not sentences | `data_pipeline.py:261-267` | punkt applied per physical line | **DEFECT, DISCLOSED in §8.4.** Not comparable with §8.2, §8.2.1 or §8.3. The fix is to unwrap paragraphs before splitting and checkpoint on paragraph index rather than line index — resume must survive |
| under-3-words filter | `data_pipeline.py:159` | report count minus jsonl rows | Discards 38.3% of *A Doll's House* and 26.7% of *Shakespeare*. Both excluded from any regression |
| French collapse | — | `du-cote-de-chez-swann-FR` vs `swanns-way-EN` | OK — 3.7–6.0× fewer destinations at 0.92× the units. Reported in §8.4 as a property of θ and its training distribution |

## §5 — map and lenses

| claim | script | evidence | status |
|---|---|---|---|
| λ_raw, λ_grav, λ_norm, λ_orr | `discover/js/main.js` `3b7818d5bab2`, 1,293 lines | — | OK |
| four-quadrant atlas, save migration | same, `discover/js/stateManager.js` | — | OK |
| `b_j ~ Unif`, unseeded, per client | `main.js:302-308`, `distributionSize = 70` | verified against `the_sea_sailed.json`: x spans −34.998 to 34.986 | OK — no projection of θ, so §5's "no metric on J" stands. That the draw is *unseeded* is now disclosed in §5 as an open design question: every operator holds a different sky |
| rarity index | `starmap/seed_database.py` | `starmap/rarity_index.json` `e51f517ade1c` | NEW — not referenced in spec |

## §6 — reward, exchange

| claim | script | evidence | status |
|---|---|---|---|
| exchange / claim / leaderboard | `starmap/api_server.py` `b3b49354befe`, `starmap/market_ticker.py` | `exchange/` | OK |
| — | **also** `game/starmap/api_server.py` | — | **AMBIGUOUS: two trees. Identify the live one.** |
| live instance data | — | `starmap/znou_exchange.db`, `starmap/znou_exchange_backup.db`, `tools/znou_exchange.db` — three files, three hashes | **DO NOT PUBLISH without inspection — may contain player data** |

## Running the thing

| purpose | file | status |
|---|---|---|
| local launcher | `starmap/local_dev.py` `e317ffde5bda` | OK |
| server | `starmap/api_server.py` | OK |
| experiment runner (loads GPT-2) | `starmap/experiment_runner.py` `27139924f1e1`, 461 lines | OK |
| config | `starmap/config.py` | OK |
| deps | `starmap/requirements.txt` `c358cc977f8f` | **7 public-IP matches — inspect before publishing** |
| db init | `starmap/initialize_database.py`, `verify_db.py` | OK |

---

## NEW — present in the tree, absent from the spec

These are not gaps in the manifest; they are work the document does not
mention, listed so it can be either cited or explicitly set aside.

**Q10 confirmed untouched.** `starmap/test_intervention.py` and the two
`intervention_report_tsalal_*.txt` files belong to an abandoned line of work
that is out of scope for this document. §4's statement that ℛ* has no results
in it stands as written. Nothing tsalal-shaped is published or cited.

```
starmap/constellation_mapper.py · constellation_mapper2.py
starmap/const_mapper_inference.py · const_mapper_resonance.py
starmap/constellations/1st run/AXIS_OF_TEMPERATURE_*/
starmap/constellations/1st run/Physical_Cold/
  → *_similarity_matrix.txt, *_heatmap.png, *_fingerprint.png, *_report.txt
starmap/verify_landmarks.py · test_cold.py · test_cold2.py
starmap/test_lensing.py · test_lensing2.py
```
§5 claims no metric on J is asserted by any lens, and names λ_edit as the only
candidate. A similarity matrix over neurons is a metric-shaped object, and the
temperature axis is the same family as §8's `it was cold` fixtures. Resolve:
say what these measure, or amend §5. Low priority — but it is the one place
the archive and the document disagree about what has been tried.

---

## Not for publication

```
old/ · oldsite/ · game/starmap/old/ · starmap/old/    legacy trees
assets/sfx/ · assets/vid/                             ~21MB media
tools/GPTCritique.pdf · GPTCritique2.pdf              11MB peer review
tools/*feedback*.txt · Critique from Gemini.txt       review correspondence
tools/SPEC*.md × 20 · TODO*.md × 8                    drafts; ship one of each
znou_exchange*.db × 3                                 inspect first
starmap/nohup.out                                     stray log
```

---

## Open questions for the author

RESOLVED:

1. ~~Which `token_sweep`?~~ The 322-line file (`token_sweep (1).py`). It writes
   both `sweep_tokens.tsv` and `sweep_neurons.tsv` from one run, and its
   `ghost_test()` **is** §4.4 — so `locus_check.py` is not needed.
2. ~~Which `control`?~~ `control2.tsv`. Its footer reads
   `# targets 100  hit_tok 96  hit_str 88 (306s)`, which is §4.5's second
   budget row and the 88% denominator.
3. ~~Which `api_server.py`?~~ `starmap/`. `local_dev.py` computes SITE_ROOT one
   level above itself and expects `index.html`, `discover/`, `exchange/` there,
   which is only true of `starmap/`. `game/starmap/` is a stale duplicate.
4. ~~Coverage counts?~~ Exported from `MasterHitCounts` to
   `results/master_hit_counts.tsv` `2724a6dbea76…`. Reproduces §8.3 exactly
   (1452 / 1339 / 2225 / 2151) and §4.4's `n_unreached` (1620, 847), with
   407,475 hits in every quadrant. Live and backup databases produce a
   byte-identical export, so the coverage table never drifted.
   The four `master_hits_{quadrant}.bin` files (3072 × uint32) ARE published:
   `seed_database.py` rebuilds the whole database from them plus
   `rarity_index.json`. Confirmed — a freshly seeded database exports the
   identical table, same sha256. No database is shipped.
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

STILL OPEN:

7. What does `AXIS_OF_TEMPERATURE` measure, and does §5 need amending?
8. `token_sweep.py` contains a `c_fc + F.gelu` fallback that would compute
   **exact GELU, not gelu_new**, if `mlp.act` were ever absent. It never fires
   on GPT-2. Delete it or make it raise before publishing — this is the same
   confusion that put x⋆ = −0.7517 in the document until v6. Still present at
   `tools/token_sweep.py:113`, dead behind the `mlp.act` branch at line 110.
11. **§7 is normative in the document and unenforced in the code.** Eight load
   sites call `from_pretrained` with no `revision`: the §7 snippet (now fixed),
   `starmap/experiment_runner.py:25`, `tools/coordinate_ascent.py:48-49`,
   `tools/replay_cache.py:36-37`, `tools/token_sweep.py:102-103`,
   `tools/pin_stack.py:77`, `tools/unreachable_certificate.py:101`,
   `apocrypha/cold/coldchat.py:33-34`. `pin_stack.py` is the circular one — it
   computes `sha256(theta)` from an unpinned load, so the tool that certifies
   the pin does not use it. `experiment_runner.py` needs the HF model loaded
   pinned and passed as `hf_model=`, since `HookedTransformer.from_pretrained`
   takes no revision. **This must land before any re-run of §8.4**, or nineteen
   fresh caches are generated against an unpinned model.
12. What is the terminal-character entropy of §8.3's own 407,475-sentence
   corpus? WikiText-103's is 0.910 bits and §8.4's nineteen run 3.198–4.020.
   One pass over the file, no forward pass. It decides whether entropy alone
   predicts the coverage ratio or merely orders corpora. **Requires locating
   that corpus, which has not been done.**
