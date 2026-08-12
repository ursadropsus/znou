# MANIFEST — claims to artifacts

Maps every MEASURED claim in SPEC_THREE to the script that produced it and the
file that holds the result. Built from `inventory.md`, root
`znou/code/v1/znou`, 507 files, 2026-08-12.

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
| reference `D` | inline in §7 | — | OK |

## Corpus — §8, B2a

| claim | script | evidence | status |
|---|---|---|---|
| `the_sea` prepared corpus | — | `data/the_sea.json` `a5cdf1bdfe75`, 29,414 lines | OK |
| pre-preparation text | — | `data/the_sea_raw.json` `12625decc7b6`, 7,357 lines | OK |
| corpus derivation (apparatus stripped, openers added) | — | recoverable as a diff of the two above | ? |
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
| WikiText-103, `wiki103test_511`, 40k, four quadrants | lost | lost | **GAP — DATA LOST.** The run wrote one file per entry, several hundred thousand files, and took the drive with it. Not in any backup. §8.2.1's figures are read off a plot whose source no longer exists, and are still `~` and TO PIN. Either mark §8.2.1 unreproducible or re-run: 40k × 4 quadrants ≈ 160k forward passes, one TSV out. |
| corpus coverage, 407,475 sentences | — | `starmap/master_hits_{imp_r,imp_i,exp_r,exp_i}.bin`, 12,288 bytes each = 3072 × int32 | ? plausible but unconfirmed |

## §5 — map and lenses

| claim | script | evidence | status |
|---|---|---|---|
| λ_raw, λ_grav, λ_norm, λ_orr | `discover/js/main.js` `3b7818d5bab2`, 1,293 lines | — | OK |
| four-quadrant atlas, save migration | same, `discover/js/stateManager.js` | — | OK |
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

```
discover/data/caches/  — 19 full-book caches, all 2025-11-05
  Shakespeare · Ulysses · War and Peace · Swann's Way · Budding Grove ·
  Crime and Punishment · Moby-Dick (full) · Leviathan · Machiavelli ·
  Metamorphosis · Poe · The Yellow Wallpaper · Tractatus · Zarathustra ·
  Alice · Book of the Dead · A Doll's House · The King in Yellow
```
Q6's open half needs one corpus at two scales with terminal-token entropy
reported at each; §8.2.1's confound is composition versus crossover. Nineteen
cached corpora of differing terminal-token character are already on disk.
Likely answerable without GPU time.

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
5. ~~§8.2.1?~~ Marked SOURCE DATA LOST in the document. Re-run optional.

STILL OPEN:

6. Source edition of the text, and how many thematic openers were added —
   needed for §8's provenance paragraph. Readable off the raw/prepared diff.
7. What does `AXIS_OF_TEMPERATURE` measure, and does §5 need amending?
8. `token_sweep.py` contains a `c_fc + F.gelu` fallback that would compute
   **exact GELU, not gelu_new**, if `mlp.act` were ever absent. It never fires
   on GPT-2. Delete it or make it raise before publishing — this is the same
   confusion that put x⋆ = −0.7517 in the document until v6.
