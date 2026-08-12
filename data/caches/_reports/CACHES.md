# Corpus caches

Produced by `data_pipeline.py` on 2025-11-05. **The unit is not a
sentence.** The pipeline applied NLTK punkt to each physical line of
hard-wrapped source text, so units are line-bounded fragments; most
end mid-clause on an ordinary word. Coverage figures here are
therefore **not comparable with SPEC 8.2, 8.2.1 or 8.3**, whose corpus
was prepared with paragraphs unwrapped first.

These runs also predate SPEC 7's configuration discipline: no
`revision` pin, no TF32 flags, no `use_deterministic_algorithms`.
The function itself is unchanged — layer 5, `mlp.hook_post`, same four
quadrants — but the stack was torch 2.4.0+cu121 on an RTX A4500 under
Ubuntu 24.04, not the reference stack.

`units_reported` counts every fragment the pipeline saw;
`jsonl_rows` counts those that survived its under-3-words filter.

| corpus | units | rows | coverage imp_r | imp_i | final-char entropy |
|---|---:|---:|---:|---:|---:|
| alice-in-wonderland_2025-11-05 | 3381 | 2969 | 328 | 440 | 3.9971 |
| book-of-the-dead_2025-11-05 | 19700 | 14585 | 627 | 852 | 3.7379 |
| within-a-budding-grove_2025-11-05 | 25028 | 22456 | 875 | 1261 | 3.9354 |
| crime-and-punishment_2025-11-05 | 27739 | 23407 | 780 | 1132 | 3.7263 |
| a-dolls-house_2025-11-05 | 5771 | 3560 | 395 | 427 | 3.1984 |
| du-cote-de-chez-swann-FR_2025-11-05 | 18692 | 16966 | 228 | 210 | 3.6356 |
| the-king-in-yellow_2025-11-05 | 9394 | 8081 | 557 | 736 | 3.6425 |
| leviathan_2025-11-05 | 22920 | 20450 | 713 | 1034 | 3.9148 |
| the-prince_2025-11-05 | 5567 | 5014 | 510 | 652 | 3.8945 |
| the-metamorphosis_2025-11-05 | 2758 | 2447 | 364 | 474 | 3.7579 |
| moby-dick_2025-11-05 | 25186 | 21713 | 812 | 1174 | 3.9515 |
| poe-collected_2025-11-05 | 12478 | 10879 | 684 | 918 | 3.8704 |
| shakespeare-complete_2025-11-05 | 175906 | 128885 | 1121 | 1556 | 3.3023 |
| swanns-way-EN_2025-11-05 | 20290 | 18516 | 851 | 1159 | 4.0204 |
| the-yellow-wallpaper_2025-11-05 | 1082 | 944 | 214 | 217 | 3.7891 |
| tractatus_2025-11-05 | 2650 | 2277 | 262 | 336 | 3.4094 |
| ulysses_2025-11-05 | 42212 | 32368 | 909 | 1241 | 3.329 |
| war-and-peace_2025-11-05 | 68773 | 59735 | 1025 | 1564 | 3.6785 |
| wiki103_full_2025-11-05 |  | 866109 | 1501 | 1042 | 0.9105 |
| thus-spoke-zarathustra_2025-11-05 | 14738 | 12223 | 632 | 838 | 4.0119 |
