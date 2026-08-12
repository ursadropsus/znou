# J-SPACE — FORMAL SPEC

## Prologue — pilfered fragments from an unknown archive

*In-universe. Fiction. The specification begins at §0.*

Here we come to that fact that shook the Third Jove Empire to its core. This is a fact that took many decades of exploration and investigation to establish. A fact that was not readily accepted by all but a fact, like all facts, that in the end we had to face. Anoikis is artificial.

There are at least two different J-spaces right now. The one the Jove left behind that I named Anoikis and the one Anthropic described inside a language model this year. Theirs is a set of directions in the residual stream, surfaced by a Jacobian lens, sitting in the middle of the network. This is the flat floor of an MLP five blocks in. I am taking the name because it fits narratively, not so much technically. In that sense, this proposal puts the first J-space inside the second. Where it belongs.

---

## Provenance of the name

*End of fiction. Everything from here is literal.*

The borrowed term refers to a real result, and the paraphrase above is loose
enough to mislead. Precisely:

```
paper     "Verbalizable Representations Form a Global Workspace in
           Language Models"
authors    Gurnee, Sofroniew, Lindsey et al., Anthropic interpretability team
published  Transformer Circuits, 6 July 2026
           transformer-circuits.pub/2026/workspace/index.html
tool       Jacobian lens (J-lens): apply the averaged input–output Jacobian,
           then final layer norm and unembedding. Reads, for a residual-stream
           activation, its first-order effect on the probability of each
           vocabulary item at some later position.
object     J-space: the space spanned by sparse linear combinations of the
           vectors J·W_U — a low-dimensional subspace of the residual stream,
           held to approximate a "cognitive space" of intermediate variables.
           It spans a band of mid-stack layers, not a single layer.
status     open-source implementation; J-lenses for open-weights models
           published on Neuronpedia. Independently replicated on Qwen 3.6 27B.
```

**What this document borrows is the name and nothing else.** The two objects
are not the same kind of thing, and one difference is worth stating because
it is nearly an inversion:

```
J-space (Anthropic)   directions in the residual stream, individuated BY
                      their effect on output logits, spanning layers.

J (this document)     the MLP activation basis at one layer, indexed by
                      neuron. By I4 the logits are never formed at all.
```

The Anthropic construction defines its coordinates by exactly the quantity
this specification refuses to compute. That is not a defect of either, but it
does mean no result about one transfers to the other, and a reader should not
infer that Anthropic found something equivalent to a GPT-2 neuron index.

**Q10 is where the two could actually meet.** Q10 asks whether a destination
predicts causal downstream effect — which is the question J-lens was built to
answer. J-lens is cheap, open, and reported to work on open-weights models,
GPT-2 among the candidates. Running it against `J` is a concrete route to
ℛ*, not an analogy.

**No affiliation is claimed or implied.** This document is not a Fenris
output, not a DeepMind output, and not an Anthropic output. It is a proposal
by its author, written in an environment those organisations happen to share.
Keep the four distinct: *this document's work · the author's proposal ·
Fenris technology · published interpretability research.*

## Provenance of the question

The name was borrowed late. The question is older, and has a source:

```
paper      "Language Models Represent Space and Time"
authors     Wes Gurnee, Max Tegmark (MIT)
published   arXiv:2310.02207, October 2023; ICLR 2024
result      linear representations of space and time, across scales and
            robust to prompting variation, recovered from Llama-2
            activations over three spatial datasets (world, US, NYC places)
            and three temporal ones (historical figures, artworks, news
            headlines). Individual "space neurons" and "time neurons" are
            identified, reliably encoding spatial and temporal coordinates.
```

What that paper did to the author was specific and is worth recording
plainly: it made *poking a neuron by hand* look like something a person
outside a lab could do. Geography surviving inside a next-token predictor,
recoverable, sitting at an address. The prototype in §7 is roughly two years
downstream of that question, asked badly and then repeatedly.

**Almost nothing transfers, and the difference is instructive.** Gurnee and
Tegmark individuate their neurons *by regression against known external
coordinates* — a supervised probe with ground truth in hand, asking which
unit tracks latitude. `D` individuates neurons by which one wins an argmax,
with no ground truth anywhere in the construction and nothing to regress
against. Their result is evidence that some neurons in some models admit an
interpretation; it is not evidence about ℛ* at GPT-2 ℓ=5, and Q10 is not a
replication of it. What it does supply is a prior that the functional sense
of meaning (§4.6) is sometimes satisfiable, which is a reason to run Q10
rather than a partial answer to it.

**Coincidence, recorded as such.** Gurnee's name appears on both that paper
and the one this document borrows *J-space* from — the second found by
searching for a name after the object already existed. That is the projected
sense of meaning (§4.6), it is not evidence of anything, and it is noted
because it is true rather than because it signifies.

## Claim tags

Every claim-bearing section carries one of four tags. Where a section mixes
kinds, the loudest sentence is tagged inline.

```
MEASURED        a number came out of a named script on the §7 stack
DERIVED         follows from a MEASURED result plus stated assumptions
INTERPRETATION  what the author takes the result to mean
PROPOSAL        design, not result; nothing is claimed to follow
```

The tags say what kind of support a claim has. They do not say what the word
*meaning* is doing in it, and that has been a separate source of overclaim.
§4.6 disambiguates it; any sentence in this document using the word should
be readable against that section.

---

## Adjacent prior art — Midjourney `--sref` — INTERPRETATION

**Nothing in this document depends on this section.** It is recorded because
the nearest thing to the activity this specification proposes already exists,
was played by the author as though it were a game, and is not usually
described as one. No equivalence is claimed and no result transfers in either
direction.

Midjourney's `--sref <n>` takes an integer and applies a visual style to a
generation. The April 2024 announcement of `--sref random` described it as a
test of new algorithms for *exploring the latent space of visual styles*, and
as a tool to play with pending more advanced ones. Codes are found rather
than written: the documented route to a new one is to ask for a random draw
and keep what comes back.

```
DISCLOSURE   the model is closed and hosted. Everything below is read off
             public documentation, an official announcement, community
             practice and the author's own use. None of it is a statement
             about Midjourney's internals, which are not inspectable from
             here and are not claimed to be.
```

What appears structurally similar:

```
addressing   an integer selects something the model already contains, and
             the correspondence was not hand-assigned per code so far as is
             visible from outside. What a given code looks like is found by
             rendering it.

adjacency    the index appears to carry no neighbourhood structure — code n
             and code n+1 are not reported to resemble each other, which is
             §5's claim about j=10 and j=11 arrived at independently on
             someone else's model. Stated carefully: the author did not
             encounter neighbourhood structure in v6, and no systematic
             test either way is known to have been published. Descriptions
             of nearby codes varying smoothly appear to be inferred from
             the phrase "latent space" rather than measured, and the code
             range — 0 to 4,294,967,295, i.e. 2³² — is the size of a seed
             rather than evidence of a coordinate. Suggestive, not settled.

coverage     community libraries hold thousands of codes against a space of
             4.29e9 and are curated, tagged, traded and in some cases sold.
             That is an atlas in the sense of §5 — a record of what has been
             found, held socially rather than by the operator — reached
             without anyone specifying one.

landmarks    individual codes accumulate names, followings and commentary.
             The clearest instance the author is aware of is `1894659223`,
             which a public search returns: a YouTube video titled *We need
             to talk about 1894659223*, an entry in a community code dump,
             catalogue pages in Japanese and Chinese, classification under
             a "gothic" heading, and a stranger describing themselves as
             obsessed with it and writing that there is a whole world of
             horror on the other side of the numbers.

             This is the indexical sense of meaning (§4.6) forming on an
             address space that has none of the other kinds. Nobody
             involved knew what the code *was*; there is nothing to know.
             It became a place because it was distinctive and because it
             had a name that could be passed on — which appears to be the
             entire recipe, and requires neither an economy nor an
             interpretability result. Observation, not measurement: no
             count of how often this happens, or of what distinguishes the
             codes it happens to, is offered here.
```

What is inverted, and the inversions are the load-bearing part:

```
direction    `--sref` takes a number and returns an aesthetic; `D` takes a
             string and returns a number. The authored, contested object
             here is the input, not the output.

verification by I6, θ is public and `D` is fifteen lines, so any operator can
             confirm a destination offline and exactly. A style code is
             redeemable only inside a hosted model. §6.1's parity argument
             has no analogue there.

pinning      users report that codes discovered under one model version do
             not reliably reproduce under the next, and Midjourney's own
             documentation advises testing before relying on them. Whether
             or not that is precisely characterised, it is the failure mode
             §7 exists to exclude: an address space that re-means itself
             under the people who mapped it. Midjourney is not offered here
             as a cautionary tale about Midjourney — a hosted product has
             every reason to improve its model — but as the worked example
             of what an unpinned address space costs a population that has
             invested in mapping it.
```

**The author's use was the game-shaped one.** Locate a code with a strongly
distinctive default output — the ones that are easy to tell apart by eye, a
particular palette or medium or recurring motif — and then try to knock it
off that default by semantic pressure from the prompt alone. A contest
between a fixed style address and a variable text input, scored by looking.
This document is the version in which the text is the only channel and the
score is an integer.

**What this does and does not license.** That codes were mined, indexed,
named and resold is a real precedent for the concern in §6 and E2: people
chart an address space nobody designed as a game, and then commoditise the
addresses. What it is *not* is a case of mining without reward. **The reward
was the output.** A code that locks a distinctive style down consistently is
immediately usable and immediately saleable, and the aesthetic returned is
sometimes worth having on its own terms — the author's case study was a
single code producing a coherent figurative horror idiom, somewhere in the
lineage of Chung, Giger and McFarlane with oddities of its own, and
occasionally further past the platform's own taste boundaries than the
platform would likely have chosen. A decade earlier a human artist with that
idiom would have had a career on it.

That is the disanalogy that matters, and it runs the other way from the
similarities above:

```
`--sref` returns an image. It has value on arrival, so no economy has to be
designed for mining to pay.

`D` returns an integer. It has no value on arrival. Whatever a destination is
worth is assigned by the world, not by J — which is the whole of §6, and the
reason Scheme A and Scheme B are a real choice rather than a formality.
```

So the precedent supports the shape of E2 — addresses that are stable,
findable and worth something to somebody get mined and sold — without
supplying the mechanism, since the thing that made them worth something there
is absent here by construction. It does not show what happens under Scheme B,
does not bear on whether Δ is optimisable (Q13), and cannot substitute for
either measurement. Read it as a reason to run those experiments rather than
as a partial answer to them.

---

## 0. Objects

```
V     = GPT-2 BPE vocabulary,  |V| = 50257
M     = GPT-2 small, parameters θ frozen (2019 release)
d     = 768        (d_model)
m     = 3072       (d_mlp)
n_ctx = 1024
ℓ     = 5          (block index, 0-based)
J     = {0, …, m−1}          the system set,  |J| = 3072
Σ     = Unicode
Σ*_b  = { s ∈ Σ* : |τ(s)| + β_bos ≤ n_ctx }     the bounded domain
```

Block ℓ, pre-LN architecture:

```
x        ∈ ℝ^{T×d}           residual stream into MLP
x̂        = γ ⊙ (x − μ)/√(σ²+ε) + β        (ln_2;  γ, β ∈ ℝ^d)
a        = x̂ W + b                        (W ∈ ℝ^{d×m},  b ∈ ℝ^m)
A        = g(a)                           g = gelu_new,  A ∈ ℝ^{T×m}
```

`A` is the only quantity read. Nothing downstream of it is computed.
θ is a frozen mass grave of unattributed authorship, unknowable people forever in stasis.
`A` is their vigil and return.

### 0.1 Why ℓ = 5

ℓ is a **parameter of this specification, not a discovery.** Every measured
result in this document is a result at ℓ=5 and nothing here establishes that
ℓ=5 is distinguished. The honest account of how it was chosen:

```
PROVENANCE       the working prototype is built at ℓ=5. Every fixture, atlas
                 and coverage figure in §4, §8 and the live region comes from
                 that instance. ℓ=5 is where the data is because ℓ=5 is where
                 the game is.

PRIOR EXPLORATION  qualitative sweeps across GPT-2 small layers over the
                 preceding year, not recorded here in a form that can be
                 cited. Extremes were checked and rejected as unplayable:
                 at ℓ=0 the destination is close to constant — one neuron
                 takes nearly everything, the ℓ=5 explicit-probe degeneracy
                 (E1) generalised to the whole layer — and late layers
                 collapse toward next-token structure. Reported as
                 recollection, not measurement. §0.2 is the test.

INTERPRETATION   mid-stack is where lexical form has resolved into something
                 semantic but has not yet been committed to a next-token
                 decision. A layer pregnant with possibility. This is a
                 common reading of transformer depth and it is the reason
                 ℓ=5 *feels* right; it is not evidence, and no result here
                 depends on it.
```

**Nothing in the construction is specific to ℓ=5 or to GPT-2.** `D`, the
quadrants, ℛ, Δ and the atlas are defined for any (model, layer) pair with an
MLP. Changing ℓ changes every number and no definition. A reader who thinks
ℓ=7 is the interesting layer can rerun the document.

### 0.2 Layer selection — PROPOSED, NOT RUN

**This is instrument selection, not hypothesis testing.** No claim is being
tested. Twelve layers are being inspected so that one can be chosen to build
a game in, on grounds of playability, by the person building the game. The
curves are a catalogue, and reading the catalogue before choosing is the
entire point of producing it.

```
corpus     a fixed thematic corpus, held constant across layers
           (candidate: collated EVE Frontier Chronicles — in-universe
           appropriate, stylistically homogeneous, single house voice)
sweep      ℓ ∈ {0 … 11} × 4 quadrants, coverage curve |ℛ̂| vs |C|
report     final |ℛ̂| · top-3 share · visit-distribution entropy ·
           marginal discovery rate at end of corpus (new systems per 1k
           sentences) · one-token census argmax dominance per layer
```

**What a good layer looks like.** A long fat tail full of possibilities. Not
maximum coverage and emphatically not maximum entropy — a perfectly uniform
layer is the worst case, not the best. If every system is equally likely then
nothing is rare, nothing is common, there are no landmarks, no accidental
first contact and no economy: the layer is a hash with 3072 buckets. What is
wanted is structure.

```
busy core        systems reached incidentally, by ordinary language, often.
                 These are the landmarks and the onboarding.
long tail        systems that take work. These are the discoveries.
no ceiling       coverage still climbing at the end of the corpus. A layer
                 that saturates early is finished as a game whatever its
                 ceiling was.
not degenerate   a handful of neurons must not take nearly everything
                 (recollected behaviour at ℓ=0, and the ℓ=5 explicit
                 one-token probe, E1).
```

Heavy-tailed, in other words, which is what ℓ=5 imp_r already looks like. No
claim of Zipf scaling is made — nothing here has fitted an exponent, and
"roughly Zipf" in an earlier draft was a figure of speech doing the work of a
statistic. Judged by eye against the curves; the quantities worth plotting
alongside coverage are top-k share, visit-distribution entropy, and marginal
discovery rate at corpus end.

A layer is rejected only if it fails in **every** quadrant. ℓ=5 survives on
imp_r regardless of imp_i's 95% top-3 share (§8.2); the quadrants are
different instruments at the same depth.

**One constraint on how the result may later be used.** Because the sweep
selects on playability, it cannot afterwards be cited as evidence that ℓ=5 is
objectively distinguished, or that §0.1's semantic reading of mid-stack depth
is correct. Selection and confirmation cannot be the same run. If a claim of
that kind is ever wanted, it needs a separate measurement with the layer
already fixed.

**Caveat on the corpus.** A single-voice corpus is stylistically homogeneous
at its sentence endings, which by §8.2 suppresses ρ=I coverage specifically.
That is fine for comparing layers to each other under a constant corpus; it
makes the absolute figures non-comparable with §8.3's 407k-sentence numbers.
Report terminal-token entropy alongside coverage so the two can be related.

---

## 1. Destination function — DEFINITION

```
τ : Σ* → V*                          BPE encode
β_bos ∈ {0,1}                        prepend <|endoftext|>
ρ ∈ {R, I}                           readout

T = |τ(s)| + β_bos                   MUST satisfy T ≤ n_ctx, else D undefined
A(s) ∈ ℝ^{T×m}

D_R(s) = min argmax_{j∈J}  max_{1≤t≤T} A(s)[t, j]    "Resonance"
D_I(s) = min argmax_{j∈J}  A(s)[T, j]                "Inference"

D_{β,ρ} : Σ*_b → J                   4 variants
```

The MUST is load-bearing. Without it an implementation silently becomes a
truncating function and is no longer `D`.

**Tie rule.** `argmax` is total only once ties are resolved. Ties resolve to
the **lowest index j**. No exact tie has been observed — the smallest margin
in 545 arrivals is 1e-4 (§8.1) — so the rule is precautionary, but it is
normative and §7 asserts it rather than inheriting it from a library. A
backend that resolves ties differently computes a different function, not the
same function differently.

**What a destination is, normatively.** `D(s)` is **the index of the winning
neuron** — nothing more. Until Q10 runs it is not a latent location, not a
represented concept, and not a feature. Every occurrence of *system*, *locus*,
*destination*, *region* and *space* in this document is shorthand for the
winning index and carries no further commitment.

**Index convention.** Positions are 1-based in §1 and §8: position 1 is the
first token of the input, which under β_bos=1 is `<|endoftext|>`. §4.3 and §7
are 0-based where they say so. The two conventions are not mixed within a
block.

```
quadrant   β_bos   ρ   reads
imp_r        1     R   every position, BOS prepended
imp_i        1     I   final position only, BOS prepended
exp_r        0     R   every position, no BOS      ] degenerate at |s|=1,
exp_i        0     I   final position, no BOS      ]  see E1
```

```
IMPLICIT ⇔ β_bos = 1        EXPLICIT ⇔ β_bos = 0
RESONANCE ⇔ ρ = R           INFERENCE ⇔ ρ = I
```

### 1.1 Monotonicity of ρ=R

M is causal: A(s)[t, :] depends only on positions ≤ t. So for any extension
`s′ = s · u`, positions 1..T are bit-identical between the two inputs, and

```
    max_{t≤T′} A(s′)[t,j]  ≥  max_{t≤T} A(s)[t,j]      ∀j
```

**Per-neuron maxima are non-decreasing under extension.** The running max is a
ratchet: no suffix can lower any neuron's peak. D_R(s′) ≠ D_R(s) only if some
neuron in the new positions strictly outbids the standing peak, **or equals it
while holding a lower index** — the second mode exists solely because of the
§1 tie rule and has never been observed.

Consequences:

```
· D_R is not a summary of a text. It is the loudest moment in it.
· A destination set early is locked unless outbid outright. Prefixes are
  sticky in proportion to how loud they were.
· E2 (§4.3) is this property read at one threshold: 2256 wins iff nothing
  anywhere beats the BOS value 2.3609.
· Long strings sample more positions, so they more often set a high bar
  somewhere, so they concentrate on high-ceiling neurons. This is the
  mechanism behind the length confound in Q9, and behind 90.6% of single
  tokens landing on 1888.
```

ρ=I has no such property. The readout is the final position, which every
appended token replaces outright — fully overwritable, never ratcheted. The
two readouts are different functions of a text, not two views of one: the
loudest moment against the last word. §8.2 measures what that costs.

Evidence: the exact margin collisions in §8.

---

## 2. Invariants — DERIVED

```
I1  D is pure: D(s) depends on (s, θ, ℓ, β_bos, ρ) and nothing else.
I2  D is deterministic: no sampling, no temperature, no RNG.
I3  D is non-injective: |Σ*_b| ≫ |J|; fibers D⁻¹(j) are unlistable in practice.
I4  D is generative-free: g(a) is read at layer ℓ; logits are never formed.
I5  D is deterministic relative to the pinned stack of §7. MEASURED, §8:
    **D was invariant across the two tested environments** — 7353/7353
    sentences, two machines, months apart, down to a margin of 1e-6. This is
    a result about those two environments, not a general portability
    theorem; "portable across strict-fp32 stacks" is not established and
    broadens only as arms are added (§7.1).
    Δ (§8.1) is a *sensitivity margin*, not a portability error term. A
    preserved argmax bounds the differential perturbation between the two
    competing neurons, not the absolute perturbation of A.
    Residual risk is configuration rather than hardware: TF32 and reduced
    precision are the known candidates, §7 disables them, §7.1 arm 2
    measures whether it matters.
    Server-side evaluation is an authority and anti-tamper decision, not a
    numerical necessity.
I6  θ is public ⇒ D is offline-computable by anyone, up to I5. (§6 depends
    on this.)
I7  D cannot distinguish authorship. It sees a string. It has no other input.
    This is a property of the interface, not a claim that authorship is
    unknowable by other means.
```

Datamining is permitted and does not help. θ is a download, `D` is fifteen
lines, and the search in §4.5 took an evening on a laptop. None of it tells
you why `it was colder` leaves the system `it was cold` arrives at. Nothing
here was designed.

---

## 3. Domain cardinality — DERIVED

`Σ*_b` is a set of **strings**, not of token sequences. The two are not the
same size and the distinction is the subject of Q12, so §3 states a bound
rather than an equality.

```
Token-sequence space, length ≤ 1023:

    Σ_{k=0}^{1023} 50257^k  ≈  50257^1023  =  10^4809.3
                               4810 decimal digits, leading 2109023927…
```

`τ` is injective — decoding is a left inverse — so that quantity bounds
`|Σ*_b|` above. It is **not attained**: `τ` is not surjective onto `V*` (BPE
emits one canonical sequence per string, §4.5.2), so most token sequences of
length ≤1023 are the image of no string at all.

For a lower bound: every string of ≤1023 bytes encodes to ≤1023 tokens, since
each byte is itself a token. ASCII alone therefore gives 128^1023.

```
    10^2156   ≤   |Σ*_b|   ≤   10^4809.3
```

**The width of that bound is the canonicality gap.** How far below 10^4809.3 the
truth sits is exactly the canonicality question; §4.5.2 measures the same
quantity at the scale of `J` (249 systems) rather than of `Σ*`. Q12 is not a
curiosity appended to this document — it is §3 asked again where the answer
is countable.

For scale: atoms in the observable universe, 10^80. Chess games, 10^120. Go
positions, 10^170. Planck volumes in the observable universe, 10^185. The
lower bound clears the last of these by 10^1971.

**Finite.** Reachability is therefore *trivially decidable by enumeration* —
as is every finite lookup problem, which is why decidability is not the
interesting property. The open question is whether reachability admits a
**tractable or compact certificate**. Enumeration is infeasible by ~4800
orders of magnitude, which is why §4 exists.

Note what this number is not. It is a bound on the size of the domain, not
the size of the problem. At least `10^2156` inputs land on 3072 outputs; by I3 the map is
spectacularly non-injective, and the fibers do the compressing. §4.5 reaches
2805 of those outputs in eight hours on one laptop. The search space is
absurd and the object it maps onto fits on a screen.

---

## 4. Reachability — MEASURED / OPEN

Three levels not to be conflated. Each represents a different claim.

```
ℛ_{β,ρ}   = im(D_{β,ρ}) ⊆ J              reachable        — open
ℛ̂_C       = { D(s) : s ∈ C }             observed under C — measured
ℛ*        = functionally meaningful      — untouched, see Q10
```

**ℛ* is the foundational one, and it is the one with no results in it.**
Everything else in this document is a statement about a deterministic
classifier over activation maxima. That is a well-defined object and the
measurements are real. But the words this document uses throughout — *system*,
*locus*, *destination*, *region*, *space* — all quietly assume that winning an
argmax at ℓ=5 corresponds to something the network computes with. Nothing here
establishes that.

Q10 is the test: patch, ablate or amplify `A[:,j]` and measure downstream
effect. Until it runs, read every occurrence of those words as *the neuron
that won*, and nothing more. It is listed tenth in §9 for historical reasons
and belongs first.

That instruction is a prohibition, not a distinction, and it has been doing
the work of one. §4.6 separates the senses of *meaning* this document keeps
colliding: Q10 settles one of them and is independent of two others.

```
j ∈ ℛ   : certificate = any s with D(s) = j.        |certificate| = 1 string
j ∉ ℛ   : certificate = ? ← the open problem
```

A system reachable only at probability 1e-15 under natural language is, for
an operator, not reachable. **Intrinsic reachability and distributional
accessibility are different properties, and this document is mostly about the
second.** §4.5 attacks ℛ directly and returns no evidence of unreachability
under the budgets tried; what it finds instead is a cost gradient. §4.1 is retained because
the attempt to prove the first kind of claim, and its failure, is part of the
record.

### 4.1 Sound unreachability certificate — DERIVED

Let `u = (x − μ)/√(σ²+ε)`. Then

```
1ᵀu = 0,   ‖u‖ = √d · σ/√(σ²+ε) ≤ √d
```

so `u ∈ S = {u : 1ᵀu = 0, ‖u‖ ≤ √d}` — independent of x, θ, and position t.

Define, for neuron j:

```
v_j = γ ⊙ W[:,j]  ∈ ℝ^d
c_j = βᵀW[:,j] + b_j  ∈ ℝ
P   = I − (1/d)·11ᵀ
```

Then `a_j = uᵀv_j + c_j`, and since `max_{u∈S} uᵀv_j = √d‖Pv_j‖`:

```
    L_j = −√d‖Pv_j‖ + c_j   ≤   a_j   ≤   √d‖Pv_j‖ + c_j = U_j        (†)
```

**Both bounds are exactly attained**, for `Pv_j ≠ 0`, at
`u* = ±√d · Pv_j/‖Pv_j‖`; if `Pv_j = 0` both extrema equal `c_j` (verified
numerically to 10 decimal places) — attained *within S*. (†) is the **exact
support-function bound for the relaxation `S`** — tight within `S`, and
claiming nothing about bounds derivable from LayerNorm by other routes (a
different treatment of γ, β, ε or of finite dimension may give more). `S` is a
relaxation: real `u` are outputs of blocks 0..ℓ−1 and need not include `u*`.
That gap is the whole story of §4.1.1.

Let `x⋆ = argmin g`. For `g = gelu_new`, i.e.

```
g(x) = 0.5·x·(1 + tanh( √(2/π)·(x + 0.044715·x³) ))

x⋆ = −0.752461      g(x⋆) = −0.170041
```

`g` has no interior maximum — it decreases to `x⋆` and increases monotonically
thereafter, approaching 0 from below as x → −∞ — so on `[L,U]`:

```
ĝ(j) = max( g(L_j), g(U_j) )              upper bound on achievable activation
ǧ(j) = g(x⋆)                     if L_j ≤ x⋆ ≤ U_j
       min( g(L_j), g(U_j) )     otherwise     guaranteed floor
```

```
    Θ = max_{j'∈J} ǧ(j')

    ĝ(j) < Θ   ⟹   j ∉ ℛ_{β,ρ}   for all β, ρ                        (‡)
```

*Proof.* If `ĝ(j) < Θ = ǧ(j⋆)` then `A[t,j] < A[t,j⋆]` for every input and every
position t. So j is never the argmax at any position, hence never the argmax of
the per-position max (ρ=R) nor at t=T (ρ=I). ∎

### 4.1.1 (‡) is vacuous at ℓ=5 — MEASURED

`g = gelu_new` attains a global minimum `g(x⋆) ≈ −0.170041`. Therefore:

```
    L_j ≤ x⋆ ≤ U_j  ∀j   ⟹   ǧ(j) = g(x⋆) ∀j
                         ⟹   Θ = g(x⋆) = min g
                         ⟹   ĝ(j) ≥ Θ ∀j          (Θ is the global min of g)
                         ⟹   {j : ĝ(j) < Θ} = ∅   necessarily
```

So (‡) can only fire if **some neuron cannot be driven into the GELU dip**:

```
    non-vacuity requires   ∃j :  L_j > x⋆
      i.e.                 ∃j :  c_j − √d‖Pv_j‖ > −0.752461
```

Measured, `gpt2`, ℓ=5, `unreachable_certificate.py`:

```
pre-activation range over all j : [−58.810, +57.215]
Θ                               : −0.170041   ( = g(x⋆) exactly)
|{j : L_j > x⋆}|                : 0
|{j : ĝ(j) < Θ}|                : 0
min_j ( ĝ(j) − Θ )              : +10.008860
```

Stated positively: **under the LayerNorm-only relaxation, every neuron's
feasible preactivation interval `[L_j, U_j]` contains the GELU minimum, by a
margin of at least 10.0.** This is a statement about `S`, not about what the
real GPT-2 computation can produce: it does not establish that any actual
input drives any given neuron into the dip. No neuron is anywhere
near being isolated by the LN constraint, so every floor collapses to `min g`
and no neuron dominates any other. **This certificate is structurally
incapable of firing at this layer** — a fact about the relaxation, not about
GPT-2. Not tested at other ℓ (`--all-layers`).

Cause: `S` permits `u` to align exactly with any single `v_j`. Real `u` do not.
The relaxation over-estimates every `U_j` and under-estimates every `L_j` by a
wide margin.

**Consequence.** The cheap route to certifying `j ∉ ℛ` is closed *for this
relaxation*. Any proof must propagate bounds through blocks 0..ℓ−1 (`S₃` in
§4.2), which is an open research problem, not a script.

### 4.2 Tighter relaxations — OPEN

```
S₁ = S                                     §4.1, sound, closed-form
S₂ = conv{ u observed over corpus C }      sound only w.r.t. C
S₃ = { u : u = LN(x), x ∈ im(blocks 0..ℓ−1) }   the true set; unknown
```

---

### 4.3 One-token census — MEASURED, complete

`token_sweep.py`, `gpt2`, ℓ=5, all 50,257 v ∈ V. Not an estimate.

```
BOS probe.  A(<|endoftext|>)[0, :] is constant under every implicit input.
    argmax = 2256  at  2.3609
    next   = 1888 (1.564), 2787 (0.893), 1450 (0.854), 1048 (0.849)
```

```
quadrant        |ℛ̂ ∩ one-token|       busiest destinations
exp_r ≡ exp_i        1 / 3072          2256  (50257/50257 = 100%)
imp_r              171 / 3072          1888 (45555), 1790 (2953), 2256 (640)
imp_i              193 / 3072          1888 (45964), 1790 (3073),  464 (262)
```

**E1. The one-token explicit probe family is degenerate.** All 50,257 tokens
map to 2256. Position 0 carries token identity and positional information;
what it lacks is *preceding-token context*. One neuron dominates regardless of
token. `|ℛ̂| = 1`. This is a fact about the probe, not about `D`.

**E2. 2256 is the BOS footprint, not a destination.** Under ρ=R the max runs
over positions, and position 1 of every implicit input contributes a fixed
value `v⋆` at neuron 2256, where `v⋆ = A(<|endoftext|>)[1, 2256] ≈ 2.3609`
(the full fp32 value is the normative one; four decimals are for reading).
An input lands on 2256 iff no neuron beats `v⋆` anywhere in it, and no neuron
of lower index ties it:

```
    D_{1,R}(s) = 2256   ⟺   max_{t≥2} max_j A[t,j] ≤ v⋆
                        and  no j < 2256 attains v⋆ at any t
```

The second clause is an artefact of the §1 tie rule and has never fired.
Stated with `<` and a rounded constant, as in earlier drafts, this was an
approximation presented as an equivalence.

640/50257 single tokens (1.3%) fail that test. 2256 is a null result — *signal
too weak* — an artefact of the start-of-text marker, not of anything semantic.
It recurs in §4.5 as the floor a failed search sits on.

**E3. One-token doors are rare.** ≤6.3% of systems are reachable by any single
token in any quadrant. 90.6% of tokens land on 1888 under ρ=R. Multi-token
structure is required for essentially all of the map.

### 4.4 Activation suppression under the one-token probe — MEASURED

`max_v A(v)[·, j]` over all 50,257 single tokens, split by whether the corpus
ever reached j. Mann-Whitney, `rank_sum(reached, unreached)`:

```
quad     n_unreached   med(reached)   med(unreached)        z            p
imp_r           1620         1.5696           1.0827   +18.00     1.8e-72
imp_i            847         1.3767           0.9976   +16.05     5.4e-58
exp_r           1733        -0.0160           0.0295    -5.04     4.7e-07   ✗
exp_i            921        -0.0103           0.0287    -1.39     1.6e-01   ✗
```

**Positive z ⇒ neurons absent from ℛ̂_C are measurably quieter under this
probe.** For both implicit quadrants this is overwhelming — and overwhelming
is not the interesting quantity.

**Effect size — MEASURED, Cliff's δ off `sweep_neurons.tsv`.** δ is the
probability that a randomly chosen reached neuron is louder than a randomly
chosen unreached one, rescaled to [−1, 1]. 95% CI by bootstrap, 2000
resamples:

```
quad        δ        95% CI          P(reached louder)   magnitude
imp_r   +0.376   [+0.336, +0.413]         0.688          medium
imp_i   +0.374   [+0.334, +0.413]         0.687          medium
exp_r   −0.106   [−0.148, −0.065]         0.447          negligible   ✗
exp_i   −0.032   [−0.082, +0.013]         0.484          negligible   ✗
```

**This is a real but partial separation, and the z badly oversold it.** Pick a
reached neuron and an unreached one at random: the reached one is louder about
69% of the time, against 50% by chance. The distributions overlap heavily. In
imp_r, 173 neurons sit in the loudest quartile and were never reached, while
269 sit in the quietest quartile and were reached anyway. Spearman correlation
between maximum single-token activation and corpus hit count is ρ = 0.32
(imp_r) and 0.35 (imp_i) — a tendency, not a rule.

**Consequence.** *Some* of the corpus's failure to reach 736 neurons is
associated with those neurons being quieter under this probe. Most of the
variance is not explained by loudness. Whatever else is going on — context
length, token composition, the BOS competition of §4.2 — is the larger part.
Read §4.4 as one contributing factor identified, not as the explanation.

**On n and multiple testing.** There are four tests here, one per quadrant,
not 3072 — no familywise correction is called for. The concern the large n
raises is the opposite one: at n=3072 any non-zero difference reaches
significance, so p and z carry almost no information about magnitude. δ is
reported because it is the quantity that does.

**Selection.** *Reached* is defined by an argmax event, so the two groups are
conditioned on the outcome and a positive δ is partly built in: a neuron that
never wins anything is likelier to be one that cannot get loud. Two things
limit how much of δ this explains. The split comes from 407,475 multi-token
sentences while the activation proxy comes from 50,257 single tokens, so they
are different regimes rather than the same measurement twice. And only 171 of
3072 neurons ever win a single-token argmax at all, of which 157 are also
corpus-reached — too few to account for a group of 1452. The caveat stands
without dissolving the result.

**Scope.** This establishes that the two groups differ under the one-token
probe distribution. It does not establish a mechanism, and it is not evidence
of unreachability — a neuron requiring four tokens of context reads as quiet
here. §4.5 is the direct test.

`✗` — exp rows are void: by E1 the explicit single-token regime is degenerate,
so `exp_max` measures position-0 behaviour and is not a proxy for anything.
Both δ are negative, i.e. pointing the wrong way, which is what a void
measurement should look like.

Source of the split: `results/master_hit_counts.tsv` (§7.2), the same corpus
coverage table §8.3 reports. `n_unreached` here and the unreached counts there
are the same numbers.

### 4.5 Coordinate ascent against ℛ — MEASURED, imp_r

`coordinate_ascent.py`. Gradient-guided discrete search (GCG-style) over token
sequences, candidate pool restricted to the 49,905 round-trip-safe tokens.
`dom(D)` is `Σ*_b`, not `V*`, so every token-level hit is re-evaluated on the
string it decodes to. Only that counts as a certificate. Reported throughout:

```
hit_tok   a token sequence exists that lands on j
hit_str   the text it decodes to still lands on j    ← the certificate
```

**Calibration.** 100 systems reached exactly once each by a single novel
(§8), all therefore in ℛ by construction:

```
budget                                 hit_tok   hit_str
len 8,  sweeps 4, restarts 2             89        74
len 16, sweeps 8, restarts 4             96        88
```

**Calibration recovery rate is 88% at the second budget.** Every figure below
is reported against that denominator. It is not "recall" in the usual sense:
the true population of reachable systems is unknown, which is the whole
question, so this is search success on a set known reachable by construction.

**The calibration set is plausibly skewed hard, and the skew has a sign.** A
system reached *exactly once* by one novel is selected for rarity under
natural language, and §4.5.1 shows rarity co-varies with search cost. If that
selection makes them harder for this searcher, 88% understates recovery on a
target drawn uniformly from J, the expected count under ℛ = J is above 2703,
and the observed shortfall is larger than 15.

Plausibly, not demonstrably: selection on natural-language rarity does not by
itself prove difficulty for *this particular search algorithm*. Establishing
it needs recovery rate plotted against prior discovery count, which has not
been run. Stated here because it cuts against the conclusion below rather
than for it.

**Sweep over all of J**, imp_r, second budget, 3072 targets:

```
hit_str   2688 / 3072   (87.5%)
hit_tok   2900 / 3072
residue    384  =  212 phantom (hit_tok, no string)  +  172 no route
```

87.5% against a calibrated recovery rate of 88.0%. The searcher fails at the
same rate over J as it does over targets known to be reachable. If every
system were reachable, a searcher of this rate would return ≈2703; it returned
2688, a gap of 15.

**Where the control standard error comes from.** `p̂ = 0.88` is estimated from
n=100, so `se(p̂) = √(.88·.12/100) = 0.0325`, and the uncertainty it induces in
the predicted count is `3072 · se(p̂) = 99.8`. The binomial term at N=3072 is
`√(3072·.88·.12) = 18.0`. Combined, `√(99.8² + 18.0²) = 101.4`. A gap of 15
against ~101 is not a signal.

**The sweep is consistent with ℛ = J and returns no evidence of
unreachability at these budgets.** It does not establish that ℛ = J: a
universal claim over 3072 systems is not provable by a search that fails 12%
of the time on targets it is known to be able to reach. Thirty-four of the 384
were reached by *Moby-Dick*: at minimum 9% of the residue is demonstrably
search failure.

**The 172 at `len 32, sweeps 20, restarts 10`** — twelve times the work per
target, ~164 s each:

```
hit_str   117 / 172   (68%)
hit_tok   154 / 172   (89.5%)
          37 phantom     18 no route found
```

**Internal control.** System 2078 appears in that final 18. It fell to this
same budget on the previous day, and *Moby-Dick* reaches it in one sentence.
Same searcher, same settings, different seed, different answer.

`INTERPRETATION` — this is evidence against a hard search barrier at 2078 and
a demonstration that the residue is at least partly seed-dependent. It does
**not** establish that the whole residue is seed noise; one instance cannot
carry that. The measurement that would is cheap and is not yet run: n seeds
against each of the surviving 18, reporting `P(hit | target, budget, seed)`.
Until then the residue is **not established as structure and not established
as noise.**

**Cumulative, imp_r:**

```
string-reached   2688 + 117 = 2805 / 3072   (91.3%)
token-reached    2900 + 154 = 3054 / 3072   (99.4%)
canonicality gap      212  +  37 =  249          ( 8.1%)
no route known                  18          ( 0.6%)
```

### 4.5.1 Access cost — MEASURED

Median Δ of the certificate, grouped by how hard the route was to find:

```
Melville routes, natural English        0.1676     n=545
sweep, light budget                     0.0810     n=2688
hard-172, heavy budget                  0.0313     n=117
```

Monotone. **Cost of access and precariousness of arrival are the same
gradient.** 91% of the heavy-budget certificates land within 0.1 of a
different system. A system that takes hours to find is found on a knife edge.

`REFUTED` — earlier drafts continued: *by I5 most of them will not survive a
different GPU.* §8 measures the opposite. Arrivals at Δ=1e-6 survived a change
of machine; a Δ=0.03 certificate is four orders of magnitude clear of the
observed perturbation. **Heavy-budget routes are precarious in the sense that
a small edit to the string moves them (Q5), not in the sense that a different
computer disagrees about them.** The two kinds of fragility were conflated
and only the first is established.

This is the central result. Not that any system cannot be reached — no search
run here has produced evidence of one — but that reaching costs, and what it
costs is legible: string length, search effort, and margin, all moving
together. Intrinsic reachability and distributional accessibility come apart,
and §4.5.1 is the measure of the gap.

### 4.5.2 The canonicality gap — MEASURED, upper bound

**Naming.** Earlier drafts called this the *phantom* gap, which presumes the
answer: that the 249 are artefacts of an unconstrained search rather than
real token/string discrepancies. That is Q12, not a result. Called the
**canonicality gap** until Q12 runs; "phantom" becomes available as an
interpretation if most of it disappears.

249 systems (8.1%) have a known token route and no known string. The searcher
operates over `V*`; `D` is defined over `Σ*_b`; and `τ` is not surjective —
BPE emits exactly one **canonical** token sequence per string, and the
optimiser is free to construct non-canonical ones that no text produces.

This is not evidence of anything yet. The search was never constrained to
canonical sequences, so the gap is an upper bound on the phenomenon and mostly
measures the wrong search space. A canonically-constrained ascent would close
an unknown fraction of it. The pruning argument is cheap to build but rests on a proposition that this
document has not stated formally, and should before any implementation
depends on it:

```
CLAIMED   canonical tokenizations are prefix-closed under the
          non-recovering property of BPE, so a violation is detectable the
          moment it appears and unfixable thereafter ⇒ pruning is sound.

OPEN      "prefix-closed" is ambiguous between at least three propositions:
            (a) the set of canonical token sequences is prefix-closed;
            (b) every prefix of a canonical sequence is canonical;
            (c) a sequence that has ceased to be extendable to any canonical
                encoding can never become extendable again.
          Pruning soundness needs (c). (a) and (b) are neither equivalent to
          it nor to each other. State and prove the one required, against
          this tokenizer implementation, before relying on it.
```

What remains after that construction is Q12, and it is the only candidate for
genuine unreachability left standing in this document.

### 4.6 Four senses of "meaning" — INTERPRETATION

This document uses *meaning* in exactly one sense and has never said so. The
instruction in §4 — read *system* and *destination* as **the neuron that
won** — is the right instruction, but it suppresses the other senses rather
than locating them, and a suppressed sense comes back in the next draft.
Four are in play. They are not competing answers to one question; they are
different questions that share a word.

```
FUNCTIONAL    does the network compute with j? Does intervening on A[:,j]
              change downstream behaviour, relative to matched controls?
              TEST     Q10. Causal, and the only one with an experiment
                       designed for it.
              STATUS   unrun. Everything about ℛ* waits here.

REFERENTIAL   does j correspond to a describable property of the input —
              can the rule be written down without evaluating A?
              TEST     §8.1's cheap-predictor benchmark, and it points
                       upstream where Q10 points downstream.
              STATUS   predicted negative, unrun. Split out as Q10' below,
                       because it was folded into Q10 and is not the same
                       question: a neuron may be causally load-bearing and
                       referentially opaque, or referentially transparent
                       and causally inert.

INDEXICAL     does j mean something to a population, by accreted use?
              TEST     none required — it is observed or it is not.
              STATUS   available now, and unaffected by the answer to Q10.

PROJECTED     what does a reader find in an arrival, having gone looking?
              TEST     none possible. Not a claim-bearing sense.
              STATUS   present throughout, including in this document.
```

**One rule, and it runs one way.** The indexical and projected senses may
never be offered as evidence for the functional or referential ones. Almost
every overclaim withdrawn between v6 and v14 was some version of that
crossing. The guard is kept; what changes is that it is now implemented as
*say which sense* rather than *do not use the word*.

**The indexical sense does not wait on Q10, and this is not a consolation
prize.** Jita 4-4 is a station in a system of unremarkable statistics that
became the market of its game because it became the market of its game. Its
meaning is accreted use and nothing else — a Schelling point with a decade of
history stacked on it, a property of the population and not of the map. 1888
is a candidate for that role: reached by 90.6% of single tokens under ρ=R
(§4.3), a frequent winner with no established meaning in any other sense
(Q11). If Q10 comes back flat — destinations are computationally inert argmax
winners and there is no ℛ* — the indexical sense is untouched. A trade hub
does not need its coordinates to signify.

The prior-art section records an instance of this happening on an address
space with no semantics at all, no ledger and no economy attached.
Distinctiveness plus a shareable name appears to be the whole recipe.

**Consequence for congestion.** Concentration on 1888 has been read as an
onboarding defect — a new operator's first strings all landing in the same
handful of systems. Under the indexical sense that reading is at least
incomplete: Jita is the most congested place in its game and is not a bug
there. The problem that remains is narrower and more tractable — whether the
first session gives an operator a reason to leave the busy core, and a
legible sense that leaving is possible. That is a different design problem
from *too many arrivals land in one place*, and a smaller one.

**The projected sense, stated once and then left alone.** Two examples from
this document's own production, offered as neither result nor argument. The
fixture set in §8 turns on `it was cold` and reaches, through Melville, a
sentence about Iceland — and the game this proposal is written for descends
from one made in Reykjavík, which no part of the search was looking for. The
tightest arrival in the entire corpus, Δ=0.000001 at j=1888, is a line about
something that fled and became transfigured into a still subtler form.

```
CAUTION   7353 sentences over 3072 destinations, read by someone with the
          context to notice, will produce coincidences at some rate. That
          rate has not been computed and no attempt is made here to
          establish that these exceed it. They are ornament.
```

They are recorded anyway, for one reason: they are part of why the work
continued, and a document that omits them misdescribes how it was produced.
A stronger position is available — that *place* bestows meaning, and that
Iceland arriving unbidden is therefore not merely coincidence — and it has
its own literature. Nothing here depends on it, and none of it is cited.

---

## 5. Map — DERIVED (rendering) / PROPOSAL (lens family)

One atlas per quadrant — `atlas[protocol_mode]` — so a system's visit count is
quadrant-local. `L = 70` (`distributionSize`), `R = L/2`.

```
b_j ~ Unif([−L/2, L/2]³)     genesis position, i.i.d. per neuron, unseeded
h_j ∈ ℕ                      visits to j in the active atlas
```

`b_j` is arbitrary and carries no information. Index adjacency is not spatial
adjacency: j=10 and j=11 are unrelated, and so are their genesis positions.
Nothing in the render claims otherwise.

A **lens** λ maps `(b_j, h_j) → p_j ∈ ℝ³`. Four are implemented:

```
λ_raw    p_j = b_j
         Genesis only. Both direction and radius are noise.

λ_grav   pull = min(h_j/10, 1)
         p_j  = b_j · (1 − 0.8·pull)
         Absolute visits, saturating at 10. Radial contraction along the
         genesis ray, floor at 0.2·|b_j|. Retains |b_j|, so radius mixes
         visitation with noise.

λ_norm   prox = (h_max − h_j)/(h_max − h_min)      over discovered only
         p_j  = b̂_j · prox · R      discovered
              = b̂_j · R             undiscovered ⇒ outer shell (see below)
         Discards |b_j| entirely; keeps only the genesis direction b̂_j.
         Radius becomes a pure function of visit rank. Busiest → origin.
         Undefined and auto-disabled for <2 discovered, or h_max = h_min.

λ_orr    Top 11 by h_j. Rank 0 → origin. Ranks 1..10 coplanar at radius
         prox·R, angle (rank−1)·2π/10, plane slerped on an 11 s cycle.
         Remainder fall back to λ_norm. Asserts a hierarchy, not a geometry.
         Not a lens in the sense of the other three: λ_raw, λ_grav and
         λ_norm encode measured quantities, λ_orr arranges a ranking.
         Read as annotation layout.
```

**What the family is for.** Genesis hands you a direction and a radius, both
arbitrary. Visitation hands you no geometry at all. Each lens is a decision
about which of the two to keep. λ_norm is the honest one: it throws away the
arbitrary radius and rebuilds it from visitation, so **angular position carries
no information and radial position carries only visit rank**. Centre means
most-visited. There is no third claim in the picture.

**The outer shell is a null result, not a frontier.** It means *zero observed
visits in this atlas* — it does not mean rare, remote, difficult, exotic, or
low-probability. A system sits there because nobody has been, which may be
because nobody has tried. This must be legible in the render itself and not
only in this paragraph, because the picture will otherwise say the opposite:
an unvisited shell around a busy core reads as a frontier whatever the prose
says. Undiscovered systems should be visually distinguishable from
least-visited ones. λ_grav keeps `|b_j|` and is therefore prettier and less
honest; both exist so the operator can see the difference.

Consequently `nbr_{λ₁}(j) ≠ nbr_{λ₂}(j)` in general and no metric on J is
claimed by any lens.

What *is* claimed: **under λ_norm the map is a radial encoding of visit rank
over an arbitrary angular embedding.** Not a spatial rendering of the visit
distribution — the distinction matters because readers convert pictures into
geometry faster than they read caveats. That is the same object §6.1 calls
`P_π`. Two operators comparing atlases are comparing empirical distributions
over J — which is why N3 (§6.1) is a measurement of something players already
do by eye.

The map is not a claim about J. It is a record of what has been found, which
is the only thing anyone will actually have.

### 5.1 Lenses not yet built — PROPOSAL

One lens family was built. It has already shown three things — coverage
(λ_raw), visit concentration (λ_grav, λ_norm), hierarchy (λ_orr). The object
supports more. These are sketches, not specifications, and the list is not
meant to be closed.

From data already recorded:

```
λ_margin   radius from Δ rather than h_j. Knife edges to the rim, basins to
           the centre. Renders how nearly the arrival went elsewhere. Not
           portability — §8 measures destinations as stable across stacks.
λ_route    radius from min |τ(s)| over known routes to j. The 171 one-token
           doors near centre; systems known only by 32-token search out at
           the shell. §4.5.1 shows this gradient is real and that it
           co-varies with Δ, so λ_route and λ_margin will largely agree —
           which is itself the finding.
λ_quad     partition by which quadrants can see j at all. §8.3 gives union
           2336 against imp_r 1452. That partition exists. Nobody has looked.
```

One new field each:

```
λ_perp     radius from perplexity of the cheapest known route under a fixed
           reference LM. Draws the intrinsic/distributional axis directly:
           English-reachable at centre, token-salad-only at the rim.
λ_first    colour by first finder. Systems reached by exactly one operator
           ever are the incidental tail, rendered rather than argued.
```

Real geometry, for anyone who wants it:

```
λ_edit     nbr(j) = { j′ : ∃ s,s′ with D(s)=j, D(s′)=j′, d(s,s′) = 1 }
           Adjacency induced by the string space rather than imposed on J.
           The cold / colder / coldest family is one edge in this graph.
           Expensive to sample. §5 claims no metric on J — this is the one
           candidate, and this is how you would build it.
```

Mesocosm:

```
λ_diverge  two atlases overlaid, colour by log( P_h / P_a ). N3 as a picture.
           The view §6.1 exists to justify.
```

---

## 6. Reward — PROPOSAL

**Design hypothesis, not theorem.** Holds under: generator throughput
dominating human throughput at equal effective sampling cost; reward
monotone in first-arrival; locations roughly substitutable in value.

```
π       player policy
G       machine string generator, throughput r_G
κ       traversal capacity (undock → haul → hold → return), r_G ≫ κ
```

```
Scheme A:  reward = f(first s such that D(s) = j)
           ⇒ argmax_π E[reward] tends to π = G
           ⇒ human sample contaminated in proportion to |reward|

Scheme B:  reward = f(in-world actions after arrival at j)
           ⇒ value(s) = option value of access to j
           ⇒ marginal value of the (k+1)-th location → 0 for k ≫ κ
           ⇒ π = G yields locations, not payout
```

By I7, `D` cannot tell who wrote the string. Do not try to authenticate the
author. Make the valuable part happen after the string. ⇒ adopt Scheme B.
Split discovery ledger from exploitation ledger accordingly.

Whether j *means* anything is Q10, and Scheme B does not wait on the answer:
in-world value is assigned by the economy, not by J.

**Scheme B has a known weakness: discovery stops mattering.** Past k ≫ κ the
marginal location is worth nothing, so the discovery ledger is a trophy case.
The repair is to notice that a discovery is not a location — it is a *route*,
and routes are not interchangeable. Δ (§8.1) is computed at every arrival and
is currently thrown away.

**Arrival margin is not route quality.** Δ is defined by the competition at
the endpoint, not by the route:

```
arrival margin   Δ = A_max − A_2nd.  MEASURED, endpoint-local. Two unrelated
                 strings reaching the same j can share a Δ; a three-word
                 string and a thirty-token string can share a Δ. Δ says how
                 close the arrival came to resolving elsewhere. Nothing else.

route quality    CONSTRUCTED. Some f(Δ, |τ(s)|, semantic plausibility under a
                 reference LM, measured repeatability). Not yet defined, and
                 not derivable from Δ alone.
```

Conflating the two was doing real work in earlier drafts and is withdrawn.
A route to a common system with Δ=1.2 may be worth more than a route to a
rare one with Δ=0.004 — but that is a claim about *route quality*, and it
holds only if margin turns out to predict the properties one actually wants,
which is Q5 for stability and Q13 for manufacturability. Neither has run.

If it does hold, it restores scarcity to discovery without restoring Scheme
A's contamination: a generator emitting 10⁶ strings still yields mostly
locations, and locations are the thing that saturates under κ.

**Locations saturate is itself an assumption.** It holds if exploitation
capacity κ is the binding constraint. If locations are instead defensible,
colonisable, monopolisable or usable as infrastructure, their marginal value
may not decay, and Scheme B's argument weakens accordingly. Stated as a
condition, not a result.

**Conditional on Q13.** If Δ turns out to be directly optimisable — if a
searcher told to maximise margin rather than merely arrive can lift a Δ=0.03
route to Δ=1.5 — then route quality is manufacturable and every mechanic in
§6.2 collapses. Test before building.

**Ledger visibility is a fork, not a knob.** Public ledger: shared prior,
clean comparison across populations, no information economy. Private or
tradeable ledger: very much this game, but discovery distributions become
path-dependent on social structure and §6.1 stops measuring cognition. These
are different shards, or different experiments. Choose deliberately.

### 6.1 Mesocosm & interface parity — PROPOSAL

J-Space provides a normalised channel for comparing biological (π_h) and
autonomous (π_a) explorers. Unlike motor-dependent benchmarks (APM, reaction
time), both populations emit strings into the same deterministic function.

**Parity is by publication, not by secrecy.** By I6, θ is a download and `D`
is fifteen lines. Both populations can run offline search and both should be
assumed to. Nobody is privileged because nobody is excluded. A design that
assumes the operator will not run the model locally has not met the operator.
§7.2 makes this checkable rather than asserted: the corpus, the caches, the
search scripts and the replay are all published, so an operator can reproduce
the document's own results before deciding whether to trust its server.

**Everyone holds the exact map. WITHDRAWN — refuted, §8.** Earlier drafts
proposed that the server holds the exact map while players hold an
approximation, because floating-point disagreement would concentrate at
low-Δ arrivals and make knowing and confirming come apart. That claim is
false. §8 measures 7353/7353 agreement across two machines at margins down to
1e-6. Under strict fp32 there is no disagreement to build on, and "this
requires no enforcement, it is a property of floating point" is withdrawn
with it.

**The correct reading is stronger than the mechanic it replaces.** By I6 θ is
a download; by I5 as now measured, anyone running fp32 computes `D` exactly,
not approximately. There is no informational asymmetry between server and
operator at all. Therefore:

```
· no mechanic may depend on an operator being unable to confirm a
  destination offline. They can, exactly, every time.
· this is independently the argument for Scheme B (§6). Scheme A fails
  because D cannot authenticate an author (I7); it fails again because
  D cannot be withheld (I5, I6). Two separate routes, one conclusion.
· the DESTINATION is computationally stable: a Δ=0.03 route found on a
  laptop resolved identically on the other tested stack. That is not the
  same as the ROUTE being a stable good — route quality could still
  depend on edits, prefix, retokenisation, context or model version.
  Q5 tests that, and §6.2 depends on it.
```

The residual is configuration, not secrecy: a client running TF32 will
disagree with the server, and that is a bug to detect and report, not a
feature to sell.

'Latent Novelty Generation' is the umbrella term. It is not one measurement.
Claims cite one of:

```
N1  first-arrival count      |ℛ̂_π \ ℛ̂_prior|
N2  tail mass                P_π( j ∈ tail ),  tail = bottom quartile of n_j in C
N3  divergence               D_JS( P_π ‖ P_ref ),  P_ref = corpus distribution
```

Coverage (which systems are ever reached) and distribution (how often each is
reached) are different results. Two populations can converge on the same set
and still spend their discoveries in different regions. Report both.

---

### 6.2 What Δ could carry — PROPOSAL, conditional on Q13

Sketches, in the manner of §5.1. Δ is measured at every arrival and currently
discarded. Observed nonzero Δ spans roughly four orders of magnitude, from
1e-4 to 1.65, and, by
§4.5.1, co-varies with cost of access. That is an unusual thing for a game to
have lying around: a second axis, orthogonal to rarity, produced by the same
act as the arrival and not assertable by design.

**Direction matters more than magnitude.** The intuitive assignment — thin
margin, poor arrival, less loot — makes the payout gradient run parallel to
the difficulty gradient, and the frontier becomes strictly dominated. Nobody
goes. Inverted, thin margin paying more, it becomes a risk gradient: common
systems reached comfortably and paying little, rare systems reached on a knife
edge and paying accordingly. Skill then has somewhere to live — a *fat*-margin
route to a *thin*-margin system is the hard and valuable object, which is Q4
with a market attached.

**Δ is not spatial.** It is tempting to render it as landing near or far from
some centre. There is no centre; §5 is explicit that J has no metric. What Δ
actually says is that the arrival did not fully resolve — the string was close
to designating a different system entirely. Not close enough for other
hardware to disagree; §8 settles that. Close enough that a different *string*
would have gone elsewhere, which is Q5's question and the interesting one.
That is stranger and truer than a fringe, and it survives datamining.

Three things it could carry, not exclusive:

```
yield         monotone in 1/Δ. Simplest. Weakest.
persistence   Δ above a threshold ⇒ the arrival supports structure; below,
              the operator takes what can be carried and the site does not
              outlast the visit. One trip, two categories of outcome.
repeatability HYPOTHESIS: Δ predicts whether a route is a key or an anecdote.
              Fat: hand it to the corp and it works for all of them. Thin: it
              happened, and may not again. This would make routes tradeable
              goods with a measured quality, which is the version §6 needs.

              Not established, and not obviously true. A fat-margin route may
              still be fragile to punctuation, prefix, retokenisation or
              surrounding context; a thin-margin route may be robust to all
              of them. Margin is endpoint-local; repeatability is a property
              of the neighbourhood of the string. Q5 measures the second
              directly and is the test of whether the first predicts it.
```

The third is the one that produces something worth fighting over, and it is
the one most exposed to Q13.

### 6.3 Competence without comprehension — PROPOSAL

**The operator is not required to understand `D`, and designing as though
they were would be a mistake.** This document is written in the language of
the thing it measures, which is arithmetic, and that is the right language
for a specification. It is the wrong language for the activity. The relevant
question is not whether an operator can compute a destination but whether
they can *develop a feel for one*, and those come apart.

Two existence proofs, both from games that ask for skill against real
mathematics without asking for the mathematics:

```
Sailwind    the player navigates by sun, stars, landmark, swell and dead
            reckoning. Competent island-to-island passage does not require
            computing an azimuth. What it requires is that the sky behave
            consistently enough that a habit becomes a method.

KSP         the player develops working intuitions for orbital transfer and
            delta-v budgets without solving anything. The intuitions are
            real — they predict outcomes — and they are not the model.
```

In both cases the game converts mathematics into something a person can hold
in language, in the sense of an internal running commentary: *the sun is high
so we are near noon so we are running roughly west.* **This proposal runs
that conversion backwards.** The operator supplies language and the mechanic
returns an integer. The arithmetic is not being made felt; feeling is being
made arithmetic. Whether the same kind of competence can form across that
inversion is not obvious and is not assumed here.

**The condition it depends on, stated as a condition.** Intuition of the
Sailwind kind requires that nearby actions have related outcomes. Turn the
tiller slightly and arrive slightly elsewhere. If `D` does not have that
property — if a one-character edit relocates the destination arbitrarily —
then no habit can become a method, no operator ever gets better, and the
mechanic is a lottery with an elaborate ticket. If it does have it, at least
locally and at least sometimes, then there is something to learn and the
§4.5.1 cost gradient is the shape of the thing being learned.

```
Q5   measures exactly this. P( D(s') = D(s) | edit distance ≤ k ) is not
     only "is this a hash"; it is whether operator skill is possible at
     all. Unrun.

Q4   asks whether a practised policy outperforms uniform sampling into the
     tail. That is the same question observed from the outside, on people
     rather than on strings.
```

Neither has run, so this section claims nothing and predicts nothing. It
records what the design is betting on, so that a negative result is
recognisable as one rather than absorbed. The fixtures in §8 are the only
relevant evidence and they cut both ways: `it was cold` → 38 and
`it was coldest` → 38 is the encouraging case, a suffix that changes nothing;
`it was colder` → 1888 sits between them and is the discouraging one. Three
strings, one inflection apart, two destinations. Δ is what says how close
each of those calls was, and §1.1 says why the collisions happen where they
do — but three fixtures are an anecdote, and Q5 is the measurement.

**Design consequence, if it holds.** The instrument an operator needs is not
an explanation of `J`. It is a legible feedback signal — where the string
went, how nearly it went elsewhere, and how that compares to the last
attempt. §5's atlas and Δ are exactly that signal, and they are already
computed. Nothing has to be explained to anybody for this to be playable;
something has to be *shown*.

## 7. Reference implementation — NORMATIVE

Normative. `D` is defined relative to this stack (I5).

**Weights and execution stack are separate provenance.** θ is the original
2019 GPT-2 small release; the computation is a 2026 software stack. "GPT-2
small (2019)" describes the parameters, not the arithmetic.

```
weights        original GPT-2 small release, θ frozen
artifact       hf.co/openai-community/gpt2   (alias: gpt2 — same revision)
revision       607a30d783dfa663caf39e06633721c8d4cfcd7e
n_params       124,439,808

sha256(theta, as loaded, fp32)
               113687a222f8cf98039222c27b39aaf716493e5e8c1db94ea4e6544e0814088c

sha256 of artifacts
  model.safetensors      248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707
  vocab.json             196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783
  merges.txt             1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5
  tokenizer.json         8414cab924d8b9b33013f0d221c5862f365ee9be39c5c2bfae8a5a9e970478a6
  tokenizer_config.json  5e04eb606e3a1583530a42e36c2a6b6615c86f34fe77e44d9ddeb43ff940931f
  config.json            0daed7749b4f02b8f76240d5444551d7b08712dab4d0adb8239c56ba823bb7b4
  generation_config.json ed0b32ac72c0f5f44a719abb2d7786ea5146c871f83717b7f2018065954de02b
```

`"gpt2"` alone is a moving pointer and is not sufficient to define `D`.
Load with `revision="607a30d7…"` or the pin is decorative.

**θ is the value that matters most.** `sha256(theta)` is taken over the
state dict — names, dtypes, shapes and bytes, sorted — after casting to fp32.
It identifies the parameters themselves and survives repo restructuring,
format migration and download path. The file hashes go stale if Hugging Face
reorganises; this one does not. `pin_stack.py` computes it, and any
implementation claiming to compute `D` should reproduce it.

```
python 3.11.15 · torch 2.9.0+cu128 · transformers 4.57.1
forward dtype fp32 · device: reference is server-side, single GPU

GPU      NVIDIA GeForce RTX 5060 Laptop   compute capability 12.0
driver   610.88
CUDA     12.8 (as reported by torch)      cuDNN  91002
cuBLAS   ships inside the torch 2.9.0+cu128 wheel; pinning the torch
         build pins it. There is no separate version to record.
OS       Windows 10.0.26200
```

```
MUST be set explicitly — these are not all default:

torch.backends.cuda.matmul.allow_tf32   = False    # torch default: False
torch.backends.cudnn.allow_tf32         = False    # torch default: TRUE
torch.set_float32_matmul_precision("highest")
torch.use_deterministic_algorithms(True)

tokenizer: GPT2TokenizerFast("gpt2", revision as above),
           add_prefix_space=False, no truncation, no padding,
           no added special tokens
```

**Two of these differ from the defaults, and one silently.** `matmul.allow_tf32`
already defaults to False on torch 2.x — which is most of why §8's two stacks
agreed bit-exactly without anyone having decided they should. `cudnn.allow_tf32`
defaults to **True**, and is left true by any implementation that does not set
it. It happens not to matter for GPT-2, which has no convolutions and routes
nothing through cuDNN kernels, but the spec pins it because a reader cannot
be expected to derive that.

**The TF32 flags are set through a deprecated API.** torch 2.9 warns that
`allow_tf32` will be removed after this version in favour of
`torch.backends.cuda.matmul.fp32_precision = "ieee"`. §7 pins torch 2.9.0
exactly so the legacy form is correct *here*, but any implementation on a
later torch must set the new form instead — and a stack that sets neither
runs TF32 by default on Ampere and later, which is the one configuration
known to change `D` silently. §7.1 arm 2 measures how much.

**TF32 is not a performance setting here.** It raises relative GEMM error by
roughly two orders of magnitude, into the range where it is comparable to the
low decile of the Δ distribution (§8.1). Left on, it does not make `D` faster;
it makes it a different function.

**Definition of `A`, independent of implementation path.** `A` is the output
tensor of block ℓ's MLP activation function, taken **before** `c_proj`. Under
this stack that tensor is obtained at `mdl.transformer.h[5].mlp.act`; the hook
path is how it is reached, not what it is. An implementation exposing the same
tensor by another route implements `D`.

```python
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ELL, BOS, N_CTX = 5, 50256, 1024
tok = GPT2TokenizerFast.from_pretrained("gpt2")
mdl = GPT2LMHeadModel.from_pretrained("gpt2").to(torch.float32).eval()
torch.set_grad_enabled(False)

buf = {}
mdl.transformer.h[ELL].mlp.act.register_forward_hook(
    lambda m, i, o: buf.__setitem__("A", o.detach())
)

def D(s: str, bos: bool = True, rho: str = "R") -> int:
    ids = tok(s)["input_ids"]
    if bos:
        ids = [BOS] + ids
    assert 0 < len(ids) <= N_CTX, "outside dom(D)"      # §1 MUST
    with torch.inference_mode():
        mdl(input_ids=torch.tensor([ids]))
    A = buf["A"][0]                                     # (T, 3072), fp32
    assert A.dtype is torch.float32                     # §7: not a cast
    v = A.max(0).values if rho == "R" else A[-1]
    return int(v.argmax())                              # §1 tie rule, asserted below
```

Casting to fp32 *after* a lower-precision forward does not reproduce fp32.
The forward dtype is part of the definition, which is why the assertion is on
the captured tensor rather than on a cast of it.

**Tie rule is asserted, not inherited.** `torch.argmax` documents
first-maximal-index, but that guarantee has varied by version and backend, and
§1 makes it normative. A start-up check belongs in the server:

```python
t = torch.tensor([0.0, 1.0, 1.0])
assert int(t.argmax()) == 1, "backend violates §1 tie rule"
```

`buf` is a module-level dict and is not thread-safe. Adequate for a reference;
a server evaluating concurrently must scope the capture per call.

### 7.1 Cross-stack perturbation — PARTIALLY MEASURED

```
ARM 1  different machine, different date, strict fp32     DONE, §8
       7353/7353 agree · tightest surviving margin 1e-6
       ⇒ perturbation < ~1e-6 · §6.1 withdrawn · I5 restated

ARM 2  fp32 with TF32 enabled                             NOT RUN
ARM 3  CPU fp32 vs CUDA fp32                              NOT RUN
ARM 4  torch / transformers major version bump            NOT RUN
```

Arm 1 answered the design question and closed a mechanic. The remaining arms
are no longer about whether to build something; they are about **what to
detect and refuse**. The expectation is that arm 2 produces flips and arms 3
and 4 do not, which would make TF32 the single configuration that silently
changes `D`.

```
fixtures   the 211 arrivals with Δ < 0.01, and the 18 below Δ < 0.001 (§8.1)
report     max |A − A_ref| over all (t, j)     the perturbation
           flip count against Δ                the consequence
action     any arm that flips becomes a startup assertion in §7, not a
           caveat in prose
```

### 7.2 Published artifacts — NORMATIVE

By I6 nothing here is withheld, and §6.1's parity argument is worth only as
much as that is actually true. Every MEASURED claim in this document is
therefore backed by a named file in the accompanying bundle, and `MANIFEST.md`
maps claims to files with sha256 for each. A claim with no file behind it is
marked as such in both places rather than left to be discovered.

```
tools/
  pin_stack.py              §7      recomputes every hash in this section
  replay_cache.py           §8      the cross-environment replay
  token_sweep.py            §4.3    one-token census, all 50257 tokens
  unreachable_certificate.py §4.1.1 the (‡) bound and its vacuity
  coordinate_ascent.py      §4.5    the search
  export_hits.py            §4.4    read-only extraction of the coverage
                                    table from the live database
  hasher.py                         file digests

data/
  the_sea_raw.json                  the corpus before preparation
  the_sea.json                      the corpus as used
  the_sea_implicit_resonance.json   the cache replayed in §8
  the_sea_explicit_resonance.json   the other three quadrants,
  the_sea_implicit_inference.json     used by §8.2
  the_sea_explicit_inference.json

results/
  sweep_tokens.tsv          §4.3    50257 rows
  sweep_neurons.tsv         §4.4    3072 rows; B4's Cliff's δ derives here
  ascent_all.tsv            §4.5    3072 targets
  control.tsv               §4.5    the 100-system calibration
  hard172.txt               §4.5    the heavy-budget set
  reached_imp_r.txt         §4.5    545 systems
  melville_routes_imp_r.txt §4.5.1  545 first-arrival routes
  margins.tsv               §8.1    n=545 margin distribution
  replay_results.tsv        §8      7353 rows, per-sentence
  fixtures18.txt            §8      the arrivals below Δ=0.001
  safe_tokens.pt            §4.5    the 49905-token candidate pool
  master_hit_counts.tsv     §4.4    corpus hits per neuron per quadrant,
                            §8.3    3072 × 4 rows. Both sections rest on it
```

**On `master_hit_counts.tsv`.** §4.4's split into *reached* and *unreached*,
and §8.3's coverage figures, were both computed against a table inside the
live instance's database. The database is withheld; the table is not operator
data. It is published separately by `export_hits.py`, and it reproduces §8.3
exactly — 1452 / 1339 / 2225 / 2151 reached, 407,475 hits in every quadrant —
and §4.4's `n_unreached` column, 1620 and 847. The identical total across all
four quadrants establishes something §8.2 assumes without stating: the four
quadrants were evaluated over the *same* 407,475 sentences, so the
cross-quadrant comparison is one corpus read four ways rather than four runs.

**Scope of the bundle.** The playable prototype ships alongside these
artifacts rather than separately, but it is not evidence for anything here:
nothing in §1–§9 depends on it, and §5 and §6 describe it rather than
measuring it. It is included because §6.1's parity argument is worth more if
the thing can actually be run.

The prototype's database is **not** shipped — it carries operator activity —
and does not need to be. `seed_database.py` rebuilds it from
`rarity_index.json` and four `master_hits_{quadrant}.bin` files, each 3072 ×
uint32, which are published. That reconstruction was checked: a database
seeded from those files exports a coverage table byte-identical to the one
exported from the live instance, `sha256 2724a6db…`, and reproduces §8.3.
**An operator therefore reaches the document's own coverage figures from
published files alone, without receiving any database from its author.**

**One caveat on §4.3 and §4.4.** `token_sweep.py` predates §7's configuration
discipline. It does not pin `revision`, does not set the TF32 flags, does not
call `use_deterministic_algorithms`, and does not assert the captured tensor's
dtype. Under §7's own argument the practical exposure is small — matmul TF32
already defaults to False on torch 2.x and GPT-2 routes nothing through cuDNN
— but the census was not produced under the normative stack, and a reader
should not assume otherwise. Re-running it under §7 is cheap and has not been
done.

**Verification path for a reader who trusts nothing.** Run `pin_stack.py` and
compare against §7. Hash `the_sea_implicit_resonance.json` and compare against
§8. Run `replay_cache.py` on your own machine: that reproduces the central
result of §8 against a cache generated on hardware neither of us controls, and
it is the one experiment in this document that a third party can run without
re-deriving anything.

**What is not published, and why.** The live instance's database is withheld:
it holds operator activity from the hosted prototype, and the one table in it
that this document's claims depend on is exported separately, above. The
§8.2.1 WikiText run is not published because it no longer exists (see §8.2.1).
The prototype itself is a separate release. Nothing else is held back.

---

## 8. Fixtures — MEASURED

**Cross-environment agreement — MEASURED, `replay_cache.py`.** The corpus is
`the_sea`: five added opening phrases followed by the sentences of
*Moby-Dick*, evaluated up to the last line of the Epilogue and no further. It
is not the raw Gutenberg text and should not be cited as such; the exact
composition is below.
`sha256(the_sea_implicit_resonance.json)[:16] = e2e0c5166a5a0518`.

**Provenance of the corpus — MEASURED, by diff.** *Moby-Dick* is public
domain; Project Gutenberg's licence attaches to their added apparatus. Source
text `moby_dick.txt`, 1,246,660 bytes, a Gutenberg plain-text edition with the
front matter already removed — the `*** START` marker is absent, the `*** END`
marker survives at byte 1,191,770.

Preparation was: split into sentences, prepend five opening phrases. Both
states ship (§7.2), so the following is checkable rather than recalled.

```
the_sea.json            7394 entries — 7393 strings and one malformed []
  [0:5]                 the five added openers, verbatim:
                          "Mellybean"
                          "Where the cosmic winds whisper secrets."
                          "A forgotten star chart."
                          "The echo of a distant song."
                          "A planet made of glass."
  [5:7353]              7348 Melville sentences, ending at the last line of
                        the Epilogue ("the devious-cruising Rachel...")
  [7353:7393]           40 sentences of Gutenberg licence text
  [7393]                []   malformed, an artifact of the split

evaluated               [0:7353] — the openers and the complete novel.
                        Gutenberg mentions inside that range: zero.
```

**The run stopped exactly at the novel's end.** The corpus §8 reports is the
first 7353 entries, and entry 7353 is where the licence text begins, so no
apparatus was evaluated. The claim that apparatus was removed holds of the
measurement. It does not hold of the *file*, which retains 40 licence
sentences past the boundary; they are published as they are rather than
silently trimmed, since trimming would produce a file that never existed.

Sentence lengths in the evaluated range: min 9 characters, median 127, mean
158, max 2764. Eight sentences exceed 1024 characters — well under 1024
*tokens*, so no truncation against the context window occurs.

**One known defect.** `moby_dick.txt` opens with a UTF-8 BOM, which attached
to the first sentence and caused the splitter to drop it. *Call me Ishmael.*
is not in the corpus and never entered any measurement. Nothing downstream
depends on it, and it is recorded because a reader comparing the corpus to the
novel will notice.

The stored atlas cache was generated months earlier on a different laptop
under an unrecorded stack. It was replayed sentence by sentence against §7:

```
stack B (replay)  python 3.11.15 · torch 2.9.0+cu128 · CUDA 12.8
                  NVIDIA RTX 5060 Laptop GPU · Windows 10.0.26200
                  TF32 disabled, fp32 matmul precision highest
stack A (cache)   different machine, ~months earlier, stack not recorded

per sentence      7353 / 7353 agree      0 disagreements   (100.000%)
per system set    545 / 545              +0 / −0
```

**Per-sentence, not per-system.** Earlier drafts reported only the 545-system
set agreement. That is much weaker than it reads: 7,353 sentences collapse to
545 systems, so a sentence could move between two systems already in the set
and leave the set identical. The per-sentence figure is the one that measures
anything.

**What the agreement bounds, and what it does not.** Agreement held at the
tightest arrival in the corpus:

```
Δ = 0.000001   j=1888    "When you think it fled, it may have but become
                          transfigured into some still subtler form."
Δ < 0.001  :  18 sentences, all agree
Δ < 0.01   : 211 sentences, all agree
```

Write `δ_j` for the difference in `A[·,j]` between the two stacks, `w` for
the winner and `r` for the runner-up. A preserved argmax constrains only

```
    |δ_w − δ_r|  <  Δ        the DIFFERENTIAL perturbation
```

and says nothing about `max_j |δ_j|`, the absolute one. Both activations may
have moved by far more than Δ in the same direction with the ordering intact.
An earlier draft of this section claimed an absolute bound of ~1e-6 and
asserted bit-identical activations; **both are withdrawn — neither follows
from destination agreement.**

What is established:

```
D was invariant across the two tested environments, at every one of 7353
sentences, down to a margin of 1e-6. At the tightest arrival the
differential perturbation between the two competing neurons was below 1e-6.
```

**The absolute perturbation for this comparison is not merely unmeasured — it
is unmeasurable.** The cache records `{sentence, neuron_id}`; the activation
tensors from stack A no longer exist. §7.1 arms 2–4 can report
`max_j |δ_j|` because both sides are runnable; arm 1 never can.

**On stack A's configuration.** TF32 reduces mantissa precision and can
produce materially different results, so a TF32-generated cache could
plausibly have entered the Δ<0.01 regime. Whether it would have produced
flips, and how many, is not derivable from a generic error ratio — PyTorch's
own documentation notes TF32 error depends on hardware and workload. Arm 2
measures it. The observed agreement is *consistent with* stack A having been
strict fp32 and is not proof of it.

`D(·, bos=True, rho=R)`, ℓ=5 — 18/18 agreement with the live region.
Δ measured, `--fixtures`:

```
                                                                   dest       Δ
it was cold                                                          38  0.2598
it was colder                                                      1888  0.3017
it was coldest                                                       38  0.2598
it was frigid                                                      2094  0.1207
it was freezing                                                    1888  0.2053
it was icy                                                         1888  0.3017
It was                                                             1888  0.7714
It was c                                                           2874  0.5017
It was co                                                          1888  0.7344
It was col                                                         1888  0.7714
It was cold                                                          38  0.3014
hello                                                              1888  0.0896
lol                                                                1888  0.6605
ah blah                                                            1888  0.3923
open cheese                                                        2256  0.1169
cold gums                                                           945  0.1274
"It was cold as Iceland—no fire at all—the landlord said he couldn’t
 afford it."                                                        688  0.1558
It was cold as Iceland—no fire at all—the landlord said he couldn’t
 afford it.                                                          38  0.1366
```

Median 0.3014 — these are hand-written short strings and sit well above the
corpus median of 0.1676 (§8.1). `open cheese` at Δ=0.1169 came within a tenth
of exceeding 2.3609 and becoming a discovery rather than a null result.

**Three exact collisions, to four decimals:**

```
0.2598   it was cold    ·  it was coldest
0.3017   it was colder  ·  it was icy
0.7714   It was         ·  It was col
```

Not coincidence — this is §1.1. `it was coldest` is `it was cold` plus `est`;
the peak was set in the prefix, `est` outbids nothing, and both the winner and
the runner-up are carried through unchanged. `it was colder` and `it was icy`
collide because both land on a peak set in the shared `it was` prefix. The
ratchet is directly visible in the fourth decimal place.

Encoding is part of the key: U+2014 EM DASH, U+2019 RIGHT SINGLE QUOTATION
MARK. Trailing whitespace is part of the key. The two Iceland rows wrap for
layout only; each is one line. `It was` has no trailing space.

### 8.1 Margin — MEASURED

`Δ(s) = A_max − A_2nd`, imp_r, two different samples of the same corpus.
They are different objects and should not be quoted interchangeably:

```
n=545   one sentence per system reached (first arrival at each)
min 0.0001 · p10 0.026 · p25 0.071 · med 0.168 · p75 0.323 · p90 0.509 · max 1.65

Δ < 0.01 :  24 (4%)
Δ < 0.05 : 101 (19%)
Δ < 0.10 : 184 (34%)
Δ < 0.25 : 359 (66%)
```

```
n=7353  every sentence in the_sea  (replay_cache.py, §8)
min 0.000001 · p10 0.032 · p25 0.091 · med 0.203 · p75 0.385 · p90 0.617 · max 2.31

Δ < 0.001 :   18 (0.24%)
Δ < 0.005 :  109 (1.48%)
Δ < 0.01  :  211 (2.87%)
Δ < 0.05  : 1081 (14.7%)
Δ < 0.10  : 2016 (27.4%)
Δ < 0.25  : 4289 (58.3%)
```

The full-corpus sample reaches four decades lower at the minimum, which is
what makes §8's agreement result as tight as it is.

**Most arrivals are narrow.** A third land within 0.1 of losing.

**What a small Δ means.** The fixtures give `it was cold → 38`,
`it was colder → 1888`, `it was coldest → 38`. An operator who wants 38 cannot
be handed a rule for that: the rule would have to be the arithmetic. Δ is what
turns that complaint into a measurement. Where Δ is large the destination
survives inflection, punctuation and paraphrase, and some compact rule may
exist. Where Δ is small the destination was settled by a near-tie between two
neurons.

`INTERPRETATION` — earlier drafts said *nothing short of evaluating `A`
predicts which one won*. A small observed margin does not imply the absence of
a simpler predictor, and that claim is withdrawn. What replaces it is
falsifiable, once the benchmark is stated precisely:

```
task        binary: given s in the low-Δ subset, name which of the two
            competing neurons won. Chance = 50%.
split       held-out low-Δ strings; the Δ threshold defining the subset is
            fixed independently of the test set.
predictors  token-level rule · linear probe on the input embedding ·
            lookup over terminal tokens · small surrogate model
prediction  none beats 50% by a meaningful margin.
```

The binary framing is load-bearing: predicting among all 3072 puts chance at
~0.03%, where a trivial class prior beats it. **Selection caveat** — the
subset is defined by `A_max − A_2nd` being small, so its members are already
selected on the activation output, and a predictor may exploit features
correlated with *membership in the low-Δ set* rather than supplying any
general rule. Hence the held-out split and the independently-fixed threshold.

If a cheap predictor does win, the "string is a key, not a description"
reading fails and the compact rule exists after all. That is a better result
than the claim it replaces, whichever way it lands.

A third of arrivals sit in that regime. Q5 measures how far the boundary is
from any given string.

Δ therefore does two jobs: it separates basins from knife edges, and it is
the mechanical quantity **proposed to test** the *you cannot write the rule
down* hypothesis. It is not itself evidence for that hypothesis; the
prediction below is what makes it falsifiable.

A third job was claimed in earlier drafts — that Δ is the portability error
term — and it is withdrawn as refuted. Δ bounds *how large a perturbation
would be needed*; §8 measures the perturbation actually present across two
machines at below ~1e-6, which is smaller than every margin in the corpus.
Nothing flips. Δ is a sensitivity margin with no observed sensitivity to
hardware; it remains a real measure of how nearly the arrival went elsewhere,
and of how little of the string's content the destination is a summary of.

### 8.2 The lens changes the map — MEASURED

Same book. Same layer. Same BOS. Only ρ differs:

```
                imp_r        imp_i
distinct          545           57
top-3 share       31%          95%
```

Cause, measured over the same 7,353 sentences under imp_i:

```
final char   sentences   distinct systems   top system (share)
    .            6182           20            1821 (48%)
    !             666           29            1790 (61%)
    ?             492           18            1790 (72%)
```

Under ρ=I only the final position is read, so for prose the destination is
steered largely by the terminal token. A novel is homogeneous at its sentence
endings and collapses. Context still modulates — 20 is not 1 — but coverage
under ρ=I tracks **terminal-token diversity**, while ρ=R is sensitive to
activations accumulated across every position of the string. That is an
architectural statement, not a semantic one: it does not establish that ρ=R
tracks *meaning*. This is the mechanism behind Q6's inversion: 38
hand-written lines end in varied and often unterminated ways; 407k scraped
sentences carry a far wider spread of terminal tokens than any single author.

### 8.2.1 Coverage curves, WikiText-103 — MEASURED, SOURCE DATA LOST

`wiki103test_511`, ℓ=5, all four quadrants, |C| = 40,000 sentences. Read off
the run plot; exact values to be taken from the log.

**The log does not survive.** The run emitted one file per entry — several
hundred thousand of them — and the drive it was written to did not survive the
run. No copy is in any backup, and the figures below are read off a plot whose
underlying data no longer exists. **This is the only section of this document
whose numbers cannot be reproduced from the published bundle (§7.2),** and the
approximations below are the strongest form in which they can be stated.
Treat them as indicative and not as MEASURED in the sense the tag usually
carries here.

**A different WikiText run does survive and has not been examined.**
`wiki103_partial870k_2025-11-05` holds a 44MB per-sentence JSONL with all four
quadrant destinations, plus checkpoint hit-count arrays. It is not this run —
it is larger, later, and differently sampled — so it cannot restore the
figures below. It could replace them. Until someone reads it, this section
stays as stated.

Re-running is the repair and is not expensive: 40,000 sentences × 4 quadrants
is ~160k forward passes, and Q6 wants a run at two scales on one corpus in any
case (below). One TSV, not four hundred thousand files.

```
|C|          exp_r    imp_r    exp_i    imp_i
 5,000        ~478     ~462      ~88     ~110
40,000        ~975     ~945     ~275     ~305
```

Three things follow, none of them previously stated.

**ρ is the dominant lever; β_bos is not.** The two ρ=R curves lie almost on
top of each other, as do the two ρ=I curves. Readout choice changes coverage
by a factor of ~3; BOS changes it by a few percent. The four quadrants are
better described as two regimes with a minor perturbation.

**Neither regime has saturated at 40k.** Both curve families are still
climbing at the end of the corpus, ρ=R at roughly a third of J. Q3's
saturation question is not answerable from any corpus this size.

**This inverts §8.3, and the inversion is Q6.** Here ρ=R covers ~3× ρ=I. At
§8.3's 407k sentences the ordering is reversed: exp_i 2151 and imp_i 2225
against exp_r 1339 and imp_r 1452. Two candidate explanations, and this pair
of runs cannot separate them:

```
(a) crossover in |C|.  ρ=I accumulates coverage through terminal-token
    diversity (§8.2), which is slow at first and does not exhaust; ρ=R
    front-loads and then flattens. A crossover somewhere in 40k–407k.

(b) corpus composition.  §8.3's corpus carries a wider spread of terminal
    tokens than WikiText-103, which by §8.2 lifts ρ=I directly and would
    produce the same ordering with no crossover at all.
```

Separating them requires running the *same* corpus out to both scales and
reporting terminal-token entropy at each. That is the remaining half of Q6.

### 8.3 Corpus coverage — MEASURED

|C| ≈ 407,475 sentences, ℓ=5:

```
              |ℛ̂_C|
exp_r        1339
imp_r        1452
exp_i        2151
imp_i        2225
union        2336        never reached in any quadrant: 736
```

736 is a property of C and nothing else. Targeted search reaches 91.3% of J
as strings and 99.4% as token sequences (§4.5); the systems C never visited
are not systems that cannot be visited. Earlier drafts of this document
treated 736 as a candidate unreachable set. That reading is withdrawn.

Verified against `results/master_hit_counts.tsv` (§7.2): all four quadrants
report 407,475 hits, so these are one corpus read four ways. Verified twice —
the same table exported from the live instance and from a database seeded from
scratch out of the published `master_hits_{quadrant}.bin` files agree
byte-for-byte, `sha256 2724a6db…`.

---

## 9. Open

Question IDs are stable and referenced throughout; they are not a priority
order. Read in this order instead:

```
BEFORE ALL     §4.6  which sense of "meaning" a question is asking about.
                     Q10 and Q10' settle two of the four; two others are
                     not questions this document can ask
FOUNDATIONAL   Q10   does a destination mean anything, functionally? (§4, ℛ*)
CHEAPEST       Q13   is Δ optimisable? — smallest experiment, largest
                     design consequence; §6.2 and E2 both hang on it
SHARPEST       Q12   canonicality — the only live candidate for genuine
                     unreachability
PLAYABILITY    Q5    whether operator skill is possible at all (§6.3).
                     Independent of Q10: a mechanic can be learnable
                     without any destination meaning anything
THEN           Q10' · Q6 · Q9 · Q1''' · the rest
```

```
Q1  Is {j : ĝ(j) < Θ} nonempty?                       ANSWERED: no, ℓ=5.
      Vacuous by construction — see §4.1.1. Closed.

Q1' Do neurons absent from ℛ̂_C have systematically lower activation ceilings
      under the one-token probe?
      ANSWERED (measured, not ĝ): yes, decisively. §4.4.
      imp_r z=+18.00 p=1.8e-72 ;  imp_i z=+16.05 p=5.4e-58
      Effect size pending. Not a proof of unreachability.

Q1'' Does any tightening of S (§4.2) make (‡) non-vacuous?
      Minimum requirement: ∃j with L_j > x⋆ = −0.752461.

Q1'''Does the §4.4 effect survive multi-token probes? A neuron needing 4
      tokens of context reads as quiet in the one-token census.

Q2  Under a specified input distribution P(s), what is the asymptotic support
      of D and the expected discovery curve E[|ℛ̂_C|] per quadrant?
      (Unstated P(s) makes |C| → ∞ ill-posed.)

Q3  Does |ℛ̂| saturate below 3072 under natural-language P(s), and at what
      rate? Distinct from Q2 by being an estimation problem, and note that
      it is now a question about *corpora*, not about ℛ: §4.5 finds no
      unreachable set.

Q4  ∃ practised operator policy π_h with
      P(D(π_h) ∈ tail) > P(D(Unif) ∈ tail)  ?          (skill gradient)
      §6.3: this is whether operator competence exists, measured on people.

Q5  Stability. P( D(s') = D(s) | edit distance ≤ k ) over character edits,
      token swaps, whitespace, punctuation, paraphrase. This is Q4
      operationalised, and the answer to "is this just a hash".
      §6.3: it is also whether operator competence is *possible*. If nearby
      strings do not have related destinations, no habit becomes a method
      and no amount of design repairs it. Load-bearing for playability in a
      way its position in this list does not convey.

Q6  Does ρ=I dominate ρ=R in coverage at all |C|, or does it invert?
      PARTIALLY ANSWERED: §8.2, §8.2.1. ρ=I coverage tracks terminal-token
      diversity; ρ=R tracks whole-string content. The ordering is observed
      both ways — ρ=R ahead 3:1 on WikiText-103 at 40k, ρ=I ahead at 407k on
      the scraped corpus — but the two runs differ in corpus as well as in
      scale, so *crossover* and *composition* are confounded and neither is
      established. Remaining: run one corpus at both scales, report
      terminal-token entropy at each, and predict the crossover |C| (if any)
      from entropy alone.
      Also open from §8.2.1: β_bos moves coverage by a few percent while ρ
      moves it threefold. Whether the quadrant model is really a 2×2 or a
      2×(minor perturbation) is worth stating once measured.
      NOTE: §8.2.1's source data is lost, so the 40k figures cannot be
      re-examined and the crossover comparison must be rebuilt from a fresh
      run rather than from the archive. The nineteen full-book caches held
      alongside `the_sea` are a corpus-composition instrument that has never
      been read: terminal-token entropy per corpus against coverage per
      corpus is computable from files already on disk, with no forward pass
      at all, and would separate composition from scale for the ρ=I half.

Q7  Coverage asymmetry. Map ℛ̂(π_h) against ℛ̂(π_a) under identical channel
      restriction. Do different architectures induce divergent samplers over
      J — regions reachable only by particular cognitive geometries?
      REQUIRES: at least one condition with matched elicitation. Humans
      typing chat against agents prompted to explore measures two task
      framings, not two intelligences. Report N3 against shared P_ref, not
      only against each other.

Q8  Incidental vs targeted. Compare 'human-optimised' (π_ho) strings against
      'human-incidental' (π_hi) — ambient, task-irrelevant language.
      PARTIALLY ANSWERED, negatively: §4.5. No set of systems accessible to
      incidental human language and closed to search was found, under the
      search regime and budgets used. Candidates fell as the budget rose;
      part of the surviving residue is demonstrably seed-dependent (2078).
      The universal form — that no such set exists — is not established and
      is not establishable by a searcher with a 12% failure rate on known
      targets.
      The surviving question is cost, and it is sharp: Melville reaches a
      system in eleven words at Δ=0.17; ascent reaches the same class of
      system in 32 tokens of salad at Δ=0.03, after ~164 s of GPU. Quantify
      that exchange rate. What does a human buy, per system, that search
      must pay for?

Q9  Mode collapse. Does π_a saturate high-activation attractors (1888) while
      under-reaching the incidental tail?
      REQUIRES: matched |τ(s)|. By §1.1 the ρ=R running max is a ratchet, so
      longer strings set a high bar more often and concentrate on
      high-ceiling neurons regardless of what wrote them. Bucket by token
      count or the result is baked in — the confound has a derivation, not
      just a suspicion.
      Length is not the only channel, and the chain should be measured link
      by link rather than jumped:

        |τ(s)| → terminal-token distribution → lexical distribution
               → syntax → activation distribution → destination

      §8.2 already establishes the second link for ρ=I. The rest is open.
      NOTE: 'agent output is longer and more fluent than player chat' is an
      assumption, not a measurement. It is also the assumption the whole
      question rests on. Measure it first — it is a corpus statistic and
      costs nothing.

Q10 ℛ*. **The functional sense (§4.6).** Does destination identity predict
      causal downstream sensitivity under intervention — patch, ablate, or
      amplify A[:,j] and measure the effect? This distinguishes "wins the
      argmax" from "is a computational locus", and it is the step that makes
      any of this interpretability rather than a lookup table.
      Two versions, and only the second is the one wanted:
        correlational — does destination correlate with downstream effect?
        causal        — does intervening on the winning neuron change
                        downstream behaviour?
      CRITERION: for strings mapping to the same j, intervention on A[:,j]
      must produce a reproducible downstream effect *relative to matched
      control neurons*. Finding that 1888 has a large causal effect on its
      own establishes something about 1888, not about destination identity
      being meaningful.
      SCOPE: settles the functional sense only. The indexical sense (§4.6)
      does not wait on it and is unaffected by either outcome.
      PRIOR: Gurnee & Tegmark (2023) recover interpretable single neurons
      by supervised regression against known coordinates. Q10 has no such
      ground truth and cannot be run that way; see *Provenance of the
      question*. Their result is a prior that this sense is sometimes
      satisfiable, not a finding about `J`.

Q10' **The referential sense (§4.6).** Does j correspond to a describable
      property of the input — is there a compact rule, short of evaluating
      A, that names the destination?
      Split out because it was previously folded into Q10 and is a different
      question, pointing upstream where Q10 points downstream. A neuron may
      be causally load-bearing and referentially opaque, or the reverse;
      neither result implies the other.
      The benchmark is already stated: §8.1's binary winner-vs-runner-up
      task on the low-Δ subset, chance 50%, held-out split, threshold fixed
      independently, with the selection caveat recorded there. §8.1 predicts
      no cheap predictor beats chance. Unrun.
      A positive result does not refute Q10 and is not evidence about ℛ*.
      It refutes "the string is a key, not a description", which is a
      separate claim this document has already withdrawn once.

Q11 1888. Margin distribution, positional dependence, token-frequency
      correlation, behaviour at other ℓ. Currently a frequent winner with no
      established meaning.

Q12 Canonicality. Is  im(D|Σ*_b)  a proper subset of  im(D|V*) ?
      Equivalently: does any system admit a token route and no string route —
      an address blocked not by θ but by the tokenizer standing in the door?
      §4.5.2 gives an upper bound of 249 from an unconstrained search. The
      test is a canonically-constrained ascent against those 249. The
      literature has the parts — canonical vs non-canonical tokenization,
      the non-recovering property — and does not appear to have asked the
      coverage question, because it takes an address space to make it
      visible.

Q13 Is Δ optimisable? Re-run §4.5 with the objective changed from *reach j*
      to *reach j with maximal margin*. If the hard set can be lifted from
      Δ≈0.03 to Δ≈1.5 by search alone, route quality is manufacturable and
      §6.2 is void. If Δ is bounded per-system by something structural, the
      tiering in §6.2 is load-bearing and the bound itself is a result.
      Cheapest experiment in this document with the largest design
      consequence.
```

## Epilogue — A Rider's Inheritance

*In-universe. Fiction.*


There had been greater success in understanding the functioning of the wormhole gates. These were clearly designed to be operated, as it were, manually. Certainly, it was only possible to operate the gates with the assistance of formidably powerful and dedicated computer systems. Regardless, there was no artificial intelligence, or anything like it, in control of the gates or even connected in any way to their fundamental workings. The designers, whoever they were, had been scrupulous in keeping the intelligent systems that did exist throughout the lattice entirely firewalled off from gate operating systems.

The intent was quite clear: the gates were only to be opened by living, breathing human beings.
