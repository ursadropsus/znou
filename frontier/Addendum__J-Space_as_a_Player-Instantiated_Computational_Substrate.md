# Addendum: J-Space as a Player-Instantiated Computational Substrate

**Status:** PROPOSAL. Nothing in this addendum is claim-bearing about `D`, θ, or
the model.
**Scope:** Frontier deployment and player-created infrastructure.
**Relationship to the core specification:** This addendum does not alter the
definition, measurement protocol, or scientific claims of J-Space. It considers
how a deterministic computational object of the form `D(s) → j` might be
instantiated inside an extensible virtual world, and what that instantiation
would require of the specification.

It uses the claim tags of the main document. Every section here is PROPOSAL
unless it is restating a MEASURED result from the spec, in which case the
section reference is given and the number is the spec's, not this document's.

---

## A.0 What this addendum inherits

An earlier draft of this addendum treated game design as unclaimed territory.
It is not. §6, §6.1, §6.2, §6.3 and §4.6 already constitute a design argument
with results, conditions and several explicit withdrawals in it. This addendum
inherits that argument rather than re-deriving it. Five inherited commitments
govern everything below.

```
INHERITED-1   Scheme B, not Scheme A (§6). Reward attaches to what happens
              after arrival, never to the act of producing a string that
              arrives. Two independent derivations: D cannot authenticate an
              author (I7), and D cannot be withheld (I5, I6).

INHERITED-2   The one-way rule (§4.6). The indexical and projected senses of
              "meaning" may never be offered as evidence for the functional or
              referential senses. This addendum operates almost entirely in the
              indexical sense and is therefore the most likely place in the
              project for that crossing to occur.

INHERITED-3   No metric on J (§5). No lens claims one. λ_edit (§5.1) is the one
              candidate for a real geometry and is unbuilt. Language in this
              addendum that sounds spatial — route, navigation, distance — is
              shorthand for reachability, not adjacency.

INHERITED-4   D is invariant across two tested environments (I5, §8), which is
              not a portability theorem and does not become one by being
              deployed.

INHERITED-5   Δ is endpoint-local and its usefulness is conditional on Q13
              (§6.2). If margin is directly optimisable, every mechanic that
              rests on it collapses.
```

Where this addendum has anything to add, it is because of one asymmetry: the
spec's design sections assume the designers control the consequences. In an
extensible world they do not. That is the question this document exists to ask.

---

## A.1 Motivation

The core specification deliberately treats `D` as a computational object rather
than a game mechanic. Given a string `s`:

```
D(s) = j ,    j ∈ {0, …, 3071}
```

The scientific questions concern properties of this mapping: reachability,
robustness, input sensitivity, destination distribution, causal relevance, and
whether the structure is learnable by humans or artificial agents.

A separate question follows from the intended deployment environment:

> **What happens if inhabitants of a programmable virtual world are given
> access to the computation, but are not given a prescribed interpretation of
> its outputs?**

This matters for EVE Frontier because Frontier is explicitly designed around
third-party development, Smart Assemblies, composability, and player-created
functionality: players are meant to create structures, economic loops, missions
and game modes while the underlying rules of the universe remain constrained by
its digital physics.

The purpose here is not to propose J-Space as an official Frontier feature. It
is to investigate a weaker and more interesting proposition:

> **J-Space may be a computational substrate that independent inhabitants could
> instantiate into their own infrastructure, applications, economies and social
> practices.**

If so, much of the resulting game does not need to be designed in advance. What
does need to be designed in advance is the boundary — which is §A.13.

---

## A.2 Frontier as an instantiation environment

Frontier has migrated from its original Ethereum-based architecture to Sui.
Smart Assemblies are programmable, on-chain objects governed by Move programs,
and third-party developers can write code that runs on structures inside the
live universe. There are two distinct integration surfaces, and the difference
matters more for J-Space than any Assembly type does:

```
IN-WORLD    Move programs running on Smart Assemblies. Constrained by what
            can execute inside the game's consensus machinery.

EXTERNAL    Tools connecting to the live universe through the official API —
            maps, coordination dashboards, analytics services.
```

The second is the plausible home for a GPT-2 oracle. See §A.4.

CCP is additionally using Walrus for data availability and Seal for native data
access controls. Seal is the obvious primitive for the attestation and
selective-disclosure mechanisms sketched in §A.4 and §A.7, and its existence is
the reason those sketches are less speculative than the previous draft assumed.

The important property for J-Space is not any individual Assembly type. It is
the separation between:

1. the object provided by the game,
2. the programmable rules governing its interaction, and
3. the application or social purpose imposed by its owner.

Which gives the architectural analogy:

```
language  →  J-Space destination  →  player-defined rule  →  Frontier consequence
```

The interpretation of the destination is external to `D`. The consequence is
external to both.

---

## A.3 Three layers

### A.3.1 The Oracle — belongs to the specification

```
s ↦ D(s)
```

Model revision, tokenizer, inference procedure, readout, precision, and
everything else required for reproducibility. §7 is normative here. The oracle
does not need to know anything about Frontier, and §A.13 says it must not learn.

### A.3.2 The Adapter — belongs to whoever builds it

A Frontier application consumes a verified `(s, j)` result and associates it
with an in-game condition. The precise mechanism depends on the current SDK,
Assembly APIs, deployment permissions and Digital Physics constraints, none of
which the specification should assume.

**A worked example of how to get this wrong.** The previous draft used
`D(s) = 1888` as its illustration, and it is the worst available choice. §4.3
measures that 90.6% of single tokens land on 1888 under ρ=R. A gate conditioned
on reaching 1888 opens to very nearly any word a passer-by types. It is not a
lock; it is a door marked *push*.

The general lesson is not "pick a rarer destination." It is that **destination
identity is not an access-control primitive at all**, for reasons the spec
already measured:

```
I3 + §3        D is spectacularly non-injective. Fibers are unlistable, but
               they are also enormous. Common destinations have accidental
               preimages in ordinary text.

§4.5           Targeted search reaches 2805 of 3072 destinations in eight
               hours on one laptop — 91.3% as strings, 99.4% as token
               sequences. Any destination a designer can name, a searcher can
               reach overnight.

I5 + I6        A preimage, once presented, is public and permanently valid.
               There is no revocation and no rate limit that is not social.
```

A preimage is a solved puzzle, not a secret. Adapters should be designed on the
assumption that any stated destination condition will be satisfied by someone
within a day, and by everyone thereafter.

### A.3.3 The Social Layer — §4.6's indexical sense

Once a destination has a persistent in-game consequence, players can assign
meanings to it. One group treats `j` as a waypoint; another as a shibboleth;
another builds an economy around access; another a mythology.

This is not a new idea introduced by this addendum. It is the **indexical
sense** of §4.6, which defines it more carefully than anything here does, gives
it a test status (*none required — it is observed or it is not*), and grounds it
in Jita 4-4: a station in a system of unremarkable statistics that became the
market of its game because it became the market of its game. §4.6 also records
the Midjourney `--sref` prior art, where the same accretion happened on an
address space with no semantics, no ledger and no economy attached.
Distinctiveness plus a shareable name appears to be the whole recipe.

A destination can therefore possess **projected and indexical meaning without
possessing functional or referential meaning in the model.** §4.6's one-way rule
governs: that fact may never be run backwards. See §A.10 and §A.13.

---

## A.4 Verification without native model execution

The canonical implementation relies on frozen GPT-2 inference under §7's pinned
stack. Smart Assemblies are Move programs on Sui. There is no basis for
assuming an arbitrary PyTorch/Transformers stack executes inside a Smart
Assembly, and no reason to want it to: a deployment model that puts the model
into the game's consensus machinery inherits every reproducibility risk in §7
as a consensus risk.

Instead, treat J-Space as an externally reproducible computation. At minimum a
claimant provides `(s, j)` with `D(s) = j`, and any independent verifier
reproduces it. The external-API surface described in §A.2 is where such a
verifier belongs. Signed attestations, replicated verification services, or
Seal-mediated disclosure are implementation possibilities, not requirements of
the specification.

The property to preserve:

> Frontier need not understand GPT-2 for Frontier inhabitants to build systems
> around GPT-2-derived outputs.

**The failure mode is not cost, it is silent disagreement.** The previous draft
anticipated verification being *cumbersome*. Cumbersome is survivable. The
unsurvivable case is two honest parties computing different answers. §6.1 states
the residual risk precisely — *the residual is configuration, not secrecy: a
client running TF32 will disagree with the server, and that is a bug to detect
and report, not a feature to sell.* Under a deployment where a Smart Assembly
releases assets on a claimed pair, "a bug to detect and report" becomes a
custody dispute with no adjudicator.

Three consequences, all of which fall on the specification rather than on
Frontier:

```
1  Any verifier must assert its own configuration before asserting a result.
   §7 disables TF32 and reduced precision; a verifier that does not check this
   at runtime is not a verifier.

2  I5 is a result about two tested environments. A third arm (§7.1) is a
   prerequisite for infrastructural use, not an optional strengthening. See
   §A.11.

3  Disagreement must be detectable rather than merely improbable. Publishing a
   fixture set with expected destinations — §8's 7353 sentences already are
   one — lets any adapter self-test before it holds anything of value.
```

---

## A.5 Determinism, and the exact scope of it

If independent actors use J-Space as an infrastructural primitive, the valuable
property is:

```
D_A(s) = D_B(s)     for independent implementations A and B
```

The model pinning and replay validation of §7 and §8 therefore serve a second
purpose: they establish a candidate **canonical oracle**.

**State the scope honestly, because deployment will not.** I5 says `D` was
invariant across the two tested environments — 7353/7353 sentences, two
machines, months apart, down to a margin of 1e-6 — and says in the same breath
that this is *a result about those two environments, not a general portability
theorem*. "Portable across strict-fp32 stacks" is not established. A deployment
that describes the oracle as canonical is making a claim one arm wider than the
evidence, and the way to fix that is to add the arm, not to soften the wording.

Determinism is also not trustworthiness. A deterministic oracle can be
faithfully reproduced while being scientifically uninteresting, strategically
exploitable, or unsuitable for the application built on it. The three
properties are independent and the specification only supplies the first.

---

## A.6 Consequence design: why preimage-bounties are the wrong primitive

The obvious use of J-Space is to make destinations into keys. §6 has already
analysed this and rejected it, and the analysis does not weaken when the payout
is onchain — it sharpens, because the reward acquires a market price.

```
Scheme A   reward = f(first s such that D(s) = j)
           ⇒ argmax_π E[reward] tends to π = G, the machine generator
           ⇒ the human sample is contaminated in proportion to |reward|

Scheme B   reward = f(in-world actions after arrival at j)
           ⇒ value(s) = option value of access to j
           ⇒ π = G yields locations, not payout
```

"A Smart Assembly releases an item to anyone who provides a valid preimage for
`j`" is Scheme A. So is "a corporation issues reputation for discovering
previously unknown destinations." Both pay for the string. By I7 the adapter
cannot tell who wrote it, and by I6 the model is a download, so the payout goes
to whoever runs the largest search — which §4.5 puts at an evening on a laptop
for most of the address space.

**The Scheme B form of the same mechanisms.** In each case the fix is to move
the consequence past the arrival:

```
NOT   the gate opens because you presented a preimage
BUT   the gate opens to anyone; what is scarce is what lies beyond it and
      what you can hold once you are there

NOT   reputation for discovering a destination
BUT   reputation for operating one — the discovery ledger records, the
      exploitation ledger pays (§6)

NOT   a market in preimages
BUT   a market in the things access enables, which saturate under traversal
      capacity κ rather than under search throughput r_G
```

**The one honest argument for revisiting this, which the specification itself
supplies.** §6 flags that *locations saturate is itself an assumption*, holding
only if exploitation capacity κ is the binding constraint, and weakening "if
locations are instead defensible, colonisable, monopolisable or usable as
infrastructure." Frontier is precisely a world in which they would be. If a
destination can be held, fortified, taxed and inherited, the marginal value of
the (k+1)-th location may not decay, and Scheme B's argument softens.

That is a real argument and it is the strongest thing this addendum has to say
about §6. It is not an argument for Scheme A. It is an argument that in a world
with property, the scarce object migrates from *the route* to *the tenure* —
which is still after the arrival, and still not the string.

**Puzzle chains are the exception worth naming.** A sequence in which each
solved destination reveals the condition for the next is Scheme A in form but
carries no economic contamination if the payout is narrative rather than
material. The distinction is not the mechanism, it is `|reward|`. §6's
contamination term is proportional to it.

---

## A.7 What is actually scarce — PROPOSAL, conditional on Q13

Not all destinations cost the same to reach. This is not a possibility; §4.5.1
measured it:

```
Melville routes, natural English        median Δ  0.1676     n=545
sweep, light budget                               0.0810     n=2688
hard-172, heavy budget                            0.0313     n=117
```

Monotone, and the spec's reading is that **cost of access and precariousness of
arrival are the same gradient** — 91% of heavy-budget certificates land within
0.1 of a different system. The previous draft treated search cost and arrival
margin as two independent axes of scarcity. They are one axis measured twice.

A second, genuinely independent axis exists and is *rarity of reach by natural
language*: §8.3's 407,475-sentence corpus reaches 2336 of 3072 destinations in
union, leaving 736 never visited — which is a property of that corpus and not a
property of J, since §4.5 reaches almost all of them under search.

The suggested hierarchy:

```
destination  →  preimage  →  route  →  mastery
```

**The middle two links are exactly what has not been established.** Whether a
route is a durable good depends on Q5 (does a small edit preserve the
destination?) and Q13 (can margin be manufactured by search alone?). §6.2 is
explicit: if a searcher told to maximise margin can lift a Δ=0.03 route to
Δ=1.5, route quality is manufacturable and every mechanic resting on it
collapses. §6 also separates the two objects that the word *route* conflates —
arrival margin Δ is endpoint-local and MEASURED; route quality is CONSTRUCTED,
some `f(Δ, |τ(s)|, plausibility under a reference LM, measured repeatability)`,
not yet defined and not derivable from Δ alone.

Until Q13 runs, this section proposes nothing that should be built.

If it does hold, the scarce object is not the destination and not the preimage.
It is **knowledge of how to reach a destination reliably** — which is compatible
with a player-driven world precisely because knowledge can be copied, withheld,
sold, falsified, monopolised or rendered obsolete by better methods.

---

## A.8 The route as a player-created resource — conditional on Q5

If Q5 establishes that controlled modification of an input produces non-random
and learnable change in destination, a route can be represented not as a pair
but as a sequence:

```
s₀ → j₀ ,  s₁ → j₁ ,  … ,  sₙ → jₙ
```

where successive strings differ by some transformation. Different players might
find different such sequences between the same endpoints; some short, some
robust, some requiring unusual linguistic knowledge, some easy for language
models and hard for humans, some the reverse.

**This object already has a name in the specification.** §5.1 sketches `λ_edit`:

```
nbr(j) = { j′ : ∃ s,s′ with D(s)=j, D(s′)=j′, d(s,s′) = 1 }
```

— adjacency induced by the string space rather than imposed on J, and
explicitly the *one* candidate for a genuine metric on J in a document that
otherwise claims none. It is expensive to sample and unbuilt. Any Frontier
route economy is an economy over `λ_edit`'s edge set, and the spec already
describes how one would construct it.

§6.3 states the dependency in the plainest available terms: intuition of the
Sailwind kind requires that nearby actions have related outcomes, and if a
one-character edit relocates the destination arbitrarily then no habit becomes
a method, no operator improves, and the mechanic is a lottery with an elaborate
ticket. The fixtures cut both ways — `it was cold` → 38 and `it was coldest` →
38 encourage; `it was colder` → 1888 sitting between them discourages.

The scientific experiments determine whether these distinctions exist. The game
environment determines whether anybody cares. Neither substitutes for the other.

---

## A.9 Human and agent ecologies — and why deployment cannot measure this

Both populations can be given the same interface, `s → D(s)`, and compared by
their induced distributions `P_human(j)` and `P_agent(j)` rather than by a
single score. They may discover different portions of the address space, develop
different linguistic strategies, exploit different robustness, find routes the
other does not.

**A live economy cannot be the instrument for this, and the reason is
derivable rather than merely suspected.** The previous draft asserted that the
infrastructure privileges neither population because "the oracle simply
evaluates strings." In the relevant sense that is false. §1.1 makes the ρ=R
running max a ratchet, so longer strings clear a high bar more often and
concentrate on high-ceiling neurons *regardless of what wrote them*. Q9 states
the confound as a requirement — bucket by token count or the result is baked in
— and Q7 adds the elicitation requirement: humans typing chat against agents
prompted to explore measures two task framings, not two intelligences.

Nobody in a live economy buckets by token count or matches elicitation. So:

```
The mesocosm (§6.1) remains a controlled instrument standing beside any
deployment. It is not replaced by one, and observational data harvested from
a deployed adapter is not evidence about Q7 or Q9.
```

This is a firewall item and appears again in §A.13.

Note also that the load-bearing assumption underneath Q9 — that agent output is
longer and more fluent than player chat — is still unmeasured, and is a corpus
statistic costing nothing.

---

## A.10 Player sovereignty and emergent semantics

The model supplies an index. Players supply interpretation. Infrastructure
supplies consequences.

```
model output  →  social interpretation  →  material consequence
```

This is compatible with Frontier's design philosophy, in which the world's
underlying rules stay stable while inhabitants build new experiences on top of
them. J-Space need not prescribe what a destination *means*; meanings can
emerge through coordination. If enough inhabitants independently agree that
some `j` is valuable, dangerous, sacred or strategic, that interpretation
becomes consequential even if nothing about neuron `j` intrinsically encodes the
concept.

**Two conditions this places on the project, neither of which is optional.**

**First, coordination requires shared referents, which the current render does
not provide.** §5 discloses that `b_j` is drawn per client at first run and
persisted in that client's saved state, so two operators hold different skies:
angular descriptions do not transfer, screenshots are not comparable, and a data
purge regenerates the sky and destroys any constellation an operator has learned
to see. §5 records this as an open design question and notes that deriving `b_j`
from a hash of `j` keeps exactly zero information from θ, leaving §5's claims
untouched either way.

This addendum's thesis decides that question. A substrate whose entire proposed
value is accreted shared meaning cannot be rendered in a sky that no two
inhabitants share. Seed it. See §A.11.

**Second, the one-way rule (§4.6) is at its most fragile here.** Once a
destination has an onchain consequence and a corporation's mythology attached,
"`j` means cold" becomes a load-bearing social fact that is indistinguishable
in conversation from a claim about the model. §4.6 records that almost every
overclaim withdrawn between v6 and v14 was some version of that crossing. The
guard runs one direction only:

```
indexical, projected  ↛  functional, referential
```

Emergent meaning is an empirical question about player behaviour. It is never
evidence about model semantics, however much of it accumulates, and however
materially consequential it becomes.

---

## A.11 Consequences for the published artifact

Three items the specification currently treats as optional become prerequisites
if the deployment hypothesis is taken seriously. They are listed here so that a
reader of this addendum knows what it is asking for.

```
SEED b_j                    §5, open design question. Decided by §A.10:
                            derive from a hash of j into the same cube.
                            Universal, shareable, stable, and information-
                            preserving-of-nothing. §5's claims unaffected.

THIRD-ENVIRONMENT REPLAY    §7.1. I5 covers two environments. The §8.4 runs
                            were made on torch 2.4.0+cu121 / RTX A4500 /
                            Ubuntu 24.04 and are reconstructible from the
                            RunPod snapshot; replaying under §7's current
                            stack widens the invariance claim across a torch
                            major version and two GPU architectures. Cheapest
                            available strengthening of the claim this
                            addendum leans on hardest (§A.5).

PUBLISHED FIXTURE SET       §8's 7353 sentences with expected destinations,
                            packaged as a conformance test any adapter can
                            run before holding anything of value (§A.4).
```

None of these changes `D`. All three are things a stranger building on the
oracle would need and currently would not have.

---

## A.12 Failure modes

The previous draft listed five. Two of them are already measured and should not
be presented as risks.

### A.12.1 The mapping is effectively unlearnable — OPEN, Q5

If small linguistic modifications produce essentially arbitrary destinations,
players cannot develop useful intuitions and J-Space functions as an oracle or
lookup system rather than a navigable space. §6.3 states this as the condition
the whole design bets on. Unrun.

### A.12.2 Search dominates understanding — OPEN, partially answered

Players or agents may find that brute-force optimisation beats linguistic
reasoning, turning J-Space into a computational resource market rather than an
exploratory environment. §4.5 and Q8 already bear on this: no set of systems
accessible to incidental human language and closed to search was found. What
survives is the *exchange rate* — Melville reaches a system in eleven words at
Δ=0.17; ascent reaches the same class in 32 tokens of salad at Δ=0.03 after
~164 s of GPU. What a human buys per system, that search must pay for, is the
open and sharp question.

That outcome would itself be informative.

### A.12.3 Concentration — MEASURED, and not a failure mode

The previous draft asked what happens *if* natural language concentrates
activity onto a small subset of neurons. It does, and the spec measured it:

```
§4.3   90.6% of single tokens land on 1888 under ρ=R.
       ≤6.3% of systems are reachable by any single token in any quadrant.
§8.3   407,475 sentences reach 2336 of 3072 in union; 736 never visited.
```

§4.6 has also already reframed what this means for a deployed world, and the
reframing is stronger than treating it as a defect: *Jita 4-4 is the most
congested place in its game and is not a bug there.* Under the indexical sense,
concentration is the mechanism by which this addendum's own thesis would come
true, not an obstacle to it. Hubs are what accreted meaning looks like.

The problem that survives is narrower and more tractable, and §4.6 states it:
whether a first session gives an operator a reason to leave the busy core, and a
legible sense that leaving is possible. That is a design problem about
onboarding, not a defect in `D`.

### A.12.4 Verification becomes the bottleneck — see §A.4

Restated there, with the correction that the dangerous case is silent
disagreement rather than cost.

### A.12.5 The underlying neurons have little causal significance — OPEN, Q10

Q10 may establish that winning-neuron identity has limited functional
significance. That would weaken any reading of J-Space as literal
model-internal geography.

It would not invalidate its use as a player-created substrate, and §4.6 says why
in advance: *if Q10 comes back flat — destinations are computationally inert
argmax winners and there is no ℛ\* — the indexical sense is untouched. A trade
hub does not need its coordinates to signify.*

This is the single most important structural property of the proposal. The
deployment hypothesis is **independent of Q10 in both directions**: a positive
Q10 does not license the mythology, and a negative Q10 does not destroy it.

---

## A.13 The firewall

The deployment hypothesis must remain independent of the scientific results.
The project must not modify `D`, the corpus, the measurement protocol, or the
interpretation of results in order to make a Frontier application more
compelling.

```
1  J-Space is not designed to produce useful game locations merely because
   such locations would be desirable.

2  Destination distributions are not artificially flattened for gameplay.

3  Hard-to-reach destinations are not made easier without documenting a change
   to the underlying computation.

4  Semantic interpretations are not presented as properties of the model
   without evidence meeting §4.6's functional or referential criteria.

5  Frontier integration is never evidence that J-Space possesses meaningful
   internal geometry — and specifically, accreted player consensus about a
   destination is indexical evidence only, under §4.6's one-way rule. However
   large the economy grows around some j, it says nothing about neuron j.

6  Observational data from a deployed adapter is not a measurement of Q7 or
   Q9. Those require matched elicitation and matched |τ(s)| (§A.9). The
   mesocosm of §6.1 stands beside any deployment and is not replaced by it.
```

Items 5 and 6 are additions to the previous draft's list and are the two the
architecture makes structurally easy to violate.

If the resulting structure is useful, that usefulness should emerge from the
measured properties of the oracle. If it is not useful, that is also a valid
experimental outcome.

---

## A.14 A possible long-term architecture

Four independently replaceable components.

**J-Space Core** — the canonical deterministic computation `D(s) → j`, under §7.

**Verification Layer** — tools by which independent parties establish that a
claimed input/output pair is valid under the canonical implementation, and,
equally important, that their own stack is configured to compute it (§A.4).

**Frontier Adapters** — player-created infrastructure consuming verified
destinations and mapping them onto supported game actions, under Scheme B
(§A.6).

**Applications** — the economic, social, navigational, competitive and cultural
systems inhabitants build on those adapters.

The scientific project defines the first layer and the constraints on the
second. The community experiments with the third and fourth. The final layer is
deliberately unspecified.

---

## A.15 The stronger hypothesis

The interesting hypothesis is not:

> "GPT-2 contains a hidden game world."

Nor:

> "J-Space should become an EVE Frontier minigame."

It is:

> **A deterministic but opaque computation derived from a language model may
> provide a sufficiently structured interface for independent agents to create
> their own navigational, economic and social systems.**

If true, the game is not authored entirely by the designers of J-Space. The
designers provide a rule. The model provides a substrate. The infrastructure
provides enforceable consequences. The inhabitants discover what the substrate
is good for.

That would make J-Space less like a game mechanic and more like a small piece of
computational physics — with the caveat §5 insists on, that *physics* here means
a stable rule and not a geometry. Its eventual meaning would be determined by
what its inhabitants learn to do with it, in the indexical sense of §4.6 and in
no other sense.

---

## A.16 Current status

No claim is made that Frontier currently supports every mechanism described
above, nor that a J-Space implementation exists within Frontier.

What is established about the environment: Frontier runs on Sui with Smart
Assemblies as programmable player-facing infrastructure written in Move;
third-party developers can deploy code that runs on structures in the live
universe; there is an official API for external tools; and Walrus and Seal
provide data availability and native access control. The stated design
philosophy emphasises composability, extensibility and player-created
experiences.

The feasibility of any particular J-Space integration is an engineering question
to be tested against the current SDK, Assembly APIs, deployment permissions,
verification mechanisms and Digital Physics constraints. Frontier's surface is
under active development and every claim in §A.2 and §A.16 should be re-checked
against current documentation before anything is built on it.

The research question comes first. Q5, Q10, Q13 and the pin all sit upstream of
every mechanism proposed here.

The game, if one emerges, comes later.

---

## Changes from the previous draft

```
NEW      §A.0   inherited commitments; the addendum now sits on §6 and §4.6
                rather than beside them
NEW      §A.11  what the deployment hypothesis asks of the published artifact
NEW      §A.13  items 5 and 6 (one-way rule; deployment ≠ mesocosm)

REWRITTEN
         §A.3.2 1888 removed as the worked example; replaced with why
                destination identity is not an access-control primitive
         §A.6   preimage-bounties identified as Scheme A and replaced with
                Scheme B forms; §6's "locations saturate is an assumption"
                surfaced as the one live counter-argument
         §A.7   cost and margin identified as one axis, not two (§4.5.1);
                whole section made conditional on Q13
         §A.9   the ratchet confound (§1.1, Q9) and the elicitation
                requirement (Q7) added; deployment excluded as instrument
         §A.10  grounded in §4.6's indexical sense; b_j seeding decided
         §A.12.3 concentration moved from hypothetical risk to measured
                finding, reframed via §4.6's Jita argument
         §A.2   Sui migration, external API, Walrus/Seal; §A.16 updated
                accordingly and made less cautious than the previous draft

FORMAT   display math moved from \[ \] to fenced code blocks, matching
         SPEC.md and rendering correctly on GitHub
         claim tags adopted from the main document
```
