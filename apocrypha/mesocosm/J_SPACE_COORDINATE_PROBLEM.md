# J-Space as a Coordinate Problem

> **Speculative note.** Nothing here is a claim about what has been measured.
> Where this document says *if*, it means *if*. The experiments that would
> settle each conditional are named as they arise, and gathered at the end.
>
> A companion note to `J_SPACE_INTEROPERABLE_DISCOVERY.md`, arriving at the
> same territory from the opposite direction: not what a shared space could be
> used for, but what would have to be true before there is a space at all.

---

## 1. The precedent nobody finds strange

Independent games already share mathematical objects and render them
differently. This is not a proposal; it is Tuesday.

Perlin noise was published once and implemented thousands of times. One game
renders it as mountains, another as cloud cover, another as cave systems, wood
grain, or the distribution of ore. Nobody regards it as remarkable that a
voxel sandbox and a flight simulator draw their terrain from a common
function. No studio negotiated with another. No asset crossed a boundary. They
agreed on a specification and diverged completely on what to do with it.

No Man's Sky supplies the architectural half of the same idea. Its universe is
deterministic from a seed, so two players standing on the same planet are not
consulting a shared database of planets — they are each regenerating the same
object from the same function. The planet is not stored anywhere. It is
*computed*, identically, on both machines.

This matters because the most obvious objection to a shared discovery layer —
*why would independent studios ever agree on a common substrate?* — has
already been answered by practice. They do it routinely, they do it without
coordination, and they do it because a published function is cheaper than an
integration.

So the ambition here is not novel in kind. What would be novel is the
*coordinate space*.

---

## 2. The generator run backwards

Procedural generation runs in one direction:

```
coordinates  →  content
```

You begin already holding the coordinate space. It is ℝ², or ℝ³, or a lattice.
It arrives with a metric attached, for free, before any function is evaluated.
"Nearby" is guaranteed by the space itself. Noise functions are then built to
be smooth over that space, because smoothness is what makes output resemble
landscape rather than static. The territory precedes the generator.

`D` runs the other way:

```
content  →  label
```

Strings go in. They have a natural metric — edit distance, and coarser
semantic ones. What comes out is an index into 3,072 neurons, and that index
set has no order whatsoever. Neuron 1847 and neuron 1848 are not adjacent in
any sense. Their numbering is an artifact of initialisation. Nothing was
arranged.

This is the whole problem, stated compactly:

> **Procedural generation shares a function defined over a territory.
> J-space shares a function into a label set, and hopes a territory falls out.**

What exists today is a reproducible naming scheme over 3,072 names. Calling
those names *locations* is a bet, not a description. The bet is that the metric
on the input side survives the mapping well enough to induce one on the output
side — that strings which are close land at destinations which are related.

That is not a philosophical question. It is `Q5`, and it is measurable: sample
first-arrival strings, perturb them at bounded edit distance, and report
`P(D(s') = D(s) | edit distance ≤ k)`. A few thousand forward passes. Until
that number exists, "nearby loci" is a phrase with no referent.

---

## 3. What makes an address portable

Two properties, and they are easy to overlook because neither is glamorous.

**Discretisation.** A 768-dimensional float vector cannot be typed into a
forum post, held in memory, tattooed, argued over in a pub, or shouted across
a Discord channel. An integer can. Embedding spaces have been proposed as
shared semantic substrates many times and remain confined to machines, partly
because a point in one is not the kind of object a person can carry. Taking an
argmax destroys almost all the information in the activation vector — and that
destruction is exactly what produces something a human can hold. `J-1847` is a
name. A vector is a file.

**Reproducibility.** A name is worthless if it means different things on
different machines. The claim that makes the address trustworthy is the
unglamorous one in §8: the same corpus evaluated on different hardware, months
apart, produced identical destinations for every sentence, down to winning
margins of 1e-6. That result reads like a housekeeping check. It is in fact the
load-bearing wall. Without it there is no shared anything — only each machine's
private opinion about where a string goes.

So the primitive is not "a neural network has interesting internal structure."
The primitive is: *a public frozen model plus a pinned procedure yields a small
integer that two strangers can independently compute and agree on.*

The pin is therefore not bureaucracy. It is the address's guarantee of
meaning, and any drift in the stack is a drift in the territory. This is why
`§7` is normative, and why its being unenforced in the current code is the
first thing that needs closing.

---

## 4. Why the space is small, and why that may be the point

Roughly 1,450 of 3,072 destinations are reached by a 407,475-sentence corpus in
one quadrant. Across four quadrants the ceiling is 12,288 addresses, and the
practical figure is far lower.

As a *territory* this is derisory. Any content-addressable scheme built on
hashing offers 2^256 addresses and guarantees no collisions. J-space offers
about the population of a village and guarantees collisions constantly. Every
popular phrasing lands somewhere already occupied.

Treated as a database, this is fatal. Treated as a *place*, it may be the most
interesting property in the design.

A space with a billion addresses is a filing system: everyone gets their own
coordinate, nobody meets, nothing accumulates. A space with fifteen hundred is
a small town. Two people who have never spoken arrive at neuron 38 by
completely different routes — one via `it was cold`, another via a landlord
who could not afford a fire — and discover they have been standing in the same
doorway. Collisions are not a failure mode of a social cartography. They are
its mechanism.

Scarcity is what allows a locus to acquire a reputation. You cannot build
folklore around an address nobody else will ever visit.

This suggests a design principle worth stating plainly: **do not expand the
address space to fix collisions.** The temptation will be to add layers,
quadrants, or models until the space is comfortably large. That trades the
only property that makes it a place for a property it does not need.

---

## 5. What is not yet true

Stated flatly, so that a reader does not have to infer it.

**There is no metric on J.** `§5` says so explicitly, and no lens in the
current implementation claims otherwise — what the lenses render is a scalar
per neuron, which is a weaker and better-supported object, as set out below.
Any phase of this proposal that requires relationships *between* loci requires
overturning that position with evidence, not with prose.

**The rendered geometry is a scalar field, not a map.** This deserves stating
precisely, because it is easy to get wrong in both directions. Under the
density and orrery lenses, a star's distance from the centre is a monotone
function of how often the loaded corpus arrives there: the dominant
destination is placed at the origin and rarer ones fall outward. That reading
is a real measurement and it is reproducible — the same corpus yields the same
centre and the same radial ordering for any operator, on any machine. Load
Moby-Dick and neuron 1888 takes the centre, because 1888 is where that corpus
predominantly arrives.

What is *not* meaningful is angular position. Direction comes from an unseeded
per-client draw, so visual adjacency between two stars carries no information
and no two operators share an angular arrangement.

This is the right way round. The map encodes what has been measured and stays
silent about what has not. It is also why `§5` survives intact rather than
being quietly contradicted by the prototype: a per-neuron arrival frequency is
a *scalar over* J, not a *metric on* J. Rendering it asserts nothing about the
relationship between any two neurons, which is exactly what `§5` declines to
claim. A scalar is enough to make a legible picture; a metric would be a much
stronger object, and none is offered.

The remaining defect is small and worth fixing for a different reason than
meaning: seeding the angular draw from a hash of the neuron index would make
screenshots fully comparable between operators and survive a data purge, while
keeping exactly zero information from θ. Nothing is claimed either way — but
right now a constellation someone learns is theirs alone.

**The similarity matrices already in the tree are an unacknowledged first
attempt at exactly this question.** `apocrypha/cold/` holds three of them. A
similarity matrix over neurons is a metric-shaped object. They should either
be explained or set aside, because at present the archive is quietly
attempting what the specification declines to claim.

**Destinations are not known to mean anything.** `Q10` — whether intervening
on a neuron changes downstream behaviour relative to matched controls — has not
been answered. The pilot data in `apocrypha/` reports causal alteration for
every neuron team tested, including teams that merely collapse into
repetition, which means it detects *any* change rather than a targeted one.
Until Q10 is answered properly, read "destination" as "the neuron that won."

---

## 6. The string is the invariant

The phrase *platform-agnostic* conceals a gap between two very different
claims, and the resolution is not the one that first suggests itself.

**Engine-agnostic** is nearly free. Publish the specification, pin the model,
and any engine can compute the same address. No technical obstacle, no
negotiation. This is the Perlin noise situation.

**Model-agnostic** looks at first like a blocker. The current address space is
not "J-space". It is *GPT-2-small, layer 5, `mlp.hook_post`, argmax*. Change
any of those and every address changes. A different model does not offer a
different view of one territory; it produces a different territory.

The mistake is to conclude from this that the model must be fixed forever, or
that the whole scheme expires when GPT-2 ages out. It does not, because **the
address is not the durable object. The string is.**

A locus is properly written as a pair:

```
(specification, address)

J-1888 @ gpt2-small · L5 · mlp.hook_post · argmax · rev 607a30d7…
```

`J-1888 @ gpt2-L5` and whatever the same string yields under some later model
are two readings of one persistent thing — the input — and they can coexist
without either being canonical. Models need not agree. The input needs only to
persist, and a string is about as durable an artifact as computing has.

This inverts the economics of the proposal. The expensive part is never the
forward pass; it is humans finding inputs worth caring about. That labour is
not model-specific. A corpus of interesting strings, with provenance, can be
re-derived under any future model by anyone who wants to, at the cost of a few
thousand forward passes. Each model becomes a lens over a shared corpus rather
than a competing claim about a shared territory.

So the progression's final phase is not "everyone agrees on GPT-2 forever". It
is: *the string corpus is the shared object, and each model is one projection
of it.*

### Where the Platonic Representation Hypothesis enters

Once the string is the invariant, convergence between models stops being a
precondition and becomes the interesting question — the one that decides how
much the multi-lens picture is worth.

The Platonic Representation Hypothesis (Huh, Cheung, Wang & Isola,
arXiv:2405.07987) argues that models trained on different data and different
modalities are converging on increasingly similar representations as they
scale, and that this convergence points at a shared statistical structure
underneath the data. Three outcomes follow for this proposal, and all three are
informative:

**If structure converges.** Two models' readings of the same string corpus show
related structure. The claim available then is not "GPT-2 has a geography" but
"there is a geography, and GPT-2 is one view of it." This is the outcome in
which the Platonic metaphor earns its name rather than decorating it: the
models are the prisoners, and the string corpus is what casts the shadows.

**If structure diverges.** The territory is model-specific. That is a finding
about representation, not a failure of the protocol — the lenses genuinely
differ, and comparing them is the entire point of keeping the string rather
than the address.

**If convergence exists but does not survive discretisation.** This is the
outcome worth naming loudest, because it is the one nobody has tested. PRH is
measured on *continuous* similarity structure — how models arrange distances
between points. This proposal takes an argmax, which discards nearly all of
that structure by design (§3). Whether representational convergence survives
the collapse to a single integer is, as far as we have found, an open
question. It is also the specific question this project is positioned to ask.

**A confound to control for.** The hypothesis is contested; later work finds
that agreement between models is sensitive to how it is measured and degrades
substantially at larger evaluation scales. That caveat bites this scheme
harder than most, because the address space is small and crowded (§4). Two
models could appear to agree simply because roughly 1,450 destinations force
collisions, not because they share structure. Any cross-model comparison must
therefore be run against a shuffled-address null, or it will manufacture the
agreement it is looking for.

### The cheap version of this experiment

It requires no new apparatus. The 545 first-arrival strings and the `Q5`
perturbation set already exist. Run the same protocol under a second model and
compare *structure* — whether strings that share a destination under one model
disproportionately share one under the other — rather than comparing addresses,
which cannot match and are not meant to. Against a shuffled null, a positive
result is a substantially stronger claim than the single-model version can
reach, and a negative one is publishable.

The honest position: engine-agnostic is available now; model-agnostic is not
required for the scheme to work, and testing it is an experiment rather than a
dependency.

---

## 7. Phases, each with a gate

The value of a staged proposal is that early stages are useful even if later
ones never arrive. That only holds if each stage names what must be true before
the next is licensed.

**Phase 0 — a reproducible name.** *Established.* A string yields the same
integer on different hardware. Gate passed by `§8`, conditional on the pin
being enforced.

**Phase 1 — shared addresses.** Two implementations independently compute the
same address from the same string. Gate: the pin enforced at every load site,
and one third-party reproduction by someone who did not write the code.

**Phase 2 — structure.** Small changes to a string produce related
destinations more often than chance. Gate: `Q5`. If the answer is at chance,
this document ends here, and Phase 1 remains a functioning curiosity — a
reproducible naming scheme, which is not nothing.

**Phase 3 — neighbourhoods.** Some relation on J is reproducible and survives
changes in prompt and measurement procedure. Gate: `Q5` plus a definition of
adjacency that is not neuron index order, plus `§5` amended to say what is
claimed.

**Phase 4 — accumulated observation.** Discoveries attach to loci with
provenance. This needs no new science, only a record format. It is the phase
most likely to be socially interesting and least likely to be technically hard.

**Phase 5 — cross-world cartography.** Independent implementations project
different worlds onto the same shared corpus. Gate: Phase 3. Note that this
phase does *not* require every implementation to use the same model — records
carry the string, so two worlds reading different specifications are still
pointing at the same object (§6). What it does require is that each record
state which reading it used.

Note that Phase 4 does not depend on Phase 3. Observations can accumulate
against reproducible names before anyone knows whether those names have
geometry. A gazetteer is useful before a map exists.

---

## 8. The portable object

The thing that travels is not an asset. It is a record, and its form matters
more than its contents:

```
locus            J-1847
specification    <model, revision, layer, hook, selection rule>
input            the string, verbatim
provenance       who, when, in which environment
interpretation   what this world made of it
evidence         what would let someone else check
```

Two fields carry the weight. **Specification** is what makes the record
meaningful to a reader who is not running your build — without it, the address
is a number in a private coordinate system. It is also what allows several
coordinate systems to coexist over one corpus: because the *input* is stored
verbatim alongside the address, the same record can be re-derived under a
different model without being reinterpreted or lost (§6). The address is a
reading; the string is the object. **Evidence** is what makes the
record a claim rather than an assertion, and it is what allows disagreement to
be productive: two worlds that interpret a locus differently are interesting,
whereas two worlds that computed it differently are simply broken, and the
record must let a reader tell those cases apart.

Everything else is commentary and can vary freely. That is the point of the
design: interpretation is local, address is not.

---

## 9. What would falsify this

A speculative document earns its place by naming the results that would end it.

- **Q5 returns chance.** Perturbed strings land no more relatedly than random
  pairs. There is no structure, only a naming scheme. Phases 2 onward are dead.
- **Q10 returns null.** Intervening on a destination neuron does nothing
  relative to matched controls. The addresses are reproducible but do not
  correspond to anything the network computes with, and "locus" was always a
  metaphor.
- **The reachable set collapses out of distribution.** Already partly observed:
  the same novel in French reaches 3.7–6.0× fewer destinations. If the space is
  largely an artifact of English training distribution, it is a territory only
  for some speakers, which is a serious problem for a shared cartography.
- **Adjacency proves procedure-dependent.** Neighbourhoods that appear under
  one prompting convention vanish under another. Then there is no territory,
  only an interaction between the model and a particular way of asking.
- **Structure fails to survive discretisation.** If cross-model comparison
  shows related continuous geometry but no relationship at all between argmax
  destinations, then the discretisation that makes addresses portable also
  destroys whatever was shared. The scheme still functions single-model, but
  the Platonic reading of it is dead.

Note what is *not* on this list: the chosen model ageing out. Because records
store the string and not merely the address (§6, §8), a corpus of discoveries
outlives any particular model and can be re-derived under a successor. The
territory has no shelf life; only a given reading of it does.

Any one of these is a publishable negative result. That is the case for running
them.

---

## 10. What this is

Not a metaverse: there is no shared world, no shared asset, no shared identity,
and nothing to negotiate.

Not a database: no canonical object sits at a coordinate, and no server needs
to hold one.

The nearest honest description is that this is an attempt at a procedural
generator whose coordinate space is meaning-shaped rather than space-shaped —
built backwards, from content toward coordinates, in the hope that a territory
emerges from a set of names.

It might not. The mapping is real, reproducible and public; the geography is a
conjecture. Those two things should never be quoted in the same breath without
the distinction attached.

But the cheap version is already available, and worth doing on its own terms: a
small, crowded, reproducible set of addresses that two strangers can compute
independently and argue about afterwards. Whether it turns out to be a map or
merely a list of names is an experiment, and the experiment is affordable.

---

*Speculative note for the Z-NOU repository. Every conditional in this document
is intended to be settled by measurement, and the measurements are named. If a
claim here has drifted loose of that discipline, it is a defect in the
document.*
