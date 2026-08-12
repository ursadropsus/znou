# Addendum: Frontier Instantiation and Rider-Created Infrastructure

## Status

This addendum is exploratory.

It does not define a required EVE Frontier integration, assert that J-Space is currently supported by EVE Frontier, or prescribe a particular game mechanic. Its purpose is to examine a possible deployment ecology for J-Space in a programmable persistent world, particularly one in which players can construct infrastructure and deploy programmable logic independently of the original authors.

The central question is not “how should J-Space be added to Frontier?”

It is:

> **If J-Space were made available as a deterministic computational primitive, what could Riders build with it without requiring its authors to design those applications themselves?**

This distinction is important to the experimental programme. J-Space should not be designed in advance to produce a successful game. If it has useful structure, that structure should be observable before its applications are prescribed.

---

## 1. Frontier as a Potential Instantiation Environment

EVE Frontier's Smart Assembly architecture provides an unusually relevant environment for considering this question.

Smart Assemblies are programmable, blockchain-backed modules which encapsulate logic, data and access control. Current documented assembly types include Smart Storage Units, Smart Gates and Smart Turrets. Their behaviour is implemented using Move on Sui, and the stated design goal is to allow players and third-party developers to extend gameplay systems through programmable infrastructure.

This creates a distinction between a conventional game mechanic and a player-instantiated computational system.

In a conventional implementation, J-Space might be exposed through a developer-authored interface:

`string → J-Space destination → game mechanic`

In a programmable environment, the authors may instead provide only the first relationship:

`string → J-Space destination`

while Riders construct the second:

`J-Space destination → game consequence`

The latter model is considerably more interesting from an experimental perspective because it does not require the authors to predict the useful applications of the substrate.

Frontier's existing Smart Assembly documentation explicitly describes player-built systems such as marketplaces, quest systems, bounty systems, arcade machines, programmable toll gates and conditional access systems. This suggests that the relevant question is not whether a programmable object can have arbitrary *meaning*, but which externally defined computations can be composed with the game's own programmable state and assets.

---

## 2. Three Layers of the System

A useful decomposition is:

**Canonical computation**

`D(s) → j`

where `s` is a string and `j ∈ {0,...,3071}` is the J-Space destination.

**Verification**

A party establishes that a claimed input/output pair is valid:

`(s, j) → valid claim`

**Instantiation**

A verified result is associated with an in-game consequence:

`valid claim → state transition`

These layers need not be implemented by the same actor.

The canonical computation may be maintained by the J-Space project.

Verification may be performed independently by Riders, community infrastructure, or a third-party service.

Instantiation may be performed by a Smart Assembly or another programmable game-system interface.

This separation is important because it avoids a requirement that the complete GPT-2 inference stack itself be executed inside Frontier's programmable environment.

The computational substrate and the game infrastructure can remain separate systems joined by a verifiable interface.

---

## 3. J-Space Does Not Necessarily Need to Become an On-Chain Computation

A naive architecture would attempt to place `D(s)` itself inside a Smart Assembly.

That is not currently assumed.

The canonical J-Space function depends on a specific frozen model and inference stack. Reproducing the result requires the exact computational environment specified elsewhere in this document. Smart Assemblies, meanwhile, are documented as Move programs running on Sui.

The more general architecture is therefore:

`Rider input`
→ `canonical J-Space implementation`
→ `destination`
→ `verification`
→ `Frontier state`

The external computation need not be trusted merely because it is external.

The important property is reproducibility.

If independent parties can run the canonical implementation and obtain the same result, then a Rider can demonstrate a result to another Rider without requiring the latter to trust the original computation.

This suggests that the reproducibility work already present in this project has a possible second function.

It is not only scientific hygiene.

It is also a prerequisite for treating J-Space as a shared computational oracle.

---

## 4. The Oracle Model

The simplest possible public interface would be:

`D(s) = j`

where `s` is a string and `j` is the canonical destination.

Nothing in this interface specifies what `j` means.

It need not correspond to a location in physical space.

It need not correspond to a semantic category.

It need not correspond to a game object.

It is simply an address in the finite output space of the function.

This deliberately leaves semantic ownership to downstream users.

A Rider may decide that destination 1888 is valuable.

A corporation may designate 1888 as a checkpoint.

A trading group may associate a commodity with successful arrival at 2741.

A puzzle designer may require a preimage for 912.

A gate operator may use a verified J-Space result as one condition for access.

None of these interpretations need to be part of the canonical definition of J-Space.

This creates a useful separation between **computational meaning** and **social meaning**.

The former is an empirical question about the model and the function.

The latter is an emergent property of the users of the function.

---

## 5. Verification Without Central Ownership

A J-Space result can be thought of as a claim:

> “This string resolves to destination `j` under the canonical J-Space specification.”

There are several possible levels of verification.

At the weakest level, the claimant provides the string and another party independently evaluates it.

At a stronger level, a third-party service evaluates the canonical implementation and publishes an attestation.

At a stronger level still, multiple independent evaluators can reproduce the result before a downstream system accepts it.

The appropriate mechanism is an implementation question and should not be prematurely fixed by the present specification.

What matters experimentally is that the underlying claim is reproducible.

This also creates an interesting property for future research: J-Space could remain permissionless at the level of *discovery* even if particular applications impose their own rules around verification.

A Rider does not need permission to discover a route.

An Assembly owner may nevertheless require proof of that route before granting an in-game consequence.

---

## 6. Smart Assemblies as J-Space Interfaces

A Smart Assembly need not understand the internal structure of GPT-2 to make use of J-Space.

It only needs an externally meaningful condition.

For example:

`destination == 1888`

could become an access condition.

A storage system could require evidence of a particular destination before releasing an item.

A gate could require a successful J-Space claim before allowing passage.

A player-built market could price goods according to a J-Space-derived property.

A puzzle system could reward a particular preimage.

A corporation could use destinations as identifiers for membership, reputation or territory.

The important point is that these are not proposed game mechanics.

They are examples of the type of composition made possible when a deterministic external computation is coupled to programmable game infrastructure.

The resulting applications belong to the builders who create them.

---

## 7. From Addresses to Infrastructure

The most interesting consequence may not be individual J-Space destinations.

It may be the emergence of infrastructure around them.

Suppose a group of Riders discovers that certain classes of linguistic transformation reliably move between destinations:

`j₁ → j₂ → j₃ → j₄`

They may construct a physical or economic system around that knowledge.

One group may sell route information.

Another may guard a particular destination.

Another may build gates whose access rules depend on destination identity.

Another may maintain a database of known preimages.

Another may deliberately conceal useful routes.

Another may attempt to find robust routes with unusually large arrival margins.

At this point, the J-Space graph is no longer merely an object of scientific measurement.

It has become an information resource.

The value of that resource would be endogenous to the users who discover and exploit it.

This is one reason not to assume in advance that every destination should have equal value.

A heterogeneous distribution of discoverability, route robustness, linguistic accessibility and social knowledge may itself become part of the ecology.

---

## 8. The Possibility of Rider-Created Games

A particularly strong version of this architecture would not expose a “J-Space game” at all.

It would expose a deterministic oracle and allow Riders to construct games around its outputs.

For example, one Rider could construct a puzzle whose solution is a string mapping to a particular destination.

Another could construct a reward mechanism around a rare or difficult-to-reach destination.

A corporation could create an initiation ritual requiring several independent J-Space discoveries.

A market could trade information about routes rather than physical goods.

A gate network could encode a discovered path through J-Space.

A competitive system could measure how quickly an agent finds a string satisfying a set of destination constraints.

None of these systems need to have been conceived by the J-Space authors.

This distinction is important.

The objective is not necessarily to make J-Space into a game.

The objective is to determine whether J-Space provides a substrate from which games, economies, puzzles, institutions or other forms of player-created infrastructure can emerge.

---

## 9. Human and Machine Inhabitants

This architecture also preserves one of the project's existing experimental opportunities.

Both human Riders and AI agents can interact with the same oracle:

`input → D → destination`

They may nevertheless discover very different regions of the resulting system.

A human may develop semantic or linguistic intuitions.

A language model may exploit token statistics.

A search algorithm may optimise directly against the oracle.

A hybrid human-agent system may discover strategies unavailable to either independently.

The resulting comparison should not be interpreted simply as an intelligence leaderboard.

A more interesting question is whether different populations construct different *maps* of the same computational substrate.

If humans and agents discover different routes to the same destinations, those routes themselves become data.

If humans discover stable linguistic transformations that automated search does not find efficiently, that is informative.

If automated systems rapidly dominate the accessible space, that is informative too.

The environment should therefore be treated as an ecology rather than a benchmark whose desired winner is known in advance.

---

## 10. Sovereignty and the Boundary of the Experiment

There is a useful conceptual symmetry here.

The J-Space authors control the definition of the oracle.

Riders control the interpretations they build around it.

Assembly owners control the rules of their own applications.

The underlying Frontier platform controls the constraints within which those applications operate.

No single participant necessarily controls the entire resulting system.

This suggests a possible meaning of “sovereign” in this context that is narrower and more useful than simply “permissionless.”

A Rider does not need authority over J-Space as a whole in order to instantiate a local interpretation of it.

They only need the ability to construct an application whose rules refer to J-Space results.

Consequently, the same computational substrate may acquire many incompatible local interpretations without requiring a canonical interpretation at the protocol level.

---

## 11. Failure Is Also a Result

This model should not be treated as evidence that J-Space will necessarily become useful infrastructure.

Several failure modes remain scientifically interesting.

If neighbouring strings produce effectively unrelated destinations, J-Space may be difficult for humans to learn.

If only brute-force search is effective, its primary value may be computational rather than linguistic.

If verification is too expensive or cumbersome, Rider-created applications may not emerge.

If a small number of destinations dominate usage, the resulting ecology may become highly concentrated.

If agents can trivially solve all useful routing problems, information about routes may have little economic value.

If the computational structure is rich but cannot be connected reliably to game state, the result may remain an interesting external experiment rather than a Frontier-native system.

None of these outcomes should be engineered away merely because they make a less compelling game.

They are observations about the substrate.

---

## 12. Relationship to Mechanistic Interpretability

This deployment model also clarifies the role of the mechanistic-interpretability questions.

Q10 asks whether the destination neuron has meaningful causal or computational significance.

That remains an important scientific question.

But Rider-created infrastructure does not require a positive answer.

A destination can be operationally useful even if the winning neuron has no coherent semantic interpretation.

Likewise, a strong result on Q5—showing that controlled changes in input produce structured changes in destination—could make the substrate useful even before the internal computational role of individual neurons is understood.

This creates two partially independent research tracks:

**What is J-Space inside the model?**

and

**What can agents do with the externally observable behaviour of J-Space?**

The first is a mechanistic-interpretability problem.

The second is a problem in learnability, interaction, emergence and game design.

A successful project need not collapse these questions into one answer.

---

## 13. Proposed Principle

The following principle may be useful for future development:

> **Define the oracle narrowly; leave the applications open.**

The canonical J-Space specification should define only what is necessary to make `D(s)` reproducible and comparable across implementations.

It should not define the economic, social or ludic meaning of destinations unless experimental evidence requires such definitions.

If Frontier or another programmable environment permits independent actors to instantiate J-Space-derived systems, those systems should be treated as downstream applications rather than part of the canonical computational object.

This preserves experimental neutrality while allowing the possibility of unusually rich emergent behaviour.

---

## 14. Open Technical Questions

Several questions should remain explicitly unresolved.

Can canonical J-Space results be verified cheaply enough for practical use?

Can an external computation be represented to a Smart Assembly in a sufficiently trustworthy way?

Which classes of Frontier state can realistically be conditioned on J-Space-derived information?

Can Riders deploy such systems without developer intervention?

What permissions, interfaces or platform constraints limit the apparent programmability of Smart Assemblies?

Can a J-Space application remain useful if its canonical oracle is hosted outside the game?

Can multiple independent implementations remain bit-for-bit compatible over time?

How should model revisions be handled if the canonical inference stack ever changes?

Would a future J-Space protocol need versioned oracles, e.g.

`J-Space/v1 → D₁(s)`

and

`J-Space/v2 → D₂(s)`?

These questions are infrastructure questions rather than evidence about J-Space itself and should not be conflated with the existing empirical questions in the main specification.

---

## 15. The Strongest Possible Outcome

The strongest outcome of this line of investigation is not that CCP implements J-Space as an official feature.

It is that no such implementation is necessary.

A sufficiently well-defined oracle could become a shared computational primitive.

Researchers could study it.

Players could discover it.

Agents could exploit it.

Riders could build infrastructure around it.

Different communities could assign different meanings to the same destinations.

Some applications could fail.

Others could become economically or socially important.

The authors could remain observers of much of what subsequently develops.

In that scenario, J-Space would not be a game mechanic authored by its creator.

It would be a small piece of computational infrastructure from which other people construct game mechanics.

That distinction is the reason Frontier is relevant to this project at all.

The interesting experiment is not:

> “Can we put GPT-2 activations into EVE Frontier?”

It is:

> **“Can a deterministic, model-derived computational substrate become a useful primitive for player-created infrastructure?”**

The answer to that question should be allowed to emerge from the interaction between the oracle, its users and the environment.

---

## 16. Current Position

No Frontier-specific implementation is currently required by J-Space.

The appropriate near-term approach is therefore observational.

First establish what structure exists in the oracle.

Then establish what humans and machines can learn about that structure.

Then determine what kinds of claims about destinations can be independently verified.

Only after those properties are understood should a particular Frontier integration be considered.

If the substrate proves sufficiently structured and learnable, Frontier provides a particularly interesting candidate environment in which its consequences could be instantiated by actors other than the original researchers.

That possibility is worth preserving without making it a prerequisite for the experiment.