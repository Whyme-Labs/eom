# Conway's law

Conway's law is an adage that states organizations which design systems are constrained to produce designs which are copies of the communication structures of these organizations. The law is named after computer programmer Melvin Conway, who introduced the idea in 1967.

## Origin

Melvin Conway was a computer scientist who observed the relationship between organizational structure and system design while working on compilers in the 1960s. He submitted a paper on the topic to Harvard Business Review in 1967, which was rejected on the grounds that he had not proved his thesis. The paper was later published in Datamation magazine. The concept was named "Conway's Law" at the 1968 National Symposium on Modular Programming, where attendees coined the term.

Conway's original formulation states: "organizations which design systems (in the broad sense used here) are constrained to produce designs which are copies of the communication structures of these organizations."

The reasoning behind this observation is straightforward: for a system to function correctly, the people who design its individual components must communicate with each other to ensure compatibility between the parts. Because communication naturally follows organizational boundaries, the architecture of the resulting system tends to mirror those same boundaries.

## Implications

The law has significant implications for software engineering and organizational design. A key insight is that the correspondence between organizational structure and system architecture is not merely incidental but structurally enforced. Teams that do not communicate freely will produce designs with hard interfaces at exactly the points where communication breaks down.

Research at MIT and Harvard Business School has validated the "mirroring hypothesis" — an equivalent formulation of Conway's Law — finding strong evidence that the product developed by a loosely-coupled organization is significantly more modular than the product from a tightly-coupled organization.

Website design provides a practical illustration: researcher Nigel Bevan observed in 1997 that organizations often produce web sites with a content and structure which mirrors the internal concerns of the organization rather than the needs of the users of the site.

Scholars have proposed stronger variations. Edward Yourdon and Larry Constantine stated that the structure of any system designed by an organization is isomorphic to the structure of the organization. James O. Coplien and Neil B. Harrison offered a prescriptive version: organizational structure should be deliberately aligned with product architecture to avoid project failures.

Opinions differ on whether the phenomenon is beneficial or harmful. Some view the organizational-technical mirroring as a natural and helpful property. Others see it as an undesirable consequence of organizational bias that constrains design flexibility. A middle perspective treats it as a necessary compromise given human cognitive limitations.

The "Inverse Conway Maneuver" has become a popular concept in discussions of microservices and team topology: rather than letting organizational structure dictate architecture, teams deliberately reorganize around the desired system architecture to nudge the design in a preferred direction.
