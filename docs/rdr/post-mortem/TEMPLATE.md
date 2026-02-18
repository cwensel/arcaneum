# Post-Mortem: RDR-[NUMBER] [TITLE]

## RDR Summary

[2-3 sentence summary of what the RDR proposed to solve
and the approach it recommended.]

## Implementation Status

[Implemented | Partially Implemented | Not Implemented |
Reverted]

[1-2 sentences on the current state, including any notable
context like "reverted after 48 hours due to Maven
incompatibility" or "code complete but CI/CD activation
pending."]

---

## Implementation vs. Plan

### What Was Implemented as Planned

- [List items where the RDR accurately predicted the
  implementation]

### What Diverged from the Plan

For each divergence:

- **[Brief label]**: The RDR planned [X]. The
  implementation did [Y] because [reason].

### Existing Infrastructure Reused Instead of New Code

- [Component planned in RDR] was replaced by
  [existing module] because [reason]

### What Was Added Beyond the Plan

- [Features, infrastructure, or tooling that were
  implemented but not anticipated by the RDR]

### What Was Planned but Not Implemented

- [Items in the RDR that were skipped, with reason
  if known]

---

## Drift Classification

Classify each divergence into one of these categories
to enable cross-RDR pattern analysis. Mark all that
apply with the count of instances.

| Category | Count | Examples | Preventable? |
| --- | --- | --- | --- |
| **Unvalidated assumption** | | | |
| **Framework API detail** | | | |
| **Missing failure mode** | | | |
| **Missing Day 2 operation** | | | |
| **Deferred critical constraint** | | | |
| **Over-specified code** | | | |
| **Under-specified architecture** | | | |
| **Scope underestimation** | | | |
| **Internal contradiction** | | | |
| **Missing cross-cutting concern** | | | |

**Preventable?** column: "Yes — source search" or
"Yes — spike" if verification would have prevented the
drift, "No" if inherently unpredictable.

### Pattern References

[For each drift category with count >= 2, reference
the applicable pattern from SYNTHESIS.md (if one
exists). This enables incremental synthesis updates.]

### Drift Category Definitions

- **Unvalidated assumption** — a claim presented as fact
  but never verified by source search or spike
- **Framework API detail** — method signatures, interface
  contracts, or config syntax wrong
- **Missing failure mode** — what breaks, what fails
  silently, recovery path not considered
- **Missing Day 2 operation** — bootstrap, CI/CD,
  removal, rollback, migration not planned
- **Deferred critical constraint** — downstream use case
  that validates the approach was out of scope
- **Over-specified code** — implementation code that was
  substantially rewritten
- **Under-specified architecture** — architectural
  decision that should have been made but wasn't
- **Scope underestimation** — sub-feature that grew into
  its own major effort
- **Internal contradiction** — research findings or stated
  principles conflicting with the proposal
- **Missing cross-cutting concern** — versioning,
  licensing, config cache, deployment model, etc.

---

## RDR Quality Assessment

### What the RDR Got Right

Focus on what was valuable and should be repeated in
future RDRs:

- [Architectural decisions, technology selections,
  trade-off analysis, research findings, etc.]

### What the RDR Missed

Focus on what should have been in the plan but wasn't:

- [Missing concerns, unconsidered constraints, wrong
  assumptions, etc.]

### What the RDR Over-specified

Focus on effort that did not contribute to a successful
implementation:

- **Code samples rewritten**: [list]
- **Deferred feature code unused**: [list]
- **Config/schema never implemented**: [list]
- **Performance targets unvalidated**: [list]
- **Alternative analysis disproportionate**: [list]

---

## Key Takeaways for RDR Process Improvement

List 3-5 actionable insights framed as improvements to
the RDR authoring process itself. Each takeaway should be:

- **Generalizable** — applicable to future RDRs beyond
  this specific topic
- **Actionable** — something an RDR author can do
  differently, not just "be more careful"
- **Evidence-based** — tied to a specific divergence or
  gap found in this post-mortem

1. **[Imperative verb phrase]**: [Explanation tied to
   evidence from this post-mortem]
2. **[Imperative verb phrase]**: [Explanation tied to
   evidence from this post-mortem]
3. **[Imperative verb phrase]**: [Explanation tied to
   evidence from this post-mortem]
