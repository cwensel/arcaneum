# Recommendation [NUMBER]: [TITLE]

> Revise during planning; lock at implementation.
> If wrong, abandon code and iterate RDR.

## Metadata

- **Date**: YYYY-MM-DD
- **Status**: Draft | Final | Implemented | Reverted |
  Abandoned | Superseded
- **Type**: Feature | Bug Fix | Technical Debt |
  Framework Workaround | Architecture
- **Priority**: High | Medium | Low
- **Related Issues**: [Links to related issues/tickets]

## Problem Statement

[What is the specific challenge or requirement?
Refine as research deepens understanding.]

## Context

### Background

[How was this discovered? Current behavior, impact,
constraints.]

### Technical Environment

[Framework versions, dependencies, architecture,
related components.]

## Research Findings

### Investigation

[What was analyzed? Code, docs, source, experiments,
standards. Cite specific locations.]

#### Dependency Source Verification

| Dependency | Source Searched? | Key Findings |
| --- | --- | --- |
| [library/service name] | Yes / No | [Signatures, constraints, or defaults confirmed or corrected] |

### Key Discoveries

[Label each finding's evidence basis:

- **Verified** — confirmed by spike/POC/experiment
- **Documented** — from official docs or source reading
- **Assumed** — needs validation before implementation]

### Critical Assumptions

[Load-bearing assumptions — if wrong, the approach fails.
Each must be verified before marking this RDR Final.]

- [ ] [Assumption 1] — **Status**: Verified | Unverified
  — **Method**: Source Search | Spike | Docs Only
- [ ] [Assumption 2] — **Status**: Verified | Unverified
  — **Method**: Source Search | Spike | Docs Only

**Method definitions**:

- **Source Search**: API verified against dependency
  source code (standard method for libraries)
- **Spike**: Behavior verified by running code
  against a live service (for opaque services only)
- **Docs Only**: Based on documentation reading alone
  (insufficient for load-bearing assumptions)

## Proposed Solution

### Approach

[Detailed description of the recommended solution.]

### Technical Design

[Architecture, component relationships, data flow,
extension points.]

**Code guidance**:

- Specify interfaces (function signatures, input/output
  types, error contracts) — not class implementations
- Mark every external API call as Verified (source
  search) or Assumed (needs validation before
  implementation)
- Do NOT include full class implementations,
  config/schema definitions, or code for deferred
  features
- Limit illustrative code to patterns that cannot be
  expressed as prose (e.g., callback signatures,
  serialization formats)

```text
// Illustrative — verify API signatures during implementation
```

### Existing Infrastructure Audit

[List existing modules that overlap with proposed
components. For each, state whether to reuse, extend,
or replace — with justification.]

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| [New component] | [Existing module path] | Reuse / Extend / Replace: [reason] |

### Decision Rationale

[Why this approach over alternatives. Key factors,
how it addresses the problem, why alternatives were
ruled out.]

## Alternatives Considered

[Full analysis for seriously evaluated alternatives.
One-sentence rejection for trivially eliminated options.]

### Alternative 1: [Name]

**Description**: [Brief description]

**Pros**:

- [Advantage 1]

**Cons**:

- [Disadvantage 1]

**Reason for rejection**: [Why this wasn't chosen]

### Briefly Rejected

- **[Alternative N]**: [One-sentence rejection]

## Trade-offs

### Consequences

[Positive and negative consequences of the chosen
approach.]

- [Consequence 1 — positive or negative]
- [Consequence 2 — positive or negative]

### Risks and Mitigations

- **Risk**: [Description]
  **Mitigation**: [How to address]

### Failure Modes

[What breaks visibly? What fails silently? Recovery
path? How does a developer diagnose the problem?]

## Implementation Plan

### Prerequisites

- [ ] All Critical Assumptions verified
- [ ] [Other prerequisites]

### Minimum Viable Validation

[The single end-to-end proof that the approach works.
Must be in scope — not deferred.]

### Phase 1: Code Implementation

#### Step 1: [Title]

[Instructions]

#### Step 2: [Title]

[Instructions]

### Phase 2: Operational Activation

[Deployment, CI/CD, credentials, shared infrastructure.
Omit if not applicable.]

#### Activation Step 1: [Title]

[Instructions]

### Day 2 Operations

[For every persistent resource this RDR creates
(collection, index, data store, config entry),
address management operations:]

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| [Resource] | In scope / Deferred / N/A | ... | ... | ... | ... |

[If any operation is marked "Deferred," justify why
it is not needed for initial usability.]

### New Dependencies

[Dependencies to add/update. For third-party: note
license and whether legal review is required.]

## Validation

### Testing Strategy

[Test scenarios and coverage goals — what to test and
what constitutes "done." For non-functional concerns
(performance, security): state measurement strategy,
not estimates.]

1. **Scenario**: [Description]
   **Expected**: [Result]

### Performance Expectations

[Do not include effort estimates or speculative
throughput targets. Rough performance metrics are
appropriate only when comparing alternatives — note
empirical data or obvious gains that support the
chosen approach over a rejected one.]

## Finalization Gate

> Complete each item with a written response before
> marking this RDR as **Final**. Written responses
> prevent rubber-stamping and produce a review record.

### Contradiction Check

[State any conflicts between Research Findings and
the Proposed Solution. If none exist, state
"No contradictions found between research findings,
design principles, and proposed solution."]

### Assumption Verification

[Confirm all Critical Assumptions are verified. List
any that remain unverified with a plan to verify
before implementation begins.]

#### API Verification

| API Call | Library | Verification |
| --- | --- | --- |
| [method/endpoint] | [library name] | Source Search / Spike / Docs Only |

### Scope Verification

[Confirm the Minimum Viable Validation is in scope
and will be executed during implementation, not
deferred. State the specific test or proof.]

### Cross-Cutting Concerns

[For each applicable concern, state how it is
addressed in this RDR. Mark the rest N/A.]

- **Versioning**: [How addressed | N/A]
- **Build tool compatibility**: [How addressed | N/A]
- **Licensing**: [How addressed | N/A]
- **Deployment model**: [How addressed | N/A]
- **IDE compatibility**: [How addressed | N/A]
- **Incremental adoption**: [How addressed | N/A]
- **Secret/credential lifecycle**: [Generation,
  storage, rotation, override | N/A]
- **Memory management**: [Peak memory estimation,
  streaming strategy, cleanup | N/A]

### Proportionality

[Is the document right-sized for the change? Flag
any sections that should be trimmed before locking.]

## References

- [Requirements/standards with section numbers]
- [Dependency docs, source paths reviewed]
- [Dependency repos searched (clone + code search)]
- [Related issues, articles, discussions]
