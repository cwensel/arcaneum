# Post-Mortem Synthesis: RDR-001 through RDR-018

Synthesized from 17 post-mortems (RDR-001 through RDR-014, RDR-016 through RDR-018).
208 total drift instances classified across 10 categories.

---

## A. Aggregate Drift Classification

| Rank | Category | Count | Percentage | RDRs With 2+ Instances |
| --- | --- | --- | --- | --- |
| 1 | Over-specified code | 35 | 16.8% | 001, 002, 003, 004, 005, 006, 008, 009, 010, 011, 013, 014, 016 |
| 2 | Scope underestimation | 27 | 13.0% | 001, 002, 003, 004, 006, 008, 009, 010, 013 |
| 3 | Under-specified architecture | 25 | 12.0% | 001, 002, 003, 004, 005, 008, 009, 013 |
| 4 | Framework API detail | 24 | 11.5% | 001, 004, 006, 009, 010, 011 |
| 4 | Missing cross-cutting concern | 24 | 11.5% | 001, 002, 003, 004, 005, 007, 009, 010 |
| 6 | Unvalidated assumption | 21 | 10.1% | 001, 002, 003, 004, 005, 006, 010, 013 |
| 7 | Missing Day 2 operation | 20 | 9.6% | 003, 004, 007, 008, 009, 010 |
| 8 | Missing failure mode | 17 | 8.2% | 004, 007, 013, 016 |
| 9 | Deferred critical constraint | 10 | 4.8% | -- |
| 10 | Internal contradiction | 5 | 2.4% | -- |

**Observation**: The top three categories (over-specified code, scope underestimation,
under-specified architecture) account for 41.8% of all drift. These are authoring-process
problems, not domain-knowledge problems -- they can be addressed by changing how RDRs
are written.

---

## B. Top Recurring Patterns

### Pattern 1: Code samples are systematically rewritten (35 instances, all 17 RDRs)

Every RDR contained at least one instance of over-specified code. The pattern has
three variants:

- **Full class implementations that were rewritten**: RDR-003 (EmbeddingClient,
  arc init), RDR-004 (LateChunker, BatchCheckpoint), RDR-009 (DualIndexer,
  sync\_directory), RDR-014 (MarkdownChunkMetadata, MarkdownMetadataSync,
  MarkdownInjectionHandler)
- **Code for deferred features that was never used**: RDR-004 (LateChunker),
  RDR-006 (MCP server), RDR-013 (EmbeddingCache, Phase 3-4 code)
- **YAML/config schemas that were never implemented**: RDR-004 (pdf\_processing
  YAML), RDR-014 (MarkdownChunkMetadata dataclass), RDR-016 (pdf\_extraction
  YAML)

**Impact**: RDR authors invested significant effort writing implementation code that
provided no value. Worse, detailed pseudocode created false confidence in designs
that did not survive contact with real APIs and existing infrastructure.

### Pattern 2: GPU and memory management missed as cross-cutting concerns (8 RDRs)

RDRs 002, 003, 004, 005, 009, 013, 014, and 016 all required GPU acceleration,
OOM recovery, or memory management that was not anticipated in the plan. RDR-013
described GPU integration as "Low complexity: just add `device='mps'`" -- the
implementation required OOM recovery with progressive batch reduction, GPU poisoning
and CPU fallback, embedding validation (NaN/Inf/zero detection), Metal command buffer
timeout handling, per-model MPS batch limits, and systematic `gc.collect()` patterns.

**Impact**: GPU/memory management became the single largest unplanned implementation
effort across the project. The streaming upload architecture (originally Phase 4 in
RDR-013) was pulled forward to Phase 2 because memory pressure made it a necessity,
not an optimization.

### Pattern 3: External API behavior assumed without spikes (13 RDRs)

Unvalidated assumptions and framework API detail errors together account for 45
drift instances. Recurring sub-patterns:

- **FastEmbed assumed sufficient for all models** (002, 003, 004, 007): Many
  high-quality models (stella, jina-code) required SentenceTransformers/PyTorch
- **MeiliSearch API methods assumed** (010, 011): Document ID constraints,
  delete\_documents filter syntax, get\_documents semantics all differed from
  pseudocode
- **Claude Code plugin API assumed stable** (001, 006): Skills system, colon
  namespacing, and command discovery all evolved after RDR research
- **MeiliSearch relevance scores assumed available** (012): Scores not exposed
  in the same way as Qdrant
- **Version pinning to unreleased versions** (008): MeiliSearch v1.32 specified
  but not available at implementation time

**Impact**: A 30-minute spike per external API would have caught the majority of
these issues. The FastEmbed assumption alone drove the most significant architectural
change in the project (dual-backend embedding system).

### Pattern 4: Day 2 operations treated as future work (20 instances, 10 RDRs)

When an RDR creates a resource, it consistently deferred list, info, delete, verify,
and backup operations to "Future Enhancements" -- then all of them were needed
immediately:

- RDR-009: corpus list/info/delete deferred; all needed immediately
- RDR-005: collection verification/repair not planned; needed after interrupted indexing
- RDR-008: index export/import not planned; needed for backup/migration
- RDR-014: cleanup command for agent-memory not built; storage accumulates indefinitely
- RDR-016: migration guide for re-indexing existing collections not created

**Impact**: Day 2 operations consistently represent 30-50% of the implementation effort
for any feature that creates persistent resources. Deferring them underestimates scope
and leaves users without essential management capabilities.

### Pattern 5: Scope systematically underestimated (27 instances, all 17 RDRs)

CLI commands grew 3-5x beyond initial plans:

- RDR-001: 9 flat commands planned, 30+ hierarchical commands implemented
- RDR-008: 4 admin commands planned, 10 implemented
- RDR-009: 2 commands planned (create, sync), 8 implemented
- RDR-018: 7 files to modify, 24 files actually touched

Code modules grew by an order of magnitude:

- Embedding client: 50-line wrapper planned (002), 1600-line GPU-aware system built
- Sync module: simple sync planned (009), 140KB module built (001)
- CLI main.py: 70-line sample in RDR-001, 787-line production file

**Impact**: Scope underestimation cascades into schedule and architecture decisions.
The recurring cause is focusing on the happy path without counting Day 2 operations,
error handling, cross-cutting integrations, and production hardening.

### Pattern 6: CLI namespace designed without growth consideration (8 RDRs)

- RDR-001/003/006: Flat commands (`create-collection`) abandoned for hierarchical
  groups (`collection create`)
- RDR-007: `arc find` became `arc search semantic` to accommodate `arc search text`
- RDR-008: `arc fulltext` became `arc indexes` to parallel `arc collection`
- RDR-009/012: `--index` became `--corpus` across multiple commands
- RDR-014: `arc inject markdown` became `arc store`

**Impact**: Command renaming breaks backwards compatibility, requires migration layers
(e.g., `resolve_corpora()` in search commands), and confuses documentation.

### Pattern 7: Speculative performance targets never validated (13 RDRs)

RDRs 002, 003, 004, 005, 007, 008, 009, 010, 011, 013, 014, 016, and 017 all
included specific performance targets (e.g., "100 PDFs/min", "query latency < 50ms",
"100-200 files/sec"). No benchmark infrastructure was created for any RDR. No target
was formally validated. Actual performance depends on GPU availability, model size,
and workload characteristics that static targets cannot capture.

**Impact**: Performance targets consumed RDR authoring effort without guiding
implementation. They created false precision and were universally ignored.

---

## C. Recommendations for Post-Mortem Template

### C.1 Add "Pattern References" subsection to Drift Classification

After the drift classification table, add a prompt:

```markdown
### Pattern References

For each drift instance with count >= 2, reference the
applicable pattern from SYNTHESIS.md (if any). This
enables incremental synthesis updates.
```

**Evidence**: The current template captures drift counts and examples per RDR
but does not link to known cross-RDR patterns, making synthesis a manual
aggregation exercise each time.

### C.2 Add "Existing Infrastructure Missed" subsection

Under "What Diverged from the Plan," add:

```markdown
### Existing Infrastructure Reused Instead of New Code

- [Component planned in RDR] was replaced by
  [existing module] because [reason]
```

**Evidence**: RDRs 007 (SearchEmbedder replaced by EmbeddingClient), 014
(MarkdownMetadataSync replaced by MetadataBasedSync), and 008 (shell script
replaced by arc container CLI) all discovered reusable infrastructure during
implementation that the RDR duplicated.

### C.3 Add "Spike Would Have Caught" column to Drift Classification

Extend the drift classification table:

```markdown
| Category | Count | Examples | Spike? |
```

Where "Spike?" is "Yes (N min)" if a brief spike would have prevented the drift,
or "No" if it was inherently unpredictable.

**Evidence**: Pattern 3 (external API assumptions) shows that 30-minute spikes
would have caught the majority of unvalidated assumption and framework API detail
drifts. Quantifying this per post-mortem makes the case for spike requirements
in the RDR template.

### C.4 Replace "What the RDR Over-specified" with structured categories

The current freeform section conflates different types of over-specification.
Replace with:

```markdown
### What the RDR Over-specified

- **Code samples rewritten**: [list]
- **Deferred feature code unused**: [list]
- **Config/schema never implemented**: [list]
- **Performance targets unvalidated**: [list]
- **Alternative analysis disproportionate**: [list]
```

**Evidence**: All 17 post-mortems contain over-specification findings, but the
freeform format makes cross-RDR comparison difficult. The five sub-categories
above cover all observed over-specification patterns.

---

## D. Recommendations for RDR Template

### D.1 Replace code sample guidance in Technical Design (targets Pattern 1)

**Current guidance** (line 67-69):

```markdown
### Technical Design

[Architecture, component relationships, data flow,
extension points. Include illustrative code snippets
for novel patterns -- not full implementations.]
```

**Proposed replacement**:

```markdown
### Technical Design

[Architecture, component relationships, data flow,
extension points.]

**Code guidance**:

- Specify interfaces (function signatures, input/output
  types, error contracts) -- not class implementations
- Mark every external API call as Verified (source
  search) or Assumed (needs validation before
  implementation)
- Do NOT include full class implementations, YAML/config
  schemas, or code for deferred features
- Limit illustrative code to patterns that cannot be
  expressed as prose (e.g., callback signatures,
  serialization formats)
```

**Evidence**: Over-specified code was the #1 drift category (35 instances). Full
class implementations were rewritten in every post-mortem. YAML schemas were
never implemented (004, 014, 016). Deferred feature code was never used (004,
006, 013).

### D.2 Add "Existing Infrastructure Audit" section (targets Patterns 1, 5)

Add after Technical Design:

```markdown
### Existing Infrastructure Audit

List existing modules that overlap with proposed
components. For each, state whether to reuse, extend,
or replace -- with justification.

| Proposed Component | Existing Module | Decision |
| --- | --- | --- |
| [New class] | [Existing module path] | Reuse / Extend / Replace: [reason] |
```

**Evidence**: RDR-007 designed a standalone SearchEmbedder when EmbeddingClient
already existed. RDR-014 designed MarkdownMetadataSync when MetadataBasedSync
already existed. RDR-008 designed a shell script when `arc container` CLI
was emerging. In each case, a 5-minute audit of existing modules would have
prevented the design of redundant components.

### D.3 Add "Day 2 Operations" section (targets Pattern 4)

Add after Implementation Plan:

```markdown
### Day 2 Operations

For every resource this RDR creates (collection, index,
corpus, file store, config entry), address:

| Resource | List | Info | Delete | Verify | Backup |
| --- | --- | --- | --- | --- | --- |
| [Resource] | [In scope / Deferred / N/A] | ... | ... | ... | ... |

If any operation is marked "Deferred," justify why it
is not needed for initial usability.
```

**Evidence**: RDR-009 deferred corpus list/info/delete to "Future Enhancements"
but all were needed immediately. RDR-005 did not plan collection verification;
it was needed after interrupted indexing. RDR-008 did not plan index
export/import; needed for backup. Pattern 4 documents 20 instances across 10 RDRs.

### D.4 Add "Failure Modes" structured section (targets Pattern 2, 8)

Add to the existing Failure Modes section explicit prompts:

```markdown
### Failure Modes

For each category, list known failure scenarios or
state "Investigated: none found."

- **Library-specific**: [What happens when the primary
  library crashes, hangs, or returns corrupt data?]
- **Hardware-specific**: [GPU OOM, device timeout,
  memory exhaustion, architecture-specific behavior?]
- **Data-specific**: [Malformed input, edge-case
  content, oversized files, encoding issues?]
- **Network-specific**: [Service down, timeout, partial
  response, auth failure?]
- **Recovery**: [For each failure above, what is the
  recovery path?]
```

**Evidence**: Missing failure modes totaled 17 instances. GPU failures on Apple
Silicon MPS were missed by 8 RDRs. Type3 font hangs (016), font digest errors
(016), PDF control characters breaking JSON (007), and MPS embedding corruption
without exceptions (013) were all discovered during implementation.

### D.5 Add "CLI Namespace Review" to Technical Design (targets Pattern 6)

When an RDR introduces CLI commands, require:

```markdown
### CLI Namespace Review

List all existing top-level command groups and their
naming pattern. Verify the proposed command(s) follow
the established pattern.

| Existing Group | Pattern |
| --- | --- |
| arc collection | noun + verb |
| arc search | verb + type |

**Proposed command**: [name]
**Follows pattern**: [Yes/No -- if No, justify]
**Growth consideration**: [How will this namespace
accommodate future subcommands?]
```

**Evidence**: `arc fulltext` was renamed to `arc indexes` (008), `arc find`
became `arc search semantic` (007), `arc inject` became `arc store` (014),
and flat commands were replaced by hierarchical groups (001, 003, 006).
A namespace review would have caught these during planning.

### D.6 Expand Finalization Gate items

Add the following gate items:

```markdown
### API Verification

List every external API call in pseudocode. For each,
state how it was verified. Source search (checking
dependency source code) is the standard method for
libraries and open-source dependencies. A spike
(running code against a live service) is for opaque
services where source code is unavailable. Any call
marked "Assumed" must include a plan to verify before
implementation begins.

| API Call | Library | Status |
| --- | --- | --- |
| [call] | [lib] | Source Search / Spike / Assumed |

### Day 2 Operations Check

Confirm the Day 2 Operations table is complete. If any
"Deferred" items exist, confirm they are not needed for
initial usability.

### CLI Namespace Consistency

If this RDR introduces CLI commands, confirm the CLI
Namespace Review section is complete and the proposed
commands follow established patterns.

### Hardware and Device Strategy

If this RDR involves ML model inference, embedding
generation, or other compute-intensive workloads,
confirm the plan addresses:

- Device selection (GPU/CPU)
- Memory management and OOM recovery
- Batch sizing strategy
- Failure modes specific to target hardware
```

**Evidence**: Pattern 3 (API assumptions without spikes or source search) drove 45
drift instances. Of these, the majority were API signatures and constraints that
could have been caught by searching dependency source code (5-10 min) without
requiring a full runtime spike. Pattern 4 (missing Day 2 operations) drove 20
instances. Pattern 6 (CLI naming) caused rework in 8 RDRs. Pattern 2 (GPU/memory)
was the largest unplanned effort in the project.

### D.7 Revise Cross-Cutting Concerns checklist

**Current list**: Versioning, Build tool compatibility, Licensing, Deployment
model, IDE compatibility, Incremental adoption.

**Proposed additions**:

```markdown
- **Secret/credential lifecycle**: [Generation, storage,
  rotation, override | N/A]
- **GPU/device management**: [Device selection, OOM
  recovery, batch strategy | N/A]
- **CLI namespace consistency**: [Follows existing
  patterns | N/A]
- **Memory management**: [Peak memory estimation,
  streaming strategy, cleanup | N/A]
- **Interaction logging**: [Integration with
  InteractionLogger | N/A]
```

**Evidence**: Secret lifecycle was missed in RDR-008 (MeiliSearch API key
auto-generation needed). GPU/device management was missed in 8 RDRs.
CLI namespace was inconsistent in 8 RDRs. Memory management was a cross-cutting
concern in RDRs 004, 005, 009, 013, 014, 016. Interaction logging (018) was
needed across 24 files.

### D.8 Add performance target guidance

Add to the Validation section:

```markdown
### Performance Expectations

State performance expectations qualitatively
(e.g., "bulk indexing should not require more than
2x the dataset size in memory") rather than with
specific numeric targets unless backed by benchmark
measurements.

If specific targets are stated, they MUST be marked:
- **Measured**: [benchmark methodology and results]
- **Estimated**: [basis for estimate, to be validated
  during implementation]

Do NOT include speculative throughput numbers without
a measurement plan.
```

**Evidence**: 13 of 17 RDRs included specific performance targets. Zero were
formally validated. No benchmark infrastructure was created. Pattern 7
documents this across all post-mortems.

---

## E. Process Recommendations

### E.1 Require source verification for external API integration (targets Pattern 3)

**Change**: Add to the Research Guidance section of README.md:

> When an RDR depends on external API behavior (third-party libraries,
> service APIs, plugin systems), verify APIs against dependency
> source code before locking the RDR. Clone the repo and use
> `arc corpus sync` + `arc search text/semantic` to confirm method
> signatures, constraints, and defaults (5-10 min per dependency).
> This is the **standard verification method** for libraries and
> open-source dependencies.
>
> For opaque services where source code is unavailable, a spike
> (running code against the live service) serves the same purpose.
>
> Documentation-only research is insufficient for load-bearing
> assumptions — method signatures and behavior from docs are
> frequently wrong in detail.

**Evidence**: FastEmbed model coverage (002, 003), MeiliSearch filter syntax
(010, 011), Claude Code plugin discovery (001, 006), and Qdrant query\_points
API (007) were all assumed from documentation and wrong. The majority were
API signatures and constraints that could have been caught by searching
dependency source code without a full runtime spike.

### E.2 Add "Reuse Audit" step to the Research phase (targets Patterns 1, 5)

**Change**: Add to the Workflow section of README.md:

> During the Research phase, audit existing modules for components that
> overlap with the proposed design. Use `grep` or code search to identify
> existing classes, functions, and patterns that the new RDR might reuse
> or extend. Document findings in the "Existing Infrastructure Audit"
> section.

**Evidence**: RDR-007, 008, and 014 all designed components that duplicated
existing infrastructure. This step takes 5-10 minutes and prevents
designing redundant code.

### E.3 Treat all code samples as illustrative unless source-verified (targets Pattern 1)

**Change**: Update the "On code examples" guidance in README.md:

> **On code examples**: All code in RDRs is illustrative by default.
> Mark framework API calls as Verified (source search) or Assumed
> (needs validation). Do not include full class implementations --
> specify behavior and interfaces instead. Do not include code for
> deferred features. Code for the current scope should demonstrate
> architectural patterns, not production-ready implementations.

**Evidence**: Over-specified code was the #1 drift category (35 instances,
16.8% of all drift). The current guidance says "not full implementations"
but every RDR included them anyway. Stronger language and explicit
prohibitions are needed.

### E.4 Scope estimates must count Day 2 operations and integration surface

**Change**: Add to the When to Create an RDR section:

> When estimating scope, count the full command surface (not just the
> happy-path commands), all Day 2 operations (list, info, delete, verify,
> backup for each resource), and the cross-cutting integration surface
> (logging, error handling, GPU management for each affected file).
> A heuristic: multiply the happy-path estimate by 2.5-3x to account
> for Day 2 operations and production hardening.

**Evidence**: RDR-009 estimated 16-20 hours for 2 commands; 8 commands were
needed. RDR-018 estimated 7 files to modify; 24 were touched. RDR-001
planned 9 CLI commands; 30+ were built. Pattern 5 documents systematic
3-5x scope growth.

### E.5 Lock CLI namespace decisions before individual command RDRs

**Change**: Add to the Research Guidance section:

> If a project will have multiple RDRs that introduce CLI commands,
> create a namespace decision early (in the first CLI-related RDR or
> a dedicated architecture RDR). The decision should specify: the
> group hierarchy, the naming convention (noun-verb vs verb-type),
> and how the namespace will accommodate known future commands.
> Subsequent RDRs should reference this decision rather than making
> independent naming choices.

**Evidence**: 8 RDRs made independent CLI naming decisions that conflicted
with each other, requiring renaming (`fulltext` to `indexes`, `find` to
`search semantic`, `inject` to `store`, `--index` to `--corpus`) and
backwards-compatibility layers.
