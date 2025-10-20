# Recommendation Data Records (RDRs)

This directory contains Recommendation Data Records (RDRs) - detailed technical recommendations and implementation plans.

## Index

| ID  | Title                                                                                                    | Status         | Priority |
|-----|----------------------------------------------------------------------------------------------------------|----------------|----------|
| 001 | [Claude Code Marketplace Project Structure](RDR-001-project-structure.md)                               | Recommendation | High     |
| 002 | [Qdrant Server Setup with Client-side Embeddings](RDR-002-qdrant-server-setup.md)                       | Recommendation | High     |
| 003 | [CLI Tool for Qdrant Collection Creation with Named Vectors](RDR-003-collection-creation.md)            | Recommendation | High     |
| 004 | [Bulk PDF Indexing with OCR Support](RDR-004-pdf-bulk-indexing.md)                                      | Recommendation | High     |
| 005 | [Git-Aware Source Code Indexing with AST Chunking](RDR-005-source-code-indexing.md)                     | Recommendation | High     |
| 006 | [Claude Code Marketplace Plugin and CLI Integration](RDR-006-claude-code-integration.md)                | Recommendation | High     |
| 007 | [Semantic Search CLI for Qdrant Collections](RDR-007-semantic-search.md)                                | Recommendation | High     |


## What are RDRs?

RDRs are ADR-inspired documents that capture detailed implementation plans *before* coding begins. Unlike traditional
Architecture Decision Records (ADRs) which document decisions already made, RDRs serve as:

- **Planning Documents**: Detailed blueprints for complex implementations
- **Iteration Artifacts**: Living documents refined through review cycles
- **AI Collaboration Tools**: Structured guides for working with Claude and other AI assistants
- **Knowledge Repository**: Preserved research and analysis for future reference

## Purpose

- **Plan Before Coding**: Create detailed implementation plans that can be reviewed and refined
- **Future Reference**: Enable any Claude instance or developer to understand and implement solutions
- **Knowledge Preservation**: Document research findings, technical analysis, and design rationale
- **Implementation Ready**: Provide complete, actionable instructions with full context
- **Framework Limitations**: Record workarounds for third-party framework constraints

## When to Create an RDR

Create an RDR when:

- Planning a complex multi-step feature implementation
- Addressing significant technical debt
- Working around framework or library limitations
- Making architectural decisions with trade-offs
- The implementation requires substantial research
- You want to iterate on an approach before committing to code

## RDR Workflow

### 1. Problem Identification

- Recognize a complex implementation need
- Identify constraints and requirements

### 2. Research & Analysis

- Investigate existing codebase patterns
- Research framework documentation
- Analyze similar solutions

### 3. RDR Creation

- Write initial RDR following the template
- Include all research findings
- Document proposed solution in detail

### 4. Review & Iteration

- Share with team or Claude for feedback
- Refine approach based on input
- Update RDR with improvements

### 5. Implementation

- Use RDR as implementation guide
- Update RDR status when implemented
- Record any deviations or learnings

## Format

Each RDR follows this structure:

### Metadata

- **ID**: Sequential number (e.g., 003)
- **Date**: Creation date
- **Status**: Recommendation | In Progress | Implemented | Superseded
- **Priority**: High | Medium | Low
- **Type**: Feature | Bug Fix | Technical Debt | Framework Workaround

### Content Sections

1. **Problem Statement**: Clear description of the challenge
2. **Context**: Background information and constraints
3. **Research Findings**: Analysis and investigation results
4. **Proposed Solution**: Detailed implementation approach
5. **Alternatives Considered**: Other approaches evaluated
6. **Trade-offs and Consequences**: Pros, cons, and impacts
7. **Implementation Plan**: Step-by-step guide
8. **Validation**: How to verify the solution

## Working with Claude

RDRs are particularly valuable when working with Claude in plan mode:

```
# Initial request
"Create an RDR for implementing [feature description]"

# Iteration
"Update the RDR to consider [specific concern or alternative]"

# Implementation
"Implement the solution from RDR-003"
```

This approach ensures:

- Detailed planning before code changes
- Opportunity for review and refinement
- Clear implementation instructions
- Preserved context for future work

## Usage Guidelines

These documents are designed to be:

- **Actionable**: Contain sufficient detail for implementation
- **Self-contained**: Include all necessary context
- **Iterative**: Can be refined before implementation
- **Preserved**: Stored in git for future reference