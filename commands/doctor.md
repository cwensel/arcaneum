---
description: Verify Arcaneum setup and prerequisites
argument-hint: [--verbose] [--json]
---

Check that all Arcaneum prerequisites are met and the system is ready for use.

**Arguments:**

- --verbose: Show detailed diagnostic information
- --json: Output JSON format

**Checks Performed:**

- Python version (>= 3.12 required)
- Required Python dependencies installed
- Qdrant server connectivity and health
- MeiliSearch server connectivity (if configured)
- Embedding model availability
- Write permissions for temporary files
- Environment variable configuration

**Examples:**

```text
/doctor
/doctor --verbose
/doctor --json
```

**Execution:**

```bash
cd ${CLAUDE_PLUGIN_ROOT}
arc doctor $ARGUMENTS
```

**Note:** This diagnostic command helps troubleshoot setup issues. I'll run all
checks and present a summary showing which requirements are met (✅) and which
need attention (❌), along with specific guidance for fixing any problems found.
