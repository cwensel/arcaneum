# Arcaneum Claude Code Plugin

CLI tools for semantic and full-text search across Qdrant and MeiliSearch vector databases.

## Features

- **Semantic search**: Vector-based search for conceptual queries
- **Full-text search**: Keyword-based search for exact matches
- **Multi-format support**: Code repositories, markdown documents, PDFs
- **Auto-activating skill**: Claude automatically searches when relevant

## Installation

```bash
/plugin marketplace add cwensel/arcaneum
/plugin install arc
```

## Sandbox Configuration

The `arc` CLI requires network access to communicate with Qdrant and MeiliSearch backends.
If you have Claude Code's sandbox enabled, you must configure it to allow `arc` to run.

### Option 1: Exclude `arc` from Sandbox (Recommended)

Add `arc` to `excludedCommands` so it runs outside the sandbox with normal permission flow.

Add to your `~/.claude/settings.json` or `.claude/settings.json`:

```json
{
  "sandbox": {
    "enabled": true,
    "autoAllowBashIfSandboxed": true,
    "excludedCommands": ["arc"]
  },
  "permissions": {
    "allow": [
      "Bash(arc:*)",
      "Bash(arc search:*)",
      "Bash(arc collection:*)"
    ]
  }
}
```

### Option 2: Allow Local Network Access

If your Qdrant/MeiliSearch run on localhost, allow local binding:

```json
{
  "sandbox": {
    "enabled": true,
    "autoAllowBashIfSandboxed": true,
    "network": {
      "allowLocalBinding": true
    }
  },
  "permissions": {
    "allow": [
      "Bash(arc:*)"
    ]
  }
}
```

Note: `allowLocalBinding` is macOS only.

### Settings File Locations

| File | Scope | Checked into git |
| ------------------------------ | -------------------- | ---------------- |
| `~/.claude/settings.json` | All projects | N/A |
| `.claude/settings.json` | This project, shared | Yes |
| `.claude/settings.local.json` | This project, local | No |

## Usage

The plugin includes an auto-activating skill. When you mention searching a collection or
request semantic search, Claude will automatically use `arc`.

### Manual Commands

```bash
# List collections
arc collection list

# Semantic search
arc search semantic "authentication flow" --corpus MyCode --limit 10

# Multi-corpus search
arc search semantic "authentication" --corpus Code --corpus Docs

# Full-text search
arc search text "validateToken" --corpus MyCode --limit 10
```

## Requirements

- `arc` CLI installed and in PATH (`pip install arcaneum`)
- Qdrant and/or MeiliSearch backend running
- Collections indexed with content

## Agent Integration

If you're creating a custom agent that uses this plugin, add the following to your agent's
`allowedTools` frontmatter:

```yaml
allowedTools:
  - "Bash(arc:*)"
```

This permits all arc subcommands (search, collection, index, etc.).

Example agent frontmatter:

```yaml
---
name: my-research-agent
description: Agent that searches indexed collections
allowedTools:
  - "Bash(arc:*)"
  - "Read"
  - "Grep"
---
```

Without this permission, the agent will fail when attempting to run arc commands.

## License

MIT
