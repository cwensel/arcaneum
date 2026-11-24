---
name: arc-container
description: Docker container management for Qdrant (and optional MeiliSearch) services. Use when user mentions starting services, stopping services, checking container status, viewing logs, restarting containers, or resetting database services.
allowed-tools: Bash(arc:*), Read
---

# Container Management

Manages the Qdrant Docker container for local development, plus MeiliSearch when `meilisearch.enabled` is true
(a new install runs Qdrant only).

```bash
# Start services
arc container start              # Start Qdrant (and MeiliSearch, when enabled)
arc container start --service qdrant       # Start only Qdrant
arc container start --service meilisearch  # Start only MeiliSearch (when enabled)

# Stop services
arc container stop
arc container stop --service qdrant
arc container stop --service meilisearch

# Restart services
arc container restart
arc container restart --service qdrant

# Check status
arc container status
arc container status --json

# View logs
arc container logs                    # Tail logs from the running services
arc container logs --service qdrant   # Qdrant logs only
arc container logs --lines 100        # Show more lines

# Reset (delete all data)
arc container reset --confirm         # Reset all enabled services
arc container reset --service qdrant --confirm  # Reset only Qdrant
```

## Optional MeiliSearch

MeiliSearch is optional. `meilisearch.enabled` in `~/.config/arcaneum/config.yaml`
decides whether `arc container start`, `restart`, `status`, `upgrade` and `backup`
include it. When the key is absent, a stored MeiliSearch key (or
`MEILISEARCH_API_KEY`) means an existing user and keeps it on; a new user gets
Qdrant only. The first `arc container start` records the choice in the config.

```yaml
meilisearch:
  enabled: false   # true to run MeiliSearch again
```

While it is `false`, `arc container start` refuses to finish if any corpus still
uses MeiliSearch for text search, naming each corpus and its
`arc corpus migrate-text <corpus> --to qdrant` command.

## Service URLs

After starting:

- **Qdrant**: <http://localhost:6333> (REST API), <http://localhost:6334> (gRPC)
- **MeiliSearch** (when enabled): <http://localhost:7700>

## Troubleshooting

```bash
# Check if Docker is running
docker info

# Check service health
arc container status

# View recent errors
arc container logs --lines 50

# Full reset if needed
arc container reset --confirm
arc container start
```
