---
name: arc-container
description: Docker container management for Qdrant and MeiliSearch services. Use when user mentions starting services, stopping services, checking container status, viewing logs, restarting containers, or resetting database services.
allowed-tools: Bash(arc:*), Read
---

# Container Management

Manages Qdrant and MeiliSearch Docker containers for local development.

```bash
# Start services
arc container start              # Start both Qdrant and MeiliSearch
arc container start --service qdrant       # Start only Qdrant
arc container start --service meilisearch  # Start only MeiliSearch

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
arc container logs                    # Tail logs from both services
arc container logs --service qdrant   # Qdrant logs only
arc container logs --lines 100        # Show more lines

# Reset (delete all data)
arc container reset --confirm         # Reset both services
arc container reset --service qdrant --confirm  # Reset only Qdrant
```

## Service URLs

After starting:
- **Qdrant**: http://localhost:6333 (REST API), http://localhost:6334 (gRPC)
- **MeiliSearch**: http://localhost:7700

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
