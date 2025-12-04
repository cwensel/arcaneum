---
description: Manage container services (Qdrant, MeiliSearch)
argument-hint: <start|stop|status|logs|restart|reset> [options]
---

Manage Docker container services for Qdrant and MeiliSearch.

**Subcommands:**

- start: Start all services
- stop: Stop all services
- status: Show service status and health
- logs: View service logs
- restart: Restart services
- reset: Delete all data and reset (WARNING: destructive)

**Arguments:**

- --follow, -f: Follow log output (logs command only)
- --tail <n>: Number of log lines to show (logs command, default: 100)
- --confirm: Confirm data deletion (reset command only)

**Examples:**

```text
/container start
/container status
/container logs
/container logs --follow
/container stop
/container restart
/container reset --confirm
```

**Execution:**

```bash
arc container $ARGUMENTS
```

**Note:** Container management commands check Docker availability and provide
helpful error messages if Docker is not running. I'll show you:

- Service startup confirmation with URLs
- Health check status for Qdrant
- Data directory locations and sizes
- Log output for debugging

**Data Locations (Docker Volumes):**

- Qdrant storage: `qdrant-arcaneum-storage`
- Qdrant snapshots: `qdrant-arcaneum-snapshots`
- All data persists across container restarts
- Use `docker volume ls --filter name=arcaneum` to view volumes

**Related:**

- Implemented in arcaneum-167, renamed in arcaneum-169
- Replaces old scripts/qdrant-manage.sh
- Part of simplified Docker management (arcaneum-158)
