# Deployment Configuration

## Docker Compose

**Note:** It's recommended to use the `arc container` CLI commands instead of running docker-compose directly.

### Using the arc CLI (Recommended)

```bash
# Start services
arc container start

# Stop services
arc container stop

# Check status
arc container status

# View logs
arc container logs

# Restart services
arc container restart
```

### Using docker-compose Directly

If you prefer direct docker-compose access:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

Or from the deploy directory:

```bash
cd deploy
docker compose up -d
```

## Configuration

- **Qdrant**: Port 6333 (REST), 6334 (gRPC)
- **Qdrant image**: `qdrant/qdrant:v1.18.0`
- **MeiliSearch image**: `getmeili/meilisearch:v1.12`
- **Storage**: Docker named volumes managed by the `arcaneum` compose project
- **Snapshots**: Docker named volumes managed by the `arcaneum` compose project

**Note:** Embedding models are stored in `~/.arcaneum/models` on the host machine. Qdrant doesn't need access to these models since it only stores/searches vectors, not generates them.

## Data Location

Embedding model caches are stored in `~/.arcaneum/`:

```
~/.arcaneum/
├── models/              # Embedding model cache
```

When started with `arc container`, Qdrant and MeiliSearch data are stored in
Docker named volumes:

```
arcaneum_qdrant-arcaneum-storage
arcaneum_qdrant-arcaneum-snapshots
arcaneum_meilisearch-arcaneum-data
arcaneum_meilisearch-arcaneum-dumps
arcaneum_meilisearch-arcaneum-snapshots
```

**Benefits:**
- Managed by Docker and isolated from source checkouts
- Easy inspection with `arc container status`
- Survives `arc container stop` and plain `docker compose down`
- Stable volume names across image-only upgrades

Do not run `docker compose down --volumes` unless you intentionally want to
delete indexed data. Image upgrades should keep the volume names in
`docker-compose.yml` unchanged. Prefer `arc container` over direct
`docker compose` commands so the compose project name, and therefore the
Docker volume names, stay stable.

## Migration from Old Setup

If you were using the old `./deploy/qdrant_storage` volumes:

```bash
# Stop services first
arc container stop

# Data should already be in ~/.arcaneum/ from migration
# Start with new configuration
arc container start
```
