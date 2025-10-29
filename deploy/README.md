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
- **Storage**: `~/.arcaneum/data/qdrant` (host path)
- **Snapshots**: `~/.arcaneum/data/qdrant_snapshots` (host path)

**Note:** Embedding models are stored in `~/.arcaneum/models` on the host machine. Qdrant doesn't need access to these models since it only stores/searches vectors, not generates them.

## Data Location

All persistent data is stored in `~/.arcaneum/`:

```
~/.arcaneum/
├── models/              # Embedding model cache
├── data/
│   ├── qdrant/         # Qdrant vector database
│   └── qdrant_snapshots/  # Backup snapshots
```

**Benefits:**
- Predictable, user-accessible location
- Easy backup/restore (just backup ~/.arcaneum/)
- Survives docker-compose down
- No hidden Docker volumes

## Migration from Old Setup

If you were using the old `./deploy/qdrant_storage` volumes:

```bash
# Stop services first
arc container stop

# Data should already be in ~/.arcaneum/ from migration
# Start with new configuration
arc container start
```
