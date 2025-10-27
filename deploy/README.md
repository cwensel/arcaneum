# Deployment Configuration

## Docker Compose

Start Qdrant server:

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
- **Storage**: ./qdrant_storage
- **Snapshots**: ./qdrant_snapshots
- **Models**: ./models_cache (shared with container)

## Quick Commands

```bash
# Start services
docker compose -f deploy/docker-compose.yml up -d

# Stop services
docker compose -f deploy/docker-compose.yml down

# View logs
docker compose -f deploy/docker-compose.yml logs -f

# Restart
docker compose -f deploy/docker-compose.yml restart
```

## Alternative: Symlink at Root

For convenience, create a symlink:

```bash
ln -s deploy/docker-compose.yml docker-compose.yml
```

Then use standard commands:
```bash
docker compose up -d
```
