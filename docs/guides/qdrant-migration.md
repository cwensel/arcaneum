# Qdrant Migration Guide: Bind Mounts to Named Volumes

This guide walks through migrating Qdrant from macOS bind mounts to Docker named volumes to eliminate the
"Unrecognized filesystem" warning and prevent data corruption.

## Migration Options

There are two approaches for migrating Qdrant collections:

| Method                     | Best For                         | Features                 |
| -------------------------- | -------------------------------- | ------------------------ |
| **Snapshots** (this guide) | Same Qdrant version, full backup | Native format, fastest   |
| **Export/Import**          | Cross-machine, selective export  | Portable, path remapping |

For cross-machine migration with path adjustments, see the **Export/Import Alternative** section below.

## Problem

**Symptoms:**

```text
WARN qdrant: There is a potential issue with the filesystem for storage path ./storage.
Details: Unrecognized filesystem - cannot guarantee data safety
```

**Root Cause:**

- Qdrant running in Docker on macOS using bind mounts (`~/.arcaneum/data/qdrant`)
- APFS (macOS filesystem) accessed through Docker's virtualization layer
- Known issue: vectors can become zero-filled on container restart
- Qdrant cannot guarantee POSIX filesystem semantics with APFS bind mounts

**Solution:**

Use Docker named volumes instead of bind mounts. Named volumes store data on a Linux filesystem (ext4) inside Docker
Desktop's VM, eliminating cross-platform filesystem issues.

## Migration Steps

### 1. Create Backup

Run the backup script to create snapshots of all collections:

```bash
./scripts/qdrant-backup.sh
```

**What this does:**

- Lists all Qdrant collections via API
- Creates snapshot for each collection using Qdrant's native snapshot feature
- Copies snapshots from container to `~/.arcaneum/backups/qdrant-snapshots-TIMESTAMP/`
- Creates manifest file documenting the backup

**Backup location:**

```text
~/.arcaneum/backups/qdrant-snapshots-2025-11-05-195000/
├── manifest.txt
├── collection1-2025-11-05-1950.snapshot
├── collection2-2025-11-05-1950.snapshot
└── ...
```

### 2. Stop Qdrant

Stop the current container:

```bash
docker compose -f deploy/docker-compose.yml down
```

This removes the container but does NOT delete the bind mount data in `~/.arcaneum/data/qdrant`.

### 3. Remove Old Data (Optional)

If starting fresh without restoring:

```bash
rm -rf ~/.arcaneum/data/qdrant
rm -rf ~/.arcaneum/data/qdrant_snapshots
```

**Note:** The docker-compose.yml has already been updated with named volumes and v1.15.5.

### 4. Start Qdrant with Named Volumes

Start Qdrant with the updated configuration:

```bash
docker compose -f deploy/docker-compose.yml up -d
```

**What this does:**

- Downloads Qdrant v1.15.5 image (if not cached)
- Creates named volumes automatically:
  - `qdrant-arcaneum-storage`
  - `qdrant-arcaneum-snapshots`
- Stores data on Linux ext4 filesystem inside Docker VM
- Starts with empty collections (no data yet)

**Verify no warnings:**

```bash
docker logs qdrant-arcaneum 2>&1 | grep -i filesystem
```

You should see NO "Unrecognized filesystem" warnings.

### 5. Restore from Backup

Restore collections from snapshots:

```bash
./scripts/qdrant-restore.sh ~/.arcaneum/backups/qdrant-snapshots-TIMESTAMP/
```

**What this does:**

- Waits for Qdrant to be ready (health check)
- Copies each snapshot file into the container's snapshots volume
- Restores each collection via Qdrant's recovery API
- Verifies point and vector counts for each collection

**Verify restoration:**

```bash
curl http://localhost:6333/collections
```

Check that all collections are present with correct point counts.

## Understanding Named Volumes

### Storage Location

**Physical location on macOS:**

```text
~/Library/Containers/com.docker.docker/Data/vms/0/
```

Data is stored inside Docker Desktop's Linux VM, NOT directly on macOS filesystem.

### Accessing Data

You cannot browse named volumes directly from Finder. Use Docker commands:

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect qdrant-arcaneum-storage

# List files in volume
docker run --rm -v qdrant-arcaneum-storage:/data alpine ls -la /data

# Copy data out
docker cp qdrant-arcaneum:/qdrant/storage ./backup/
```

### Volume Persistence

**Named volumes persist independently:**

- Survive `docker compose down`
- Survive container deletion
- Survive system reboots
- Only deleted with explicit `docker compose down -v` or `docker volume rm`

**To delete volumes:**

```bash
# Remove containers AND volumes
docker compose -f deploy/docker-compose.yml down -v

# Or manually delete
docker volume rm qdrant-arcaneum-storage
docker volume rm qdrant-arcaneum-snapshots
```

## Backup Best Practices

### Regular Backups

Create a cron job or scheduled task:

```bash
# Add to crontab (daily at 2 AM)
0 2 * * * /path/to/arcaneum/scripts/qdrant-backup.sh

# Or run manually before risky operations
./scripts/qdrant-backup.sh
```

### Backup Storage

Keep backups:

- On different physical disk than Docker volumes
- In cloud storage (S3, Dropbox, etc.)
- With regular cleanup of old backups (keep last N)

### Restore Testing

Periodically test restoration to verify backups work:

```bash
# Restore to test instance
QDRANT_HOST=http://localhost:7333 \
CONTAINER_NAME=qdrant-test \
./scripts/qdrant-restore.sh ~/.arcaneum/backups/qdrant-snapshots-TIMESTAMP/
```

## What Changed

### Before (Bind Mounts)

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.15.4
    volumes:
      - ~/.arcaneum/data/qdrant:/qdrant/storage
      - ~/.arcaneum/data/qdrant_snapshots:/qdrant/snapshots
```

**Issues:**

- APFS filesystem warning
- Risk of vector corruption on restart
- Cross-platform virtualization overhead

### After (Named Volumes)

```yaml
services:
  qdrant:
    image: qdrant/qdrant:v1.15.5
    volumes:
      - qdrant-arcaneum-storage:/qdrant/storage
      - qdrant-arcaneum-snapshots:/qdrant/snapshots

volumes:
  qdrant-arcaneum-storage:
    driver: local
  qdrant-arcaneum-snapshots:
    driver: local
```

**Benefits:**

- No filesystem warnings
- Data stored on Linux ext4 (native, tested)
- No corruption risk
- Better performance

## Version Upgrade

### v1.15.4 → v1.15.5

**Critical fixes in v1.15.5:**

- Fixed segment corruption from unflushed mutable ID tracker files
- Resolved data race issues during snapshot creation
- Improved point movement handling

**Upgrade notes:**

- No breaking changes
- Snapshots from v1.15.4 work in v1.15.5
- Recommended for all users on v1.15.x

## Troubleshooting

### Snapshots Don't Restore

**Check container logs:**

```bash
docker logs qdrant-arcaneum
```

**Common issues:**

- Snapshot file format mismatch (unlikely between v1.15.4 and v1.15.5)
- Insufficient disk space in Docker VM
- Container not fully started (wait longer)

**Solution:**

- Verify snapshot files are not corrupted: `ls -lh backup/*.snapshot`
- Increase Docker Desktop VM disk size in settings
- Wait for health check: `curl http://localhost:6333/healthz`

### Collections Empty After Restore

**Verify restore succeeded:**

```bash
curl http://localhost:6333/collections/{collection_name}
```

**Check:**

- `points_count` should match backup manifest
- `vectors_count` should match backup manifest
- `status` should be "green"

**If counts are zero:**

- Restore may have failed silently
- Check container logs for errors
- Try manual restore with different snapshot

### Named Volumes Full

**Check volume size:**

```bash
docker system df -v
```

**Clean up:**

```bash
# Remove unused volumes (careful!)
docker volume prune

# Increase Docker Desktop VM disk allocation in settings
```

### Can't Access Data from macOS

This is expected behavior with named volumes. Use:

```bash
# View files
docker run --rm -v qdrant-arcaneum-storage:/data alpine ls -la /data

# Copy to macOS
docker cp qdrant-arcaneum:/qdrant/storage ./local-copy/

# Or use Qdrant's snapshot feature
curl -X POST http://localhost:6333/collections/{name}/snapshots
docker cp qdrant-arcaneum:/qdrant/snapshots/{snapshot} ./
```

## Export/Import Alternative

For cross-machine migration or when you need selective export with path remapping, use the CLI
export/import commands instead of snapshots.

### When to Use Export/Import

- Migrating to a different machine with different paths
- Sharing collections with team members
- Selective backup (specific repos or file patterns)
- Collections that need path adjustments

### Export/Import Workflow

**On source machine:**

```bash
# Option 1: Detached export (strips root prefix)
arc collection export MyCode -o shareable.arcexp --detach

# Option 2: Full export (keeps absolute paths)
arc collection export MyCode -o backup.arcexp

# Selective export (specific repo only)
arc collection export MyCode -o arcaneum.arcexp --repo arcaneum#main
```

**Transfer file to target machine, then import:**

```bash
# Import detached export with new root
arc collection import shareable.arcexp --attach /home/newuser/projects

# Import with path remapping
arc collection import backup.arcexp \
    --remap /Users/olduser:/home/newuser \
    --into MyCode-migrated
```

### Export vs Snapshots Comparison

| Feature          | Snapshots     | Export/Import      |
| ---------------- | ------------- | ------------------ |
| Format           | Native binary | Portable `.arcexp` |
| Speed            | Fastest       | Slightly slower    |
| Selective export | No            | Yes (filters)      |
| Path remapping   | No            | Yes                |
| Cross-version    | Limited       | Yes                |
| Docker required  | Yes           | No                 |

See [CLI Reference](cli-reference.md#export-collection) for full export/import documentation.

## References

- [Qdrant Filesystem Requirements](https://qdrant.tech/documentation/)
- [GitHub Issue #6676: Vector corruption on macOS/Windows](https://github.com/qdrant/qdrant/issues/6676)
- [Docker Named Volumes Documentation](https://docs.docker.com/storage/volumes/)
- [Qdrant Snapshots API](https://qdrant.tech/documentation/concepts/snapshots/)

## Summary

**Migration achieves:**

- ✓ Eliminates "Unrecognized filesystem" warning
- ✓ Prevents vector corruption on container restart
- ✓ Uses Qdrant-tested filesystem (ext4)
- ✓ Upgrades to v1.15.5 with critical bug fixes
- ✓ Maintains data integrity via snapshot backup/restore
- ✓ Better performance (no cross-platform virtualization)

**Going forward:**

- Use `./scripts/qdrant-backup.sh` for regular backups
- Named volumes persist across container lifecycles
- Data safely stored in Docker-managed Linux filesystem
- No manual management of `~/.arcaneum/data/qdrant` needed
