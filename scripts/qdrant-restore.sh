#!/usr/bin/env bash
# Qdrant Restore Script
# Restores collections from snapshot backups

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <backup_directory>"
  echo ""
  echo "Example: $0 ~/.arcaneum/backups/qdrant-snapshots-2025-11-05-195000"
  exit 1
fi

BACKUP_PATH="$1"
QDRANT_HOST="${QDRANT_HOST:-http://localhost:6333}"
CONTAINER_NAME="${CONTAINER_NAME:-qdrant-arcaneum}"

echo "=== Qdrant Restore Script ==="
echo "Qdrant Host: $QDRANT_HOST"
echo "Backup Directory: $BACKUP_PATH"
echo ""

# Verify backup directory exists
if [ ! -d "$BACKUP_PATH" ]; then
  echo "ERROR: Backup directory not found: $BACKUP_PATH"
  exit 1
fi

# Find all snapshot files
SNAPSHOTS=$(find "$BACKUP_PATH" -name "*.snapshot" -type f)

if [ -z "$SNAPSHOTS" ]; then
  echo "ERROR: No snapshot files found in $BACKUP_PATH"
  exit 1
fi

echo "Found snapshot files:"
echo "$SNAPSHOTS" | while read snapshot; do
  echo "  - $(basename "$snapshot")"
done
echo ""

# Wait for Qdrant to be ready
echo "Checking Qdrant availability..."
for i in {1..30}; do
  if curl -s -f "$QDRANT_HOST/healthz" > /dev/null 2>&1; then
    echo "✓ Qdrant is ready"
    break
  fi
  if [ $i -eq 30 ]; then
    echo "ERROR: Qdrant not responding after 30 seconds"
    exit 1
  fi
  echo "  Waiting for Qdrant... ($i/30)"
  sleep 1
done
echo ""

# Restore each snapshot
echo "$SNAPSHOTS" | while read snapshot_file; do
  snapshot_name=$(basename "$snapshot_file")
  # Extract collection name by removing timestamp pattern: -NNNNNN...-YYYY-MM-DD-HH-MM-SS.snapshot
  collection_name=$(echo "$snapshot_name" | sed 's/-[0-9]\{13,16\}-[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}-[0-9]\{2\}\.snapshot$//')

  echo "Restoring collection: $collection_name"
  echo "  From snapshot: $snapshot_name"

  # Copy snapshot to container's snapshots directory
  echo "  Copying snapshot to container..."
  docker cp "$snapshot_file" "$CONTAINER_NAME:/qdrant/snapshots/$snapshot_name"

  # Restore via API
  echo "  Restoring via API..."
  RESTORE_RESPONSE=$(curl -s -X PUT "$QDRANT_HOST/collections/$collection_name/snapshots/recover" \
    -H "Content-Type: application/json" \
    -d "{\"location\": \"file:///qdrant/snapshots/$snapshot_name\"}")

  # Check response
  if echo "$RESTORE_RESPONSE" | grep -q '"status":"ok"'; then
    echo "  ✓ Restored successfully"

    # Get collection info
    INFO=$(curl -s "$QDRANT_HOST/collections/$collection_name")
    VECTORS_COUNT=$(echo "$INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['vectors_count'])" 2>/dev/null || echo "unknown")
    POINTS_COUNT=$(echo "$INFO" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['points_count'])" 2>/dev/null || echo "unknown")

    echo "  Collection info: $POINTS_COUNT points, $VECTORS_COUNT vectors"
  else
    echo "  ✗ ERROR: Restore failed"
    echo "  Response: $RESTORE_RESPONSE"
  fi

  echo ""
done

echo "=== Restore Complete ==="
echo ""
echo "Verify collections:"
echo "  curl $QDRANT_HOST/collections"
