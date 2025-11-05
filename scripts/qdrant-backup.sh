#!/usr/bin/env bash
# Qdrant Backup Script
# Creates snapshots of all collections and copies them to local disk

set -euo pipefail

QDRANT_HOST="${QDRANT_HOST:-http://localhost:6333}"
BACKUP_DIR="${BACKUP_DIR:-$HOME/.arcaneum/backups}"
TIMESTAMP=$(date +%Y-%m-%d-%H%M%S)
BACKUP_PATH="$BACKUP_DIR/qdrant-snapshots-$TIMESTAMP"
CONTAINER_NAME="${CONTAINER_NAME:-qdrant-arcaneum}"

echo "=== Qdrant Backup Script ==="
echo "Qdrant Host: $QDRANT_HOST"
echo "Backup Directory: $BACKUP_PATH"
echo ""

# Create backup directory
mkdir -p "$BACKUP_PATH"

# Get list of all collections
echo "Fetching collection list..."
COLLECTIONS=$(curl -s "$QDRANT_HOST/collections" | \
  python3 -c "import sys, json; print('\n'.join([c['name'] for c in json.load(sys.stdin)['result']['collections']]))")

if [ -z "$COLLECTIONS" ]; then
  echo "No collections found!"
  exit 0
fi

echo "Found collections:"
echo "$COLLECTIONS"
echo ""

# Create snapshot for each collection
for collection in $COLLECTIONS; do
  echo "Creating snapshot for: $collection"

  # Create snapshot via API
  SNAPSHOT_RESPONSE=$(curl -s -X POST "$QDRANT_HOST/collections/$collection/snapshots")
  SNAPSHOT_NAME=$(echo "$SNAPSHOT_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['result']['name'])")

  if [ -z "$SNAPSHOT_NAME" ]; then
    echo "ERROR: Failed to create snapshot for $collection"
    echo "Response: $SNAPSHOT_RESPONSE"
    continue
  fi

  echo "  Snapshot created: $SNAPSHOT_NAME"

  # Wait for snapshot to be written to disk
  echo "  Waiting for snapshot to be written..."
  sleep 2

  # Copy snapshot from container to local disk (snapshots are in collection subdirectories)
  echo "  Copying snapshot to $BACKUP_PATH/"
  docker cp "$CONTAINER_NAME:/qdrant/snapshots/$collection/$SNAPSHOT_NAME" "$BACKUP_PATH/"

  # Verify file exists
  if [ -f "$BACKUP_PATH/$SNAPSHOT_NAME" ]; then
    SIZE=$(du -h "$BACKUP_PATH/$SNAPSHOT_NAME" | cut -f1)
    echo "  ✓ Backed up successfully ($SIZE)"
  else
    echo "  ✗ ERROR: Snapshot file not found after copy"
  fi

  echo ""
done

# Create manifest file
echo "Creating backup manifest..."
cat > "$BACKUP_PATH/manifest.txt" <<EOF
Qdrant Backup Manifest
======================
Backup Date: $(date)
Qdrant Host: $QDRANT_HOST
Container: $CONTAINER_NAME

Collections Backed Up:
$COLLECTIONS

Snapshot Files:
$(ls -lh "$BACKUP_PATH"/*.snapshot 2>/dev/null || echo "No snapshots found")
EOF

echo "Backup manifest:"
cat "$BACKUP_PATH/manifest.txt"
echo ""

echo "=== Backup Complete ==="
echo "Backup location: $BACKUP_PATH"
echo ""
echo "To restore, use: ./scripts/qdrant-restore.sh $BACKUP_PATH"
