#!/usr/bin/env python3
"""
Enable on-disk storage for Qdrant collections.

This script updates all collections to store vectors and HNSW indexes on disk
instead of in memory. This reduces memory usage by ~90% with acceptable
performance trade-offs for desktop use.

The script handles collections with named vectors (multiple vector types per collection).

Usage:
    python3 scripts/qdrant-enable-disk-storage.py [--dry-run]
"""

import argparse
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParamsDiff, HnswConfigDiff


def main():
    parser = argparse.ArgumentParser(description="Enable on-disk storage for Qdrant collections")
    parser.add_argument(
        "--host",
        default="localhost",
        help="Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=6333,
        help="Qdrant port (default: 6333)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes"
    )
    args = parser.parse_args()

    # Connect to Qdrant
    print(f"Connecting to Qdrant at {args.host}:{args.port}...")
    try:
        client = QdrantClient(host=args.host, port=args.port, timeout=60)
    except Exception as e:
        print(f"Error: Failed to connect to Qdrant: {e}", file=sys.stderr)
        return 1

    # Get all collections
    try:
        collections = client.get_collections().collections
    except Exception as e:
        print(f"Error: Failed to list collections: {e}", file=sys.stderr)
        return 1

    if not collections:
        print("No collections found")
        return 0

    print(f"\nFound {len(collections)} collections")
    print("=" * 80)

    # Process each collection
    success_count = 0
    error_count = 0

    for collection in collections:
        collection_name = collection.name

        try:
            # Get current collection info
            info = client.get_collection(collection_name)

            print(f"\n{collection_name}:")
            print(f"  Segments: {info.segments_count}")
            print(f"  Points:   {info.points_count}")

            # Check if collection uses named vectors or single vector
            vectors_config = info.config.params.vectors

            if isinstance(vectors_config, dict):
                # Named vectors (multiple vector types)
                print(f"  Vectors:  {len(vectors_config)} named vectors")

                # Build updated vectors config with on_disk=True
                new_vectors_config = {}
                for vector_name, vector_config in vectors_config.items():
                    current_on_disk = getattr(vector_config, 'on_disk', False)
                    print(f"    - {vector_name}: size={vector_config.size}, on_disk={current_on_disk}")

                    new_vectors_config[vector_name] = VectorParamsDiff(
                        on_disk=True  # Enable on-disk storage
                    )

            else:
                # Single vector config (less common)
                print(f"  Vectors:  Single vector config")
                current_on_disk = getattr(vectors_config, 'on_disk', False)
                print(f"    size={vectors_config.size}, on_disk={current_on_disk}")

                new_vectors_config = VectorParamsDiff(
                    on_disk=True
                )

            # Check current HNSW config
            hnsw_config = info.config.hnsw_config
            hnsw_on_disk = getattr(hnsw_config, 'on_disk', False)
            print(f"  HNSW:     on_disk={hnsw_on_disk}")

            if args.dry_run:
                print(f"  → [DRY RUN] Would enable on-disk storage for vectors and HNSW")
                success_count += 1
                continue

            # Update collection with on-disk storage
            print(f"  → Enabling on-disk storage...")
            client.update_collection(
                collection_name=collection_name,
                vectors_config=new_vectors_config,
                hnsw_config=HnswConfigDiff(
                    m=hnsw_config.m,
                    ef_construct=hnsw_config.ef_construct,
                    on_disk=True  # Enable on-disk HNSW
                )
            )

            print(f"  ✓ On-disk storage enabled")
            print(f"    Note: Qdrant will rebuild segments in background")
            print(f"    Memory usage will decrease as rebuild completes")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"Summary: {success_count} succeeded, {error_count} failed")

    if not args.dry_run and success_count > 0:
        print("\nNext steps:")
        print("  1. Monitor memory usage:")
        print("     docker stats qdrant-arcaneum")
        print("  2. Wait 5-10 minutes for segment rebuilding")
        print("  3. Proceed with scalar quantization for further memory reduction")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
