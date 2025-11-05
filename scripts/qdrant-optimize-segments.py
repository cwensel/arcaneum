#!/usr/bin/env python3
"""
Optimize Qdrant collection segments for low-memory operation.

This script updates optimizer configuration to consolidate segments from
their current count down to the target (default: 2 segments per collection).

The optimizer automatically merges segments in the background after config update.
No forced compaction needed - Qdrant handles it gracefully.

Usage:
    python3 scripts/qdrant-optimize-segments.py [--dry-run]
"""

import argparse
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import OptimizersConfigDiff


def main():
    parser = argparse.ArgumentParser(description="Optimize Qdrant collection segments")
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
        "--target-segments",
        type=int,
        default=2,
        help="Target number of segments (default: 2)"
    )
    parser.add_argument(
        "--max-segment-size-kb",
        type=int,
        default=100000,
        help="Max segment size in KB (default: 100000)"
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
            current_segments = info.segments_count
            optimizer_status = info.optimizer_status

            print(f"\n{collection_name}:")
            print(f"  Current segments: {current_segments}")
            print(f"  Target segments:  {args.target_segments}")
            print(f"  Optimizer status: {optimizer_status}")

            if current_segments <= args.target_segments:
                print(f"  → Already at or below target, skipping")
                success_count += 1
                continue

            reduction = current_segments - args.target_segments
            print(f"  → Will consolidate {reduction} segments ({current_segments}→{args.target_segments})")

            if args.dry_run:
                print(f"  → [DRY RUN] Would update optimizer config")
                success_count += 1
                continue

            # Update optimizer configuration
            client.update_collection(
                collection_name=collection_name,
                optimizer_config=OptimizersConfigDiff(
                    default_segment_number=args.target_segments,
                    max_segment_size=args.max_segment_size_kb
                )
            )

            print(f"  ✓ Optimizer config updated")
            print(f"    Note: Segment consolidation happens automatically in background")
            print(f"    Estimated time: {reduction * 2}-{reduction * 5} minutes")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"Summary: {success_count} succeeded, {error_count} failed")

    if not args.dry_run and success_count > 0:
        print("\nNext steps:")
        print("  1. Monitor segment consolidation progress:")
        print("     python3 scripts/qdrant-monitor-segments.py")
        print("  2. Wait for all collections to reach target segment count")
        print("  3. Proceed with on-disk storage optimization")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
