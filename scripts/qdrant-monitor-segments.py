#!/usr/bin/env python3
"""
Monitor Qdrant segment consolidation progress.

Usage:
    python3 scripts/qdrant-monitor-segments.py [--watch] [--interval SECONDS]
"""

import argparse
import sys
import time
from qdrant_client import QdrantClient


def get_collection_stats(client):
    """Get segment counts for all collections."""
    collections = client.get_collections().collections
    stats = []

    for collection in collections:
        info = client.get_collection(collection.name)
        stats.append({
            'name': collection.name,
            'segments': info.segments_count,
            'points': info.points_count,
            'status': info.optimizer_status
        })

    return stats


def print_stats(stats, show_header=True):
    """Print collection statistics in a table."""
    if show_header:
        print(f"{'Collection':<25} {'Segments':>10} {'Points':>10} {'Status':>15}")
        print("=" * 65)

    for stat in stats:
        print(f"{stat['name']:<25} {stat['segments']:>10} {stat['points']:>10} {stat['status']:>15}")


def main():
    parser = argparse.ArgumentParser(description="Monitor Qdrant segment consolidation")
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
        "--watch",
        action="store_true",
        help="Continuously monitor (Ctrl+C to stop)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Update interval in seconds for --watch mode (default: 30)"
    )
    parser.add_argument(
        "--target",
        type=int,
        default=2,
        help="Target segment count (default: 2)"
    )
    args = parser.parse_args()

    # Connect to Qdrant
    try:
        client = QdrantClient(host=args.host, port=args.port, timeout=60)
    except Exception as e:
        print(f"Error: Failed to connect to Qdrant: {e}", file=sys.stderr)
        return 1

    try:
        iteration = 0
        while True:
            if args.watch and iteration > 0:
                print(f"\n{'-' * 65}")
                print(f"Update {iteration} at {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'-' * 65}")

            stats = get_collection_stats(client)
            print_stats(stats, show_header=(iteration == 0 or args.watch))

            # Check if all collections reached target
            all_done = all(stat['segments'] <= args.target for stat in stats)

            if all_done:
                print(f"\n✓ All collections at or below target ({args.target} segments)")
                if args.watch:
                    print("  Consolidation complete!")
                return 0

            if not args.watch:
                # Show which collections still need work
                pending = [stat for stat in stats if stat['segments'] > args.target]
                if pending:
                    print(f"\n⏳ {len(pending)} collection(s) still consolidating:")
                    for stat in pending:
                        remaining = stat['segments'] - args.target
                        print(f"   {stat['name']}: {stat['segments']} → {args.target} ({remaining} to merge)")
                return 0

            iteration += 1
            time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
