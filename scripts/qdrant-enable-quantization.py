#!/usr/bin/env python3
"""
Enable scalar quantization for Qdrant collections.

This script enables int8 scalar quantization for all collections, providing
4x memory reduction with ~99% accuracy preservation. Quantized vectors are
kept in RAM while original vectors remain on disk.

Usage:
    python3 scripts/qdrant-enable-quantization.py [--dry-run]
"""

import argparse
import sys
from qdrant_client import QdrantClient
from qdrant_client.models import ScalarQuantization, ScalarQuantizationConfig, ScalarType


def main():
    parser = argparse.ArgumentParser(description="Enable scalar quantization for Qdrant collections")
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
        "--quantile",
        type=float,
        default=0.99,
        help="Quantization quantile (default: 0.99)"
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
            print(f"  Segments:       {info.segments_count}")
            print(f"  Points:         {info.points_count}")

            # Check current quantization config
            current_quant = info.config.quantization_config
            if current_quant:
                print(f"  Quantization:   Already enabled ({current_quant.type})")
                success_count += 1
                continue

            print(f"  Quantization:   Not enabled")

            if args.dry_run:
                print(f"  → [DRY RUN] Would enable int8 scalar quantization (quantile={args.quantile})")
                success_count += 1
                continue

            # Enable scalar quantization
            print(f"  → Enabling int8 scalar quantization...")
            client.update_collection(
                collection_name=collection_name,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=args.quantile,
                        always_ram=True  # Keep quantized vectors in RAM
                    )
                )
            )

            print(f"  ✓ Quantization enabled")
            print(f"    Config: int8, quantile={args.quantile}, always_ram=True")
            print(f"    Memory: 4x reduction (float32 → int8)")
            print(f"    Accuracy: ~99% preserved")
            success_count += 1

        except Exception as e:
            print(f"  ✗ Error: {e}", file=sys.stderr)
            error_count += 1

    # Summary
    print("\n" + "=" * 80)
    print(f"Summary: {success_count} succeeded, {error_count} failed")

    if not args.dry_run and success_count > 0:
        print("\nQuantization complete!")
        print("  - Quantized vectors stored in RAM (4x smaller)")
        print("  - Original vectors remain on disk")
        print("  - Monitor memory usage to verify reduction")
        print("\nNext steps:")
        print("  1. Monitor memory usage over 24-48 hours")
        print("  2. Verify query performance acceptable")
        print("  3. Consider reducing Docker memory limit if stable")

    return 0 if error_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
