#!/usr/bin/env python3
"""List files whose chunks are duplicated inside Qdrant.

Counts real chunk points per file and compares them against the cached
``chunk_count`` recorded in that file's manifest. A file holding more real
points than its manifest claims was indexed more than once under
non-deterministic point IDs, so the later pass appended copies instead of
overwriting them.

`arc corpus verify` does not catch this: it checks chunk_index *coverage*
using a set, so duplicate indices collapse and the file looks complete.
`arc corpus parity` does not catch it either -- both systems agree once the
duplicates have been copied to MeiliSearch, which is correct behaviour for a
cross-system check.

Usage:
    python scripts/find-duplicate-chunks.py <CorpusName>
    python scripts/find-duplicate-chunks.py --all
"""

import sys
from collections import Counter, defaultdict

from arcaneum.cli.sync import create_qdrant_client
from arcaneum.indexing.collection_metadata import metadata_exclusion_filter
from arcaneum.indexing.common.sync import MetadataBasedSync


def scan(qdrant, corpus):
    """Return (scanned_points, list of (file_path, manifest_count, real_count, dup_indices))."""
    manifest = {
        path: payload.get("chunk_count", 0)
        for path, payload in MetadataBasedSync(qdrant)
        .get_file_manifest_snapshot(corpus)
        .items()
    }

    real = Counter()
    indices_by_file = defaultdict(list)
    offset = None
    scanned = 0

    while True:
        points, offset = qdrant.scroll(
            collection_name=corpus,
            scroll_filter=metadata_exclusion_filter(),
            limit=1000,
            offset=offset,
            with_payload=["file_path", "chunk_index"],
            with_vectors=False,
        )
        if not points:
            break
        for point in points:
            payload = point.payload or {}
            file_path = payload.get("file_path")
            if file_path:
                real[file_path] += 1
                indices_by_file[file_path].append(payload.get("chunk_index"))
        scanned += len(points)
        if offset is None:
            break

    findings = []
    for file_path, count in real.items():
        duplicated = [
            index
            for index, seen in Counter(indices_by_file[file_path]).items()
            if seen > 1 and index is not None
        ]
        expected = manifest.get(file_path)
        if duplicated or (expected is not None and count != expected):
            findings.append((file_path, expected, count, len(duplicated)))

    return scanned, len(real), sorted(findings, key=lambda row: -row[2])


def report(corpus, scanned, file_count, findings):
    print(f"\ncorpus={corpus}  scanned {scanned} chunk points across {file_count} files")
    if not findings:
        print("  No duplicated or drifted files found.")
        return
    print(f"  {len(findings)} file(s) with duplicated chunks or manifest drift:\n")
    for file_path, expected, count, duplicated in findings:
        ratio = f"{count / expected:.1f}x" if expected else "n/a"
        print(
            f"    manifest={expected!s:<7} real={count:<7} {ratio:<6} "
            f"dup_indices={duplicated:<6} {file_path}"
        )
    print("\n  Repair (--force deletes existing chunks before re-indexing):")
    for file_path, *_ in findings:
        print(f'    arc corpus sync {corpus} "{file_path}" --force')


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1

    qdrant = create_qdrant_client()

    if args[0] == "--all":
        corpora = [c.name for c in qdrant.get_collections().collections]
    else:
        corpora = args

    total_findings = 0
    for corpus in sorted(corpora):
        try:
            scanned, file_count, findings = scan(qdrant, corpus)
        except Exception as error:  # noqa: BLE001 - report and continue the sweep
            print(f"\ncorpus={corpus}  skipped: {error}")
            continue
        report(corpus, scanned, file_count, findings)
        total_findings += len(findings)

    print(f"\nTotal affected files: {total_findings}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
