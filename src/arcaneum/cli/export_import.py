"""
Collection export/import for cross-machine migration (RDR-017).

This module provides portable export and import functionality for Qdrant collections,
enabling migration between machines and selective export capabilities.

Supported formats:
- Binary (.arcexp): Compact msgpack+numpy format with gzip compression (default)
- JSONL (.jsonl): Human-readable debug format (opt-in)

Key features:
- Streaming I/O for memory efficiency with large collections
- Path filtering (--include/--exclude) using fnmatch globs
- Repo filtering (--repo) for code collections
- Detached exports (--detach) for shareable archives with relative paths
- Path remapping (--attach, --remap) for cross-machine migration
"""

import fnmatch
import gzip
import json
import logging
import os
import struct
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import msgpack
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchValue,
    PointStruct,
    VectorParams,
)

from arcaneum.indexing.collection_metadata import (
    METADATA_POINT_ID,
    get_collection_metadata,
    set_collection_metadata,
)

logger = logging.getLogger(__name__)

# Binary format constants
MAGIC = b"ARCE"  # Arcaneum Export
VERSION = 1
EOF_MARKER = None


@dataclass
class ExportHeader:
    """Header information for export files."""

    collection_name: str
    collection_type: Optional[str]
    model: Optional[str]
    vector_config: Dict[str, Dict[str, Any]]  # {name: {size, distance}}
    point_count: int
    root_prefix: Optional[str]  # Common path prefix (auto-detected)
    detached: bool  # True if paths are relative (root stripped)
    exported_at: str  # ISO timestamp
    version: int = VERSION

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "collection_name": self.collection_name,
            "collection_type": self.collection_type,
            "model": self.model,
            "vector_config": self.vector_config,
            "point_count": self.point_count,
            "root_prefix": self.root_prefix,
            "detached": self.detached,
            "exported_at": self.exported_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExportHeader":
        """Create from dictionary."""
        return cls(
            collection_name=data["collection_name"],
            collection_type=data.get("collection_type"),
            model=data.get("model"),
            vector_config=data["vector_config"],
            point_count=data["point_count"],
            root_prefix=data.get("root_prefix"),
            detached=data.get("detached", False),
            exported_at=data["exported_at"],
            version=data.get("version", 1),
        )


@dataclass
class ExportResult:
    """Result of an export operation."""

    output_path: str
    exported_count: int
    skipped_count: int
    file_size_bytes: int
    collection_name: str
    collection_type: Optional[str]
    detached: bool
    root_prefix: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "output_path": self.output_path,
            "exported_count": self.exported_count,
            "skipped_count": self.skipped_count,
            "file_size_bytes": self.file_size_bytes,
            "collection_name": self.collection_name,
            "collection_type": self.collection_type,
            "detached": self.detached,
            "root_prefix": self.root_prefix,
        }


@dataclass
class ImportResult:
    """Result of an import operation."""

    collection_name: str
    imported_count: int
    collection_type: Optional[str]
    source_file: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON output."""
        return {
            "collection_name": self.collection_name,
            "imported_count": self.imported_count,
            "collection_type": self.collection_type,
            "source_file": self.source_file,
        }


def build_export_filter(
    includes: Tuple[str, ...],
    excludes: Tuple[str, ...],
    repos: Tuple[str, ...],
) -> Tuple[Optional[Filter], Optional[Callable[[Dict], bool]]]:
    """Build Qdrant filter and path filter function for export.

    Args:
        includes: Glob patterns for file_path inclusion
        excludes: Glob patterns for file_path exclusion
        repos: Repo name or repo#branch filters (code collections)

    Returns:
        Tuple of (scroll_filter, path_filter_func)
        - scroll_filter: Qdrant Filter for repo filtering (None if no repo filter)
        - path_filter_func: Function to filter points by path patterns (None if no patterns)
    """
    scroll_filter = None
    path_filter_func = None

    # Build Qdrant filter for repo filtering (code collections)
    if repos:
        conditions = []
        for repo in repos:
            if "#" in repo:
                # repo#branch format - filter on git_project_identifier
                conditions.append(
                    FieldCondition(
                        key="git_project_identifier",
                        match=MatchValue(value=repo),
                    )
                )
            else:
                # repo name only - filter on git_project_name
                conditions.append(
                    FieldCondition(
                        key="git_project_name",
                        match=MatchValue(value=repo),
                    )
                )
        # Multiple repos = OR (should match any)
        if len(conditions) == 1:
            scroll_filter = Filter(must=conditions)
        else:
            scroll_filter = Filter(should=conditions)

    # Build path filter function for include/exclude patterns
    if includes or excludes:

        def path_filter(payload: Dict) -> bool:
            file_path = payload.get("file_path", "")
            if not file_path:
                return True  # No path to filter on

            # Include check: if any includes specified, must match at least one
            if includes:
                matches_include = any(
                    fnmatch.fnmatch(file_path, pattern) for pattern in includes
                )
                if not matches_include:
                    return False

            # Exclude check: must not match any exclude pattern
            if excludes:
                matches_exclude = any(
                    fnmatch.fnmatch(file_path, pattern) for pattern in excludes
                )
                if matches_exclude:
                    return False

            return True

        path_filter_func = path_filter

    return scroll_filter, path_filter_func


def detect_root_prefix(file_paths: List[str]) -> Optional[str]:
    """Detect common root prefix from file paths.

    Args:
        file_paths: List of absolute file paths

    Returns:
        Common directory prefix, or None if no common prefix
    """
    if not file_paths:
        return None

    # Filter to absolute paths only
    abs_paths = [p for p in file_paths if p.startswith("/")]
    if not abs_paths:
        return None

    # Find common prefix
    prefix = os.path.commonpath(abs_paths)

    # Ensure prefix is a directory (not partial filename match)
    if prefix and not prefix.endswith("/"):
        prefix = os.path.dirname(prefix)

    return prefix if prefix and prefix != "/" else None


def strip_root_prefix(path: str, prefix: str) -> str:
    """Strip root prefix from path, returning relative path.

    Args:
        path: Absolute file path
        prefix: Root prefix to strip

    Returns:
        Relative path with prefix removed
    """
    if path.startswith(prefix):
        rel_path = path[len(prefix) :]
        # Remove leading slash
        if rel_path.startswith("/"):
            rel_path = rel_path[1:]
        return rel_path
    return path


def attach_root_prefix(path: str, prefix: str) -> str:
    """Attach root prefix to relative path.

    Args:
        path: Relative file path
        prefix: Root prefix to prepend

    Returns:
        Absolute path with prefix
    """
    if path.startswith("/"):
        return path  # Already absolute
    # Ensure prefix ends with / for proper joining
    if not prefix.endswith("/"):
        prefix = prefix + "/"
    return prefix + path


def remap_path(path: str, remaps: List[Tuple[str, str]]) -> str:
    """Apply path remapping substitutions.

    Args:
        path: Original file path
        remaps: List of (old_prefix, new_prefix) tuples

    Returns:
        Remapped path (first matching substitution applied)
    """
    for old, new in remaps:
        if path.startswith(old):
            return new + path[len(old) :]
    return path


class BaseExporter(ABC):
    """Base class for collection exporters."""

    def __init__(self, client: QdrantClient):
        self.client = client

    @abstractmethod
    def export(
        self,
        collection_name: str,
        output_path: Path,
        scroll_filter: Optional[Filter] = None,
        path_filter: Optional[Callable[[Dict], bool]] = None,
        detach: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExportResult:
        """Export collection to file."""
        pass

    def _get_collection_info(
        self, collection_name: str
    ) -> Tuple[Dict[str, Dict[str, Any]], int]:
        """Get collection vector config and point count.

        Returns:
            Tuple of (vector_config, point_count)
        """
        info = self.client.get_collection(collection_name)
        point_count = info.points_count

        vector_config = {}
        if hasattr(info.config.params, "vectors") and isinstance(
            info.config.params.vectors, dict
        ):
            for name, params in info.config.params.vectors.items():
                vector_config[name] = {
                    "size": params.size,
                    "distance": str(params.distance),
                }

        return vector_config, point_count

    def _scroll_points(
        self,
        collection_name: str,
        scroll_filter: Optional[Filter] = None,
        path_filter: Optional[Callable[[Dict], bool]] = None,
        include_metadata_point: bool = True,
    ) -> Iterator[Any]:
        """Scroll through collection points with optional filtering.

        Args:
            collection_name: Collection to scroll
            scroll_filter: Qdrant filter for server-side filtering
            path_filter: Client-side path filter function
            include_metadata_point: If True, always include metadata point

        Yields:
            Point objects matching filters
        """
        # First, yield metadata point if requested (always include regardless of filters)
        if include_metadata_point:
            metadata_points = self.client.retrieve(
                collection_name=collection_name,
                ids=[METADATA_POINT_ID],
                with_payload=True,
                with_vectors=True,
            )
            if metadata_points:
                yield metadata_points[0]

        # Scroll through remaining points
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=scroll_filter,
                with_vectors=True,
                with_payload=True,
                limit=100,
                offset=offset,
            )

            if not points:
                break

            for point in points:
                # Skip metadata point (already yielded)
                if str(point.id) == METADATA_POINT_ID:
                    continue

                # Apply path filter if specified
                if path_filter and point.payload:
                    if not path_filter(point.payload):
                        continue

                yield point

            if offset is None:
                break


class BinaryExporter(BaseExporter):
    """Export collection to compressed binary format (.arcexp)."""

    def export(
        self,
        collection_name: str,
        output_path: Path,
        scroll_filter: Optional[Filter] = None,
        path_filter: Optional[Callable[[Dict], bool]] = None,
        detach: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExportResult:
        """Export collection to .arcexp binary format.

        Args:
            collection_name: Name of collection to export
            output_path: Output file path
            scroll_filter: Qdrant filter for repo filtering
            path_filter: Function to filter by path patterns
            detach: If True, strip root prefix and store relative paths
            progress_callback: Optional callback(current, total) for progress

        Returns:
            ExportResult with export statistics
        """
        # Get collection info
        vector_config, point_count = self._get_collection_info(collection_name)
        metadata = get_collection_metadata(self.client, collection_name)

        # First pass: collect file paths for root prefix detection if detaching
        root_prefix = None
        if detach:
            file_paths = []
            for point in self._scroll_points(
                collection_name, scroll_filter, path_filter, include_metadata_point=False
            ):
                if point.payload:
                    fp = point.payload.get("file_path")
                    if fp:
                        file_paths.append(fp)
            root_prefix = detect_root_prefix(file_paths)
            logger.info(f"Detected root prefix: {root_prefix}")

        # Create header
        header = ExportHeader(
            collection_name=collection_name,
            collection_type=metadata.get("collection_type"),
            model=metadata.get("model"),
            vector_config=vector_config,
            point_count=point_count,
            root_prefix=root_prefix,
            detached=detach,
            exported_at=datetime.now().isoformat(),
        )

        exported_count = 0
        skipped_count = 0

        with gzip.open(output_path, "wb") as f:
            # Write magic bytes and version
            f.write(MAGIC)
            f.write(struct.pack("B", VERSION))

            # Write header
            header_bytes = msgpack.packb(header.to_dict())
            f.write(struct.pack("<I", len(header_bytes)))
            f.write(header_bytes)

            # Stream points
            for point in self._scroll_points(
                collection_name, scroll_filter, path_filter
            ):
                # Serialize point
                point_data = self._serialize_point(point, root_prefix if detach else None)
                f.write(msgpack.packb(point_data))
                exported_count += 1

                if progress_callback:
                    progress_callback(exported_count, point_count)

            # Write EOF marker
            f.write(msgpack.packb(EOF_MARKER))

        # Get file size
        file_size = output_path.stat().st_size

        return ExportResult(
            output_path=str(output_path),
            exported_count=exported_count,
            skipped_count=skipped_count,
            file_size_bytes=file_size,
            collection_name=collection_name,
            collection_type=metadata.get("collection_type"),
            detached=detach,
            root_prefix=root_prefix,
        )

    def _serialize_point(
        self, point: Any, root_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Serialize point to msgpack-compatible format.

        Args:
            point: Qdrant point object
            root_prefix: If provided, strip this prefix from paths

        Returns:
            Dictionary with id, vectors (as binary), and payload
        """
        # Convert vectors to binary
        vectors_binary = {}
        if isinstance(point.vector, dict):
            for name, vec in point.vector.items():
                vectors_binary[name] = np.array(vec, dtype=np.float32).tobytes()
        else:
            # Single unnamed vector
            vectors_binary["_default"] = np.array(
                point.vector, dtype=np.float32
            ).tobytes()

        # Process payload
        payload = dict(point.payload) if point.payload else {}

        # Strip root prefix from paths if detaching
        if root_prefix:
            if "file_path" in payload:
                payload["file_path"] = strip_root_prefix(payload["file_path"], root_prefix)
            if "git_project_root" in payload:
                payload["git_project_root"] = strip_root_prefix(
                    payload["git_project_root"], root_prefix
                )

        return {
            "id": str(point.id),
            "vectors": vectors_binary,
            "payload": payload,
        }


class BinaryImporter:
    """Import collection from compressed binary format (.arcexp)."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def import_collection(
        self,
        input_path: Path,
        target_name: Optional[str] = None,
        attach_root: Optional[str] = None,
        path_remaps: Optional[List[Tuple[str, str]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ImportResult:
        """Import collection from .arcexp binary format.

        Args:
            input_path: Input file path
            target_name: Target collection name (uses original if not specified)
            attach_root: Root path to prepend to relative paths (for detached exports)
            path_remaps: List of (old, new) path substitutions
            progress_callback: Optional callback(current, total) for progress

        Returns:
            ImportResult with import statistics
        """
        with gzip.open(input_path, "rb") as f:
            # Validate magic bytes
            magic = f.read(4)
            if magic != MAGIC:
                raise ValueError(f"Invalid file format: expected ARCE magic, got {magic}")

            # Read version
            version = struct.unpack("B", f.read(1))[0]
            if version > VERSION:
                raise ValueError(
                    f"Unsupported format version: {version} (max supported: {VERSION})"
                )

            # Read header
            header_len = struct.unpack("<I", f.read(4))[0]
            header_data = msgpack.unpackb(f.read(header_len))
            header = ExportHeader.from_dict(header_data)

            # Determine target collection name
            collection_name = target_name or header.collection_name

            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                raise ValueError(
                    f"Collection '{collection_name}' already exists. "
                    "Use --into to specify a different name."
                )
            except Exception as e:
                if "not found" not in str(e).lower() and "doesn't exist" not in str(e).lower():
                    raise

            # Create collection with original configuration
            self._create_collection(collection_name, header)

            # Import points in batches
            batch = []
            batch_size = 100
            imported_count = 0

            unpacker = msgpack.Unpacker(f, raw=False)
            for point_data in unpacker:
                if point_data is None:  # EOF marker
                    break

                point = self._deserialize_point(
                    point_data, header, attach_root, path_remaps
                )
                batch.append(point)

                if len(batch) >= batch_size:
                    self.client.upsert(collection_name=collection_name, points=batch)
                    imported_count += len(batch)
                    if progress_callback:
                        progress_callback(imported_count, header.point_count)
                    batch = []

            # Upsert remaining batch
            if batch:
                self.client.upsert(collection_name=collection_name, points=batch)
                imported_count += len(batch)
                if progress_callback:
                    progress_callback(imported_count, header.point_count)

        return ImportResult(
            collection_name=collection_name,
            imported_count=imported_count,
            collection_type=header.collection_type,
            source_file=str(input_path),
        )

    def _create_collection(self, collection_name: str, header: ExportHeader) -> None:
        """Create collection with configuration from header.

        Args:
            collection_name: Name for new collection
            header: Export header with vector configuration
        """
        # Build vectors config
        vectors_config = {}
        for name, config in header.vector_config.items():
            distance_str = config.get("distance", "Cosine")
            # Handle Distance enum string format (e.g., "Distance.COSINE")
            if "COSINE" in distance_str.upper():
                distance = Distance.COSINE
            elif "EUCLID" in distance_str.upper():
                distance = Distance.EUCLID
            elif "DOT" in distance_str.upper():
                distance = Distance.DOT
            else:
                distance = Distance.COSINE

            vectors_config[name] = VectorParams(
                size=config["size"],
                distance=distance,
            )

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

        # Set collection metadata if available
        if header.collection_type and header.model:
            set_collection_metadata(
                client=self.client,
                collection_name=collection_name,
                collection_type=header.collection_type,
                model=header.model,
            )

    def _deserialize_point(
        self,
        point_data: Dict[str, Any],
        header: ExportHeader,
        attach_root: Optional[str] = None,
        path_remaps: Optional[List[Tuple[str, str]]] = None,
    ) -> PointStruct:
        """Deserialize point from binary format.

        Args:
            point_data: Serialized point data
            header: Export header
            attach_root: Root path to prepend (for detached exports)
            path_remaps: Path substitutions to apply

        Returns:
            PointStruct ready for upsert
        """
        # Reconstruct vectors from binary
        vectors = {}
        for name, vec_bytes in point_data["vectors"].items():
            vec = np.frombuffer(vec_bytes, dtype=np.float32).tolist()
            vectors[name] = vec

        # Process payload
        payload = point_data.get("payload", {})

        # Apply path transformations
        if header.detached and attach_root:
            # Attach root to relative paths
            if "file_path" in payload:
                payload["file_path"] = attach_root_prefix(payload["file_path"], attach_root)
            if "git_project_root" in payload:
                payload["git_project_root"] = attach_root_prefix(
                    payload["git_project_root"], attach_root
                )
        elif path_remaps:
            # Apply explicit path remappings
            if "file_path" in payload:
                payload["file_path"] = remap_path(payload["file_path"], path_remaps)
            if "git_project_root" in payload:
                payload["git_project_root"] = remap_path(
                    payload["git_project_root"], path_remaps
                )

        return PointStruct(
            id=point_data["id"],
            vector=vectors,
            payload=payload,
        )


class JsonlExporter(BaseExporter):
    """Export collection to JSONL format for debugging."""

    def export(
        self,
        collection_name: str,
        output_path: Path,
        scroll_filter: Optional[Filter] = None,
        path_filter: Optional[Callable[[Dict], bool]] = None,
        detach: bool = False,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ExportResult:
        """Export collection to JSONL format.

        Args:
            collection_name: Name of collection to export
            output_path: Output file path
            scroll_filter: Qdrant filter for repo filtering
            path_filter: Function to filter by path patterns
            detach: If True, strip root prefix and store relative paths
            progress_callback: Optional callback(current, total) for progress

        Returns:
            ExportResult with export statistics
        """
        # Get collection info
        vector_config, point_count = self._get_collection_info(collection_name)
        metadata = get_collection_metadata(self.client, collection_name)

        # Detect root prefix if detaching
        root_prefix = None
        if detach:
            file_paths = []
            for point in self._scroll_points(
                collection_name, scroll_filter, path_filter, include_metadata_point=False
            ):
                if point.payload:
                    fp = point.payload.get("file_path")
                    if fp:
                        file_paths.append(fp)
            root_prefix = detect_root_prefix(file_paths)

        # Create header
        header = {
            "_header": True,
            "_version": VERSION,
            "collection_name": collection_name,
            "collection_type": metadata.get("collection_type"),
            "model": metadata.get("model"),
            "vector_config": vector_config,
            "point_count": point_count,
            "root_prefix": root_prefix,
            "detached": detach,
            "exported_at": datetime.now().isoformat(),
        }

        exported_count = 0

        with open(output_path, "w") as f:
            # Write header line
            f.write(json.dumps(header) + "\n")

            # Stream points
            for point in self._scroll_points(
                collection_name, scroll_filter, path_filter
            ):
                point_data = self._serialize_point(point, root_prefix if detach else None)
                f.write(json.dumps(point_data) + "\n")
                exported_count += 1

                if progress_callback:
                    progress_callback(exported_count, point_count)

        # Get file size
        file_size = output_path.stat().st_size

        return ExportResult(
            output_path=str(output_path),
            exported_count=exported_count,
            skipped_count=0,
            file_size_bytes=file_size,
            collection_name=collection_name,
            collection_type=metadata.get("collection_type"),
            detached=detach,
            root_prefix=root_prefix,
        )

    def _serialize_point(
        self, point: Any, root_prefix: Optional[str] = None
    ) -> Dict[str, Any]:
        """Serialize point to JSON-compatible format."""
        # Convert vectors to lists
        vectors = {}
        if isinstance(point.vector, dict):
            for name, vec in point.vector.items():
                vectors[name] = list(vec) if not isinstance(vec, list) else vec
        else:
            vectors["_default"] = (
                list(point.vector) if not isinstance(point.vector, list) else point.vector
            )

        # Process payload
        payload = dict(point.payload) if point.payload else {}

        # Strip root prefix if detaching
        if root_prefix:
            if "file_path" in payload:
                payload["file_path"] = strip_root_prefix(payload["file_path"], root_prefix)
            if "git_project_root" in payload:
                payload["git_project_root"] = strip_root_prefix(
                    payload["git_project_root"], root_prefix
                )

        return {
            "id": str(point.id),
            "vector": vectors,
            "payload": payload,
        }


class JsonlImporter:
    """Import collection from JSONL format."""

    def __init__(self, client: QdrantClient):
        self.client = client

    def import_collection(
        self,
        input_path: Path,
        target_name: Optional[str] = None,
        attach_root: Optional[str] = None,
        path_remaps: Optional[List[Tuple[str, str]]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> ImportResult:
        """Import collection from JSONL format.

        Args:
            input_path: Input file path
            target_name: Target collection name (uses original if not specified)
            attach_root: Root path to prepend to relative paths (for detached exports)
            path_remaps: List of (old, new) path substitutions
            progress_callback: Optional callback(current, total) for progress

        Returns:
            ImportResult with import statistics
        """
        with open(input_path, "r") as f:
            # Read header
            header_line = f.readline()
            header_data = json.loads(header_line)

            if not header_data.get("_header"):
                raise ValueError("Invalid JSONL format: missing header line")

            header = ExportHeader(
                collection_name=header_data["collection_name"],
                collection_type=header_data.get("collection_type"),
                model=header_data.get("model"),
                vector_config=header_data["vector_config"],
                point_count=header_data["point_count"],
                root_prefix=header_data.get("root_prefix"),
                detached=header_data.get("detached", False),
                exported_at=header_data["exported_at"],
                version=header_data.get("_version", 1),
            )

            # Determine target collection name
            collection_name = target_name or header.collection_name

            # Check if collection exists
            try:
                self.client.get_collection(collection_name)
                raise ValueError(
                    f"Collection '{collection_name}' already exists. "
                    "Use --into to specify a different name."
                )
            except Exception as e:
                if "not found" not in str(e).lower() and "doesn't exist" not in str(e).lower():
                    raise

            # Create collection
            self._create_collection(collection_name, header)

            # Import points in batches
            batch = []
            batch_size = 100
            imported_count = 0

            for line in f:
                line = line.strip()
                if not line:
                    continue

                point_data = json.loads(line)
                point = self._deserialize_point(
                    point_data, header, attach_root, path_remaps
                )
                batch.append(point)

                if len(batch) >= batch_size:
                    self.client.upsert(collection_name=collection_name, points=batch)
                    imported_count += len(batch)
                    if progress_callback:
                        progress_callback(imported_count, header.point_count)
                    batch = []

            # Upsert remaining batch
            if batch:
                self.client.upsert(collection_name=collection_name, points=batch)
                imported_count += len(batch)
                if progress_callback:
                    progress_callback(imported_count, header.point_count)

        return ImportResult(
            collection_name=collection_name,
            imported_count=imported_count,
            collection_type=header.collection_type,
            source_file=str(input_path),
        )

    def _create_collection(self, collection_name: str, header: ExportHeader) -> None:
        """Create collection with configuration from header."""
        # Build vectors config
        vectors_config = {}
        for name, config in header.vector_config.items():
            distance_str = config.get("distance", "Cosine")
            if "COSINE" in distance_str.upper():
                distance = Distance.COSINE
            elif "EUCLID" in distance_str.upper():
                distance = Distance.EUCLID
            elif "DOT" in distance_str.upper():
                distance = Distance.DOT
            else:
                distance = Distance.COSINE

            vectors_config[name] = VectorParams(
                size=config["size"],
                distance=distance,
            )

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_config,
            hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
        )

        # Set collection metadata if available
        if header.collection_type and header.model:
            set_collection_metadata(
                client=self.client,
                collection_name=collection_name,
                collection_type=header.collection_type,
                model=header.model,
            )

    def _deserialize_point(
        self,
        point_data: Dict[str, Any],
        header: ExportHeader,
        attach_root: Optional[str] = None,
        path_remaps: Optional[List[Tuple[str, str]]] = None,
    ) -> PointStruct:
        """Deserialize point from JSON format."""
        # Vectors are already lists in JSON
        vectors = point_data.get("vector", {})

        # Process payload
        payload = point_data.get("payload", {})

        # Apply path transformations
        if header.detached and attach_root:
            if "file_path" in payload:
                payload["file_path"] = attach_root_prefix(payload["file_path"], attach_root)
            if "git_project_root" in payload:
                payload["git_project_root"] = attach_root_prefix(
                    payload["git_project_root"], attach_root
                )
        elif path_remaps:
            if "file_path" in payload:
                payload["file_path"] = remap_path(payload["file_path"], path_remaps)
            if "git_project_root" in payload:
                payload["git_project_root"] = remap_path(
                    payload["git_project_root"], path_remaps
                )

        return PointStruct(
            id=point_data["id"],
            vector=vectors,
            payload=payload,
        )
