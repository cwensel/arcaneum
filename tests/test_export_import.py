"""Tests for collection export/import functionality (RDR-017)."""

import gzip
import json
import os
import struct
import tempfile
from pathlib import Path

import msgpack
import numpy as np
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    HnswConfigDiff,
    PointStruct,
    VectorParams,
)

from arcaneum.cli.export_import import (
    MAGIC,
    VERSION,
    BinaryExporter,
    BinaryImporter,
    ExportHeader,
    JsonlExporter,
    JsonlImporter,
    attach_root_prefix,
    build_export_filter,
    detect_root_prefix,
    remap_path,
    strip_root_prefix,
)
from arcaneum.indexing.collection_metadata import (
    METADATA_POINT_ID,
    set_collection_metadata,
    get_collection_metadata,
)


QDRANT_URL = "http://localhost:6333"


@pytest.fixture
def qdrant_client():
    """Provide Qdrant client connected to test server."""
    client = QdrantClient(url=QDRANT_URL)
    yield client
    # Cleanup: delete test collections
    try:
        collections = client.get_collections()
        for col in collections.collections:
            if col.name.startswith("test_export"):
                client.delete_collection(col.name)
    except Exception:
        pass


@pytest.fixture
def test_collection(qdrant_client):
    """Create a test collection with sample data."""
    collection_name = "test_export_source"

    # Delete if exists
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "stella": VectorParams(size=1024, distance=Distance.COSINE)
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )

    # Set collection metadata
    set_collection_metadata(
        client=qdrant_client,
        collection_name=collection_name,
        collection_type="pdf",
        model="stella",
    )

    # Add sample points
    points = []
    for i in range(10):
        vec = np.random.rand(1024).astype(np.float32).tolist()
        points.append(
            PointStruct(
                id=f"point-{i}",
                vector={"stella": vec},
                payload={
                    "file_path": f"/Users/alice/docs/report-{i}.pdf",
                    "file_hash": f"hash-{i}",
                    "chunk_index": i,
                    "chunk_count": 10,
                },
            )
        )

    qdrant_client.upsert(collection_name=collection_name, points=points)

    yield collection_name

    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


@pytest.fixture
def test_code_collection(qdrant_client):
    """Create a test collection with code-style data."""
    collection_name = "test_export_code"

    # Delete if exists
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass

    # Create collection
    qdrant_client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "jina-code-0.5b": VectorParams(size=896, distance=Distance.COSINE)
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
    )

    # Set collection metadata
    set_collection_metadata(
        client=qdrant_client,
        collection_name=collection_name,
        collection_type="code",
        model="jina-code-0.5b",
    )

    # Add sample points for two repos
    points = []
    repos = [
        ("arcaneum", "main"),
        ("other-project", "develop"),
    ]

    for repo_name, branch in repos:
        for i in range(5):
            vec = np.random.rand(896).astype(np.float32).tolist()
            points.append(
                PointStruct(
                    id=f"{repo_name}-{branch}-{i}",
                    vector={"jina-code-0.5b": vec},
                    payload={
                        "file_path": f"/Users/alice/repos/{repo_name}/src/file-{i}.py",
                        "git_project_name": repo_name,
                        "git_project_identifier": f"{repo_name}#{branch}",
                        "git_branch": branch,
                        "git_project_root": f"/Users/alice/repos/{repo_name}",
                        "chunk_index": i,
                        "chunk_count": 5,
                    },
                )
            )

    qdrant_client.upsert(collection_name=collection_name, points=points)

    yield collection_name

    # Cleanup
    try:
        qdrant_client.delete_collection(collection_name)
    except Exception:
        pass


class TestPathUtilities:
    """Tests for path manipulation utilities."""

    def test_detect_root_prefix(self):
        """Test common root prefix detection."""
        paths = [
            "/Users/alice/docs/report1.pdf",
            "/Users/alice/docs/report2.pdf",
            "/Users/alice/docs/subdir/report3.pdf",
        ]
        prefix = detect_root_prefix(paths)
        assert prefix == "/Users/alice/docs"

    def test_detect_root_prefix_no_common(self):
        """Test when no common prefix exists."""
        paths = [
            "/Users/alice/docs/report.pdf",
            "/home/bob/docs/report.pdf",
        ]
        prefix = detect_root_prefix(paths)
        # Should return None or root-level directory
        assert prefix is None or prefix == "/"

    def test_detect_root_prefix_empty(self):
        """Test with empty list."""
        assert detect_root_prefix([]) is None

    def test_strip_root_prefix(self):
        """Test stripping root prefix."""
        path = "/Users/alice/docs/report.pdf"
        prefix = "/Users/alice"
        result = strip_root_prefix(path, prefix)
        assert result == "docs/report.pdf"

    def test_strip_root_prefix_no_match(self):
        """Test when prefix doesn't match."""
        path = "/Users/alice/docs/report.pdf"
        prefix = "/home/bob"
        result = strip_root_prefix(path, prefix)
        assert result == path  # Unchanged

    def test_attach_root_prefix(self):
        """Test attaching root prefix."""
        path = "docs/report.pdf"
        prefix = "/home/bob"
        result = attach_root_prefix(path, prefix)
        assert result == "/home/bob/docs/report.pdf"

    def test_attach_root_prefix_already_absolute(self):
        """Test attaching to already absolute path."""
        path = "/absolute/path.pdf"
        prefix = "/home/bob"
        result = attach_root_prefix(path, prefix)
        assert result == path  # Unchanged

    def test_remap_path(self):
        """Test path remapping."""
        path = "/Users/alice/docs/report.pdf"
        remaps = [("/Users/alice", "/home/bob")]
        result = remap_path(path, remaps)
        assert result == "/home/bob/docs/report.pdf"

    def test_remap_path_first_match(self):
        """Test that first matching remap is used."""
        path = "/Users/alice/docs/report.pdf"
        remaps = [
            ("/Users/alice/docs", "/data/documents"),
            ("/Users/alice", "/home/bob"),
        ]
        result = remap_path(path, remaps)
        assert result == "/data/documents/report.pdf"


class TestFilterBuilding:
    """Tests for export filter construction."""

    def test_build_export_filter_no_filters(self):
        """Test with no filters."""
        scroll_filter, path_filter = build_export_filter(
            includes=(), excludes=(), repos=()
        )
        assert scroll_filter is None
        assert path_filter is None

    def test_build_export_filter_include(self):
        """Test with include pattern."""
        scroll_filter, path_filter = build_export_filter(
            includes=("*.pdf",), excludes=(), repos=()
        )
        assert scroll_filter is None
        assert path_filter is not None

        # Test filter function
        assert path_filter({"file_path": "/path/to/report.pdf"}) is True
        assert path_filter({"file_path": "/path/to/report.txt"}) is False

    def test_build_export_filter_exclude(self):
        """Test with exclude pattern."""
        scroll_filter, path_filter = build_export_filter(
            includes=(), excludes=("*/drafts/*",), repos=()
        )
        assert scroll_filter is None
        assert path_filter is not None

        # Test filter function
        assert path_filter({"file_path": "/docs/final/report.pdf"}) is True
        assert path_filter({"file_path": "/docs/drafts/report.pdf"}) is False

    def test_build_export_filter_repo(self):
        """Test with repo filter."""
        scroll_filter, path_filter = build_export_filter(
            includes=(), excludes=(), repos=("arcaneum",)
        )
        assert scroll_filter is not None
        assert path_filter is None

    def test_build_export_filter_repo_with_branch(self):
        """Test with repo#branch filter."""
        scroll_filter, path_filter = build_export_filter(
            includes=(), excludes=(), repos=("arcaneum#main",)
        )
        assert scroll_filter is not None
        assert path_filter is None


class TestExportHeader:
    """Tests for ExportHeader serialization."""

    def test_header_to_dict(self):
        """Test header serialization to dict."""
        header = ExportHeader(
            collection_name="TestCollection",
            collection_type="pdf",
            model="stella",
            vector_config={"stella": {"size": 1024, "distance": "Cosine"}},
            point_count=100,
            root_prefix=None,
            detached=False,
            exported_at="2026-01-14T12:00:00",
        )
        data = header.to_dict()
        assert data["collection_name"] == "TestCollection"
        assert data["collection_type"] == "pdf"
        assert data["model"] == "stella"
        assert data["point_count"] == 100
        assert data["detached"] is False

    def test_header_from_dict(self):
        """Test header deserialization from dict."""
        data = {
            "collection_name": "TestCollection",
            "collection_type": "pdf",
            "model": "stella",
            "vector_config": {"stella": {"size": 1024, "distance": "Cosine"}},
            "point_count": 100,
            "root_prefix": None,
            "detached": False,
            "exported_at": "2026-01-14T12:00:00",
            "version": 1,
        }
        header = ExportHeader.from_dict(data)
        assert header.collection_name == "TestCollection"
        assert header.collection_type == "pdf"
        assert header.model == "stella"
        assert header.point_count == 100


class TestBinaryFormat:
    """Tests for binary export format structure."""

    def test_binary_export_format(self, qdrant_client, test_collection):
        """Test that binary export creates valid format."""
        exporter = BinaryExporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = exporter.export(
                collection_name=test_collection,
                output_path=output_path,
            )

            assert result.exported_count > 0
            assert output_path.exists()

            # Verify format structure
            with gzip.open(output_path, "rb") as f:
                # Check magic bytes
                magic = f.read(4)
                assert magic == MAGIC

                # Check version
                version = struct.unpack("B", f.read(1))[0]
                assert version == VERSION

                # Check header
                header_len = struct.unpack("<I", f.read(4))[0]
                header_data = msgpack.unpackb(f.read(header_len))
                assert header_data["collection_name"] == test_collection
                assert header_data["collection_type"] == "pdf"
                assert header_data["model"] == "stella"
        finally:
            output_path.unlink(missing_ok=True)

    def test_binary_roundtrip(self, qdrant_client, test_collection):
        """Test export and import roundtrip preserves data."""
        exporter = BinaryExporter(qdrant_client)
        importer = BinaryImporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        target_collection = "test_export_imported"

        try:
            # Export
            export_result = exporter.export(
                collection_name=test_collection,
                output_path=output_path,
            )

            # Import to new collection
            import_result = importer.import_collection(
                input_path=output_path,
                target_name=target_collection,
            )

            # Verify counts match
            assert import_result.imported_count == export_result.exported_count

            # Verify collection metadata
            metadata = get_collection_metadata(qdrant_client, target_collection)
            assert metadata.get("collection_type") == "pdf"
            assert metadata.get("model") == "stella"

            # Verify point count
            info = qdrant_client.get_collection(target_collection)
            assert info.points_count == export_result.exported_count

        finally:
            output_path.unlink(missing_ok=True)
            try:
                qdrant_client.delete_collection(target_collection)
            except Exception:
                pass


class TestJsonlFormat:
    """Tests for JSONL export format."""

    def test_jsonl_export_format(self, qdrant_client, test_collection):
        """Test that JSONL export creates valid format."""
        exporter = JsonlExporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            output_path = Path(f.name)

        try:
            result = exporter.export(
                collection_name=test_collection,
                output_path=output_path,
            )

            assert result.exported_count > 0
            assert output_path.exists()

            # Verify format structure
            with open(output_path, "r") as f:
                # First line is header
                header = json.loads(f.readline())
                assert header["_header"] is True
                assert header["collection_name"] == test_collection
                assert header["collection_type"] == "pdf"

                # Remaining lines are points
                point_count = 0
                for line in f:
                    point = json.loads(line)
                    assert "id" in point
                    assert "vector" in point
                    assert "payload" in point
                    point_count += 1

                assert point_count == result.exported_count
        finally:
            output_path.unlink(missing_ok=True)

    def test_jsonl_roundtrip(self, qdrant_client, test_collection):
        """Test JSONL export and import roundtrip."""
        exporter = JsonlExporter(qdrant_client)
        importer = JsonlImporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
            output_path = Path(f.name)

        target_collection = "test_export_jsonl_imported"

        try:
            # Export
            export_result = exporter.export(
                collection_name=test_collection,
                output_path=output_path,
            )

            # Import to new collection
            import_result = importer.import_collection(
                input_path=output_path,
                target_name=target_collection,
            )

            # Verify counts match
            assert import_result.imported_count == export_result.exported_count

        finally:
            output_path.unlink(missing_ok=True)
            try:
                qdrant_client.delete_collection(target_collection)
            except Exception:
                pass


class TestDetachedExport:
    """Tests for detached export (relative paths)."""

    def test_detached_export(self, qdrant_client, test_collection):
        """Test export with --detach strips root prefix."""
        exporter = BinaryExporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        try:
            result = exporter.export(
                collection_name=test_collection,
                output_path=output_path,
                detach=True,
            )

            assert result.detached is True
            assert result.root_prefix is not None
            assert result.root_prefix.startswith("/Users/alice")

            # Verify paths in export are relative
            with gzip.open(output_path, "rb") as f:
                # Skip header
                f.read(4)  # magic
                f.read(1)  # version
                header_len = struct.unpack("<I", f.read(4))[0]
                header_data = msgpack.unpackb(f.read(header_len))

                assert header_data["detached"] is True
                assert header_data["root_prefix"] is not None

                # Read first point
                unpacker = msgpack.Unpacker(f, raw=False)
                point = next(unpacker)
                if point is not None:
                    file_path = point["payload"].get("file_path", "")
                    # Should not be absolute
                    assert not file_path.startswith("/")
        finally:
            output_path.unlink(missing_ok=True)

    def test_attach_on_import(self, qdrant_client, test_collection):
        """Test import with --attach prepends new root."""
        exporter = BinaryExporter(qdrant_client)
        importer = BinaryImporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        target_collection = "test_export_attached"
        new_root = "/home/bob"

        try:
            # Export with detach
            exporter.export(
                collection_name=test_collection,
                output_path=output_path,
                detach=True,
            )

            # Import with attach
            importer.import_collection(
                input_path=output_path,
                target_name=target_collection,
                attach_root=new_root,
            )

            # Verify paths have new root
            points, _ = qdrant_client.scroll(
                collection_name=target_collection,
                with_payload=True,
                with_vectors=False,
                limit=1,
            )

            # Filter out metadata point
            for point in points:
                if str(point.id) != METADATA_POINT_ID:
                    file_path = point.payload.get("file_path", "")
                    assert file_path.startswith(new_root)
                    break

        finally:
            output_path.unlink(missing_ok=True)
            try:
                qdrant_client.delete_collection(target_collection)
            except Exception:
                pass


class TestPathRemapping:
    """Tests for path remapping on import."""

    def test_remap_on_import(self, qdrant_client, test_collection):
        """Test import with --remap substitutes paths."""
        exporter = BinaryExporter(qdrant_client)
        importer = BinaryImporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        target_collection = "test_export_remapped"

        try:
            # Export without detach
            exporter.export(
                collection_name=test_collection,
                output_path=output_path,
                detach=False,
            )

            # Import with remap
            importer.import_collection(
                input_path=output_path,
                target_name=target_collection,
                path_remaps=[("/Users/alice", "/home/bob")],
            )

            # Verify paths are remapped
            points, _ = qdrant_client.scroll(
                collection_name=target_collection,
                with_payload=True,
                with_vectors=False,
                limit=1,
            )

            for point in points:
                if str(point.id) != METADATA_POINT_ID:
                    file_path = point.payload.get("file_path", "")
                    assert file_path.startswith("/home/bob")
                    assert "/Users/alice" not in file_path
                    break

        finally:
            output_path.unlink(missing_ok=True)
            try:
                qdrant_client.delete_collection(target_collection)
            except Exception:
                pass


class TestRepoFiltering:
    """Tests for repo-based filtering on export."""

    def test_filter_by_repo_name(self, qdrant_client, test_code_collection):
        """Test filtering export by repo name."""
        exporter = BinaryExporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        try:
            # Build filter for arcaneum repo only
            scroll_filter, _ = build_export_filter(
                includes=(), excludes=(), repos=("arcaneum",)
            )

            result = exporter.export(
                collection_name=test_code_collection,
                output_path=output_path,
                scroll_filter=scroll_filter,
            )

            # Should have 5 points from arcaneum + 1 metadata point
            # (metadata point is always included)
            assert result.exported_count == 6
        finally:
            output_path.unlink(missing_ok=True)

    def test_filter_by_repo_branch(self, qdrant_client, test_code_collection):
        """Test filtering export by repo#branch."""
        exporter = BinaryExporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        try:
            # Build filter for arcaneum#main only
            scroll_filter, _ = build_export_filter(
                includes=(), excludes=(), repos=("arcaneum#main",)
            )

            result = exporter.export(
                collection_name=test_code_collection,
                output_path=output_path,
                scroll_filter=scroll_filter,
            )

            # Should have 5 points from arcaneum#main + 1 metadata point
            assert result.exported_count == 6
        finally:
            output_path.unlink(missing_ok=True)


class TestImportValidation:
    """Tests for import validation."""

    def test_import_refuses_existing_collection(self, qdrant_client, test_collection):
        """Test that import refuses to overwrite existing collection."""
        exporter = BinaryExporter(qdrant_client)
        importer = BinaryImporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            output_path = Path(f.name)

        try:
            # Export
            exporter.export(
                collection_name=test_collection,
                output_path=output_path,
            )

            # Try to import into existing collection
            with pytest.raises(ValueError, match="already exists"):
                importer.import_collection(
                    input_path=output_path,
                    target_name=test_collection,
                )
        finally:
            output_path.unlink(missing_ok=True)

    def test_import_invalid_magic(self, qdrant_client):
        """Test that import rejects files with invalid magic."""
        importer = BinaryImporter(qdrant_client)

        with tempfile.NamedTemporaryFile(suffix=".arcexp", delete=False) as f:
            # Write invalid data
            with gzip.open(f.name, "wb") as gz:
                gz.write(b"XXXX")  # Invalid magic
            output_path = Path(f.name)

        try:
            with pytest.raises(ValueError, match="Invalid file format"):
                importer.import_collection(
                    input_path=output_path,
                    target_name="test_import_invalid",
                )
        finally:
            output_path.unlink(missing_ok=True)
