from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from qdrant_client.models import FieldCondition, MatchValue, PayloadSchemaType

from arcaneum.indexing.collection_metadata import (
    FILE_MANIFEST_READY_FIELD,
    FILE_MANIFEST_SCHEMA_FIELD,
    FILE_MANIFEST_SCHEMA_VERSION,
    METADATA_POINT_ID,
    user_point_count,
)
from arcaneum.indexing.common.sync import (
    FILE_MANIFEST_PAYLOAD_KEY,
    FILE_MANIFEST_PAYLOAD_VALUE,
    MetadataBasedSync,
)


def _collection_info(payload_schema=None):
    vectors = {"dense": SimpleNamespace(size=3)}
    return SimpleNamespace(
        config=SimpleNamespace(params=SimpleNamespace(vectors=vectors)),
        payload_schema=payload_schema or {},
    )


def _ready_metadata_point():
    return SimpleNamespace(
        payload={
            "is_metadata": True,
            "collection_type": "code",
            "model": "model",
            FILE_MANIFEST_READY_FIELD: True,
            FILE_MANIFEST_SCHEMA_FIELD: FILE_MANIFEST_SCHEMA_VERSION,
        }
    )


def test_manifest_id_is_stable_per_collection_and_physical_path(tmp_path):
    sync = MetadataBasedSync(MagicMock())
    path = tmp_path / "source.py"

    assert sync.file_manifest_id("code", str(path)) == sync.file_manifest_id("code", str(path))
    assert sync.file_manifest_id("code", str(path)) != sync.file_manifest_id("other", str(path))


def test_manifest_vectors_are_cached_per_collection():
    qdrant = MagicMock()
    qdrant.get_collection.return_value = _collection_info()
    sync = MetadataBasedSync(qdrant)

    sync.build_file_manifest_point("code", "/repo/a.py", "a")
    sync.build_file_manifest_point("code", "/repo/b.py", "b")

    qdrant.get_collection.assert_called_once_with("code")


def test_ready_quick_hash_scan_uses_indexed_manifests_only():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_ready_metadata_point()]
    qdrant.scroll.return_value = (
        [SimpleNamespace(payload={"file_path": "/repo/a.py", "quick_hash": "quick"})],
        None,
    )

    pairs = MetadataBasedSync(qdrant)._get_indexed_quick_hashes("code")

    assert pairs == {("/repo/a.py", "quick")}
    qdrant.create_payload_index.assert_called_once_with(
        collection_name="code",
        field_name=FILE_MANIFEST_PAYLOAD_KEY,
        field_schema=PayloadSchemaType.KEYWORD,
        wait=True,
    )
    manifest_condition = FieldCondition(
        key=FILE_MANIFEST_PAYLOAD_KEY,
        match=MatchValue(value=FILE_MANIFEST_PAYLOAD_VALUE),
    )
    assert manifest_condition in qdrant.scroll.call_args.kwargs["scroll_filter"].must
    assert qdrant.scroll.call_args.kwargs["with_payload"] == [
        "file_path",
        "quick_hash",
        "file_hash",
        "chunk_count",
        "file_size",
        "store_type",
    ]


def test_legacy_backfill_writes_one_manifest_per_physical_path_then_stamps():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [
        SimpleNamespace(
            payload={
                "is_metadata": True,
                "collection_type": "code",
                "model": "model",
            }
        )
    ]
    qdrant.get_collection.return_value = _collection_info()
    qdrant.scroll.return_value = (
        [
            SimpleNamespace(
                payload={
                    "file_path": "/repo/a.py",
                    "file_paths": ["/repo/a.py", "/copy/a.py"],
                    "file_quick_hashes": {"/repo/a.py": "qa", "/copy/a.py": "qb"},
                    "file_hash": "content",
                    "chunk_count": 1,
                }
            )
        ],
        None,
    )
    progress = MagicMock()

    count = MetadataBasedSync(qdrant).backfill_file_manifests("code", progress)

    assert count == 2
    assert progress.call_args.args == (1,)
    manifest_upsert = qdrant.upsert.call_args_list[0]
    points = manifest_upsert.kwargs["points"]
    assert {point.payload["file_path"] for point in points} == {"/repo/a.py", "/copy/a.py"}
    assert all(point.payload["is_metadata"] is True for point in points)
    assert all(
        point.payload[FILE_MANIFEST_PAYLOAD_KEY] == FILE_MANIFEST_PAYLOAD_VALUE for point in points
    )
    metadata_point = qdrant.upsert.call_args_list[-1].kwargs["points"][0]
    assert str(metadata_point.id) == METADATA_POINT_ID
    assert metadata_point.payload[FILE_MANIFEST_READY_FIELD] is True
    assert metadata_point.payload[FILE_MANIFEST_SCHEMA_FIELD] == FILE_MANIFEST_SCHEMA_VERSION


def test_failed_backfill_never_writes_readiness_stamp():
    qdrant = MagicMock()
    qdrant.scroll.side_effect = RuntimeError("scroll failed")

    with pytest.raises(RuntimeError, match="scroll failed"):
        MetadataBasedSync(qdrant).backfill_file_manifests("code")

    qdrant.upsert.assert_not_called()


def test_existing_manifest_index_conflict_is_idempotent_across_sync_instances():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_ready_metadata_point()]
    qdrant.get_collection.return_value = _collection_info()
    qdrant.create_payload_index.side_effect = RuntimeError("409 index already exists")
    qdrant.scroll.return_value = (
        [SimpleNamespace(payload={"file_path": "/repo/a.py", "quick_hash": "quick"})],
        None,
    )

    first = MetadataBasedSync(qdrant)._get_indexed_quick_hashes("code")
    second = MetadataBasedSync(qdrant)._get_indexed_file_paths_set("code")

    assert first == {("/repo/a.py", "quick")}
    assert second == {"/repo/a.py"}
    assert qdrant.create_payload_index.call_count == 2


def test_ready_manifest_scan_failure_is_not_treated_as_empty_collection():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_ready_metadata_point()]
    qdrant.get_collection.return_value = _collection_info(
        {FILE_MANIFEST_PAYLOAD_KEY: SimpleNamespace()}
    )
    qdrant.scroll.side_effect = RuntimeError("timeout")

    with pytest.raises(RuntimeError, match="Failed to scan file manifests"):
        MetadataBasedSync(qdrant)._get_indexed_quick_hashes("code")


def test_copy_manifest_rebuilds_missing_source_from_existing_chunks():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = []
    qdrant.scroll.return_value = (
        [SimpleNamespace(payload={"file_hash": "content", "chunk_count": 3})],
        None,
    )
    qdrant.get_collection.return_value = _collection_info()

    MetadataBasedSync(qdrant).copy_file_manifest(
        "code", "/old.py", "/new.py", "q", file_size=10, store_type="code"
    )

    manifest = qdrant.upsert.call_args.kwargs["points"][0].payload
    assert manifest["file_hash"] == "content"
    assert manifest["chunk_count"] == 3


def test_rename_rebuilds_missing_manifest_from_already_moved_chunks():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = []
    qdrant.scroll.return_value = (
        [SimpleNamespace(payload={"file_hash": "content", "chunk_count": 3})],
        None,
    )
    qdrant.get_collection.return_value = _collection_info()

    sync = MetadataBasedSync(qdrant)
    sync.copy_file_manifest(
        "code",
        "/old.py",
        "/new.py",
        "q",
        delete_source=True,
        file_size=10,
        store_type="code",
    )

    query_filter = qdrant.scroll.call_args.kwargs["scroll_filter"]
    assert FieldCondition(key="file_path", match=MatchValue(value="/new.py")) in query_filter.should
    manifest = qdrant.upsert.call_args.kwargs["points"][0].payload
    assert manifest["file_path"].endswith("/new.py")
    qdrant.delete.assert_called_once()


def test_rebuild_skips_chunks_missing_required_manifest_metadata():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = []
    qdrant.scroll.return_value = ([SimpleNamespace(payload={"chunk_count": 3})], None)

    MetadataBasedSync(qdrant).copy_file_manifest("code", "/old.py", "/new.py", "q")

    qdrant.upsert.assert_not_called()


def test_get_unindexed_files_propagates_ready_manifest_scan_failure(tmp_path):
    source = tmp_path / "a.py"
    source.write_text("pass\n")
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_ready_metadata_point()]
    qdrant.get_collection.return_value = _collection_info(
        {FILE_MANIFEST_PAYLOAD_KEY: SimpleNamespace()}
    )
    qdrant.scroll.side_effect = RuntimeError("timeout")

    with pytest.raises(RuntimeError, match="Failed to scan file manifests"):
        MetadataBasedSync(qdrant).get_unindexed_files("code", [source])


def test_ready_chunk_counts_are_read_from_manifests():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_ready_metadata_point()]
    qdrant.get_collection.return_value = _collection_info(
        {FILE_MANIFEST_PAYLOAD_KEY: SimpleNamespace()}
    )
    qdrant.scroll.return_value = (
        [SimpleNamespace(payload={"file_path": "/repo/a.py", "chunk_count": 7})],
        None,
    )

    assert MetadataBasedSync(qdrant).get_chunk_counts_by_file("code") == {"/repo/a.py": 7}
    assert qdrant.scroll.call_args.kwargs["with_payload"] == [
        "file_path",
        "quick_hash",
        "file_hash",
        "chunk_count",
        "file_size",
        "store_type",
    ]


def test_ready_manifest_snapshot_is_reused_across_sync_checks():
    qdrant = MagicMock()
    qdrant.retrieve.return_value = [_ready_metadata_point()]
    qdrant.get_collection.return_value = _collection_info(
        {FILE_MANIFEST_PAYLOAD_KEY: SimpleNamespace()}
    )
    qdrant.scroll.return_value = (
        [
            SimpleNamespace(
                payload={
                    "file_path": "/repo/a.py",
                    "quick_hash": "quick",
                    "file_hash": "content",
                    "chunk_count": 7,
                    "file_size": 10,
                    "store_type": "code",
                }
            )
        ],
        None,
    )
    sync = MetadataBasedSync(qdrant)

    assert sync._get_indexed_quick_hashes("code") == {("/repo/a.py", "quick")}
    assert sync._get_indexed_file_paths_set("code") == {"/repo/a.py"}
    assert sync.get_indexed_paths_by_content_hash("code") == {"content": ["/repo/a.py"]}
    assert sync.get_chunk_counts_by_file("code") == {"/repo/a.py": 7}
    assert qdrant.scroll.call_count == 1


def test_chunk_content_hash_query_explicitly_excludes_reserved_points():
    qdrant = MagicMock()
    qdrant.scroll.return_value = ([], None)

    MetadataBasedSync(qdrant).find_file_by_content_hash("code", "hash")

    query_filter = qdrant.scroll.call_args.kwargs["scroll_filter"]
    assert FieldCondition(key="is_metadata", match=MatchValue(value=True)) in query_filter.must_not


def test_user_point_count_excludes_collection_metadata_and_manifests():
    qdrant = MagicMock()
    qdrant.count.return_value = SimpleNamespace(count=12)
    collection_info = SimpleNamespace(
        points_count=113,
        payload_schema={FILE_MANIFEST_PAYLOAD_KEY: SimpleNamespace()},
    )

    assert user_point_count(qdrant, "code", collection_info) == 100

    manifest_condition = FieldCondition(
        key=FILE_MANIFEST_PAYLOAD_KEY,
        match=MatchValue(value=FILE_MANIFEST_PAYLOAD_VALUE),
    )
    assert manifest_condition in qdrant.count.call_args.kwargs["count_filter"].must


def test_user_point_count_without_manifest_index_excludes_only_metadata():
    qdrant = MagicMock()
    collection_info = SimpleNamespace(points_count=113, payload_schema={})

    assert user_point_count(qdrant, "code", collection_info) == 112
    qdrant.count.assert_not_called()
