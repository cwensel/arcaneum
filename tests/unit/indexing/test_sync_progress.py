from unittest.mock import Mock, call, patch

from arcaneum.cli.sync import _metadata_scan_progress_callback
from arcaneum.indexing.common.sync import MetadataBasedSync


def test_indexed_quick_hash_scan_reports_each_page():
    first_page = [
        Mock(payload={"file_path": f"/code/{i}.py", "quick_hash": str(i)}) for i in range(1000)
    ]
    second_page = [
        Mock(payload={"file_path": f"/code/{i}.py", "quick_hash": str(i)})
        for i in range(1000, 1250)
    ]
    qdrant = Mock()
    qdrant.scroll.side_effect = [(first_page, "next"), (second_page, None)]
    progress = Mock()

    MetadataBasedSync(qdrant)._get_indexed_quick_hashes("code", progress_callback=progress)

    assert [call.args[0] for call in progress.call_args_list] == [1000, 1250]


def test_indexed_quick_hash_scan_handles_dict_payloads():
    qdrant = Mock()
    qdrant.scroll.return_value = (
        [
            Mock(
                payload={
                    "file_path": "/code/primary.py",
                    "quick_hash": "old",
                    "file_quick_hashes": {
                        "/code/primary.py": "primary",
                        "/code/alias.py": "alias",
                    },
                }
            )
        ],
        None,
    )

    result = MetadataBasedSync(qdrant)._get_indexed_quick_hashes("code")

    assert result == {
        ("/code/primary.py", "primary"),
        ("/code/alias.py", "alias"),
    }


def test_progress_callback_failure_does_not_discard_scan_results():
    qdrant = Mock()
    qdrant.scroll.return_value = (
        [Mock(payload={"file_path": "/code/a.py", "quick_hash": "a"})],
        None,
    )

    result = MetadataBasedSync(qdrant)._get_indexed_quick_hashes(
        "code", progress_callback=Mock(side_effect=RuntimeError("display failed"))
    )

    assert result == {("/code/a.py", "a")}


def test_metadata_scan_progress_reports_user_facing_milestones():
    with patch("arcaneum.cli.sync.print_info") as print_info:
        progress = _metadata_scan_progress_callback(
            verbose=True,
            output_json=False,
            force=False,
            parity=False,
        )
        for count in (1000, 2000, 10000, 20000):
            progress(count)

    assert print_info.call_args_list == [
        call("Checking existing Qdrant metadata for unchanged files..."),
        call("  Scanned 1,000 existing Qdrant chunks..."),
        call("  Scanned 10,000 existing Qdrant chunks..."),
        call("  Scanned 20,000 existing Qdrant chunks..."),
    ]


def test_metadata_scan_progress_is_disabled_for_parity_mode():
    with patch("arcaneum.cli.sync.print_info") as print_info:
        progress = _metadata_scan_progress_callback(
            verbose=True,
            output_json=False,
            force=False,
            parity=True,
        )

    assert progress is None
    print_info.assert_not_called()
