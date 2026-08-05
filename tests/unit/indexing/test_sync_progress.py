from unittest.mock import Mock

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
