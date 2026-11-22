"""One cite-ready hit schema for semantic and text search (RDR-026 Q2, Q4, Q7).

Fixture hits stand in for a PDF, a markdown and a code corpus. Each fixture is
fed through both search modes' ``to_data`` so the tests compare what a consumer
of either mode receives.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arcaneum.fulltext.backend import MeiliTextBackend
from arcaneum.retrieval.hits import (
    build_cite,
    build_locator,
    heading_slug,
    parse_cite,
    read_source_roots,
)
from arcaneum.search.searcher import SearchResult

HOME = "/home/reader"
HIT_KEYS = {"corpus", "chunk_id", "locator", "cite", "score", "snippet", "snippet_truncated"}
DATA_KEYS = {"schema_version", "query", "corpora", "missing_corpora", "limit", "offset", "hits"}


@pytest.fixture(autouse=True)
def home(monkeypatch):
    monkeypatch.setenv("HOME", HOME)


PDF = {
    "file_path": f"{HOME}/Refs/Database-Books/db(31).pdf",
    "filename": "db(31).pdf",
    "page_number": 89,
    "chunk_index": 5,
    "quality_manifest": {"method": "pymupdf_normalized"},
    "content": "B-trees keep keys sorted.",
}
MARKDOWN = {
    "file_path": "/data/notes/guides/setup.md",
    "section": "Setup > Install Qdrant",
    "chunk_index": 2,
    "content": "Run the container.",
}
CODE = {
    "file_path": "/work/arcaneum/src/arcaneum/cli/errors.py",
    "programming_language": "python",
    "chunk_index": 0,
    "content": "class ArcaneumError(Exception):",
}
FIXTURES = {"pdf": ("DevRef", PDF), "markdown": ("Notes", MARKDOWN), "code": ("Arcaneum", CODE)}


def _semantic_result(corpus, payload, roots=None, score=0.8, point_id="p1"):
    from arcaneum.retrieval.semantic import SemanticSearchResult

    hit = SearchResult(
        score=score,
        collection=corpus,
        location=payload["file_path"],
        content=payload["content"],
        metadata=dict(payload),
        point_id=point_id,
    )
    return SemanticSearchResult(
        query="q",
        corpora=[corpus],
        limit=10,
        offset=0,
        results=[hit],
        source_roots={corpus: list(roots or [])},
    )


def _text_result(corpus, payload, roots=None, score=0.97, doc_id="p1"):
    from arcaneum.retrieval.text import TextSearchResult

    hit = {
        **payload,
        "id": doc_id,
        "_corpus": corpus,
        "_rankingScore": score,
        "_formatted": {"content": "<em>" + payload["content"] + "</em>"},
    }
    return TextSearchResult(
        query="q",
        corpora=[corpus],
        limit=10,
        offset=0,
        hits=[hit],
        source_roots={corpus: list(roots or [])},
    )


@pytest.mark.parametrize("kind", sorted(FIXTURES))
class TestBothModesShareOneSchema:
    def test_same_data_and_hit_keys(self, kind):
        corpus, payload = FIXTURES[kind]
        semantic = _semantic_result(corpus, payload).to_data()
        text = _text_result(corpus, payload).to_data()

        assert set(semantic) == set(text) == DATA_KEYS
        assert semantic["schema_version"] == text["schema_version"] == 1
        assert set(semantic["hits"][0]) == set(text["hits"][0]) == HIT_KEYS

    def test_store_internals_stay_out_unless_verbose(self, kind):
        corpus, payload = FIXTURES[kind]
        for result in (_semantic_result(corpus, payload), _text_result(corpus, payload)):
            hit = result.to_data()["hits"][0]
            for leaked in (
                "quality_manifest",
                "_formatted",
                "_corpus",
                "_rankingScore",
                "metadata",
            ):
                assert leaked not in hit

            verbose = result.to_data(verbose=True)["hits"][0]
            assert set(verbose) == HIT_KEYS | {"metadata"}
            assert verbose["metadata"]["file_path"] == payload["file_path"]
            assert "content" not in verbose["metadata"]
            assert not any(key.startswith("_") for key in verbose["metadata"])

    def test_cite_parses_back_to_corpus_and_locator(self, kind):
        corpus, payload = FIXTURES[kind]
        for result in (_semantic_result(corpus, payload), _text_result(corpus, payload)):
            hit = result.to_data()["hits"][0]
            parsed = parse_cite(hit["cite"])
            assert parsed.corpus == hit["corpus"] == corpus
            assert parsed.locator == hit["locator"]
            assert hit["cite"] == f"{corpus}:{hit['locator']}"
            assert not any(ch.isspace() for ch in hit["cite"])

    def test_both_modes_cite_one_chunk_the_same_way(self, kind):
        corpus, payload = FIXTURES[kind]
        semantic = _semantic_result(corpus, payload).to_data()["hits"][0]
        text = _text_result(corpus, payload).to_data()["hits"][0]
        assert semantic["cite"] == text["cite"]
        assert semantic["chunk_id"] == text["chunk_id"] == "p1"

    @pytest.mark.parametrize(
        "budget, expected_len, truncated",
        [(None, 500, True), (10, 10, True), (0, 1500, False), (5000, 1500, False)],
    )
    def test_snippet_bound_sets_the_truncation_flag(self, kind, budget, expected_len, truncated):
        corpus, payload = FIXTURES[kind]
        long_payload = {**payload, "content": "x" * 1500}
        for result in (_semantic_result(corpus, long_payload), _text_result(corpus, long_payload)):
            hit = result.to_data(max_content_chars=budget)["hits"][0]
            assert len(hit["snippet"]) == expected_len
            assert hit["snippet_truncated"] is truncated


class TestCiteGrammar:
    def test_pdf_cites_the_page_relative_to_the_longest_root(self):
        roots = [f"{HOME}/Refs", f"{HOME}/Refs/Database-Books"]
        assert build_cite("DevRef", PDF, roots) == "DevRef:db(31).pdf#page=89"

    def test_pdf_without_a_root_folds_home(self):
        assert build_cite("DevRef", PDF) == "DevRef:~/Refs/Database-Books/db(31).pdf#page=89"

    def test_root_matches_only_on_a_path_boundary(self):
        assert build_cite("DevRef", PDF, [f"{HOME}/Re"]) == (
            "DevRef:~/Refs/Database-Books/db(31).pdf#page=89"
        )

    def test_path_outside_home_stays_absolute(self):
        assert build_cite("Notes", MARKDOWN) == "Notes:/data/notes/guides/setup.md#install-qdrant"

    def test_markdown_cites_the_deepest_heading_slug(self):
        assert build_cite("Notes", MARKDOWN, ["/data/notes/"]) == (
            "Notes:guides/setup.md#install-qdrant"
        )

    def test_markdown_indexer_header_path_cites_the_heading(self):
        # ``arc index markdown`` stores the heading trail as ``header_path``.
        payload = {k: v for k, v in MARKDOWN.items() if k != "section"}
        payload["header_path"] = "Setup > Install Qdrant"
        assert build_cite("Notes", payload, ["/data/notes"]) == (
            "Notes:guides/setup.md#install-qdrant"
        )

    def test_filesystem_root_is_a_valid_source_root(self):
        assert build_cite("Notes", MARKDOWN, ["/"]) == (
            "Notes:data/notes/guides/setup.md#install-qdrant"
        )

    def test_chunk_before_any_heading_has_no_fragment(self):
        payload = {**MARKDOWN, "section": None}
        assert build_cite("Notes", payload, ["/data/notes"]) == "Notes:guides/setup.md"

    def test_semantic_code_hit_cites_the_path_alone(self):
        assert build_cite("Arcaneum", CODE, ["/work/arcaneum"]) == (
            "Arcaneum:src/arcaneum/cli/errors.py"
        )

    def test_code_hit_with_a_line_range_cites_it(self):
        payload = {**CODE, "start_line": 12, "end_line": 48}
        assert build_cite("Libs", payload, ["/work/arcaneum"]) == (
            "Libs:src/arcaneum/cli/errors.py#L12-48"
        )

    def test_start_line_alone_is_not_a_range(self):
        payload = {**CODE, "start_line": 12}
        assert build_cite("Libs", payload, ["/work/arcaneum"]) == (
            "Libs:src/arcaneum/cli/errors.py"
        )

    def test_hash_percent_and_space_are_escaped_and_nothing_else(self):
        payload = {"file_path": "/data/my notes/a#b%c (1).md", "content": ""}
        cite = build_cite("Notes", payload, ["/data"])
        assert cite == "Notes:my%20notes/a%23b%25c%20(1).md"
        parsed = parse_cite(cite)
        assert parsed.path == "my notes/a#b%c (1).md"
        assert parsed.fragment is None

    def test_other_whitespace_in_a_path_still_round_trips(self):
        # Only ``#``, ``%`` and space are escaped (RDR-026 Q4); a cite holding
        # another whitespace character must still parse back to its path.
        payload = {"file_path": "/data/Design\u00a0Notes.pdf", "page_number": 3, "content": ""}
        parsed = parse_cite(build_cite("Docs", payload, ["/data"]))
        assert parsed.path == "Design\u00a0Notes.pdf"
        assert parsed.fragment == "page=3"

    def test_parse_splits_corpus_path_and_fragment(self):
        parsed = parse_cite("DevRef:~/Refs/Database-Books/db(31).pdf#page=89")
        assert parsed.corpus == "DevRef"
        assert parsed.locator == "~/Refs/Database-Books/db(31).pdf#page=89"
        assert parsed.path == "~/Refs/Database-Books/db(31).pdf"
        assert parsed.fragment == "page=89"

    @pytest.mark.parametrize(
        "cite",
        ["no-colon", ":path.md", "bad corpus:path.md", "C:a.md#x#y", "C:a.md#", "C:", "C:a.md#Bad"],
    )
    def test_parse_rejects_what_the_grammar_does_not_produce(self, cite):
        with pytest.raises(ValueError):
            parse_cite(cite)

    def test_locator_is_the_cite_without_its_corpus(self):
        assert build_locator(PDF, [f"{HOME}/Refs"]) == "Database-Books/db(31).pdf#page=89"

    @pytest.mark.parametrize(
        "section, slug",
        [
            ("Setup > Install Qdrant", "install-qdrant"),
            (["Setup", "Install Qdrant!"], "install-qdrant"),
            ("## Background", "background"),
            ("Step 2: Run it", "step-2-run-it"),
            ("!!!", None),
            (None, None),
            ([], None),
        ],
    )
    def test_heading_slug(self, section, slug):
        assert heading_slug(section) == slug


class TestSourceRoots:
    def _client(self, payload):
        client = MagicMock()
        client.retrieve.return_value = [SimpleNamespace(payload=payload)]
        return client

    def test_reads_roots_from_collection_metadata(self):
        client = self._client({"source_roots": ["/data/notes", "/work/arcaneum"]})
        assert read_source_roots(client, "Notes") == ["/data/notes", "/work/arcaneum"]

    @pytest.mark.parametrize("value", [None, "not-a-list", [1, None, "/ok"]])
    def test_malformed_roots_keep_only_strings(self, value):
        client = self._client({"source_roots": value})
        assert read_source_roots(client, "Notes") == (["/ok"] if isinstance(value, list) else [])

    def test_unreachable_qdrant_gives_no_roots(self):
        client = MagicMock()
        client.retrieve.side_effect = ConnectionError("refused")
        assert read_source_roots(client, "Notes") == []

    def test_no_client_gives_no_roots(self):
        assert read_source_roots(None, "Notes") == []


class TestSearchCoreRoots:
    def test_semantic_search_reads_each_corpus_roots(self, monkeypatch):
        from arcaneum.retrieval import semantic as module

        client = MagicMock()
        client.collection_exists.return_value = True
        client.retrieve.return_value = [SimpleNamespace(payload={"source_roots": [f"{HOME}/Refs"]})]
        hit = SearchResult(0.9, "DevRef", "", PDF["content"], dict(PDF), "p1")
        monkeypatch.setattr(module, "search_collection", lambda **kwargs: [hit])
        monkeypatch.setattr(module, "acquire_embedder_slot", nullcontext)

        result = module.search_semantic("q", ["DevRef"], client=client, embedder=MagicMock())

        assert result.to_data()["hits"][0]["cite"] == "DevRef:Database-Books/db(31).pdf#page=89"

    def _meili(self, hits_by_index):
        client = MagicMock()
        client.health_check.return_value = True
        client.index_exists.return_value = True
        client.search.side_effect = lambda index, query, **kwargs: {
            "hits": [dict(hit) for hit in hits_by_index[index]],
            "estimatedTotalHits": len(hits_by_index[index]),
            "processingTimeMs": 1,
        }
        return client

    def test_text_search_requests_and_reports_the_ranking_score(self):
        from arcaneum.retrieval import search_text

        meili = self._meili({"DevRef": [{**PDF, "id": "p1", "_rankingScore": 0.9993}]})
        result = search_text("q", ["DevRef"], text_backend=MeiliTextBackend(meili))

        assert meili.search.call_args.kwargs["show_ranking_score"] is True
        assert result.to_data()["hits"][0]["score"] == 0.9993

    def test_text_search_reads_roots_from_qdrant(self):
        from arcaneum.retrieval import search_text

        qdrant = MagicMock()
        qdrant.retrieve.return_value = [SimpleNamespace(payload={"source_roots": [f"{HOME}/Refs"]})]
        meili = self._meili({"DevRef": [{**PDF, "id": "p1"}]})
        result = search_text(
            "q", ["DevRef"], text_backend=MeiliTextBackend(meili), qdrant_client=qdrant
        )

        assert result.to_data()["hits"][0]["cite"] == "DevRef:Database-Books/db(31).pdf#page=89"

    def test_text_search_with_qdrant_down_cites_the_absolute_path(self):
        from arcaneum.retrieval import search_text

        qdrant = MagicMock()
        qdrant.retrieve.side_effect = ConnectionError("refused")
        meili = self._meili({"DevRef": [{**PDF, "id": "p1"}]})
        result = search_text(
            "q", ["DevRef"], text_backend=MeiliTextBackend(meili), qdrant_client=qdrant
        )

        data = result.to_data()
        assert data["hits"][0]["cite"] == "DevRef:~/Refs/Database-Books/db(31).pdf#page=89"
        # The hits may be a migrated corpus's stale index, so Qdrant is named (arcaneum#178).
        assert data["service_errors"] == ["Qdrant: refused"]

    def test_a_failed_root_read_stops_further_root_reads(self):
        from arcaneum.retrieval import search_text

        qdrant = MagicMock()
        qdrant.retrieve.side_effect = TimeoutError("timed out")
        meili = self._meili({"A": [{**CODE, "id": "a1"}], "B": [{**PDF, "id": "b1"}]})
        result = search_text(
            "q", ["A", "B"], text_backend=MeiliTextBackend(meili), qdrant_client=qdrant
        )

        assert qdrant.retrieve.call_count == 1
        assert [hit["cite"].startswith(("A:/work/", "B:~/")) for hit in result.to_data()["hits"]] == [
            True,
            True,
        ]

    def test_multi_corpus_text_stays_round_robin_not_score_ordered(self):
        from arcaneum.retrieval import search_text

        meili = self._meili(
            {
                "A": [
                    {**CODE, "id": "a1", "_rankingScore": 0.2},
                    {**CODE, "id": "a2", "_rankingScore": 0.1},
                ],
                "B": [{**CODE, "id": "b1", "_rankingScore": 0.99}],
            }
        )
        hits = search_text("q", ["A", "B"], text_backend=MeiliTextBackend(meili)).to_data()["hits"]

        assert [(h["corpus"], h["chunk_id"]) for h in hits] == [
            ("A", "a1"),
            ("B", "b1"),
            ("A", "a2"),
        ]
        assert [h["score"] for h in hits] == [0.2, 0.99, 0.1]


def test_fulltext_client_passes_show_ranking_score():
    from arcaneum.fulltext.client import FullTextClient

    client = FullTextClient.__new__(FullTextClient)
    index = MagicMock()
    client.get_index = lambda name: index

    client.search("Idx", "q", show_ranking_score=True)
    assert index.search.call_args.args[1]["showRankingScore"] is True

    client.search("Idx", "q")
    assert "showRankingScore" not in index.search.call_args.args[1]


class TestSyncRecordsSourceRoots:
    def test_merges_new_roots_after_the_recorded_ones(self, monkeypatch):
        from arcaneum.cli import sync

        updates = {}
        monkeypatch.setattr(
            sync, "get_collection_metadata", lambda client, corpus: {"source_roots": ["/a", "/b"]}
        )
        monkeypatch.setattr(
            sync,
            "update_collection_metadata",
            lambda client, corpus, **kwargs: updates.update(kwargs),
        )

        sync._stamp_last_sync_metadata(MagicMock(), "Notes", source_roots=["/b", "/c", "/a"])

        assert updates["source_roots"] == ["/a", "/b", "/c"]
        assert "last_sync" in updates

    def test_no_new_roots_leaves_source_roots_alone(self, monkeypatch):
        from arcaneum.cli import sync

        updates = {}
        monkeypatch.setattr(sync, "get_collection_metadata", lambda client, corpus: {})
        monkeypatch.setattr(
            sync,
            "update_collection_metadata",
            lambda client, corpus, **kwargs: updates.update(kwargs),
        )

        sync._stamp_last_sync_metadata(MagicMock(), "Notes")

        assert "source_roots" not in updates
        assert "last_sync" in updates

    def test_roots_are_each_directory_and_each_code_git_toplevel(self, tmp_path):
        from pathlib import Path

        from arcaneum.cli import sync

        roots = sync._sync_source_roots(
            [Path("/data/notes"), Path("/work")],
            "code",
            ["/work/arcaneum", "/work/libs"],
            ["/work/arcaneum"],
        )
        assert roots == ["/data/notes", "/work", "/work/arcaneum", "/work/libs"]

        assert sync._sync_source_roots([Path("/data/notes")], "pdf", ["/x"], ["/y"]) == [
            "/data/notes"
        ]


def test_directory_sync_records_its_directory_as_a_source_root(tmp_path):
    from contextlib import ExitStack
    from unittest.mock import patch

    from arcaneum.cli import sync

    meili = MagicMock()
    meili.health_check.return_value = True
    meili.index_exists.return_value = True
    stamps = []
    targets = {
        "create_qdrant_client": MagicMock(return_value=MagicMock()),
        "get_meili_client": MagicMock(return_value=meili),
        "get_collection_type": MagicMock(return_value="markdown"),
        "get_collection_metadata": MagicMock(return_value={"model": "arctic-m"}),
        "_maybe_backfill_legacy_prompt_policy": MagicMock(
            side_effect=lambda q, c, t, m, metadata, o: (metadata, [])
        ),
        "MetadataBasedSync": MagicMock(),
        "_ensure_file_manifests": MagicMock(),
        "interaction_logger": MagicMock(),
        "_stamp_last_sync_metadata": lambda qdrant, corpus, roots=(): stamps.append(
            (corpus, roots)
        ),
    }
    with ExitStack() as stack:
        for name, value in targets.items():
            stack.enter_context(patch.object(sync, name, value))
        try:
            sync._sync_directory_locked(
                "Notes",
                (str(tmp_path),),
                None,
                "arctic-m",
                None,
                False,
                False,
                None,
                None,
                True,
                None,
                False,
                True,  # output_json
            )
        except SystemExit:
            pass

    assert stamps == [("Notes", [str(tmp_path.resolve())])]


def test_file_only_code_sync_records_the_file_git_root(tmp_path):
    from contextlib import ExitStack
    from unittest.mock import patch

    from arcaneum.cli import sync

    source = tmp_path / "a.py"
    source.write_text("x = 1\n")
    meili = MagicMock()
    meili.health_check.return_value = True
    meili.index_exists.return_value = True
    stamps = []
    targets = {
        "create_qdrant_client": MagicMock(return_value=MagicMock()),
        "get_meili_client": MagicMock(return_value=meili),
        "get_collection_type": MagicMock(return_value="code"),
        "get_collection_metadata": MagicMock(return_value={"model": "jina-code"}),
        "_maybe_backfill_legacy_prompt_policy": MagicMock(
            side_effect=lambda q, c, t, m, metadata, o: (metadata, [])
        ),
        "MetadataBasedSync": MagicMock(),
        "_ensure_file_manifests": MagicMock(),
        "interaction_logger": MagicMock(),
        "_find_git_root": MagicMock(return_value="/work/repo"),
        # Nothing left to index, so the sync stamps and returns early.
        "order_files": MagicMock(return_value=[]),
        "_stamp_last_sync_metadata": lambda qdrant, corpus, roots=(): stamps.append(
            (corpus, roots)
        ),
    }
    with ExitStack() as stack:
        for name, value in targets.items():
            stack.enter_context(patch.object(sync, name, value))
        try:
            sync._sync_directory_locked(
                "Code",
                (str(source),),
                None,
                "jina-code",
                None,
                False,
                False,
                None,
                None,
                True,
                None,
                False,
                True,  # output_json
            )
        except SystemExit:
            pass

    assert stamps == [("Code", ["/work/repo"])]
