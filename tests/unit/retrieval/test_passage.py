"""get_passage: a bounded window of text around a search hit (RDR-026 Q8).

The fakes below hold only what the passage lookup reads, and fail the test on
any index or settings write: a passage is a read-only lookup.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from meilisearch.errors import MeilisearchApiError

from arcaneum.cli.errors import (
    InvalidArgumentError,
    ServiceDownError,
    UnknownChunkError,
    UnknownCorpusError,
)
from arcaneum.fulltext.backend import MeiliTextBackend
from arcaneum.indexing.collection_metadata import METADATA_POINT_ID

GUIDE = "/root/docs/guide.md"
OTHER = "/root/docs/other.md"
ROOTS = ["/root/docs"]
WORDS = [f"w{n:04d}" for n in range(400)]
STEP, SIZE = 20, 30  # each chunk overlaps the next by 10 words
CHUNKS = 6


def _uuid(n: int) -> str:
    return f"00000000-0000-4000-8000-{n:012d}"


def _words(start: int, end: int) -> str:
    return " ".join(WORDS[start:end])


def _chunk_text(i: int) -> str:
    return _words(i * STEP, i * STEP + SIZE)


def _window_text(first: int, last: int) -> str:
    return _words(first * STEP, last * STEP + SIZE)


def _chunk(i, *, chunk_id=None, file_path=GUIDE, text=None, **extra):
    return {
        "id": chunk_id or _uuid(i),
        "file_path": file_path,
        "chunk_index": i,
        "section": "Guide > Setup",
        "text": _chunk_text(i) if text is None else text,
        **extra,
    }


def _guide():
    return [_chunk(i) for i in range(CHUNKS)] + [
        _chunk(i, chunk_id=_uuid(100 + i), file_path=OTHER) for i in range(3)
    ]


class FakeQdrant:
    """Qdrant with points per collection; refuses every write."""

    def __init__(self, collections, roots=None, down=False):
        self.collections = collections  # name -> list of chunk dicts (id + payload)
        self.roots = roots or {}
        self.down = down
        self.calls = []

    def _check(self, method):
        self.calls.append(method)
        if self.down:
            raise ConnectionError("[Errno 61] Connection refused")

    def collection_exists(self, collection_name):
        self._check("collection_exists")
        return collection_name in self.collections

    @staticmethod
    def _point(chunk):
        return SimpleNamespace(
            id=chunk["id"], payload={k: v for k, v in chunk.items() if k != "id"}
        )

    def retrieve(self, collection_name, ids, with_payload=True, with_vectors=False):
        self._check("retrieve")
        assert with_vectors is False
        wanted = [str(i) for i in ids]
        if METADATA_POINT_ID in wanted:
            roots = self.roots.get(collection_name)
            if roots is None:
                return []
            payload = {"is_metadata": True, "source_roots": roots}
            return [SimpleNamespace(id=METADATA_POINT_ID, payload=payload)]
        return [
            self._point(chunk)
            for chunk in self.collections.get(collection_name, [])
            if chunk["id"] in wanted
        ]

    def scroll(
        self,
        collection_name,
        scroll_filter=None,
        limit=10,
        offset=None,
        with_payload=True,
        with_vectors=False,
    ):
        self._check("scroll")
        assert with_vectors is False
        matched = []
        for chunk in self.collections.get(collection_name, []):
            ok = True
            for condition in scroll_filter.must:
                value = chunk.get(condition.key)
                if condition.match is not None:
                    ok = ok and value == condition.match.value
                if condition.range is not None:
                    bounds = condition.range
                    ok = ok and isinstance(value, int)
                    ok = ok and (bounds.gte is None or value >= bounds.gte)
                    ok = ok and (bounds.lte is None or value <= bounds.lte)
            if ok:
                matched.append(self._point(chunk))
        start = offset or 0
        page = matched[start : start + limit]
        following = start + limit if start + limit < len(matched) else None
        return page, following

    def create_payload_index(self, *_args, **_kwargs):
        raise AssertionError("a passage lookup created a payload index")

    def upsert(self, *_args, **_kwargs):
        raise AssertionError("a passage lookup wrote to Qdrant")


class FakeMeili:
    """MeiliSearch with documents per index; refuses every write."""

    def __init__(self, indexes, healthy=True, max_total_hits=1000):
        self.indexes = indexes  # name -> list of documents
        self.healthy = healthy
        self.max_total_hits = max_total_hits
        self.calls = []

    def health_check(self):
        self.calls.append("health_check")
        return self.healthy

    def index_exists(self, name):
        self.calls.append("index_exists")
        return name in self.indexes

    def get_document(self, index_name, document_id):
        self.calls.append("get_document")
        for document in self.indexes.get(index_name, []):
            if document["id"] == document_id:
                return dict(document)
        return None

    def search(
        self,
        index_name,
        query,
        filter=None,
        limit=20,
        offset=0,
        attributes_to_retrieve=None,
        **_kwargs,
    ):
        self.calls.append("search")
        assert query == ""
        prefix = 'file_path = "'
        assert filter.startswith(prefix) and filter.endswith('"'), filter
        path = json.loads(filter[len("file_path = ") :])
        matched = [d for d in self.indexes.get(index_name, []) if d.get("file_path") == path]
        reachable = matched[: self.max_total_hits]
        hits = []
        for document in reachable[offset : offset + limit]:
            if attributes_to_retrieve:
                document = {k: v for k, v in document.items() if k in attributes_to_retrieve}
            hits.append(dict(document))
        return {"hits": hits, "estimatedTotalHits": len(matched), "processingTimeMs": 1}

    def update_index_settings(self, *_args, **_kwargs):
        raise AssertionError("a passage lookup changed MeiliSearch settings")


def _meili_docs(chunks):
    """MeiliSearch documents for chunks: ``content`` in place of Qdrant's ``text``."""
    documents = []
    for chunk in chunks:
        document = {k: v for k, v in chunk.items() if k != "text"}
        document["content"] = chunk["text"]
        documents.append(document)
    return documents


@pytest.fixture
def stores():
    chunks = _guide()
    qdrant = FakeQdrant({"Docs": chunks}, roots={"Docs": ROOTS})
    meili = FakeMeili({"Docs": _meili_docs(chunks)})
    return SimpleNamespace(qdrant=qdrant, meili=meili)


def _passage(stores, chunk_id, corpus="Docs", **options):
    from arcaneum.retrieval import get_passage

    return get_passage(
        corpus,
        chunk_id,
        qdrant_client=stores.qdrant,
        text_backend=MeiliTextBackend(stores.meili),
        **options,
    )


def _indexes(result):
    return [chunk["chunk_index"] for chunk in result.to_data()["chunks"]]


class TestWindow:
    def test_a_middle_chunk_gets_its_neighbours_in_order_stitched(self, stores):
        result = _passage(stores, _uuid(2))
        data = result.to_data()

        assert set(data) == {
            "schema_version",
            "corpus",
            "chunk_id",
            "cite",
            "first_chunk_id",
            "last_chunk_id",
            "chunks",
            "text",
            "truncated",
        }
        assert data["schema_version"] == 1
        assert data["corpus"] == "Docs"
        assert data["chunk_id"] == _uuid(2)
        assert data["chunks"] == [{"chunk_id": _uuid(i), "chunk_index": i} for i in (1, 2, 3)]
        assert data["first_chunk_id"] == _uuid(1)
        assert data["last_chunk_id"] == _uuid(3)
        # The 10-word overlap between neighbours appears once.
        assert data["text"] == _window_text(1, 3)
        assert data["truncated"] is False
        # The hit chunk's cite, from the shared builder with the recorded root.
        assert data["cite"] == "Docs:guide.md#setup"
        assert result.source == "qdrant"

    def test_the_first_chunk_has_nothing_before_it(self, stores):
        result = _passage(stores, _uuid(0), before=1, after=1)
        assert _indexes(result) == [0, 1]
        assert result.to_data()["first_chunk_id"] == _uuid(0)
        assert result.to_data()["text"] == _window_text(0, 1)

    def test_the_last_chunk_has_nothing_after_it(self, stores):
        result = _passage(stores, _uuid(CHUNKS - 1), before=1, after=1)
        assert _indexes(result) == [CHUNKS - 2, CHUNKS - 1]
        assert result.to_data()["last_chunk_id"] == _uuid(CHUNKS - 1)
        assert result.to_data()["text"] == _window_text(CHUNKS - 2, CHUNKS - 1)

    def test_no_neighbours_is_the_chunk_alone(self, stores):
        data = _passage(stores, _uuid(3), before=0, after=0).to_data()
        assert data["chunks"] == [{"chunk_id": _uuid(3), "chunk_index": 3}]
        assert data["text"] == _chunk_text(3)

    def test_the_widest_window_stays_within_the_file(self, stores):
        result = _passage(stores, _uuid(2), before=5, after=5)
        assert _indexes(result) == list(range(CHUNKS))
        assert all(chunk["chunk_id"] != _uuid(100) for chunk in result.to_data()["chunks"])
        assert result.to_data()["text"] == _window_text(0, CHUNKS - 1)

    def test_max_chars_bounds_the_text_and_marks_it(self, stores):
        data = _passage(stores, _uuid(2), max_chars=50).to_data()
        assert data["text"] == _window_text(1, 3)[:50]
        assert data["truncated"] is True
        assert [chunk["chunk_index"] for chunk in data["chunks"]] == [1, 2, 3]

    def test_zero_max_chars_is_unlimited(self, stores):
        data = _passage(stores, _uuid(2), max_chars=0).to_data()
        assert data["text"] == _window_text(1, 3)
        assert data["truncated"] is False

    def test_the_window_never_creates_an_index_or_writes(self, stores):
        _passage(stores, _uuid(2), before=5, after=5)
        assert set(stores.qdrant.calls) <= {"collection_exists", "retrieve", "scroll"}
        assert set(stores.meili.calls) <= {"health_check", "index_exists", "get_document", "search"}

    def test_writes_nothing_to_stdout(self, stores, capsys):
        _passage(stores, _uuid(2))
        assert capsys.readouterr().out == ""

    def test_a_qdrant_window_past_the_scroll_bound_is_truncated(self):
        # An unversioned seed admits every version's chunks at its path; when
        # they fill the bounded scroll before its neighbours, the window is
        # incomplete and must say so.
        chunks = [
            _chunk(i, chunk_id=_uuid(10_000 + n * 10 + i), git_version_identifier=f"v{n}")
            for n in range(200)
            for i in (1, 2, 3)
        ]
        chunks += [_chunk(i) for i in range(CHUNKS)]
        stores = SimpleNamespace(
            qdrant=FakeQdrant({"Docs": chunks}, roots={"Docs": ROOTS}), meili=FakeMeili({})
        )

        data = _passage(stores, _uuid(2)).to_data()
        assert data["truncated"] is True

    def test_the_default_meilisearch_client_generates_no_key(self, monkeypatch):
        import arcaneum.cli.utils as cli_utils
        from arcaneum.retrieval import get_passage

        chunks = [_chunk(i, chunk_id=f"meili-{i}") for i in range(CHUNKS)]
        meili = FakeMeili({"Docs": _meili_docs(chunks)})
        options = []

        def create_meili_client(*_args, **kwargs):
            options.append(kwargs)
            return meili

        monkeypatch.setattr(cli_utils, "create_meili_client", create_meili_client)
        get_passage("Docs", "meili-2", qdrant_client=FakeQdrant({}))
        assert options == [{"generate_key": False}]

    def test_the_default_qdrant_client_migrates_no_config(self, monkeypatch, stores):
        import arcaneum.cli.utils as cli_utils
        from arcaneum.retrieval import get_passage

        migrations, options = [], []

        def resolve_config_path(migrate=True):
            migrations.append(migrate)
            return "config.yaml"

        def create_qdrant_client(*_args, **kwargs):
            options.append(kwargs)
            return stores.qdrant

        monkeypatch.setattr(cli_utils, "resolve_config_path", resolve_config_path)
        monkeypatch.setattr(cli_utils, "create_qdrant_client", create_qdrant_client)
        get_passage("Docs", _uuid(2), text_backend=MeiliTextBackend(stores.meili))
        assert migrations == [False]
        assert options == [{"config_path": "config.yaml", "for_search": True}]


class TestArguments:
    @pytest.mark.parametrize(
        "options, field",
        [
            ({"before": 6}, "before"),
            ({"before": -1}, "before"),
            ({"after": 6}, "after"),
            ({"after": -1}, "after"),
            ({"max_chars": -1}, "max_chars"),
        ],
    )
    def test_out_of_bounds_is_bad_argument_naming_the_field(self, stores, options, field):
        with pytest.raises(InvalidArgumentError) as exc_info:
            _passage(stores, _uuid(2), **options)
        assert exc_info.value.code == "bad-argument"
        assert field in str(exc_info.value)
        assert stores.qdrant.calls == [] and stores.meili.calls == []

    def test_an_empty_chunk_id_is_bad_argument(self, stores):
        with pytest.raises(InvalidArgumentError) as exc_info:
            _passage(stores, "")
        assert "chunk_id" in str(exc_info.value)


class TestUnknown:
    def test_an_unknown_id_is_unknown_chunk(self, stores):
        with pytest.raises(UnknownChunkError) as exc_info:
            _passage(stores, _uuid(99))
        assert exc_info.value.code == "unknown-chunk"
        assert exc_info.value.exit_code == 3
        assert _uuid(99) in str(exc_info.value)

    @pytest.mark.parametrize("chunk_id", ["../../keys", "not an id", "a/b"])
    def test_an_id_neither_store_can_hold_is_unknown_chunk_without_a_lookup(self, stores, chunk_id):
        with pytest.raises(UnknownChunkError):
            _passage(stores, chunk_id)
        # Neither a Qdrant point id nor a MeiliSearch document id: never sent.
        assert "retrieve" not in stores.qdrant.calls
        assert "get_document" not in stores.meili.calls

    def test_the_collection_metadata_point_is_not_a_chunk(self, stores):
        with pytest.raises(UnknownChunkError):
            _passage(stores, METADATA_POINT_ID)

    def test_a_corpus_in_neither_store_is_unknown_corpus(self, stores):
        with pytest.raises(UnknownCorpusError) as exc_info:
            _passage(stores, _uuid(2), corpus="Nope")
        assert "Nope" in str(exc_info.value)


class TestEitherStore:
    def test_a_dual_indexed_chunk_gives_the_same_window_from_either_store(self, stores):
        from_qdrant = _passage(stores, _uuid(2)).to_data()
        stores.qdrant.down = True
        from_meili = _passage(stores, _uuid(2))

        assert from_meili.source == "meilisearch"
        data = from_meili.to_data()
        assert data["chunks"] == from_qdrant["chunks"]
        assert data["text"] == from_qdrant["text"]
        # Without Qdrant there is no recorded root: the cite is absolute.
        assert data["cite"] == "Docs:/root/docs/guide.md#setup"

    def test_a_text_hit_id_for_a_backfilled_file_takes_its_window_from_meilisearch(self):
        # MeiliSearch-to-Qdrant backfill re-chunks the file under fresh Qdrant
        # ids, so the MeiliSearch ordinals select different content in Qdrant.
        qdrant_chunks = [
            _chunk(i, chunk_id=_uuid(200 + i), text=f"rechunked {i}") for i in range(CHUNKS)
        ]
        meili_chunks = [_chunk(i, chunk_id=f"meili-{i}") for i in range(CHUNKS)]
        stores = SimpleNamespace(
            qdrant=FakeQdrant({"Docs": qdrant_chunks}, roots={"Docs": ROOTS}),
            meili=FakeMeili({"Docs": _meili_docs(meili_chunks)}),
        )

        result = _passage(stores, "meili-3")
        assert result.source == "meilisearch"
        assert result.to_data()["chunks"] == [
            {"chunk_id": f"meili-{i}", "chunk_index": i} for i in (2, 3, 4)
        ]
        assert result.to_data()["text"] == _window_text(2, 4)
        assert "rechunked" not in result.to_data()["text"]
        assert "scroll" not in stores.qdrant.calls
        # Qdrant still supplies the cite's root.
        assert result.to_data()["cite"] == "Docs:guide.md#setup"

    @pytest.mark.parametrize("qdrant_down", [False, True])
    def test_a_window_never_crosses_a_version(self, qdrant_down):
        chunks = []
        for version in ("v1", "v2"):
            for i in range(CHUNKS):
                chunks.append(
                    _chunk(
                        i,
                        chunk_id=_uuid((1 if version == "v1" else 2) * 1000 + i),
                        text=f"{version} " + _chunk_text(i),
                        git_project_identifier="proj#main",
                        git_version_identifier=version,
                    )
                )
        # The same path indexed without a version must not join either window.
        chunks.append(_chunk(2, chunk_id=_uuid(3002), text="unversioned"))
        stores = SimpleNamespace(
            qdrant=FakeQdrant({"Docs": chunks}, roots={"Docs": ROOTS}, down=qdrant_down),
            meili=FakeMeili({"Docs": _meili_docs(chunks)}),
        )

        data = _passage(stores, _uuid(2002), before=5, after=5).to_data()
        assert [c["chunk_id"] for c in data["chunks"]] == [_uuid(2000 + i) for i in range(CHUNKS)]
        assert "v1" not in data["text"] and "unversioned" not in data["text"]

    def test_a_definition_without_a_chunk_index_is_its_own_passage(self):
        definition = {
            "id": "pkg_parse_py_parse_12",
            "file_path": "/root/code/pkg/parse.py",
            "start_line": 12,
            "end_line": 48,
            "content": "def parse(text):\n    return text\n",
        }
        stores = SimpleNamespace(qdrant=FakeQdrant({}), meili=FakeMeili({"Defs": [definition]}))

        data = _passage(stores, "pkg_parse_py_parse_12", corpus="Defs", before=5, after=5).to_data()
        assert data["chunks"] == [{"chunk_id": "pkg_parse_py_parse_12", "chunk_index": None}]
        assert data["first_chunk_id"] == data["last_chunk_id"] == "pkg_parse_py_parse_12"
        assert data["text"] == definition["content"]
        assert data["cite"] == "Defs:/root/code/pkg/parse.py#L12-48"
        assert "search" not in stores.meili.calls

    def test_qdrant_down_and_meilisearch_without_the_chunk_is_service_down(self, stores):
        stores.qdrant.down = True
        with pytest.raises(ServiceDownError) as exc_info:
            _passage(stores, _uuid(99))
        assert exc_info.value.code == "service-down"
        assert "Qdrant" in str(exc_info.value)

    def test_both_stores_down_is_service_down_naming_both(self, stores):
        stores.qdrant.down = True
        stores.meili.healthy = False
        with pytest.raises(ServiceDownError) as exc_info:
            _passage(stores, _uuid(2))
        assert "Qdrant" in str(exc_info.value) and "MeiliSearch" in str(exc_info.value)

    def test_a_meilisearch_window_past_its_returned_chunks_is_truncated(self):
        # Paging stops at maxTotalHits, so a big file lists only its first
        # chunks; a window reaching past them cannot be told from the file's end.
        chunks = [_chunk(i, chunk_id=f"meili-{i}") for i in range(CHUNKS)]
        stores = SimpleNamespace(
            qdrant=FakeQdrant({}),
            meili=FakeMeili({"Docs": _meili_docs(chunks)}, max_total_hits=3),
        )

        data = _passage(stores, "meili-3").to_data()
        assert [c["chunk_index"] for c in data["chunks"]] == [2, 3]
        assert data["truncated"] is True

        whole = _passage(stores, "meili-1").to_data()
        assert [c["chunk_index"] for c in whole["chunks"]] == [0, 1, 2]
        assert whole["truncated"] is False


class TestStitch:
    def _chunks(self, *texts):
        from arcaneum.retrieval.passage import PassageChunk

        return [
            PassageChunk(chunk_id=str(i), chunk_index=i, text=text, payload={})
            for i, text in enumerate(texts)
        ]

    def test_the_longest_overlap_is_trimmed(self):
        from arcaneum.retrieval.passage import stitch

        overlap = "shared tail of the first chunk"
        head = "alpha beta gamma delta epsilon zeta eta theta "
        tail = " iota kappa lambda mu nu xi omicron pi rho sigma"
        chunks = self._chunks(head + overlap, overlap + tail)
        assert stitch(chunks) == head + overlap + tail

    def test_chunks_without_an_overlap_join_with_a_blank_line(self):
        from arcaneum.retrieval.passage import stitch

        assert stitch(self._chunks("first part.", "second part.")) == "first part.\n\nsecond part."

    def test_a_short_coincidental_match_is_not_trimmed(self):
        from arcaneum.retrieval.passage import stitch

        # "the " ends one chunk and starts the next by chance; trimming it would drop text.
        assert stitch(self._chunks("over the ", "the end")) == "over the \n\nthe end"

    def test_an_overlap_may_be_most_of_both_chunks(self):
        from arcaneum.retrieval.passage import stitch

        shared = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"
        first, second = "h " + shared, shared + " t"
        assert stitch(self._chunks(first, second)) == "h " + shared + " t"

    def test_line_based_code_chunks_sharing_a_long_line_stitch_back_to_their_source(self):
        from arcaneum.indexing.ast_chunker import ASTCodeChunker
        from arcaneum.retrieval.passage import stitch

        # The line fallback keeps whole lines as overlap: here the shared middle
        # line is most of each chunk.
        source = "\n".join(["a" * 299 + "1", "b" * 999 + "2", "c" * 299 + "3"])
        chunker = ASTCodeChunker(chunk_size=400, chunk_overlap=20)
        pieces = [c.content for c in chunker.chunk_code("x.py", source, force_line_based=True)]
        assert len(pieces) == 2
        assert stitch(self._chunks(*pieces)) == source

    def test_repeated_line_based_code_chunks_are_kept(self):
        from arcaneum.indexing.ast_chunker import ASTCodeChunker
        from arcaneum.retrieval.passage import stitch

        # Three identical long lines become three chunks that share no
        # overlap: every chunk adds text past its overlap, so a whole-chunk
        # match is repetition, not overlap.
        source = "\n".join(["# " + "abcdefghij" * 100] * 3)
        chunker = ASTCodeChunker(chunk_size=400, chunk_overlap=20)
        pieces = [c.content for c in chunker.chunk_code("x.py", source, force_line_based=True)]
        assert len(pieces) == 3 and len(set(pieces)) == 1
        assert stitch(self._chunks(*pieces)) == "\n\n".join(pieces)

    def test_a_short_last_chunk_from_the_pdf_chunker_stitches_back_to_its_source(self):
        from arcaneum.indexing.pdf.chunker import PDFChunker
        from arcaneum.retrieval.passage import stitch

        # The last chunk (629 characters) overlaps its full-size neighbour by
        # 503, more than half of itself. Corpus sync stores no offsets, so the
        # text match must find it.
        source = " ".join(f"w{n:05d}" for n in range(500))
        pieces = [chunk.text for chunk in PDFChunker({"chunk_size": 1024}).chunk(source, {})]
        assert len(pieces) == 2 and len(pieces[1]) < len(pieces[0]) // 2
        assert stitch(self._chunks(*pieces)) == source

    def test_chunks_that_are_not_consecutive_are_not_trimmed(self):
        from arcaneum.retrieval.passage import PassageChunk, stitch

        overlap = "a long enough shared run of text"
        chunks = [
            PassageChunk(chunk_id="a", chunk_index=1, text="x " + overlap, payload={}),
            PassageChunk(chunk_id="b", chunk_index=3, text=overlap + " y", payload={}),
        ]
        assert stitch(chunks) == "x " + overlap + "\n\n" + overlap + " y"

    def test_character_offsets_splice_the_chunks(self):
        from arcaneum.retrieval.passage import PassageChunk, stitch

        source = "0123456789 the quick brown fox jumps over the lazy dog"
        first, second = source[:30], source[24:]
        chunks = [
            PassageChunk(
                chunk_id="a",
                chunk_index=0,
                text=first,
                payload={"chunk_start_char": 0, "chunk_end_char": 30},
            ),
            PassageChunk(
                chunk_id="b",
                chunk_index=1,
                text=second,
                payload={"chunk_start_char": 24, "chunk_end_char": len(source)},
            ),
        ]
        # A 6-character overlap, below the text-match floor, still splices by offset.
        assert stitch(chunks) == source

    def test_offsets_that_disagree_with_the_text_fall_back(self):
        from arcaneum.retrieval.passage import PassageChunk, stitch

        chunks = [
            PassageChunk(
                chunk_id="a",
                chunk_index=0,
                text="first",
                payload={"chunk_start_char": 0, "chunk_end_char": 5},
            ),
            PassageChunk(
                chunk_id="b",
                chunk_index=1,
                text="second",
                payload={"chunk_start_char": 2, "chunk_end_char": 8},
            ),
        ]
        assert stitch(chunks) == "first\n\nsecond"


class TestMeiliGetDocument:
    def _client(self, index):
        from arcaneum.fulltext.client import FullTextClient

        client = FullTextClient.__new__(FullTextClient)
        client.get_index = lambda _name: index
        return client

    def test_a_document_comes_back_as_a_dict(self):
        index = SimpleNamespace(
            get_document=lambda document_id: SimpleNamespace(id=document_id, content="text")
        )
        assert self._client(index).get_document("Docs", "a1") == {"id": "a1", "content": "text"}

    def test_a_missing_document_is_none(self):
        body = {
            "message": "Document `a1` not found.",
            "code": "document_not_found",
            "type": "invalid_request",
            "link": "https://docs.meilisearch.com/errors#document_not_found",
        }

        def get_document(_document_id):
            raise MeilisearchApiError(
                "missing", SimpleNamespace(status_code=404, text=json.dumps(body))
            )

        index = SimpleNamespace(get_document=get_document)
        assert self._client(index).get_document("Docs", "a1") is None
