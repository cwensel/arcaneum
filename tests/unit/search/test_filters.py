"""The one filter dialect shared by semantic and text search (RDR-026 Q6).

Each neutral form parses once and translates to both a Qdrant ``Filter`` and a
MeiliSearch expression. Native forms pass through to their own mode only.
"""

import pytest
from qdrant_client.http import models

from arcaneum.search.filters import (
    FilterError,
    NativeFilter,
    NeutralFilter,
    check_filterable,
    describe_filter,
    parse_filter,
    qdrant_text_filterable_fields,
    semantic_filterable_fields,
    to_meilisearch_filter,
    to_qdrant_filter,
    to_qdrant_text_filter,
)


def _match(key, value):
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def _digits(key, text):
    """A digit-only value matches the keyword or the integer payload."""
    return models.Filter(should=[_match(key, text), _match(key, int(text))])


# (filter, expected Qdrant must-conditions, expected MeiliSearch expression)
NEUTRAL_TABLE = [
    ("language=python", [_match("language", "python")], 'language = "python"'),
    ("language = python", [_match("language", "python")], 'language = "python"'),
    (
        "language=python,project=backend",
        [_match("language", "python"), _match("project", "backend")],
        'language = "python" AND project = "backend"',
    ),
    ("filename=db(31).pdf", [_match("filename", "db(31).pdf")], 'filename = "db(31).pdf"'),
    (
        "section=Getting started",
        [_match("section", "Getting started")],
        'section = "Getting started"',
    ),
    ("page_number=89", [_digits("page_number", "89")], 'page_number = "89"'),
    ("language:match:python", [_match("language", "python")], 'language = "python"'),
    (
        "language:in:python|java",
        [models.FieldCondition(key="language", match=models.MatchAny(any=["python", "java"]))],
        'language IN ["python", "java"]',
    ),
    (
        "page_number:in:89|90",
        [
            models.Filter(
                should=[
                    models.FieldCondition(
                        key="page_number", match=models.MatchAny(any=["89", "90"])
                    ),
                    models.FieldCondition(key="page_number", match=models.MatchAny(any=[89, 90])),
                ]
            )
        ],
        'page_number IN ["89", "90"]',
    ),
    (
        "page_number:gte:80",
        [models.FieldCondition(key="page_number", range=models.Range(gte=80))],
        'page_number >= "80"',
    ),
    (
        "page_number:gt:80",
        [models.FieldCondition(key="page_number", range=models.Range(gt=80))],
        'page_number > "80"',
    ),
    (
        "page_number:lte:1.5",
        [models.FieldCondition(key="page_number", range=models.Range(lte=1.5))],
        'page_number <= "1.5"',
    ),
    (
        "page_number:lt:-2",
        [models.FieldCondition(key="page_number", range=models.Range(lt=-2))],
        'page_number < "-2"',
    ),
    (
        "file_path:match:C:/src/x.py",
        [_match("file_path", "C:/src/x.py")],
        'file_path = "C:/src/x.py"',
    ),
    (
        "filename=db(31).pdf,page_number:gte:80",
        [
            _match("filename", "db(31).pdf"),
            models.FieldCondition(key="page_number", range=models.Range(gte=80)),
        ],
        'filename = "db(31).pdf" AND page_number >= "80"',
    ),
    ('{"filename": "db(31).pdf"}', [_match("filename", "db(31).pdf")], 'filename = "db(31).pdf"'),
    (
        '{"filename": "say \\"hi\\" (2).pdf"}',
        [_match("filename", 'say "hi" (2).pdf')],
        'filename = "say \\"hi\\" (2).pdf"',
    ),
    (
        '{"file_path": "C:\\\\docs\\\\a b.md"}',
        [_match("file_path", "C:\\docs\\a b.md")],
        'file_path = "C:\\\\docs\\\\a b.md"',
    ),
    (
        '{"language": ["python", "java"]}',
        [models.FieldCondition(key="language", match=models.MatchAny(any=["python", "java"]))],
        'language IN ["python", "java"]',
    ),
    ('{"page_number": 89}', [_digits("page_number", "89")], 'page_number = "89"'),
    ('{"is_image_pdf": true}', [_match("is_image_pdf", True)], 'is_image_pdf = "true"'),
    (
        '{"filename": "a.pdf", "page_number": 3}',
        [_match("filename", "a.pdf"), _digits("page_number", "3")],
        'filename = "a.pdf" AND page_number = "3"',
    ),
]


def _range(key, **bounds):
    return models.FieldCondition(key=key, range=models.Range(**bounds))


# The grown forms (RDR-027 Q6): quoted values, AND joins and symbolic
# comparisons, covering every form in remote-arc/search-usage-filters.md and
# the interaction logs.
GROWN_TABLE = [
    ('filename="a, b.pdf"', [_match("filename", "a, b.pdf")], 'filename = "a, b.pdf"'),
    ("filename='x.pdf'", [_match("filename", "x.pdf")], 'filename = "x.pdf"'),
    ("filename='a, b.pdf'", [_match("filename", "a, b.pdf")], 'filename = "a, b.pdf"'),
    ('filename = "db(31).pdf"', [_match("filename", "db(31).pdf")], 'filename = "db(31).pdf"'),
    (
        'filename = "db(31).pdf" AND page_number > 80',
        [_match("filename", "db(31).pdf"), _range("page_number", gt=80)],
        'filename = "db(31).pdf" AND page_number > "80"',
    ),
    (
        'page_number > 80 AND filename = "db(31).pdf"',
        [_range("page_number", gt=80), _match("filename", "db(31).pdf")],
        'page_number > "80" AND filename = "db(31).pdf"',
    ),
    (
        "filename='V' AND page_number>=39 AND page_number<=43",
        [_match("filename", "V"), _range("page_number", gte=39), _range("page_number", lte=43)],
        'filename = "V" AND page_number >= "39" AND page_number <= "43"',
    ),
    (
        "filename = x AND chunk_index = 3",
        [_match("filename", "x"), _digits("chunk_index", "3")],
        'filename = "x" AND chunk_index = "3"',
    ),
    (
        "language = python AND project = myapp",
        [_match("language", "python"), _match("project", "myapp")],
        'language = "python" AND project = "myapp"',
    ),
    ("file_hash = 'abc'", [_match("file_hash", "abc")], 'file_hash = "abc"'),
    (
        'filename="INTRO TO X.pdf"',
        [_match("filename", "INTRO TO X.pdf")],
        'filename = "INTRO TO X.pdf"',
    ),
    (
        "filename='Ideas IN Motion 10 TO 20.pdf' AND page_number >= 2",
        [_match("filename", "Ideas IN Motion 10 TO 20.pdf"), _range("page_number", gte=2)],
        'filename = "Ideas IN Motion 10 TO 20.pdf" AND page_number >= "2"',
    ),
    ('title="say \\"hi\\""', [_match("title", 'say "hi"')], 'title = "say \\"hi\\""'),
    ("title='it\\'s'", [_match("title", "it's")], 'title = "it\'s"'),
    ('file_path="C:\\\\docs"', [_match("file_path", "C:\\docs")], 'file_path = "C:\\\\docs"'),
    ("author=O'Brien", [_match("author", "O'Brien")], 'author = "O\'Brien"'),
    ("section=Rock 'n' roll", [_match("section", "Rock 'n' roll")], 'section = "Rock \'n\' roll"'),
    (
        "filename=The '80s.md,section=O'Brien",
        [_match("filename", "The '80s.md"), _match("section", "O'Brien")],
        'filename = "The \'80s.md" AND section = "O\'Brien"',
    ),
    ("filename=draft (1).md", [_match("filename", "draft (1).md")], 'filename = "draft (1).md"'),
    ("page_number = 3", [_digits("page_number", "3")], 'page_number = "3"'),
    ('page_number = "3"', [_digits("page_number", "3")], 'page_number = "3"'),
    ("page_number > 3", [_range("page_number", gt=3)], 'page_number > "3"'),
    ("page_number >= 3", [_range("page_number", gte=3)], 'page_number >= "3"'),
    ("page_number < 3", [_range("page_number", lt=3)], 'page_number < "3"'),
    ("page_number <= 3", [_range("page_number", lte=3)], 'page_number <= "3"'),
    ("page_number<1.5", [_range("page_number", lt=1.5)], 'page_number < "1.5"'),
    ('page_number > "80"', [_range("page_number", gt=80)], 'page_number > "80"'),
    ("page_number > 10", [_range("page_number", gt=10)], 'page_number > "10"'),
    ("page_number < 10", [_range("page_number", lt=10)], 'page_number < "10"'),
    ("page_number >= 10", [_range("page_number", gte=10)], 'page_number >= "10"'),
    (
        "filename=a.pdf, page_number >= 2 AND page_number < 5",
        [_match("filename", "a.pdf"), _range("page_number", gte=2), _range("page_number", lt=5)],
        'filename = "a.pdf" AND page_number >= "2" AND page_number < "5"',
    ),
    (
        'file_path:match:"C:/a, b.py"',
        [_match("file_path", "C:/a, b.py")],
        'file_path = "C:/a, b.py"',
    ),
    (
        "language:in:'a, b'|c",
        [models.FieldCondition(key="language", match=models.MatchAny(any=["a, b", "c"]))],
        'language IN ["a, b", "c"]',
    ),
    ('page_number:gte:"80"', [_range("page_number", gte=80)], 'page_number >= "80"'),
]


class TestNeutralForms:
    @pytest.mark.parametrize("text, must, meili", NEUTRAL_TABLE, ids=[t[0] for t in NEUTRAL_TABLE])
    def test_translates_to_both_backends(self, text, must, meili):
        parsed = parse_filter(text)

        assert isinstance(parsed, NeutralFilter)
        assert to_qdrant_filter(parsed) == models.Filter(must=must)
        assert to_meilisearch_filter(parsed) == meili

    @pytest.mark.parametrize("text, must, meili", GROWN_TABLE, ids=[t[0] for t in GROWN_TABLE])
    def test_grown_forms_translate_to_both_backends(self, text, must, meili):
        parsed = parse_filter(text)

        assert isinstance(parsed, NeutralFilter)
        assert to_qdrant_filter(parsed) == models.Filter(must=must)
        assert to_meilisearch_filter(parsed) == meili

    def test_contains_translates_for_qdrant(self):
        parsed = parse_filter("file_path:contains:/src/")

        assert to_qdrant_filter(parsed) == models.Filter(
            must=[models.FieldCondition(key="file_path", match=models.MatchText(text="/src/"))]
        )

    def test_contains_is_refused_for_meilisearch(self):
        with pytest.raises(FilterError, match="contains"):
            to_meilisearch_filter(parse_filter("file_path:contains:/src/"))

    def test_fields_lists_each_key_once_in_order(self):
        parsed = parse_filter("b=1,a=2,b:gte:0")

        assert parsed.fields == ["b", "a"]

    def test_empty_filter_is_none(self):
        assert parse_filter("") is None
        assert parse_filter("   ") is None
        assert to_qdrant_filter(None) is None
        assert to_meilisearch_filter(None) is None

    def test_description_names_each_condition(self):
        parsed = parse_filter("filename=db(31).pdf,page_number:gte:80,language:in:python|java")

        assert describe_filter(parsed) == (
            "filename=db(31).pdf AND page_number>=80 AND language in [python, java]"
        )


class TestNativeForms:
    def test_qdrant_json_passes_through_to_semantic(self):
        text = '{"must": [{"key": "chunk_index", "match": {"value": 3}}]}'
        parsed = parse_filter(text)

        assert isinstance(parsed, NativeFilter)
        assert parsed.backend == "qdrant"
        assert to_qdrant_filter(parsed) == models.Filter(must=[_match("chunk_index", 3)])

    def test_qdrant_json_is_refused_by_text(self):
        parsed = parse_filter('{"must_not": [{"key": "a", "match": {"value": "b"}}]}')

        with pytest.raises(FilterError, match="Qdrant"):
            to_meilisearch_filter(parsed)

    def test_malformed_qdrant_json_is_refused(self):
        with pytest.raises(FilterError):
            to_qdrant_filter(parse_filter('{"must": [{"nope": 1}]}'))

    NATIVE_MEILISEARCH = [
        "language = python OR language = java",
        'filename = "a.pdf" OR filename = "b.pdf"',
        "NOT language = python",
        "page_number 10 TO 20",
        'language IN ["python", "java"]',
        "tags EXISTS",
        "tags IS NULL",
        "tags IS EMPTY",
        "language != python",
        'filename != "a.pdf" AND page_number > 3',
        "(page_number > 3 AND page_number < 5)",
        "page_number > 3 AND (page_number < 5)",
    ]

    @pytest.mark.parametrize("text", NATIVE_MEILISEARCH)
    def test_meilisearch_expression_passes_through_to_text(self, text):
        parsed = parse_filter(text)

        assert isinstance(parsed, NativeFilter)
        assert parsed.backend == "meilisearch"
        assert to_meilisearch_filter(parsed) == text

    @pytest.mark.parametrize("text", NATIVE_MEILISEARCH)
    def test_meilisearch_expression_is_refused_by_semantic(self, text):
        with pytest.raises(FilterError, match="MeiliSearch"):
            to_qdrant_filter(parse_filter(text))

    def test_refusal_gives_qdrant_json_and_the_grown_forms(self):
        with pytest.raises(FilterError) as exc_info:
            to_qdrant_filter(parse_filter("language = python OR language = java"))

        message = str(exc_info.value)
        for needle in ("must", "should", "must_not", "AND", ">="):
            assert needle in message

    def test_lowercase_words_stay_neutral(self):
        parsed = parse_filter("section=Getting to know it and more")

        assert isinstance(parsed, NeutralFilter)


class TestUnparseable:
    @pytest.mark.parametrize(
        "text, needle",
        [
            ("language:like:python", "operator"),
            ("page_number:gte:ten", "number"),
            ("page_number:gte:nan", "number"),
            ("=python", "key"),
            ("bad key=python", "key"),
            ("language=", "value"),
            ("language:in:", "value"),
            ("language:in:python||java", "value"),
            ("language:in:python,java", "|"),
            ("language=python,", "term"),
            ("justaword", "term"),
            ("a:b", "term"),
            ("{not json", "JSON"),
            ("[1, 2]", "term"),
            ('{"filename": {"nested": 1}}', "filename"),
            ('{"filename": null}', "filename"),
            ('{"filename": []}', "filename"),
            ('{"filename": [["a"]]}', "filename"),
            ('{"page_number": 1.5}', "page_number"),
            ('{"": "x"}', "key"),
            ("{}", "empty"),
            ('filename="x', "quote"),
            ("filename='x.pdf", "quote"),
            ('filename="a" b', "quote"),
            ("page_number > ten", "number"),
            ('path>"a:b:c"', "number"),
            ("> 3", "key"),
            ("filename=", "value"),
            ('filename=""', "value"),
            ("a AND", "term"),
            ("AND a=1", "term"),
        ],
    )
    def test_is_refused_not_dropped(self, text, needle):
        with pytest.raises(FilterError) as exc_info:
            parsed = parse_filter(text)
            to_qdrant_filter(parsed)

        assert needle in str(exc_info.value)

    def test_filter_error_is_a_value_error(self):
        assert issubclass(FilterError, ValueError)


class TestFilterableFields:
    def test_known_fields_pass(self):
        check_filterable(parse_filter("filename=a.pdf"), ["filename", "file_path"], "DevRef")

    def test_unknown_field_names_the_field_and_the_filterable_ones(self):
        with pytest.raises(FilterError) as exc_info:
            check_filterable(parse_filter("chunk_index=3"), ["filename", "file_path"], "DevRef")

        message = str(exc_info.value)
        assert "'chunk_index' is not filterable" in message
        assert "DevRef" in message
        assert "filename" in message
        assert "file_path" in message

    def test_native_forms_are_not_checked(self):
        check_filterable(parse_filter("chunk_index != 3"), ["filename"], "DevRef")

    def test_grown_forms_are_checked(self):
        with pytest.raises(FilterError, match="'chunk_index' is not filterable"):
            check_filterable(parse_filter("chunk_index > 3"), ["filename"], "DevRef")

    def test_pdf_semantic_fields_are_the_index_filterable_attributes(self):
        fields = semantic_filterable_fields("pdf")

        assert "filename" in fields
        assert "page_number" in fields

    def test_code_semantic_fields_use_qdrant_payload_names(self):
        # to_qdrant_point stores language/project/branch under other names.
        fields = semantic_filterable_fields("code")

        assert "programming_language" in fields
        assert "git_project_name" in fields
        assert "git_branch" in fields
        assert "language" not in fields
        assert "file_path" in fields

    @pytest.mark.parametrize("collection_type", [None, "", "mystery", 3])
    def test_untyped_collection_has_no_field_list(self, collection_type):
        assert semantic_filterable_fields(collection_type) is None


# MeiliSearch names a Qdrant text filter accepts, as to_qdrant_point stores them.
_PAYLOAD_NAMES = {"language": "programming_language", "project": "git_project_name"}


def _payload_named(condition):
    """``condition`` with each MeiliSearch field name as its Qdrant payload name."""
    if isinstance(condition, models.Filter):
        return condition.model_copy(update={"should": [_payload_named(c) for c in condition.should]})
    return condition.model_copy(update={"key": _PAYLOAD_NAMES.get(condition.key, condition.key)})


class TestQdrantTextFilter:
    """A filter on a Qdrant text corpus (RDR-027 Q6)."""

    @pytest.mark.parametrize(
        "text, must, _meili",
        NEUTRAL_TABLE + GROWN_TABLE,
        ids=[t[0] for t in NEUTRAL_TABLE + GROWN_TABLE],
    )
    def test_neutral_and_grown_forms_translate_as_semantic_search_does(self, text, must, _meili):
        assert to_qdrant_text_filter(parse_filter(text)) == models.Filter(
            must=[_payload_named(c) for c in must]
        )

    def test_meilisearch_names_become_payload_names(self):
        parsed = parse_filter("language=python,project=app,branch=main")

        assert to_qdrant_text_filter(parsed) == models.Filter(
            must=[
                _match("programming_language", "python"),
                _match("git_project_name", "app"),
                _match("git_branch", "main"),
            ]
        )

    def test_qdrant_json_passes_through_unrenamed(self):
        text = '{"should": [{"key": "chunk_index", "match": {"value": 3}}]}'

        assert to_qdrant_text_filter(parse_filter(text)) == models.Filter(
            should=[_match("chunk_index", 3)]
        )

    def test_contains_is_match_text(self):
        assert to_qdrant_text_filter(parse_filter("file_path:contains:/src/")) == models.Filter(
            must=[models.FieldCondition(key="file_path", match=models.MatchText(text="/src/"))]
        )

    @pytest.mark.parametrize("text", TestNativeForms.NATIVE_MEILISEARCH)
    def test_meilisearch_syntax_is_refused_with_the_json_form(self, text):
        with pytest.raises(FilterError) as exc_info:
            to_qdrant_text_filter(parse_filter(text))

        message = str(exc_info.value)
        assert "MeiliSearch" in message
        for needle in ("must", "should", "must_not"):
            assert needle in message

    def test_the_json_hint_names_a_field_every_corpus_type_stores(self):
        """Native Qdrant JSON names payload fields, so the example must not use ``language``."""
        with pytest.raises(FilterError) as exc_info:
            to_qdrant_text_filter(parse_filter("NOT language = python"))

        assert '"key": "language"' not in str(exc_info.value)
        assert '"key": "filename"' in str(exc_info.value)


class TestQdrantTextFilterableFields:
    def test_pdf_chunk_and_section_fields_pass(self):
        fields = qdrant_text_filterable_fields("pdf")

        check_filterable(parse_filter("chunk_index=3,section_type=table"), fields, "Pdfs")
        check_filterable(parse_filter("page_number >= 2,filename=a.pdf"), fields, "Pdfs")

    @pytest.mark.parametrize("field", ["page", "language"])
    def test_pdf_refuses_a_field_no_chunk_carries_by_name(self, field):
        with pytest.raises(FilterError, match=f"'{field}' is not filterable"):
            check_filterable(
                parse_filter(f"{field}=x"), qdrant_text_filterable_fields("pdf"), "Pdfs"
            )

    def test_code_accepts_meilisearch_and_payload_names(self):
        fields = qdrant_text_filterable_fields("code")

        for text in ("project=x", "git_project_name=x", "language=python", "line_number > 3"):
            check_filterable(parse_filter(text), fields, "Code")

    def test_markdown_fields(self):
        fields = qdrant_text_filterable_fields("markdown")

        for text in ("section=Intro", "tags=a", "headings=b", "title=c", "chunk_count=2"):
            check_filterable(parse_filter(text), fields, "Notes")

    def test_includes_the_semantic_fields(self):
        for corpus_type in ("pdf", "code", "markdown"):
            assert set(semantic_filterable_fields(corpus_type)) <= set(
                qdrant_text_filterable_fields(corpus_type)
            )

    @pytest.mark.parametrize("collection_type", [None, "", "mystery", 3])
    def test_untyped_collection_has_no_field_list(self, collection_type):
        assert qdrant_text_filterable_fields(collection_type) is None

    def test_semantic_fields_are_unchanged(self):
        assert "chunk_index" not in semantic_filterable_fields("pdf")
        assert "section_type" not in semantic_filterable_fields("pdf")
