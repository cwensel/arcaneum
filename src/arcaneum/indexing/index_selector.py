"""Select indexed files by their stored metadata (`arc corpus sync --from-index`).

A selector is a comma-separated list of terms that must all hold for a file to
be selected. Each term compares one stored payload field:

    method=pymupdf_normalized
    method=pymupdf_normalized,version<0.10.0
    method=pymupdf_normalized,!fallback
    quality_manifest.page_coverage.page_count=3

`!key` selects records that do not store the field at all, as in Kubernetes
label selectors. Shorthand keys name the fields worth re-indexing by; any
other key is read as a dotted payload path. Matching happens client-side over a full scroll so that
combined extraction methods and version ordering compare correctly.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .collection_metadata import FILE_MANIFEST_PAYLOAD_KEY, FILE_MANIFEST_PAYLOAD_VALUE


class IndexSelectorError(ValueError):
    """A selector that cannot be parsed or cannot match this collection."""


@dataclass(frozen=True)
class _Field:
    path: str
    match: str = "equal"  # equal | component | version


SHORTHANDS: Dict[str, _Field] = {
    # Combined methods ("pymupdf4llm_markdown+pymupdf_normalized") record each
    # extractor that contributed text, so match any component.
    "method": _Field("quality_manifest.extraction_method", match="component"),
    "fallback": _Field("quality_manifest.fallback_reason"),
    "ocr": _Field("quality_manifest.ocr.reason"),
    "warning": _Field("quality_manifest.quality_warnings"),
    "version": _Field("app_version", match="version"),
}

_TERM = re.compile(r"^\s*([A-Za-z_][\w.]*)\s*(<=|>=|=|<|>)\s*(.*?)\s*$")
_ABSENT_TERM = re.compile(r"^\s*!\s*([A-Za-z_][\w.]*)\s*$")
_MISSING = object()
_PATH_KEYS = {"file_path", "canonical_path", FILE_MANIFEST_PAYLOAD_KEY}


@dataclass(frozen=True)
class SelectorTerm:
    key: str
    field: _Field
    op: str
    value: str


def parse_index_selector(expression: str) -> List[SelectorTerm]:
    """Parse a selector expression into terms, rejecting malformed input."""
    if not expression or not expression.strip():
        raise IndexSelectorError("Selector is empty; expected key=value (e.g. method=...)")

    terms = []
    for raw in expression.split(","):
        if raw.strip().startswith("!"):
            absent = _ABSENT_TERM.match(raw)
            if not absent:
                raise IndexSelectorError(
                    f"Selector term {raw.strip()!r} takes no value; !key selects records "
                    "that do not store key"
                )
            key = absent.group(1)
            terms.append(
                SelectorTerm(key=key, field=SHORTHANDS.get(key, _Field(key)), op="!", value="")
            )
            continue
        found = _TERM.match(raw)
        if not found:
            raise IndexSelectorError(
                f"Cannot parse selector term {raw.strip()!r}; expected key=value"
            )
        key, op, value = found.groups()
        if not value:
            raise IndexSelectorError(f"Selector term {raw.strip()!r} has no value")
        field = SHORTHANDS.get(key, _Field(key))
        if op != "=" and field.match != "version":
            raise IndexSelectorError(
                f"Comparison {op!r} is only supported for version; use = for {key!r}"
            )
        terms.append(SelectorTerm(key=key, field=field, op=op, value=value))
    return terms


def select_indexed_paths(
    qdrant: Any,
    collection_name: str,
    expression: str,
    page_size: int = 1000,
) -> List[str]:
    """Return the sorted canonical source paths whose stored metadata matches.

    Raises IndexSelectorError when a term names a field no indexed point
    carries, since that is a typo or a field this corpus type never records,
    not a legitimately empty selection.
    """
    terms = parse_index_selector(expression)
    payload_keys = sorted({t.field.path.split(".")[0] for t in terms} | _PATH_KEYS)

    selected = set()
    seen_fields = set()
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=collection_name,
            limit=page_size,
            offset=offset,
            with_payload=payload_keys,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            source = _source_path(payload)
            if not source:
                continue
            values = [
                (
                    _lookup_absence(payload, t.field.path)
                    if t.op == "!"
                    else _lookup(payload, t.field.path)
                )
                for t in terms
            ]
            for term, value in zip(terms, values):
                if term.op != "!" and value is not _MISSING:
                    seen_fields.add(term.field.path)
            if all(_matches(t, v) for t, v in zip(terms, values)):
                selected.add(source)
        if offset is None:
            break

    # A field no record stores is the expected case for !key, not a typo.
    absent = [t for t in terms if t.op != "!" and t.field.path not in seen_fields]
    if absent:
        names = ", ".join(
            f"{t.field.path} ({t.key}=)" if t.key in SHORTHANDS else t.field.path for t in absent
        )
        raise IndexSelectorError(
            f"No indexed point in '{collection_name}' records {names}; check the key, "
            "or this corpus type (or the arc version that indexed it) does not store it"
        )
    return sorted(selected)


def _source_path(payload: Dict[str, Any]) -> Optional[str]:
    """Return the path to re-index: chunks live under the canonical path."""
    if payload.get(FILE_MANIFEST_PAYLOAD_KEY) == FILE_MANIFEST_PAYLOAD_VALUE:
        return payload.get("canonical_path") or payload.get("file_path")
    return payload.get("file_path")


def _lookup(payload: Dict[str, Any], path: str) -> Any:
    value: Any = payload
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _lookup_absence(payload: Dict[str, Any], path: str) -> bool:
    """Return True when the record holding the field exists without the field.

    A point with no quality manifest at all says nothing about whether its
    manifest records fallback_reason, so it does not match `!fallback`.
    """
    parent_path, _, leaf = path.rpartition(".")
    parent = _lookup(payload, parent_path) if parent_path else payload
    return isinstance(parent, dict) and leaf not in parent


def _matches(term: SelectorTerm, value: Any) -> bool:
    if term.op == "!":
        return bool(value)
    if value is _MISSING:
        return False
    if term.field.match == "version":
        return _compare_versions(value, term.op, term.value)
    if isinstance(value, list):
        return any(_equals(item, term.value) for item in value)
    if term.field.match == "component" and isinstance(value, str):
        return _equals(value, term.value) or any(
            _equals(part, term.value) for part in re.split(r"[+:]", value)
        )
    return _equals(value, term.value)


def _equals(value: Any, expected: str) -> bool:
    if value is None:
        return expected.lower() in ("none", "null")
    if isinstance(value, bool):
        return str(value).lower() == expected.lower()
    if isinstance(value, (int, float)):
        try:
            return float(value) == float(expected)
        except ValueError:
            return False
    return str(value).lower() == expected.lower()


def _version_key(value: Any) -> Optional[Tuple[int, ...]]:
    numbers = re.findall(r"\d+", str(value)) if value is not None else []
    return tuple(int(n) for n in numbers) or None


def _compare_versions(value: Any, op: str, expected: str) -> bool:
    actual_key = _version_key(value)
    expected_key = _version_key(expected)
    if actual_key is None or expected_key is None:
        return False
    # 0.10 and 0.10.0 are the same release.
    width = max(len(actual_key), len(expected_key))
    actual_key += (0,) * (width - len(actual_key))
    expected_key += (0,) * (width - len(expected_key))
    return {
        "=": actual_key == expected_key,
        "<": actual_key < expected_key,
        "<=": actual_key <= expected_key,
        ">": actual_key > expected_key,
        ">=": actual_key >= expected_key,
    }[op]
