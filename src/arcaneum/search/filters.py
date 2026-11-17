"""One metadata filter dialect for semantic and text search (RDR-026 Q6).

``parse_filter`` reads a ``--filter`` string once into a backend-neutral
filter, which translates to a Qdrant ``Filter`` (``to_qdrant_filter``) and to
a MeiliSearch expression (``to_meilisearch_filter``). Neutral forms:

- ``key=value[,key=value]``, whitespace around ``=`` trimmed;
- ``key:op:value``, op one of ``match``, ``in``, ``gte``, ``gt``, ``lte``,
  ``lt`` or ``contains``; ``in`` takes ``|``-separated values, since ``,``
  separates terms;
- flat JSON ``{"field": value | [values]}``, where a list means ``in``.

Native forms pass through to their own mode only. JSON with a top-level
``must``, ``should`` or ``must_not`` is Qdrant's; a string with a MeiliSearch
keyword, ``!=``, ``>``, ``<`` or a double quote is MeiliSearch's. Anything
that does not parse raises ``FilterError``; no term is dropped.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, List, Optional, Sequence, Tuple, Union

if TYPE_CHECKING:
    # qdrant_client is imported where it is used, keeping it off the `arc` start
    # path (RDR-026 Q11).
    from qdrant_client.http import models


class FilterError(ValueError):
    """The filter does not parse, or cannot be sent to the requested backend."""


Scalar = Union[str, bool]

OPERATORS = ("match", "in", "gte", "gt", "lte", "lt", "contains")
_RANGE_SYMBOLS = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<"}
_QDRANT_NATIVE_KEYS = ("must", "should", "must_not")

# A MeiliSearch expression: one of its keywords as an upper-case word, a
# comparison operator, or a double quote (RDR-026 Q6).
_MEILI_NATIVE = re.compile(r'\b(?:AND|OR|NOT|TO|IN|EXISTS)\b|\bIS\s+(?:NULL|EMPTY)\b|!=|[<>"]')
_KEY = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]*$")
_EXTENDED_TERM = re.compile(r"^([^:=]+):([^:=]*):(.*)$", re.DOTALL)
_DIGITS = re.compile(r"^-?\d+$")

# Qdrant payload names for MeiliSearch filterable attributes that
# schema/document.py::to_qdrant_point stores under another name.
_QDRANT_FIELD_NAMES = {
    "language": "programming_language",
    "project": "git_project_name",
    "branch": "git_branch",
}


@dataclass(frozen=True)
class Condition:
    """One neutral term: ``key op value``."""

    key: str
    op: str
    value: Union[Scalar, float, int, Tuple[Scalar, ...]]


@dataclass(frozen=True)
class NeutralFilter:
    """Conditions that must all hold, translatable to either backend."""

    conditions: Tuple[Condition, ...]

    @property
    def fields(self) -> List[str]:
        """Each filtered key once, in first-use order."""
        return list(dict.fromkeys(c.key for c in self.conditions))


@dataclass(frozen=True)
class NativeFilter:
    """A backend's own filter, accepted only by that backend."""

    backend: str  # "qdrant" (a JSON dict) or "meilisearch" (an expression)
    expression: Any


ParsedFilter = Union[NeutralFilter, NativeFilter]


def parse_filter(filter_arg: Optional[str]) -> Optional[ParsedFilter]:
    """Parse a ``--filter`` string.

    Returns:
        A ``NeutralFilter``, a ``NativeFilter``, or None for an empty filter

    Raises:
        FilterError: the filter does not parse
    """
    if filter_arg is None or not filter_arg.strip():
        return None
    text = filter_arg.strip()
    if text.startswith("{"):
        return _parse_json(text)
    if _MEILI_NATIVE.search(text):
        return NativeFilter("meilisearch", text)
    return NeutralFilter(tuple(_parse_terms(text)))


def _check_key(key: str, source: str) -> str:
    if not key:
        raise FilterError(f"Missing key in {source!r}")
    if not _KEY.match(key):
        raise FilterError(
            f"Invalid key {key!r} in {source!r}: use letters, digits, '_', '.' or '-'"
        )
    return key


def _parse_terms(text: str) -> Iterable[Condition]:
    previous_op = None
    for term in text.split(","):
        term = term.strip()
        if not term:
            raise FilterError(f"Empty term in {text!r}")
        extended = _EXTENDED_TERM.match(term)
        if extended:
            key, op, value = (part.strip() for part in extended.groups())
            condition = _extended_condition(_check_key(key, term), op, value, term)
        elif "=" in term:
            key, value = (part.strip() for part in term.split("=", 1))
            _check_key(key, term)
            if not value:
                raise FilterError(f"Missing value in {term!r}")
            condition = Condition(key, "match", value)
        else:
            hint = " (separate 'in' values with '|')" if previous_op == "in" else ""
            raise FilterError(
                f"Unparseable term {term!r}: expected key=value or key:op:value{hint}"
            )
        previous_op = condition.op
        yield condition


def _parse_number(value: str, source: str) -> Union[int, float]:
    try:
        number = float(value)
    except ValueError:
        number = math.nan
    if not math.isfinite(number):
        raise FilterError(f"Range needs a number, got {value!r} in {source!r}")
    return int(number) if number.is_integer() else number


def _extended_condition(key: str, op: str, value: str, term: str) -> Condition:
    if op not in OPERATORS:
        raise FilterError(
            f"Unknown operator {op!r} in {term!r}; use one of: {', '.join(OPERATORS)}"
        )
    if op == "in":
        values = tuple(v.strip() for v in value.split("|"))
        if not value or any(not v for v in values):
            raise FilterError(f"Empty value in {term!r}")
        return Condition(key, "in", values)
    if not value:
        raise FilterError(f"Missing value in {term!r}")
    if op in _RANGE_SYMBOLS:
        return Condition(key, op, _parse_number(value, term))
    return Condition(key, op, value)


def _json_scalar(key: str, value: Any) -> Scalar:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    raise FilterError(
        f"Field {key!r} needs a string, integer or boolean (or a list of them), got "
        f"{json.dumps(value)}; use key:gte:N for a range"
    )


def _parse_json(text: str) -> ParsedFilter:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise FilterError(f"Invalid JSON filter: {e}") from e
    if not isinstance(data, dict):
        raise FilterError("A JSON filter must be an object")
    if any(key in data for key in _QDRANT_NATIVE_KEYS):
        return NativeFilter("qdrant", data)
    if not data:
        raise FilterError("A JSON filter must not be empty")
    conditions = []
    for key, value in data.items():
        _check_key(key, text)
        if isinstance(value, list):
            if not value or any(isinstance(v, (list, dict)) for v in value):
                raise FilterError(
                    f"Field {key!r} needs a non-empty list of strings, integers or booleans"
                )
            conditions.append(Condition(key, "in", tuple(_json_scalar(key, v) for v in value)))
        else:
            conditions.append(Condition(key, "match", _json_scalar(key, value)))
    return NeutralFilter(tuple(conditions))


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------


def _qdrant_match(key: str, value: Scalar):
    from qdrant_client.http import models

    condition = models.FieldCondition(key=key, match=models.MatchValue(value=value))
    if isinstance(value, str) and _DIGITS.match(value):
        # A neutral value has no type, and Qdrant matches "89" only against a
        # keyword payload: match an integer payload (page_number) as well.
        return models.Filter(
            should=[
                condition,
                models.FieldCondition(key=key, match=models.MatchValue(value=int(value))),
            ]
        )
    return condition


def _qdrant_any(key: str, values: Sequence[Scalar]):
    from qdrant_client.http import models

    if any(isinstance(v, bool) for v in values):
        return models.Filter(should=[_qdrant_match(key, v) for v in values])
    condition = models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))
    numbers = [int(v) for v in values if _DIGITS.match(v)]
    if not numbers:
        return condition
    return models.Filter(
        should=[condition, models.FieldCondition(key=key, match=models.MatchAny(any=numbers))]
    )


def _qdrant_condition(condition: Condition):
    from qdrant_client.http import models

    if condition.op == "match":
        return _qdrant_match(condition.key, condition.value)
    if condition.op == "in":
        return _qdrant_any(condition.key, condition.value)
    if condition.op == "contains":
        return models.FieldCondition(
            key=condition.key, match=models.MatchText(text=condition.value)
        )
    return models.FieldCondition(
        key=condition.key, range=models.Range(**{condition.op: condition.value})
    )


def to_qdrant_filter(parsed: Optional[ParsedFilter]) -> Optional[models.Filter]:
    """Translate a parsed filter for semantic search.

    Raises:
        FilterError: a MeiliSearch expression, or malformed Qdrant JSON
    """
    from qdrant_client.http import models

    if parsed is None:
        return None
    if isinstance(parsed, NativeFilter):
        if parsed.backend != "qdrant":
            raise FilterError(
                f"{parsed.expression!r} is a MeiliSearch expression, which only text search "
                "accepts. Semantic search takes key=value, key:op:value (e.g. "
                'page_number:gt:10) or JSON (e.g. {"filename": "a.pdf"}).'
            )
        try:
            return models.Filter(**parsed.expression)
        except Exception as e:
            raise FilterError(f"Invalid Qdrant filter structure: {e}") from e
    return models.Filter(must=[_qdrant_condition(c) for c in parsed.conditions])


# ---------------------------------------------------------------------------
# MeiliSearch
# ---------------------------------------------------------------------------


def _meili_value(value: Union[Scalar, int, float]) -> str:
    if isinstance(value, bool):
        text = "true" if value else "false"
    else:
        text = str(value)
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _meili_condition(condition: Condition) -> str:
    if condition.op == "match":
        return f"{condition.key} = {_meili_value(condition.value)}"
    if condition.op == "in":
        values = ", ".join(_meili_value(v) for v in condition.value)
        return f"{condition.key} IN [{values}]"
    if condition.op == "contains":
        raise FilterError(
            f"Text search cannot filter with 'contains' ({condition.key}:contains:...): "
            "MeiliSearch's CONTAINS is an experimental feature. Use semantic search, or match "
            "the whole value."
        )
    return f"{condition.key} {_RANGE_SYMBOLS[condition.op]} {_meili_value(condition.value)}"


def to_meilisearch_filter(parsed: Optional[ParsedFilter]) -> Optional[str]:
    """Translate a parsed filter for text search, quoting every value.

    Raises:
        FilterError: Qdrant JSON, or a ``contains`` term
    """
    if parsed is None:
        return None
    if isinstance(parsed, NativeFilter):
        if parsed.backend != "meilisearch":
            raise FilterError(
                "A JSON filter with must/should/must_not is a Qdrant filter, which only "
                'semantic search accepts. Text search takes JSON such as {"filename": "a.pdf"}, '
                "key=value, key:op:value, or a MeiliSearch expression."
            )
        return parsed.expression
    return " AND ".join(_meili_condition(c) for c in parsed.conditions)


# ---------------------------------------------------------------------------
# Field check and descriptions
# ---------------------------------------------------------------------------


def unfilterable_field_message(field: str, filterable: Sequence[str], index_name: str) -> str:
    """Name an unfilterable field and list the index's filterable ones."""
    lines = [f"Filter attribute '{field}' is not filterable."]
    if not filterable:
        lines.append(
            f"Index '{index_name}' has no filterable attributes configured; "
            "filters cannot be used with this index."
        )
        return "\n".join(lines)
    lines.append(f"Available filterable attributes for index '{index_name}':")
    lines.extend(f"  - {attr}" for attr in filterable)
    for attr in filterable:
        if field.lower() in attr.lower() or attr.lower() in field.lower():
            lines.append(f'Hint: Try --filter "{attr}=<value>"')
            break
    return "\n".join(lines)


def check_filterable(
    parsed: Optional[ParsedFilter], filterable: Sequence[str], index_name: str
) -> None:
    """Refuse a neutral filter on a field outside ``filterable``.

    Native filters are the escape hatch and are not checked.

    Raises:
        FilterError: naming the field and listing the filterable ones
    """
    if not isinstance(parsed, NeutralFilter):
        return
    for field in parsed.fields:
        if field not in filterable:
            raise FilterError(unfilterable_field_message(field, filterable, index_name))


def semantic_filterable_fields(collection_type: Any) -> Optional[List[str]]:
    """Fields a neutral semantic filter may name for a corpus type.

    Qdrant filters any payload field, so the list is the type's MeiliSearch
    ``filterableAttributes`` under their Qdrant payload names: one list for
    both modes. None (no check) for an untyped or unknown collection.
    """
    if not isinstance(collection_type, str) or not collection_type:
        return None
    from ..fulltext.indexes import get_index_settings

    try:
        settings = get_index_settings(collection_type)
    except ValueError:
        return None
    return [_QDRANT_FIELD_NAMES.get(f, f) for f in settings.get("filterableAttributes", [])]


def _describe_value(value: Any) -> str:
    if isinstance(value, tuple):
        return "[" + ", ".join(_describe_value(v) for v in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def describe_filter(parsed: Optional[ParsedFilter]) -> str:
    """Human-readable description of a parsed filter."""
    if parsed is None:
        return "No filters applied"
    if isinstance(parsed, NativeFilter):
        if parsed.backend == "qdrant":
            return build_filter_description(to_qdrant_filter(parsed))
        return parsed.expression
    parts = []
    for c in parsed.conditions:
        if c.op == "match":
            parts.append(f"{c.key}={_describe_value(c.value)}")
        elif c.op == "in":
            parts.append(f"{c.key} in {_describe_value(c.value)}")
        elif c.op == "contains":
            parts.append(f"{c.key} contains '{c.value}'")
        else:
            parts.append(f"{c.key}{_RANGE_SYMBOLS[c.op]}{_describe_value(c.value)}")
    return " AND ".join(parts)


def build_filter_description(filter_obj: Optional[models.Filter]) -> str:
    """Generate human-readable description of a Qdrant filter.

    Args:
        filter_obj: Qdrant Filter object

    Returns:
        Human-readable filter description
    """
    from qdrant_client.http import models

    if not filter_obj:
        return "No filters applied"

    descriptions = []

    if filter_obj.must:
        for condition in filter_obj.must:
            if isinstance(condition, models.FieldCondition):
                key = condition.key
                if condition.match:
                    if isinstance(condition.match, models.MatchValue):
                        descriptions.append(f"{key}={condition.match.value}")
                    elif isinstance(condition.match, models.MatchAny):
                        values = ",".join(str(v) for v in condition.match.any)
                        descriptions.append(f"{key} in [{values}]")
                    elif isinstance(condition.match, models.MatchText):
                        descriptions.append(f"{key} contains '{condition.match.text}'")
                elif condition.range:
                    range_desc = []
                    if condition.range.gte is not None:
                        range_desc.append(f">={condition.range.gte}")
                    if condition.range.gt is not None:
                        range_desc.append(f">{condition.range.gt}")
                    if condition.range.lte is not None:
                        range_desc.append(f"<={condition.range.lte}")
                    if condition.range.lt is not None:
                        range_desc.append(f"<{condition.range.lt}")
                    descriptions.append(f"{key} {' and '.join(range_desc)}")

    return " AND ".join(descriptions) if descriptions else "Complex filter"
