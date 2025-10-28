"""Metadata filter DSL parser for search (RDR-007)."""

from qdrant_client.http import models
import json
from typing import Optional


def parse_filter(filter_arg: str) -> Optional[models.Filter]:
    """Parse filter from CLI argument.

    Supports three formats:
    1. Simple: key=value,key=value (e.g., "language=python,project=backend")
    2. Extended: key:op:value (e.g., "language:in:python,java" or "priority:gte:5")
    3. JSON: Full Qdrant filter as JSON (e.g., '{"must": [...]}')

    Args:
        filter_arg: Filter string from CLI

    Returns:
        Qdrant Filter object or None if filter_arg is empty

    Raises:
        ValueError: If filter format is invalid
    """
    if not filter_arg:
        return None

    # Detect format
    if filter_arg.startswith('{'):
        return parse_json_filter(filter_arg)
    elif ':' in filter_arg and not filter_arg.count(':') == filter_arg.count('='):
        # Extended format uses colons, simple uses equals
        # If we have colons but not all are in key=value pairs, it's extended
        return parse_extended_filter(filter_arg)
    else:
        return parse_simple_filter(filter_arg)


def parse_simple_filter(filter_str: str) -> Optional[models.Filter]:
    """Parse simple key=value,key=value format.

    Example: "language=python,project=backend"

    Args:
        filter_str: Comma-separated key=value pairs

    Returns:
        Qdrant Filter with must conditions
    """
    conditions = []

    for pair in filter_str.split(','):
        pair = pair.strip()
        if '=' not in pair:
            continue

        key, value = pair.split('=', 1)
        conditions.append(
            models.FieldCondition(
                key=key.strip(),
                match=models.MatchValue(value=value.strip())
            )
        )

    return models.Filter(must=conditions) if conditions else None


def parse_json_filter(json_str: str) -> models.Filter:
    """Parse Qdrant JSON filter format.

    Example: '{"must": [{"key": "language", "match": {"value": "python"}}]}'

    Args:
        json_str: JSON string representing Qdrant filter

    Returns:
        Qdrant Filter object

    Raises:
        ValueError: If JSON is invalid or doesn't match Filter schema
    """
    try:
        filter_dict = json.loads(json_str)
        return models.Filter(**filter_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON filter: {e}")
    except Exception as e:
        raise ValueError(f"Invalid filter structure: {e}")


def parse_extended_filter(filter_str: str) -> Optional[models.Filter]:
    """Parse extended DSL: key:op:value.

    Supported operators:
    - match (exact): language:match:python
    - in (multiple values): language:in:python,java
    - gte, gt, lte, lt (range): priority:gte:5
    - contains (text search): path:contains:/src/

    Example: "language:in:python,java,priority:gte:2"

    Args:
        filter_str: Comma-separated key:op:value terms

    Returns:
        Qdrant Filter with must conditions
    """
    conditions = []

    for term in filter_str.split(','):
        term = term.strip()
        parts = term.split(':', 2)

        if len(parts) != 3:
            continue

        key, op, value = [p.strip() for p in parts]

        if op == 'in':
            # Multiple values: language:in:python,java
            # Note: Value might contain commas, so we need to handle this
            # We'll split on comma but this is a known limitation
            values = [v.strip() for v in value.split(',')]
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=values)
                )
            )
        elif op in ('gte', 'gt', 'lte', 'lt'):
            # Range query: priority:gte:5
            try:
                numeric_value = float(value)
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        range=models.Range(**{op: numeric_value})
                    )
                )
            except ValueError:
                # If not numeric, skip this condition
                continue
        elif op == 'contains':
            # Text search: path:contains:/src/
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchText(text=value)
                )
            )
        elif op == 'match':
            # Explicit exact match: language:match:python
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value)
                )
            )

    return models.Filter(must=conditions) if conditions else None


def build_filter_description(filter_obj: Optional[models.Filter]) -> str:
    """Generate human-readable description of a filter.

    Args:
        filter_obj: Qdrant Filter object

    Returns:
        Human-readable filter description
    """
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
