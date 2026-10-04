"""Plain search output and explicit content budgets shared by both backends."""


def limit_content(content: str, max_chars: int) -> tuple[str, bool]:
    """Return at most max_chars source characters; zero means unlimited."""
    if max_chars < 0:
        raise ValueError("max_chars must be nonnegative")
    truncated = max_chars > 0 and len(content) > max_chars
    return (content[:max_chars] if truncated else content), truncated


def content_preview(content: str, max_chars: int) -> str:
    """Preserve source whitespace and mark truncation outside the content budget."""
    content, truncated = limit_content(content, max_chars)
    return content + ("\n[truncated]" if truncated else "")


def limit_json_content(result: dict, max_chars: int) -> dict:
    """Copy a hit and bound content, including duplicate payload/highlight text.

    Only used for an explicit budget, leaving legacy JSON output unchanged.
    The marker covers truncation in any of the known content fields.
    """
    output = dict(result)
    truncated = False
    if isinstance(output.get("content"), str):
        output["content"], truncated = limit_content(output["content"], max_chars)
    for key in ("metadata", "_formatted"):
        nested = output.get(key)
        if isinstance(nested, dict) and isinstance(nested.get("content"), str):
            nested = dict(nested)
            nested["content"], nested_truncated = limit_content(nested["content"], max_chars)
            output[key] = nested
            truncated |= nested_truncated
    output["content_truncated"] = truncated
    return output


def format_compact_results(
    query: str,
    corpora: list[str],
    results: list[dict],
    offset: int = 0,
    max_content_chars: int = 500,
) -> str:
    """Render normalized hits without terminal styling, wrapping or indentation."""
    lines = [f"Q: {query} | corpus: {', '.join(corpora)} | returned: {len(results)}"]
    for rank, result in enumerate(results, offset + 1):
        lines.append("---")
        header = f"[{rank}] {result['location']}"
        if len(corpora) > 1:
            header += f" | corpus: {result['corpus']}"
        if result.get("score") is not None:
            header += f" score={result['score']}"
        lines.append(header)
        lines.append(content_preview(result.get("content") or "", max_content_chars))
    return "\n".join(lines)
