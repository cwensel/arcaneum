"""Read-only arc MCP server over stdio for review sessions (RDR-026 Q1).

``arc mcp`` serves the retrieval core (``arcaneum.retrieval``) in process, on
the official ``mcp`` SDK's low-level ``Server``. Review sessions launch it fresh
for every review, so it starts fast, never prompts, and keeps stdout for
protocol messages alone:

- **stdout guard**: before the SDK or the core is imported, fd 1 is saved and
  pointed at stderr, and only the saved descriptor carries protocol messages.
  print, a raw fd 1 write, logging, warnings, rich and tqdm all reach stderr.
- **Start**: no prompts, migrations, config writes, container checks or model
  downloads (the root CLI callback skips its migration for ``mcp``). A tool
  call writes nothing either: its clients read a legacy config in place and use
  only an existing MeiliSearch key.
- **Allowlist**: ``--corpus`` (repeatable) or ``ARC_MCP_CORPORA``
  (comma-separated); the flag wins. With one, a listing shows only the allowed
  corpora and any other name is ``unknown-corpus``.
- **Per call**: the core runs in a worker thread under ``--timeout``; a stalled
  call is abandoned (every core call is read-only) and reported as ``timeout``.
  ``--max-output-chars`` bounds the serialized ``data`` of every result.
- **Semantic**: the first ``search_semantic`` call loads the query embedder,
  which then stays resident, holding an embedder slot, until 5 minutes idle
  (``retrieval.semantic.SemanticSession``). It never downloads a model or
  stamps a prompt policy; an uncached model or an embedder busy past the
  call's time is ``semantic-unavailable``.
- **Launch config**: ``--print-config claude|codex`` prints a client config
  that names an absolute command (``launch_command``), so a client started
  under launchd's fixed PATH still finds arc, and exits before any server
  start.
- **Results**: a success carries ``data`` as ``structuredContent`` and as one
  JSON text block; a failure sets ``isError`` with ``{code, message}`` and the
  text ``<code>: <message>``, the codes of ``arcaneum.cli.errors``. The output
  cap shortens a failure's message too.

Importing this module loads neither the SDK nor the backends' client libraries.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

from .cli.errors import (
    GENERIC_ERROR_CODE,
    ArcaneumError,
    InvalidArgumentError,
    OperationTimeoutError,
    ServiceDownError,
    UnknownCorpusError,
)

if TYPE_CHECKING:
    import mcp_types

logger = logging.getLogger(__name__)

SERVER_NAME = "arc"
ALLOWLIST_ENV = "ARC_MCP_CORPORA"
DEFAULT_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_OUTPUT_CHARS = 40_000
MIN_MAX_OUTPUT_CHARS = 4_000

LIST_CORPORA_TOOL: dict[str, Any] = {
    "name": "list_corpora",
    "title": "List corpora",
    "description": (
        "List the arc corpora this server can search: name, type (pdf, code or "
        "markdown), sync status, item count and description. Same data as "
        "`arc corpus list --json`."
    ),
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}

# MCP-only policy for the search tools (RDR-026 Q2, Q3): the server refuses a
# limit above the maximum rather than clamping it, since the inputSchema
# advertises the bound. The CLI's own defaults differ (no snippet cap in JSON).
SEARCH_LIMIT_DEFAULT = 10
SEARCH_LIMIT_MAX = 25
SEARCH_CONTENT_CHARS_DEFAULT = 1200

SEARCH_TEXT_TOOL: dict[str, Any] = {
    "name": "search_text",
    "title": "Search text",
    "description": (
        "Ranked full-text search for keywords, identifiers and quoted phrases "
        "across one or more arc corpora. Each hit carries a cite "
        "(<corpus>:<locator>) and a snippet. Same data as "
        "`arc search text --json`."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Keywords or identifiers; wrap a phrase in double quotes for an "
                    "exact match. No regular expressions."
                ),
            },
            "corpora": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "minItems": 1,
                "description": "The corpora to search, as list_corpora names them.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": SEARCH_LIMIT_MAX,
                "default": SEARCH_LIMIT_DEFAULT,
                "description": "Maximum number of hits.",
            },
            "offset": {
                "type": "integer",
                "minimum": 0,
                "default": 0,
                "description": "Number of hits to skip, for paging.",
            },
            "filter": {
                "type": "string",
                "description": (
                    "Metadata filter: key=value, key:op:value, a JSON object such as "
                    '{"language": "python"}, or a MeiliSearch expression.'
                ),
            },
            "max_content_chars": {
                "type": "integer",
                "minimum": 0,
                "default": SEARCH_CONTENT_CHARS_DEFAULT,
                "description": "Maximum characters per snippet; 0 means unlimited.",
            },
        },
        "required": ["query", "corpora"],
        "additionalProperties": False,
    },
}

SEARCH_SEMANTIC_TOOL: dict[str, Any] = {
    "name": "search_semantic",
    "title": "Search semantic",
    "description": (
        "Vector search for concepts and paraphrases across one or more arc "
        "corpora. Each hit carries a cite (<corpus>:<locator>) and a snippet. "
        "Same data as `arc search semantic --json`. Uses only embedding models "
        "already in the local cache; on semantic-unavailable, use search_text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            **SEARCH_TEXT_TOOL["input_schema"]["properties"],
            "query": {
                "type": "string",
                "description": "A natural-language description of what to find.",
            },
            "filter": {
                "type": "string",
                "description": (
                    "Metadata filter: key=value, key:op:value, a JSON object such as "
                    '{"language": "python"}, or a Qdrant filter object (must, should, '
                    "must_not)."
                ),
            },
            "score_threshold": {
                "type": "number",
                "description": "Minimum similarity score; weaker hits are dropped.",
            },
        },
        "required": ["query", "corpora"],
        "additionalProperties": False,
    },
}


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


def allowlist_names(flag_corpora: Iterable[str], environ: Mapping[str, str]) -> list[str]:
    """The allowlisted corpus names in order: ``--corpus`` names, else ``ARC_MCP_CORPORA``.

    Empty when neither names a corpus. Repeated names appear once.
    """
    names = [name.strip() for name in flag_corpora if name.strip()]
    if not names:
        names = [name.strip() for name in environ.get(ALLOWLIST_ENV, "").split(",") if name.strip()]
    return list(dict.fromkeys(names))


def resolve_allowlist(
    flag_corpora: Iterable[str], environ: Mapping[str, str]
) -> frozenset[str] | None:
    """The corpus allowlist: ``--corpus`` names, else ``ARC_MCP_CORPORA``.

    Returns None (every corpus allowed) when neither names a corpus.
    """
    return frozenset(allowlist_names(flag_corpora, environ)) or None


def require_allowed(allowlist: frozenset[str] | None, names: Iterable[str]) -> None:
    """Raise ``unknown-corpus`` naming every requested corpus outside the allowlist."""
    if allowlist is None:
        return
    refused = sorted({name for name in names if name not in allowlist})
    if refused:
        served = ", ".join(sorted(allowlist))
        raise UnknownCorpusError(
            f"Unknown corpus: {', '.join(refused)} (this server serves {served})"
        )


# ---------------------------------------------------------------------------
# Launch config (--print-config)
# ---------------------------------------------------------------------------

PRINT_CONFIG_TARGETS = ("claude", "codex")
# The module an interpreter runs as arc. Not ``arcaneum`` (a package
# ``__main__.py``): the repo's .gitignore ignores ``_*`` files bar __init__.py.
LAUNCH_MODULE = "arcaneum.cli.main"


def _shebang_interpreter(script: str) -> str | None:
    """The interpreter ``script``'s shebang names, or None when it has no shebang."""
    try:
        with open(script, "rb") as handle:
            first = handle.readline(1024)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    words = first[2:].split()
    return os.fsdecode(words[0]) if words else ""


def _absolute_interpreter(interpreter: str) -> bool:
    """True when ``interpreter`` is an existing executable named by absolute path.

    ``/usr/bin/env`` does not count: env searches PATH, which a
    launchd-started client does not share with the shell that printed the
    config. ``/bin/sh`` (pip's wrapper for a long interpreter path) does.
    """
    return (
        os.path.isabs(interpreter)
        and os.path.basename(interpreter) != "env"
        and os.path.isfile(interpreter)
        and os.access(interpreter, os.X_OK)
    )


def launch_command(argv0: str | None = None, executable: str | None = None) -> list[str]:
    """An absolute command that starts this ``arc``, with no PATH lookup.

    The running script (``argv0``, resolved) when it is executable and its
    shebang names an absolute interpreter, as an installed console script's
    does. An executable script whose shebang needs PATH (a checkout's
    ``bin/arc``, ``#!/usr/bin/env python3``) runs under the running
    interpreter, so the script keeps its own ``sys.path`` setup (``bin/arc``
    puts the checkout's ``src`` first). Otherwise (``python -m
    arcaneum.cli.main``) the running interpreter with ``-m arcaneum.cli.main``.
    """
    argv0 = sys.argv[0] if argv0 is None else argv0
    executable = sys.executable if executable is None else executable
    if argv0 and os.path.isfile(argv0) and os.access(argv0, os.X_OK):
        script = os.path.realpath(argv0)
        interpreter = _shebang_interpreter(script)
        if interpreter is not None:
            if _absolute_interpreter(interpreter):
                return [script]
            return [executable, script]
    return [executable, "-m", LAUNCH_MODULE]


def _format_seconds(seconds: float) -> str:
    """``seconds`` as text that parses back to the same float (``20.0`` as ``20``)."""
    text = repr(float(seconds))
    return text[:-2] if text.endswith(".0") else text


def server_args(
    corpora: Iterable[str] = (),
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> list[str]:
    """The ``arc mcp`` arguments that serve ``corpora`` under these limits.

    Defaults are left out, so the launched server tracks arc's own.
    """
    args = ["mcp"]
    for name in corpora:
        args += ["--corpus", name]
    if timeout != DEFAULT_TIMEOUT_SECONDS:
        args += ["--timeout", _format_seconds(timeout)]
    if max_output_chars != DEFAULT_MAX_OUTPUT_CHARS:
        args += ["--max-output-chars", str(max_output_chars)]
    return args


def _toml_value(value: str | list[str]) -> str:
    # A JSON string or array of strings is a TOML basic string or array, once
    # DEL, which JSON leaves raw and TOML forbids, is escaped. Non-ASCII stays
    # raw: JSON's surrogate-pair escapes are not valid TOML.
    return json.dumps(value, ensure_ascii=False).replace("\x7f", "\\u007f")


def render_config(
    target: str,
    command: list[str],
    *,
    corpora: Iterable[str] = (),
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
) -> str:
    """A client config snippet that launches ``command`` as the ``arc`` MCP server.

    ``claude``: a ``--mcp-config`` JSON document. ``codex``: an
    ``[mcp_servers.arc]`` TOML table, followed by comments giving the same
    settings as ``codex -c`` overrides.

    Raises:
        ValueError: ``target`` is not one of ``PRINT_CONFIG_TARGETS``
    """
    executable, prefix = command[0], command[1:]
    args = [*prefix, *server_args(corpora, timeout, max_output_chars)]
    if target == "claude":
        config = {"mcpServers": {SERVER_NAME: {"command": executable, "args": args}}}
        return json.dumps(config, ensure_ascii=False, indent=2)
    if target == "codex":
        import shlex

        table = f"mcp_servers.{SERVER_NAME}"
        overrides = [
            f"#   -c {shlex.quote(f'{table}.command={_toml_value(executable)}')}",
            f"#   -c {shlex.quote(f'{table}.args={_toml_value(args)}')}",
        ]
        return "\n".join(
            [
                f"[{table}]",
                f"command = {_toml_value(executable)}",
                f"args = {_toml_value(args)}",
                "",
                "# The same settings as codex -c overrides:",
                *overrides,
            ]
        )
    raise ValueError(f"unknown --print-config target: {target}")


# ---------------------------------------------------------------------------
# Results and the output cap
# ---------------------------------------------------------------------------


def serialize(data: Any) -> str:
    """The JSON text block of a result; ``--max-output-chars`` bounds its length."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _cap_tail(
    data: dict[str, Any], key: str, omitted_key: str, max_chars: int, refusal: str
) -> dict[str, Any]:
    """Fit ``data`` into ``max_chars`` by dropping whole ``data[key]`` items from the tail.

    A capped result gains ``output_truncated: true`` and ``omitted_key``.

    Raises:
        InvalidArgumentError: ``refusal``, when the result does not fit even with no items
    """
    if len(serialize(data)) <= max_chars:
        return data
    entries = data[key]

    def keep(count: int) -> dict[str, Any]:
        return {
            **data,
            key: entries[:count],
            "output_truncated": True,
            omitted_key: len(entries) - count,
        }

    if len(serialize(keep(0))) > max_chars:
        raise InvalidArgumentError(refusal)
    # The serialized size grows with the kept prefix: find the longest that fits.
    low, high = 0, len(entries) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if len(serialize(keep(middle))) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return keep(low)


def cap_listing(data: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Fit a listing into ``max_chars`` by dropping whole entries from the tail.

    A capped listing gains ``output_truncated: true`` and ``omitted_corpora``.

    Raises:
        InvalidArgumentError: the listing does not fit even with no entries
    """
    refusal = (
        f"list_corpora's result does not fit --max-output-chars {max_chars} even with no entries"
    )
    return _cap_tail(data, "corpora", "omitted_corpora", max_chars, refusal)


def cap_search(data: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Fit a search result into ``max_chars`` by dropping whole hits from the tail.

    Applied after ``max_content_chars`` bounds each snippet, so no hit is cut
    mid-way. A capped result gains ``output_truncated: true`` and
    ``omitted_hits``.

    Raises:
        InvalidArgumentError: the result does not fit even with no hits (a
            query or corpus list longer than the cap)
    """
    refusal = (
        f"the search result does not fit --max-output-chars {max_chars} even with "
        "no hits; shorten the query or name fewer corpora"
    )
    return _cap_tail(data, "hits", "omitted_hits", max_chars, refusal)


def success_result(data: dict[str, Any]) -> mcp_types.CallToolResult:
    import mcp_types

    text = serialize(data)
    return mcp_types.CallToolResult(
        content=[mcp_types.TextContent(text=text)], structured_content=json.loads(text)
    )


TRUNCATED_MARKER = " [truncated]"


def fit_message(code: str, message: str, max_chars: int) -> str:
    """Shorten ``message`` so the error's ``{code, message}`` fits ``max_chars``.

    A shortened message ends with ``TRUNCATED_MARKER``. The serialized pair is
    longer than its ``<code>: <message>`` text, so the text fits as well.
    """

    def overflow(text: str) -> int:
        return len(serialize({"code": code, "message": text})) - max_chars

    if overflow(message) <= 0:
        return message
    keep = len(message)
    while keep > 0 and overflow(message[:keep] + TRUNCATED_MARKER) > 0:
        keep -= overflow(message[:keep] + TRUNCATED_MARKER)
    return message[: max(keep, 0)] + TRUNCATED_MARKER


def error_result(
    code: str, message: str, max_chars: int = DEFAULT_MAX_OUTPUT_CHARS
) -> mcp_types.CallToolResult:
    import mcp_types

    message = fit_message(code, message, max_chars)
    return mcp_types.CallToolResult(
        is_error=True,
        content=[mcp_types.TextContent(text=f"{code}: {message}")],
        structured_content={"code": code, "message": message},
    )


def _service_errors_down(service_errors: list[str]) -> ServiceDownError:
    """``service-down`` naming each service a listing could not reach.

    The core lists the reachable service's corpora when one fails (the CLI
    prints that partial listing); a review session gets the failure instead,
    since a half listing reads as corpora that exist in one system only.
    """
    reasons = []
    for entry in service_errors:
        service, _, detail = entry.partition(": ")
        reasons.append(f"{service} is unreachable: {detail}" if detail else entry)
    return ServiceDownError("; ".join(reasons))


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def _qdrant_client():
    """A Qdrant client from the config as it stands: a legacy config is read, not migrated."""
    from .cli import utils

    return utils.create_qdrant_client(config_path=utils.resolve_config_path(migrate=False))


def _qdrant_search_client():
    """``_qdrant_client`` with the search timeout, as ``arc search semantic`` makes it."""
    from .cli import utils

    return utils.create_qdrant_client(
        config_path=utils.resolve_config_path(migrate=False), for_search=True
    )


def _meili_client():
    """A MeiliSearch client on the existing key: none is generated or stored."""
    from .cli import utils

    return utils.create_meili_client(generate_key=False)


def _source_roots_client():
    """A Qdrant client for reading cite roots, or None when one cannot be made.

    Text search does not need Qdrant: a failed read only means absolute-path
    cites (RDR-026 Q4). It gets the CLI's short timeout, and a legacy config
    is read, not migrated.
    """
    from .cli import utils
    from .cli.fulltext import SOURCE_ROOTS_TIMEOUT_SECONDS

    try:
        return utils.create_qdrant_client(
            config_path=utils.resolve_config_path(migrate=False),
            timeout=SOURCE_ROOTS_TIMEOUT_SECONDS,
        )
    except Exception as error:
        logger.debug("No Qdrant client for source roots: %s", error)
        return None


def _reject_arguments(tool: str, arguments: Mapping[str, Any], allowed: Iterable[str]) -> None:
    unexpected = sorted(set(arguments) - set(allowed))
    if unexpected:
        raise InvalidArgumentError(f"{tool} got unexpected arguments: {', '.join(unexpected)}")


def _string_argument(arguments: Mapping[str, Any], name: str, *, required: bool) -> str | None:
    value = arguments.get(name)
    if value is None:
        if required:
            raise InvalidArgumentError(f"{name} is required")
        return None
    if not isinstance(value, str):
        raise InvalidArgumentError(f"{name} must be a string, got {type(value).__name__}")
    return value


def _corpora_argument(arguments: Mapping[str, Any]) -> list[str]:
    value = arguments.get("corpora")
    if value is None:
        raise InvalidArgumentError("corpora is required: name one or more corpora")
    if not isinstance(value, list) or not value:
        raise InvalidArgumentError("corpora must be a non-empty array of corpus names")
    if not all(isinstance(name, str) and name for name in value):
        raise InvalidArgumentError("corpora must hold only non-empty corpus names")
    return value


def _integer_argument(
    arguments: Mapping[str, Any],
    name: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """An integer argument within its advertised bounds; out of bounds is refused, not clamped."""
    value = arguments.get(name)
    if value is None:
        return default
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentError(f"{name} must be an integer, got {value!r}")
    if value < minimum:
        raise InvalidArgumentError(f"{name} must be at least {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise InvalidArgumentError(f"{name} must be at most {maximum}, got {value}")
    return value


def _number_argument(arguments: Mapping[str, Any], name: str) -> float | None:
    value = arguments.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidArgumentError(f"{name} must be a number, got {value!r}")
    return float(value)


def _search_arguments(tool: dict[str, Any], arguments: Mapping[str, Any]) -> dict[str, Any]:
    """The arguments both search tools share, within their advertised bounds."""
    _reject_arguments(tool["name"], arguments, tool["input_schema"]["properties"])
    return {
        "query": _string_argument(arguments, "query", required=True),
        "corpora": _corpora_argument(arguments),
        "limit": _integer_argument(
            arguments, "limit", SEARCH_LIMIT_DEFAULT, minimum=1, maximum=SEARCH_LIMIT_MAX
        ),
        "offset": _integer_argument(arguments, "offset", 0, minimum=0),
        "filter": _string_argument(arguments, "filter", required=False) or None,
        "max_content_chars": _integer_argument(
            arguments, "max_content_chars", SEARCH_CONTENT_CHARS_DEFAULT, minimum=0
        ),
    }


# The core gives up this long before the per-call deadline (at most 1 s, a tenth
# of a short timeout), so a busy embedder reports semantic-unavailable itself
# rather than racing the deadline's timeout.
def _semantic_report_margin(timeout: float) -> float:
    return min(1.0, timeout / 10)


class ArcTools:
    """The tool implementations, independent of the transport."""

    def __init__(
        self,
        *,
        allowlist: frozenset[str] | None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    ):
        self.allowlist = allowlist
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        # One event per call whose worker was left running, set when it returns.
        self._abandoned: list[threading.Event] = []
        # Made on the first semantic call, so start loads no embedder (RDR-026 Q9).
        self._semantic: Any = None
        self._semantic_lock = threading.Lock()

    def tool_definitions(self) -> list[mcp_types.Tool]:
        import mcp_types

        read_only = mcp_types.ToolAnnotations(read_only_hint=True)
        return [
            mcp_types.Tool(**definition, annotations=read_only)
            for definition in (LIST_CORPORA_TOOL, SEARCH_TEXT_TOOL, SEARCH_SEMANTIC_TOOL)
        ]

    def abandoned_calls_running(self) -> bool:
        """True while an abandoned call (timed out or cancelled) still runs."""
        self._abandoned = [done for done in self._abandoned if not done.is_set()]
        return bool(self._abandoned)

    async def call(
        self, name: str, arguments: Mapping[str, Any] | None
    ) -> mcp_types.CallToolResult:
        handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
            "list_corpora": self.list_corpora,
            "search_text": self.search_text,
            "search_semantic": self.search_semantic,
        }
        handler = handlers.get(name)
        if handler is None:
            import mcp_types
            from mcp.shared.exceptions import MCPError

            raise MCPError(mcp_types.INVALID_PARAMS, f"Unknown tool: {name}")
        try:
            data = await self._run(name, handler, dict(arguments or {}))
        except ArcaneumError as error:
            return error_result(error.code, str(error), self.max_output_chars)
        except Exception as error:  # an unclassified core failure
            logger.warning("%s failed", name, exc_info=True)
            return error_result(
                GENERIC_ERROR_CODE, f"{type(error).__name__}: {error}", self.max_output_chars
            )
        return success_result(data)

    async def _run(self, name, handler, arguments):
        import anyio
        import anyio.to_thread

        done = threading.Event()

        def work():
            try:
                return handler(arguments)
            finally:
                done.set()

        try:
            try:
                with anyio.fail_after(self.timeout) as scope:
                    return await anyio.to_thread.run_sync(work, abandon_on_cancel=True)
            finally:
                # The deadline, a client's cancel or the transport closing can
                # each abandon the worker mid-call; serve() must not wait on it.
                if not done.is_set():
                    self._abandoned.append(done)
        except TimeoutError:
            if not scope.cancel_called:
                raise  # the core's own timeout, not the per-call deadline
            raise OperationTimeoutError(
                f"{name} exceeded the {self.timeout:g} s per-call timeout"
            ) from None

    def list_corpora(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        _reject_arguments("list_corpora", arguments, ())
        from .retrieval import list_corpora

        listing = list_corpora(qdrant_factory=_qdrant_client, meili_factory=_meili_client)
        if listing.service_errors:
            raise _service_errors_down(listing.service_errors)
        data = listing.to_data()
        if self.allowlist is not None:
            data["corpora"] = [c for c in data["corpora"] if c["name"] in self.allowlist]
        return cap_listing(data, self.max_output_chars)

    def search_text(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """``arc search text --json``'s ``data`` under the MCP bounds and the output cap.

        Every argument and the allowlist are checked before any client is made.
        """
        args = _search_arguments(SEARCH_TEXT_TOOL, arguments)
        require_allowed(self.allowlist, args["corpora"])

        from .retrieval import search_text

        result = search_text(
            args["query"],
            args["corpora"],
            filter=args["filter"],
            limit=args["limit"],
            offset=args["offset"],
            client=_meili_client(),
            qdrant_client=_source_roots_client(),
        )
        data = result.to_data(max_content_chars=args["max_content_chars"])
        return cap_search(data, self.max_output_chars)

    @property
    def _semantic_session(self):
        """The session's resident embedder, made on first use."""
        with self._semantic_lock:
            if self._semantic is None:
                from .retrieval.semantic import SemanticSession

                self._semantic = SemanticSession()
            return self._semantic

    def search_semantic(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """``arc search semantic --json``'s ``data`` under the MCP bounds and the output cap.

        Every argument and the allowlist are checked before any client is made.
        The search never downloads a model or stamps a prompt policy, and
        reports an uncached model or a busy embedder as ``semantic-unavailable``
        (RDR-026 Q9).
        """
        deadline = time.monotonic() + self.timeout - _semantic_report_margin(self.timeout)
        args = _search_arguments(SEARCH_SEMANTIC_TOOL, arguments)
        score_threshold = _number_argument(arguments, "score_threshold")
        require_allowed(self.allowlist, args["corpora"])

        result = self._semantic_session.search(
            args["query"],
            args["corpora"],
            deadline=deadline,
            filter=args["filter"],
            limit=args["limit"],
            offset=args["offset"],
            score_threshold=score_threshold,
            client=_qdrant_search_client(),
        )
        data = result.to_data(max_content_chars=args["max_content_chars"])
        return cap_search(data, self.max_output_chars)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------


def guard_stdout() -> int:
    """Point fd 1 at stderr and return a descriptor for the original stdout.

    Everything that writes to stdout afterwards, at the Python or the fd level
    and in child processes, reaches stderr; only the returned descriptor
    reaches the client.
    """
    sys.stdout.flush()
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    return protocol_fd


def _configure_logging() -> None:
    logging.basicConfig(
        stream=sys.stderr,
        level=logging.WARNING,
        format="arc mcp: %(levelname)s %(name)s: %(message)s",
    )


async def _serve(tools: ArcTools, protocol_fd: int) -> None:
    import io

    import anyio
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    from . import __version__

    async def list_tools(_ctx, _params):
        import mcp_types

        return mcp_types.ListToolsResult(tools=tools.tool_definitions())

    async def call_tool(_ctx, params):
        return await tools.call(params.name, params.arguments)

    server = Server(
        SERVER_NAME,
        version=__version__,
        on_list_tools=list_tools,
        on_call_tool=call_tool,
    )
    # closefd=False: an abandoned worker may outlive the transport, and the
    # descriptor must not be recycled under it.
    wire = io.TextIOWrapper(
        os.fdopen(protocol_fd, "wb", closefd=False), encoding="utf-8", newline="\n"
    )
    async with stdio_server(stdout=anyio.wrap_file(wire)) as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def serve(
    *,
    corpora: Iterable[str] = (),
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_output_chars: int = DEFAULT_MAX_OUTPUT_CHARS,
    environ: Mapping[str, str] | None = None,
) -> int:
    """Serve the arc tools over stdio until the client closes stdin."""
    protocol_fd = guard_stdout()
    _configure_logging()

    import anyio

    tools = ArcTools(
        allowlist=resolve_allowlist(corpora, os.environ if environ is None else environ),
        timeout=timeout,
        max_output_chars=max_output_chars,
    )
    anyio.run(_serve, tools, protocol_fd)
    if tools.abandoned_calls_running():
        # An abandoned read is still blocked in its worker thread. The client
        # has gone, so exit now rather than wait for it.
        sys.stderr.flush()
        os._exit(0)
    return 0
