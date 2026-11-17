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


# ---------------------------------------------------------------------------
# Allowlist
# ---------------------------------------------------------------------------


def resolve_allowlist(
    flag_corpora: Iterable[str], environ: Mapping[str, str]
) -> frozenset[str] | None:
    """The corpus allowlist: ``--corpus`` names, else ``ARC_MCP_CORPORA``.

    Returns None (every corpus allowed) when neither names a corpus.
    """
    names = [name.strip() for name in flag_corpora if name.strip()]
    if not names:
        names = [name.strip() for name in environ.get(ALLOWLIST_ENV, "").split(",") if name.strip()]
    return frozenset(names) or None


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
# Results and the output cap
# ---------------------------------------------------------------------------


def serialize(data: Any) -> str:
    """The JSON text block of a result; ``--max-output-chars`` bounds its length."""
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def cap_listing(data: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Fit a listing into ``max_chars`` by dropping whole entries from the tail.

    A capped listing gains ``output_truncated: true`` and ``omitted_corpora``.

    Raises:
        InvalidArgumentError: the listing does not fit even with no entries
    """
    if len(serialize(data)) <= max_chars:
        return data
    corpora = data["corpora"]

    def keep(count: int) -> dict[str, Any]:
        return {
            **data,
            "corpora": corpora[:count],
            "output_truncated": True,
            "omitted_corpora": len(corpora) - count,
        }

    if len(serialize(keep(0))) > max_chars:
        raise InvalidArgumentError(
            f"list_corpora's result does not fit --max-output-chars {max_chars} "
            "even with no entries"
        )
    # The serialized size grows with the kept prefix: find the longest that fits.
    low, high = 0, len(corpora) - 1
    while low < high:
        middle = (low + high + 1) // 2
        if len(serialize(keep(middle))) <= max_chars:
            low = middle
        else:
            high = middle - 1
    return keep(low)


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


def _meili_client():
    """A MeiliSearch client on the existing key: none is generated or stored."""
    from .cli import utils

    return utils.create_meili_client(generate_key=False)


def _reject_arguments(tool: str, arguments: Mapping[str, Any], allowed: Iterable[str]) -> None:
    unexpected = sorted(set(arguments) - set(allowed))
    if unexpected:
        raise InvalidArgumentError(f"{tool} got unexpected arguments: {', '.join(unexpected)}")


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

    def tool_definitions(self) -> list[mcp_types.Tool]:
        import mcp_types

        read_only = mcp_types.ToolAnnotations(read_only_hint=True)
        return [mcp_types.Tool(**LIST_CORPORA_TOOL, annotations=read_only)]

    def abandoned_calls_running(self) -> bool:
        """True while an abandoned call (timed out or cancelled) still runs."""
        self._abandoned = [done for done in self._abandoned if not done.is_set()]
        return bool(self._abandoned)

    async def call(
        self, name: str, arguments: Mapping[str, Any] | None
    ) -> mcp_types.CallToolResult:
        handlers: dict[str, Callable[[Mapping[str, Any]], dict[str, Any]]] = {
            "list_corpora": self.list_corpora,
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
