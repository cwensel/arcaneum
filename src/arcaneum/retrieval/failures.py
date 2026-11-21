"""Classify backend failures into the retrieval error codes (RDR-026 Q3)."""

from __future__ import annotations

from ..cli.errors import ServiceDownError


def _connection_error_types() -> tuple[type[BaseException], ...]:
    # Imported on demand: these libraries are loaded by the time a client has
    # failed, and the core stays cheap to import.
    import httpx
    import requests

    from ..fulltext.backend import connection_error_types as meili_connection_error_types

    return (
        ConnectionError,
        httpx.ConnectError,
        httpx.ConnectTimeout,
        requests.exceptions.ConnectionError,
        *meili_connection_error_types(),
    )


def is_connection_failure(exc: BaseException) -> bool:
    """True when ``exc``, or an error it wraps, means the service is unreachable.

    Follows qdrant-client's ``ResponseHandlingException.source`` and the
    ``__cause__``/``__context__`` chain, since callers such as
    ``search_collection`` re-raise transport errors as plain exceptions.
    """
    connection_types = _connection_error_types()
    seen: set[int] = set()
    pending: list[BaseException | None] = [exc]
    while pending:
        current = pending.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, connection_types):
            return True
        source = getattr(current, "source", None)
        if isinstance(source, BaseException):
            pending.append(source)
        pending.extend([current.__cause__, current.__context__])
    return False


def service_down(service: str, exc: BaseException | None = None) -> ServiceDownError:
    """Build a ``service-down`` error that names the unreachable service."""
    detail = f": {exc}" if exc is not None else ""
    return ServiceDownError(f"{service} is unreachable{detail}")
