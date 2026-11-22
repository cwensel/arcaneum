"""Corpus management commands for dual indexing (RDR-009).

This module implements the 'corpus create' command that creates both a Qdrant
collection and a MeiliSearch index in a single operation.
"""

import logging
import sys
from datetime import datetime, timezone

from qdrant_client.models import HnswConfigDiff
from rich.console import Console

from ..cli.corpus_defaults import DEFAULT_MODELS_BY_CORPUS_TYPE
from ..cli.corpus_lock import acquire_corpus_lock
from ..cli.errors import InvalidArgumentError, ResourceNotFoundError
from ..cli.interaction_logger import interaction_logger
from ..cli.output import print_error, print_info, print_json
from ..cli.utils import (
    build_vectors_config,
    create_meili_client,
    create_qdrant_client,
    get_model_dimensions,
)
from ..fulltext.backend import MeiliTextBackend
from ..fulltext.indexes import get_index_settings
from ..indexing.collection_metadata import (
    DEFAULT_TEXT_BACKEND,
    get_collection_metadata,
    metadata_exclusion_filter,
    set_collection_metadata,
    text_backend_of,
    update_collection_metadata,
    user_point_count,
    validate_text_backend,
)
from ..indexing.common.sync import MetadataBasedSync
from ..retrieval import list_corpora
from ..retrieval.qdrant_text import QdrantTextBackend
from ..retrieval.corpora import (
    UNKNOWN_LEGACY,
    _item_unit_for_corpus_type,
    _payload_file_paths,
)

console = Console()
logger = logging.getLogger(__name__)


def default_models_for_type(corpus_type: str) -> str:
    """Return the stable default embedding model list for a corpus type."""
    try:
        return DEFAULT_MODELS_BY_CORPUS_TYPE[corpus_type]
    except KeyError as e:
        raise InvalidArgumentError(f"Unknown corpus type: {corpus_type}") from e


def map_corpus_type_to_canonical(corpus_type: str) -> str:
    """Map corpus type aliases to canonical names.

    Args:
        corpus_type: Type from CLI (pdf, code, markdown)

    Returns:
        Canonical type name for MeiliSearch (pdf-docs, source-code, markdown-docs)
    """
    type_map = {
        "pdf": "pdf-docs",
        "code": "source-code",
        "markdown": "markdown-docs",
    }
    return type_map.get(corpus_type, corpus_type)


def _format_last_sync_for_table(last_sync, last_sync_status: str) -> str:
    """Return a compact, human-readable sync timestamp for table output."""
    if not last_sync:
        return "never synced" if last_sync_status == "never_synced" else UNKNOWN_LEGACY

    if not isinstance(last_sync, str):
        return str(last_sync)

    try:
        parsed = datetime.fromisoformat(last_sync.replace("Z", "+00:00"))
    except ValueError:
        return last_sync

    timestamp = parsed.replace(microsecond=0).strftime("%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        return timestamp
    if parsed.utcoffset() == timezone.utc.utcoffset(None):
        return f"{timestamp}Z"
    return f"{timestamp}{parsed.strftime('%z')}"


def _format_list_item_count(corpus):
    """Format detailed corpus-list item counts."""
    if corpus["item_count"] is None:
        if corpus.get("file_count") is None:
            return UNKNOWN_LEGACY
        parts = []
    else:
        parts = [f"{corpus['item_count']} {corpus['item_unit']}"]
    if corpus.get("file_count") is not None:
        parts.append(f"{corpus['file_count']} {corpus['file_unit']}")
    return ", ".join(parts)


def _format_chunk_summary(corpus):
    """Format corpus-list chunk counts in one column.

    A Qdrant text corpus shows its text index's chunks (T) beside its points.
    """
    q_chunks = corpus["qdrant_chunks"]
    if "meili_chunks" not in corpus:
        t_chunks = corpus["text_chunks"]
        return str(q_chunks) if q_chunks == t_chunks else f"Q: {q_chunks} / T: {t_chunks}"
    m_chunks = corpus["meili_chunks"]
    if q_chunks == m_chunks:
        return str(q_chunks)
    return f"Q: {q_chunks} / M: {m_chunks}"


def list_corpora_command(details: bool, output_json: bool):
    """List all corpora with parity status.

    A corpus is a paired Qdrant collection + MeiliSearch index. This command
    formats ``arcaneum.retrieval.list_corpora``, which discovers the pairs and
    their sync status and writes nothing (RDR-026 Q10).

    Args:
        details: Show extended listing columns, including exact item counts
        output_json: Output as JSON
    """
    from rich.table import Table

    from ..config import meili_corpora_conflict, resolve_meili_settings

    interaction_logger.start("corpus", "list")

    try:
        settings = resolve_meili_settings()
        listing = list_corpora(
            details=details,
            qdrant_factory=create_qdrant_client,
            text_factory=lambda: MeiliTextBackend(create_meili_client()),
            meilisearch_enabled=settings.enabled,
        )
    except Exception as e:
        interaction_logger.finish(error=str(e))
        raise

    corpora = listing.corpora
    # MeiliSearch disabled while a corpus still searches text on it (arcaneum#183).
    conflict = meili_corpora_conflict(
        settings, [c["name"] for c in corpora if c["text_backend"] == "meili"]
    )
    if not output_json:
        for service_error in listing.service_errors:
            print_error(f"Could not connect to {service_error}")
        if conflict:
            print_error(conflict)

    # Output
    if output_json:
        data = listing.to_data()
        if conflict:
            data["warnings"] = [conflict]
        print_json("success", f"Found {len(corpora)} corpora", data)
    else:
        if not corpora:
            print_info("No corpora found")
            print_info("Create one with: arc corpus create <name> --type <pdf|code|markdown>")
        else:
            table = Table(title="Corpora", expand=True)
            table.add_column("Name", style="cyan", no_wrap=True)
            table.add_column("Type", style="blue", no_wrap=True)
            table.add_column("Text", style="blue", no_wrap=True)
            if details:
                table.add_column("Model", style="magenta", no_wrap=True)
            table.add_column("Status", style="green", no_wrap=True)
            if details:
                table.add_column("Last Sync", style="yellow", no_wrap=True)
            table.add_column("Items", style="yellow", justify="right", no_wrap=True)
            if details:
                table.add_column("Chunks", style="yellow", justify="right", no_wrap=True)
            table.add_column("Description", style="dim", ratio=1)

            for c in corpora:
                # Format status with color
                status = c["status"]
                if status == "synced":
                    status_str = "[green]synced[/green]"
                elif status == "chunk_mismatch":
                    status_str = "[yellow]chunk_mismatch[/yellow]"
                elif status == "qdrant_only":
                    status_str = "[yellow]qdrant_only[/yellow]"
                elif status == "coverage_gap":
                    status_str = "[yellow]coverage_gap[/yellow]"
                elif status == "text_index_missing":
                    status_str = "[red]text_index_missing[/red]"
                else:
                    status_str = "[yellow]meili_only[/yellow]"

                row = [
                    c["name"],
                    c["type"] or "—",
                    c["text_backend"],
                ]
                if details:
                    row.append(c["model_summary"])
                row.append(status_str)
                if details:
                    row.append(_format_last_sync_for_table(c["last_sync"], c["last_sync_status"]))
                row.append(_format_list_item_count(c))
                if details:
                    row.append(_format_chunk_summary(c))
                row.append(c["description"] or "—")
                table.add_row(*row)

            console.print(table)

    interaction_logger.finish(result_count=len(corpora))


def delete_corpus_command(
    name: str,
    confirm: bool,
    output_json: bool,
    lock_wait: bool = True,
    lock_timeout: float | None = None,
):
    """Delete both Qdrant collection and MeiliSearch index for a corpus.

    Args:
        name: Corpus name (used for both collection and index)
        confirm: Skip confirmation prompt
        output_json: If True, output JSON format
        lock_wait: If False, fail immediately when the corpus lock is held
        lock_timeout: Seconds to wait for the lock (None = default)
    """
    import click

    # Start interaction logging (RDR-018)
    interaction_logger.start("corpus", "delete", corpus=name)

    qdrant_deleted = False
    meili_deleted = False
    errors = []

    try:
        # Check what exists before asking for confirmation
        qdrant_exists = False
        meili_exists = False

        try:
            qdrant = create_qdrant_client()
            qdrant.get_collection(name)
            qdrant_exists = True
        except Exception:
            pass

        try:
            meili = create_meili_client()
            if meili.health_check() and meili.index_exists(name):
                meili_exists = True
        except Exception:
            pass

        if not qdrant_exists and not meili_exists:
            raise ResourceNotFoundError(f"Corpus '{name}' not found")

        # Confirmation prompt
        if not confirm:
            if output_json:
                raise InvalidArgumentError("--confirm flag required for non-interactive deletion")

            parts = []
            if qdrant_exists:
                parts.append("Qdrant collection")
            if meili_exists:
                parts.append("MeiliSearch index")

            if not click.confirm(
                f"Delete {' and '.join(parts)} for corpus '{name}'? This cannot be undone."
            ):
                print_info("Cancelled.")
                return

        # Take the corpus write lock only now: acquiring before the prompt above
        # would block a running sync on a human answering y/n. From here on the
        # work is destructive, so an in-flight sync must not interleave with it
        # (kata htmw).
        with acquire_corpus_lock(name, wait=lock_wait, timeout=lock_timeout, quiet=output_json):
            # Delete Qdrant collection
            if qdrant_exists:
                try:
                    qdrant = create_qdrant_client()
                    qdrant.delete_collection(name)
                    qdrant_deleted = True
                    if not output_json:
                        console.print(f"[green]✅ Deleted Qdrant collection '{name}'[/green]")
                except Exception as e:
                    errors.append(f"Qdrant: {e}")
                    if not output_json:
                        print_error(f"Failed to delete Qdrant collection: {e}")

            # Delete MeiliSearch index
            if meili_exists:
                try:
                    meili = create_meili_client()
                    meili.delete_index(name)
                    meili_deleted = True
                    if not output_json:
                        console.print(f"[green]✅ Deleted MeiliSearch index '{name}'[/green]")
                except Exception as e:
                    errors.append(f"MeiliSearch: {e}")
                    if not output_json:
                        print_error(f"Failed to delete MeiliSearch index: {e}")

        # Output results
        if output_json:
            data = {
                "corpus": name,
                "qdrant_deleted": qdrant_deleted,
                "meili_deleted": meili_deleted,
                "errors": errors if errors else None,
            }
            if errors:
                print_json("partial", f"Partially deleted corpus '{name}'", data)
            else:
                print_json("success", f"Deleted corpus '{name}'", data)
        else:
            if errors:
                print_error(f"Corpus '{name}' partially deleted. Errors: {', '.join(errors)}")
            else:
                console.print(f"\n[green]✅ Corpus '{name}' deleted[/green]")

        # Log operation (RDR-018)
        if errors:
            interaction_logger.finish(error="; ".join(errors))
        else:
            interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to delete corpus: {e}", output_json)
        sys.exit(1)


def create_corpus_command(
    name: str,
    corpus_type: str,
    models: str | None,
    description: str | None,
    output_json: bool,
    text_backend: str | None = None,
):
    """Create the Qdrant collection and the corpus's text index.

    This implements the first command of the 2-command workflow:
    1. corpus create (this command) - creates both systems
    2. corpus sync - indexes documents to both systems

    Args:
        name: Corpus name (used for both collection and index)
        corpus_type: Type of corpus (pdf, code, markdown)
        models: Comma-separated list of embedding models. If None, inferred
            from corpus_type using stable defaults.
        output_json: If True, output JSON format
        text_backend: ``meili`` (a MeiliSearch index) or ``qdrant`` (a text
            index on the collection's ``text`` payload; MeiliSearch is never
            contacted), recorded in the collection metadata (RDR-027 Q2).
            None takes ``corpus.default_text_backend`` (inferred when absent:
            ``meili`` for an existing MeiliSearch user, else ``qdrant``);
            ``meili`` is refused while ``meilisearch.enabled`` is false
            (arcaneum#183)
    """
    from ..config import resolve_meili_settings

    if not models:
        models = default_models_for_type(corpus_type)
    settings = resolve_meili_settings()
    if text_backend is None:
        text_backend = settings.default_text_backend

    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "corpus",
        "create",
        corpus=name,
        corpus_type=corpus_type,
        models=models,
        description=description,
        text_backend=text_backend,
    )

    try:
        try:
            validate_text_backend(text_backend)
        except ValueError as e:
            raise InvalidArgumentError(str(e))
        if text_backend == "meili" and not settings.enabled:
            raise InvalidArgumentError(
                f"Cannot create '{name}' on MeiliSearch: {settings.disabled_reason()}, "
                "or create it with --text-backend qdrant"
            )
        if not output_json:
            print_info(f"Creating corpus '{name}'")
            print_info(f"Type: {corpus_type}, Models: {models}, Text backend: {text_backend}")

        # Parse and validate models, build vectors config
        model_list = [m.strip() for m in models.split(",")]
        try:
            vectors_config = build_vectors_config(model_list)
        except ValueError as e:
            raise InvalidArgumentError(str(e))

        # Step 1: Create Qdrant collection
        if not output_json:
            print_info("Step 1/2: Creating Qdrant collection...")

        qdrant = create_qdrant_client()

        try:
            qdrant.create_collection(
                collection_name=name,
                vectors_config=vectors_config,
                hnsw_config=HnswConfigDiff(m=16, ef_construct=100),
                on_disk_payload=True,
            )

            # Set collection metadata (type and model)
            set_collection_metadata(
                client=qdrant,
                collection_name=name,
                collection_type=corpus_type,
                model=models,
                description=description,
                text_backend=text_backend,
            )
            # Sync deletes, renames and passage windows filter on these.
            MetadataBasedSync(qdrant).ensure_corpus_payload_indexes(name)
            if text_backend == "qdrant":
                QdrantTextBackend(qdrant).ensure(name, corpus_type)

            if not output_json:
                console.print(f"[green]✅ Qdrant collection '{name}' created[/green]")

        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str:
                raise InvalidArgumentError(f"Qdrant collection '{name}' already exists")
            raise

        # Step 2: Create MeiliSearch index, unless the collection serves text.
        if text_backend == "meili":
            if not output_json:
                print_info("Step 2/2: Creating MeiliSearch index...")

            meili = create_meili_client()

            try:
                # Verify MeiliSearch is available
                if not meili.health_check():
                    raise ResourceNotFoundError(
                        "MeiliSearch server not available. "
                        "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
                    )

                # Check if index already exists
                if meili.index_exists(name):
                    raise InvalidArgumentError(f"MeiliSearch index '{name}' already exists")

                # Get settings based on corpus type
                canonical_type = map_corpus_type_to_canonical(corpus_type)
                settings = get_index_settings(canonical_type)

                # Create index with settings
                meili.create_index(name, primary_key="id", settings=settings)

                if not output_json:
                    console.print(f"[green]✅ MeiliSearch index '{name}' created[/green]")

            except InvalidArgumentError:
                raise
            except ResourceNotFoundError:
                raise
            except Exception as e:
                # MeiliSearch failed but Qdrant succeeded
                if not output_json:
                    print_error(f"MeiliSearch index creation failed: {e}")
                    print_info(f"Note: Qdrant collection '{name}' was created successfully")
                    print_info(
                        "You may need to create the MeiliSearch index manually or delete the Qdrant collection"
                    )
                raise

        # Success output
        data = {
            "corpus": name,
            "type": corpus_type,
            "description": description,
            "models": model_list,
            "vectors": {m: get_model_dimensions(m) for m in model_list},
            "qdrant_collection": name,
            "text_backend": text_backend,
            "meilisearch_index": name if text_backend == "meili" else None,
        }

        if output_json:
            print_json(
                "success", f"Corpus '{name}' created with {len(model_list)} models", data=data
            )
        else:
            console.print(f"\n[green]✅ Corpus '{name}' ready for indexing![/green]")
            for model in model_list:
                dims = get_model_dimensions(model)
                console.print(f"  • {model}: {dims}D")
            console.print(f"\n[dim]Next: arc corpus sync {name} /path/to/files[/dim]")

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to create corpus: {e}", output_json)
        sys.exit(1)


def update_corpus_command(
    name: str,
    description: str | None,
    clear_description: bool,
    output_json: bool,
):
    """Update corpus metadata without reindexing documents.

    Args:
        name: Corpus name
        description: New description text. Empty string is preserved.
        clear_description: Remove the stored description
        output_json: If True, output JSON format
    """
    interaction_logger.start("corpus", "update", corpus=name)

    try:
        if description is None and not clear_description:
            raise InvalidArgumentError(
                "Nothing to update. Pass --description or --clear-description."
            )
        if description is not None and clear_description:
            raise InvalidArgumentError(
                "--description and --clear-description are mutually exclusive"
            )

        qdrant = create_qdrant_client()
        try:
            qdrant.get_collection(name)
        except Exception as e:
            raise ResourceNotFoundError(f"Corpus '{name}' not found") from e

        updates = {
            "description": None if clear_description else description,
        }
        metadata = update_collection_metadata(qdrant, name, **updates)

        data = {
            "corpus": name,
            "description": metadata.get("description"),
        }
        if output_json:
            print_json("success", f"Corpus '{name}' updated", data=data)
        else:
            if clear_description:
                console.print(f"[green]✅ Cleared description for '{name}'[/green]")
            else:
                console.print(f"[green]✅ Updated description for '{name}'[/green]")

        interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to update corpus: {e}", output_json)
        sys.exit(1)


def _get_qdrant_item_count(client, collection_name: str, collection_type: str) -> int:
    """Get count of unique items in a Qdrant collection.

    Scrolls through collection to count the natural item type:
    repositories for code collections, files/documents otherwise.

    Args:
        client: Qdrant client
        collection_name: Collection name
        collection_type: Type of collection (pdf, code, markdown)

    Returns:
        Count of unique items
    """
    unique_items = set()
    offset = None
    payload_fields = (
        ["git_project_identifier"] if collection_type == "code" else ["file_path", "file_paths"]
    )

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=metadata_exclusion_filter(),
            limit=1000,
            offset=offset,
            with_payload=payload_fields,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            if point.payload:
                if collection_type == "code":
                    item_id = point.payload.get("git_project_identifier")
                    if item_id:
                        unique_items.add(item_id)
                else:
                    for item_id in _payload_file_paths(point.payload):
                        unique_items.add(item_id)

        if offset is None:
            break

    return len(unique_items)


def _get_meili_item_count(client, index_name: str, collection_type: str) -> int:
    """Get count of unique items in a MeiliSearch index.

    Args:
        client: MeiliSearch FullTextClient
        index_name: Index name
        collection_type: Type of corpus

    Returns:
        Count of unique repositories for code, unique file paths otherwise
    """
    if collection_type != "code":
        file_paths = client.get_all_file_paths(index_name)
        return len(file_paths)

    projects = {
        doc["git_project_identifier"]
        for doc in MeiliTextBackend(client).iter_documents(index_name, ["git_project_identifier"])
        if doc.get("git_project_identifier")
    }

    return len(projects)


def _print_text_index_coverage(name: str, coverage) -> None:
    """The human-readable text coverage of a Qdrant text corpus (RDR-027 Q8)."""
    console.print("\n[bold]Qdrant Text Index:[/bold]")
    if coverage is None:
        console.print("  [yellow]Not readable[/yellow]")
        return
    if coverage.indexed is None:
        console.print("  [red]Missing: arc search text fails on this corpus[/red]")
    else:
        console.print(
            f"  Chunks: {coverage.indexed:,} of {coverage.searchable_chunks:,} searchable"
        )
    if coverage.zero_token_chunks:
        console.print(
            f"  Zero-token chunks: {coverage.zero_token_chunks:,} "
            "[dim](yield no token; a chunking finding, not a gap)[/dim]"
        )
    if not coverage.complete:
        console.print(f"  [dim]Fix with: arc corpus repair {name} (preview with --dry-run)[/dim]")


def corpus_info_command(name: str, output_json: bool):
    """Show combined information about a corpus (Qdrant + MeiliSearch).

    Displays unified view of a dual-index corpus including:
    - Item counts (unique files or repos indexed)
    - Chunk counts (total chunks across both systems)
    - Parity status

    Args:
        name: Corpus name (used for both collection and index)
        output_json: If True, output JSON format
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("corpus", "info", corpus=name)

    qdrant_info = None
    meili_info = None
    errors = []
    qdrant_client = None

    try:
        # Gather Qdrant collection info
        qdrant_client = create_qdrant_client()
        try:
            col_info = qdrant_client.get_collection(name)
            metadata = get_collection_metadata(qdrant_client, name)
            collection_type = metadata.get("collection_type")

            # Build vectors dict
            vectors = {}
            if hasattr(col_info.config.params, "vectors") and isinstance(
                col_info.config.params.vectors, dict
            ):
                for vector_name, vector_params in col_info.config.params.vectors.items():
                    vectors[vector_name] = {
                        "size": vector_params.size,
                        "distance": str(vector_params.distance),
                    }

            chunk_count = user_point_count(qdrant_client, name, col_info)

            # Get unique item count (files or repos)
            item_count = _get_qdrant_item_count(qdrant_client, name, collection_type)

            qdrant_info = {
                "name": name,
                "status": str(col_info.status),
                "item_count": item_count,
                "chunk_count": chunk_count,
                "type": collection_type,
                "model": metadata.get("model"),
                "description": metadata.get("description"),
                "text_backend": text_backend_of(metadata),
                "vectors": vectors,
                "hnsw_config": {
                    "m": col_info.config.hnsw_config.m,
                    "ef_construct": col_info.config.hnsw_config.ef_construct,
                },
            }
        except Exception as e:
            errors.append(f"Qdrant: {e}")

        # A corpus Qdrant does not hold is a MeiliSearch index (RDR-027 Q2).
        text_backend = qdrant_info["text_backend"] if qdrant_info else DEFAULT_TEXT_BACKEND
        coverage = None
        if text_backend == "qdrant":
            # Its text lives in Qdrant: coverage replaces parity, and
            # MeiliSearch is never read (RDR-027 Q8).
            try:
                text_index = QdrantTextBackend(qdrant_client, {name: qdrant_info["type"]})
                # A display path: reuse the chunk count, not a full-scan exact count.
                coverage = text_index.coverage(
                    name, files=False, chunks=qdrant_info["chunk_count"]
                )
            except Exception as e:
                errors.append(f"Qdrant text index: {e}")
        else:
            # Gather MeiliSearch index info
            meili = create_meili_client()
            try:
                if not meili.health_check():
                    errors.append("MeiliSearch: Server not available")
                elif not meili.index_exists(name):
                    errors.append(f"MeiliSearch: Index '{name}' not found")
                else:
                    stats = meili.get_index_stats(name)

                    # Get unique item count
                    item_count = _get_meili_item_count(meili, name, collection_type)
                    chunk_count = stats.get("numberOfDocuments", 0)

                    meili_info = {
                        "name": name,
                        "item_count": item_count,
                        "chunk_count": chunk_count,
                        "is_indexing": stats.get("isIndexing", False),
                    }
            except Exception as e:
                if "not available" not in str(e).lower():
                    errors.append(f"MeiliSearch: {e}")

        # If neither system has the corpus, it's an error
        if qdrant_info is None and meili_info is None:
            raise ResourceNotFoundError(
                f"Corpus '{name}' not found. "
                f"Create with: arc corpus create {name} --type <pdf|code|markdown>"
            )

        # Compute parity status based on item counts
        parity = {"status": "unknown"}
        if text_backend == "qdrant":
            if coverage is not None:
                parity = {"status": coverage.status, "qdrant_items": qdrant_info["item_count"]}
        elif qdrant_info and meili_info:
            qdrant_items = qdrant_info["item_count"]
            meili_items = meili_info["item_count"]
            diff = abs(qdrant_items - meili_items)

            if diff == 0:
                parity = {
                    "status": "synced",
                    "qdrant_items": qdrant_items,
                    "meili_items": meili_items,
                }
            else:
                parity = {
                    "status": "out_of_sync",
                    "qdrant_items": qdrant_items,
                    "meili_items": meili_items,
                    "difference": diff,
                }
        elif qdrant_info:
            parity = {"status": "qdrant_only", "qdrant_items": qdrant_info["item_count"]}
        elif meili_info:
            parity = {"status": "meili_only", "meili_items": meili_info["item_count"]}

        # Determine item label based on type
        collection_type = qdrant_info.get("type") if qdrant_info else "unknown"
        item_label = _item_unit_for_corpus_type(collection_type)
        if coverage is not None:
            text_chunks = coverage.indexed or 0
        else:
            text_chunks = meili_info["chunk_count"] if meili_info else None

        # Output
        if output_json:
            data = {
                "corpus": name,
                "type": collection_type,
                "description": (qdrant_info.get("description") if qdrant_info else None),
                "text_backend": text_backend,
                "text_chunks": text_chunks,
                "item_label": item_label,
                "parity": parity,
                "qdrant": qdrant_info,
                "meilisearch": meili_info,
                "errors": errors if errors else None,
            }
            if text_backend == "qdrant":
                data["coverage"] = coverage.to_data() if coverage is not None else None
            print_json("success", f"Corpus '{name}' information", data=data)
        else:
            # Header
            console.print(f"\n[bold cyan]Corpus: {name}[/bold cyan]")
            console.print(f"Type: [bold]{collection_type}[/bold]")
            console.print(f"Text backend: [bold]{text_backend}[/bold]")
            if qdrant_info and qdrant_info.get("description") is not None:
                console.print(f"Description: {qdrant_info['description']}")

            # Parity status
            if parity["status"] == "synced":
                console.print(
                    f"Status: [green]synced[/green] ({parity['qdrant_items']:,} {item_label})"
                )
            elif parity["status"] == "out_of_sync":
                console.print(
                    f"Status: [red]out of sync[/red] (Qdrant: {parity['qdrant_items']:,}, MeiliSearch: {parity['meili_items']:,} {item_label})"
                )
            elif parity["status"] == "qdrant_only":
                console.print(
                    f"Status: [yellow]Qdrant only[/yellow] ({parity['qdrant_items']:,} {item_label})"
                )
            elif parity["status"] == "meili_only":
                console.print(
                    f"Status: [yellow]MeiliSearch only[/yellow] ({parity['meili_items']:,} {item_label})"
                )
            elif parity["status"] == "coverage_gap":
                console.print(
                    f"Status: [yellow]coverage gap[/yellow] (the text index holds "
                    f"{coverage.indexed:,} of {coverage.searchable_chunks:,} searchable chunks)"
                )
            elif parity["status"] == "text_index_missing":
                console.print("Status: [red]text index missing[/red]")

            # Qdrant section
            if qdrant_info:
                console.print("\n[bold]Qdrant Collection:[/bold]")
                console.print(f"  {item_label.capitalize()}: {qdrant_info['item_count']:,}")
                console.print(f"  Chunks: {qdrant_info['chunk_count']:,}")
                console.print(f"  Status: {qdrant_info['status']}")
                if qdrant_info["vectors"]:
                    console.print("  Models:")
                    for model_name, vec_info in qdrant_info["vectors"].items():
                        console.print(
                            f"    - {model_name}: {vec_info['size']}D ({vec_info['distance']})"
                        )
            else:
                console.print("\n[yellow]Qdrant Collection: not found[/yellow]")

            # Text section: the Qdrant text index, or the MeiliSearch index
            if text_backend == "qdrant":
                _print_text_index_coverage(name, coverage)
            elif meili_info:
                console.print("\n[bold]MeiliSearch Index:[/bold]")
                console.print(f"  {item_label.capitalize()}: {meili_info['item_count']:,}")
                console.print(f"  Chunks: {meili_info['chunk_count']:,}")
                if meili_info["is_indexing"]:
                    console.print("  Status: [yellow]indexing[/yellow]")
            else:
                console.print("\n[yellow]MeiliSearch Index: not found[/yellow]")

            # Errors/warnings
            if errors:
                console.print(f"\n[dim]Warnings: {', '.join(errors)}[/dim]")

            # Usage hint
            console.print("\n[dim]Search with:[/dim]")
            console.print(f'[dim]  arc search semantic "query" --corpus {name}[/dim]')
            console.print(f'[dim]  arc search text "query" --corpus {name}[/dim]')

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except ResourceNotFoundError:
        interaction_logger.finish(error="corpus not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to get corpus info: {e}", output_json)
        sys.exit(1)


PARITY_ITEM_STATUSES = ("synced", "mismatch", "qdrant_only", "meili_only")
# A Qdrant text corpus's items share its coverage status (RDR-027 Q8).
COVERAGE_ITEM_STATUSES = ("synced", "coverage_gap", "text_index_missing", "unknown")
_ITEM_STATUS_COLOR = {"synced": "green", "mismatch": "red", "text_index_missing": "red"}


def _scan_qdrant_parity_items(qdrant, name: str, collection_type: str | None) -> dict:
    """Group Qdrant chunks per item and count them.

    Items are keyed by git_project_identifier for code collections and by
    file_path for every other type, matching _scan_meili_parity_items.
    """
    if collection_type == "code":
        id_field = "git_project_identifier"
        payload_fields = [
            "git_project_name",
            "git_project_identifier",
            "git_branch",
            "git_commit_hash",
            "git_remote_url",
        ]
    else:
        id_field = "file_path"
        payload_fields = ["file_path", "file_hash", "file_size", "filename"]

    qdrant_items = {}
    offset = None
    while True:
        points, offset = qdrant.scroll(
            collection_name=name,
            scroll_filter=metadata_exclusion_filter(),
            limit=1000,
            offset=offset,
            with_payload=payload_fields,
            with_vectors=False,
        )

        if not points:
            break

        for point in points:
            if not point.payload:
                continue

            item_id = point.payload.get(id_field)
            if item_id and item_id not in qdrant_items:
                qdrant_items[item_id] = {
                    **{k: point.payload.get(k) for k in payload_fields},
                    "qdrant_chunks": 1,
                }
            elif item_id:
                qdrant_items[item_id]["qdrant_chunks"] += 1

        if offset is None:
            break

    return qdrant_items


def _qdrant_parity_items_from_verification(verification) -> dict | None:
    """Per-file Qdrant point counts taken from a file-level verification result.

    The verifier has already scrolled every point, counting each chunk_index
    and its surplus copies, so actual_chunks + duplicate_chunk_count equals
    what _scan_qdrant_parity_items would count, without a second scan.
    Returns None for code collections, which verify per project instead.
    """
    if verification.collection_type == "code":
        return None
    items = {}
    for file in verification.files:
        points = file.actual_chunks + file.duplicate_chunk_count
        if points:
            items[file.file_path] = {"qdrant_chunks": points}
    return items


def _scan_meili_parity_items(
    meili, name: str, collection_type: str | None, *, ids_only: bool = False
) -> dict:
    """Group MeiliSearch documents per item and count them.

    Uses get_documents() instead of search() to avoid the maxTotalHits cap.
    With ids_only, fetches just the grouping fields in large pages: enough
    for a parity comparison and many times faster on large indexes.
    """
    batch_size = 10000 if ids_only else 1000

    # Determine fields to fetch based on collection type
    if ids_only:
        doc_fields = (
            ["git_project_identifier"] if collection_type == "code" else ["file_path", "filename"]
        )
    elif collection_type == "code":
        doc_fields = [
            "file_path",
            "filename",
            "language",
            "programming_language",
            "git_project_identifier",
            "git_project_name",
            "git_branch",
        ]
    else:
        doc_fields = ["file_path", "filename", "language", "programming_language"]

    meili_items = {}
    for doc in MeiliTextBackend(meili).iter_documents(name, doc_fields, batch_size=batch_size):
        # For code collections, group by git_project_identifier
        # For other types, group by file_path
        if collection_type == "code":
            item_id = doc.get("git_project_identifier")
        else:
            item_id = doc.get("file_path") or doc.get("filename")

        if item_id and item_id not in meili_items:
            meili_items[item_id] = {
                "file_path": doc.get("file_path"),
                "filename": doc.get("filename"),
                "language": doc.get("language") or doc.get("programming_language"),
                "git_project_identifier": doc.get("git_project_identifier"),
                "git_project_name": doc.get("git_project_name"),
                "git_branch": doc.get("git_branch"),
                "meili_chunks": 1,
            }
        elif item_id:
            meili_items[item_id]["meili_chunks"] += 1

    return meili_items


def _classify_parity_item(qdrant_chunks: int, meili_chunks: int) -> str:
    """Return an item's parity status from its per-store chunk counts."""
    if qdrant_chunks > 0 and meili_chunks > 0:
        return "synced" if qdrant_chunks == meili_chunks else "mismatch"
    if qdrant_chunks > 0:
        return "qdrant_only"
    return "meili_only"


def _summarize_parity_items(items: list[dict], statuses=PARITY_ITEM_STATUSES) -> dict:
    """Count items per parity status."""
    return {status: sum(1 for item in items if item["status"] == status) for status in statuses}


def _compare_parity_items(qdrant_items: dict, meili_items: dict) -> dict:
    """Compare per-item chunk counts across both stores.

    Returns the corpus parity status, a per-status summary, and the items
    that are not synced.
    """
    items = []
    for item_id in sorted(set(qdrant_items) | set(meili_items)):
        q_chunks = qdrant_items.get(item_id, {}).get("qdrant_chunks", 0)
        m_chunks = meili_items.get(item_id, {}).get("meili_chunks", 0)
        items.append(
            {
                "id": item_id,
                "qdrant_chunks": q_chunks,
                "meili_chunks": m_chunks,
                "status": _classify_parity_item(q_chunks, m_chunks),
            }
        )

    summary = _summarize_parity_items(items)
    unsynced = [item for item in items if item["status"] != "synced"]
    return {
        "status": "mismatch" if unsynced else "synced",
        "summary": summary,
        "items": unsynced,
    }


def corpus_items_command(name: str, output_json: bool):
    """List all indexed items in a corpus with parity status.

    Shows items from both Qdrant collection and MeiliSearch index,
    with chunk counts from each system for comparison.

    Args:
        name: Corpus name (used for both collection and index)
        output_json: If True, output JSON format
    """
    from rich.table import Table
    from ..indexing.collection_metadata import get_collection_metadata

    # Start interaction logging (RDR-018)
    interaction_logger.start("corpus", "items", corpus=name)

    qdrant_items = {}
    meili_items = {}
    collection_type = None
    text_backend = DEFAULT_TEXT_BACKEND
    coverage = None
    zero_token_items = {}
    errors = []

    try:
        # Gather Qdrant collection items
        try:
            qdrant = create_qdrant_client()
            metadata = get_collection_metadata(qdrant, name)
            collection_type = metadata.get("collection_type")
            text_backend = text_backend_of(metadata)
            qdrant_items = _scan_qdrant_parity_items(qdrant, name, collection_type)
            if text_backend == "qdrant":
                # Coverage replaces parity; MeiliSearch is never read (RDR-027 Q8).
                text_index = QdrantTextBackend(qdrant, {name: collection_type})
                coverage = text_index.coverage(name, files=False)
                zero_token_items = text_index.zero_token_counts(
                    name, "git_project_identifier" if collection_type == "code" else "file_path"
                )
        except Exception as e:
            errors.append(f"Qdrant: {e}")

        # Gather MeiliSearch index items
        if text_backend != "qdrant":
            try:
                meili = create_meili_client()
                if not meili.health_check():
                    errors.append("MeiliSearch: Server not available")
                elif not meili.index_exists(name):
                    errors.append(f"MeiliSearch: Index '{name}' not found")
                else:
                    meili_items = _scan_meili_parity_items(meili, name, collection_type)
            except Exception as e:
                if "not available" not in str(e).lower():
                    errors.append(f"MeiliSearch: {e}")

        # If neither system has the corpus, check if it's a data issue or missing corpus
        if not qdrant_items and not meili_items:
            # Check if corpus exists but has data quality issues
            corpus_exists = False
            try:
                qdrant = create_qdrant_client()
                col_info = qdrant.get_collection(name)
                if col_info.points_count > 0:
                    corpus_exists = True
            except Exception:
                pass

            if not corpus_exists and text_backend != "qdrant":
                try:
                    meili = create_meili_client()
                    if meili.index_exists(name):
                        stats = meili.get_index_stats(name)
                        if stats.get("numberOfDocuments", 0) > 0:
                            corpus_exists = True
                except Exception:
                    pass

            if corpus_exists:
                # Corpus exists but items couldn't be grouped
                if collection_type == "code":
                    errors.append("Code chunks missing git_project_identifier field for grouping")
                else:
                    errors.append("Chunks missing file_path field for grouping")
            else:
                raise ResourceNotFoundError(
                    f"Corpus '{name}' not found. "
                    f"Create with: arc corpus create {name} --type <pdf|code|markdown>"
                )

        # Merge items from both systems
        all_item_ids = set(qdrant_items.keys()) | set(meili_items.keys())
        merged_items = []

        for item_id in all_item_ids:
            q_item = qdrant_items.get(item_id, {})
            m_item = meili_items.get(item_id, {})

            # Determine parity status for this item
            q_chunks = q_item.get("qdrant_chunks", 0)
            m_chunks = m_item.get("meili_chunks", 0)

            if text_backend == "qdrant":
                # The text index is not countable per item: an item shares
                # its corpus's coverage status.
                zero_token = zero_token_items.get(item_id, 0)
                merged = {
                    "id": item_id,
                    "qdrant_chunks": q_chunks,
                    "searchable_chunks": q_chunks - zero_token,
                    "zero_token_chunks": zero_token,
                    "status": coverage.status if coverage is not None else "unknown",
                }
            else:
                # Merge metadata from both sources
                merged = {
                    "id": item_id,
                    "qdrant_chunks": q_chunks,
                    "meili_chunks": m_chunks,
                    "status": _classify_parity_item(q_chunks, m_chunks),
                }

            # Add type-specific fields (prefer Qdrant, fall back to MeiliSearch)
            if collection_type == "code":
                merged["git_project_name"] = q_item.get("git_project_name") or m_item.get(
                    "git_project_name"
                )
                merged["git_branch"] = q_item.get("git_branch") or m_item.get("git_branch")
                merged["git_commit_hash"] = q_item.get("git_commit_hash")
            else:
                merged["filename"] = q_item.get("filename") or m_item.get("filename")
                merged["file_size"] = q_item.get("file_size")

            merged_items.append(merged)

        # Sort items
        if collection_type == "code":
            merged_items.sort(key=lambda x: x.get("git_project_name") or x["id"])
        else:
            merged_items.sort(key=lambda x: x.get("filename") or x["id"])

        # Calculate summary stats
        statuses = COVERAGE_ITEM_STATUSES if text_backend == "qdrant" else PARITY_ITEM_STATUSES
        summary = _summarize_parity_items(merged_items, statuses)

        # Output
        if output_json:
            data = {
                "corpus": name,
                "type": collection_type,
                "text_backend": text_backend,
                "item_count": len(merged_items),
                "summary": summary,
                "items": merged_items,
                "errors": errors if errors else None,
            }
            if text_backend == "qdrant":
                data["coverage"] = coverage.to_data() if coverage is not None else None
            print_json("success", f"Found {len(merged_items)} items in corpus '{name}'", data)
        else:
            # Header
            console.print(f"\n[bold cyan]Corpus: {name}[/bold cyan]")
            type_str = (
                f"[bold]{collection_type}[/bold]" if collection_type else "[yellow]unknown[/yellow]"
            )
            console.print(f"Type: {type_str}")
            console.print(f"Items: {len(merged_items)}")

            # Summary
            summary_parts = [
                f"[{_ITEM_STATUS_COLOR.get(status, 'yellow')}]{count} {status}"
                f"[/{_ITEM_STATUS_COLOR.get(status, 'yellow')}]"
                for status, count in summary.items()
                if count > 0
            ]
            if summary_parts:
                label = "Coverage" if text_backend == "qdrant" else "Parity"
                console.print(f"{label}: {', '.join(summary_parts)}\n")
            # The second count column: MeiliSearch's chunks, or a Qdrant text
            # corpus's searchable chunks.
            if text_backend == "qdrant":
                second_column, second_key = "S", "searchable_chunks"
            else:
                second_column, second_key = "M", "meili_chunks"

            if not merged_items:
                print_info("No items found")
            else:
                if collection_type == "code":
                    table = Table(title="Indexed Repositories")
                    table.add_column("Project", style="cyan")
                    table.add_column("Branch", style="green")
                    table.add_column("Commit", style="dim")
                    table.add_column("Q", style="yellow", justify="right")
                    table.add_column(second_column, style="yellow", justify="right")
                    table.add_column("Status", style="magenta")

                    for item in merged_items:
                        # Format status with color
                        status = item["status"]
                        if status == "synced":
                            status_str = "[green]synced[/green]"
                        elif status == "mismatch":
                            status_str = "[red]mismatch[/red]"
                        else:
                            status_str = f"[yellow]{status}[/yellow]"

                        table.add_row(
                            item.get("git_project_name") or item["id"],
                            item.get("git_branch") or "-",
                            item["git_commit_hash"][:12] if item.get("git_commit_hash") else "-",
                            str(item["qdrant_chunks"]),
                            str(item[second_key]),
                            status_str,
                        )
                    console.print(table)
                else:
                    table = Table(title="Indexed Files")
                    table.add_column("File", style="cyan", no_wrap=False)
                    table.add_column("Size", style="dim")
                    table.add_column("Q", style="yellow", justify="right")
                    table.add_column(second_column, style="yellow", justify="right")
                    table.add_column("Status", style="magenta")

                    for item in merged_items:
                        # Format file size
                        size = item.get("file_size", 0)
                        if size:
                            if size > 1024 * 1024:
                                size_str = f"{size / (1024 * 1024):.1f}MB"
                            elif size > 1024:
                                size_str = f"{size / 1024:.1f}KB"
                            else:
                                size_str = f"{size}B"
                        else:
                            size_str = "-"

                        # Format status with color
                        status = item["status"]
                        if status == "synced":
                            status_str = "[green]synced[/green]"
                        elif status == "mismatch":
                            status_str = "[red]mismatch[/red]"
                        else:
                            status_str = f"[yellow]{status}[/yellow]"

                        display_name = item.get("filename") or item["id"]
                        table.add_row(
                            display_name,
                            size_str,
                            str(item["qdrant_chunks"]),
                            str(item[second_key]),
                            status_str,
                        )
                    console.print(table)

            # Errors/warnings
            if errors:
                console.print(f"\n[dim]Warnings: {', '.join(errors)}[/dim]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(result_count=len(merged_items))

    except ResourceNotFoundError:
        interaction_logger.finish(error="corpus not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to list corpus items: {e}", output_json)
        sys.exit(1)


def corpus_verify_command(
    name: str,
    project: str | None,
    verbose: bool,
    output_json: bool,
    *,
    deep: bool = False,
    strict: bool = False,
):
    """Verify corpus health across both Qdrant and MeiliSearch.

    Performs fsck-like integrity checks on both systems and reports:
    - Qdrant collection health (chunk completeness)
    - MeiliSearch index health (document accessibility)
    - Parity status between the two systems

    Args:
        name: Corpus name (used for both collection and index)
        project: Optional project identifier filter (code collections only)
        verbose: Show detailed file-level results
        output_json: If True, output JSON format
        deep: Compare persisted PDF metadata with source files on disk
        strict: Exit 1 when verification reports issues (CI/agent mode)
    """
    from arcaneum.indexing.verify import CollectionVerifier

    # Start interaction logging (RDR-018)
    interaction_logger.start("corpus", "verify", corpus=name, project=project)

    qdrant_result = None
    meili_result = None
    text_backend = DEFAULT_TEXT_BACKEND
    coverage = None
    errors = []

    try:
        # Step 1: Verify Qdrant collection
        if not output_json:
            console.print(f"[dim]Verifying Qdrant collection '{name}'...[/dim]")

        try:
            qdrant = create_qdrant_client()
            verifier = CollectionVerifier(qdrant)
            qdrant_result = verifier.verify_collection(
                name, project_filter=project, verbose=verbose, deep=deep
            )
            text_backend = text_backend_of(get_collection_metadata(qdrant, name))
        except Exception as e:
            errors.append(f"Qdrant: {e}")

        if text_backend == "qdrant":
            # Step 2: the text index's coverage replaces MeiliSearch and
            # parity, and MeiliSearch is never read (RDR-027 Q8).
            if not output_json:
                console.print(f"[dim]Checking the text index of '{name}'...[/dim]")
            try:
                text_index = QdrantTextBackend(qdrant, {name: qdrant_result.collection_type})
                coverage = text_index.coverage(name, files=False)
            except Exception as e:
                errors.append(f"Qdrant text index: {e}")
        else:
            # Step 2: Verify MeiliSearch index
            if not output_json:
                console.print(f"[dim]Verifying MeiliSearch index '{name}'...[/dim]")

            try:
                meili = create_meili_client()

                if not meili.health_check():
                    errors.append("MeiliSearch: Server not available")
                elif not meili.index_exists(name):
                    errors.append(f"MeiliSearch: Index '{name}' not found")
                else:
                    # Perform health checks on MeiliSearch
                    stats = meili.get_index_stats(name)
                    settings = meili.get_index_settings(name)

                    meili_issues = []
                    meili_warnings = []

                    doc_count = stats.get("numberOfDocuments", 0)
                    is_indexing = stats.get("isIndexing", False)

                    if is_indexing:
                        meili_warnings.append("Index is currently processing documents")

                    # Check searchable/filterable attributes
                    searchable = settings.get("searchableAttributes", [])
                    if not searchable or searchable == ["*"]:
                        meili_warnings.append("Searchable attributes not explicitly defined")

                    filterable = settings.get("filterableAttributes", [])
                    if not filterable:
                        meili_warnings.append("No filterable attributes defined")

                    # Try sample retrieval
                    sample_accessible = False
                    try:
                        sample_results = meili.search(name, "", limit=1)
                        if sample_results.get("hits"):
                            sample_accessible = True
                    except Exception as e:
                        meili_issues.append(f"Failed to retrieve sample document: {e}")

                    meili_result = {
                        "is_healthy": len(meili_issues) == 0,
                        "document_count": doc_count,
                        "is_indexing": is_indexing,
                        "sample_accessible": sample_accessible,
                        "issues": meili_issues,
                        "warnings": meili_warnings,
                    }
            except Exception as e:
                if "not available" not in str(e).lower():
                    errors.append(f"MeiliSearch: {e}")

        # If neither system has the corpus, it's an error
        if qdrant_result is None and meili_result is None:
            raise ResourceNotFoundError(
                f"Corpus '{name}' not found. "
                f"Create with: arc corpus create {name} --type <pdf|code|markdown>"
            )

        # Compute overall health and parity
        overall_healthy = True
        parity_status = "unknown"
        parity = None

        if text_backend == "qdrant":
            # Unhealthy on a coverage gap or a missing index; zero-token
            # chunks are a chunking finding, not a gap.
            overall_healthy = qdrant_result.is_healthy
            if coverage is not None:
                parity = coverage.to_data()
                parity_status = coverage.status
                overall_healthy = overall_healthy and coverage.complete
        elif qdrant_result and meili_result:
            qdrant_healthy = qdrant_result.is_healthy
            meili_healthy = meili_result["is_healthy"]
            overall_healthy = qdrant_healthy and meili_healthy

            # Same per-item chunk comparison as 'arc corpus items'
            if not output_json:
                console.print("[dim]Comparing items across both systems...[/dim]")
            try:
                collection_type = qdrant_result.collection_type
                qdrant_items = _qdrant_parity_items_from_verification(qdrant_result)
                if qdrant_items is None:
                    qdrant_items = _scan_qdrant_parity_items(qdrant, name, collection_type)
                meili_items = _scan_meili_parity_items(meili, name, collection_type, ids_only=True)
                if project:
                    qdrant_items = {k: v for k, v in qdrant_items.items() if k == project}
                    meili_items = {k: v for k, v in meili_items.items() if k == project}
                parity = _compare_parity_items(qdrant_items, meili_items)
            except Exception as e:
                reason = f"Parity scan failed: {e}"
                parity = {"status": "unknown", "reason": reason}
                errors.append(reason)

            parity_status = parity["status"]
            if parity_status == "mismatch":
                overall_healthy = False
        elif qdrant_result:
            overall_healthy = qdrant_result.is_healthy
            parity_status = "qdrant_only"
        elif meili_result:
            overall_healthy = meili_result["is_healthy"]
            parity_status = "meili_only"
        overall_healthy = overall_healthy and not errors

        quality_summary = None
        if qdrant_result and qdrant_result.collection_type == "pdf":
            quality_counts = {
                "garbled_files": getattr(qdrant_result, "garbled_items", 0),
                "dropout_files": getattr(qdrant_result, "dropout_items", 0),
                "dropped_chunks": getattr(qdrant_result, "dropped_chunks", 0),
                "sub_floor_chunks": getattr(qdrant_result, "sub_floor_chunks", 0),
                "duplicate_source_groups": getattr(qdrant_result, "duplicate_source_groups", 0),
                "stale_policy_files": getattr(qdrant_result, "stale_policy_items", 0),
                "stale_extraction_files": getattr(qdrant_result, "stale_extraction_items", 0),
                "incomplete_files": qdrant_result.incomplete_items,
                "quality_manifest_gaps": getattr(qdrant_result, "quality_manifest_gaps", 0),
            }
            quality_summary = {
                **quality_counts,
                "warnings": [
                    {"code": code, "count": count}
                    for code, count in quality_counts.items()
                    if count
                ],
                "mode": "deep" if deep else "persisted",
            }

        # Output results
        if output_json:
            if coverage is not None:
                text_chunks = coverage.indexed or 0
            else:
                text_chunks = meili_result["document_count"] if meili_result else None
            data = {
                "corpus": name,
                "text_backend": text_backend,
                "text_chunks": text_chunks,
                "overall_healthy": overall_healthy,
                "parity_status": parity_status,
                "parity": parity,
                "qdrant": None,
                "meilisearch": None,
                "errors": errors if errors else None,
                "service_errors": errors,
                "quality": quality_summary,
            }

            if qdrant_result:
                data["qdrant"] = {
                    "collection": qdrant_result.collection_name,
                    "type": qdrant_result.collection_type,
                    "is_healthy": qdrant_result.is_healthy,
                    "total_points": qdrant_result.total_points,
                    "total_items": qdrant_result.total_items,
                    "complete_items": qdrant_result.complete_items,
                    "incomplete_items": qdrant_result.incomplete_items,
                    "duplicate_items": qdrant_result.duplicate_items,
                    "schema_version": qdrant_result.schema_version,
                    "app_version": qdrant_result.app_version,
                    "errors": qdrant_result.errors,
                }
                if qdrant_result.collection_type != "code":
                    data["qdrant"]["files"] = [
                        {
                            "file_path": file.file_path,
                            "is_complete": file.is_complete,
                            "has_duplicate_chunks": file.has_duplicate_chunks,
                            "duplicate_chunk_count": file.duplicate_chunk_count,
                            "quality_manifest": file.quality_manifest,
                        }
                        for file in qdrant_result.files
                        # Duplicated files are reported even without a quality
                        # manifest; they are otherwise invisible in JSON output.
                        if getattr(file, "quality_manifest", None) or file.has_duplicate_chunks
                    ]

            if meili_result:
                data["meilisearch"] = meili_result

            status = "success" if overall_healthy else "warning"
            msg = (
                f"Corpus '{name}' is healthy" if overall_healthy else f"Corpus '{name}' has issues"
            )
            print_json(status, msg, data)
        else:
            # Human-readable output
            console.print(f"\n[bold cyan]Corpus: {name}[/bold cyan]")

            # Overall status
            if overall_healthy:
                console.print("[green]Overall Status: Healthy[/green]")
            else:
                console.print("[yellow]Overall Status: Issues detected[/yellow]")

            if quality_summary is not None:
                console.print("\n[bold]PDF Quality:[/bold]")
                console.print(
                    "  "
                    + ", ".join(
                        f"{label.replace('_', ' ')}: {count}"
                        for label, count in quality_summary.items()
                        if label not in {"warnings", "mode"}
                    )
                )
                console.print(f"  Mode: {quality_summary['mode']}")

            # Qdrant section
            console.print("\n[bold]Qdrant Collection:[/bold]")
            if qdrant_result:
                if qdrant_result.errors:
                    console.print("  [yellow]Verification errors:[/yellow]")
                    for error in qdrant_result.errors:
                        console.print(f"    [yellow]- {error}[/yellow]")
                if qdrant_result.is_healthy:
                    console.print("  [green]Status: Healthy[/green]")
                    console.print(
                        f"  Items: {qdrant_result.total_items} ({qdrant_result.complete_items} complete)"
                    )
                elif qdrant_result.errors and qdrant_result.incomplete_items == 0:
                    console.print("  [yellow]Status: Verification errors[/yellow]")
                    console.print(
                        f"  Items: {qdrant_result.total_items} ({qdrant_result.complete_items} complete)"
                    )
                else:
                    console.print(
                        f"  [yellow]Status: {qdrant_result.incomplete_items} incomplete items[/yellow]"
                    )
                    console.print(
                        f"  Items: {qdrant_result.total_items} ({qdrant_result.complete_items} complete, {qdrant_result.incomplete_items} incomplete)"
                    )

                    # Show incomplete items summary
                    if verbose and qdrant_result.collection_type == "code":
                        for proj in qdrant_result.projects:
                            if not proj.is_complete:
                                console.print(
                                    f"  [dim]• {proj.project_name}: {proj.completion_percentage:.1f}% complete[/dim]"
                                )
                    elif verbose:
                        for file in qdrant_result.files[:5]:
                            if not file.is_complete:
                                # A duplicated file has every chunk present, so
                                # a completion percentage would read 100% and
                                # hide the reason it is unhealthy.
                                if file.has_duplicate_chunks:
                                    detail = f"{file.duplicate_chunk_count} duplicate chunks"
                                else:
                                    detail = f"{file.completion_percentage:.1f}% complete"
                                console.print(f"  [dim]• {file.file_path}: {detail}[/dim]")
                        if qdrant_result.incomplete_items > 5:
                            console.print(
                                f"  [dim]... and {qdrant_result.incomplete_items - 5} more[/dim]"
                            )
            else:
                console.print("  [yellow]Not found or not accessible[/yellow]")

            # Text section: a Qdrant text corpus's coverage stands in for the
            # MeiliSearch and parity sections.
            if text_backend == "qdrant":
                _print_verify_coverage(name, coverage, verbose)
            else:
                _print_verify_meili_and_parity(name, meili_result, parity, parity_status, verbose)

            # Errors
            if errors:
                console.print("\n[red]Errors:[/red]")
                for error in errors:
                    console.print(f"  [red]• {error}[/red]")

            # Summary
            if overall_healthy and not errors:
                console.print("\n[green]All checks passed[/green]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            overall_healthy=overall_healthy,
            qdrant_healthy=qdrant_result.is_healthy if qdrant_result else None,
            meili_healthy=meili_result["is_healthy"] if meili_result else None,
        )
        if strict and not overall_healthy:
            sys.exit(1)

    except ResourceNotFoundError:
        interaction_logger.finish(error="corpus not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to verify corpus: {e}", output_json)
        sys.exit(1)


def _print_verify_coverage(name: str, coverage, verbose: bool) -> None:
    """Verify's text coverage section for a Qdrant text corpus (RDR-027 Q8)."""
    console.print("\n[bold]Text Coverage:[/bold]")
    if coverage is None:
        console.print("  [yellow]Status: unknown[/yellow]")
        return
    color = "green" if coverage.complete else "red"
    console.print(f"  [{color}]Status: {coverage.status}[/{color}]")
    if coverage.indexed is not None:
        console.print(
            f"  Text index: {coverage.indexed:,} of {coverage.searchable_chunks:,} searchable chunks"
        )
    if coverage.zero_token_chunks:
        console.print(
            f"  Zero-token chunks: {coverage.zero_token_chunks:,} "
            "(yield no token; a chunking finding, not a gap)"
        )
        shown = coverage.flagged if verbose else coverage.flagged[:5]
        for path, index in shown:
            console.print(f"  [dim]• {path} chunk {index}[/dim]")
    if not coverage.complete:
        console.print(f"  [dim]Fix with: arc corpus repair {name} (preview with --dry-run)[/dim]")


def _print_verify_meili_and_parity(name, meili_result, parity, parity_status, verbose) -> None:
    """Verify's MeiliSearch and parity sections for a MeiliSearch corpus."""
    # MeiliSearch section
    console.print("\n[bold]MeiliSearch Index:[/bold]")
    if meili_result:
        if meili_result["is_healthy"]:
            console.print("  [green]Status: Healthy[/green]")
        else:
            console.print("  [yellow]Status: Issues detected[/yellow]")
            for issue in meili_result["issues"]:
                console.print(f"    [red]• {issue}[/red]")

        console.print(f"  Documents: {meili_result['document_count']:,}")
        if meili_result["is_indexing"]:
            console.print("  [yellow]Currently indexing...[/yellow]")
        if meili_result["sample_accessible"]:
            console.print("  [green]Sample retrieval: OK[/green]")

        if meili_result["warnings"]:
            console.print("  Warnings:")
            for warning in meili_result["warnings"]:
                console.print(f"    [dim]• {warning}[/dim]")
    else:
        console.print("  [yellow]Not found or not accessible[/yellow]")

    # Parity section
    console.print("\n[bold]Parity:[/bold]")
    if parity is None:
        console.print(f"  [yellow]Status: {parity_status}[/yellow]")
    elif parity_status == "unknown":
        console.print(f"  [yellow]Status: unknown ({parity['reason']})[/yellow]")
    else:
        summary_str = ", ".join(
            f"{count} {status}" for status, count in parity["summary"].items() if count
        )
        color = "green" if parity_status == "synced" else "red"
        console.print(f"  [{color}]Status: {parity_status}[/{color}]")
        console.print(f"  Items: {summary_str or '0'}")
        shown = parity["items"] if verbose else parity["items"][:5]
        for item in shown:
            console.print(
                f"  [dim]• {item['id']}: {item['status']} "
                f"(qdrant {item['qdrant_chunks']}, meili {item['meili_chunks']})[/dim]"
            )
        if len(parity["items"]) > len(shown):
            console.print(
                f"  [dim]... and {len(parity['items']) - len(shown)} more "
                f"(see: arc corpus items {name})[/dim]"
            )
        if parity_status == "mismatch":
            console.print(
                f"  [dim]Fix with: arc corpus repair {name} (preview with --dry-run)[/dim]"
            )
