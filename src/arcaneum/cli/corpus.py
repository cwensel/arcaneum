"""Corpus management commands for dual indexing (RDR-009).

This module implements the 'corpus create' command that creates both a Qdrant
collection and a MeiliSearch index in a single operation.
"""

import sys
from typing import Dict, Any

import click
from rich.console import Console
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

from ..cli.output import print_json, print_error, print_info, print_success
from ..cli.utils import create_qdrant_client
from ..cli.interaction_logger import interaction_logger
from ..cli.errors import InvalidArgumentError, ResourceNotFoundError
from ..embeddings.client import EMBEDDING_MODELS
from ..fulltext.indexes import get_index_settings
from ..indexing.collection_metadata import set_collection_metadata, get_collection_metadata, CollectionType

console = Console()


def get_meili_client():
    """Get MeiliSearch client from environment or auto-generated key."""
    import os
    from ..paths import get_meilisearch_api_key
    from ..fulltext.client import FullTextClient

    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    api_key = get_meilisearch_api_key()
    return FullTextClient(url, api_key)


def get_model_dimensions(model_name: str) -> int:
    """Get vector dimensions for an embedding model.

    Args:
        model_name: Model identifier

    Returns:
        Number of dimensions

    Raises:
        ValueError: If model is unknown
    """
    if model_name not in EMBEDDING_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(EMBEDDING_MODELS.keys())}"
        )
    return EMBEDDING_MODELS[model_name]["dimensions"]


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


def list_corpora_command(verbose: bool, output_json: bool):
    """List all corpora with parity status.

    A corpus is a paired Qdrant collection + MeiliSearch index.
    This command discovers all such pairs and shows their sync status.

    Args:
        verbose: Show detailed information (models, chunks)
        output_json: Output as JSON
    """
    from rich.table import Table

    interaction_logger.start("corpus", "list")

    try:
        # Gather Qdrant collections
        qdrant_collections = {}
        try:
            qdrant = create_qdrant_client()
            collections = qdrant.get_collections()
            for col in collections.collections:
                metadata = get_collection_metadata(qdrant, col.name)
                col_info = qdrant.get_collection(col.name)
                # Subtract 1 for the metadata point
                chunk_count = col_info.points_count - 1 if col_info.points_count > 0 else 0
                qdrant_collections[col.name] = {
                    "type": metadata.get("collection_type"),
                    "model": metadata.get("model"),
                    "chunks": chunk_count,
                }
        except Exception as e:
            if not output_json:
                print_error(f"Could not connect to Qdrant: {e}")

        # Gather MeiliSearch indexes
        meili_indexes = {}
        try:
            meili = get_meili_client()
            if meili.health_check():
                indexes = meili.list_indexes()
                for idx in indexes:
                    name = idx['uid']
                    try:
                        stats = meili.get_index_stats(name)
                        meili_indexes[name] = {
                            "chunks": stats.get('numberOfDocuments', 0),
                        }
                    except Exception:
                        meili_indexes[name] = {"chunks": 0}
        except Exception as e:
            if not output_json:
                print_error(f"Could not connect to MeiliSearch: {e}")

        # Build unified corpus list
        all_names = set(qdrant_collections.keys()) | set(meili_indexes.keys())
        corpora = []

        for name in sorted(all_names):
            q_info = qdrant_collections.get(name)
            m_info = meili_indexes.get(name)

            # Determine parity status
            if q_info and m_info:
                status = "synced"
            elif q_info:
                status = "qdrant_only"
            else:
                status = "meili_only"

            corpus_type = q_info.get("type") if q_info else None
            model = q_info.get("model") if q_info else None
            q_chunks = q_info.get("chunks", 0) if q_info else 0
            m_chunks = m_info.get("chunks", 0) if m_info else 0

            corpora.append({
                "name": name,
                "type": corpus_type,
                "model": model,
                "status": status,
                "qdrant_chunks": q_chunks,
                "meili_chunks": m_chunks,
            })

        # Output
        if output_json:
            print_json("success", f"Found {len(corpora)} corpora", {"corpora": corpora})
        else:
            if not corpora:
                print_info("No corpora found")
                print_info("Create one with: arc corpus create <name> --type <pdf|code|markdown>")
            else:
                table = Table(title="Corpora")
                table.add_column("Name", style="cyan")
                table.add_column("Type", style="blue")
                table.add_column("Status", style="green")
                if verbose:
                    table.add_column("Model", style="magenta")
                    table.add_column("Q Chunks", style="yellow")
                    table.add_column("M Chunks", style="yellow")

                for c in corpora:
                    # Format status with color
                    status = c["status"]
                    if status == "synced":
                        status_str = "[green]synced[/green]"
                    elif status == "qdrant_only":
                        status_str = "[yellow]qdrant_only[/yellow]"
                    else:
                        status_str = "[yellow]meili_only[/yellow]"

                    row = [
                        c["name"],
                        c["type"] or "—",
                        status_str,
                    ]
                    if verbose:
                        row.extend([
                            c["model"] or "—",
                            str(c["qdrant_chunks"]),
                            str(c["meili_chunks"]),
                        ])
                    table.add_row(*row)

                console.print(table)

        interaction_logger.finish(result_count=len(corpora))

    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to list corpora: {e}", output_json)
        sys.exit(1)


def create_corpus_command(
    name: str,
    corpus_type: str,
    models: str,
    output_json: bool
):
    """Create both Qdrant collection and MeiliSearch index.

    This implements the first command of the 2-command workflow:
    1. corpus create (this command) - creates both systems
    2. corpus sync - indexes documents to both systems

    Args:
        name: Corpus name (used for both collection and index)
        corpus_type: Type of corpus (pdf, code, markdown)
        models: Comma-separated list of embedding models
        output_json: If True, output JSON format
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "corpus", "create",
        corpus=name,
        corpus_type=corpus_type,
        models=models,
    )

    try:
        if not output_json:
            print_info(f"Creating corpus '{name}'")
            print_info(f"Type: {corpus_type}, Models: {models}")

        # Parse and validate models
        model_list = [m.strip() for m in models.split(',')]
        for model in model_list:
            if model not in EMBEDDING_MODELS:
                error_msg = f"Unknown model: {model}. Available: {list(EMBEDDING_MODELS.keys())}"
                raise InvalidArgumentError(error_msg)

        # Build vectors config for Qdrant
        vectors_config = {}
        for model in model_list:
            vectors_config[model] = VectorParams(
                size=get_model_dimensions(model),
                distance=Distance.COSINE,
            )

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
                model=models
            )

            if not output_json:
                console.print(f"[green]✅ Qdrant collection '{name}' created[/green]")

        except Exception as e:
            error_str = str(e).lower()
            if "already exists" in error_str:
                raise InvalidArgumentError(f"Qdrant collection '{name}' already exists")
            raise

        # Step 2: Create MeiliSearch index
        if not output_json:
            print_info("Step 2/2: Creating MeiliSearch index...")

        meili = get_meili_client()

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
            meili.create_index(name, primary_key='id', settings=settings)

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
                print_info("You may need to create the MeiliSearch index manually or delete the Qdrant collection")
            raise

        # Success output
        data = {
            "corpus": name,
            "type": corpus_type,
            "models": model_list,
            "vectors": {m: get_model_dimensions(m) for m in model_list},
            "qdrant_collection": name,
            "meilisearch_index": name,
        }

        if output_json:
            print_json(
                "success",
                f"Corpus '{name}' created with {len(model_list)} models",
                data=data
            )
        else:
            console.print(f"\n[green]✅ Corpus '{name}' ready for indexing![/green]")
            for model in model_list:
                dims = get_model_dimensions(model)
                console.print(f"  • {model}: {dims}D")
            console.print(f"\n[dim]Next: arc corpus sync /path/to/files --corpus {name}[/dim]")

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to create corpus: {e}", output_json)
        sys.exit(1)


def _get_qdrant_item_count(client, collection_name: str, collection_type: str) -> int:
    """Get count of unique items (files or repos) in a Qdrant collection.

    Scrolls through collection to count unique file_path or git_project_identifier values.

    Args:
        client: Qdrant client
        collection_name: Collection name
        collection_type: Type of collection (pdf, code, markdown)

    Returns:
        Count of unique items
    """
    unique_items = set()
    offset = None

    # Determine which field to use based on collection type
    if collection_type == "code":
        id_field = "git_project_identifier"
    else:
        id_field = "file_path"

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=[id_field],
            with_vectors=False
        )

        if not points:
            break

        for point in points:
            if point.payload:
                item_id = point.payload.get(id_field)
                if item_id:
                    unique_items.add(item_id)

        if offset is None:
            break

    return len(unique_items)


def _get_meili_item_count(client, index_name: str) -> int:
    """Get count of unique file_path values in a MeiliSearch index.

    Args:
        client: MeiliSearch FullTextClient
        index_name: Index name

    Returns:
        Count of unique file paths
    """
    file_paths = client.get_all_file_paths(index_name)
    return len(file_paths)


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
            if hasattr(col_info.config.params, 'vectors') and isinstance(col_info.config.params.vectors, dict):
                for vector_name, vector_params in col_info.config.params.vectors.items():
                    vectors[vector_name] = {
                        "size": vector_params.size,
                        "distance": str(vector_params.distance),
                    }

            # Subtract 1 for the metadata point
            chunk_count = col_info.points_count - 1 if col_info.points_count > 0 else 0

            # Get unique item count (files or repos)
            item_count = _get_qdrant_item_count(qdrant_client, name, collection_type)

            qdrant_info = {
                "name": name,
                "status": str(col_info.status),
                "item_count": item_count,
                "chunk_count": chunk_count,
                "type": collection_type,
                "model": metadata.get("model"),
                "vectors": vectors,
                "hnsw_config": {
                    "m": col_info.config.hnsw_config.m,
                    "ef_construct": col_info.config.hnsw_config.ef_construct,
                },
            }
        except Exception as e:
            errors.append(f"Qdrant: {e}")

        # Gather MeiliSearch index info
        meili = get_meili_client()
        try:
            if not meili.health_check():
                errors.append("MeiliSearch: Server not available")
            elif not meili.index_exists(name):
                errors.append(f"MeiliSearch: Index '{name}' not found")
            else:
                stats = meili.get_index_stats(name)

                # Get unique item count
                item_count = _get_meili_item_count(meili, name)
                chunk_count = stats.get('numberOfDocuments', 0)

                meili_info = {
                    "name": name,
                    "item_count": item_count,
                    "chunk_count": chunk_count,
                    "is_indexing": stats.get('isIndexing', False),
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
        if qdrant_info and meili_info:
            qdrant_items = qdrant_info["item_count"]
            meili_items = meili_info["item_count"]
            diff = abs(qdrant_items - meili_items)

            if diff == 0:
                parity = {"status": "synced", "qdrant_items": qdrant_items, "meili_items": meili_items}
            else:
                parity = {"status": "out_of_sync", "qdrant_items": qdrant_items, "meili_items": meili_items, "difference": diff}
        elif qdrant_info:
            parity = {"status": "qdrant_only", "qdrant_items": qdrant_info["item_count"]}
        elif meili_info:
            parity = {"status": "meili_only", "meili_items": meili_info["item_count"]}

        # Determine item label based on type
        collection_type = qdrant_info.get("type") if qdrant_info else "unknown"
        item_label = "repos" if collection_type == "code" else "files"

        # Output
        if output_json:
            data = {
                "corpus": name,
                "type": collection_type,
                "item_label": item_label,
                "parity": parity,
                "qdrant": qdrant_info,
                "meilisearch": meili_info,
                "errors": errors if errors else None,
            }
            print_json("success", f"Corpus '{name}' information", data=data)
        else:
            # Header
            console.print(f"\n[bold cyan]Corpus: {name}[/bold cyan]")
            console.print(f"Type: [bold]{collection_type}[/bold]")

            # Parity status
            if parity["status"] == "synced":
                console.print(f"Status: [green]synced[/green] ({parity['qdrant_items']:,} {item_label})")
            elif parity["status"] == "out_of_sync":
                console.print(f"Status: [red]out of sync[/red] (Qdrant: {parity['qdrant_items']:,}, MeiliSearch: {parity['meili_items']:,} {item_label})")
            elif parity["status"] == "qdrant_only":
                console.print(f"Status: [yellow]Qdrant only[/yellow] ({parity['qdrant_items']:,} {item_label})")
            elif parity["status"] == "meili_only":
                console.print(f"Status: [yellow]MeiliSearch only[/yellow] ({parity['meili_items']:,} {item_label})")

            # Qdrant section
            if qdrant_info:
                console.print(f"\n[bold]Qdrant Collection:[/bold]")
                console.print(f"  {item_label.capitalize()}: {qdrant_info['item_count']:,}")
                console.print(f"  Chunks: {qdrant_info['chunk_count']:,}")
                console.print(f"  Status: {qdrant_info['status']}")
                if qdrant_info["vectors"]:
                    console.print(f"  Models:")
                    for model_name, vec_info in qdrant_info["vectors"].items():
                        console.print(f"    - {model_name}: {vec_info['size']}D ({vec_info['distance']})")
            else:
                console.print(f"\n[yellow]Qdrant Collection: not found[/yellow]")

            # MeiliSearch section
            if meili_info:
                console.print(f"\n[bold]MeiliSearch Index:[/bold]")
                console.print(f"  {item_label.capitalize()}: {meili_info['item_count']:,}")
                console.print(f"  Chunks: {meili_info['chunk_count']:,}")
                if meili_info['is_indexing']:
                    console.print(f"  Status: [yellow]indexing[/yellow]")
            else:
                console.print(f"\n[yellow]MeiliSearch Index: not found[/yellow]")

            # Errors/warnings
            if errors:
                console.print(f"\n[dim]Warnings: {', '.join(errors)}[/dim]")

            # Usage hint
            console.print(f"\n[dim]Search with:[/dim]")
            console.print(f"[dim]  arc search semantic \"query\" --collection {name}[/dim]")
            console.print(f"[dim]  arc search text \"query\" --index {name}[/dim]")

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except ResourceNotFoundError:
        interaction_logger.finish(error="corpus not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to get corpus info: {e}", output_json)
        sys.exit(1)
