"""Collection management CLI commands (RDR-003 with RDR-006 enhancements)."""

import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, HnswConfigDiff

from arcaneum.config import load_config, ArcaneumConfig, DEFAULT_MODELS
from arcaneum.embeddings.client import EMBEDDING_MODELS
from arcaneum.indexing.collection_metadata import (
    set_collection_metadata,
    get_collection_metadata,
    get_collection_type,
    CollectionType,
)
from arcaneum.cli.errors import InvalidArgumentError, ResourceNotFoundError
from arcaneum.cli.interaction_logger import interaction_logger
from arcaneum.cli.output import print_json, print_error, print_success
from arcaneum.cli.utils import create_qdrant_client

console = Console()


def get_distance(distance_str: str) -> Distance:
    """Convert distance string to Qdrant Distance enum.

    Args:
        distance_str: Distance metric name (cosine, euclid, dot)

    Returns:
        Distance enum value
    """
    distance_map = {
        "cosine": Distance.COSINE,
        "euclid": Distance.EUCLID,
        "dot": Distance.DOT,
    }
    return distance_map.get(distance_str.lower(), Distance.COSINE)


def create_collection_command(
    name: str,
    model: str,
    hnsw_m: int,
    hnsw_ef: int,
    on_disk: bool,
    output_json: bool,
    collection_type: str = None,
):
    """Create a new Qdrant collection with named vectors.

    Args:
        name: Collection name
        model: Embedding model to use (or comma-separated list). If None, inferred from collection_type.
        hnsw_m: HNSW m parameter
        hnsw_ef: HNSW ef_construct parameter
        on_disk: Store vectors on disk
        output_json: Output as JSON
        collection_type: Type of collection ("pdf", "code", or "markdown")
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "collection", "create",
        collection=name,
        collection_type=collection_type,
        model=model,
    )

    try:
        # Infer model from collection_type if not provided
        if model is None:
            if collection_type is None:
                raise InvalidArgumentError(
                    "Either --model must be specified, or --type must be specified to infer the model. "
                    "Model inference works with: --type pdf (stella), --type code (jina-code-0.5b), --type markdown (stella)"
                )

            # Map collection type to default model
            type_to_model = {
                "pdf": "stella",
                "code": "jina-code-0.5b",  # Updated to SOTA Sept 2025 model (896D, 32K context)
                "markdown": "stella"
            }
            model = type_to_model.get(collection_type)
            if model is None:
                raise InvalidArgumentError(f"Unknown collection type: {collection_type}")

        # Parse models (support comma-separated list)
        model_list = [m.strip() for m in model.split(',')]

        # Validate models
        for m in model_list:
            if m not in EMBEDDING_MODELS:
                error_msg = f"Unknown model: {m}. Available: {list(EMBEDDING_MODELS.keys())}"
                raise InvalidArgumentError(error_msg)

        # Build vectors config
        vectors_config = {}
        for m in model_list:
            model_info = EMBEDDING_MODELS[m]
            vectors_config[m] = VectorParams(
                size=model_info["dimensions"],
                distance=Distance.COSINE,
            )

        # Connect to Qdrant
        client = create_qdrant_client()

        # Validate collection type if provided
        if collection_type:
            CollectionType.validate(collection_type)

        # Create collection
        client.create_collection(
            collection_name=name,
            vectors_config=vectors_config,
            hnsw_config=HnswConfigDiff(m=hnsw_m, ef_construct=hnsw_ef),
            on_disk_payload=on_disk,
        )

        # Set collection metadata (including type)
        if collection_type:
            set_collection_metadata(
                client=client,
                collection_name=name,
                collection_type=collection_type,
                model=model
            )

        # Output success
        data = {
            "collection": name,
            "type": collection_type,
            "models": model_list,
            "vectors": {m: EMBEDDING_MODELS[m]["dimensions"] for m in model_list},
            "hnsw": {"m": hnsw_m, "ef_construct": hnsw_ef},
            "on_disk_payload": on_disk,
        }

        if output_json:
            type_str = f" (type: {collection_type})" if collection_type else ""
            print_json("success", f"Created collection '{name}'{type_str} with {len(model_list)} models", data)
        else:
            type_str = f" (type: {collection_type})" if collection_type else ""
            console.print(f"[green]✅ Created collection '{name}'{type_str} with {len(model_list)} models[/green]")
            for m in model_list:
                dims = EMBEDDING_MODELS[m]["dimensions"]
                console.print(f"  • {m}: {dims}D")

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise  # Re-raise our custom exceptions to be handled by main()
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to create collection: {e}", output_json)
        sys.exit(1)


def list_collections_command(verbose: bool, output_json: bool):
    """List all Qdrant collections.

    Args:
        verbose: Show detailed information
        output_json: Output as JSON
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("collection", "list")

    try:
        client = create_qdrant_client()
        collections = client.get_collections()

        if output_json:
            result = []
            for col in collections.collections:
                col_info = client.get_collection(col.name)
                metadata = get_collection_metadata(client, col.name)
                vectors = {}
                if hasattr(col_info.config.params, 'vectors') and isinstance(col_info.config.params.vectors, dict):
                    for vector_name, vector_params in col_info.config.params.vectors.items():
                        vectors[vector_name] = {
                            "size": vector_params.size,
                            "distance": str(vector_params.distance),
                        }
                result.append({
                    "name": col.name,
                    "model": metadata.get("model"),
                    "type": metadata.get("collection_type"),
                    "points_count": col_info.points_count,
                    "vectors": vectors,
                })
            print_json("success", f"Found {len(result)} collections", {"collections": result})
        else:
            table = Table(title="Qdrant Collections")
            table.add_column("Name", style="cyan")
            table.add_column("Model", style="magenta")
            table.add_column("Points", style="yellow")
            if verbose:
                table.add_column("Type", style="blue")
                table.add_column("Vectors", style="green")

            for col in collections.collections:
                col_info = client.get_collection(col.name)
                metadata = get_collection_metadata(client, col.name)
                model = metadata.get("model", "—")
                collection_type = metadata.get("collection_type", "—")
                row = [col.name, model, str(col_info.points_count)]

                if verbose:
                    row.append(collection_type)
                    vectors_str = ""
                    if hasattr(col_info.config.params, 'vectors') and isinstance(col_info.config.params.vectors, dict):
                        vectors_list = [f"{name}({params.size}D)" for name, params in col_info.config.params.vectors.items()]
                        vectors_str = ", ".join(vectors_list)
                    row.append(vectors_str)

                table.add_row(*row)

            console.print(table)

        # Log successful operation (RDR-018)
        interaction_logger.finish(result_count=len(collections.collections))

    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to list collections: {e}", output_json)
        sys.exit(1)


def delete_collection_command(name: str, confirm: bool, output_json: bool):
    """Delete a Qdrant collection.

    Args:
        name: Collection name to delete
        confirm: Skip confirmation prompt
        output_json: Output as JSON
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("collection", "delete", collection=name)

    try:
        if not confirm:
            if output_json:
                raise InvalidArgumentError("--confirm flag required for non-interactive deletion")

            response = console.input(f"[yellow]Delete collection '{name}'? This cannot be undone. (yes/no): [/yellow]")
            if response.lower() != 'yes':
                console.print("Cancelled.")
                return

        client = create_qdrant_client()
        client.delete_collection(name)

        if output_json:
            print_json("success", f"Deleted collection '{name}'", {"deleted": name})
        else:
            console.print(f"[green]✅ Deleted collection '{name}'[/green]")

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise  # Re-raise our custom exceptions
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to delete collection: {e}", output_json)
        sys.exit(1)


def info_collection_command(name: str, output_json: bool):
    """Show detailed information about a collection.

    Args:
        name: Collection name
        output_json: Output as JSON
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("collection", "info", collection=name)

    try:
        client = create_qdrant_client()
        info = client.get_collection(name)

        # Get collection metadata (includes type and model)
        metadata = get_collection_metadata(client, name)
        collection_type = metadata.get("collection_type")
        model = metadata.get("model")

        if output_json:
            vectors = {}
            if hasattr(info.config.params, 'vectors') and isinstance(info.config.params.vectors, dict):
                for vector_name, vector_params in info.config.params.vectors.items():
                    vectors[vector_name] = {
                        "size": vector_params.size,
                        "distance": str(vector_params.distance),
                    }
            data = {
                "name": name,
                "type": collection_type,
                "model": model,
                "points_count": info.points_count,
                "status": str(info.status),
                "vectors": vectors,
                "hnsw_config": {
                    "m": info.config.hnsw_config.m,
                    "ef_construct": info.config.hnsw_config.ef_construct,
                },
            }
            print_json("success", f"Collection '{name}' information", data)
        else:
            console.print(f"\n[bold cyan]Collection: {name}[/bold cyan]")
            if collection_type:
                console.print(f"Type: [bold]{collection_type}[/bold]")
            else:
                console.print(f"Type: [yellow]untyped[/yellow]")
            if model:
                console.print(f"Model: [bold magenta]{model}[/bold magenta]")
            else:
                console.print(f"Model: [yellow]not set[/yellow]")
            console.print(f"Points: {info.points_count}")
            console.print(f"Status: {info.status}")
            console.print(f"\n[bold]Vectors:[/bold]")

            if hasattr(info.config.params, 'vectors') and isinstance(info.config.params.vectors, dict):
                for vector_name, vector_params in info.config.params.vectors.items():
                    console.print(f"  • {vector_name}: {vector_params.size}D ({vector_params.distance})")

            console.print(f"\n[bold]HNSW Config:[/bold]")
            hnsw = info.config.hnsw_config
            console.print(f"  m: {hnsw.m}")
            console.print(f"  ef_construct: {hnsw.ef_construct}")

        # Log successful operation (RDR-018)
        interaction_logger.finish()

    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to get collection info: {e}", output_json)
        sys.exit(1)


def items_collection_command(name: str, output_json: bool):
    """List all indexed files/repos in a collection.

    Args:
        name: Collection name
        output_json: Output as JSON
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("collection", "items", collection=name)

    try:
        client = create_qdrant_client()

        # Get collection type to determine how to list items
        collection_type = get_collection_type(client, name)

        # Scroll through collection and collect unique items
        items_by_id = {}  # Use dict to deduplicate
        offset = None

        while True:
            # Determine which fields to fetch based on collection type
            if collection_type == "pdf" or collection_type == "markdown":
                payload_fields = ["file_path", "file_hash", "file_size", "page_count", "filename"]
            elif collection_type == "code":
                payload_fields = ["git_project_name", "git_project_identifier", "git_branch",
                                "git_commit_hash", "git_remote_url", "file_path"]
            else:
                # Untyped collection - try to get both file and git fields
                payload_fields = ["file_path", "file_hash", "filename", "git_project_name",
                                "git_project_identifier", "git_branch", "git_commit_hash"]

            points, offset = client.scroll(
                collection_name=name,
                limit=100,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False
            )

            if not points:
                break

            # Collect unique items based on type
            for point in points:
                if not point.payload:
                    continue

                if collection_type == "code":
                    # For source code, group by git_project_identifier
                    identifier = point.payload.get("git_project_identifier")
                    if identifier and identifier not in items_by_id:
                        items_by_id[identifier] = {
                            "git_project_name": point.payload.get("git_project_name"),
                            "git_project_identifier": identifier,
                            "git_branch": point.payload.get("git_branch"),
                            "git_commit_hash": point.payload.get("git_commit_hash"),
                            "git_remote_url": point.payload.get("git_remote_url"),
                            "chunk_count": 1
                        }
                    elif identifier:
                        items_by_id[identifier]["chunk_count"] += 1
                else:
                    # For PDF/markdown, group by file_path
                    file_path = point.payload.get("file_path")
                    if file_path and file_path not in items_by_id:
                        items_by_id[file_path] = {
                            "file_path": file_path,
                            "file_hash": point.payload.get("file_hash"),
                            "file_size": point.payload.get("file_size"),
                            "page_count": point.payload.get("page_count"),
                            "filename": point.payload.get("filename"),
                            "chunk_count": 1
                        }
                    elif file_path:
                        items_by_id[file_path]["chunk_count"] += 1

            if offset is None:
                break

        items_list = list(items_by_id.values())

        # Output results
        if output_json:
            data = {
                "collection": name,
                "type": collection_type,
                "item_count": len(items_list),
                "items": items_list
            }
            print_json("success", f"Found {len(items_list)} items in collection '{name}'", data)
        else:
            console.print(f"\n[bold cyan]Collection: {name}[/bold cyan]")
            type_str = f"[bold]{collection_type}[/bold]" if collection_type else "[yellow]untyped[/yellow]"
            console.print(f"Type: {type_str}")
            console.print(f"Items: {len(items_list)}\n")

            if collection_type == "code":
                # Display as table for source code
                table = Table(title="Indexed Repositories")
                table.add_column("Project", style="cyan")
                table.add_column("Branch", style="green")
                table.add_column("Commit", style="yellow")
                table.add_column("Chunks", style="magenta")

                for item in sorted(items_list, key=lambda x: x["git_project_name"]):
                    table.add_row(
                        item["git_project_name"],
                        item["git_branch"],
                        item["git_commit_hash"][:12] if item["git_commit_hash"] else "N/A",
                        str(item["chunk_count"])
                    )
                console.print(table)
            else:
                # Display as table for PDFs/markdown
                table = Table(title="Indexed Files")
                table.add_column("File", style="cyan", no_wrap=False)
                table.add_column("Size", style="yellow")
                table.add_column("Chunks", style="magenta")

                for item in sorted(items_list, key=lambda x: x.get("filename", x["file_path"])):
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
                        size_str = "N/A"

                    # Show filename or full path
                    display_name = item.get("filename") or item["file_path"]

                    table.add_row(
                        display_name,
                        size_str,
                        str(item["chunk_count"])
                    )
                console.print(table)

        # Log successful operation (RDR-018)
        interaction_logger.finish(result_count=len(items_list))

    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to list collection items: {e}", output_json)
        sys.exit(1)


def verify_collection_command(
    name: str,
    project: str | None,
    verbose: bool,
    output_json: bool,
):
    """Verify collection integrity (fsck-like check).

    Args:
        name: Collection name
        project: Optional project identifier filter (code collections only)
        verbose: Show detailed file-level results
        output_json: Output as JSON
    """
    # Start interaction logging (RDR-018)
    interaction_logger.start("collection", "verify", collection=name, project=project)

    try:
        from arcaneum.indexing.verify import CollectionVerifier

        client = create_qdrant_client()
        verifier = CollectionVerifier(client)

        console.print(f"[dim]Scanning collection '{name}'...[/dim]")
        result = verifier.verify_collection(name, project_filter=project, verbose=verbose)

        if output_json:
            # Build JSON output
            data = {
                "collection": result.collection_name,
                "type": result.collection_type,
                "total_points": result.total_points,
                "total_items": result.total_items,
                "complete_items": result.complete_items,
                "incomplete_items": result.incomplete_items,
                "is_healthy": result.is_healthy,
                "errors": result.errors,
            }

            if result.collection_type == "code":
                data["projects"] = []
                for proj in result.projects:
                    proj_data = {
                        "identifier": proj.identifier,
                        "project_name": proj.project_name,
                        "branch": proj.branch,
                        "commit_hash": proj.commit_hash,
                        "total_files": proj.total_files,
                        "complete_files": proj.complete_files,
                        "completion_percentage": round(proj.completion_percentage, 1),
                        "is_complete": proj.is_complete,
                    }
                    if not proj.is_complete and verbose:
                        proj_data["incomplete_files"] = [
                            {
                                "file_path": f.file_path,
                                "expected_chunks": f.expected_chunks,
                                "actual_chunks": f.actual_chunks,
                                "missing_indices": f.missing_indices,
                            }
                            for f in proj.incomplete_files
                        ]
                    data["projects"].append(proj_data)
            else:
                data["files"] = []
                for file in result.files:
                    file_data = {
                        "file_path": file.file_path,
                        "expected_chunks": file.expected_chunks,
                        "actual_chunks": file.actual_chunks,
                        "completion_percentage": round(file.completion_percentage, 1),
                        "is_complete": file.is_complete,
                    }
                    if not file.is_complete:
                        file_data["missing_indices"] = file.missing_indices
                    data["files"].append(file_data)

            # Include items needing repair
            data["needs_repair"] = result.get_items_needing_repair()

            status = "success" if result.is_healthy else "warning"
            msg = (
                f"Collection '{name}' is healthy"
                if result.is_healthy
                else f"Collection '{name}' has {result.incomplete_items} incomplete items"
            )
            print_json(status, msg, data)
        else:
            # Human-readable output
            console.print(f"\n[bold cyan]Collection: {name}[/bold cyan]")
            type_str = f"[bold]{result.collection_type}[/bold]" if result.collection_type else "[yellow]untyped[/yellow]"
            console.print(f"Type: {type_str}")
            console.print(f"Total points: {result.total_points:,}")
            console.print(f"Total items: {result.total_items}")

            if result.is_healthy:
                console.print(f"\n[green]Collection is healthy - all {result.complete_items} items complete[/green]")
            else:
                console.print(f"\n[yellow]Found {result.incomplete_items} incomplete items[/yellow]")
                console.print(f"Complete: {result.complete_items}, Incomplete: {result.incomplete_items}")

                # Show incomplete items
                if result.collection_type == "code":
                    table = Table(title="Incomplete Projects")
                    table.add_column("Project", style="cyan")
                    table.add_column("Branch", style="green")
                    table.add_column("Files", style="yellow")
                    table.add_column("Completion", style="magenta")

                    for proj in result.projects:
                        if not proj.is_complete:
                            table.add_row(
                                proj.project_name,
                                proj.branch,
                                f"{proj.complete_files}/{proj.total_files}",
                                f"{proj.completion_percentage:.1f}%",
                            )

                            # Show file details if verbose
                            if verbose:
                                for f in proj.incomplete_files[:5]:  # Limit to first 5
                                    console.print(
                                        f"  [dim]{f.file_path}: "
                                        f"{f.actual_chunks}/{f.expected_chunks} chunks "
                                        f"(missing: {f.missing_indices[:5]}{'...' if len(f.missing_indices) > 5 else ''})[/dim]"
                                    )
                                if len(proj.incomplete_files) > 5:
                                    console.print(f"  [dim]... and {len(proj.incomplete_files) - 5} more files[/dim]")

                    console.print(table)
                else:
                    table = Table(title="Incomplete Files")
                    table.add_column("File", style="cyan", no_wrap=False)
                    table.add_column("Chunks", style="yellow")
                    table.add_column("Completion", style="magenta")

                    for file in result.files:
                        if not file.is_complete:
                            table.add_row(
                                file.file_path,
                                f"{file.actual_chunks}/{file.expected_chunks}",
                                f"{file.completion_percentage:.1f}%",
                            )

                    console.print(table)

                # Show repair hint
                needs_repair = result.get_items_needing_repair()
                if needs_repair:
                    console.print(f"\n[dim]To repair, re-index the following items:[/dim]")
                    for item in needs_repair[:10]:
                        console.print(f"  [yellow]{item}[/yellow]")
                    if len(needs_repair) > 10:
                        console.print(f"  [dim]... and {len(needs_repair) - 10} more[/dim]")

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=result.total_items,
            is_healthy=result.is_healthy,
            incomplete_items=result.incomplete_items,
        )

    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to verify collection: {e}", output_json)
        sys.exit(1)

