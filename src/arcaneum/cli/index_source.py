"""CLI command for source code indexing (RDR-005)."""

import sys
import logging
import signal
from pathlib import Path
from typing import Optional

from rich.console import Console

from .interaction_logger import interaction_logger
from .logging_config import setup_logging_default, setup_logging_verbose, setup_logging_debug
from .utils import set_process_priority, create_qdrant_client
from arcaneum.indexing.source_code_pipeline import SourceCodeIndexer
from arcaneum.indexing.qdrant_indexer import QdrantIndexer
from arcaneum.indexing.collection_metadata import (
    backfill_embedding_prompt_policy,
    validate_collection_type,
    set_collection_metadata,
    get_collection_metadata,
    get_vector_names,
    CollectionType,
    prompt_policy_can_be_backfilled,
    prompt_policy_issues,
    update_collection_metadata,
)
from arcaneum.embeddings.client import prompt_policy_model_key_for_name
from arcaneum.embeddings.model_cache import get_cached_model
from arcaneum.paths import get_models_dir

console = Console()


def index_source_command(
    path: str,
    from_file: str,
    collection: str,
    model: str,
    embedding_batch_size: int,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    depth: Optional[int],
    process_priority: str,
    not_nice: bool,
    force: bool,
    prune: bool,
    no_gpu: bool,
    verify: bool,
    streaming: bool,
    verbose: bool,
    debug: bool,
    profile: bool,
    output_json: bool,
):
    """Index source code to Qdrant collection (from RDR-005).

    Args:
        path: Directory containing git repositories (or None if using from_file)
        from_file: Path to file containing list of source code file paths, or "-" for stdin
        collection: Target collection name
        model: Embedding model (jina-code, jina-v2-code, stella)
        embedding_batch_size: Batch size for embedding generation
        chunk_size: Target chunk size in tokens (default: 400)
        chunk_overlap: Overlap between chunks in tokens (default: 20)
        depth: Git discovery depth (None = unlimited)
        process_priority: Process scheduling priority (low, normal, high)
        not_nice: Disable process priority reduction for worker processes
        force: Force reindex all projects
        no_gpu: Disable GPU acceleration (use CPU only)
        verify: Verify collection integrity and repair incomplete items after indexing
        verbose: Verbose output
        debug: Debug mode (show all library warnings)
        profile: Show pipeline performance profiling
        output_json: Output JSON format

    Note:
        For corporate networks with SSL issues, set environment variables:
        - HF_HUB_OFFLINE=1 (offline mode)
        - PYTHONHTTPSVERIFY=0 (disable SSL verification)
        See docs/testing/offline-mode.md for details.
    """
    # Setup logging (centralized configuration)
    if debug:
        setup_logging_debug()
    elif verbose:
        setup_logging_verbose()
    else:
        setup_logging_default()

    logger = logging.getLogger(__name__)

    # Set process priority early
    set_process_priority(process_priority, disable_worker_nice=not_nice)

    # Auto-detect optimal settings
    # Note: File workers is hardcoded to 1 due to embedding lock (arcaneum-6pvk)
    # Embedding generation is serialized when file_workers > 1 to prevent GPU conflicts
    actual_file_workers = 1  # Serialized by embedding lock
    # GPU models ignore embedding_workers (single-threaded is faster)
    actual_embedding_workers = 1  # GPU: single-threaded optimal

    # Note: Batch size auto-tuning happens AFTER model retrieval (see below)
    # to ensure we know the actual model name for proper batch size estimation

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "index",
        "code",
        collection=collection,
        path=path,
        from_file=from_file,
        depth=depth,
        force=force,
    )

    try:
        # Handle file list if provided
        file_list = None
        source_dir = None

        if from_file:
            from .utils import read_file_list

            # Code indexing doesn't restrict by extension - accepts all source files
            file_list = read_file_list(from_file, allowed_extensions=None)
            if not file_list:
                raise ValueError("No valid files found in the provided list")
            # Use parent directory of first file as base directory for reporting
            source_dir = file_list[0].parent
        else:
            from pathlib import Path

            source_dir = Path(path)
            if not source_dir.exists():
                raise ValueError(f"Path does not exist: {path}")

        if verbose:
            console.print("[cyan]Connecting to Qdrant[/cyan]")

        qdrant_client = create_qdrant_client()
        explicit_model = model is not None

        # Retrieve model from collection metadata if not provided
        if model is None:
            metadata = get_collection_metadata(qdrant_client, collection)
            if not metadata or "model" not in metadata:
                raise ValueError(
                    f"Collection '{collection}' has no model metadata. "
                    "Please create the collection with 'arc collection create --type code' first."
                )
            model = prompt_policy_model_key_for_name(metadata["model"]) or metadata["model"]
        else:
            # Warn about deprecated --model flag
            console.print(
                "[yellow]⚠️  Warning: --model flag is deprecated. "
                "Model is now set at collection creation time. "
                "Please use 'arc collection create --type code' instead.[/yellow]"
            )

        # Auto-tune batch size if not explicitly set by user
        # This happens AFTER model retrieval to ensure proper model-aware estimation
        if embedding_batch_size is None:
            if not no_gpu:
                # GPU mode: auto-tune based on available memory
                try:
                    from arcaneum.utils.memory import (
                        get_gpu_memory_info,
                        estimate_safe_batch_size_v2,
                    )

                    available_bytes, total_bytes, device_type = get_gpu_memory_info()

                    if available_bytes is not None:
                        embedding_batch_size = estimate_safe_batch_size_v2(
                            model_name=model,
                            available_gpu_bytes=available_bytes,
                            pipeline_overhead_gb=0.3,
                            safety_factor=0.6,
                            device_type=device_type,
                        )
                    else:
                        embedding_batch_size = 128  # Fallback
                except Exception:
                    embedding_batch_size = 128  # Fallback on error
            else:
                # CPU mode: use conservative default
                embedding_batch_size = 128

        qdrant_indexer = QdrantIndexer(qdrant_client)

        # Create embedding client with persistent model caching (RDR-013 Phase 2, arcaneum-pwd5)
        # get_cached_model ensures models persist for the process lifetime,
        # saving 7-8 seconds on subsequent CLI invocations within the same session
        # CPU is the stable default; --gpu opts into accelerator embedding.
        embedding_client = get_cached_model(
            model_name=model, cache_dir=str(get_models_dir()), use_gpu=not no_gpu
        )

        # Show configuration at start (if verbose)
        if verbose:
            console.print(f"\n[bold blue]Source Code Indexing Configuration[/bold blue]")
            console.print(f"  Collection: {collection} (type: code)")
            console.print(f"  Model: {model}")

            # Show GPU/CPU info
            device_info = embedding_client.get_device_info()
            if no_gpu:
                console.print(f"  Device: CPU (GPU acceleration disabled)")
            elif device_info["gpu_available"]:
                console.print(
                    f"  [green]Device: {device_info['device'].upper()} (GPU acceleration enabled)[/green]"
                )
            else:
                console.print(f"  Device: CPU (GPU not available)")

            # Show parallelism configuration
            console.print(f"  File processing: {actual_file_workers} workers")
            console.print(
                f"  Embedding: {actual_embedding_workers} workers, batch size {embedding_batch_size}"
            )

            # Show process priority
            if process_priority != "normal":
                console.print(f"  Process Priority: {process_priority}")

            console.print()

        # Check/create collection and determine vector name
        if not qdrant_indexer.collection_exists(collection):
            if verbose:
                console.print(
                    f"[yellow]Collection '{collection}' does not exist, creating...[/yellow]"
                )

            # Map model names to actual embedding models for NEW collections
            model_map = {
                "jina-code": "jinaai/jina-embeddings-v2-base-code",  # 768D - code-specific
                "jina-v2-code": "jinaai/jina-embeddings-v2-base-code",  # 768D
                "jina-v3": "jinaai/jina-embeddings-v3",  # 1024D - multilingual
                "jina-base-en": "jinaai/jina-embeddings-v2-base-en",  # 768D - English-only
                "stella": "dunzhang/stella_en_1.5B_v5",  # 1024D
                "bge": "BAAI/bge-large-en-v1.5",  # 1024D
            }
            embedding_model = model_map.get(model, "jinaai/jina-embeddings-v2-base-code")

            # Determine vector size
            vector_sizes = {
                "sentence-transformers/all-MiniLM-L6-v2": 384,
                "BAAI/bge-small-en-v1.5": 384,
                "BAAI/bge-base-en-v1.5": 768,
                "BAAI/bge-large-en-v1.5": 1024,
                "jinaai/jina-embeddings-v2-base-code": 768,
                "jinaai/jina-embeddings-v2-base-en": 768,
                "jinaai/jina-embeddings-v3": 1024,
                "dunzhang/stella_en_1.5B_v5": 1024,
            }
            vector_size = vector_sizes.get(embedding_model, 768)
            metadata_model = {
                "jina-v2-code": "jina-code",
            }.get(model, model)

            qdrant_indexer.create_collection(collection, vector_size=vector_size)

            # Set type metadata for auto-created collections
            set_collection_metadata(
                client=qdrant_client,
                collection_name=collection,
                collection_type=CollectionType.CODE,
                model=metadata_model,
            )
            vector_name = model  # Use specified model for new collection
            if verbose:
                console.print(
                    f"[green]✓ Collection created (type: code, vector: {vector_name})[/green]"
                )
        else:
            # Validate collection type
            validate_collection_type(qdrant_client, collection, CollectionType.CODE)

            # Auto-detect vector name from collection
            vector_names = get_vector_names(qdrant_client, collection)
            if vector_names:
                vector_name = vector_names[0]  # Use first vector
                if verbose:
                    console.print(
                        f"[green]✓ Collection '{collection}' exists "
                        f"(type: code, vector: {vector_name})[/green]"
                    )

                # Map vector name back to actual embedding model
                # Must match EMBEDDING_MODELS dimensions in embeddings/client.py
                vector_to_model_map = {
                    "bge": "BAAI/bge-large-en-v1.5",  # 1024D
                    "stella": "dunzhang/stella_en_1.5B_v5",  # 1024D
                    "jina-code": "jinaai/jina-embeddings-v2-base-code",  # 768D - code-specific
                    "jina": "jinaai/jina-embeddings-v2-base-code",  # 768D
                    "jina-v3": "jinaai/jina-embeddings-v3",  # 1024D - multilingual
                    "jina-base-en": "jinaai/jina-embeddings-v2-base-en",  # 768D - English-only
                    "jina-code-0.5b": "jinaai/jina-code-embeddings-0.5b",  # 896D - SOTA code model
                    "jina-code-1.5b": "jinaai/jina-code-embeddings-1.5b",  # 1536D - SOTA code model
                }
                embedding_model = vector_to_model_map.get(
                    vector_name, "jinaai/jina-embeddings-v2-base-code"
                )
                if verbose:
                    console.print(f"[cyan]Auto-detected embedding model:[/cyan] {embedding_model}")
            else:
                vector_name = model
                model_map = {
                    "jina-code": "sentence-transformers/all-MiniLM-L6-v2",
                    "bge": "BAAI/bge-large-en-v1.5",
                }
                embedding_model = model_map.get(model, model)
                console.print(f"[green]✓ Collection '{collection}' exists (type: code)[/green]")

            if explicit_model and vector_names and model != vector_name:
                raise ValueError(
                    f"Explicit model '{model}' does not match existing vector "
                    f"'{vector_name}' for collection '{collection}'. Create a "
                    "new collection for a different embedding model."
                )

        metadata = get_collection_metadata(qdrant_client, collection)
        issues = prompt_policy_issues(metadata, model)
        if issues and prompt_policy_can_be_backfilled(metadata, [model]):
            metadata = backfill_embedding_prompt_policy(
                qdrant_client,
                collection,
                CollectionType.CODE,
                model,
            )
            issues = []
            if verbose:
                console.print(
                    "[cyan]Backfilled legacy embedding prompt-policy metadata "
                    "(no documents reindexed)[/cyan]"
                )
        full_force_candidate = force and file_list is None and depth is None
        if issues and not full_force_candidate:
            raise ValueError(
                "Collection embedding policy is stale before source indexing: "
                + "; ".join(issues)
                + ". Run an unrestricted full directory index with --force to "
                "reindex and recertify the corpus before incremental indexing "
                "or repair."
            )

        # Create indexer (pass embedding client for GPU support)
        indexer = SourceCodeIndexer(
            qdrant_indexer=qdrant_indexer,
            embedding_client=embedding_client,
            embedding_model_id=model,  # Use model ID (e.g., "jina-code", "stella")
            chunk_size=chunk_size or 400,  # Default: 400 tokens for 8K context models
            chunk_overlap=chunk_overlap or 20,  # Default: 20 tokens overlap
            vector_name=vector_name,  # Use auto-detected or specified vector name
            parallel_workers=actual_file_workers,  # File processing parallelism
            embedding_workers=actual_embedding_workers,  # Embedding generation parallelism
            embedding_batch_size=embedding_batch_size,  # Embedding batch size
            streaming=streaming,  # Stream embeddings to Qdrant immediately
        )

        # Pre-verify if requested - find incomplete items to include in indexing
        repair_targets = None
        if verify:
            from arcaneum.indexing.verify import CollectionVerifier

            if verbose or not output_json:
                console.print("[dim]Pre-indexing verification...[/dim]")

            verifier = CollectionVerifier(qdrant_client)
            pre_verification = verifier.verify_collection(collection, verbose=False)

            if not pre_verification.is_healthy:
                incomplete_identifiers = pre_verification.get_items_needing_repair()
                if incomplete_identifiers:
                    repair_targets = set(incomplete_identifiers)
                    if verbose or not output_json:
                        console.print(
                            f"[yellow]Found {len(repair_targets)} incomplete items to repair: {list(repair_targets)}[/yellow]\n"
                        )

        # Capture indexed paths BEFORE indexing so we can detect orphans
        # (indexed files no longer on disk) on a force, full-directory run.
        from arcaneum.indexing.common.sync import MetadataBasedSync

        path_sync = MetadataBasedSync(qdrant_client)
        pre_run_paths = set()
        if force and file_list is None:
            pre_run_paths = path_sync._get_indexed_file_paths_set(collection)
            if issues:
                source_root = source_dir.resolve()
                outside_root = [
                    p
                    for p in pre_run_paths
                    if not Path(p).resolve().is_relative_to(source_root)
                ]
                if outside_root:
                    raise ValueError(
                        "Collection embedding policy is stale before source "
                        "indexing: existing indexed files are outside the "
                        "requested source root. Re-run --force from the full "
                        "corpus root before incremental indexing or repair."
                    )

        # Index directory (include repair targets if any)
        stats = indexer.index_directory(
            input_path=source_dir,
            collection_name=collection,
            depth=depth,
            force=force,
            show_progress=verbose,
            verbose=verbose,
            file_list=file_list,
            profile=profile,
            repair_targets=repair_targets,  # Projects to force re-index even if commit unchanged
        )

        # Orphan-aware prompt-policy stamp gate (C3/C4)
        if force and file_list is None:
            from ..indexing.collection_metadata import (
                prune_orphans_and_stamp,
            )

            on_disk_paths = path_sync._get_indexed_file_paths_set(collection)
            on_disk_paths = {p for p in on_disk_paths if Path(p).exists()}
            # Coverage for certification is the set of files this run actually
            # processed. Existence under source_dir is not enough: a still-on-
            # disk repo may be undiscovered, metadata extraction may fail, or a
            # collection may span roots while this run only reindexed one.
            covered_paths = set(stats.get("covered_paths", []))
            gate_stats = {
                "files": stats.get("files_processed", 0),
                # Real per-file failure count from the pipeline; a reindex with
                # errors must not certify the collection (job-1921 Fix C).
                "errors": stats.get("errors", 0),
            }
            prune_warn = (
                None if output_json else (lambda m: console.print(f"[yellow]⚠ {m}[/yellow]"))
            )
            certification = prune_orphans_and_stamp(
                qdrant=qdrant_client,
                sync=path_sync,
                collection_name=collection,
                collection_type=CollectionType.CODE,
                model=model,
                force=force,
                file_list=file_list,
                stats=gate_stats,
                on_disk_paths=on_disk_paths,
                pre_run_paths=pre_run_paths,
                prune=prune,
                covered_paths=covered_paths,
                warn=prune_warn,
            )
            if issues and not certification.get("stamped"):
                raise ValueError(
                    "Full force source indexing did not certify the updated "
                    "embedding policy: "
                    + "; ".join(issues)
                    + ". Remove stale/orphaned vectors and rerun --force before "
                    "incremental indexing."
                )
            if certification.get("stamped") and metadata.get("model") != model:
                update_collection_metadata(qdrant_client, collection, model=model)

        # Post-verify if requested
        verification_result = None
        if verify:
            if verbose or not output_json:
                console.print("\n[dim]Post-indexing verification...[/dim]")

            verification_result = verifier.verify_collection(collection, verbose=verbose)

            if verification_result.is_healthy:
                if verbose or not output_json:
                    console.print(
                        f"[green]Collection verified - all {verification_result.complete_items} items complete[/green]"
                    )
            else:
                if verbose or not output_json:
                    still_incomplete = verification_result.get_items_needing_repair()
                    console.print(
                        f"[yellow]Warning: {len(still_incomplete)} items still incomplete after indexing[/yellow]"
                    )
                    for item in still_incomplete[:5]:
                        console.print(f"  [yellow]{item}[/yellow]")

            # Add verification results to stats
            stats["verification"] = {
                "is_healthy": verification_result.is_healthy,
                "total_items": verification_result.total_items,
                "complete_items": verification_result.complete_items,
                "incomplete_items": verification_result.incomplete_items,
                "repaired": len(repair_targets) if repair_targets else 0,
            }

        # Output
        if output_json:
            import json

            print(json.dumps(stats, indent=2))

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=stats.get("projects_processed", 0),
            files=stats.get("files_processed", 0),
            chunks=stats.get("chunks_uploaded", 0),
        )

        sys.exit(0)

    except KeyboardInterrupt:
        interaction_logger.finish(error="interrupted by user")
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        interaction_logger.finish(error=str(e))
        logger.error(f"Indexing failed: {e}", exc_info=verbose)
        if not output_json:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        else:
            import json

            print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)
