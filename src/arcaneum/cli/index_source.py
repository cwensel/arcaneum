"""CLI command for source code indexing (RDR-005)."""

import sys
import os
import logging
import signal
from typing import Optional

from rich.console import Console
from rich import print as rprint

from .logging_config import setup_logging_default, setup_logging_verbose, setup_logging_debug
from .utils import set_process_priority, create_qdrant_client
from arcaneum.indexing.source_code_pipeline import SourceCodeIndexer
from arcaneum.indexing.qdrant_indexer import QdrantIndexer
from arcaneum.indexing.collection_metadata import (
    validate_collection_type,
    set_collection_metadata,
    get_vector_names,
    CollectionType
)
from arcaneum.embeddings.client import EmbeddingClient
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
    no_gpu: bool,
    verbose: bool,
    debug: bool,
    output_json: bool
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
        verbose: Verbose output
        debug: Debug mode (show all library warnings)
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

    # Auto-tune batch size if not explicitly set by user
    if embedding_batch_size is None:
        if not no_gpu:
            # GPU mode: auto-tune based on available memory
            try:
                from arcaneum.utils.memory import get_gpu_memory_info, estimate_safe_batch_size_v2
                memory_info = get_gpu_memory_info()
                if memory_info:
                    available_bytes = memory_info.get('available', 0)
                    device_type = memory_info.get('device_type', 'cuda')
                    embedding_batch_size = estimate_safe_batch_size_v2(
                        available_memory_bytes=available_bytes,
                        device_type=device_type
                    )
                else:
                    embedding_batch_size = 256  # Fallback
            except Exception:
                embedding_batch_size = 256  # Fallback on error
        else:
            # CPU mode: use conservative default
            embedding_batch_size = 256

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

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

        # Connect to Qdrant (use defaults)
        qdrant_url = 'localhost'
        qdrant_port = 6333
        if verbose:
            console.print(f"[cyan]Connecting to Qdrant:[/cyan] {qdrant_url}:{qdrant_port}")

        # Use HTTP client (consistent with other CLI commands)
        qdrant_url_full = f"http://{qdrant_url}:{qdrant_port}"
        qdrant_client = create_qdrant_client(url=qdrant_url_full)

        # Retrieve model from collection metadata if not provided
        if model is None:
            from arcaneum.indexing.collection_metadata import get_collection_metadata
            metadata = get_collection_metadata(qdrant_client, collection)
            if not metadata or 'model' not in metadata:
                raise ValueError(
                    f"Collection '{collection}' has no model metadata. "
                    "Please create the collection with 'arc collection create --type code' first."
                )
            model = metadata['model']
        else:
            # Warn about deprecated --model flag
            console.print(
                "[yellow]⚠️  Warning: --model flag is deprecated. "
                "Model is now set at collection creation time. "
                "Please use 'arc collection create --type code' instead.[/yellow]"
            )

        qdrant_indexer = QdrantIndexer(qdrant_client)

        # Create embedding client with GPU support and persistent model caching (RDR-013 Phase 2, arcaneum-pwd5)
        # get_cached_model ensures models persist for the process lifetime,
        # saving 7-8 seconds on subsequent CLI invocations within the same session
        # GPU enabled by default, disabled with --no-gpu flag
        embedding_client = get_cached_model(
            model_name=model,
            cache_dir=str(get_models_dir()),
            use_gpu=not no_gpu
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
            elif device_info['gpu_available']:
                console.print(f"  [green]Device: {device_info['device'].upper()} (GPU acceleration enabled)[/green]")
            else:
                console.print(f"  Device: CPU (GPU not available)")

            # Show parallelism configuration
            console.print(f"  File processing: {actual_file_workers} workers")
            console.print(f"  Embedding: {actual_embedding_workers} workers, batch size {embedding_batch_size}")

            # Show process priority
            if process_priority != "normal":
                console.print(f"  Process Priority: {process_priority}")

            console.print()

        # Check/create collection and determine vector name
        if not qdrant_indexer.collection_exists(collection):
            if verbose:
                console.print(f"[yellow]Collection '{collection}' does not exist, creating...[/yellow]")

            # Map model names to actual embedding models for NEW collections
            model_map = {
                'jina-code': 'jinaai/jina-embeddings-v2-base-code',  # 768D - code-specific
                'jina-v2-code': 'jinaai/jina-embeddings-v2-base-code',  # 768D
                'jina-v3': 'jinaai/jina-embeddings-v3',  # 1024D - multilingual
                'jina-base-en': 'jinaai/jina-embeddings-v2-base-en',  # 768D - English-only
                'stella': 'dunzhang/stella_en_1.5B_v5',     # 1024D
                'bge': 'BAAI/bge-large-en-v1.5',            # 1024D
            }
            embedding_model = model_map.get(model, 'jinaai/jina-embeddings-v2-base-code')

            # Determine vector size
            vector_sizes = {
                'sentence-transformers/all-MiniLM-L6-v2': 384,
                'BAAI/bge-small-en-v1.5': 384,
                'BAAI/bge-base-en-v1.5': 768,
                'BAAI/bge-large-en-v1.5': 1024,
                'jinaai/jina-embeddings-v2-base-code': 768,
                'jinaai/jina-embeddings-v2-base-en': 768,
                'jinaai/jina-embeddings-v3': 1024,
                'dunzhang/stella_en_1.5B_v5': 1024,
            }
            vector_size = vector_sizes.get(embedding_model, 768)

            qdrant_indexer.create_collection(collection, vector_size=vector_size)

            # Set type metadata for auto-created collections
            set_collection_metadata(
                client=qdrant_client,
                collection_name=collection,
                collection_type=CollectionType.CODE,
                model=embedding_model
            )
            vector_name = model  # Use specified model for new collection
            if verbose:
                console.print(f"[green]✓ Collection created (type: code, vector: {vector_name})[/green]")
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
                    'bge': 'BAAI/bge-large-en-v1.5',        # 1024D
                    'stella': 'dunzhang/stella_en_1.5B_v5',  # 1024D
                    'jina-code': 'jinaai/jina-embeddings-v2-base-code',  # 768D - code-specific
                    'jina': 'jinaai/jina-embeddings-v2-base-code',  # 768D
                    'jina-v3': 'jinaai/jina-embeddings-v3',  # 1024D - multilingual
                    'jina-base-en': 'jinaai/jina-embeddings-v2-base-en',  # 768D - English-only
                }
                embedding_model = vector_to_model_map.get(vector_name, 'jinaai/jina-embeddings-v2-base-code')
                if verbose:
                    console.print(f"[cyan]Auto-detected embedding model:[/cyan] {embedding_model}")
            else:
                vector_name = model
                model_map = {
                    'jina-code': 'sentence-transformers/all-MiniLM-L6-v2',
                    'bge': 'BAAI/bge-large-en-v1.5',
                }
                embedding_model = model_map.get(model, model)
                console.print(f"[green]✓ Collection '{collection}' exists (type: code)[/green]")

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
            embedding_batch_size=embedding_batch_size  # Embedding batch size
        )

        # Index directory
        stats = indexer.index_directory(
            input_path=source_dir,
            collection_name=collection,
            depth=depth,
            force=force,
            show_progress=verbose,
            verbose=verbose,
            file_list=file_list
        )

        # Output
        if output_json:
            import json
            print(json.dumps(stats, indent=2))

        sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT

    except Exception as e:
        logger.error(f"Indexing failed: {e}", exc_info=verbose)
        if not output_json:
            console.print(f"\n[bold red]Error:[/bold red] {e}")
        else:
            import json
            print(json.dumps({"error": str(e)}, indent=2))
        sys.exit(1)
