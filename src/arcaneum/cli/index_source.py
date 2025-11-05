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
from arcaneum.paths import get_models_dir

console = Console()


def index_source_command(
    path: str,
    collection: str,
    model: str,
    file_workers: Optional[int],
    file_worker_mult: float,
    embedding_workers: int,
    embedding_worker_mult: float,
    embedding_batch_size: int,
    chunk_size: Optional[int],
    chunk_overlap: Optional[int],
    depth: Optional[int],
    process_priority: str,
    max_perf: bool,
    force: bool,
    no_gpu: bool,
    verbose: bool,
    debug: bool,
    output_json: bool
):
    """Index source code to Qdrant collection (from RDR-005).

    Args:
        path: Directory containing git repositories
        collection: Target collection name
        model: Embedding model (jina-code, jina-v2-code, stella)
        file_workers: Number of parallel workers for file processing (None = cpu_count // 2)
        file_worker_mult: File worker multiplier
        embedding_workers: Number of parallel workers for embedding generation
        embedding_worker_mult: Embedding worker multiplier
        embedding_batch_size: Batch size for embedding generation
        chunk_size: Target chunk size in tokens (default: 400)
        chunk_overlap: Overlap between chunks in tokens (default: 20)
        depth: Git discovery depth (None = unlimited)
        process_priority: Process scheduling priority (low, normal, high)
        max_perf: Maximum performance preset
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
    set_process_priority(process_priority)

    # Apply --max-perf preset (sets defaults before precedence logic)
    if max_perf:
        if embedding_worker_mult is None:
            embedding_worker_mult = 1.0
        if embedding_batch_size == 200:  # Default value
            embedding_batch_size = 500
        if process_priority == "normal":  # Default value
            process_priority = "low"
            set_process_priority(process_priority)  # Re-apply with new priority

    # Compute embedding_workers with precedence: absolute → multiplier → default (0.5)
    from multiprocessing import cpu_count
    if embedding_workers is not None:
        # Absolute value specified, use it
        actual_embedding_workers = max(1, embedding_workers)
        embedding_worker_source = f"{actual_embedding_workers} (absolute)"
    elif embedding_worker_mult is not None:
        # Multiplier specified, compute from cpu_count
        actual_embedding_workers = max(1, int(cpu_count() * embedding_worker_mult))
        embedding_worker_source = f"{actual_embedding_workers} (cpu_count × {embedding_worker_mult})"
    else:
        # Default: 0.5 multiplier (half of CPU cores)
        actual_embedding_workers = max(1, int(cpu_count() * 0.5))
        embedding_worker_source = f"{actual_embedding_workers} (cpu_count × 0.5, default)"

    # Compute file_workers with precedence: absolute → multiplier → default (1)
    # Note: Source code DOES use file parallelism (ProcessPoolExecutor in source_code_pipeline.py)
    if file_workers is not None:
        # Absolute value specified, use it
        actual_file_workers = max(1, file_workers)
        file_worker_source = f"{actual_file_workers} (absolute)"
    elif file_worker_mult is not None:
        # Multiplier specified, compute from cpu_count
        actual_file_workers = max(1, int(cpu_count() * file_worker_mult))
        file_worker_source = f"{actual_file_workers} (cpu_count × {file_worker_mult})"
    else:
        # Default: 1 worker (sequential processing)
        actual_file_workers = 1
        file_worker_source = f"{actual_file_workers} (default, sequential)"

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Connect to Qdrant (use defaults)
        qdrant_url = 'localhost'
        qdrant_port = 6333
        if verbose:
            console.print(f"[cyan]Connecting to Qdrant:[/cyan] {qdrant_url}:{qdrant_port}")

        # Use HTTP client (consistent with other CLI commands)
        qdrant_url_full = f"http://{qdrant_url}:{qdrant_port}"
        qdrant_client = create_qdrant_client(url=qdrant_url_full)

        qdrant_indexer = QdrantIndexer(qdrant_client)

        # Create embedding client with GPU support (RDR-013 Phase 2)
        # GPU enabled by default, disabled with --no-gpu flag
        embedding_client = EmbeddingClient(
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
            preset_suffix = " [max-perf preset]" if max_perf else ""
            console.print(f"  File processing: {file_worker_source} workers{preset_suffix}")
            console.print(f"  Embedding: {embedding_worker_source} workers, batch size {embedding_batch_size}{preset_suffix}")

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
            input_path=path,
            collection_name=collection,
            depth=depth,
            force=force,
            show_progress=verbose,
            verbose=verbose
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
