"""CLI commands for markdown indexing (RDR-014)."""

import click
from pathlib import Path
from rich.console import Console
import logging
import sys
import json
import os
import signal

from .logging_config import setup_logging_default, setup_logging_verbose, setup_logging_debug
from .utils import create_qdrant_client
from ..config import load_config, DEFAULT_MODELS
from ..embeddings.client import EmbeddingClient
from ..embeddings.model_cache import get_cached_model
from ..indexing.markdown.pipeline import MarkdownIndexingPipeline
from ..indexing.collection_metadata import validate_collection_type, CollectionType, get_vector_names
from qdrant_client import QdrantClient

console = Console()
logger = logging.getLogger(__name__)


def index_markdown_command(
    path: str,
    collection: str,
    model: str,
    embedding_batch_size: int,
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool,
    exclude: tuple,
    qdrant_url: str,
    process_priority: str,
    force: bool,
    no_gpu: bool,
    offline: bool,
    verbose: bool,
    debug: bool,
    output_json: bool
):
    """Index markdown files to Qdrant collection.

    Args:
        path: Directory containing markdown files
        collection: Target collection name
        model: Embedding model to use
        embedding_batch_size: Batch size for embedding generation
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        process_priority: Process scheduling priority
        force: Force reindex all files
        no_gpu: Disable GPU acceleration
        offline: Use cached models only (no network calls)
        verbose: Verbose output
        debug: Debug mode (show all library warnings)
        output_json: Output JSON format
    """
    # Import utilities
    from .utils import set_process_priority

    # Set process priority early
    set_process_priority(process_priority)

    # Auto-detect optimal settings
    actual_file_workers = 1
    actual_embedding_workers = 1

    # Enable offline mode if requested
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # Setup logging (centralized configuration)
    if debug:
        setup_logging_debug()
    elif verbose:
        setup_logging_verbose()
    else:
        setup_logging_default()

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        markdown_dir = Path(path)
        if not markdown_dir.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Initialize Qdrant client early to retrieve model from collection metadata
        from arcaneum.paths import get_models_dir
        qdrant = create_qdrant_client(url=qdrant_url)

        # Retrieve model from collection metadata if not provided
        if model is None:
            from arcaneum.indexing.collection_metadata import get_collection_metadata
            metadata = get_collection_metadata(qdrant, collection)
            if not metadata or 'model' not in metadata:
                raise ValueError(
                    f"Collection '{collection}' has no model metadata. "
                    "Please create the collection with 'arc collection create --type markdown' first."
                )
            model = metadata['model']
        else:
            # Warn about deprecated --model flag
            console.print(
                "[yellow]⚠️  Warning: --model flag is deprecated. "
                "Model is now set at collection creation time. "
                "Please use 'arc collection create --type markdown' instead.[/yellow]"
            )

        # Use default model config
        if model not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(DEFAULT_MODELS.keys())}")

        model_config = DEFAULT_MODELS[model]
        model_dict = {
            'chunk_size': chunk_size or model_config.chunk_size,
            'chunk_overlap': chunk_overlap or model_config.chunk_overlap,
            'vector_name': getattr(model_config, 'vector_name', None),
        }

        # Initialize embedding client with persistent model caching (arcaneum-pwd5)
        # get_cached_model ensures models are cached for the process lifetime,
        # saving 7-8 seconds on subsequent CLI invocations within the same session
        embeddings = get_cached_model(
            model_name=model,
            cache_dir=str(get_models_dir()),
            use_gpu=not no_gpu
        )

        # Validate collection type (must be 'markdown' or untyped)
        try:
            validate_collection_type(qdrant, collection, CollectionType.MARKDOWN)
        except Exception as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            else:
                console.print(f"[red]❌ {e}[/red]")
            sys.exit(1)

        # Detect vector name from existing collection (if it exists)
        try:
            vector_names = get_vector_names(qdrant, collection)
            if vector_names:
                # Collection exists with named vectors - use first one
                model_dict['vector_name'] = vector_names[0]
        except Exception:
            # Collection doesn't exist yet, use default vector_name from model_config
            pass

        # Build exclude patterns from CLI args
        exclude_patterns = list(exclude) if exclude else []
        # Add default excludes
        default_excludes = ['**/node_modules/**', '**/.git/**', '**/venv/**']
        for pattern in default_excludes:
            if pattern not in exclude_patterns:
                exclude_patterns.append(pattern)

        # Create pipeline with custom exclude patterns and worker configuration
        pipeline = MarkdownIndexingPipeline(
            qdrant_client=qdrant,
            embedding_client=embeddings,
            batch_size=300,  # Optimized from 100 (arcaneum-6pvk: reduce upload rate)
            exclude_patterns=exclude_patterns,
            file_workers=actual_file_workers,
            embedding_workers=actual_embedding_workers,
            embedding_batch_size=embedding_batch_size,
        )

        # Show configuration
        if not output_json:
            from arcaneum.embeddings.client import EMBEDDING_MODELS
            actual_model = EMBEDDING_MODELS.get(model, {}).get('name', model)
            model_desc = EMBEDDING_MODELS.get(model, {}).get('description', '')

            console.print(f"\n[bold blue]Markdown Indexing Configuration[/bold blue]")
            console.print(f"  Collection: {collection} (type: markdown)")
            if model_desc:
                console.print(f"  Model: {model} → {actual_model}")
                console.print(f"    ({model_desc})")
            else:
                console.print(f"  Model: {actual_model}")

            # Show GPU/CPU info
            device_info = embeddings.get_device_info()
            if no_gpu:
                console.print(f"  Device: CPU (GPU acceleration disabled)")
            elif device_info['gpu_available']:
                console.print(f"  [green]Device: {device_info['device'].upper()} (GPU acceleration enabled)[/green]")
            else:
                console.print(f"  Device: CPU (GPU not available)")

            # Show parallelism configuration
            console.print(f"  File processing: {actual_file_workers} workers (sequential, parallelism planned)")
            console.print(f"  Embedding: {actual_embedding_workers} workers, batch size {embedding_batch_size}")

            console.print(f"  Chunk size: {model_dict['chunk_size']} tokens")
            console.print(f"  Chunk overlap: {model_dict['chunk_overlap']} tokens")
            console.print(f"  Mode: {'Force reindex' if force else 'Incremental sync'}")

            # Show process priority
            if process_priority != "normal":
                console.print(f"  Process Priority: {process_priority}")

            if offline or os.environ.get('HF_HUB_OFFLINE') == '1':
                console.print(f"  [yellow]Mode: Offline (cached models only)[/yellow]")

            console.print()

        # Index directory
        stats = pipeline.index_directory(
            markdown_dir=markdown_dir,
            collection_name=collection,
            model_name=model,
            model_config=model_dict,
            force_reindex=force,
            verbose=verbose,
            chunk_size=model_dict['chunk_size'],
            chunk_overlap=model_dict['chunk_overlap'],
            recursive=recursive
        )

        # Output results
        if output_json:
            print(json.dumps(stats))
        elif not verbose:
            # Summary already printed by pipeline
            pass

        sys.exit(0)

    except KeyboardInterrupt:
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)
    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)


def store_command(
    file: str,
    collection: str,
    model: str,
    title: str,
    category: str,
    tags: str,
    metadata: str,
    chunk_size: int,
    chunk_overlap: int,
    verbose: bool,
    output_json: bool
):
    """Store agent-generated content for long-term memory.

    Args:
        file: Path to markdown file (or '-' for stdin)
        collection: Target collection name
        model: Embedding model to use
        title: Document title
        category: Document category
        tags: Comma-separated tags
        metadata: Additional metadata as JSON
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks
        verbose: Verbose output
        output_json: Output JSON format
    """
    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

    try:
        # Read content
        if file == '-':
            content = sys.stdin.read()
            filename = "stdin"
        else:
            file_path = Path(file)
            if not file_path.exists():
                raise ValueError(f"File does not exist: {file}")
            content = file_path.read_text()
            filename = file_path.name

        # Parse metadata
        meta = {}
        if title:
            meta['title'] = title
        if category:
            meta['category'] = category
        if tags:
            meta['tags'] = [t.strip() for t in tags.split(',')]
        if metadata:
            try:
                custom_meta = json.loads(metadata)
                meta.update(custom_meta)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON metadata: {e}")

        # Use default model config
        if model not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(DEFAULT_MODELS.keys())}")

        model_config = DEFAULT_MODELS[model]
        model_dict = {
            'chunk_size': chunk_size or model_config.chunk_size,
            'chunk_overlap': chunk_overlap or model_config.chunk_overlap,
            'vector_name': getattr(model_config, 'vector_name', None),
        }

        # Initialize clients
        from arcaneum.paths import get_models_dir
        qdrant = create_qdrant_client()
        # Use cached model for persistent model loading (arcaneum-pwd5)
        embeddings = get_cached_model(
            model_name=model,
            cache_dir=str(get_models_dir()),
            use_gpu=False
        )

        # Validate collection type
        try:
            validate_collection_type(qdrant, collection, CollectionType.MARKDOWN)
        except Exception as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            else:
                console.print(f"[red]❌ {e}[/red]")
            sys.exit(1)

        # Detect vector name from existing collection (if it exists)
        try:
            vector_names = get_vector_names(qdrant, collection)
            if vector_names:
                # Collection exists with named vectors - use first one
                model_dict['vector_name'] = vector_names[0]
        except Exception:
            # Collection doesn't exist yet, use default vector_name from model_config
            pass

        # Create pipeline (no exclude patterns needed for injection)
        pipeline = MarkdownIndexingPipeline(
            qdrant_client=qdrant,
            embedding_client=embeddings,
            batch_size=100,
        )

        # Show configuration
        if not output_json and verbose:
            console.print(f"\n[bold blue]Markdown Injection Configuration[/bold blue]")
            console.print(f"  Collection: {collection}")
            console.print(f"  Source: {filename}")
            console.print(f"  Size: {len(content)} chars")
            console.print()

        # Store content
        stats = pipeline.inject_content(
            content=content,
            collection_name=collection,
            model_name=model,
            model_config=model_dict,
            metadata=meta,
            chunk_size=model_dict['chunk_size'],
            chunk_overlap=model_dict['chunk_overlap'],
            persist=True  # Always persist for agent memory
        )

        # Output results
        if output_json:
            print(json.dumps(stats))
        else:
            console.print(f"✅ Stored {stats['chunks']} chunks")
            if stats.get('persisted') and not verbose:
                console.print(f"📁 {stats['path']}")

        sys.exit(0)

    except Exception as e:
        if output_json:
            print(json.dumps({"error": str(e)}))
        else:
            console.print(f"[red]❌ Error: {e}[/red]")
        if verbose:
            logger.exception("Full traceback:")
        sys.exit(1)
