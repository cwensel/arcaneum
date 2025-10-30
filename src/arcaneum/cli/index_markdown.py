"""CLI commands for markdown indexing (RDR-014)."""

import click
from pathlib import Path
from rich.console import Console
import logging
import sys
import json
import os
import signal

# Suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from ..config import load_config, DEFAULT_MODELS
from ..embeddings.client import EmbeddingClient
from ..indexing.markdown.pipeline import MarkdownIndexingPipeline
from ..indexing.collection_metadata import validate_collection_type, CollectionType, get_vector_names
from qdrant_client import QdrantClient

console = Console()
logger = logging.getLogger(__name__)


def index_markdown_command(
    path: str,
    collection: str,
    model: str,
    chunk_size: int,
    chunk_overlap: int,
    recursive: bool,
    exclude: tuple,
    qdrant_url: str,
    force: bool,
    offline: bool,
    verbose: bool,
    output_json: bool
):
    """Index markdown files to Qdrant collection.

    Args:
        path: Directory containing markdown files
        collection: Target collection name
        model: Embedding model to use
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        force: Force reindex all files
        offline: Use cached models only (no network calls)
        verbose: Verbose output
        output_json: Output JSON format
    """
    # Enable offline mode if requested
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # Setup logging
    if verbose:
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('qdrant_client').setLevel(logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
        logging.getLogger('arcaneum').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('qdrant_client').setLevel(logging.ERROR)
        logging.getLogger('fastembed').setLevel(logging.WARNING)

    # Allow HuggingFace/transformers download progress to show through
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('huggingface_hub').setLevel(logging.WARNING)

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        markdown_dir = Path(path)
        if not markdown_dir.exists():
            raise ValueError(f"Path does not exist: {path}")

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
        qdrant = QdrantClient(url=qdrant_url)
        embeddings = EmbeddingClient(cache_dir=str(get_models_dir()))

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

        # Create pipeline with custom exclude patterns
        pipeline = MarkdownIndexingPipeline(
            qdrant_client=qdrant,
            embedding_client=embeddings,
            batch_size=100,
            exclude_patterns=exclude_patterns,
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
            console.print(f"  Chunk size: {model_dict['chunk_size']} tokens")
            console.print(f"  Chunk overlap: {model_dict['chunk_overlap']} tokens")
            console.print(f"  Mode: {'Force reindex' if force else 'Incremental sync'}")
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
        qdrant = QdrantClient(url='http://localhost:6333')
        embeddings = EmbeddingClient(cache_dir=str(get_models_dir()))

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
