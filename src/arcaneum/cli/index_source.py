"""CLI command for source code indexing (RDR-005)."""

import sys
import os
import logging
import signal
from typing import Optional

# Suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from rich.console import Console
from rich import print as rprint

from arcaneum.indexing.source_code_pipeline import SourceCodeIndexer
from arcaneum.indexing.qdrant_indexer import QdrantIndexer, create_qdrant_client
from arcaneum.indexing.collection_metadata import (
    validate_collection_type,
    set_collection_metadata,
    get_vector_names,
    CollectionType
)

console = Console()


def index_source_command(
    path: str,
    collection: str,
    model: str,
    workers: int,
    depth: Optional[int],
    force: bool,
    verbose: bool,
    output_json: bool
):
    """Index source code to Qdrant collection (from RDR-005).

    Args:
        path: Directory containing git repositories
        collection: Target collection name
        model: Embedding model (jina-code, jina-v2-code, stella)
        workers: Parallel workers (not yet implemented)
        depth: Git discovery depth (None = unlimited)
        force: Force reindex all projects
        verbose: Verbose output
        output_json: Output JSON format
    """
    # Setup logging - minimal output by default
    if verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(levelname)s:%(name)s:%(message)s'
        )
    else:
        # Default: Only warnings and errors, no INFO logs
        logging.basicConfig(
            level=logging.WARNING,
            format='%(levelname)s: %(message)s'
        )
        # Suppress all library INFO logs
        logging.getLogger('arcaneum').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('qdrant_client').setLevel(logging.ERROR)
        logging.getLogger('fastembed').setLevel(logging.ERROR)

    logger = logging.getLogger(__name__)

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        # Connect to Qdrant (use defaults)
        qdrant_url = 'localhost'
        qdrant_port = 6333
        qdrant_grpc_port = 6334

        if verbose:
            console.print(f"[cyan]Connecting to Qdrant:[/cyan] {qdrant_url}:{qdrant_port}")

        qdrant_client = create_qdrant_client(
            url=qdrant_url,
            port=qdrant_port,
            grpc_port=qdrant_grpc_port,
            prefer_grpc=True
        )

        qdrant_indexer = QdrantIndexer(qdrant_client)

        # Check/create collection and determine vector name
        if not qdrant_indexer.collection_exists(collection):
            if verbose:
                console.print(f"[yellow]Collection '{collection}' does not exist, creating...[/yellow]")

            # Map model names to FastEmbed models for NEW collections
            model_map = {
                'jina-code': 'BAAI/bge-small-en-v1.5',      # 384D - use bge-small as placeholder
                'jina-v2-code': 'BAAI/bge-small-en-v1.5',   # 384D
                'stella': 'BAAI/bge-large-en-v1.5',         # 1024D
                'bge': 'BAAI/bge-large-en-v1.5',            # 1024D
            }
            embedding_model = model_map.get(model, 'BAAI/bge-small-en-v1.5')

            # Determine vector size
            vector_sizes = {
                'sentence-transformers/all-MiniLM-L6-v2': 384,
                'BAAI/bge-small-en-v1.5': 384,
                'BAAI/bge-large-en-v1.5': 1024,
            }
            vector_size = vector_sizes.get(embedding_model, 384)

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

                # Map vector name back to FastEmbed model
                # Must match EMBEDDING_MODELS dimensions in embeddings/client.py
                vector_to_model_map = {
                    'bge': 'BAAI/bge-large-en-v1.5',        # 1024D
                    'stella': 'BAAI/bge-large-en-v1.5',     # 1024D (stella uses bge-large)
                    'jina-code': 'BAAI/bge-small-en-v1.5',  # 384D placeholder
                    'jina': 'BAAI/bge-base-en-v1.5',        # 768D (jina uses bge-base as placeholder)
                }
                embedding_model = vector_to_model_map.get(vector_name, 'sentence-transformers/all-MiniLM-L6-v2')
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

        # Create indexer
        indexer = SourceCodeIndexer(
            qdrant_indexer=qdrant_indexer,
            embedding_model=embedding_model,
            chunk_size=400,  # 400 tokens for 8K context models
            vector_name=vector_name  # Use auto-detected or specified vector name
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
