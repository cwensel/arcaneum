"""CLI command for PDF indexing (RDR-004)."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
import logging
import sys
import json
import os
import signal

# Suppress tokenizers parallelism warning
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from ..config import load_config, DEFAULT_MODELS
from ..embeddings.client import EmbeddingClient
from ..indexing.uploader import PDFBatchUploader
from ..indexing.collection_metadata import validate_collection_type, CollectionType
from qdrant_client import QdrantClient

console = Console()
logger = logging.getLogger(__name__)


def index_pdfs_command(
    path: str,
    collection: str,
    model: str,
    workers: int,
    ocr_enabled: bool,
    ocr_language: str,
    force: bool,
    batch_across_files: bool,
    offline: bool,
    verbose: bool,
    output_json: bool
):
    """Index PDF files to Qdrant collection.

    Args:
        path: Directory containing PDF files
        collection: Target collection name
        model: Embedding model to use
        workers: Number of parallel workers
        ocr_enabled: Enable OCR for scanned PDFs
        ocr_language: OCR language code
        force: Force reindex all files
        offline: Use cached models only (no network calls)
        verbose: Verbose output
        output_json: Output JSON format
    """
    # Enable offline mode if requested (blocks all HuggingFace network calls)
    if offline:
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # Setup logging
    if verbose:
        # Verbose: Show INFO but not DEBUG (too noisy)
        logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
        # Suppress DEBUG from libraries
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('httpcore').setLevel(logging.WARNING)
        logging.getLogger('qdrant_client').setLevel(logging.INFO)
    else:
        # Normal: Clean output, only warnings and errors
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

        # Suppress all INFO logs for clean output
        logging.getLogger('arcaneum').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('qdrant_client').setLevel(logging.ERROR)
        # Keep fastembed at WARNING to allow download progress bars
        logging.getLogger('fastembed').setLevel(logging.WARNING)

    # Set up signal handler for Ctrl-C
    def signal_handler(sig, frame):
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    signal.signal(signal.SIGINT, signal_handler)

    try:
        pdf_dir = Path(path)
        if not pdf_dir.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Use default model config
        if model not in DEFAULT_MODELS:
            raise ValueError(f"Unknown model: {model}. Available: {list(DEFAULT_MODELS.keys())}")

        model_config = DEFAULT_MODELS[model]
        model_dict = {
            'chunk_size': model_config.chunk_size,
            'chunk_overlap': model_config.chunk_overlap,
            'char_to_token_ratio': 3.3,
            'late_chunking': model in ['stella', 'modernbert', 'jina'],  # bge doesn't support
        }

        # Initialize clients
        qdrant = QdrantClient(url="http://localhost:6333")
        embeddings = EmbeddingClient(cache_dir="./models_cache")

        # Validate collection type (must be 'pdf' or untyped)
        try:
            validate_collection_type(qdrant, collection, CollectionType.PDF)
        except Exception as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            else:
                console.print(f"[red]❌ {e}[/red]")
            sys.exit(1)

        # Create uploader
        uploader = PDFBatchUploader(
            qdrant_client=qdrant,
            embedding_client=embeddings,
            batch_size=100,
            parallel_workers=workers,
            max_retries=5,
            ocr_enabled=ocr_enabled,
            ocr_engine='tesseract',
            ocr_language=ocr_language,
            ocr_threshold=100,
            batch_across_files=batch_across_files,
        )

        # Show configuration at start
        if not output_json:
            # Get actual model name being used
            from arcaneum.embeddings.client import EMBEDDING_MODELS
            actual_model = EMBEDDING_MODELS.get(model, {}).get('name', model)
            model_desc = EMBEDDING_MODELS.get(model, {}).get('description', '')

            console.print(f"\n[bold blue]PDF Indexing Configuration[/bold blue]")
            console.print(f"  Collection: {collection} (type: pdf)")
            if model_desc:
                console.print(f"  Model: {model} → {actual_model}")
                console.print(f"    ({model_desc})")
            else:
                console.print(f"  Embedding: {actual_model}")
            if ocr_enabled:
                console.print(f"  OCR: tesseract ({ocr_language})")
            console.print(f"  Pipeline: PDF → Extract → [OCR if needed] → Chunk → Embed → Upload")
            if batch_across_files:
                console.print(f"  Upload: Batched across files (100 chunks)")
            else:
                console.print(f"  Upload: Atomic per-document (safer)")
            if offline:
                console.print(f"  [yellow]Mode: Offline (cached models only)[/yellow]")
            console.print()

        # Index PDFs
        stats = uploader.index_directory(
            pdf_dir=pdf_dir,
            collection_name=collection,
            model_name=model,
            model_config=model_dict,
            force_reindex=force,
            verbose=verbose
        )

        # Output results
        if output_json:
            result = {
                "success": True,
                "collection": collection,
                "model": model,
                "stats": stats
            }
            print(json.dumps(result, indent=2))
        else:
            # Minimal output by default (matches index-source style)
            if not verbose:
                console.print(f"\n✓ Indexed {stats['files']} PDF(s): {stats['chunks']} chunks")
                if stats['errors'] > 0:
                    console.print(f"⚠ {stats['errors']} errors occurred")
            else:
                # Verbose: Show detailed table
                console.print("\n[bold green]✓ Indexing Complete[/bold green]")

                table = Table(title="Indexing Results")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="magenta")

                table.add_row("Files Processed", str(stats['files']))
                table.add_row("Chunks Uploaded", str(stats['chunks']))
                table.add_row("Errors", str(stats['errors']))

                console.print(table)

                if stats['errors'] > 0:
                    console.print(f"\n[yellow]⚠ {stats['errors']} errors occurred[/yellow]")

    except KeyboardInterrupt:
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    except Exception as e:
        if output_json:
            result = {
                "success": False,
                "error": str(e)
            }
            print(json.dumps(result, indent=2))
            sys.exit(1)
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
            sys.exit(1)
