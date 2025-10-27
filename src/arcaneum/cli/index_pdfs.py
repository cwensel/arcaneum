"""CLI command for PDF indexing (RDR-004)."""

import click
from pathlib import Path
from rich.console import Console
from rich.table import Table
import logging
import sys
import json
import os

from ..config import load_config, DEFAULT_MODELS
from ..embeddings.client import EmbeddingClient
from ..indexing.uploader import PDFBatchUploader
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
        # Verbose: Show everything
        logging.basicConfig(level=logging.DEBUG, format='%(levelname)s:%(name)s:%(message)s')
    else:
        # Normal: Clean output, only warnings and errors
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

        # Suppress all INFO logs for clean output
        logging.getLogger('arcaneum').setLevel(logging.WARNING)
        logging.getLogger('httpx').setLevel(logging.ERROR)
        logging.getLogger('qdrant_client').setLevel(logging.ERROR)
        logging.getLogger('fastembed').setLevel(logging.ERROR)

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
        )

        if not output_json:
            console.print(f"\n[bold blue]Indexing PDFs[/bold blue]")
            console.print(f"  Directory: {pdf_dir}")
            console.print(f"  Collection: {collection}")
            console.print(f"  Model: {model}")
            if ocr_enabled:
                console.print(f"  OCR: {ocr_language}")
            if offline:
                console.print(f"  [yellow]Mode: Offline (cached models only)[/yellow]")
            console.print()

        # Index PDFs
        stats = uploader.index_directory(
            pdf_dir=pdf_dir,
            collection_name=collection,
            model_name=model,
            model_config=model_dict,
            force_reindex=force
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
