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

from .logging_config import setup_logging_default, setup_logging_verbose, setup_logging_debug
from .utils import set_process_priority, create_qdrant_client
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
    file_workers: int,
    file_worker_mult: float,
    embedding_workers: int,
    embedding_worker_mult: float,
    embedding_batch_size: int,
    no_ocr: bool,
    ocr_language: str,
    ocr_workers: int,
    normalize_only: bool,
    preserve_images: bool,
    process_priority: str,
    max_perf: bool,
    force: bool,
    batch_across_files: bool,
    no_gpu: bool,
    offline: bool,
    verbose: bool,
    debug: bool,
    output_json: bool
):
    """Index PDF files to Qdrant collection.

    Args:
        path: Directory containing PDF files
        collection: Target collection name
        model: Embedding model to use
        workers: Number of parallel workers
        embedding_workers: Number of parallel workers for embedding generation
        embedding_batch_size: Batch size for embedding generation
        no_ocr: Disable OCR (enabled by default)
        ocr_language: OCR language code
        ocr_workers: Number of parallel OCR workers (None = cpu_count)
        normalize_only: Skip markdown conversion, only normalize whitespace (RDR-016)
        preserve_images: Extract images for multimodal search (RDR-016)
        process_priority: Process scheduling priority (low, normal, high)
        force: Force reindex all files
        batch_across_files: Batch uploads across files
        no_gpu: Disable GPU acceleration (use CPU only)
        offline: Use cached models only (no network calls)
        verbose: Verbose output
        debug: Debug mode (show all library warnings)
        output_json: Output JSON format

    Note:
        For corporate networks with SSL issues, set environment variables:
        - HF_HUB_OFFLINE=1 (offline mode)
        - PYTHONHTTPSVERIFY=0 (disable SSL verification)
        See docs/testing/offline-mode.md for details.
    """
    # Invert no_ocr flag to get ocr_enabled
    ocr_enabled = not no_ocr

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

    # Compute file_workers with precedence: absolute → multiplier → default (1)
    from multiprocessing import cpu_count
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

    # Compute embedding_workers with precedence: absolute → multiplier → default (0.5)
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

    # Enable offline mode if requested (blocks all HuggingFace network calls)
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
        pdf_dir = Path(path)
        if not pdf_dir.exists():
            raise ValueError(f"Path does not exist: {path}")

        # Initialize Qdrant client early to retrieve model from collection metadata
        from arcaneum.paths import get_models_dir
        qdrant = create_qdrant_client()

        # Retrieve model from collection metadata if not provided
        if model is None:
            from arcaneum.indexing.collection_metadata import get_collection_metadata
            metadata = get_collection_metadata(qdrant, collection)
            if not metadata or 'model' not in metadata:
                raise ValueError(
                    f"Collection '{collection}' has no model metadata. "
                    "Please create the collection with 'arc collection create --type pdf' first."
                )
            model = metadata['model']
        else:
            # Warn about deprecated --model flag
            console.print(
                "[yellow]⚠️  Warning: --model flag is deprecated. "
                "Model is now set at collection creation time. "
                "Please use 'arc collection create --type pdf' instead.[/yellow]"
            )

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

        # Initialize embedding client
        embeddings = EmbeddingClient(cache_dir=str(get_models_dir()), use_gpu=not no_gpu)

        # Validate collection type (must be 'pdf' or untyped)
        try:
            validate_collection_type(qdrant, collection, CollectionType.PDF)
        except Exception as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            else:
                console.print(f"[red]❌ {e}[/red]")
            sys.exit(1)

        # Create uploader with file parallelism (arcaneum-108, RDR-016)
        uploader = PDFBatchUploader(
            qdrant_client=qdrant,
            embedding_client=embeddings,
            batch_size=100,
            parallel_workers=4,  # Upload parallelism (implementation detail)
            max_retries=5,
            ocr_enabled=ocr_enabled,
            ocr_engine='tesseract',
            ocr_language=ocr_language,
            ocr_threshold=100,
            ocr_workers=ocr_workers,
            embedding_workers=actual_embedding_workers,
            embedding_batch_size=embedding_batch_size,
            batch_across_files=batch_across_files,
            file_workers=actual_file_workers,  # PDF file parallelism
            pdf_timeout=600,  # 10 minute timeout per PDF
            ocr_page_timeout=60,  # 1 minute timeout per OCR page
            embedding_timeout=300,  # 5 minute timeout for embeddings
            markdown_conversion=not normalize_only,  # RDR-016: markdown by default
            preserve_images=preserve_images,  # RDR-016: images off by default
        )

        # Pre-load model to avoid "hang" during first file processing (similar to markdown indexing)
        if not output_json:
            # Check if model is cached
            is_cached = embeddings.is_model_cached(model)
            if not is_cached:
                console.print(f"⬇️  Downloading {model} model (first time only)...", style="yellow")
            else:
                console.print(f"📦 Loading {model} model from cache...", style="blue")

        # Load model now (separate phase from file processing)
        embeddings.get_model(model)

        if not output_json:
            console.print()

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

            # Show GPU/CPU info
            device_info = embeddings.get_device_info()
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

            # Show extraction strategy (RDR-016)
            if normalize_only:
                console.print(f"  Extraction: Normalization-only (47-48% token savings, no structure)")
            else:
                console.print(f"  Extraction: Markdown conversion (quality-first, semantic structure)")

            if preserve_images:
                console.print(f"  Images: Preserved for multimodal search")

            if ocr_enabled:
                from multiprocessing import cpu_count
                workers_display = ocr_workers if ocr_workers else cpu_count()
                console.print(f"  OCR: tesseract ({ocr_language}, {workers_display} parallel workers)")
            else:
                console.print(f"  OCR: disabled")
            console.print(f"  Pipeline: PDF → Extract → [OCR if needed] → Chunk → Embed → Upload")
            if batch_across_files:
                console.print(f"  Upload: Batched across files (100 chunks)")
            else:
                console.print(f"  Upload: Atomic per-document (safer)")

            # Show process priority
            if process_priority != "normal":
                console.print(f"  Process Priority: {process_priority}")

            if offline:
                console.print(f"  [yellow]Mode: Offline (cached models only)[/yellow]")
            # Check if offline mode set via environment
            elif os.environ.get('HF_HUB_OFFLINE') == '1':
                console.print(f"  [yellow]Mode: Offline (HF_HUB_OFFLINE=1)[/yellow]")
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
