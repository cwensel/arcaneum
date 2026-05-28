"""CLI command for PDF indexing (RDR-004)."""

from pathlib import Path
from rich.console import Console
from rich.table import Table
import logging
import sys
import json
import os
import signal

from .interaction_logger import interaction_logger
from .logging_config import setup_logging_default, setup_logging_verbose, setup_logging_debug
from .utils import set_process_priority, create_qdrant_client
from ..config import DEFAULT_MODELS
from ..embeddings.model_cache import get_cached_model
from ..indexing.uploader import PDFBatchUploader
from ..indexing.collection_metadata import (
    validate_collection_type,
    CollectionType,
    MultiRootPruneError,
)
from .errors import EXIT_INVALID_ARGS

console = Console()
logger = logging.getLogger(__name__)


def index_pdfs_command(
    path: str,
    from_file: str,
    collection: str,
    model: str,
    embedding_batch_size: int,
    no_ocr: bool,
    ocr_language: str,
    ocr_workers: int,
    normalize_only: bool,
    preserve_images: bool,
    process_priority: str,
    not_nice: bool,
    force: bool,
    prune: bool,
    no_gpu: bool,
    offline: bool,
    randomize: bool,
    verify: bool,
    streaming: bool,
    verbose: bool,
    debug: bool,
    output_json: bool
):
    """Index PDF files to Qdrant collection.

    Args:
        path: Directory containing PDF files (or None if using from_file)
        from_file: Path to file containing list of PDF paths, or "-" for stdin
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
        not_nice: Disable process priority reduction for worker processes
        force: Force reindex all files
        no_gpu: Disable GPU acceleration (use CPU only)
        offline: Use cached models only (no network calls)
        verify: Verify collection integrity after indexing
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
    set_process_priority(process_priority, disable_worker_nice=not_nice)

    # Auto-detect optimal settings
    # Note: File workers is hardcoded to 1 due to embedding lock (arcaneum-6pvk)
    # Embedding generation is serialized when file_workers > 1 to prevent GPU conflicts
    actual_file_workers = 1  # Serialized by embedding lock (arcaneum-3fs3)

    # GPU models ignore embedding_workers (single-threaded is faster)
    # CPU models use ThreadPoolExecutor, but benefit is limited
    actual_embedding_workers = 1  # GPU: single-threaded optimal, CPU: ignored in lock

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

    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "index", "pdf",
        collection=collection,
        path=path,
        from_file=from_file,
        force=force,
    )

    try:
        # Handle file list if provided
        file_list = None
        pdf_dir = None

        if from_file:
            from .utils import read_file_list
            file_list = read_file_list(from_file, allowed_extensions={'.pdf'})
            if not file_list:
                raise ValueError("No valid PDF files found in the provided list")
            # Use parent directory of first file as base directory for reporting
            pdf_dir = file_list[0].parent
        else:
            pdf_dir = Path(path)
            if not pdf_dir.exists():
                raise ValueError(f"Path does not exist: {path}")

        # Initialize Qdrant client ONCE and reuse throughout indexing (connection pooling optimization)
        # This avoids TCP connection overhead for each operation
        qdrant = create_qdrant_client()

        # Retrieve model from collection metadata
        from arcaneum.paths import get_models_dir

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
            'char_to_token_ratio': model_config.char_to_token_ratio,
            'late_chunking': model_config.late_chunking,
        }

        # Initialize embedding client with persistent model caching (arcaneum-pwd5)
        # get_cached_model ensures models are cached for the process lifetime,
        # saving 7-8 seconds on subsequent CLI invocations within the same session
        embeddings = get_cached_model(
            model_name=model,
            cache_dir=str(get_models_dir()),
            use_gpu=not no_gpu
        )

        # Validate collection type (must be 'pdf' or untyped)
        try:
            validate_collection_type(qdrant, collection, CollectionType.PDF)
        except Exception as e:
            if output_json:
                print(json.dumps({"error": str(e)}))
            else:
                console.print(f"[red]❌ {e}[/red]")
            sys.exit(1)

        # Auto-tune batch size if not explicitly set by user
        if embedding_batch_size is None:
            if not no_gpu:
                # GPU mode: auto-tune based on available memory
                from arcaneum.utils.memory import get_gpu_memory_info, estimate_safe_batch_size_v2

                available_bytes, total_bytes, device_type = get_gpu_memory_info()

                if available_bytes is not None:
                    # Calculate optimal batch size
                    # Note: pipeline_overhead_gb is minimal (0.3GB) because PDF processing is CPU-based
                    auto_batch_size = estimate_safe_batch_size_v2(
                        model_name=model,
                        available_gpu_bytes=available_bytes,
                        pipeline_overhead_gb=0.3,  # Minimal GPU overhead (PDF extraction/chunking use CPU)
                        safety_factor=0.6,
                        device_type=device_type  # Pass device type for MPS vs CUDA logic
                    )

                    if not output_json:
                        available_gb = available_bytes / (1024 ** 3)
                        total_gb = total_bytes / (1024 ** 3)
                        console.print(
                            f"[blue]🔧 Auto-tuned batch size: {auto_batch_size} "
                            f"(GPU: {available_gb:.1f}GB / {total_gb:.1f}GB total, {device_type.upper()})[/blue]"
                        )

                    embedding_batch_size = auto_batch_size
                else:
                    # GPU memory detection failed, use fallback
                    embedding_batch_size = 128
                    if not output_json:
                        console.print(
                            "[yellow]⚠️  GPU memory detection failed, using default batch size: 128[/yellow]"
                        )
            else:
                # CPU mode: use conservative default
                embedding_batch_size = 128
        else:
            # User explicitly set batch size - respect it but check if it seems risky
            if not no_gpu:
                from arcaneum.utils.memory import get_gpu_memory_info, estimate_safe_batch_size_v2

                available_bytes, total_bytes, device_type = get_gpu_memory_info()

                if available_bytes is not None:
                    safe_batch_size = estimate_safe_batch_size_v2(
                        model_name=model,
                        available_gpu_bytes=available_bytes,
                        pipeline_overhead_gb=0.3,  # Minimal GPU overhead
                        safety_factor=0.6,
                        device_type=device_type  # Pass device type for MPS vs CUDA logic
                    )

                    if embedding_batch_size > safe_batch_size and not output_json:
                        available_gb = available_bytes / (1024 ** 3)
                        console.print(
                            f"\n[yellow]⚠️  WARNING: Batch size {embedding_batch_size} may exceed available GPU memory[/yellow]"
                        )
                        console.print(f"   GPU: {device_type.upper()}, Available: {available_gb:.1f}GB")
                        console.print(f"   Recommended batch size: {safe_batch_size}")
                        console.print(f"   Consider: --embedding-batch-size {safe_batch_size} or --no-gpu\n")

        # Create uploader with file parallelism (arcaneum-108, RDR-016)
        uploader = PDFBatchUploader(
            qdrant_client=qdrant,
            embedding_client=embeddings,
            batch_size=300,  # Optimized from 100 (arcaneum-6pvk: reduce upload rate)
            parallel_workers=2,  # Reduced from 4 (arcaneum-6pvk: reduce connection pressure)
            ocr_enabled=ocr_enabled,
            ocr_language=ocr_language,
            ocr_threshold=100,
            ocr_workers=ocr_workers,
            embedding_workers=actual_embedding_workers,
            embedding_batch_size=embedding_batch_size,
            file_workers=actual_file_workers,  # PDF file parallelism (arcaneum-6pvk: hardcoded to 1)
            pdf_timeout=600,  # 10 minute timeout per PDF
            ocr_page_timeout=60,  # 1 minute timeout per OCR page
            embedding_timeout=300,  # 5 minute timeout for embeddings
            markdown_conversion=not normalize_only,  # RDR-016: markdown by default
            preserve_images=preserve_images,  # RDR-016: images off by default
            streaming=streaming,  # Stream embeddings to Qdrant immediately
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
            console.print(f"  File processing: {actual_file_workers} workers")
            console.print(f"  Embedding: {actual_embedding_workers} workers, batch size {embedding_batch_size}")

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

        # Capture indexed paths BEFORE indexing so we can detect orphans
        # (indexed files no longer on disk) on a force, full-directory run.
        pre_run_paths = set()
        if force and file_list is None:
            pre_run_paths = uploader.sync._get_indexed_file_paths_set(collection)

        # Index PDFs
        stats = uploader.index_directory(
            pdf_dir=pdf_dir,
            collection_name=collection,
            model_name=model,
            model_config=model_dict,
            force_reindex=force,
            randomize=randomize,
            verbose=verbose,
            file_list=file_list
        )

        # Orphan-aware prompt-policy stamp gate (C3/C4)
        if force and file_list is None:
            from ..indexing.collection_metadata import prune_orphans_and_stamp

            # An orphan is an indexed file that no longer exists on disk. Use
            # existence, not this run's discovery scope, so a scope-limited run
            # does NOT treat still-existing indexed files outside the current
            # scan as orphans and delete them. Mirrors the source path's
            # symmetric, existence-based on_disk set.
            on_disk_paths = {p for p in pre_run_paths if Path(p).exists()}
            # Files this run actually re-embedded (its discovery scope, always
            # recursive for PDFs). Any still-existing indexed PDF not covered
            # bars certification (stale vectors).
            covered_paths = {str(p.absolute()) for p in pdf_dir.rglob("*.pdf")}
            prune_warn = (
                None
                if output_json
                else (lambda m: console.print(f"[yellow]⚠ {m}[/yellow]"))
            )
            prune_orphans_and_stamp(
                qdrant=qdrant,
                sync=uploader.sync,
                collection_name=collection,
                collection_type=CollectionType.PDF,
                model=model,
                force=force,
                file_list=file_list,
                stats=stats,
                on_disk_paths=on_disk_paths,
                pre_run_paths=pre_run_paths,
                prune=prune,
                indexed_dir=str(pdf_dir.absolute()),
                covered_paths=covered_paths,
                warn=prune_warn,
            )

        # Post-verify if requested
        verification_result = None
        if verify:
            from arcaneum.indexing.verify import CollectionVerifier

            if not output_json:
                console.print("\n[dim]Verifying collection integrity...[/dim]")

            verifier = CollectionVerifier(qdrant)
            verification_result = verifier.verify_collection(collection, verbose=verbose)

            if verification_result.is_healthy:
                if not output_json:
                    console.print(f"[green]✓ Collection verified - all {verification_result.complete_items} files complete[/green]")
            else:
                incomplete = verification_result.get_items_needing_repair()
                if not output_json:
                    console.print(f"[yellow]⚠ Found {len(incomplete)} incomplete files[/yellow]")
                    for item in incomplete[:5]:
                        console.print(f"  [yellow]{item}[/yellow]")
                    if len(incomplete) > 5:
                        console.print(f"  [dim]... and {len(incomplete) - 5} more[/dim]")
                    console.print("[dim]Re-run with --force to repair incomplete files[/dim]")

            stats["verification"] = {
                "is_healthy": verification_result.is_healthy,
                "total_files": verification_result.total_items,
                "complete_files": verification_result.complete_items,
                "incomplete_files": verification_result.incomplete_items,
            }

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

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=stats.get('files', 0),
            chunks=stats.get('chunks', 0),
            errors=stats.get('errors', 0),
        )

    except KeyboardInterrupt:
        interaction_logger.finish(error="interrupted by user")
        console.print("\n\nIndexing interrupted by user")
        sys.exit(130)

    except MultiRootPruneError as e:
        # A refused --prune on a multi-root collection is an invalid-argument
        # condition, not an unexpected failure: exit with the invalid-args code.
        interaction_logger.finish(error=str(e))
        if output_json:
            print(json.dumps({"success": False, "error": str(e)}, indent=2))
        else:
            console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(EXIT_INVALID_ARGS)

    except Exception as e:
        interaction_logger.finish(error=str(e))
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
