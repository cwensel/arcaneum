"""Directory sync command for dual indexing (RDR-009).

This module implements the 'corpus sync' command that indexes documents
to both Qdrant and MeiliSearch in a single operation.
"""

import gc
import logging
import multiprocessing as mp
import os

# Suppress tokenizers fork warning - must be set before any tokenizers import
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import sys
import hashlib
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import List, Optional, Set, Dict, Any, Tuple
from uuid import uuid4

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from ..cli.output import print_json, print_error, print_info
from ..cli.utils import create_qdrant_client
from ..cli.interaction_logger import interaction_logger
from ..cli.errors import InvalidArgumentError, ResourceNotFoundError
from ..embeddings.client import EmbeddingClient, EMBEDDING_MODELS
from ..fulltext.client import FullTextClient
from ..schema.document import DualIndexDocument
from ..indexing.dual_indexer import DualIndexer
from ..indexing.collection_metadata import get_collection_type, get_collection_metadata
from ..indexing.common.sync import MetadataBasedSync, compute_quick_hash
from ..indexing.git_operations import GitProjectDiscovery
from ..config import DEFAULT_MODELS

console = Console()
logger = logging.getLogger(__name__)


def _chunk_code_file_worker(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Process a single code file: read and chunk using AST.

    Module-level function for ProcessPoolExecutor pickling.

    Args:
        file_path: Path to source code file
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        Tuple of (file_path, list of chunk dicts, error or None)
    """
    # Lower process priority to avoid starving main process
    if os.environ.get('ARCANEUM_DISABLE_WORKER_NICE') != '1':
        try:
            if hasattr(os, 'nice'):
                os.nice(10)
        except Exception:
            pass

    try:
        from ..indexing.ast_chunker import ASTCodeChunker

        # Read file
        file_p = Path(file_path)
        try:
            code = file_p.read_text(encoding='utf-8', errors='replace')
        except Exception as e:
            return (file_path, [], f"Read error: {e}")

        if not code.strip():
            return (file_path, [], None)

        # Determine language from extension
        ext_to_lang = {
            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
            '.java': 'java', '.go': 'go', '.rs': 'rust', '.rb': 'ruby',
            '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
        }
        language = ext_to_lang.get(file_p.suffix.lower(), 'unknown')

        # Chunk using AST
        chunker = ASTCodeChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = chunker.chunk_code(file_path, code)

        # Convert to serializable dicts
        chunk_dicts = []
        for i, chunk in enumerate(chunks):
            chunk_dicts.append({
                'text': chunk.content,
                'metadata': {
                    'file_path': file_path,
                    'filename': file_p.name,
                    'language': language,
                    'chunk_index': i,
                    'method': chunk.method,
                }
            })

        return (file_path, chunk_dicts, None)

    except Exception as e:
        return (file_path, [], str(e))


def get_meili_client() -> FullTextClient:
    """Get MeiliSearch client from environment or auto-generated key."""
    from ..paths import get_meilisearch_api_key

    url = os.environ.get('MEILISEARCH_URL', 'http://localhost:7700')
    api_key = get_meilisearch_api_key()
    return FullTextClient(url, api_key)


def discover_files(
    directory: Path,
    file_types: Optional[str],
    corpus_type: str
) -> List[Path]:
    """Discover files to index based on corpus type and file filters.

    Args:
        directory: Directory to scan
        file_types: Comma-separated file extensions (e.g., ".py,.js")
        corpus_type: Type of corpus (pdf, code, markdown)

    Returns:
        List of file paths to index
    """
    # Determine extensions to look for
    if file_types:
        extensions = set(ext.strip().lower() for ext in file_types.split(','))
        # Ensure extensions start with '.'
        extensions = set(e if e.startswith('.') else f'.{e}' for e in extensions)
    else:
        # Default extensions based on corpus type
        type_extensions = {
            "pdf": {".pdf"},
            "code": {".py", ".js", ".ts", ".java", ".go", ".rs", ".rb", ".cpp", ".c", ".h", ".hpp"},
            "markdown": {".md", ".markdown"},
        }
        extensions = type_extensions.get(corpus_type, set())

    if not extensions:
        logger.warning(f"No file extensions defined for corpus type: {corpus_type}")
        return []

    # Discover files
    files = []
    for ext in extensions:
        pattern = f"*{ext}"  # rglob already handles recursive search
        found = list(directory.rglob(pattern))
        files.extend(found)

    # Sort for consistent ordering
    files.sort()
    return files


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file for change detection."""
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:16]  # First 16 chars is enough


def chunk_pdf_file(file_path: Path, model_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chunk a PDF file using existing PDF chunking logic.

    Args:
        file_path: Path to PDF file
        model_config: Model configuration with chunk_size, etc.

    Returns:
        List of chunk dicts with 'text' and 'metadata'
    """
    from ..indexing.pdf.chunker import PDFChunker
    from ..indexing.pdf.extractor import PDFExtractor

    # Extract text from PDF using PDFExtractor class
    extractor = PDFExtractor()
    text, metadata = extractor.extract(file_path)

    if not text or not text.strip():
        logger.warning(f"No text extracted from {file_path}")
        return []

    # Create chunker and chunk the text
    chunker = PDFChunker(model_config)
    base_metadata = {
        'file_path': str(file_path),
        'filename': file_path.name,
        'page_boundaries': metadata.get('page_boundaries', []),
    }

    chunks = chunker.chunk(text, base_metadata)

    return [{'text': c.text, 'metadata': c.metadata} for c in chunks]


def chunk_markdown_file(file_path: Path, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Chunk a markdown file using semantic markdown chunking.

    Args:
        file_path: Path to markdown file
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of chunk dicts with 'text' and 'metadata'
    """
    from ..indexing.markdown.chunker import SemanticMarkdownChunker

    text = file_path.read_text(encoding='utf-8', errors='replace')
    if not text.strip():
        return []

    chunker = SemanticMarkdownChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    base_metadata = {
        'file_path': str(file_path),
        'filename': file_path.name,
    }

    chunks = chunker.chunk(text, base_metadata)
    return [{'text': c.text, 'metadata': c.metadata} for c in chunks]


def chunk_code_file(file_path: Path, chunk_size: int, chunk_overlap: int) -> List[Dict[str, Any]]:
    """Chunk a source code file using AST-aware chunking.

    Args:
        file_path: Path to source file
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens

    Returns:
        List of chunk dicts with 'text' and 'metadata'
    """
    from ..indexing.ast_chunker import ASTCodeChunker

    text = file_path.read_text(encoding='utf-8', errors='replace')
    if not text.strip():
        return []

    # Determine language from extension
    ext_to_lang = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
    }
    language = ext_to_lang.get(file_path.suffix.lower(), 'unknown')

    chunker = ASTCodeChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = chunker.chunk(text, language)

    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            'text': chunk.content,
            'metadata': {
                'file_path': str(file_path),
                'filename': file_path.name,
                'language': language,
                'chunk_index': i,
                'method': chunk.method,
            }
        })

    return result


def sync_directory_command(
    corpus: str,
    directories: tuple,
    models: str,
    file_types: Optional[str],
    force: bool,
    verify: bool,
    text_workers: Optional[int],
    verbose: bool,
    output_json: bool
):
    """Sync directories to both Qdrant and MeiliSearch.

    This implements the second command of the 2-command workflow:
    1. corpus create - creates both systems
    2. corpus sync (this command) - indexes documents to both systems

    Args:
        corpus: Corpus name (must exist)
        directories: Tuple of directories to sync
        models: Comma-separated list of embedding models
        file_types: File extensions to index (e.g., ".py,.js")
        force: If True, reindex all files (bypass change detection)
        verify: If True, verify collection integrity after indexing
        text_workers: Number of parallel workers for code chunking (None=auto, 0/1=sequential)
        verbose: If True, show detailed progress
        output_json: If True, output JSON format
    """
    # Calculate effective text workers
    if text_workers is None:
        effective_text_workers = max(1, cpu_count() // 2)
    elif text_workers <= 1:
        effective_text_workers = 1  # Sequential
    else:
        effective_text_workers = text_workers
    # Start interaction logging (RDR-018)
    interaction_logger.start(
        "corpus", "sync",
        corpus=corpus,
        directories=list(directories),
        models=models,
        file_types=file_types,
    )

    try:
        # Validate and resolve all directories
        dir_paths = []
        for directory in directories:
            dir_path = Path(directory).resolve()
            if not dir_path.exists():
                raise InvalidArgumentError(f"Directory not found: {directory}")
            if not dir_path.is_dir():
                raise InvalidArgumentError(f"Not a directory: {directory}")
            dir_paths.append(dir_path)

        if not output_json:
            if len(dir_paths) == 1:
                print_info(f"Syncing '{directories[0]}' to corpus '{corpus}'")
            else:
                print_info(f"Syncing {len(dir_paths)} directories to corpus '{corpus}'")

        # Initialize clients
        qdrant = create_qdrant_client()
        meili = get_meili_client()

        # Verify corpus exists in both systems
        try:
            qdrant.get_collection(corpus)
        except Exception:
            raise ResourceNotFoundError(
                f"Qdrant collection '{corpus}' not found. "
                f"Create it first with: arc corpus create {corpus} --type <type>"
            )

        if not meili.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        if not meili.index_exists(corpus):
            raise ResourceNotFoundError(
                f"MeiliSearch index '{corpus}' not found. "
                f"Create it first with: arc corpus create {corpus} --type <type>"
            )

        # Get corpus type and configured models from collection metadata
        corpus_type = get_collection_type(qdrant, corpus)
        metadata = get_collection_metadata(qdrant, corpus)
        configured_models = metadata.get('model', models)

        if not corpus_type:
            corpus_type = 'pdf'  # Default
            logger.warning(f"Collection type not set, defaulting to {corpus_type}")

        if not output_json:
            print_info(f"Corpus type: {corpus_type}")
            print_info(f"Models: {configured_models}")
            print_info(f"Dual indexing to:")
            print_info(f"  - Qdrant collection: {corpus} (semantic search)")
            print_info(f"  - MeiliSearch index: {corpus} (full-text search)")

        # Parse models
        model_list = [m.strip() for m in configured_models.split(',')]

        # Discover files from all directories
        files = []
        for dir_path in dir_paths:
            dir_files = discover_files(dir_path, file_types, corpus_type)
            files.extend(dir_files)

        if not files:
            if output_json:
                print_json("success", "No files to index", data={"indexed": 0})
            else:
                print_info("No files found to index")
            interaction_logger.finish(result_count=0)
            return

        if not output_json:
            dir_word = "directory" if len(dir_paths) == 1 else "directories"
            print_info(f"Found {len(files)} files across {len(dir_paths)} {dir_word}")

        # Apply change detection (skip already indexed files) unless --force
        already_indexed_count = 0
        meili_backfill_paths = []  # Files in Qdrant but missing from MeiliSearch
        qdrant_backfill_paths = []  # Files in MeiliSearch but missing from Qdrant

        if force:
            if not output_json:
                print_info("Force mode: reindexing all files")
        else:
            if not output_json:
                print_info("Checking index parity between Qdrant and MeiliSearch...")

            # Get file paths from both systems
            sync_manager = MetadataBasedSync(qdrant)
            meili_file_paths = meili.get_all_file_paths(corpus)
            qdrant_file_paths = sync_manager._get_indexed_file_paths_set(corpus)

            # Calculate set operations
            in_both_systems = qdrant_file_paths & meili_file_paths
            missing_from_meili = qdrant_file_paths - meili_file_paths
            missing_from_qdrant = meili_file_paths - qdrant_file_paths

            # Files in Qdrant but not in MeiliSearch need backfill
            if missing_from_meili:
                meili_backfill_paths = [p for p in missing_from_meili if Path(p).exists()]
                if not output_json and meili_backfill_paths:
                    print_info(f"Found {len(meili_backfill_paths)} files in Qdrant missing from MeiliSearch")

            # Files in MeiliSearch but not in Qdrant need backfill
            if missing_from_qdrant:
                qdrant_backfill_paths = [p for p in missing_from_qdrant if Path(p).exists()]
                if not output_json and qdrant_backfill_paths:
                    print_info(f"Found {len(qdrant_backfill_paths)} files in MeiliSearch missing from Qdrant")

            # Check for new/modified files not in either system
            # Convert discovered files to absolute path strings for comparison
            discovered_file_paths = {str(f.absolute()) for f in files}
            all_indexed_paths = qdrant_file_paths | meili_file_paths
            new_file_paths = discovered_file_paths - all_indexed_paths

            # Filter to only process new files (not in either system)
            files_to_process = [f for f in files if str(f.absolute()) in new_file_paths]

            # Count files in both systems (truly skipped)
            already_indexed_count = len(in_both_systems & discovered_file_paths)

            if not output_json:
                if already_indexed_count > 0:
                    print_info(f"Skipping {already_indexed_count} files already in both systems")
                if len(files_to_process) > 0:
                    print_info(f"Processing {len(files_to_process)} new files")

            files = files_to_process

        # If no new files but there are files to backfill, continue
        if not files and not meili_backfill_paths and not qdrant_backfill_paths:
            if output_json:
                print_json("success", "All files already indexed", data={
                    "indexed": 0,
                    "skipped": already_indexed_count
                })
            else:
                print_info("All files are already indexed in both systems (use --force to reindex)")
            interaction_logger.finish(result_count=0)
            return

        # Process files
        total_indexed = 0
        total_chunks = 0
        total_qdrant = 0
        total_meili = 0

        # Only initialize embedding infrastructure if there are new files to process
        if files:
            # Initialize embedding client
            use_gpu = not os.environ.get('ARC_NO_GPU', '').lower() in ('1', 'true')
            embedding_client = EmbeddingClient(use_gpu=use_gpu)

            # Initialize git discovery for code corpora
            git_discovery = GitProjectDiscovery() if corpus_type == 'code' else None
            git_metadata_cache: Dict[str, Any] = {}  # Cache by git root path

            # Create dual indexer
            dual_indexer = DualIndexer(
                qdrant_client=qdrant,
                meili_client=meili,
                collection_name=corpus,
                index_name=corpus
            )

            # Get model config for chunking
            first_model = model_list[0]
            if first_model in DEFAULT_MODELS:
                model_config = DEFAULT_MODELS[first_model].__dict__
            else:
                # Fallback config
                model_config = {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'char_to_token_ratio': 3.3,
                }

            # Pre-chunk code files in parallel if workers > 1
            pre_chunked_code_files: Dict[str, List[Dict[str, Any]]] = {}
            if corpus_type == 'code' and effective_text_workers > 1:
                if not output_json:
                    print_info(f"Parallel chunking {len(files)} code files with {effective_text_workers} workers...")
                pre_chunked_code_files = _parallel_chunk_code_files(
                    [str(f) for f in files],
                    model_config.get('chunk_size', 400),
                    model_config.get('chunk_overlap', 20),
                    effective_text_workers,
                    verbose,
                    output_json,
                    console,
                )
                if not output_json:
                    print_info(f"Pre-chunked {len(pre_chunked_code_files)} files")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                task = progress.add_task("Indexing...", total=len(files))

                for file_path in files:
                    progress.update(task, description=f"Processing {file_path.name}...")

                    try:
                        # Chunk file based on corpus type
                        if verbose and not output_json:
                            progress.console.print(f"[dim]Extracting text from {file_path.name}...[/dim]")

                        if corpus_type == 'pdf':
                            chunks = chunk_pdf_file(file_path, model_config)
                        elif corpus_type == 'markdown':
                            chunks = chunk_markdown_file(
                                file_path,
                                model_config.get('chunk_size', 512),
                                model_config.get('chunk_overlap', 50)
                            )
                        elif corpus_type == 'code':
                            # Use pre-chunked data if available
                            file_path_str = str(file_path)
                            if file_path_str in pre_chunked_code_files:
                                chunks = pre_chunked_code_files[file_path_str]
                            else:
                                chunks = chunk_code_file(
                                    file_path,
                                    model_config.get('chunk_size', 400),
                                    model_config.get('chunk_overlap', 20)
                                )
                        else:
                            logger.warning(f"Unknown corpus type: {corpus_type}, skipping {file_path}")
                            continue

                        if not chunks:
                            if verbose and not output_json:
                                progress.console.print(f"[yellow]  No text extracted from {file_path.name}[/yellow]")
                            progress.advance(task)
                            continue

                        if verbose and not output_json:
                            progress.console.print(f"[dim]  Created {len(chunks)} chunks, generating embeddings...[/dim]")

                        # Build dual index documents
                        documents = []
                        file_hash = compute_file_hash(file_path)
                        quick_hash = compute_quick_hash(file_path)

                        for i, chunk in enumerate(chunks):
                            # Generate embeddings for all models
                            vectors = {}
                            for model in model_list:
                                embeddings = embedding_client.embed([chunk['text']], model)
                                # Handle both list and numpy array returns
                                if hasattr(embeddings, 'tolist'):
                                    vectors[model] = embeddings[0].tolist()
                                else:
                                    vectors[model] = list(embeddings[0])

                            # Create document with shared metadata
                            doc = DualIndexDocument(
                                id=str(uuid4()),
                                content=chunk['text'],
                                file_path=str(file_path.absolute()),  # Use absolute path for change detection
                                filename=file_path.name,
                                file_extension=file_path.suffix,
                                chunk_index=i,
                                chunk_count=len(chunks),
                                file_hash=file_hash,
                                file_size=file_path.stat().st_size,
                                quick_hash=quick_hash,
                                vectors=vectors,
                            )

                            # Add type-specific metadata
                            chunk_meta = chunk.get('metadata', {})

                            if corpus_type == 'pdf':
                                doc.page_number = chunk_meta.get('page_number')
                                doc.document_type = 'pdf'

                            elif corpus_type == 'markdown':
                                doc.language = 'markdown'
                                doc.section = chunk_meta.get('header_path')
                                if chunk_meta.get('has_code_blocks'):
                                    doc.tags = ['has-code']

                            elif corpus_type == 'code':
                                doc.language = chunk_meta.get('language', 'unknown')
                                doc.line_number = chunk_meta.get('line_number')

                                # Extract git metadata for code files
                                if git_discovery:
                                    # Find git root for this file
                                    file_abs = str(file_path.absolute())
                                    git_root = None
                                    for parent in [file_path.absolute()] + list(file_path.absolute().parents):
                                        if (parent / '.git').exists():
                                            git_root = str(parent)
                                            break

                                    if git_root:
                                        # Get cached metadata or extract it
                                        if git_root not in git_metadata_cache:
                                            git_meta = git_discovery.extract_metadata(git_root)
                                            git_metadata_cache[git_root] = git_meta
                                        else:
                                            git_meta = git_metadata_cache[git_root]

                                        if git_meta:
                                            doc.project = git_meta.project_name
                                            doc.branch = git_meta.branch
                                            doc.git_project_identifier = git_meta.identifier

                            documents.append(doc)

                        # Index to both systems
                        if documents:
                            if verbose and not output_json:
                                progress.console.print(f"[dim]  Indexing {len(documents)} chunks to Qdrant + MeiliSearch...[/dim]")

                            qdrant_count, meili_count = dual_indexer.index_batch(documents)
                            total_chunks += len(documents)
                            total_qdrant += qdrant_count
                            total_meili += meili_count

                            if verbose and not output_json:
                                progress.console.print(f"[green]  ✓ {file_path.name}: {len(chunks)} chunks → Qdrant({qdrant_count}) + MeiliSearch({meili_count})[/green]")

                        total_indexed += 1

                    except Exception as e:
                        logger.error(f"Failed to process {file_path}: {e}")
                        if not output_json:
                            progress.console.print(f"[yellow]Warning: Failed to process {file_path.name}: {e}[/yellow]")

                    progress.advance(task)

        # Backfill MeiliSearch for files already in Qdrant but missing from MeiliSearch
        meili_backfilled = 0
        meili_backfill_chunks = 0
        meili_backfill_failed = 0

        if meili_backfill_paths and not force:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(meili_backfill_paths)} files to MeiliSearch...[/blue]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling...", total=len(meili_backfill_paths))
                meili_backfilled, meili_backfill_chunks, meili_backfill_failed = _backfill_qdrant_to_meili(
                    qdrant, meili, corpus, meili_backfill_paths,
                    verbose, output_json, progress, backfill_task,
                    fetch_workers=effective_text_workers,
                )

            total_meili += meili_backfill_chunks

        # Backfill Qdrant for files in MeiliSearch but missing from Qdrant
        # This requires re-processing the files since we need embeddings
        qdrant_backfilled = 0
        qdrant_backfill_chunks = 0
        qdrant_backfill_failed = 0

        if qdrant_backfill_paths and not force:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(qdrant_backfill_paths)} files to Qdrant (requires embedding)...[/blue]")

            # Need embedding client for Qdrant backfill
            use_gpu = not os.environ.get('ARC_NO_GPU', '').lower() in ('1', 'true')
            embedding_client = EmbeddingClient(use_gpu=use_gpu)

            # Get model config for chunking
            first_model = model_list[0]
            if first_model in DEFAULT_MODELS:
                model_config = DEFAULT_MODELS[first_model].__dict__
            else:
                model_config = {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'char_to_token_ratio': 3.3,
                }

            # Pre-chunk code files for backfill in parallel if workers > 1
            backfill_pre_chunked: Dict[str, List[Dict[str, Any]]] = {}
            if corpus_type == 'code' and effective_text_workers > 1:
                if not output_json:
                    print_info(f"Parallel chunking {len(qdrant_backfill_paths)} code files for backfill...")
                backfill_pre_chunked = _parallel_chunk_code_files(
                    qdrant_backfill_paths,
                    model_config.get('chunk_size', 400),
                    model_config.get('chunk_overlap', 20),
                    effective_text_workers,
                    verbose,
                    output_json,
                    console,
                )

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling Qdrant...", total=len(qdrant_backfill_paths))

                for file_path_str in qdrant_backfill_paths:
                    file_path = Path(file_path_str)
                    progress.update(backfill_task, description=f"Backfilling {file_path.name}...")

                    try:
                        # Chunk file based on corpus type
                        if corpus_type == 'pdf':
                            chunks = chunk_pdf_file(file_path, model_config)
                        elif corpus_type == 'markdown':
                            chunks = chunk_markdown_file(
                                file_path,
                                model_config.get('chunk_size', 512),
                                model_config.get('chunk_overlap', 50)
                            )
                        elif corpus_type == 'code':
                            # Use pre-chunked data if available
                            if file_path_str in backfill_pre_chunked:
                                chunks = backfill_pre_chunked[file_path_str]
                            else:
                                chunks = chunk_code_file(
                                    file_path,
                                    model_config.get('chunk_size', 400),
                                    model_config.get('chunk_overlap', 20)
                                )
                        else:
                            progress.advance(backfill_task)
                            continue

                        if not chunks:
                            progress.advance(backfill_task)
                            continue

                        # Build Qdrant documents with embeddings
                        from qdrant_client.models import PointStruct
                        points = []
                        file_hash = compute_file_hash(file_path)
                        quick_hash = compute_quick_hash(file_path)

                        for i, chunk in enumerate(chunks):
                            # Generate embeddings for all models
                            vectors = {}
                            for model in model_list:
                                embeddings = embedding_client.embed([chunk['text']], model)
                                if hasattr(embeddings, 'tolist'):
                                    vectors[model] = embeddings[0].tolist()
                                else:
                                    vectors[model] = list(embeddings[0])

                            # Build payload
                            payload = {
                                "text": chunk['text'],
                                "file_path": str(file_path.absolute()),
                                "filename": file_path.name,
                                "file_extension": file_path.suffix,
                                "chunk_index": i,
                                "chunk_count": len(chunks),
                                "file_hash": file_hash,
                                "file_size": file_path.stat().st_size,
                                "quick_hash": quick_hash,
                            }

                            # Add type-specific metadata
                            chunk_meta = chunk.get('metadata', {})
                            if corpus_type == 'pdf':
                                if chunk_meta.get('page_number'):
                                    payload["page_number"] = chunk_meta["page_number"]
                                payload["document_type"] = "pdf"

                            points.append(PointStruct(
                                id=str(uuid4()),
                                vector=vectors,
                                payload=payload
                            ))

                        # Upload to Qdrant
                        if points:
                            qdrant.upsert(collection_name=corpus, points=points, wait=True)
                            qdrant_backfill_chunks += len(points)

                            if verbose and not output_json:
                                progress.console.print(f"[green]  ✓ {file_path.name}: {len(points)} chunks → Qdrant[/green]")

                        qdrant_backfilled += 1

                    except Exception as e:
                        qdrant_backfill_failed += 1
                        logger.error(f"Failed to backfill {file_path_str} to Qdrant: {e}")
                        if not output_json:
                            progress.console.print(f"[yellow]Warning: Failed to backfill {file_path.name}: {e}[/yellow]")

                    progress.advance(backfill_task)

            total_qdrant += qdrant_backfill_chunks

        # Output results
        data = {
            "corpus": corpus,
            "directories": [str(p) for p in dir_paths],
            "files_indexed": total_indexed,
            "files_skipped": already_indexed_count,
            "meili_backfilled": meili_backfilled,
            "meili_backfill_failed": meili_backfill_failed,
            "qdrant_backfilled": qdrant_backfilled,
            "qdrant_backfill_failed": qdrant_backfill_failed,
            "total_chunks": total_chunks,
            "qdrant_indexed": total_qdrant,
            "meili_indexed": total_meili,
            "meili_backfilled_chunks": meili_backfill_chunks,
            "qdrant_backfilled_chunks": qdrant_backfill_chunks,
            "models": model_list,
        }

        if output_json:
            print_json("success", f"Indexed {total_indexed} files ({total_chunks} chunks)", data=data)
        else:
            console.print(f"\n[green]✅ Sync complete for corpus '{corpus}'[/green]")

            # New files (indexed to both systems)
            if total_indexed > 0:
                console.print(f"   New files (both systems): {total_indexed} files ({total_chunks} chunks)")

            # Already in both systems
            if already_indexed_count > 0:
                console.print(f"   Already synced:           {already_indexed_count} files")

            # MeiliSearch backfill (from Qdrant)
            if meili_backfilled > 0 or meili_backfill_failed > 0:
                status = f"{meili_backfilled} files ({meili_backfill_chunks} chunks)"
                if meili_backfill_failed > 0:
                    status += f" [yellow]({meili_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to MeiliSearch: {status}")

            # Qdrant backfill (requires re-embedding)
            if qdrant_backfilled > 0 or qdrant_backfill_failed > 0:
                status = f"{qdrant_backfilled} files ({qdrant_backfill_chunks} chunks)"
                if qdrant_backfill_failed > 0:
                    status += f" [yellow]({qdrant_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to Qdrant:      {status}")

            # Totals
            console.print(f"\n   Total in Qdrant:      {total_qdrant} vectors")
            console.print(f"   Total in MeiliSearch: {total_meili} documents")

            console.print(f"\n[dim]Search with:[/dim]")
            console.print(f"  arc search semantic \"your query\" --collection {corpus}")
            console.print(f"  arc search text \"your query\" --index {corpus}")

        # Post-verify if requested (uses same verifier as index commands)
        verification_result = None
        if verify:
            from ..indexing.verify import CollectionVerifier

            if not output_json:
                console.print("\n[dim]Verifying collection integrity...[/dim]")

            verifier = CollectionVerifier(qdrant)
            verification_result = verifier.verify_collection(corpus, verbose=verbose)

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

            data["verification"] = {
                "is_healthy": verification_result.is_healthy,
                "total_files": verification_result.total_items,
                "complete_files": verification_result.complete_items,
                "incomplete_files": verification_result.incomplete_items,
            }

        # Log successful operation (RDR-018)
        interaction_logger.finish(
            result_count=total_indexed,
            total_chunks=total_chunks,
        )

    except (InvalidArgumentError, ResourceNotFoundError):
        interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to sync directory: {e}", output_json)
        sys.exit(1)


def _fetch_qdrant_chunks_for_file(
    qdrant,
    corpus: str,
    file_path_str: str,
) -> Tuple[str, List[Dict[str, Any]], Optional[str]]:
    """Fetch chunks for a single file from Qdrant.

    Args:
        qdrant: Qdrant client
        corpus: Collection name
        file_path_str: File path to fetch

    Returns:
        Tuple of (file_path, list of meili docs, error or None)
    """
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    try:
        # Paginate through all points for this file
        all_points = []
        offset = None

        while True:
            points, next_offset = qdrant.scroll(
                collection_name=corpus,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="file_path",
                            match=MatchValue(value=file_path_str)
                        )
                    ]
                ),
                limit=1000,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )

            if points:
                all_points.extend(points)

            # Check if there are more results
            if next_offset is None or not points:
                break
            offset = next_offset

        if not all_points:
            return (file_path_str, [], None)

        # Convert Qdrant points to MeiliSearch documents
        meili_docs = []
        for point in all_points:
            payload = point.payload
            meili_doc = {
                "id": str(point.id),
                "content": payload.get("text", ""),
                "file_path": payload.get("file_path", ""),
                "filename": payload.get("filename", ""),
                "file_extension": payload.get("file_extension", ""),
                "chunk_index": payload.get("chunk_index", 0),
            }

            # Add optional fields
            if payload.get("page_number"):
                meili_doc["page_number"] = payload["page_number"]
            if payload.get("programming_language"):
                meili_doc["language"] = payload["programming_language"]
            if payload.get("document_type"):
                meili_doc["document_type"] = payload["document_type"]

            # Add git metadata fields for code corpora
            if payload.get("git_project_identifier"):
                meili_doc["git_project_identifier"] = payload["git_project_identifier"]
            if payload.get("git_project_name"):
                meili_doc["git_project_name"] = payload["git_project_name"]
            if payload.get("git_branch"):
                meili_doc["git_branch"] = payload["git_branch"]
            if payload.get("git_commit_hash"):
                meili_doc["git_commit_hash"] = payload["git_commit_hash"]

            meili_docs.append(meili_doc)

        return (file_path_str, meili_docs, None)

    except Exception as e:
        return (file_path_str, [], str(e))


def _fetch_git_metadata_for_files(
    qdrant,
    corpus: str,
    file_paths_needed: Set[str],
    console,
    output_json: bool,
) -> Dict[str, Dict[str, Any]]:
    """Fetch git metadata by scrolling Qdrant once, bounded to needed files.

    Instead of making one request per file, this scrolls through all Qdrant
    points once and only stores metadata for files in the needed set.
    This is O(total_points) but with far fewer requests than O(files_needed).

    Args:
        qdrant: Qdrant client
        corpus: Corpus/collection name
        file_paths_needed: Set of file paths that need git metadata
        console: Rich console for output
        output_json: JSON output mode (suppresses progress)

    Returns:
        Dict mapping file_path -> git metadata dict
    """
    git_metadata_by_file: Dict[str, Dict[str, Any]] = {}
    offset = None
    points_scanned = 0

    while True:
        points, offset = qdrant.scroll(
            collection_name=corpus,
            limit=1000,
            offset=offset,
            with_payload=["file_path", "git_project_identifier",
                         "git_project_name", "git_branch", "git_commit_hash"],
            with_vectors=False
        )
        if not points:
            break

        for point in points:
            file_path = point.payload.get("file_path")
            # Only store if file is in our needed set and has git metadata
            if file_path in file_paths_needed and point.payload.get("git_project_identifier"):
                if file_path not in git_metadata_by_file:
                    git_metadata_by_file[file_path] = {
                        "git_project_identifier": point.payload.get("git_project_identifier"),
                        "git_project_name": point.payload.get("git_project_name"),
                        "git_branch": point.payload.get("git_branch"),
                        "git_commit_hash": point.payload.get("git_commit_hash"),
                    }

        points_scanned += len(points)
        if not output_json and points_scanned % 10000 == 0:
            console.print(f"[dim]  Scanned {points_scanned} Qdrant points...[/dim]")

        if offset is None:
            break

    return git_metadata_by_file


def _repair_meili_metadata(
    qdrant,
    meili: FullTextClient,
    corpus: str,
    verbose: bool,
    output_json: bool,
    console,
) -> Tuple[int, int]:
    """Repair MeiliSearch documents with missing git metadata from Qdrant.

    For code corpora, finds MeiliSearch documents missing git_project_identifier
    and updates them with metadata from corresponding Qdrant points.

    Args:
        qdrant: Qdrant client
        meili: MeiliSearch client
        corpus: Corpus name
        verbose: Show verbose output
        output_json: JSON output mode
        console: Rich console for output

    Returns:
        Tuple of (documents_updated, documents_failed)
    """
    # Step 1: Find MeiliSearch documents missing git_project_identifier
    # Use get_documents API (not search) to iterate through ALL documents
    docs_to_repair = []
    batch_offset = 0
    batch_size = 1000

    if not output_json:
        console.print(f"[dim]Scanning MeiliSearch for documents missing git metadata...[/dim]")

    index = meili.get_index(corpus)
    while True:
        result = index.get_documents({
            'offset': batch_offset,
            'limit': batch_size,
            'fields': ['id', 'file_path', 'chunk_index', 'git_project_identifier']
        })

        # Handle both dict and object results
        if hasattr(result, 'results'):
            docs = result.results
        else:
            docs = result.get('results', [])

        if not docs:
            break

        for doc in docs:
            # Get field values handling both dict and object
            if isinstance(doc, dict):
                has_git_id = bool(doc.get('git_project_identifier'))
                doc_id = doc.get('id')
                file_path = doc.get('file_path')
                chunk_index = doc.get('chunk_index', 0)
            else:
                has_git_id = bool(getattr(doc, 'git_project_identifier', None))
                doc_id = getattr(doc, 'id', None)
                file_path = getattr(doc, 'file_path', None)
                chunk_index = getattr(doc, 'chunk_index', 0)

            # Check if git_project_identifier is missing
            if not has_git_id:
                docs_to_repair.append({
                    'id': doc_id,
                    'file_path': file_path,
                    'chunk_index': chunk_index,
                })

        batch_offset += len(docs)
        if len(docs) < batch_size:
            break

    if not docs_to_repair:
        if not output_json:
            console.print(f"[green]✓ No documents need metadata repair[/green]")
        return 0, 0

    if not output_json:
        console.print(f"[blue]Found {len(docs_to_repair)} documents to repair[/blue]")

    # Step 2: Build a map of file_path -> git metadata from Qdrant
    # Get unique file paths that need repair
    file_paths_needed = set(d['file_path'] for d in docs_to_repair if d['file_path'])

    if not output_json:
        console.print(f"[dim]Scanning Qdrant for git metadata ({len(file_paths_needed)} files needed)...[/dim]")

    # Single scroll through Qdrant, bounded to needed files
    git_metadata_by_file = _fetch_git_metadata_for_files(
        qdrant, corpus, file_paths_needed, console, output_json
    )

    if not git_metadata_by_file:
        if not output_json:
            console.print(f"[yellow]No git metadata found in Qdrant to use for repair[/yellow]")
        return 0, 0

    if not output_json:
        console.print(f"[dim]Found git metadata for {len(git_metadata_by_file)} files[/dim]")

    # Step 3: Build update documents
    updates = []
    for doc in docs_to_repair:
        file_path = doc.get('file_path')
        if file_path and file_path in git_metadata_by_file:
            git_meta = git_metadata_by_file[file_path]
            update_doc = {
                'id': doc['id'],
                **git_meta
            }
            updates.append(update_doc)

    if not updates:
        if not output_json:
            console.print(f"[yellow]No documents matched for update[/yellow]")
        return 0, 0

    # Step 4: Update MeiliSearch documents in batches
    if not output_json:
        console.print(f"[blue]Updating {len(updates)} documents in MeiliSearch...[/blue]")

    docs_updated = 0
    docs_failed = 0
    update_batch_size = 1000

    try:
        index = meili.get_index(corpus)

        for i in range(0, len(updates), update_batch_size):
            batch = updates[i:i + update_batch_size]
            try:
                # Use update_documents which will update existing docs by id
                task = index.update_documents(batch)
                # Wait for task to complete using meilisearch client
                meili.client.wait_for_task(task.task_uid)
                docs_updated += len(batch)
            except Exception as e:
                logger.error(f"Failed to update batch: {e}")
                docs_failed += len(batch)

        if not output_json:
            console.print(f"[green]✓ Updated {docs_updated} documents with git metadata[/green]")
            if docs_failed > 0:
                console.print(f"[yellow]  {docs_failed} documents failed to update[/yellow]")

    except Exception as e:
        logger.error(f"Failed to repair metadata: {e}")
        if not output_json:
            console.print(f"[red]Failed to repair metadata: {e}[/red]")
        return docs_updated, len(updates) - docs_updated

    return docs_updated, docs_failed


def _backfill_qdrant_to_meili(
    qdrant,
    meili: FullTextClient,
    corpus: str,
    file_paths: List[str],
    verbose: bool,
    output_json: bool,
    progress,
    backfill_task,
    batch_size: int = 1000,
    fetch_workers: int = 8,
) -> tuple:
    """Backfill files from Qdrant to MeiliSearch.

    Copies chunk data from Qdrant to MeiliSearch without needing file access.
    Uses parallel fetches from Qdrant and uploads to MeiliSearch.

    ATOMICITY: Each file is uploaded as a complete unit. If the process is
    interrupted, files are either fully indexed or not at all - no partial
    file indexing. Re-running parity will pick up any incomplete files.

    Args:
        qdrant: Qdrant client
        meili: MeiliSearch client
        corpus: Corpus name
        file_paths: List of file paths to backfill
        verbose: Show verbose output
        output_json: JSON output mode
        progress: Rich progress instance
        backfill_task: Progress task ID
        batch_size: Max documents per upload batch (default: 1000)
        fetch_workers: Number of parallel threads for Qdrant fetches (default: 8)

    Returns:
        Tuple of (files_success, chunks_success, files_failed)
    """
    import threading

    files_success = 0
    files_failed = 0
    chunks_success = 0

    # Track completed file uploads (for atomicity)
    upload_tasks_by_file: Dict[str, List[int]] = {}  # file_path -> [task_uids]
    completed_files: Set[str] = set()
    stats_lock = threading.Lock()

    def upload_file_chunks(file_path: str, docs: List[Dict[str, Any]]) -> bool:
        """Upload all chunks for a single file atomically.

        Returns True if upload was queued successfully.
        """
        if not docs:
            return True

        try:
            index = meili.get_index(corpus)
            task_uids = []

            # Upload in batches, but track all tasks for this file
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                task = index.add_documents(batch)
                task_uids.append(task.task_uid)

            # Track tasks for this file
            with stats_lock:
                upload_tasks_by_file[file_path] = task_uids

            if verbose and not output_json:
                progress.console.print(
                    f"[dim]  Queued {len(docs)} chunks for {Path(file_path).name} "
                    f"({len(task_uids)} batch{'es' if len(task_uids) > 1 else ''})[/dim]"
                )
            return True

        except Exception as e:
            logger.error(f"Failed to queue upload for {file_path}: {e}")
            return False

    # Parallel fetch from Qdrant and queue uploads
    progress.update(backfill_task, description=f"Fetching & uploading ({fetch_workers} threads)...")

    with ThreadPoolExecutor(max_workers=fetch_workers) as executor:
        # Submit all fetch tasks
        futures = {
            executor.submit(_fetch_qdrant_chunks_for_file, qdrant, corpus, fp): fp
            for fp in file_paths
        }

        # Process results as they complete - upload each file atomically
        for future in as_completed(futures):
            file_path_str = futures[future]
            try:
                fp, docs, error = future.result()
                if error:
                    with stats_lock:
                        files_failed += 1
                    logger.error(f"Failed to fetch {fp} from Qdrant: {error}")
                    if not output_json:
                        progress.console.print(f"[yellow]Warning: Failed to fetch {Path(fp).name}: {error}[/yellow]")
                elif docs:
                    # Upload all chunks for this file as a unit
                    if upload_file_chunks(fp, docs):
                        with stats_lock:
                            files_success += 1
                            chunks_success += len(docs)
                    else:
                        with stats_lock:
                            files_failed += 1
                else:
                    # No chunks found, still count as success
                    with stats_lock:
                        files_success += 1
            except Exception as e:
                with stats_lock:
                    files_failed += 1
                logger.error(f"Failed to process {file_path_str}: {e}")

            progress.advance(backfill_task)

    # Wait for all upload tasks to complete, tracking per-file success
    all_task_uids = []
    for task_uids in upload_tasks_by_file.values():
        all_task_uids.extend(task_uids)

    if all_task_uids:
        progress.update(backfill_task, description=f"Waiting for {len(all_task_uids)} uploads to complete...")

        # Wait for all tasks
        task_results: Dict[int, bool] = {}  # task_uid -> success
        for task_uid in all_task_uids:
            try:
                result = meili.client.wait_for_task(task_uid, timeout_in_ms=180000)
                status = getattr(result, 'status', None) or (result.get('status') if isinstance(result, dict) else None)
                task_results[task_uid] = (status == 'succeeded')
                if status == 'failed':
                    error = getattr(result, 'error', None) or (result.get('error') if isinstance(result, dict) else None)
                    logger.error(f"Upload task {task_uid} failed: {error}")
            except Exception as e:
                task_results[task_uid] = False
                logger.error(f"Upload task {task_uid} error: {e}")

        # Check which files completed successfully (all their tasks succeeded)
        files_with_failed_uploads = []
        for file_path, task_uids in upload_tasks_by_file.items():
            all_succeeded = all(task_results.get(uid, False) for uid in task_uids)
            if all_succeeded:
                completed_files.add(file_path)
            else:
                files_with_failed_uploads.append(file_path)
                # Adjust counts - file upload failed
                with stats_lock:
                    files_success -= 1
                    files_failed += 1
                    # We don't know exact chunk count that failed, but this is conservative

        if files_with_failed_uploads:
            if not output_json:
                progress.console.print(f"[yellow]Warning: {len(files_with_failed_uploads)} files had upload failures[/yellow]")
                if verbose:
                    for fp in files_with_failed_uploads[:5]:
                        progress.console.print(f"[yellow]  - {Path(fp).name}[/yellow]")
                    if len(files_with_failed_uploads) > 5:
                        progress.console.print(f"[yellow]  ... and {len(files_with_failed_uploads) - 5} more[/yellow]")

            # Clean up partial uploads from MeiliSearch so they can be retried
            # This ensures failed files aren't seen as "synced" on the next parity run
            if not output_json:
                progress.console.print(f"[dim]Cleaning up partial uploads for failed files...[/dim]")
            try:
                meili.delete_documents_by_file_paths(corpus, files_with_failed_uploads)
                if not output_json:
                    progress.console.print(f"[dim]  Cleaned up {len(files_with_failed_uploads)} partial file uploads[/dim]")
            except Exception as cleanup_error:
                logger.warning(f"Failed to clean up partial uploads: {cleanup_error}")
                if not output_json:
                    progress.console.print(
                        f"[yellow]Warning: Could not clean up partial uploads: {cleanup_error}[/yellow]"
                    )
                    progress.console.print(
                        f"[yellow]These files may appear synced but have incomplete data. "
                        f"Run parity again to retry.[/yellow]"
                    )

    if verbose and not output_json:
        progress.console.print(f"[green]  ✓ Uploaded {chunks_success} chunks for {len(completed_files)} files to MeiliSearch[/green]")

    return files_success, chunks_success, files_failed


def _parallel_chunk_code_files(
    file_paths: List[str],
    chunk_size: int,
    chunk_overlap: int,
    workers: int,
    verbose: bool,
    output_json: bool,
    console,
) -> Dict[str, List[Dict[str, Any]]]:
    """Pre-chunk code files in parallel using ProcessPoolExecutor.

    Args:
        file_paths: List of file paths to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between chunks in tokens
        workers: Number of parallel workers
        verbose: Show verbose output
        output_json: JSON output mode
        console: Rich console for output

    Returns:
        Dict mapping file_path -> list of chunk dicts
    """
    chunked_files = {}
    errors = []

    if workers <= 1:
        # Sequential mode
        for file_path in file_paths:
            file_path_str, chunks, error = _chunk_code_file_worker(
                file_path, chunk_size, chunk_overlap
            )
            if error:
                errors.append((file_path_str, error))
            elif chunks:
                chunked_files[file_path_str] = chunks
        return chunked_files

    # Parallel mode
    try:
        ctx = mp.get_context('fork') if sys.platform != 'win32' else mp.get_context('spawn')
    except ValueError:
        ctx = mp.get_context('spawn')

    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = {
            executor.submit(
                _chunk_code_file_worker,
                fp, chunk_size, chunk_overlap
            ): fp
            for fp in file_paths
        }

        for future in as_completed(futures):
            try:
                file_path_str, chunks, error = future.result()
                if error:
                    errors.append((file_path_str, error))
                    if verbose and not output_json:
                        console.print(f"[yellow]  Chunking error for {Path(file_path_str).name}: {error}[/yellow]")
                elif chunks:
                    chunked_files[file_path_str] = chunks
            except Exception as e:
                fp = futures[future]
                errors.append((fp, str(e)))
                logger.error(f"Worker exception for {fp}: {e}")

        # Clean up
        del futures

    gc.collect()

    if verbose and not output_json and errors:
        console.print(f"[yellow]  {len(errors)} files had chunking errors[/yellow]")

    return chunked_files


def _backfill_meili_to_qdrant(
    qdrant,
    embedding_client: EmbeddingClient,
    corpus: str,
    corpus_type: str,
    model_list: List[str],
    model_config: Dict[str, Any],
    file_paths: List[str],
    verbose: bool,
    output_json: bool,
    progress,
    backfill_task,
    text_workers: int = 1,
) -> tuple:
    """Backfill files from MeiliSearch to Qdrant.

    Re-chunks files and generates embeddings. Requires file access.
    Files that don't exist on disk are skipped with a warning.

    Args:
        qdrant: Qdrant client
        embedding_client: Embedding client
        corpus: Corpus name
        corpus_type: Type of corpus (pdf, code, markdown)
        model_list: List of embedding model names
        model_config: Chunking configuration
        file_paths: List of file paths to backfill
        verbose: Show verbose output
        output_json: JSON output mode
        progress: Rich progress instance
        backfill_task: Progress task ID
        text_workers: Number of parallel workers for code chunking (1=sequential)

    Returns:
        Tuple of (files_success, chunks_success, files_failed, skipped_paths)
    """
    from qdrant_client.models import PointStruct

    files_success = 0
    chunks_success = 0
    files_failed = 0
    skipped_paths = []

    # For code corpora with parallel workers, pre-chunk all files in parallel
    pre_chunked_files: Dict[str, List[Dict[str, Any]]] = {}
    if corpus_type == 'code' and text_workers > 1:
        # Filter to existing files first
        existing_files = []
        for fp in file_paths:
            if Path(fp).exists():
                existing_files.append(fp)
            else:
                skipped_paths.append(fp)

        if existing_files:
            progress.update(backfill_task, description=f"Parallel chunking {len(existing_files)} code files...")
            pre_chunked_files = _parallel_chunk_code_files(
                existing_files,
                model_config.get('chunk_size', 400),
                model_config.get('chunk_overlap', 20),
                text_workers,
                verbose,
                output_json,
                progress.console,
            )
            if verbose and not output_json:
                progress.console.print(f"[dim]  Pre-chunked {len(pre_chunked_files)} files in parallel[/dim]")

    for file_path_str in file_paths:
        file_path = Path(file_path_str)
        progress.update(backfill_task, description=f"Backfilling {file_path.name}...")

        # Check if file exists on disk (already checked for pre-chunked code)
        if file_path_str in skipped_paths:
            if not output_json:
                progress.console.print(f"[yellow]  ⚠ {file_path.name}: File not found, skipping[/yellow]")
            progress.advance(backfill_task)
            continue

        if not file_path.exists():
            skipped_paths.append(file_path_str)
            if not output_json:
                progress.console.print(f"[yellow]  ⚠ {file_path.name}: File not found, skipping[/yellow]")
            progress.advance(backfill_task)
            continue

        try:
            # Chunk file based on corpus type
            chunks = None

            if corpus_type == 'code' and file_path_str in pre_chunked_files:
                # Use pre-chunked results
                chunks = pre_chunked_files[file_path_str]
            elif corpus_type == 'pdf':
                chunks = chunk_pdf_file(file_path, model_config)
            elif corpus_type == 'markdown':
                chunks = chunk_markdown_file(
                    file_path,
                    model_config.get('chunk_size', 512),
                    model_config.get('chunk_overlap', 50)
                )
            elif corpus_type == 'code':
                # Sequential chunking for code (workers=1 or file not in pre-chunked)
                chunks = chunk_code_file(
                    file_path,
                    model_config.get('chunk_size', 400),
                    model_config.get('chunk_overlap', 20)
                )
            else:
                progress.advance(backfill_task)
                continue

            if not chunks:
                progress.advance(backfill_task)
                continue

            # Build Qdrant documents with embeddings
            points = []
            file_hash = compute_file_hash(file_path)
            quick_hash = compute_quick_hash(file_path)

            for i, chunk in enumerate(chunks):
                # Generate embeddings for all models
                vectors = {}
                # Get text from chunk - handle both dict (pre-chunked) and Chunk object formats
                chunk_text = chunk.get('text') if isinstance(chunk, dict) else chunk.content
                for model in model_list:
                    embeddings = embedding_client.embed([chunk_text], model)
                    if hasattr(embeddings, 'tolist'):
                        vectors[model] = embeddings[0].tolist()
                    else:
                        vectors[model] = list(embeddings[0])

                # Build payload
                payload = {
                    "text": chunk_text,
                    "file_path": str(file_path.absolute()),
                    "filename": file_path.name,
                    "file_extension": file_path.suffix,
                    "chunk_index": i,
                    "chunk_count": len(chunks),
                    "file_hash": file_hash,
                    "file_size": file_path.stat().st_size,
                    "quick_hash": quick_hash,
                }

                # Add type-specific metadata
                chunk_meta = chunk.get('metadata', {}) if isinstance(chunk, dict) else {}
                if corpus_type == 'pdf':
                    if chunk_meta.get('page_number'):
                        payload["page_number"] = chunk_meta["page_number"]
                    payload["document_type"] = "pdf"

                points.append(PointStruct(
                    id=str(uuid4()),
                    vector=vectors,
                    payload=payload
                ))

            # Upload to Qdrant
            if points:
                qdrant.upsert(collection_name=corpus, points=points, wait=True)
                chunks_success += len(points)

                if verbose and not output_json:
                    progress.console.print(f"[green]  ✓ {file_path.name}: {len(points)} chunks → Qdrant[/green]")

            files_success += 1

        except Exception as e:
            files_failed += 1
            logger.error(f"Failed to backfill {file_path_str} to Qdrant: {e}")
            if not output_json:
                progress.console.print(f"[yellow]Warning: Failed to backfill {file_path.name}: {e}[/yellow]")

        progress.advance(backfill_task)

    return files_success, chunks_success, files_failed, skipped_paths


def _create_missing_meili_index(
    corpus_name: str,
    output_json: bool = False
) -> bool:
    """Create MeiliSearch index for a qdrant_only corpus.

    Reads corpus type from Qdrant metadata and creates index with appropriate settings.

    Args:
        corpus_name: Name of the corpus to create index for
        output_json: If True, suppress console output

    Returns:
        True if created successfully, False otherwise
    """
    from ..fulltext.indexes import get_index_settings

    try:
        # Get Qdrant client and read metadata
        qdrant = create_qdrant_client()
        corpus_type = get_collection_type(qdrant, corpus_name)

        if not corpus_type:
            logger.warning(f"Cannot determine corpus type for '{corpus_name}'")
            if not output_json:
                console.print(f"[yellow]Warning: Cannot determine corpus type for '{corpus_name}'[/yellow]")
            return False

        # Map corpus type to index settings
        # corpus_type is 'pdf', 'code', or 'markdown'
        # get_index_settings accepts these as aliases
        settings = get_index_settings(corpus_type)

        # Create MeiliSearch index
        meili = get_meili_client()

        if not meili.health_check():
            logger.error("MeiliSearch server not available")
            if not output_json:
                console.print("[red]Error: MeiliSearch server not available[/red]")
            return False

        meili.create_index(corpus_name, primary_key='id', settings=settings)

        if not output_json:
            console.print(f"[green]Created MeiliSearch index '{corpus_name}' (type: {corpus_type})[/green]")

        logger.info(f"Created MeiliSearch index '{corpus_name}' with type '{corpus_type}'")
        return True

    except Exception as e:
        logger.error(f"Failed to create MeiliSearch index for '{corpus_name}': {e}")
        if not output_json:
            console.print(f"[red]Error creating MeiliSearch index for '{corpus_name}': {e}[/red]")
        return False


def _discover_corpora_for_parity() -> List[Dict[str, Any]]:
    """Discover all corpora and their parity status.

    Returns a list of corpus info dicts with keys:
        - name: Corpus name
        - status: 'synced', 'qdrant_only', 'meili_only', or 'chunk_mismatch'
        - qdrant_chunks: Number of chunks in Qdrant
        - meili_chunks: Number of chunks in MeiliSearch
        - to_meili: Number of files to backfill to MeiliSearch
        - to_qdrant: Number of files to backfill to Qdrant
    """
    from ..fulltext.client import FullTextClient

    # Gather Qdrant collections
    qdrant_collections = {}
    try:
        qdrant = create_qdrant_client()
        collections = qdrant.get_collections()
        for col in collections.collections:
            metadata = get_collection_metadata(qdrant, col.name)
            col_info = qdrant.get_collection(col.name)
            # Subtract 1 for the metadata point
            chunk_count = col_info.points_count - 1 if col_info.points_count > 0 else 0
            qdrant_collections[col.name] = {
                "type": metadata.get("collection_type"),
                "model": metadata.get("model"),
                "chunks": chunk_count,
            }
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant: {e}")

    # Gather MeiliSearch indexes
    meili_indexes = {}
    try:
        meili = get_meili_client()
        if meili.health_check():
            indexes = meili.list_indexes()
            for idx in indexes:
                name = idx['uid']
                try:
                    stats = meili.get_index_stats(name)
                    meili_indexes[name] = {
                        "chunks": stats.get('numberOfDocuments', 0),
                    }
                except Exception:
                    meili_indexes[name] = {"chunks": 0}
    except Exception as e:
        logger.warning(f"Could not connect to MeiliSearch: {e}")

    # Build unified corpus list
    all_names = set(qdrant_collections.keys()) | set(meili_indexes.keys())
    corpora = []

    for name in sorted(all_names):
        q_info = qdrant_collections.get(name)
        m_info = meili_indexes.get(name)

        # Determine parity status
        if q_info and m_info:
            # Both exist - check if chunks match
            q_chunks = q_info.get("chunks", 0)
            m_chunks = m_info.get("chunks", 0)
            if q_chunks == m_chunks:
                status = "synced"
            else:
                status = "chunk_mismatch"
        elif q_info:
            status = "qdrant_only"
        else:
            status = "meili_only"

        q_chunks = q_info.get("chunks", 0) if q_info else 0
        m_chunks = m_info.get("chunks", 0) if m_info else 0

        # Calculate files to backfill
        to_meili = max(0, q_chunks - m_chunks) if status == "chunk_mismatch" else (q_chunks if status == "qdrant_only" else 0)
        to_qdrant = max(0, m_chunks - q_chunks) if status == "chunk_mismatch" else (m_chunks if status == "meili_only" else 0)

        corpora.append({
            "name": name,
            "status": status,
            "qdrant_chunks": q_chunks,
            "meili_chunks": m_chunks,
            "to_meili": to_meili,
            "to_qdrant": to_qdrant,
        })

    return corpora


def _parity_all_corpora(
    dry_run: bool,
    verify: bool,
    repair_metadata: bool,
    text_workers: Optional[int],
    qdrant_timeout: int,
    create_missing: bool,
    confirm: bool,
    verbose: bool,
    output_json: bool,
    effective_text_workers: int,
):
    """Process parity for all discovered corpora.

    Shows a summary table and asks for confirmation before proceeding.
    If create_missing is True, creates MeiliSearch indexes for qdrant_only corpora.
    """
    import click
    from rich.table import Table

    interaction_logger.start("corpus", "parity_all", dry_run=dry_run)

    try:
        if not output_json:
            print_info("Discovering corpora...")

        corpora = _discover_corpora_for_parity()

        if not corpora:
            if output_json:
                print_json("success", "No corpora found", data={"corpora": []})
            else:
                print_info("No corpora found.")
            interaction_logger.finish(result_count=0)
            return

        # Separate corpora into categories:
        # - can_process: chunk_mismatch (both sides exist, just need sync)
        # - incomplete: qdrant_only or meili_only (need manual creation first)
        # - synced: already in parity
        can_process = [c for c in corpora if c["status"] == "chunk_mismatch"]
        incomplete = [c for c in corpora if c["status"] in ("qdrant_only", "meili_only")]

        # Handle --create-missing: create MeiliSearch indexes for qdrant_only corpora
        indexes_created = []
        if create_missing:
            qdrant_only = [c for c in incomplete if c["status"] == "qdrant_only"]
            meili_only = [c for c in incomplete if c["status"] == "meili_only"]

            if qdrant_only:
                if not output_json:
                    console.print(f"\n[blue]Creating missing MeiliSearch indexes for {len(qdrant_only)} qdrant_only corpora...[/blue]")

                for c in qdrant_only:
                    if dry_run:
                        if not output_json:
                            console.print(f"  [dim]Would create MeiliSearch index for '{c['name']}'[/dim]")
                        indexes_created.append(c['name'])
                    else:
                        if _create_missing_meili_index(c['name'], output_json):
                            indexes_created.append(c['name'])
                            # Update corpus status and move to can_process
                            c['status'] = 'chunk_mismatch'
                            can_process.append(c)

                # Remove successfully created from incomplete
                incomplete = [c for c in incomplete if c['name'] not in indexes_created]

            if meili_only and not output_json:
                console.print(f"\n[yellow]Warning: {len(meili_only)} meili_only corpora cannot be auto-created (require --type and --model)[/yellow]")
                for c in meili_only:
                    console.print(f"  [yellow]Skipping '{c['name']}' - create Qdrant collection first with: arc corpus create {c['name']} --type <type>[/yellow]")

        # Build summary table
        if not output_json:
            table = Table(title="Parity Summary")
            table.add_column("Corpus", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("To MeiliSearch", style="yellow")
            table.add_column("To Qdrant", style="yellow")

            for c in corpora:
                status = c["status"]
                name = c["name"]
                # Check if index was created first (for dry-run display)
                if name in indexes_created:
                    if dry_run:
                        status_str = "[blue]would_create_index[/blue]"
                    else:
                        status_str = "[blue]index_created[/blue]"
                elif status == "synced":
                    status_str = "[green]synced[/green]"
                elif status == "qdrant_only":
                    status_str = "[dim yellow]qdrant_only (skipped)[/dim yellow]"
                elif status == "meili_only":
                    status_str = "[dim yellow]meili_only (skipped)[/dim yellow]"
                else:
                    status_str = "[yellow]chunk_mismatch[/yellow]"

                to_meili = f"{c['to_meili']} chunks" if c['to_meili'] > 0 else "0"
                to_qdrant = f"{c['to_qdrant']} chunks" if c['to_qdrant'] > 0 else "0"

                table.add_row(name, status_str, to_meili, to_qdrant)

            console.print()
            console.print(table)

            # Note about incomplete corpora
            if incomplete:
                console.print(f"\n[dim]Note: {len(incomplete)} corpora exist only on one side and will be skipped.[/dim]")
                console.print(f"[dim]Create the missing index/collection first with: arc corpus create <name> --type <type>[/dim]")

        # Dry-run mode: show summary and exit
        if dry_run:
            if output_json:
                data = {
                    "dry_run": True,
                    "corpora": corpora,
                    "can_process": len(can_process),
                    "incomplete_skipped": len(incomplete),
                    "indexes_would_create": indexes_created,
                }
                print_json("success", "Dry run complete", data=data)
            else:
                console.print(f"\n[bold yellow]DRY RUN - No changes will be made[/bold yellow]")
                if indexes_created:
                    console.print(f"\nWould create {len(indexes_created)} MeiliSearch indexes")
                # Total to process = existing can_process + newly created indexes
                total_would_process = len(can_process) + len(indexes_created)
                if total_would_process > 0:
                    console.print(f"\nWould process parity for {total_would_process} corpora")
                elif not indexes_created:
                    console.print(f"\n[green]✓ All complete corpora are already in parity[/green]")
            interaction_logger.finish(result_count=0)
            return

        # If nothing to process, we're done
        if not can_process:
            if output_json:
                data = {
                    "corpora": corpora,
                    "incomplete_skipped": len(incomplete),
                }
                print_json("success", "All complete corpora are already in parity", data=data)
            else:
                console.print(f"\n[green]✓ All complete corpora are already in parity[/green]")
            interaction_logger.finish(result_count=0)
            return

        # Confirmation prompt (unless --confirm or --json)
        if not confirm:
            if output_json:
                raise InvalidArgumentError(
                    "--confirm flag required when processing all corpora with --json output"
                )
            console.print()
            if not click.confirm(f"Proceed with parity for {len(can_process)} corpora?"):
                print_info("Cancelled.")
                interaction_logger.finish(result_count=0)
                return

        # Process each corpus
        results = []
        for i, c in enumerate(can_process, 1):
            if not output_json:
                console.print(f"\n[bold]Processing {i}/{len(can_process)}: {c['name']}[/bold]")

            try:
                _parity_single_corpus(
                    corpus=c["name"],
                    dry_run=False,  # Already handled dry_run above
                    verify=verify,
                    repair_metadata=repair_metadata,
                    effective_text_workers=effective_text_workers,
                    qdrant_timeout=qdrant_timeout,
                    create_missing=False,  # Already handled create_missing above
                    verbose=verbose,
                    output_json=output_json,
                    all_corpora_mode=True,  # Suppress per-corpus logging
                )
                results.append({"corpus": c["name"], "status": "success"})
            except Exception as e:
                logger.error(f"Failed to process corpus {c['name']}: {e}")
                results.append({"corpus": c["name"], "status": "failed", "error": str(e)})
                if not output_json:
                    console.print(f"[red]Failed to process {c['name']}: {e}[/red]")

        # Final summary
        successes = sum(1 for r in results if r["status"] == "success")
        failures = sum(1 for r in results if r["status"] == "failed")

        if output_json:
            print_json("success", f"Parity completed for {successes} corpora", data={
                "processed": len(can_process),
                "successes": successes,
                "failures": failures,
                "incomplete_skipped": len(incomplete),
                "indexes_created": indexes_created,
                "results": results,
            })
        else:
            summary = f"\n[green]✅ Parity completed: {successes} successful, {failures} failed[/green]"
            if indexes_created:
                summary += f"\n[blue]   ({len(indexes_created)} MeiliSearch indexes created)[/blue]"
            if incomplete:
                summary += f"\n[dim]   ({len(incomplete)} incomplete corpora skipped)[/dim]"
            console.print(summary)

        interaction_logger.finish(result_count=successes)

    except InvalidArgumentError:
        interaction_logger.finish(error="invalid argument")
        raise
    except Exception as e:
        interaction_logger.finish(error=str(e))
        print_error(f"Failed to process all corpora: {e}", output_json)
        sys.exit(1)


def _parity_single_corpus(
    corpus: str,
    dry_run: bool,
    verify: bool,
    repair_metadata: bool,
    effective_text_workers: int,
    qdrant_timeout: int,
    create_missing: bool,
    verbose: bool,
    output_json: bool,
    all_corpora_mode: bool = False,
):
    """Check and restore parity for a single corpus.

    Args:
        corpus: Corpus name
        dry_run: If True, show what would be backfilled without making changes
        verify: If True, verify chunk counts match for files in both systems
        repair_metadata: If True, update MeiliSearch docs with missing git metadata from Qdrant
        effective_text_workers: Number of parallel workers for code chunking
        qdrant_timeout: Timeout in seconds for Qdrant operations
        create_missing: If True, create missing MeiliSearch index for qdrant_only corpus
        verbose: If True, show detailed progress
        output_json: If True, output JSON format
        all_corpora_mode: If True, suppress interaction logging (called from _parity_all_corpora)
    """
    if not all_corpora_mode:
        interaction_logger.start(
            "corpus", "parity",
            corpus=corpus,
            dry_run=dry_run,
        )

    try:
        if not output_json:
            print_info(f"Checking parity for corpus '{corpus}'...")

        # Initialize clients with extended timeout for batch operations
        qdrant = create_qdrant_client(timeout=qdrant_timeout)
        meili = get_meili_client()

        # Verify corpus exists in both systems
        try:
            qdrant.get_collection(corpus)
        except Exception:
            raise ResourceNotFoundError(
                f"Qdrant collection '{corpus}' not found. "
                f"Create it first with: arc corpus create {corpus} --type <type>"
            )

        if not meili.health_check():
            raise ResourceNotFoundError(
                "MeiliSearch server not available. "
                "Start with: docker compose -f deploy/docker-compose.yml up -d meilisearch"
            )

        index_created = False
        meili_index_exists = meili.index_exists(corpus)
        if not meili_index_exists:
            if create_missing:
                # Attempt to create missing index
                if dry_run:
                    if not output_json:
                        console.print(f"[dim]Would create MeiliSearch index for '{corpus}'[/dim]")
                    index_created = True
                    # In dry-run mode with missing index, show what would happen and exit
                    corpus_type = get_collection_type(qdrant, corpus)
                    sync_manager = MetadataBasedSync(qdrant)
                    qdrant_file_paths = sync_manager._get_indexed_file_paths_set(corpus)
                    if output_json:
                        print_json("success", "Dry run complete", data={
                            "corpus": corpus,
                            "dry_run": True,
                            "would_create_index": True,
                            "corpus_type": corpus_type,
                            "would_backfill_to_meili": len(qdrant_file_paths),
                        })
                    else:
                        console.print(f"\n[bold yellow]DRY RUN - No changes will be made[/bold yellow]")
                        console.print(f"\nWould create MeiliSearch index and backfill {len(qdrant_file_paths)} files")
                    if not all_corpora_mode:
                        interaction_logger.finish(result_count=0)
                    return
                else:
                    if _create_missing_meili_index(corpus, output_json):
                        index_created = True
                    else:
                        raise ResourceNotFoundError(
                            f"Failed to create MeiliSearch index '{corpus}'. "
                            f"Create it manually with: arc corpus create {corpus} --type <type>"
                        )
            else:
                raise ResourceNotFoundError(
                    f"MeiliSearch index '{corpus}' not found. "
                    f"Create it first with: arc corpus create {corpus} --type <type> "
                    f"or use --create-missing to auto-create"
                )

        # Get corpus metadata
        corpus_type = get_collection_type(qdrant, corpus)
        metadata = get_collection_metadata(qdrant, corpus)
        configured_models = metadata.get('model', 'stella')
        model_list = [m.strip() for m in configured_models.split(',')]

        if not corpus_type:
            corpus_type = 'pdf'
            logger.warning(f"Collection type not set, defaulting to {corpus_type}")

        if not output_json:
            print_info(f"Corpus type: {corpus_type}, Models: {configured_models}")

        # Get file paths from both systems
        sync_manager = MetadataBasedSync(qdrant)
        qdrant_file_paths = sync_manager._get_indexed_file_paths_set(corpus)
        meili_file_paths = meili.get_all_file_paths(corpus)

        # Calculate set operations
        in_both = qdrant_file_paths & meili_file_paths
        missing_from_meili = qdrant_file_paths - meili_file_paths
        missing_from_qdrant = meili_file_paths - qdrant_file_paths

        # Verify chunk counts if requested
        chunk_mismatches = []
        if verify and in_both:
            if not output_json:
                console.print(f"\n[dim]Verifying chunk counts for {len(in_both)} files...[/dim]")

            # Get chunk counts from both systems
            qdrant_chunk_counts = sync_manager.get_chunk_counts_by_file(corpus)
            meili_chunk_counts = meili.get_chunk_counts_by_file(corpus)

            # Show diagnostic info
            total_qdrant_chunks = sum(qdrant_chunk_counts.values())
            total_meili_chunks = sum(meili_chunk_counts.values())
            if not output_json:
                console.print(f"[dim]  Qdrant total chunks: {total_qdrant_chunks} across {len(qdrant_chunk_counts)} files[/dim]")
                console.print(f"[dim]  MeiliSearch total chunks: {total_meili_chunks} across {len(meili_chunk_counts)} files[/dim]")

            # Compare counts for files in both systems
            for file_path in in_both:
                qdrant_count = qdrant_chunk_counts.get(file_path, 0)
                meili_count = meili_chunk_counts.get(file_path, 0)
                if qdrant_count != meili_count:
                    chunk_mismatches.append({
                        'file_path': file_path,
                        'qdrant_chunks': qdrant_count,
                        'meili_chunks': meili_count,
                    })

            # Also check for files in the counts but not in file_paths (edge case)
            qdrant_only_in_counts = set(qdrant_chunk_counts.keys()) - meili_file_paths
            meili_only_in_counts = set(meili_chunk_counts.keys()) - qdrant_file_paths
            if verbose and not output_json:
                if qdrant_only_in_counts:
                    console.print(f"[dim]  Files in Qdrant counts but not MeiliSearch file_paths: {len(qdrant_only_in_counts)}[/dim]")
                if meili_only_in_counts:
                    console.print(f"[dim]  Files in MeiliSearch counts but not Qdrant file_paths: {len(meili_only_in_counts)}[/dim]")

            if chunk_mismatches:
                # Move mismatched files from "in_both" to the appropriate backfill list
                # Files with more chunks in Qdrant need MeiliSearch backfill
                # Files with more chunks in MeiliSearch need Qdrant backfill (re-index)
                for mismatch in chunk_mismatches:
                    fp = mismatch['file_path']
                    in_both.discard(fp)
                    # Qdrant is source of truth - backfill to MeiliSearch
                    missing_from_meili.add(fp)

        # Report status
        if not output_json:
            console.print(f"\n[bold]Index Status:[/bold]")
            console.print(f"  Files in both systems:     {len(in_both)}")
            console.print(f"  Files in Qdrant only:      {len(missing_from_meili)}")
            console.print(f"  Files in MeiliSearch only: {len(missing_from_qdrant)}")

            if chunk_mismatches:
                console.print(f"\n[yellow]Chunk count mismatches:      {len(chunk_mismatches)} files[/yellow]")
                if verbose:
                    for m in chunk_mismatches[:10]:
                        console.print(
                            f"  [yellow]{Path(m['file_path']).name}: "
                            f"Qdrant={m['qdrant_chunks']}, MeiliSearch={m['meili_chunks']}[/yellow]"
                        )
                    if len(chunk_mismatches) > 10:
                        console.print(f"  [yellow]... and {len(chunk_mismatches) - 10} more[/yellow]")

        # Check which files missing from Qdrant exist on disk
        qdrant_backfill_paths = []
        qdrant_skip_paths = []
        for path in missing_from_qdrant:
            if Path(path).exists():
                qdrant_backfill_paths.append(path)
            else:
                qdrant_skip_paths.append(path)

        meili_backfill_paths = list(missing_from_meili)

        # Dry-run mode: report and exit
        if dry_run:
            if not output_json:
                console.print(f"\n[bold yellow]DRY RUN - No changes will be made[/bold yellow]")

                if meili_backfill_paths:
                    console.print(f"\nWould backfill to MeiliSearch: {len(meili_backfill_paths)} files")
                    if verbose:
                        for p in meili_backfill_paths[:10]:
                            console.print(f"  {Path(p).name}")
                        if len(meili_backfill_paths) > 10:
                            console.print(f"  ... and {len(meili_backfill_paths) - 10} more")

                if qdrant_backfill_paths:
                    console.print(f"\nWould backfill to Qdrant: {len(qdrant_backfill_paths)} files")
                    if verbose:
                        for p in qdrant_backfill_paths[:10]:
                            console.print(f"  {Path(p).name}")
                        if len(qdrant_backfill_paths) > 10:
                            console.print(f"  ... and {len(qdrant_backfill_paths) - 10} more")

                if qdrant_skip_paths:
                    console.print(f"\n[yellow]Would skip (file not found): {len(qdrant_skip_paths)} files[/yellow]")
                    if verbose:
                        for p in qdrant_skip_paths[:10]:
                            console.print(f"  [yellow]{Path(p).name}[/yellow]")
                        if len(qdrant_skip_paths) > 10:
                            console.print(f"  ... and {len(qdrant_skip_paths) - 10} more")

                if not meili_backfill_paths and not qdrant_backfill_paths:
                    console.print(f"\n[green]✓ Indexes are already in parity[/green]")
            else:
                data = {
                    "corpus": corpus,
                    "dry_run": True,
                    "files_in_both": len(in_both),
                    "would_backfill_to_meili": len(meili_backfill_paths),
                    "would_backfill_to_qdrant": len(qdrant_backfill_paths),
                    "would_skip_not_found": len(qdrant_skip_paths),
                    "skipped_files": qdrant_skip_paths,
                }
                print_json("success", "Dry run complete", data=data)

            if not all_corpora_mode:
                interaction_logger.finish(result_count=0)
            return

        # Nothing to do (unless repair_metadata is requested)
        if not meili_backfill_paths and not qdrant_backfill_paths and not repair_metadata:
            if output_json:
                data = {
                    "corpus": corpus,
                    "files_in_both": len(in_both),
                    "meili_backfilled": 0,
                    "qdrant_backfilled": 0,
                }
                print_json("success", "Indexes are already in parity", data=data)
            else:
                console.print(f"\n[green]✓ Indexes are already in parity[/green]")
            if not all_corpora_mode:
                interaction_logger.finish(result_count=0)
            return

        # Delete mismatched files from MeiliSearch before re-syncing
        if chunk_mismatches:
            mismatched_paths = [m['file_path'] for m in chunk_mismatches]
            if not output_json:
                console.print(f"\n[dim]Deleting {len(mismatched_paths)} mismatched files from MeiliSearch...[/dim]")
            try:
                meili.delete_documents_by_file_paths(corpus, mismatched_paths)
            except Exception as e:
                logger.warning(f"Failed to delete mismatched files: {e}")
                if not output_json:
                    console.print(f"[yellow]Warning: Could not delete mismatched files: {e}[/yellow]")

        # Backfill MeiliSearch from Qdrant
        meili_backfilled = 0
        meili_backfill_chunks = 0
        meili_backfill_failed = 0

        if meili_backfill_paths:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(meili_backfill_paths)} files to MeiliSearch...[/blue]")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling...", total=len(meili_backfill_paths))
                meili_backfilled, meili_backfill_chunks, meili_backfill_failed = _backfill_qdrant_to_meili(
                    qdrant, meili, corpus, meili_backfill_paths,
                    verbose, output_json, progress, backfill_task,
                    fetch_workers=effective_text_workers,
                )

        # Backfill Qdrant from MeiliSearch
        qdrant_backfilled = 0
        qdrant_backfill_chunks = 0
        qdrant_backfill_failed = 0

        if qdrant_backfill_paths:
            if not output_json:
                console.print(f"\n[blue]Backfilling {len(qdrant_backfill_paths)} files to Qdrant (requires embedding)...[/blue]")

            # Initialize embedding client
            use_gpu = not os.environ.get('ARC_NO_GPU', '').lower() in ('1', 'true')
            embedding_client = EmbeddingClient(use_gpu=use_gpu)

            # Get model config for chunking
            first_model = model_list[0]
            if first_model in DEFAULT_MODELS:
                model_config = DEFAULT_MODELS[first_model].__dict__
            else:
                model_config = {
                    'chunk_size': 512,
                    'chunk_overlap': 50,
                    'char_to_token_ratio': 3.3,
                }

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
                disable=output_json,
            ) as progress:
                backfill_task = progress.add_task("Backfilling Qdrant...", total=len(qdrant_backfill_paths))
                qdrant_backfilled, qdrant_backfill_chunks, qdrant_backfill_failed, _ = _backfill_meili_to_qdrant(
                    qdrant, embedding_client, corpus, corpus_type, model_list, model_config,
                    qdrant_backfill_paths, verbose, output_json, progress, backfill_task,
                    text_workers=effective_text_workers,
                )

        # Repair metadata if requested (for code corpora)
        metadata_repaired = 0
        metadata_repair_failed = 0
        if repair_metadata and corpus_type == 'code':
            if not output_json:
                console.print(f"\n[blue]Repairing git metadata in MeiliSearch...[/blue]")
            metadata_repaired, metadata_repair_failed = _repair_meili_metadata(
                qdrant, meili, corpus, verbose, output_json, console
            )

        # Output results
        data = {
            "corpus": corpus,
            "index_created": index_created,
            "files_in_both": len(in_both),
            "meili_backfilled": meili_backfilled,
            "meili_backfill_chunks": meili_backfill_chunks,
            "meili_backfill_failed": meili_backfill_failed,
            "qdrant_backfilled": qdrant_backfilled,
            "qdrant_backfill_chunks": qdrant_backfill_chunks,
            "qdrant_backfill_failed": qdrant_backfill_failed,
            "qdrant_skipped_not_found": len(qdrant_skip_paths),
            "skipped_files": qdrant_skip_paths,
            "metadata_repaired": metadata_repaired,
            "metadata_repair_failed": metadata_repair_failed,
        }

        if output_json:
            print_json("success", f"Parity restored for corpus '{corpus}'", data=data)
        else:
            console.print(f"\n[green]✅ Parity restored for corpus '{corpus}'[/green]")
            if index_created:
                console.print(f"   [blue]MeiliSearch index created[/blue]")

            if meili_backfilled > 0 or meili_backfill_failed > 0:
                status = f"{meili_backfilled} files ({meili_backfill_chunks} chunks)"
                if meili_backfill_failed > 0:
                    status += f" [yellow]({meili_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to MeiliSearch: {status}")

            if qdrant_backfilled > 0 or qdrant_backfill_failed > 0:
                status = f"{qdrant_backfilled} files ({qdrant_backfill_chunks} chunks)"
                if qdrant_backfill_failed > 0:
                    status += f" [yellow]({qdrant_backfill_failed} failed)[/yellow]"
                console.print(f"   Backfilled to Qdrant:      {status}")

            if qdrant_skip_paths:
                console.print(f"   [yellow]Skipped (not found):      {len(qdrant_skip_paths)} files[/yellow]")

            if metadata_repaired > 0 or metadata_repair_failed > 0:
                status = f"{metadata_repaired} documents"
                if metadata_repair_failed > 0:
                    status += f" [yellow]({metadata_repair_failed} failed)[/yellow]"
                console.print(f"   Metadata repaired:         {status}")

        if not all_corpora_mode:
            interaction_logger.finish(
                result_count=meili_backfilled + qdrant_backfilled,
            )

    except (InvalidArgumentError, ResourceNotFoundError):
        if not all_corpora_mode:
            interaction_logger.finish(error="invalid argument or resource not found")
        raise
    except Exception as e:
        if not all_corpora_mode:
            interaction_logger.finish(error=str(e))
        raise


def parity_command(
    corpus: Optional[str],
    dry_run: bool,
    verify: bool,
    repair_metadata: bool,
    text_workers: Optional[int],
    qdrant_timeout: int,
    create_missing: bool,
    confirm: bool,
    verbose: bool,
    output_json: bool
):
    """Check and restore parity between Qdrant and MeiliSearch.

    When corpus is provided, operates on a single corpus.
    When corpus is None, discovers all corpora and processes them with confirmation.

    Compares indexed files in both systems and backfills missing entries:
    - Qdrant -> MeiliSearch: Copies metadata (no file access needed)
    - MeiliSearch -> Qdrant: Re-chunks and embeds files (requires file access)

    Files that don't exist on disk are skipped with a warning.

    Args:
        corpus: Corpus name (optional - if None, process all corpora)
        dry_run: If True, show what would be backfilled without making changes
        verify: If True, verify chunk counts match for files in both systems
        repair_metadata: If True, update MeiliSearch docs with missing git metadata from Qdrant
        text_workers: Number of parallel workers for code chunking (None=auto, 0/1=sequential)
        qdrant_timeout: Timeout in seconds for Qdrant operations
        create_missing: If True, create missing MeiliSearch indexes for qdrant_only corpora
        confirm: If True, skip confirmation prompt when processing all corpora
        verbose: If True, show detailed progress
        output_json: If True, output JSON format
    """
    # Calculate effective workers
    if text_workers is None:
        effective_text_workers = max(1, cpu_count() // 2)
    elif text_workers <= 1:
        effective_text_workers = 1  # Sequential
    else:
        effective_text_workers = text_workers

    if corpus:
        # Single corpus mode
        _parity_single_corpus(
            corpus=corpus,
            dry_run=dry_run,
            verify=verify,
            repair_metadata=repair_metadata,
            effective_text_workers=effective_text_workers,
            qdrant_timeout=qdrant_timeout,
            create_missing=create_missing,
            verbose=verbose,
            output_json=output_json,
            all_corpora_mode=False,
        )
    else:
        # All corpora mode
        _parity_all_corpora(
            dry_run=dry_run,
            verify=verify,
            repair_metadata=repair_metadata,
            text_workers=text_workers,
            qdrant_timeout=qdrant_timeout,
            create_missing=create_missing,
            confirm=confirm,
            verbose=verbose,
            output_json=output_json,
            effective_text_workers=effective_text_workers,
        )
